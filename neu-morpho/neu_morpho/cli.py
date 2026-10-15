"""neu-morpho: index -> allowlist -> meshes, locally or on SLURM.

The command-line entry point for this package.

    neu-morpho run ...              # the pipeline
    neu-morpho progress <work-dir>  # live per-stage counts
    neu-morpho run-report <work-dir># self-contained HTML summary
    neu-morpho measure ...          # morphometry over PUBLISHED output

Run ``run`` ON A WORKSTATION, in a session that outlives your terminal: it starts a
dask cluster whose workers are SLURM jobs (dask submits the sbatch itself via
``scale()``), then runs the requested stages against it. Every stage is idempotent,
so re-running the same command resumes.

    # 0. look at the pyramid first, and pick your scales from real metadata
    neu-morpho run --src /path/to/seg --describe

    # 1. small ROI, locally, to see it work end to end
    neu-morpho run --src ... \\
        --dst /path/to/morpho/segmentation --work-dir /path/to/morpho \\
        --config dask-local --workers 4 \\
        --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh

    # 2. the same ROI on SLURM, surviving logout, publishing to s3
    nohup env PYTHONUNBUFFERED=1 neu-morpho run --src ... \\
        --dst s3://bucket/sample3/segmentation --work-dir /path/to/morpho \\
        --config dask-slurm-example --config ~/my-site.yaml --workers 48 \\
        --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh > run.log 2>&1 &
    squeue -u "$USER"        # watch your jobs (read-only; don't poll in a tight loop)

``python -m neu_morpho`` is equivalent to ``neu-morpho``. ``PYTHONUNBUFFERED=1`` is
the console-script equivalent of ``python -u``; without it the log lags the run.
Keep that log OFF a shared network filesystem such as CephFS: unbuffered, every line is
its own write, and once another node reads the file (``tail -f``) each one becomes a
synchronous round trip. A local disk or your home directory is fine.

**--config takes a bundled template name or a path, and is repeatable**, deep-merged
left to right — so a site config carries only the keys that differ. The templates ship
with **blockrun**, next to ``start_dask``.

**--dst is the published volume; --work-dir is everything else.** --dst holds the
segmentation scales with meshes and skeletons inside it, and may be an ``s3://``
URL. --work-dir must be an ordinary filesystem path: it holds the stage-1
fragments, the sqlite metrics DB and the appended JSONL manifests, none of which
work over an object store.

Because the two are separate, a manifest can outlive the data it describes —
clear --dst and a resumed run would skip everything and report success having
written nothing. Each stage refuses to resume when its manifest records completed
work but its output ``info`` is gone; --no-resume is the explicit override.

The ROI is in the voxels of **--roi-scale, which is required with --roi** — the same
six numbers name a box 4x smaller per pyramid level, and the wrong scale silently
processes the wrong region rather than failing. It filters blocks on the *global*
grid, so widening it later re-uses everything already done rather than redoing it
(see neu_morpho/roi.py). Drop --roi for the whole volume.

Scales are integers (--mesh-scale / --index-scale); the voxel size
of each is read from the source metadata, never assumed to be 2**scale — that
assumption is what misaligns meshes against skeletons on non-standard pyramids.

Skeletons come from **neu-skel**, a separate GPL-licensed package: ``neu-skel run``
adds them to a work dir this command has indexed, with the same --src, --dst and
--work-dir.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time

from blockrun import bundled_configs, load_config

from neu_morpho.metrics_db import MetricsDB
from neu_morpho.roi import parse_roi
from neu_vol import describe_scales as describe

log = logging.getLogger("neu-morpho")

from neu_morpho.pipeline import STAGE_DOC, STAGES  # noqa: E402  (re-exported for the docs)


def stages_help(indent: str = "  ", width: int = 79, stage_doc=None) -> str:
    """The stage list as ``--help`` prints it: one wrapped paragraph per stage.

    For the ``run`` subparser's description (``RawDescriptionHelpFormatter`` keeps the
    layout); argparse would re-wrap it inside ``--stages``' own help.
    """
    import textwrap

    stage_doc = STAGE_DOC if stage_doc is None else stage_doc
    label = max(len(name) for name in stage_doc) + 2
    out = ["Stages, in the order they run (--stages):", ""]
    for name, doc in stage_doc.items():
        lines = textwrap.wrap(doc, width=width - len(indent) - label)
        out.append(f"{indent}{name:<{label}}{lines[0]}")
        out += [f"{indent}{'':<{label}}{line}" for line in lines[1:]]
        out.append("")
    return "\n".join(out)


def _configs(args) -> list[str]:
    """``--config`` values, defaulting to blockrun's bundled local template; resolution
    and validation are ``blockrun.dask_config``'s."""
    return args.config or ["dask-local"]


def add_run_args(p, *, stages=STAGES, default_stages="index,mesh") -> None:
    """The ``run`` arguments for ``stages``, on parser ``p``.

    An option only one built-in stage reads is added only with that stage, and a single
    stage gets no ``--stages``. ``neu-skel run`` passes ``stages=("skel",)`` and adds
    its own options.
    """
    p.add_argument("--src", required=True, help="segmentation volume (path or s3://...)")
    p.add_argument("--dst", help="the published precomputed VOLUME: labels with "
                                 "meshes/skeletons inside it. Local path or s3://...; "
                                 "bookkeeping goes to --work-dir")
    p.add_argument("--work-dir", help="run directory: stage-1 fragments, manifests, "
                                      "metrics DB, failures, run summary. Must be a "
                                      "filesystem path. Defaults to the PARENT of a "
                                      "local --dst; required when --dst is remote")
    p.add_argument("--describe", action="store_true",
                   help="print the source pyramid and exit (no cluster, no writes)")

    if len(stages) > 1:
        p.add_argument("--stages", default=default_stages, metavar="LIST",
                       help=f"comma-separated subset of {','.join(stages)}, described "
                            f"under 'Stages' above. They run in that order, and each is "
                            f"idempotent and resumable, so re-running a subset continues "
                            f"rather than redoing it (default: {default_stages})")
    else:
        p.set_defaults(stages=stages[0])
    if "seg" in stages:
        p.add_argument("--seg-encoding", default="compressed_segmentation",
                       help="encoding for the exported segmentation (or 'raw')")
        p.add_argument("--seg-scales", default="all",
                       help="source scales to copy in the seg stage: 'all', a range "
                            "like '1-6', or a comma list. Scale 0 usually dominates "
                            "the cost by an order of magnitude — check the log line")
    p.add_argument("--roi", help="z0,y0,x0,z1,y1,x1 voxels, at --roi-scale")
    p.add_argument("--roi-scale", type=int, default=None,
                   help="scale the --roi values are given in. REQUIRED with --roi: "
                        "the same six numbers name a different box at every scale, "
                        "and guessing wrong silently processes the wrong region")

    if "index" in stages:
        p.add_argument("--index-scale", type=int, default=2,
                       help="scale to scan for bboxes")
    if "mesh" in stages:
        p.add_argument("--mesh-scale", type=int, default=2)
    p.add_argument("--block", default="256,256,256", help="block shape (z,y,x) voxels")

    p.add_argument("--min-voxels", type=int, default=0,
                   help="only process bodies with at least this many voxels, counted "
                        "at the index scale by the index stage")
    p.add_argument("--limit-bodies", type=int, default=None,
                   help="cap the allowlist to the N largest bodies (needs the index "
                        "stage too)")
    p.add_argument("--allowlist", help="explicit body-id file (overrides --min-voxels)")
    p.add_argument("--no-metrics-db", action="store_true",
                   help="skip the per-body metrics DB (its enrichment is observational; "
                        "with an explicit --allowlist the index stage is unnecessary too)")

    p.add_argument("--occupancy-scale", type=int, default=5,
                   help="coarse scale used to skip empty blocks; -1 disables")
    p.add_argument("--occupancy-dilate", type=int, default=1,
                   help="grow the occupied set by N blocks. Keep >=1: a coarse scale "
                        "misses sparse blocks and the miss does not converge, so an "
                        "un-dilated filter silently skips real data")

    p.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH",
                   help=f"dask config: a bundled template ({', '.join(bundled_configs())}) "
                        f"or a path to your own YAML. Repeatable, deep-merged left to "
                        f"right, so an overlay need only carry what differs from the "
                        f"template. Default: dask-local")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--serial", action="store_true",
                   help="no dask at all — run in this process (smallest smoke test)")
    p.add_argument("--no-resume", action="store_true", help="start over, ignoring manifests")
    p.add_argument("--max-consecutive-failures", type=int, default=10,
                   help="stop a stage after this many per-body failures in a row "
                        "(0 disables; systemic errors abort immediately regardless)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan (scales, ROI, block counts) and exit")
    p.add_argument("--store-logs", action="store_true",
                   help="keep TensorStore's benign S3 credential-chain logging "
                        "(suppressed by default; real errors are never suppressed)")


def build_parser() -> argparse.ArgumentParser:
    """The full ``neu-morpho`` parser, built but not run; the docs render the CLI reference
    from it."""
    top = argparse.ArgumentParser(
        prog="neu-morpho", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "run", help="index -> allowlist -> meshes",
        description=f"{__doc__}\n{stages_help()}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(func=_main)
    add_run_args(p)
    # --- progress -----------------------------------------------------------
    q = sub.add_parser("progress", help="live progress of a run, from its work dir",
                       description="Per-stage task counts against the plan, for a run "
                                   "in flight. Reads only; re-run to refresh.")
    q.add_argument("work_dir")
    q.add_argument("--sample", type=int, default=None, metavar="SECONDS",
                   help="sample the manifests twice this far apart to estimate a rate")
    q.set_defaults(func=_cmd_progress)

    # --- measure ------------------------------------------------------------
    # A SEPARATE command, not a `run --stages` entry: these read published output and
    # write tables, where `run` reads a source and writes a volume.
    m = sub.add_parser("measure", help="per-body morphometry over published output",
                       description=_MEASURE_DESC,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    msub = m.add_subparsers(dest="measure_command", required=True)

    v = msub.add_parser("volumes", help="per-body voxel count and volume",
                        description=_MEASURE_VOLUMES_DESC,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    v.add_argument("--src", required=True, help="segmentation to read (path or URL)")
    v.add_argument("--db", required=True, help="metrics DB to accumulate into")
    v.add_argument("--level", type=int, default=2,
                   help="pyramid level to read. Its voxel size comes from the source "
                        "metadata, never from 2**level")
    v.add_argument("--block", type=int, default=512, help="cube block edge, in voxels")
    v.add_argument("--keep", action="append", default=None, metavar="SRC",
                   help="restrict which bodies are RECORDED (not which blocks are read). "
                        "A file of ids, or a neuroglancer_segment_properties URL. "
                        "Repeatable; the id sets are unioned")
    v.add_argument("--with-bbox", action="store_true",
                   help="also record each body's bounding box. Roughly 10x the cost on a "
                        "dense block, and only worth it if something reads the bbox")
    v.add_argument("--roi", default=None, help="z0,y0,x0,z1,y1,x1 in --level voxels")
    v.add_argument("--background", type=int, default=0,
                   help="label to treat as background. Annotation tools that number from "
                        "0 make this 1")
    v.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--serial", action="store_true", help="no dask; run in this process")
    v.add_argument("--no-resume", action="store_true")
    v.add_argument("--dry-run", action="store_true")
    v.add_argument("--store-logs", action="store_true")
    v.set_defaults(func=_cmd_measure_volumes)

    k = msub.add_parser("skeletons", help="cable length and topology per body",
                        description=_MEASURE_SKEL_DESC,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    k.add_argument("--volume", required=True,
                   help="volume the skeletons live under (path or URL)")
    k.add_argument("--db", required=True)
    k.add_argument("--skeleton-dir", default="skeleton",
                   help="subdirectory named by the volume's info 'skeletons' key")
    k.add_argument("--bodies", action="append", required=True, metavar="SRC",
                   help="which bodies to measure: an id file or a "
                        "neuroglancer_segment_properties URL. Repeatable, unioned")
    k.add_argument("--require-radii", action="store_true",
                   help="fail on a body whose skeleton carries no radius attribute, "
                        "instead of recording NaN diameters. Use on a volume this "
                        "package wrote, where missing radii mean something went wrong")
    k.add_argument("--batch", type=int, default=200, help="bodies per task")
    k.add_argument("--threads", type=int, default=8, help="concurrent reads per task")
    k.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH")
    k.add_argument("--workers", type=int, default=4)
    k.add_argument("--serial", action="store_true")
    k.add_argument("--no-resume", action="store_true")
    k.add_argument("--no-retry-failed", action="store_true",
                   help="treat a previously failed body as done")
    k.add_argument("--dry-run", action="store_true")
    k.add_argument("--store-logs", action="store_true")
    k.set_defaults(func=_cmd_measure_skeletons)

    cm = msub.add_parser("compartments", help="per-body volume split by semantic label",
                         description=_MEASURE_COMPARTMENTS_DESC,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    cm.add_argument("--src", required=True, help="segmentation")
    cm.add_argument("--semantic", required=True,
                    help="semantic label volume, on the same grid at --level")
    cm.add_argument("--db", required=True)
    cm.add_argument("--level", type=int, default=2, help="segmentation level to read")
    cm.add_argument("--semantic-level", type=int, default=None,
                    help="semantic level with the SAME voxel size; found by matching the "
                         "voxel size if omitted, which is safer than assuming an index")
    cm.add_argument("--block", type=int, default=512)
    cm.add_argument("--keep", action="append", default=None, metavar="SRC")
    cm.add_argument("--occupancy-scale", type=int, default=None, metavar="LEVEL",
                    help="restrict to blocks non-empty at this coarse level OF THE "
                         "SEGMENTATION. Safe in a way an anatomical ROI is not, and "
                         "--verify proves afterwards that nothing was missed")
    cm.add_argument("--occupancy-dilate", type=int, default=1)
    cm.add_argument("--roi", default=None)
    cm.add_argument("--background", type=int, default=0)
    cm.add_argument("--verify", action="store_true",
                    help="after the run, check each body's per-label sum against the "
                         "total the volume sweep recorded")
    cm.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH")
    cm.add_argument("--workers", type=int, default=4)
    cm.add_argument("--serial", action="store_true")
    cm.add_argument("--no-resume", action="store_true")
    cm.add_argument("--dry-run", action="store_true")
    cm.add_argument("--store-logs", action="store_true")
    cm.set_defaults(func=_cmd_measure_compartments)

    g = msub.add_parser("progress", help="live progress of a measure run, from its DB",
                        description="Blocks done against the total each stage recorded "
                                    "before dispatch. Reads only; re-run to refresh.")
    g.add_argument("db")
    g.set_defaults(func=_cmd_measure_progress)

    # --- run-report ---------------------------------------------------------
    q = sub.add_parser("run-report", help="self-contained HTML summary of a run",
                       description="Write a human-readable HTML summary of a run from "
                                   "its work dir. Works on an in-flight run too.")
    q.add_argument("work_dir")
    q.add_argument("-o", "--out", default=None,
                   help="output HTML (default: <work-dir>/report.html)")
    q.set_defaults(func=_cmd_run_report)

    _add_help_command(sub)

    # `run`'s own subparser, so `check_run_args` reports errors against it.
    top._run_parser = p
    return top


def _add_help_command(sub) -> None:
    """``neu-morpho help [COMMAND ...]`` — the same text ``--help`` prints.

    It re-parses with ``--help`` rather than reaching into the parser, so it cannot drift,
    nested subcommands work, and an unknown name gets argparse's own error.
    """
    q = sub.add_parser(
        "help", help="what `--help` prints, as a subcommand",
        description="Print the help for the tool, or for one of its subcommands.\n\n"
                    "`neu-morpho help run` and `neu-morpho run --help` are the same "
                    "thing; nested subcommands work too (`help measure run`).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("command", nargs="*", metavar="COMMAND",
                   help="the subcommand to describe (default: the whole tool)")
    q.set_defaults(func=cmd_help)


def cmd_help(args) -> int:
    """Hand the arguments back to the parser as ``--help``, and report its exit code."""
    try:
        build_parser().parse_args([*args.command, "--help"])
    except SystemExit as e:                      # what --help always does
        return int(e.code or 0)
    return 0


def _parse_args(argv=None):
    top = build_parser()
    args = top.parse_args(argv)
    if args.command == "run":
        check_run_args(top._run_parser, args)
    return args


def check_run_args(p, args) -> None:
    """The ``run`` checks argparse cannot express, reported as errors of parser ``p``."""
    # No default for --roi-scale: an ROI read at the wrong scale processes the wrong
    # region, and nothing downstream can tell it from a deliberately smaller ROI.
    if args.roi and args.roi_scale is None:
        p.error("--roi-scale is required with --roi: the same six values name a "
                "different box at every pyramid level, and the wrong one silently "
                "processes the wrong region. Pass the scale the numbers are in "
                "(e.g. --roi-scale 2).")
    if args.roi_scale is not None and not args.roi:
        p.error("--roi-scale has no effect without --roi")

    # Validate here, so a mistyped --config fails before any remote metadata read.
    if not args.serial:
        load_config(_configs(args))


def _cmd_progress(args) -> int:
    """Live per-stage progress, from the stdlib-only ``_scripts.run_progress``."""
    from neu_morpho._scripts import run_progress

    argv = [args.work_dir] + (["--sample", str(args.sample)] if args.sample else [])
    return run_progress.main(argv) or 0


_MEASURE_DESC = """\
Per-body morphometry over PUBLISHED output.

Separate from `run` on purpose: `run` reads a source and writes a volume, while these
read a finished volume and write tables. The thing being measured is often not the
thing this pipeline produced.

    neu-morpho measure volumes  --src ... --db ...   # per-body voxel count + volume
    neu-morpho measure progress <db>                 # how far along that is
"""

_MEASURE_VOLUMES_DESC = """\
Count each body's voxels and accumulate per-body volume into a metrics DB.

Reads EVERY block by default. An ROI built from a compartment mask omits somata that
sit outside it, and dividing a truncated volume by a complete cable length is wrong in
a way nothing downstream can see; an empty block is cheap enough that reading all of
them is the affordable choice as well as the safe one. Pass --roi only when the region
really is the subject.

--keep restricts which bodies are RECORDED, never which blocks are read, so the rows it
writes are complete. Without it a fragmented segmentation records every single-voxel
speck, and each one costs a driver-side upsert.

--with-bbox switches to the index scan, which also records bounding boxes. That
costs an argwhere plus an argsort over every labelled voxel — ~30 s against ~3 s per
dense block — so it is opt-in. The two paths both accumulate into voxel_count and
refuse to share a DB, since running both would double every count.
"""


def _cmd_measure_volumes(args) -> int:
    from blockrun import iter_blocks
    from neu_vol import read_scales, scale_spec
    from neu_vol.logs import quiet_store_logs
    from neu_morpho.measure.driver import resolve_keep

    with quiet_store_logs(not args.store_logs):
        scales = read_scales(args.src)
        if not 0 <= args.level < len(scales):
            raise SystemExit(f"--level {args.level} out of range "
                             f"(source has {len(scales)} levels)")
        s = scales[args.level]
        log.info("source L%d %s @ %s nm  origin %s", s.index, s.shape, s.voxel_size,
                 s.origin_nm)
        keep = resolve_keep(args.keep)
        if keep is not None:
            log.info("keep: %d bodies from %d source(s)", len(keep), len(args.keep))
        n_blocks = sum(1 for _ in iter_blocks(s.shape, (args.block,) * 3))
        log.info("%d blocks of %d^3%s -> %s", n_blocks, args.block,
                 " (+bbox)" if args.with_bbox else "", args.db)
        if args.dry_run:
            log.info("--dry-run: nothing executed")
            return 0

        t = time.time()
        common = dict(keep=keep, roi=args.roi, resume=not args.no_resume)
        if args.serial:
            out = _measure_dispatch(args, scale_spec(args.src, args.level), s,
                                    client=None, **common)
        else:
            from blockrun import start_dask
            cfg = args.config or ["dask-local"]
            with start_dask(args.workers, cfg, label="measure-volumes",
                            logger=log) as client:
                out = _measure_dispatch(args, scale_spec(args.src, args.level), s,
                                        client=client, **common)
        out["minutes"] = round((time.time() - t) / 60, 2)
        log.info("measure volumes: %s", out)
    return 0


def _measure_dispatch(args, spec, s, *, client, keep, roi, resume):
    """--with-bbox reuses the index scan; the default is the counts-only sweep."""
    if args.with_bbox:
        from neu_morpho.ops import index_segments

        return index_segments(spec, args.db, scan_voxel_size=s.voxel_size,
                              scan_scale=s.index, block_shape=(args.block,) * 3,
                              roi=roi, keep=keep, client=client, resume=resume)
    from neu_morpho.measure.driver import sweep_volumes
    return sweep_volumes(spec, args.db, voxel_size=s.voxel_size, block=args.block,
                         keep=keep, roi=roi, background=args.background,
                         client=client, resume=resume)


_MEASURE_SKEL_DESC = """\
Cable length, topology and (where the source has radii) diameter statistics, per body.

The per-BODY pass, where `volumes` is per-block, and the differences are deliberate.
Bodies are batched into tasks because each read is one small object and per-task
dispatch would otherwise dominate; within a batch, reads run on threads, and they all
share one opened store. A body that fails is RECORDED and skipped rather than killing
the run — the opposite of the block sweep's fail-fast, because one unreadable skeleton
says nothing about the others, whereas one skipped block truncates every body in it.

A body with no published skeleton is recorded `absent` and never retried. A body that
raised is recorded `failed` and IS retried next run. Those are different states and
collapsing them would make a transient error permanent.
"""


def _cmd_measure_skeletons(args) -> int:
    from neu_vol.logs import quiet_store_logs
    from neu_morpho.measure.driver import resolve_keep
    from neu_morpho.measure.skeletons import sweep_skeletons

    with quiet_store_logs(not args.store_logs):
        bodies = resolve_keep(args.bodies)
        if not bodies:
            raise SystemExit("--bodies resolved to no ids")
        log.info("%d bodies; skeletons under %s/%s", len(bodies), args.volume,
                 args.skeleton_dir)
        if args.dry_run:
            log.info("--dry-run: nothing executed")
            return 0

        t = time.time()
        common = dict(bodies=sorted(bodies), skeleton_dir=args.skeleton_dir,
                      require_radii=args.require_radii, batch=args.batch,
                      threads=args.threads, resume=not args.no_resume,
                      retry_failed=not args.no_retry_failed)
        if args.serial:
            out = sweep_skeletons(args.volume, args.db, client=None, **common)
        else:
            from blockrun import start_dask
            with start_dask(args.workers, args.config or ["dask-local"],
                            label="measure-skeletons", logger=log) as client:
                out = sweep_skeletons(args.volume, args.db, client=client, **common)
        out["minutes"] = round((time.time() - t) / 60, 2)
        log.info("measure skeletons: %s", out)
    return 0


_MEASURE_COMPARTMENTS_DESC = """\
Split each body's volume by semantic compartment (nucleus, soma, neuropil, ...).

Reads the segmentation block and the semantic block over the SAME voxel indices and counts
(body, label) pairs in one pass. Requires the two volumes to share a grid at the chosen
level -- same voxel size, same origin -- which is checked, not assumed.

Label 0 is counted too, so `sum over labels == the body's total voxel count` is an exact
identity. --verify checks that against the totals a previous `measure volumes` run
recorded. That is what makes --occupancy-scale safe to use: a coarse grid CAN drop a block
that holds data, and normally you would never learn which, but the volume sweep read every
block, so its totals are ground truth and any deficit is named rather than silent.

Note the distinction from an anatomical ROI, which is NOT safe here: a compartment mask
omits somata lying outside it, whereas a soma is segmentation and so is non-zero at any
coarse level of the segmentation itself.

non_somatic = total - (nucleus + soma), so nothing needs measuring twice.
"""


def _cmd_measure_compartments(args) -> int:
    import numpy as np

    from neu_vol import read_scales, scale_spec
    from neu_vol.backends.base import open_backend
    from neu_vol.logs import quiet_store_logs
    from neu_morpho.measure.compartments import (semantic_label_names, somatic_labels,
                                                 sweep_compartments, verify_compartments)
    from neu_morpho.measure.driver import resolve_keep

    with quiet_store_logs(not args.store_logs):
        seg_scales = read_scales(args.src)
        if not 0 <= args.level < len(seg_scales):
            raise SystemExit(f"--level {args.level} out of range")
        s = seg_scales[args.level]

        sem_scales = read_scales(args.semantic)
        if args.semantic_level is not None:
            sem = sem_scales[args.semantic_level]
        else:
            match = [x for x in sem_scales
                     if np.allclose(x.voxel_size, s.voxel_size)]
            if not match:
                raise SystemExit(
                    f"the semantic volume has no level at {s.voxel_size} nm "
                    f"(it has {[x.voxel_size[0] for x in sem_scales]}); pass "
                    f"--semantic-level explicitly if you know what you are doing")
            sem = match[0]
        # Same voxel size is necessary but not sufficient — a shared ORIGIN is what makes
        # a voxel index mean the same thing in both.
        if not np.allclose(s.origin_nm, sem.origin_nm):
            raise SystemExit(f"grids do not share an origin: segmentation "
                             f"{s.origin_nm} vs semantic {sem.origin_nm}")
        log.info("seg  L%d %s @ %s", s.index, s.shape, s.voxel_size)
        log.info("sem  L%d %s @ %s", sem.index, sem.shape, sem.voxel_size)
        names = semantic_label_names(args.semantic)
        log.info("semantic labels: %s", names)
        log.info("somatic = %s -> labels %s", list(("nucleus", "soma")),
                 somatic_labels(args.semantic))

        keep = resolve_keep(args.keep)
        blocks = None
        if args.occupancy_scale is not None:
            from neu_morpho.occupancy import occupied_blocks
            occ = seg_scales[args.occupancy_scale]
            arr = open_backend(scale_spec(args.src, occ.index)).read_region(
                tuple(slice(0, int(d)) for d in occ.shape))
            grid_shape = [int(math.ceil(d / args.block)) for d in s.shape]
            blocks = occupied_blocks(arr, occ_voxel_size=occ.voxel_size,
                                     mesh_voxel_size=s.voxel_size,
                                     block_shape=(args.block,) * 3,
                                     grid_shape=grid_shape,
                                     dilate=args.occupancy_dilate)
            log.info("occupancy L%d dilate %d: %d of %d blocks",
                     occ.index, args.occupancy_dilate, len(blocks),
                     int(np.prod(grid_shape)))
        if args.dry_run:
            log.info("--dry-run: nothing executed")
            return 0

        t = time.time()
        common = dict(sem_shape=sem.shape, block=args.block, keep=keep, blocks=blocks,
                      roi=args.roi, background=args.background,
                      resume=not args.no_resume)
        specs = (scale_spec(args.src, s.index), scale_spec(args.semantic, sem.index))
        if args.serial:
            out = sweep_compartments(*specs, args.db, client=None, **common)
        else:
            from blockrun import start_dask
            with start_dask(args.workers, args.config or ["dask-local"],
                            label="measure-compartments", logger=log) as client:
                out = sweep_compartments(*specs, args.db, client=client, **common)
        out["minutes"] = round((time.time() - t) / 60, 2)
        log.info("measure compartments: %s", out)

        if args.verify:
            v = verify_compartments(args.db)
            log.info("verify: %d bodies, %d mismatched, %d voxels missing (%.4f%%)",
                     v["n_bodies"], v["n_mismatched"], v["voxels_missing"],
                     100 * v["fraction_missing"])
            if v["n_mismatched"]:
                log.warning("worst (body, total, summed): %s", v["worst"])
    return 0


def _cmd_measure_progress(args) -> int:
    if not os.path.exists(args.db):
        raise SystemExit(f"no such metrics DB: {args.db}")
    # Read-only: a default open runs DDL, whose write lock can kill a running sweep.
    db = MetricsDB(args.db, read_only=True)
    try:
        meta, counts = db.read_stage_meta(), db.stage_counts()
        bodies = db.con.execute(
            "SELECT COUNT(*) FROM bodies WHERE voxel_count>0").fetchone()[0]
        # From the registry, so every registered stage is shown.
        rows = [st for st in MetricsDB.BLOCK_STAGES if counts.get(st) or st in meta]
        if not rows:
            print(f"{args.db}: no measure stage has run")
            return 0
        for st in rows:
            done, m = counts.get(st, 0), meta.get(st, {})
            total = m.get("total")
            if total:
                frac = done / total
                bar = "#" * int(24 * frac) + "." * (24 - int(24 * frac))
                print(f"{st:6s} [{bar}] {done:>7d}/{total:<7d} {100*frac:5.1f}%")
            else:
                # A stage that ran before the total was recorded, or a killed driver.
                print(f"{st:6s} {done:>7d} blocks done, total unrecorded")
            if m:
                print(f"       block {m.get('block_shape')}  voxel {m.get('voxel_size')}"
                      f"  keep {m.get('n_keep')}  started {m.get('started')}")
        print(f"bodies with a voxel count: {bodies}")
        skel = db.skel_counts()
        if skel:
            total = (meta.get("skel") or {}).get("total")
            seen = sum(skel.values())
            head = f"skel   {seen}/{total}" if total else f"skel   {seen} bodies"
            # Statuses are shown separately because they are not interchangeable:
            # `absent` is terminal, `failed` is retried on the next run.
            print(f"{head}  " + "  ".join(f"{k} {v}" for k, v in sorted(skel.items())))
    finally:
        db.close()
    return 0


def _cmd_run_report(args) -> int:
    from neu_morpho._scripts import run_report

    argv = [args.work_dir] + (["-o", args.out] if args.out else [])
    return run_report.main(argv) or 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from neu_vol.logs import quiet_store_logs

    # TensorStore's S3 stack logs its credential-provider chain at ERROR severity
    # on success; on a real run that buried the actual output. Real failures
    # (PERMISSION_DENIED and friends) are never suppressed — see logs.NEVER_DROP.
    with quiet_store_logs(not getattr(args, "store_logs", False)):
        return args.func(args)


def _main(args) -> int:
    return run_main(args)


def run_main(args, *, extra_stages=None, runners=None, extra=None,
             label: str = "neu-morpho") -> int:
    """``run`` from parsed arguments: plan, then run on a cluster or serially.

    ``extra_stages`` / ``runners`` / ``extra`` pass through to :mod:`neu_morpho.pipeline`
    for a stage another package provides.
    """
    from neu_morpho import pipeline

    if args.describe:
        print(describe(args.src))
        return 0
    if not args.dst:
        raise SystemExit("--dst is required unless --describe")

    seg_scales = None
    raw_scales = getattr(args, "seg_scales", None)
    if raw_scales and raw_scales != "all":
        if "-" in raw_scales:
            a, _, b = raw_scales.partition("-")
            seg_scales = list(range(int(a), int(b) + 1))
        else:
            seg_scales = [int(x) for x in raw_scales.split(",")]
    # Options add_run_args left out for these stages keep pipeline.plan's defaults.
    stage_opts = {k: getattr(args, k) for k in ("index_scale", "mesh_scale", "seg_encoding")
                  if hasattr(args, k)}
    try:
        plan = pipeline.plan(
            args.src, args.dst, work_dir=args.work_dir,
            stages=[s.strip() for s in args.stages.split(",") if s.strip()],
            block=tuple(int(v) for v in args.block.split(",")),
            extra_stages=extra_stages, roi=parse_roi(args.roi), roi_scale=args.roi_scale,
            seg_scales=seg_scales, occupancy_scale=args.occupancy_scale,
            occupancy_dilate=args.occupancy_dilate, allowlist=args.allowlist,
            min_voxels=args.min_voxels, limit_bodies=args.limit_bodies,
            metrics_db=not args.no_metrics_db,
            max_consecutive_failures=args.max_consecutive_failures,
            resume=not args.no_resume, extra=extra, **stage_opts)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    if args.serial:
        log.info("serial mode: no dask client")
        summary = pipeline.run(plan, runners=runners)
    else:
        # Imported here: it pulls in dask.distributed (~1 s), which `progress` and
        # `run-report` never need.
        from blockrun import start_dask

        with start_dask(args.workers, _configs(args), label=label,
                        effective_config_path=plan.work_dir) as client:
            summary = pipeline.run(plan, client=client, runners=runners)
    # Isolated per-body failures do not stop the run; exit non-zero so a scripted
    # pipeline does not treat a partial result as clean.
    return 1 if summary["n_failed_bodies"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
