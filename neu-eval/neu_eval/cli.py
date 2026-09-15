"""The ``neu-eval`` command.

One subcommand so far. ``compare`` reads a reference and a segmentation as two registered
pieces, scores them, and writes the metrics plus a ranked disagreement table you fill in and
hand back.

Every heavy import is inside the handler that needs it, so ``--help`` stays fast and the
docs site — which imports this module to render the CLI reference, with no conda and no
tensorstore — can do so at all.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import __version__

logger = logging.getLogger(__name__)

_EPILOG = """\
examples:
  # one annotated crop against a segmentation, read registered to it
  neu-eval compare --reference gt.h5:/crop_01 --segmentation SEG_URL --level 0 --out cmp/

  # every crop in the container, plus a pooled score over all of them
  neu-eval compare --reference gt.h5 --all-datasets --segmentation SEG_URL --out cmp/

  # after filling in the verdict column, rescore without the reference's own errors
  neu-eval compare --reference gt.h5:/crop_01 --segmentation SEG_URL \\
      --adjudicated cmp/disagreements.csv --out cmp/

  # step through the worst disagreements in a viewer. The coordinates are nanometres,
  # so `annotate` needs a frame: --volume is better than --voxel-size, since it also
  # bounds-checks. Then compose the layer with the volumes.
  neu-glance annotate --points cmp/annotations.csv --nm --volume REF_URL \\
      --name disagreements --format layer --out layer.json
  neu-glance gen --image IMAGE_URL --seg REF_URL --seg SEG_URL --layer layer.json \\
      --format url
"""


def build_parser() -> argparse.ArgumentParser:
    """The parser. **Name and signature are a contract with the docs site**, which imports
    this module and renders ``--help`` from the real object so the reference cannot drift."""
    p = argparse.ArgumentParser(
        prog="neu-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare a segmentation against a reference labeling.\n\n"
            "Produces the standard metrics (variation of information with its split/merge\n"
            "decomposition, adapted Rand, IoU matching, fragmentation, completeness,\n"
            "covering) and a ranked, located list of the disagreements behind them.\n\n"
            "Because a reference is not always right, the disagreement table round-trips:\n"
            "it goes out with a blank `verdict` column, you fill it in while looking at the\n"
            "voxels, and passing it back with --adjudicated reports a second score that\n"
            "excludes the pairs you judged to be reference defects — alongside the raw one,\n"
            "never instead of it."),
        epilog=_EPILOG)
    p.add_argument("--version", action="version", version=f"neu-eval {__version__}")
    p.add_argument("--store-logs", action="store_true",
                   help="do not filter the store's benign credential-provider chatter")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    q = sub.add_parser(
        "compare", help="score a segmentation against a reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Read two labelings of the same physical box and score them.\n\n"
            "The segmentation is read with `crop=<the reference piece>`, so the two are\n"
            "registered by construction — the reference's physical box is converted using\n"
            "the segmentation level's own voxel size and origin, never an assumed 2**level\n"
            "factor, because real pyramids are anisotropic."),
        epilog=_EPILOG)
    q.add_argument("--reference", required=True, metavar="SRC",
                   help="the reference labeling: PATH or PATH:/DATASET, or a source URL")
    q.add_argument("--segmentation", required=True, metavar="SRC",
                   help="the labeling being assessed, read registered to the reference")
    q.add_argument("--level", type=int, default=0,
                   help="level of the segmentation to read (default: 0)")
    q.add_argument("--reference-level", type=int, default=0,
                   help="level of the reference, for a multiscale reference (default: 0)")
    q.add_argument("--all-datasets", action="store_true",
                   help="score every array in the reference container, and pool them")
    q.add_argument("--out", metavar="DIR", default=".",
                   help="where to write the results (default: the working directory)")
    q.add_argument("--labels", default="gt,seg", metavar="A,B",
                   help="names for the two sides, used throughout the output "
                        "(default: gt,seg)")
    q.add_argument("--ignore-reference", default="0", metavar="IDS",
                   help="comma-separated label values to drop on the reference side; 0 is "
                        "unannotated, so it is the default. '' keeps everything")
    q.add_argument("--ignore-segmentation", default="", metavar="IDS",
                   help="label values to drop on the segmentation side. Empty by default: "
                        "a 0 there means 'assigned to no body' inside a region somebody "
                        "did annotate, which is an error and not an absence")
    q.add_argument("--top", type=int, default=200,
                   help="how many disagreements to report (default: 200; 0 for all)")
    q.add_argument("--min-frac", type=float, default=0.05,
                   help="a pair must hold this fraction of either side to count as "
                        "structural (default: 0.05). Below it is speck overlap, and these "
                        "volumes have a great deal of that")
    q.add_argument("--min-voxels", type=int, default=1,
                   help="a pair must hold at least this many voxels (default: 1)")
    q.add_argument("--rank", choices=("severity", "fraction"), default="severity",
                   help="order the disagreements by VOI contribution (default) or by how "
                        "much of a body or segment the pair accounts for. Severity "
                        "inherits VOI's size weighting, so a large body losing a sliver "
                        "outranks a small one cut in half; --rank fraction is the one to "
                        "use when the list matters more than the metric")
    q.add_argument("--no-locate", action="store_true",
                   help="skip the per-pair distance transform, so no coordinates. Faster "
                        "when only the scalars are wanted")
    q.add_argument("--adjudicated", metavar="CSV",
                   help="a disagreement table with its verdict column filled in; adds the "
                        "adjudicated score beside the raw one")
    q.add_argument("--voxel-size", metavar="Z,Y,X",
                   help="voxel size in nm for a reference that records none")
    q.add_argument("--dtype", default="uint64",
                   help="cast both sides to this on read (default: uint64), so crops "
                        "exported by different tools compare")
    q.add_argument("--no-scatter-check", action="store_true",
                   help="skip the check for reference labels scattered far beyond their "
                        "own size, which is what an un-relabelled multi-crop reference "
                        "looks like")
    q.set_defaults(func=cmd_compare)

    h = sub.add_parser("help", help="show help for a command")
    h.add_argument("topic", nargs="?")
    h.set_defaults(func=cmd_help)
    return p


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


def _ids(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(",") if v.strip() != "")


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(",") if v.strip() != "")


def _paired_source(src: str, name: str | None) -> str:
    """``src`` addressed to dataset ``name`` when it is a container that has one.

    Under ``--all-datasets`` the reference is a container of crops, and the segmentation is
    usually one volume that every crop is read out of — so it is passed through unchanged.
    But it can *also* be a container of matching crops (comparing two versions of an
    annotation, say), and then each reference crop has to be paired with its namesake
    instead of with the whole file, which raises "needs one array and this is a container
    of 13".

    Detected rather than flagged: an explicit ``:/dataset`` is left alone, a path that is
    not a readable HDF5 file is left alone, and a container is addressed only if it actually
    holds that name.
    """
    if name is None or ":/" in src:
        return src
    try:
        import h5py

        if not Path(src).is_file() or not h5py.is_hdf5(src):
            return src
        with h5py.File(src, "r") as handle:
            if name in handle and hasattr(handle[name], "shape"):
                return f"{src}:/{name}"
    except (OSError, ImportError):
        return src
    return src


def cmd_compare(args) -> int:
    """Score one reference against one segmentation, or every crop in a container."""
    import h5py
    from neu_vol import read_piece

    from . import compare as compare_mod, overlap, tables

    labels = tuple(x.strip() for x in args.labels.split(","))
    if len(labels) != 2:
        raise SystemExit(f"--labels needs two comma-separated names, got {args.labels!r}")
    out = Path(args.out)
    top = None if args.top == 0 else args.top
    voxel_size = _floats(args.voxel_size) if args.voxel_size else None

    read_kwargs = dict(kind="segmentation", dtype=args.dtype)
    if voxel_size:
        read_kwargs["voxel_size"] = voxel_size

    datasets: list[str | None]
    if args.all_datasets:
        path = args.reference.split(":/")[0]
        with h5py.File(path, "r") as handle:
            datasets = sorted(k for k, v in handle.items() if hasattr(v, "shape"))
        if not datasets:
            raise SystemExit(f"{path} holds no arrays")
        logger.info("scoring %d datasets from %s", len(datasets), path)
    else:
        datasets = [None]

    verdicts = None
    if args.adjudicated:
        _, verdicts = tables.read_disagreements(args.adjudicated)
        logger.info("read %d verdicts from %s", len(verdicts), args.adjudicated)

    summaries, all_rows, contingencies = [], [], []
    for name in datasets:
        src = args.reference if name is None else f"{args.reference.split(':/')[0]}:/{name}"
        reference = read_piece(src, level=args.reference_level, **read_kwargs)
        segmentation = read_piece(_paired_source(args.segmentation, name),
                                  level=args.level, crop=reference, **read_kwargs)
        report = compare_mod.compare(
            reference, segmentation, labels=labels,
            ignore_a=_ids(args.ignore_reference), ignore_b=_ids(args.ignore_segmentation),
            top=top, min_frac=args.min_frac, min_voxels=args.min_voxels,
            rank=args.rank, locate=not args.no_locate, verdicts=verdicts,
            check_scattered=not args.no_scatter_check)

        tag = name or reference.name or "piece"
        print(f"\n== {tag}")
        print(report.headline())
        for warning in report.warnings:
            print(f"  ! {warning}")

        row = {"piece": tag, **report.summary}
        if report.adjudicated_summary is not None:
            row.update({f"adj_{k}": v for k, v in report.adjudicated_summary.items()})
        summaries.append(row)
        all_rows.extend({"piece": tag, **r} for r in report.rows)
        contingencies.append(report.contingency)

    if len(contingencies) > 1:
        # Separately annotated crops have independent label spaces, so this is `pool`, not
        # a sum — see neu_eval.overlap.pool for why the difference is not cosmetic.
        pooled = overlap.pool(contingencies)
        pooled_summary = compare_mod.summarize(pooled)
        # No adjudicated column for the pooled row: `pool` renumbers, so a verdict keyed on
        # a real body id has nothing to match. Adjudicate per piece and read those rows.
        print(f"\n== pooled over {len(contingencies)} pieces "
              f"(ids renumbered, so scalars only)")
        print(f"{labels[0]} vs {labels[1]}: VOI {pooled_summary['voi']:.4f} "
              f"(split {pooled_summary['voi_split']:.4f}, "
              f"merge {pooled_summary['voi_merge']:.4f}), "
              f"Rand err {pooled_summary['rand_error']:.4f}, "
              f"covering {pooled_summary['covering']:.4f} | "
              f"{pooled_summary['n_scored']:,} voxels scored "
              f"({pooled_summary['frac_scored']:.1%})")
        summaries.append({"piece": "POOLED", **pooled_summary})

    written = [tables.write_summary(summaries, out / "summary.parquet")]
    if all_rows:
        written.append(tables.write_disagreements(
            all_rows, out / "disagreements.csv", labels=labels, verdicts=verdicts))
        if not args.no_locate:
            written.append(tables.annotation_csv(
                all_rows, out / "annotations.csv", labels=labels))
    if len(contingencies) == 1:
        written.append(tables.write_table(
            tables.per_label_frame(contingencies[0]), out / "per_label.parquet"))

    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    if all_rows and not args.no_locate:
        print(f"\nStep through them:")
        print(f"  neu-glance annotate --points {out / 'annotations.csv'} --nm "
              f"--volume {args.reference.split(':/')[0]} \\")
        print(f"      --name disagreements --format layer --out {out / 'layer.json'}")
        print(f"  neu-glance gen --seg {args.reference.split(':/')[0]} "
              f"--seg {args.segmentation} --layer {out / 'layer.json'} --format url")
        print(f"Adjudicate:         fill in the verdict column of "
              f"{out / 'disagreements.csv'}, then re-run with --adjudicated")
    return 0


def cmd_help(args) -> int:
    parser = build_parser()
    if not args.topic:
        parser.print_help()
        return 0
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    choices = actions[0].choices if actions else {}
    if args.topic not in choices:
        print(f"no such command: {args.topic}. Try: {', '.join(sorted(choices))}")
        return 2
    choices[args.topic].print_help()
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    from neu_vol.logs import quiet_store_logs

    with quiet_store_logs(not args.store_logs):
        return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
