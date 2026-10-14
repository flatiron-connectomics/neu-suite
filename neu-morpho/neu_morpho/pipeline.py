"""The ``neu-morpho run`` pipeline as a library call: seg -> index -> allowlist -> mesh -> skel.

:func:`plan` validates the arguments against the source's own pyramid and counts the
blocks each stage will process; :func:`run` executes a plan, on a dask ``client`` or
serially in this process. Starting the cluster is the caller's, between the two, so a
dry run never starts one::

    p = pipeline.plan(src, dst, work_dir=work, roi=(0, 0, 0, 512, 2048, 2048), roi_scale=2)
    summary = pipeline.run(p)               # serial; pass client= to run on dask

``dst`` is the published volume (labels, with meshes and skeletons inside it) and may
be an object store; ``work_dir`` is a filesystem path for the fragments, the metrics DB
and the manifests.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

log = logging.getLogger("neu-morpho")

#: Above this many indexed bodies, an unfiltered mesh/skel run logs a warning.
UNFILTERED_WARN_BODIES = 100_000

#: One line per stage, and the only place they are described: ``--stages`` help, the
#: published CLI reference and the docs cheat sheet are built from this.
STAGE_DOC = {
    "seg": "copy the ROI's labels out as a precomputed volume, so the meshes and "
           "skeletons can be viewed against the segmentation they came from. Optional, "
           "and off by default — it is pure I/O and usually the most expensive stage.",
    "index": "scan the volume once and record every body's bounding box and voxel count "
             "in a SQLite metrics DB. REQUIRED FIRST unless the DB already exists: mesh "
             "and skel crop each body by its bbox and cannot ask the volume for one, and "
             "the size filter that builds the allowlist reads the counts.",
    "mesh": "multi-resolution Draco meshes, per body. Two phases — fragments per "
            "(body, block), then one mesh assembled per body — so a re-run with only "
            "this stage reuses the fragments already on disk.",
    "skel": "skeletons, per body (the neutu tracer by default), in the same two phases "
            "and with its own manifest. Reads bounding boxes from the metrics DB.",
}
STAGES = tuple(STAGE_DOC)
DEFAULT_STAGES = ("index", "mesh", "skel")


@dataclass
class Plan:
    """What :func:`run` will do. Built by :func:`plan`; every scale and ROI resolved."""

    src: str
    dst: str
    work_dir: str
    stages: tuple[str, ...]
    block: tuple[int, int, int]
    scales: dict[str, Any]                  #: ``index`` / ``mesh`` / ``skel`` -> ScaleInfo
    finest: Any                             #: the source's level 0
    rois: dict[str, tuple | None]           #: each stage's ROI in its own scale's voxels
    roi: tuple | None
    roi_scale: int | None
    seg_scales: list[int] | None
    occupancy_spec: dict | None
    occupancy_voxel_size: tuple | None
    occupancy_scale: int | None
    occupancy_dilate: int
    planned_blocks: dict[str, int]
    allowlist: str | None
    min_voxels: int
    limit_bodies: int | None
    metrics_db: bool
    tracer: str
    neutu_cost: str
    seg_encoding: str
    max_consecutive_failures: int
    resume: bool

    @property
    def db_path(self) -> str | None:
        return f"{self.work_dir}/metrics.db" if self.metrics_db else None

    @property
    def allowlist_path(self) -> str:
        return self.allowlist or f"{self.work_dir}/allowlist.csv"


def ng_source(volume_dir: str) -> str:
    """The neuroglancer source URL for a volume — local paths need a file:// prefix."""
    from neu_vol import is_local

    return (f"precomputed://file://{volume_dir}" if is_local(volume_dir)
            else f"precomputed://{volume_dir}")


def _blocks_in(shape, block, roi, voxel_size=None, occ=None) -> int:
    """How many blocks a stage will process, after the ROI and occupancy filters both."""
    from blockrun import iter_blocks

    from .occupancy import occupied_blocks
    from .roi import clip_to_shape, filter_blocks

    blocks = filter_blocks(iter_blocks(shape, block), clip_to_shape(roi, shape))
    if occ is None:
        return len(blocks)
    arr, occ_voxel, dilate = occ
    grid = tuple(-(-shape[a] // block[a]) for a in range(3))
    keep = occupied_blocks(arr, occ_voxel_size=occ_voxel, mesh_voxel_size=voxel_size,
                           block_shape=block, grid_shape=grid, allowlist=None,
                           dilate=dilate)
    return sum(1 for b in blocks if b.index in keep)


def plan(src: str, dst: str, *, work_dir: str | None = None,
         stages: Sequence[str] = DEFAULT_STAGES,
         block: Sequence[int] = (256, 256, 256),
         index_scale: int = 2, mesh_scale: int = 2, skel_scale: int = 2,
         roi: Sequence[int] | None = None, roi_scale: int | None = None,
         seg_scales: Sequence[int] | None = None,
         occupancy_scale: int | None = 5, occupancy_dilate: int = 1,
         allowlist: str | None = None, min_voxels: int = 0,
         limit_bodies: int | None = None, metrics_db: bool = True,
         tracer: str | None = None, neutu_cost: str | None = None,
         seg_encoding: str = "compressed_segmentation",
         max_consecutive_failures: int = 10, resume: bool = True) -> Plan:
    """Resolve and check a run against ``src``'s pyramid; logs the plan. Writes nothing.

    ``roi`` is ``(z0, y0, x0, z1, y1, x1)`` in the voxels of ``roi_scale``, which is
    required with it: the same six numbers name a different box at every level. Scales
    are pyramid indices, their voxel sizes read from the source. ``occupancy_scale``
    ``None`` or negative disables the empty-block prefilter; keep ``occupancy_dilate``
    at 1 or more, since a coarse scale misses sparse blocks. ``seg_scales`` ``None``
    copies every scale in the seg stage. ``tracer`` / ``neutu_cost`` default to
    :class:`~neu_morpho.config.SkeletonConfig`'s.

    Raises ``ValueError`` for an argument the source or the other arguments rule out.
    """
    from neu_vol import describe_scales, is_local, read_scales, scale_spec

    from .config import SkeletonConfig
    from .roi import scale_roi

    dst = dst.rstrip("/")
    if work_dir:
        work = work_dir.rstrip("/")
    elif is_local(dst):
        # A local dst's parent: <run>/segmentation beside <run>/chunked and metrics.db.
        work = os.path.dirname(os.path.abspath(dst))
    else:
        raise ValueError(
            f"work_dir is required when dst is remote ({dst}). It holds the stage-1 "
            "fragments, the sqlite metrics DB and the JSONL manifests, none of which "
            "work over an object store.")
    if not is_local(work):
        raise ValueError(f"work_dir must be a filesystem path, got {work}")

    stages = tuple(stages)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; pick from {list(STAGES)}")
    if "index" in stages and not metrics_db:
        raise ValueError("metrics_db=False is incompatible with the index stage, whose "
                         "only output is that DB")
    if roi is not None and roi_scale is None:
        raise ValueError("roi_scale is required with roi: the same six values name a "
                         "different box at every pyramid level")
    if roi_scale is not None and roi is None:
        raise ValueError("roi_scale has no effect without roi")

    scales = read_scales(src)
    for name, idx in (("index", index_scale), ("mesh", mesh_scale), ("skel", skel_scale),
                      ("roi", roi_scale)):
        if idx is not None and not 0 <= idx < len(scales):
            raise ValueError(f"{name}_scale {idx} out of range (source has {len(scales)})")
    by_stage = {"index": scales[index_scale], "mesh": scales[mesh_scale],
                "skel": scales[skel_scale]}

    # One physical box everywhere: the ROI converted into each stage's own voxels.
    roi = tuple(int(v) for v in roi) if roi is not None else None
    rois = {}
    for name, s in by_stage.items():
        if roi is None:
            rois[name] = None
            continue
        base = scales[roi_scale]
        rois[name] = scale_roi(roi, tuple(base.voxel_size[a] / s.voxel_size[a]
                                          for a in range(3)))

    occ_spec = occ_voxel = occ = None
    if occupancy_scale is not None and occupancy_scale >= 0:
        if not 0 <= occupancy_scale < len(scales):
            raise ValueError(f"occupancy_scale {occupancy_scale} out of range")
        o = scales[occupancy_scale]
        occ_spec, occ_voxel = scale_spec(src, o.index), o.voxel_size
        from neu_vol.backends.base import open_backend

        be = open_backend(occ_spec)
        occ = (be.read_region(tuple(slice(0, x) for x in be.shape)), occ_voxel,
               occupancy_dilate)
    else:
        occupancy_scale = None

    block = tuple(int(v) for v in block)
    planned = {name: _blocks_in(s.shape, block, rois[name], s.voxel_size, occ)
               for name, s in by_stage.items()}
    defaults = SkeletonConfig()
    p = Plan(src=src, dst=dst, work_dir=work, stages=stages, block=block,
             scales=by_stage, finest=scales[0], rois=rois, roi=roi, roi_scale=roi_scale,
             seg_scales=list(seg_scales) if seg_scales is not None else None,
             occupancy_spec=occ_spec, occupancy_voxel_size=occ_voxel,
             occupancy_scale=occupancy_scale, occupancy_dilate=occupancy_dilate,
             planned_blocks=planned, allowlist=allowlist, min_voxels=min_voxels,
             limit_bodies=limit_bodies, metrics_db=metrics_db,
             tracer=tracer or defaults.tracer, neutu_cost=neutu_cost or defaults.neutu_cost,
             seg_encoding=seg_encoding, max_consecutive_failures=max_consecutive_failures,
             resume=resume)

    log.info("source pyramid:\n%s", describe_scales(src))
    for name, s in by_stage.items():
        log.info("%-5s  scale %d  voxel %s nm  shape %s  -> %d blocks",
                 name, s.index, s.voxel_size, s.shape, planned[name])
    log.info("stages=%s roi=%s occupancy=%s", list(stages), roi,
             f"scale {occupancy_scale} dilate {occupancy_dilate}" if occ else "off")
    # The tracer decides what the skeletons are, so it belongs in the log too.
    if "skel" in stages:
        log.info("tracer        = %s%s", p.tracer,
                 f" (cost={p.neutu_cost})" if p.tracer == "neutu" else "")
    log.info("dst (volume)  = %s", dst)
    log.info("work dir      = %s", work)
    return p


def _write_run_plan(p: Plan, command: str) -> None:
    """``run_plan.json``: the denominators ``neu-morpho progress`` reports against."""
    from .ops.export_roi_seg import DEFAULT_COPY_BLOCK, scale_cost

    mesh = p.scales["mesh"]
    seg = [{"scale": c["scale"], "shape": list(c["shape"]),
            "blocks": math.prod(-(-c["shape"][a] // DEFAULT_COPY_BLOCK[a]) for a in range(3))}
           for c in scale_cost(p.src, p.rois["mesh"], mesh.voxel_size,
                               block_shape=p.block, scale_indices=p.seg_scales)
           ] if "seg" in p.stages else []
    with open(f"{p.work_dir}/run_plan.json", "w") as f:
        json.dump({
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": command, "dst": p.dst, "work_dir": p.work_dir,
            "stages": list(p.stages), "block_shape": list(p.block),
            "roi": list(p.roi) if p.roi else None,
            "roi_scale": p.roi_scale,                 # the ROI is meaningless without it
            "scales": {k: s.index for k, s in p.scales.items()},
            "tracer": p.tracer, "neutu_cost": p.neutu_cost,
            "planned_blocks": p.planned_blocks, "seg_scales": seg,
        }, f, indent=2)


def run(p: Plan, *, client: Any = None, command: str | None = None) -> dict:
    """Execute ``p``'s stages; returns the run summary, also written to ``run_summary.json``.

    ``client`` is a dask client, or ``None`` to run serially in this process. Per-body
    failures do not stop a stage; they are counted in ``n_failed_bodies`` and retried by
    the next run. Afterwards the volume's ``info`` is pointed at whichever of the mesh and
    skeleton subresources exist, so one neuroglancer layer carries all three.
    """
    from neu_vol import exists, scale_spec

    from .config import MeshConfig, OutputConfig, SkeletonConfig
    from .metrics_db import MetricsDB
    from .ops import export_roi_seg, index_segments, meshify, skeletonize_segments
    from .ops.export_roi_seg import scale_cost
    from .precomputed import link_subresources

    command = command if command is not None else " ".join(sys.argv)
    work, src, stages = p.work_dir, p.src, p.stages
    idx_s, mesh_s, skel_s = p.scales["index"], p.scales["mesh"], p.scales["skel"]
    db_path, allowlist_path = p.db_path, p.allowlist_path
    out = OutputConfig(dst=p.dst, work_dir=work)

    os.makedirs(work, exist_ok=True)
    _write_run_plan(p, command)
    t_start = time.time()
    mesh_cfg = MeshConfig(mesh_scale=mesh_s.index, block_shape=p.block)
    skel_cfg = SkeletonConfig(skeleton_scale=skel_s.index, block_shape=p.block,
                              anisotropy=skel_s.voxel_size,
                              tracer=p.tracer, neutu_cost=p.neutu_cost)
    occupancy = dict(occupancy_spec=p.occupancy_spec,
                     occupancy_voxel_size=p.occupancy_voxel_size,
                     occupancy_dilate=p.occupancy_dilate)
    summaries: dict[str, dict] = {}
    timing: dict[str, float] = {}

    if "seg" in stages:
        t = time.time()
        for c in scale_cost(src, p.rois["mesh"], mesh_s.voxel_size,
                            block_shape=p.block, scale_indices=p.seg_scales):
            log.info("  seg scale %d (%3.0f nm): %-22s %6.2f Gvox  ~%6.2f GB raw",
                     c["scale"], c["voxel_size"][0], str(c["shape"]),
                     c["n_voxels"] / 1e9, c["raw_gb"])
        summaries["seg"] = export_roi_seg(
            src, out.volume_dir(), roi=p.rois["mesh"],
            roi_voxel_size=mesh_s.voxel_size, scale_indices=p.seg_scales,
            block_shape=p.block, encoding=p.seg_encoding,
            progress_path=f"{work}/progress.seg.jsonl", client=client, resume=p.resume)
        timing["seg"] = (time.time() - t) / 60
        log.info("seg: %d scales, %.2f Gvox -> %s  (%.1f min)",
                 len(summaries["seg"]["scales"]), summaries["seg"]["n_voxels_total"] / 1e9,
                 summaries["seg"]["out_dir"], timing["seg"])

    if "index" in stages:
        from .allowlist import load_allowlist

        t = time.time()
        # An explicit allowlist restricts which bodies get RECORDED; every block is still
        # read in full. What it saves is the driver-side upsert per speck.
        summaries["index"] = index_segments(
            scale_spec(src, idx_s.index), db_path,
            scan_voxel_size=idx_s.voxel_size, scan_scale=idx_s.index,
            fullres_factor=idx_s.factor_from(p.finest), block_shape=p.block,
            roi=p.rois["index"], keep=load_allowlist(p.allowlist) if p.allowlist else None,
            client=client, resume=p.resume)
        timing["index"] = (time.time() - t) / 60
        log.info("index: %s  (%.1f min)", summaries["index"], timing["index"])

    # Size-filter the indexed bodies into an allowlist the later stages honour.
    if db_path and p.allowlist is None and (p.min_voxels or p.limit_bodies):
        db = MetricsDB(db_path)
        n = db.write_allowlist(allowlist_path, min_voxels=p.min_voxels,
                               limit=p.limit_bodies)
        db.close()
        log.info("allowlist: %d bodies -> %s", n, allowlist_path)
    allow = allowlist_path if os.path.exists(allowlist_path) else None
    if allow is None and db_path and os.path.exists(db_path) and (
            "mesh" in stages or "skel" in stages):
        db = MetricsDB(db_path)
        n_all = db.con.execute("SELECT COUNT(*) FROM bodies").fetchone()[0]
        db.close()
        if n_all > UNFILTERED_WARN_BODIES:
            log.warning(
                "no min_voxels or allowlist: meshing/skeletonizing ALL %d indexed bodies. "
                "An over-segmentation is mostly tiny segments, each costing a directory "
                "and a fragment file in the work dir; consider min_voxels (in voxels at "
                "the index scale).", n_all)

    if "mesh" in stages:
        t = time.time()
        summaries["mesh"] = meshify(
            scale_spec(src, mesh_s.index), out, mesh_cfg,
            mesh_voxel_size=mesh_s.voxel_size, allowlist=allow, roi=p.rois["mesh"],
            db_path=db_path, max_consecutive_failures=p.max_consecutive_failures,
            client=client, resume=p.resume, **occupancy)
        timing["mesh"] = (time.time() - t) / 60
        log.info("mesh: %s  (%.1f min)", summaries["mesh"], timing["mesh"])

    if "skel" in stages:
        t = time.time()
        summaries["skel"] = skeletonize_segments(
            scale_spec(src, skel_s.index), out, skel_cfg,
            allowlist=allow, roi=p.rois["skel"], db_path=db_path,
            fusion_stats_path=f"{work}/fusion_stats.jsonl",
            max_consecutive_failures=p.max_consecutive_failures,
            client=client, resume=p.resume, **occupancy)
        timing["skel"] = (time.time() - t) / 60
        log.info("skel: %s  (%.1f min)", summaries["skel"], timing["skel"])

    # After the stages, because the seg stage rewrites info. A subresource is there when
    # its own info is, probed through the kvstore since the volume may be remote.
    vol = out.volume_dir()
    linked: dict[str, str] = {}
    if exists(vol, "info"):
        subs = {k: d for k, d in (("mesh", out.mesh_dir), ("skeletons", out.skeleton_dir))
                if exists(vol, d, "info")}
        if subs:
            linked = link_subresources(vol, **subs)
            log.info("volume info -> %s", linked)
        log.info("neuroglancer source: %s", ng_source(vol))
    else:
        log.warning("no volume at %s — run the seg stage to export the labels, otherwise "
                    "meshes/skeletons have no volume to attach to", vol)

    failed = {stage: s.get("failed_bodies") or [] for stage, s in summaries.items()}
    for stage, bodies in failed.items():
        if bodies:
            log.warning("%s: %d bodies FAILED and were skipped: %s%s", stage, len(bodies),
                        bodies[:10], " ..." if len(bodies) > 10 else "")
            log.warning("%s: tracebacks in %s; re-run to retry them",
                        stage, summaries[stage].get("failures_path"))

    wall = (time.time() - t_start) / 60
    summary = {
        "timing_min": {**{k: round(v, 2) for k, v in timing.items()},
                       "total": round(wall, 2)},
        "started": datetime.fromtimestamp(t_start).astimezone().isoformat(timespec="seconds"),
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "volume": vol,
        "work_dir": work,
        "neuroglancer_source": ng_source(vol),
        "linked_subresources": linked,
        "n_failed_bodies": sum(len(v) for v in failed.values()),
        "stages": {k: dict(v) for k, v in summaries.items()},
    }
    with open(f"{work}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("timing (min): %s", summary["timing_min"])
    log.info("done -> %s", f"{work}/run_summary.json")
    return summary
