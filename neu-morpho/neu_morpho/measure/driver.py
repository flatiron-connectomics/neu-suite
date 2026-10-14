"""Block-mapped per-body volume over a published segmentation.

Fills `voxel_count` / `volume_nm3` in a `MetricsDB`, block by block, resumably; the
pure per-block work is in `sweep.py`.

- **No occupancy or ROI prefilter by default**: a compartment-mask ROI omits somata
  outside it, and ``V/L`` would divide a truncated volume by complete cable. An empty
  block costs about a tenth of a dense one, so reading everything is affordable.
- **Counts, not bboxes** (see `MetricsDB.apply_counts_block`).
- **Per-block retry of transient errors**; a persistent one still stops the run.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map, iter_blocks

from ..metrics_db import MetricsDB
from .. import roi as _roi
from .sweep import DEFAULT_BLOCK, count_labels


def _block_key(index) -> str:
    return "_".join(str(int(i)) for i in index)


def resolve_keep(sources: Iterable[str] | None) -> set[int] | None:
    """Union the body ids named by each source; ``None`` for "record every label".

    A source is a file of ids (one per line, or a first CSV column), or a
    ``neuroglancer_segment_properties`` source (a directory or URL with an ``info``) whose
    ``ids`` are taken; ``os.path.isfile`` tells them apart. Several may be given, as a
    dataset may split its properties across sources. A properties source is usually a
    better cohort gate than a size threshold: only named bodies have properties.
    """
    if not sources:
        return None
    import os

    from ..allowlist import load_allowlist

    out: set[int] = set()
    for src in sources:
        src = str(src)
        if os.path.isfile(src):
            got = load_allowlist(src)
            if got:
                out |= got
            continue
        from neu_vol import location
        info = location.read_json(src, "info")
        if info is None:
            raise ValueError(f"--keep {src}: not a file, and no 'info' found there")
        kind = info.get("@type")
        if kind != "neuroglancer_segment_properties":
            raise ValueError(
                f"--keep {src} is a '{kind}', not a neuroglancer_segment_properties "
                f"source. A segmentation volume's own info carries no id list.")
        out |= {int(i) for i in info["inline"]["ids"]}
    return out


def _count_block(block, *, seg_spec: dict, keep: frozenset | None,
                 background: int = 0, attempts: int = 5) -> tuple:
    """``(key, {body: voxel_count})`` for one block. The whole block is the retry unit."""
    from neu_vol.backends.base import open_backend
    from neu_vol.retry import with_retry

    key = _block_key(block.index)

    def read_and_count():
        seg = open_backend(seg_spec).read_region(block.region)
        return count_labels(seg, background=background)

    counts = with_retry(read_and_count, attempts=attempts, label=f"block {key}")
    if keep is not None:
        counts = {k: v for k, v in counts.items() if k in keep}
    return (key, counts)


def sweep_volumes(
    seg_spec: dict,
    db_path: str,
    *,
    voxel_size: Sequence[float],
    block: int | Sequence[int] = DEFAULT_BLOCK,
    keep: Iterable[int] | None = None,
    roi: Sequence[int] | str | None = None,
    background: int = 0,
    attempts: int = 5,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Count each body's voxels across ``seg_spec`` and accumulate into ``db_path``.

    ``voxel_size`` (nm, zyx) is the read level's own voxel size, from
    ``neu_vol.read_scales`` (pyramids may be anisotropic, so never ``2**level``). ``keep``
    restricts which bodies are recorded, not which blocks are read, so its rows are
    complete. Refuses a DB that already holds an index scan: both fill ``voxel_count``.
    """
    shape = tuple(int(s) for s in np.atleast_1d(block)) if not np.isscalar(block) else None
    block_shape = shape or (int(block),) * 3
    keep = None if keep is None else frozenset(int(b) for b in keep)
    voxel_volume = float(np.prod(voxel_size))

    from neu_vol.backends.base import open_backend
    vol_shape = open_backend(seg_spec).shape
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), vol_shape)
    blocks = _roi.filter_blocks(iter_blocks(vol_shape, block_shape), roi)

    db = MetricsDB(db_path)
    try:
        n_indexed = db.con.execute("SELECT COUNT(*) FROM index_progress").fetchone()[0]
        if n_indexed:
            raise ValueError(
                f"{db_path} already holds an index scan ({n_indexed} blocks). Both it and "
                f"this sweep accumulate into voxel_count, so running both would double "
                f"every count. Use a separate DB, or read the counts that are already there.")
        done = db.done_sweep_blocks() if resume else db.reset_sweep()
        todo = [b for b in blocks if _block_key(b.index) not in done]
        # Before dispatch, so a reader has a denominator mid-run.
        db.record_stage_meta("sweep", total=len(blocks), block_shape=block_shape,
                             voxel_size=voxel_size,
                             n_keep=None if keep is None else len(keep))

        worker = functools.partial(_count_block, seg_spec=seg_spec, keep=keep,
                                   background=background, attempts=attempts)

        def on_result(batch):                     # driver-side, single writer
            for key, counts in batch:
                db.apply_counts_block(key, counts, voxel_volume)

        block_map(todo, worker, client=client, npartitions=npartitions,
                  on_result=on_result)
        n_bodies = db.con.execute(
            "SELECT COUNT(*) FROM bodies WHERE voxel_count>0").fetchone()[0]
    finally:
        db.close()
    return {"db": db_path, "n_blocks": len(blocks), "processed": len(todo),
            "n_bodies": n_bodies, "block_shape": block_shape,
            "n_kept": None if keep is None else len(keep)}
