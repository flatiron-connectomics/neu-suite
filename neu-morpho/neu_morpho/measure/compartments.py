"""Per-body volume split by semantic compartment, from two aligned volumes.

Gives each body its own somatic fraction, for the ``variant`` column. Reads the
segmentation and semantic blocks over the same voxel indices and counts
``(body_id, semantic_label)`` pairs:

- **The grids must coincide** (same voxel size and origin at the chosen level, differing
  only in far-edge padding); checked at open time.
- **Label 0 is counted too**, so ``sum over labels == the body's total`` is exact, and
  :func:`verify_compartments` checks it against a volume sweep's totals, which shows
  whether a block filter lost anything.

``non_somatic = total - (nucleus + soma)``.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map, iter_blocks

from ..metrics_db import MetricsDB
from .. import roi as _roi
from .sweep import DEFAULT_BLOCK

#: The semantic labels that make a cell body, by name; :func:`somatic_labels` reads their
#: ids from the source, since datasets number them differently.
SOMATIC_NAMES = ("nucleus", "soma")

#: A body id and a small semantic label pack into one uint64 key, so the joint histogram
#: is one `np.unique`.
_LABEL_BITS = 4
_LABEL_MAX = (1 << _LABEL_BITS) - 1


def semantic_label_names(semantic_url: str) -> dict[int, str]:
    """``{label: name}`` from the semantic volume's own segment_properties."""
    from neu_vol import location

    info = location.read_json(semantic_url, "info")
    sub = info.get("segment_properties")
    if not sub:
        raise ValueError(f"{semantic_url}/info declares no segment_properties, so its "
                         f"label numbering cannot be read and must not be guessed")
    inline = location.read_json(semantic_url, sub, "info")["inline"]
    values = inline["properties"][0]["values"]
    return {int(i): str(v) for i, v in zip(inline["ids"], values)}


def somatic_labels(semantic_url: str, names: Iterable[str] = SOMATIC_NAMES) -> list[int]:
    """The label ids for ``names``, raising if any is missing."""
    by_name = {v: k for k, v in semantic_label_names(semantic_url).items()}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"{semantic_url} has no label named {missing}; it publishes "
                         f"{sorted(by_name)}")
    return [by_name[n] for n in names]


def _block_key(index) -> str:
    return "_".join(str(int(i)) for i in index)


def joint_counts(seg: np.ndarray, sem: np.ndarray,
                 background: int = 0) -> dict[tuple[int, int], int]:
    """``{(body_id, semantic_label): voxels}`` for one block.

    Background *segmentation* is dropped (no body there); background *semantic* is kept,
    because "this body's tissue carries no compartment label" is an answer.
    """
    seg = np.asarray(seg)
    sem = np.asarray(sem)
    if seg.shape != sem.shape:
        raise ValueError(f"segmentation block {seg.shape} and semantic block "
                         f"{sem.shape} differ; the two grids are not aligned")
    nz = seg != background
    if not nz.any():
        return {}
    bodies = seg[nz].astype(np.uint64, copy=False)
    labels = sem[nz].astype(np.uint64, copy=False)
    hi = int(labels.max())
    if hi > _LABEL_MAX:
        raise ValueError(f"semantic label {hi} exceeds {_LABEL_MAX}; the packed key "
                         f"would collide with the body id")
    keys, counts = np.unique((bodies << np.uint64(_LABEL_BITS)) | labels,
                            return_counts=True)
    return {(int(k >> _LABEL_BITS), int(k & _LABEL_MAX)): int(c)
            for k, c in zip(keys, counts)}


def _compartment_block(block, *, seg_spec: dict, sem_spec: dict, sem_shape,
                       keep: frozenset | None, background: int, attempts: int) -> tuple:
    from neu_vol.backends.base import open_backend
    from neu_vol.retry import with_retry

    key = _block_key(block.index)

    def read_and_count():
        seg = open_backend(seg_spec).read_region(block.region)
        # Clip to the semantic volume, then pad, in case its extent differs.
        reg = tuple(slice(s.start, min(s.stop, int(d)))
                    for s, d in zip(block.region, sem_shape))
        sem = open_backend(sem_spec).read_region(reg)
        if sem.shape != seg.shape:
            pad = [(0, a - b) for a, b in zip(seg.shape, sem.shape)]
            sem = np.pad(sem, pad)
        return joint_counts(seg, sem, background=background)

    out = with_retry(read_and_count, attempts=attempts, label=f"block {key}")
    if keep is not None:
        out = {bl: n for bl, n in out.items() if bl[0] in keep}
    return (key, out)


def sweep_compartments(
    seg_spec: dict,
    sem_spec: dict,
    db_path: str,
    *,
    sem_shape: Sequence[int],
    block: int = DEFAULT_BLOCK,
    keep: Iterable[int] | None = None,
    blocks: Iterable[tuple] | None = None,
    roi: Sequence[int] | str | None = None,
    background: int = 0,
    attempts: int = 5,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Count ``(body, semantic_label)`` pairs into ``db_path``'s compartment table.

    ``blocks`` optionally restricts the work to a set of block indices, such as an occupancy
    grid from the segmentation's own coarse level (which, unlike an anatomical ROI, cannot
    omit a soma). :func:`verify_compartments` shows afterwards whether anything was missed.
    """
    keep = None if keep is None else frozenset(int(b) for b in keep)
    block_shape = (int(block),) * 3

    from neu_vol.backends.base import open_backend
    vol_shape = open_backend(seg_spec).shape
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), vol_shape)
    all_blocks = _roi.filter_blocks(iter_blocks(vol_shape, block_shape), roi)
    if blocks is not None:
        want = {tuple(int(v) for v in b) for b in blocks}
        all_blocks = [b for b in all_blocks if tuple(int(v) for v in b.index) in want]

    db = MetricsDB(db_path)
    try:
        done = db.done_compartment_blocks() if resume else db.reset_compartments()
        todo = [b for b in all_blocks if _block_key(b.index) not in done]
        db.record_stage_meta("compartments", total=len(all_blocks),
                            block_shape=block_shape,
                            n_keep=None if keep is None else len(keep))

        worker = functools.partial(_compartment_block, seg_spec=seg_spec,
                                   sem_spec=sem_spec,
                                   sem_shape=tuple(int(v) for v in sem_shape),
                                   keep=keep, background=background, attempts=attempts)

        def on_result(batch):
            for key, counts in batch:
                db.apply_compartment_block(key, counts)

        block_map(todo, worker, client=client, npartitions=npartitions,
                  on_result=on_result)
        n_bodies = db.con.execute(
            "SELECT COUNT(DISTINCT body_id) FROM body_compartments").fetchone()[0]
    finally:
        db.close()
    return {"db": db_path, "n_blocks": len(all_blocks), "processed": len(todo),
            "n_bodies": n_bodies, "n_kept": None if keep is None else len(keep)}


def verify_compartments(db_path: str, *, tolerance: int = 0) -> dict:
    """Check each body's per-label sum against the total a volume sweep recorded.

    A coarse occupancy grid can drop a block holding data; the volume sweep read every
    block, so its totals are ground truth, and this names any body that falls short.
    """
    db = MetricsDB(db_path, read_only=True)
    try:
        rows = db.con.execute(
            "SELECT b.body_id, b.voxel_count, COALESCE(SUM(c.voxel_count), 0) "
            "FROM bodies b LEFT JOIN body_compartments c ON c.body_id = b.body_id "
            "WHERE b.voxel_count > 0 GROUP BY b.body_id").fetchall()
    finally:
        db.close()
    bad = [(int(b), int(t), int(s)) for b, t, s in rows if abs(int(t) - int(s)) > tolerance]
    deficit = sum(t - s for _, t, s in bad)
    total = sum(int(t) for _, t, _ in rows)
    return {"n_bodies": len(rows), "n_mismatched": len(bad),
            "voxels_missing": int(deficit),
            "fraction_missing": (deficit / total) if total else 0.0,
            "worst": sorted(bad, key=lambda r: r[1] - r[2], reverse=True)[:10]}
