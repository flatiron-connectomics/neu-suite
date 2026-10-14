"""Cable length and topology for many bodies, from published skeletons.

The per-body counterpart of `driver.sweep_volumes`:

- **Tasks are batches of bodies**: one small read per body would be dominated by dispatch
  overhead. Within a batch, reads run on threads and share one opened store.
- **Failures are isolated**: one unreadable skeleton says nothing about the others.
- **Absence is not failure**: a body with no published skeleton is recorded ``absent`` and
  never retried; one that raised is ``failed`` and retried next run.

Without radii, ``max_radius_nm`` is NaN while cable length and topology still land.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map

from ..metrics_db import MetricsDB

#: Bodies per task. Large enough to amortise dispatch, small enough that a batch is a
#: useful resume granule and that one slow body cannot stall a worker for long.
DEFAULT_BATCH = 200

#: Concurrent reads within a batch. I/O-bound, so well above the core count.
DEFAULT_THREADS = 8

#: Columns written per body: the same four a skeleton stage writes into a run, so the two
#: are directly comparable. Richer rows belong in an export table, not this DB.
_COLUMNS = ("cable_length_nm", "n_branches", "n_tips", "max_radius_nm")


def skeleton_columns(skeleton) -> dict:
    """The four per-body skeleton columns of the metrics DB, from a :class:`neu_lib.Skeleton`.

    The one definition, shared by this pass over published skeletons and by whatever
    writes skeletons into a run (``neu-skel``). ``max_radius_nm`` is NaN without radii.
    """
    from .metrics import cable_length_nm, max_radius_nm, topology

    top = topology(skeleton)
    return {"cable_length_nm": cable_length_nm(skeleton), "n_branches": top["n_branches"],
            "n_tips": top["n_tips"], "max_radius_nm": max_radius_nm(skeleton)}


def _measure_one(body_id: int, *, volume: str, skeleton_dir: str,
                 require_radii: bool) -> tuple:
    """``(body_id, status, payload)`` for one body. Never raises."""
    from neu_lib import Skeleton

    from neu_vol.readback import read_body_skeleton
    try:
        got = read_body_skeleton(volume, body_id, skeleton_dir,
                                 require_radii=require_radii)
        if got is None:
            return (body_id, "absent", None)
        verts_xyz, edges, radii = got
        # The reader returns stored xyz; everything here is zyx.
        skel = Skeleton(vertices_zyx_nm=np.asarray(verts_xyz, dtype=float)[:, ::-1],
                        edges=edges, radii_nm=radii, name=str(body_id))
        return (body_id, "written", skeleton_columns(skel))
    except Exception as exc:                                   # noqa: BLE001 - isolated
        return (body_id, "failed", f"{type(exc).__name__}: {exc}"[:400])


def _measure_batch(bodies: Sequence[int], *, volume: str, skeleton_dir: str,
                   require_radii: bool, threads: int) -> list[tuple]:
    from concurrent.futures import ThreadPoolExecutor

    fn = functools.partial(_measure_one, volume=volume, skeleton_dir=skeleton_dir,
                           require_radii=require_radii)
    if threads <= 1:
        return [fn(b) for b in bodies]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(fn, bodies))


def sweep_skeletons(
    volume: str,
    db_path: str,
    *,
    bodies: Iterable[int],
    skeleton_dir: str = "skeleton",
    require_radii: bool = False,
    batch: int = DEFAULT_BATCH,
    threads: int = DEFAULT_THREADS,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
    retry_failed: bool = True,
) -> dict:
    """Measure each body's published skeleton into ``db_path``.

    ``resume`` skips bodies already recorded ``written`` or ``absent``; ``retry_failed``
    (default) attempts ``failed`` ones again.
    """
    bodies = [int(b) for b in bodies]
    db = MetricsDB(db_path)
    try:
        done = db.done_skel_bodies(include_failed=not retry_failed) if resume else \
            db.reset_skel()
        todo = [b for b in bodies if b not in done]
        db.record_stage_meta("skel", total=len(bodies), n_keep=len(bodies))

        batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        worker = functools.partial(_measure_batch, volume=volume,
                                   skeleton_dir=skeleton_dir,
                                   require_radii=require_radii, threads=threads)

        counts = {"written": 0, "absent": 0, "failed": 0}

        def on_result(results):                    # driver-side, single writer
            for batch_out in results:
                rows = []
                for body_id, status, payload in batch_out:
                    counts[status] = counts.get(status, 0) + 1
                    rows.append((body_id, status,
                                 None if status != "failed" else payload,
                                 payload if status == "written" else None))
                db.apply_skel_batch(rows)

        block_map(batches, worker, client=client, npartitions=npartitions,
                  on_result=on_result)
    finally:
        db.close()
    return {"db": db_path, "n_bodies": len(bodies), "processed": len(todo),
            "skeleton_dir": skeleton_dir, **counts}
