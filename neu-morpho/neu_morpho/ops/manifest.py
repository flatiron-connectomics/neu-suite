"""Manifest status policy shared by the ops.

- :func:`is_complete`: resume skips *successful* tasks. blockrun's ``Manifest.is_done``
  tests key presence, so a recorded ``failed`` would read as done and never be retried.
- :func:`guarded`: per-task fault isolation, for **stage-2 workers only**.
- :func:`check_selection`: a resume may not ask stage 1 for bodies it already ran
  without.
- :func:`check_grid`: nor resume stage 1 on a different block grid.

Stage-2 tasks are one body each and independent, so a bad one costs that body and is
recorded. Stage-1 tasks are blocks, and stage 2 aggregates across them: a skipped block
would truncate every body passing through it while the output looks complete. So stage
1 fails fast, and resume makes the relaunch cheap.
"""

from __future__ import annotations

import errno
import traceback
from typing import Any, Callable

FAILED = "failed"

# Exceptions that say "the environment is broken", not "this body is odd" — every
# remaining task would fail the same way, so isolating them just burns the run.
_SYSTEMIC_TYPES = (MemoryError, ImportError)
# OSError is mostly transient (a flaky read is worth isolating); these errnos are
# not — they will not fix themselves before the next task.
_SYSTEMIC_ERRNOS = frozenset({errno.ENOSPC,     # no space left on device
                              errno.EDQUOT,     # over disk quota
                              errno.EROFS})     # read-only filesystem


class StageAborted(RuntimeError):
    """Stage stopped early by the failure breaker (not a per-task failure)."""


class StaleManifest(RuntimeError):
    """A manifest records completed work whose output is no longer there."""


def check_manifest_matches_output(manifest: Any, out_dir: str, *, stage: str,
                                  progress_path: str, resume: bool) -> None:
    """Refuse to resume a manifest that has outlived the data it describes.

    The manifest lives in the work dir and the data possibly in an object store, so
    clearing the destination would leave a manifest saying every task is done: the run
    would write nothing and report success. The probe is the stage's own ``info``, not
    the segmentation's, so meshing into a volume whose labels were never exported is
    not mistaken for a cleared destination.

    Only checked when resuming; ``resume=False`` is an explicit fresh start.
    """
    if not resume:
        return
    recorded = sum(manifest.counts().values())
    if not recorded:
        return                                   # nothing to be stale about
    from ..precomputed import volume_exists

    if volume_exists(out_dir):
        return
    raise StaleManifest(
        f"{progress_path} records {recorded} completed {stage} task(s), but there is "
        f"no 'info' at {out_dir} — the destination looks like it was cleared while "
        f"the manifest survived. Resuming would skip every task and report success "
        f"having written nothing. Re-run with --no-resume to start over (or delete "
        f"the manifest), or point --dst back at the intended destination.")


class SelectionChanged(RuntimeError):
    """A resume asks for bodies that stage 1's completed blocks were not run for."""


def check_selection(manifest: Any, group: str, allow: set[int] | None, *, stage: str,
                    progress_path: str, record: bool) -> None:
    """Refuse a resume for bodies the completed stage-1 blocks were not run for.

    A completed block is never revisited, so a body outside the selection it ran with
    gets no fragments: stage 2 finds nothing for it and the run reports success. A
    later run may ask for that selection or a subset of it. ``record`` is whether this
    run processes stage-1 blocks: its selection is then recorded in the manifest, since
    the remaining blocks see only that (``None`` for every label).
    """
    complete = sum(1 for k in manifest.done_keys(group) if is_complete(manifest, group, k))
    recorded = manifest.meta(group)
    before = recorded.get("selection") if complete else None
    if complete and "selection" not in recorded:
        return                                   # stage 1 ran before selections were kept
    if before is not None:
        if allow is None:
            asked = f"every label, where they were run for {len(before)} bodies"
        else:
            added = sorted(allow - set(before))
            asked = (f"{len(added)} bodies they were not run for (e.g. "
                     f"{', '.join(map(str, added[:5]))})") if added else None
        if asked:
            raise SelectionChanged(
                f"{progress_path}: {complete} {stage} stage-1 blocks are complete, and "
                f"this run asks for {asked}. A complete block is not revisited, so those "
                f"bodies would get no {stage} output while the run reports success. Use "
                f"a new --work-dir for them, or --no-resume to redo stage 1 for this "
                f"selection.")
    if record:
        manifest.record_meta(group, selection=None if allow is None else sorted(allow))


class GridChanged(RuntimeError):
    """A resume would reuse completed stage-1 blocks on a different block grid."""


def grid_record(block_shape, voxel_size, **settings) -> dict:
    """The block grid stage 1 runs on, and any ``settings`` its blocks' output depends on
    (a mesh's smoothing and decimation), as recorded and compared."""
    return {"block_shape": [int(v) for v in block_shape],
            "voxel_size": [float(v) for v in voxel_size], **settings}


def grid_changes(before: dict, now: dict) -> dict:
    """``{field: (before, now)}`` over the fields both records have: a record from before a
    field was recorded is not refused for it."""
    return {k: (before[k], now[k]) for k in now if k in before and before[k] != now[k]}


def grid_mismatch(stage: str, where: str, complete: int, changes: dict) -> str:
    what = ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changes.items())
    return (f"{where}: {complete} {stage} stage-1 blocks are complete, and this run changes "
            f"{what}. Completed blocks are not revisited, so their output would be reused "
            f"as if made this way. Use a new --work-dir, or --no-resume to redo stage 1.")


def check_grid(manifest: Any, group: str, grid: dict, *, stage: str, progress_path: str,
               record: bool) -> None:
    """Refuse a resume whose block grid or stage-1 settings (``grid_record``) differ from
    those its completed blocks ran with; ``record`` (this run processes stage-1 blocks)
    notes them in the manifest. A manifest from before they were recorded is not checked.
    """
    complete = sum(1 for k in manifest.done_keys(group) if is_complete(manifest, group, k))
    before = manifest.meta(group).get("grid")
    changes = grid_changes(before, grid) if complete and before is not None else {}
    if changes:
        raise GridChanged(grid_mismatch(stage, progress_path, complete, changes))
    if record:
        manifest.record_meta(group, grid=grid)


def is_systemic(exc: BaseException) -> bool:
    """Would this error recur for every remaining task?

    ``MemoryError`` counts as systemic even though one enormous body can cause it: a
    process that has hit OOM is not worth trusting for the remaining tasks.
    """
    if isinstance(exc, _SYSTEMIC_TYPES):
        return True
    return isinstance(exc, OSError) and exc.errno in _SYSTEMIC_ERRNOS


def is_complete(manifest: Any, group: str, key: Any) -> bool:
    """Has this task finished *successfully*? (``failed`` records must be retried.)"""
    return manifest.status(group, key) not in (None, FAILED)


def group_counts(manifest: Any, group: str) -> dict[str, int]:
    """Per-group status tallies.

    ``Manifest.counts()`` aggregates every group, conflating blocks with bodies.
    """
    out: dict[str, int] = {}
    for key in manifest.done_keys(group):
        status = manifest.status(group, key)
        out[status] = out.get(status, 0) + 1
    return out


def guarded(worker: Callable[[Any], tuple], key: Any) -> tuple:
    """Run a stage-2 worker, converting an exception into a ``failed`` result.

    Workers return ``(key, status, metrics, info)``; on failure ``info`` carries the
    traceback and a ``systemic`` flag for the breaker. ``KeyboardInterrupt`` and
    ``SystemExit`` are not caught.
    """
    try:
        return worker(key)
    except Exception as exc:                       # noqa: BLE001 - deliberate isolation
        return (key, FAILED, None, {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                    "systemic": is_systemic(exc)})


class FailureBreaker:
    """Stops a stage when failures stop looking like isolated bad bodies.

    Triggers: a systemic exception (:func:`is_systemic`), at once, or
    ``max_consecutive`` failures in a row (in completion order); a success resets the
    streak. It records a reason and :meth:`check` raises, so a batch's successes are
    applied first.
    """

    def __init__(self, max_consecutive: int = 10):
        self.max_consecutive = max_consecutive
        self.consecutive = 0
        self.total = 0
        self.reason: str | None = None

    def success(self) -> None:
        self.consecutive = 0

    def failure(self, key: Any, info: dict | None) -> None:
        info = info or {}
        self.total += 1
        self.consecutive += 1
        if self.reason is not None:
            return
        if info.get("systemic"):
            self.reason = (f"systemic error on task {key} — every remaining task would "
                           f"likely fail the same way: {info.get('error')}")
        elif self.max_consecutive and self.consecutive >= self.max_consecutive:
            self.reason = (f"{self.consecutive} consecutive task failures, most recently "
                           f"on {key}: {info.get('error')}")

    def check(self) -> None:
        """Raise :class:`StageAborted` if a trigger fired."""
        if self.reason is not None:
            raise StageAborted(
                f"{self.reason}. Stopping the stage rather than failing every remaining "
                f"task; {self.total} failure(s) recorded so far. Progress is durable — "
                f"fix the cause and re-run to resume (failed tasks are retried).")


def write_failures(path: str, failures: list[dict]) -> str | None:
    """Dump per-task failures as JSONL (tracebacks would bloat the run summary)."""
    if not failures:
        return None
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")
    return path
