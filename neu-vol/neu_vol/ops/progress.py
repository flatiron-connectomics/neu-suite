"""How far a conversion has got, per level: from its run manifest or from storage.

The manifest is what the run *recorded*: a local append-only JSONL, cheap to read even
for a remote volume (1.5 s for 772k records, against tens of seconds of S3 listing).
Storage is what is *there*, counted with one listing per level. They diverge if objects
were deleted underneath, or if a different run wrote the volume.
"""

from __future__ import annotations

import glob
import json
import math
import os

# Statuses meaning the block will not be retried. "failed" is excluded: a resumed run
# retries failures, so counting them as done would overstate progress.
DONE_STATUSES = ("written", "empty", "skipped")


def manifest_counts(path: str) -> dict[int, dict] | None:
    """Per-level progress from a run's manifest, or ``None`` if absent.

    ``{level: {"counts": {status: n}, "total": int|None, "task_shape": [...]|None,
    "grid": (max index + 1 per axis)}}``.

    The unit is the **task**, which may cover many destination chunks; the run records
    each level's task ``total`` before dispatch. ``grid``, the largest block index seen
    per axis, stands in for manifests written without one.

    Statuses are kept apart rather than summed: ``empty`` means a block read as entirely
    the fill value, which is normal for sparse data and an alarm when it is every block.

    A torn final line is normal — the writer may be appending — so it is skipped.
    """
    if not path or not os.path.exists(path):
        return None
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            group = rec.get("group", rec.get("level"))
            if not isinstance(group, int):
                continue
            lv = out.setdefault(group, {"counts": {}, "total": None,
                                        "task_shape": None, "grid": None})
            if "status" not in rec:                      # a meta line: the denominator
                meta = rec.get("meta") or {}
                if isinstance(meta.get("total"), int):
                    lv["total"] = meta["total"]
                if meta.get("task_shape"):
                    lv["task_shape"] = tuple(meta["task_shape"])
                continue
            status = str(rec["status"])
            lv["counts"][status] = lv["counts"].get(status, 0) + 1
            key = rec.get("key", rec.get("index"))
            if isinstance(key, list) and all(isinstance(i, int) for i in key):
                lv["grid"] = (tuple(key) if lv["grid"] is None else
                              tuple(max(a, b) for a, b in zip(lv["grid"], key)))
    for lv in out.values():
        if lv["grid"] is not None:
            lv["grid"] = tuple(i + 1 for i in lv["grid"])
    return out


def manifest_plan(path: str) -> dict[int, dict]:
    """``{level: planned row}`` from the run's ``plan`` record, ``{}`` if it has none.

    A run records the plan before its first level: each level's shape, task shape and
    ``grid_total``, with ``upper_bound`` where ``--sparse`` will drop empty tasks once the
    level below exists. The latest record wins, as a resume records it again.
    """
    if not path or not os.path.exists(path):
        return {}
    plan: list = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("group") == "plan" and "status" not in rec:
                plan = (rec.get("meta") or {}).get("levels") or plan
    return {int(p["level"]): p for p in plan if isinstance(p, dict) and "level" in p}


def task_total(lv: dict, chunk_total: int, chunk: tuple, level: int) -> tuple[int, str]:
    """The denominator for a level's manifest count, and a note if it isn't the chunk grid.

    The manifest counts tasks and ``chunk_total`` is the destination chunk grid. They
    differ at level 0 when the source reads in units coarser than the destination chunk.
    In order of preference: the recorded ``total``; the grid implied by the largest
    recorded block index (exact once the level finished, a lower bound before); the
    chunk grid.
    """
    if lv.get("total"):
        total = lv["total"]
        task = lv.get("task_shape")
        if total != chunk_total and task:
            per = math.prod(max(1, t // c) for t, c in zip(task, chunk))
            return total, (f"level {level} is counted in tasks of "
                           f"{tuple(task)}, {per} destination chunks each "
                           f"({chunk_total:,} chunks in {total:,} tasks)")
        return total, ""
    grid = lv.get("grid")
    if grid and math.prod(grid) < chunk_total:
        implied = math.prod(grid)
        return implied, (
            f"level {level}: this manifest predates the recorded task total, so "
            f"{implied:,} is inferred from the recorded block indices (grid {grid}, "
            f"against {chunk_total:,} chunks). Exact if the level finished, a lower "
            f"bound if it is still running — so the % is an upper bound")
    return chunk_total, ""


def other_manifests(manifest_path: str) -> list[str]:
    """Manifests beside ``manifest_path``, such as a rebuild's ``.regen-from-<n>`` one."""
    suffix = ".progress.jsonl"
    stem = manifest_path[: -len(suffix)] if manifest_path.endswith(suffix) else manifest_path
    return sorted(p for p in glob.glob(f"{stem}*{suffix}") if p != manifest_path)


def stored_chunks(volume: str, fmt: str, level: int, scale_key: str | None) -> int:
    """Count the chunk objects stored for one level.

    zarr v3 keeps each level's chunks (or shards) under ``<level>/c/``; precomputed keys
    each scale's chunks under its scale key beside one shared ``info``.
    """
    from ..location import list_keys

    if fmt == "zarr3":
        keys = list_keys(volume, str(level))
        return sum(1 for k in keys if k.startswith("c/"))
    keys = list_keys(volume, scale_key or "")
    # A precomputed chunk key looks like `0-128_0-128_0-128`.
    return sum(1 for k in keys if "_" in k.rsplit("/", 1)[-1] and "-" in k.rsplit("/", 1)[-1])


def _precomputed_scale_keys(volume: str) -> dict[int, str]:
    """``{level: scale key}``, finest first."""
    from ..scales import finest_first, precomputed_info

    scales = (precomputed_info(volume) or {}).get("scales", [])
    return {i: scales[j]["key"] for i, j in enumerate(finest_first(scales))}


def level_progress(volume: str, *, progress_path: str | None = None,
                   storage: bool = False) -> dict:
    """Blocks done per level of ``volume``, against the number each level has.

    Counted from the run manifest (``progress_path``, defaulting to where the run puts
    it) unless ``storage`` is set or no manifest exists, then from a storage listing.
    Returns ``{"manifest_path", "source": "manifest"|"storage", "levels": [row, ...],
    "notes": [...]}``, each row ``{"level", "shape", "chunk", "done", "total",
    "written", "empty", "planned", "upper_bound"}``; ``chunk`` is ``None`` where the level's
    chunking is unknown, and ``written`` / ``empty`` are ``None`` when counting from storage.

    From a manifest that records the run's plan, levels not yet created are rows too
    (``planned``), counted against their task grid; ``upper_bound`` marks a grid that
    ``--sparse`` will shrink when the level starts.

    Raises ``FileNotFoundError`` if nothing at ``volume`` is a volume, and
    ``ValueError`` for a single-array format, which has no chunks to count.
    """
    from ..detect import PRECOMPUTED_GZ, detect_backend, require_chunked_volume
    from ..location import default_progress_path
    from ..source_metadata import existing_levels

    volume = volume.rstrip("/")
    manifest_path = progress_path or default_progress_path(volume)
    manifest = None if storage else manifest_counts(manifest_path)

    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")
    require_chunked_volume(fmt, volume, "neu-vol progress")
    if fmt == PRECOMPUTED_GZ:
        # Only the chunk keys differ; shapes and scale keys come from `info`.
        fmt = "neuroglancer_precomputed"
    levels = existing_levels(volume, fmt)
    scale_keys = (_precomputed_scale_keys(volume)
                  if fmt == "neuroglancer_precomputed" else {})

    rows: list[dict] = []
    notes: list[str] = []
    for level, lv_info in sorted(levels.items()):
        shape, chunk = lv_info["shape"], lv_info["chunks"]
        row = {"level": level, "shape": shape, "chunk": chunk or None, "done": None,
               "total": None, "written": None, "empty": None, "planned": False,
               "upper_bound": False}
        rows.append(row)
        if not chunk:
            continue
        total = math.prod(math.ceil(s / c) for s, c in zip(shape, chunk))
        if manifest is not None:
            lv = manifest.get(level, {})
            by_status = lv.get("counts", {})
            row["done"] = sum(n for s, n in by_status.items() if s in DONE_STATUSES)
            row["written"] = by_status.get("written", 0)
            row["empty"] = by_status.get("empty", 0)
            total, note = task_total(lv, total, chunk, level)
            if note:
                notes.append(note)
        else:
            row["done"] = stored_chunks(volume, fmt, level, scale_keys.get(level))
        row["total"] = total

    plan = manifest_plan(manifest_path) if manifest is not None else {}
    ahead = [p for lvl, p in sorted(plan.items()) if lvl not in levels]
    for p in ahead:
        rows.append({"level": int(p["level"]), "shape": tuple(p["shape"]),
                     "chunk": tuple(p["task_shape"]), "done": 0, "total": int(p["grid_total"]),
                     "written": 0, "empty": 0, "planned": True,
                     "upper_bound": bool(p.get("upper_bound"))})
    if any(p.get("upper_bound") for p in ahead):
        notes.append("levels not started yet are counted against their full task grid; "
                     "with --sparse each shrinks when it starts (tasks with empty input are "
                     "dropped), so those totals, and the TOTAL, are upper bounds")
    return {"manifest_path": manifest_path,
            "source": "manifest" if manifest is not None else "storage",
            "levels": rows, "notes": notes}
