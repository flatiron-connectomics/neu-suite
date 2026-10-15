"""A work dir's run records: one directory per run, and a current view built from them.

    <work_dir>/runs/<run id>/run_plan.json      written when the run starts
    <work_dir>/runs/<run id>/run_summary.json   written when it ends, completed or aborted
    <work_dir>/run_plan.json, run_summary.json  the current view

A run's own files are written once and never merged, so every run sharing a work dir
(``neu-morpho run --stages index,mesh``, ``neu-skel run``, a retry) stays on record in
full. The view holds, for each stage, the latest run that executed it, naming that run,
and lists every run. A run with a plan and no summary is still going, or was killed.

Imports only the standard library: ``neu-morpho progress`` reads through it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable

RUNS = "runs"
PLAN = "run_plan.json"
SUMMARY = "run_summary.json"

#: A stage whose latest run has written no summary: still running, or killed.
UNFINISHED = "unfinished"


def new_run_id(work_dir: str, started: datetime, stages: Iterable[str]) -> str:
    """``<YYYYmmddTHHMMSS.mmm>-<stages>``, suffixed ``-2``, ``-3`` … if taken.

    Ids sort in start order, which is the order the view applies runs in.
    """
    stamp = f"{started:%Y%m%dT%H%M%S}.{started.microsecond // 1000:03d}"
    base = f"{stamp}-{'-'.join(stages)}"
    run_id, n = base, 1
    while os.path.exists(os.path.join(work_dir, RUNS, run_id)):
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _write(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, path)          # a reader never sees half a file


def _read(path: str) -> dict | None:
    try:
        with open(path) as f:
            out = json.load(f)
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def write_run_plan(work_dir: str, run_id: str, record: dict) -> None:
    _write(os.path.join(work_dir, RUNS, run_id, PLAN), record)


def write_run_summary(work_dir: str, run_id: str, record: dict) -> None:
    _write(os.path.join(work_dir, RUNS, run_id, SUMMARY), record)


def runs(work_dir: str) -> list[tuple[str, dict, dict | None]]:
    """``(run id, plan, summary or None)`` for every recorded run, oldest first."""
    root = os.path.join(work_dir, RUNS)
    try:
        ids = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for run_id in ids:
        plan = _read(os.path.join(root, run_id, PLAN))
        if plan is not None:
            out.append((run_id, plan, _read(os.path.join(root, run_id, SUMMARY))))
    return out


#: Plan fields that describe one stage; every other field is the run's, copied to each.
_PER_STAGE_PLAN = ("scales", "planned_blocks")
_RUN_ONLY_PLAN = ("run", "started", "command", "dst", "work_dir", "stages", "seg_scales")


def rebuild_view(work_dir: str, dst: str) -> tuple[dict, dict]:
    """Rewrite the work dir's ``run_plan.json`` and ``run_summary.json`` from its runs to
    ``dst``; returns ``(plan view, summary view)``."""
    recorded = [r for r in runs(work_dir) if r[1].get("dst") == dst]
    latest: dict[str, tuple[str, dict, dict | None]] = {}
    for rec in recorded:
        for stage in rec[1].get("stages", []):
            latest.pop(stage, None)              # re-inserted: keeps run order
            latest[stage] = rec

    plan_stages, summary_stages = {}, {}
    for stage, (run_id, plan, summary) in latest.items():
        about = {"id": run_id, "command": plan.get("command"), "started": plan.get("started")}
        entry = {"run": about,
                 "scale": (plan.get("scales") or {}).get(stage),
                 "planned_blocks": (plan.get("planned_blocks") or {}).get(stage),
                 **{k: v for k, v in plan.items()
                    if k not in _RUN_ONLY_PLAN and k not in _PER_STAGE_PLAN}}
        if stage == "seg":
            entry["seg_scales"] = plan.get("seg_scales") or []
        plan_stages[stage] = entry

        if summary is None:
            status, finished, minutes, result = UNFINISHED, None, None, {}
        else:
            # A stage that finished before its run aborted is complete all the same.
            result = (summary.get("stages") or {}).get(stage)
            status = "completed" if result is not None else summary.get("status")
            finished = summary.get("finished")
            minutes = (summary.get("timing_min") or {}).get(stage)
            result = result or {}
        summary_stages[stage] = {**result, "run": {**about, "finished": finished,
                                                   "minutes": minutes, "status": status}}

    plan_view = {
        "dst": dst, "work_dir": work_dir, "stages": plan_stages,
        "runs": [{"id": i, "started": p.get("started"), "command": p.get("command"),
                  "stages": p.get("stages", [])} for i, p, _ in recorded],
    }
    finished = [s for _, _, s in recorded if s is not None]
    last = finished[-1] if finished else {}
    summary_view = {
        "volume": last.get("volume"), "work_dir": work_dir,
        "neuroglancer_source": last.get("neuroglancer_source"),
        "linked_subresources": last.get("linked_subresources", {}),
        "n_failed_bodies": sum(len(s.get("failed_bodies") or [])
                               for s in summary_stages.values()),
        "stages": summary_stages,
        "runs": [{"id": i, "started": p.get("started"), "command": p.get("command"),
                  "stages": p.get("stages", []),
                  "status": s.get("status", "completed") if s else UNFINISHED,
                  "finished": s.get("finished") if s else None,
                  "timing_min": s.get("timing_min") if s else None,
                  "n_failed_bodies": s.get("n_failed_bodies") if s else None}
                 for i, p, s in recorded],
    }
    _write(os.path.join(work_dir, PLAN), plan_view)
    _write(os.path.join(work_dir, SUMMARY), summary_view)
    return plan_view, summary_view


# --------------------------------------------------------------------------- #
# Reading the view
# --------------------------------------------------------------------------- #
def planned_blocks(plan_view: dict) -> dict[str, Any]:
    return {s: e.get("planned_blocks") for s, e in (plan_view.get("stages") or {}).items()}


def seg_scales(plan_view: dict) -> list:
    return ((plan_view.get("stages") or {}).get("seg") or {}).get("seg_scales") or []


def unfinished_stages(summary_view: dict | None) -> list[str]:
    """Stages whose latest run has written no summary."""
    return [s for s, e in ((summary_view or {}).get("stages") or {}).items()
            if (e.get("run") or {}).get("status") == UNFINISHED]
