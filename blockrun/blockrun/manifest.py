"""Driver-side, single-writer completion manifest.

The driver is the only writer, so there is no concurrency to coordinate: as
per-task results stream back it appends one JSON line per task recording its
status. On resume it reads the manifest and callers filter out already-done tasks
before dispatch — no per-object existence scan.

Keys are generic:
  - ``group``: any JSON-serializable group id (e.g. a pyramid level int, or a
    single 0 when there's only one group).
  - ``key``:   any JSON-serializable task identity — a scalar (e.g. a segment id)
    or a sequence (e.g. a block-grid index tuple).
  - ``status``: an opaque string. neu-vol uses ``written`` / ``empty`` /
    ``skipped``; other consumers can use their own (e.g. ``failed``).

A second kind of line records facts **about** a group rather than a task within
it — ``{"group": g, "meta": {...}}``, written by :meth:`Manifest.record_meta`.
Its purpose is the denominator: only the driver knows how many tasks a group has,
and without it a reader can only guess, which is how ``neu-vol progress`` came to
divide a count of tasks by a count of chunks. A line carrying no ``status`` is
never a task, so old readers skip it and old files simply have no meta.
"""

from __future__ import annotations

import json
import os
from typing import Any, Hashable, Iterable


#: Sentinel for "every group" — ``None`` is a legal group id, so it cannot be it.
_ALL = object()


def _norm(key: Any) -> Hashable:
    """Normalize a key to a hashable form (lists -> tuples; scalars unchanged)."""
    if isinstance(key, list):
        return tuple(key)
    return key


class Manifest:
    """Append-only JSONL record of completed tasks, keyed by ``(group, key)``."""

    def __init__(self, path: str | None):
        self.path = path
        self._done: dict[tuple[Any, Hashable], str] = {}
        self._meta: dict[Any, dict] = {}
        self._fh = None

    # -- read -------------------------------------------------------------
    def load(self) -> "Manifest":
        self._done.clear()
        self._meta.clear()
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line from a hard crash
                    group = rec.get("group", rec.get("level"))
                    if "status" not in rec:
                        # Not a task record. Later meta wins, so a resumed run's
                        # totals replace the ones the interrupted run wrote.
                        if isinstance(rec.get("meta"), dict):
                            self._meta.setdefault(group, {}).update(rec["meta"])
                        continue
                    key = _norm(rec.get("key", rec.get("index")))
                    self._done[(group, key)] = rec["status"]
        return self

    def done_keys(self, group: Any) -> set:
        """All recorded keys for ``group``."""
        return {key for (g, key) in self._done if g == group}

    def is_done(self, group: Any, key: Any) -> bool:
        return (group, _norm(key)) in self._done

    def status(self, group: Any, key: Any) -> str | None:
        return self._done.get((group, _norm(key)))

    def counts(self, group: Any = _ALL) -> dict[str, int]:
        """``{status: count}`` over every group, or over one group if given."""
        out: dict[str, int] = {}
        for (g, _key), status in self._done.items():
            if group is _ALL or g == group:
                out[status] = out.get(status, 0) + 1
        return out

    def meta(self, group: Any) -> dict:
        """Facts recorded about ``group`` (e.g. its task ``total``); ``{}`` if none."""
        return dict(self._meta.get(group, {}))

    # -- write (driver only) ----------------------------------------------
    def reset(self) -> "Manifest":
        """Truncate the manifest (fresh run): clear in-memory + on-disk records."""
        self._done.clear()
        self._meta.clear()
        if self.path:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            open(self.path, "w").close()
        return self

    def _open(self) -> None:
        if self.path and self._fh is None:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "a")

    def record_meta(self, group: Any, **fields: Any) -> None:
        """Record facts *about* ``group`` — most usefully its task ``total``.

        Write this **before dispatching** the group's tasks: a reader watching a
        live run needs the denominator from the start, and a run that dies
        halfway should still leave behind what it was trying to do.
        """
        if not fields:
            return
        if self.path:
            self._open()
            self._fh.write(json.dumps({"group": group, "meta": fields}) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        self._meta.setdefault(group, {}).update(fields)

    def record(self, group: Any, results: Iterable[tuple[Any, str]]) -> None:
        """Record ``(key, status)`` results for a group; durable (flush+fsync)."""
        results = list(results)
        if self.path:
            self._open()
            for key, status in results:
                self._fh.write(json.dumps({"group": group, "key": key, "status": status}) + "\n")
        for key, status in results:
            self._done[(group, _norm(key))] = status
        if self._fh:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
