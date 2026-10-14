"""Per-body metrics database (SQLite).

One row per body, progressively enriched: the index scan fills bbox / voxel_count /
volume; meshing and skeletonization update their columns. Bbox is in **full-resolution
voxels**; consumers convert to their own scale.

Each block's contribution and its "done" mark land in **one transaction**, so a crash
cannot double-count a summed metric on resume. The driver is the sole writer.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Mapping, Sequence

_BBOX = ["z0", "y0", "x0", "z1", "y1", "x1"]
# enrichment columns other stages may set
_EXTRA = ["cable_length_nm", "n_branches", "n_tips", "max_radius_nm",
          "mesh_area_nm2", "mesh_verts", "n_mesh_components"]


class MetricsDB:
    def __init__(self, path: str, *, read_only: bool = False, busy_timeout_ms: int = 30_000):
        """Open the metrics DB. ``read_only`` opens it WITHOUT writing to it.

        **A reader must not take a write lock**: the default open runs DDL, which locks
        the DB and can kill a running sweep with ``database is locked``. Read-only tools
        pass ``read_only=True``, which skips the DDL and opens with ``mode=ro``.

        **A writer waits rather than fails**: ``busy_timeout`` turns contention with a
        concurrent reader into a pause.
        """
        self.read_only = bool(read_only)
        if self.read_only:
            self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self.con.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            return
        self.con = sqlite3.connect(path)
        self.con.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.con.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{c} INTEGER" for c in _BBOX)
        extra = ", ".join(f"{c} REAL" for c in _EXTRA)
        self.con.execute(
            f"CREATE TABLE IF NOT EXISTS bodies (body_id INTEGER PRIMARY KEY, {cols}, "
            f"voxel_count INTEGER DEFAULT 0, volume_nm3 REAL DEFAULT 0, {extra})")
        self.con.execute("CREATE TABLE IF NOT EXISTS index_progress (block TEXT PRIMARY KEY)")
        # The counts-only sweep keeps its own progress table: it fills voxel_count with
        # the same block keys, so a shared table would make a second pass skip every block.
        # Separate tables let `neu_morpho.measure.driver` refuse to mix the two.
        self.con.execute("CREATE TABLE IF NOT EXISTS sweep_progress (block TEXT PRIMARY KEY)")
        # A progress table counts tasks done; only the driver knows the total, so it
        # records it here, and `measure progress` can report a fraction.
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS stage_meta ("
            "stage TEXT PRIMARY KEY, total INTEGER, block_shape TEXT, level INTEGER, "
            "voxel_size TEXT, n_keep INTEGER, started TEXT)")
        self.con.commit()

    # -- index scan (reduction) -------------------------------------------
    def done_blocks(self) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT block FROM index_progress")}

    def reset_index(self) -> None:
        self.con.execute("DELETE FROM bodies")
        self.con.execute("DELETE FROM index_progress")
        self.con.commit()

    def apply_index_block(self, block_key: str,
                          partials: Mapping[int, tuple], voxel_volume_nm3: float) -> None:
        """Merge one block's {body_id: (z0,y0,x0,z1,y1,x1,count)} and mark it done, atomically.

        bbox merges by min/max; voxel_count and volume accumulate. No-op if the block was
        already applied. A pre-existing row may have a NULL bbox (another stage created
        it), and SQLite's ``min(NULL, 5)`` is ``NULL``, hence the explicit NULL check.
        """
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            if cur.execute("SELECT 1 FROM index_progress WHERE block=?", (block_key,)).fetchone():
                cur.execute("COMMIT")
                return
            for body_id, (z0, y0, x0, z1, y1, x1, count) in partials.items():
                cur.execute(
                    "INSERT INTO bodies (body_id, z0,y0,x0,z1,y1,x1, voxel_count, volume_nm3) "
                    "VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(body_id) DO UPDATE SET "
                    "z0=iif(z0 IS NULL,excluded.z0,min(z0,excluded.z0)), "
                    "y0=iif(y0 IS NULL,excluded.y0,min(y0,excluded.y0)), "
                    "x0=iif(x0 IS NULL,excluded.x0,min(x0,excluded.x0)), "
                    "z1=iif(z1 IS NULL,excluded.z1,max(z1,excluded.z1)), "
                    "y1=iif(y1 IS NULL,excluded.y1,max(y1,excluded.y1)), "
                    "x1=iif(x1 IS NULL,excluded.x1,max(x1,excluded.x1)), "
                    "voxel_count=voxel_count+excluded.voxel_count, "
                    "volume_nm3=volume_nm3+excluded.volume_nm3",
                    (int(body_id), z0, y0, x0, z1, y1, x1, int(count), count * voxel_volume_nm3))
            cur.execute("INSERT INTO index_progress (block) VALUES (?)", (block_key,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -- stage metadata (the progress DENOMINATOR) -------------------------
    def record_stage_meta(self, stage: str, *, total: int, block_shape=None,
                          level: int | None = None, voxel_size=None,
                          n_keep: int | None = None) -> None:
        """Record a stage's task total BEFORE dispatch. See the table comment."""
        import datetime as _dt
        self.con.execute(
            "INSERT INTO stage_meta (stage, total, block_shape, level, voxel_size, "
            "n_keep, started) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(stage) DO UPDATE SET total=excluded.total, "
            "block_shape=excluded.block_shape, level=excluded.level, "
            "voxel_size=excluded.voxel_size, n_keep=excluded.n_keep, "
            "started=excluded.started",
            (str(stage), int(total),
             None if block_shape is None else ",".join(str(int(v)) for v in block_shape),
             None if level is None else int(level),
             None if voxel_size is None else ",".join(str(float(v)) for v in voxel_size),
             None if n_keep is None else int(n_keep),
             _dt.datetime.now().astimezone().isoformat(timespec="seconds")))
        self.con.commit()

    def _has_table(self, name: str) -> bool:
        """A read-only open cannot CREATE TABLE IF NOT EXISTS, so ask instead."""
        return self.con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone() is not None

    def read_stage_meta(self) -> dict[str, dict]:
        if not self._has_table("stage_meta"):
            return {}                       # a DB written before totals were recorded
        cols = ("total", "block_shape", "level", "voxel_size", "n_keep", "started")
        return {r[0]: dict(zip(cols, r[1:]))
                for r in self.con.execute(
                    f"SELECT stage, {','.join(cols)} FROM stage_meta")}

    #: stage name -> its progress table. The one place a block-mapped stage is registered:
    #: `stage_counts` and `neu-morpho measure progress` both derive from it.
    BLOCK_STAGES = {"index": "index_progress",
                    "sweep": "sweep_progress",
                    "compartments": "compartment_progress"}

    def stage_counts(self) -> dict[str, int]:
        """Blocks completed per block-mapped stage."""
        return {stage: (self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        if self._has_table(table) else 0)
                for stage, table in self.BLOCK_STAGES.items()}

    # -- per-body compartment split ----------------------------------------
    def _ensure_compartments(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS body_compartments ("
            "body_id INTEGER, label INTEGER, voxel_count INTEGER, "
            "PRIMARY KEY (body_id, label))")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS compartment_progress (block TEXT PRIMARY KEY)")
        self.con.commit()

    def done_compartment_blocks(self) -> set[str]:
        if not self._has_table("compartment_progress"):
            return set()
        return {r[0] for r in self.con.execute("SELECT block FROM compartment_progress")}

    def reset_compartments(self) -> set[str]:
        self._ensure_compartments()
        self.con.execute("DELETE FROM compartment_progress")
        self.con.execute("DELETE FROM body_compartments")
        self.con.commit()
        return set()

    def apply_compartment_block(self, block_key: str, counts) -> None:
        """Merge one block's ``{(body_id, label): voxels}`` and mark it done, atomically."""
        self._ensure_compartments()
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            if cur.execute("SELECT 1 FROM compartment_progress WHERE block=?",
                           (block_key,)).fetchone():
                cur.execute("COMMIT")
                return
            for (body_id, label), n in counts.items():
                cur.execute(
                    "INSERT INTO body_compartments (body_id, label, voxel_count) "
                    "VALUES (?,?,?) ON CONFLICT(body_id, label) DO UPDATE SET "
                    "voxel_count=voxel_count+excluded.voxel_count",
                    (int(body_id), int(label), int(n)))
            cur.execute("INSERT INTO compartment_progress (block) VALUES (?)", (block_key,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def compartment_counts(self) -> dict[int, int]:
        """Total voxels per semantic label, across bodies."""
        if not self._has_table("body_compartments"):
            return {}
        return {int(r[0]): int(r[1]) for r in self.con.execute(
            "SELECT label, SUM(voxel_count) FROM body_compartments GROUP BY label")}

    # -- per-body skeleton pass --------------------------------------------
    def _ensure_skel_status(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS skel_status ("
            "body_id INTEGER PRIMARY KEY, status TEXT, detail TEXT)")
        self.con.commit()

    def done_skel_bodies(self, *, include_failed: bool = False) -> set[int]:
        """Bodies not worth attempting again.

        Status is filtered, not tested for presence, so a recorded ``failed`` is retried.
        ``absent`` is terminal: a body with no published skeleton will not grow one.
        """
        if not self._has_table("skel_status"):
            return set()
        keep = ("written", "absent", "failed") if include_failed else ("written", "absent")
        q = f"SELECT body_id FROM skel_status WHERE status IN ({','.join('?' * len(keep))})"
        return {int(r[0]) for r in self.con.execute(q, keep)}

    def reset_skel(self) -> set[int]:
        self._ensure_skel_status()
        self.con.execute("DELETE FROM skel_status")
        cols = ", ".join(f"{c}=NULL" for c in
                         ("cable_length_nm", "n_branches", "n_tips", "max_radius_nm"))
        self.con.execute(f"UPDATE bodies SET {cols}")
        self.con.commit()
        return set()

    def apply_skel_batch(self, rows) -> None:
        """Apply one batch of ``(body_id, status, detail, columns)`` atomically."""
        self._ensure_skel_status()
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            for body_id, status, detail, cols in rows:
                if cols:
                    self._upsert(int(body_id), cols)
                cur.execute(
                    "INSERT INTO skel_status (body_id, status, detail) VALUES (?,?,?) "
                    "ON CONFLICT(body_id) DO UPDATE SET status=excluded.status, "
                    "detail=excluded.detail",
                    (int(body_id), str(status), detail))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def skel_counts(self) -> dict[str, int]:
        if not self._has_table("skel_status"):
            return {}
        return {r[0]: r[1] for r in self.con.execute(
            "SELECT status, COUNT(*) FROM skel_status GROUP BY status")}

    # -- counts-only sweep (no bbox) ---------------------------------------
    def done_sweep_blocks(self) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT block FROM sweep_progress")}

    def reset_sweep(self) -> set[str]:
        self.con.execute("DELETE FROM sweep_progress")
        self.con.execute("UPDATE bodies SET voxel_count=0, volume_nm3=0")
        self.con.commit()
        return set()

    def apply_counts_block(self, block_key: str, counts: Mapping[int, int],
                           voxel_volume_nm3: float) -> None:
        """Merge one block's ``{body_id: voxel_count}`` and mark it done, atomically.

        The bbox-free counterpart of :meth:`apply_index_block`, for the morphometry sweep,
        about 10x cheaper per dense block. Rows created here have NULL bbox and skeleton
        columns: this pass did not measure them.
        """
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            if cur.execute("SELECT 1 FROM sweep_progress WHERE block=?", (block_key,)).fetchone():
                cur.execute("COMMIT")
                return
            for body_id, count in counts.items():
                cur.execute(
                    "INSERT INTO bodies (body_id, voxel_count, volume_nm3) VALUES (?,?,?) "
                    "ON CONFLICT(body_id) DO UPDATE SET "
                    "voxel_count=voxel_count+excluded.voxel_count, "
                    "volume_nm3=volume_nm3+excluded.volume_nm3",
                    (int(body_id), int(count), count * voxel_volume_nm3))
            cur.execute("INSERT INTO sweep_progress (block) VALUES (?)", (block_key,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -- queries -----------------------------------------------------------
    def get_bbox(self, body_id: int) -> tuple | None:
        r = self.con.execute(
            f"SELECT {','.join(_BBOX)} FROM bodies WHERE body_id=?", (int(body_id),)).fetchone()
        return tuple(r) if r else None

    def crop_at_scale(self, body_id: int, factor_zyx: Sequence[int], *,
                      margin_vox: int = 0, clip_shape: Sequence[int] | None = None) -> tuple | None:
        """A body's bbox at a read scale, as (z0,y0,x0,z1,y1,x1) voxels (+ margin).

        ``factor_zyx`` = full-res voxels per read-scale voxel.
        """
        bb = self.get_bbox(body_id)
        if bb is None:
            return None
        fz, fy, fx = factor_zyx
        lo = [bb[0] // fz, bb[1] // fy, bb[2] // fx]
        hi = [-(-bb[3] // fz), -(-bb[4] // fy), -(-bb[5] // fx)]   # ceil
        lo = [max(0, v - margin_vox) for v in lo]
        hi = [v + margin_vox for v in hi]
        if clip_shape is not None:
            hi = [min(h, int(s)) for h, s in zip(hi, clip_shape)]
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def bodies_by_size(self, min_voxels: int = 0, limit: int | None = None) -> list[int]:
        q = "SELECT body_id FROM bodies WHERE voxel_count>=? ORDER BY voxel_count DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [r[0] for r in self.con.execute(q, (int(min_voxels),))]

    def write_allowlist(self, path: str, min_voxels: int = 0, limit: int | None = None) -> int:
        ids = self.bodies_by_size(min_voxels, limit)
        with open(path, "w") as f:
            f.write("body_id\n")
            f.writelines(f"{i}\n" for i in ids)
        return len(ids)

    # -- enrichment (mesh / skeleton stages, via the single-writer driver) --
    def update_body(self, body_id: int, **cols) -> None:
        """Set enrichment columns for a body, inserting the row if it is new.

        The values go in the INSERT as well as the DO UPDATE, so a body the index scan
        never saw is inserted with them.
        """
        if not cols:
            return
        self._upsert(body_id, cols)
        self.con.commit()

    def _upsert(self, body_id: int, cols: Mapping) -> None:
        names = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        assignments = ", ".join(f"{k}=excluded.{k}" for k in cols)
        self.con.execute(
            f"INSERT INTO bodies (body_id, {names}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(body_id) DO UPDATE SET {assignments}",
            (int(body_id), *cols.values()))

    def update_bodies(self, rows: Iterable[tuple[int, Mapping]]) -> int:
        """Upsert many bodies in **one transaction with one commit**.

        ``update_body`` commits per call; in a pipeline that is one fsync per body in the
        driver, the sole writer, while workers sit idle.
        """
        n = 0
        for body_id, cols in rows:
            if cols:
                self._upsert(body_id, cols)
                n += 1
        if n:
            self.con.commit()
        return n

    def to_csv(self, path: str) -> None:
        import csv
        cur = self.con.execute("SELECT * FROM bodies ORDER BY body_id")
        names = [d[0] for d in cur.description]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(cur)

    def commit(self) -> None:
        self.con.commit()

    def close(self) -> None:
        self.con.commit()
        self.con.close()
