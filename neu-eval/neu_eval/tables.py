"""Writing results out, and reading a reviewed disagreement table back in.

pandas and pyarrow arrive with ``neu-eval[report]`` and are imported **inside** functions,
so ``import neu_eval`` does not pay for them and neither will ``--help``. Same rule
``neu_morpho.measure`` follows, and a test asserts it.

**The file formats differ on purpose, and the reason is who edits which.** Metrics go to
parquet: nobody hand-edits them, and parquet round-trips a nullable integer column where csv
does not. The disagreement table goes to **csv**, because a human opens it and fills in the
verdict column, and a parquet file is not something you can do that to.

**Which makes csv's own hazard load-bearing here: a spreadsheet destroys a uint64 body id.**
Excel and LibreOffice both parse a long integer as a float64, so a 19-digit id comes back
rounded — the failure `neu_morpho.measure.cohort` already hit from the other direction.
So ids are written as **text**, and ``pair_key`` rather than the id columns is the join key
on the way back in. A reviewer who reformats a column still cannot break the join.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .adjudicate import VERDICT_COLUMN
from .overlap import Contingency

#: Columns written as text in the csv, whatever their Python type. Everything that is or
#: could be a uint64 label id.
_TEXT_COLUMNS = ("pair_key",)


def write_summary(rows: Mapping | Sequence[Mapping], path: str | Path) -> str:
    """One parquet file of metric rows. Accepts a single row or many."""
    import pandas as pd

    frame = pd.DataFrame([rows] if isinstance(rows, Mapping) else list(rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)


def per_label_frame(c: Contingency):
    """Per-label VOI contributions and sizes, one row per label on each side.

    A long frame with a ``side`` column rather than two frames, because the two sides are
    measured the same way and a single table is what a groupby wants.
    """
    import pandas as pd

    from .voxel import per_label_voi

    merge_by_a, split_by_b = per_label_voi(c)
    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()

    records = []
    for ids, counts, blame, side in ((a_ids, a_counts, merge_by_a, c.labels[0]),
                                     (b_ids, b_counts, split_by_b, c.labels[1])):
        for label, n in zip(ids, counts):
            records.append({
                "side": side,
                "label_id": int(label),
                "n_voxels": int(n),
                "voi_contribution": float(blame.get(int(label), 0.0)),
            })
    frame = pd.DataFrame(records)
    return frame.sort_values(["side", "voi_contribution"], ascending=[True, False])


def write_disagreements(rows: Iterable[Mapping], path: str | Path, *,
                        labels: Sequence[str] = ("gt", "seg"),
                        verdicts: Mapping[str, str] | None = None) -> str:
    """The ranked disagreements as csv, with a ``verdict`` column to fill in.

    Written with the stdlib ``csv`` module rather than pandas: this is the one output that
    does not need the ``report`` extra, and it would be perverse for the file a reviewer
    edits to be the one requiring an optional dependency.

    ``verdicts`` pre-fills the column, so re-running a comparison preserves a review that
    has already happened instead of blanking it — which, the first time it blanks somebody's
    afternoon of work, is not recoverable.
    """
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The UNION of every row's keys, in first-seen order — not `rows[0].keys()`, which is
    # what this did and which silently dropped any column a later row added. Rows compared
    # under different side names (one table per vendor, say) have different id columns, and
    # `DictWriter(extrasaction="ignore")` discarded the ones missing from row zero: 450 of
    # 600 rows came out with no segment id at all, and nothing said so.
    header: list[str] = []
    for row in rows:
        for key in row:
            # `shapes` is nested geometry for the viewer (see shape_csvs); it has no
            # meaning as one spreadsheet cell and would only bury the verdict column.
            if key not in header and key not in _NOT_TABULAR:
                header.append(key)
    if not header:
        header = ["pair_key", "kind", f"{labels[0]}_id", f"{labels[1]}_id",
                  "n_voxels", "severity"]
    if VERDICT_COLUMN not in header:
        header.append(VERDICT_COLUMN)
    if "note" not in header:
        header.append("note")

    verdicts = verdicts or {}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.setdefault(VERDICT_COLUMN, verdicts.get(str(row.get("pair_key", "")), ""))
            out.setdefault("note", "")
            for column in header:
                # Ids and keys as text; a spreadsheet rounds a long integer to float64.
                if column.endswith("_id") or column in _TEXT_COLUMNS:
                    out[column] = str(out.get(column, ""))
            writer.writerow(out)

    # A legend the reviewer can read without finding the docs.
    legend = path.with_suffix(".README.txt")
    legend.write_text(
        "Fill in the `verdict` column for the rows you inspect, then pass this file back\n"
        "with `neu-eval compare --adjudicated <this file>`.\n\n"
        "Allowed verdicts:\n"
        f"  {labels[0]}_correct   the reference is right; the other side got it wrong\n"
        f"  {labels[1]}_correct   the other side is right; the REFERENCE is defective here\n"
        "  ambiguous     looked at, genuinely unclear (still scored -- see below)\n"
        "  skip          not looked at\n"
        "  <blank>       not looked at\n\n"
        f"Only `{labels[1]}_correct` rows leave the denominator on a rescore. `ambiguous`\n"
        "is still scored on purpose: \"I could not tell\" is not evidence the reference was\n"
        "wrong, and excluding it is how an adjudicated score drifts upward.\n\n"
        "Ids are written as TEXT. If you open this in a spreadsheet, keep them that way --\n"
        "a long integer parsed as a number comes back rounded and no longer names a body.\n"
        "`pair_key` is the join key, so reformatting an id column cannot break the round\n"
        "trip.\n\n"
        "`z_nm`/`y_nm`/`x_nm` are world coordinates: paste them into a viewer. The point\n"
        "sits on the false cut or join (`point_at` says which), not in the overlap.\n")
    return str(path)


def read_disagreements(path: str | Path) -> tuple[list[dict], dict[str, str]]:
    """``(rows, verdicts)`` from a csv written by :func:`write_disagreements`.

    ``verdicts`` maps ``pair_key`` to the verdict text, ready for
    :func:`neu_eval.adjudicate.apply`. Rows come back as read — strings — because the only
    thing this needs to be right about is the key and the verdict, and re-typing the numeric
    columns from a file a human has edited invites a parse failure over a column nobody uses.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and VERDICT_COLUMN not in rows[0]:
        raise ValueError(
            f"{path} has no {VERDICT_COLUMN!r} column, so it is not an adjudication file. "
            f"Columns: {', '.join(rows[0])}")
    verdicts = {row["pair_key"]: (row.get(VERDICT_COLUMN) or "").strip()
                for row in rows if row.get("pair_key")}
    return rows, verdicts


#: Row keys that are not columns. Written by their own writers.
_NOT_TABULAR = ("shapes",)

#: Which shape each disagreement kind is drawn as, per side. A merge is a LINE across the
#: false join -- it connects the two things that should not be connected. A split is an
#: ELLIPSOID over the false cut -- a surface where there should be no boundary, sized by
#: how big the cut is. A tangle is both errors, so it gets both.
SHAPE_OF = {"merge": "lines", "split": "ellipsoids"}


def shape_csvs(rows: Iterable[Mapping], out_dir: str | Path, prefix: str, *,
               labels: Sequence[str] = ("gt", "seg"),
               relationships: Sequence[str] = ()) -> dict[str, dict[str, str]]:
    """The located shapes as the CSVs ``neu-glance annotate`` reads, one set per kind.

    Returns ``{"merges": {"lines": path, "points": path}, "splits": {"ellipsoids": ...},
    "tangles": {"lines": ..., "ellipsoids": ...}}``, only for files that have rows. Each
    kind is its own set because each becomes its own layer, so merges, splits and tangles
    can be toggled independently.

    Coordinates are **nanometres** (so ``annotate --nm``). Each row carries a
    ``segments:<name>`` column per entry of ``relationships`` -- the ids
    :func:`neu_eval.disagree.sample_ids` found at the shape's two anchors in that
    labelling -- for ``annotate --link <name>=<layer>``. A relationship with nothing there
    is written empty, not left out, because the viewer pairs id lists by position.

    A row ``locate`` could not shape (no seam: a disconnected fragment) is kept as a POINT
    in its kind's set rather than dropped, so every row of the table has an annotation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_key, b_key = f"{labels[0]}_id", f"{labels[1]}_id"
    rels = list(relationships)
    groups = {"merge": "merges", "split": "splits", "tangle": "tangles"}
    buckets: dict[tuple[str, str], list[list]] = {}

    def seg_cols(ids):
        return [" ".join((ids or {}).get(r, [])) for r in rels]

    for row in rows:
        group = groups.get(row.get("kind"))
        if group is None:
            continue
        drawn = False
        for s in row.get("shapes") or ():
            ids = s.get("ids", {})
            a_here, b_here = ids.get(labels[0]), ids.get(labels[1])
            if s["side"] == "merge":
                what = (f"merge: {labels[1]} {row[b_key]} joins {labels[0]} "
                        f"{' + '.join(a_here) if a_here else row[a_key]}")
                coords = [*s["p0_nm"], *s["p1_nm"]]
            else:
                what = (f"split: {labels[0]} {row[a_key]} cut into {labels[1]} "
                        f"{' + '.join(b_here) if b_here else row[b_key]}")
                coords = [*s["centre_nm"], *s["radii_nm"]]
            desc = f"{what}  (sev {float(row.get('severity', 0)):.4f}, #{row.get('group_rank', 1)})"
            ident = f"{row['pair_key']}:{s['side']}"
            buckets.setdefault((group, SHAPE_OF[s["side"]]), []).append(
                [*coords, ident, desc, *seg_cols(ids)])
            drawn = True
        if not drawn and "z_nm" in row:
            desc = (f"{row['kind']} {row['pair_key']} (no seam found: "
                    f"{row.get('point_at', 'overlap')})")
            buckets.setdefault((group, "points"), []).append(
                [row["z_nm"], row["y_nm"], row["x_nm"], f"{row['pair_key']}:point", desc,
                 *seg_cols({labels[0]: [str(row[a_key])], labels[1]: [str(row[b_key])]})])

    heads = {"lines": ["z0", "y0", "x0", "z1", "y1", "x1"],
             "ellipsoids": ["z", "y", "x", "rz", "ry", "rx"],
             "points": ["z", "y", "x"]}
    written: dict[str, dict[str, str]] = {}
    for (group, kind), body in buckets.items():
        path = out_dir / f"{prefix}-{group}.{kind}.csv"
        with path.open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow([*heads[kind], "id", "description",
                        *[f"segments:{r}" for r in rels]])
            w.writerows(body)
        written.setdefault(group, {})[kind] = str(path)
    return written


def write_table(frame, path: str | Path) -> str:
    """A DataFrame to parquet, matching ``neu_morpho.measure.tables.write_table``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)


def annotation_csv(rows: Iterable[Mapping], path: str | Path, *,
                   labels: Sequence[str] = ("gt", "seg")) -> str:
    """The located rows as a point list ``neu-glance annotate --points`` can read.

    **The column names are neu-glance's contract, not ours**: ``z``, ``y``, ``x`` required,
    ``id`` / ``description`` / ``segments`` optional. This wrote ``z_nm``/``y_nm``/``x_nm``
    at first, which its reader rejects outright — caught only by running the command that
    three of our own docs had been advertising. Coordinates here are **nanometres**, so the
    command needs ``--nm`` and a frame (``--volume`` or ``--voxel-size``).

    ``segments`` carries both ids of the pair, so clicking the annotation selects the bodies
    it is about rather than leaving you to type them. neuroglancer looks them up in the
    segmentation the annotation layer is linked to, so the id belonging to the other side
    simply finds nothing — harmless, and cheaper than emitting two files.

    Points, not boxes, and they become **local** annotations: only local ones appear in the
    Annotations tab, which is what makes a ranked worst-first list steppable with ``[`` and
    ``]``. A precomputed annotation source renders in the viewport and lists nothing.

    Writing the table rather than importing neu-glance is the whole interface — the same
    abstention neu-draw keeps from neu-mark by taking synapses as tables.
    """
    rows = [row for row in rows if "z_nm" in row]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a_key, b_key = f"{labels[0]}_id", f"{labels[1]}_id"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z", "y", "x", "description", "segments"])
        for row in rows:
            ids = [str(row[k]) for k in (a_key, b_key) if k in row]
            writer.writerow([
                row["z_nm"], row["y_nm"], row["x_nm"],
                f"{row['kind']} {row['pair_key']} sev={row['severity']:.4f} "
                f"frac={row.get('worst_fraction', 0):.2f} "
                f"@{row.get('point_at', 'overlap')}",
                " ".join(ids),
            ])
    return str(path)
