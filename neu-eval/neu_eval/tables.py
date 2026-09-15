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

    header = list(rows[0].keys()) if rows else [
        "pair_key", "kind", f"{labels[0]}_id", f"{labels[1]}_id", "n_voxels", "severity"]
    if VERDICT_COLUMN not in header:
        header.append(VERDICT_COLUMN)
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
        "`z_nm`/`y_nm`/`x_nm` are world coordinates: paste them into a viewer. The point is\n"
        "the deepest interior voxel of the overlap, so it lands inside the disagreement\n"
        "rather than on its edge.\n")
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


def write_table(frame, path: str | Path) -> str:
    """A DataFrame to parquet, matching ``neu_morpho.measure.tables.write_table``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)


def annotation_csv(rows: Iterable[Mapping], path: str | Path) -> str:
    """The located rows as a point list ``neu-glance annotate`` can read.

    Three columns of nanometres and a description, which is the contract that keeps this
    package from importing neu-glance: the table is the interface, exactly as neu-draw takes
    synapses as tables rather than importing neu-mark.

    These become **local** annotations in a viewer state, and that is the point — a
    precomputed annotation source renders in the viewport but puts zero rows in the
    Annotations tab, so there is nothing to click through and ``[``/``]`` do not step. A
    ranked worst-first list is exactly the thing you want to step through, so it has to be
    local.
    """
    rows = [row for row in rows if "z_nm" in row]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_nm", "y_nm", "x_nm", "description"])
        for row in rows:
            writer.writerow([
                row["z_nm"], row["y_nm"], row["x_nm"],
                f"{row['kind']} {row['pair_key']} sev={row['severity']:.4f}",
            ])
    return str(path)
