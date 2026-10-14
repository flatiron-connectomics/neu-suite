"""The table model: elements and relationships as DataFrames.

**Coordinates are always named columns** (``z``, ``y``, ``x``, ``to_z``, ...). DVID stores
positions as ``[x, y, z]``, the suite holds zyx in memory and precomputed stores xyz, and
a mix-up mirrors positions through the z=x diagonal with no error. Named columns cannot be
transposed silently; :func:`positions_zyx` and :func:`positions_xyz` are the only places
an ``(N, 3)`` array is built from a table, and the caller states the order.

A DVID relationship points at a coordinate, not a body (``{"Rel": "PostSynTo", "To":
[x, y, z]}``); the partner body is found by joining that coordinate against the fetched
elements, so it resolves only when both ends' bodies were fetched. :func:`match_rate`
reports how much resolved: a low rate means the body list is too small, and otherwise
looks like a sparse connectome. Relationships are kept one row each, unmodified, and
:func:`connections` derives the de-duplicated synapse view, using both directions since
each is anchored on the body list at one end only.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: In-memory axis order for this package, and the column order of every table it writes.
AXES = ("z", "y", "x")

#: DVID's own order, which is what ``Pos`` and ``To`` are in.
DVID_AXES = ("x", "y", "z")

#: Coordinate dtype; int32 spans ±2.1e9 voxels.
COORD_DTYPE = np.int32

#: Body ids are uint64 in DVID and must stay uint64 on disk, which is why parquet is the
#: default output format (see :mod:`neu_mark.io`).
BODY_DTYPE = np.uint64

#: Synaptic relationship names, and which end each makes the element: ``PostSynTo`` means
#: the element is the PSD and the target the T-bar; ``PreSynTo`` the reverse. Taken from the
#: relationship rather than the element's ``Kind``, so a mislabelled element cannot invert
#: an edge.
SYNAPTIC_RELS = {"PostSynTo": ("post", "pre"), "PreSynTo": ("pre", "post")}


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame({
        "body": pd.Series(dtype=BODY_DTYPE),
        **{a: pd.Series(dtype=COORD_DTYPE) for a in AXES},
        "kind": pd.Series(dtype="string"),
    })


def _empty_rels() -> pd.DataFrame:
    return pd.DataFrame({
        "rel": pd.Series(dtype="string"),
        **{a: pd.Series(dtype=COORD_DTYPE) for a in AXES},
        **{f"to_{a}": pd.Series(dtype=COORD_DTYPE) for a in AXES},
        "from_body": pd.Series(dtype=BODY_DTYPE),
        "to_body": pd.Series(dtype="UInt64"),
    })


def elements_to_frames(elements: Sequence[Mapping[str, Any]],
                       body: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One body's ``/label`` response -> (points, relationships).

    Elements have ``Pos``, ``Kind``, ``Tags``, ``Prop`` and ``Rels``; every ``Prop`` key
    becomes a column. (``neuclease.load_elements_as_dataframe`` drops the element's body.)
    """
    if not elements:
        return _empty_points(), _empty_rels()

    rows: list[dict] = []
    rel_rows: list[dict] = []
    for el in elements:
        pos = el.get("Pos")
        if pos is None or len(pos) != 3:
            raise ValueError(
                f"element in body {body} has Pos={pos!r}; expected three coordinates")
        # xyz -> zyx, by name.
        x, y, z = (int(v) for v in pos)
        row: dict[str, Any] = {"body": int(body), "z": z, "y": y, "x": x,
                               "kind": el.get("Kind")}
        tags = el.get("Tags") or []
        # A comma-joined string, since a list column does not round-trip through csv.
        row["tags"] = ",".join(str(t) for t in tags) if tags else ""
        for key, value in (el.get("Prop") or {}).items():
            # Usually `conf`, `user`, `annotation`; any other key is kept too.
            row[str(key)] = value
        rows.append(row)

        for rel in el.get("Rels") or []:
            to = rel.get("To")
            if to is None or len(to) != 3:
                # A deleted partner leaves a null target; kept as a row with no target, so
                # dangling references can be counted.
                tx = ty = tz = None
            else:
                tx, ty, tz = (int(v) for v in to)
            rel_rows.append({"rel": rel.get("Rel"), "z": z, "y": y, "x": x,
                             "to_z": tz, "to_y": ty, "to_x": tx,
                             "from_body": int(body)})

    points = pd.DataFrame(rows)
    rels = pd.DataFrame(rel_rows) if rel_rows else _empty_rels()
    return points, rels


def _coerce_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].astype(BODY_DTYPE)
    for a in AXES:
        df[a] = df[a].astype(COORD_DTYPE)
    if "kind" in df:
        df["kind"] = df["kind"].astype("string")
    if "conf" in df:
        # DVID stores confidence as a string; float32 is enough.
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce").astype(np.float32)
    lead = ["body", *AXES, "kind", "tags"]
    rest = [c for c in df.columns if c not in lead]
    return df[[c for c in lead if c in df.columns] + rest]


def combine(frames: Iterable[tuple[pd.DataFrame, pd.DataFrame]]
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-body ``(points, rels)`` pairs and resolve partner bodies."""
    point_parts, rel_parts = [], []
    for pts, rels in frames:
        if len(pts):
            point_parts.append(pts)
        if len(rels):
            rel_parts.append(rels)

    points = (_coerce_points(pd.concat(point_parts, ignore_index=True))
              if point_parts else _empty_points())
    rels = (pd.concat(rel_parts, ignore_index=True) if rel_parts else _empty_rels())
    return points, resolve_partner_bodies(points, rels)


def position_index(points: pd.DataFrame) -> pd.Series:
    """``(z, y, x) -> body`` for every fetched element.

    Raises if two elements share a position, which would make the partner join ambiguous.
    """
    if not len(points):
        return pd.Series(dtype=BODY_DTYPE)
    dup = points.duplicated(subset=list(AXES)).sum()
    if dup:
        raise ValueError(
            f"{dup} elements share a position with another element. A position is the "
            f"join key for relationships, so duplicates would make partner resolution "
            f"ambiguous. This should not happen within one annotation instance.")
    return points.set_index(list(AXES))["body"]


def resolve_partner_bodies(points: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    """Fill ``to_body`` by joining each relationship's target against ``points``.

    ``NA`` where the target was not fetched. Nullable ``UInt64``, never float, which would
    round large body ids.
    """
    rels = rels.copy()
    if not len(rels):
        rels["to_body"] = pd.Series(dtype="UInt64")
        return rels

    for col in ("from_body",):
        rels[col] = rels[col].astype(BODY_DTYPE)
    lut = position_index(points)
    keys = pd.MultiIndex.from_arrays(
        [rels[f"to_{a}"] for a in AXES], names=list(AXES))
    rels["to_body"] = pd.Series(lut.reindex(keys).to_numpy(), index=rels.index,
                                dtype="Float64").astype("UInt64")
    for a in AXES:
        rels[a] = rels[a].astype(COORD_DTYPE)
        # to_* stays nullable: a dangling relationship has no target coordinate.
        rels[f"to_{a}"] = rels[f"to_{a}"].astype("Int32")
    lead = ["rel", *AXES, *[f"to_{a}" for a in AXES], "from_body", "to_body"]
    return rels[[c for c in lead if c in rels.columns]
                + [c for c in rels.columns if c not in lead]]


def connections(rels: pd.DataFrame) -> pd.DataFrame:
    """The oriented, de-duplicated synapse view over the raw relationship table.

    One row per distinct ``(tbar, psd)`` coordinate pair, whichever direction it was seen
    from. Columns ``pre_z/pre_y/pre_x``, ``post_z/post_y/post_x``, ``pre_body``,
    ``post_body``.

    Both relationship directions are used, then de-duplicated. Non-synaptic relationships
    are ignored here and stay in the raw table.
    """
    cols = {}
    frames = []
    for name, (here, there) in SYNAPTIC_RELS.items():
        part = rels[rels["rel"] == name]
        if not len(part):
            continue
        cols = {}
        for a in AXES:
            cols[f"{here}_{a}"] = part[a]
            cols[f"{there}_{a}"] = part[f"to_{a}"]
        cols[f"{here}_body"] = part["from_body"]
        cols[f"{there}_body"] = part["to_body"]
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame(columns=[f"{s}_{a}" for s in ("pre", "post") for a in AXES]
                            + ["pre_body", "post_body"])

    out = pd.concat(frames, ignore_index=True)
    order = [f"pre_{a}" for a in AXES] + [f"post_{a}" for a in AXES]
    out = out[order + ["pre_body", "post_body"]]
    # A pair seen from both ends appears twice; keep whichever bodies each row knows.
    out = (out.groupby(order, dropna=False, as_index=False)
              .agg({"pre_body": "first", "post_body": "first"}))
    return out


def enrich_connections(conns: pd.DataFrame, points: pd.DataFrame,
                       columns: Sequence[str] = ("conf",)) -> pd.DataFrame:
    """Bring per-element columns onto a connections table as ``pre_<col>`` / ``post_<col>``.

    Joined from ``points`` on the coordinate (confidence, ROI, ...). A value for a partner
    that was not fetched stays NaN rather than 0, which would claim zero confidence; NaN
    also fails shader comparisons, so a ``conf < threshold`` filter keeps such lines.
    """
    missing = [c for c in columns if c not in points.columns]
    if missing:
        raise KeyError(
            f"points has no column(s) {', '.join(missing)}; it has "
            f"{', '.join(map(str, points.columns))}")
    lut = points.drop_duplicates(subset=list(AXES)).set_index(list(AXES))[list(columns)]
    out = conns.copy()
    for side in ("pre", "post"):
        keys = pd.MultiIndex.from_arrays(
            [conns[f"{side}_{a}"] for a in AXES], names=list(AXES))
        got = lut.reindex(keys)
        for col in columns:
            out[f"{side}_{col}"] = got[col].to_numpy()
    return out


def match_rate(conns: pd.DataFrame) -> dict[str, Any]:
    """How much of the connectivity the body list actually resolved.

    A low fraction looks the same as a sparse connectome once the tables are written.
    """
    total = int(len(conns))
    if not total:
        return {"pairs": 0, "both_ends": 0, "one_end": 0, "fraction": None}
    both = int((conns["pre_body"].notna() & conns["post_body"].notna()).sum())
    one = int((conns["pre_body"].notna() ^ conns["post_body"].notna()).sum())
    return {"pairs": total, "both_ends": both, "one_end": one,
            "fraction": both / total}


def body_roi_counts(points: pd.DataFrame, *, kind_column: str = "kind") -> pd.DataFrame:
    """Per body and kind, how many synapses fall in each ROI. Long form, lossless.

    A group-by over ``points``, in the shape a tagging rule wants: where a neuron's
    outputs sit often identifies it.
    """
    if "roi" not in points.columns:
        raise KeyError(
            "no 'roi' column: fetch the points with a ROI set (neu-mark points --rois, or "
            "notebook.points(..., rois=[...])) to label each synapse with its neuropil.")
    have = [c for c in ("body", kind_column, "roi") if c in points.columns]
    counts = (points[have].dropna(subset=["roi"])
              .value_counts(dropna=True).rename("synapses").reset_index())
    return counts.sort_values(["body", "synapses"], ascending=[True, False],
                              ignore_index=True)


def keyvalues_to_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """``{body_key: json}`` from a keyvalue instance -> one row per body.

    Records are ragged, so columns are the union of keys, with missing fields null.
    ``instance`` is the neuron's name (``CAm(L)``), not a DVID data instance; the name is
    kept to match neuclease and neuprint. The raw record is kept in a ``json`` column.
    """
    rows = []
    for key, value in values.items():
        if value is None:
            continue
        if not str(key).isdigit():
            # Skip non-body keys (schema documents, config).
            continue
        body = int(key)
        record = dict(value) if isinstance(value, Mapping) else {"value": value}
        stated = record.get("bodyid", record.get("body ID"))
        if stated is not None and int(stated) != body:
            raise ValueError(
                f"body annotation under key {key!r} says bodyid={stated!r}. The key is "
                f"authoritative here, and a mismatch means the instance is inconsistent "
                f"rather than that one of them should be preferred.")
        rows.append({"body": body,
                     **{k: v for k, v in record.items() if k != "bodyid"},
                     "json": value})
    if not rows:
        return pd.DataFrame({"body": pd.Series(dtype=BODY_DTYPE),
                             "json": pd.Series(dtype="object")})

    df = pd.DataFrame(rows)
    df["body"] = df["body"].astype(BODY_DTYPE)
    # Plain strings, not a fixed categorical vocabulary that would reject a new status.
    for col in ("status", "instance", "user", "instance_user", "type"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    lead = ["body", "status", "instance", "type", "user", "instance_user"]
    rest = [c for c in df.columns if c not in lead and c != "json"]
    return df[[c for c in lead if c in df.columns] + rest + ["json"]].sort_values(
        "body", ignore_index=True)


# --------------------------------------------------------------------------- #
# the only coordinate-array builders
# --------------------------------------------------------------------------- #
def positions_zyx(df: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """``(N, 3)`` array of positions in **zyx** order.

    With :func:`positions_xyz`, the only place named coordinate columns become an array.
    """
    return np.stack([df[f"{prefix}{a}"].to_numpy() for a in AXES], axis=-1)


def positions_xyz(df: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """``(N, 3)`` array of positions in **xyz** order, as DVID and precomputed store them."""
    return np.stack([df[f"{prefix}{a}"].to_numpy() for a in reversed(AXES)], axis=-1)
