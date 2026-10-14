"""Local annotation layers: authored from coordinates, or from a volume's occupancy boxes.

Pure: plain data in, plain data out, no store access. The occupancy analysis itself is
``neu_vol.ops.annotate.labeled_regions``.

Local annotations, carried inline in the state, appear in the viewer's Annotations tab,
where they can be clicked through and stepped with ``[`` / ``]``. A precomputed annotation
source renders but lists nothing there (its frontend class iterates as empty), so it suits
sets too large to carry inline (:mod:`neu_glance.sources`), and a local layer suits a
subset to step through.

Coordinates are zyx throughout and flipped to xyz only in :func:`build_annotation`; a
wrong flip gives a valid annotation in the wrong place.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

# Neuroglancer's annotation type strings (`AnnotationType` in src/annotation/index.ts),
# used by `render` to keep each annotation on one line.
ANNOTATION_TYPES = ("point", "line", "axis_aligned_bounding_box", "ellipsoid",
                    "polyline")

# Unit names to metres. An unknown unit leaves the layer unitless (with a warning), and a
# unitless layer does not align with layers that have physical units.
_UNIT_METRES = {
    "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9,
    "um": 1e-6, "µm": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "m": 1.0, "meter": 1.0, "metre": 1.0,
}

#: Each kind: neuroglancer's ``type`` string and its geometry fields (from
#: ``src/annotation/index.ts``; a box is two corners, not an origin and size). ``polyline``
#: is absent because a CSV row cannot carry a variable number of points.
KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "point": ("point", ("point",)),
    "box": ("axis_aligned_bounding_box", ("pointA", "pointB")),
    "line": ("line", ("pointA", "pointB")),
    "ellipsoid": ("ellipsoid", ("center", "radii")),
}

#: The zyx column names each kind needs, one tuple per geometry field above.
CSV_COLUMNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}

#: Columns any kind may carry. ``segments`` links an annotation to body ids (strings).
OPTIONAL_COLUMNS = ("id", "description", "segments")

#: Prefix of a per-relationship column, ``segments:<name>``, so one annotation can name
#: segments in several segmentation layers. The plain ``segments`` column is the default
#: relationship.
RELATIONSHIP_PREFIX = "segments:"
DEFAULT_RELATIONSHIP = "segments"

#: How many of a kind's leading coordinate groups are positions (an ellipsoid's second
#: group is radii, an extent).
POSITION_GROUPS = {"point": 1, "box": 2, "line": 2, "ellipsoid": 1}


def output_dimensions(voxel_size_zyx, units: str | None) -> tuple[dict, str | None]:
    """``{"x": [scale, "m"], ...}`` for the layer, and a warning if units are unknown.

    Declared on the layer, so it can be pasted into any state of the same volume.
    """
    if voxel_size_zyx is None:
        return ({d: [1, ""] for d in "xyz"},
                "no voxel size recorded: the layer is unitless, so it will only line "
                "up if the viewer's own dimensions are voxels. Pass --voxel-size.")
    metres = _UNIT_METRES.get(str(units or "nm").lower())
    if metres is None:
        return ({d: [1, ""] for d in "xyz"},
                f"unrecognised unit {units!r}: the layer is unitless and may not line "
                f"up. Pass --voxel-size to state the size in nm.")
    xyz = tuple(voxel_size_zyx)[::-1]
    return {d: [float(v) * metres, "m"] for d, v in zip("xyz", xyz)}, None


def positions(record: dict):
    """The position tuples of a record, skipping any that are extents (zyx)."""
    return record["coords"][:POSITION_GROUPS[record["kind"]]]


def read_annotation_csv(text: str, kind: str, *, source: str = "<csv>") -> list[dict]:
    """Rows of a CSV as annotation records: ``{"kind", "coords", id?, ...}``.

    ``coords`` is one zyx tuple per geometry field of ``kind`` (one for a point, two for a
    box), in the file's units; :func:`rescale` converts them. Columns are found by name,
    not position, and unknown columns are ignored.
    """
    import csv
    import io

    if kind not in CSV_COLUMNS:
        raise ValueError(f"unknown annotation kind {kind!r}; known: {sorted(KINDS)}")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{source}: no data rows (a header naming the columns is "
                         f"required)")
    needed = [c for group in CSV_COLUMNS[kind] for c in group]
    present = {(k or "").strip() for k in rows[0]}
    missing = [c for c in needed if c not in present]
    if missing:
        raise ValueError(
            f"{source}: {kind} needs column(s) {', '.join(missing)} — expected "
            f"{', '.join(needed)}{' plus any of ' + ', '.join(OPTIONAL_COLUMNS)}. "
            f"Found: {', '.join(sorted(present)) or '(no header)'}")

    records = []
    for n, row in enumerate(rows, start=2):        # row 1 is the header
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        try:
            coords = tuple(tuple(float(clean[c]) for c in group)
                           for group in CSV_COLUMNS[kind])
        except ValueError as e:
            raise ValueError(f"{source} line {n}: {e}") from None
        rec: dict[str, Any] = {"kind": kind, "coords": coords}
        if clean.get("id"):
            rec["id"] = clean["id"]
        if clean.get("description"):
            rec["description"] = clean["description"]
        if clean.get("segments"):
            rec["segments"] = _parse_segments(clean["segments"], f"{source} line {n}")
        rels = {}
        for key, value in clean.items():
            if not key.startswith(RELATIONSHIP_PREFIX):
                continue
            name = key[len(RELATIONSHIP_PREFIX):].strip()
            if not name:
                raise ValueError(f"{source}: column {key!r} names no relationship; "
                                 f"write it as segments:<name>, e.g. segments:gt")
            # Kept even when empty: the column's presence declares the relationship.
            rels[name] = _parse_segments(value, f"{source} line {n}") if value else []
        if rels:
            rec["relationships"] = rels
        records.append(rec)
    return records


def record_relationships(records: Sequence[dict]) -> list[str]:
    """Every relationship the records name, in first-seen order. ``segments`` stands for
    the plain column."""
    seen: dict[str, None] = {}
    for r in records:
        if r.get("segments"):
            seen.setdefault(DEFAULT_RELATIONSHIP, None)
        for name in r.get("relationships", {}):
            seen.setdefault(name, None)
    return list(seen)


def _parse_segments(value: str, where: str) -> list[str]:
    """Segment ids, however they were separated. Kept as strings, as the state wants."""
    ids = [p for p in value.replace("|", " ").replace(",", " ").split() if p]
    for i in ids:
        # A float means a spreadsheet turned a long id into 1.23e+18.
        if not i.isdigit():
            raise ValueError(f"{where}: segment id {i!r} is not a whole number")
    return ids


def rescale(records: list[dict], factor_zyx: Sequence[float]) -> list[dict]:
    """Multiply every coordinate of every record by ``factor_zyx`` (per axis).

    Covers voxels at another level to level-0 voxels, and nm to level-0 voxels. Radii scale
    like positions.
    """
    out = []
    for r in records:
        out.append({**r, "coords": tuple(tuple(c * f for c, f in zip(group, factor_zyx))
                                         for group in r["coords"])})
    return out


def build_annotation(record: dict, ident: str,
                     relationships: Sequence[str] | None = None) -> dict:
    """One annotation object, zyx in and xyz out.

    A box's corners are sorted per axis, since a reversed pair renders as nothing; a line's
    order is its direction and is kept.

    ``relationships`` is the layer's declared order, and ``segments`` gets one id list per
    relationship in that order: neuroglancer matches them by position, not name. Without
    it, only the default relationship is written.
    """
    kind = record["kind"]
    type_name, fields = KINDS[kind]
    coords = list(record["coords"])
    if kind == "box":
        lo, hi = coords
        coords = [tuple(min(a, b) for a, b in zip(lo, hi)),
                  tuple(max(a, b) for a, b in zip(lo, hi))]
    ann: dict[str, Any] = {"type": type_name, "id": ident}
    if record.get("description"):
        ann["description"] = record["description"]
    for field, group in zip(fields, coords):
        ann[field] = [float(v) for v in tuple(group)[::-1]]          # zyx -> xyz
    if relationships is not None:
        rels = record.get("relationships", {})
        per = [[str(s) for s in (rels.get(name)
                                 or (record.get("segments") if name == DEFAULT_RELATIONSHIP
                                     else None)
                                 or [])]
               for name in relationships]
        if any(per):
            ann["segments"] = per
    elif record.get("segments"):
        # One list for the default relationship. Strings: a uint64 id does not survive
        # a JSON number.
        ann["segments"] = [[str(s) for s in record["segments"]]]
    return ann


def local_layer(annotations: list[dict], dims: dict, *, name: str = "annotations",
                color: str = "#ffee00", links: dict[str, str] | None = None) -> dict:
    """The layer envelope for inline (``local://annotations``) annotations.

    ``links`` maps each relationship name to the segmentation layer it refers to. Without
    it the segment ids are stored but inert: neuroglancer does not know which layer they
    belong to. Its order must match the ``relationships`` given to :func:`build_annotation`.
    """
    layer = {
        "type": "annotation",
        "name": name,
        # opens the layer panel straight onto the clickable list
        "tab": "annotations",
        "source": {"url": "local://annotations",
                   "transform": {"outputDimensions": dims}},
        "annotationColor": color,
        "annotations": annotations,
    }
    if links:
        layer["annotationRelationships"] = [str(k) for k in links]
        layer["linkedSegmentationLayer"] = {str(k): str(v) for k, v in links.items()}
    return layer


def boxes_layer(regions: list[dict], dims: dict, *, name: str = "regions",
                color: str = "#ffee00", kind: str = "box",
                label: str = "r") -> dict:
    """The occupancy layer: one annotation per region of ``labeled_regions``.

    ``regions`` are zyx ``{"lo", "hi"}``. (:func:`neu_glance.sources.annotation_layer` is
    the unrelated layer for a precomputed annotation source.)
    """
    annotations = []
    for i, r in enumerate(regions):
        lo, hi = tuple(r["lo"]), tuple(r["hi"])
        ident = f"{label}{i:02d}"
        extent = "x".join(str(hi[a] - lo[a]) for a in range(3))
        note = f"{ident}  {extent} vox"
        if r.get("n_labels") is not None:
            note += f"  {r['n_labels']} labels"
        record = ({"kind": "point",
                   "coords": (tuple((lo[a] + hi[a]) / 2 for a in range(3)),)}
                  if kind == "point" else {"kind": "box", "coords": (lo, hi)})
        annotations.append(build_annotation({**record, "description": note}, ident))
    return local_layer(annotations, dims, name=name, color=color)


def render(obj: Any) -> str:
    """``json.dumps`` with each annotation kept on a single line.

    For pasting into neuroglancer's JSON editor, where a person reads it.
    """
    holes: dict[str, str] = {}

    def fold(o):
        if isinstance(o, dict):
            if o.get("type") in ANNOTATION_TYPES:
                # A plain ASCII placeholder: json.dumps would escape a control character,
                # and the substitution below would then miss it.
                key = f"__neu_glance_annotation_{len(holes)}__"
                holes[key] = json.dumps(o, separators=(", ", ": "))
                return key
            return {k: fold(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fold(v) for v in o]
        return o

    text = json.dumps(fold(obj), indent=1)
    for key, line in holes.items():
        text = text.replace(f'"{key}"', line)
    return text
