"""Read a volume's ``neuroglancer_segment_properties`` subresource: per-segment labels,
tags, strings and numbers.

:func:`read_segment_properties` returns plain lists in one read;
:func:`segment_properties_frame` turns that into a DataFrame indexed by segment id, with
each tag facet as its own column. Only the ``inline`` form is readable.

The format pairs every property's values with ``ids`` by position, so a length mismatch
raises rather than attaching values to the wrong segments.
"""

from __future__ import annotations

from typing import Any, Mapping

from .location import read_json

AT_TYPE = "neuroglancer_segment_properties"

__all__ = ["read_segment_properties", "segment_properties_frame", "SegmentPropertiesProblem"]


class SegmentPropertiesProblem(ValueError):
    """A volume has no readable segment properties: none declared, the wrong ``@type``, no
    ``inline`` block, or values not paired one-to-one with ids."""


def properties_dir(volume: str, info: Mapping[str, Any] | None = None) -> str:
    """The subdirectory a volume's ``info`` names for ``segment_properties``. Raises when
    there is none, rather than guessing a name that would find nothing."""
    info = info if info is not None else read_json(volume, "info")
    name = (info or {}).get("segment_properties")
    if not name:
        present = ", ".join(k for k in ("mesh", "skeletons", "segment_properties")
                            if (info or {}).get(k)) or "none"
        raise SegmentPropertiesProblem(
            f"{volume} declares no 'segment_properties' subresource in its info (it has: "
            f"{present}), so there is nothing to read")
    return str(name)


def read_segment_properties(volume: str, *, info: Mapping[str, Any] | None = None) -> dict:
    """Everything a volume's ``segment_properties`` carries, in ONE read.

    Returns ``{"ids", "labels", "label_id", "tags", "numbers", "strings", "vocabulary"}``:
    ids as strings, labels parallel to them (the ids themselves when there is no label
    property), ``label_id`` the label property's own id (``None`` without one), tags
    resolved from indices to strings, and each ``number`` and ``string`` property under its
    own id.
    """
    directory = properties_dir(volume, info)
    where = f"{volume.rstrip('/')}/{directory}"
    document = read_json(where, "info")
    declared = (document or {}).get("@type")
    if declared != AT_TYPE:
        raise SegmentPropertiesProblem(
            f"{where} declares @type {declared!r}, not {AT_TYPE!r} — that is not a segment "
            f"properties source")
    inline = (document or {}).get("inline")
    if not inline:
        raise SegmentPropertiesProblem(
            f"{where} carries no `inline` block; only inline segment properties are readable")

    ids = [str(i) for i in inline.get("ids") or []]
    out: dict[str, Any] = {"ids": ids, "labels": [], "label_id": None, "tags": [],
                           "numbers": {}, "strings": {}, "vocabulary": []}
    for prop in inline.get("properties") or []:
        values = prop.get("values") or []
        if len(values) != len(ids):
            raise SegmentPropertiesProblem(
                f"{where} property {prop.get('id')!r} has {len(values)} values for "
                f"{len(ids)} ids; the format pairs them by position, so a mismatch would "
                f"attach them to the wrong segments")
        kind = prop.get("type")
        if kind == "label":
            out["labels"] = [str(v) for v in values]
            out["label_id"] = str(prop.get("id"))
        elif kind == "tags":
            vocabulary = [str(t) for t in prop.get("tags") or []]
            out["vocabulary"] = vocabulary
            out["tags"] = [tuple(vocabulary[i] for i in row) for row in values]
        elif kind == "number":
            out["numbers"][str(prop.get("id"))] = list(values)
        elif kind == "string":
            out["strings"][str(prop.get("id"))] = [str(v) for v in values]
    if not out["labels"]:
        out["labels"] = list(ids)
    if not out["tags"]:
        out["tags"] = [()] * len(ids)
    return out


def segment_properties_frame(volume_or_read: str | Mapping[str, Any], *,
                             info: Mapping[str, Any] | None = None):
    """Segment properties as a DataFrame indexed by segment id (uint64, named ``body``).

    Columns: the label under its property id; each string and number property; ``tags``
    (a tuple per segment); and the tags split by facet. A tag ``facet:value`` goes to
    column ``tag:<facet>`` — a string (missing where the segment has none), or a tuple
    when some segment carries several values of that facet. A bare tag ``name`` becomes
    the boolean column ``flag:<name>``.

    Accepts a volume location, or what :func:`read_segment_properties` returned. Needs
    pandas (``pip install "neu-vol[pandas]"``).
    """
    import numpy as np
    import pandas as pd

    read = (read_segment_properties(volume_or_read, info=info)
            if isinstance(volume_or_read, str) else volume_or_read)
    index = pd.Index(np.asarray([int(i) for i in read["ids"]], dtype=np.uint64),
                     name="body")
    columns: dict[str, Any] = {}
    if read.get("label_id"):
        columns[read["label_id"]] = pd.array(read["labels"], dtype="string")
    for name, values in read["strings"].items():
        columns[name] = pd.array([v if v != "" else None for v in values], dtype="string")
    for name, values in read["numbers"].items():
        columns[name] = np.asarray(values)
    columns["tags"] = list(read["tags"])

    facets: dict[str, list[list[str]]] = {}
    flags: dict[str, np.ndarray] = {}
    for tag in read["vocabulary"]:
        if ":" in tag:
            facets.setdefault(tag.split(":", 1)[0], [[] for _ in read["ids"]])
        else:
            flags[tag] = np.zeros(len(read["ids"]), dtype=bool)
    for row, tags in enumerate(read["tags"]):
        for tag in tags:
            if ":" in tag:
                facet, value = tag.split(":", 1)
                facets[facet][row].append(value)
            else:
                flags[tag][row] = True
    for facet, rows in facets.items():
        if any(len(v) > 1 for v in rows):
            columns[f"tag:{facet}"] = [tuple(v) for v in rows]
        else:
            columns[f"tag:{facet}"] = pd.array([v[0] if v else None for v in rows],
                                               dtype="string")
    for name, mask in flags.items():
        columns[f"flag:{name}"] = mask
    return pd.DataFrame(columns, index=index)
