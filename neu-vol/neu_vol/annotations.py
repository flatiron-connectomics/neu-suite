"""The ``neuroglancer_annotations_v1`` format: encoding records, and reading a source back.

The encoders serve writers (neu-mark's synapse source writes one LINE per connection, T-bar
to PSD); the readers serve anything consuming a published source: one annotation by id
(:func:`read_annotation`), one segment's annotations through a relationship index
(:func:`read_related`), or the whole source as columns (:func:`read_all`).

Byte layout, little-endian, checked against a published male-CNS source. Per annotation:
geometry (for LINE, two ``float32`` xyz triples), then properties in descending size
(4-, 2-, then 1-byte), then zero padding to 4 bytes. The single-annotation (``by_id``)
encoding then adds, per relationship, a ``uint32`` count and that many ``uint64`` ids; the
multi-annotation encoding instead starts with a ``uint64`` count and ends with all the ids.

**Property order matters:** a reader infers offsets from the declared order, and a file
written in any other order parses into misaligned values with no error, so
:func:`sort_properties` fixes it. Properties cannot be uint64; full body ids go in
relationships, with a truncated ``uint32`` copy as a property if a shader needs one.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

AT_TYPE = "neuroglancer_annotations_v1"

#: Lowercase, as published sources write it (the spec document says "LINE").
LINE = "line"
POINT = "point"

#: Bytes per property type, and the numpy dtype used to encode it. ``rgb``/``rgba`` are in
#: the format but unused here.
PROPERTY_TYPES: dict[str, tuple[int, str]] = {
    "float32": (4, "<f4"), "uint32": (4, "<u4"), "int32": (4, "<i4"),
    "uint16": (2, "<u2"), "int16": (2, "<i2"),
    "uint8": (1, "u1"), "int8": (1, "i1"),
}

#: Geometry floats per annotation type.
GEOMETRY_FLOATS = {LINE: 6, POINT: 3}


def sort_properties(properties: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Properties in the order the format requires: descending size, the caller's order
    kept within a size."""
    known = [p for p in properties]
    for p in known:
        if p.get("type") not in PROPERTY_TYPES:
            raise ValueError(
                f"property {p.get('id')!r} has type {p.get('type')!r}; the format allows "
                f"{', '.join(sorted(PROPERTY_TYPES))} (notably NOT uint64 — store a wide id "
                f"as a relationship, and a truncated copy as uint32 if a shader needs it)")
    return sorted((dict(p) for p in known),
                  key=lambda p: -PROPERTY_TYPES[p["type"]][0])


def record_size(annotation_type: str, properties: Sequence[Mapping[str, Any]]) -> int:
    """Bytes per annotation in a group: geometry + properties + padding to 4 bytes."""
    geometry = 4 * GEOMETRY_FLOATS[annotation_type]
    props = sum(PROPERTY_TYPES[p["type"]][0] for p in properties)
    return geometry + props + (-(geometry + props) % 4)


def _property_columns(properties: Sequence[Mapping[str, Any]],
                      values: Mapping[str, Any], n: int) -> list[np.ndarray]:
    columns = []
    for p in properties:
        if p["id"] not in values:
            raise KeyError(f"no values supplied for property {p['id']!r}")
        _size, dtype = PROPERTY_TYPES[p["type"]]
        column = np.asarray(values[p["id"]])
        if column.shape != (n,):
            raise ValueError(
                f"property {p['id']!r} has {column.shape} values, expected ({n},)")
        columns.append(column.astype(dtype, copy=False))
    return columns


def encode_group(geometry: np.ndarray, ids: Sequence[int], *,
                 annotation_type: str = LINE,
                 properties: Sequence[Mapping[str, Any]] = (),
                 values: Mapping[str, Any] | None = None) -> bytes:
    """The multiple-annotation encoding: a spatial cell, or one relationship's annotations.

    ``geometry`` is ``(n, floats)`` in xyz. Relationship ids are not part of this encoding;
    see :func:`encode_single`.
    """
    props = list(properties)
    n = len(ids)
    floats = GEOMETRY_FLOATS[annotation_type]
    geometry = np.asarray(geometry, dtype="<f4")
    if geometry.shape != (n, floats):
        raise ValueError(f"geometry is {geometry.shape}, expected ({n}, {floats}) for "
                         f"{annotation_type!r}")

    stride = record_size(annotation_type, props)
    body = np.zeros((n, stride), dtype="u1")
    body[:, :4 * floats] = geometry.reshape(n, -1).view("u1")

    offset = 4 * floats
    for p, column in zip(props, _property_columns(props, values or {}, n)):
        size = PROPERTY_TYPES[p["type"]][0]
        body[:, offset:offset + size] = column.view("u1").reshape(n, size)
        offset += size
    # Bytes from `offset` to `stride` stay zero: the padding.

    return (struct.pack("<Q", n) + body.tobytes()
            + np.asarray(ids, dtype="<u8").tobytes())


def encode_single(geometry: Sequence[float], *, annotation_type: str = LINE,
                  properties: Sequence[Mapping[str, Any]] = (),
                  values: Mapping[str, Any] | None = None,
                  relationships: Sequence[Sequence[int]] = ()) -> bytes:
    """The single-annotation encoding, for the ``by_id`` index.

    ``relationships`` is one sequence of segment ids per declared relationship, in the
    declared order.
    """
    props = list(properties)
    floats = GEOMETRY_FLOATS[annotation_type]
    geom = np.asarray(geometry, dtype="<f4").reshape(-1)
    if geom.size != floats:
        raise ValueError(f"geometry has {geom.size} floats, expected {floats}")

    out = bytearray(geom.tobytes())
    for p in props:
        _size, dtype = PROPERTY_TYPES[p["type"]]
        if p["id"] not in (values or {}):
            raise KeyError(f"no value supplied for property {p['id']!r}")
        out += np.asarray((values or {})[p["id"]], dtype=dtype).tobytes()
    out += b"\x00" * (-len(out) % 4)

    for related in relationships:
        related = list(related)
        out += struct.pack("<I", len(related))
        out += np.asarray(related, dtype="<u8").tobytes()
    return bytes(out)


def decode_group(raw: bytes, *, annotation_type: str = LINE,
                 properties: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Invert :func:`encode_group`. Returns ``{"ids", "geometry", "values"}``."""
    props = list(properties)
    floats = GEOMETRY_FLOATS[annotation_type]
    stride = record_size(annotation_type, props)
    (n,) = struct.unpack_from("<Q", raw, 0)
    expected = 8 + n * stride + n * 8
    if len(raw) != expected:
        raise ValueError(f"{len(raw)} bytes for {n} annotations at stride {stride}; "
                         f"expected {expected}")
    body = np.frombuffer(raw, dtype="u1", count=n * stride, offset=8).reshape(n, stride)
    ids = np.frombuffer(raw, dtype="<u8", count=n, offset=8 + n * stride)
    geometry = body[:, :4 * floats].copy().view("<f4").reshape(n, floats)

    values, offset = {}, 4 * floats
    for p in props:
        size, dtype = PROPERTY_TYPES[p["type"]]
        values[p["id"]] = body[:, offset:offset + size].copy().view(dtype).reshape(n)
        offset += size
    return {"ids": ids, "geometry": geometry, "values": values}


def decode_single(raw: bytes, *, annotation_type: str = LINE,
                  properties: Sequence[Mapping[str, Any]] = (),
                  n_relationships: int = 0) -> dict[str, Any]:
    """Invert :func:`encode_single`. Returns ``{"geometry", "values", "relationships"}``."""
    props = list(properties)
    floats = GEOMETRY_FLOATS[annotation_type]
    geometry = np.frombuffer(raw, dtype="<f4", count=floats, offset=0)

    values, offset = {}, 4 * floats
    for p in props:
        size, dtype = PROPERTY_TYPES[p["type"]]
        values[p["id"]] = np.frombuffer(raw, dtype=dtype, count=1, offset=offset)[0]
        offset += size
    offset += -offset % 4

    relationships = []
    for _ in range(n_relationships):
        (count,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        relationships.append(
            np.frombuffer(raw, dtype="<u8", count=count, offset=offset).tolist())
        offset += 8 * count
    if offset != len(raw):
        raise ValueError(f"{len(raw) - offset} trailing bytes after {n_relationships} "
                         f"relationships; the record is not the shape declared")
    return {"geometry": geometry, "values": values, "relationships": relationships}


def build_info(*, lower_bound: Sequence[float], upper_bound: Sequence[float],
               voxel_size_xyz: Sequence[float], annotation_type: str = LINE,
               properties: Sequence[Mapping[str, Any]] = (),
               relationships: Sequence[Mapping[str, Any]] = (),
               by_id: Mapping[str, Any] | None = None,
               spatial: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The annotation source's ``info``.

    Every member is required (``relationships`` and ``spatial`` may be empty).
    ``dimensions`` is written x, y, z, matching the stored geometry, in metres (8 nm is
    ``8e-9``).
    """
    props = sort_properties(properties)
    return {
        "@type": AT_TYPE,
        "dimensions": {axis: [float(size) * 1e-9, "m"]
                       for axis, size in zip("xyz", voxel_size_xyz)},
        "lower_bound": [float(v) for v in lower_bound],
        "upper_bound": [float(v) for v in upper_bound],
        "annotation_type": annotation_type,
        "properties": props,
        "relationships": [dict(r) for r in relationships],
        "by_id": dict(by_id or {"key": "by_id"}),
        "spatial": [dict(s) for s in spatial],
    }


def enum_property(prop_id: str, labels: Sequence[str], *, dtype: str = "int16",
                  description: str | None = None) -> dict[str, Any]:
    """A property whose integer values index ``labels``.

    By convention index 0 means "none" (``<unspecified>``), so a shader can test for it.
    """
    return {"id": prop_id, "type": dtype,
            **({"description": description} if description else {}),
            "enum_values": list(range(len(labels))),
            "enum_labels": [str(v) for v in labels]}


# --------------------------------------------------------------------------- #
# reading a source
# --------------------------------------------------------------------------- #
def read_info(src: str) -> dict[str, Any]:
    """A source's ``info``, refused unless it declares this format."""
    from .location import read_json

    info = read_json(src, "info")
    if info is None:
        raise FileNotFoundError(f"no annotation info at {str(src).rstrip('/')}")
    if info.get("@type") != AT_TYPE:
        raise ValueError(f"{src} declares @type {info.get('@type')!r}, not {AT_TYPE!r}")
    return info


def _read_entry(src: str, index: Mapping[str, Any], chunk_id: int) -> bytes | None:
    from . import sharded
    from .location import read_bytes

    if "sharding" in index:
        return sharded.read_one(src, index["sharding"], chunk_id, index["key"])
    return read_bytes(src, index["key"], str(int(chunk_id)))


def read_annotation(src: str, annotation_id: int, *, info: Mapping[str, Any] | None = None
                    ) -> dict[str, Any] | None:
    """One annotation from the ``by_id`` index, decoded; ``None`` when the id is absent."""
    info = info if info is not None else read_info(src)
    raw = _read_entry(src, {"key": "by_id", **info["by_id"]}, annotation_id)
    if raw is None:
        return None
    return decode_single(raw, annotation_type=info["annotation_type"],
                         properties=info["properties"],
                         n_relationships=len(info["relationships"]))


def read_related(src: str, relationship: str, segment: int, *,
                 info: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """One segment's annotations from a relationship index, decoded — a viewer's keyed
    fetch. ``None`` when the segment has none."""
    info = info if info is not None else read_info(src)
    rel = next((r for r in info["relationships"] if r["id"] == relationship), None)
    if rel is None:
        raise KeyError(f"no relationship {relationship!r}; this source declares "
                       + ", ".join(repr(r["id"]) for r in info["relationships"]))
    raw = _read_entry(src, rel, segment)
    if raw is None:
        return None
    return decode_group(raw, annotation_type=info["annotation_type"],
                        properties=info["properties"])


def _by_id_entries(src: str, info: Mapping[str, Any], threads: int):
    from . import sharded
    from .location import list_keys, read_bytes

    index = {"key": "by_id", **info["by_id"]}
    if "sharding" in index:
        yield from sharded.read_all(src, index["sharding"], index["key"], threads=threads)
        return
    for name in list_keys(src, index["key"]):
        raw = read_bytes(src, index["key"], name)
        if raw is not None:
            yield int(name), raw


def read_all(src: str, *, info: Mapping[str, Any] | None = None,
             threads: int = 8) -> dict[str, Any]:
    """Every annotation in a source, as columns, from its ``by_id`` index.

    Returns ``{"ids", "geometry", "values", "relationships"}``: ``ids`` uint64 ``(n,)``,
    ``geometry`` float32 ``(n, floats)`` in xyz, ``values`` one array per property, and
    ``relationships`` ``{name: (offsets, segments)}`` — annotation ``i`` relates to
    ``segments[offsets[i]:offsets[i + 1]]``, uint64, so an empty relationship is
    representable. ``by_id`` is the index that carries relationships at full uint64
    width; the spatial index has only the properties. Rows follow the index's storage
    order, not the ids.
    """
    info = info if info is not None else read_info(src)
    props = list(info["properties"])
    atype = info["annotation_type"]
    floats = GEOMETRY_FLOATS[atype]
    stride = record_size(atype, props)
    names = [r["id"] for r in info["relationships"]]

    ids: list[int] = []
    prefixes = bytearray()
    counts: list[list[int]] = [[] for _ in names]
    related: list[list[bytes]] = [[] for _ in names]
    for chunk_id, raw in _by_id_entries(src, info, threads):
        ids.append(chunk_id)
        prefixes += raw[:stride]
        offset = stride
        for r in range(len(names)):
            (count,) = struct.unpack_from("<I", raw, offset)
            offset += 4
            counts[r].append(count)
            related[r].append(raw[offset:offset + 8 * count])
            offset += 8 * count
        if offset != len(raw):
            raise ValueError(f"annotation {chunk_id}: {len(raw) - offset} bytes left after "
                             f"{len(names)} relationships; the record is not the shape "
                             f"its info declares")

    n = len(ids)
    body = np.frombuffer(bytes(prefixes), dtype="u1").reshape(n, stride)
    values, offset = {}, 4 * floats
    for p in props:
        size, dtype = PROPERTY_TYPES[p["type"]]
        values[p["id"]] = body[:, offset:offset + size].copy().view(dtype).reshape(n)
        offset += size
    relationships = {}
    for r, name in enumerate(names):
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts[r], out=offsets[1:])
        relationships[name] = (offsets, np.frombuffer(b"".join(related[r]), dtype="<u8"))
    return {"ids": np.asarray(ids, dtype=np.uint64),
            "geometry": body[:, :4 * floats].copy().view("<f4").reshape(n, floats),
            "values": values, "relationships": relationships}
