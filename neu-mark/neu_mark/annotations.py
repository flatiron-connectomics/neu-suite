"""Encoding ``neuroglancer_annotations_v1`` LINE annotations.

One annotation per synaptic connection, a line from the T-bar to the PSD, as published
synapse sources do. A shader can still draw either endpoint alone
(``setLineWidth(0.0); setEndpointMarkerSize(pre, 0.0)`` shows only T-bars).

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

    ``geometry`` is ``(n, floats)`` in xyz, from :func:`neu_mark.tables.positions_xyz`.
    Relationship ids are not part of this encoding; see :func:`encode_single`.
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
    """Invert :func:`encode_group`. Returns ``{"ids", "geometry", "values"}``.

    Used to check a written source against the table it came from.
    """
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
