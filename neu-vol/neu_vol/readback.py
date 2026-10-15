"""Read a body's mesh and skeleton out of a published precomputed volume, local or remote.

The writers are in neu-morpho (``neu_morpho.precomputed``); reading needs no meshing
stack, so it is here, where every package that views or measures a volume can use it.
Formats: ``neuroglancer_multilod_draco`` and ``neuroglancer_legacy_mesh`` meshes,
``neuroglancer_skeletons``, each sharded or not.

Coordinates come back in physical nm: each subresource's ``info`` ``transform`` is
applied. DracoPy (meshes) and osteoid (skeletons) are imported on first use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import location

__all__ = ["read_body_skeleton", "read_body_mesh", "mesh_body_ids", "mesh_transform",
           "skeleton_transform", "UnsupportedSubresource"]


class UnsupportedSubresource(ValueError):
    """A mesh or skeleton subresource is in a form this module cannot decode.

    Distinct from a body with no mesh, which is an ordinary ``None``: anything batching
    reads should let this propagate rather than count it as a missing body.
    """


def read_body_skeleton(volume: str, body_id: int, skeleton_dir: str = "skeleton",
                       *, require_radii: bool = True):
    """``(vertices_xyz_nm, edges, radii_nm)`` for one body, or ``None`` if absent.

    Vertices come back xyz, the order the format stores, matching :func:`read_body_mesh`
    so the two overlay without a flip; the rest of the suite holds zyx in memory. They
    are physical nm, mapped through the skeleton ``info``'s ``transform`` as the mesh
    reader maps its own. Sharded and unsharded skeletons are both read.

    Raises if the skeleton has no usable ``radius`` attribute. With none declared,
    osteoid returns one ``-1`` per vertex, which passes a length check and renders as an
    inverted tube, so negative radii are refused. ``require_radii=False`` returns
    ``radii_nm=None`` instead, for a source that publishes centrelines only.
    """
    from osteoid import Skeleton

    info = _subresource_info(volume, skeleton_dir)
    _check_supported(info, volume, skeleton_dir, "skeleton")
    sharding = info.get("sharding")
    if sharding is not None:
        blob = _shard_reader(volume, skeleton_dir, sharding).read(int(body_id))
    else:
        blob = location.read_bytes(volume, skeleton_dir, str(int(body_id)))
    if blob is None:
        return None
    skel = Skeleton.from_precomputed(blob, segid=int(body_id))
    vertices = np.asarray(skel.vertices, dtype=float)
    matrix = _transform_matrix(info)
    if matrix is not None:
        vertices = vertices @ matrix[:, :3].T + matrix[:, 3]
    raw = getattr(skel, "radii", None)
    radii = None if raw is None else np.asarray(raw, dtype=float)
    edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)
    absent = (radii is None or len(radii) != len(vertices)
              or (len(radii) and not np.isfinite(radii).all())
              or (len(radii) and radii.min() < 0))
    if absent and not require_radii:
        return vertices, edges, None
    if radii is None or len(radii) != len(vertices):
        raise ValueError(
            f"body {body_id}: skeleton has no usable 'radius' attribute "
            f"({0 if radii is None else len(radii)} radii for "
            f"{len(vertices)} vertices)")
    if len(radii) and not np.isfinite(radii).all():
        raise ValueError(f"body {body_id}: skeleton has non-finite radii")
    if len(radii) and radii.min() < 0:
        raise ValueError(
            f"body {body_id}: skeleton has negative radii (min {radii.min():g}). "
            f"All -1 means {volume}/{skeleton_dir}/info declares no 'radius' "
            f"vertex attribute, so there is nothing to read. Pass "
            f"require_radii=False for a source that publishes centrelines only.")
    scale = _radius_scale(matrix, volume, skeleton_dir, require_radii)
    if scale is None:
        return vertices, edges, None
    return vertices, edges, radii * scale


def _radius_scale(matrix, volume: str, skeleton_dir: str, require_radii: bool):
    """The factor carrying a skeleton's radii into nm alongside its vertices.

    A radius is a length, so only a uniform linear scale carries it. Returns that factor
    (``1.0`` for identity), or ``None`` meaning the radii must be dropped: under a
    non-uniform transform there is no single radius, so this raises when radii are
    required and returns ``None`` otherwise.
    """
    if matrix is None:
        return 1.0
    linear = matrix[:, :3]
    scales = np.sqrt((linear ** 2).sum(axis=0))
    if np.allclose(linear, np.diag(scales)) and np.allclose(scales, scales[0]):
        return float(scales[0])
    if not require_radii:
        return None
    raise ValueError(
        f"{volume}/{skeleton_dir}/info declares a non-uniform transform "
        f"{matrix.tolist()}, so its radii have no single value in nm. Pass "
        f"require_radii=False for the centreline, and scale the radii yourself if the "
        f"anisotropy is intended.")


#: One :class:`~neu_vol.sharded.ShardReader` per (volume, subresource directory), keeping
#: its parsed indexes, so a batch of bodies from one shard does not re-read them (8 bodies:
#: 0.70 s with a reader each, 0.10 s sharing one). A race builds a reader twice, harmlessly.
_SHARD_READERS: dict[tuple, Any] = {}


def _shard_reader(volume: str, subdir: str, sharding):
    """A cached reader for one sharded subresource — mesh or skeleton alike."""
    import json

    from .sharded import ShardReader

    cache_key = (str(volume), str(subdir), json.dumps(dict(sharding), sort_keys=True))
    reader = _SHARD_READERS.get(cache_key)
    if reader is None:
        reader = _SHARD_READERS[cache_key] = ShardReader(volume, sharding, subdir)
    return reader


def _subresource_info(volume: str, subdir: str) -> dict:
    """A subresource's parsed ``info``; ``{}`` when there is none."""
    import json

    raw = location.read_bytes(volume, subdir, "info")
    return json.loads(raw) if raw else {}


#: The ``@type``\ s each reader here understands. ``neuroglancer_legacy_mesh`` is the
#: single-resolution format published ROI and neuropil sources often use.
_SUPPORTED_TYPES = {"mesh": ("neuroglancer_multilod_draco", "neuroglancer_legacy_mesh"),
                    "skeleton": ("neuroglancer_skeletons",)}

#: The single-resolution mesh format, handled by :func:`_read_legacy_mesh`.
LEGACY_MESH = "neuroglancer_legacy_mesh"


def _check_supported(info, volume: str, subdir: str, kind: str) -> None:
    """Refuse a subresource whose declared ``@type`` this module cannot read.

    Otherwise every body would read as absent, indistinguishable from a volume with no
    meshes. An ``info`` with no ``@type``, or none at all, is allowed through.
    """
    declared = (info or {}).get("@type")
    expected = _SUPPORTED_TYPES[kind]
    if declared is None or declared in expected:
        return
    understood = " or ".join(repr(e) for e in expected)
    raise UnsupportedSubresource(
        f"{volume}/{subdir}/info declares '@type' {declared!r}, but this reader "
        f"understands only {understood}. Refusing rather than reporting every "
        f"body absent, which is what reading it with the wrong format would look like.")


def _read_legacy_mesh(volume: str, body_id: int, mesh_dir: str):
    """``(vertices_xyz, faces)`` for a ``neuroglancer_legacy_mesh`` body, or ``None``.

    ``<body>:0`` is a JSON manifest naming fragment files relative to the mesh directory;
    each fragment is a little-endian ``uint32`` vertex count, that many ``float32`` xyz
    triples, then ``uint32`` triangle indices. Fragments are concatenated with faces
    rebased. A fragment shorter than its vertex count requires is refused as truncated.
    """
    import json

    manifest = location.read_bytes(volume, mesh_dir, f"{int(body_id)}:0")
    if manifest is None:
        return None
    fragments = (json.loads(manifest) or {}).get("fragments") or []

    verts, faces, offset = [], [], 0
    for name in fragments:
        blob = location.read_bytes(volume, mesh_dir, str(name))
        if blob is None:
            raise ValueError(
                f"body {body_id}: manifest names fragment {name!r}, which is not in "
                f"{volume}/{mesh_dir}. The manifest and the store disagree")
        count = int(np.frombuffer(blob, "<u4", 1)[0])
        needed = 4 + 12 * count
        if len(blob) < needed:
            raise ValueError(
                f"body {body_id}: fragment {name!r} declares {count} vertices, which needs "
                f"{needed} bytes, but it is {len(blob)} — truncated")
        v = np.frombuffer(blob, "<f4", count * 3, 4).reshape(-1, 3).astype(float)
        indices = np.frombuffer(blob, "<u4", (len(blob) - needed) // 4, needed)
        verts.append(v)
        faces.append(indices.reshape(-1, 3).astype(np.int64) + offset)
        offset += count
    if not verts:
        return None
    return np.concatenate(verts), np.concatenate(faces)


def _transform_matrix(info):
    """A subresource ``transform`` as a ``(3, 4)`` matrix, or ``None`` when it is identity,
    so the caller skips the multiply."""
    values = (info or {}).get("transform")
    if values is None:
        return None
    matrix = np.asarray(values, dtype=float).reshape(3, 4)
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    return None if np.array_equal(matrix, identity) else matrix


def _parse_manifest(manifest: bytes):
    """The multi-resolution mesh manifest, split into the pieces a re-emit needs.

    Layout: ``chunk_shape[3]`` and ``grid_origin[3]`` as float32, ``num_lods`` as uint32,
    then ``lod_scales[num_lods]`` and ``vertex_offsets[num_lods][3]`` as float32,
    ``num_fragments_per_lod[num_lods]`` as uint32, and finally per LOD a block of
    ``fragment_positions[3][n]`` followed by ``fragment_offsets[n]`` — the latter being
    byte SIZES despite the name.
    """
    num_lods = int(np.frombuffer(manifest, "<u4", 1, 24)[0])
    scales_end = 28 + num_lods * 4
    offsets_end = scales_end + num_lods * 3 * 4
    counts = np.frombuffer(manifest, "<u4", num_lods, offsets_end)

    pos = offsets_end + num_lods * 4
    blocks, sizes = [], []
    for count in counts:
        start = pos
        pos += int(count) * 3 * 4
        sizes.append(np.frombuffer(manifest, "<u4", int(count), pos).astype(np.int64))
        pos += int(count) * 4
        blocks.append(manifest[start:pos])
    return {
        "head": manifest[:24],
        "num_lods": num_lods,
        "lod_scales": manifest[28:scales_end],
        "vertex_offsets": manifest[scales_end:offsets_end],
        "counts": counts,
        "blocks": blocks,
        "sizes": sizes,
    }


def _single_lod_manifest(parsed, want: int) -> bytes:
    """Re-emit the manifest declaring only LOD ``want``, with the earlier LODs empty.

    Kept, not renumbered: a fragment's cell size is ``chunk_shape * 2**lod`` from its
    position in the manifest, so relabelling LOD 3 as 0 would shrink the mesh 8x.
    """
    counts = np.zeros(want + 1, dtype="<u4")
    counts[want] = parsed["counts"][want]
    return b"".join([
        parsed["head"],
        np.uint32(want + 1).tobytes(),
        parsed["lod_scales"][:(want + 1) * 4],
        parsed["vertex_offsets"][:(want + 1) * 3 * 4],
        counts.tobytes(),
        parsed["blocks"][want],
    ])


def _sharded_object(volume: str, body_id: int, mesh_dir: str, sharding, lod):
    """``(fragment_data, manifest, lod)`` for one body and one LOD, or ``None``.

    Only the manifest is an indexed shard entry; the fragment data sits immediately
    before it, at ``[manifest_offset - total_size, manifest_offset)`` with LODs in order,
    which is why this needs :class:`~neu_vol.sharded.ShardReader`. Only the requested LOD
    is fetched (the coarsest is typically under 1% of the bytes), and the manifest is
    re-emitted to match.
    """
    reader = _shard_reader(volume, mesh_dir, sharding)
    found = reader.locate(int(body_id))
    if found is None:
        return None
    shard, offset, _size = found
    manifest = reader.read(int(body_id))
    if manifest is None:
        return None

    parsed = _parse_manifest(manifest)
    present = [i for i, c in enumerate(parsed["counts"]) if int(c)]
    if not present:
        return None
    want = present[-1] if lod is None else int(lod)
    if want not in present:
        raise ValueError(f"body {body_id}: LOD {want} not present; have {present}")

    per_lod = [int(s.sum()) for s in parsed["sizes"]]
    total = sum(per_lod)
    start = offset - total + sum(per_lod[:want])

    # Checked first: a negative start would surface as an obscure byte-range error.
    if total > offset:
        raise ValueError(
            f"body {body_id}: manifest claims {total} bytes of mesh fragment data but "
            f"its own entry starts at byte {offset} of {shard}, so the data cannot "
            f"precede it. {volume}/{mesh_dir} is not a well-formed sharded mesh.")

    data = reader.read_range(shard, start, start + per_lod[want])
    if data is None or len(data) != per_lod[want]:
        raise ValueError(
            f"body {body_id}: LOD {want} fragment data is "
            f"{0 if data is None else len(data)} bytes, manifest says {per_lod[want]}. "
            f"The manifest and the data it points at disagree, so {volume}/{mesh_dir} "
            f"is not a well-formed sharded mesh.")
    return data, _single_lod_manifest(parsed, want), want


def read_body_mesh(volume: str, body_id: int, lod: int | None = None,
                   mesh_dir: str = "mesh"):
    """``(vertices_xyz_nm, faces, lod)`` for one LOD, or ``None`` if absent.

    ``lod`` defaults to the coarsest present, the cheapest to draw; pass 0 for full
    detail. The LOD's octree fragments are concatenated with face indices rebased.
    Sharded and unsharded meshes are both read, by the ``sharding`` key of the mesh
    ``info``; a sharded read fetches only the requested LOD.

    Vertices are always physical nm: stored coordinates go through the mesh ``info``'s
    ``transform``, with no opt-out. neu-morpho writes identity; a foreign source may not
    (one published source declares ``diag(16)``). :func:`mesh_transform` reports it.
    """
    info = _subresource_info(volume, mesh_dir)
    _check_supported(info, volume, mesh_dir, "mesh")

    if info.get("@type") == LEGACY_MESH:
        # Single-resolution: only LOD 0 exists.
        if lod not in (None, 0):
            raise ValueError(
                f"body {body_id}: {volume}/{mesh_dir} is {LEGACY_MESH}, which has a single "
                f"resolution — LOD {lod} does not exist; pass lod=0 or None")
        legacy = _read_legacy_mesh(volume, body_id, mesh_dir)
        if legacy is None:
            return None
        vertices, all_faces = legacy
        matrix = _transform_matrix(info)
        if matrix is not None:
            vertices = vertices @ matrix[:, :3].T + matrix[:, 3]
        return vertices, all_faces, 0

    sharding = info.get("sharding")
    if sharding is not None:
        staged = _sharded_object(volume, body_id, mesh_dir, sharding, lod)
        if staged is None:
            return None
        # The re-emitted manifest declares only the LOD fetched, so ask for that one
        # rather than for the caller's `lod`, which may have been None.
        data, manifest, lod = staged
    else:
        data = location.read_bytes(volume, mesh_dir, str(int(body_id)))
        manifest = location.read_bytes(volume, mesh_dir, f"{int(body_id)}.index")
        if data is None or manifest is None:
            return None

    bits = int(info.get("vertex_quantization_bits", 16))
    decoded = _decode_lod(manifest, data, bits, lod, body_id)
    if decoded is None:
        return None
    vertices, faces, want = decoded
    matrix = _transform_matrix(info)
    if matrix is not None:
        vertices = vertices @ matrix[:, :3].T + matrix[:, 3]
    return vertices, faces, want


def mesh_body_ids(volume: str, mesh_dir: str = "mesh") -> list[int]:
    """Sorted ids of the bodies with a mesh under ``volume/mesh_dir``; ``[]`` for none.

    A multi-resolution mesh counts once its ``<id>.index`` manifest exists (neu-morpho
    writes it after the fragment data); a legacy mesh, once its ``<id>:0`` manifest does.
    One listing of the directory. A sharded mesh raises :class:`UnsupportedSubresource`:
    enumerating one means reading every minishard index (about a billion ids and 3.7 GB
    of index reads for one published fly CNS), so name the bodies explicitly instead.
    """
    import re

    info = _subresource_info(volume, mesh_dir)
    _check_supported(info, volume, mesh_dir, "mesh")
    if info.get("sharding") is not None:
        raise UnsupportedSubresource(
            f"{volume}/{mesh_dir} is sharded, so its body ids can only be enumerated by "
            f"reading every shard's minishard indexes. Pass the bodies explicitly "
            f"(an allowlist) instead.")
    declared = info.get("@type")
    patterns = {"neuroglancer_multilod_draco": r"(\d+)\.index",
                LEGACY_MESH: r"(\d+):0"}
    wanted = [patterns[declared]] if declared in patterns else list(patterns.values())
    match = re.compile("|".join(f"(?:{p})" for p in wanted) + "$").fullmatch
    ids = set()
    for key in location.list_keys(volume, mesh_dir):
        m = match(key.rsplit("/", 1)[-1])
        if m:
            ids.add(int(next(g for g in m.groups() if g is not None)))
    return sorted(ids)


def _decode_lod(manifest: bytes, data: bytes, bits: int, lod: int | None, body_id: int):
    """``(vertices_xyz, faces, lod)`` of one LOD of a multi-LOD Draco object, or ``None``.

    Stored-model coordinates, before the ``transform``. ``lod`` ``None`` is the coarsest
    LOD holding data. Each fragment is Draco-decoded and dequantized against its octree
    cell, ``chunk_shape * 2**lod`` wide at ``grid_origin + position * cell + vertex_offset``;
    fragments of other LODs are skipped rather than decoded.
    """
    import DracoPy

    parsed = _parse_manifest(manifest)
    present = [i for i, sizes in enumerate(parsed["sizes"]) if int(sizes.sum())]
    if not present:
        return None
    want = present[-1] if lod is None else int(lod)
    if want not in present:
        raise ValueError(f"body {body_id}: LOD {want} not present; have {present}")

    chunk = np.frombuffer(manifest, "<f4", 3, 0).astype(float)
    origin = np.frombuffer(manifest, "<f4", 3, 12).astype(float)
    offset = np.frombuffer(parsed["vertex_offsets"], "<f4").astype(float).reshape(-1, 3)[want]
    n = int(parsed["counts"][want])
    positions = np.frombuffer(parsed["blocks"][want], "<u4", n * 3).reshape(3, n).T
    cell = chunk * 2 ** want
    max_q = float((1 << bits) - 1)

    pos = sum(int(sizes.sum()) for sizes in parsed["sizes"][:want])
    verts, faces, base = [], [], 0
    for position, size in zip(positions.astype(np.int64), parsed["sizes"][want]):
        blob = data[pos:pos + int(size)]
        pos += int(size)
        if not size:
            continue
        if len(blob) != size:
            raise ValueError(
                f"body {body_id}: LOD {want} fragment data ends {int(size) - len(blob)} "
                f"bytes short of what its manifest declares")
        mesh = DracoPy.decode(blob)
        q = np.asarray(mesh.points, dtype=float)
        verts.append(origin + position * cell + offset + cell * (q / max_q))
        faces.append(np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3) + base)
        base += len(q)
    return np.concatenate(verts), np.concatenate(faces), want


def mesh_transform(volume: str, mesh_dir: str = "mesh"):
    """The mesh ``info``'s stored-model to nm ``transform``, as a ``(3, 4)`` matrix.

    Identity included. :func:`read_body_mesh` has already applied it; this is for
    reporting it and recovering stored-model coordinates. Raises when there is no
    ``info`` at the location, since an identity answer to a mistyped path would be
    believed.
    """
    return _declared_transform(volume, mesh_dir, "mesh", "mesh")


def skeleton_transform(volume: str, skeleton_dir: str = "skeleton"):
    """The skeleton ``info``'s ``transform``, as a ``(3, 4)`` matrix.

    Separate from :func:`mesh_transform` because a source declares the two independently.
    """
    return _declared_transform(volume, skeleton_dir, "skeleton", "skeletons")


def _declared_transform(volume: str, subdir: str, kind: str, info_key: str):
    import json

    raw = location.read_bytes(volume, subdir, "info")
    if raw is None:
        raise FileNotFoundError(
            f"no {kind} info at {volume}/{subdir}/info, so there is no transform to "
            f"report. Pass the VOLUME and its {kind} subdirectory separately — the "
            f"subdirectory comes from the volume info's {info_key!r} key.")
    values = json.loads(raw).get("transform")
    if values is None:
        return np.hstack([np.eye(3), np.zeros((3, 1))])
    return np.asarray(values, dtype=float).reshape(3, 4)
