"""Pyramid levels from a volume's own metadata: the one parser per format.

- **precomputed**: each entry of ``info["scales"]`` has its own ``resolution`` (xyz nm),
  ``size`` (xyz voxels) and optional ``voxel_offset`` (in that level's voxels).
- **zarr / OME-NGFF**: each multiscale dataset has its own ``scale`` transform and an
  optional ``translation``, which is already in physical units.

Voxel sizes come from the metadata, never from ``2 ** level``: real pyramids are
anisotropic. Levels are ordered finest first, so index 0 is full resolution. Every module
that reads ``info`` or an OME ``zarr.json`` goes through the functions here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from neu_lib import Frame, ScaleInfo


# --------------------------------------------------------------------------- #
# precomputed
# --------------------------------------------------------------------------- #
def precomputed_info(location: str | Mapping[str, Any]) -> dict | None:
    """The volume's ``info`` document, or ``None`` if it has none."""
    from .location import read_bytes

    raw = read_bytes(location, "info")
    return None if raw is None else json.loads(raw)


def finest_first(scales: Sequence[Mapping[str, Any]]) -> list[int]:
    """Indices into ``info["scales"]``, finest resolution first."""
    return sorted(range(len(scales)), key=lambda i: tuple(scales[i]["resolution"]))


def precomputed_level(scale: Mapping[str, Any]) -> tuple[tuple[int, ...], Frame]:
    """``(shape_zyx, frame)`` for one entry of ``info["scales"]``."""
    voxel = tuple(float(v) for v in scale["resolution"][::-1])           # xyz -> zyx
    off = tuple(int(v) for v in scale.get("voxel_offset", (0, 0, 0))[::-1])
    shape = tuple(int(v) for v in scale["size"][::-1])
    return shape, Frame(voxel_size_nm=voxel,
                        origin_nm=tuple(o * v for o, v in zip(off, voxel)))


# --------------------------------------------------------------------------- #
# zarr / OME-NGFF
# --------------------------------------------------------------------------- #
def read_zarr_json(location: str | Mapping[str, Any], *parts: str) -> dict | None:
    """A ``zarr.json`` document, or ``None`` if absent."""
    from .location import read_bytes

    raw = read_bytes(location, *parts, "zarr.json")
    return None if raw is None else json.loads(raw)


def ome_multiscale(meta: Mapping[str, Any] | None) -> dict | None:
    """The first OME multiscale of a group ``zarr.json``; ``None`` for anything else."""
    if not meta or meta.get("node_type") != "group":
        return None
    ome = meta.get("attributes", {}).get("ome")
    return ome["multiscales"][0] if ome else None


def spatial_axes(multiscale: Mapping[str, Any]) -> list[int]:
    """Positions of the ``space`` axes in a multiscale's axis list."""
    return [i for i, a in enumerate(multiscale["axes"]) if a.get("type") == "space"]


def ome_levels(multiscale: Mapping[str, Any]) -> list[tuple[str, Frame]]:
    """``(path, frame)`` per dataset, in listed order, over the spatial axes only."""
    spatial = spatial_axes(multiscale)
    out = []
    for ds in multiscale["datasets"]:
        transforms = ds["coordinateTransformations"]
        scale = next(t["scale"] for t in transforms if t["type"] == "scale")
        translation = next((t["translation"] for t in transforms
                            if t["type"] == "translation"), None)
        out.append((str(ds["path"]), Frame(
            voxel_size_nm=tuple(float(scale[a]) for a in spatial),
            origin_nm=(tuple(float(translation[a]) for a in spatial)
                       if translation else (0.0, 0.0, 0.0)))))
    return out


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #
def read_scales(spec: str | Mapping[str, Any]) -> list[ScaleInfo]:
    """All pyramid levels of a source, finest first. Raises if metadata is absent."""
    from .logs import quiet_reads

    with quiet_reads():
        return _read_scales(spec)


def _read_scales(spec: str | Mapping[str, Any]) -> list[ScaleInfo]:
    from .location import as_directory, spec_kvstore
    from .mapped import TAG as MAPPED, is_descriptor, read_descriptor
    from .source_metadata import detect_backend

    # A mapped view relabels its source and changes nothing else, so its levels are
    # the source's.
    if isinstance(spec, str) and is_descriptor(spec):
        return _read_scales(read_descriptor(spec)["source"])
    if not isinstance(spec, str) and spec.get("backend") == MAPPED:
        return _read_scales(spec["source_location"])

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = as_directory(spec_kvstore(spec))
    # detect_backend takes a location, not a spec: a spec dict would lose the driver.
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return _precomputed_scales(kv)
    if backend in ("zarr3", "zarr2"):
        return _zarr_scales(kv)
    raise ValueError(
        f"cannot read scale metadata for backend {backend!r}; "
        "pass voxel sizes explicitly instead")


def _precomputed_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    info = precomputed_info(kv)
    if info is None:
        raise ValueError(f"no precomputed 'info' at {kv}")
    scales = info["scales"]
    out = []
    for i, j in enumerate(finest_first(scales)):
        shape, frame = precomputed_level(scales[j])
        out.append(ScaleInfo(index=i, shape=shape, frame=frame,
                             key=str(scales[j].get("key", ""))))
    return out


def _zarr_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    meta = read_zarr_json(kv)
    if meta is None:
        raise ValueError(f"no 'zarr.json' at {kv} (zarr v2 OME groups not supported here)")
    ms = ome_multiscale(meta)
    if ms is None:
        raise ValueError("source is a bare zarr array, not an OME multiscale group; "
                         "pass voxel sizes explicitly")
    spatial = spatial_axes(ms)
    out = []
    for i, (path, frame) in enumerate(ome_levels(ms)):
        sub = read_zarr_json(kv, path)
        shape = (tuple(int(sub["shape"][a]) for a in spatial) if sub else (0, 0, 0))
        out.append(ScaleInfo(index=i, shape=shape, frame=frame, key=path))
    return out


def scale_spec(spec: str | Mapping[str, Any], scale_index: int) -> dict:
    """A read spec pinned to one pyramid level.

    precomputed selects the level with ``scale_index``; zarr addresses the level's
    subgroup by path. Use this rather than writing a spec by hand: an unrecognised key
    is silently ignored, so ``{"scale": 2}`` opens full resolution and reads the wrong
    place, failing as empty data rather than an error.
    """
    from .location import as_directory, join, spec_kvstore
    from .source_metadata import detect_backend

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = as_directory(spec_kvstore(spec))
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return {"backend": "neuroglancer_precomputed", "kvstore": kv, "scale_index": scale_index}
    scales = read_scales(spec)
    if not 0 <= scale_index < len(scales):
        raise IndexError(f"scale {scale_index} out of range (source has {len(scales)})")
    return {"backend": backend or "zarr3", "kvstore": join(kv, scales[scale_index].key)}


def describe_scales(spec: str | Mapping[str, Any]) -> str:
    """Human-readable pyramid listing, from one ``info`` or ``zarr.json``.

    Much cheaper than :func:`neu_vol.describe`, which opens every level.
    """
    scales = read_scales(spec)
    lines = [f"{'scale':>5}  {'shape (z,y,x)':>24}  {'voxel nm (z,y,x)':>22}  key"]
    for s in scales:
        lines.append(f"{s.index:>5}  {str(s.shape):>24}  "
                     f"{str(tuple(round(v, 3) for v in s.voxel_size)):>22}  {s.key}")
    return "\n".join(lines)
