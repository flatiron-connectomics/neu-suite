"""The only module that reads anything. Everything else takes arrays.

Reads meshes and skeletons out of a neuroglancer-precomputed volume, resolves a pyramid
level to a :class:`~em_viz.geometry.Frame`, turns a synapse table into point arrays, and
builds the ``inside()`` predicates that :meth:`Skeleton.crop` takes.

Two things it refuses to guess.

**Subresource directory names come from the volume's own ``info``.** The precomputed
spec has the volume name its subdirectories, and the names are *not* fixed — sample3's
info reads ``{"mesh": "mesh", "skeletons": "skeleton"}``, where the key is plural and the
directory singular. Hardcoding ``"skeletons"`` finds nothing, and finding nothing is
indistinguishable from a body having no skeleton.

**A scale index is resolved through real per-axis voxel sizes**, never ``2 ** level``
(invariant 1). Meshes and skeletons need no frame at all — the format stores them in nm
already, and both of sample3's subresources declare an identity ``transform`` — but
anything coming from the voxel grid does.
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from . import cache as _cache
from .logs import quiet_reads
from .geometry import BBox, Frame, Mesh, Skeleton

#: Threads, not processes: the work is IO-bound, and tensorstore's S3 credential
#: bootstrap is per-process (invariant 8), which a thread pool inherits for free.
DEFAULT_THREADS = 8


def _quiet(fn):
    """Filter benign store logging for the duration of a read.

    Attached to every entry point here rather than left to the caller. The sibling CLIs
    do the same thing by wrapping `main()` (with `--store-logs` to opt out); em-viz has
    no CLI, so its entry points are these functions. `em_viz.logs.enabled = False` is
    the opt-out, and the filter is a deny-list — a real failure still prints.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with quiet_reads():
            return fn(*args, **kwargs)
    return wrapper


class MissingSubresource(RuntimeError):
    """The volume's ``info`` declares no meshes or no skeletons."""


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

@_quiet
def volume_info(volume: str) -> dict:
    """The volume's own ``info``, as a dict."""
    from em_volume_tools import read_json

    return read_json(volume, "info")


def subresource_dir(volume: str, kind: str, info: Optional[Mapping] = None) -> str:
    """The subdirectory holding ``kind`` (``"mesh"`` or ``"skeletons"``), from ``info``.

    Raises rather than defaulting. A default that happens to be wrong reads back as
    "this body has no mesh" for every body, which looks like missing data.
    """
    if kind not in ("mesh", "skeletons"):
        raise ValueError(f"kind must be 'mesh' or 'skeletons', got {kind!r}")
    info = volume_info(volume) if info is None else info
    name = info.get(kind)
    if not name:
        raise MissingSubresource(
            f"{volume}/info declares no {kind!r}, so there is nothing to read. "
            f"(em-seg-morpho's `link_subresources` is what adds that key.)")
    return str(name)


#: Pyramid metadata per volume. A volume's scales cannot change under a live session,
#: and `volume_frame(v, 0)`, `(v, 1)`, `(v, 2)` would otherwise re-read `info` each time.
_SCALES: dict[str, list] = {}


@_quiet
def scales(volume: str, *, refresh: bool = False) -> list:
    """Every pyramid level, finest first, each with its own real voxel size. Memoized."""
    if refresh or volume not in _SCALES:
        from em_seg_morpho.scales import read_scales

        _SCALES[volume] = read_scales(volume)
    return _SCALES[volume]


@_quiet
def volume_frame(volume: str, level: int = 0) -> Frame:
    """The :class:`~em_viz.geometry.Frame` of one pyramid level.

    Reads the level's own ``resolution`` from the metadata. **Never ``2 ** level``** —
    real pyramids are anisotropic, and the factor that looks right for sample3 (which
    is isotropic) is wrong for the common ``(1, 2, 2)`` shape.
    """
    levels = scales(volume)
    for scale in levels:
        if scale.index == level:
            return Frame(voxel_size_nm=tuple(scale.voxel_size))
    raise IndexError(
        f"{volume} has no level {level}; it has {[s.index for s in levels]}")


# --------------------------------------------------------------------------- #
# bodies
# --------------------------------------------------------------------------- #

@_quiet
def body_skeleton(volume: str, body_id: int, *, cache: Any = None,
                  name: Optional[str] = None, skeleton_dir: Optional[str] = None,
                  info: Optional[Mapping] = None) -> Optional[Skeleton]:
    """One body's skeleton, or ``None`` if the volume holds none for it.

    Vertices arrive xyz (the order the format stores) and are flipped to zyx once, by
    :meth:`Skeleton.from_precomputed`.
    """
    store = _cache.resolve(cache)
    key = ("skeleton", volume, int(body_id))
    if key in store:
        return _named(store[key], name, body_id)

    from em_seg_morpho.readback import read_body_skeleton

    directory = skeleton_dir or subresource_dir(volume, "skeletons", info)
    raw = read_body_skeleton(volume, int(body_id), skeleton_dir=directory)
    if raw is None:
        return None
    skeleton = Skeleton.from_precomputed(*raw, name=str(body_id))
    store[key] = skeleton
    return _named(skeleton, name, body_id)


@_quiet
def body_mesh(volume: str, body_id: int, *, lod: Optional[int] = None, cache: Any = None,
              name: Optional[str] = None, mesh_dir: Optional[str] = None,
              info: Optional[Mapping] = None) -> Optional[Mesh]:
    """One body's mesh at one LOD, or ``None`` if the volume holds none for it.

    ``lod`` defaults to the **coarsest** present, which is the cheapest to draw and is
    usually what a whole-neuron view wants; pass 0 for full detail.
    """
    store = _cache.resolve(cache)
    key = ("mesh", volume, int(body_id), lod)
    if key in store:
        return _named(store[key], name, body_id)

    from em_seg_morpho.readback import read_body_mesh

    directory = mesh_dir or subresource_dir(volume, "mesh", info)
    raw = read_body_mesh(volume, int(body_id), lod=lod, mesh_dir=directory)
    if raw is None:
        return None
    vertices_xyz, faces, _ = raw
    mesh = Mesh.from_precomputed(vertices_xyz, faces, name=str(body_id))
    store[key] = mesh
    return _named(mesh, name, body_id)


@_quiet
def body_skeletons(volume: str, body_ids: Iterable[int], *, cache: Any = None,
                   names: Optional[Mapping[int, str]] = None,
                   threads: int = DEFAULT_THREADS, skip_missing: bool = True,
                   **kwargs) -> dict[int, Skeleton]:
    """Skeletons for many bodies, fetched concurrently. Missing bodies are omitted."""
    return _fetch_many(body_skeleton, "skeleton", volume, body_ids, cache=cache,
                       names=names, threads=threads, skip_missing=skip_missing, **kwargs)


@_quiet
def body_meshes(volume: str, body_ids: Iterable[int], *, cache: Any = None,
                names: Optional[Mapping[int, str]] = None,
                threads: int = DEFAULT_THREADS, skip_missing: bool = True,
                **kwargs) -> dict[int, Mesh]:
    """Meshes for many bodies, fetched concurrently. Missing bodies are omitted."""
    return _fetch_many(body_mesh, "mesh", volume, body_ids, cache=cache, names=names,
                       threads=threads, skip_missing=skip_missing, **kwargs)


def _fetch_many(fetch: Callable, kind: str, volume: str, body_ids: Iterable[int], *,
                cache: Any, names: Optional[Mapping[int, str]], threads: int,
                skip_missing: bool, **kwargs) -> dict:
    ids = list(dict.fromkeys(int(b) for b in body_ids))
    if not ids:
        return {}
    store = _cache.resolve(cache, default=_cache.MemoryCache())
    # Read `info` ONCE and pass it down. Otherwise every body re-reads it — one S3
    # round trip per body, for a file that cannot have changed during the call.
    shared = dict(kwargs)
    shared.setdefault("info", volume_info(volume))

    def one(body_id: int):
        try:
            return body_id, fetch(volume, body_id, cache=store,
                                  name=(names or {}).get(body_id), **shared)
        except Exception:
            if skip_missing:
                return body_id, None
            raise

    if threads and threads > 1 and len(ids) > 1:
        with ThreadPoolExecutor(max_workers=min(threads, len(ids))) as pool:
            pairs = list(pool.map(one, ids))
    else:
        pairs = [one(b) for b in ids]

    found = {b: obj for b, obj in pairs if obj is not None}
    missing = [b for b in ids if b not in found]
    if missing and not skip_missing:
        raise KeyError(f"{volume} holds no {kind} for {len(missing)} bodies: "
                       f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    return found


def _named(obj, name: Optional[str], body_id: int):
    """Apply the caller's label without mutating the cached object."""
    if name is None or obj is None or obj.name == name:
        return obj
    import dataclasses

    return dataclasses.replace(obj, name=name)


# --------------------------------------------------------------------------- #
# derived geometry
# --------------------------------------------------------------------------- #

def skeleton_tube(skeleton: Skeleton, sides: int = 8,
                  name: Optional[str] = None) -> Mesh:
    """A skeleton as a solid tube, one truncated cone per edge, using its radii.

    The alternative to a line: ``LineSegmentMaterial`` draws a constant screen-space
    thickness, so it says nothing about calibre. This makes the per-vertex radius
    visible, and a radius that does not fit inside the body shows up as the tube
    breaking the surface.

    Built in **xyz** and flipped back on the way out, so the triangle winding matches
    what a precomputed mesh's would be — the round trip through
    :meth:`Mesh.from_precomputed` is what keeps the normals facing outward.
    """
    if skeleton.radii_nm is None:
        raise ValueError(
            f"skeleton {skeleton.name!r} carries no radii, so it has no tube. "
            f"Draw it as lines, or read a skeleton whose info declares a 'radius' "
            f"vertex attribute.")
    from em_seg_morpho.readback import frustum_mesh
    from .geometry import to_xyz

    vertices, faces = frustum_mesh(to_xyz(skeleton.vertices_zyx_nm), skeleton.edges,
                                   skeleton.radii_nm, sides=sides)
    return Mesh.from_precomputed(vertices, faces,
                                 name=name if name is not None else skeleton.name)


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #

def points_from_table(table: Any, *, frame: Optional[Frame] = None,
                      columns: Sequence[str] = ("z", "y", "x"),
                      prefix: str = "") -> np.ndarray:
    """An ``(N, 3)`` zyx nm array from a table with named coordinate columns.

    ``columns`` is read **in the order given**, so the default is already zyx and
    em-annotation's tables need no reordering. Pass ``frame`` when the table holds voxel
    indices rather than nm — DVID's do; a precomputed annotation source's do not.

    Coordinates are always named columns, never a positional array: that is the one
    place the axis order is decided, and a mirrored synapse is a valid synapse in the
    wrong place.
    """
    names = [f"{prefix}{c}" for c in columns]
    missing = [n for n in names if not _has_column(table, n)]
    if missing:
        raise KeyError(f"table has no column(s) {missing}; "
                       f"looked for {names} (prefix={prefix!r})")
    coords = np.column_stack([np.asarray(table[n], dtype=np.float64) for n in names])
    return np.ascontiguousarray(frame.to_nm(coords) if frame is not None else coords)


def synapse_points(table: Any, *, frame: Optional[Frame] = None,
                   kind_column: str = "kind",
                   columns: Sequence[str] = ("z", "y", "x"),
                   prefix: str = "") -> dict[str, np.ndarray]:
    """Split a synapse table by kind into ``{kind: (N, 3) zyx nm}``.

    Keys are whatever the column holds — DVID's are ``PreSyn`` and ``PostSyn``. A table
    with no kind column becomes a single ``"points"`` entry, so a plain coordinate list
    still works.
    """
    coords = points_from_table(table, frame=frame, columns=columns, prefix=prefix)
    if not _has_column(table, kind_column):
        return {"points": coords}
    kinds = np.asarray(table[kind_column], dtype=object)
    return {str(k): coords[kinds == k] for k in dict.fromkeys(kinds.tolist())}


# --------------------------------------------------------------------------- #
# region predicates
# --------------------------------------------------------------------------- #

def box_predicate(box: BBox) -> Callable[[np.ndarray], np.ndarray]:
    """``inside()`` for a nm bounding box. The simplest region there is."""
    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        return box.contains(np.asarray(points_zyx_nm))
    return inside


def mask_predicate(mask_zyx: np.ndarray, frame: Frame) -> Callable[[np.ndarray], np.ndarray]:
    """``inside()`` for a dense boolean region at some resolution.

    A dense array is the right representation *here* and would not be for a body mask:
    ROIs live at a coarse level, where the whole volume is small. sample3 at level 5 is
    about 352x281x430, some 43 MB as ``bool`` — while the same volume at level 0 is
    a million times that, which is why em-viz has no general mask type.

    Points outside the array are outside the region, not an error: a skeleton normally
    extends past any one ROI, and that is the question being asked.
    """
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D zyx, got shape {mask.shape}")

    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        voxels = np.floor(frame.to_voxel(np.asarray(points_zyx_nm))).astype(np.int64)
        ok = np.all((voxels >= 0) & (voxels < np.asarray(mask.shape)), axis=1)
        out = np.zeros(len(voxels), dtype=bool)
        if ok.any():
            inside_voxels = voxels[ok]
            out[ok] = mask[inside_voxels[:, 0], inside_voxels[:, 1],
                           inside_voxels[:, 2]]
        return out
    return inside


def _has_column(table: Any, column: str) -> bool:
    try:
        return column in table
    except TypeError:
        return False
