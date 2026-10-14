"""The only module that reads data; everything else takes arrays.

Reads meshes and skeletons from a neuroglancer-precomputed volume, resolves a pyramid
level to a :class:`~neu_lib.Frame`, and turns a synapse table into point arrays.

Subdirectory names come from the volume's own ``info``, never assumed: they vary (one
volume declares ``{"mesh": "mesh", "skeletons": "skeleton"}``), and a wrong guess reads
back as every body having no skeleton. A level's voxel size comes from the metadata,
never ``2 ** level``. Meshes and skeletons are stored in nm and need no frame.
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from . import cache as _cache
from .logs import quiet_reads
from neu_lib import BBox, Frame, Mesh, Skeleton

#: Threads, not processes: the work is IO-bound, and tensorstore's S3 credentials are set
#: up per process, which threads share.
DEFAULT_THREADS = 8


def _quiet(fn):
    """Filter benign store logging during a read (see :mod:`neu_draw.logs`)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with quiet_reads():
            return fn(*args, **kwargs)
    return wrapper


class MissingSubresource(RuntimeError):
    """The volume's ``info`` declares no meshes or no skeletons."""


def _structural_errors():
    """Exceptions meaning the source or the call is wrong, not that a body is absent.

    ``skip_missing`` must not swallow these: a volume with no meshes declared, a format
    the reader cannot decode, or a bad keyword (``TypeError``, identical for every body in
    a batch) would otherwise return an empty dict that looks like a clean answer.
    """
    from neu_vol.readback import UnsupportedSubresource

    return (MissingSubresource, UnsupportedSubresource, TypeError)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

@_quiet
def volume_info(volume: str) -> dict:
    """The volume's own ``info``, as a dict."""
    from neu_vol import read_json

    return read_json(volume, "info")


def subresource_dir(volume: str, kind: str, info: Optional[Mapping] = None) -> str:
    """The subdirectory holding ``kind`` (``"mesh"`` or ``"skeletons"``), from ``info``.

    Raises rather than guessing, since a wrong name would read back as every body
    having no mesh.
    """
    if kind not in ("mesh", "skeletons"):
        raise ValueError(f"kind must be 'mesh' or 'skeletons', got {kind!r}")
    info = volume_info(volume) if info is None else info
    name = info.get(kind)
    if not name:
        raise MissingSubresource(
            f"{volume}/info declares no {kind!r}, so there is nothing to read. "
            f"(`neu_vol.link_subresources` adds that key.)")
    return str(name)


#: Pyramid metadata per volume, read once per session.
_SCALES: dict[str, list] = {}


@_quiet
def scales(volume: str, *, refresh: bool = False) -> list:
    """Every pyramid level, finest first, each with its own voxel size
    (``neu_vol.read_scales``). Memoized; ``refresh=True`` re-reads."""
    if refresh or volume not in _SCALES:
        from neu_vol import read_scales

        _SCALES[volume] = read_scales(volume)
    return _SCALES[volume]


@_quiet
def volume_frame(volume: str, level: int = 0) -> Frame:
    """The :class:`~neu_lib.Frame` (voxel size and origin) of one pyramid level, from the
    volume's metadata."""
    levels = scales(volume)
    for scale in levels:
        if scale.index == level:
            return scale.frame
    raise IndexError(
        f"{volume} has no level {level}; it has {[s.index for s in levels]}")


# --------------------------------------------------------------------------- #
# bodies
# --------------------------------------------------------------------------- #

@_quiet
def body_skeleton(volume: str, body_id: int, *, cache: Any = None,
                  name: Optional[str] = None, skeleton_dir: Optional[str] = None,
                  info: Optional[Mapping] = None,
                  require_radii: bool = True) -> Optional[Skeleton]:
    """One body's skeleton, or ``None`` if the volume holds none for it.

    ``require_radii=False`` accepts a source that publishes centrelines only, with every
    radius the ``-1`` sentinel (FlyEM's male-CNS does).
    """
    store = _cache.resolve(cache)
    key = ("skeleton", volume, int(body_id), require_radii)
    if key in store:
        return _named(store[key], name, body_id)

    from neu_vol.readback import read_body_skeleton

    directory = skeleton_dir or subresource_dir(volume, "skeletons", info)
    raw = read_body_skeleton(volume, int(body_id), skeleton_dir=directory,
                             require_radii=require_radii)
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

    ``lod`` defaults to the coarsest present, the cheapest to draw; pass 0 for full
    detail.
    """
    store = _cache.resolve(cache)
    key = ("mesh", volume, int(body_id), lod)
    if key in store:
        return _named(store[key], name, body_id)

    from neu_vol.readback import read_body_mesh

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
    """Skeletons for many bodies, fetched concurrently. Missing bodies are omitted.

    ``skip_missing`` covers only bodies the volume does not hold. A volume declaring no
    skeletons, or storing them in a format the reader cannot decode, raises either way.
    """
    return _fetch_many(body_skeleton, "skeleton", volume, body_ids, cache=cache,
                       names=names, threads=threads, skip_missing=skip_missing, **kwargs)


@_quiet
def body_meshes(volume: str, body_ids: Iterable[int], *, cache: Any = None,
                names: Optional[Mapping[int, str]] = None,
                threads: int = DEFAULT_THREADS, skip_missing: bool = True,
                **kwargs) -> dict[int, Mesh]:
    """Meshes for many bodies, fetched concurrently. Missing bodies are omitted.

    ``skip_missing`` covers only bodies the volume does not hold. A volume declaring no
    meshes, or storing them in a format the reader cannot decode, raises either way.
    """
    return _fetch_many(body_mesh, "mesh", volume, body_ids, cache=cache, names=names,
                       threads=threads, skip_missing=skip_missing, **kwargs)


def _fetch_many(fetch: Callable, kind: str, volume: str, body_ids: Iterable[int], *,
                cache: Any, names: Optional[Mapping[int, str]], threads: int,
                skip_missing: bool, **kwargs) -> dict:
    ids = list(dict.fromkeys(int(b) for b in body_ids))
    if not ids:
        return {}
    store = _cache.resolve(cache, default=_cache.MemoryCache())
    # Read `info` once for the batch rather than once per body.
    shared = dict(kwargs)
    shared.setdefault("info", volume_info(volume))

    structural = _structural_errors()

    def one(body_id: int):
        try:
            return body_id, fetch(volume, body_id, cache=store,
                                  name=(names or {}).get(body_id), **shared)
        except structural:
            raise                       # never a missing body; see _structural_errors
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
# points
# --------------------------------------------------------------------------- #

def points_from_table(table: Any, *, frame: Optional[Frame] = None,
                      columns: Sequence[str] = ("z", "y", "x"),
                      prefix: str = "") -> np.ndarray:
    """An ``(N, 3)`` zyx nm array from a table with named coordinate columns.

    ``columns`` are read in the order given, so the default yields zyx from neu-mark's
    tables. Pass ``frame`` when the table holds voxel indices (DVID's do) rather than nm.
    Columns are looked up by name, never by position, so the axis order is explicit.
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

    Keys are the column's values (DVID's are ``PreSyn`` and ``PostSyn``). A table with no
    kind column becomes a single ``"points"`` entry.
    """
    coords = points_from_table(table, frame=frame, columns=columns, prefix=prefix)
    if not _has_column(table, kind_column):
        return {"points": coords}
    kinds = np.asarray(table[kind_column], dtype=object)
    return {str(k): coords[kinds == k] for k in dict.fromkeys(kinds.tolist())}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _has_column(table: Any, column: str) -> bool:
    try:
        return column in table
    except TypeError:
        return False
