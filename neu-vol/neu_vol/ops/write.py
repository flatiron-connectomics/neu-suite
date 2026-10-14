"""Write one subvolume into an existing volume, at a voxel offset and one level.

The companion to :mod:`neu_vol.ops.create`: the frame exists, and each piece (an image
stack, an HDF5 dataset, a region of another volume) is placed into it. **Single-scale**:
only the named level is written, since a piece's coarse representation is a separate
decision (averaging a label map invents ids); run ``neu-vol downsample`` afterwards.
Serial, in this process; no dask, no manifest.

**Partial-chunk writes race.** A region off the destination's chunk grid makes
tensorstore read-modify-write its boundary chunks, and two such writes into a shared chunk
at once lose one update. The plan reports alignment and the CLI says so.

Tiles are cut on the global destination chunk grid (sized to the source's read unit, as
in ``convert``), so no two tiles share a destination chunk.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..backends.base import Region, open_backend
from ..detect import detect_backend, require_chunked_volume
from ..source_metadata import (existing_levels, level_spec, location_spec,
                               read_level_voxel_sizes)
from ._multiscale import DEFAULT_TASK_BYTES, _elapsed, plan_task_shape

logger = logging.getLogger(__name__)


def source_spec(src: str | Mapping[str, Any], src_format: str | None = None,
                dataset: str | None = None) -> dict[str, Any]:
    """A backend spec for ``src``, detecting the format when it is not given.

    :func:`~neu_vol.detect.detect_backend` then
    :func:`~neu_vol.source_metadata.location_spec`, with the ``dataset`` of an HDF5
    container threaded through. A ``dvid://`` URL resolves here at level 0 with the node
    as given; :func:`~neu_vol.ops.pack.pack_hdf5` takes its levels and read options.
    """
    if isinstance(src, Mapping):
        return dict(src)
    src = str(src)
    fmt = src_format or detect_backend(src)
    if fmt is None:
        raise ValueError(
            f"could not tell what {src!r} is; pass src_format= "
            f"(image_stack, hdf5, zarr3, neuroglancer_precomputed, ...)")
    return location_spec(src, fmt, dataset=dataset)


def container_sources(src: str | Mapping[str, Any], pattern: str = "*", *,
                      src_format: str | None = None) -> list[dict[str, Any]]:
    """Every volumetric dataset of an HDF5 container, as one source spec each.

    For a file of several pieces, each with its own ``voxel_offset``::

        write_subvolumes(volume, container_sources("gt_v1_eval_cleaned.h5"))

    ``pattern`` is an fnmatch glob, tried against the **full path** (``/edge_z07801``) and
    against the **basename** (``edge_z07801``). Datasets come back sorted. **A pattern that
    matches nothing raises**, listing what is there, rather than writing nothing.
    """
    import posixpath
    from fnmatch import fnmatch

    from ..backends.hdf5 import datasets

    # Not `source_spec`, which resolves the *sole* dataset and so raises for a container.
    base: dict[str, Any] = dict(src) if isinstance(src, Mapping) else {}
    path = str(base.get("path") if base else src)
    fmt = base.get("backend") if base else (src_format or detect_backend(path))
    if fmt != "hdf5":
        raise ValueError(
            f"expanding every dataset applies to an HDF5 container, and {path!r} is "
            f"{fmt or 'not one'} — a single array, not a bag of them")

    found = sorted(datasets(path))
    chosen = [d for d in found
              if fnmatch(d, pattern) or fnmatch(posixpath.basename(d), pattern)]
    if not chosen:
        raise ValueError(
            f"no dataset in {path} matches {pattern!r}; it holds "
            + (", ".join(found) if found else "no 3D+ dataset"))
    return [{**base, **location_spec(path, "hdf5", dataset=d)} for d in chosen]


def resolve_offset(backend, offset, *, field: str, order: str | None, ndim: int):
    """The offset to write at, as ``(offset, provenance)``, taking it from the source
    if none was given.

    Any backend may expose ``stored_offset(name)`` (HDF5 does), used when no offset is
    given.

    ``order`` is the axis order of the numbers, ``"zyx"`` or ``"xyz"``: ``voxel_offset`` is
    precomputed's field name, where it means xyz, and a wrong guess mirrors the piece
    through the z=x diagonal. ``order=None`` reads ``stored_axes()`` if the source has it
    (``neu-vol to-hdf5`` records it), else assumes zyx. The provenance string says which.
    """
    if order is not None and order not in ("zyx", "xyz"):
        raise ValueError(f"offset order must be 'zyx' or 'xyz', got {order!r}")
    order_from = "given"
    if order is None:
        stated = getattr(backend, "stored_axes", None)
        found = stated() if stated is not None else None
        if found and found[0] in ("zyx", "xyz"):
            order, order_from = found[0], f"recorded in the source, {found[1]}"
        elif found:
            raise ValueError(
                f"the source records axes {found[0]!r} ({found[1]}), which is neither "
                f"'zyx' nor 'xyz'; pass the order explicitly")
        else:
            order, order_from = "zyx", "this package's default, the source records none"
    provenance = "given"
    if offset is None:
        read = getattr(backend, "stored_offset", None)
        found = read(field) if read is not None else None
        if found is None:
            raise ValueError(
                f"no offset given, and this source records none. Pass offset=/--offset, "
                f"or use a source that stores one: an HDF5 file is searched for "
                f"{field!r} in the dataset's attributes, the root group's attributes, "
                f"and a top-level dataset of that name")
        offset, provenance = found[0], f"from the source, {found[1]}"
    offset = tuple(int(o) for o in offset)
    if len(offset) != ndim:
        raise ValueError(f"offset {offset} ({provenance}) has {len(offset)} entries "
                         f"but the data is {ndim}-D")
    # Only an order the source stated is worth reporting.
    note = f", {order_from}" if order_from.startswith("recorded") else ""
    if order == "xyz":
        offset = offset[::-1]
        provenance += f" (read as xyz{note}, reversed to zyx)"
    elif note:
        provenance += f" (read as zyx{note})"
    return offset, provenance


def _offset_at_level(volume: str, fmt: str, offset: Sequence[int],
                     from_level: int, to_level: int) -> tuple[int, ...]:
    """Re-express a voxel offset given at ``from_level`` in ``to_level``'s voxels.

    From each level's **recorded** voxel size, never ``2**level`` (pyramids may be
    anisotropic). A non-integral result is an error rather than a rounding.
    """
    if from_level == to_level:
        return tuple(int(o) for o in offset)
    sizes = read_level_voxel_sizes({"backend": fmt, "path": volume})
    if not sizes or max(from_level, to_level) >= len(sizes):
        raise ValueError(
            f"{volume} does not record per-level voxel sizes for levels "
            f"{from_level} and {to_level}, so an offset cannot be converted between "
            f"them; give the offset in level-{to_level} voxels")
    out = []
    for axis, (o, vf, vt) in enumerate(zip(offset, sizes[from_level], sizes[to_level])):
        scaled = o * vf / vt
        if abs(scaled - round(scaled)) > 1e-6:
            raise ValueError(
                f"offset {tuple(offset)} at level {from_level} is not a whole number "
                f"of level-{to_level} voxels on axis {axis} ({scaled:g}); "
                f"level {from_level} voxel {vf:g} nm, level {to_level} voxel {vt:g} nm")
        out.append(int(round(scaled)))
    return tuple(out)


def _tiles(start: Sequence[int], stop: Sequence[int], unit: Sequence[int]) -> list[Region]:
    """Cut ``[start, stop)`` into tiles on the global grid of multiples of ``unit``.

    ``unit`` is a whole number of chunks, so every interior cut falls on a chunk boundary
    and no two tiles share a chunk; only edge tiles of an unaligned region are partial.
    """
    per_axis = []
    for a, b, u in zip(start, stop, unit):
        cuts = [a]
        nxt = (a // u + 1) * u
        while nxt < b:
            cuts.append(nxt)
            nxt += u
        cuts.append(b)
        per_axis.append(list(zip(cuts[:-1], cuts[1:])))
    return [tuple(slice(lo, hi) for lo, hi in combo)
            for combo in itertools.product(*per_axis)]


def _misaligned_axes(start, stop, shape, chunk) -> list[int]:
    """Axes where the region's edges fall inside a destination chunk.

    :func:`neu_lib.grid.misaligned_axes`, shared with ``neu-vol align-bbox``; an edge at
    the end of the volume counts as aligned.
    """
    from neu_lib.grid import misaligned_axes

    return misaligned_axes(start, stop, chunk, extent=shape)


def plan_subvolume_write(
    volume: str,
    src: str | Mapping[str, Any],
    offset: Sequence[int] | None = None,
    *,
    level: int = 0,
    offset_level: int | None = None,
    offset_field: str = "voxel_offset",
    voxel_size_field: str = "voxel_size",
    offset_order: str | None = None,
    background: Sequence[int] | None = None,
    src_format: str | None = None,
    dataset: str | None = None,
    cast: bool = False,
    max_bytes: int = DEFAULT_TASK_BYTES,
) -> dict:
    """Everything :func:`write_subvolume` would do, resolved and checked. Writes nothing.

    ``offset`` may be ``None``, in which case it is read from the source — see
    :func:`resolve_offset`.
    """
    volume = volume.rstrip("/")
    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(
            f"no volume found at {volume} — create one first with `neu-vol create`")
    # The destination must be a volume: HDF5 files and image stacks are read-only here.
    require_chunked_volume(fmt, volume, "neu-vol write")

    levels = existing_levels(volume, fmt)
    if level not in levels:
        raise ValueError(f"{volume} has no level {level}; present: {sorted(levels)}")
    dst_spec = level_spec(volume, fmt, level)
    dst = open_backend(dst_spec)

    src_spec = source_spec(src, src_format, dataset)
    if background:
        # Before writing: an all-background block of 1s is not all-fill, so it would be
        # stored.
        src_spec = {"backend": "remap", "source": dict(src_spec),
                    "values": [int(v) for v in background], "to": 0}
    src_backend = open_backend(src_spec)
    src_shape = tuple(int(s) for s in src_backend.shape)

    if len(src_shape) != len(dst.shape):
        raise ValueError(
            f"source is {len(src_shape)}-D {src_shape} but level {level} is "
            f"{len(dst.shape)}-D {tuple(dst.shape)}; a channel axis has to match "
            f"(this op does not add or drop one)")

    offset, provenance = resolve_offset(src_backend, offset, field=offset_field,
                                        order=offset_order, ndim=len(src_shape))
    start = _offset_at_level(volume, fmt, offset,
                             level if offset_level is None else offset_level, level)
    stop = tuple(a + s for a, s in zip(start, src_shape))
    dst_shape = tuple(int(s) for s in dst.shape)
    if any(a < 0 for a in start) or any(b > s for b, s in zip(stop, dst_shape)):
        raise ValueError(
            f"the subvolume does not fit: {src_shape} at {start} would span "
            f"{tuple(zip(start, stop))} in a level-{level} volume of {dst_shape}")

    # A piece that records its scale is checked against the level: one extracted at
    # level 1 and written to level 0 fits, at the wrong resolution.
    scale_note = None
    read_size = getattr(src_backend, "stored_voxel_size", None)
    stored_size = read_size(voxel_size_field) if read_size is not None else None
    if stored_size:
        from ..source_metadata import read_level_voxel_sizes

        per_level = read_level_voxel_sizes({"backend": fmt, "path": volume}) or []
        if level < len(per_level):
            want = tuple(float(v) for v in per_level[level])
            got = tuple(stored_size[0])[-len(want):]
            if not np.allclose(got, want):
                scale_note = (
                    f"the source records a voxel size of {got} ({stored_size[1]}) but "
                    f"level {level} of this volume is {want}. The piece will be placed as "
                    f"given — nothing rescales it — so check --level if it was extracted "
                    f"from another one")
                logger.warning("%s", scale_note)

    src_dtype, dst_dtype = np.dtype(src_backend.dtype), np.dtype(dst.dtype)
    if src_dtype != dst_dtype and not cast and not np.can_cast(src_dtype, dst_dtype, "safe"):
        raise ValueError(
            f"source dtype {src_dtype} does not fit destination dtype {dst_dtype} "
            f"without possible loss; pass cast=True (--cast) to do it anyway")

    chunk = tuple(int(c) for c in dst.chunks)
    extent = tuple(b - a for a, b in zip(start, stop))
    try:
        src_chunks = tuple(int(c) for c in src_backend.chunks)
    except Exception:                       # a source need not expose chunking
        src_chunks = None
    unit = plan_task_shape(src_chunks or chunk, chunk, extent,
                           itemsize=dst_dtype.itemsize, max_bytes=max_bytes)
    tiles = _tiles(start, stop, unit)

    return {
        "volume": volume, "format": fmt, "level": level,
        "dst_spec": dst_spec, "src_spec": src_spec,
        "dst_shape": dst_shape, "dst_chunk": chunk, "dst_dtype": str(dst_dtype),
        "src_shape": src_shape, "src_dtype": str(src_dtype), "src_chunks": src_chunks,
        "offset": offset, "offset_from": provenance, "offset_order": offset_order,
        "start": start, "stop": stop, "task_shape": tuple(unit),
        "tiles": tiles, "num_tiles": len(tiles),
        "nbytes": math.prod(extent) * dst_dtype.itemsize,
        "misaligned_axes": _misaligned_axes(start, stop, dst_shape, chunk),
        "scale_note": scale_note,
    }


def _warn_alignment(plan: dict) -> None:
    if plan["misaligned_axes"]:
        logger.warning(
            "region %s-%s is not aligned to level %d's %s chunks on axes %s: the edge "
            "chunks are read-modify-written. Safe here (the chunk's existing data is "
            "merged, not replaced), but do NOT run two such writes concurrently into "
            "chunks they share — one update would be lost silently.",
            plan["start"], plan["stop"], plan["level"], plan["dst_chunk"],
            plan["misaligned_axes"])


def _overlaps(a: dict, b: dict) -> bool:
    return (a["level"] == b["level"] and
            all(a0 < b1 and b0 < a1 for a0, a1, b0, b1
                in zip(a["start"], a["stop"], b["start"], b["stop"])))


def _warn_overlaps(plans: Sequence[dict]) -> list[tuple[int, int]]:
    """Report pieces that cover the same voxels. Later wins, and that is rarely meant.

    Not an error (it may be deliberate), but a mistyped offset that buries an earlier
    piece would otherwise leave no trace.
    """
    clashes = [(i, j) for i in range(len(plans)) for j in range(i + 1, len(plans))
               if _overlaps(plans[i], plans[j])]
    for i, j in clashes:
        logger.warning("sources %d and %d overlap (%s-%s and %s-%s at level %d); "
                       "the later one wins where they meet",
                       i, j, plans[i]["start"], plans[i]["stop"],
                       plans[j]["start"], plans[j]["stop"], plans[i]["level"])
    return clashes


def _execute(plan: dict) -> dict:
    """Read the source tile by tile and write each into the destination."""
    src_backend = open_backend(plan["src_spec"])
    dst = open_backend(plan["dst_spec"])
    out_dtype = np.dtype(plan["dst_dtype"])
    start = plan["start"]
    total = plan["num_tiles"]
    every = 1 if total <= 32 else max(1, total // 20)

    t0 = time.monotonic()
    for i, region in enumerate(plan["tiles"], 1):
        src_region = tuple(slice(s.start - o, s.stop - o) for s, o in zip(region, start))
        data = src_backend.read_region(src_region)
        if data.dtype != out_dtype:
            data = data.astype(out_dtype)
        dst.write_region(region, data)
        if i % every == 0 or i == total:
            logger.info("wrote tile %d/%d %s", i, total,
                        tuple((s.start, s.stop) for s in region))
    elapsed = time.monotonic() - t0
    logger.info("wrote %s %s into %s level %d at %s in %s",
                plan["src_shape"], plan["dst_dtype"], plan["volume"], plan["level"],
                start, _elapsed(elapsed))
    return {**plan, "written": total, "dry_run": False, "seconds": elapsed}


def write_subvolume(volume: str, src: str | Mapping[str, Any],
                    offset: Sequence[int] | None = None,
                    *, dry_run: bool = False, **kw: Any) -> dict:
    """Write ``src`` into ``volume`` at voxel ``offset`` of one level.

    ``offset=None`` takes it from the source when the source records one. See
    :func:`plan_subvolume_write` for the arguments and the checks. Returns the plan
    plus what happened.
    """
    plan = plan_subvolume_write(volume, src, offset, **kw)
    _warn_alignment(plan)
    if dry_run:
        return {**plan, "written": 0, "dry_run": True}
    return _execute(plan)


def write_subvolumes(volume: str, srcs: Sequence[Any],
                     offsets: Sequence[Sequence[int] | None] | None = None,
                     *, dry_run: bool = False, **kw: Any) -> list[dict]:
    """Write several subvolumes into one volume. Returns one result per source.

    **Every source is planned before any is written**, so a bad offset in the last file
    is caught while the volume is untouched. Writing then fails fast.

    ``offsets`` is ``None`` (every source supplies its own) or one entry per source.
    """
    srcs = list(srcs)
    if offsets is None:
        offsets = [None] * len(srcs)
    offsets = list(offsets)
    if len(offsets) != len(srcs):
        raise ValueError(
            f"{len(srcs)} source(s) but {len(offsets)} offset(s): give one offset per "
            f"source, or none at all and let each source supply its own")

    plans = [plan_subvolume_write(volume, s, o, **kw) for s, o in zip(srcs, offsets)]
    for plan in plans:
        _warn_alignment(plan)
    clashes = _warn_overlaps(plans)
    if dry_run:
        return [{**p, "written": 0, "dry_run": True, "overlaps": clashes} for p in plans]
    return [{**_execute(p), "overlaps": clashes} for p in plans]
