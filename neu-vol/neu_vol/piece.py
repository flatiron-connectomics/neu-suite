"""Read a box out of any source as a :class:`neu_lib.Piece`, and write one back.

A piece carries its array with the frame saying where it is (voxel size and origin in nm),
so a crop cannot lose its position on the way through::

    piece = read_piece("crops.h5:/crop_01")                     # frame from the file
    image = read_piece("s3://bucket/em", level=2, crop=piece)   # the SAME physical box
    write_piece(cleaned, "crops_cleaned.h5")                    # ...and back out again

A crop given as another piece, or in nm, is converted through the target level's own voxel
size and origin, so it transfers between levels and between volumes. :func:`write_piece`
writes an in-memory piece in the same layout :func:`~neu_vol.pack_hdf5` uses, so
``neu-vol write`` can place either with no arguments.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: How much a single read may pull into memory unless asked for explicitly. A whole
#: level 0 of an EM volume can be terabytes, and reading it would look like a hung kernel.
DEFAULT_MAX_BYTES = 4 * 1024 ** 3


def _crop_request(crop: Any, what: str = "crop"):
    """``(voxel_box, nm_bounds)`` — at most one of them.

    Voxels are resolved here; nanometres cannot be, because converting them needs the
    target level's own voxel size *and* origin, neither known until it is opened.
    """
    if crop is None:
        return None, None
    bounds = getattr(crop, "bounds_nm", None)
    if bounds is not None:
        return None, tuple(bounds)
    if isinstance(crop, Mapping):
        if "nm" not in crop:
            raise ValueError(f"{what} mapping must carry 'nm': (lo, hi); got {crop!r}")
        return None, tuple(crop["nm"])
    flat = tuple(crop)
    if len(flat) == 6 and all(isinstance(v, (int, float)) for v in flat):
        lo, hi = tuple(int(v) for v in flat[:3]), tuple(int(v) for v in flat[3:])
    elif len(flat) == 2:
        lo, hi = tuple(int(v) for v in flat[0]), tuple(int(v) for v in flat[1])
    else:
        raise ValueError(
            f"{what} must be ((z0,y0,x0), (z1,y1,x1)) or (z0,y0,x0,z1,y1,x1) in whole "
            f"VOXELS of the level being read, a Piece to take the same physical box as, "
            f"or {{'nm': (lo, hi)}} — got {crop!r}")
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError(f"{what} corners are zyx, so 3 values each; got {lo} / {hi}")
    return (lo, hi), None


def read_piece(src: str | Mapping[str, Any], kind: str | None = None, *,
               level: int = 0, crop: Any = None, dataset: str | None = None,
               src_format: str | None = None, dtype: str | type | None = None,
               voxel_size: Sequence[float] | None = None,
               name: str | None = None,
               max_bytes: int | None = DEFAULT_MAX_BYTES,
               backend: Any = None):
    """The voxels at ``src``, with the frame that says where they are.

    ``src`` is ``PATH`` or ``PATH:/DATASET``, the second selecting an array inside an HDF5
    container (only a leading slash after the last colon counts, so ``s3://`` is left
    alone). A spec mapping or an open ``backend=`` is accepted too.

    ``crop`` is a box in whole voxels of the level read (``((lo), (hi))`` or the flat six),
    anything carrying ``bounds_nm`` such as another :class:`~neu_lib.Piece`, or
    ``{"nm": (lo, hi)}``. A physical box is converted with this level's own voxel size and
    origin and grown outward to whole voxels.

    ``kind`` overrides what the source records; unrecorded, it stays ``None`` rather than
    being guessed from the dtype. ``voxel_size`` overrides the recorded scale and is
    required for a source that records none. ``dtype`` casts after the read, warning on a
    narrowing cast, which would silently wrap label ids.

    The piece is named ``stem`` or ``stem/dataset`` after its source (:func:`piece_name`),
    unless ``name`` is given. ``max_bytes`` refuses a read larger than the cap, naming the
    size and a level that would fit; ``None`` lifts it. Store logging is filtered for the
    call.
    """
    from neu_lib import Frame, Piece

    from .backends.base import open_backend
    from .logs import quiet_reads
    from .source_metadata import (describe, level_spec, location_spec,
                                  read_level_voxel_sizes, read_source_metadata,
                                  require_one_array)

    box, box_nm = _crop_request(crop)
    if isinstance(src, Mapping):
        spec, path, fmt = dict(src), str(src.get("path") or src.get("source") or src), None
        fmt = src.get("backend")
        meta_from_spec = True
    else:
        # The LAST colon; a tail of `//…` is a scheme's own, not a dataset.
        path, _, tail = str(src).rpartition(":")
        # `len(path) > 1` also rules out a Windows drive letter, cheaply.
        if not tail.startswith("/") or tail.startswith("//") or len(path) < 2:
            path, tail = str(src), ""
        dataset = dataset or (tail or None)
        spec, fmt, meta_from_spec = None, src_format, False

    with quiet_reads():
        if backend is not None:
            spec = dict(backend.to_spec())
            fmt = fmt or spec.get("backend")
            dataset = spec.get("dataset") or dataset
            meta = read_source_metadata(spec) or {}
        elif meta_from_spec:
            meta = read_source_metadata(spec) or {}
        elif fmt is None:
            described = describe(path, dataset=dataset or None)
            if described["shape"] is None:
                require_one_array(described, path, "read_piece")
            fmt, spec = described["format"], described["spec"]
            meta = described["meta"] or {}
            dataset = described.get("dataset") or dataset
        else:
            # A format named outright, for a file whose name detection would not recognise.
            spec = location_spec(path, fmt, dataset=dataset or None)
            meta = read_source_metadata(spec) or {}
            dataset = spec.get("dataset") or dataset

        single = fmt in ("hdf5", "image_stack")
        if level and single:
            raise ValueError(
                f"level {level} needs a multiscale volume; {path} is {fmt}, a single array")

        # A zarr OME group is not an array, so level 0 is read through the metadata's
        # `data_spec`, which names the level's subdirectory.
        if backend is not None or single or meta_from_spec:
            read_spec = spec
        elif level:
            read_spec = level_spec(path, fmt, level, dataset=dataset or None)
        elif meta.get("data_spec"):
            read_spec = meta["data_spec"]
        else:
            read_spec = spec          # a bare array: it has no levels to descend into
        source = backend or open_backend(read_spec)
        shape = tuple(int(s) for s in source.shape)

        per_level = read_level_voxel_sizes(spec) or []
        voxel = (tuple(float(v) for v in voxel_size) if voxel_size
                 else (tuple(per_level[level]) if level < len(per_level) else None)
                 or (tuple(meta["voxel_size"]) if meta.get("voxel_size") else None))
        if voxel is None:
            raise ValueError(
                f"{path} records no voxel size, so nothing here knows its physical scale; "
                f"pass voxel_size=(z, y, x)")

        channel_axis = len(shape) == 4
        spatial = shape[1:] if channel_axis else shape
        recorded = (tuple(float(o) for o in meta["offset"])
                    if meta.get("offset") else (0.0, 0.0, 0.0))

        if box_nm is not None:
            box = _physical_to_voxels(box_nm, Frame(voxel_size_nm=voxel,
                                                   origin_nm=recorded),
                                      spatial, path=path, level=level)

        lo = (0, 0, 0)
        if box:
            lo, hi = box
            for axis, (start, stop, extent) in enumerate(zip(lo, hi, spatial)):
                if not 0 <= start < stop <= extent:
                    hint = (" — a crop in NANOMETRES looks like this"
                            if any(v > e * 4 for v, e in zip(hi, spatial)) else "")
                    raise ValueError(
                        f"crop {lo}:{hi} does not fit {path}'s level-{level} extent "
                        f"{spatial} on axis {axis}. The box is in whole VOXELS of that "
                        f"level, zyx, half-open{hint}")
            region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        else:
            region = tuple(slice(0, int(s)) for s in spatial)
        if channel_axis:
            region = (slice(0, shape[0]),) + region
        if max_bytes is not None:
            _refuse_if_too_big(region, source.dtype, max_bytes, path=path, level=level,
                               per_level=per_level, spatial=spatial, cropped=bool(box))

        # A crop of a positioned source lands at the sum of the two origins.
        origin = tuple(o + i * v for o, i, v in zip(recorded, lo, voxel))
        array = source.read_region(region)
        if dtype is not None:
            array = _cast(array, dtype, path=path, dataset=dataset)
        return Piece(array=array,
                     frame=Frame(voxel_size_nm=voxel, origin_nm=origin),
                     kind=kind or meta.get("kind"),
                     name=name or piece_name(path, dataset))


def _cast(array, dtype, *, path: str, dataset: str | None):
    """``array`` as ``dtype``, warning when the cast is not ``np.can_cast``-safe.

    Warned, not refused: only the caller knows the range their labels use. The check
    compares dtypes, not voxels.
    """
    import numpy as np

    want = np.dtype(dtype)
    if array.dtype == want:
        return array
    if not np.can_cast(array.dtype, want, "safe"):
        logger.warning(
            "reading %s%s as %s from %s is not a safe cast: a value that does not fit "
            "wraps to a different one, and for label ids that is silent — the result is "
            "another plausible id. Widening (uint32 -> uint64) is always safe",
            path, f":{dataset}" if dataset else "", want, array.dtype)
    return array.astype(want)


def _host_block(block):
    """One block of a piece's array as a host numpy array.

    Per block, so a lazy array (dask, zarr, h5py) is not materialised whole; a cupy array
    needs ``.get()``, since its ``__array__`` raises.
    """
    import numpy as np

    if type(block).__module__.split(".")[0] == "cupy":
        return np.asarray(block.get())
    return np.asanyarray(block)


def write_piece(piece, out: str, *, dataset: str | None = None,
                dtype: Any = None, truncate: bool = False,
                chunk: Sequence[int] | None = None,
                compression: str | None = "gzip", overwrite: bool = False,
                dry_run: bool = False,
                voxel_size_field: str | None = None,
                offset_field: str | None = None,
                block_bytes: int | None = None) -> dict:
    """Write a :class:`neu_lib.Piece` to an HDF5 file, frame and position included.

    The inverse of :func:`read_piece`::

        piece = read_piece("crops.h5:/crop_01")
        cleaned = piece.apply(dust).apply(dilate)
        write_piece(cleaned, "crops_cleaned.h5")    # -> /crop_01, same frame

    The piece supplies the voxel size, the voxel offset (``piece.origin_voxel``, which
    raises rather than rounding a non-whole origin), ``axes="zyx"``, ``units="nm"`` and its
    ``kind``. ``dataset`` defaults to the last component of ``piece.name`` when it has one
    (``"crops/crop_01"`` -> ``/crop_01``), else :data:`~neu_vol.ops.pack.DEFAULT_DATASET`.

    An existing file is added to when its recorded frame matches; a frame that disagrees,
    or a dataset name already in use without ``overwrite``, is refused
    (:func:`~neu_vol.ops.pack.check_hdf5_target`). Writing is blocked, so a lazy array
    streams. ``dry_run`` returns the plan without writing. A ``dtype`` too narrow for the
    values is refused unless ``truncate`` (:func:`~neu_vol.ops.pack.check_fits`).

    ``out`` must be a local path with an HDF5 extension: h5py has no object-store driver,
    and HDF5 files are recognised by name. ``neu-vol write <volume> --src <file>`` then
    places the piece with no offset argument.
    """
    import numpy as np

    from blockrun import iter_blocks

    from .backends.hdf5 import HDF5_EXTENSIONS, require_local_path
    from .logs import quiet_reads
    from .ops.pack import (DEFAULT_BLOCK_BYTES, DEFAULT_OFFSET_FIELD,
                           DEFAULT_VOXEL_SIZE_FIELD, _block_shape, check_hdf5_target,
                           hdf5_dataset_name, write_hdf5_array)

    voxel_size_field = voxel_size_field or DEFAULT_VOXEL_SIZE_FIELD
    offset_field = offset_field or DEFAULT_OFFSET_FIELD

    out = require_local_path(out, "the HDF5 file to write")
    if not out.lower().endswith(HDF5_EXTENSIONS):
        raise ValueError(
            f"{out!r} does not end in one of {', '.join(HDF5_EXTENSIONS)}, and this writes "
            f"an HDF5 file. The extension is not cosmetic: HDF5 has no marker object, so "
            f"`describe`, `neu-vol info` and `neu-vol write` recognise a container BY NAME "
            f"and would not recognise this one. To write into a volume instead, write the "
            f"piece to an .h5 and place it with `neu-vol write <volume> --src <file>`")

    # A piece is zyx nanometres by construction.
    axes = ("z", "y", "x")
    units = "nm"
    voxel_size = tuple(float(v) for v in piece.voxel_size_nm)
    voxel_offset = tuple(int(v) for v in piece.origin_voxel)

    name = hdf5_dataset_name(dataset if dataset is not None
                             else _dataset_from_piece_name(piece.name))
    shape = piece.shape
    out_dtype = np.dtype(dtype or piece.dtype)
    frame = {"voxel_size": voxel_size, "units": units, "axes": axes}

    _existing, existing_datasets, replacing = check_hdf5_target(
        out, name, frame, overwrite=overwrite, voxel_size_field=voxel_size_field)
    others = [d for d in existing_datasets if d != name]
    block = _block_shape(shape, out_dtype.itemsize, block_bytes or DEFAULT_BLOCK_BYTES)
    plan = {
        "out": out, "dataset": name, "piece": piece.name, "shape": shape,
        "dtype": str(out_dtype), "has_channels": piece.channel_axis,
        "voxel_size": voxel_size, "voxel_offset": voxel_offset, "units": units,
        "axes": axes, "kind": piece.kind, "chunk": tuple(chunk) if chunk else None,
        "compression": compression, "nbytes": float(out_dtype.itemsize) * math.prod(shape),
        "blocks": len(list(iter_blocks(shape, block))), "block_shape": block,
        "replacing": replacing, "appending": bool(existing_datasets) and not replacing,
        "other_datasets": others,
    }
    if dry_run:
        return plan

    if others:
        # INFO, where `pack_hdf5` warns: writing crops back one at a time into one file is
        # the normal use here.
        logger.info("%s now holds %d volumetric datasets (%s); readers must name one "
                    "— `neu-vol write --dataset %s`, or HDF5Backend's `dataset` key",
                    out, len(others) + 1, ", ".join(others + [name]), name)

    array = piece.array
    # A notebook entry point, and a lazy array may read a store to fill each block.
    with quiet_reads():
        write_hdf5_array(out, name, lambda region: _host_block(array[region]),
                         shape=shape, dtype=out_dtype, voxel_size=voxel_size,
                         voxel_offset=voxel_offset, units=units, axes=axes,
                         kind=piece.kind, chunk=chunk, compression=compression,
                         replacing=replacing, block=block,
                         has_channels=piece.channel_axis,
                         voxel_size_field=voxel_size_field, offset_field=offset_field,
                         truncate=truncate)
    logger.info("wrote %s -> %s%s (%s %s, %d block(s))", piece.name or "piece", out, name,
                shape, out_dtype, plan["blocks"])
    return plan


def _dataset_from_piece_name(name: str | None) -> str | None:
    """The array half of a piece's name (after the slash :func:`piece_name` puts in for a
    dataset), or ``None`` if it names no array."""
    import posixpath

    if not name or "/" not in str(name):
        return None
    return posixpath.basename(str(name).strip("/")) or None


#: Suffixes stripped when naming a piece after its source. A volume's directory name is
#: the useful part; the format marker is not.
_NAME_SUFFIXES = (".h5", ".hdf5", ".hdf", ".he5", ".zarr", ".precomputed", ".n5")


def piece_name(path: str, dataset: str | None = None) -> str:
    """What to call a piece read from ``path``: ``stem`` or ``stem/dataset``.

    Both halves, since crops from different files often share a dataset name and crops in
    one file share a stem, and a viewer keys layers by name.
    """
    import os
    import posixpath

    stem = os.path.basename(str(path).rstrip("/")) or str(path)
    for suffix in _NAME_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    inner = posixpath.basename(str(dataset).strip("/")) if dataset else ""
    return f"{stem}/{inner}" if inner else (stem or "piece")


def _size(nbytes: float) -> str:
    """Bytes as the unit a human would use, so a small cap does not read as "0.0 GiB"."""
    for unit, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if nbytes >= scale:
            return f"{nbytes / scale:,.1f} {unit}"
    return f"{nbytes:.0f} B"


def _refuse_if_too_big(region, dtype, max_bytes: int, *, path: str, level: int,
                       per_level, spatial, cropped: bool) -> None:
    """Raise before reading, naming the size and a level that would fit."""
    import numpy as np

    extents = [s.stop - s.start for s in region]
    nbytes = math.prod(extents) * np.dtype(dtype).itemsize
    if nbytes <= max_bytes:
        return

    # Which coarser level WOULD fit, predicted from the recorded per-level voxel sizes
    # rather than by opening anything — the ratio to level 0 gives each level's extent.
    fits = ""
    if per_level and level < len(per_level):
        here = per_level[level]
        for i in range(level + 1, len(per_level)):
            factor = [c / f for c, f in zip(per_level[i], here)]
            shrunk = math.prod(max(1, e / f) for e, f in zip(extents, factor + [1.0]))
            if shrunk * np.dtype(dtype).itemsize <= max_bytes:
                fits = (f" level={i} would be about "
                        f"{_size(shrunk * np.dtype(dtype).itemsize)}.")
                break

    raise ValueError(
        f"reading {tuple(extents)} from {path} at level {level} is "
        f"{_size(nbytes)}, over the {_size(max_bytes)} cap — and "
        f"the failure this prevents is a HANG, not an error: a read that size does not stop, "
        f"it just never finishes, which in a notebook looks like a wedged kernel."
        + ("" if cropped else " Pass crop= to take a box out of it.")
        + fits
        + " Raise max_bytes= (or pass None) if you mean it.")


def _physical_to_voxels(bounds_nm, frame, spatial, *, path: str, level: int):
    """A physical box as a voxel box in ``frame``, clamped to ``spatial``.

    The fraction kept is reported, warning when most of the box is lost: that usually
    means the box came from a different volume.
    """
    box = frame.voxel_box(bounds_nm)
    clamped = (tuple(max(0, v) for v in box.lo),
               tuple(min(e, v) for v, e in zip(box.hi, spatial)))
    logger.info("crop %s nm -> level-%d voxels %s:%s", bounds_nm, level, *clamped)
    if any(b <= a for a, b in zip(*clamped)):
        raise ValueError(
            f"the physical box {bounds_nm} nm does not overlap {path}'s level-{level} "
            f"extent {spatial} at {frame.voxel_size_nm} nm/voxel starting "
            f"{frame.origin_nm} nm. If the box came from another dataset's crop, the two "
            f"are not the same volume")
    asked = math.prod(b - a for a, b in zip(box.lo, box.hi))
    got = math.prod(b - a for a, b in zip(*clamped))
    if got < asked:
        where = [f"{'zyx'[a]} {box.lo[a]}:{box.hi[a]} -> {clamped[0][a]}:{clamped[1][a]}"
                 for a in range(3)
                 if (box.lo[a], box.hi[a]) != (clamped[0][a], clamped[1][a])]
        mostly = got * 2 < asked
        (logger.warning if mostly else logger.info)(
            "the physical box does not fit %s's level-%d extent %s and was clipped to "
            "%.0f%% of it (%s)%s", path, level, spatial, 100.0 * got / asked,
            "; ".join(where),
            ". Losing most of a box usually means it came from a different dataset than "
            "this volume" if mostly else "")
    return clamped
