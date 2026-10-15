"""Pack a small volume into one HDF5 file, described well enough to be written back.

The inverse of ``ops/write.py``: that places a piece into a large volume, this produces
the piece, recording **where it belongs and at what scale** so ``neu-vol write`` can place
it without anyone re-typing an offset.

The source is anything readable: an image stack, an HDF5 file (whose recorded frame becomes
the default), or a volume, in which case ``level`` and ``crop_start``/``crop_stop`` take a
box out of one level and the crop origin becomes the recorded ``voxel_offset``.

Recorded: ``voxel_offset`` (whole voxels) on the dataset; ``voxel_size`` / ``units`` /
``axes`` on the root and the dataset, so either alone is self-describing; ``offset`` (the
same place in ``units``) on the dataset. ``axes`` makes ``voxel_offset``'s order knowable:
the name is precomputed's, where it means xyz, and a wrong guess mirrors the piece through
the z=x diagonal.

Reads are blocked, so a volume larger than expected still packs. Serial and in-process.
"""

from __future__ import annotations

import logging
import math
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from blockrun import iter_blocks

from ..backends.base import open_backend
from ..detect import PRECOMPUTED_GZ
from .write import source_spec

logger = logging.getLogger(__name__)

#: The dataset a file gets when the caller names none, the same default the reader uses.
DEFAULT_DATASET = "/data"

#: Frame attributes that must agree when adding a dataset to an existing file. Not
#: ``voxel_offset``, which every dataset has its own of.
FRAME_ATTRS = ("voxel_size", "units", "axes")

#: Default names for the two fields other writers most often spell differently; both are
#: settable per call, for reading and writing.
DEFAULT_VOXEL_SIZE_FIELD = "voxel_size"
DEFAULT_OFFSET_FIELD = "voxel_offset"

#: Ceiling on one read, so packing a volume that is larger than advertised streams instead
#: of dying. Blocks cover whole z-slabs, the natural read unit of an image stack.
DEFAULT_BLOCK_BYTES = 1 * 1024 ** 3

#: Attribute recording what the voxels mean (:data:`neu_lib.KINDS`), on the dataset only.
#: A uint8 label array cannot be told from an image by dtype, and treating labels as an
#: image averages ids into ids that were never in the data; read back only when the value
#: is one this package writes.
KIND_ATTR = "kind"

#: Sources served over HTTP rather than read from a store; their levels come from the
#: server's metadata.
SERVER_FORMATS = ("dvid", "webknossos")


def resolve_source(src: str | dict, src_format: str | None = None, level: int = 0,
                   dataset: str | None = None,
                   voxel_size_field: str = "voxel_size",
                   offset_field: str = "voxel_offset",
                   options: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """The spec for the level to read, and what the source says about its own frame.

    ``options`` are the server-source read options of
    :func:`~neu_vol.ops.convert.with_source_options`. Three kinds of source:

    * a **server source** (DVID, webknossos) — its levels, voxel sizes and kind come from
      the server's metadata, and the spec read is the one that metadata resolves: for a
      DVID branch ref, the concrete node, so every block comes from the same version.
    * a **multiscale volume** — resolved through
      :func:`~neu_vol.source_metadata.level_spec`, which addresses a level correctly in
      either format; its per-level voxel size, units and kind come back too.
    * **anything else** (an image stack, a bare array, an HDF5 file): no levels, so a
      ``level`` other than 0 is an error. An HDF5 file's recorded frame is read.
    """
    from ..detect import detect_backend
    from ..source_metadata import (existing_levels, level_spec, location_spec,
                                   read_level_voxel_sizes, read_source_metadata)
    from .convert import with_source_options

    fmt = None if isinstance(src, dict) else (src_format or detect_backend(str(src)))
    if options and any(options.values()) and fmt not in SERVER_FORMATS:
        with_source_options({"backend": fmt}, **options)       # raises, naming the option
    if fmt in SERVER_FORMATS:
        loc = with_source_options(location_spec(str(src), fmt), **(options or {}))
        per_level = read_level_voxel_sizes(loc) or []
        if per_level and not 0 <= level < len(per_level):
            raise ValueError(f"{src} has no level {level} "
                             f"(present: 0-{len(per_level) - 1})")
        meta = read_source_metadata({**loc, "scale_index": int(level)})
        frame = {"units": meta.get("units") or "nm"}
        if per_level:
            frame["voxel_size"] = tuple(per_level[level])
        if meta.get("kind"):
            frame["kind"] = meta["kind"]
        return meta["data_spec"], frame
    if fmt not in ("zarr3", "neuroglancer_precomputed", PRECOMPUTED_GZ):
        if level:
            raise ValueError(
                f"--level {level} needs a multiscale volume; {src!r} is "
                f"{fmt or src_format or 'not one'}, which has no levels")
        spec = source_spec(src, src_format, dataset)
        frame = {}
        if spec.get("backend") == "hdf5":
            # A repack must keep what the file says about itself: a piece that arrives
            # knowing where it belongs must not leave claiming the origin.
            backend = open_backend(spec)
            size = backend.stored_voxel_size(voxel_size_field)
            if size:
                frame["voxel_size"] = tuple(size[0])
                logger.info("voxel_size taken from the source, %s", size[1])
            order = backend.stored_axes()
            if order:
                frame["axes"] = tuple(order[0])
                logger.info("axes taken from the source, %s", order[1])
            meaning = backend.stored_kind()
            if meaning:
                frame["kind"] = meaning[0]
                logger.info("kind taken from the source, %s", meaning[1])
            found = backend.stored_offset(offset_field)
            if found:
                # `resolve_offset` owns the axis-order rule: recorded `axes`, else zyx.
                from .write import resolve_offset

                value, prov = resolve_offset(backend, None, field=offset_field,
                                             order=None, ndim=len(found[0]))
                frame["voxel_offset"] = tuple(value)
                logger.info("voxel_offset taken %s", prov)
        return spec, frame

    volume = str(src).rstrip("/")
    per_level = read_level_voxel_sizes({"backend": fmt, "path": volume}) or []
    levels = existing_levels(volume, "neuroglancer_precomputed"
                             if fmt == PRECOMPUTED_GZ else fmt)
    if not levels:
        # A bare zarr array is detected as zarr3 but has no levels to address.
        if level:
            raise ValueError(f"{volume} has no levels (it looks like a bare array, not a "
                             f"multiscale volume), so --level {level} means nothing")
        return source_spec(src, src_format, dataset), {}
    if level not in levels:
        raise ValueError(f"{volume} has no level {level} (present: {sorted(levels)})")
    meta = read_source_metadata({"backend": fmt, "path": volume}) or {}
    frame = {}
    if level < len(per_level):
        frame["voxel_size"] = tuple(per_level[level])
    if meta.get("units"):
        frame["units"] = meta["units"]
    if meta.get("kind"):
        # A recorded kind is safe to carry over; inferring one from a dtype is not.
        frame["kind"] = meta["kind"]
    return level_spec(volume, fmt, level), frame


def _needs_staging(src: Any, src_format: str | None, out: str) -> bool:
    from ..backends.hdf5 import names_remote_hdf5
    from ..location import is_local

    return names_remote_hdf5(src, src_format) or not is_local(out)


@contextmanager
def _staged(src: Any, src_format: str | None, out: str, *,
            dry_run: bool) -> Iterator[tuple[Any, str]]:
    """Local stand-ins for a remote HDF5 ``src`` and a remote ``out``.

    ``out`` is uploaded when the block exits cleanly, unless ``dry_run``.
    """
    from ..backends.hdf5 import local_copy, names_remote_hdf5, staged_output
    from ..location import is_local

    with ExitStack() as stack:
        if names_remote_hdf5(src, src_format):
            logger.info("reading %s from a temporary local copy", src)
            src = stack.enter_context(local_copy(src))
        if not is_local(out):
            logger.info("writing %s through a temporary local copy", out)
            out = stack.enter_context(staged_output(out, upload=not dry_run))
        yield src, out


def _as_given(plan: dict, src: Any, local_src: Any, out: str) -> dict:
    """``plan`` naming the locations the caller gave rather than the temporary copies."""
    plan = dict(plan, out=out)
    if local_src != src:
        plan["src_spec"] = _with_path(plan["src_spec"], local_src, src)
    return plan


def _with_path(spec: dict, old: str, new: str) -> dict:
    """``spec`` with every ``path`` equal to ``old`` replaced by ``new``, views included."""
    out = {k: (_with_path(v, old, new) if isinstance(v, dict) else v)
           for k, v in spec.items()}
    if out.get("path") == old:
        out["path"] = new
    return out


def _block_shape(shape: Sequence[int], itemsize: int,
                 max_bytes: int = DEFAULT_BLOCK_BYTES) -> tuple[int, ...]:
    """Whole slabs along the slowest axis, sized to ``max_bytes``. At least one plane."""
    shape = tuple(int(s) for s in shape)
    plane = max(1, math.prod(shape[1:]) * itemsize)
    n = max(1, min(shape[0], max_bytes // plane))
    return (n,) + shape[1:]


def _read_frame(h5file, voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD) -> dict:
    """The frame recorded at a file's root, in comparable form. Empty if it records none.

    Keyed by this package's names, whatever the file calls them.
    """
    out = {}
    for key in FRAME_ATTRS:
        attr = voxel_size_field if key == "voxel_size" else key
        if attr in h5file.attrs:
            value = h5file.attrs[attr]
            if key == "units":
                out[key] = value.decode() if isinstance(value, bytes) else str(value)
            elif key == "axes":
                out[key] = _axes_tuple(value)
            else:
                out[key] = tuple(float(v) for v in np.asarray(value).ravel())
    return out


def _axes_tuple(value: Any) -> tuple[str, ...]:
    """An axis-order attribute as ``("z","y","x")``, through the reader's own parser."""
    from ..backends.hdf5 import axes_string

    return tuple(axes_string(value))


def _frame_mismatch(existing: dict, wanted: dict) -> list[str]:
    """Human-readable differences between two frames, in the order of FRAME_ATTRS."""
    bad = []
    for key in FRAME_ATTRS:
        if key not in existing:
            continue
        old, new = existing[key], wanted[key]
        same = (np.allclose(old, new) if key == "voxel_size"
                else tuple(old) == tuple(new))
        if not same:
            bad.append(f"{key}: file has {old}, this would write {new}")
    return bad


# --------------------------------------------------------------------------- #
# The file layout, in one place: `pack_hdf5` and `neu_vol.write_piece` both write through
# these three functions, so the two writers cannot drift apart.
# --------------------------------------------------------------------------- #
def hdf5_dataset_name(dataset: str | None) -> str:
    """A dataset name as a rooted path, defaulting to :data:`DEFAULT_DATASET`.

    Rooted, the form ``h5py`` reports from ``visititems``, so ``"gt/piece0"`` and
    ``"/gt/piece0"`` compare equal.
    """
    name = dataset or DEFAULT_DATASET
    return name if name.startswith("/") else "/" + name


def check_hdf5_target(out: str, name: str, frame: dict, *, overwrite: bool = False,
                      voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD
                      ) -> tuple[dict, list[str], bool]:
    """What an existing ``out`` allows: ``(its frame, its datasets, replacing)``.

    An existing file is **added to** rather than replaced; each dataset keeps its own
    ``voxel_offset``. Refused: a frame that disagrees with the file's, and a dataset name
    already in use unless ``overwrite``. A missing file answers with empties.
    """
    import os

    import h5py

    if not os.path.exists(out):
        return {}, [], False

    with h5py.File(out, "r") as f:
        existing_frame = _read_frame(f, voxel_size_field)
        found: list[str] = []
        f.visititems(lambda n, o: found.append("/" + n)
                     if isinstance(o, h5py.Dataset) and o.ndim >= 3 else None)
        existing_datasets = sorted(found)

    mismatch = _frame_mismatch(existing_frame, frame)
    if mismatch:
        raise ValueError(
            f"{out} already records a different frame, so adding to it would make one "
            f"file describe two coordinate systems: " + "; ".join(mismatch))
    replacing = name in existing_datasets
    if replacing and not overwrite:
        raise FileExistsError(
            f"{out} already has a dataset at {name}. Give another name, or pass "
            f"overwrite=True to replace it.")
    return existing_frame, existing_datasets, replacing


def check_fits(data: np.ndarray, dtype: Any, region: Sequence[slice] | None = None) -> None:
    """Raise unless every value of ``data`` lies within ``dtype``'s range.

    A cast that cannot lose range (uint16 -> uint64) is not inspected. Only range is
    checked: integer -> float precision and float -> integer fractions are not.
    """
    dtype = np.dtype(dtype)
    if data.size == 0 or np.can_cast(data.dtype, dtype, "safe"):
        return
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
    elif dtype.kind == "f":
        info = np.finfo(dtype)
    else:
        return
    outside = (data < info.min) | (data > info.max)
    n = int(np.count_nonzero(outside))
    if not n:
        return
    where = ("" if region is None else
             " in block [" + ", ".join(f"{s.start}:{s.stop}" for s in region) + "]")
    raise ValueError(
        f"{n} value(s){where} do not fit {dtype} (range {info.min}..{info.max}; this "
        f"block holds {np.nanmin(data)}..{np.nanmax(data)}), and the cast would change "
        f"them. Choose a wider dtype, renumber the source first, or pass truncate=True "
        f"(--truncate) to cast anyway.")


def write_hdf5_array(out: str, name: str, read_region, *, shape: Sequence[int],
                     dtype: Any, voxel_size: Sequence[float],
                     voxel_offset: Sequence[int], units: str = "nm",
                     axes: Sequence[str] = ("z", "y", "x"), kind: str | None = None,
                     chunk: Sequence[int] | None = None,
                     compression: str | None = "gzip", replacing: bool = False,
                     block: Sequence[int] | None = None,
                     has_channels: bool = False,
                     voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD,
                     offset_field: str = DEFAULT_OFFSET_FIELD,
                     truncate: bool = False) -> None:
    """Create ``out``'s ``name`` from ``read_region``, and record the frame beside it.

    ``read_region(region) -> ndarray`` is called once per block, so the whole array is
    never held. ``voxel_offset`` describes the spatial axes; a leading channel axis gets 0.
    ``voxel_size`` / ``units`` / ``axes`` go on the root and the dataset; ``voxel_offset``,
    ``offset`` and ``kind`` on the dataset alone (see the module docstring).

    A value outside ``dtype``'s range raises (:func:`check_fits`) unless ``truncate``. The
    dataset is written under a temporary name and moved into place only once complete, so
    a refused or failed write leaves the file as it was, a replaced dataset included.
    """
    import os

    import h5py

    from ..backends.base import release_backends

    # HDF5 refuses to open a file for writing while a cached reader holds it.
    release_backends(out)

    shape = tuple(int(s) for s in shape)
    out_dtype = np.dtype(dtype)
    axes = tuple(axes)
    voxel_size = tuple(float(v) for v in voxel_size)
    voxel_offset = tuple(int(v) for v in voxel_offset)

    chunk_full = tuple(chunk) if chunk else tuple(min(64, s) for s in shape)
    if len(chunk_full) != len(shape):
        raise ValueError(f"chunk {chunk_full} does not match the source's {len(shape)} axes")
    chunk_full = tuple(min(int(c), int(s)) for c, s in zip(chunk_full, shape))
    block = tuple(block) if block else _block_shape(shape, out_dtype.itemsize)

    created = not os.path.exists(out)
    try:
        with h5py.File(out, "a") as f:
            partial = name.rstrip("/") + ".partial"
            if partial in f:
                del f[partial]                     # left by a run killed mid-write
            dset = f.create_dataset(partial, shape=shape, dtype=out_dtype,
                                    chunks=chunk_full,
                                    **({"compression": compression} if compression else {}))
            try:
                for b in iter_blocks(shape, block):
                    data = read_region(b.region)
                    if str(data.dtype) != str(out_dtype):
                        if not truncate:
                            check_fits(data, out_dtype, b.region)
                        data = data.astype(out_dtype)
                    dset[b.region] = data
            except BaseException:
                del f[partial]
                raise

            for target in (f, dset):
                target.attrs[voxel_size_field] = np.asarray(voxel_size, dtype="float64")
                target.attrs["units"] = units
                target.attrs["axes"] = "".join(axes)
            full_offset = ((0,) + voxel_offset) if has_channels else voxel_offset
            dset.attrs[offset_field] = np.asarray(full_offset, dtype="int64")
            dset.attrs["offset"] = np.asarray(
                [o * v for o, v in zip(voxel_offset, voxel_size)], dtype="float64")
            if kind is not None:
                dset.attrs[KIND_ATTR] = str(kind)
            if replacing:
                del f[name]
            f.move(partial, name)
    except BaseException:
        if created and os.path.exists(out):
            os.remove(out)
        raise


def pack_hdf5(
    src: str | dict,
    out: str,
    *,
    voxel_size: Sequence[float] | None = None,
    voxel_offset: Sequence[int] | None = None,
    units: str | None = None,
    axes: Sequence[str] = ("z", "y", "x"),
    level: int = 0,
    crop_start: Sequence[int] | None = None,
    crop_stop: Sequence[int] | None = None,
    dataset: str | None = None,
    src_dataset: str | None = None,
    src_format: str | None = None,
    supervoxels: bool = False,
    prefer_locked: bool = False,
    wk_mapping: str | None = None,
    voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD,
    offset_field: str = DEFAULT_OFFSET_FIELD,
    background: Sequence[int] | None = None,
    kind: str | None = None,
    dtype: str | None = None,
    truncate: bool = False,
    chunk: Sequence[int] | None = None,
    compression: str | None = "gzip",
    overwrite: bool = False,
    dry_run: bool = False,
    max_bytes: int = DEFAULT_BLOCK_BYTES,
    warn_others: bool = True,
) -> dict:
    """Write ``src`` into ``out`` as an HDF5 dataset carrying its frame and position.

    ``dataset`` defaults to :data:`DEFAULT_DATASET`. An existing file is **added to**
    rather than replaced, provided its recorded frame matches; a dataset name already in
    use needs another ``dataset`` or ``overwrite=True``.

    ``voxel_offset`` is in the same axis order as ``axes``, whole voxels, and names the
    position of the piece's ``(0, ...)`` corner in the volume it belongs to.

    ``kind`` records what the voxels mean (:data:`KIND_ATTR`) and defaults to what the
    source recorded; an image stack or bare array records none, so pass it or leave it
    unrecorded. It is never guessed from the dtype.

    A DVID or webknossos source (``dvid://`` / ``wk://``) takes ``level`` like any
    pyramid, records its voxel size and kind, and takes the same read options as
    :func:`~neu_vol.convert`: ``supervoxels`` and ``prefer_locked`` for DVID,
    ``wk_mapping`` for webknossos. Give it a crop; a whole instance is far too large
    for one file.

    h5py reads and writes only filesystems, so an HDF5 ``src`` or an ``out`` on an object
    store (``s3://`` / ``gs://``) is worked on as a temporary local copy: downloaded first
    (``out`` only if it exists, so it can be added to), and ``out`` uploaded once the write
    succeeds. :mod:`neu_vol.transfer` says how the bytes move.

    ``dtype`` narrower than the source's is refused if any value falls outside its range
    (:func:`check_fits`); ``truncate=True`` casts anyway, as numpy's ``astype`` does.

    To write an array that is **already in memory**, use
    :func:`neu_vol.write_piece` — same file layout, no round trip through a store.
    To repack several datasets of one file, :func:`pack_hdf5_datasets`.
    """
    params = dict(locals())
    if _needs_staging(src, src_format, out):
        with _staged(src, src_format, out, dry_run=dry_run) as (local_src, local_out):
            plan = pack_hdf5(**{**params, "src": local_src, "out": local_out})
        return _as_given(plan, src, local_src, out)

    from ..backends.hdf5 import require_local_path

    out = require_local_path(out, "the HDF5 file to write")
    name = hdf5_dataset_name(dataset)
    axes = tuple(axes)

    spec, frame = resolve_source(src, src_format, level, src_dataset, voxel_size_field,
                                 offset_field,
                                 options={"supervoxels": supervoxels,
                                          "prefer_locked": prefer_locked,
                                          "wk_mapping": wk_mapping})
    from ..backends.base import spec_names_path

    if spec_names_path(spec, out):
        # The source is read while the destination is open for writing, which HDF5 does
        # not allow on one file.
        raise ValueError(
            f"src and out are the same file ({out}), which HDF5 cannot do: the source is "
            f"read while the destination is open for writing, and one file cannot be both. "
            f"Pack into a new file, or `read_piece` the array and `write_piece` it back.")
    if voxel_size is None:
        voxel_size = frame.get("voxel_size")
        if voxel_size is None:
            raise ValueError(
                "voxel_size is required: this source records no physical scale — an image "
                "stack, an HDF5 file or a bare array records none — and attaching one is "
                "the point of packing")
    if units is None:
        units = frame.get("units") or "nm"
    voxel_size = tuple(float(v) for v in voxel_size)

    backend = open_backend(spec)
    shape = tuple(int(s) for s in backend.shape)
    src_dtype = np.dtype(backend.dtype)
    out_dtype = np.dtype(dtype or src_dtype)
    n_spatial = len(axes)
    if len(shape) not in (n_spatial, n_spatial + 1):
        raise ValueError(f"source is {len(shape)}-D, which is neither {n_spatial} spatial "
                         f"axes {axes} nor those plus a leading channel axis")
    has_channels = len(shape) == n_spatial + 1

    # Background remap under the crop, as in `convert`: it corrects the data itself. The
    # crop's origin becomes the piece's voxel_offset unless the caller gives one.
    if background:
        spec = {"backend": "remap", "source": dict(spec),
                "values": [int(v) for v in background], "to": 0}
        backend = open_backend(spec)
        logger.info("treating %s as background: replaced with 0 while packing",
                    [int(v) for v in background])

    crop_origin = None
    if crop_start is not None or crop_stop is not None:
        from .convert import _resolve_crop

        spec, shape, crop_origin = _resolve_crop(
            spec, shape, start=crop_start, stop=crop_stop, n_spatial=n_spatial,
            has_channels=has_channels, pad_value=0, clip=True)
        backend = open_backend(spec)
    if voxel_offset is None:
        # Where the source says it sits, plus where in it the crop started.
        base = frame.get("voxel_offset") or (0,) * n_spatial
        voxel_offset = (tuple(b + c for b, c in zip(base, crop_origin))
                        if crop_origin is not None else tuple(base))
    voxel_offset = tuple(int(v) for v in voxel_offset)
    if not (len(voxel_size) == len(voxel_offset) == len(axes)):
        raise ValueError(f"voxel_size {voxel_size}, voxel_offset {voxel_offset} and axes "
                         f"{axes} must describe the same number of axes")

    if kind is None:
        kind = frame.get("kind")

    # The frame describes the spatial axes; a channel axis is not a physical dimension.
    frame = {"voxel_size": voxel_size, "units": units, "axes": axes}

    _existing_frame, existing_datasets, replacing = check_hdf5_target(
        out, name, frame, overwrite=overwrite, voxel_size_field=voxel_size_field)

    others = [d for d in existing_datasets if d != name]
    block = _block_shape(shape, out_dtype.itemsize, max_bytes)
    plan = {
        "out": out, "dataset": name, "src_spec": spec, "shape": shape,
        "dtype": str(out_dtype), "src_dtype": str(src_dtype),
        "narrowing": not np.can_cast(src_dtype, out_dtype, "safe"),
        "truncate": truncate, "has_channels": has_channels,
        "voxel_size": voxel_size, "voxel_offset": voxel_offset, "units": units,
        "axes": axes, "kind": kind, "chunk": tuple(chunk) if chunk else None,
        "compression": compression, "nbytes": float(out_dtype.itemsize) * math.prod(shape),
        "blocks": len(list(iter_blocks(shape, block))), "block_shape": block,
        "replacing": replacing, "appending": bool(existing_datasets) and not replacing,
        "other_datasets": others, "level": level, "crop_origin": crop_origin,
    }
    if dry_run:
        return plan

    # A second volumetric dataset means every later reader has to name one.
    if others and warn_others:
        logger.warning("%s will hold %d volumetric datasets (%s); readers must name one "
                       "— `neu-vol write --dataset %s`, or HDF5Backend's `dataset` key",
                       out, len(others) + 1, ", ".join(others + [name]), name)

    write_hdf5_array(out, name, backend.read_region, shape=shape, dtype=out_dtype,
                     voxel_size=voxel_size, voxel_offset=voxel_offset, units=units,
                     axes=axes, kind=kind, chunk=chunk, compression=compression,
                     replacing=replacing, block=block, has_channels=has_channels,
                     voxel_size_field=voxel_size_field, offset_field=offset_field,
                     truncate=truncate)
    logger.info("packed %s -> %s%s (%s %s, %d block(s))", spec, out, name, shape,
                out_dtype, plan["blocks"])
    return plan


def pack_hdf5_datasets(src: str, out: str, datasets: Sequence[str] | None = None, *,
                       src_format: str | None = None, dry_run: bool = False,
                       **kwargs) -> list[dict]:
    """Repack several datasets of one HDF5 file into ``out``, each under its own name.

    ``datasets`` defaults to every volumetric dataset in ``src``. Each keeps the frame and
    ``voxel_offset`` it records; the other :func:`pack_hdf5` arguments (``dtype``,
    ``truncate``, ``chunk``, ``compression``, ``overwrite``, ...) apply to all of them. A
    remote ``src`` or ``out`` is transferred once for the whole set, not once per dataset.
    """
    from ..backends.hdf5 import datasets as volumetric
    from ..detect import detect_file_backend

    chosen_here = ("dataset", "src_dataset")
    per_dataset = ("voxel_offset", "crop_start", "crop_stop")
    if any(k in kwargs for k in chosen_here):
        raise TypeError(f"{'/'.join(chosen_here)} are chosen per dataset here; pass `datasets`")
    if any(kwargs.get(k) is not None for k in per_dataset):
        raise ValueError("an offset or a crop describes ONE dataset; repack datasets one at a "
                         "time with pack_hdf5 to give them")

    with _staged(src, src_format, out, dry_run=dry_run) as (local_src, local_out):
        if (src_format or detect_file_backend(local_src)) != "hdf5":
            raise ValueError(f"{src} is not an HDF5 file, so it has no datasets to repack")
        names = list(datasets) if datasets else volumetric(local_src)
        if not names:
            raise ValueError(f"{src} holds no volumetric dataset")
        plans = [pack_hdf5(local_src, local_out, src_dataset=n, dataset=n,
                           src_format="hdf5", dry_run=dry_run, warn_others=False, **kwargs)
                 for n in names]
    return [_as_given(p, src, local_src, out) for p in plans]
