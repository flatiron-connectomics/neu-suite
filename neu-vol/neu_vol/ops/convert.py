"""Convert an existing volume (any source backend) into a multiscale zarr v3 or
neuroglancer-precomputed volume, whole or cropped to a box.

Generalizes ``ingest``: the source is any registered backend. Coordinate metadata
(``voxel_size``/``offset``/``units``/``axes``) is read from the source when it
carries it — OME-NGFF zarr groups, precomputed ``info``, a DVID instance, and an HDF5
file that records its own frame beside the array; anything the caller passes explicitly
overrides the read value. Sources that record nothing (bare arrays, image stacks, a
plain HDF5 file) require the caller to supply at least ``voxel_size``.

``crop_start``/``crop_stop`` restrict the copy to a box **without changing the model
space**: the physical offset shifts by the crop origin, so the output lands on top of the
source in a viewer. The crop view wraps the resolved ``data_spec``, so a cropped copy
reads through the reader detection chose. ``ops/roi.py`` is a thin wrapper over this.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ..detect import detect_backend, require_populated_scale
from ..source_metadata import location_spec, read_source_metadata
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def _resolve_crop(
    data_spec: dict,
    src_shape: Sequence[int],
    *,
    start: Sequence[int] | None,
    stop: Sequence[int] | None,
    n_spatial: int,
    has_channels: bool,
    pad_value: float,
    clip: bool,
) -> tuple[dict, tuple[int, ...], tuple[int, ...]]:
    """A crop view over ``data_spec``: ``(spec, output shape, spatial start)``.

    ``start``/``stop`` are voxel indices into ``data_spec``'s own (level-0) grid, in
    spatial ``(z, y, x)``; a leading channel axis is always taken in full. Either may
    be ``None``, meaning the volume's own bound, so a half-open box needs only the end
    that matters.

    ``clip`` (the default) trims the box to the source extent; without it the margin is
    filled with ``pad_value``, the crop-and-pad contract of :func:`~neu_vol.extract_roi`.
    """
    spatial = tuple(int(s) for s in (src_shape[1:] if has_channels else src_shape))
    start = (0,) * n_spatial if start is None else tuple(int(v) for v in start)
    stop = spatial if stop is None else tuple(int(v) for v in stop)
    if len(start) != n_spatial or len(stop) != n_spatial:
        raise ValueError(
            f"crop start/stop must have {n_spatial} spatial entries, got "
            f"{start} / {stop}"
        )
    if clip:
        c_start = tuple(max(0, a) for a in start)
        c_stop = tuple(min(b, d) for b, d in zip(stop, spatial))
        if (c_start, c_stop) != (start, stop):
            logger.info("crop clipped to the source extent: [%s:%s] -> [%s:%s]",
                        start, stop, c_start, c_stop)
            start, stop = c_start, c_stop
    out_spatial = tuple(b - a for a, b in zip(start, stop))
    if any(s <= 0 for s in out_spatial):
        raise ValueError(
            f"empty crop: start={start} stop={stop} (source spatial shape {spatial})"
        )
    origin = ((0,) + start) if has_channels else start
    out_shape = ((int(src_shape[0]),) + out_spatial) if has_channels else out_spatial
    spec = {"backend": "crop", "source": dict(data_spec), "origin": list(origin),
            "shape": list(out_shape), "pad_value": pad_value}
    return spec, out_shape, start


def _resolve_masks(
    data_spec: dict,
    src_shape: Sequence[int],
    boxes: Sequence[Sequence[Sequence[int]]],
    *,
    value: float,
    n_spatial: int,
    has_channels: bool,
) -> dict:
    """A mask view over ``data_spec``, blanking each spatial ``(lo, hi)`` box.

    Boxes are spatial ``(z, y, x)`` in the source's own level-0 voxels; a channel axis is
    always masked in full, since "exclude this region" is not a per-channel statement.

    **A box that misses the volume raises**: that is what a box in the wrong axis order or
    scale looks like, and copying everything instead must not pass quietly.
    """
    spatial = tuple(int(s) for s in (src_shape[1:] if has_channels else src_shape))
    full = []
    for lo, hi in boxes:
        if len(lo) != n_spatial or len(hi) != n_spatial:
            raise ValueError(f"mask box must have {n_spatial} spatial entries per corner, "
                             f"got {tuple(lo)} / {tuple(hi)}")
        lo = tuple(int(v) for v in lo)
        hi = tuple(int(v) for v in hi)
        if any(b <= a for a, b in zip(lo, hi)):
            raise ValueError(f"empty mask box {lo}:{hi}")
        if any(b <= 0 or a >= d for a, b, d in zip(lo, hi, spatial)):
            raise ValueError(
                f"mask box {lo}:{hi} does not intersect the volume (spatial shape "
                f"{spatial}), so nothing would be excluded. Check the axis order (these "
                f"are z,y,x) and the scale the box is in.")
        if has_channels:
            lo, hi = (0,) + lo, (int(src_shape[0]),) + hi
        full.append([list(lo), list(hi)])
    logger.info("masking %d region(s) with fill value %r", len(full), value)
    return {"backend": "mask", "source": dict(data_spec), "boxes": full, "value": value}


def with_source_options(spec: dict, *, prefer_locked: bool = False,
                        supervoxels: bool = False, wk_mapping: str | None = None) -> dict:
    """``spec`` with the server-source read options set, each only when asked for.

    ``prefer_locked`` and ``supervoxels`` apply to DVID, ``wk_mapping`` to webknossos. An
    option given for another source raises rather than being dropped: dropping it would
    read a different segmentation than the one asked for, and nothing downstream could
    tell. Unset options leave the spec's own values alone.
    """
    backend = spec.get("backend")
    for asked, wanted, name in ((prefer_locked, "dvid", "prefer_locked=True"),
                                (supervoxels, "dvid", "supervoxels=True"),
                                (wk_mapping, "webknossos", "wk_mapping")):
        if asked and backend != wanted:
            raise ValueError(f"{name} applies to {'DVID' if wanted == 'dvid' else wanted} "
                             f"sources only, but the source is {backend!r}")
    spec = dict(spec)
    if prefer_locked:
        spec["prefer_locked"] = True
    if supervoxels:
        spec["supervoxels"] = True
    if wk_mapping:
        spec["mapping"] = str(wk_mapping)
    return spec


def convert(
    src: str | dict,
    dst: str,
    *,
    voxel_size: Sequence[float] | None = None,
    src_format: str | None = None,
    src_level: int | None = None,
    units: str | None = None,
    axes: Sequence[str] | None = None,
    offset: Sequence[float] | None = None,
    has_channels: bool | None = None,
    crop_start: Sequence[int] | None = None,
    crop_stop: Sequence[int] | None = None,
    pad_value: float = 0,
    clip_crop: bool = True,
    mask_boxes: Sequence[Sequence[Sequence[int]]] | None = None,
    mask_value: float = 0,
    background: Sequence[int] | None = None,
    supervoxels: bool = False,
    prefer_locked: bool = False,
    wk_mapping: str | None = None,
    crop_to_data: bool = True,
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str | None = None,
    encoding: str | None = None,
    compressed_segmentation_block_size: Sequence[int] = (8, 8, 8),
    multiscale: bool = True,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    resume: bool = False,
    verify: bool = False,
    sparse: bool = False,
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Convert ``src`` into a multiscale volume at ``dst``, whole or cropped.

    ``src`` is a path or a full backend spec dict. For a path, ``src_format`` is
    auto-detected unless given: ``info``->precomputed, ``zarr.json``->zarr3,
    ``.zarray``/``.zgroup``->zarr2, a ``dvid://`` URL->dvid, and — from the name, since
    neither has a marker object — an HDF5 file or a glob, file or directory of 2D slices.
    Metadata not passed explicitly is taken from the source where available;
    ``voxel_size`` is required if the source carries none, which an image stack always
    does and an HDF5 file does unless it records its own frame.

    With ``resume=True`` an interrupted run continues, skipping already-done blocks
    instead of recreating them.

    ``background`` names the values the source uses for background (manual segmentation
    numbered from 0 makes it 1) and replaces them with 0 as the source is read. Afterwards
    is too late: an all-background block of 1s is not all-fill, so it would be stored.

    ``sparse=True`` skips **pyramid** tasks whose input holds no stored chunk, which on a
    sparse volume is nearly all of them. The level-0 copy is never skipped, since its
    source is foreign.

    ``mask_boxes`` is a list of spatial ``(lo, hi)`` pairs to *exclude*: everything else
    is copied and those regions are written as ``mask_value``. Because the pyramid is
    derived from the output's own level 0, the hole propagates to every level. A box that
    does not intersect the volume raises rather than quietly copying everything.

    ``crop_start``/``crop_stop`` copy only ``src[crop_start:crop_stop]`` (spatial
    ``(z, y, x)`` voxels of the source's level 0, either bound optional). The output's
    physical offset shifts by the crop origin, so it stays in the source's coordinate
    frame. The pyramid is built **from the cropped level 0**, so for a crop origin that is
    not a multiple of the cumulative factor, coarse voxels straddle the source's differently.

    ``crop_to_data`` (default on) crops a source that records the box its data lies in (a
    webknossos layer's bounding box) to that box when no crop is given, keeping the frame.
    ``False`` keeps the full origin-anchored extent.
    """
    if isinstance(src, dict):
        src_spec = dict(src)
    else:
        fmt = src_format or detect_backend(src)
        if fmt is None:
            raise ValueError(
                f"could not detect source format at {src!r}: no info / zarr.json / "
                f".zarray marker, and the name is neither an HDF5 file nor a glob, "
                f"file or directory of 2D slices. Pass src_format= explicitly"
            )
        # `location_spec` knows every source's address form.
        src_spec = location_spec(src, fmt)

    if src_level is not None:
        # Which source scale becomes the output's level 0 (default: the finest that
        # stores data), at that scale's own voxel size, never an assumed 2**level.
        src_spec["scale_index"] = int(src_level)

    src_spec = with_source_options(src_spec, prefer_locked=prefer_locked,
                                   supervoxels=supervoxels, wk_mapping=wk_mapping)
    meta = read_source_metadata(src_spec)

    # The array/scale to actually read (level 0 of an OME group / the finest precomputed
    # scale that stores data, or `src_level` if it was named).
    data_spec = meta["data_spec"] if meta else src_spec

    # Refuse, before any work, a scale that was declared but never written: it reads as
    # the fill value everywhere and the run would write a volume of zeros. A mapped view
    # is checked through its source.
    require_populated_scale(data_spec["source"] if data_spec.get("backend") == "mapped"
                            else data_spec, op="convert")

    # Warned here, not in the metadata reader, which read-only inspection also uses.
    if data_spec.get("backend") == "dvid" and meta:
        from ..dvid import resolve_node, spec_url

        if not resolve_node(data_spec)["locked"]:
            logger.warning(
                "dvid node %s is OPEN (not locked), so its data can change while this "
                "runs and this export is not reproducible. provenance.json will name "
                "the node. For an immutable snapshot re-run with --dvid-locked, which "
                "takes the newest locked ancestor. Source: %s",
                data_spec.get("uuid"), spec_url(data_spec))

    # The viewer shows a webknossos segmentation through its default mapping, so a
    # supervoxel export differs from what was looked at — legitimately (it is the input
    # for comparing views), but never silently.
    if data_spec.get("backend") == "webknossos" and meta:
        if data_spec.get("mapping"):
            logger.info("webknossos: reading with mapping %s, applied by the server",
                        data_spec["mapping"])
        elif meta.get("default_mapping"):
            logger.warning(
                "webknossos: reading SUPERVOXELS (no mapping). The viewer shows this layer "
                "through %r; pass --wk-mapping %s for those ids",
                meta["default_mapping"], meta["default_mapping"])

    # Resolve coordinate metadata: explicit arg > source metadata > default.
    if voxel_size is None:
        if not meta:
            raise ValueError(
                "voxel_size is required: source has no readable coordinate metadata"
            )
        voxel_size = meta["voxel_size"]
    if axes is None:
        axes = meta["spatial_axes"] if meta else ("z", "y", "x")
    if units is None:
        units = (meta.get("units") if meta else None) or "nm"
    if offset is None and meta:
        offset = meta["offset"]
    if has_channels is None and meta is not None:
        has_channels = meta["has_channels"]
    if kind is None:
        # Explicit > what the source records > image. Treating a segmentation as an
        # image would average label ids into ids that were never in the data.
        kind = (meta or {}).get("kind") or "image"
        if meta and meta.get("kind"):
            logger.info("kind=%s, from the source's own metadata", kind)

    if (crop_to_data and crop_start is None and crop_stop is None
            and meta and meta.get("data_box")):
        crop_start, crop_stop = (tuple(int(v) for v in c) for c in meta["data_box"])
        logger.info("cropping to the source's data box [%s:%s] (crop_to_data; the "
                    "output keeps the source's frame)", crop_start, crop_stop)

    src_backend = open_backend(data_spec)
    src_shape = src_backend.shape
    n_spatial = len(axes)

    if has_channels is None:
        has_channels = len(src_shape) == n_spatial + 1
    if len(src_shape) != n_spatial + (1 if has_channels else 0):
        raise ValueError(
            f"source ndim {len(src_shape)} incompatible with {n_spatial} spatial axes "
            f"{'+ channel' if has_channels else ''}"
        )
    num_channels = int(src_shape[0]) if has_channels else 1

    # The views wrap the resolved data_spec, so reads go through the reader detection
    # chose. Mask first, crop over it, so mask boxes are in the source's own coordinates.
    read_spec, read_shape = data_spec, src_shape
    out_offset = tuple(offset) if offset else (0.0,) * n_spatial
    if background:
        # Innermost: it corrects the data itself, before any box is laid over it.
        read_spec = {"backend": "remap", "source": dict(read_spec),
                     "values": [int(v) for v in background], "to": 0}
        logger.info("treating %s as background: replaced with 0 as the source is read",
                    [int(v) for v in background])
    if mask_boxes:
        read_spec = _resolve_masks(read_spec, src_shape, mask_boxes, value=mask_value,
                                   n_spatial=n_spatial, has_channels=has_channels)
    if crop_start is not None or crop_stop is not None:
        read_spec, read_shape, start = _resolve_crop(
            read_spec, src_shape, start=crop_start, stop=crop_stop,
            n_spatial=n_spatial, has_channels=has_channels, pad_value=pad_value,
            clip=clip_crop)
        out_offset = tuple(o + a * v for o, a, v in zip(out_offset, start, voxel_size))
        logger.info("crop [%s:%s] of %s -> shape=%s offset=%s nm",
                    start, tuple(a + b for a, b in zip(start, read_shape[-n_spatial:])),
                    src_shape, read_shape, out_offset)
    logger.info("convert %s -> %s | shape=%s dtype=%s channels=%s voxel_size=%s",
                data_spec, dst, read_shape, src_backend.dtype,
                num_channels if has_channels else 1, tuple(voxel_size))

    summary = materialize_multiscale(
        src_spec=read_spec,
        src_shape=read_shape,
        src_dtype=str(src_backend.dtype),
        dst=dst,
        profile=profile,
        voxel_size=tuple(voxel_size),
        offset=out_offset,
        units=units,
        spatial_axes=tuple(axes),
        has_channels=has_channels,
        num_channels=num_channels,
        dtype=dtype,
        kind=kind,
        encoding=encoding,
        compressed_segmentation_block_size=tuple(compressed_segmentation_block_size),
        multiscale=multiscale,
        factors=factors,
        max_levels=max_levels,
        min_dim=min_dim,
        name=name,
        chunk=chunk,
        shard=shard,
        client=client,
        npartitions=npartitions,
        delete_existing=delete_existing,
        resume=resume,
        verify=verify,
        sparse=sparse,
        progress_path=progress_path,
        validate=validate,
    )

    # After the data, so a run that died half way leaves no record claiming a volume.
    # The spec is the resolved one: for DVID the concrete node, not the branch ref.
    from . import provenance as _provenance

    _provenance.write(dst, _provenance.build_record(
        src_spec=(meta or {}).get("provenance_spec") or src_spec,
        dst=dst,
        kind=kind,
        voxel_size=list(voxel_size),
        crop_start=list(crop_start) if crop_start is not None else None,
        crop_stop=list(crop_stop) if crop_stop is not None else None,
        mask_boxes=[[list(lo), list(hi)] for lo, hi in mask_boxes] if mask_boxes else None,
        background=list(background) if background else None,
        num_levels=summary.get("num_levels"),
        status_counts=summary.get("status_counts"),
    ))
    return summary
