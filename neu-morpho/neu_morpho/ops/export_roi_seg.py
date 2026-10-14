"""Copy the ROI's segmentation into the output as a precomputed volume.

So meshes and skeletons can be inspected against the labels they came from. Each scale
carries its ``voxel_offset``, so the labels land in the same physical-nm space as the
meshes.

**Every source scale is copied, not regenerated**: each is cropped directly into the
matching destination scale, where downsampling from level 0 would re-read far more data
and might not reproduce the source's own rule.

Cost is uneven: scale 0 usually dominates by an order of magnitude. :func:`scale_cost`
reports it and ``scale_indices`` restricts the range. By default the region is
**expanded to whole blocks**, matching what stage 1 meshed (roi.py).
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any, Sequence

import numpy as np

from blockrun import Manifest, block_map, iter_blocks

from .manifest import check_manifest_matches_output, group_counts, is_complete

logger = logging.getLogger(__name__)

# The only precomputed profile neu-vol ships; the name is about where it
# is usually written, not a requirement to be on S3.
PRECOMPUTED_PROFILE = "s3-neuroglancer"
# Copy in chunks larger than the stored chunk (128^3) to keep the task count sane.
# Must stay a multiple of it so concurrent writers never share a chunk.
DEFAULT_COPY_BLOCK = (512, 512, 512)


def block_align(roi: Sequence[int], block_shape: Sequence[int],
                shape: Sequence[int]) -> tuple[int, ...]:
    """Expand an ROI to whole blocks of the global grid, clipped to the volume."""
    lo = [int(roi[a]) // block_shape[a] * block_shape[a] for a in range(3)]
    hi = [min(int(shape[a]),
              math.ceil(int(roi[3 + a]) / block_shape[a]) * block_shape[a]) for a in range(3)]
    return tuple(lo + hi)


def _region_at(nm_lo: np.ndarray, nm_hi: np.ndarray, voxel_size: Sequence[float],
               shape: Sequence[int]) -> tuple[list[int], list[int]]:
    """The nm box expressed in one scale's voxels, clipped to that scale's shape."""
    v = np.asarray(voxel_size, float)
    lo = np.floor(nm_lo / v).astype(int)
    hi = np.ceil(nm_hi / v).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape, int))
    return lo.tolist(), hi.tolist()


def source_levels(src, roi_voxel_size) -> list[tuple]:
    """``[(ScaleInfo, read_spec)]`` for the source, finest first.

    A source with no pyramid metadata (a bare array, an HDF5 dataset) is one level at
    ``roi_voxel_size``.
    """
    from neu_vol.backends.base import open_backend

    from neu_lib import Frame, ScaleInfo
    from neu_vol import read_scales, scale_spec

    try:
        return [(s, scale_spec(src, s.index)) for s in read_scales(src)]
    except (ValueError, KeyError, IndexError):
        spec = dict(src) if isinstance(src, dict) else {"path": src}
        shape = tuple(int(x) for x in open_backend(spec).shape)
        logger.info("no pyramid metadata at the source; treating it as a single level "
                    "at %s nm", tuple(roi_voxel_size))
        # Origin zero: such a source states no offset either.
        return [(ScaleInfo(index=0, shape=shape,
                           frame=Frame(voxel_size_nm=tuple(
                               float(v) for v in roi_voxel_size))), spec)]


def scale_cost(src, roi, roi_voxel_size, *, block_shape=(256, 256, 256),
               align_to_blocks: bool = True, scale_indices=None) -> list[dict]:
    """Per-scale shape and byte estimate for an export — check before running one."""
    levels = source_levels(src, roi_voxel_size)
    if scale_indices is None:
        scale_indices = range(len(levels))
    nm_lo, nm_hi = _roi_nm([s for s, _ in levels], roi, roi_voxel_size,
                           block_shape, align_to_blocks)
    out = []
    for si in sorted(scale_indices):
        s = levels[si][0]
        lo, hi = _region_at(nm_lo, nm_hi, s.voxel_size, s.shape)
        shape = [b - a for a, b in zip(lo, hi)]
        n = int(np.prod(shape)) if all(x > 0 for x in shape) else 0
        out.append({"scale": si, "voxel_size": s.voxel_size, "shape": tuple(shape),
                    "n_voxels": n, "raw_gb": n * 8 / 1e9})
    return out


def _roi_nm(scales, roi, roi_voxel_size, block_shape, align_to_blocks):
    """The ROI as a physical-nm box, block-aligned on the grid it was quoted in."""
    from ..roi import clip_to_shape, parse_roi

    # find the scale the roi is quoted in, so alignment uses the right grid/shape
    ref = min(scales, key=lambda s: sum(abs(a - b) for a, b in
                                        zip(s.voxel_size, roi_voxel_size)))
    region = clip_to_shape(parse_roi(roi), ref.shape)
    if region is None:
        region = (0, 0, 0, *ref.shape)
    elif align_to_blocks:
        region = block_align(region, block_shape, ref.shape)
    v = np.asarray(roi_voxel_size, float)
    return np.asarray(region[:3], float) * v, np.asarray(region[3:], float) * v


def _copy_block(block, *, src_spec: dict, dst_spec: dict, src_origin: Sequence[int],
                attempts: int = 5) -> tuple:
    """Copy one block, retrying transient store failures.

    Unlike meshing's stage 1, a copy block feeds no aggregation and is idempotent, so a
    transient store error is retried rather than ending a long copy. ``"written"`` /
    ``"empty"`` describe the data: TensorStore writes no object for an all-zero chunk.
    """
    from neu_vol.backends.base import open_backend
    from neu_vol.retry import with_retry

    src_region = tuple(slice(o + s.start, o + s.stop)
                       for o, s in zip(src_origin, block.region))

    def copy() -> bool:
        data = open_backend(src_spec).read_region(src_region)
        open_backend(dst_spec).write_region(block.region, data)
        return bool(data.any())

    nonzero = with_retry(copy, attempts=attempts, label=f"block {block.index}")
    return (block.index, "written" if nonzero else "empty")


def export_roi_seg(
    src: dict | str,
    out_dir: str,
    *,
    roi: Sequence[int] | str | None,
    roi_voxel_size: Sequence[float],
    scale_indices: Sequence[int] | None = None,
    block_shape: Sequence[int] = (256, 256, 256),
    align_to_blocks: bool = True,
    copy_block: Sequence[int] = DEFAULT_COPY_BLOCK,
    encoding: str | None = "compressed_segmentation",
    progress_path: str | None = None,
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    resume: bool = True,
) -> dict:
    """Write the ROI's labels to ``out_dir`` as a multiscale precomputed volume.

    ``roi`` is in the voxels of the scale whose voxel size is ``roi_voxel_size``;
    ``None`` copies the whole volume. ``scale_indices`` selects which source
    scales to copy (default: all) — see :func:`scale_cost` first, since scale 0
    usually dominates.
    """
    from neu_vol.backends.base import open_backend
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import precomputed_create_spec

    levels = source_levels(src, roi_voxel_size)
    if scale_indices is None:
        scale_indices = list(range(len(levels)))
    scale_indices = sorted(int(i) for i in scale_indices)
    for si in scale_indices:
        if not 0 <= si < len(levels):
            raise IndexError(f"scale {si} out of range (source has {len(levels)})")

    nm_lo, nm_hi = _roi_nm([s for s, _ in levels], roi, roi_voxel_size,
                           block_shape, align_to_blocks)
    src_dtype = str(open_backend(levels[scale_indices[0]][1]).dtype)

    # The manifest is appended to, so it needs a filesystem; the driver passes one in
    # the work dir.
    if progress_path:
        progress = progress_path
    else:
        from neu_vol import is_local
        if not is_local(out_dir):
            raise ValueError(
                f"progress_path is required when out_dir is remote ({out_dir}): "
                "the manifest is an appended JSONL file and needs a filesystem")
        progress = out_dir.rstrip("/") + "/../progress.seg.jsonl"
    manifest = Manifest(progress)
    manifest.load() if resume else manifest.reset()
    # The manifest lives apart from the data, so a cleared destination is refused.
    check_manifest_matches_output(manifest, out_dir, stage="seg",
                                  progress_path=progress, resume=resume)

    written: list[dict] = []
    try:
        for dest_i, si in enumerate(scale_indices):
            s, src_spec = levels[si]
            lo, hi = _region_at(nm_lo, nm_hi, s.voxel_size, s.shape)
            shape = [b - a for a, b in zip(lo, hi)]
            if any(x <= 0 for x in shape):
                logger.warning("scale %d: ROI does not intersect this level, skipping", si)
                continue

            spec = precomputed_create_spec(
                PRECOMPUTED_PROFILE, out_dir, shape, src_dtype,
                resolution_zyx=s.voxel_size, scale_index=dest_i, type_="segmentation",
                encoding=encoding, voxel_offset_zyx=lo)
            be = TensorStoreBackend.open_or_create(
                spec, resume=resume,
                # precomputed scales share one info: only scale 0 may delete it
                delete_existing=(delete_existing and dest_i == 0))
            dst_spec = be.to_spec()

            group = f"seg-{si}"
            blocks = list(iter_blocks(tuple(shape), tuple(copy_block)))
            todo = [b for b in blocks if not (resume and is_complete(manifest, group, b.index))]
            n = int(np.prod(shape))
            logger.info("scale %d (%.0f nm): %s = %.2f Gvox, %d blocks (%d to do) "
                        "offset %s", si, s.voxel_size[0], tuple(shape), n / 1e9,
                        len(blocks), len(todo), tuple(lo))

            worker = functools.partial(_copy_block, src_spec=src_spec,
                                       dst_spec=dst_spec, src_origin=tuple(lo))
            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=lambda r, g=group: manifest.record(g, r))
            written.append({"scale": si, "shape": tuple(shape), "voxel_offset": tuple(lo),
                            "n_voxels": n, "voxel_size": s.voxel_size,
                            "counts": group_counts(manifest, group)})
    finally:
        manifest.close()

    return {"out_dir": out_dir, "scales": written,
            "n_voxels_total": sum(w["n_voxels"] for w in written),
            "region_nm": (tuple(nm_lo.tolist()), tuple(nm_hi.tolist())),
            "progress_path": progress}
