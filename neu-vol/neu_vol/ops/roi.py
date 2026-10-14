"""Extract / crop / pad a region of interest into a new multiscale volume.

A thin wrapper over :func:`~neu_vol.convert`, adding the crop-*and*-pad contract:
``start`` may be negative and ``stop`` may exceed the source extent, and the margin is
filled with ``pad_value`` rather than trimmed. Metadata, including the source's offset,
comes from the source as in ``convert``, so the ROI lands in the source's frame.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .convert import convert

logger = logging.getLogger(__name__)


def extract_roi(
    src: str | dict,
    dst: str,
    start: Sequence[int],
    stop: Sequence[int],
    voxel_size: Sequence[float] | None = None,
    *,
    src_format: str | None = None,
    pad_value: float = 0,
    units: str | None = None,
    axes: Sequence[str] | None = None,
    offset: Sequence[float] | None = None,
    has_channels: bool | None = None,
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str = "image",
    encoding: str | None = None,
    compressed_segmentation_block_size: Sequence[int] = (8, 8, 8),
    multiscale: bool = False,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    resume: bool = False,
    verify: bool = False,
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Extract ``src[start:stop]`` (spatial) into a new volume at ``dst``.

    ``start`` may be negative and ``stop`` may exceed the source extent; the
    out-of-bounds margin is filled with ``pad_value`` (crop *and* pad in one). To
    trim to the volume instead — the right default when copying real data — call
    ``convert(..., crop_start=, crop_stop=)``, which clips.

    ``voxel_size``/``units``/``axes``/``offset`` are read from the source when it
    carries them and only need passing for a source that does not (bare arrays, HDF5,
    image stacks) or to override what it says.
    """
    return convert(
        src, dst, voxel_size=voxel_size, src_format=src_format, units=units,
        axes=axes, offset=offset, has_channels=has_channels,
        crop_start=start, crop_stop=stop, pad_value=pad_value, clip_crop=False,
        profile=profile, chunk=chunk, shard=shard, dtype=dtype, kind=kind,
        encoding=encoding,
        compressed_segmentation_block_size=compressed_segmentation_block_size,
        multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, name=name, client=client, npartitions=npartitions,
        delete_existing=delete_existing, resume=resume, verify=verify,
        progress_path=progress_path, validate=validate,
    )
