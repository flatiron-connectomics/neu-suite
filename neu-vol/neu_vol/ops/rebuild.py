"""Regenerate a multiscale pyramid in place, from a level you trust.

When the coarse end of a pyramid is wrong — blank, aliased, or built from a bad
intermediate — the fix is to re-derive it, not to reconvert. Downsampling is
**cascaded** (level N is built from level N-1, not from level 0), so a single bad
level poisons everything above it and re-deriving must start at or below the
lowest good one.

    rebuild_pyramid("s3://bucket/prefix/volume", start_level=2, kind="image")

Levels at or below ``start_level`` are never opened for writing. Everything above is
rebuilt with the original factors, since the schedule is recomputed from level 0.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ..detect import detect_backend, require_chunked_volume
from ..source_metadata import read_source_metadata
from ..location import default_progress_path
from ..profiles import PROFILES
from ._multiscale import materialize_multiscale
from .create import DEFAULT_MAX_LEVELS

logger = logging.getLogger(__name__)


def resolve_max_levels(recorded: Sequence[Any] | None) -> tuple[int, str]:
    """The level cap for a rebuild, and why — from the volume where it says anything.

    A pyramid's depth is the volume's own, so a rebuild matches it; a conversion's default
    would leave the top levels of a deeper pyramid untouched. A volume recording one level
    or none has no pyramid to match (``write`` and ``relabel`` tell you to run
    ``downsample`` to build one), so it gets the conversion default.
    """
    n = len(recorded or ())
    if n > 1:
        return n, "matching the volume's recorded levels"
    return DEFAULT_MAX_LEVELS, ("the volume records no pyramid to match, so building a "
                                "default one — pass --max-levels to choose its depth")


def _profile_for(fmt: str) -> str:
    for name, prof in PROFILES.items():
        if prof.format == fmt:
            return name
    raise ValueError(f"no storage profile for format {fmt!r}")


def rebuild_pyramid(
    dst: str,
    *,
    start_level: int,
    kind: str | None = None,
    profile: str | None = None,
    voxel_size: Sequence[float] | None = None,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int | None = None,
    min_dim: int = 128,
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    encoding: str | None = None,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = False,
    verify: bool = False,
    sparse: bool = False,
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Rebuild levels ``start_level+1 …`` of the volume at ``dst``, in place.

    ``start_level`` is the level to derive from — it is read, never written. **0 is
    valid and is the ordinary case**: rebuild the whole pyramid from full resolution.

    ``kind`` picks the reducer: ``"image"`` is a mask-weighted mean (anti-aliased),
    ``"segmentation"`` a label-preserving mode. It defaults to whatever the volume
    already records (precomputed ``info["type"]``, OME multiscales ``type``), since
    averaging label ids invents ids. If the volume records nothing usable, pass it.

    Metadata (shape, voxel size, offset, axes) comes from the existing volume.
    ``voxel_size`` overrides it, for a volume whose recorded scale is itself the
    thing that was wrong.

    The progress manifest defaults to a name distinct from a conversion's, which would
    mark the levels being rebuilt as already done.
    """
    if start_level < 0:
        raise ValueError(f"start_level must be >= 0, got {start_level}")

    fmt = detect_backend(dst)
    if fmt is None:
        raise ValueError(
            f"could not detect a volume at {dst!r} (no info / zarr.json / .zarray)")
    # A format that cannot have a pyramid is the wrong argument, not a broken volume.
    require_chunked_volume(fmt, dst, "neu-vol downsample")
    spec = {"backend": fmt, "path": dst}
    meta = read_source_metadata(spec)
    if not meta and voxel_size is None:
        raise ValueError(
            f"{dst!r} has no readable coordinate metadata; pass voxel_size=")

    # Level 0 of the existing volume defines the pyramid the schedule must match.
    level0_spec = meta["data_spec"] if meta else spec
    level0 = open_backend(level0_spec)
    axes = tuple(meta["spatial_axes"]) if meta else ("z", "y", "x")
    has_channels = meta["has_channels"] if meta else len(level0.shape) == len(axes) + 1
    voxel_size = tuple(voxel_size if voxel_size is not None else meta["voxel_size"])
    offset = tuple(meta["offset"]) if (meta and meta.get("offset")) else (0.0,) * len(axes)

    if kind is None:
        kind = (meta or {}).get("kind")
        if kind is None:
            raise ValueError(
                f"{dst!r} does not record whether it is image or segmentation data, "
                "so the downsampling reducer cannot be inferred; pass "
                "kind='image' (mask-weighted mean) or kind='segmentation' "
                "(label-preserving mode). Averaging label ids would invent ids "
                "that are not in the data.")
        logger.info("kind=%r taken from the volume's metadata", kind)
    if kind not in ("image", "segmentation"):
        raise ValueError(f"kind must be 'image' or 'segmentation', got {kind!r}")

    if max_levels is None:
        from ..source_metadata import read_level_voxel_sizes

        max_levels, why = resolve_max_levels(read_level_voxel_sizes(spec))
        logger.info("max_levels=%d (%s)", max_levels, why)

    # Keep the existing chunking by default: silently rechunking only the
    # regenerated levels would leave the pyramid internally inconsistent.
    if chunk is None:
        try:
            existing = tuple(int(c) for c in level0.chunks)
            chunk = existing[1:] if has_channels else existing
        except Exception:                       # backend without chunk info
            chunk = None

    if progress_path is None:
        progress_path = default_progress_path(dst).replace(
            ".progress.jsonl", f".regen-from-{start_level}.progress.jsonl")

    logger.info("rebuild %s from level %d | level0 shape=%s voxel_size=%s kind=%s",
                dst, start_level, tuple(level0.shape), voxel_size, kind)

    return materialize_multiscale(
        # Never read (no level-0 copy in a rebuild); the signature is shared with convert.
        src_spec=level0_spec,
        src_shape=level0.shape,
        src_dtype=str(level0.dtype),
        dst=dst,
        profile=profile or _profile_for(fmt),
        voxel_size=voxel_size,
        offset=offset,
        units=(meta.get("units") if meta else None) or "nm",
        spatial_axes=axes,
        has_channels=has_channels,
        num_channels=int(level0.shape[0]) if has_channels else 1,
        dtype=None,
        kind=kind,
        encoding=encoding,
        multiscale=True,
        factors=factors,
        max_levels=max_levels,
        min_dim=min_dim,
        name=name,
        chunk=chunk,
        shard=shard,
        client=client,
        npartitions=npartitions,
        delete_existing=False,         # never destroy a volume we are repairing
        resume=resume,
        verify=verify,
        sparse=sparse,
        progress_path=progress_path,
        validate=validate,
        # An int, never None, which would mean "convert" and try to create level 0.
        seed_level=start_level,
    )
