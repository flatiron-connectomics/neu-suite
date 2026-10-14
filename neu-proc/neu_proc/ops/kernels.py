"""The one place a physical length becomes voxels.

Parameters are declared in nanometres and functions take voxels; the conversion lives only
here, for two reasons:

- A halo must be derived from the converted parameter. A Gaussian whose halo assumes
  ``truncate=3.0`` while scipy uses 4.0 is wrong in a shell around every block seam, which
  shows as faint block edges rather than as an error.
- A voxel count means nothing without a :class:`~neu_lib.Frame`: ``sigma_vox=2`` is
  different smoothing at level 0 and level 2, and along z and x on an anisotropic volume.

Nothing here imports scipy.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: scipy's default ``gaussian_filter(truncate=)``. A caller passing a different
#: ``truncate`` to the filter must pass the same one to :func:`gaussian_halo`.
GAUSSIAN_TRUNCATE = 4.0


def per_axis(value: float | Sequence[float], rank: int = 3) -> tuple[float, ...]:
    """A scalar or per-axis sequence as a length-``rank`` tuple, zyx; a scalar is repeated."""
    if np.isscalar(value):
        return (float(value),) * rank
    out = tuple(float(v) for v in value)          # type: ignore[union-attr]
    if len(out) != rank:
        raise ValueError(f"expected a scalar or {rank} values (zyx), got {tuple(value)}")
    return out


def to_voxels(length_nm: float | Sequence[float], voxel_size_nm: Sequence[float],
              *, rank: int = 3) -> tuple[float, ...]:
    """A physical length, per axis, in voxels of ``voxel_size_nm``. Not rounded: a sigma
    stays fractional, while a footprint radius is rounded by its caller."""
    want = per_axis(length_nm, rank)
    voxel = per_axis(voxel_size_nm, rank)
    if any(v <= 0 for v in voxel):
        raise ValueError(f"voxel size must be positive on every axis, got {voxel}")
    return tuple(w / v for w, v in zip(want, voxel))


def volume_to_voxels(volume_nm3: float, voxel_size_nm) -> float:
    """A volume in nm^3 as a voxel count, dividing by the product of the voxel sizes (a
    40x8x8 nm voxel is 2560 nm^3, so 1 um^3 is 390 voxels). Not rounded; the caller picks
    ``ceil`` or ``floor``."""
    voxel = per_axis(voxel_size_nm)
    if any(v <= 0 for v in voxel):
        raise ValueError(f"voxel size must be positive on every axis, got {voxel}")
    return float(volume_nm3) / (voxel[0] * voxel[1] * voxel[2])


def gaussian_halo(sigma_vox: float | Sequence[float], *,
                  truncate: float = GAUSSIAN_TRUNCATE,
                  rank: int = 3) -> tuple[int, ...]:
    """The read margin a Gaussian of this sigma needs, per axis, in whole voxels.

    ``ceil(truncate * sigma)``, scipy's kernel radius. Pass the same voxel sigma and
    ``truncate`` the filter will get.
    """
    return tuple(int(math.ceil(truncate * s))
                 for s in per_axis(sigma_vox, rank))


def ball(radius_vox: float | Sequence[float], *, rank: int = 3) -> np.ndarray:
    """A boolean footprint of this per-axis radius in voxels: an ellipsoid when the radii
    differ, which is what a physical ball becomes on an anisotropic volume."""
    radii = per_axis(radius_vox, rank)
    if any(r < 0 for r in radii):
        raise ValueError(f"radius must not be negative, got {radii}")
    half = tuple(int(math.floor(r)) for r in radii)
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    # A zero radius adds no extent on its axis (and must not divide by zero).
    total = sum((g / r) ** 2 for g, r in zip(grids, radii) if r > 0)
    return np.asarray(total <= 1.0) if np.ndim(total) else np.ones((1,) * rank, bool)


def box(size_vox: float | Sequence[float], *, rank: int = 3) -> tuple[int, ...]:
    """An odd per-axis footprint size in whole voxels, at least 1. Odd, because an even
    window shifts a centred filter's result by half a voxel."""
    out = []
    for s in per_axis(size_vox, rank):
        n = max(1, int(round(s)))
        out.append(n if n % 2 else n + 1)
    return tuple(out)
