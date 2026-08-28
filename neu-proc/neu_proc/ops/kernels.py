"""The one place a physical length becomes voxels.

Every ``Op`` declares its parameters in **nanometres** and every function takes **voxels**;
this module is the whole of the boundary between them, and keeping it to one module is the
point rather than a tidiness preference. Two failures it exists to prevent:

- **A halo derived from anything other than the converted parameter is wrong.** A Gaussian
  whose halo assumes ``truncate=3.0`` while scipy's default is ``4.0`` is wrong in a shell
  around every block seam — which shows up as faint block edges in the output, not as an
  error. So the halo is computed *here*, from the same voxel number the filter will be given,
  and never declared beside it.
- **A voxel count is meaningless without a frame.** ``sigma_vox=2`` is a different physical
  smoothing at level 0 than at level 2, and a different one along z than along x on an
  anisotropic volume — and real pyramids are anisotropic, ``(1, 2, 2)`` being common. So a
  voxel number cannot be allowed to exist before a :class:`~neu_lib.Frame` does, which is
  what ``Op.bind(frame)`` enforces and what this module implements.

Nothing here imports scipy: converting a length needs arithmetic and a frame, and keeping it
scipy-free means the conversion can be tested without the filters and reasoned about beside
the halo it produces.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: scipy's own default for ``gaussian_filter(truncate=)``, restated because the halo has to
#: agree with it and a mismatch is invisible. If a caller passes a different ``truncate`` to
#: the filter, it must pass the same one here.
GAUSSIAN_TRUNCATE = 4.0


def per_axis(value: float | Sequence[float], rank: int = 3) -> tuple[float, ...]:
    """A scalar or per-axis sequence as a length-``rank`` tuple, zyx.

    A scalar broadcasts, which is right for an isotropic request and wrong to *assume* — so
    the nm-native layer above should hand down a real per-axis value derived from the frame,
    and this only widens what a caller wrote.
    """
    if np.isscalar(value):
        return (float(value),) * rank
    out = tuple(float(v) for v in value)          # type: ignore[union-attr]
    if len(out) != rank:
        raise ValueError(f"expected a scalar or {rank} values (zyx), got {tuple(value)}")
    return out


def to_voxels(length_nm: float | Sequence[float], voxel_size_nm: Sequence[float],
              *, rank: int = 3) -> tuple[float, ...]:
    """A physical length, per axis, in voxels of ``voxel_size_nm``. Not rounded.

    Left as floats because the two consumers want different roundings: a Gaussian's sigma is
    genuinely fractional, while a structuring element's radius has to become whole voxels.
    Rounding here would force one of them to be wrong.
    """
    want = per_axis(length_nm, rank)
    voxel = per_axis(voxel_size_nm, rank)
    if any(v <= 0 for v in voxel):
        raise ValueError(f"voxel size must be positive on every axis, got {voxel}")
    return tuple(w / v for w, v in zip(want, voxel))


def volume_to_voxels(volume_nm3: float, voxel_size_nm) -> float:
    """A physical volume in nm^3 as a voxel count. Not rounded, and not per-axis.

    The other kind of conversion :func:`to_voxels` does not do: a **volume** divides by the
    *product* of the three voxel sizes, where a length scales per axis. Using the length
    conversion here would be wrong by two factors of the voxel size — a 40x8x8 nm voxel is
    2560 nm^3, not 40, so a 1 um^3 threshold is 390 voxels and not 25,000.

    Unrounded because the caller decides: a size threshold wants ``ceil`` or ``floor``
    depending on which way it should err, and rounding here would take that away.
    """
    voxel = per_axis(voxel_size_nm)
    if any(v <= 0 for v in voxel):
        raise ValueError(f"voxel size must be positive on every axis, got {voxel}")
    return float(volume_nm3) / (voxel[0] * voxel[1] * voxel[2])


def gaussian_halo(sigma_vox: float | Sequence[float], *,
                  truncate: float = GAUSSIAN_TRUNCATE,
                  rank: int = 3) -> tuple[int, ...]:
    """The read margin a Gaussian of this sigma needs, per axis, in whole voxels.

    ``ceil(truncate * sigma)`` — scipy's own kernel radius, which is what the filter reads
    beyond the voxel it is writing. Derived from the **converted** sigma and from the same
    ``truncate`` the filter will use, because those are the two ways this number goes
    silently wrong.
    """
    return tuple(int(math.ceil(truncate * s))
                 for s in per_axis(sigma_vox, rank))


def ball(radius_vox: float | Sequence[float], *, rank: int = 3) -> np.ndarray:
    """A boolean ellipsoid footprint of this per-axis radius, in voxels.

    An *ellipsoid*, not a ball, whenever the radii differ — which is the anisotropic case and
    the one that matters: a spherical footprint in voxel space is an ellipsoid in tissue, so
    a morphology radius given in nm must become different voxel counts per axis or the
    operation is physically lopsided.
    """
    radii = per_axis(radius_vox, rank)
    if any(r < 0 for r in radii):
        raise ValueError(f"radius must not be negative, got {radii}")
    half = tuple(int(math.floor(r)) for r in radii)
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    # A zero radius on an axis contributes no extent and must not divide by zero.
    total = sum((g / r) ** 2 for g, r in zip(grids, radii) if r > 0)
    return np.asarray(total <= 1.0) if np.ndim(total) else np.ones((1,) * rank, bool)


def box(size_vox: float | Sequence[float], *, rank: int = 3) -> tuple[int, ...]:
    """An odd per-axis footprint size in whole voxels, at least 1.

    Odd because a median or a box filter centred on a voxel needs a symmetric window; an
    even size would shift the result by half a voxel, consistently and invisibly.
    """
    out = []
    for s in per_axis(size_vox, rank):
        n = max(1, int(round(s)))
        out.append(n if n % 2 else n + 1)
    return tuple(out)
