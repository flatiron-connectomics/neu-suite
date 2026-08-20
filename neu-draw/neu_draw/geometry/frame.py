"""Voxel indices to physical nanometres, and the one place axis order flips.

Two conventions hold everywhere in this package, and both are load-bearing:

**One model space: physical nanometres.** Geometry never carries an integer scale, and
never derives a factor from ``2 ** scale``. Real pyramids are anisotropic — halving x
and y while leaving z alone is ordinary — so a scale index means nothing without the
source's own per-level voxel sizes. Resolving an index to a :class:`Frame` is the
``sources`` layer's job, from metadata it has read; by the time anything reaches
``geometry`` it is already in nm.

**zyx in memory, xyz at the boundary.** Every array here is zyx. The renderer wants
xyz, and :func:`to_xyz` is the only conversion — one call site per drawable in the
backend, rather than a flip scattered through the geometry. Getting this wrong mirrors
everything through the z=x diagonal, which looks like plausible data in the wrong place
rather than like an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def to_xyz(points_zyx: np.ndarray) -> np.ndarray:
    """Reverse the last axis: the single zyx→xyz conversion, for the renderer.

    Returns a C-contiguous ``float32`` array, which is what a GPU buffer wants and what
    a reversed view is not.
    """
    arr = np.asarray(points_zyx)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) array, got shape {arr.shape}")
    return np.ascontiguousarray(arr[:, ::-1], dtype=np.float32)


@dataclass(frozen=True)
class Frame:
    """Maps voxel indices at some level to physical nanometres. zyx.

    ``nm = index * voxel_size_nm + origin_nm``. The origin is where the level's voxel
    ``(0, 0, 0)`` sits in the shared nm space, which is how a cropped volume stays in
    its parent's frame.

    Anisotropy is the reason this is a per-axis tuple rather than a number, and the
    reason it exists at all rather than an ``int`` scale.
    """

    voxel_size_nm: tuple[float, float, float]
    origin_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        size = tuple(float(v) for v in self.voxel_size_nm)
        origin = tuple(float(v) for v in self.origin_nm)
        if len(size) != 3 or len(origin) != 3:
            raise ValueError(
                f"a frame is 3-D and zyx: voxel_size_nm={size} origin_nm={origin}")
        if any(v <= 0 for v in size):
            raise ValueError(f"voxel size must be positive on every axis, got {size}")
        object.__setattr__(self, "voxel_size_nm", size)
        object.__setattr__(self, "origin_nm", origin)

    @classmethod
    def identity(cls) -> "Frame":
        """One nm per voxel at the origin — for data already in nm."""
        return cls((1.0, 1.0, 1.0))

    @property
    def is_isotropic(self) -> bool:
        return len(set(self.voxel_size_nm)) == 1

    def to_nm(self, points_zyx: Sequence | np.ndarray) -> np.ndarray:
        """Voxel indices → nm. Accepts a single point or an ``(N, 3)`` array."""
        arr = np.asarray(points_zyx, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected zyx points, got shape {np.shape(points_zyx)}")
        out = arr * np.asarray(self.voxel_size_nm) + np.asarray(self.origin_nm)
        return out[0] if single else out

    def to_voxel(self, points_nm: Sequence | np.ndarray) -> np.ndarray:
        """nm → fractional voxel indices. Not rounded: rounding is the caller's policy."""
        arr = np.asarray(points_nm, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected zyx points, got shape {np.shape(points_nm)}")
        out = (arr - np.asarray(self.origin_nm)) / np.asarray(self.voxel_size_nm)
        return out[0] if single else out
