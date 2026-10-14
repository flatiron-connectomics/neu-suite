"""Voxel indices to physical nanometres (:class:`Frame`), and the zyx-to-xyz flip.

A level's voxel size comes from the source's metadata, never from ``2 ** level``: real
pyramids are anisotropic (halving x and y but not z is common). Arrays are zyx;
:func:`to_xyz` is the one conversion to a renderer's xyz, and getting the order wrong
mirrors data through the z=x diagonal without any error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def to_xyz(points_zyx: np.ndarray) -> np.ndarray:
    """``(N, 3)`` zyx points as xyz, in a new C-contiguous ``float32`` array (what a GPU
    buffer needs; a reversed view is not contiguous)."""
    arr = np.asarray(points_zyx)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) array, got shape {arr.shape}")
    return np.ascontiguousarray(arr[:, ::-1], dtype=np.float32)


@dataclass(frozen=True)
class Frame:
    """Maps voxel indices at one level to physical nanometres, zyx, per axis:
    ``nm = index * voxel_size_nm + origin_nm``.

    ``origin_nm`` is where voxel ``(0, 0, 0)`` sits, which is how a crop stays in its
    parent's coordinates.
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

    def voxel_box(self, bounds_nm, *, outward: bool = True) -> "BBox":
        """A physical ``(lo, hi)`` in nm, as a box in this frame's voxels. Use it to find
        the same place at another level, or in a crop's parent.

        ``outward=True`` floors ``lo`` and ceils ``hi``, so the result contains the
        physical box even when this frame is coarser (a 200-voxel box at 8 nm covers 25 or
        26 voxels at 64 nm, depending on where it starts). ``outward=False`` rounds each end
        to the nearest voxel instead.
        """
        import math

        from .grid import BBox

        lo_nm, hi_nm = bounds_nm
        if len(lo_nm) != len(self.voxel_size_nm) or len(hi_nm) != len(lo_nm):
            raise ValueError(
                f"bounds must be a ({len(self.voxel_size_nm)},) pair in nm, got "
                f"{tuple(lo_nm)} / {tuple(hi_nm)}")
        lo, hi = [], []
        for a, (start, stop) in enumerate(zip(lo_nm, hi_nm)):
            v, origin = self.voxel_size_nm[a], self.origin_nm[a]
            first, last = (start - origin) / v, (stop - origin) / v
            if outward:
                lo.append(math.floor(first + 1e-9))
                hi.append(math.ceil(last - 1e-9))
            else:
                lo.append(int(round(first)))
                hi.append(int(round(last)))
        return BBox(lo=tuple(lo), hi=tuple(hi))

    def to_voxel(self, points_nm: Sequence | np.ndarray) -> np.ndarray:
        """nm → fractional voxel indices, unrounded."""
        arr = np.asarray(points_nm, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected zyx points, got shape {np.shape(points_nm)}")
        out = (arr - np.asarray(self.origin_nm)) / np.asarray(self.voxel_size_nm)
        return out[0] if single else out
