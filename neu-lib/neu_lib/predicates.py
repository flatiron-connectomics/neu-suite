"""Region tests: builders for the ``inside()`` callable that :meth:`Skeleton.crop` takes.

An ``InsideFn`` takes an ``(N, 3)`` array of zyx nm points and returns a bool per point.
Any callable of that shape works, so a caller can supply regions this module does not
build (a mesh hull, a distance field). Points beyond a region's extent are outside it,
not an error.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .frame import Frame
from .grid import BBox
from .skeleton import InsideFn


def box_predicate(box: BBox) -> InsideFn:
    """``inside()`` for a bounding box in nm."""
    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        return box.contains(np.asarray(points_zyx_nm))
    return inside


def mask_predicate(mask_zyx: np.ndarray, frame: Frame) -> InsideFn:
    """``inside()`` for a dense boolean mask, zyx, whose voxel size and origin are given by
    ``frame``. Meant for small masks: a compartment or ROI at a coarse level, or a crop."""
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D zyx, got shape {mask.shape}")

    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        voxels = np.floor(frame.to_voxel(np.asarray(points_zyx_nm))).astype(np.int64)
        ok = np.all((voxels >= 0) & (voxels < np.asarray(mask.shape)), axis=1)
        out = np.zeros(len(voxels), dtype=bool)
        if ok.any():
            inside_voxels = voxels[ok]
            out[ok] = mask[inside_voxels[:, 0], inside_voxels[:, 1],
                           inside_voxels[:, 2]]
        return out

    return inside


def union(*predicates: InsideFn) -> InsideFn:
    """Inside any of the predicates; ``union()`` of nothing is inside nothing. The
    predicates need not share a resolution or frame."""
    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_zyx_nm)
        out = np.zeros(len(points), dtype=bool)
        for predicate in predicates:
            out |= np.asarray(predicate(points), dtype=bool)
        return out
    return inside
