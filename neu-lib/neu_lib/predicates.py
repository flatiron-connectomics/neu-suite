"""Region tests: build the ``inside()`` that :meth:`Skeleton.crop` takes.

An ``InsideFn`` answers one question for an ``(N, 3)`` array of zyx nm points — is each
one in the region — and that is the whole interface. Keeping it a plain callable is what
lets a caller supply a region this package has never heard of (a mesh hull, a distance
field, a polygon) without anything here changing.

These live beside their consumer on purpose. They were in ``neu_draw.sources``, a tier
*above* the packages that want to exclude a compartment before measuring, so the builders
sat above the thing that uses them and nothing below could reach either.

**Points outside a region's own extent are outside the region, not an error.** A skeleton
normally runs past any one ROI or compartment mask, and that is precisely the question
being asked of it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .frame import Frame
from .grid import BBox
from .skeleton import InsideFn


def box_predicate(box: BBox) -> InsideFn:
    """``inside()`` for a nm bounding box. The simplest region there is."""
    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        return box.contains(np.asarray(points_zyx_nm))
    return inside


def mask_predicate(mask_zyx: np.ndarray, frame: Frame) -> InsideFn:
    """``inside()`` for a dense boolean region at some resolution.

    A dense array is the right representation *here* and would be wrong for a body mask:
    a compartment or ROI is read at a coarse level, or in a small box around something
    that localises it, so the array is small. The same volume at full resolution is a
    million times that, which is why there is no general mask type in this package.

    ``frame`` is the mask's own — its voxel size and origin — so a mask read out of a
    crop places correctly instead of being assumed to start at nm zero.
    """
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
    """Inside any of them. ``union()`` of nothing is inside nothing.

    Exists so several compartments compose without a caller rasterising them together:
    excluding soma *and* nucleus is two masks, and merging them into one array would
    require they share a resolution and a frame, which they need not.
    """
    def inside(points_zyx_nm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_zyx_nm)
        out = np.zeros(len(points), dtype=bool)
        for predicate in predicates:
            out |= np.asarray(predicate(points), dtype=bool)
        return out
    return inside
