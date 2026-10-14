"""Spherical morphological opening, label-aware: :func:`opening`.

An erosion then a dilation, to remove thin protrusions and one-voxel bridges.
``fastmorph.spherical_open`` does it in one pass without merging bodies.

**The result is not guaranteed to be a subset of the input**, unlike a mathematical
opening, because fastmorph's erosion and dilation read ``radius`` differently. On a 10^3
box, ``radius=1`` erodes nothing and dilates to 1600 voxels, so it is a pure dilation;
``radius=2`` removes a one-voxel shell but returns 1384 voxels. Sizes measured after an
opening are therefore not comparable to sizes before it. :func:`opening` warns at a radius
too small to erode.

``radius`` is in the units of ``anisotropy`` when given, otherwise in voxels.
"""

from __future__ import annotations

import warnings

from fastmorph import spherical_open

from .backend import cpu_only, like

#: Below this radius (times the smallest ``anisotropy``), fastmorph's erosion removes
#: nothing and the call is a pure dilation, so it warns.
_INERT_EROSION_BELOW = 2.0


def opening(arr, radius: float, *, anisotropy=None, **kwargs):
    """Erode then dilate the labels in ``arr``, removing thin structure.

        opening(labels, 2)                              # a one-voxel shell off and back
        opening(labels, 80, anisotropy=(40, 8, 8))      # 80 nm, on an anisotropic volume

    ``radius`` is in the units of ``anisotropy`` when given, otherwise voxels. The result
    may be larger than the input (see the module docstring).

    Runs on the CPU; a GPU array is copied off and back with a warning.
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    if anisotropy is not None:
        anisotropy = tuple(float(a) for a in anisotropy)
        if len(anisotropy) != 3:
            raise ValueError(
                f"anisotropy is one value per axis (zyx); got {anisotropy}")
        if any(a <= 0 for a in anisotropy):
            raise ValueError(f"anisotropy must be positive on every axis; got {anisotropy}")
    if radius < _INERT_EROSION_BELOW * min(anisotropy or (1.0,)):
        warnings.warn(
            f"opening(radius={radius}) erodes nothing at this scale, so it is a pure "
            f"dilation — measured: a 10^3 box comes back 1600 voxels. Use radius >= "
            f"{_INERT_EROSION_BELOW * min(anisotropy or (1.0,)):g} to remove anything, or "
            f"`dilate` if growth is what you want",
            stacklevel=2)
    opened = spherical_open(cpu_only(arr, "fastmorph.spherical_open"), radius=radius,
                            anisotropy=anisotropy, **kwargs)
    return like(opened, arr)
