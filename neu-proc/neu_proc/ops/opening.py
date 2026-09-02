"""Spherical morphological opening: :func:`opening`.

An erosion followed by a dilation, which is the standard way to remove a thin protrusion or
a one-voxel bridge without shrinking what is left. ``fastmorph.spherical_open`` does it in
one pass and is label-aware, so bodies do not merge across the erosion the way a binary
implementation would.

**It is NOT anti-extensive, and a mathematical opening is.** ``γ(X) ⊆ X`` is the property
the name implies — an opening can only take voxels away — and fastmorph's does not have it,
because its erosion and its dilation read ``radius`` differently. Measured on a 10^3 box:

===========  =========  ==========  ==========
``radius``   erode      dilate      open
===========  =========  ==========  ==========
1.0          1000 (—)   1600        **1600**
2.0          512        2328        **1384**
===========  =========  ==========  ==========

So at ``radius=1`` the erosion removes nothing while the dilation still grows the shape by a
6-neighbourhood: ``opening(arr, radius=1)`` is a **pure dilation**, the opposite of what the
word promises. At ``radius=2`` a one-voxel shell does come off — a thin spike is genuinely
removed, which is the point — but the survivor comes back 38% larger than it went in.

Neither number is a bug here to fix; they are what the library does, and both are useful as
long as nobody is counting on the subset property. What would be a bug is a caller assuming
``opening`` can only remove: a size measured after opening is not comparable to one measured
before, and a shape that grew into its neighbour's background changes what touches what.
:func:`opening` therefore reports the radius at which the erosion is inert rather than
letting it pass as a no-op.

``radius`` is in the units of ``anisotropy`` — measured, the same convention
``dilate``'s ``max_distance`` follows for ``sampling`` (:mod:`neu_proc.ops.dilate`): on a
32^3 array with one 4^3 seed, ``radius=2`` leaves 8 voxels while
``radius=2, anisotropy=(4, 1, 1)`` leaves 16, and ``radius=8, anisotropy=(4, 1, 1)`` — the
same 2 voxels along z — erodes it away entirely. Without ``anisotropy`` the radius is
counted in voxels, which on an anisotropic volume is lopsided in tissue while looking
symmetric in the array.
"""

from __future__ import annotations

import warnings

from fastmorph import spherical_open

from .backend import host_only, like

#: Below this, ``fastmorph``'s erosion removes nothing (measured; see the module docstring),
#: so the call is a dilation wearing the wrong name. Warned about rather than refused —
#: growing labels by one voxel is a legitimate thing to want, just not under this name.
_INERT_EROSION_BELOW = 2.0


def opening(arr, radius: float, *, anisotropy=None, **kwargs):
    """Erode then dilate the labels in ``arr``, removing thin structure.

        opening(labels, 2)                              # a one-voxel shell off and back
        opening(labels, 80, anisotropy=(40, 8, 8))      # 80 nm, on an anisotropic volume

    ``radius`` is in the units of ``anisotropy`` when that is given and in voxels when it is
    not — see the module docstring, which also records why the result is **not** guaranteed
    to be a subset of the input. It is required rather than defaulted: every useful property
    of this op depends on it, including whether the erosion happens at all.

    fastmorph has no device implementation, so a device array is copied off and back with a
    warning; the answer comes back where the input lived, the same contract
    :func:`~neu_proc.ops.dilate.dilate` follows.
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
    opened = spherical_open(host_only(arr, "fastmorph.spherical_open"), radius=radius,
                            anisotropy=anisotropy, **kwargs)
    return like(opened, arr)
