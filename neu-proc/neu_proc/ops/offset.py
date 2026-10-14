"""Shift an array by whole voxels, keeping its shape: :func:`offset`.

The far side is filled by one of two methods:

* ``extend`` repeats the edge plane outward, adding only what the boundary already holds;
  the default, and right for a label volume.
* ``const`` writes ``fill_const`` (0, background, by default).

There is no "dilate" fill: ``edt`` dilation is unbounded and would fill interior
background too, and even bounded it would invent labels where the edge is background.
Fill the margin, then dilate explicitly if that is wanted.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Fill method -> accepted spellings, so an unknown name is refused with the list.
FILL_METHODS = {
    "extend": ("extend", "edge", "replicate"),
    "const": ("const", "constant", "empty", "background", "bg"),
}


def _resolve_fill(fill_method: str) -> str:
    wanted = str(fill_method).lower()
    for name, spellings in FILL_METHODS.items():
        if wanted in spellings:
            return name
    raise ValueError(
        f"unknown fill_method {fill_method!r}; known: "
        + ", ".join(f"{n} ({'/'.join(s)})" for n, s in FILL_METHODS.items()))


def offset(arr, offset, fill_method: str = "extend", fill_const=0) -> np.ndarray:
    """``arr`` shifted by ``offset`` whole voxels, **keeping its shape**.

        offset(labels, 1)                       # one voxel along every axis
        offset(labels, (2, 0, -3))              # per axis, either direction
        offset(labels, 4, "const", fill_const=0)

    ``offset`` is a scalar for all three axes or one value per axis, zyx, positive toward
    higher indices. What falls off one side is dropped and the other side is filled by
    ``fill_method`` (:data:`FILL_METHODS`).

    The origin is not adjusted: through ``Piece.apply`` the shifted data keeps the old frame.
    If the shift should move the data in physical space, pass a ``frame=`` with the origin
    moved accordingly.
    """
    shift = np.asarray(list(offset) if isinstance(offset, Iterable) else [offset] * 3,
                       dtype=int)
    if shift.shape != (3,):
        raise ValueError(
            f"offset is one value per axis (zyx) or a scalar for all three; got "
            f"{tuple(shift.tolist())}")
    if arr.ndim != 3:
        raise ValueError(f"offset works on a 3-D zyx array; got {arr.ndim}-D {arr.shape}")
    if np.any(np.abs(shift) >= np.asarray(arr.shape)):
        raise ValueError(
            f"offset {tuple(shift.tolist())} is at least as large as the array {arr.shape} "
            f"on some axis, which would shift every voxel out of view")

    method = _resolve_fill(fill_method)
    # Drop what shifts out, then pad the other side. Not np.roll, which wraps data around to
    # the opposite face.
    kept = tuple(slice(-x, None) if x < 0 else slice(None, -x or None) for x in shift)
    pad = tuple((x, 0) if x > 0 else (0, -x) for x in shift)
    if method == "extend":
        return np.pad(arr[kept], pad, mode="edge")
    return np.pad(arr[kept], pad, mode="constant", constant_values=fill_const)
