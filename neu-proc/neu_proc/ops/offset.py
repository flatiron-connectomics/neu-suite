"""Shift an array by whole voxels: :func:`offset`.

A shift is not a crop — the shape is preserved, so the far side has to be filled with
something, and *what* is the only interesting decision here. Two answers, and they mean
different things about the data:

* ``extend`` repeats the edge plane outward. Nothing is invented that was not already at
  the boundary, which is the right default for a label volume: the voxels it adds are a
  claim the data already made one plane over.
* ``const`` writes a fixed value, ``fill_const`` (0 by default, i.e. background). Honest
  about knowing nothing, at the cost of a slab of background that no annotator put there.

**There is deliberately no "dilate" fill.** It reads well — pad with background, then let
the labels grow into it — and it is wrong twice over. `dilate`'s default ``edt`` method is
*unbounded* (see :mod:`neu_proc.ops.dilate`), so it would fill every interior background
voxel too, not just the new margin. And even bounded it would not agree with ``extend``:
where an edge voxel is background, edge-repetition carries that background outward while a
nearest-label fill invents a label there, and in a corner the nearest source voxel is
diagonal rather than perpendicular. Fill the margin, then dilate yourself if that is what
you want.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Accepted spellings of each fill method, and what it does. A table rather than a chain of
#: comparisons so an unknown name is refused *with the list* — the house pattern
#: (`dilate.METHODS`, `neu_vol.profiles.get_profile`).
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

    ``offset`` is a scalar applied to all three axes or one value per axis, zyx, positive
    meaning toward higher indices. What falls off one side is dropped and the other side is
    filled by ``fill_method`` (:data:`FILL_METHODS`) — the shape is invariant, which is what
    makes this composable with anything else that expects the array's own grid.

    **The frame does not come along**, and that is the caller's to deal with: shifting the
    voxels by one is the same statement as moving the origin by one voxel the other way, and
    a `Piece` that did neither would be misplaced by exactly the shift. This layer is arrays
    in and arrays out (invariant NM-SPACE lives one layer up), so use
    ``piece.apply(...)`` with a ``frame=`` when the shift is meant to be physical rather
    than an in-place resample.
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
    # Drop what shifts out, then pad the other side back to the original width. Two steps
    # rather than np.roll: rolling WRAPS, so a body leaving one face reappears on the
    # opposite one — plausible-looking output with a piece of tissue teleported across the
    # volume.
    kept = tuple(slice(-x, None) if x < 0 else slice(None, -x or None) for x in shift)
    pad = tuple((x, 0) if x > 0 else (0, -x) for x in shift)
    if method == "extend":
        return np.pad(arr[kept], pad, mode="edge")
    return np.pad(arr[kept], pad, mode="constant", constant_values=fill_const)
