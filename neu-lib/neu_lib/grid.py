"""Aligning a box to a block grid, per axis, and the :class:`BBox` type.

Which grid matters depends on the caller: a store's chunks or shards (a partial write is a
read-modify-write, and two concurrent partial writes into one chunk lose one of them), a
pyramid's cumulative downsampling factor, or the per-axis LCM of two chunkings. This module
only does the integer arithmetic, independently per axis, because real grids are
anisotropic: ``(1, 2, 2)`` pyramids and ``(128, 2048, 2048)`` chunks are both common.

**Boxes are half-open**, ``[lo, hi)``: a ``hi`` already on a block boundary stays there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

#: How each end of the box moves. Rounding is per axis and independent.
MODES = ("outer", "inner", "nearest", "origin")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _nearest(value: int, block: int) -> int:
    """``value`` rounded to the closest multiple of ``block``, halves upward (not to
    even, as :func:`round` would)."""
    return math.floor(value / block + 0.5) * block


def align_box(lo: Sequence[int], hi: Sequence[int], block: Sequence[int],
              mode: str = "outer") -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Align the half-open box ``[lo, hi)`` to the ``block`` grid, per axis.

    - ``outer`` floors ``lo`` and ceils ``hi``, so no requested voxel is lost. It never
      fails.
    - ``inner`` ceils ``lo`` and floors ``hi``, so the result lies inside the request.
    - ``nearest`` rounds both ends.
    - ``origin`` rounds ``lo`` and keeps the extent, for a fixed-size crop; ``hi`` is then
      generally off the grid.

    Raises ``ValueError`` if an axis would be left empty (``inner`` on a box spanning no
    whole block, ``nearest`` on one whose ends round together).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {', '.join(MODES)}")
    if not (len(lo) == len(hi) == len(block)):
        raise ValueError(f"rank mismatch: lo={tuple(lo)} hi={tuple(hi)} "
                         f"block={tuple(block)}")
    if any(b <= 0 for b in block):
        raise ValueError(f"block must be positive on every axis, got {tuple(block)}")

    out_lo, out_hi = [], []
    for axis, (a, b, k) in enumerate(zip(lo, hi, block)):
        a, b, k = int(a), int(b), int(k)
        if mode == "outer":
            na, nb = (a // k) * k, _ceil_div(b, k) * k
        elif mode == "inner":
            na, nb = _ceil_div(a, k) * k, (b // k) * k
        elif mode == "nearest":
            na, nb = _nearest(a, k), _nearest(b, k)
        else:                                            # origin: keep the extent
            na = _nearest(a, k)
            nb = na + (b - a)
        if nb <= na:
            raise ValueError(
                f"aligning [{a}, {b}) to a {k}-voxel grid with mode {mode!r} leaves "
                f"axis {axis} empty ({na}:{nb}): the box spans no whole block there. "
                f"Use mode 'outer', which only ever grows a box.")
        out_lo.append(na)
        out_hi.append(nb)
    return tuple(out_lo), tuple(out_hi)


def clamp_box(lo: Sequence[int], hi: Sequence[int],
              extent: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Trim the box to ``[0, extent)`` per axis. An edge clamped to the end of the volume
    may fall off the grid, which is fine (see :func:`misaligned_axes`)."""
    return (tuple(max(0, int(a)) for a in lo),
            tuple(min(int(b), int(e)) for b, e in zip(hi, extent)))


def misaligned_axes(lo: Sequence[int], hi: Sequence[int], block: Sequence[int],
                    extent: Sequence[int] | None = None) -> list[int]:
    """Axes where an edge of ``[lo, hi)`` falls inside a block.

    With ``extent`` given, an edge at the end of the volume counts as aligned: the final
    block is partial anyway, so there is no neighbouring data to read-modify-write.
    """
    bad = []
    for axis, (a, b, k) in enumerate(zip(lo, hi, block)):
        at_end = extent is not None and int(b) == int(extent[axis])
        if int(a) % int(k) or (int(b) % int(k) and not at_end):
            bad.append(axis)
    return bad


def lcm_grid(*grids: Sequence[int]) -> tuple[int, ...]:
    """Per-axis least common multiple of several block shapes: a box on this grid is on
    all of them, e.g. ``lcm_grid((128, 128, 128), (1, 2, 2)) == (128, 128, 128)``."""
    if not grids:
        raise ValueError("lcm_grid needs at least one grid")
    if len({len(g) for g in grids}) != 1:
        raise ValueError(f"rank mismatch between grids: {[tuple(g) for g in grids]}")
    return tuple(math.lcm(*(int(g[a]) for g in grids)) for a in range(len(grids[0])))


@dataclass(frozen=True, order=False)
class BBox:
    """A half-open box ``[lo, hi)`` with integer bounds, of any rank.

    Frozen and hashable; every operation returns a new box. Axis order is the caller's
    (the suite uses zyx).
    """

    lo: tuple[int, ...]
    hi: tuple[int, ...]

    def __post_init__(self) -> None:
        lo = tuple(int(v) for v in self.lo)
        hi = tuple(int(v) for v in self.hi)
        if len(lo) != len(hi):
            raise ValueError(f"rank mismatch: lo={lo} hi={hi}")
        if not lo:
            raise ValueError("a box needs at least one axis")
        if any(b < a for a, b in zip(lo, hi)):
            raise ValueError(f"hi must not be below lo on any axis: lo={lo} hi={hi}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_points(cls, points: Any) -> "BBox":
        """The smallest box containing every point of an ``(N, D)`` array. ``hi`` is
        ``max + 1``, so the box contains the extreme points. Raises on an empty set; use
        :meth:`empty` for an empty box."""
        import numpy as np

        arr = np.asarray(points)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(
                f"expected a non-empty (N, D) array of points, got shape {arr.shape}")
        lo = np.floor(arr.min(axis=0)).astype(np.int64)
        hi = np.floor(arr.max(axis=0)).astype(np.int64) + 1
        return cls(tuple(lo.tolist()), tuple(hi.tolist()))

    @classmethod
    def empty(cls, ndim: int = 3) -> "BBox":
        """A box with zero extent on every axis — the identity for :meth:`union`."""
        return cls((0,) * ndim, (0,) * ndim)

    # -- shape -----------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.lo)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.lo, self.hi))

    @property
    def size(self) -> int:
        """Voxel count. Zero exactly when the box is empty."""
        return math.prod(self.shape)

    # -- as arrays, for arithmetic ---------------------------------------------

    @property
    def lo_array(self) -> Any:
        """``lo`` as an int64 array."""
        import numpy as np

        return np.asarray(self.lo, dtype=np.int64)

    @property
    def hi_array(self) -> Any:
        """``hi`` as an int64 array."""
        import numpy as np

        return np.asarray(self.hi, dtype=np.int64)

    @property
    def center(self) -> Any:
        """The midpoint, as a float array. Not rounded: a box of odd extent has no integer
        centre."""
        return (self.lo_array + self.hi_array) / 2.0

    def is_empty(self) -> bool:
        return any(b <= a for a, b in zip(self.lo, self.hi))

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.lo, self.hi))

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """``lo, hi = box`` — the two-sequence form the functions above take."""
        return iter((self.lo, self.hi))

    def __repr__(self) -> str:
        return f"BBox(lo={self.lo}, hi={self.hi}, shape={self.shape})"

    # -- combination -----------------------------------------------------------

    def _check(self, other: "BBox") -> None:
        if not isinstance(other, BBox):
            raise TypeError(f"expected a BBox, got {type(other).__name__}")
        if other.ndim != self.ndim:
            raise ValueError(f"rank mismatch: {self.ndim} vs {other.ndim}")

    def union(self, other: "BBox") -> "BBox":
        """The smallest box containing both. An empty operand is ignored, so
        ``reduce(BBox.union, boxes, BBox.empty())`` works."""
        self._check(other)
        if other.is_empty():
            return self
        if self.is_empty():
            return other
        return BBox(tuple(min(a, b) for a, b in zip(self.lo, other.lo)),
                    tuple(max(a, b) for a, b in zip(self.hi, other.hi)))

    def intersect(self, other: "BBox") -> "BBox":
        """The overlap, or an empty box at the origin when there is none."""
        self._check(other)
        lo = tuple(max(a, b) for a, b in zip(self.lo, other.lo))
        hi = tuple(min(a, b) for a, b in zip(self.hi, other.hi))
        if any(b <= a for a, b in zip(lo, hi)):
            return BBox.empty(self.ndim)
        return BBox(lo, hi)

    def translate(self, offset: Sequence[float]) -> "BBox":
        """Move the box. A fractional offset is floored, like the grid arithmetic in this
        module, rather than truncated toward zero: a shift of -0.5 moves the box by -1, not
        by nothing."""
        if len(offset) != self.ndim:
            raise ValueError(f"rank mismatch: offset={tuple(offset)} box ndim={self.ndim}")
        shift = [math.floor(d) for d in offset]
        return BBox(tuple(a + d for a, d in zip(self.lo, shift)),
                    tuple(b + d for b, d in zip(self.hi, shift)))

    def contains(self, points: Any) -> Any:
        """Whether each point falls in ``[lo, hi)``.

        A single point gives a ``bool``; an ``(N, D)`` array gives a ``(N,)`` bool array.
        """
        import numpy as np

        arr = np.asarray(points)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.ndim:
            raise ValueError(
                f"expected points of rank {self.ndim}, got shape {np.shape(points)}")
        inside = np.all((arr >= np.asarray(self.lo)) & (arr < np.asarray(self.hi)),
                        axis=1)
        return bool(inside[0]) if single else inside

    # -- the grid arithmetic above, as methods ---------------------------------

    def aligned(self, block: Sequence[int], mode: str = "outer") -> "BBox":
        """:func:`align_box`, as a method."""
        return BBox(*align_box(self.lo, self.hi, block, mode=mode))

    def clamped(self, extent: Sequence[int]) -> "BBox":
        """:func:`clamp_box`, as a method. Raises if nothing is left, which usually means
        the box is in the wrong frame; use :meth:`intersect` when no overlap is a valid
        answer."""
        lo, hi = clamp_box(self.lo, self.hi, extent)
        if any(b <= a for a, b in zip(lo, hi)):
            raise ValueError(
                f"clamping {self!r} to extent {tuple(extent)} leaves nothing: the box "
                f"lies outside the volume on at least one axis. Use intersect() if an "
                f"empty result is meaningful here.")
        return BBox(lo, hi)

    def misaligned_axes(self, block: Sequence[int],
                        extent: Sequence[int] | None = None) -> list[int]:
        """:func:`misaligned_axes`, as a method."""
        return misaligned_axes(self.lo, self.hi, block, extent=extent)
