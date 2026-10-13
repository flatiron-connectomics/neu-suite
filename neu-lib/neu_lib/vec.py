"""A 3D coordinate that does arithmetic: positions, offsets and extents in one type.

## Why this exists

Everything geometric in this suite is a triple, and until now those triples were tuples —
``BBox.lo``, ``BBox.hi``, the ``offset_zyx_nm`` arguments. Tuples do not subtract, so lining
two objects up meant writing the arithmetic out by hand::

    offset = tuple(a - b for a, b in zip(other.bbox.lo, mesh.bbox.lo))   # every time

which is tedious exactly when you are doing it most: placing cells from different datasets
in one scene.

## Why not just use a numpy array

Because a bare ``ndarray`` makes a *coordinate* and a *point cloud* the same type. A ``(3,)``
and an ``(N, 3)`` broadcast against each other happily, so a rank mistake produces a plausible
answer rather than an error — which is the shape of the zyx/xyz confusion this package spends
so much effort pinning down. :class:`Vec3` is ``(3,)`` or it does not exist.

## The property that makes it free to adopt

**A ``Vec3`` is accepted anywhere a sequence is**, because it implements ``__len__``,
``__getitem__``, ``__iter__`` and ``__array__``. So ``mesh.translate(Vec3(0, 0, 500))`` and
``box.aligned(Vec3(64, 64, 64))`` work with no change to either signature, and nothing that
already takes a tuple has to learn about this type. It is additive in the same way
:class:`~neu_lib.grid.BBox` was: the functions it serves neither use it nor know about it.

## Axis order

``z``, ``y``, ``x``, matching the package's in-memory convention everywhere else — see the
module docstring for :mod:`neu_lib`. :attr:`xyz` is the one conversion, for the renderer and
for anything writing to disk; :func:`~neu_lib.frame.to_xyz` remains the equivalent for
``(N, 3)`` arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Vec3:
    """Three components in the package's zyx order. Immutable, hashable, and arithmetic.

    ``+``, ``-``, ``*``, ``/`` and unary ``-`` accept another :class:`Vec3`, any length-3
    sequence, or a scalar (broadcast to all three). Anything else defers to the other
    operand — so ``Vec3(...) * np.zeros((5, 3))`` gives the ``(5, 3)`` array numpy would,
    rather than an error from here.
    """

    z: float
    y: float
    x: float

    def __post_init__(self) -> None:
        for axis in ("z", "y", "x"):
            object.__setattr__(self, axis, float(getattr(self, axis)))

    # -- construction ----------------------------------------------------------

    @classmethod
    def of(cls, value: Any) -> "Vec3":
        """From a :class:`Vec3`, a length-3 sequence, or a scalar broadcast to all three.

        The one coercion, used by every operator, so "what counts as a vector here" is
        answered in a single place.
        """
        if isinstance(value, Vec3):
            return value
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            v = float(value)                                  # type: ignore[arg-type]
            return cls(v, v, v)
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"a Vec3 needs three components, got shape {arr.shape}. An (N, 3) array is "
                f"a set of points, not a coordinate — keep it as an array.")
        return cls(*arr.tolist())

    @classmethod
    def zero(cls) -> "Vec3":
        return cls(0.0, 0.0, 0.0)

    # -- sequence and array interop --------------------------------------------

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        return (self.z, self.y, self.x)[index]

    def __iter__(self) -> Iterator[float]:
        return iter((self.z, self.y, self.x))

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = np.array([self.z, self.y, self.x], dtype=dtype or np.float64)
        return arr

    @property
    def zyx(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Reversed, for the renderer and for anything the precomputed format stores."""
        return (self.x, self.y, self.z)

    @property
    def array(self) -> np.ndarray:
        """A fresh ``(3,)`` float64 array. Mutating it cannot affect this vector."""
        return np.array([self.z, self.y, self.x], dtype=np.float64)

    def rounded(self) -> "Vec3":
        """Each component to the nearest integer, halves away from zero.

        Not :func:`round`, which rounds halves to even, so ``0.5`` and ``1.5`` would land on
        the same value — the same reason :func:`neu_lib.grid._nearest` does its own thing.
        """
        return Vec3(*(math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5) for v in self))

    # -- arithmetic ------------------------------------------------------------

    def _pair(self, other: Any):
        try:
            return Vec3.of(other)
        except (ValueError, TypeError):
            return None

    def __add__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else Vec3(self.z + o.z, self.y + o.y,
                                                     self.x + o.x)

    def __sub__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else Vec3(self.z - o.z, self.y - o.y,
                                                     self.x - o.x)

    def __mul__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else Vec3(self.z * o.z, self.y * o.y,
                                                     self.x * o.x)

    def __truediv__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else Vec3(self.z / o.z, self.y / o.y,
                                                     self.x / o.x)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else o - self

    def __rtruediv__(self, other: Any) -> Any:
        o = self._pair(other)
        return NotImplemented if o is None else o / self

    def __neg__(self) -> "Vec3":
        return Vec3(-self.z, -self.y, -self.x)

    def norm(self) -> float:
        """Euclidean length — the distance, when this is the difference of two positions."""
        return math.sqrt(self.z * self.z + self.y * self.y + self.x * self.x)

    def __repr__(self) -> str:
        return f"Vec3(z={self.z:g}, y={self.y:g}, x={self.x:g})"
