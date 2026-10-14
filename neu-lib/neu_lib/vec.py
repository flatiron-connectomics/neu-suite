"""A 3D coordinate that does arithmetic: :class:`Vec3`, for positions, offsets and extents.

    offset = Vec3.of(other.bbox.lo) - Vec3.of(mesh.bbox.lo)
    mesh.translate(offset)

Unlike a ``(3,)`` numpy array, a ``Vec3`` cannot be broadcast against an ``(N, 3)`` point
array by accident: building one from anything but three components raises. It works
anywhere a length-3 sequence is accepted, since it supports ``len``, indexing, iteration
and ``np.asarray``. Components are zyx; :attr:`Vec3.xyz` gives the reversed order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Vec3:
    """Three components, zyx. Frozen and hashable.

    ``+``, ``-``, ``*``, ``/`` and unary ``-`` take another :class:`Vec3`, a length-3
    sequence, or a scalar (applied to all three). Any other operand is left to its own
    type, so ``Vec3(...) * np.zeros((5, 3))`` gives the ``(5, 3)`` array numpy would.
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
        """From a :class:`Vec3`, a length-3 sequence, or a scalar applied to all three."""
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
        """The components reversed, as renderers and the precomputed format expect."""
        return (self.x, self.y, self.z)

    @property
    def array(self) -> np.ndarray:
        """A new ``(3,)`` float64 array."""
        return np.array([self.z, self.y, self.x], dtype=np.float64)

    def rounded(self) -> "Vec3":
        """Each component to the nearest integer, halves away from zero (not to even, as
        :func:`round` would)."""
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
        """Euclidean length."""
        return math.sqrt(self.z * self.z + self.y * self.y + self.x * self.x)

    def __repr__(self) -> str:
        return f"Vec3(z={self.z:g}, y={self.y:g}, x={self.x:g})"
