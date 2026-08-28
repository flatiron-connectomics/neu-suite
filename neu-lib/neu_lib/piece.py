"""An array and the frame that says where it is: :class:`Piece`.

The type that was missing. :class:`~neu_lib.scale.ScaleInfo` pairs a *shape* with a
:class:`~neu_lib.frame.Frame` and answers "what does a voxel at this level mean"; this
pairs the *data* with one and answers "where is this box, and what is in it". Everywhere
that read a crop used to carry the two alongside each other — an array here, a voxel size
and an origin there — and every such pair is a chance to drop the origin, which is
invariant 1's silent failure: the data lands at nm zero instead of on top of what it came
from, and nothing raises.

    piece.shape          # (364, 244, 244)
    piece.bbox           # BBox in the frame's voxels — where it sits in its parent
    piece.bounds_nm      # the same box in nm, as floats
    piece.crop(box)      # a sub-piece, its origin shifted to match

**One coordinate space, and it is the frame's voxels.** ``bbox`` and ``crop`` both speak
it, not array indices — a piece read from voxel 7819 of a level reports ``lo = 7819``,
because that is the number a caller needs to place it back, compare it with a sibling, or
hand to a crop argument. Array indices are always ``0..shape`` and answer nothing.

**``bbox`` is integers and ``bounds_nm`` is floats, deliberately named apart.**
:class:`~neu_lib.grid.BBox` has integer bounds — it is a grid box, and every grid
operation on it assumes that — so a box in nanometres cannot be one without truncating,
which for a 0.5 nm voxel or a fractional origin would silently move the box. The two names
differ so nobody expects box arithmetic on the physical one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .frame import Frame
from .grid import BBox


@dataclass(frozen=True)
class Piece:
    """A 3-D array (optionally with a leading channel axis) and its :class:`Frame`.

    ``frame.origin_nm`` is where the array's ``(0, 0, 0)`` sits in the shared nm space, so
    a piece cropped out of a volume stays in that volume's frame. That is the whole point
    of the type: the array and the origin cannot get separated.

    Frozen like the rest of neu-lib's types, so a piece can be passed around without
    anyone worrying about who owns it — but note the **array itself is not copied and not
    read-only**, so ``piece.array[...] = 0`` still works and still mutates whatever the
    array was a view of. Frozen means the pairing is fixed, not that the voxels are.
    """

    array: np.ndarray
    frame: Frame

    def __post_init__(self) -> None:
        arr = np.asanyarray(self.array)
        if arr.ndim not in (3, 4):
            raise ValueError(
                f"a piece is 3-D zyx, optionally with a leading channel axis, so 3-D or "
                f"4-D — got {arr.ndim}-D {arr.shape}")
        object.__setattr__(self, "array", arr)

    # -- what it is ------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """The array's full shape, channel axis included."""
        return tuple(int(s) for s in self.array.shape)

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """The three spatial extents, zyx, whether or not there is a channel axis."""
        shape = self.shape
        return shape[1:] if self.channel_axis else shape           # type: ignore[return-value]

    @property
    def channel_axis(self) -> bool:
        """Whether the leading axis is channels — decided by rank, since a piece is 3-D."""
        return self.array.ndim == 4

    @property
    def channels(self) -> int:
        return int(self.array.shape[0]) if self.channel_axis else 1

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.array.dtype)

    @property
    def voxel_size_nm(self) -> tuple[float, float, float]:
        return self.frame.voxel_size_nm

    @property
    def origin_nm(self) -> tuple[float, float, float]:
        return self.frame.origin_nm

    # -- where it is -----------------------------------------------------------
    @property
    def origin_voxel(self) -> tuple[int, int, int]:
        """Where the array starts, in the frame's voxels. Non-integral is an error.

        This is the number an HDF5 piece records as ``voxel_offset`` and the one to hand
        to a crop argument. It **raises** rather than rounding when the origin is not on
        the voxel grid, because rounding would shift the piece by up to half a voxel
        against the thing it is meant to line up with — drift that nothing downstream can
        detect.
        """
        out = []
        for axis, (o, v) in enumerate(zip(self.origin_nm, self.voxel_size_nm)):
            exact = o / v
            if abs(exact - round(exact)) > 1e-6:
                raise ValueError(
                    f"origin {self.origin_nm} nm is not a whole number of "
                    f"{self.voxel_size_nm} nm voxels ({exact:g} on axis {axis}), so this "
                    f"piece has no integer voxel origin. `bounds_nm` is always available")
            out.append(int(round(exact)))
        return tuple(out)                                          # type: ignore[return-value]

    @property
    def bbox(self) -> BBox:
        """Where the piece sits, as a half-open box in the **frame's voxels**.

        Not array indices: a piece read from voxel 7819 of a level reports ``lo = 7819``.
        That is what makes two pieces of one volume comparable and what a crop argument
        wants.
        """
        lo = self.origin_voxel
        return BBox(lo=lo, hi=tuple(a + s for a, s in zip(lo, self.spatial_shape)))

    @property
    def bounds_nm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``(lo, hi)`` in nanometres, as floats. Deliberately not a :class:`BBox`.

        ``BBox`` has integer bounds and its operations assume a grid, so a physical box
        cannot be one without truncating — which for a fractional voxel size or origin
        would move the box silently.
        """
        lo = tuple(float(o) for o in self.origin_nm)
        hi = tuple(o + s * v for o, s, v in
                   zip(lo, self.spatial_shape, self.voxel_size_nm))
        return lo, hi                                              # type: ignore[return-value]

    def to_nm(self, points_zyx: Sequence | np.ndarray) -> np.ndarray:
        """Voxel coordinates **of this piece's array** to nm. Convenience over the frame."""
        return self.frame.to_nm(points_zyx)

    # -- deriving another one --------------------------------------------------
    def crop(self, box: Any) -> "Piece":
        """A sub-piece, in the **frame's voxels** — the same space :attr:`bbox` reports.

        Absolute, not array-relative, and that is the point: the two spaces differ by the
        origin, and a caller who has just read ``piece.bbox`` and narrowed it would
        otherwise get a box silently offset by the origin. The result's frame carries the
        shifted origin, so the sub-piece is still in the parent's coordinates.

        ``box`` is a :class:`BBox` or any ``(lo, hi)`` pair.
        """
        want = box if isinstance(box, BBox) else BBox(lo=tuple(box[0]), hi=tuple(box[1]))
        mine = self.bbox
        if len(want.lo) != 3:
            raise ValueError(f"a crop box is 3-D zyx; got {want.lo} / {want.hi}")
        for axis in range(3):
            if not (mine.lo[axis] <= want.lo[axis] < want.hi[axis] <= mine.hi[axis]):
                raise ValueError(
                    f"crop {want.lo}:{want.hi} is not inside this piece's "
                    f"{mine.lo}:{mine.hi} on axis {axis}. The box is in the frame's "
                    f"voxels, the same space `bbox` reports — not array indices")
        local = tuple(slice(want.lo[a] - mine.lo[a], want.hi[a] - mine.lo[a])
                      for a in range(3))
        if self.channel_axis:
            local = (slice(0, self.channels),) + local
        origin = tuple(float(o) + (want.lo[a] - mine.lo[a]) * self.voxel_size_nm[a]
                       for a, o in enumerate(self.origin_nm))
        return Piece(array=self.array[local],
                     frame=replace(self.frame, origin_nm=origin))

    def with_frame(self, frame: Frame) -> "Piece":
        """The same array in a different frame — for a source that recorded none."""
        return Piece(array=self.array, frame=frame)

    def __repr__(self) -> str:
        try:
            where = f" at {self.origin_voxel}"
        except ValueError:                       # an origin off the voxel grid
            where = f" at {self.origin_nm} nm"
        return (f"Piece({self.spatial_shape}"
                + (f", {self.channels}ch" if self.channel_axis else "")
                + f", {self.dtype}, {self.voxel_size_nm} nm{where})")
