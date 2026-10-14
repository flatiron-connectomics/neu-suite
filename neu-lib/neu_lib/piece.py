"""An array and the frame that says where it is: :class:`Piece`.

    piece.shape          # (364, 244, 244)
    piece.bbox           # BBox in the frame's voxels: where it sits in its parent
    piece.bounds_nm      # the same box in nm, as floats
    piece.crop(box)      # a sub-piece, its origin shifted to match
    piece.kind           # "segmentation": what the voxels mean, if the source said
    piece.name           # "gt/vol_03700": what it is called, if the source said
    piece.to_numpy()     # the voxels in CPU memory

Keeping the array and its origin in one object is the point: carried separately, the
origin is easy to drop, and the data then lands at nm zero with no error.

**Coordinates are the frame's voxels**, not array indices: a piece read from voxel 7819 of
a level has ``bbox.lo == 7819``, the number needed to place it back or compare it with a
sibling. ``bbox`` is an integer :class:`~neu_lib.grid.BBox`; ``bounds_nm`` is floats and
deliberately not a ``BBox``, since rounding a physical box to integers would move it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np

from .frame import Frame
from .grid import BBox


def _is_gpu_array(value: Any) -> bool:
    """True if ``value`` is a cupy array. Checked by module name, so cupy is never imported."""
    return type(value).__module__.split(".")[0] == "cupy"


def _to_numpy(value: Any) -> np.ndarray:
    """``value`` as a CPU numpy array. A cupy array is copied off the GPU with its own
    ``.get()``, since ``np.asarray`` refuses one; anything else (zarr, h5py, dask) converts
    through ``__array__``, which for a lazy array means reading it."""
    if _is_gpu_array(value):
        getter = getattr(value, "get", None)
        if not callable(getter):
            raise TypeError(
                f"{type(value).__name__} looks like a GPU array but has no .get(); "
                f"convert it to numpy yourself before building a Piece")
        return np.asarray(getter())
    return np.asanyarray(value)


def _as_array(value: Any) -> Any:
    """``value`` unchanged if it has ``shape``, ``dtype`` and ``ndim`` (numpy, cupy, dask,
    zarr, h5py); otherwise, a list or scalar, coerced with numpy."""
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "ndim")):
        return value
    return np.asanyarray(value)

#: What the voxels mean. It decides whether coarsening may average (image, probability) or
#: must take a mode (segmentation): averaging label ids invents ids that were never in the
#: data. Precomputed records it as ``info["type"]``, OME-NGFF as the multiscales ``type``.
KINDS = ("image", "probability", "segmentation")

#: "Not given", for arguments where ``None`` is itself a meaningful value.
_UNSET = object()


@dataclass(frozen=True)
class Piece:
    """A 3-D zyx array, optionally with a leading channel axis, and its :class:`Frame`.

    ``frame.origin_nm`` is where the array's ``(0, 0, 0)`` sits in nm, so a piece cropped
    out of a volume stays in that volume's coordinates.

    ``kind`` (one of :data:`KINDS`, or ``None`` if unknown) is never guessed from the dtype:
    a uint8 label array looks exactly like an image. ``name`` is a label for display, not
    an identity; :func:`neu_vol.read_piece` fills both in from the source.

    The array may be any array type with ``shape``, ``dtype`` and ``ndim``, including cupy,
    dask, zarr and h5py. The piece is frozen but the **array is not copied or made
    read-only**: ``piece.array[...] = 0`` mutates it, and whatever it is a view of. Use
    :meth:`copy` before editing in place.
    """

    array: np.ndarray
    frame: Frame
    kind: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        arr = _as_array(self.array)
        if arr.ndim not in (3, 4):
            raise ValueError(
                f"a piece is 3-D zyx, optionally with a leading channel axis, so 3-D or "
                f"4-D — got {arr.ndim}-D {arr.shape}")
        object.__setattr__(self, "array", arr)
        if self.kind is not None and self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS} or None, got {self.kind!r}")

    @property
    def is_segmentation(self) -> bool:
        """Whether coarsening this must take a mode rather than a mean. False when ``kind``
        is unknown."""
        return self.kind == "segmentation"

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
        """Whether there is a leading channel axis (the array is 4-D)."""
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
        """Where the array starts, in the frame's voxels (what HDF5 pieces record as
        ``voxel_offset``). Raises if the origin is not on the voxel grid, rather than
        rounding and shifting the piece by up to half a voxel."""
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
        """Where the piece sits, as a half-open box in the frame's voxels (not array
        indices)."""
        lo = self.origin_voxel
        return BBox(lo=lo, hi=tuple(a + s for a, s in zip(lo, self.spatial_shape)))

    @property
    def bounds_nm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``(lo, hi)`` in nm, as floats. Not a :class:`BBox`, whose bounds are integers."""
        lo = tuple(float(o) for o in self.origin_nm)
        hi = tuple(o + s * v for o, s, v in
                   zip(lo, self.spatial_shape, self.voxel_size_nm))
        return lo, hi                                              # type: ignore[return-value]

    def to_nm(self, points_zyx: Sequence | np.ndarray) -> np.ndarray:
        """Voxel coordinates of this piece's array, zyx, to nm."""
        return self.frame.to_nm(points_zyx)

    # -- deriving another one --------------------------------------------------
    def crop(self, box: Any) -> "Piece":
        """The sub-piece inside ``box`` (a :class:`BBox` or ``(lo, hi)``), given in the
        frame's voxels like :attr:`bbox`, not in array indices. The result's origin is
        shifted to match, so it stays in the parent's coordinates."""
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
        return replace(self, array=self.array[local],
                       frame=replace(self.frame, origin_nm=origin))

    def with_frame(self, frame: Frame) -> "Piece":
        """The same array in a different frame — for a source that recorded none."""
        return replace(self, frame=frame)

    def with_kind(self, kind: str | None) -> "Piece":
        """The same array and frame, saying what the voxels mean."""
        return replace(self, kind=kind)

    def to_numpy(self) -> np.ndarray:
        """The voxels as a CPU numpy array; for a cupy array, a copy off the GPU.

            plt.imshow(piece.to_numpy()[32])

        :meth:`to_cpu` does the same but keeps the piece. There is deliberately no
        ``to_gpu``: it would need cupy, and this package depends on numpy alone. Use
        ``neu_proc.ops.backend.to_gpu``.
        """
        return _to_numpy(self.array)

    def to_cpu(self) -> "Piece":
        """The same piece with its array in CPU memory; unchanged if it already is."""
        if isinstance(self.array, np.ndarray) and not _is_gpu_array(self.array):
            return self
        return replace(self, array=_to_numpy(self.array))

    def with_name(self, name: str | None) -> "Piece":
        """The same piece under a different name."""
        return replace(self, name=name)

    def apply(self, fn: Any, *args, frame: Frame | None = None,
              kind: Any = _UNSET, name: Any = _UNSET, **kwargs) -> "Piece":
        """``fn(array, *args, **kwargs)`` as a new piece, keeping everything else.

            piece.apply(lambda a: a > 0.5, kind="segmentation")
            piece.apply(scipy.ndimage.gaussian_filter, sigma=2)
            piece.apply(dilate).apply(label, kind="segmentation")

        The frame, kind and name carry over, with two guards:

        - **A change of spatial shape needs ``frame=``.** A 2x downsample halves the shape
          and doubles the voxel size; keeping the old frame would place the result at half
          its real size. No factor is guessed, since real pyramids are anisotropic.
        - **A change of dtype needs ``kind=``**: a member of :data:`KINDS`, ``"same"``, or
          ``None`` for unknown. A dtype change is the visible sign that the meaning may have
          changed (a probability map thresholded to a mask, a mask labelled), and carrying
          ``"probability"`` through it would later allow averaging label ids.

        ``name`` carries over unless given; ``name=None`` clears it.
        """
        out = _as_array(fn(self.array, *args, **kwargs))
        if kind is _UNSET:
            if out.dtype != self.dtype:
                raise ValueError(
                    f"{getattr(fn, '__name__', fn)} changed the dtype from {self.dtype} to "
                    f"{out.dtype}, which is the signal that it may have changed what the "
                    f"voxels MEAN — a thresholded probability map is a segmentation, and "
                    f"one still labelled {self.kind!r} would later authorise averaging "
                    f"label ids. Pass kind=: one of {', '.join(KINDS)}, or \"same\" if the "
                    f"meaning is unchanged, or None if it is no longer known.")
            kind = self.kind
        elif kind == "same":
            kind = self.kind
        if frame is None:
            was = self.spatial_shape
            now = out.shape[1:] if out.ndim == 4 else out.shape
            if tuple(now) != tuple(was):
                raise ValueError(
                    f"{getattr(fn, '__name__', fn)} changed the spatial shape from {was} to "
                    f"{tuple(now)}, so the frame no longer describes it — a 2x downsample "
                    f"halves the shape AND doubles the voxel size, and carrying the old "
                    f"frame through would place the result at half its real size with "
                    f"nothing to catch it. Pass frame= with the new voxel size and origin.")
        return replace(self, array=out, frame=frame or self.frame, kind=kind,
                       name=self.name if name is _UNSET else name)

    def copy(self, **changes) -> "Piece":
        """A copy, with any field replaced — and **the array copied too**.

            out = piece.copy(array=gaussian(piece.array))   # a transform, same frame
            out = piece.copy(kind="probability")            # relabel what it is
            out = piece.copy(); out.array[mask] = 0         # then edit in place, safely

        Unlike ``dataclasses.replace``, which pairs the *same* array with new metadata, so
        that editing the result edits the original. Passing ``array=`` skips the copy. The
        frame is shared, which is safe because it is frozen. Fields are validated as usual.
        """
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            hint = ""
            if unknown & {"voxel_size", "voxel_size_nm", "origin", "origin_nm"}:
                # The likely typo: these live on the Frame, not on the Piece.
                hint = (" Voxel size and origin belong to the frame: "
                        "copy(frame=replace(piece.frame, voxel_size_nm=...)).")
            raise TypeError(
                f"Piece has no field(s) {', '.join(sorted(unknown))}; it has "
                f"{', '.join(f.name for f in fields(self))}.{hint}")
        if "array" not in changes:
            # The array's own .copy(), so a cupy array is copied on the GPU.
            source = self.array
            changes["array"] = (source.copy() if hasattr(source, "copy")
                                else np.array(source, copy=True, subok=True))
        return replace(self, **changes)

    def __repr__(self) -> str:
        try:
            where = f" at {self.origin_voxel}"
        except ValueError:                       # an origin off the voxel grid
            where = f" at {self.origin_nm} nm"
        return (f"Piece({self.name + ', ' if self.name else ''}{self.spatial_shape}"
                + (f", {self.channels}ch" if self.channel_axis else "")
                + f", {self.dtype}"
                + (f", {self.kind}" if self.kind else "")
                + f", {self.voxel_size_nm} nm{where})")
