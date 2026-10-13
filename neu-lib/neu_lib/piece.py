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
    piece.kind           # "segmentation" — what the voxels mean, where the source said
    piece.name           # "gt/vol_03700" — what it is called, where the source said
    piece.to_numpy()     # the voxels in CPU memory, for something that needs numpy

``kind`` is optional and rides along because it is a fact about the **data**, not about
any renderer: it decides whether coarsening may average or must take a mode, and averaging
label ids invents ids that were never in the data. A source that records it (precomputed's
``info["type"]``, OME's multiscales ``type``) fills it in; one that does not leaves it
``None``, which is honest. It is never derived from the dtype — a uint8 label array looks
exactly like an image that way, and that is the mistake the field exists to carry past.

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

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np

from .frame import Frame
from .grid import BBox


def _is_gpu_array(value: Any) -> bool:
    """True if ``value`` lives in GPU memory, by the type's defining module.

    One check, shared by :func:`_to_numpy` and :meth:`Piece.to_cpu`, so the two cannot
    disagree about what "already in CPU memory" means — which they did: ``to_cpu`` tested
    ``isinstance(np.ndarray)``, so a GPU array that happened to subclass ndarray was
    reported as already home and never converted.

    By module rather than ``isinstance``, because importing cupy to ask whether an array is
    a numpy array would make every CPU-only install pay for a CUDA stack.
    """
    return type(value).__module__.split(".")[0] == "cupy"


def _to_numpy(value: Any) -> np.ndarray:
    """``value`` as a CPU numpy array, whatever kind of array it is.

    **numpy cannot do this on its own for a GPU array.** ``np.asarray`` on a cupy array
    raises ``TypeError: Implicit conversion to a NumPy array is not allowed`` — deliberately,
    because the alternative is copying gigabytes off a GPU because someone passed the wrong
    thing to a plotting call.

    What makes it possible here without importing cupy is that the conversion is a **method
    on the array**: ``.get()``. So this package keeps its numpy-only rule and still converts.
    The GPU is checked for first rather than by catching numpy's TypeError, because that
    path also emits a spurious ``__array__`` deprecation warning that would reach the caller
    looking like a bug in their code.

    Anything else — a zarr array, an open h5py dataset, a dask array — converts through
    ``__array__`` as usual, which for the lazy ones means reading or computing it.
    """
    if _is_gpu_array(value):
        getter = getattr(value, "get", None)
        if not callable(getter):
            raise TypeError(
                f"{type(value).__name__} looks like a GPU array but has no .get(); "
                f"convert it to numpy yourself before building a Piece")
        return np.asarray(getter())
    return np.asanyarray(value)


def _as_array(value: Any) -> Any:
    """``value`` unchanged if it is already an array, else coerced with numpy.

    **Not ``np.asanyarray`` unconditionally**, which is what this used to be: cupy makes
    ``__array__`` raise on purpose, to stop a GPU array being copied to the CPU by
    accident. So coercing every input meant a ``Piece`` could not hold a cupy array at all
    — ``TypeError: Implicit conversion to a NumPy array is not allowed`` — and the same goes
    for anything else lazy or foreign: a dask array, a zarr array, an open h5py dataset.

    Anything carrying ``shape``, ``dtype`` and ``ndim`` is taken at its word. A list or a
    scalar still goes through numpy, which is the case the coercion was for.

    This package stays numpy-only: recognising a foreign array by its attributes needs no
    import of whatever produced it.
    """
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "ndim")):
        return value
    return np.asanyarray(value)

#: What the voxels of an array MEAN, which is not a rendering choice — it decides whether
#: coarsening may average (image, probability) or must take a mode (segmentation), and
#: averaging label ids invents ids that were never in the data. Recorded by precomputed as
#: ``info["type"]`` and by OME as the multiscales ``type``; the vocabulary lives here
#: rather than in whichever package happens to read it, so a viewer and a downsampler
#: cannot end up with two lists.
KINDS = ("image", "probability", "segmentation")

#: ``Piece.apply``'s default for ``kind``, distinct from ``None`` because ``None`` is itself
#: a legal value meaning "no longer known". Not exported: the three things a caller writes
#: are a member of :data:`KINDS`, ``"same"``, or ``None``.
_UNSET = object()


@dataclass(frozen=True)
class Piece:
    """A 3-D array (optionally with a leading channel axis) and its :class:`Frame`.

    ``frame.origin_nm`` is where the array's ``(0, 0, 0)`` sits in the shared nm space, so
    a piece cropped out of a volume stays in that volume's frame. That is the whole point
    of the type: the array and the origin cannot get separated.

    ``name`` is what this piece is *called* — a label, not identity, and nothing here
    depends on it being unique or even present. It rides along for the same reason ``kind``
    does: whoever read the piece knew where it came from, and a consumer three calls later
    does not. :func:`neu_vol.read_piece` fills it in from the source.

    **The array need not be a numpy array.** Anything carrying ``shape``, ``dtype`` and
    ``ndim`` is taken as one — a cupy array on the GPU, a dask or zarr array, an open h5py
    dataset — because coercing everything meant a GPU array could not be held at all
    (cupy makes ``__array__`` raise on purpose, to stop an accidental copy to the CPU).
    Nothing here imports any of them; recognising an array by its attributes needs no
    knowledge of what made it.

    Frozen like the rest of neu-lib's types, so a piece can be passed around without
    anyone worrying about who owns it — but note the **array itself is not copied and not
    read-only**, so ``piece.array[...] = 0`` still works and still mutates whatever the
    array was a view of. Frozen means the pairing is fixed, not that the voxels are.
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
        """Whether coarsening this must take a mode rather than a mean.

        ``None`` reads as False, and that asymmetry is deliberate: a piece that does not
        know what it is should not claim to be labels, and the caller that cares is the one
        that has to establish it. What it must **not** do is guess from the dtype — a uint8
        label array looks exactly like an image that way.
        """
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
        return replace(self, array=self.array[local],
                       frame=replace(self.frame, origin_nm=origin))

    def with_frame(self, frame: Frame) -> "Piece":
        """The same array in a different frame — for a source that recorded none."""
        return replace(self, frame=frame)

    def with_kind(self, kind: str | None) -> "Piece":
        """The same array and frame, saying what the voxels mean."""
        return replace(self, kind=kind)

    def to_numpy(self) -> np.ndarray:
        """The voxels as a CPU numpy array, for handing to something that needs one.

            plt.imshow(piece.to_numpy()[32])
            neu_vol.pack_hdf5(...)          # h5py cannot take a GPU array

        A no-op when the array is already numpy. For a cupy array this **copies off the
        GPU**, which is the expensive direction — so it is a method you call rather than
        something that happens to you, and :meth:`to_cpu` is the version that keeps the
        piece intact.

        There is no ``to_gpu`` here, and that asymmetry is not an oversight: coming *to*
        numpy is a method on the array (``.get()``), so this package can do it without
        importing anything, while going *to* the GPU needs cupy itself — which would put a
        compiled CUDA dependency under this package's numpy-only promise. That lives in
        ``neu_proc.ops.backend.to_gpu``.
        """
        return _to_numpy(self.array)

    def to_cpu(self) -> "Piece":
        """The same piece with its array in CPU memory. Unchanged if it already is.

        The form to use mid-chain, since it keeps the frame, kind and name — where
        :meth:`to_numpy` hands out the bare array.
        """
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

        The point is that the frame, the name and the kind survive a chain of transforms —
        which is the reason this type exists, since an array on its own forgets where it is
        after the first operation.

        **A transform that changes the shape must supply the new ``frame``.** Shape and
        frame are two halves of one statement about where the voxels are: a 2x downsample
        halves the shape *and* doubles the voxel size, so carrying the old frame through
        would put the data at half its real size, silently and with nothing to check it
        against. Rather than guess a factor — real pyramids are anisotropic, and a crop
        changes the origin instead of the size — this refuses and asks.

        **A transform that changes the dtype must say what the result is.** Nothing here can
        detect a change of meaning — a threshold written ``(a > 0.5).astype("float32")``
        changes neither shape nor dtype — but a dtype change is the signal that is available,
        and it catches the two that matter: a probability map thresholded to a mask
        (float32 -> uint8) and a mask labelled (uint8 -> uint32). Inheriting
        ``"probability"`` through either would later authorise averaging label ids into ids
        that were never in the data. So a dtype change with no ``kind`` raises, and the three
        answers are a member of :data:`KINDS`, ``"same"`` to say the meaning is unchanged,
        or ``None`` to say it is no longer known.

        Full dtype equality, not ``dtype.kind``: a widening cast like ``uint8 -> uint32`` is
        exactly the connected-components case, and the cost of a false alarm is twelve
        characters while the cost of a miss is silent.

        The two guards are the same shape — shape change wants ``frame=``, dtype change
        wants ``kind=`` — because those are the two observable signals that a transform did
        something the metadata cannot follow on its own.

        ``name`` is inherited unless given, and ``name=None`` **clears** it rather than
        meaning "not specified" — the same sentinel treatment ``kind`` gets, so the two
        override arguments behave alike and neither has a value that quietly does nothing.

        Nothing here imports a transform library, and it must stay that way — this package
        is numpy and nothing else. ``fn`` is any callable, which is also what the planned
        neu-proc gives you: an ``Op`` bound to a frame *is* a callable, so
        ``piece.apply(op.bind(piece.frame))`` needs no hook on either side.
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

        Copying the array is the whole reason this exists rather than
        ``dataclasses.replace``, which re-pairs the *same* array with new metadata — so an
        in-place edit on what looks like a copy reaches back into the original, and into
        whatever the original was itself a view of. Passing ``array=`` skips the copy, since
        then you brought your own.

        The frame is shared rather than copied, which is safe because it is frozen and holds
        only numbers.

        Every field is validated as usual, so ``copy(kind="labels")`` raises rather than
        producing a piece nothing can interpret.
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
            # The array's OWN copy, not `np.array(..., copy=True)`: that would refuse a cupy
            # array for the same reason `_as_array` exists, and a GPU array should be
            # copied on the GPU rather than round-tripped through the CPU.
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
