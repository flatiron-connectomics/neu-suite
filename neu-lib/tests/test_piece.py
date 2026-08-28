"""An array that carries its frame: :class:`neu_lib.Piece`.

The type exists so an array and its origin cannot get separated. Every place that read a
crop used to pass the two alongside each other, and each such pair is a chance to drop the
origin — invariant 1's silent failure, where the data lands at nm zero instead of on top of
what it came from and nothing raises.

The tests are mostly about **which coordinate space** each accessor speaks, because that is
the only thing about this type that can be subtly wrong.
"""

import numpy as np
import pytest

from neu_lib import BBox, Frame, Piece


def _piece(shape=(8, 16, 16), voxel=(40.0, 8.0, 8.0), origin=(400.0, 160.0, 240.0),
           dtype="uint64"):
    return Piece(array=np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape),
                 frame=Frame(voxel_size_nm=voxel, origin_nm=origin))


# --------------------------------------------------------------------------- #
# what it is
# --------------------------------------------------------------------------- #
def test_a_piece_is_three_dimensional_or_three_plus_channels():
    assert _piece().spatial_shape == (8, 16, 16)
    with_channels = Piece(array=np.zeros((3, 8, 16, 16)),
                          frame=Frame(voxel_size_nm=(8, 8, 8)))
    assert with_channels.channel_axis and with_channels.channels == 3
    assert with_channels.spatial_shape == (8, 16, 16)
    assert not _piece().channel_axis and _piece().channels == 1

    for bad in [np.zeros((4, 4)), np.zeros((2, 3, 4, 5, 6))]:
        with pytest.raises(ValueError, match="3-D or 4-D"):
            Piece(array=bad, frame=Frame(voxel_size_nm=(8, 8, 8)))


def test_frozen_fixes_the_PAIRING_not_the_voxels():
    """The array is not copied and not made read-only — freezing the dataclass keeps the
    array and its frame together, which is the invariant worth enforcing."""
    import dataclasses

    p = _piece()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.frame = Frame(voxel_size_nm=(1, 1, 1))
    p.array[0, 0, 0] = 99          # still allowed, and documented as such
    assert p.array[0, 0, 0] == 99


# --------------------------------------------------------------------------- #
# where it is — the part that can be subtly wrong
# --------------------------------------------------------------------------- #
def test_bbox_is_in_the_FRAMES_voxels_not_array_indices():
    """A piece read from voxel 10 of a level reports lo = 10. Array indices are always
    0..shape and answer nothing — they cannot place the piece back, compare it with a
    sibling, or be handed to a crop argument."""
    p = _piece()                                     # origin 400/40, 160/8, 240/8
    assert p.origin_voxel == (10, 20, 30)
    assert p.bbox == BBox(lo=(10, 20, 30), hi=(18, 36, 46))
    assert p.bbox.hi == tuple(a + s for a, s in zip(p.bbox.lo, p.spatial_shape))


def test_bounds_nm_is_floats_and_deliberately_not_a_bbox():
    """`BBox` has integer bounds and its operations assume a grid, so a physical box cannot
    be one without truncating — which for a fractional voxel size would move it."""
    lo, hi = _piece().bounds_nm
    assert lo == (400.0, 160.0, 240.0)
    assert hi == (400.0 + 8 * 40, 160.0 + 16 * 8, 240.0 + 16 * 8)
    assert not isinstance(_piece().bounds_nm, BBox)

    # a voxel size that could not survive integer bounds
    fine = _piece(voxel=(0.5, 0.5, 0.5), origin=(0.25, 0.0, 0.0))
    assert fine.bounds_nm[0] == (0.25, 0.0, 0.0)


def test_an_origin_off_the_voxel_grid_has_no_integer_bbox():
    """Rounding would shift the piece by up to half a voxel against whatever it is meant to
    line up with — drift nothing downstream can detect. `bounds_nm` still works."""
    p = _piece(voxel=(8.0, 8.0, 8.0), origin=(4.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="not a whole number"):
        p.origin_voxel
    with pytest.raises(ValueError, match="not a whole number"):
        p.bbox
    assert p.bounds_nm[0] == (4.0, 0.0, 0.0), "the physical answer is always available"
    assert "nm" in repr(p), "and the repr falls back to it rather than raising"


# --------------------------------------------------------------------------- #
# crop
# --------------------------------------------------------------------------- #
def test_crop_speaks_the_same_space_bbox_reports():
    """Absolute, not array-relative. The two differ by the origin, so a caller who reads
    `piece.bbox`, narrows it, and passes it back would otherwise get a box silently offset
    by the origin — the worst kind of off-by-N."""
    p = _piece()
    sub = p.crop(((12, 24, 34), (16, 28, 38)))
    assert sub.spatial_shape == (4, 4, 4)
    assert sub.bbox == BBox(lo=(12, 24, 34), hi=(16, 28, 38))
    # ...and the voxels are the right ones
    np.testing.assert_array_equal(sub.array, p.array[2:6, 4:8, 4:8])
    # the origin moved with it, so the sub-piece is still in the parent's frame
    assert sub.frame.origin_nm == (12 * 40.0, 24 * 8.0, 34 * 8.0)
    assert sub.frame.voxel_size_nm == p.frame.voxel_size_nm


def test_crop_takes_a_bbox_or_a_pair():
    p = _piece()
    box = BBox(lo=(12, 24, 34), hi=(16, 28, 38))
    assert p.crop(box).bbox == p.crop((box.lo, box.hi)).bbox


def test_crop_outside_the_piece_is_refused_and_says_which_space(caplog):
    p = _piece()
    # array indices, which is the mistake this message is for
    with pytest.raises(ValueError, match="not array indices"):
        p.crop(((0, 0, 0), (4, 4, 4)))
    with pytest.raises(ValueError, match="not inside"):
        p.crop(((12, 24, 34), (99, 28, 38)))


def test_crop_keeps_a_channel_axis_whole():
    p = Piece(array=np.zeros((3, 8, 16, 16)),
              frame=Frame(voxel_size_nm=(8, 8, 8), origin_nm=(0, 0, 0)))
    sub = p.crop(((0, 0, 0), (4, 8, 8)))
    assert sub.shape == (3, 4, 8, 8), "channels are never sliced"
    assert sub.channels == 3


# --------------------------------------------------------------------------- #
# Frame.voxel_box — the conversion between frames
# --------------------------------------------------------------------------- #
def test_voxel_box_puts_a_physical_box_into_another_frames_voxels():
    """The only way to say "the same place" across frames, since nm is the one shared
    model space. This is what makes "show me the image under this ground-truth crop"
    expressible: the two have different voxel sizes AND different origins, so no voxel box
    transfers between them."""
    gt = Piece(array=np.zeros((364, 244, 244), "uint64"),
               frame=Frame(voxel_size_nm=(8.0, 8.0, 8.0),
                           origin_nm=(62552.0, 71832.0, 27832.0)))
    assert gt.bbox.lo == (7819, 8979, 3479)

    # the same box, read at level 0 of the parent (8 nm, origin 0) -> identity
    assert Frame(voxel_size_nm=(8.0, 8.0, 8.0)).voxel_box(gt.bounds_nm) == gt.bbox
    # ...and at level 2 (32 nm)
    coarse = Frame(voxel_size_nm=(32.0, 32.0, 32.0)).voxel_box(gt.bounds_nm)
    assert coarse.lo == (1954, 2244, 869) and coarse.hi == (2046, 2306, 931)


def test_voxel_box_grows_OUTWARD_so_the_result_contains_the_box():
    """A box that is not a whole number of coarse voxels must not lose a face. Rounding to
    nearest would, and it would look like a correct read one voxel short."""
    frame = Frame(voxel_size_nm=(10.0, 10.0, 10.0))
    box = frame.voxel_box(((5.0, 5.0, 5.0), (25.0, 25.0, 25.0)))
    assert box.lo == (0, 0, 0) and box.hi == (3, 3, 3), "0..30 nm contains 5..25"
    nearest = frame.voxel_box(((5.0, 5.0, 5.0), (25.0, 25.0, 25.0)), outward=False)
    assert nearest.lo == (0, 0, 0) and nearest.hi == (2, 2, 2)
    # an exact multiple is unaffected by the choice
    exact = ((0.0, 0.0, 0.0), (30.0, 30.0, 30.0))
    assert frame.voxel_box(exact) == frame.voxel_box(exact, outward=False)


def test_voxel_box_accounts_for_the_TARGETS_own_origin():
    """A level with a voxel_offset does not start at nm zero, and ignoring that shifts
    every converted box by the offset."""
    shifted = Frame(voxel_size_nm=(8.0, 8.0, 8.0), origin_nm=(800.0, 0.0, 0.0))
    box = shifted.voxel_box(((800.0, 0.0, 0.0), (864.0, 64.0, 64.0)))
    assert box.lo == (0, 0, 0) and box.hi == (8, 8, 8)


def test_voxel_box_checks_the_rank():
    with pytest.raises(ValueError, match="pair in nm"):
        Frame(voxel_size_nm=(8, 8, 8)).voxel_box(((0, 0), (1, 1)))


# --------------------------------------------------------------------------- #
# name — a label, not identity
# --------------------------------------------------------------------------- #
def test_the_name_rides_along_and_survives_every_derivation():
    """It exists for the same reason `kind` does: whoever read the piece knew where it came
    from, and a consumer three calls later does not."""
    p = _piece()
    assert p.name is None, "optional, and absent is fine"
    named = p.with_name("gt/vol_03700")
    assert named.name == "gt/vol_03700"
    assert named.crop(((12, 24, 34), (16, 28, 38))).name == "gt/vol_03700"
    assert named.with_kind("image").name == "gt/vol_03700"
    assert named.with_frame(Frame(voxel_size_nm=(1, 1, 1))).name == "gt/vol_03700"
    assert "gt/vol_03700" in repr(named)


def test_a_derivation_keeps_everything_it_was_not_asked_to_change():
    """`replace` rather than a fresh Piece, so a field added later cannot be silently
    dropped by one of these three."""
    p = Piece(array=np.zeros((4, 4, 4)), frame=Frame(voxel_size_nm=(8, 8, 8)),
              kind="segmentation", name="a")
    assert p.with_name("b").kind == "segmentation"
    assert p.with_kind("image").name == "a"
    assert p.with_frame(Frame(voxel_size_nm=(2, 2, 2))).kind == "segmentation"


# --------------------------------------------------------------------------- #
# copy — for an array transform that keeps everything else
# --------------------------------------------------------------------------- #
def test_copy_copies_the_ARRAY_which_is_the_point():
    """`dataclasses.replace` re-pairs the *same* array with new metadata, so an in-place
    edit on what looks like a copy reaches back into the original — and into whatever the
    original was itself a view of. That is the whole reason this method exists."""
    p = _piece()
    c = p.copy()
    assert c.array is not p.array
    c.array[0, 0, 0] = 999
    assert p.array[0, 0, 0] != 999, "the original must not move"
    # everything else rides along
    assert c.frame == p.frame and c.kind == p.kind and c.name == p.name


def test_copy_takes_a_transformed_array_without_copying_it_again():
    """Passing `array=` means you brought your own, so there is nothing to protect."""
    p = _piece()
    out = p.array.astype("float32") / 2
    c = p.copy(array=out, kind="probability")
    assert c.array is out, "no needless copy of a fresh array"
    assert c.dtype == np.dtype("float32") and c.kind == "probability"
    assert c.frame == p.frame and c.name == p.name


def test_copy_replaces_any_field_and_still_validates():
    p = _piece().with_name("gt/a").with_kind("segmentation")
    assert p.copy(name="other").name == "other"
    assert p.copy(kind=None).kind is None
    assert p.copy(frame=Frame(voxel_size_nm=(1, 1, 1))).voxel_size_nm == (1.0, 1.0, 1.0)
    # a bad value is refused rather than producing a piece nothing can interpret
    with pytest.raises(ValueError, match="kind must be one of"):
        p.copy(kind="labels")
    with pytest.raises(ValueError, match="3-D or 4-D"):
        p.copy(array=np.zeros((4, 4)))


def test_copy_names_the_fields_when_given_something_else():
    """`voxel_size` is the likely typo — it belongs to the frame, not the piece — and a bare
    TypeError about an unexpected keyword would not say so."""
    p = _piece()
    with pytest.raises(TypeError, match="belong to the frame"):
        p.copy(voxel_size=(8, 8, 8))
    with pytest.raises(TypeError, match="has no field"):
        p.copy(nope=1)


def test_copy_shares_the_frame_which_is_safe():
    """Frozen and only numbers, so there is nothing to protect against."""
    p = _piece()
    assert p.copy().frame is p.frame


# --------------------------------------------------------------------------- #
# apply — a transform that keeps the frame
# --------------------------------------------------------------------------- #
def test_apply_keeps_the_frame_name_and_kind_through_a_transform():
    """The reason the type exists: an array on its own forgets where it is after the first
    operation."""
    p = _piece().with_name("aff/data").with_kind("probability")
    out = p.apply(lambda a: a * 2)
    assert out.frame == p.frame and out.name == p.name and out.kind == p.kind
    np.testing.assert_array_equal(out.array, p.array * 2)
    assert out.array is not p.array


def test_apply_forwards_arguments_so_a_plain_array_function_just_works():
    ndimage = pytest.importorskip("scipy.ndimage")
    p = _piece(dtype="float32")
    out = p.apply(ndimage.gaussian_filter, sigma=1)
    assert out.spatial_shape == p.spatial_shape and out.frame == p.frame


def test_a_transform_that_changes_the_DTYPE_must_say_what_the_result_is():
    """Nothing can detect a change of meaning — `(a > 0.5).astype("float32")` changes
    neither shape nor dtype — but a dtype change is the signal that IS available, and it
    catches the two that matter: a probability map thresholded to a mask, and a mask
    labelled. Inheriting "probability" through either would later authorise averaging label
    ids into ids that were never in the data."""
    p = _piece(dtype="float32").with_kind("probability")

    with pytest.raises(ValueError, match="changed the dtype"):
        p.apply(lambda a: (a > 0).astype("uint8"))

    out = p.apply(lambda a: (a > 0).astype("uint8"), kind="segmentation")
    assert out.kind == "segmentation" and out.dtype == np.dtype("uint8")
    # "same" is the explicit acknowledgement, None is "no longer known"
    assert p.apply(lambda a: a.astype("float64"), kind="same").kind == "probability"
    assert p.apply(lambda a: a.astype("float64"), kind=None).kind is None
    # ...and an unchanged dtype inherits silently, which is the ordinary filter case
    assert p.apply(lambda a: a * 1).kind == "probability"


def test_the_dtype_guard_compares_FULL_dtypes_not_just_the_kind_character():
    """`uint8 -> uint32` is the connected-components case — a mask becoming labels — which
    `dtype.kind` would miss since both are unsigned. A false alarm costs twelve characters;
    a miss is silent."""
    p = _piece(dtype="uint8").with_kind("segmentation")
    with pytest.raises(ValueError, match="uint8 to uint32"):
        p.apply(lambda a: a.astype("uint32"))
    assert p.apply(lambda a: a.astype("uint32"), kind="same").kind == "segmentation"


def test_None_is_a_legal_kind_and_distinct_from_not_saying():
    """Which is why the default is a sentinel rather than None: `kind=None` means "no longer
    known", and that has to be sayable."""
    p = _piece(dtype="float32").with_kind("image")
    assert p.apply(lambda a: a.astype("uint8"), kind=None).kind is None
    assert p.apply(lambda a: a * 1).kind == "image", "not saying is not the same as None"


def test_a_transform_that_changes_the_SHAPE_must_supply_the_frame():
    """Shape and frame are two halves of one statement about where the voxels are. A 2x
    downsample halves the shape and doubles the voxel size; carrying the old frame through
    would place the result at half its real size, silently. The factor is not guessed —
    real pyramids are anisotropic, and a crop changes the origin instead of the size."""
    p = _piece()
    with pytest.raises(ValueError, match="changed the spatial shape"):
        p.apply(lambda a: a[::2, ::2, ::2])

    coarser = Frame(voxel_size_nm=(80.0, 16.0, 16.0), origin_nm=p.origin_nm)
    out = p.apply(lambda a: a[::2, ::2, ::2], frame=coarser)
    assert out.spatial_shape == (4, 8, 8)
    assert out.voxel_size_nm == (80.0, 16.0, 16.0)


def test_a_channel_reduction_counts_as_a_shape_change():
    """Rank 4 -> 3 drops the channel axis, and the guard compares SPATIAL shapes so it is
    not fooled by the rank changing on its own."""
    p = Piece(array=np.zeros((3, 4, 4, 4)), frame=Frame(voxel_size_nm=(8, 8, 8)))
    out = p.apply(lambda a: a.max(axis=0))      # 3 channels -> 1, spatial unchanged
    assert out.spatial_shape == (4, 4, 4) and not out.channel_axis
    with pytest.raises(ValueError, match="changed the spatial shape"):
        p.apply(lambda a: a[:, ::2, ::2, ::2])


def test_apply_name_is_inherited_renamed_or_CLEARED():
    """`name=None` clears rather than meaning "not specified" — the same sentinel treatment
    `kind` gets, so neither override argument has a value that quietly does nothing."""
    p = _piece().with_name("aff/data")
    assert p.apply(lambda a: a * 1).name == "aff/data"
    assert p.apply(lambda a: a * 1, name="mask").name == "mask"
    assert p.apply(lambda a: a * 1, name=None).name is None


def test_apply_chains():
    p = _piece(dtype="float32").with_kind("probability")
    out = (p.apply(lambda a: a + 1)
            .apply(lambda a: (a > 1).astype("uint8"), kind="segmentation", name="mask"))
    assert out.kind == "segmentation" and out.name == "mask"
    assert out.frame == p.frame


# --------------------------------------------------------------------------- #
# to_numpy / to_host — the one direction that needs no cupy
# --------------------------------------------------------------------------- #
class _FakeDeviceArray(np.ndarray):
    """A numpy array that claims cupy's module and converts through `.get()`.

    Enough to test the whole path with no GPU: the device branch is chosen by the type's
    defining module, and the conversion is a method on the array.
    """
    __module__ = "cupy"

    def get(self):
        return np.asarray(self).view(np.ndarray)


def test_to_numpy_is_a_no_op_for_a_host_array():
    p = _piece()
    assert p.to_numpy() is p.array
    assert p.to_host() is p, "and to_host returns the same piece, not a copy"


def test_to_numpy_converts_a_device_array_through_its_own_get():
    """**numpy cannot do this on its own.** `np.asarray` on a cupy array raises
    `TypeError: Implicit conversion to a NumPy array is not allowed` — deliberately, so that
    nobody copies gigabytes off a GPU by passing the wrong thing to a plotting call. What
    makes it possible without importing cupy is that the conversion is a *method*."""
    data = np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4)
    p = Piece(data.view(_FakeDeviceArray), Frame(voxel_size_nm=(8, 8, 8)),
              kind="image", name="gpu")
    out = p.to_numpy()
    assert type(out) is np.ndarray, "a plain ndarray, not the device subclass"
    np.testing.assert_array_equal(out, data)


def test_to_host_keeps_the_frame_kind_and_name():
    """The form to use mid-chain, where to_numpy hands out the bare array."""
    data = np.zeros((4, 4, 4), "uint8")
    p = Piece(data.view(_FakeDeviceArray),
              Frame(voxel_size_nm=(40, 8, 8), origin_nm=(80, 0, 0)),
              kind="segmentation", name="gt/a")
    host = p.to_host()
    assert type(host.array) is np.ndarray
    assert host.frame == p.frame and host.kind == p.kind and host.name == p.name
    assert host.to_host() is host, "idempotent once on the host"


def test_a_device_array_with_no_get_says_so():
    """Rather than falling through to numpy, which would raise a TypeError naming
    `__array__` and nothing about what to do."""
    class NoGet(np.ndarray):
        __module__ = "cupy"

    p = Piece(np.zeros((4, 4, 4)).view(NoGet), Frame(voxel_size_nm=(8, 8, 8)))
    with pytest.raises(TypeError, match="no .get()"):
        p.to_numpy()


def test_there_is_no_to_device_and_that_is_deliberate():
    """Coming *to* numpy is a method on the array, so this package can do it without
    importing anything; going *to* a device needs cupy itself, which is why that lives in
    `neu_proc.ops.backend.to_device`."""
    assert not hasattr(_piece(), "to_device")
