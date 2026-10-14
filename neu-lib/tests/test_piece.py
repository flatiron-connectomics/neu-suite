"""Piece: an array that carries its frame. Mostly about which coordinate space each
accessor speaks, since that is what can be subtly wrong."""

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
    """The dataclass is frozen, but the array is neither copied nor read-only."""
    import dataclasses

    p = _piece()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.frame = Frame(voxel_size_nm=(1, 1, 1))
    p.array[0, 0, 0] = 99          # still allowed, and documented as such
    assert p.array[0, 0, 0] == 99


# --------------------------------------------------------------------------- #
# where it is
# --------------------------------------------------------------------------- #
def test_bbox_is_in_the_FRAMES_voxels_not_array_indices():
    p = _piece()                                     # origin 400/40, 160/8, 240/8
    assert p.origin_voxel == (10, 20, 30)
    assert p.bbox == BBox(lo=(10, 20, 30), hi=(18, 36, 46))
    assert p.bbox.hi == tuple(a + s for a, s in zip(p.bbox.lo, p.spatial_shape))


def test_bounds_nm_is_floats_and_deliberately_not_a_bbox():
    lo, hi = _piece().bounds_nm
    assert lo == (400.0, 160.0, 240.0)
    assert hi == (400.0 + 8 * 40, 160.0 + 16 * 8, 240.0 + 16 * 8)
    assert not isinstance(_piece().bounds_nm, BBox)

    # a voxel size that could not survive integer bounds
    fine = _piece(voxel=(0.5, 0.5, 0.5), origin=(0.25, 0.0, 0.0))
    assert fine.bounds_nm[0] == (0.25, 0.0, 0.0)


def test_an_origin_off_the_voxel_grid_has_no_integer_bbox():
    """It raises rather than rounding the piece by up to half a voxel."""
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
    """Absolute voxels, not array indices, and the origin moves with the crop."""
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


def test_crop_outside_the_piece_is_refused_and_says_which_space():
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
# Frame.voxel_box: the conversion between frames
# --------------------------------------------------------------------------- #
def test_voxel_box_puts_a_physical_box_into_another_frames_voxels():
    """A GT crop and the level it came from differ in voxel size AND origin."""
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
    """Rounding to nearest would drop a face and look like a correct read."""
    frame = Frame(voxel_size_nm=(10.0, 10.0, 10.0))
    box = frame.voxel_box(((5.0, 5.0, 5.0), (25.0, 25.0, 25.0)))
    assert box.lo == (0, 0, 0) and box.hi == (3, 3, 3), "0..30 nm contains 5..25"
    nearest = frame.voxel_box(((5.0, 5.0, 5.0), (25.0, 25.0, 25.0)), outward=False)
    assert nearest.lo == (0, 0, 0) and nearest.hi == (2, 2, 2)
    # an exact multiple is unaffected by the choice
    exact = ((0.0, 0.0, 0.0), (30.0, 30.0, 30.0))
    assert frame.voxel_box(exact) == frame.voxel_box(exact, outward=False)


def test_voxel_box_accounts_for_the_TARGETS_own_origin():
    shifted = Frame(voxel_size_nm=(8.0, 8.0, 8.0), origin_nm=(800.0, 0.0, 0.0))
    box = shifted.voxel_box(((800.0, 0.0, 0.0), (864.0, 64.0, 64.0)))
    assert box.lo == (0, 0, 0) and box.hi == (8, 8, 8)


def test_voxel_box_checks_the_rank():
    with pytest.raises(ValueError, match="pair in nm"):
        Frame(voxel_size_nm=(8, 8, 8)).voxel_box(((0, 0), (1, 1)))


# --------------------------------------------------------------------------- #
# name: a label, not identity
# --------------------------------------------------------------------------- #
def test_the_name_rides_along_and_survives_every_derivation():
    p = _piece()
    assert p.name is None, "optional, and absent is fine"
    named = p.with_name("gt/vol_03700")
    assert named.name == "gt/vol_03700"
    assert named.crop(((12, 24, 34), (16, 28, 38))).name == "gt/vol_03700"
    assert named.with_kind("image").name == "gt/vol_03700"
    assert named.with_frame(Frame(voxel_size_nm=(1, 1, 1))).name == "gt/vol_03700"
    assert "gt/vol_03700" in repr(named)


def test_a_derivation_keeps_everything_it_was_not_asked_to_change():
    p = Piece(array=np.zeros((4, 4, 4)), frame=Frame(voxel_size_nm=(8, 8, 8)),
              kind="segmentation", name="a")
    assert p.with_name("b").kind == "segmentation"
    assert p.with_kind("image").name == "a"
    assert p.with_frame(Frame(voxel_size_nm=(2, 2, 2))).kind == "segmentation"


# --------------------------------------------------------------------------- #
# copy
# --------------------------------------------------------------------------- #
def test_copy_copies_the_ARRAY_which_is_the_point():
    """Unlike `dataclasses.replace`, editing the copy must not reach the original."""
    p = _piece()
    c = p.copy()
    assert c.array is not p.array
    c.array[0, 0, 0] = 999
    assert p.array[0, 0, 0] != 999, "the original must not move"
    # everything else rides along
    assert c.frame == p.frame and c.kind == p.kind and c.name == p.name


def test_copy_takes_a_transformed_array_without_copying_it_again():
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
    with pytest.raises(ValueError, match="kind must be one of"):
        p.copy(kind="labels")
    with pytest.raises(ValueError, match="3-D or 4-D"):
        p.copy(array=np.zeros((4, 4)))


def test_copy_names_the_fields_when_given_something_else():
    """`voxel_size` is the likely typo, and belongs to the frame."""
    p = _piece()
    with pytest.raises(TypeError, match="belong to the frame"):
        p.copy(voxel_size=(8, 8, 8))
    with pytest.raises(TypeError, match="has no field"):
        p.copy(nope=1)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def test_apply_keeps_the_frame_name_and_kind_through_a_transform():
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
    """A thresholded probability map is a segmentation; inheriting "probability" would
    later allow averaging label ids."""
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
    """uint8 -> uint32 (a mask becoming labels) has the same `dtype.kind`."""
    p = _piece(dtype="uint8").with_kind("segmentation")
    with pytest.raises(ValueError, match="uint8 to uint32"):
        p.apply(lambda a: a.astype("uint32"))
    assert p.apply(lambda a: a.astype("uint32"), kind="same").kind == "segmentation"


def test_a_transform_that_changes_the_SHAPE_must_supply_the_frame():
    """A 2x downsample halves the shape and doubles the voxel size; no factor is guessed."""
    p = _piece()
    with pytest.raises(ValueError, match="changed the spatial shape"):
        p.apply(lambda a: a[::2, ::2, ::2])

    coarser = Frame(voxel_size_nm=(80.0, 16.0, 16.0), origin_nm=p.origin_nm)
    out = p.apply(lambda a: a[::2, ::2, ::2], frame=coarser)
    assert out.spatial_shape == (4, 8, 8)
    assert out.voxel_size_nm == (80.0, 16.0, 16.0)


def test_a_channel_reduction_counts_as_a_shape_change():
    """The guard compares SPATIAL shapes, so a rank change alone does not fool it."""
    p = Piece(array=np.zeros((3, 4, 4, 4)), frame=Frame(voxel_size_nm=(8, 8, 8)))
    out = p.apply(lambda a: a.max(axis=0))      # 3 channels -> 1, spatial unchanged
    assert out.spatial_shape == (4, 4, 4) and not out.channel_axis
    with pytest.raises(ValueError, match="changed the spatial shape"):
        p.apply(lambda a: a[:, ::2, ::2, ::2])


def test_apply_name_is_inherited_renamed_or_CLEARED():
    p = _piece().with_name("aff/data")
    assert p.apply(lambda a: a * 1).name == "aff/data"
    assert p.apply(lambda a: a * 1, name="mask").name == "mask"
    assert p.apply(lambda a: a * 1, name=None).name is None


# --------------------------------------------------------------------------- #
# to_numpy / to_cpu
# --------------------------------------------------------------------------- #
class _FakeGpuArray(np.ndarray):
    """A numpy array that claims cupy's module and converts through `.get()`."""
    __module__ = "cupy"

    def get(self):
        return np.asarray(self).view(np.ndarray)


def test_to_numpy_is_a_no_op_for_a_cpu_array():
    p = _piece()
    assert p.to_numpy() is p.array
    assert p.to_cpu() is p, "and to_cpu returns the same piece, not a copy"


def test_to_numpy_converts_a_gpu_array_through_its_own_get():
    data = np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4)
    p = Piece(data.view(_FakeGpuArray), Frame(voxel_size_nm=(8, 8, 8)),
              kind="image", name="gpu")
    out = p.to_numpy()
    assert type(out) is np.ndarray, "a plain ndarray, not the GPU subclass"
    np.testing.assert_array_equal(out, data)


def test_to_cpu_keeps_the_frame_kind_and_name():
    data = np.zeros((4, 4, 4), "uint8")
    p = Piece(data.view(_FakeGpuArray),
              Frame(voxel_size_nm=(40, 8, 8), origin_nm=(80, 0, 0)),
              kind="segmentation", name="gt/a")
    on_cpu = p.to_cpu()
    assert type(on_cpu.array) is np.ndarray
    assert on_cpu.frame == p.frame and on_cpu.kind == p.kind and on_cpu.name == p.name
    assert on_cpu.to_cpu() is on_cpu, "idempotent once in CPU memory"


def test_a_gpu_array_with_no_get_says_so():
    class NoGet(np.ndarray):
        __module__ = "cupy"

    p = Piece(np.zeros((4, 4, 4)).view(NoGet), Frame(voxel_size_nm=(8, 8, 8)))
    with pytest.raises(TypeError, match="no .get()"):
        p.to_numpy()


def test_there_is_no_to_gpu_and_that_is_deliberate():
    """Going to the GPU needs cupy, which would break the numpy-only rule; it lives in
    `neu_proc.ops.backend.to_gpu`. Do not "complete" the asymmetry."""
    assert not hasattr(_piece(), "to_gpu")
    assert not hasattr(_piece(), "to_device"), "nor under the old name"
