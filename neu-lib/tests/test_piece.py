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
