"""Vec3: accepted wherever a sequence is, and never a point cloud."""

import numpy as np
import pytest

from neu_lib import BBox, Mesh, Skeleton, Vec3


def test_it_is_accepted_wherever_a_sequence_was():
    mesh = Mesh(np.zeros((3, 3), "f4"), np.array([[0, 1, 2]]))
    assert mesh.translate(Vec3(0, 0, 500)).vertices_zyx_nm[0].tolist() == [0, 0, 500]

    skeleton = Skeleton(np.zeros((2, 3), "f4"), np.array([[0, 1]]))
    assert skeleton.translate(Vec3(1, 0, 0)).vertices_zyx_nm[0].tolist() == [1, 0, 0]

    box = BBox((1, 1, 1), (100, 100, 100))
    assert box.aligned(Vec3(64, 64, 64)).hi == (128, 128, 128)
    assert box.translate(Vec3(10, 0, 0)).lo == (11, 1, 1)


def test_it_satisfies_the_sequence_and_array_protocols():
    v = Vec3(1, 2, 3)
    assert len(v) == 3
    assert v[0] == 1 and v[-1] == 3
    assert list(v) == [1.0, 2.0, 3.0]
    assert np.asarray(v).tolist() == [1.0, 2.0, 3.0]
    assert np.asarray(v).dtype == np.float64
    z, y, x = v
    assert (z, y, x) == (1.0, 2.0, 3.0)


def test_the_array_it_hands_out_is_a_copy():
    v = Vec3(1, 2, 3)
    arr = v.array
    arr[0] = 99
    assert v.z == 1.0


# --------------------------------------------------------------------------- #
# a coordinate is not a point cloud
# --------------------------------------------------------------------------- #
def test_an_n_by_3_array_is_refused_rather_than_broadcast():
    """A (3,) would broadcast against an (N, 3); a Vec3 refuses one."""
    with pytest.raises(ValueError, match="not a coordinate"):
        Vec3.of(np.ones((5, 3)))
    with pytest.raises(ValueError, match="three components"):
        Vec3.of((1, 2))


def test_it_defers_to_numpy_rather_than_refusing_an_array_operand():
    got = Vec3(1, 1, 1) * np.ones((4, 3))
    assert isinstance(got, np.ndarray) and got.shape == (4, 3)


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_arithmetic_with_a_vec_a_sequence_or_a_scalar():
    assert Vec3(1, 2, 3) + Vec3(1, 1, 1) == Vec3(2, 3, 4)
    assert Vec3(1, 2, 3) - (1, 1, 1) == Vec3(0, 1, 2)
    assert Vec3(1, 2, 3) * 2 == Vec3(2, 4, 6)
    assert Vec3(4, 4, 4) / 2 == Vec3(2, 2, 2)
    assert -Vec3(1, 2, 3) == Vec3(-1, -2, -3)
    # reflected forms: `target - here`, where `here` may be a tuple
    assert (10, 10, 10) - Vec3(1, 2, 3) == Vec3(9, 8, 7)
    assert 2 * Vec3(1, 2, 3) == Vec3(2, 4, 6)
    assert Vec3.of(5) == Vec3(5, 5, 5)
    assert Vec3.zero() == Vec3(0, 0, 0)
    assert Vec3(0, 3, 4).norm() == 5.0


def test_it_is_hashable_so_it_can_key_a_dict():
    assert len({Vec3(1, 2, 3), Vec3(1, 2, 3), Vec3(0, 0, 0)}) == 2


def test_rounded_does_not_round_halves_to_even():
    assert Vec3(0.5, 1.5, 2.5).rounded() == Vec3(1, 2, 3)
    assert Vec3(-0.5, -1.5, -2.5).rounded() == Vec3(-1, -2, -3)


# --------------------------------------------------------------------------- #
# axis order
# --------------------------------------------------------------------------- #
def test_the_components_are_zyx_and_xyz_is_the_one_conversion():
    v = Vec3(1, 2, 3)
    assert v.zyx == (1.0, 2.0, 3.0)
    assert v.xyz == (3.0, 2.0, 1.0)


# --------------------------------------------------------------------------- #
# BBox's array views
# --------------------------------------------------------------------------- #
def test_bbox_hands_out_arrays_that_actually_subtract():
    a = BBox((0, 0, 0), (10, 10, 10))
    b = BBox((100, 100, 100), (110, 110, 110))
    assert (b.lo_array - a.lo_array).tolist() == [100, 100, 100]
    assert (b.center - a.center).tolist() == [100, 100, 100]
    assert a.lo_array.dtype == np.int64


def test_the_centre_is_unrounded_and_rank_agnostic():
    assert BBox((0, 0, 0), (11, 11, 11)).center.tolist() == [5.5, 5.5, 5.5]
    box = BBox((0, 0), (10, 20))
    assert box.center.tolist() == [5.0, 10.0]
    with pytest.raises(ValueError, match="three components"):
        Vec3.of(box.center)


@pytest.mark.parametrize("offset, lo", [
    ((0.5, 0.5, 0.5), (0, 0, 0)),
    ((-0.5, -0.5, -0.5), (-1, -1, -1)),       # truncating would not move it at all
    ((1.5, 1.5, 1.5), (1, 1, 1)),
    ((-1.5, -1.5, -1.5), (-2, -2, -2)),
    ((5, -5, 0), (5, -5, 0)),                  # whole numbers move exactly
])
def test_translate_floors_a_fractional_offset(offset, lo):
    moved = BBox((0, 0, 0), (10, 10, 10)).translate(offset)
    assert moved.lo == lo and moved.shape == (10, 10, 10)
