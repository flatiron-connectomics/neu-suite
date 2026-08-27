"""Vec3: a coordinate that does arithmetic, and is accepted wherever a tuple was.

The type exists because ``BBox.lo`` and ``BBox.hi`` are tuples, which do not subtract — so
placing two objects relative to one another meant writing the arithmetic out by hand. The
tests that matter are therefore not about the arithmetic (which is obvious) but about the
two things that make it safe to introduce: it goes everywhere a sequence goes, and it
refuses to be a point cloud.
"""

import numpy as np
import pytest

from neu_lib import BBox, Mesh, Skeleton, Vec3


# --------------------------------------------------------------------------- #
# the property that makes it free to adopt
# --------------------------------------------------------------------------- #
def test_it_is_accepted_wherever_a_sequence_was():
    """No existing signature had to change. If this breaks, the type stops being additive
    and every call site that takes a coordinate needs to learn about it."""
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
    """Otherwise a caller mutating it would silently edit a value declared immutable."""
    v = Vec3(1, 2, 3)
    arr = v.array
    arr[0] = 99
    assert v.z == 1.0


# --------------------------------------------------------------------------- #
# a coordinate is not a point cloud
# --------------------------------------------------------------------------- #
def test_an_n_by_3_array_is_refused_rather_than_broadcast():
    """The reason this is a type and not a bare ndarray. A (3,) and an (N, 3) broadcast
    against each other happily, so a rank mistake yields a plausible answer instead of an
    error — the shape of the zyx/xyz confusion this package works hard to pin down."""
    with pytest.raises(ValueError, match="not a coordinate"):
        Vec3.of(np.ones((5, 3)))
    with pytest.raises(ValueError, match="three components"):
        Vec3.of((1, 2))


def test_it_defers_to_numpy_rather_than_refusing_an_array_operand():
    """Returning NotImplemented rather than raising, so numpy's own broadcasting still
    works — `Vec3 * (N, 3)` is a meaningful thing to ask for."""
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


def test_the_reflected_forms_work_too():
    """`target - here` is the natural way to write an offset, and `here` may be the tuple."""
    assert (10, 10, 10) - Vec3(1, 2, 3) == Vec3(9, 8, 7)
    assert 2 * Vec3(1, 2, 3) == Vec3(2, 4, 6)


def test_a_scalar_broadcasts_to_every_axis():
    assert Vec3.of(5) == Vec3(5, 5, 5)
    assert Vec3.zero() == Vec3(0, 0, 0)


def test_it_is_hashable_so_it_can_key_a_dict():
    assert len({Vec3(1, 2, 3), Vec3(1, 2, 3), Vec3(0, 0, 0)}) == 2


def test_norm_is_the_distance_when_it_is_a_difference():
    assert Vec3(0, 3, 4).norm() == 5.0


def test_rounded_does_not_round_halves_to_even():
    """`round()` sends 0.5 and 1.5 both to even neighbours, so two points half a voxel
    apart would land on the same one. Same reason `grid._nearest` does its own thing."""
    assert Vec3(0.5, 1.5, 2.5).rounded() == Vec3(1, 2, 3)
    assert Vec3(-0.5, -1.5, -2.5).rounded() == Vec3(-1, -2, -3)


# --------------------------------------------------------------------------- #
# axis order
# --------------------------------------------------------------------------- #
def test_the_components_are_zyx_and_xyz_is_the_one_conversion():
    """Matching the package's in-memory convention. Getting this backwards mirrors an
    object through the z=x diagonal, which is a valid-looking result in the wrong place."""
    v = Vec3(1, 2, 3)
    assert v.zyx == (1.0, 2.0, 3.0)
    assert v.xyz == (3.0, 2.0, 1.0)


# --------------------------------------------------------------------------- #
# BBox, as the producer of coordinates
# --------------------------------------------------------------------------- #
def test_bbox_hands_out_arrays_that_actually_subtract():
    """The original complaint: `lo` is a tuple, so every offset was a zip and a genexp."""
    a = BBox((0, 0, 0), (10, 10, 10))
    b = BBox((100, 100, 100), (110, 110, 110))
    assert (b.lo_array - a.lo_array).tolist() == [100, 100, 100]
    assert (b.center - a.center).tolist() == [100, 100, 100]
    assert a.lo_array.dtype == np.int64


def test_the_centre_is_not_rounded_to_a_voxel():
    """A box of odd extent has no integer centre, and rounding here would put two objects
    up to a voxel per axis out of register — visible only as slight misalignment."""
    assert BBox((0, 0, 0), (11, 11, 11)).center.tolist() == [5.5, 5.5, 5.5]


def test_centre_and_arrays_stay_rank_agnostic_like_the_rest_of_the_class():
    box = BBox((0, 0), (10, 20))
    assert box.center.tolist() == [5.0, 10.0]
    with pytest.raises(ValueError, match="three components"):
        Vec3.of(box.center)


def test_translate_floors_a_fractional_offset_rather_than_truncating():
    """`int()` truncates toward zero, so -1.5 moved by -1 while +1.5 moved by +1, and
    ±0.5 moved by nothing — the same displacement landing differently by sign. Latent
    while every offset was an integer; live as soon as centring produces fractional ones."""
    box = BBox((0, 0, 0), (10, 10, 10))
    assert box.translate((0.5, 0.5, 0.5)).lo == (0, 0, 0)
    assert box.translate((-0.5, -0.5, -0.5)).lo == (-1, -1, -1)
    assert box.translate((1.5, 1.5, 1.5)).lo == (1, 1, 1)
    assert box.translate((-1.5, -1.5, -1.5)).lo == (-2, -2, -2)


def test_translating_by_whole_numbers_is_unchanged():
    """The behaviour every existing caller relies on."""
    box = BBox((0, 0, 0), (10, 10, 10))
    assert box.translate((5, -5, 0)).lo == (5, -5, 0)
    assert box.translate((5, -5, 0)).hi == (15, 5, 10)
