"""Shifting an array by whole voxels: :func:`neu_proc.ops.offset.offset`.

Each failure gives a plausible array: a changed shape, data wrapping round the far face, or
``extend`` repeating a slab instead of the edge plane.
"""

import numpy as np
import pytest

from neu_proc.ops.offset import offset


def _ramp(shape=(6, 5, 4)):
    """Distinct per-plane values on every axis, so a shift is visible and a wrap obvious."""
    return np.arange(np.prod(shape), dtype="uint32").reshape(shape)


# --------------------------------------------------------------------------- #
# the shape is invariant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shift", [1, 2, 3, -1, -2, -3, (1, -2, 3), (0, 0, 0)])
def test_the_shape_never_changes(shift):
    a = _ramp()
    assert offset(a, shift).shape == a.shape
    assert offset(a, shift).dtype == a.dtype


def test_a_scalar_shifts_every_axis():
    a = _ramp()
    per_axis = offset(a, (2, 2, 2))
    np.testing.assert_array_equal(offset(a, 2), per_axis)


def test_zero_is_the_identity():
    a = _ramp()
    np.testing.assert_array_equal(offset(a, 0), a)
    np.testing.assert_array_equal(offset(a, (0, 0, 0), "const"), a)


# --------------------------------------------------------------------------- #
# which way, and what fills in behind
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shift,moved,source", [
    (1, slice(1, None), slice(None, -1)),          # positive: toward higher indices
    (-1, slice(None, -1), slice(1, None)),
])
def test_the_direction_of_a_shift(shift, moved, source):
    a = _ramp()
    np.testing.assert_array_equal(offset(a, (shift, 0, 0))[moved], a[source])


def test_extend_repeats_the_edge_plane_not_a_slab():
    """Only visible for |offset| > 1."""
    a = _ramp()
    out = offset(a, (3, 0, 0))

    for z in range(3):
        np.testing.assert_array_equal(out[z], a[0], f"plane {z} is a copy of the edge plane")
    np.testing.assert_array_equal(out[3:], a[:-3])


def test_const_fills_with_the_value_asked_for():
    a = _ramp() + 1                          # no zeros, so the fill is unmistakable
    out = offset(a, (2, 0, 0), "const")
    assert (out[:2] == 0).all(), "background by default"
    assert (offset(a, (2, 0, 0), "const", fill_const=99)[:2] == 99).all()


def test_nothing_wraps_around_the_far_face():
    """As `np.roll` would."""
    a = np.zeros((8, 4, 4), "uint32")
    a[0] = 5                                 # only the first plane is labelled
    out = offset(a, (-2, 0, 0))              # ...shifted off the near face entirely
    assert not (out == 5).any(), "the labelled plane left the array and did not reappear"


def test_every_axis_shifts_independently():
    a = _ramp((5, 5, 5))
    out = offset(a, (1, -1, 0))
    np.testing.assert_array_equal(out[1:, :-1, :], a[:-1, 1:, :])


# --------------------------------------------------------------------------- #
# what it refuses
# --------------------------------------------------------------------------- #
def test_a_bad_fill_method_is_refused_with_the_list():
    """"dilate" was removed as a fill method: it filled every interior background voxel."""
    with pytest.raises(ValueError, match="unknown fill_method") as caught:
        offset(_ramp(), 1, "dilate")
    assert "extend" in str(caught.value) and "const" in str(caught.value), \
        "the refusal names the methods that do exist"


def test_the_spellings_each_method_answers_to():
    a = _ramp()
    for name in ("extend", "edge", "replicate", "EXTEND"):
        np.testing.assert_array_equal(offset(a, 1, name), offset(a, 1, "extend"))
    for name in ("const", "constant", "background", "bg"):
        np.testing.assert_array_equal(offset(a, 1, name), offset(a, 1, "const"))


def test_an_offset_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one value per axis"):
        offset(_ramp(), (1, 2))


def test_an_offset_that_would_empty_the_array_is_refused():
    """A shift by the full extent would return pure fill."""
    with pytest.raises(ValueError, match="out of view"):
        offset(_ramp((6, 5, 4)), (6, 0, 0))
    with pytest.raises(ValueError, match="out of view"):
        offset(_ramp((6, 5, 4)), (0, 0, -4))


def test_a_non_3d_array_is_refused():
    with pytest.raises(ValueError, match="3-D zyx"):
        offset(np.zeros((4, 4), "uint32"), 1)
