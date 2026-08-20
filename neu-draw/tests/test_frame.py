"""Nanometres and axis order — the two conventions everything else assumes.

Both fail silently when broken. A wrong voxel size puts a body at the wrong scale, which
looks like a body; a reversed axis order mirrors it through the z=x diagonal, which
looks like a different body. Neither raises.
"""

import numpy as np
import pytest

from neu_draw.geometry import Frame, to_xyz


def test_anisotropy_is_per_axis_not_a_factor():
    """The reason a Frame exists instead of an int scale. A `(1, 2, 2)` pyramid — halve
    x and y, leave z — is ordinary, and `2 ** scale` cannot express it."""
    frame = Frame(voxel_size_nm=(40.0, 8.0, 8.0))
    assert not frame.is_isotropic
    assert list(frame.to_nm(np.array([1, 1, 1]))) == [40.0, 8.0, 8.0]


def test_round_trip_through_voxels_and_back():
    frame = Frame(voxel_size_nm=(40.0, 8.0, 8.0), origin_nm=(1000.0, -32.0, 0.0))
    points = np.array([[0, 0, 0], [3, 10, 7], [-2, 1, 4]], dtype=float)
    assert np.allclose(frame.to_voxel(frame.to_nm(points)), points)


def test_origin_places_a_cropped_level_in_its_parents_frame():
    frame = Frame((8.0, 8.0, 8.0), origin_nm=(800.0, 0.0, 0.0))
    assert list(frame.to_nm(np.array([0, 0, 0]))) == [800.0, 0.0, 0.0]


def test_to_voxel_does_not_round():
    """Rounding is the caller's policy; doing it here would hide a half-voxel offset."""
    frame = Frame((8.0, 8.0, 8.0))
    assert frame.to_voxel(np.array([4.0, 12.0, 0.0])).tolist() == [0.5, 1.5, 0.0]


def test_a_single_point_stays_a_single_point():
    frame = Frame((8.0, 8.0, 8.0))
    assert frame.to_nm(np.array([1, 2, 3])).shape == (3,)
    assert frame.to_nm(np.array([[1, 2, 3]])).shape == (1, 3)


def test_a_frame_rejects_a_non_positive_voxel():
    with pytest.raises(ValueError, match="positive"):
        Frame((8.0, 0.0, 8.0))


def test_to_xyz_reverses_and_returns_a_real_buffer():
    """A reversed numpy view is not contiguous, and a GPU buffer must be."""
    out = to_xyz(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert out.tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]
    assert out.flags["C_CONTIGUOUS"] and out.dtype == np.float32


def test_to_xyz_is_its_own_inverse():
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.array_equal(to_xyz(to_xyz(points)), points)
