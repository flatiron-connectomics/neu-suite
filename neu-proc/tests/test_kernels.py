"""The nm-to-voxel boundary: :mod:`neu_proc.ops.kernels`.

Each failure here gives a plausible array: rounding in the wrong place, a halo from a
different ``truncate`` than the filter uses (faint block edges at every seam), a footprint
spherical in voxels on an anisotropic volume, or an even window shifting results by half a
voxel. ``volume_to_voxels`` is tested in ``test_dust.py``.
"""

import numpy as np
import pytest

from neu_proc.ops.kernels import (GAUSSIAN_TRUNCATE, ball, box, gaussian_halo, per_axis,
                                  to_voxels)


# --------------------------------------------------------------------------- #
# per_axis and to_voxels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,rank,expected", [
    (2, 3, (2.0, 2.0, 2.0)), ((1, 4, 4), 3, (1.0, 4.0, 4.0)), (2, 2, (2.0, 2.0)),
])
def test_per_axis_broadcasts_a_scalar_and_returns_floats(value, rank, expected):
    got = per_axis(value, rank=rank)
    assert got == expected
    assert all(isinstance(v, float) for v in got)


def test_the_wrong_number_of_values_is_refused():
    """A 2-tuple for 3 axes is a zyx/xy mix-up, not something to pad."""
    with pytest.raises(ValueError, match="3 values"):
        per_axis((4, 4))


@pytest.mark.parametrize("length,voxel,expected", [
    (80, (40, 8, 8), (2.0, 10.0, 10.0)),          # per axis
    ((80, 80, 160), (40, 8, 8), (2.0, 10.0, 20.0)),
    (100, (40, 8, 8), (2.5, 12.5, 12.5)),         # not rounded
    (1, (8, 8, 8), (0.125, 0.125, 0.125)),
])
def test_to_voxels_divides_per_axis_and_does_not_round(length, voxel, expected):
    """Unrounded, because a Gaussian sigma is fractional while a radius must become whole
    voxels; each consumer rounds for itself."""
    assert to_voxels(length, voxel) == pytest.approx(expected)


@pytest.mark.parametrize("voxel", [(8, 0, 8), (8, -8, 8)])
def test_a_nonsense_voxel_size_is_refused(voxel):
    with pytest.raises(ValueError, match="positive"):
        to_voxels(80, voxel)


# --------------------------------------------------------------------------- #
# gaussian_halo: it must agree with the filter's own truncate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sigma,kwargs,expected", [
    (2.0, {}, (8, 8, 8)),                         # ceil(4.0 * sigma)
    ((0.5, 2.0, 2.0), {}, (2, 8, 8)),
    (2.0, {"truncate": 3.0}, (6, 6, 6)),          # a different truncate, passed to both
    (1.1, {}, (5, 5, 5)),                         # rounds up, never short
    (0.0, {}, (0, 0, 0)),
])
def test_the_halo_is_the_filters_kernel_radius_in_whole_voxels(sigma, kwargs, expected):
    got = gaussian_halo(sigma, **kwargs)
    assert got == expected
    assert all(isinstance(v, int) for v in got)


def test_the_default_truncate_matches_scipy():
    """A halo assuming a smaller truncate than scipy's is short at every block seam."""
    import inspect

    from scipy import ndimage

    default = inspect.signature(ndimage.gaussian_filter).parameters["truncate"].default
    assert GAUSSIAN_TRUNCATE == default


# --------------------------------------------------------------------------- #
# ball: an ellipsoid whenever the radii differ
# --------------------------------------------------------------------------- #
def test_an_isotropic_radius_gives_a_symmetric_ball():
    f = ball(2)
    assert f.shape == (5, 5, 5) and f.dtype == bool
    assert f[2, 2, 2], "the centre is in it"
    assert not f[0, 0, 0], "the corner is not: a ball, not a box"
    for axis in range(3):
        np.testing.assert_array_equal(f, np.flip(f, axis))


def test_differing_radii_give_an_ellipsoid():
    """A radius in nm becomes different voxel counts per axis on an anisotropic volume."""
    f = ball((1, 5, 5))
    assert f.shape == (3, 11, 11)
    assert f[1, 5, 0] and f[1, 5, 10], "reaches the full radius along x"
    assert f[0, 5, 5], "and 1 voxel along z"
    assert not f[0, 5, 1], "an ellipsoid, not a box"
    zs, ys, xs = np.nonzero(f)
    assert np.ptp(zs) == 2 and np.ptp(xs) == 10
    assert f.sum() < f.size


@pytest.mark.parametrize("radius,shape", [((0, 3, 3), (1, 7, 7)), (0, (1, 1, 1))])
def test_a_zero_radius_gives_no_extent_on_that_axis(radius, shape):
    """No division by zero for a 2-D footprint in a 3-D array, or a single voxel."""
    f = ball(radius)
    assert f.shape == shape
    assert f[tuple(n // 2 for n in shape)]


def test_a_negative_radius_is_refused():
    with pytest.raises(ValueError, match="negative"):
        ball((1, -1, 1))


def test_a_fractional_radius_floors_to_whole_voxels_of_extent():
    """The extent is whole voxels, but the inclusion test stays fractional."""
    assert ball(2.9).shape == (5, 5, 5)
    assert int(ball(2.9).sum()) > int(ball(2.0).sum())


# --------------------------------------------------------------------------- #
# box: odd, so the window stays centred
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,expected", [
    (4, (5, 5, 5)), (3, (3, 3, 3)), ((2, 6, 6), (3, 7, 7)),   # forced odd
    (0, (1, 1, 1)), ((0, 3, 3), (1, 3, 3)),                   # at least one
    ((2.4, 3.6, 5.0), (3, 5, 5)),                             # rounded, then odd
])
def test_box_sizes_are_odd_whole_and_at_least_one(size, expected):
    """An even window would shift the result by half a voxel."""
    got = box(size)
    assert got == expected
    assert all(isinstance(n, int) for n in got)
