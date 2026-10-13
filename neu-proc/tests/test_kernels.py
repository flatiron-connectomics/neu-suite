"""The nm-to-voxel boundary: :mod:`neu_proc.ops.kernels`.

This is the one module in the package where invariant 1 — one model space, physical
nanometres — is actually implemented rather than respected, so the tests are about the ways
a conversion goes *silently* wrong. Every failure here produces a plausible array of the
right shape and dtype:

- rounding in the wrong place, so a fractional sigma becomes an integer one;
- a halo derived from a different ``truncate`` than the filter uses, which is wrong in a
  shell around every block seam and reads as faint block edges;
- a footprint that is spherical in *voxels* on an anisotropic volume, which is an ellipsoid
  in tissue — physically lopsided while looking symmetric in the array;
- an even-sized window, which shifts the result by half a voxel, consistently.

``volume_to_voxels`` is tested in ``test_dust.py`` instead, beside the operation whose
threshold it converts — the hazard there is specific to dusting (a volume divides by the
*product* of the voxel sizes, so using the length form is wrong by two factors).
"""

import math

import numpy as np
import pytest

from neu_proc.ops.kernels import (GAUSSIAN_TRUNCATE, ball, box, gaussian_halo, per_axis,
                                  to_voxels)


# --------------------------------------------------------------------------- #
# per_axis: a scalar broadcasts, a sequence is checked
# --------------------------------------------------------------------------- #
def test_a_scalar_broadcasts_and_a_sequence_passes_through():
    assert per_axis(2) == (2.0, 2.0, 2.0)
    assert per_axis((1, 4, 4)) == (1.0, 4.0, 4.0)
    assert per_axis(2, rank=2) == (2.0, 2.0)


def test_the_wrong_number_of_values_is_refused():
    """Not silently padded or truncated: a 2-tuple where 3 axes are expected is a zyx/xy
    mix-up, and quietly accepting it would apply the wrong value to an axis."""
    with pytest.raises(ValueError, match="expected a scalar or 3 values"):
        per_axis((4, 4))


def test_everything_comes_back_as_floats():
    """So the two consumers can round differently — see `to_voxels`."""
    assert all(isinstance(v, float) for v in per_axis((1, 4, 4)))


# --------------------------------------------------------------------------- #
# to_voxels: per axis, and deliberately not rounded
# --------------------------------------------------------------------------- #
def test_a_length_divides_PER_AXIS():
    """The anisotropic case is the point: 80 nm is 2 voxels along a 40 nm axis and 10 along
    an 8 nm one. A single number here would be wrong on two axes out of three."""
    assert to_voxels(80, (40, 8, 8)) == (2.0, 10.0, 10.0)
    assert to_voxels((80, 80, 160), (40, 8, 8)) == (2.0, 10.0, 20.0)


def test_the_result_is_NOT_ROUNDED():
    """Because the two consumers want different roundings: a Gaussian's sigma is genuinely
    fractional, while a structuring element's radius has to become whole voxels. Rounding
    here would force one of them to be wrong."""
    assert to_voxels(100, (40, 8, 8)) == (2.5, 12.5, 12.5)
    assert to_voxels(1, (8, 8, 8))[0] == pytest.approx(0.125)


def test_a_nonsense_voxel_size_is_refused():
    """Rather than producing an inf or a negative extent, both of which survive into a
    footprint and fail somewhere unrecognisable."""
    with pytest.raises(ValueError, match="positive on every axis"):
        to_voxels(80, (8, 0, 8))
    with pytest.raises(ValueError, match="positive on every axis"):
        to_voxels(80, (8, -8, 8))


# --------------------------------------------------------------------------- #
# gaussian_halo: it must agree with the filter's own truncate
# --------------------------------------------------------------------------- #
def test_the_halo_is_scipys_own_kernel_radius():
    """`ceil(truncate * sigma)` — what the filter reads beyond the voxel it is writing."""
    assert gaussian_halo(2.0) == (8, 8, 8)                # 4.0 * 2
    assert gaussian_halo((0.5, 2.0, 2.0)) == (2, 8, 8)


def test_the_default_truncate_MATCHES_SCIPY():
    """The failure this guards is invisible: a halo assuming `truncate=3.0` while scipy's
    default is 4.0 is too small in a shell around every block seam, which shows up as faint
    block edges in the output rather than as an error."""
    from scipy import ndimage

    import inspect

    default = inspect.signature(ndimage.gaussian_filter).parameters["truncate"].default
    assert GAUSSIAN_TRUNCATE == default


def test_a_different_truncate_must_be_passed_to_BOTH():
    """The parameter exists so a caller who changes the filter's truncate can change the
    halo with it. Pinned so it cannot quietly become decorative."""
    assert gaussian_halo(2.0, truncate=3.0) == (6, 6, 6)


def test_the_halo_ROUNDS_UP_and_is_whole_voxels():
    """A read margin must never be short, so this is the one conversion that rounds, and it
    rounds outward."""
    got = gaussian_halo(1.1)
    assert got == (math.ceil(4.0 * 1.1),) * 3 == (5, 5, 5)
    assert all(isinstance(v, int) for v in got)


def test_a_zero_sigma_needs_no_margin():
    assert gaussian_halo(0.0) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# ball: an ELLIPSOID whenever the radii differ
# --------------------------------------------------------------------------- #
def test_an_isotropic_radius_gives_a_symmetric_ball():
    f = ball(2)
    assert f.shape == (5, 5, 5) and f.dtype == bool
    assert f[2, 2, 2], "the centre is in it"
    assert not f[0, 0, 0], "the corner is not — that is what makes it a ball, not a box"
    # Symmetric under a flip on every axis.
    for axis in range(3):
        np.testing.assert_array_equal(f, np.flip(f, axis))


def test_differing_radii_give_an_ELLIPSOID_not_a_ball():
    """Which is the anisotropic case and the one that matters: a spherical footprint in
    voxel space is an ellipsoid in tissue, so a morphology radius given in nm must become
    different voxel counts per axis or the operation is physically lopsided."""
    f = ball((1, 5, 5))
    assert f.shape == (3, 11, 11)
    assert f[1, 5, 0] and f[1, 5, 10], "reaches the full radius along x"
    assert f[0, 5, 5], "and 1 voxel along z, which is that axis' whole radius"
    # The shape of the thing, not just its bounding box: at the z extreme the budget is
    # spent, so nothing off-axis is included there. A per-axis BOX would include it.
    assert not f[0, 5, 1], "an ellipsoid, not a box"
    zs, ys, xs = np.nonzero(f)
    assert np.ptp(zs) == 2 and np.ptp(xs) == 10, "wider than it is tall, by the radii"
    assert f.sum() < f.size


def test_a_zero_radius_on_an_axis_contributes_no_extent_and_does_not_divide_by_zero():
    """The 2-D-footprint-in-a-3-D-array case, e.g. an in-plane operation on an anisotropic
    volume. Left unguarded this is a division by zero inside the ellipsoid equation."""
    f = ball((0, 3, 3))
    assert f.shape == (1, 7, 7)
    assert f[0, 3, 3]


def test_an_all_zero_radius_is_a_single_voxel():
    f = ball(0)
    assert f.shape == (1, 1, 1) and f.all()


def test_a_negative_radius_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        ball((1, -1, 1))


def test_a_fractional_radius_floors_to_whole_voxels_of_extent():
    """The footprint has to be an integer number of voxels across; the *test* stays
    fractional, so a radius of 2.9 admits voxels a ball of radius 2 would not."""
    assert ball(2.9).shape == (5, 5, 5)
    assert int(ball(2.9).sum()) > int(ball(2.0).sum())


# --------------------------------------------------------------------------- #
# box: odd, so the window stays centred
# --------------------------------------------------------------------------- #
def test_sizes_are_forced_ODD():
    """An even size would shift the result by half a voxel — consistently, in one direction,
    and invisibly, which is why this is not left to the caller."""
    assert box(4) == (5, 5, 5)
    assert box(3) == (3, 3, 3)
    assert box((2, 6, 6)) == (3, 7, 7)
    assert all(n % 2 for n in box((2, 4, 8)))


def test_a_size_is_at_least_one():
    """A zero-size window is not a smaller window, it is no window at all."""
    assert box(0) == (1, 1, 1)
    assert box((0, 3, 3)) == (1, 3, 3)


def test_sizes_are_whole_ints():
    assert all(isinstance(n, int) for n in box((2.4, 3.6, 5.0)))
