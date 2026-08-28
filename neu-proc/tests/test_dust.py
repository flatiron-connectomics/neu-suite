"""Removing small disconnected specks: :func:`neu_proc.ops.dust.dust`.

The two things worth pinning are both about *what* is being counted: connected components
rather than label totals, and a total voxel count rather than any per-axis extent. Both
produce identically-shaped output whichever way they go, so a mistake in either is invisible
in the result.
"""

import warnings

import numpy as np
import pytest

import neu_proc
from neu_proc.ops import backend
from neu_proc.ops.kernels import volume_to_voxels


def _slabs(sizes=(2, 8, 30, 200)):
    """One connected component per entry, of that many voxels, well separated in z."""
    a = np.zeros((4 * (len(sizes) + 1), 10, 10), "uint32")
    for i, n in enumerate(sizes, start=1):
        flat = np.zeros(400, "uint32")
        flat[:n] = i
        a[i * 4:i * 4 + 4] = flat.reshape(4, 10, 10)
    return a


def _sizes(x):
    return sorted(np.unique(x[x > 0], return_counts=True)[1].tolist())


# --------------------------------------------------------------------------- #
# what gets counted
# --------------------------------------------------------------------------- #
def test_the_threshold_is_a_TOTAL_voxel_count():
    """Not an extent along any axis: a long thin component of N voxels and a compact one of N
    voxels are treated alike, which is what "how much of it is there" should mean."""
    a = np.zeros((10, 10, 10), "uint32")
    a[0, 0, 0:9] = 1                         # 9 voxels in a line
    a[5:7, 5:7, 5:7] = 2                     # 8 voxels in a cube
    assert _sizes(a) == [8, 9]
    assert _sizes(neu_proc.dust(a, 9)) == [9], "the line survives, the cube does not"
    assert _sizes(neu_proc.dust(a, 8)) == [8, 9], "both are above 8"


def test_dust_removes_components_not_whole_LABELS():
    """A speck carrying the same id as a big body elsewhere goes, and the body stays. This is
    the behaviour that matters here: one body in this dataset measured 344 components at
    scale 1, most of them single-voxel specks carrying the label."""
    a = np.zeros((10, 10, 10), "uint32")
    a[1:5, 1:5, 1:5] = 7                     # 64 voxels
    a[9, 9, 9] = 7                           # the same LABEL, one voxel, far away
    out = neu_proc.dust(a, 10)
    assert out[9, 9, 9] == 0, "the speck went"
    assert int((out == 7).sum()) == 64, "...and the body carrying the same id stayed"


def test_a_touching_neighbour_is_its_own_component():
    """The property a binary-mask implementation cannot have, and the reason there is no GPU
    path: `cupyx.scipy.ndimage.label` works on a mask, so two different bodies that touch
    become one component and the small one survives. Measured: a 4-voxel label adjacent to a
    125-voxel one is kept by that approach and removed here."""
    a = np.zeros((10, 10, 10), "uint32")
    a[1:6, 1:6, 1:6] = 5                     # 125 voxels
    a[6, 1:3, 1:3] = 9                       # 4 voxels, face-adjacent to label 5
    out = neu_proc.dust(a, 10)
    assert _sizes(out) == [125]
    assert not (out == 9).any()


# --------------------------------------------------------------------------- #
# the range, and its direction
# --------------------------------------------------------------------------- #
def test_min_voxels_keeps_at_least_that_many():
    assert _sizes(neu_proc.dust(_slabs(), 10)) == [30, 200]
    assert _sizes(neu_proc.dust(_slabs(), 3)) == [8, 30, 200]


def test_a_band_drops_BOTH_tails():
    """cc3d reads a pair as the range to keep, so a max also removes the large — which is no
    longer dusting, and worth a test since "remove the big things too" is rarely meant."""
    assert _sizes(neu_proc.dust(_slabs(), 5, max_voxels=50)) == [8, 30]


def test_invert_keeps_only_the_dust():
    """The useful way to *look* at what you are about to throw away."""
    assert _sizes(neu_proc.dust(_slabs(), 10, invert=True)) == [2, 8]


def test_connectivity_changes_what_counts_as_one_component():
    """26 is the default and the more conservative choice for dusting: it merges specks into
    neighbours where it can, so it removes fewer of them. 6 finds more components."""
    a = np.zeros((6, 6, 6), "uint32")
    a[1, 1, 1] = 1
    a[2, 2, 2] = 1                           # corner-touching: one component at 26, two at 6
    assert _sizes(neu_proc.dust(a, 2, connectivity=26)) == [2]
    assert _sizes(neu_proc.dust(a, 2, connectivity=6)) == []


def test_no_bound_at_all_is_refused():
    with pytest.raises(ValueError, match="give min_voxels"):
        neu_proc.dust(_slabs())
    with pytest.raises(ValueError, match="keeps nothing"):
        neu_proc.dust(_slabs(), 50, max_voxels=5)


# --------------------------------------------------------------------------- #
# the physical conversion, which belongs one layer up
# --------------------------------------------------------------------------- #
def test_a_volume_divides_by_the_PRODUCT_of_the_voxel_sizes():
    """The conversion `to_voxels` does not do. Using the length form here would be wrong by
    two factors of the voxel size — a 40x8x8 nm voxel is 2560 nm^3, not 40."""
    assert volume_to_voxels(1e9, (40, 8, 8)) == pytest.approx(1e9 / 2560)
    assert volume_to_voxels(1e9, (40, 8, 8)) == pytest.approx(390625.0)
    # isotropic, for a sanity check that is easy to do in the head
    assert volume_to_voxels(8000, (10, 10, 10)) == pytest.approx(8.0)
    # a single number, never per-axis
    assert isinstance(volume_to_voxels(1e6, (8, 8, 8)), float)


def test_volume_to_voxels_refuses_a_nonsense_voxel_size():
    with pytest.raises(ValueError, match="positive on every axis"):
        volume_to_voxels(1e6, (8, 0, 8))


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #
def test_a_device_array_is_moved_with_a_warning():
    """cc3d has no GPU build, so this is the `host_only` path — loud, because the array came
    to the device to be fast and a copy back is the slowest thing that can happen to it."""
    if not backend.gpu_available():
        pytest.skip("no GPU backend")
    backend._warned.discard("cc3d.dust")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = neu_proc.dust(backend.to_device(_slabs()), 10)
    assert not backend.is_device_array(out)
    assert caught and "no GPU implementation" in str(caught[0].message)
    assert _sizes(out) == [30, 200], "and the answer is still right"
