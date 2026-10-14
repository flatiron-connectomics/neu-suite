"""Spherical opening: :func:`neu_proc.ops.opening.opening`.

The surprising fact: fastmorph's opening is not anti-extensive, so it can return more
voxels than it was given.
"""

import warnings

import numpy as np
import pytest

from neu_proc.ops import backend
from neu_proc.ops.opening import opening


def _box_with_spike(shape=(20, 20, 20)):
    """A 10^3 body with a one-voxel-thick spike — the thing an opening is for."""
    a = np.zeros(shape, "uint32")
    a[5:15, 5:15, 5:15] = 7
    a[15:19, 9, 9] = 7
    return a


# --------------------------------------------------------------------------- #
# what it does
# --------------------------------------------------------------------------- #
def test_opening_removes_a_thin_spike():
    a = _box_with_spike()
    out = opening(a, 2)
    assert not (out[16:19, 9, 9] != 0).any(), "the spike is gone"
    assert (out[9, 9, 9] == 7), "and the body it grew out of is still there"


def test_opening_is_not_anti_extensive():
    """fastmorph's erosion and dilation read `radius` differently, so a 10^3 box grows
    (to 1384 voxels at radius 2, with fastmorph 1.8)."""
    box = np.zeros((20, 20, 20), "uint32")
    box[5:15, 5:15, 5:15] = 7
    out = opening(box, 2)

    assert int(((out != 0) & (box == 0)).sum()) > 0, "grew into background"
    assert int((out != 0).sum()) > int((box != 0).sum())


def test_radius_1_is_a_pure_dilation_and_says_so():
    """At radius 1 the erosion removes nothing and the dilation still grows."""
    box = np.zeros((20, 20, 20), "uint32")
    box[5:15, 5:15, 5:15] = 7

    with pytest.warns(UserWarning, match="erodes nothing"):
        out = opening(box, 1)
    assert ((out != 0) >= (box != 0)).all(), "nothing removed"
    assert int((out != 0).sum()) > int((box != 0).sum()), "purely grown"


def test_the_radius_follows_anisotropy():
    """fastmorph's convention, which `opening` passes through: the radius is in the units
    of ``anisotropy``."""
    a = np.zeros((32, 32, 32), "uint32")
    a[14:18, 14:18, 14:18] = 1

    from fastmorph import spherical_erode

    assert int((spherical_erode(a, radius=2.0) != 0).sum()) == 8
    assert int((spherical_erode(a, radius=2.0, anisotropy=(4.0, 1.0, 1.0)) != 0).sum()) == 16
    # 8 nm of a 4 nm z-voxel is the same 2 voxels, and erodes the seed away entirely
    assert int((spherical_erode(a, radius=8.0, anisotropy=(4.0, 1.0, 1.0)) != 0).sum()) == 0


def test_labels_do_not_merge_across_the_operation():
    """Two bodies one voxel apart stay two bodies."""
    a = np.zeros((20, 20, 20), "uint32")
    a[2:8, 2:8, 2:8] = 3
    a[9:15, 2:8, 2:8] = 4
    out = opening(a, 2)
    assert set(np.unique(out)) == {0, 3, 4}


# --------------------------------------------------------------------------- #
# what it refuses
# --------------------------------------------------------------------------- #
def test_a_nonsense_radius_is_refused():
    with pytest.raises(ValueError, match="radius"):
        opening(_box_with_spike(), 0)
    with pytest.raises(ValueError, match="radius"):
        opening(_box_with_spike(), -2)


def test_a_nonsense_anisotropy_is_refused():
    a = _box_with_spike()
    with pytest.raises(ValueError, match="one value per axis"):
        opening(a, 2, anisotropy=(8.0, 8.0))
    with pytest.raises(ValueError, match="positive"):
        opening(a, 2, anisotropy=(8.0, 0.0, 8.0))


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #
def test_a_device_array_is_moved_with_a_warning_and_comes_back():
    """fastmorph has no GPU build: the array is copied to the CPU with a warning, and the
    result goes back to the GPU, where the input lived."""
    if not backend.gpu_available():
        pytest.skip("no GPU backend")
    backend._warned.discard("fastmorph.spherical_open")
    a = backend.to_gpu(_box_with_spike())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = opening(a, 2)
    assert any("no GPU implementation" in str(w.message) for w in caught)
    assert backend.is_gpu_array(out), "handed in a GPU array, handed one back"
    assert not (out.get()[16:19, 9, 9] != 0).any(), "and the answer is still right"
