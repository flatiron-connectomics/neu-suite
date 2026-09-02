"""Spherical opening: :func:`neu_proc.ops.opening.opening`.

The fact worth pinning is the surprising one: **fastmorph's opening is not
anti-extensive**, so it can hand back more voxels than it was given. A caller assuming the
mathematical property (``γ(X) ⊆ X``) gets a size measured after opening that is not
comparable to one measured before, and a shape that has grown into its neighbour's
background — both invisible in the output.
"""

import warnings

import numpy as np
import pytest

import neu_proc
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
    """The point of the operation: the spike is thinner than the structuring element, so
    the erosion breaks it and the dilation cannot bring it back."""
    a = _box_with_spike()
    out = opening(a, 2)
    assert not (out[16:19, 9, 9] != 0).any(), "the spike is gone"
    assert (out[9, 9, 9] == 7), "and the body it grew out of is still there"


def test_opening_is_NOT_anti_extensive():
    """Measured, and the reason the module docstring leads with it: fastmorph's erosion and
    dilation read `radius` differently, so a 10^3 box comes back 38% larger at radius 2.
    Nothing here fixes that; the test exists so nobody later assumes the subset property."""
    box = np.zeros((20, 20, 20), "uint32")
    box[5:15, 5:15, 5:15] = 7
    out = opening(box, 2)

    added = int(((out != 0) & (box == 0)).sum())
    assert added == 384, "grew into background rather than only losing voxels"
    assert int((out != 0).sum()) == 1384 > int((box != 0).sum()) == 1000


def test_radius_1_is_a_pure_dilation_and_says_so():
    """At radius 1 the erosion removes nothing while the dilation still grows by a
    6-neighbourhood, so the call does the opposite of what the name promises."""
    box = np.zeros((20, 20, 20), "uint32")
    box[5:15, 5:15, 5:15] = 7

    with pytest.warns(UserWarning, match="erodes nothing"):
        out = opening(box, 1)
    assert int((out != 0).sum()) == 1600, "1000 -> 1600: purely grown"


def test_the_radius_follows_anisotropy():
    """The same convention `dilate`'s `max_distance` follows for `sampling`, and the same
    trap: on an anisotropic volume a voxel-counted radius is lopsided in tissue while
    looking symmetric in the array."""
    a = np.zeros((32, 32, 32), "uint32")
    a[14:18, 14:18, 14:18] = 1

    from fastmorph import spherical_erode

    assert int((spherical_erode(a, radius=2.0) != 0).sum()) == 8
    assert int((spherical_erode(a, radius=2.0, anisotropy=(4.0, 1.0, 1.0)) != 0).sum()) == 16
    # 8 nm of a 4 nm z-voxel is the same 2 voxels, and erodes the seed away entirely
    assert int((spherical_erode(a, radius=8.0, anisotropy=(4.0, 1.0, 1.0)) != 0).sum()) == 0


def test_labels_do_not_merge_across_the_operation():
    """Label-aware, which a binary implementation is not: two bodies one voxel apart stay
    two bodies rather than becoming one through the dilation half."""
    a = np.zeros((20, 20, 20), "uint32")
    a[2:8, 2:8, 2:8] = 3
    a[9:15, 2:8, 2:8] = 4
    out = opening(a, 2)
    assert set(np.unique(out)) == {0, 3, 4}


# --------------------------------------------------------------------------- #
# what it refuses
# --------------------------------------------------------------------------- #
def test_a_nonsense_radius_is_refused():
    with pytest.raises(ValueError, match="radius must be positive"):
        opening(_box_with_spike(), 0)
    with pytest.raises(ValueError, match="radius must be positive"):
        opening(_box_with_spike(), -2)


def test_a_nonsense_anisotropy_is_refused():
    a = _box_with_spike()
    with pytest.raises(ValueError, match="one value per axis"):
        opening(a, 2, anisotropy=(8.0, 8.0))
    with pytest.raises(ValueError, match="positive on every axis"):
        opening(a, 2, anisotropy=(8.0, 0.0, 8.0))


# --------------------------------------------------------------------------- #
# the backend and the export
# --------------------------------------------------------------------------- #
def test_a_device_array_is_moved_with_a_warning_and_comes_back():
    """fastmorph has no device build, so this is the `host_only` path — loud, because the
    array came to the device to be fast. The answer returns **where the input lived**, which
    is `like`'s contract and what `dilate`'s own fastmorph methods do; `dust` differs only
    because cc3d's path never stages at all."""
    if not backend.gpu_available():
        pytest.skip("no GPU backend")
    backend._warned.discard("fastmorph.spherical_open")
    a = backend.to_device(_box_with_spike())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = opening(a, 2)
    assert any("no GPU implementation" in str(w.message) for w in caught)
    assert backend.is_device_array(out), "handed in a device array, handed one back"
    assert not (out.get()[16:19, 9, 9] != 0).any(), "and the answer is still right"


def test_opening_is_reachable_from_the_top_level():
    a = _box_with_spike()
    np.testing.assert_array_equal(neu_proc.opening(a, 2), opening(a, 2))
