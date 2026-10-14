"""Removing small disconnected specks: :func:`neu_proc.ops.dust.dust`.

What is counted matters most: connected components rather than label totals, and a total
voxel count rather than an extent. Either mistake leaves output of the right shape.
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
def test_the_threshold_is_a_total_voxel_count():
    """A thin line and a compact cube of similar voxel count are judged alike."""
    a = np.zeros((10, 10, 10), "uint32")
    a[0, 0, 0:9] = 1                         # 9 voxels in a line
    a[5:7, 5:7, 5:7] = 2                     # 8 voxels in a cube
    assert _sizes(a) == [8, 9]
    assert _sizes(neu_proc.dust(a, 9)) == [9], "the line survives, the cube does not"
    assert _sizes(neu_proc.dust(a, 8)) == [8, 9], "both are above 8"


def test_dust_removes_components_not_whole_labels():
    """A speck sharing a big body's id goes, and the body stays."""
    a = np.zeros((10, 10, 10), "uint32")
    a[1:5, 1:5, 1:5] = 7                     # 64 voxels
    a[9, 9, 9] = 7                           # the same LABEL, one voxel, far away
    out = neu_proc.dust(a, 10)
    assert out[9, 9, 9] == 0, "the speck went"
    assert int((out == 7).sum()) == 64, "...and the body carrying the same id stayed"


def test_a_touching_neighbour_is_its_own_component():
    """Components are per label: a small label touching a big one is still removed (a
    binary-mask labelling, as cupyx's, would merge and keep it)."""
    a = np.zeros((10, 10, 10), "uint32")
    a[1:6, 1:6, 1:6] = 5                     # 125 voxels
    a[6, 1:3, 1:3] = 9                       # 4 voxels, face-adjacent to label 5
    out = neu_proc.dust(a, 10)
    assert _sizes(out) == [125]
    assert not (out == 9).any()


# --------------------------------------------------------------------------- #
# the range, and its direction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("args,kwargs,kept", [
    ((10,), {}, [30, 200]),
    ((3,), {}, [8, 30, 200]),
    ((5,), {"max_voxels": 50}, [8, 30]),          # a band drops both tails
    ((10,), {"invert": True}, [2, 8]),            # invert keeps only the dust
])
def test_the_kept_range(args, kwargs, kept):
    assert _sizes(neu_proc.dust(_slabs(), *args, **kwargs)) == kept


def test_connectivity_changes_what_counts_as_one_component():
    """6 (the default) splits corner-touching voxels into separate components; 26 joins
    them, so it removes fewer specks."""
    a = np.zeros((6, 6, 6), "uint32")
    a[1, 1, 1] = 1
    a[2, 2, 2] = 1                           # corner-touching: one component at 26, two at 6
    assert _sizes(neu_proc.dust(a, 2, connectivity=26)) == [2]
    assert _sizes(neu_proc.dust(a, 2, connectivity=6)) == []
    assert _sizes(neu_proc.dust(a, 2)) == [], "6 by default, so the pair is two components"


def test_no_bound_at_all_is_refused():
    with pytest.raises(ValueError, match="min_voxels"):
        neu_proc.dust(_slabs())
    with pytest.raises(ValueError, match="keeps nothing"):
        neu_proc.dust(_slabs(), 50, max_voxels=5)


# --------------------------------------------------------------------------- #
# the physical conversion, which belongs one layer up
# --------------------------------------------------------------------------- #
def test_a_volume_divides_by_the_product_of_the_voxel_sizes():
    """A 40x8x8 nm voxel is 2560 nm^3; the result is one number, not per axis."""
    assert volume_to_voxels(1e9, (40, 8, 8)) == pytest.approx(390625.0)
    assert volume_to_voxels(8000, (10, 10, 10)) == pytest.approx(8.0)
    assert isinstance(volume_to_voxels(1e6, (8, 8, 8)), float)


def test_volume_to_voxels_refuses_a_nonsense_voxel_size():
    with pytest.raises(ValueError, match="positive on every axis"):
        volume_to_voxels(1e6, (8, 0, 8))


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #
def test_a_device_array_is_moved_with_a_warning():
    """cc3d has no GPU build, so a GPU array is copied to the CPU with a warning."""
    if not backend.gpu_available():
        pytest.skip("no GPU backend")
    backend._warned.discard("cc3d.dust")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = neu_proc.dust(backend.to_gpu(_slabs()), 10)
    assert not backend.is_gpu_array(out)
    assert caught and "no GPU implementation" in str(caught[0].message)
    assert _sizes(out) == [30, 200], "and the answer is still right"


# --------------------------------------------------------------------------- #
# exclude_boundary: a component reaching a face may continue outside the crop, so its size
# is unknown and it is spared. The rule applies per component, not per label.
# --------------------------------------------------------------------------- #
def _two_blobs_one_label(shape=(12, 12, 12)):
    """Label 7 on the z=0 face and in the interior, plus interior label 9; all 3 voxels."""
    a = np.zeros(shape, "uint32")
    a[0, 1:4, 1] = 7                         # on the face
    a[6, 6:9, 6] = 7                         # interior, same label
    a[6, 2:5, 2] = 9                         # interior, its own label
    return a


def test_a_component_reaching_a_face_is_spared():
    a = np.zeros((10, 10, 10), "uint32")
    a[0, 0, 0:3] = 1                         # touches z=0
    a[5, 5, 5:8] = 2                         # wholly inside
    out = neu_proc.dust(a, 10, exclude_boundary=True)

    assert (out[0, 0, 0:3] == 1).all(), "on a face: unjudgeable, so kept"
    assert (out[5, 5, 5:8] == 0).all(), "wholly inside: judged and removed"


def test_the_rule_is_per_component_not_per_label():
    """Sharing an id with a face-touching fragment must not spare an interior speck."""
    out = neu_proc.dust(_two_blobs_one_label(), 10, exclude_boundary=True)

    assert (out[0, 1:4, 1] == 7).all(), "label 7 on the face survives"
    assert (out[6, 6:9, 6] == 0).all(), "label 7's INTERIOR speck still goes"
    assert (out[6, 2:5, 2] == 0).all(), "and so does label 9's, which never touched a face"


def test_reaching_a_face_spares_a_component_however_far_it_extends():
    """Touching a face spares a component however deep it runs; being one voxel short of
    the face does not."""
    a = np.zeros((20, 10, 10), "uint32")
    a[0:15, 5, 5] = 1                        # reaches z=0 and runs deep inside
    a[1:6, 8, 8] = 2                         # near the face, but not touching it
    out = neu_proc.dust(a, 100, exclude_boundary=True)

    assert (out[0:15, 5, 5] == 1).all()
    assert (out[1:6, 8, 8] == 0).all(), "one voxel short of the face is still interior"


def test_exclude_boundary_only_spares_more():
    """Everything the plain call keeps is still kept."""
    a = _two_blobs_one_label()
    a[4:8, 4:8, 8:11] = 5                    # a big blob, kept on size alone
    plain = neu_proc.dust(a, 10)
    spared = neu_proc.dust(a, 10, exclude_boundary=True)

    assert ((plain != 0) <= (spared != 0)).all()
    np.testing.assert_array_equal(spared[plain != 0], plain[plain != 0])


def test_invert_with_exclude_boundary_shows_what_would_go():
    """The two answers stay disjoint: a spared face-touching speck is in neither."""
    a = _two_blobs_one_label()
    dust_only = neu_proc.dust(a, 10, exclude_boundary=True, invert=True)
    kept = neu_proc.dust(a, 10, exclude_boundary=True)

    assert (dust_only[6, 6:9, 6] == 7).all() and (dust_only[6, 2:5, 2] == 9).all()
    assert (dust_only[0, 1:4, 1] == 0).all(), "spared, so not part of what would go"
    assert not ((dust_only != 0) & (kept != 0)).any(), "disjoint"


def test_a_passthrough_keyword_is_refused_rather_than_dropped():
    """This path does not call cc3d.dust, so a cc3d.dust keyword would be ignored."""
    with pytest.raises(TypeError, match="in_place"):
        neu_proc.dust(_two_blobs_one_label(), 10, exclude_boundary=True, in_place=True)


def test_exclude_boundary_agrees_with_the_face_slices_it_stands_in_for():
    """The bounding-box test the implementation uses must equal the six face slices."""
    rng = np.random.default_rng(4)
    a = (rng.random((16, 16, 16)) < 0.25).astype("uint32")
    a *= rng.integers(1, 4, a.shape, dtype="uint32")      # a few labels, tangled together

    import cc3d

    cc = cc3d.connected_components(a, connectivity=6)
    on_a_face = set()
    for face in (cc[0], cc[-1], cc[:, 0], cc[:, -1], cc[:, :, 0], cc[:, :, -1]):
        on_a_face |= set(np.unique(face).tolist())
    expected = a.copy()
    for comp in np.unique(cc):
        if comp == 0 or comp in on_a_face:
            continue
        if (cc == comp).sum() < 5:
            expected[cc == comp] = 0

    np.testing.assert_array_equal(neu_proc.dust(a, 5, exclude_boundary=True), expected)
