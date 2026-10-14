"""Label dilation: :func:`neu_proc.ops.dilate.dilate`.

Two things dominate: ``edt`` is unbounded unless ``max_distance`` is given (a Voronoi
partition, not a dilation), and the three methods take disjoint parameters. The GPU path is
in ``test_backend.py``.
"""

import numpy as np
import pytest

from neu_proc.ops.dilate import METHODS, accepted, dilate


def _seed(shape=(16, 16, 16), at=(7, 7, 7), size=2, label=5):
    """One cube of ``label`` in an otherwise empty array."""
    a = np.zeros(shape, "uint32")
    sl = tuple(slice(c, c + size) for c in at)
    a[sl] = label
    return a


# --------------------------------------------------------------------------- #
# edt is unbounded, which is the surprising part
# --------------------------------------------------------------------------- #
def test_unbounded_edt_leaves_no_background():
    """Every background voxel takes the nearest label: one seed fills the array, two
    partition it."""
    a = _seed()
    out = dilate(a, "edt")
    assert not (out == 0).any()
    assert set(np.unique(out).tolist()) == {5}

    b = _seed(label=5, at=(2, 2, 2))
    b[12:14, 12:14, 12:14] = 9
    parts = dilate(b, "edt")
    assert not (parts == 0).any()
    assert set(np.unique(parts).tolist()) == {5, 9}


def test_max_distance_is_what_makes_edt_a_dilation():
    a = _seed(shape=(32, 32, 32), at=(15, 15, 15))
    grown = dilate(a, "edt", max_distance=5)
    assert (grown > 0).sum() < grown.size, "bounded, so background survives"
    assert (grown > 0).sum() > (a > 0).sum(), "...and it did grow"
    # strictly monotone in the bound
    sizes = [int((dilate(a, "edt", max_distance=d) > 0).sum()) for d in (1, 3, 5, 9)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 4


def test_a_bounded_edt_never_erases_the_seed():
    """Even a bound under one voxel keeps the seed: the mask applies to background only."""
    a = _seed()
    out = dilate(a, "edt", max_distance=0.5)
    assert int((out == 5).sum()) >= int((a == 5).sum())


def test_max_distance_units_follow_sampling():
    """Without ``sampling`` the bound is in voxels; with it, in sampling's units, so on an
    anisotropic voxel the growth differs per axis."""
    a = _seed(shape=(32, 32, 32), at=(15, 15, 15))
    in_voxels = dilate(a, "edt", max_distance=5)
    in_nm = dilate(a, "edt", max_distance=80, sampling=(40, 8, 8))
    assert int((in_voxels > 0).sum()) != int((in_nm > 0).sum())

    # 80 nm reaches 2 voxels in z (40 nm) and 10 in x (8 nm): wider than tall.
    zs, ys, xs = np.nonzero(in_nm)
    assert np.ptp(zs) < np.ptp(xs)


def test_sampling_alone_does_not_bound_anything():
    out = dilate(_seed(), "edt", sampling=(40, 8, 8))
    assert not (out == 0).any()


# --------------------------------------------------------------------------- #
# the three methods differ
# --------------------------------------------------------------------------- #
def test_the_three_methods_disagree_by_construction():
    """Unbounded edt fills all 4096 voxels of this fixture; fastmorph and spherical grow by
    a shape, and spherical less than fastmorph."""
    a = _seed()
    got = {m: int((dilate(a, m) > 0).sum()) for m in METHODS}
    assert got["edt"] == a.size
    assert got["fastmorph"] < got["edt"] and got["spherical"] < got["fastmorph"]
    assert got["spherical"] > int((a > 0).sum()), "the bounded methods still grow"


def test_no_method_mutates_its_input():
    for method in METHODS:
        a = _seed()
        before = a.copy()
        dilate(a, method)
        np.testing.assert_array_equal(a, before, err_msg=f"{method} mutated its input")


# --------------------------------------------------------------------------- #
# the entry point: name resolution and the disjoint parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelling,expect", [
    ("euclidean", "edt"), ("EDT", "edt"), ("fm", "fastmorph"),
    ("sphere", "spherical"), ("spher", "spherical"),
])
def test_method_spellings_resolve(spelling, expect):
    a = _seed()
    np.testing.assert_array_equal(dilate(a, spelling), dilate(a, expect))


def test_an_unknown_method_is_refused_with_the_list():
    with pytest.raises(ValueError, match="unknown dilation method") as exc:
        dilate(_seed(), "dilate")
    for name in METHODS:
        assert name in str(exc.value)


def test_a_parameter_on_the_wrong_method_says_which_one_takes_it():
    """The libraries' own TypeErrors name their functions, not this one."""
    with pytest.raises(TypeError, match="radius=") as exc:
        dilate(_seed(), "edt", radius=3)
    assert "spherical" in str(exc.value), "and it points at the method that does"

    with pytest.raises(TypeError, match="sampling="):
        dilate(_seed(), "spherical", sampling=(40, 8, 8))


def test_max_distance_is_refused_on_an_already_bounded_method():
    """Ignoring it on the other methods would look like an honoured request."""
    with pytest.raises(TypeError, match="'edt'"):
        dilate(_seed(), "fastmorph", max_distance=3)


def test_accepted_is_introspected_and_disjoint():
    assert "sampling" in accepted("edt")
    assert "radius" in accepted("spherical")
    assert "iterations" in accepted("fastmorph")
    for name in METHODS:
        others = set().union(*(set(accepted(m)) for m in METHODS if m != name))
        assert not (set(accepted(name)) & others - {"parallel"}), f"{name} overlaps"


def test_reserved_keywords_are_not_offered():
    """`dilate` sets these itself; return_indices=False would break the nearest-label step."""
    for name in ("return_indices", "return_distances", "input"):
        assert name not in accepted("edt")


def test_through_a_piece_the_frame_and_kind_ride_along():
    from neu_lib import Frame, Piece

    p = Piece(_seed(), Frame(voxel_size_nm=(40, 8, 8)), kind="segmentation", name="seg")
    out = p.apply(dilate, "edt", max_distance=3)
    assert out.frame == p.frame and out.kind == "segmentation" and out.name == "seg"
    assert int((out.array > 0).sum()) > int((p.array > 0).sum())
