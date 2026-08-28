"""Label dilation: :func:`neu_proc.ops.dilate.dilate`.

Three methods behind one entry point, and the tests are shaped by what that arrangement can
get wrong rather than by the arithmetic, which the underlying libraries own. Two things
dominate:

- **``edt`` is unbounded by default**, so the natural reading of "dilate" is wrong for it.
  A test that only checked "the labels got bigger" would pass against a Voronoi partition
  of the whole array.
- **The methods take disjoint parameters**, so putting one on the wrong method is the easy
  mistake, and the libraries' own TypeErrors name *their* functions rather than this one.

`test_backend.py` covers the device path and the host/device agreement; this file is about
the entry point.
"""

import numpy as np
import pytest

import neu_proc
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
def test_unbounded_edt_leaves_NO_background():
    """Not a dilation by a radius at all — a Voronoi partition of the array. Every
    background voxel takes the nearest label, so with one seed the whole array becomes it."""
    a = _seed()
    out = dilate(a, "edt")
    assert not (out == 0).any(), "no background left"
    assert set(np.unique(out).tolist()) == {5}

    # Two seeds partition the array between them rather than growing by a radius.
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
    # Monotone in the bound, which is the property that catches a comparison flipped to `<`.
    sizes = [int((dilate(a, "edt", max_distance=d) > 0).sum()) for d in (1, 3, 5, 9)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 4


def test_a_bounded_edt_never_erases_the_seed():
    """A foreground voxel indexes itself, so its distance is 0 and it survives any bound —
    including one smaller than a voxel, which is the case that would clear the array if the
    mask were applied to everything rather than only to background."""
    a = _seed()
    out = dilate(a, "edt", max_distance=0.5)
    assert int((out == 5).sum()) >= int((a == 5).sum())


def test_max_distance_units_FOLLOW_sampling():
    """The same number means different things, and this is the whole reason the `Op` layer
    above declares lengths in nanometres. Without `sampling` the bound is counted in voxels;
    with it, in whatever `sampling` is in — so on an anisotropic volume voxel-counted growth
    is lopsided in tissue while looking symmetric in the array."""
    a = _seed(shape=(32, 32, 32), at=(15, 15, 15))
    in_voxels = dilate(a, "edt", max_distance=5)
    in_nm = dilate(a, "edt", max_distance=80, sampling=(40, 8, 8))
    assert int((in_voxels > 0).sum()) != int((in_nm > 0).sum())

    # And the anisotropy is real: with 40 nm along z and 8 nm along x, an 80 nm bound
    # reaches 2 voxels in z and 10 in x, so the result is WIDER than it is tall.
    zs, ys, xs = np.nonzero(in_nm)
    assert np.ptp(zs) < np.ptp(xs), "an isotropic voxel count would make these equal"


def test_sampling_alone_does_not_bound_anything():
    """`sampling` describes the geometry; `max_distance` is the bound. Passing only the
    former must still fill the array, or the two have been conflated."""
    out = dilate(_seed(), "edt", sampling=(40, 8, 8))
    assert not (out == 0).any()


# --------------------------------------------------------------------------- #
# the other two methods, and how they differ
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["fastmorph", "spherical"])
def test_the_fastmorph_methods_are_BOUNDED_by_their_own_parameters(method):
    """Where `edt` fills everything, these grow by a shape — so a test comparing the three
    on one array is what shows they are not interchangeable."""
    a = _seed()
    out = dilate(a, method)
    assert (out == 0).any(), "still background left"
    assert int((out > 0).sum()) > int((a > 0).sum())


def test_the_three_methods_disagree_by_construction():
    """Measured on this fixture: unbounded edt fills all 4096 voxels, fastmorph 128,
    spherical 64. Pinned because "they all dilate" is exactly the reasoning that would let
    one entry point quietly resolve to the wrong implementation."""
    a = _seed()
    got = {m: int((dilate(a, m) > 0).sum()) for m in METHODS}
    assert got["edt"] == a.size
    assert got["fastmorph"] < got["edt"] and got["spherical"] < got["fastmorph"]


def test_no_method_mutates_its_input():
    """Measured: neither fastmorph function mutates, so `dilate` needs no defensive copy —
    which is worth a test, since the alternative is a caller's array changing under them."""
    for method in METHODS:
        a = _seed()
        before = a.copy()
        dilate(a, method)
        np.testing.assert_array_equal(a, before, err_msg=f"{method} mutated its input")


# --------------------------------------------------------------------------- #
# the entry point: name resolution and the disjoint parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelling,expect", [
    ("edt", "edt"), ("euclidean", "edt"), ("EDT", "edt"),
    ("fastmorph", "fastmorph"), ("fm", "fastmorph"),
    ("spherical", "spherical"), ("sphere", "spherical"), ("spher", "spherical"),
])
def test_method_spellings_resolve(spelling, expect):
    from neu_proc.ops.dilate import _resolve

    assert _resolve(spelling) == expect


def test_an_unknown_method_is_refused_WITH_THE_LIST():
    """Before the table, a typo fell through every branch and raised
    `UnboundLocalError: cannot access local variable 'dilated'` — which names nothing
    useful. The list is what makes the message actionable."""
    with pytest.raises(ValueError, match="unknown dilation method") as exc:
        dilate(_seed(), "dilate")
    for name in METHODS:
        assert name in str(exc.value)


def test_a_parameter_on_the_WRONG_METHOD_says_which_one_takes_it():
    """The underlying libraries already refuse it, but their TypeError names *their*
    function — so `dilate(a, "edt", radius=3)` reads as a bug in scipy."""
    with pytest.raises(TypeError, match=r"does not take radius=") as exc:
        dilate(_seed(), "edt", radius=3)
    assert "spherical" in str(exc.value), "and it points at the method that does"

    with pytest.raises(TypeError, match=r"does not take sampling="):
        dilate(_seed(), "spherical", sampling=(40, 8, 8))


def test_max_distance_is_refused_on_an_ALREADY_BOUNDED_method():
    """It bounds the edt method specifically. Silently ignoring it on the others would make
    a request that looks honoured and is not."""
    with pytest.raises(TypeError, match="bounds the 'edt' method"):
        dilate(_seed(), "fastmorph", max_distance=3)


def test_accepted_is_introspected_from_the_IMPLEMENTATION():
    """Read from the library rather than listed, so it cannot drift — all three targets have
    real signatures, the C extensions included."""
    assert "sampling" in accepted("edt")
    assert "radius" in accepted("spherical")
    assert "iterations" in accepted("fastmorph")
    # Disjoint, which is the property the error messages above rely on.
    for name in METHODS:
        others = set().union(*(set(accepted(m)) for m in METHODS if m != name))
        assert not (set(accepted(name)) & others - {"parallel"}), f"{name} overlaps"


def test_reserved_keywords_are_not_offered():
    """`dilate` sets these itself, so a caller passing one would be fighting it — and
    `return_indices=False` would break the whole nearest-label trick."""
    for name in ("return_indices", "return_distances", "input"):
        assert name not in accepted("edt")


# --------------------------------------------------------------------------- #
# through the package, and through a Piece
# --------------------------------------------------------------------------- #
def test_the_lazy_top_level_export_is_the_same_function():
    assert neu_proc.dilate is dilate


def test_through_a_piece_the_frame_and_kind_ride_along():
    """`Piece.apply` is where the metadata discipline lives, so a function here stays dumb
    about both: same dtype and same shape, so neither a `kind=` nor a `frame=` is needed."""
    from neu_lib import Frame, Piece

    p = Piece(_seed(), Frame(voxel_size_nm=(40, 8, 8)), kind="segmentation", name="seg")
    out = p.apply(dilate, "edt", max_distance=3)
    assert out.frame == p.frame and out.kind == "segmentation" and out.name == "seg"
    assert int((out.array > 0).sum()) > int((p.array > 0).sum())
