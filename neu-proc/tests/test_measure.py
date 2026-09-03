"""Readings over a label array: :mod:`neu_proc.ops.measure`.

The module was written without tests and parked (see `NOTES-neu-proc.md`), and every bug
it turned out to have was one shape: **a label value used where an index was meant.** The
configurable `background=` that caused three of them is gone — 0 is background, always —
so what is left to pin is the index/label boundary in the two places it still exists:
`sizes` resolving a renumbered index back to its label, and `component_sizes` grouping
component ids under labels.

The fixture deliberately uses labels 7, 1, 9 rather than 1, 2, 3. A self-renumbering
fixture makes `renumber` the identity and hides the whole class.
"""

import numpy as np

from neu_proc.ops.measure import component_sizes, flat_component_sizes, sizes


def _labelled():
    """Labels 7, 1, 9 in three slabs of 16 voxels, over a 0 background of 16.

    Renumbering is {0:0, 7:1, 9:3, 1:2}, so a label's value and its index disagree.
    """
    arr = np.zeros((4, 4, 4), "uint64")
    arr[0] = 7
    arr[1] = 1
    arr[2] = 9
    return arr


# --------------------------------------------------------------------------- #
# sizes
# --------------------------------------------------------------------------- #
def test_label_zero_is_never_reported():
    assert sizes(_labelled()) == {7: 16, 1: 16, 9: 16}


def test_labels_are_reported_under_their_own_values():
    """Not under their renumbered indices — 7 must not come back as 1."""
    got = sizes(_labelled())
    assert set(got) == {7, 1, 9}


def test_a_label_absent_from_the_array_is_not_reported_at_all():
    """`cc3d.statistics` counts every index up to the maximum value present, so a gappy
    label set reported the gaps as real labels of size 0 — `{1: 16, 2: 0, ..., 9: 16}`."""
    got = sizes(_labelled(), skip_relabel=True)
    assert got == {7: 16, 1: 16, 9: 16}
    assert not any(size == 0 for size in got.values())


def test_skip_relabel_agrees_with_the_relabelled_path():
    """It is an optimisation, not a semantic change; the two used to disagree."""
    arr = _labelled()
    assert sizes(arr) == sizes(arr, skip_relabel=True)


def test_an_empty_volume_reports_nothing():
    assert sizes(np.zeros((4, 4, 4), "uint64")) == {}


def test_the_background_count_is_still_derivable():
    """What the removed `background=None` was for, in one line and with no parameter."""
    arr = _labelled()
    assert arr.size - sum(sizes(arr).values()) == 16


def test_sort_orders_by_size():
    arr = np.zeros((4, 4, 4), "uint64")
    arr[0] = 7           # 16
    arr[1, :2] = 1       # 8
    arr[2, :1] = 9       # 4
    assert list(sizes(arr, sort=True)) == [7, 1, 9]
    assert list(sizes(arr, sort=True, descending=False)) == [9, 1, 7]


# --------------------------------------------------------------------------- #
# exclude_margin
# --------------------------------------------------------------------------- #
def test_exclude_margin_drops_labels_absent_from_the_interior():
    arr = np.zeros((6, 6, 6), "uint64")
    arr[0, 0, 0] = 7                 # in the margin only
    arr[2:4, 2:4, 2:4] = 9           # in the interior
    got = sizes(arr, exclude_margin=2)
    assert 7 not in got
    assert got[9] == 8


def test_exclude_margin_reports_the_WHOLE_size_of_a_surviving_label():
    """It filters which labels are reported, it does not re-count them inside the box —
    a label straddling the boundary keeps its full voxel count."""
    arr = np.zeros((6, 6, 6), "uint64")
    arr[1:4, 2, 2] = 9               # 3 voxels, one of them in the margin
    assert sizes(arr, exclude_margin=2) == {9: 3}


def test_exclude_margin_leaves_the_input_untouched():
    """It used to copy the array and blank the copy; a view must not write through."""
    arr = _labelled()
    before = arr.copy()
    sizes(arr, exclude_margin=1)
    assert np.array_equal(arr, before)


def test_a_margin_that_swallows_the_volume_reports_nothing():
    arr = np.full((4, 4, 4), 9, "uint64")
    assert sizes(arr, exclude_margin=2) == {}


# --------------------------------------------------------------------------- #
# component_sizes
# --------------------------------------------------------------------------- #
def test_components_are_grouped_under_their_own_label():
    arr = np.zeros((8, 8, 8), "uint64")
    arr[0:2, 0:2, 0:2] = 7           # one blob of 8
    arr[5:7, 5:7, 5:7] = 7           # a second, disjoint blob of 8
    arr[0:2, 5:8, 0:2] = 9           # one blob of 12
    got = component_sizes(arr)
    assert set(got) == {7, 9}
    assert np.array_equal(got[7], [8, 8])
    assert np.array_equal(got[9], [12])


def test_component_sizes_are_returned_largest_first():
    arr = np.zeros((8, 8, 8), "uint64")
    arr[0, 0, 0] = 7                 # 1
    arr[3:5, 3:5, 3:5] = 7           # 8
    arr[7, 0:3, 0] = 7               # 3
    assert np.array_equal(component_sizes(arr)[7], [8, 3, 1])


def test_the_background_is_not_reported_as_a_component():
    """cc3d gives the zero voxels component id 0, which must not surface as a label."""
    arr = np.zeros((8, 8, 8), "uint64")
    arr[0:2, 0:2, 0:2] = 7
    assert set(component_sizes(arr)) == {7}


def test_the_scatter_agrees_with_the_per_label_mask():
    """Pins the optimisation against the definition it replaced, on a field with many
    labels and many components each."""
    import cc3d

    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 12, size=(8, 8, 8), dtype="uint64")
    arr = np.kron(coarse, np.ones((3, 3, 3), dtype="uint64"))

    arr_cc = cc3d.connected_components(arr, connectivity=26)
    cc_sizes = sizes(arr_cc, sort=False, skip_relabel=True)
    expected = {}
    for label in np.unique(arr):
        if label == 0:
            continue
        comps = np.unique(arr_cc[arr == label])
        expected[int(label)] = np.array(
            sorted((cc_sizes[i] for i in comps if i != 0), reverse=True))

    got = component_sizes(arr)
    assert set(got) == set(expected)
    for label in expected:
        assert np.array_equal(got[label], expected[label]), f"label {label}"


def test_component_sizes_keys_are_plain_ints():
    """So a caller can look one up with a literal; the keys used to be numpy scalars."""
    got = component_sizes(_labelled())
    assert all(type(k) is int for k in got)


def test_sorting_components_orders_by_component_count():
    arr = np.zeros((8, 8, 8), "uint64")
    arr[0, 0, 0] = 7
    arr[0, 2, 0] = 7
    arr[0, 4, 0] = 7                 # 7 has three components
    arr[4:6, 4:6, 4:6] = 9           # 9 has one
    assert list(component_sizes(arr, sort=True)) == [7, 9]
    assert list(component_sizes(arr, sort=True, descending=False)) == [9, 7]


# --------------------------------------------------------------------------- #
# flat_component_sizes
# --------------------------------------------------------------------------- #
def test_flat_component_sizes_pools_every_component():
    arr = np.zeros((8, 8, 8), "uint64")
    arr[0:2, 0:2, 0:2] = 7           # 8
    arr[5:7, 5:7, 5:7] = 7           # 8
    arr[0, 5, 0] = 9                 # 1
    assert sorted(flat_component_sizes(arr).tolist()) == [1, 8, 8]
    assert flat_component_sizes(arr, sort=True).tolist() == [8, 8, 1]


def test_flat_component_sizes_can_still_call_the_module_level_sizes():
    """Its local accumulator used to be named `sizes`, shadowing the function."""
    from neu_proc.ops import measure

    assert callable(measure.sizes)
    assert flat_component_sizes(_labelled()).size == 3


def test_flat_component_sizes_of_an_empty_volume_is_empty():
    assert flat_component_sizes(np.zeros((4, 4, 4), "uint64")).size == 0


# --------------------------------------------------------------------------- #
# the namespace
# --------------------------------------------------------------------------- #
def test_measure_is_reachable_as_a_submodule():
    """Exposed via `_SUBMODULES` as a namespace of readings, not as loose names."""
    import neu_proc.ops as ops

    assert ops.measure.sizes is sizes
