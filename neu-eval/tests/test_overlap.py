"""The contingency table: canonical form, the ignore rules, and additivity.

``test_adding_block_tables_equals_the_whole_volume_table`` is the one the rest of the
package leans on. The blockwise sweep is built on the claim that summing block tables gives
the volume's table, so that claim is pinned here, before anything depends on it — the same
reason ``NEU-PROC-PLAN``'s equivalence battery comes before the ops it checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval import overlap
from neu_eval.overlap import Contingency, contingency


def _pair(seed=0, shape=(12, 10, 9), n_a=7, n_b=11):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_a, size=shape, dtype=np.uint64)
    b = rng.integers(0, n_b, size=shape, dtype=np.uint64)
    return a, b


# -- canonical form -------------------------------------------------------------------

def test_pairs_come_back_sorted_with_no_duplicates_and_no_empty_cells():
    a, b = _pair()
    c = contingency(a, b)
    order = np.lexsort((c.b_ids, c.a_ids))
    assert np.array_equal(order, np.arange(c.n_pairs))
    combined = np.stack([c.a_ids, c.b_ids], axis=1)
    assert len(np.unique(combined, axis=0)) == c.n_pairs
    assert (c.counts > 0).all()


def test_counts_account_for_every_voxel():
    a, b = _pair()
    c = contingency(a, b)
    assert c.n_scored + c.n_ignored == a.size


def test_the_table_is_the_same_by_either_counting_path(monkeypatch):
    """The dense-histogram and sorted-key paths must agree exactly.

    They are chosen on size, so a divergence would show up only on large inputs — which is
    where nobody checks a table by hand.
    """
    a, b = _pair()
    dense = contingency(a, b)
    monkeypatch.setattr(overlap, "DENSE_CELL_LIMIT", 0)
    sparse = contingency(a, b)
    assert dense == sparse


def test_a_single_label_pair_is_one_row():
    a = np.full((4, 4, 4), 3, dtype=np.uint64)
    b = np.full((4, 4, 4), 9, dtype=np.uint64)
    c = contingency(a, b)
    assert c.n_pairs == 1
    assert int(c.a_ids[0]) == 3 and int(c.b_ids[0]) == 9
    assert c.n_scored == 64


def test_an_all_ignored_region_gives_an_empty_table_rather_than_raising():
    a = np.zeros((3, 3, 3), dtype=np.uint64)
    b = np.arange(27, dtype=np.uint64).reshape(3, 3, 3)
    c = contingency(a, b)
    assert c.n_pairs == 0
    assert c.n_scored == 0
    assert c.n_ignored == 27


# -- additivity, the master test ------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adding_block_tables_equals_the_whole_volume_table(axis, seed):
    a, b = _pair(seed=seed)
    whole = contingency(a, b)
    cut = a.shape[axis] // 2
    lo = (slice(None),) * axis + (slice(None, cut),)
    hi = (slice(None),) * axis + (slice(cut, None),)
    assert contingency(a[lo], b[lo]) + contingency(a[hi], b[hi]) == whole


def test_additivity_holds_across_many_blocks_and_through_sum():
    a, b = _pair(shape=(16, 8, 8))
    whole = contingency(a, b)
    blocks = [
        contingency(a[z:z + 4, y:y + 4, :], b[z:z + 4, y:y + 4, :])
        for z in range(0, 16, 4) for y in range(0, 8, 4)
    ]
    assert len(blocks) == 8
    # `sum` starts from the integer 0, which is how a driver-side reduction gets written.
    assert sum(blocks) == whole


def test_addition_is_order_independent():
    a, b = _pair(shape=(9, 6, 6))
    parts = [contingency(a[i::3], b[i::3]) for i in range(3)]
    assert parts[0] + parts[1] + parts[2] == parts[2] + parts[0] + parts[1]


def test_addition_refuses_tables_that_name_their_sides_differently():
    a, b = _pair(shape=(6, 6, 6))
    left = contingency(a, b, labels=("gt", "seg"))
    right = contingency(a, b, labels=("seg", "gt"))
    with pytest.raises(ValueError, match="different sides"):
        left + right


def test_addition_refuses_tables_that_dropped_different_labels():
    a, b = _pair(shape=(6, 6, 6))
    with pytest.raises(ValueError, match="dropped different labels"):
        contingency(a, b) + contingency(a, b, ignore_a=())


# -- ignore and mask ------------------------------------------------------------------

def test_ignored_labels_leave_the_count_and_are_tallied():
    a = np.array([[[1, 1, 0, 0]]], dtype=np.uint64)
    b = np.array([[[5, 5, 5, 5]]], dtype=np.uint64)
    c = contingency(a, b)
    assert c.n_pairs == 1
    assert c.n_scored == 2
    assert c.n_ignored == 2


def test_the_ignore_default_is_asymmetric():
    """0 on side a is unannotated; 0 on side b is a body the segmentation failed to assign."""
    a = np.array([[[1, 1]]], dtype=np.uint64)
    b = np.array([[[0, 7]]], dtype=np.uint64)
    c = contingency(a, b)
    assert c.n_scored == 2
    assert set(int(v) for v in c.b_ids) == {0, 7}


def test_a_mask_restricts_the_scored_region():
    a = np.ones((2, 2, 2), dtype=np.uint64)
    b = np.arange(8, dtype=np.uint64).reshape(2, 2, 2) + 1
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0] = True
    c = contingency(a, b, mask=mask)
    assert c.n_scored == 4
    assert c.n_ignored == 4


def test_ignoring_more_labels_can_only_shrink_the_scored_region():
    a, b = _pair()
    wide = contingency(a, b)
    narrow = contingency(a, b, ignore_a=(0, 1), ignore_b=(2,))
    assert narrow.n_scored < wide.n_scored
    assert narrow.n_scored + narrow.n_ignored == a.size


# -- transpose ------------------------------------------------------------------------

def test_transpose_swaps_the_sides_and_round_trips():
    a, b = _pair()
    c = contingency(a, b, labels=("gt", "seg"))
    t = c.T
    assert t.labels == ("seg", "gt")
    assert t.n_scored == c.n_scored
    assert t.T == c


def test_transpose_is_not_the_kernel_run_the_other_way_round():
    """Because the ignore defaults follow the ARGUMENT, not the table."""
    a = np.array([[[1, 1, 0, 0]]], dtype=np.uint64)
    b = np.array([[[0, 2, 2, 2]]], dtype=np.uint64)
    assert contingency(a, b).T != contingency(b, a)


# -- selection ------------------------------------------------------------------------

def test_restrict_keeps_only_the_named_labels_and_banks_the_rest_as_ignored():
    a, b = _pair()
    c = contingency(a, b)
    keep_a = [int(c.a_ids[0])]
    r = c.restrict(a_ids=keep_a)
    assert set(int(v) for v in r.a_ids) == set(keep_a)
    assert r.n_scored + r.n_ignored == c.n_scored + c.n_ignored


def test_keep_pairs_needs_one_flag_per_pair():
    a, b = _pair(shape=(5, 5, 5))
    c = contingency(a, b)
    with pytest.raises(ValueError, match="one flag per pair"):
        c.keep_pairs(np.ones(c.n_pairs + 1, dtype=bool))


# -- input guards ---------------------------------------------------------------------

def test_mismatched_shapes_are_refused_by_name():
    with pytest.raises(ValueError, match="same shape"):
        contingency(np.zeros((2, 2, 2), np.uint64), np.zeros((2, 2, 3), np.uint64))


def test_a_float_labeling_is_refused():
    with pytest.raises(TypeError, match="float label array"):
        contingency(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint64))


def test_negative_ids_are_refused_rather_than_wrapped():
    a = np.array([[-1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="negative values"):
        contingency(a, np.ones((1, 2), np.uint64))


def test_labels_of_different_integer_widths_still_compare():
    a = np.array([[1, 2]], dtype=np.uint16)
    b = np.array([[1, 2]], dtype=np.int64)
    assert contingency(a, b).n_pairs == 2


# -- construction guards --------------------------------------------------------------

def test_parallel_arrays_are_required():
    with pytest.raises(ValueError, match="parallel"):
        Contingency(a_ids=np.zeros(2, np.uint64), b_ids=np.zeros(3, np.uint64),
                    counts=np.zeros(2, np.uint64))


def test_repr_says_what_is_in_it():
    a, b = _pair(shape=(4, 4, 4))
    text = repr(contingency(a, b, labels=("gt", "seg")))
    assert "gt=" in text and "seg=" in text and "scored=" in text
