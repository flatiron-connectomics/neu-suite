"""Matching, fragmentation, completeness and covering."""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval.assign import (
    HUNGARIAN_MAX_LABELS, best_match, completeness, covering, fragmentation, iou, summary,
)
from neu_eval.overlap import contingency


def _c(a, b, **kwargs):
    a = np.asarray(a, dtype=np.uint64).reshape(1, 1, -1)
    b = np.asarray(b, dtype=np.uint64).reshape(1, 1, -1)
    kwargs.setdefault("ignore_a", ())
    return contingency(a, b, **kwargs)


# -- iou -------------------------------------------------------------------------------

def test_iou_is_one_for_an_exact_match():
    c = _c([1, 1, 2, 2], [5, 5, 6, 6])
    assert np.allclose(iou(c), 1.0)


def test_iou_is_hand_computable_for_a_partial_overlap():
    # a=1 covers 3 voxels, b=5 covers 2, sharing 2 -> 2 / (3 + 2 - 2) = 2/3.
    c = _c([1, 1, 1], [5, 5, 6])
    values = dict(zip(zip(c.a_ids.tolist(), c.b_ids.tolist()), iou(c)))
    assert values[(1, 5)] == pytest.approx(2 / 3)
    assert values[(1, 6)] == pytest.approx(1 / 3)


def test_iou_measures_against_the_scored_region_only():
    """A body half outside a mask is measured against the half that was scored."""
    a = np.array([[[1, 1, 1, 1]]], dtype=np.uint64)
    b = np.array([[[5, 5, 9, 9]]], dtype=np.uint64)
    mask = np.array([[[True, True, False, False]]])
    c = contingency(a, b, ignore_a=(), mask=mask)
    assert np.allclose(iou(c), 1.0)


# -- matching ---------------------------------------------------------------------------

def test_a_clean_one_to_one_labeling_matches_completely():
    c = _c([1, 1, 2, 2, 3, 3], [5, 5, 6, 6, 7, 7])
    match = best_match(c)
    assert match.a_to_b == {1: 5, 2: 6, 3: 7}
    assert match.b_to_a == {5: 1, 6: 2, 7: 3}
    assert match.n_matched == 3


def test_a_merge_leaves_one_body_unmatched():
    """One segment cannot be two bodies, which is the point of the one-to-one rule."""
    c = _c([1, 1, 2, 2], [5, 5, 5, 5])
    match = best_match(c)
    assert match.n_matched == 1
    assert len(match.unmatched_a(c)) == 1
    assert match.unmatched_b(c) == []


def test_matching_by_iou_and_by_overlap_disagree_where_size_does():
    """A small tight body and a big sprawling one compete for the same segment.

    a=1 is 4 voxels, 3 of them in b=5. a=2 is 20 voxels scattered over four segments, 6 of
    them in b=5. By raw overlap b=5 belongs to a=2 (6 voxels beats 3), which strands a=1 on
    its 1-voxel second choice. By IoU it belongs to a=1 (0.30 beats 0.26), because 3 of 4
    voxels is a fit and 6 of 20 is a coincidence.
    """
    a = [1] * 4 + [2] * 20
    b = [5, 5, 5, 7] + [5] * 6 + [8] * 5 + [9] * 5 + [10] * 4
    c = _c(a, b)
    assert best_match(c, by="overlap").a_to_b == {1: 7, 2: 5}
    assert best_match(c, by="iou").a_to_b == {1: 5, 2: 8}


def test_min_iou_refuses_a_marginal_pair():
    # One segment covers both bodies: iou 0.25 with a=1 and 0.75 with a=2.
    c = _c([1] * 10 + [2] * 30, [5] * 40)
    assert best_match(c).a_to_b == {2: 5}
    assert best_match(c, min_iou=0.8).n_matched == 0


def test_the_greedy_order_does_not_depend_on_the_sort():
    """Ties are broken by id, so the same table always gives the same match."""
    c = _c([1, 2], [5, 6])
    assert best_match(c).a_to_b == best_match(c).a_to_b == {1: 5, 2: 6}


def test_the_optimal_assignment_can_beat_greedy():
    """Greedy takes the single best pair first and may strand a better total.

    a=1 overlaps b=5 by 5 and b=6 by 4; a=2 overlaps only b=5, by 4. Greedy takes (1,5)
    and leaves a=2 unmatched; the optimal assignment takes (1,6) and (2,5).
    """
    a = [1] * 9 + [2] * 4
    b = [5] * 5 + [6] * 4 + [5] * 4
    c = _c(a, b)
    greedy = best_match(c, by="overlap", method="greedy")
    optimal = best_match(c, by="overlap", method="hungarian")
    assert greedy.n_matched == 1
    assert optimal.n_matched == 2
    assert optimal.a_to_b == {1: 6, 2: 5}


def test_the_optimal_assignment_never_pairs_labels_that_share_no_voxels():
    c = _c([1, 2], [5, 6])
    match = best_match(c, method="hungarian")
    assert match.a_to_b == {1: 5, 2: 6}


def test_the_optimal_assignment_is_capped_rather_than_left_to_hang():
    n = HUNGARIAN_MAX_LABELS + 1
    c = _c(np.arange(1, n + 1), np.arange(1, n + 1))
    with pytest.raises(ValueError, match="capped"):
        best_match(c, method="hungarian")


def test_an_unknown_method_or_key_is_refused_by_name():
    c = _c([1], [5])
    with pytest.raises(ValueError, match="greedy"):
        best_match(c, method="optimal")
    with pytest.raises(ValueError, match="'iou' or 'overlap'"):
        best_match(c, by="dice")


def test_an_empty_table_matches_nothing():
    assert best_match(_c([], [])).n_matched == 0


# -- fragmentation and completeness -----------------------------------------------------

def test_fragmentation_counts_the_extra_segments():
    assert fragmentation(_c([1, 1, 1, 1], [5, 5, 6, 6])) == 1
    assert fragmentation(_c([1, 1, 2, 2], [5, 5, 5, 5])) == -1
    assert fragmentation(_c([1, 1, 2, 2], [5, 5, 6, 6])) == 0


def test_completeness_takes_the_largest_labels_first():
    # a: body 1 = 6 voxels, body 2 = 3, body 3 = 1.
    a = [1] * 6 + [2] * 3 + [3]
    b = list(range(10, 20))
    c = _c(a, b)
    assert completeness(c, 0.5).n_a_kept == 1      # 6 >= 5
    assert completeness(c, 0.9).n_a_kept == 2      # 6 + 3 >= 9
    assert completeness(c, 1.0).n_a_kept == 3


def test_completeness_thresholds_the_two_sides_independently():
    a = [1] * 8 + [2] * 2
    b = [5] * 10
    c = _c(a, b)
    comp = completeness(c, 0.9)
    assert comp.n_a_kept == 2 and comp.n_b_kept == 1
    assert comp.fragmentation == -1


def test_a_threshold_outside_the_unit_interval_is_refused():
    c = _c([1], [5])
    for bad in (0.0, 1.5, -1):
        with pytest.raises(ValueError, match="at must be"):
            completeness(c, bad)


# -- covering ---------------------------------------------------------------------------

def test_covering_is_one_for_a_perfect_labeling():
    assert covering(_c([1, 1, 2, 2], [5, 5, 6, 6])) == pytest.approx(1.0)


def test_covering_is_weighted_by_body_size():
    """A big body captured perfectly outweighs a small one that was not."""
    a = [1] * 9 + [2]
    b = [5] * 9 + [5]
    c = _c(a, b)
    # body 1: iou 9/10. body 2: iou 1/10. weights 9 and 1.
    assert covering(c) == pytest.approx((9 * 0.9 + 1 * 0.1) / 10)


def test_covering_is_zero_on_an_empty_table():
    assert covering(_c([], [])) == 0.0


# -- summary ----------------------------------------------------------------------------

def test_summary_carries_a_threshold_column_per_requested_level():
    c = _c([1, 1, 2, 2], [5, 5, 6, 7], labels=("gt", "seg"))
    row = summary(c, at=(0.5, 0.9))
    assert {"fragmentation_at50", "fragmentation_at90", "n_gt_at90", "n_seg_at90"} <= set(row)
    assert "covering" in row and "frac_gt_matched_iou50" in row


def test_summary_is_flat_and_numeric():
    c = _c([1, 1, 2], [5, 6, 6])
    assert all(isinstance(v, (int, float)) for v in summary(c).values())
