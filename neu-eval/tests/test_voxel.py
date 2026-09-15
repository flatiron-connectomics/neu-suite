"""VOI and adapted Rand, including the two tests that pin which half is which.

Every number asserted here is hand-computable. The point of doing it that way rather than
against a reference implementation is that a reference implementation shares the one
mistake this module is most likely to make — getting the direction backwards — and would
agree with us enthusiastically about it.
"""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval.overlap import contingency
from neu_eval.voxel import Rand, Voi, pair_voi, per_label_voi, rand, summary, voi


def _c(a, b, **kwargs):
    """A table over two flat label lists, with nothing ignored."""
    a = np.asarray(a, dtype=np.uint64).reshape(1, 1, -1)
    b = np.asarray(b, dtype=np.uint64).reshape(1, 1, -1)
    kwargs.setdefault("ignore_a", ())
    return contingency(a, b, **kwargs)


# -- the direction --------------------------------------------------------------------

def test_a_pure_split_is_all_split_and_no_merge():
    """One reference body cut in two: H(b|a) = 1 bit, H(a|b) = 0."""
    c = _c([1, 1, 1, 1], [1, 1, 2, 2])
    result = voi(c)
    assert result.split == pytest.approx(1.0)
    assert result.merge == pytest.approx(0.0)


def test_a_pure_merge_is_all_merge_and_no_split():
    """Two reference bodies fused into one: H(a|b) = 1 bit, H(b|a) = 0."""
    c = _c([1, 1, 2, 2], [1, 1, 1, 1])
    result = voi(c)
    assert result.merge == pytest.approx(1.0)
    assert result.split == pytest.approx(0.0)


def test_split_and_merge_swap_under_transpose():
    c = _c([1, 1, 1, 1], [1, 1, 2, 2])
    forward, backward = voi(c), voi(c.T)
    assert forward.split == pytest.approx(backward.merge)
    assert forward.merge == pytest.approx(backward.split)


def test_a_simultaneous_split_and_merge_scores_both():
    c = _c([1, 1, 2, 2], [1, 2, 1, 2])
    result = voi(c)
    assert result.split == pytest.approx(1.0)
    assert result.merge == pytest.approx(1.0)
    assert result.total == pytest.approx(2.0)


# -- identity and invariance ----------------------------------------------------------

def test_perfect_agreement_scores_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 9, size=(6, 6, 6), dtype=np.uint64)
    c = contingency(a, a, ignore_a=())
    assert voi(c).total == pytest.approx(0.0)
    assert rand(c).f_score == pytest.approx(1.0)
    assert rand(c).error == pytest.approx(0.0)


def test_renaming_either_side_changes_nothing():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 7, size=(5, 5, 5), dtype=np.uint64)
    b = rng.integers(1, 9, size=(5, 5, 5), dtype=np.uint64)
    base = summary(contingency(a, b, ignore_a=()))

    # A bijection on each side's ids, which no metric may notice.
    a2 = (a * np.uint64(1000) + np.uint64(7))
    b2 = (b * np.uint64(31) + np.uint64(2))
    renamed = summary(contingency(a2, b2, ignore_a=()))
    for key, value in base.items():
        assert renamed[key] == pytest.approx(value), key


def test_adding_ignored_voxels_moves_no_metric():
    a = np.array([[[1, 1, 2, 2]]], dtype=np.uint64)
    b = np.array([[[1, 1, 3, 3]]], dtype=np.uint64)
    padded_a = np.concatenate([a, np.zeros((1, 1, 6), np.uint64)], axis=2)
    padded_b = np.concatenate([b, np.full((1, 1, 6), 9, np.uint64)], axis=2)

    before, after = summary(contingency(a, b)), summary(contingency(padded_a, padded_b))
    for key in ("voi", "voi_split", "voi_merge", "rand_f_score"):
        assert after[key] == pytest.approx(before[key]), key
    assert after["n_ignored"] == before["n_ignored"] + 6
    assert after["frac_scored"] < before["frac_scored"]


def test_an_empty_table_scores_zero_rather_than_nan():
    c = _c([], [])
    assert voi(c).total == 0.0
    assert rand(c).f_score == 1.0
    assert summary(c)["frac_scored"] == 0.0


# -- adapted Rand ---------------------------------------------------------------------

def test_rand_on_a_pure_split_is_hand_computable():
    """b refines a: every pair b claims is right, and it keeps a third of a's pairs.

    agree = 2 pairs, in_b = 2, in_a = 6 -> precision 1, recall 1/3, F 1/2.
    """
    result = rand(_c([1, 1, 1, 1], [1, 1, 2, 2]))
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1 / 3)
    assert result.f_score == pytest.approx(0.5)
    assert result.error == pytest.approx(0.5)


def test_rand_on_a_pure_merge_swaps_precision_and_recall():
    result = rand(_c([1, 1, 2, 2], [1, 1, 1, 1]))
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == pytest.approx(1.0)


def test_a_group_of_one_contributes_no_pairs():
    """Singletons must not create pairs they do not have."""
    result = rand(_c([1, 2, 3], [4, 5, 6]))
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


# -- decomposition ---------------------------------------------------------------------

def test_the_per_label_decomposition_sums_to_the_total():
    rng = np.random.default_rng(2)
    a = rng.integers(1, 6, size=(7, 7, 7), dtype=np.uint64)
    b = rng.integers(1, 8, size=(7, 7, 7), dtype=np.uint64)
    c = contingency(a, b, ignore_a=())
    merge_by_a, split_by_b = per_label_voi(c)
    result = voi(c)
    assert sum(merge_by_a.values()) == pytest.approx(result.merge)
    assert sum(split_by_b.values()) == pytest.approx(result.split)


def test_the_per_pair_decomposition_sums_to_the_total():
    rng = np.random.default_rng(3)
    a = rng.integers(1, 5, size=(6, 6, 6), dtype=np.uint64)
    b = rng.integers(1, 7, size=(6, 6, 6), dtype=np.uint64)
    c = contingency(a, b, ignore_a=())
    merge_terms, split_terms = pair_voi(c)
    result = voi(c)
    assert merge_terms.sum() == pytest.approx(result.merge)
    assert split_terms.sum() == pytest.approx(result.split)
    assert merge_terms.shape == (c.n_pairs,)


def test_the_body_blamed_for_a_merge_is_the_one_that_got_merged():
    # a bodies 1 and 2 are fused by b's single segment 5; body 3 is left alone.
    c = _c([1, 1, 2, 2, 3, 3], [5, 5, 5, 5, 6, 6])
    merge_by_a, _ = per_label_voi(c)
    assert merge_by_a[1] > 0
    assert merge_by_a[2] > 0
    assert merge_by_a[3] == pytest.approx(0.0)


def test_the_segment_blamed_for_a_split_is_the_one_that_split():
    c = _c([1, 1, 1, 1, 2, 2], [5, 5, 6, 6, 7, 7])
    _, split_by_b = per_label_voi(c)
    assert split_by_b[5] > 0
    assert split_by_b[6] > 0
    assert split_by_b[7] == pytest.approx(0.0)


# -- summary ---------------------------------------------------------------------------

def test_summary_names_its_columns_after_the_sides():
    c = _c([1, 1, 2], [1, 2, 2], labels=("gt", "seg"))
    row = summary(c)
    assert "n_gt" in row and "n_seg" in row
    assert row["n_scored"] == 3


def test_summary_is_flat_and_json_able():
    c = _c([1, 1, 2], [1, 2, 2])
    row = summary(c)
    assert all(isinstance(v, (int, float)) for v in row.values())


def test_voi_and_rand_dataclasses_expose_their_derived_fields():
    assert Voi(0.25, 0.75).total == pytest.approx(1.0)
    assert Rand(1.0, 1.0).error == pytest.approx(0.0)
    assert Rand(0.0, 0.0).f_score == 0.0
