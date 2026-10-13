"""The verdict round-trip: what gets excluded, what deliberately does not, and the guards."""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval import adjudicate
from neu_eval.overlap import contingency
from neu_eval.voxel import summary


def _c(a, b, labels=("gt", "seg")):
    a = np.asarray(a, dtype=np.uint64).reshape(1, 1, -1)
    b = np.asarray(b, dtype=np.uint64).reshape(1, 1, -1)
    return contingency(a, b, ignore_a=(), labels=labels)


# -- what the vocabulary is -------------------------------------------------------------

def test_the_verdicts_follow_the_side_names():
    assert adjudicate.verdict_names(("gt", "seg"))[:2] == ("gt_correct", "seg_correct")
    assert adjudicate.verdict_names(("v1", "v2"))[:2] == ("v1_correct", "v2_correct")


def test_the_default_exclusion_follows_the_side_names_too():
    """The bug this caught: a constant `seg_correct` matched nothing under other names.

    With --labels v1,v2 the reviewer writes `v2_correct`, no verdict matched, and the
    adjudicated score came back exactly equal to the raw one with nothing to say why.
    """
    assert adjudicate.default_exclude(("gt", "seg")) == ("seg_correct",)
    assert adjudicate.default_exclude(("v1", "v2")) == ("v2_correct",)


def test_nothing_says_right_anywhere_in_the_vocabulary():
    """`right` reads as a tuple position; the verdicts say `correct`."""
    for name in adjudicate.verdict_names(("gt", "seg")):
        assert "right" not in name


# -- exclusion --------------------------------------------------------------------------

def test_excluding_a_pair_moves_its_voxels_out_of_the_denominator():
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    before = summary(c)
    after = summary(adjudicate.apply(c, {"1:5": "seg_correct"}))
    assert after["n_scored"] == before["n_scored"] - 5
    assert after["n_ignored"] == before["n_ignored"] + 5
    assert after["frac_scored"] < before["frac_scored"]


def test_the_region_total_is_conserved():
    """`n_scored + n_ignored` still totals the box, however much is adjudicated away.

    Which is what lets a report state the price of the correction instead of quietly
    shrinking the thing the score is about.
    """
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    total = c.n_scored + c.n_ignored
    after = adjudicate.apply(c, {"1:5": "seg_correct", "2:5": "seg_correct"})
    assert after.n_scored + after.n_ignored == total


def test_only_the_reference_is_wrong_verdict_excludes_by_default():
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    for verdict in ("gt_correct", "ambiguous", "skip", ""):
        assert adjudicate.apply(c, {"1:5": verdict}) == c, verdict
    assert adjudicate.apply(c, {"1:5": "seg_correct"}) != c


def test_ambiguous_is_kept_on_purpose_and_can_be_excluded_explicitly():
    """"I could not tell" is not evidence the reference was wrong.

    Excluding it by default is how an adjudicated score drifts upward one judgement call at
    a time — so it takes an explicit `exclude=`.
    """
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    assert adjudicate.apply(c, {"1:5": "ambiguous"}) == c
    widened = adjudicate.apply(c, {"1:5": "ambiguous"}, exclude=("ambiguous",))
    assert widened.n_scored < c.n_scored


def test_a_verdict_for_a_pair_that_is_not_in_the_table_is_ignored():
    """A verdict file outlives the table it came from, and min_frac legitimately changes
    which pairs appear."""
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    assert adjudicate.apply(c, {"999:888": "seg_correct"}) == c


def test_no_verdicts_at_all_is_a_no_op():
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    assert adjudicate.apply(c, {}) is c


def test_excluding_everything_leaves_an_empty_but_valid_table():
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    empty = adjudicate.apply(c, {"1:5": "seg_correct", "2:5": "seg_correct"})
    assert empty.n_pairs == 0
    assert summary(empty)["frac_scored"] == 0.0


# -- guards -----------------------------------------------------------------------------

def test_an_unrecognised_verdict_is_refused_rather_than_ignored():
    """A typo would otherwise read as "not excluded", which is the direction that flatters."""
    c = _c([1] * 5 + [2] * 5, [5] * 10)
    with pytest.raises(ValueError, match="unrecognised verdict"):
        adjudicate.apply(c, {"1:5": "sg_corect"})


def test_the_refusal_names_what_was_allowed():
    c = _c([1] * 5 + [2] * 5, [5] * 10, labels=("v1", "v2"))
    with pytest.raises(ValueError, match="v2_correct"):
        adjudicate.apply(c, {"1:5": "seg_correct"})


def test_verdicts_are_checked_against_the_tables_own_side_names():
    c = _c([1] * 5 + [2] * 5, [5] * 10, labels=("v1", "v2"))
    assert adjudicate.apply(c, {"1:5": "v2_correct"}).n_scored < c.n_scored


# -- tally -------------------------------------------------------------------------------

def test_the_tally_counts_every_verdict_and_the_unreviewed():
    tally = adjudicate.summarize(
        {"1:5": "seg_correct", "2:5": "seg_correct", "3:5": "ambiguous", "4:5": ""},
        labels=("gt", "seg"))
    assert tally["seg_correct"] == 2
    assert tally["ambiguous"] == 1
    assert tally["unreviewed"] == 1
    assert tally["gt_correct"] == 0
