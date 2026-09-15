"""`compare` end to end on synthetic pieces, plus the registration and scatter guards."""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval import compare as compare_mod
from neu_eval.compare import compare
from neu_eval.overlap import contingency, pool


def _piece(arr, *, voxel_size=(8.0, 8.0, 8.0), origin=(0.0, 0.0, 0.0)):
    from neu_lib import Frame, Piece

    return Piece(np.asarray(arr, dtype=np.uint64),
                 Frame(voxel_size_nm=voxel_size, origin_nm=origin), kind="segmentation")


def _split_pair(shape=(8, 8, 8)):
    """One reference body, cut in half by the segmentation."""
    a = np.ones(shape, dtype=np.uint64)
    b = np.full(shape, 5, dtype=np.uint64)
    b[shape[0] // 2:] = 6
    return _piece(a), _piece(b)


# -- the happy path ---------------------------------------------------------------------

def test_compare_produces_metrics_rows_and_a_headline():
    a, b = _split_pair()
    report = compare(a, b, ignore_a=())
    assert report.summary["voi_split"] > 0
    assert report.summary["voi_merge"] == pytest.approx(0.0)
    assert len(report.rows) == 2
    assert report.kind_counts["split"] == 2
    assert "VOI" in report.headline()


def test_every_row_carries_a_world_coordinate_by_default():
    a, b = _split_pair()
    for row in compare(a, b, ignore_a=()).rows:
        assert {"z_nm", "y_nm", "x_nm", "z_vox"} <= set(row)


def test_locate_can_be_turned_off():
    a, b = _split_pair()
    rows = compare(a, b, ignore_a=(), locate=False).rows
    assert rows and "z_nm" not in rows[0]


def test_the_summary_is_one_flat_numeric_row():
    a, b = _split_pair()
    row = compare(a, b, ignore_a=(), locate=False).summary
    assert all(isinstance(v, (int, float)) for v in row.values())
    assert {"voi", "rand_error", "covering", "fragmentation", "frac_scored"} <= set(row)


def test_identical_labelings_score_perfectly_and_report_nothing():
    a = _piece(np.arange(64, dtype=np.uint64).reshape(4, 4, 4) + 1)
    report = compare(a, a, ignore_a=())
    assert report.summary["voi"] == pytest.approx(0.0)
    assert report.rows == []


def test_the_side_names_reach_every_output():
    a, b = _split_pair()
    report = compare(a, b, ignore_a=(), labels=("truth", "auto"))
    assert report.labels == ("truth", "auto")
    assert "n_truth" in report.summary and "n_auto" in report.summary
    assert "truth_id" in report.rows[0]
    assert "truth vs auto" in report.headline()


# -- adjudication through compare -------------------------------------------------------

def test_verdicts_add_a_second_score_without_replacing_the_first():
    a, b = _split_pair()
    raw = compare(a, b, ignore_a=(), locate=False)
    key = raw.rows[0]["pair_key"]
    adjudicated = compare(a, b, ignore_a=(), locate=False,
                          verdicts={key: "seg_correct"})
    assert adjudicated.summary == raw.summary, "the raw score is untouched"
    assert adjudicated.adjudicated_summary["frac_scored"] < raw.summary["frac_scored"]
    assert adjudicated.verdict_tally["seg_correct"] == 1
    assert "adjudicated" in adjudicated.headline()


def test_no_verdicts_means_no_adjudicated_fields():
    a, b = _split_pair()
    report = compare(a, b, ignore_a=(), locate=False)
    assert report.adjudicated is None
    assert report.adjudicated_summary is None
    assert "adjudicated" not in report.headline()


def test_verdicts_that_match_nothing_say_so_in_the_headline():
    """Because "changed nothing" and "matched nothing" are the same numbers.

    A verdict file outlives the table it came from, so it can be full of perfectly valid
    verdicts that name no pair here.
    """
    a, b = _split_pair()
    report = compare(a, b, ignore_a=(), locate=False,
                     verdicts={"999:888": "seg_correct"})
    assert report.verdict_tally["excluded_pairs"] == 0
    assert "nothing was excluded" in report.headline()


def test_a_verdict_that_does_match_reports_the_pair_it_removed():
    a, b = _split_pair()
    raw = compare(a, b, ignore_a=(), locate=False)
    report = compare(a, b, ignore_a=(), locate=False,
                     verdicts={raw.rows[0]["pair_key"]: "seg_correct"})
    assert report.verdict_tally["excluded_pairs"] == 1
    assert "nothing was excluded" not in report.headline()


# -- registration guard -----------------------------------------------------------------

def test_mismatched_shapes_are_refused_with_the_fix_in_the_message():
    a = _piece(np.ones((4, 4, 4), np.uint64))
    b = _piece(np.ones((4, 4, 5), np.uint64))
    with pytest.raises(ValueError, match="read_piece"):
        compare(a, b)


def test_two_pieces_in_different_places_are_refused():
    """Same shape, different physical box — comparing them measures the offset."""
    a = _piece(np.ones((4, 4, 4), np.uint64), origin=(0.0, 0.0, 0.0))
    b = _piece(np.ones((4, 4, 4), np.uint64), origin=(1000.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="different physical boxes"):
        compare(a, b)


def test_sub_voxel_drift_is_tolerated():
    """read_piece grows a physical box outward to whole voxels, so two levels of one
    volume legitimately differ by less than a voxel."""
    a = _piece(np.ones((4, 4, 4), np.uint64), origin=(0.0, 0.0, 0.0))
    b = _piece(np.ones((4, 4, 4), np.uint64), origin=(4.0, 0.0, 0.0))
    compare(a, b, ignore_a=())


def test_the_frame_check_can_be_waived():
    a = _piece(np.ones((4, 4, 4), np.uint64), origin=(0.0, 0.0, 0.0))
    b = _piece(np.ones((4, 4, 4), np.uint64), origin=(1000.0, 0.0, 0.0))
    compare(a, b, ignore_a=(), check_frames=False)


# -- the scatter warning ----------------------------------------------------------------

def test_a_scattered_reference_label_is_reported_not_raised():
    """What an un-relabelled multi-crop reference looks like: one id in two far corners."""
    arr = np.zeros((80, 80, 80), dtype=np.uint64)
    arr[:4, :4, :4] = 7
    arr[76:, 76:, 76:] = 7          # same id, opposite corner: 128 voxels, 80^3 box
    arr[20:60, 20:60, 20:60] = 8    # a normal compact body
    a = _piece(arr)
    b = _piece(np.where(arr > 0, 5, 0).astype(np.uint64))

    report = compare(a, b, locate=False)
    assert any("scattered" in w for w in report.warnings)
    assert any("neu-vol relabel" in w for w in report.warnings)


def test_a_compact_reference_produces_no_scatter_warning():
    arr = np.zeros((80, 80, 80), dtype=np.uint64)
    arr[20:60, 20:60, 20:60] = 8
    a = _piece(arr)
    b = _piece(np.where(arr > 0, 5, 0).astype(np.uint64))
    assert compare(a, b, locate=False).warnings == []


def test_the_scatter_check_can_be_turned_off():
    arr = np.zeros((80, 80, 80), dtype=np.uint64)
    arr[:4, :4, :4] = 7
    arr[76:, 76:, 76:] = 7
    a = _piece(arr)
    b = _piece(np.where(arr > 0, 5, 0).astype(np.uint64))
    assert compare(a, b, locate=False, check_scattered=False).warnings == []


def test_tiny_labels_are_never_flagged_as_scattered():
    """A two-voxel label in opposite corners has a huge ratio and tells you nothing."""
    arr = np.zeros((80, 80, 80), dtype=np.uint64)
    arr[0, 0, 0] = 7
    arr[79, 79, 79] = 7
    arr[20:60, 20:60, 20:60] = 8
    a = _piece(arr)
    b = _piece(np.where(arr > 0, 5, 0).astype(np.uint64))
    assert compare(a, b, locate=False).warnings == []


def test_a_non_overlapping_pair_is_reported_rather_than_scored_silently():
    a = _piece(np.zeros((4, 4, 4), np.uint64))
    b = _piece(np.full((4, 4, 4), 5, np.uint64))
    report = compare(a, b, locate=False)
    assert any("no labels co-occur" in w for w in report.warnings)


# -- pooling -----------------------------------------------------------------------------

def test_pooling_keeps_separate_crops_separate():
    """The whole point: two crops that both number their bodies from 1.

    Added directly, id 1 in crop A and id 1 in crop B would fuse into one "body" — the
    chimera bug in table form, producing a plausible pooled number from a partition that
    does not exist. Pooled, each crop's labels get their own range.
    """
    one = contingency(
        np.asarray([1, 1, 2, 2], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 5, 6, 6], np.uint64).reshape(1, 1, -1), ignore_a=())
    two = contingency(
        np.asarray([1, 1, 2, 2], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 5, 6, 6], np.uint64).reshape(1, 1, -1), ignore_a=())

    summed = one + two
    pooled = pool([one, two])
    assert summed.n_a == 2, "addition treats the shared ids as one body"
    assert pooled.n_a == 4, "pooling gives each crop its own labels"
    assert pooled.n_scored == summed.n_scored == 8


def test_a_perfect_pair_of_crops_pools_to_a_perfect_score():
    from neu_eval.voxel import voi

    crop = contingency(
        np.asarray([1, 1, 2, 2], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 5, 6, 6], np.uint64).reshape(1, 1, -1), ignore_a=())
    assert voi(pool([crop, crop])).total == pytest.approx(0.0)


def test_pooling_sums_what_was_ignored():
    crop = contingency(
        np.asarray([1, 1, 0, 0], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 5, 6, 6], np.uint64).reshape(1, 1, -1))
    pooled = pool([crop, crop, crop])
    assert pooled.n_ignored == 3 * crop.n_ignored
    assert pooled.n_scored == 3 * crop.n_scored


def test_pooling_refuses_tables_that_disagree_about_their_sides():
    left = contingency(np.ones((1, 1, 2), np.uint64), np.ones((1, 1, 2), np.uint64),
                       ignore_a=(), labels=("gt", "seg"))
    right = left.T
    with pytest.raises(ValueError, match="different sides"):
        pool([left, right])


def test_pooling_nothing_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        pool([])


def test_pooling_a_single_table_preserves_its_scalars():
    from neu_eval.voxel import voi

    crop = contingency(
        np.asarray([1, 1, 1, 2], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 6, 6, 6], np.uint64).reshape(1, 1, -1), ignore_a=())
    assert voi(pool([crop])).total == pytest.approx(voi(crop).total)


def test_pooling_empty_tables_gives_an_empty_table():
    empty = contingency(np.zeros((1, 1, 4), np.uint64), np.zeros((1, 1, 4), np.uint64))
    pooled = pool([empty, empty])
    assert pooled.n_pairs == 0
    assert pooled.n_ignored == 8


# -- summarize ---------------------------------------------------------------------------

def test_summarize_merges_both_metric_families():
    c = contingency(
        np.asarray([1, 1, 2, 2], np.uint64).reshape(1, 1, -1),
        np.asarray([5, 5, 6, 7], np.uint64).reshape(1, 1, -1), ignore_a=())
    row = compare_mod.summarize(c)
    assert "voi" in row and "covering" in row and "fragmentation_at90" in row
