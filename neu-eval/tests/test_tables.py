"""Writers and the csv round trip. The parquet half skips without the `report` extra."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from neu_eval import disagree, tables
from neu_eval.overlap import contingency


def _c(a, b, labels=("gt", "seg")):
    a = np.asarray(a, dtype=np.uint64).reshape(1, 1, -1)
    b = np.asarray(b, dtype=np.uint64).reshape(1, 1, -1)
    return contingency(a, b, ignore_a=(), labels=labels)


def _rows():
    return disagree.rows(_c([1] * 10, [5] * 5 + [6] * 5))


# -- the csv, which needs no optional dependency -----------------------------------------

def test_the_disagreement_csv_needs_no_pandas(tmp_path, monkeypatch):
    """The one file a reviewer edits must not require an optional extra to produce."""
    monkeypatch.setitem(__import__("sys").modules, "pandas", None)
    path = tables.write_disagreements(_rows(), tmp_path / "d.csv")
    assert csv.DictReader(open(path)).fieldnames is not None


def test_the_csv_has_a_blank_verdict_and_note_column(tmp_path):
    path = tables.write_disagreements(_rows(), tmp_path / "d.csv")
    rows = list(csv.DictReader(open(path)))
    assert all(row["verdict"] == "" for row in rows)
    assert all(row["note"] == "" for row in rows)


def test_ids_are_written_as_text(tmp_path):
    """A spreadsheet parses a long integer as float64 and hands back a rounded id.

    So the ids go out as text and `pair_key` is the join key — a reviewer who reformats a
    column still cannot break the round trip.
    """
    huge = 2**63 + 12345
    rows = [{"pair_key": f"{huge}:5", "kind": "merge", "gt_id": huge, "seg_id": 5,
             "n_voxels": 10, "severity": 0.5}]
    path = tables.write_disagreements(rows, tmp_path / "d.csv")
    text = open(path).read()
    assert str(huge) in text
    back, verdicts = tables.read_disagreements(path)
    assert back[0]["gt_id"] == str(huge)
    assert f"{huge}:5" in verdicts


def test_the_round_trip_recovers_the_verdicts(tmp_path):
    path = tables.write_disagreements(_rows(), tmp_path / "d.csv")
    rows = list(csv.DictReader(open(path)))
    rows[0]["verdict"] = "seg_correct"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _, verdicts = tables.read_disagreements(path)
    assert verdicts[rows[0]["pair_key"]] == "seg_correct"
    assert all(v == "" for k, v in verdicts.items() if k != rows[0]["pair_key"])


def test_rewriting_preserves_a_review_already_done(tmp_path):
    """Re-running a comparison must not blank an afternoon of work."""
    path = tmp_path / "d.csv"
    rows = _rows()
    verdicts = {rows[0]["pair_key"]: "gt_correct"}
    tables.write_disagreements(rows, path, verdicts=verdicts)
    _, back = tables.read_disagreements(path)
    assert back[rows[0]["pair_key"]] == "gt_correct"


def test_a_legend_is_written_beside_the_csv(tmp_path):
    path = tables.write_disagreements(_rows(), tmp_path / "d.csv", labels=("v1", "v2"))
    legend = (tmp_path / "d.README.txt").read_text()
    assert "v2_correct" in legend and "v1_correct" in legend
    assert "TEXT" in legend, "the spreadsheet hazard is stated where a reviewer will see it"
    del path


def test_a_file_without_a_verdict_column_is_refused_by_name(tmp_path):
    path = tmp_path / "not-adjudication.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="verdict"):
        tables.read_disagreements(path)


def test_an_empty_row_set_still_writes_a_usable_header(tmp_path):
    path = tables.write_disagreements([], tmp_path / "d.csv", labels=("gt", "seg"))
    assert "verdict" in (csv.DictReader(open(path)).fieldnames or [])


# -- the annotation point list -----------------------------------------------------------

def test_the_annotation_csv_is_three_coordinates_and_a_description(tmp_path):
    a = np.ones((8, 8, 8), dtype=np.uint64)
    b = np.full((8, 8, 8), 5, dtype=np.uint64)
    b[4:] = 6
    from neu_lib import Frame, Piece

    frame = Frame(voxel_size_nm=(8.0, 8.0, 8.0))
    rows = disagree.locate(disagree.rows(contingency(a, b, ignore_a=())),
                           Piece(a, frame), Piece(b, frame))
    path = tables.annotation_csv(rows, tmp_path / "ann.csv")
    got = list(csv.DictReader(open(path)))
    assert list(got[0]) == ["z_nm", "y_nm", "x_nm", "description"]
    assert "split" in got[0]["description"]


def test_unlocated_rows_are_skipped_rather_than_written_as_zeros(tmp_path):
    path = tables.annotation_csv(_rows(), tmp_path / "ann.csv")
    assert list(csv.DictReader(open(path))) == []


# -- parquet, behind the extra -----------------------------------------------------------

def test_the_summary_parquet_round_trips(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from neu_eval.compare import summarize

    row = summarize(_c([1] * 10, [5] * 5 + [6] * 5))
    path = tables.write_summary(row, tmp_path / "s.parquet")
    back = pd.read_parquet(path)
    assert len(back) == 1
    assert back["voi"].iloc[0] == pytest.approx(row["voi"])


def test_many_summary_rows_go_in_one_file(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from neu_eval.compare import summarize

    c = _c([1] * 10, [5] * 5 + [6] * 5)
    path = tables.write_summary([{"piece": "a", **summarize(c)},
                                 {"piece": "b", **summarize(c)}], tmp_path / "s.parquet")
    assert list(pd.read_parquet(path)["piece"]) == ["a", "b"]


def test_the_per_label_frame_is_long_with_a_side_column(tmp_path):
    pytest.importorskip("pandas")

    c = _c([1] * 10, [5] * 5 + [6] * 5)
    frame = tables.per_label_frame(c)
    assert set(frame["side"]) == {"gt", "seg"}
    assert set(frame.columns) == {"side", "label_id", "n_voxels", "voi_contribution"}
    # The gt body took all the split blame; the two segments share the merge blame of zero.
    assert frame[frame.side == "seg"]["voi_contribution"].sum() > 0
