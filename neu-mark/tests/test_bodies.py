"""Reading the body list, the required input to every command."""

import pandas as pd
import pytest

from neu_mark import bodies


@pytest.mark.parametrize("given, expected", [
    ("3,1,2", [1, 2, 3]),
    ("3 1 2", [1, 2, 3]),
    ("10", [10]),
    ("5,5,1,5", [1, 5]),           # sorted and de-duplicated
    ([9, 8, 8], [8, 9]),
])
def test_inline_lists_and_iterables(given, expected):
    assert bodies.load(given) == expected


def test_text_file_one_per_line_with_comments(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# traced neurons\n13481220\n18126428  # MT(L)\n\n77263210\n")
    assert bodies.load(str(p)) == [13481220, 18126428, 77263210]


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"name": ["a", "b"], "body": [7, 3]}),     # a recognisably named column
    pd.DataFrame({"whatever": [7, 3]}),                     # a single column needs no name
])
def test_csv_columns(tmp_path, frame):
    p = tmp_path / "b.csv"
    frame.to_csv(p, index=False)
    assert bodies.load(str(p)) == [3, 7]


def test_parquet_round_trips_uint64(tmp_path):
    p = tmp_path / "b.parquet"
    pd.DataFrame({"body": [2**63 + 7, 3]}, dtype="uint64").to_parquet(p)
    assert bodies.load(str(p)) == [3, 2**63 + 7]


def test_this_packages_own_bodies_table_is_a_valid_input(tmp_path):
    from neu_mark import tables

    df = tables.keyvalues_to_frame({"5": {"bodyid": 5}, "9": {"bodyid": 9}})
    p = tmp_path / "bodies.parquet"
    df.drop(columns=["json"]).to_parquet(p)
    assert bodies.load(str(p)) == [5, 9]


def test_an_ambiguous_or_wrong_column_names_what_to_do(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="--body-column"):
        bodies.load(str(p))
    assert bodies.load(str(p), column="b") == [2]
    with pytest.raises(ValueError, match="no column 'zzz'"):
        bodies.load(str(p), column="zzz")


def test_a_non_numeric_token_names_the_line(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("1\nnot-a-body\n")
    with pytest.raises(ValueError, match="line 2"):
        bodies.load(str(p))


def test_a_missing_file_that_looks_like_a_path_is_not_read_as_ids():
    with pytest.raises(FileNotFoundError):
        bodies.load("/no/such/file.csv")


def test_summarise_is_one_line():
    assert bodies.summarise([]) == "0 bodies"
    assert bodies.summarise([7]) == "1 body (7)"
    assert bodies.summarise([3, 9]) == "2 bodies (3..9)"
