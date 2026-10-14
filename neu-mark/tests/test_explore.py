"""The notebook-facing inspection functions, on realistic dirty strings: a trailing '?'
expressing doubt, trailing whitespace, and a body with no instance at all."""

import pandas as pd
import pytest

from neu_mark import explore as ex

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "Tm(R)_truncated?",
                           "irrelevant ", "LN_C5(L)_NCL", "PGNG_CV", None],
                          dtype="string"),
    "type": pd.Series(["Tm2", None, "Tm", None, "LN", None, None], dtype="string"),
})


@pytest.mark.parametrize("raw, expected", [
    (" Tm(R)_truncated? ", "Tm(R)_truncated"),
    ("irrelevant ", "irrelevant"),
    (None, None),
    ("   ", None),
    (float("nan"), None),
])
def test_normalize_strips_whitespace_and_doubt_marks(raw, expected):
    assert ex.normalize(raw) == expected


def test_instances_counts_bodies_and_drops_missing():
    out = ex.instances(BODIES)
    assert list(out.columns) == ["instance", "bodies"]
    # 6 distinct after normalization; the None row is not counted
    assert int(out["bodies"].sum()) == 6
    assert "Tm(R)_truncated" in set(out["instance"])       # the '?' is gone


def test_tokens_trailing_finds_the_suffix_vocabulary():
    out = ex.tokens(BODIES).set_index("token")["bodies"].to_dict()
    assert out["fragment"] == 1 and out["truncated"] == 1
    assert out["NCL"] == 1 and out["CV"] == 1


def test_dropping_the_side_stops_it_burying_the_suffixes():
    """A parenthesized side is the trailing token of any string ending in one."""
    with_side = ex.tokens(BODIES).set_index("token")["bodies"].to_dict()
    assert with_side.get("L") == 1                       # 'Tm2_A2(L)' ends with (L)
    without = ex.tokens(BODIES, drop=r"\((L|R)\)").set_index("token")["bodies"].to_dict()
    assert "L" not in without and "R" not in without
    assert without["A2"] == 1                            # the real trailing token


def test_tokens_positions():
    lead = ex.tokens(BODIES, position="leading").set_index("token")["bodies"].to_dict()
    assert lead["VLNP"] == 1 and lead["PGNG"] == 1
    every = ex.tokens(BODIES, position="all").set_index("token")["bodies"].to_dict()
    assert every["Tm2"] == 1 and every["A2"] == 1        # both tokens of one string


def test_tokens_reports_distinct_strings_as_well_as_bodies():
    """A suffix on many distinct strings is vocabulary; on one string it is a name."""
    out = ex.tokens(BODIES).set_index("token")
    assert out.loc["fragment", "strings"] == 1


def test_an_unknown_position_is_refused():
    with pytest.raises(ValueError, match="trailing, leading or all"):
        ex.tokens(BODIES, position="sideways")


def test_a_missing_column_names_what_is_there():
    with pytest.raises(KeyError, match="no 'nope' column"):
        ex.instances(BODIES, "nope")


def test_variants_shows_what_normalization_repaired():
    """These vanish from every other view, so they get their own report."""
    out = ex.variants(BODIES)
    pairs = dict(zip(out["raw"], out["normalized"]))
    assert pairs["Tm(R)_truncated?"] == "Tm(R)_truncated"
    assert pairs["irrelevant "] == "irrelevant"
    # clean strings are absent
    assert "Tm2_A2(L)" not in pairs


def test_near_finds_a_misspelling_and_excludes_the_exact_token_unless_asked():
    frame = pd.DataFrame({"instance": ["X_fragmnet", "Y_fragment", "Z_fragment"]})
    out = ex.near(ex.tokens(frame), "fragment")
    assert out["token"].tolist() == ["fragmnet"] and out["bodies"].tolist() == [1]

    exact = pd.DataFrame({"instance": ["Y_fragment"]})
    assert ex.near(ex.tokens(exact), "fragment").empty
    assert not ex.near(ex.tokens(exact), "fragment", include_exact=True).empty


# --------------------------------------------------------------------------- #
# the rule contract
# --------------------------------------------------------------------------- #
def test_a_rule_may_return_a_scalar_a_sequence_or_none():
    rules = {
        "one": lambda r: "x",
        "many": lambda r: ["a", "b"],
        "none": lambda r: None,
        "empty": lambda r: [],
    }
    out = ex.apply_rules(BODIES, rules)
    assert out["one"].iloc[0] == ("x",)
    assert out["many"].iloc[0] == ("a", "b")
    assert out["none"].iloc[0] == () and out["empty"].iloc[0] == ()


def test_multi_valued_facets_are_first_class():
    out = ex.coverage(BODIES, {"col": lambda r: ["D2", "E3"]})
    assert out.loc[0, "multi_valued"] == len(BODIES)
    assert out.loc[0, "values"] == 2 * len(BODIES)


def test_missing_values_reach_a_rule_as_none_not_pd_na():
    """`pd.NA or ''` raises, so the obvious rule body would fail on a body with no instance."""
    seen = []

    def rule(r):
        seen.append(r["instance"])
        return str(r["instance"] or "")[:2] or None

    ex.apply_rules(BODIES, {"first_two": rule})
    assert seen[-1] is None
    assert not any(x is pd.NA for x in seen)


def test_a_rule_that_raises_says_it_should_have_returned_none():
    with pytest.raises(RuntimeError, match="should return None, not raise"):
        ex.apply_rules(BODIES, {"bad": lambda r: 1 / 0})


def test_apply_rules_index_cannot_misalign_with_a_reindexed_frame():
    odd = BODIES.set_index(pd.Index(range(100, 107), name="weird"))
    out = ex.apply_rules(odd, {"x": lambda r: r["instance"]})
    assert list(out.index) == list(range(len(odd)))


def test_coverage_reports_the_number_to_watch():
    rules = {"side": lambda r: ("L" if "(L)" in str(r["instance"] or "") else None)}
    out = ex.coverage(BODIES, rules).iloc[0]
    assert out["bodies"] == 2                     # Tm2_A2(L) and LN_C5(L)_NCL
    assert out["coverage"] == pytest.approx(2 / 7)
    assert out["distinct"] == 1 and "L (2)" in out["top"]


def test_unparsed_ranks_what_to_fix_next_by_body_count():
    frame = pd.DataFrame({"instance": ["AGNG"] * 5 + ["VLNP(L)"] * 2 + ["Tm2(L)"]})
    rules = {"side": lambda r: ("L" if "(L)" in str(r["instance"] or "") else None)}
    out = ex.unparsed(frame, rules, "side")
    assert out.iloc[0]["instance"] == "AGNG" and out.iloc[0]["bodies"] == 5
    assert ex.unparsed(BODIES, {"all": lambda r: "x"}, "all").empty


def test_unparsed_names_the_rules_it_has():
    with pytest.raises(KeyError, match="no rule named 'zzz'"):
        ex.unparsed(BODIES, {"a": lambda r: None}, "zzz")


# --------------------------------------------------------------------------- #
# comparing a curated field against a parse
# --------------------------------------------------------------------------- #
def test_compare_classifies_the_relation_without_judging_it():
    """A `type` coarser than the parse is a difference of grain, not an error, so the
    relation is reported and the judgment left to whoever knows the dataset."""
    core = lambda r: str(r["instance"] or "").split("(")[0] or None
    out = ex.compare(BODIES, "type", core)
    rel = dict(zip(out["body"], out["relation"]))
    assert rel[1] == "derived_is_longer"       # type=Tm2, derived=Tm2_A2
    assert rel[3] == "exact"                   # type=Tm, derived=Tm
    assert rel[2] == "field_missing"            # no type on VLNP(R)_fragment
    assert set(out["relation"]) <= set(ex.RELATIONS)


def test_compare_keeps_the_raw_string_for_reading_the_disagreements():
    core = lambda r: "ZZZ"
    out = ex.compare(BODIES, "type", core)
    row = out[out["body"] == 1].iloc[0]
    assert row["relation"] == "unrelated"
    assert row["instance"] == "Tm2_A2(L)"      # so it can be eyeballed
