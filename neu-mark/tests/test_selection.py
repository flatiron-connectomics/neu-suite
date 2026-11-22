"""The criteria beyond a synapse count: connection pairs and names."""

import numpy as np
import pandas as pd
import pytest

from neu_mark import rule, selection
from neu_mark.rules import RuleSet


def test_pairs_count_each_end_and_leave_out_unknown_bodies():
    conns = pd.DataFrame({"pre_body": np.array([1, 1, 1, 2, 0], np.uint64),
                          "post_body": np.array([2, 3, 1, 3, 3], np.uint64)})
    pairs = selection.pair_counts(conns)
    # body 1: three as pre, plus once as post of its own autapse
    assert pairs.to_dict() == {1: 4, 3: 3, 2: 2}
    assert 0 not in pairs.index
    assert pairs.index.dtype == np.uint64


def test_pairs_keep_bodies_above_2_to_the_53():
    big = np.uint64(2**60 + 1)
    conns = pd.DataFrame({"pre_body": np.array([big, big], np.uint64),
                          "post_body": np.array([7, 8], np.uint64)})
    assert selection.pair_counts(conns)[big] == 2


@rule(drop=True)
def noise(r):
    """not data"""
    return "noise" if "irrelevant" in (r["instance"] or "") else None


@rule(tag=False)
def cell_type(r):
    """a typed name"""
    name = r["instance"] or ""
    return None if not name or "fragment" in name else name


RULES = RuleSet([noise, cell_type], label="instance")


def _records(**names):
    return pd.DataFrame({"body": np.array([int(b[1:]) for b in names], np.uint64),
                         "instance": pd.Series(list(names.values()), dtype="string")})


def test_a_name_counts_when_the_rule_gives_a_value_and_nothing_drops_it():
    records = _records(b1="Tm1(L)", b2="SNP_fragment", b3="Tm1_irrelevant", b4=None)
    named = selection.named_bodies(records, RULES, "cell_type")
    assert named.to_dict() == {1: "Tm1(L)"}
    assert named.index.dtype == np.uint64


def test_a_field_no_record_has_is_missing_not_an_error():
    @rule(tag=False, needs=["type"])
    def typed(r):
        """by the type field"""
        return r["type"]

    records = _records(b1="Tm1(L)")
    assert selection.named_bodies(records, RuleSet([typed], label="instance"),
                                  "typed").empty


def test_a_drop_rule_cannot_name_a_body():
    with pytest.raises(ValueError, match="drop rule"):
        selection.named_bodies(_records(b1="x"), RULES, "noise")


def test_admitted_lists_only_what_the_count_missed():
    counted = pd.DataFrame({"body": np.array([1, 2], np.uint64)})
    pairs = pd.Series({1: 90, 3: 60, 4: 10})
    named = pd.Series({2: "A", 5: "B"})
    got = selection.admitted(counted, pairs=pairs, min_pairs=50, named=named)
    assert got == {"by_pairs": [1, 3], "by_name": [2, 5], "extra": [3, 5]}


@rule(prefix="")
def glia(r):
    """glia"""
    return "glia" if "glia" in (r["instance"] or "") else None


def test_excluded_names_the_first_rule_that_fires_and_ignores_other_drops():
    records = _records(b1="glia_irrelevant", b2="Tm1_irrelevant", b3="Tm1(L)")
    ruleset = RuleSet([noise, glia, cell_type], label="instance")
    got = selection.excluded_bodies(records, ruleset, ["glia", "noise"])
    assert got.to_dict() == {1: "glia", 2: "noise"}
    # noise is not asked for here, so it does not exclude
    assert selection.excluded_bodies(records, ruleset, ["glia"]).to_dict() == {1: "glia"}
    assert got.index.dtype == np.uint64
