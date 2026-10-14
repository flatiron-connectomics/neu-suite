"""The @rule framework: declaration, validation, and testing a rule in isolation."""

import re
import sys

import pandas as pd
import pytest

from neu_mark import rules
from neu_mark.rules import Rule, RuleSet, rule

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "LN_C5(L)_NCL", None],
                          dtype="string"),
    "type": pd.Series(["Tm2", None, "LN", None], dtype="string"),
})

SIDE = re.compile(r"\((L|R)\)")


@rule
def side(r):
    """Hemisphere, from the parenthesized (L)/(R)."""
    m = SIDE.search(str(r["instance"] or ""))
    return m.group(1) if m else None


@rule(multi=True)
def column(r):
    """Column labels; genuinely multi-valued."""
    return re.findall(r"_([A-Z]\d+)(?=$|[_(])", str(r["instance"] or ""))


@rule(prefix="", consumes=["ncl", "nucleus"])
def nucleated(r):
    """Cell body in the volume. Normalizes an ALIAS, so its value is not its token."""
    lowered = {t.lower() for t in rules.explore.split_tokens(r["instance"])}
    return "nucleated" if lowered & {"ncl", "nucleus"} else None


# --------------------------------------------------------------------------- #
# declaration
# --------------------------------------------------------------------------- #
def test_declaration_bare_and_called():
    assert isinstance(side, Rule) and isinstance(column, Rule)
    assert (side.name, side.multi, column.name, column.multi) == ("side", False, "column", True)
    assert side.description == "Hemisphere, from the parenthesized (L)/(R)."
    assert side(pd.Series({"instance": "Tm2_A2(L)"})) == "L"      # still callable
    assert side.needs == ("instance",)

    @rule(needs=["instance", "type"])
    def both(r):
        return None

    assert both.needs == ("instance", "type")


# --------------------------------------------------------------------------- #
# testing a rule
# --------------------------------------------------------------------------- #
def test_test_evaluates_one_string_positionally():
    assert side.test("Tm2_A2(L)") == "L"
    assert side.test("AGNG") is None
    assert side.test(None) is None
    assert column.test("L1/L3_D4(R)") == ["D4"]


def test_test_takes_other_fields_by_keyword_and_passes_missing_ones_as_none():
    """None rather than pd.NA, so `r["x"] or ""` behaves as it does in a run."""
    seen = {}

    @rule(needs=["instance", "type"])
    def curated(r):
        seen["type"] = r["type"]
        return r["type"]

    assert curated.test("anything", type="Tm2") == "Tm2"
    curated.test("X")
    assert seen["type"] is None


# --------------------------------------------------------------------------- #
# the multi declaration
# --------------------------------------------------------------------------- #
def test_a_single_valued_rule_returning_several_values_is_an_error():
    """The findall-instead-of-search slip, which passes until a string matches twice."""
    @rule
    def oops(r):
        return re.findall(r"([A-Z])", str(r["instance"] or ""))

    assert oops.values(pd.Series({"instance": "abc_D"})) == ("D",)
    with pytest.raises(ValueError, match="single-valued"):
        oops.values(pd.Series({"instance": "A_B"}))


def test_a_multi_valued_rule_may_return_any_number():
    assert column.values(pd.Series({"instance": "X_A1_B2"})) == ("A1", "B2")
    assert column.values(pd.Series({"instance": "AGNG"})) == ()


# --------------------------------------------------------------------------- #
# RuleSet
# --------------------------------------------------------------------------- #
def test_a_ruleset_keeps_declaration_order():
    rs = RuleSet([side, column])
    assert rs.names == ["side", "column"] and len(rs) == 2
    assert rs["side"] is side


@pytest.mark.parametrize("twin", [side, rule(lambda r: None, name="side")],
                         ids=["same-rule", "same-name"])
def test_duplicate_names_in_one_declaration_are_refused(twin):
    with pytest.raises(ValueError, match="duplicate rule names"):
        RuleSet([side, twin])


def test_an_undecorated_function_is_refused_with_the_fix():
    with pytest.raises(TypeError, match="decorate it with @rule"):
        RuleSet([lambda r: None])


def test_a_plain_mapping_is_accepted():
    rs = RuleSet({"x": lambda r: "v"})
    assert rs.names == ["x"] and rs.explain("anything") == {"x": "v"}


def test_reading_an_undeclared_column_names_the_rule_and_the_fix():
    """Otherwise it is a bare KeyError on the one row lacking the column."""
    @rule
    def group(r):
        """reads `type` without saying so"""
        return r["type"]

    with pytest.raises(KeyError, match=r"does not declare"):
        group.test("Tm2_A2\\(L\\)")
    with pytest.raises(KeyError, match=r"needs=\['instance', 'type'\]"):
        RuleSet([group]).explain("Tm2")


def test_a_declared_column_that_is_missing_raises_the_original_error():
    """Only an undeclared read is rewritten."""
    @rule(needs=["type"])
    def group(r):
        """declares it properly"""
        return r["nope"] if r["type"] is None else r["type"]

    with pytest.raises(KeyError) as caught:
        group.test(type=None)
    assert "does not declare" in str(caught.value)   # 'nope' is the undeclared one


def test_unmet_needs_are_reported_up_front_naming_the_rule():
    @rule(needs=["instance", "roi_post_top"])
    def needs_roi(r):
        return None

    with pytest.raises(KeyError, match="needs_roi needs roi_post_top"):
        RuleSet([needs_roi]).apply(BODIES)


def test_a_rules_repr_delimits_its_fields():
    head, description = repr(nucleated).split(" — ", 1)
    assert head.split(" | ") == ["<Rule nucleated", "tag bare", "needs=instance",
                                 "consumes=[ncl, nucleus]"]
    assert description.startswith("Cell body in the volume.")
    assert "| multi" in repr(column) and "multi" not in repr(side)


def test_a_ruleset_repr_aligns_its_rules_into_columns():
    """`side` declares no extras, so its line also checks padding is kept."""
    text = repr(RuleSet([side, column, nucleated], source="wasp_rules.py"))
    header, *lines = text.splitlines()
    assert header == "<RuleSet 3 rules, from wasp_rules.py>"
    assert [line.split()[0] for line in lines] == ["side", "column", "nucleated"]
    assert len({line.index("needs=") for line in lines}) == 1
    assert len({line.index("—") for line in lines}) == 1
    assert repr(RuleSet([])) == "<RuleSet 0 rules>"


def test_a_notebook_renders_the_table_and_a_terminal_the_reprs():
    rs = RuleSet([side, column])
    html = rs._repr_html_()
    assert "<table" in html and "column" in html
    assert repr(rs).startswith("<RuleSet 2 rules")
    assert isinstance(rs.describe(), pd.DataFrame)


def test_describe_reports_what_every_rule_declared():
    @rule(drop=True)
    def noise(r):
        """junk"""
        return None

    frame = RuleSet([side, column, nucleated, noise]).describe()
    assert list(frame["rule"]) == ["side", "column", "nucleated", "noise"]
    assert list(frame["kind"]) == ["tag", "tag", "tag", "drop"]
    by_rule = frame.set_index("rule")
    assert bool(by_rule.loc["column", "multi"]) is True
    assert by_rule.loc["side", "description"] == "Hemisphere, from the parenthesized (L)/(R)."
    assert by_rule.loc["side", "consumes"] == "derived"
    assert by_rule.loc["nucleated", "consumes"] == "ncl, nucleus"


@pytest.mark.parametrize("args, expected", [
    (("Tm2_A2(L)",), {"side": "L", "column": ("A2",)}),
    (("LN_C5(L)_NCL",), {"side": "L", "column": ("C5",)}),
    (("AGNG",), {"side": None, "column": ()}),
    ((), {"side": None, "column": ()}),
])
def test_explain_reports_each_facet_scalar_or_tuple_by_declaration(args, expected):
    assert RuleSet([side, column]).explain(*args) == expected


def test_apply_and_coverage_carry_the_declared_facts():
    rs = RuleSet([side, column])
    assert rs.apply(BODIES)["side"].tolist() == [("L",), ("R",), ("L",), ()]

    cov = rs.coverage(BODIES).set_index("rule")
    assert cov.loc["side", "bodies"] == 3
    assert bool(cov.loc["column", "multi"]) is True
    assert cov.loc["side", "description"].startswith("Hemisphere")


def test_unparsed_delegates_and_ranks_by_bodies():
    out = RuleSet([side]).unparsed(BODIES, "side")
    assert out.empty or "instance" in out.columns


def test_keep_is_an_explicit_allowlist():
    """A drop-list would let a new field reach a viewer by being forgotten."""
    assert RuleSet([side], keep=["instance", "type"]).keep == ("instance", "type")
    assert RuleSet([side]).keep == ()


# --------------------------------------------------------------------------- #
# loading a rules module by path
# --------------------------------------------------------------------------- #
MODULE = '''
import re
from neu_mark import rule

KEEP = ["instance", "type"]

@rule
def side(r):
    """Side."""
    m = re.search(r"\\((L|R)\\)", str(r["instance"] or ""))
    return m.group(1) if m else None

@rule(multi=True)
def column(r):
    """Columns."""
    return re.findall(r"_([A-Z]\\d+)(?=$|[_(])", str(r["instance"] or ""))
'''


def test_a_module_of_decorated_functions_needs_no_rules_variable(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE)
    rs = rules.from_module(str(path))
    assert rs.names == ["side", "column"]
    assert rs.keep == ("instance", "type")
    assert rs.source == str(path)
    assert rs.explain("Tm2_A2(L)") == {"side": "L", "column": ("A2",)}


def test_an_explicit_rules_list_wins(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE + "\nRULES = [column]\n")
    assert rules.from_module(str(path)).names == ["column"]


def test_a_module_with_no_rules_or_no_file_is_a_clear_error(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("X = 1\n")
    with pytest.raises(ValueError, match="defines no rules"):
        rules.from_module(str(path))
    with pytest.raises(FileNotFoundError, match="no rules module"):
        rules.from_module(str(tmp_path / "nope.py"))


def test_a_broken_module_does_not_linger_in_sys_modules(tmp_path):
    """So fixing the file and re-loading works."""
    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        rules.from_module(str(path))
    assert not [m for m in sys.modules if m.endswith("_broken")]

    path.write_text(MODULE)
    assert rules.from_module(str(path)).names == ["side", "column"]


# --------------------------------------------------------------------------- #
# drop rules
# --------------------------------------------------------------------------- #
def test_a_drop_rule_is_never_tagged_and_partitions_the_set():
    @rule(drop=True)
    def noise(r):
        """junk"""
        return None

    assert noise.drop is True and noise.tag is False
    rs = RuleSet([noise, side, column])
    assert [r.name for r in rs.drops] == ["noise"]
    assert [r.name for r in rs.tagged] == ["side", "column"]


def test_a_multi_valued_drop_rule_is_refused():
    with pytest.raises(ValueError, match="drop=True and multi=True"):
        @rule(drop=True, multi=True)
        def noise(r):
            """junk"""
            return ["a", "b"]


# --------------------------------------------------------------------------- #
# consumed / remainder
# --------------------------------------------------------------------------- #
def test_a_rule_reports_the_tokens_it_accounted_for():
    """`nucleated` returns an alias of its token, so it declares `consumes`."""
    row = rules._row({"instance": "Tm2_A2(L)"}, needs=["instance"])
    assert side.consumed(row) == ("L",)
    assert column.consumed(row) == ("A2",)
    ncl = rules._row({"instance": "LN_C5(L)_NCL"}, needs=["instance"])
    assert nucleated.consumed(ncl) == ("NCL",)
    assert RuleSet([side, column, nucleated]).consumed("LN_C5(L)_NCL") == {
        "side": ("L",), "column": ("C5",), "nucleated": ("NCL",)}


@pytest.mark.parametrize("ruleset, text, expected", [
    (nucleated, "LN_C5(L)_NCL", "LN_C5_L"),
    # token-granular: removing the substring "L" would eat LC10's own L
    (side, "LC10_C5(L)", "LC10_C5"),
    (side, "L1/L3_D4(R)", "L1/L3_D4"),
    (RuleSet([side, column, nucleated]), "Tm2_A2(L)", "Tm2"),
    (RuleSet([side, column, nucleated]), "LN_C5(L)_NCL", "LN"),
    (RuleSet([nucleated, column, side]), "LN_C5(L)_NCL", "LN"),     # order-independent
    (RuleSet([nucleated]), "NCL", None),
    # separators collapse and margins strip; "(L)" becomes a plain separator
    (RuleSet([column]), "Tm2_A2(L)", "Tm2_L"),
    (RuleSet([side]), "_Tm2__(L)_", "Tm2"),
])
def test_remainder(ruleset, text, expected):
    assert ruleset.remainder(text) == expected


def test_a_rule_that_does_not_read_the_string_consumes_nothing_from_it():
    """`group` returns the curated `type`, often the very token a remainder should keep."""
    @rule(needs=["type"])
    def group(r):
        """curated cell type"""
        return r["type"]

    row = rules._row({"instance": "Tm2_A2(L)", "type": "Tm2"}, needs=["instance", "type"])
    assert group.consumed(row) == ()
    assert RuleSet([side, column, group]).remainder(instance="Tm2_A2(L)", type="Tm2") == "Tm2"


def test_consumes_accepts_a_callable_for_anything_irregular():
    @rule(consumes=lambda r: ["weird"] if "x" in (r["instance"] or "") else [])
    def odd(r):
        """fires on x, accounts for `weird`"""
        return "x" if "x" in (r["instance"] or "") else None

    assert odd.remainder("x_weird_Tm2") == "x_Tm2"


# --------------------------------------------------------------------------- #
# override
# --------------------------------------------------------------------------- #
def test_override_replaces_by_name_and_keeps_the_position():
    """Position matters because a later rule may read what an earlier one produced."""
    @rule(name="side")
    def other(r):
        """lowercased side"""
        return "l"

    merged = RuleSet([side, column]).override([other])
    assert merged.names == ["side", "column"]
    assert merged["side"].description == "lowercased side"


def test_override_appends_rules_that_are_new():
    @rule
    def glomerulus(r):
        """antennal lobe glomerulus"""
        return None

    assert RuleSet([side]).override([glomerulus]).names == ["side", "glomerulus"]


def test_override_leaves_both_operands_alone():
    """A builtin RuleSet is module state; mutating it would leak one run's plugin."""
    base = RuleSet([side, column])
    base.override([rule(lambda r: None, name="side")])
    assert base["side"] is side


# --------------------------------------------------------------------------- #
# reserved names
# --------------------------------------------------------------------------- #
def test_a_reserved_name_must_declare_tag_false():
    """A tagged rule named `label` would mint `label:…` tags and never set the property."""
    with pytest.raises(ValueError, match="reserved rule name"):
        rule(lambda r: "x", name="label")
    assert rule(lambda r: "x", name="label", tag=False).name == "label"


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_provenance_embeds_the_source_and_what_each_rule_declared(tmp_path):
    """The text, not just a hash, so the module can be recovered later."""
    import hashlib

    path = tmp_path / "r.py"
    path.write_text(MODULE)
    record = rules.provenance(rules.from_module(str(path)))

    assert record["names"] == ["side", "column"]
    assert [f["path"] for f in record["files"]] == [str(path.resolve())]
    entry = record["files"][0]
    assert entry["text"] == MODULE
    assert entry["sha256"] == hashlib.sha256(MODULE.encode()).hexdigest()

    declared = {r["name"]: r for r in record["rules"]}
    assert declared["column"]["multi"] is True
    assert declared["column"]["description"] == "Columns."
    assert declared["side"]["needs"] == ["instance"]


def test_provenance_skips_the_builtins():
    from neu_mark.segprops_rules import BUILTIN

    assert rules.provenance(BUILTIN)["files"] == []


def test_provenance_can_omit_the_text_and_says_when_it_did(tmp_path):
    path = tmp_path / "r.py"
    path.write_text(MODULE + "\n#" + "x" * 5000 + "\n")
    rs = rules.from_module(str(path))

    entry = rules.provenance(rs, embed=False)["files"][0]
    assert "text" not in entry and entry["sha256"]

    entry = rules.provenance(rs, max_embed_bytes=100)["files"][0]
    assert "text" not in entry
    assert "max_embed_bytes=100" in entry["text_omitted"]
