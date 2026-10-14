"""Building the neuroglancer segment_properties document.

The format allows one `tags` property, tag values as increasing indices, no spaces, and
matches case-insensitively. A document can satisfy all that and still be wrong in ways only
a viewer shows, so what is testable is pinned here.
"""

import re

import pandas as pd
import pytest

from neu_mark import rule, segprops

BODIES = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 5, 6, 7], dtype="uint64"),
    "instance": pd.Series(["Tm2_A2(L)", "VLNP(R)_fragment", "LN_C5(L)_NCL",
                           "PGNG_CV", "irrelevant", "glia", None], dtype="string"),
    "type": pd.Series(["Tm2", None, "LN", None, None, None, None], dtype="string"),
})

COUNTS = pd.DataFrame({
    "body": pd.Series([1, 2, 3, 4, 6, 7], dtype="uint64"),
    "pre": [10, 0, 5, 1, 0, 0], "post": [90, 3, 20, 2, 1, 0],
    "syn": [100, 3, 25, 3, 1, 0],
})


def _built(**kw):
    return segprops.build(BODIES, **kw)


def _props(info):
    return {p["id"]: p for p in info["inline"]["properties"]}


def _labels(info):
    return dict(zip(info["inline"]["ids"], _props(info)["instance"]["values"]))


def _tags_of(result, body):
    tags = _props(result["info"])["tags"]
    i = result["info"]["inline"]["ids"].index(str(body))
    return {tags["tags"][j] for j in tags["values"][i]}


# --------------------------------------------------------------------------- #
# the pd.NA trap
# --------------------------------------------------------------------------- #
def test_a_missing_type_produces_no_group_tag():
    """`str(pd.NA)` is the truthy "<NA>"; a hand-rolled check once tagged every body
    `group-<na>`, reading as full coverage of a field populated on 5%."""
    result = _built()
    tags = _props(result["info"])["tags"]["tags"]
    assert not [t for t in tags if "na" == t or "<na>" in t or "nan" in t]
    assert "group:Tm2" in tags and "group:LN" in tags
    assert sum(1 for t in tags if t.startswith("group:")) == 2
    assert result["report"]["coverage"]["group"]["bodies"] == 2


@pytest.mark.parametrize("missing", [pd.NA, None, float("nan"), "   "])
def test_normalize_tag_handles_every_flavour_of_missing(missing):
    assert segprops.normalize_tag(missing) is None


# --------------------------------------------------------------------------- #
# what the format requires
# --------------------------------------------------------------------------- #
def _check_property_counts(info):
    props = info["inline"]["properties"]
    for kind in ("label", "tags", "description"):
        assert sum(1 for p in props if p["type"] == kind) <= 1, kind


def _check_tag_values(info):
    tags = _props(info)["tags"]
    for row in tags["values"]:
        assert all(isinstance(i, int) for i in row)
        assert row == sorted(row) and len(row) == len(set(row))
        assert all(0 <= i < len(tags["tags"]) for i in row)


def _check_tags_property(info):
    tags = _props(info)["tags"]
    assert len(tags["tag_descriptions"]) == len(tags["tags"])
    assert "description" not in tags              # forbidden on `tags`, allowed elsewhere
    assert all(" " not in t and not t.startswith("#") for t in tags["tags"])


def _check_ids(info):
    assert all(isinstance(i, str) and i.isdigit() for i in info["inline"]["ids"])


def _check_at_type(info):
    from neu_vol.ops.subresources import SUBRESOURCE_TYPES

    assert info["@type"] == SUBRESOURCE_TYPES["segment_properties"]


@pytest.mark.parametrize("check", [_check_property_counts, _check_tag_values,
                                   _check_tags_property, _check_ids, _check_at_type])
def test_the_document_satisfies_the_format(check):
    check(_built()["info"])


def test_tag_case_is_preserved_and_collisions_are_reported_not_folded():
    """Matching is case-insensitive, so two tags differing in case are one viewer chip.
    Folding would be lossy; the collisions are reported instead."""
    tags = _props(_built()["info"])["tags"]["tags"]
    assert "group:Tm2" in tags
    assert segprops.case_collisions(["group:Tm2", "group:TM2", "side:L"]) == {
        "group:tm2": ["group:Tm2", "group:TM2"]}
    assert segprops.case_collisions(["group:Tm2", "side:L"]) == {}


def test_facet_tags_use_a_colon_and_flags_stay_bare():
    """With a hyphen, a facet value is indistinguishable from a standalone flag."""
    tags = _props(_built()["info"])["tags"]["tags"]
    assert {t.split(":", 1)[0] for t in tags if ":" in t} == {"group", "side"}
    for flag in ("fragment", "truncated", "nucleated", "cervical", "glia"):
        assert all(":" not in t for t in tags if t == flag)


# --------------------------------------------------------------------------- #
# facets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body, label", [("1", "Tm2_A2(L)"), ("3", "LN_C5(L)_NCL"),
                                         ("7", "7")])
def test_the_label_is_the_raw_instance_string_or_the_id(body, label):
    """A blank label would be indistinguishable from a missing property in the viewer."""
    assert _labels(_built()["info"])[body] == label


@pytest.mark.parametrize("body, expected", [
    (1, {"group:Tm2", "side:L"}),
    (2, {"side:R", "fragment"}),
    (3, {"group:LN", "side:L", "nucleated"}),
    (4, {"cervical"}),
    (6, {"glia"}),                                 # glia are kept and tagged by default
])
def test_each_facet_lands_on_the_right_body(body, expected):
    assert _tags_of(_built(), body) == expected


@pytest.mark.parametrize("instance, label, expected", [
    ("Tm(R)_truncated?", "Tm(R)_truncated", {"side:R", "truncated"}),   # doubt mark dropped
    ("DPM-like(L)", "DPM-like(L)", {"side:L"}),                         # -like is not a tag
])
def test_instance_suffixes(instance, label, expected):
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"), "instance": [instance]})
    result = segprops.build(frame)
    assert _props(result["info"])["instance"]["values"] == [label]
    assert _tags_of(result, 1) == expected


def test_no_builtin_parses_a_column_out_of_the_instance():
    """Which part of a name is a column is dataset knowledge, so it lives in a rules module."""
    result = _built()
    assert not [t for t in _props(result["info"])["tags"]["tags"] if t.startswith("col")]
    assert "column" not in result["report"]["coverage"]


def test_noise_is_excluded_entirely_not_merely_untagged():
    result = _built()
    assert "5" not in result["info"]["inline"]["ids"]        # 'irrelevant'
    assert result["report"]["excluded"]["irrelevant"] == 1


def test_everything_excluded_is_an_error_not_an_empty_document():
    frame = pd.DataFrame({"body": pd.Series([1], dtype="uint64"), "instance": ["irrelevant"]})
    with pytest.raises(ValueError, match="every body was excluded"):
        segprops.build(frame)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def test_numbers_are_uint32_because_float32_would_round_them():
    """float32 is exact only to 2^24, and real bodies reach 86 million voxels."""
    info = _built(counts=COUNTS, sizes={1: 86_315_764, 2: 4_510_070})["info"]
    for name in ("pre", "post", "syn", "voxels"):
        p = _props(info)[name]
        assert p["type"] == "number" and p["data_type"] == "uint32", name
    assert dict(zip(info["inline"]["ids"], _props(info)["voxels"]["values"]))["1"] == 86_315_764


def test_number_values_are_parallel_to_ids():
    info = _built(counts=COUNTS)["info"]
    ids, pre = info["inline"]["ids"], _props(info)["pre"]["values"]
    assert len(pre) == len(ids)
    assert dict(zip(ids, pre))["1"] == 10 and dict(zip(ids, pre))["3"] == 5

    partial = _built(counts=COUNTS.iloc[:1])["info"]            # missing bodies get zero
    assert len(_props(partial)["pre"]["values"]) == len(partial["inline"]["ids"])


def test_numbers_are_omitted_when_not_supplied():
    assert [p["id"] for p in _built()["info"]["inline"]["properties"]] == ["instance", "tags"]


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def test_it_writes_where_the_info_key_will_point(tmp_path):
    from neu_vol.location import read_json

    dst = str(tmp_path / "vol")
    assert segprops.write(dst, _built()["info"]) == "segment_properties/info"
    assert read_json(dst, "segment_properties", "info")["@type"] == segprops.AT_TYPE


def test_the_module_never_opens_a_file_directly():
    """Every write goes through `location`, or an s3 destination silently writes nothing."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(segprops))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in called


# --------------------------------------------------------------------------- #
# rules: adding, overriding, replacing
# --------------------------------------------------------------------------- #
def test_a_plugin_rule_is_added_alongside_the_builtins():
    @rule(multi=True)
    def column(r):
        """optic lobe column"""
        return re.findall(r"_([A-Z]\d+)(?=$|[_(])", r["instance"] or "")

    result = segprops.build(BODIES, rules=[column])
    assert _tags_of(result, 1) == {"group:Tm2", "side:L", "column:A2"}
    assert "side" in result["report"]["coverage"]


def test_a_plugin_rule_overrides_a_builtin_by_name_and_keeps_its_position():
    @rule
    def side(r):
        """hemisphere, lowercased"""
        m = re.search(r"\((L|R)\)", r["instance"] or "")
        return m.group(1).lower() if m else None

    result = segprops.build(BODIES, rules=[side])
    assert _tags_of(result, 1) == {"group:Tm2", "side:l"}
    names = result["report"]["rules"]["names"]
    assert names.index("side") == 1 and names.count("side") == 1


def test_rules_only_drops_the_builtins():
    @rule
    def side(r):
        """hemisphere"""
        return "L" if "(L)" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[side], rules_only=True)
    assert result["report"]["rules"]["names"] == ["side"]
    assert result["report"]["excluded"] == {}             # `noise` went with the builtins
    assert "5" in result["info"]["inline"]["ids"]


def test_the_prefix_colon_is_added_once_however_the_rule_spells_it():
    @rule(prefix="col:", name="col")
    def col(r):
        """column"""
        return "A2" if "A2" in (r["instance"] or "") else None

    assert "col:A2" in _tags_of(segprops.build(BODIES, rules=[col]), 1)


def test_a_drop_rule_excludes_the_body_and_counts_its_reason():
    @rule(drop=True)
    def tiny(r):
        """bodies the dataset marks as specks"""
        return "speck" if "PGNG" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[tiny])
    assert "4" not in result["info"]["inline"]["ids"]
    assert result["report"]["excluded"]["speck"] == 1
    assert result["report"]["coverage"]["tiny"]["kind"] == "drop"


def test_a_dropped_body_never_runs_the_tag_rules():
    """Otherwise coverage would be counted against the wrong denominator."""
    seen = []

    @rule
    def watcher(r):
        """records which bodies reached the tag rules"""
        seen.append(r["instance"])
        return None

    segprops.build(BODIES, rules=[watcher])
    assert "irrelevant" not in seen and "Tm2_A2(L)" in seen


def test_keep_glia_false_turns_the_glia_rule_into_a_drop():
    kept, dropped = segprops.build(BODIES), segprops.build(BODIES, keep_glia=False)
    assert kept["report"]["coverage"]["glia"]["kind"] == "tag"
    assert dropped["report"]["coverage"]["glia"]["kind"] == "drop"
    assert dropped["report"]["excluded"]["glia"] == 1
    assert "6" not in dropped["info"]["inline"]["ids"]


def test_keep_glia_false_drops_whatever_the_current_glia_rule_matches():
    """A dataset that spells glia differently redefines the rule."""
    @rule(prefix="")
    def glia(r):
        """glia, as this dataset spells it"""
        return "glia" if "PGNG" in (r["instance"] or "") else None

    dropped = segprops.build(BODIES, rules=[glia], keep_glia=False)
    assert "4" not in dropped["info"]["inline"]["ids"]
    assert "6" in dropped["info"]["inline"]["ids"]


# --------------------------------------------------------------------------- #
# the reserved rules, and tag=False
# --------------------------------------------------------------------------- #
def test_a_label_rule_names_the_segments():
    @rule(name="label", tag=False)
    def label(r):
        """the name a viewer shows"""
        return (r["type"] or None) and f"cell {r['type']}"

    labels = _labels(segprops.build(BODIES, rules=[label])["info"])
    assert labels["1"] == "cell Tm2"
    assert labels["2"] == "VLNP(R)_fragment"      # falls back to the instance...
    assert labels["7"] == "7"                     # ...and then to the id


def test_a_label_keeps_its_spaces_because_it_is_not_a_tag():
    @rule(name="label", tag=False)
    def label(r):
        """the name a viewer shows"""
        return "LC10 anterior"

    info = segprops.build(BODIES, rules=[label])["info"]
    assert _props(info)["instance"]["values"][0] == "LC10 anterior"


def test_a_tagged_rule_named_label_is_refused():
    with pytest.raises(ValueError, match="reserved"):
        @rule(name="label")
        def label(r):
            """oops"""
            return "x"


def test_a_quiet_rule_is_computed_and_reported_but_emits_nothing():
    """`tag=False` is the seam for a future number or string property."""
    @rule(tag=False)
    def depth(r):
        """soma depth, one day a number property"""
        return 7 if "(L)" in (r["instance"] or "") else None

    result = segprops.build(BODIES, rules=[depth])
    assert result["report"]["coverage"]["depth"] == {
        "bodies": 2, "fraction": 2 / 6, "distinct": 1, "kind": "property",
        "description": "soma depth, one day a number property"}
    assert not [t for t in _props(result["info"])["tags"]["tags"] if "depth" in t]
    assert [p["id"] for p in result["info"]["inline"]["properties"]] == ["instance", "tags"]


def test_tag_descriptions_come_from_the_rule_that_minted_the_tag():
    tags = _props(_built()["info"])["tags"]
    described = dict(zip(tags["tags"], tags["tag_descriptions"]))
    assert described["fragment"] == "a fragment, not a completely traced cell"
    assert described["side:L"] == "hemisphere, from the (L)/(R) in the name"


def test_a_rule_needing_an_absent_column_warns_rather_than_raising():
    """A list where nobody has a `type` has no `type` column at all, which is normal."""
    result = segprops.build(BODIES.drop(columns=["type"]))
    assert result["report"]["absent_columns"] == ["type"]
    assert result["report"]["coverage"]["group"]["bodies"] == 0


# --------------------------------------------------------------------------- #
# the vocabulary guard
# --------------------------------------------------------------------------- #
def _many(n):
    """n bodies with n distinct instances that no builtin rule fires on."""
    return pd.DataFrame({"body": pd.Series(range(1, n + 1), dtype="uint64"),
                         "instance": pd.Series([f"cell{i}" for i in range(n)],
                                               dtype="string")})


@rule
def echo(r):
    """a rule that hands the whole instance string back"""
    return r["instance"]


@rule
def coarse(r):
    """two bodies per value, so a fraction below 1 can admit it"""
    return "g" + str(int(r["instance"].removeprefix("cell")) // 2)


def test_a_rule_echoing_the_instance_string_is_refused_by_name():
    """A chip per body is spec-legal and unusable. The message names the rule and how the
    limit was computed."""
    with pytest.raises(ValueError, match=r"'echo' with 2000 distinct") as caught:
        segprops.build(_many(2000), rules=[echo])
    assert "10% of 2000, floored at 100" in str(caught.value)


def test_raising_the_fraction_admits_a_genuinely_diverse_dataset():
    """`coarse` is 50% distinct: refused at the 10% default, admitted at 60%."""
    with pytest.raises(ValueError, match="'coarse' with 1000 distinct"):
        segprops.build(_many(2000), rules=[coarse])
    report = segprops.build(_many(2000), rules=[coarse], max_tags=0.6)["report"]
    assert (report["tags"], report["tag_limit"], report["max_tags"]) == (1000, 1200, 0.6)


def test_a_short_list_is_not_judged_by_the_fraction():
    result = segprops.build(_many(40), rules=[echo])
    assert result["report"]["tags"] == 40
    assert result["report"]["tag_limit"] == segprops.MIN_TAG_ALLOWANCE


@pytest.mark.parametrize("setting, expected", [
    (0.10, (2111, "fraction")),
    (2111, (2111, "count")),
    (1, (1, "count")),
    (1.0, (1, "count")),             # chosen by value: TOML and JSON cannot tell 1 from 1.0
    (25, (25, "count")),             # an explicit count is never raised to the floor
])
def test_a_value_below_one_is_a_fraction_and_one_or_above_is_a_count(setting, expected):
    assert segprops.MIN_TAG_ALLOWANCE > 25
    assert segprops.resolve_tag_limit(setting, 21116) == expected


def test_only_a_converted_fraction_gets_the_floor():
    assert segprops.resolve_tag_limit(0.001, 21116)[0] == segprops.MIN_TAG_ALLOWANCE


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_a_non_positive_limit_is_refused(bad):
    with pytest.raises(ValueError, match="must be positive"):
        segprops.resolve_tag_limit(bad, 100)


def test_an_absolute_count_caps_the_vocabulary():
    with pytest.raises(ValueError, match="explicit count of 25"):
        segprops.build(_many(2000), rules=[echo], max_tags=25)
    assert segprops.build(_many(2000), rules=[echo], max_tags=5000)["report"]["tags"] == 2000


def test_too_many_facets_is_refused_even_though_the_vocabulary_is_tiny():
    """Many facets of one tag each: a small vocabulary, and nothing to group on."""
    rules = [rule(lambda r, _i=i: "x", name=f"facet{i}") for i in range(segprops.MAX_FACETS + 2)]
    with pytest.raises(ValueError, match="distinct tag facets"):
        segprops.build(BODIES, rules=rules)


def test_the_report_carries_the_document_size():
    report = _built()["report"]
    assert report["bytes"] > 0
    assert report["facets"] == ["", "group", "side"]
