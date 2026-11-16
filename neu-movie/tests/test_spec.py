"""The spec loader: defaults, the set forms, and every error a typo should raise."""

import re
import tomllib

import pytest

from conftest import make_spec
from neu_movie import spec as spec_mod
from neu_movie.spec import SpecError


def test_the_docstring_example_parses():
    example = re.search(r"\n\n(    view = .*?)\n\n\*\*Modes", spec_mod.__doc__, re.S).group(1)
    raw = tomllib.loads("\n".join(line[4:] for line in example.splitlines()))
    spec = spec_mod.parse(raw, "/specs/example.toml")
    assert [b["sets"] for b in spec["beats"]] == [["LA", "Tm"], ["R7p"]]
    assert spec["sets"]["Tm"]["exclude"] == ["82159813"]
    assert spec["sets"]["LA"]["query"] == {"without": ["fragment"], "label": "^L[1-9]"}


def test_defaults_name_and_view_path(tmp_path):
    spec = make_spec([{"name": "A", "ids": [1]}], tmp=tmp_path)
    assert spec["name"] == "movie"
    assert spec["view"] == str(tmp_path / "view.json")
    assert spec["timing"]["fade"] == 1.5 and spec["camera"]["style"] == "orbit"
    assert spec["head"] is None and spec["neuropils"] is None


def test_load_reads_toml(tmp_path):
    path = tmp_path / "pop.toml"
    path.write_text('view = "v.json"\n[neurons]\nlayer = "seg"\n'
                    '[[beat]]\nname = "A"\nids = [3, 1]\ncolor = "#AABBCC"\n')
    spec = spec_mod.load(path)
    assert spec["name"] == "pop"
    assert spec["sets"]["A"] == {"name": "A", "regions": [], "exclude": [],
                                 "color": "#aabbcc", "ids": ["3", "1"], "query": None}


def test_a_toml_syntax_error_is_a_spec_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("view = \n")
    with pytest.raises(SpecError, match="bad.toml"):
        spec_mod.load(path)


def test_query_defaults_merge_under_the_set(tmp_path):
    spec = make_spec([{"name": "T", "label": "^Tm", "without": ["truncated"]}],
                     neurons={"layer": "seg", "query": {"without": ["fragment"], "limit": 9}})
    assert spec["sets"]["T"]["query"] == {"without": ["truncated"], "limit": 9, "label": "^Tm"}


def test_a_set_named_again_is_the_same_set():
    spec = make_spec([{"sets": [{"name": "A", "label": "^A"}]},
                      {"sets": [{"name": "A"}, {"name": "B", "ids": [2]}]}])
    assert [b["sets"] for b in spec["beats"]] == [["A"], ["A", "B"]]
    assert list(spec["sets"]) == ["A", "B"]


def test_repeating_an_identical_definition_is_allowed():
    spec = make_spec([{"name": "A", "ids": [1]}, {"name": "A", "ids": [1]}])
    assert len(spec["beats"]) == 2


@pytest.mark.parametrize("raw, beats, message", [
    ({"colour": 1}, None, r"unknown key\(s\) colour"),
    ({"timing": {"fade_s": 1}}, None, r"\[timing\]: unknown key\(s\) fade_s"),
    ({"mode": "stack"}, None, "mode 'stack'"),
    ({"camera": {"style": "dolly"}}, None, "camera style 'dolly'"),
    ({"neurons": {}}, None, "`layer` is required"),
    ({"neurons": {"layer": "s", "query": {"lable": "x"}}}, None, "query: unknown key"),
    ({"head": {"id": 3}}, None, "one of `library` or `volume`"),
    ({"head": {"library": "x", "volume": "y", "id": 3}}, None, "one of `library` or `volume`"),
    ({"head": {"library": "x"}}, None, r"\[head\]: `id` is required"),
    ({}, [], r"no `\[\[beat\]\]`"),
    ({}, [{"ids": [1]}], "needs a `name`"),
    ({}, [{"name": "A", "lable": "^A"}], r"unknown key\(s\) lable"),
    ({}, [{"name": "A"}], "neither `ids` nor a query"),
    ({}, [{"name": "A", "ids": [1], "label": "^A"}], "`ids` or a query, not both"),
    ({}, [{"name": "A", "ids": [1], "color": "red"}], "is not #rrggbb"),
    ({}, [{"title": "t", "name": "A", "sets": [{"name": "B", "ids": [1]}]}],
     "takes only `title`"),
    ({}, [{"sets": [{"name": "A", "ids": [1]}, {"name": "A"}]}], "listed twice"),
    ({}, [{"name": "A", "ids": [1]}, {"name": "A", "ids": [2]}], "defined differently"),
    ({"mode": "accumulate"}, [{"name": "A", "ids": [1]}, {"name": "A"}],
     "accumulate mode.*: A"),
])
def test_mistakes_fail_loudly(raw, beats, message):
    with pytest.raises(SpecError, match=message):
        make_spec([{"name": "Z", "ids": [9]}] if beats is None else beats, **raw)


def test_a_missing_view_is_an_error():
    with pytest.raises(SpecError, match="`view`"):
        spec_mod.parse({"neurons": {"layer": "s"}, "beat": [{"name": "A", "ids": [1]}]},
                       "/specs/m.toml")


def test_defaults_are_not_shared_between_specs():
    a = make_spec([{"name": "A", "ids": [1]}])
    a["neurons"]["query"]["label"] = "changed"
    b = make_spec([{"name": "A", "ids": [1]}])
    assert b["neurons"]["query"] == {}
