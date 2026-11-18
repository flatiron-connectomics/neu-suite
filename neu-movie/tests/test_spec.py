"""The spec loader: defaults, the set forms, and every error a typo should raise."""

import re
import tomllib

import pytest

from conftest import make_spec
from neu_movie import spec as spec_mod
from neu_movie.spec import SpecError


def test_the_docstring_example_parses():
    example = re.search(r"\n\n(    mode = .*?)\n\n\*\*Modes", spec_mod.__doc__, re.S).group(1)
    raw = tomllib.loads("\n".join(line[4:] for line in example.splitlines()))
    spec = spec_mod.parse(raw, "/specs/example.toml")
    assert [b["sets"] for b in spec["beats"]] == [["LA", "Tm"], ["R7p"]]
    assert spec["sets"]["Tm"]["exclude"] == ["82159813"]
    assert spec["sets"]["LA"]["query"] == {"without": ["fragment"], "label": "^L[1-9]"}


def test_defaults_name_and_view(tmp_path):
    spec = make_spec([{"name": "A", "ids": [1]}], tmp=tmp_path)
    assert spec["name"] == "movie"
    assert spec["view"] == {"position": [8.0, 16.0, 24.0], "orientation": [0.0, 0.0, 0.0, 1.0],
                            "height": 8000.0}
    assert spec["timing"]["fade"] == 1.5 and spec["camera"]["style"] == "orbit"
    assert spec["head"] is None and spec["neuropils"] is None


def test_load_reads_toml(tmp_path):
    path = tmp_path / "pop.toml"
    path.write_text('[view]\nposition = [1, 2, 3]\nheight = 100\n'
                    '[neurons]\nvolume = "s3://my-bucket/seg"\n'
                    '[[beat]]\nname = "A"\nids = [3, 1]\ncolor = "#AABBCC"\n')
    spec = spec_mod.load(path)
    assert spec["name"] == "pop"
    assert spec["sets"]["A"] == {"name": "A", "neuropils": [], "exclude": [],
                                 "color": "#aabbcc", "ids": ["3", "1"], "query": None}


def test_a_toml_syntax_error_is_a_spec_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("view = \n")
    with pytest.raises(SpecError, match="bad.toml"):
        spec_mod.load(path)


def test_query_defaults_merge_under_the_set(tmp_path):
    spec = make_spec([{"name": "T", "label": "^Tm", "without": ["truncated"]}],
                     neurons={"library": "lib", "query": {"without": ["fragment"], "limit": 9}})
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
    ({"neurons": {}}, None, r"\[neurons\]: give one of `library` or `volume`"),
    ({"neurons": {"library": "l", "volume": "v"}}, None, "give one of `library` or `volume`"),
    ({"neurons": {"library": "l", "query": {"lable": "x"}}}, None, "query: unknown key"),
    ({"view": {"position": [1, 2], "height": 1}}, None, r"`position` must be 3 numbers"),
    ({"view": {"position": [1, 2, 3], "height": 0}}, None, "`height` must be a number above"),
    ({"view": {"position": [1, 2, 3], "height": 1, "orientation": [0, 0, 1]}}, None,
     "`orientation` must be 4 numbers"),
    ({"camera": {"up": [0, 1]}}, None, r"`up` must be 3 numbers"),
    ({"camera": {"loop": True, "orbit_seconds": 10}}, None, "takes no `orbit_seconds`"),
    ({"camera": {"orbit_seconds": 0}}, None, "`orbit_seconds` must be a number above 0"),
    ({"neuropils": {"library": "l"}}, [{"name": "Z", "ids": [9], "neuropils": ["LA"]}],
     "with `library`, give `path`"),
    ({}, [{"name": "Z", "ids": [9], "neuropils": ["LA"]}], "no \\[neuropils\\] section"),
    ({"neuropils": {"volume": "v", "colors": {"ME": "#ff0000"}}},
     [{"name": "Z", "ids": [9], "neuropils": ["LA"]}], "colors: no set shows ME"),
    ({"neuropils": {"volume": "v", "colors": {"LA": "red"}}},
     [{"name": "Z", "ids": [9], "neuropils": ["LA"]}], "colors: colour 'red' is not"),
    ({"head": {"id": 3}}, None, "one of `library` or `volume`"),
    ({"head": {"library": "x", "volume": "y", "id": 3}}, None, "one of `library` or `volume`"),
    ({"head": {"library": "x"}}, None, r"\[head\]: `id` is required"),
    ({}, [], r"no `\[\[beat\]\]`"),
    ({}, [{"ids": [1]}], "needs a `name`"),
    ({}, [{"name": "A", "lable": "^A"}], r"unknown key\(s\) lable"),
    ({"neuropils": {"volume": "v"}}, [{"name": "A", "ids": [1], "regions": ["LA"]}],
     r"unknown key\(s\) regions"),                     # the earlier names, outside a build
    ({"labels": {"legend_header": "x"}}, None, r"unknown key\(s\) legend_header"),
    ({}, [{"name": "A"}], "neither `ids` nor a query"),
    ({}, [{"name": "A", "ids": [1], "label": "^A"}], "`ids` or a query, not both"),
    ({}, [{"name": "A", "ids": [1], "color": "red"}], "is not #rrggbb"),
    ({}, [{"title": "t", "name": "A", "sets": [{"name": "B", "ids": [1]}]}],
     "takes only `title`"),
    ({}, [{"sets": [{"name": "A", "ids": [1]}, {"name": "A"}]}], "listed twice"),
    ({}, [{"name": "A", "ids": [1]}, {"name": "A", "ids": [2]}], "defined differently"),
    ({"mode": "accumulate"}, [{"name": "A", "ids": [1]}, {"name": "A"}],
     "accumulate mode.*: A"),
    ({"timing": {"repeat": 0}}, None, "`repeat` must be a whole number"),
    ({"timing": {"repeat": 1.5}}, None, "`repeat` must be a whole number"),
    ({"mode": "accumulate", "timing": {"repeat": 2}}, None, "`repeat` needs mode"),
    ({"camera": {"style": "follow", "loop": True}}, None, "`loop` is for the orbit camera"),
    ({"camera": {"frame": "neurons"}}, None, "frame 'neurons' is not one of"),
    ({"camera": {"style": "glide", "loop": True}}, None, "`loop` is for the orbit camera"),
    ({"light": {"fill": -1}}, None, "`fill` must be a number"),
    ({"light": {"fill": "bright"}}, None, "`fill` must be a number"),
])
def test_mistakes_fail_loudly(raw, beats, message):
    with pytest.raises(SpecError, match=message):
        make_spec([{"name": "Z", "ids": [9]}] if beats is None else beats, **raw)


def test_a_missing_view_is_an_error():
    with pytest.raises(SpecError, match=r"\[view\] needs `position` and `height`"):
        spec_mod.parse({"neurons": {"library": "s"}, "beat": [{"name": "A", "ids": [1]}]},
                       "/specs/m.toml")


def test_a_view_file_says_how_to_make_the_table():
    with pytest.raises(SpecError, match="`neu-movie camera front.json` prints one"):
        spec_mod.parse({"view": "front.json", "neurons": {"library": "s"},
                        "beat": [{"name": "A", "ids": [1]}]}, "/specs/m.toml")


def test_defaults_are_not_shared_between_specs():
    a = make_spec([{"name": "A", "ids": [1]}])
    a["neurons"]["query"]["label"] = "changed"
    b = make_spec([{"name": "A", "ids": [1]}])
    assert b["neurons"]["query"] == {}


def test_repeat_plays_the_beats_over_again():
    spec = make_spec([{"title": "one", "name": "A", "ids": [1]},
                      {"title": "two", "name": "B", "ids": [2]}], timing={"repeat": 3})
    assert [b["title"] for b in spec["beats"]] == ["one", "two"] * 3
    assert set(spec["sets"]) == {"A", "B"}


def test_up_and_neuropil_colours_are_normalised():
    spec = make_spec([{"name": "A", "ids": [1], "neuropils": ["LA"]}],
                     camera={"style": "follow", "up": [0, 1, 0]},
                     neuropils={"library": "l", "path": "rois", "colors": {"LA": "#AABBCC"}})
    assert spec["camera"]["up"] == [0.0, 1.0, 0.0]
    assert spec["neuropils"]["colors"] == {"LA": "#aabbcc"}
