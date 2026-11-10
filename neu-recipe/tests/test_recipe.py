"""Recipe parsing: substitution, the [recipe] table, step selection, warnings."""

from __future__ import annotations

from pathlib import Path

import pytest

from neu_recipe.recipe import RecipeError, load, select, warnings

LIBRARY = {"lib/seg": {"name": "lib/seg", "location": "s3://my-bucket/seg",
                       "export_uuid": "abc123", "voxel_size_nm": [8, 8, 8]}}


def entry(name):
    if name not in LIBRARY:
        raise RecipeError(f"volume {name!r}: not in the library")
    return LIBRARY[name]


def write_recipe(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pipelines" / "demo.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


def resolve(tmp_path, text, **kw):
    return load(write_recipe(tmp_path, text), entry=kw.pop("entry", entry), **kw)


def test_vars_use_volumes_and_earlier_vars(tmp_path):
    r = resolve(tmp_path, '''
[volumes]
seg = "lib/seg"
[vars]
work = "/w/{seg.name}"
bodies = "{work}/b.csv"
dvid = "dvid://host/{seg.export_uuid}/labels"
[[step]]
name = "a"
run = """tool --src {seg} --out {bodies}
         --name 'two words' --lit 'x_{{uuid:8}}.csv'"""
''')
    assert r["vars"] == {"work": "/w/lib/seg", "bodies": "/w/lib/seg/b.csv",
                         "dvid": "dvid://host/abc123/labels"}
    assert r["steps"][0]["argv"] == ["tool", "--src", "s3://my-bucket/seg", "--out",
                                     "/w/lib/seg/b.csv", "--name", "two words",
                                     "--lit", "x_{uuid:8}.csv"]


@pytest.mark.parametrize("text, match", [
    ('[vars]\na = "{b}"\nb = "x"\n[[step]]\nname = "s"\nrun = "t"', r"\{b\} is not defined"),
    ('[vars]\na = "x"\n[[step]]\nname = "s"\nrun = "t {typo}"', r"step s: \{typo\}"),
    ('[volumes]\nseg = "lib/seg"\n[[step]]\nname = "s"\nrun = "t {seg.nope}"', "no field"),
    ('[volumes]\nseg = "lib/seg"\n[[step]]\nname = "s"\nrun = "t {seg.voxel_size_nm}"',
     "not a string"),
    ('[vars]\na = "x"\n[[step]]\nname = "s"\nrun = "t {a.b}"', "a is not a volume"),
    ('[volumes]\nseg = "lib/nope"\n[[step]]\nname = "s"\nrun = "t"', "not in the library"),
    ('[volumes]\nseg = "lib/seg"\n[vars]\nseg = "x"\n[[step]]\nname = "s"\nrun = "t"',
     "shadows a volume"),
    ('[[step]]\nname = "s"\nafter = ["x"]\nrun = "t"', r"unknown key\(s\) \['after'\]"),
    ('[[step]]\nname = "s"\nrun = "t"\n[[step]]\nname = "s"\nrun = "u"', "used twice"),
    ('[[step]]\nrun = "t"', "needs both name and run"),
    ('[vars]\na = "x"', "no \\[\\[step\\]\\]"),
    ('[step_defaults]\na = 1', "unknown table"),
    ('[recipe]\nroot = "."\n[[step]]\nname = "s"\nrun = "t"', r"\[recipe\]: unknown key"),
])
def test_a_bad_recipe_fails_before_anything_runs(tmp_path, text, match):
    with pytest.raises(RecipeError, match=match):
        resolve(tmp_path, text)


def test_steps_run_from_the_recipes_directory_unless_it_says_otherwise(tmp_path):
    r = resolve(tmp_path, '[[step]]\nname = "s"\nrun = "t"')
    assert r["cwd"] == (tmp_path / "pipelines").resolve()
    r = resolve(tmp_path, '[recipe]\ncwd = ".."\n[[step]]\nname = "s"\nrun = "t"')
    assert r["cwd"] == tmp_path.resolve()


def _library_dir(tmp_path, name="lib"):
    d = tmp_path / name
    d.mkdir()
    (d / "fly.toml").write_text(
        '[volume."fly/seg"]\nkind = "segmentation"\nlocation = "s3://my-bucket/seg/"\n')
    return d


RECIPE_USING_LIB = '[volumes]\nseg = "fly/seg"\n[[step]]\nname = "s"\nrun = "t {seg}"'


def test_the_library_comes_from_the_recipe_relative_to_it(tmp_path, monkeypatch):
    monkeypatch.delenv("NEU_RECIPE_LIBRARY", raising=False)
    _library_dir(tmp_path)
    path = write_recipe(tmp_path, '[recipe]\nlibrary = "../lib"\n' + RECIPE_USING_LIB)
    r = load(path)
    assert r["steps"][0]["argv"] == ["t", "s3://my-bucket/seg"]
    assert r["library"] == (tmp_path / "lib").resolve()


def test_an_explicit_library_overrides_the_recipes(tmp_path):
    other = _library_dir(tmp_path, "other")
    path = write_recipe(tmp_path, '[recipe]\nlibrary = "../missing"\n' + RECIPE_USING_LIB)
    assert load(path, library=other)["library"] == other


def test_the_library_falls_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NEU_RECIPE_LIBRARY", str(_library_dir(tmp_path)))
    r = load(write_recipe(tmp_path, RECIPE_USING_LIB))
    assert r["volumes"]["seg"]["location"] == "s3://my-bucket/seg"


def test_a_recipe_without_volumes_needs_no_library(tmp_path, monkeypatch):
    monkeypatch.delenv("NEU_RECIPE_LIBRARY", raising=False)
    r = load(write_recipe(tmp_path, '[[step]]\nname = "s"\nrun = "t"'))
    assert r["library"] is None and r["volumes"] == {}


def test_no_library_at_all_is_a_recipe_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NEU_RECIPE_LIBRARY", raising=False)
    with pytest.raises(RecipeError, match="NEU_RECIPE_LIBRARY"):
        load(write_recipe(tmp_path, RECIPE_USING_LIB))


def test_from_and_only_select_steps(tmp_path):
    steps = resolve(tmp_path, "".join(f'[[step]]\nname = "{n}"\nrun = "t"\n'
                                      for n in "abc"))["steps"]
    assert [s["name"] for s in select(steps, start="b")] == ["b", "c"]
    assert [s["name"] for s in select(steps, only="b")] == ["b"]
    with pytest.raises(RecipeError, match="no step 'z'"):
        select(steps, only="z")


def test_a_dvid_branch_is_flagged_and_a_node_is_not(tmp_path):
    steps = resolve(tmp_path, '''
[[step]]
name = "branch"
run = "t --src dvid://host/abc123:main/labels"
[[step]]
name = "node"
run = "t --src dvid://host/abc123/labels"
''')["steps"]
    found = warnings(steps)
    assert len(found) == 1 and found[0].startswith("step branch:")
