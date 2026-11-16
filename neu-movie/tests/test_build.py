"""Builds on disk, the site config, the render job, and the inspection links. No Blender and
no Slurm: the steps that run them are checked up to the command line."""

import json
import subprocess

import pytest

from neu_movie import build as build_mod
from neu_movie import site as site_mod
from neu_movie.spec import SpecError

SPEC = """\
view = "../views/view.json"
mode = "replace"
[neurons]
layer = "seg"
[timing]
start = 1.0
every = 5.0
fade = 1.0
hold = 2.0
[labels]
legend_header = "Types:"
[[beat]]
title = "one"
sets = [{ name = "A", ids = [1] }, { name = "B", ids = [2] }]
[[beat]]
title = "two"
name = "C"
ids = [3]
"""

VIEW = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
        "position": [1, 2, 3], "projectionScale": 1000,
        "layers": [{"type": "segmentation", "name": "seg", "source": "precomputed://x"}]}


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.delenv(site_mod.ENV_VAR, raising=False)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "neu-movie.toml"
    path.write_text('blender = "tools/blender"\nneuvid = "/opt/neuVid/neuVid"\n'
                    'meshes = "~/meshes"\n'
                    '[slurm]\noptions = ["--partition=gpu", "--gpus-per-node=4"]\n'
                    'time = "02:00:00"\n')
    return path


@pytest.fixture
def build_dir(tmp_path, site):
    """A build as `build` leaves it, without the Blender files."""
    d = tmp_path / "movies" / "renders" / "movie_20260101T0000"
    d.mkdir(parents=True)
    (d / "spec.toml").write_text(SPEC)
    (d / "view.json").write_text(json.dumps(VIEW))
    (d / "build.json").write_text(json.dumps({"seconds": 16.5, "frames": 495,
                                              "smooth": None, "name": "movie"}))
    (d / "sets.json").write_text(json.dumps({"volume": "precomputed://x", "limit": None, "sets": {
        "A": {"color": "#1f77b4", "ids": ["1"]}, "C": {"color": "#ff7f0e", "ids": ["3"]}}}))
    (d / "movie.json").write_text(json.dumps({"fps": 30}))
    return d


def test_site_is_found_above_the_start_and_paths_are_relative_to_it(build_dir, site):
    s = site_mod.load(build_dir)
    assert s.path == site
    assert s.blender == site.parent / "tools" / "blender"
    assert str(s.neuvid) == "/opt/neuVid/neuVid"
    assert s.library is None and s.ffmpeg_setup == ""
    assert s.slurm["time"] == "02:00:00"


def test_the_env_var_wins_and_a_missing_key_is_named(tmp_path, monkeypatch, site):
    other = tmp_path / "other.toml"
    other.write_text('blender = "/b"\n')
    monkeypatch.setenv(site_mod.ENV_VAR, str(other))
    s = site_mod.load(tmp_path)
    assert s.path == other
    with pytest.raises(site_mod.SiteError, match="sets no `neuvid`"):
        s.neuvid


def test_no_site_config_says_where_to_put_one(tmp_path, monkeypatch):
    monkeypatch.delenv(site_mod.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(site_mod.SiteError, match="NEU_MOVIE_CONFIG"):
        site_mod.load(tmp_path)


def test_open_build_reads_its_own_copy_of_the_view(build_dir):
    d, spec, info, recorded = build_mod.open_build(build_dir / "build.json")
    assert d == build_dir
    assert spec["view"] == str(build_dir / "view.json")
    assert info["frames"] == 495 and set(recorded["sets"]) == {"A", "C"}


def test_open_build_refuses_a_folder_that_is_not_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a build"):
        build_mod.open_build(tmp_path)


def test_relabel_uses_the_recorded_sets_and_colours(build_dir):
    (build_dir / "spec.toml").write_text(SPEC.replace('"one"', '"first"'))
    build_mod.relabel(build_dir)
    doc = json.loads((build_dir / "movie_labels.json").read_text())
    texts = [a["text"] for c, a in doc["animation"] if c == "label"]
    assert "first" in texts and "A" in texts and "C" in texts
    assert "B" not in texts                       # resolved to nothing when built


def test_unit_nm_needs_isotropic_dimensions():
    assert build_mod.unit_nm(VIEW) == pytest.approx(8.0)
    flat = dict(VIEW, dimensions=dict(VIEW["dimensions"], z=[4e-8, "m"]))
    with pytest.raises(SpecError, match="anisotropic"):
        build_mod.unit_nm(flat)


def test_a_layer_not_in_the_view_lists_the_ones_that_are():
    with pytest.raises(SpecError, match="no layer 'cells'; it has 'seg'"):
        build_mod._layer(VIEW, "cells")


def test_head_volume_from_a_url():
    spec = {"head": {"volume": "s3://my-bucket/rois/", "library": None, "path": "/rois/head/"}}
    assert build_mod.head_volume(spec, None) == "s3://my-bucket/rois/rois/head"


def test_new_build_is_named_by_spec_and_minute(tmp_path):
    spec = {"path": str(tmp_path / "pop.toml"), "name": "pop"}
    d = build_mod.new_build(spec)
    assert d.parent == tmp_path / "renders" and d.name.startswith("pop_2")
    with pytest.raises(FileExistsError):
        build_mod.new_build(spec)


def test_job_script(build_dir, site):
    job = build_mod.job_script(build_dir, first=10, last=20)
    text = job.read_text()
    assert job == build_dir / "render.sbatch"
    assert "#SBATCH --partition=gpu\n#SBATCH --gpus-per-node=4\n" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "first=10; last=20; total=495" in text
    assert f"export NEU_MOVIE_CONFIG={site}" in text
    assert "--skipExisting" in text and "-m neu_movie finish" in text
    assert subprocess.run(["bash", "-n", str(job)]).returncode == 0
    assert "--time=00:30:00" in build_mod.job_script(build_dir, time_limit="00:30:00").read_text()


def test_submit_prints_how_to_follow_the_job(build_dir, monkeypatch, capsys):
    calls = []

    def sbatch(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 4242\n",
                                           stderr="")
    monkeypatch.setattr(build_mod.subprocess, "run", sbatch)
    assert build_mod.submit(build_dir) == "Submitted batch job 4242"
    assert calls == [["sbatch", str(build_dir / "render.sbatch")]]
    err = capsys.readouterr().err
    assert f"tail -f {build_dir}/slurm-4242.out" in err
    assert f'echo "$(ls {build_dir}/frames | wc -l) of 495 frames"' in err


def test_a_dry_run_submits_nothing(build_dir, monkeypatch, capsys):
    monkeypatch.setattr(build_mod.subprocess, "run", pytest.fail)
    assert build_mod.submit(build_dir, dry_run=True) is None
    assert "sbatch " in capsys.readouterr().err


def test_frame_numbers(tmp_path):
    for name in ("0002.png", "0010.png", "0001.png", "notes.txt", "x.png"):
        (tmp_path / name).write_text("")
    assert build_mod.frame_numbers(tmp_path) == [1, 2, 10]


def test_links_put_each_beats_sets_in_their_own_coloured_layers(build_dir):
    pytest.importorskip("neu_glance")
    from neu_movie import links
    _, spec, _, recorded = build_mod.open_build(build_dir)
    ids = {n: s["ids"] for n, s in recorded["sets"].items()}
    colours = {n: s["color"] for n, s in recorded["sets"].items()}
    states = links.beat_states(spec, VIEW, ids, colours)
    assert [title for title, _ in states] == ["one", "two"]
    layers = {lyr["name"]: lyr for lyr in states[0][1]["layers"]}
    assert layers["A"]["segmentDefaultColor"] == "#1f77b4"
    assert "B" not in layers
    assert VIEW["layers"][0]["name"] == "seg" and len(VIEW["layers"]) == 1
    url = links.beat_links(spec, VIEW, ids, colours)[1][1]
    assert url.startswith("http")
