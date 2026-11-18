"""Builds on disk, the site config, the render job, and the inspection links. No Blender and
no Slurm: the steps that run them are checked up to the command line."""

import json
import subprocess

import pytest

from neu_movie import build as build_mod
from neu_movie import site as site_mod
from neu_movie.spec import SpecError

SPEC = """\
mode = "replace"
[view]
position = [8, 16, 24]
height = 8000
[neurons]
volume = "s3://my-bucket/seg"
[timing]
start = 1.0
every = 5.0
fade = 1.0
hold = 2.0
[labels]
neuron_legend_header = "Types:"
[[beat]]
title = "one"
sets = [{ name = "A", ids = [1] }, { name = "B", ids = [2] }]
[[beat]]
title = "two"
name = "C"
ids = [3]
"""

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


def test_open_build_reads_its_own_spec(build_dir):
    d, spec, info, recorded = build_mod.open_build(build_dir / "build.json")
    assert d == build_dir
    assert spec["view"]["position"] == [8.0, 16.0, 24.0]
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


def test_unit_nm_is_the_volumes_finest_voxel_and_must_be_a_cube(monkeypatch):
    from types import SimpleNamespace

    import neu_vol

    def scales(voxel):
        return lambda volume: [SimpleNamespace(frame=SimpleNamespace(voxel_size_nm=voxel))]
    monkeypatch.setattr(neu_vol, "read_scales", scales((8.0, 8.0, 8.0)))
    assert build_mod.unit_nm("s3://my-bucket/seg") == 8.0
    monkeypatch.setattr(neu_vol, "read_scales", scales((40.0, 8.0, 8.0)))
    with pytest.raises(SpecError, match="anisotropic"):
        build_mod.unit_nm("s3://my-bucket/seg")


def test_a_location_is_a_volume_or_library_entry_with_a_path_under_it(tmp_path, monkeypatch):
    section = {"volume": "s3://my-bucket/v1/", "library": None, "path": "/rois/head/"}
    assert build_mod.location(section, None) == "s3://my-bucket/v1/rois/head"
    assert build_mod.location(dict(section, path=""), None) == "s3://my-bucket/v1"
    from neu_recipe import library
    monkeypatch.setattr(library, "location", lambda name, directory: f"s3://my-bucket/{name}")
    site = type("Site", (), {"library": tmp_path})()
    section = {"volume": None, "library": "fly/v2", "path": "rois/neuropils"}
    assert build_mod.location(section, site) == "s3://my-bucket/fly/v2/rois/neuropils"


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


def test_an_orbiting_build_fixes_its_orbit(build_dir, site, monkeypatch):
    ran = []
    monkeypatch.setattr(build_mod, "blender", lambda site, script, *a, **k: ran.append(
        (script.name, [str(x) for x in a])))
    for name in ("movie.blend", "movieAnim.blend"):
        (build_dir / name).write_bytes(b"x")
    s = site_mod.load(build_dir)
    build_mod.make_blend(build_dir, s, {"name": "movie", "smooth": None, "frames": 495,
                                        "orbit": True, "loop": True})
    assert ran[-1] == ("orbit.py", ["-i", str(build_dir / "movie.json"), "--loop"])
    ran.clear()
    build_mod.make_blend(build_dir, s, {"name": "movie", "smooth": None, "frames": 495,
                                        "orbit": True, "loop": False})
    assert ran[-1] == ("orbit.py", ["-i", str(build_dir / "movie.json")])
    ran.clear()
    build_mod.make_blend(build_dir, s, {"name": "movie", "smooth": None, "frames": 495})
    assert [r[0] for r in ran] == ["importMeshes.py", "addAnimation.py"]


def test_a_build_with_a_fill_light_adds_it_last(build_dir, site, monkeypatch):
    ran = []
    monkeypatch.setattr(build_mod, "blender", lambda site, script, *a, **k: ran.append(
        (script.name, [str(x) for x in a])))
    for name in ("movie.blend", "movieAnim.blend"):
        (build_dir / name).write_bytes(b"x")
    s = site_mod.load(build_dir)
    build_mod.make_blend(build_dir, s, {"name": "movie", "smooth": None, "frames": 495,
                                        "orbit": True, "loop": True, "fill": 2.5})
    assert [r[0] for r in ran] == ["importMeshes.py", "addAnimation.py", "orbit.py",
                                   "fill_light.py"]
    assert ran[-1][1] == ["-i", str(build_dir / "movie.json"), "--strength", "2.5"]
    ran.clear()
    build_mod.make_blend(build_dir, s, {"name": "movie", "smooth": None, "frames": 495,
                                        "fill": 0})
    assert "fill_light.py" not in [r[0] for r in ran]


def test_a_missing_blender_script_fails_before_blender_runs(build_dir, site, tmp_path):
    with pytest.raises(FileNotFoundError, match="no Blender script"):
        build_mod.blender(site_mod.load(build_dir), tmp_path / "gone.py")


def test_relabel_uses_the_recorded_neuropil_colours(build_dir):
    (build_dir / "spec.toml").write_text(
        SPEC.replace("[timing]", '[neuropils]\nvolume = "s3://my-bucket/rois"\n[timing]')
            .replace("ids = [3]", 'ids = [3]\nneuropils = ["LA(L)"]'))
    recorded = json.loads((build_dir / "sets.json").read_text())
    recorded["neuropils"] = {"LA(L)": {"id": "2", "color": "#1b9e77"}}
    (build_dir / "sets.json").write_text(json.dumps(recorded))
    build_mod.relabel(build_dir)
    doc = json.loads((build_dir / "movie_labels.json").read_text())
    la = [a for c, a in doc["animation"] if c == "label" and a["text"] == "LA(L)"]
    assert len(la) == 1 and la[0]["align"] == "right"


def test_an_older_build_with_the_earlier_key_names_still_relabels(build_dir):
    (build_dir / "spec.toml").write_text(
        SPEC.replace("neuron_legend_header", "legend_header")
            .replace("[timing]", '[neuropils]\nvolume = "s3://my-bucket/rois"\n[timing]')
            .replace("ids = [3]", 'ids = [3]\nregions = ["LA(L)"]'))
    recorded = json.loads((build_dir / "sets.json").read_text())
    recorded["neuropils"] = "s3://my-bucket/rois"
    recorded["regions"] = {"LA(L)": {"id": "2", "color": "#1b9e77"}}
    (build_dir / "sets.json").write_text(json.dumps(recorded))
    build_mod.relabel(build_dir)
    texts = [a["text"] for c, a in json.loads((build_dir / "movie_labels.json").read_text())
             ["animation"] if c == "label"]
    assert "Types:" in texts and "LA(L)" in texts


@pytest.mark.parametrize("labels", [True, False])
def test_finish_encodes_the_movie_and_a_labelled_one(build_dir, monkeypatch, labels):
    encoded, labelled = [], []
    monkeypatch.setattr(build_mod, "encode",
                        lambda d, site, info, frames, out: encoded.append((frames.name, out.name)))
    monkeypatch.setattr(build_mod, "label_frames",
                        lambda d, site, info, frames, out, processes: labelled.append(out.name))
    (build_dir / "movie_labels.json").write_text("{}")
    build_mod.finish(build_dir, labels=labels)
    assert encoded[0] == ("frames", "movie.mp4")
    assert encoded[1:] == ([("frames-labeled", "movie_labeled.mp4")] if labels else [])
    assert labelled == (["frames-labeled"] if labels else [])
