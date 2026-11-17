"""The command line: the parser the docs render, errors as messages, and `check`."""

import subprocess
import sys

import pytest

from neu_movie import build as build_mod
from neu_movie.cli import build_parser, main


def test_the_parser_has_its_subcommands():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert list(sub.choices) == ["camera", "check", "build", "animate", "stills", "submit",
                                 "relabel", "finish"]


def test_help_imports_no_store_or_viewer_code():
    out = subprocess.run([sys.executable, "-c",
                          "import sys; from neu_movie.cli import build_parser; build_parser(); "
                          "print(sorted(m for m in ('neu_vol', 'neu_glance', 'neu_recipe') "
                          "if m in sys.modules))"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_a_bad_spec_is_a_message_not_a_traceback(tmp_path, capsys):
    path = tmp_path / "m.toml"
    path.write_text('mode = "stack"\n[view]\nposition = [0, 0, 0]\nheight = 1\n'
                    '[neurons]\nlibrary = "s"\n')
    assert main(["check", str(path)]) == 1
    assert "neu-movie: m.toml: mode 'stack'" in capsys.readouterr().err


def test_check_prints_the_schedule(tmp_path, monkeypatch, capsys):
    path = tmp_path / "m.toml"
    path.write_text('[view]\nposition = [0, 0, 0]\nheight = 8\n[neurons]\nlibrary = "s"\n'
                    '[[beat]]\ntitle = "first"\nname = "A"\nids = [1, 2]\n'
                    '[[beat]]\nname = "B"\nlabel = "^B"\n')
    (tmp_path / "site.toml").write_text('blender = "/b"\n')
    monkeypatch.setenv("NEU_MOVIE_CONFIG", str(tmp_path / "site.toml"))
    world = {"volume": "s3://my-bucket/seg", "unit": 8.0, "ids": {"A": ["1", "2"]},
             "colours": {"A": "#1f77b4", "B": "#ff7f0e"}, "roi_volume": None, "regions": {},
             "head_volume": None}
    monkeypatch.setattr(build_mod, "resolve", lambda spec, site, limit=None: world)
    assert main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "m: replace, camera orbit, 2 bodies" in out
    assert "first" in out and "A (2, #1f77b4)" in out and "B (none)" in out
    assert "frames" in out


@pytest.mark.parametrize("argv", [["submit", "b", "--frames", "1-10"], ["finish", "b"],
                                  ["animate", "s", "--dry-run", "--time", "01:00:00"]])
def test_arguments_parse(argv):
    build_parser().parse_args(argv)


def test_camera_prints_a_view_block_from_a_state(tmp_path, capsys):
    import json
    state = {"dimensions": {a: [8e-9, "m"] for a in "xyz"}, "position": [10, 20, 30],
             "projectionOrientation": [0, 0, 0, 1], "projectionScale": 1000, "layers": []}
    (tmp_path / "s.json").write_text(json.dumps(state))
    assert main(["camera", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "[view]\nposition = [80, 160, 240]" in out
    assert "orientation = [0, 0, 0, 1]" in out and "height = 8000 " in out
    assert "# up = [0, -1, 0]" in out          # neuroglancer's y points down the screen
