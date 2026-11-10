"""The command line: the parser the docs render, and the `library` subcommand."""

from __future__ import annotations

import subprocess
import sys

import pytest

from neu_recipe import library as lib
from neu_recipe.cli import build_parser, main


def test_the_parser_has_the_three_subcommands():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert list(sub.choices) == ["run", "status", "library"]


def test_help_runs_without_importing_the_runner():
    out = subprocess.run([sys.executable, "-c",
                          "import sys; from neu_recipe.cli import build_parser; "
                          "build_parser(); print('neu_recipe.runner' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


@pytest.fixture
def library_dir(tmp_path):
    (tmp_path / "fly.toml").write_text(
        '[volume."fly/seg"]\nkind = "segmentation"\nlocation = "s3://my-bucket/seg"\n')
    (tmp_path / "worm.toml").write_text(
        '[volume."worm/img"]\nkind = "image"\nlocation = "s3://my-bucket/img"\n')
    return tmp_path


def test_library_lists_volumes_and_filters_by_specimen(library_dir, capsys):
    assert main(["library", "--library", str(library_dir)]) == 0
    out = capsys.readouterr().out
    assert "fly/seg" in out and "worm/img" in out
    assert main(["library", "worm", "--library", str(library_dir)]) == 0
    out = capsys.readouterr().out
    assert "worm/img" in out and "fly/seg" not in out


def test_library_check_exits_non_zero_on_a_miss(library_dir, monkeypatch, capsys):
    monkeypatch.setattr(lib, "probe", lambda loc: "info" if loc.endswith("seg") else None)
    with pytest.raises(SystemExit, match="1 location"):
        main(["library", "--check", "--library", str(library_dir)])
    out = capsys.readouterr().out
    assert "ok   fly/seg" in out and "MISS worm/img" in out


def test_an_unknown_specimen_is_an_error(library_dir):
    with pytest.raises(SystemExit, match="no volumes for specimen"):
        main(["library", "fish", "--library", str(library_dir)])
