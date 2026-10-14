"""What the docs build needs from the CLI: the parser is reachable without running it,
and importing the CLI loads no conda-only package (the docs job installs from PyPI)."""

import argparse
import subprocess
import sys

CONDA_ONLY = ["vol2mesh", "dvidutils", "kimimaro", "DracoPy", "osteoid", "tensorstore",
              "h5py"]


def test_build_parser_returns_the_parser_without_running_it():
    from neu_morpho.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "neu-morpho"


def _subcommands(parser):
    return next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices


def test_every_subcommand_renders_usage():
    """Top level and the nested `measure` group, which the docs render page by page."""
    from neu_morpho import cli

    top = _subcommands(cli.build_parser())
    assert set(top) == {"run", "progress", "measure", "run-report", "help"}
    inner = _subcommands(top["measure"])
    assert set(inner) == {"volumes", "skeletons", "compartments", "progress"}
    for name, sub in [*top.items(), *(("measure " + k, v) for k, v in inner.items())]:
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_help_is_a_subcommand_as_well_as_a_flag(capsys):
    """`neu-morpho help run` prints exactly what `neu-morpho run --help` prints."""
    import pytest

    from neu_morpho import cli

    assert cli.main(["help"]) == 0
    assert "usage: neu-morpho" in capsys.readouterr().out

    assert cli.main(["help", "run"]) == 0
    printed = capsys.readouterr().out
    assert "usage: neu-morpho run" in printed

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--help"])
    assert capsys.readouterr().out == printed


def test_help_reaches_a_NESTED_subcommand(capsys):
    """`help measure volumes` reaches the nested group."""
    from neu_morpho import cli

    assert cli.main(["help", "measure", "volumes"]) == 0
    assert "usage: neu-morpho measure volumes" in capsys.readouterr().out

    assert cli.main(["help", "measure", "nosuch"]) == 2
    assert "invalid choice: 'nosuch'" in capsys.readouterr().err


def test_every_stage_is_described_in_run_help():
    """`STAGE_DOC` feeds `run --help`, the CLI reference and the cheat sheet, so a stage
    without an entry would go undocumented in all three."""
    from neu_morpho.cli import STAGE_DOC, STAGES, build_parser

    assert tuple(STAGE_DOC) == STAGES
    assert all(doc.strip() for doc in STAGE_DOC.values())
    rendered = build_parser()._run_parser.format_help()
    for name in STAGES:
        assert name in rendered.split("Stages")[1], f"{name} missing from run --help"


def test_importing_the_cli_needs_no_conda_only_package():
    """In a fresh interpreter: no conda-only package, and no dask (startup latency)."""
    probe = CONDA_ONLY + ["dask", "distributed"]
    code = (
        "import sys; import neu_morpho.cli; "
        f"print(','.join(m for m in {probe!r} "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    got = set(filter(None, out.split(",")))

    conda = sorted(got & set(CONDA_ONLY))
    assert not conda, (
        f"importing neu_morpho.cli now pulls in {conda}. The docs build installs from "
        f"PyPI only, and these are conda-only on flyem-forge, so this would break it. "
        f"Defer the import into the function that needs it.")
    heavy = sorted(got & {"dask", "distributed"})
    assert not heavy, (
        f"importing neu_morpho.cli now pulls in {heavy}, which is ~1 s added to every "
        f"invocation — including `neu-morpho progress`, which only reads JSONL. Import "
        f"start_dask inside the branch that starts a cluster, not at module scope.")
