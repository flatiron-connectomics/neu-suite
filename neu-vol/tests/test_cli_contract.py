"""What the documentation build depends on, pinned here so it cannot quietly break.

The docs site renders the CLI reference from the real ``ArgumentParser``, so the parser
must be reachable without running it, and importing the module must not need the conda-only
half of the environment.
"""

import argparse
import subprocess
import sys

# conda-only on flyem-forge, or heavy enough that pulling them in would change the docs
# job from "pip install" to "build a conda environment"
CONDA_ONLY = ["vol2mesh", "dvidutils", "kimimaro", "DracoPy", "osteoid", "tensorstore",
              "h5py",
              # neuclease needs libdvid-cpp and vigra, conda-only on flyem-forge, so it
              # can never be a pip dep — and `import neuclease.dvid` costs ~9 s, which
              # would land on every `neu-vol --help`. Both reasons point the same way:
              # the DVID backend must defer its import into the function that needs it.
              "neuclease", "libdvid"]


def test_build_parser_returns_the_parser_without_running_it():
    from neu_vol.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "neu-vol"


def test_every_subcommand_is_reachable_from_the_parser():
    """The reference is built by walking this, so a missing subcommand goes undocumented."""
    from neu_vol import cli

    parser = cli.build_parser()
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert set(subs) == {"info", "convert", "copy", "downsample", "create", "write",
                         "to-hdf5", "progress",
                         "align-bbox", "relabel", "mask-by-value", "wk-mappings",
                         "wk-ls", "help"}
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_help_is_a_subcommand_as_well_as_a_flag(capsys):
    """`neu-vol help write` prints exactly what `neu-vol write --help` does."""
    import pytest

    from neu_vol import cli

    assert cli.main(["help"]) == 0
    assert "usage: neu-vol" in capsys.readouterr().out

    assert cli.main(["help", "write"]) == 0
    subcommand = capsys.readouterr().out
    assert "usage: neu-vol write" in subcommand
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["write", "--help"])
    assert capsys.readouterr().out == subcommand


def test_help_for_something_that_is_not_a_command_lists_the_real_ones(capsys):
    from neu_vol import cli

    assert cli.main(["help", "nosuch"]) == 2, "argparse's own exit code for a bad choice"
    assert "invalid choice: 'nosuch'" in capsys.readouterr().err


def test_importing_the_cli_needs_no_conda_only_package():
    """Nor dask. In a subprocess, since this session has already imported them."""
    probe = CONDA_ONLY + ["dask", "distributed"]
    code = (
        "import sys; import neu_vol.cli; "
        f"print(','.join(m for m in {probe!r} "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    got = set(filter(None, out.split(",")))

    conda = sorted(got & set(CONDA_ONLY))
    assert not conda, (
        f"importing neu_vol.cli now pulls in {conda}. The docs build installs "
        f"from PyPI only; these are conda-only or heavy, so this would break it. Defer "
        f"the import into the function that needs it.")
    heavy = sorted(got & {"dask", "distributed"})
    assert not heavy, (
        f"importing neu_vol.cli now pulls in {heavy}, which is ~1 s added to "
        f"every invocation — including `neu-vol info`, which never builds a cluster. "
        f"Import start_dask inside _client(), not at module scope.")
