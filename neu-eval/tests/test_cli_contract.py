"""What the CLI promises: the parser exists, `--help` is cheap, and the flags are wired.

``build_parser`` is a **hard contract with the docs site**, which imports this module and
renders the reference from the real parser object so the published `--help` cannot drift
from the code. The site builds with no conda and no tensorstore, so importing the module
must not reach for either — which is what the ``sys.modules`` assertions below are for, and
the reason neu-vol and neu-morpho carry the same pair of tests.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from neu_eval import cli


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# -- the docs contract -------------------------------------------------------------------

def test_build_parser_exists_and_is_named_what_the_docs_look_for():
    parser = cli.build_parser()
    assert parser.prog == "neu-eval"


def test_importing_the_cli_does_not_pull_in_a_store_stack():
    """The docs job has no tensorstore, so an import-time dependency breaks the build."""
    leaked = _run(
        "import sys, neu_eval.cli;"
        "print(','.join(m for m in ['neu_vol','tensorstore','h5py','pandas','dask'] "
        "if m in sys.modules))")
    assert leaked == ""


def test_help_does_not_pay_for_the_reading_stack():
    leaked = _run(
        "import sys; from neu_eval.cli import build_parser; build_parser();"
        "print(','.join(m for m in ['neu_vol','tensorstore','h5py','pandas'] "
        "if m in sys.modules))")
    assert leaked == ""


# -- the subcommands ---------------------------------------------------------------------

def test_compare_is_the_subcommand_and_it_requires_both_sides():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["compare"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["compare", "--reference", "x.h5"])


def test_the_parsed_defaults_match_the_library_defaults():
    """A CLI default that drifts from the function's is a silent behaviour change."""
    from neu_eval.compare import compare
    import inspect

    args = cli.build_parser().parse_args(
        ["compare", "--reference", "a.h5", "--segmentation", "b.h5"])
    signature = inspect.signature(compare).parameters
    assert args.min_frac == signature["min_frac"].default
    assert args.min_voxels == signature["min_voxels"].default
    assert args.top == signature["top"].default
    assert args.rank == signature["rank"].default


def test_the_reference_side_ignores_zero_by_default_and_the_other_does_not():
    args = cli.build_parser().parse_args(
        ["compare", "--reference", "a.h5", "--segmentation", "b.h5"])
    assert cli._ids(args.ignore_reference) == (0,)
    assert cli._ids(args.ignore_segmentation) == ()


def test_an_unknown_rank_is_refused_by_argparse():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["compare", "--reference", "a.h5", "--segmentation", "b.h5",
             "--rank", "voxels"])


def test_help_lists_the_commands():
    assert cli.cmd_help(cli.build_parser().parse_args(["help"])) == 0


def test_help_for_an_unknown_topic_is_an_error_not_a_crash():
    assert cli.cmd_help(cli.build_parser().parse_args(["help", "nope"])) == 2


# -- the little parsers ------------------------------------------------------------------

def test_id_lists_tolerate_an_empty_string():
    assert cli._ids("") == ()
    assert cli._ids("0") == (0,)
    assert cli._ids("0,3, 7") == (0, 3, 7)


def test_voxel_sizes_parse_as_floats_not_ints():
    """They are nanometres and 6.8 is a real voxel size."""
    assert cli._floats("6.8,8,8") == (6.8, 8.0, 8.0)


# -- dataset pairing ---------------------------------------------------------------------

def test_a_plain_source_is_passed_through_unchanged():
    assert cli._paired_source("s3://my-bucket/vol", "crop_a") == "s3://my-bucket/vol"
    assert cli._paired_source("some.h5:/already", "crop_a") == "some.h5:/already"
    assert cli._paired_source("some.h5", None) == "some.h5"


def test_a_container_holding_the_name_is_addressed_to_it(tmp_path):
    """The --all-datasets case where BOTH sides are per-crop containers."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    path = tmp_path / "seg.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("crop_a", data=np.zeros((2, 2, 2), np.uint64))
    assert cli._paired_source(str(path), "crop_a") == f"{path}:/crop_a"
    assert cli._paired_source(str(path), "absent") == str(path)


def test_a_non_hdf5_file_is_left_alone(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not hdf5")
    assert cli._paired_source(str(path), "crop_a") == str(path)
