"""`neu-morpho run` argument handling: checks argparse cannot express, and --config."""

import pytest


def _driver():
    from neu_morpho import cli

    return cli



BASE_ARGV = ["run", "--src", "x", "--dst", "y", "--work-dir", "z"]



@pytest.mark.parametrize("extra, ok", [
    (["--roi", "0,0,0,64,64,64"], False),        # an ROI must say which scale it is in
    (["--roi-scale", "2"], False),               # a scale with no ROI is a mistake
    (["--roi", "0,0,0,64,64,64", "--roi-scale", "2"], True),
    ([], True),
], ids=["roi-without-scale", "scale-without-roi", "both", "neither"])
def test_roi_and_roi_scale_go_together(extra, ok):
    """An ROI read at the wrong scale processes a fraction of the volume and still
    succeeds, so --roi and --roi-scale are only accepted together."""
    parse = _driver()._parse_args
    if not ok:
        with pytest.raises(SystemExit):
            parse(BASE_ARGV + extra)
        return
    args = parse(BASE_ARGV + extra)
    assert (args.roi is None) == (args.roi_scale is None)



def test_config_defaults_to_the_bundled_local_template_and_is_repeatable(tmp_path):
    cli = _driver()
    assert cli._configs(cli._parse_args(BASE_ARGV)) == ["dask-local"]
    # a real file, because every --config value is resolved at parse time
    site = tmp_path / "site.yaml"
    site.write_text("jobqueue:\n  slurm:\n    account: your-account\n")
    args = cli._parse_args(BASE_ARGV + ["--config", "dask-slurm-example",
                                        "--config", str(site)])
    assert cli._configs(args) == ["dask-slurm-example", str(site)]



def test_bad_config_fails_at_parse_time_not_at_cluster_start():
    """A --config typo fails at parse time, before any network read of --src."""
    cli = _driver()
    with pytest.raises((SystemExit, FileNotFoundError), match="dask-local"):
        cli._parse_args(BASE_ARGV + ["--config", "definitely-not-a-config"])



def test_serial_mode_does_not_require_a_config():
    """--serial runs in-process with no cluster, so a config is meaningless there."""
    cli = _driver()
    args = cli._parse_args(BASE_ARGV + ["--serial", "--config", "not-a-real-config"])
    assert args.serial



def test_a_subcommand_is_required():
    """A bare `neu-morpho` must not silently mean `run`."""
    cli = _driver()
    with pytest.raises(SystemExit):
        cli._parse_args([])

