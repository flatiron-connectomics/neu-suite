"""What the documentation build depends on: the parser is reachable without running
anything, and every subcommand is reachable from it, or it never gets documented."""

import argparse

import pytest


def test_every_subcommand_is_reachable_from_the_parser():
    from neu_mark import cli

    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser) and parser.prog == "neu-mark"
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert "help" in subs
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_help_is_a_subcommand_as_well_as_a_flag(capsys):
    from neu_mark import cli

    assert cli.main(["help"]) == 0
    assert "usage: neu-mark" in capsys.readouterr().out

    assert cli.main(["help", "points"]) == 0
    printed = capsys.readouterr().out
    assert "usage: neu-mark points" in printed
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["points", "--help"])
    assert capsys.readouterr().out == printed          # one rendering, not two


def test_help_for_something_that_is_not_a_command_lists_the_real_ones(capsys):
    from neu_mark import cli

    assert cli.main(["help", "nosuch"]) == 2
    assert "invalid choice: 'nosuch'" in capsys.readouterr().err


NOISE = (b"E1016 01:27:34.416375 3591857 AuthCredentialsProvider:6146] static: Profile "
         b"credentials provider could not load a profile at .\n")


@pytest.mark.parametrize("flag, shown", [([], False), (["--store-logs"], True)])
def test_the_s3_drivers_benign_lines_are_filtered_unless_asked(monkeypatch, capfd, flag,
                                                              shown):
    import os

    from neu_mark import cli

    def noisy(args):
        os.write(2, NOISE)
        os.write(2, b"a real message\n")
        return 0

    monkeypatch.setattr(cli, "cmd_info", noisy)
    assert cli.main([*flag, "info", "--src", "dvid://dvid.example.org/abc/syn"]) == 0
    err = capfd.readouterr().err
    assert ("AuthCredentialsProvider" in err) is shown and "a real message" in err
