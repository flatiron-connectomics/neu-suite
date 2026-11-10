"""neu-recipe: run a recipe of commands in order, and record every run."""

from __future__ import annotations

import argparse
import contextlib
import shlex
import sys
from pathlib import Path

DESCRIPTION = """\
Run a recipe: an ordered list of command lines, with volumes named in a library and
variables substituted in. Steps run one at a time, the first failure stops the run, and
every run leaves a record beside the recipe (runs/<recipe>_<time>/).

    neu-recipe run recipe.toml --dry-run      # resolve and print; run nothing
    neu-recipe run recipe.toml                # run every step
    neu-recipe run recipe.toml --from mesh    # resume at a step
    neu-recipe run recipe.toml --only skel    # one step
    neu-recipe status recipe.toml             # the latest run
    neu-recipe library --check                # list the volumes, probing each
"""


def _library_arg(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument("--library", type=Path, metavar="DIR",
                   help=f"the volume library directory {what} (default: $NEU_RECIPE_LIBRARY)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neu-recipe", description=DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a recipe's steps, recording the run",
                       description="Run a recipe's steps in file order, one at a time. "
                                   "The first step to exit non-zero stops the run.")
    r.add_argument("recipe", type=Path)
    r.add_argument("--dry-run", action="store_true",
                   help="resolve and print the commands; run and write nothing")
    which = r.add_mutually_exclusive_group()
    which.add_argument("--from", dest="start", metavar="STEP",
                       help="start at this step and run the rest")
    which.add_argument("--only", metavar="STEP", help="run just this step")
    r.add_argument("--log-flush", type=float, default=0.0, metavar="SECONDS",
                   help="write step logs every SECONDS instead of line by line: for logs "
                        "on a shared filesystem that another machine reads while the run "
                        "writes (default: every line)")
    _library_arg(r, "(overrides the recipe's [recipe] library)")

    s = sub.add_parser("status", help="report on a recipe's latest run",
                       description="Report on the recipe's latest run: each step's state, "
                                   "and for the step running or failed, the end of its "
                                   "log and the output of its progress command. Reads "
                                   "files and runs only that command.")
    s.add_argument("recipe", type=Path)

    lb = sub.add_parser("library", help="list the volume library, optionally probing it",
                        description="List the library's volumes. --check probes each "
                                    "location for its metadata object (one request "
                                    "each; needs neu-vol).")
    lb.add_argument("specimen", nargs="?", help="only this specimen's volumes")
    lb.add_argument("--check", action="store_true",
                    help="probe each location for its metadata object")
    _library_arg(lb, "")
    return p


def _run(args, argv) -> int:
    from .recipe import RecipeError, load, select, warnings
    from .runner import run

    if args.log_flush < 0:
        raise SystemExit("--log-flush takes seconds >= 0 (0 writes every line)")
    recipe = args.recipe.resolve()
    try:
        resolved = load(recipe, library=args.library)
        steps = select(resolved["steps"], start=args.start, only=args.only)
    except (RecipeError, OSError) as e:
        raise SystemExit(f"{args.recipe}: {e}") from None

    for name, entry in resolved["volumes"].items():
        print(f"volume {name} = {entry['name']} -> {entry['location']}")
    for name, value in resolved["vars"].items():
        print(f"var    {name} = {value}")
    for line in warnings(steps):
        print(f"WARNING {line}")
    if args.dry_run:
        for s in steps:
            print(f"\n[{s['index']}] {s['name']}\n    {shlex.join(s['argv'])}")
        print(f"\ndry run: {len(steps)} step(s), nothing run, run from {resolved['cwd']}")
        return 0
    return run(recipe, resolved, steps, argv, log_flush=args.log_flush)


def _library(args) -> int:
    from . import library as lib

    try:
        volumes = lib.load(args.library)
    except lib.LibraryError as e:
        raise SystemExit(str(e)) from None
    shown = {n: v for n, v in volumes.items()
             if args.specimen is None or v["specimen"] == args.specimen}
    if not shown:
        raise SystemExit(f"no volumes for specimen {args.specimen!r}")
    quiet = contextlib.nullcontext()
    if args.check:
        with contextlib.suppress(ImportError):
            from neu_vol.logs import quiet_store_logs
            quiet = quiet_store_logs(True)
    bad = 0
    with quiet:
        try:
            for line, found in lib.listing(shown, check=args.check):
                bad += found is False
                print(line, flush=True)
        except lib.LibraryError as e:
            raise SystemExit(str(e)) from None
    if bad:
        raise SystemExit(f"{bad} location(s) have no metadata object")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args, argv)
    if args.command == "status":
        from .runner import status
        return status(args.recipe.resolve())
    return _library(args)


if __name__ == "__main__":
    raise SystemExit(main())
