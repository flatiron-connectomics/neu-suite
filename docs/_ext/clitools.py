"""Build-time generation: the cheat sheet, and the markdown pulled in from the repos.

Everything here runs on Sphinx's ``builder-inited`` event, so the generated pages exist
before the build reads them. Nothing generated is committed — the point of generating it
is that it cannot fall out of step with the code, and a committed copy would reintroduce
exactly that risk.
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path


def posixpath_dirname(rel: str) -> str:
    """The directory an included file sits in, within its own repository."""
    return posixpath.dirname(rel)

HERE = Path(__file__).resolve().parent.parent          # docs/
ROOT = HERE.parent                                     # the workspace root
GENERATED = HERE / "_generated"

# The console scripts. em-blockrun is a library and declares no [project.scripts], so it
# has no CLI page — it appears in the design notes instead.
COMMANDS = [
    ("em-vol", "em_volume_tools.cli", "the volumes themselves: inspect, convert, copy, "
                                      "create, write, renumber"),
    ("em-morpho", "em_seg_morpho.cli", "meshes and skeletons from a segmentation"),
    ("em-annot", "em_annotation.cli", "DVID annotations into tables: synapses and other "
                                      "point annotations, per-body records"),
    ("em-ngl", "em_ngl.cli", "neuroglancer states, annotation layers and links"),
]

# Markdown that lives in the repos and is included here verbatim, so it is written and
# reviewed next to the code it describes rather than copied by hand into the site.
INCLUDED = [
    ("em-blockrun/README.md", "em-blockrun-readme.md"),
    ("em-volume-tools/README.md", "em-volume-tools-readme.md"),
    ("em-volume-tools/docs/DESIGN.md", "em-volume-tools-design.md"),
    ("em-volume-tools/docs/dask-slurm-rusty.md", "dask-slurm-rusty.md"),
    ("em-seg-morpho/README.md", "em-seg-morpho-readme.md"),
    ("em-seg-morpho/docs/DESIGN.md", "em-seg-morpho-design.md"),
    ("em-annotation/README.md", "em-annotation-readme.md"),
    ("em-ngl/README.md", "em-ngl-readme.md"),
]


def _subcommands(parser: argparse.ArgumentParser):
    """``(name, subparser)`` for each subcommand, in the order the CLI declares them."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # `choices` preserves declaration order; `_choices_actions` carries the help
            # strings, which are not on the subparser itself.
            helps = {a.dest: (a.help or "") for a in action._choices_actions}
            for name, sub in action.choices.items():
                yield name, sub, helps.get(name, "")
            return


def _load(module: str) -> argparse.ArgumentParser:
    import importlib

    mod = importlib.import_module(module)
    builder = getattr(mod, "build_parser", None)
    if builder is None:
        # This site is built against the package repositories' `main`, so it is one
        # version skew away from a confusing failure: the docs can be newer than the
        # package they document. Raised by name because the underlying symptom is a
        # bare AttributeError inside a Sphinx event handler, which says nothing about
        # what to do.
        raise RuntimeError(
            f"{module} has no build_parser(). The CLI reference renders the real "
            f"ArgumentParser, so the package needs the commit that split "
            f"build_parser() out of _parse_args — and this site builds against the "
            f"package repositories' main branch, so that commit has to be PUSHED, not "
            f"just committed locally. Installed from: {getattr(mod, '__file__', '?')}")
    return builder()


def _rst_safe(text: str) -> str:
    """Mark indented blocks in help text as RST literal blocks.

    argparse descriptions are plain text and ours carry indented command examples whose
    continuation lines are indented further still. reStructuredText reads the block as a
    block quote and the deeper continuation as a nested one, which is an "Unexpected
    indentation" error — the page renders, but on an error, and it renders the examples
    as quotes rather than code.

    Prefixing each indented run with ``::`` makes it a literal block, which tolerates
    any internal indentation and displays as code. This happens **only** for the
    documentation build: `--help` in the terminal is untouched, which is the point —
    the help text is written for the terminal first.
    """
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        starts_block = (lines[i][:1] == " " and lines[i].strip()
                        and (not out or not out[-1].strip()))
        if not starts_block:
            out.append(lines[i])
            i += 1
            continue
        out += ["::", ""]
        while i < len(lines) and (not lines[i].strip() or lines[i][:1] == " "):
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _walk(parser: argparse.ArgumentParser):
    yield parser
    for _name, sub, _help in _subcommands(parser):
        yield from _walk(sub)


def documented_parser(module: str) -> argparse.ArgumentParser:
    """The real parser, with its prose made safe for the RST renderer."""
    parser = _load(module)
    for p in _walk(parser):
        for attr in ("description", "epilog"):
            value = getattr(p, attr, None)
            if value:
                setattr(p, attr, _rst_safe(value))
    return parser


def em_vol_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the em-vol page."""
    return documented_parser("em_volume_tools.cli")


def em_morpho_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the em-morpho page."""
    return documented_parser("em_seg_morpho.cli")


def em_annot_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the em-annot page."""
    return documented_parser("em_annotation.cli")


def em_ngl_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the em-ngl page."""
    return documented_parser("em_ngl.cli")


def write_cheatsheet(path: Path) -> None:
    """One page listing every subcommand with its synopsis.

    This is the page that answers "what was that flag called" without reading a
    reference. It is generated from the same parser objects the CLI uses, so a
    subcommand cannot appear here with the wrong usage, and a new one cannot be
    forgotten.
    """
    out = [
        "# Cheat sheet",
        "",
        "Every subcommand of both commands, with its synopsis. Generated from the "
        "argparse parsers at build time, so it always matches `--help`.",
        "",
    ]
    for prog, module, blurb in COMMANDS:
        parser = _load(module)
        subs = list(_subcommands(parser))
        out += [f"## `{prog}`", "", f"{blurb}", "",
                "| subcommand | what it does |", "| --- | --- |"]
        out += [f"| [`{prog} {name}`](#{prog}-{name}) | {help_} |"
                for name, _sub, help_ in subs]
        out.append("")
        out += _extras(module)
        for name, sub, help_ in subs:
            usage = " ".join(sub.format_usage().split())
            usage = usage[len("usage: "):] if usage.startswith("usage: ") else usage
            out += [f"### `{prog} {name}`", "", help_, "", "```text", usage, "```", ""]
    path.write_text("\n".join(out))


def _extras(module: str) -> list[str]:
    """Per-command detail worth having on the cheat sheet, taken from the package.

    Only `em-morpho`'s stages so far. They belong here because "which stages do I pass"
    is the question the cheat sheet exists to answer, and `--stages index,mesh,skel`
    tells you nothing about what those are — but the text is imported rather than
    restated, so the site, `--help` and the code cannot disagree.
    """
    import importlib

    doc = getattr(importlib.import_module(module), "STAGE_DOC", None)
    if not doc:
        return []
    out = ["#### Stages", "",
           "Passed to `run --stages` as a comma-separated list, and run in this order. "
           "Each is idempotent and resumable, so re-running a subset continues rather "
           "than redoing it.", "",
           "| stage | what it does |", "| --- | --- |"]
    out += [f"| `{name}` | {text} |" for name, text in doc.items()]
    return out + [""]


GITHUB = "https://github.com/flatiron-connectomics"


def _absolutize(text: str, repo: str, from_dir: str) -> str:
    """Point a repo README's relative links at GitHub.

    An included file's links (`LICENSE`, `docs/DESIGN.md`, `../em-blockrun`) resolve
    against its own repository, not against this site — so left alone they are dangling
    references that warn on every build and 404 for the reader. Rewriting them to the
    repository they mean makes them work, which is better than suppressing the warning.
    """
    import posixpath
    import re

    def fix(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#|/)", target):
            return m.group(0)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        # A sibling repo (`../em-blockrun`) leaves this repo entirely; anything else is
        # relative to the included file's own directory inside it.
        joined = posixpath.normpath(posixpath.join(from_dir, target))
        if joined.startswith("../"):
            return f"[{label}]({GITHUB}/{joined.removeprefix('../')}{anchor})"
        return f"[{label}]({GITHUB}/{repo}/blob/main/{joined}{anchor})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", fix, text)


def copy_included(dest: Path) -> None:
    """Copy repo markdown into the build tree, with relative links absolutised.

    Sphinx will not read sources from outside its source directory, and symlinking
    breaks the GitHub Actions checkout. Copying is the boring option that works both
    locally and in CI.
    """
    for rel, name in INCLUDED:
        src = ROOT / rel
        if src.exists():
            repo = rel.split("/", 1)[0]
            from_dir = posixpath_dirname(rel.split("/", 1)[1])
            (dest / name).write_text(
                _absolutize(src.read_text(), repo, from_dir))
        else:
            # A repo that is not checked out should not fail the build; the toctree
            # entry will warn on its own, which is the visible signal.
            (dest / name).write_text(
                f"# Not available\n\n`{rel}` was not present when this site was "
                f"built.\n")


def setup(app):
    def generate(_app):
        GENERATED.mkdir(exist_ok=True)
        write_cheatsheet(GENERATED / "cheatsheet.md")
        copy_included(GENERATED)

    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
