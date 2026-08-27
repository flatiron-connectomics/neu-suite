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

# The console scripts. blockrun is a library and declares no [project.scripts], so it
# has no CLI page — it appears in the design notes instead.
COMMANDS = [
    ("neu-vol", "neu_vol.cli", "the volumes themselves: inspect, convert, copy, "
                                      "downsample, create, write, renumber, pack"),
    ("neu-morpho", "neu_morpho.cli", "meshes and skeletons from a segmentation"),
    ("neu-mark", "neu_mark.cli", "DVID annotations into tables: synapses and other "
                                      "point annotations, per-body records"),
    ("neu-glance", "neu_glance.cli", "neuroglancer states, annotation layers and links"),
]

# Markdown that lives in the repos and is included here verbatim, so it is written and
# reviewed next to the code it describes rather than copied by hand into the site.
INCLUDED = [
    ("neu-lib/README.md", "neu-lib-readme.md"),
    ("blockrun/README.md", "blockrun-readme.md"),
    ("blockrun/docs/dask-slurm.md", "dask-slurm.md"),
    ("neu-vol/README.md", "neu-vol-readme.md"),
    ("neu-morpho/README.md", "neu-morpho-readme.md"),
    ("neu-morpho/docs/skeletonization.md", "neu-morpho-skeletonization.md"),
    ("neu-morpho/docs/measure-calibration.md", "neu-morpho-measure-calibration.md"),
    ("neu-mark/README.md", "neu-mark-readme.md"),
    ("neu-glance/README.md", "neu-glance-readme.md"),
    ("neu-draw/README.md", "neu-draw-readme.md"),
]


# The importable package inside each repository, **in dependency order** — which is the
# order the API reference lists them in, and the reason it is declared here rather than
# in `conf.py`: the repositories are what this file already knows about, and the
# generated landing page, the toctree and the parsed source must not disagree about
# which packages exist or how they are layered.
PACKAGES = [
    ("neu-lib", "neu_lib",
     "The vocabulary every tier shares. numpy and nothing else.",
     "`BBox`, `Frame`, `Mesh`, `Skeleton`, `ScaleInfo`, `align_box`, `skeleton_tube`"),
    ("blockrun", "blockrun",
     "The dask/SLURM substrate. Knows nothing about electron microscopy.",
     "`block_map`, `iter_blocks`, `Manifest`, `start_dask`"),
    ("neu-vol", "neu_vol",
     "Volume I/O: tensorstore backends, storage profiles, the conversion ops, "
     "source metadata.",
     "`convert`, `open_backend`, `scale_spec`, `read_scales`, `describe`, `location`"),
    ("neu-morpho", "neu_morpho",
     "Per-body meshes and skeletons, published into the volume — and read back out "
     "of it.",
     "`readback`, `measure`, `precomputed`, `occupancy`"),
    ("neu-mark", "neu_mark",
     "DVID annotations into tables, and on into neuroglancer.",
     "`notebook`, `tables`, `explore`, `rule`, `segprops`, `annsource`"),
    ("neu-glance", "neu_glance",
     "Everything a remote viewer consumes: states, layers, links, shaders.",
     "`state.build_state`, `layers`, `shaders`, `sources`"),
    ("neu-draw", "neu_draw",
     "Local 3D rendering in a notebook, on pygfx.",
     "`show`, `build_scene`, `Scene`, `sources`, `Legend`"),
]


def package_dirs() -> list[Path]:
    """The package directories `autoapi` parses, skipping any repo not checked out.

    A missing one must not fail the build: the site is assembled from seven independent
    clones, and a partial checkout is a normal state to build in locally. Which is
    exactly why the API landing page is generated rather than written — a hand-written
    toctree would cite the page of a package that was never parsed, and `-W` turns that
    into a failed build for anyone without all seven.
    """
    return [ROOT / repo / pkg for repo, pkg, _blurb, _entry in PACKAGES
            if (ROOT / repo / pkg).is_dir()]


def present_packages() -> list[tuple[str, str, str, str]]:
    """`PACKAGES`, filtered to what is actually here, in dependency order."""
    return [row for row in PACKAGES if (ROOT / row[0] / row[1]).is_dir()]


def write_api_index(path: Path) -> None:
    """The API reference's landing page: prose, cards, and the toctree of what was parsed.

    Generated for the reason `package_dirs` gives — the toctree must list exactly the
    packages that were parsed, and nothing else.
    """
    present = present_packages()
    out = [
        "# API reference",
        "",
        "Every module of every package, with its classes, functions and docstrings.",
        "",
        "This is **parsed from the source, never imported**, which is worth knowing "
        "because it is what makes the reference complete. Half of this suite cannot be "
        "installed from PyPI at all — `tensorstore`, `vol2mesh`, `dvidutils`, "
        "`kimimaro` and `neuclease` are conda-only, and `neu-draw` needs a GPU stack "
        "the documentation build has no use for. An importing reference would have to "
        "either stub all of that out or quietly omit whatever failed to import; this "
        "one reads the files.",
        "",
        "```{admonition} What it is not",
        ":class: note",
        "",
        "A curated public API. Everything is here, private helpers and all, because "
        "these packages are read as often as they are called — the reason a function "
        "does what it does is usually in its docstring, and that is the thing worth "
        "coming here for. For the surface that is meant to be *called*, start from each "
        "package's README, or from the table below.",
        "",
        "The one deliberate omission is the `cli` modules. Those are documented by the "
        "[CLI reference](../cli/index.md), rendered from the real `ArgumentParser` "
        "objects, which is a better page than an API listing of argparse plumbing.",
        "```",
        "",
        "## The packages, in dependency order",
        "",
        "Nothing lower may import from anything higher, and packages at the same tier "
        "do not import each other.",
        "",
        "::::{grid} 2",
    ]
    for _repo, pkg, blurb, entry in present:
        out += [f":::{{grid-item-card}} {{doc}}`{pkg} <../api/{pkg}/index>`",
                blurb, "", f"**Start at** {entry}", ":::"]
    out += ["::::", ""]

    # Keyed on the package each row points into, so a partial checkout drops the row
    # rather than emitting a `{doc}` reference to a page that was never generated.
    starts = [
        ("neu_lib", "do box or grid arithmetic",
         "{doc}`neu_lib.grid <../api/neu_lib/grid/index>` — `BBox`, `align_box`, "
         "`lcm_grid`"),
        ("neu_lib", "turn voxels into nanometres",
         "{doc}`neu_lib.frame <../api/neu_lib/frame/index>` — `Frame`, `to_xyz`"),
        ("blockrun", "run something per block, resumably",
         "{doc}`blockrun.engine <../api/blockrun/engine/index>` — `block_map`, "
         "`iter_blocks` — with {doc}`blockrun.manifest <../api/blockrun/manifest/index>`"),
        ("neu_vol", "open a volume and read a region",
         "{doc}`neu_vol.backends <../api/neu_vol/backends/index>` — `open_backend`, "
         "and `scale_spec` for the level"),
        ("neu_vol", "ask what a volume *is*",
         "{doc}`neu_vol.source_metadata <../api/neu_vol/source_metadata/index>`, "
         "{doc}`neu_vol.scales <../api/neu_vol/scales/index>`"),
        ("neu_vol", "read or write a store, local or `s3://`",
         "{doc}`neu_vol.location <../api/neu_vol/location/index>`"),
        ("neu_vol", "convert or copy a source",
         "{doc}`neu_vol.ops.convert <../api/neu_vol/ops/convert/index>`"),
        ("neu_morpho", "read published meshes and skeletons back",
         "{doc}`neu_morpho.readback <../api/neu_morpho/readback/index>`"),
        ("neu_morpho", "measure morphology of published output",
         "{doc}`neu_morpho.measure <../api/neu_morpho/measure/index>`"),
        ("neu_mark", "pull DVID annotations into DataFrames",
         "{doc}`neu_mark.notebook <../api/neu_mark/notebook/index>`"),
        ("neu_mark", "inspect and parse neuron names",
         "{doc}`neu_mark.explore <../api/neu_mark/explore/index>`, "
         "{doc}`neu_mark.rules <../api/neu_mark/rules/index>`"),
        ("neu_glance", "build a neuroglancer state or layer",
         "{doc}`neu_glance.state <../api/neu_glance/state/index>`, "
         "{doc}`neu_glance.layers <../api/neu_glance/layers/index>`"),
        ("neu_draw", "draw meshes and skeletons in a notebook",
         "{doc}`neu_draw.scene <../api/neu_draw/scene/index>` — `build_scene`, `Scene` "
         "— then `neu_draw.show`"),
    ]
    here = {pkg for _repo, pkg, _b, _e in present}
    out += [
        "## Where to start, by what you are doing",
        "",
        "| you want to | reach for |",
        "| --- | --- |",
    ]
    out += [f"| {want} | {where} |" for pkg, want, where in starts if pkg in here]
    out += [
        "",
        "## Two conventions that run through all of it",
        "",
        "Both fail *silently* when broken, so they are worth knowing before reading any "
        "signature:",
        "",
        "- **zyx in memory, xyz on disk.** Every region argument, `Mesh.vertices_zyx` "
        "and kimimaro's vertices are zyx. Both precomputed formats *store* xyz, and the "
        "flip happens at the boundary. Getting it wrong mirrors output through the z=x "
        "diagonal.",
        "- **One model space: physical nanometres**, via each level's real voxel size — "
        "never an assumed `2 ** level` factor, because real pyramids are anisotropic.",
        "",
        "---",
        "",
        "Looking for a module by name? The {ref}`module index <modindex>` lists every "
        "one alphabetically, and the search box knows every docstring.",
        "",
        "```{toctree}",
        ":maxdepth: 2",
        ":hidden:",
        "",
    ]
    out += [f"../api/{pkg}/index" for _repo, pkg, _b, _e in present]
    out += ["```", ""]
    path.write_text("\n".join(out))


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
        # A run already introduced by `::` needs nothing added, and adding it anyway
        # renders the marker itself as the first line of the code block. No help text
        # does this today; it is cheap to be idempotent and the failure is silent.
        introduced = any(ln.strip() for ln in reversed(out)) and next(
            ln for ln in reversed(out) if ln.strip()).rstrip().endswith("::")
        starts_block = (lines[i][:1] == " " and lines[i].strip()
                        and (not out or not out[-1].strip()) and not introduced)
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
    """Target of the ``argparse`` directive on the neu-vol page."""
    return documented_parser("neu_vol.cli")


def em_morpho_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the neu-morpho page."""
    return documented_parser("neu_morpho.cli")


def em_annot_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the neu-mark page."""
    return documented_parser("neu_mark.cli")


def neu_glance_parser() -> argparse.ArgumentParser:
    """Target of the ``argparse`` directive on the neu-glance page."""
    return documented_parser("neu_glance.cli")


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
        "Every subcommand of all four commands, with its synopsis. Generated from the "
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

    Only `neu-morpho`'s stages so far. They belong here because "which stages do I pass"
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

    An included file's links (`LICENSE`, `docs/skeletonization.md`, `../blockrun`) resolve
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
        # A sibling repo (`../blockrun`) leaves this repo entirely; anything else is
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


def _hide_source_links(app, pagename, _templatename, context, _doctree) -> None:
    """Drop the "view source" / "edit this page" links from generated pages.

    The theme emits them for every page from `source_repository` alone, so a page with
    no committed source gets a link to a file that is not in the repository — a 404 that
    looks exactly like a working link. That is true of the cheat sheet, of every included
    README, and of all ~130 pages of the API reference, whose `.rst` exists only for the
    duration of the build.

    The theme's own gate is `page_source_suffix`, so clearing it takes the template's
    "no source" branch rather than fighting it. The hand-written pages keep their links,
    which is the point: those are the ones worth editing.
    """
    prefixes = (f"{GENERATED.name}/", f"{app.config.autoapi_root}/")
    if pagename.startswith(prefixes):
        context["page_source_suffix"] = ""
        context["show_source"] = False
        context["has_source"] = False


def setup(app):
    def generate(_app):
        GENERATED.mkdir(exist_ok=True)
        write_cheatsheet(GENERATED / "cheatsheet.md")
        write_api_index(GENERATED / "api-index.md")
        copy_included(GENERATED)

    app.connect("builder-inited", generate)
    app.connect("html-page-context", _hide_source_links)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
