"""Sphinx configuration for the neu-suite documentation site.

The site assembles the suite's seven sibling repositories — deliberately separate
packages, in a strict one-way dependency — into one place to read about them, because
the thing a user needs (a command, or a function) does not correspond to a repository.
`clitools.PACKAGES` is the single list of what those seven are, in dependency order.

**Nothing here is written by hand twice.** The CLI reference renders the actual
`ArgumentParser` objects from `build_parser()` in each package, so a published flag
cannot disagree with `--help`; the API reference parses the source; the cheat sheet and
the API landing page are generated from the same declarations. That is the whole reason
this is a Sphinx site rather than a set of hand-maintained pages.

Building needs **no conda environment**, which is the constraint the whole build is
shaped around. Two different mechanisms get there:

* the *CLI* reference imports the four `cli` modules, which pull in dask, distributed,
  numpy, pandas and pyarrow and nothing heavier — never tensorstore, vol2mesh,
  dvidutils, kimimaro or neuclease, all of which are conda-only on flyem-forge. A test
  in `neu-vol` pins that property so it cannot quietly stop being true.
* the *API* reference never imports anything at all — `autoapi` parses the files — so it
  covers the conda-only modules, and covers `neu-draw` without a GPU stack in a job that
  renders no pixels.

See `.github/workflows/docs.yml`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

import clitools  # noqa: E402  — the `_ext` path above is what makes this importable

project = "neu-suite"
author = "Flatiron Institute — Center for Computational Neuroscience"
copyright = "Flatiron Institute"

extensions = [
    "myst_parser",          # the repos' docs are markdown; read them as they are
    "sphinxarg.ext",        # the generated CLI reference
    "sphinx_design",        # the cards on the landing page
    "sphinx_copybutton",    # this site is mostly commands; make them copyable
    "clitools",             # cheat sheet + copying repo markdown into the build
]

# `colon_fence` lets directives be written as ::: blocks inside markdown, and
# `deflist`/`linkify` are the two MyST niceties the existing docs already assume.
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "_ext", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "neu-suite"
html_static_path = ["_static"]

# Cropped from the pathway figure on the front page, to the medulla and the R7p neuron in
# it. A browser renders this at 16-32 px, where the whole figure is an indistinct grey —
# the green disc with a yellow mark is the only part of it that survives that downscale.
html_favicon = "_static/favicon.png"
html_theme_options = {
    "source_repository": "https://github.com/flatiron-connectomics/neu-suite/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Included repo markdown is copied in at build time and links inside it point at paths
# that only exist in its own repository; those are not broken links in the site's terms.
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    # `neu_morpho.cli` imports `link_subresources` from `neu_morpho.precomputed`, which
    # re-exports it from `neu_vol.ops.subresources` — a deliberate re-export, so that
    # editing a volume's `info` lives in the package that owns volumes while existing
    # call sites are unchanged. `autoapi` follows one hop of that statically and not
    # two across a package boundary, so this warning is a fact about static parsing
    # rather than about the code. Scoped to the one category, so a NEW unresolvable
    # import still shows up.
    "autoapi.python_import_resolution",
]

# Single backticks in a docstring mean CODE here — measured: 281 uses across the suite,
# every one of them a flag, a path, a field or a call. RST's own default would render
# them as italic title references pointing at nothing.
default_role = "literal"


# --------------------------------------------------------------------------- #
# The API reference
# --------------------------------------------------------------------------- #
# `autoapi` PARSES the source rather than importing it, and that is the whole reason it
# is used here instead of `autodoc`. Two consequences this site depends on:
#
#   * no `autodoc_mock_imports` list to keep in step with tensorstore, vol2mesh,
#     dvidutils, kimimaro, neuclease, pygfx and wgpu — none of which are installable
#     from PyPI alone, which is the constraint the whole build is built around;
#   * a package that is CHECKED OUT but not INSTALLED is documented anyway, which is
#     exactly `neu-draw`'s situation in CI: it is cloned for its README, and installing
#     it would pull a GPU stack into a job that renders no pixels.
extensions.append("autoapi.extension")

autoapi_dirs = [str(p) for p in clitools.package_dirs()]
autoapi_root = "api"
# The `cli` modules are documented by the CLI reference, rendered from the real parsers,
# and that is strictly the better page: an API page for them shows argparse plumbing.
# There is also a hard conflict. A module docstring that IS an argparse description gets
# rendered twice — raw here, and through `clitools._rst_safe` there — and the two want
# opposite things: RST needs `::` before an indented example, and `--help` then prints a
# bare `::` to the terminal. The help text is written for the terminal first, so the
# reference is what gives way.
autoapi_ignore = ["*/cli.py", "*/__main__.py"]
# The generated tree is cited by hand from `index.md`, so it sits under Reference with
# everything else rather than being appended to the sidebar on its own.
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"      # the class docstring AND __init__'s
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
