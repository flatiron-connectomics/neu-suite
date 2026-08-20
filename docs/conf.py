"""Sphinx configuration for the neu-suite documentation site.

The site assembles three sibling repositories that are deliberately separate packages —
`blockrun`, `neu-vol`, `neu-morpho` — into one place to read about them,
because the thing a user needs (a command) does not correspond to a repository.

**The CLI reference is generated, never written.** `sphinx-argparse` renders the actual
`ArgumentParser` objects from `build_parser()` in each package, so a published flag
cannot disagree with `--help`. That is the whole reason this is a Sphinx site rather
than a hand-maintained page.

Building requires the three packages to be importable, but *not* the full `neu-env`
conda environment: importing the CLI modules pulls in only dask, distributed and numpy —
no tensorstore, vol2mesh, dvidutils or kimimaro — so a plain pip install is enough. See
`.github/workflows/docs.yml`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "neu-suite"
author = "Flatiron Institute — Center for Computational Neuroscience"
copyright = "Flatiron Institute"

extensions = [
    "myst_parser",          # the repos' docs are markdown; read them as they are
    "sphinxarg.ext",        # the generated CLI reference
    "sphinx_design",        # the cards on the landing page
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
suppress_warnings = ["myst.xref_missing", "myst.header"]
