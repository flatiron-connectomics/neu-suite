"""neu-recipe — run a recipe of commands in order, and record every run.

A recipe (:mod:`neu_recipe.recipe`) is a TOML file of named steps, each a command line,
with variables and volumes substituted in. The runner (:mod:`neu_recipe.runner`) runs
the steps one at a time, stops at the first failure, and writes a run directory holding
the recipe as run, every step's log, the git state of the code involved and the exact
environment. Volumes are named in a library (:mod:`neu_recipe.library`), so a recipe
names what it reads and writes rather than where it is.

Standard library only; ``neu-recipe library --check`` additionally needs neu-vol.
"""

from __future__ import annotations

__version__ = "0.1.0"
