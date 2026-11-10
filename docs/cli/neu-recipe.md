# `neu-recipe`

From `neu-recipe`. Runs a recipe — an ordered list of the other commands' command lines —
one step at a time, and writes a record of each run beside the recipe: the commands as
resolved, every step's log and exit code, the git state of the code and the exact
environment. Volumes are named in a library rather than by location. The recipe format,
the run record and the library are described in the
[neu-recipe README](../_generated/neu-recipe-readme.md).

`python -m neu_recipe` is equivalent.

```{argparse}
:module: clitools
:func: neu_recipe_parser
:prog: neu-recipe
```
