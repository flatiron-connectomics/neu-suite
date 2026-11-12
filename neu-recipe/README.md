# neu-recipe

Run a recipe — an ordered list of command lines — and keep a record of every run, over
volumes named in a library rather than by location.

A dataset release is a dozen commands from several packages run in order: export a
segmentation, select bodies, mesh, skeletonize, publish annotations, build a viewer
state. `neu-recipe` runs them one at a time, stops at the first failure, resumes from any
step, and writes down exactly what ran: the commands as resolved, each step's log and
exit code, the git commit and uncommitted changes of every repository involved, and the
environment's exact packages.

```bash
neu-recipe run release.toml --dry-run      # resolve and print; run nothing
neu-recipe run release.toml                # run every step
neu-recipe run release.toml --from mesh    # resume at a step
neu-recipe run release.toml --only skel    # one step
neu-recipe status release.toml             # the latest run, safe to repeat while it runs
neu-recipe library --check                 # list the volume library, probing each location
```

Standard library only. `library --check` needs neu-vol (`pip install 'neu-recipe[check]'`).

## A recipe

```toml
[recipe]                    # optional; paths are relative to this file
cwd = ".."                  # where steps run (default: this file's directory)
library = "../library"      # the volume library (default: $NEU_RECIPE_LIBRARY)

[volumes]                   # names from the library
image = "fly-a/em"
seg = "fly-a/seg-v2"

[vars]                      # strings, resolved in order; may use volumes and earlier vars
work_dir = "/path/to/work/{seg.name}"
bodies = "{work_dir}/selected_bodies.csv"

[[step]]
name = "mesh"
run = """neu-morpho run --src {seg} --dst {seg} --stages index,mesh
         --allowlist {bodies} --work-dir {work_dir}"""
progress = "neu-morpho progress {work_dir}"   # optional: what `status` runs for this step

[[step]]
name = "state"
run = "neu-glance gen --image {image} --seg {seg} --format state --out {seg}/state.json"
```

- `{seg}` is the volume's location, `{seg.name}` its library name, and `{seg.<field>}`
  any other field of its library entry.
- A reference to anything undefined is an error before anything runs, so a typo never
  reaches a cluster. `{{` and `}}` are literal braces, for a placeholder that a command
  expands itself.
- A command is split as a shell would split it (quotes respected, newlines are spaces)
  and runs without a shell, from `cwd`, so relative paths mean the same thing however the
  runner is started.
- Steps run in file order. A step that exits non-zero stops the run, and the message
  says how to resume.
- A DVID branch ref (`dvid://host/<uuid>:<branch>/...`) gets a warning: a branch's node
  moves, so the run could not be repeated. Pin the node's uuid.

## What a run records

Every run (not a dry run) writes `runs/<recipe>_<YYYYmmddTHHMM>/` beside the recipe:

| file | holds |
| --- | --- |
| `recipe.toml` | the recipe as run |
| `record.json` | resolved volumes, vars and commands; per-step start, end, minutes and exit code; the git state of each repository involved. Rewritten after every step, so a killed run keeps its record |
| `env-explicit.txt` | `mamba list --explicit` (or conda's): the environment's exact packages |
| `pip-freeze.txt` | pip's view, including editable installs |
| `steps/<n>-<name>.log` | each step's combined output, also echoed to the terminal |
| `configs/` | a copy of every file a step passed as `--config` |

The repositories recorded are the one holding the recipe and every one holding an
editable install in the running environment: together, the code a step can run. Each
gets its commit, branch and the tracked files with uncommitted changes.

Steps and `progress` commands run with `$NEU_PROGRESS_DIR` set to `runs/progress/`,
where neu-vol keeps the progress manifest of a remote destination (a local one sits
beside its volume). It is shared by every run, not kept per run, so a run started with
`--from` resumes where the last one stopped. To redo a step from scratch, delete that
destination's manifest there, or pass the tool's own `--fresh`.

Stopping the runner (Ctrl-C, or `kill` on its pid) stops the running step too, rather
than leaving a driver and its cluster running unrecorded. `status` then reports the run
as stopped. Logs are written line by line; `--log-flush N` writes them every N seconds
instead, for a log on a shared filesystem that another machine reads during the run.

## Status

`neu-recipe status release.toml` reads the latest run's record: each step done, failed,
running, interrupted or pending, with its time. For the step running, or the one that
stopped the run, it shows the end of its log and runs the step's `progress` command. It
reads files and runs only those commands, so it is safe to repeat while a run is in
flight. It never queries a scheduler.

## The volume library

Recipes, configs and reports name volumes, never locations. Data moves and deliveries
get re-cut; a name is what everything refers to, and its location changes in one place.

A library is a directory of TOML files, one per specimen:

```toml
# library/fly-a.toml
[volume."fly-a/seg-v2"]
kind = "segmentation"
location = "s3://my-bucket/fly-a/seg-v2"
description = "proofread segmentation, second release"
export_uuid = "0123456789abcdef0123456789abcdef"

[volume."fly-a/em"]
kind = "image"
location = "s3://my-bucket/fly-a/em"
```

`kind` and `location` are required. Any other field is carried through, which is how a
recipe reaches it (`dvid://host/{seg.export_uuid}/labels`). Every load checks that each
name starts with its file's specimen, that names are unique, and that no two names share
a location, since two names for one volume is how two analyses silently disagree about
which one they read.

A library holds data locations, so it usually lives in a private repository beside the
recipes. A recipe names it with `[recipe] library`; scripts and notebooks use
`$NEU_RECIPE_LIBRARY`, or pass the directory:

```python
from neu_recipe import library

volumes = library.load("path/to/library")
library.location("fly-a/seg-v2", volumes)
```

## Why not Snakemake (or Make, Nextflow, ...)

Those tools decide *what* needs to run, from a graph of rules over files. That is not
the problem here, and their model fits badly:

- **"Done" is not a file.** A step writes into a volume of millions of objects on a
  store, and the tools themselves decide what is finished from their own manifests and
  resume checks. A marker file per step would be a second record of completion that can
  disagree with the first.
- **Scheduling is already handled.** Each step starts its own dask cluster from its
  `--config`; a workflow engine's cluster executor would sit on top of that, not
  replace it.
- **The record that matters is different.** For a release, what has to survive is the
  commit and uncommitted changes of each repository and the exact environment for the
  whole run, which these tools do not keep.

So the scope is deliberately small: an ordered list of commands, resolved names, a
complete record and a status report. No dependency graph, no parallel steps, no
caching. If automatic rebuilds of whatever is out of date ever become the need, that is
the point to reach for Snakemake.
