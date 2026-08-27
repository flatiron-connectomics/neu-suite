# Getting started

## The environment

One conda environment, `neu-env`, covers all seven packages, installed editable so a
change to any of them is live everywhere.

```bash
conda activate neu-env
neu-vol --help          # the volumes themselves
neu-morpho --help       # meshes and skeletons
neu-mark --help         # DVID annotations into tables
neu-glance --help       # neuroglancer states, layers and links
```

Building one from scratch:

```bash
conda create -n neu-env -c flyem-forge -c conda-forge python=3.12 \
    vol2mesh dvidutils kimimaro tensorstore zarr dask distributed dask-jobqueue \
    numpy scipy h5py tifffile imageio pandas pyarrow ngff-zarr jsonschema pyyaml
conda activate neu-env
pip install --no-deps -e ./neu-lib -e ./blockrun -e ./neu-vol -e ./neu-morpho \
                      -e ./neu-mark -e ./neu-glance -e ./neu-draw
```

```{warning}
`--no-deps` on the editable installs is load-bearing. The `pyproject.toml` files declare
real runtime dependencies, and without it pip re-resolves conda-provided binaries —
tensorstore and h5py — from PyPI, which invites an ABI mismatch that surfaces much later
as a segfault rather than an install error.
```

**Python 3.12 is required**, and it is `vol2mesh` and `dvidutils` that force it. Both are
py312-only *and* conda-only on flyem-forge: there is no py313 build and no PyPI
equivalent, so they can never be pip dependencies. For the same reason they are omitted
from `neu-morpho`'s declared dependencies.

The seven repositories must stay siblings — they depend on each other by relative
`../sibling` path.

```{note}
`dvidutils` is invisible to `importlib.metadata`. Its flyem-forge package ships no
`.dist-info`, so `pip check` and `metadata.version()` report it missing while
`import dvidutils` works fine. Not worth chasing.
```

### Only some of it needs all of that

`neu-lib` is **numpy and nothing else** — deliberately, since it holds the types every
other package names, and it is what keeps that vocabulary installable on 3.11 while the
rest of the suite is pinned to 3.12. Two other pieces have their own requirements:

`neu-draw` renders locally and so needs a GPU stack (`pygfx`, `wgpu`, `jupyter-rfb`,
`ipywidgets`); nothing else in the suite imports it. `neu-mark`'s DVID sources need
`neuclease`, which is conda-only for the same reason `vol2mesh` is — it depends on
`libdvid-cpp`, and libdvid is what inflates DVID's compressed label blocks. There is
therefore no `dvid` pip extra, and there cannot be one.

## Running on the cluster

`neu-vol` and `neu-morpho` are the two commands that distribute work, and both take
`--config`, naming either a bundled template or a path to your own YAML. It is repeatable
and deep-merged left to right, so a site config carries only the keys that differ from a
template rather than being a fork of it. Unrecognised keys raise rather than merging
silently, and the effective merged config is written into the work directory. The
templates ship with **blockrun**, next to `start_dask`, so every consumer shares one set.

`neu-mark` and `neu-glance` run in one process and need none of this.

```bash
# smoke test in one process first — no dask, no cluster
neu-vol convert --src ... --dst ... --serial --single-level

# then the real thing, surviving logout
nohup env PYTHONUNBUFFERED=1 neu-vol convert --src ... --dst s3://... \
    --config dask-slurm-example --config ~/my-site.yaml --workers 48 > run.log 2>&1 &
squeue -u "$USER"
```

`PYTHONUNBUFFERED=1` is the console-script equivalent of `python -u`. Without it the log
lags a long run in 8 KB blocks, which makes a healthy run look hung.

## Object stores

Only some destinations may be remote, and where that is true it is stated per command.
Writes go through tensorstore's kvstore layer, which needs AWS credentials bootstrapped
**per process** — the packages do this for you at every point a store is opened.

```{note}
`AuthCredentialsProvider` lines at `E` severity are **noise, not failures**. They are
emitted on every successful S3 open and record the two credential providers that failed
before the environment provider succeeded. The marker of a real problem is
`PERMISSION_DENIED` or `AccessDenied`. The commands filter the known-benign lines by
default; pass `--store-logs` to see them. `neu-draw` filters them too, without being
asked, because a notebook has no flag to pass.
```

## Where to go next

- The [cheat sheet](_generated/cheatsheet.md) — every subcommand on one page.
- The [CLI reference](cli/index.md) — flag by flag, rendered from the real parsers.
- The [API reference](_generated/api-index.md) — calling any of it from Python.
- The guides, which follow whole tasks rather than single commands: [converting a
  source](guides/converting.md), [assembling small pieces into one
  frame](guides/small-pieces.md), [ground truth to
  meshes](guides/ground-truth.md), [viewing the results in
  neuroglancer](guides/viewing.md), and [rendering locally in a
  notebook](guides/rendering.md).
