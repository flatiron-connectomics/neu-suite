# Getting started

## The environment

One conda environment, `neu-env`, covers all three packages, installed editable so a
change to any of them is live everywhere.

```bash
conda activate neu-env
neu-vol --help
neu-morpho --help
```

Building one from scratch:

```bash
conda create -n neu-env -c flyem-forge -c conda-forge python=3.12 \
    vol2mesh dvidutils kimimaro tensorstore zarr dask distributed dask-jobqueue \
    numpy scipy h5py tifffile imageio pandas pyarrow ngff-zarr jsonschema pyyaml
conda activate neu-env
pip install --no-deps -e ./blockrun -e ./neu-vol -e ./neu-morpho
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

The three repositories must stay siblings — they depend on each other by relative
`../sibling` path.

```{note}
`dvidutils` is invisible to `importlib.metadata`. Its flyem-forge package ships no
`.dist-info`, so `pip check` and `metadata.version()` report it missing while
`import dvidutils` works fine. Not worth chasing.
```

## Running on the cluster

Both commands take `--config`, naming either a bundled template or a path to your own
YAML. It is repeatable and deep-merged left to right, so a site config carries only the
keys that differ from a template rather than being a fork of it. Unrecognised keys raise
rather than merging silently, and the effective merged config is written into the work
directory.

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
`PERMISSION_DENIED` or `AccessDenied`. Both commands filter the known-benign lines by
default; pass `--store-logs` to see them.
```
