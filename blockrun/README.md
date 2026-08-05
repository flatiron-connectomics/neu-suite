# em-blockrun

Resumable, chunked **parallel maps on dask** — the same code runs locally on a
workstation or across Flatiron **Rusty/SLURM** by swapping one YAML config.

Extracted from `em-volume-tools` so it can be shared across projects (volume
conversion, meshing/skeletonization, …). Dependency-light: only `dask`,
`distributed`, `dask-jobqueue`, `pyyaml`.

```python
from em_blockrun import start_dask, iter_blocks, block_map, Manifest

blocks = list(iter_blocks(shape=(1024, 1024, 1024), chunks=(128, 128, 128)))
# A bundled template plus your own overlay, deep-merged left to right.
with start_dask(48, ["dask-slurm-example", "my-site.yaml"]) as client:
    results = block_map(blocks, work_fn, client=client)   # batched client.map + as_completed
```

- **`start_dask`** — LocalCluster or SLURMCluster from one or more configs; tears
  down (and `scancel`s) on exit.

### Cluster config

`config_path` takes a **bundled template name or a path**, or a sequence of either,
deep-merged left to right. Two templates ship: `dask-local` (the default) and
`dask-slurm-example`. **The SLURM one is a template** — copy it or overlay it with
your account, partition, sizing and log directory. Site-specific configs are
deliberately not in this repo.

Layering means a site config is the handful of keys that actually differ, rather than
a fork of the template that silently drifts from it:

```yaml
# my-site.yaml — everything else comes from dask-slurm-example
jobqueue:
  slurm:
    account: my-account
    queue: my-partition
    log-directory: /path/with/room
```

Dicts merge; scalars and lists **replace** (`job-extra-directives` is a complete
argument vector, and appending to it produces a command line nobody wrote).

**Unrecognised keys raise.** An overlay typo would otherwise deep-merge silently and
do nothing — a run using different settings than you wrote, with no error. Keys are
checked against dask's own defaults (so anything dask or dask-jobqueue accepts is
allowed, not just what the templates happen to show) and canonicalised, so
`log_directory` and `log-directory` are the same key. Pass
`effective_config_path=<run dir>` to record the merged result alongside the run.
- **`block_map`** — maps a picklable fn over items; incremental results via
  `as_completed` (`on_result` callback for durable progress), or serial when
  `client=None`.
- **`Manifest`** — single-writer JSONL completion log for resume + intent
  (`written` / `empty` / `skipped`).
- **`iter_blocks` / `Block`** — tile an N-d shape into chunk blocks.

Orchestration background and hard-won SLURM gotchas: see
`em-volume-tools/docs/dask-slurm-rusty.md`.

## Environment

One conda environment covers this repo and its consumers (`em-volume-tools`,
`em-seg-morpho`), each installed editable. All runtime deps here are pure Python
and come from conda-forge; `--no-deps` keeps pip from re-resolving them.

```bash
conda activate em-lib
pip install --no-deps -e .
python -m pytest -q
```

The combined spec lives one level up, at `em-libraries/environment.yml`.
