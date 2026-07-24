# em-blockrun

Resumable, chunked **parallel maps on dask** — the same code runs locally on a
workstation or across Flatiron **Rusty/SLURM** by swapping one YAML config.

Extracted from `em-volume-tools` so it can be shared across projects (volume
conversion, meshing/skeletonization, …). Dependency-light: only `dask`,
`distributed`, `dask-jobqueue`, `pyyaml`.

```python
from em_blockrun import start_dask, iter_blocks, block_map, Manifest

blocks = list(iter_blocks(shape=(1024, 1024, 1024), chunks=(128, 128, 128)))
with start_dask(num_workers=48, config_path="em_blockrun/configs/dask-slurm-any.yaml") as client:
    results = block_map(blocks, work_fn, client=client)   # batched client.map + as_completed
```

- **`start_dask`** — LocalCluster or SLURMCluster from one config (see
  `em_blockrun/configs/`); tears down (and `scancel`s) on exit.
- **`block_map`** — maps a picklable fn over items; incremental results via
  `as_completed` (`on_result` callback for durable progress), or serial when
  `client=None`.
- **`Manifest`** — single-writer JSONL completion log for resume + intent
  (`written` / `empty` / `skipped`).
- **`iter_blocks` / `Block`** — tile an N-d shape into chunk blocks.

Orchestration background and hard-won SLURM gotchas: see
`em-volume-tools/docs/dask-slurm-rusty.md`.

## Environment

Managed with [pixi]; the env is detached to ceph (home inode quota).

```bash
pixi install && pixi run -e dev test
```
