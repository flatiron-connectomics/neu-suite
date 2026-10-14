"""blockrun: resumable, chunked parallel maps on dask — locally or on SLURM.

A small, dependency-light substrate for block-wise jobs (volume conversion, meshing,
skeletonization):

- ``start_dask``: one context manager that yields a connected dask Client backed
  by a LocalCluster or a SLURMCluster, selected by a YAML config.
- ``block_map``: map a function over items via batched ``client.map`` +
  ``as_completed`` (incremental, durable results) or serially.
- ``Manifest``: a single-writer completion log for resume + intent tracking.
- ``iter_blocks`` / ``Block``: tile an N-d shape into chunk blocks.

Config templates are under ``blockrun/configs/``; ``docs/dask-slurm.md`` is the guide.
"""

__version__ = "0.1.0"

from .dask_config import (ConfigKeyError, bundled_configs, deep_merge, load_config,
                          resolve_config, unknown_keys, write_effective_config)
from .engine import Block, block_map, idempotent, iter_blocks
from .manifest import Manifest


# `start_dask` is imported on first access (PEP 562): `dask.distributed` costs ~1 s, and
# commands that never start a cluster (`neu-vol info`) should not pay it.
def __getattr__(name: str):
    if name == "start_dask":
        from .dask_runner import start_dask

        return start_dask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    "__version__",
    "start_dask",
    # dask config, for each consumer CLI's --config argument
    "ConfigKeyError",
    "bundled_configs",
    "deep_merge",
    "load_config",
    "resolve_config",
    "unknown_keys",
    "write_effective_config",
    "Block",
    "block_map",
    "idempotent",
    "iter_blocks",
    "Manifest",
]
