"""em-blockrun: resumable, chunked parallel maps on dask — locally or on SLURM.

A small, dependency-light substrate extracted from em-volume-tools for reuse
across projects (volume conversion, meshing/skeletonization, etc.):

- ``start_dask``: one context manager that yields a connected dask Client backed
  by a LocalCluster or a SLURMCluster, selected by a YAML config.
- ``block_map``: map a function over items via batched ``client.map`` +
  ``as_completed`` (incremental, durable results) or serially.
- ``Manifest``: a single-writer completion log for resume + intent tracking.
- ``iter_blocks`` / ``Block``: tile an N-d shape into chunk blocks.

See docs and the config templates under ``em_blockrun/configs/``.
"""

__version__ = "0.1.0"

from .dask_config import (ConfigKeyError, bundled_configs, deep_merge, load_config,
                          resolve_config, unknown_keys, write_effective_config)
from .engine import Block, block_map, idempotent, iter_blocks
from .manifest import Manifest


# `start_dask` is resolved on first access, not at import (PEP 562). Importing
# `dask_runner` eagerly costs ~1 s, nearly all of it `dask.distributed`, and it is paid
# by everything that touches this package — including `em-vol info`, which reads one
# metadata document and never starts a cluster. Nothing else here needs dask:
# `dask_config` is yaml only, and `engine.block_map` imports `as_completed` inside the
# branch that has a client. Kept lazy rather than made conditional on being under test:
# an import graph that differs between tests and production is the shape of bug this
# codebase has already shipped twice (see CLAUDE.md invariant 9).
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
    # dask config: bundled templates, layered overrides, key validation. Exported
    # because each consumer's CLI needs them for its own --config argument.
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
