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

from .dask_runner import start_dask
from .engine import Block, block_map, idempotent, iter_blocks
from .manifest import Manifest

__all__ = [
    "__version__",
    "start_dask",
    "Block",
    "block_map",
    "idempotent",
    "iter_blocks",
    "Manifest",
]
