"""neu-morpho: segment morphology from segmentation volumes.

Generate multi-resolution Draco-encoded **meshes** (via ``vol2mesh``) per segment,
written in neuroglancer-precomputed format, orchestrated across segments with
``blockrun`` and reading segmentation arrays with ``neu-vol``; read, measure and compare
meshes and **skeletons**. Skeletons themselves are made by ``neu-skel``, a separate
GPL-licensed package that runs as an extra stage of this pipeline. The pipeline runs from the ``neu-morpho`` command; from Python::

    import neu_morpho
    summary = neu_morpho.pipeline.run(neu_morpho.pipeline.plan(src, dst))  # serial
    verts_xyz_nm, faces, lod = neu_morpho.read_body_mesh(volume, body_id)   # or None
    db = neu_morpho.MetricsDB(path)                          # per-body metrics

Large segments (whose binary mask over a big bounding box would OOM) are meshed
**chunked and stitched** rather than materialized whole. See the README, and
neu-skel for the tracers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

#: Attribute name -> the module defining it, relative to this package. Resolved on first
#: use, so ``import neu_morpho`` (which the CLI does for ``__version__``) stays instant.
_EXPORTS: dict[str, str] = {
    "read_body_mesh": "readback",
    "read_body_skeleton": "readback",
    "mesh_transform": "readback",
    "skeleton_transform": "readback",
    "UnsupportedSubresource": "readback",
    "MetricsDB": "metrics_db",
    "MeshConfig": "config",
}

#: Name -> submodule exposed whole at the top level. Keys must not overlap ``_EXPORTS``.
_SUBMODULES: dict[str, str] = {
    "measure": "measure",
    "pipeline": "pipeline",
    "readback": "readback",
}

__all__ = ["__version__", *sorted({*_EXPORTS, *_SUBMODULES})]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDE completion only
    from . import measure, pipeline, readback  # noqa: F401
    from .config import MeshConfig  # noqa: F401
    from .metrics_db import MetricsDB  # noqa: F401
    from .readback import (UnsupportedSubresource, mesh_transform,  # noqa: F401
                           read_body_mesh, read_body_skeleton, skeleton_transform)


def __getattr__(name: str):
    """Resolve a top-level export or submodule on first use."""
    import importlib

    module = _SUBMODULES.get(name)
    if module is not None:
        return importlib.import_module(f".{module}", __name__)
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return list(__all__)
