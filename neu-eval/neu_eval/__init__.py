"""neu-eval: comparing a segmentation against a reference labeling.

Standard metrics (variation of information, adapted Rand, IoU matching, fragmentation)
plus a ranked, located list of where the two labelings disagree, since a disagreement is
not always the segmentation's fault.

Everything starts from :func:`neu_eval.contingency`, which reduces two label arrays to a
:class:`~neu_eval.overlap.Contingency`, a sparse ``(a_id, b_id) -> voxels`` table. Every
voxel metric is a function of that table, which is:

- **additive** over disjoint regions, so a whole-volume sweep is a block map plus a sum;
- **computed once**, after which all the metrics are cheap to recompute in a notebook;
- **symmetric**: ``c.T`` swaps the sides, and which one is "gt" is only a label.

Only the voxel pass and :func:`neu_eval.disagree.locate` touch a GPU (via
:mod:`neu_proc.ops.backend`; ``NEU_PROC_GPU=0`` turns it off); the table itself is small
and lives in CPU memory. Top-level names resolve lazily, so ``import neu_eval`` loads
neither scipy nor neu-vol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

#: name -> the module (relative to this package) that defines it.
_EXPORTS: dict[str, str] = {
    "Contingency": "overlap",
    "contingency": "overlap",
    "pool": "overlap",
    "compare": "compare",
    "Report": "compare",
}

#: name -> submodule exposed whole. Keys must not overlap `_EXPORTS` (a test checks).
_SUBMODULES: dict[str, str] = {
    "adjudicate": "adjudicate",
    "assign": "assign",
    "disagree": "disagree",
    "overlap": "overlap",
    "tables": "tables",
    "voxel": "voxel",
}

__all__ = ["__version__", *sorted({*_EXPORTS, *_SUBMODULES})]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDE completion only
    from . import adjudicate, assign, disagree, overlap, tables, voxel  # noqa: F401
    from .compare import Report, compare  # noqa: F401
    from .overlap import Contingency, contingency, pool  # noqa: F401


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
