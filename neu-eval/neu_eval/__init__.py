"""neu-eval — comparing a segmentation against a reference labeling.

Standard image-segmentation metrics (variation of information, adapted Rand, IoU matching,
fragmentation) together with the ones proposed for connectomics, and — because a
disagreement is not always the segmentation's fault — a ranked, located, adjudicable list
of *where* the two labelings differ. A score says how much; the disagreement table says
where to look.

**Everything hangs off one object.** :func:`neu_eval.contingency` reduces two label arrays
to a :class:`~neu_eval.overlap.Contingency` — a sparse table of ``(a_id, b_id) -> voxels``
— and every voxel metric in this package is a pure function of *that*, not of the arrays.
Three properties follow, and they are the reason the package is shaped this way:

- **It is additive.** ``c(region1) + c(region2) == c(region1 | region2)`` for disjoint
  regions, exactly. So a whole-volume sweep is a block map with a driver-side sum, and no
  metric code changes to get there.
- **The expensive pass happens once.** VOI, Rand, per-body decompositions, matching and
  completeness are then cheap and recomputable in a notebook, which is what makes the
  "use several metrics, none of them alone" practice affordable rather than a pass each.
- **It is symmetric.** ``c.T`` swaps the two sides. Nothing below the reporting layer calls
  either one truth; which is ``gt`` and which is ``seg`` is a *label*, passed in at the
  point where a human reads the output.

**A reduction is also what keeps the GPU question small.** A contingency table is
``O(pairs)``, not ``O(voxels)``, so it comes back to CPU memory at the boundary and stays
there. The only functions that touch a backend are the voxel pass itself and the
per-pair distance transform behind :func:`neu_eval.disagree.locate`; everything downstream
is plain numpy. Dispatch comes from :mod:`neu_proc.ops.backend` — on the array, never on a
flag — and the off switch is ``NEU_PROC_GPU=0``, reused rather than duplicated under a
second name for the same decision.

Top-level names resolve lazily (PEP 562), so importing this package pays for neither scipy
nor neu-vol. ``_EXPORTS`` maps a name to the module it is defined in; ``_SUBMODULES`` maps a
name to a module exposed whole. Laziness holds up to the first *use*, so anything behind an
optional extra belongs inside a function — which is where pandas lives here, as it does in
``neu_morpho.measure``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

#: name -> module it lives in, relative to this package. Explicit rather than a star import,
#: so lazy resolution cannot quietly start pulling a heavy module for a light name. The key
#: is looked up *inside* the module named by the value, so one module contributing two names
#: needs two entries.
_EXPORTS: dict[str, str] = {
    "Contingency": "overlap",
    "contingency": "overlap",
    "pool": "overlap",
    "compare": "compare",
    "Report": "compare",
}

#: name -> submodule to expose at top level. Separate from `_EXPORTS` because the resolution
#: differs: this returns the MODULE, where `_EXPORTS` returns a name defined inside one.
#: Same lazy contract. Keys must not collide with `_EXPORTS` — whichever were checked first
#: would silently win, so a test asserts they stay disjoint.
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
