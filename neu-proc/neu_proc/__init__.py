"""neu-proc — array processing for large 3D EM volumes.

scipy / scikit-image filters and Seung-lab label tools (cc3d, fastremap, fastmorph, edt,
fill_voids), applied to whole arrays. The fourth consumer tier alongside neu-morpho,
neu-mark and neu-glance.

**Only :mod:`neu_proc.ops` exists so far: array in, array out.** Running these ops blockwise
across a volume, with the halos and seam reconciliation that needs, is planned and not
built. An op that works on one array is the reference that runner will be checked against.

Functions take **voxel** parameters. Converting a physical length into voxels for a given
level happens in one place, :mod:`neu_proc.ops.kernels`, and never inside a function: a halo
derived from anything other than the *converted* parameter is wrong in a shell around every
seam, which reads as faint block edges rather than as an error.

Top-level names resolve lazily (PEP 562) so importing this package, or asking a future
``--help``, pays for neither scipy nor dask. Two mappings feed that: ``_EXPORTS`` for a name
defined *inside* a module (``neu_proc.dust``) and ``_SUBMODULES`` for a module exposed whole
(``from neu_proc import measure``). Laziness holds up to the first *use* of a name, so a
module reached this way still pays for whatever its body imports — which is why anything
behind an optional extra belongs inside a function, as ``neu_morpho.measure`` does with
pandas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

#: name -> module it lives in, relative to this package. Explicit rather than a star import,
#: so lazy resolution cannot quietly start pulling a heavy module for a light name. Grows as
#: `ops/` does.
#: The value is a **dotted module path relative to this package** — no slashes, no `.py`,
#: because `__getattr__` hands it to `importlib.import_module(f".{module}", __name__)`. The
#: key is the attribute name, looked up *inside* that module, so one module contributing two
#: names needs two entries.
_EXPORTS: dict[str, str] = {
    "dilate": "ops.dilate",
    "dust": "ops.dust",
    "offset": "ops.offset",
    "opening": "ops.opening",
}

#: name -> submodule to expose at top level, also relative to this package. **Separate from
#: `_EXPORTS` because the resolution differs**: this returns the MODULE, where `_EXPORTS`
#: returns a name defined inside one. There is no entry shape in `_EXPORTS` that could do
#: this — every value there is a container to reach into — and one dict with a sentinel
#: value reads as a trap. Same lazy contract: nothing is imported until asked for. For a
#: module that is a namespace rather than a single callable, so `measure.summary(...)` and
#: `measure.SOMETHING` both read at the call site.
#:
#: Keys must not collide with `_EXPORTS`; whichever is checked first would silently win, so
#: `test_every_export_resolves` asserts they stay disjoint.
_SUBMODULES: dict[str, str] = {
    "measure": "ops.measure",
}

__all__ = ["__version__", *sorted({*_EXPORTS, *_SUBMODULES})]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDE completion only
    # A lazily-resolved name is invisible to static analysis, so this is what makes
    # `neu_proc.measure` complete and typecheck. Never executed at runtime, so it costs
    # nothing and cannot become an eager import by accident.
    from .ops import measure  # noqa: F401


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
