"""neu-proc: array processing for large 3D EM volumes.

scipy and scikit-image filters and the Seung-lab label tools (cc3d, fastremap, fastmorph,
edt, fill_voids), applied to whole arrays. Only :mod:`neu_proc.ops` exists so far: array
in, array out. Running ops blockwise across a volume, with halos and seam reconciliation,
is planned; an op that works on one array is the reference that runner will be checked
against.

Functions take voxel parameters. Converting a physical length into voxels for a given
level happens in :mod:`neu_proc.ops.kernels`, never inside a function: a halo derived from
anything but the converted parameter is wrong in a shell around every seam, which shows
as faint block edges rather than as an error.

Top-level names resolve lazily (PEP 562), so importing the package pays for neither scipy
nor dask until a name is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

#: Attribute name -> the module defining it, as a dotted path relative to this package
#: (handed to ``importlib.import_module(f".{module}", __name__)``).
_EXPORTS: dict[str, str] = {
    "dilate": "ops.dilate",
    "dust": "ops.dust",
    "offset": "ops.offset",
    "opening": "ops.opening",
}

#: Name -> submodule exposed whole at the top level (``from neu_proc import measure``).
#: Keys must not overlap ``_EXPORTS``, or one would silently shadow the other; a test
#: checks this.
_SUBMODULES: dict[str, str] = {
    "measure": "ops.measure",
}

__all__ = ["__version__", *sorted({*_EXPORTS, *_SUBMODULES})]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDE completion only
    # Editors and type checkers read this block (it never runs), so every lazy name
    # must be listed here as well; a test checks the two lists agree.
    from .ops import measure  # noqa: F401
    from .ops.dilate import dilate  # noqa: F401
    from .ops.dust import dust  # noqa: F401
    from .ops.offset import offset  # noqa: F401
    from .ops.opening import opening  # noqa: F401


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
