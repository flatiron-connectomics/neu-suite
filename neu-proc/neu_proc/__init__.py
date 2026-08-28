"""neu-proc — blockwise array processing for large 3D EM volumes.

scipy / scikit-image filters and Seung-lab label tools (cc3d, fastremap, fastmorph, edt,
fill_voids) applied to whole arrays now, and blockwise across a volume later. The fourth
consumer tier alongside neu-morpho, neu-mark and neu-glance.

**Only :mod:`neu_proc.ops` exists so far, and it is deliberately the pure half.** The array
functions are the part that can be written, read and tested on their own; the halo, seam and
reconcile machinery that makes them correct across block boundaries is designed in
``NEU-PROC-PLAN.md`` and not built. Nothing here is a placeholder for it — an op that works
on one array is the *reference* the distributed runner will be checked against, which is why
the plan makes ``apply`` the test oracle.

Three layers, and collapsing the first two is the easy mistake:

===========  ==========================  =======================  =========
layer        example                     units                    can run?
===========  ==========================  =======================  =========
``Op``       ``Gaussian(sigma_nm=80)``   **nanometres**           no
``BoundOp``  ``op.bind(frame)``          voxels, **plus a halo**  yes
a function   ``gaussian(a, sigma_vox=)`` **voxels**               —
===========  ==========================  =======================  =========

The declaration is physical because a voxel count means different things at different
levels and along different axes of an anisotropic volume; the execution is in voxels because
that is what scipy takes. The conversion happens in exactly one place,
:mod:`neu_proc.ops.kernels`, reached through ``bind`` — never inside a function and never
declared beside a halo, because a halo derived from anything other than the *converted*
parameter is wrong in a shell around every seam, which reads as faint block edges rather
than as an error.

Top-level names resolve lazily (PEP 562) so importing this package, or asking a future
``--help``, pays for neither scipy nor dask.
"""

from __future__ import annotations

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
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    """Resolve a top-level export on first use."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return list(__all__)
