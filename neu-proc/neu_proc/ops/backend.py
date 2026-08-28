"""Which library runs an op: scipy on the host, or cupyx on the device.

**Dispatch is on the array, never on a flag or a parameter.** ``cupyx.scipy.ndimage``
requires a ``cupy.ndarray``, so the module cannot be swapped without moving the data::

    def dilate(array, *, radius_vox):
        ndi, array = ndimage_for(array, "grey_dilation")
        return ndi.grey_dilation(array, footprint=kernels.ball(radius_vox))

One body, both backends.

**An op handed a HOST array may still go to the device and come back, and for an expensive
op that wins.** This module first claimed the opposite — that a single op's transfer
dominates, so only a chain could pay for it — and measurement contradicted it: a dilation is
enough arithmetic that the round trip is still faster than staying on the host. That was an
assumption stated as a reason, which is worse than no reason, so it is corrected here rather
than quietly dropped.

But it is **per-op, not blanket**, because arithmetic intensity is a property of the op and
not of the caller: a dilation earns the round trip and a threshold cannot, being bandwidth-
bound either way. So each op declares its own default via :func:`stage`, a caller can
override it, and ``NEU_PROC_GPU=0`` turns the whole thing off.

**What you hand in is what you get back.** :func:`like` returns the result to the array's
original home, so ``dilate(numpy_array)`` is numpy and nothing silently moves a caller's data
onto a GPU. For a *chain*, move once yourself — ``Piece(to_device(a), frame)`` — and every op
then sees a device array and stays there, which is the case where transfers really are the
whole cost.

Three things follow from dispatching on the array rather than on a global switch:

- **No hidden state.** A global ``use_gpu`` is the thing that makes one call behave
  differently for two people, which is the argument that kept a machine-wide config out of
  neu-mark. Here the answer is a property of the array in your hand.
- **Free when cupy is absent.** :func:`is_device_array` reads ``type(a).__module__``, so
  nothing here imports cupy to discover that it is not needed — which is what keeps the
  ``import neu_proc.ops`` guard honest.
- **The enable switch governs TRANSFERS, not dispatch.** ``NEU_PROC_GPU=0`` (or
  ``backend.enabled = False``) makes :func:`to_device` a no-op, so a chain stays on the host.
  It deliberately does not disable dispatch: handed a device array anyway, the right thing is
  still to use cupyx, and a flag that silently ran the host code on device memory would just
  raise somewhere less obvious.

Measured, on an RTX A6000, for the EDT dilation — ``host`` is scipy, ``round trip`` includes
both transfers, ``on device`` is an array already there:

======  =========  ==================  ===================
size    host       round trip          already on device
======  =========  ==================  ===================
96^3     36.4 ms    2.7 ms (13.6x)      0.7 ms (52x)
144^3   140.7 ms    6.3 ms (22.3x)      1.3 ms (112x)
192^3   811.2 ms   16.4 ms (49.5x)      2.0 ms (401x)
======  =========  ==================  ===================

Two things to read off it. The round trip wins by a wide margin *and the margin grows*,
because scipy's EDT is superlinear in the voxel count while a transfer is linear — so the
break-even is below any size worth processing, for this op. And staying on the device is
another ~8x on top, which is the argument for moving once yourself around a chain rather than
relying on the per-op default.

None of that generalises to a cheap op: a threshold is bandwidth-bound, so it can only lose
the transfer. Hence ``prefer_gpu`` being the op's call.

The first call is slower than any of these — CUDA context setup and kernel compilation, once
per process, not attributable to anything here. Use :func:`synchronize` when timing, or cupy
queues the work and returns and the GPU looks impossibly fast.

**The namespaces are not identical, and the gap is small.** Measured against cupy 14.2 /
scipy 1.18, ``cupyx.scipy.ndimage`` lacks four real functions —
``distance_transform_bf``, ``distance_transform_cdt``, ``geometric_transform``,
``watershed_ift`` — out of the whole module (``distance_transform_edt`` *is* there).
:func:`ndimage_for` handles that by falling back **loudly**: it moves the array to the host
and warns once, so the op works and the performance cliff is visible. Falling back silently
would turn "I am using the GPU" into the slowest possible path with nothing to show it.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

#: Whether :func:`to_device` may move anything. Dispatch is unaffected — see the module
#: docstring. ``NEU_PROC_GPU=0`` turns it off for a process without touching code, which is
#: what a benchmark or a bisect wants.
enabled: bool = os.environ.get("NEU_PROC_GPU", "1").strip().lower() not in (
    "0", "false", "no", "off")

#: Functions cupyx lacks, warned about once each rather than on every call — a filter in a
#: loop would otherwise bury everything else.
_warned: set[str] = set()


def is_device_array(value: Any) -> bool:
    """True if ``value`` lives on a GPU, decided **without importing cupy**.

    The type's defining module, not ``isinstance``: importing cupy to ask whether an array
    is a numpy array would make every host-only install pay for a CUDA stack, and on a
    machine without one it would fail outright.
    """
    return type(value).__module__.split(".")[0] == "cupy"


def gpu_available() -> bool:
    """True if cupy imports *and* a device answers. Cached after the first call.

    Both halves are needed and the second is the one that bites: cupy installs fine on a
    machine with no driver, no device, or a CUDA version mismatch, and then fails at the
    first allocation rather than at import.
    """
    global _available
    if _available is None:
        try:
            import cupy

            _available = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception as exc:                                          # noqa: BLE001
            logger.debug("no GPU backend: %s: %s", type(exc).__name__, exc)
            _available = False
    return _available


_available: bool | None = None


def array_module(value: Any):
    """``cupy`` for a device array, ``numpy`` otherwise.

    The ``xp`` of the array-API idiom, so an op needing ``zeros_like`` or ``unique`` gets it
    from the right library without branching.
    """
    if is_device_array(value):
        import cupy

        return cupy
    import numpy

    return numpy


def ndimage_for(value: Any, need: str | None = None):
    """``(ndimage module, array)`` for ``value`` — the array moved only if it had to be.

    ``need`` names the function about to be called. When ``value`` is on the device and
    cupyx does not provide it, the array is brought **back to the host** and scipy is
    returned, with a warning once per function name. Loud rather than silent, because the
    fallback is the slowest path available and the whole reason to be on the device was
    speed; and loud rather than fatal, because the op still works and a hard failure would
    make a mixed pipeline unusable over one missing function.

    With no ``need`` the namespace is chosen and the array returned untouched.
    """
    if is_device_array(value):
        from cupyx.scipy import ndimage as cu

        if need is None or hasattr(cu, need):
            return cu, value
        if need not in _warned:
            _warned.add(need)
            warnings.warn(
                f"cupyx.scipy.ndimage has no {need!r}, so this runs on the HOST: the array "
                f"is copied off the device and back, which is slower than having stayed "
                f"there. Restructure to keep the device path, or accept the copy.",
                RuntimeWarning, stacklevel=3)
        value = to_host(value)
    from scipy import ndimage

    return ndimage, value


def stage(value: Any, *, prefer_gpu: bool = True):
    """``value`` where the op should compute — moved to the device if that is worth it.

    The op decides, through ``prefer_gpu``, because only the op knows its own arithmetic
    intensity. Pass ``False`` for anything bandwidth-bound (a threshold, a cast, a remap):
    those cannot recover a round trip and would be slower for the move.

    A device array is returned untouched, so a caller who moved once for a chain pays
    nothing here. Otherwise the usual guards apply: ``NEU_PROC_GPU=0`` or no device means
    the array comes back as it went in, so one piece of calling code runs in both places.

    Pair with :func:`like`, which returns the result to where ``value`` actually lived::

        work = stage(arr)
        ...
        return like(result, arr)
    """
    if not prefer_gpu:
        return value
    return to_device(value)


def like(result: Any, reference: Any):
    """``result`` moved to wherever ``reference`` lives — host or device.

    So an op gives back what it was handed: ``dilate(numpy_array)`` returns numpy even if it
    computed on the GPU, and nothing silently leaves a caller's data on a device they did not
    ask to use. It is also what makes :func:`stage` safe to apply by default.
    """
    if is_device_array(reference):
        return to_device(result)
    return to_host(result)


def host_only(value: Any, what: str):
    """``value`` on the host, warning once if it had to be moved.

    For a library with **no device implementation at all** — fastmorph, cc3d, fill_voids,
    edt — which is most of the Seung-lab half. Without this, handing one a device array
    fails with cupy's ``Implicit conversion to a NumPy array is not allowed``, which names
    neither the function nor what to do about it.

    Loud rather than silent for the same reason :func:`ndimage_for`'s fallback is: the array
    came to the device to be fast, and a copy back is the slowest thing that can happen to
    it. Loud rather than fatal because the op still works, and refusing would make one
    host-only function poison a whole chain.
    """
    if not is_device_array(value):
        return value
    if what not in _warned:
        _warned.add(what)
        warnings.warn(
            f"{what} has no GPU implementation, so the array is copied off the device and "
            f"back. Keep it on the host for this step, or use a device-capable alternative.",
            RuntimeWarning, stacklevel=3)
    return to_host(value)


def to_device(value: Any):
    """``value`` on the GPU, or unchanged if that is not possible or not wanted.

    Honours :data:`enabled` and :func:`gpu_available`, so a host-only machine and a
    deliberately disabled one both get the array back untouched rather than an error — which
    is what lets one piece of calling code run in both places.
    """
    if is_device_array(value):
        return value
    if not (enabled and gpu_available()):
        return value
    import cupy

    return cupy.asarray(value)


def to_host(value: Any):
    """``value`` as a numpy array, moved off the device if it is on one."""
    if not is_device_array(value):
        return value
    import cupy

    return cupy.asnumpy(value)


def synchronize() -> None:
    """Wait for the device to finish. Only needed when TIMING something.

    cupy queues work and returns, so a naive ``time.perf_counter()`` around an op measures
    the enqueue and reports a GPU as impossibly fast. Every benchmark needs this; nothing
    else does, because a subsequent read synchronizes on its own.
    """
    if not gpu_available():
        return
    import cupy

    cupy.cuda.runtime.deviceSynchronize()
