"""Which library runs an op: scipy on the CPU, or cupyx on the GPU.

Dispatch follows the array, never a flag: ``cupyx.scipy.ndimage`` needs a ``cupy.ndarray``,
so one body serves both backends::

    def dilate(array, *, radius_vox):
        ndi, array = ndimage_for(array, "grey_dilation")
        return ndi.grey_dilation(array, footprint=kernels.ball(radius_vox))

An op handed a CPU array may still compute on the GPU and copy back, when it has enough
arithmetic to pay for the transfers; each op declares this with :func:`stage`. Measured on
an RTX A6000, an EDT dilation of a 192^3 array takes 811 ms in scipy, 16 ms on the GPU
including both transfers, and 2 ms on an array already there. A bandwidth-bound op (a
threshold, a cast) cannot win the round trip and should not stage. :func:`like` returns
each result to where the input lived, so ``dilate(numpy_array)`` returns numpy. For a
chain of ops, move once with :func:`to_gpu` and every step stays on the GPU.

``NEU_PROC_GPU=0`` (or ``backend.enabled = False``) makes :func:`to_gpu` a no-op, so
nothing moves; a GPU array handed in is still processed on the GPU. ``cupyx`` lacks a few
ndimage functions (``distance_transform_bf``, ``distance_transform_cdt``,
``geometric_transform``, ``watershed_ift``); :func:`ndimage_for` then falls back to the CPU
and warns once. When timing GPU code, call :func:`synchronize` first: cupy queues work and
returns immediately, and the first call in a process also pays CUDA setup.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

#: Whether :func:`to_gpu` may move anything (dispatch is unaffected). Set from
#: ``NEU_PROC_GPU``; ``0``, ``false``, ``no`` or ``off`` disable it.
enabled: bool = os.environ.get("NEU_PROC_GPU", "1").strip().lower() not in (
    "0", "false", "no", "off")

#: Names already warned about, so each fallback warns once per process.
_warned: set[str] = set()


def is_gpu_array(value: Any) -> bool:
    """True if ``value`` is a cupy array. Checked by module name, so cupy is never imported."""
    return type(value).__module__.split(".")[0] == "cupy"


def gpu_available() -> bool:
    """True if cupy imports and a GPU device is found; cached after the first call. cupy
    can import on a machine with no GPU or driver and fail only at first use, hence the
    device check."""
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
    """``cupy`` for a GPU array, ``numpy`` otherwise (the array-API ``xp``)."""
    if is_gpu_array(value):
        import cupy

        return cupy
    import numpy

    return numpy


def ndimage_for(value: Any, need: str | None = None):
    """``(ndimage module, array)`` for ``value``.

    ``need`` names the function about to be called. If ``value`` is on the GPU and cupyx
    lacks that function, the array is moved to the CPU and scipy is returned, with a
    warning once per name. Without ``need`` the array is returned untouched.
    """
    if is_gpu_array(value):
        from cupyx.scipy import ndimage as cu

        if need is None or hasattr(cu, need):
            return cu, value
        if need not in _warned:
            _warned.add(need)
            warnings.warn(
                f"cupyx.scipy.ndimage has no {need!r}, so this runs on the CPU: the array "
                f"is copied off the GPU and back, which is slower than having stayed "
                f"there. Restructure to keep the GPU path, or accept the copy.",
                RuntimeWarning, stacklevel=3)
        value = to_cpu(value)
    from scipy import ndimage

    return ndimage, value


def stage(value: Any, *, prefer_gpu: bool = True):
    """``value`` where the op should compute: moved to the GPU if ``prefer_gpu`` and a GPU
    is enabled and available, otherwise unchanged. Pass ``prefer_gpu=False`` for a
    bandwidth-bound op. Pair with :func:`like`::

        work = stage(arr)
        ...
        return like(result, arr)
    """
    if not prefer_gpu:
        return value
    return to_gpu(value)


def like(result: Any, reference: Any):
    """``result`` moved to wherever ``reference`` lives, CPU or GPU, so an op returns the
    kind of array it was given."""
    if is_gpu_array(reference):
        return to_gpu(result)
    return to_cpu(result)


def cpu_only(value: Any, what: str):
    """``value`` on the CPU, warning once per ``what`` if it had to be moved. For libraries
    with no GPU implementation (fastmorph, cc3d, fill_voids, edt), which would otherwise
    fail with cupy's unhelpful "Implicit conversion to a NumPy array" error."""
    if not is_gpu_array(value):
        return value
    if what not in _warned:
        _warned.add(what)
        warnings.warn(
            f"{what} has no GPU implementation, so the array is copied off the GPU and "
            f"back. Keep it on the CPU for this step, or use a GPU-capable alternative.",
            RuntimeWarning, stacklevel=3)
    return to_cpu(value)


def to_gpu(value: Any):
    """``value`` on the GPU; unchanged, without error, if :data:`enabled` is false or no GPU
    is available."""
    if is_gpu_array(value):
        return value
    if not (enabled and gpu_available()):
        return value
    import cupy

    return cupy.asarray(value)


def to_cpu(value: Any):
    """``value`` as a numpy array, moved off the GPU if it is on one."""
    if not is_gpu_array(value):
        return value
    import cupy

    return cupy.asnumpy(value)


def synchronize() -> None:
    """Wait for queued GPU work to finish. Needed only when timing: cupy returns before the
    work is done, while reading a result synchronizes on its own."""
    if not gpu_available():
        return
    import cupy

    cupy.cuda.runtime.deviceSynchronize()
