"""Choosing between scipy and cupyx: :mod:`neu_proc.ops.backend`.

Dispatch is tested without a GPU, using a fake array that reports cupy as its module; the
real-GPU tests skip when no GPU backend is available.
"""

import warnings

import numpy as np
import pytest

from neu_proc.ops import backend


class FakeGpuArray(np.ndarray):
    """A numpy array whose type claims to be cupy's, which is all dispatch reads."""
    __module__ = "cupy"

    @classmethod
    def of(cls, array):
        return array.view(cls)


def _host(shape=(4, 4, 4)):
    return np.zeros(shape, "uint8")


# --------------------------------------------------------------------------- #
# dispatch, no GPU needed
# --------------------------------------------------------------------------- #
def test_dispatch_reads_the_type_and_imports_nothing():
    """Asking whether an array is on the GPU must not import cupy."""
    import subprocess
    import sys

    assert not backend.is_gpu_array(_host())
    assert backend.is_gpu_array(FakeGpuArray.of(_host()))

    out = subprocess.run(
        [sys.executable, "-c",
         "import numpy as np\n"
         "from neu_proc.ops import backend\n"
         "backend.is_gpu_array(np.zeros((2, 2, 2)))\n"
         "backend.array_module(np.zeros((2, 2, 2)))\n"
         "import sys; assert 'cupy' not in sys.modules, sorted(sys.modules)[:5]\n"
         "print('ok')"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_a_host_array_gets_scipy():
    ndi, arr = backend.ndimage_for(_host())
    assert ndi.__name__ == "scipy.ndimage"
    assert arr is not None and not backend.is_gpu_array(arr)
    assert backend.array_module(_host()).__name__ == "numpy"


def test_to_cpu_is_a_no_op_for_a_cpu_array():
    a = _host()
    assert backend.to_cpu(a) is a


def test_the_switch_governs_transfers_not_dispatch(monkeypatch):
    """With transfers disabled nothing moves, but a GPU array is still recognised."""
    monkeypatch.setattr(backend, "enabled", False)
    a = _host()
    assert backend.to_gpu(a) is a
    assert backend.is_gpu_array(FakeGpuArray.of(a))


def test_to_gpu_is_a_no_op_without_a_gpu(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    a = _host()
    assert backend.to_gpu(a) is a


def test_the_env_var_is_read_at_import(monkeypatch):
    import importlib

    monkeypatch.setenv("NEU_PROC_GPU", "0")
    reloaded = importlib.reload(backend)
    try:
        assert reloaded.enabled is False
    finally:
        monkeypatch.setenv("NEU_PROC_GPU", "1")
        importlib.reload(backend)


def test_gpu_available_is_false_rather_than_raising_when_cupy_is_broken(monkeypatch):
    monkeypatch.setattr(backend, "_available", None)
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "cupy":
            raise ImportError("no cupy")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert backend.gpu_available() is False
    monkeypatch.setattr(backend, "_available", None)


# --------------------------------------------------------------------------- #
# the real GPU
# --------------------------------------------------------------------------- #
#: Gated on `gpu_available()` rather than `importorskip("cupy")`: cupy can import and still
#: fail at the first allocation (no driver, CUDA mismatch).
needs_gpu = pytest.mark.skipif(
    not backend.gpu_available(),
    reason="no GPU backend; the CPU paths above cover the dispatch logic")


@needs_gpu
def test_a_device_array_gets_cupyx():
    dev = backend.to_gpu(_host())
    assert backend.is_gpu_array(dev)
    ndi, arr = backend.ndimage_for(dev, "grey_dilation")
    assert ndi.__name__ == "cupyx.scipy.ndimage"
    assert arr is dev, "no move when cupyx has the function"
    assert backend.array_module(dev).__name__ == "cupy"


@needs_gpu
def test_a_missing_cupyx_function_falls_back_with_a_warning():
    """cupyx lacks a few scipy functions (watershed_ift among them); falling back to the CPU
    silently would hide the slowest path."""
    backend._warned.discard("watershed_ift")
    dev = backend.to_gpu(_host())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ndi, arr = backend.ndimage_for(dev, "watershed_ift")
    assert ndi.__name__ == "scipy.ndimage"
    assert not backend.is_gpu_array(arr), "moved to the CPU so scipy can run"
    assert caught and "runs on the CPU" in str(caught[0].message)


@needs_gpu
def test_the_fallback_warns_once_per_function():
    backend._warned.discard("geometric_transform")
    dev = backend.to_gpu(_host())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            backend.ndimage_for(dev, "geometric_transform")
    assert len(caught) == 1


@needs_gpu
def test_a_round_trip_preserves_the_data():
    a = np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4)
    np.testing.assert_array_equal(backend.to_cpu(backend.to_gpu(a)), a)
    dev = backend.to_gpu(a)
    assert backend.to_gpu(dev) is dev, "already there, so nothing moves"


@needs_gpu
def test_a_piece_holds_a_device_array_and_stays_on_the_device():
    """A Piece holds a cupy array as-is, and deriving from it never moves it to the CPU."""
    from neu_lib import Frame, Piece

    dev = backend.to_gpu(np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4))
    piece = Piece(dev, Frame(voxel_size_nm=(8, 8, 8), origin_nm=(80, 0, 0)),
                  kind="segmentation")
    assert backend.is_gpu_array(piece.array)
    assert piece.bbox.lo == (10, 0, 0)
    for derived in (piece.copy(), piece.crop(((11, 1, 1), (13, 3, 3))),
                    piece.apply(lambda a: a * 2)):
        assert backend.is_gpu_array(derived.array), "never silently moved to the CPU"


@needs_gpu
def test_a_pieces_copy_is_made_on_the_device():
    from neu_lib import Frame, Piece

    dev = backend.to_gpu(np.zeros((4, 4, 4), "uint8"))
    piece = Piece(dev, Frame(voxel_size_nm=(8, 8, 8)))
    copy = piece.copy()
    copy.array[0, 0, 0] = 99
    assert int(piece.array[0, 0, 0]) == 0, "the original must not move"
    assert int(copy.array[0, 0, 0]) == 99


# --------------------------------------------------------------------------- #
# stage / like: an op may take a CPU array to the GPU and back
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("why", ["op declines", "transfers disabled", "no GPU"])
def test_stage_leaves_the_array_alone(why, monkeypatch):
    a = _host()
    if why == "op declines":
        assert backend.stage(a, prefer_gpu=False) is a
        return
    if why == "transfers disabled":
        monkeypatch.setattr(backend, "enabled", False)
    else:
        monkeypatch.setattr(backend, "gpu_available", lambda: False)
    assert backend.stage(a) is a


def test_like_returns_a_result_to_a_host_reference():
    a, result = _host(), _host()
    assert backend.like(result, a) is result, "already home, so nothing moves"


@needs_gpu
def test_stage_moves_a_host_array_to_the_device():
    a = _host()
    assert backend.is_gpu_array(backend.stage(a))
    dev = backend.to_gpu(a)
    assert backend.stage(dev) is dev, "already there: a chain pays nothing per step"


@needs_gpu
def test_like_gives_back_what_was_handed_in():
    """numpy in, numpy out, so nothing is left on a GPU the caller did not ask to use."""
    cpu, dev = _host(), backend.to_gpu(_host())
    assert not backend.is_gpu_array(backend.like(backend.to_gpu(_host()), cpu))
    assert backend.is_gpu_array(backend.like(_host(), dev))


@needs_gpu
def test_the_op_round_trip_agrees_with_the_host_path():
    import neu_proc

    seg = np.zeros((32,) * 3, "uint32")
    seg[15:17, 15:17, 15:17] = 1
    seg[4:6, 20:22, 8:10] = 2
    on_gpu = neu_proc.dilate(seg, max_distance=6, gpu=True)
    on_host = neu_proc.dilate(seg, max_distance=6, gpu=False)
    assert not backend.is_gpu_array(on_gpu), "numpy in, numpy out"
    np.testing.assert_array_equal(on_gpu, on_host)
