"""Choosing between scipy and cupyx: :mod:`neu_proc.ops.backend`.

Split so the whole design is tested **without a GPU** — CI has none, and a layer whose logic
can only be checked on one machine is a layer nobody can refactor. A fake device array
carries the one property dispatch reads (the type's defining module), which is enough for
every branch; the real-device tests are gated on cupy and skip otherwise.
"""

import warnings

import numpy as np
import pytest

from neu_proc.ops import backend


class FakeDeviceArray(np.ndarray):
    """A numpy array that claims to be cupy's.

    Dispatch reads ``type(a).__module__`` and nothing else — deliberately, so that no host
    install imports cupy to discover it does not need it — which means a subclass lying about
    that module exercises every branch. Subclassing ndarray rather than mimicking one keeps
    the rest of the assertions about real data.
    """
    __module__ = "cupy"

    @classmethod
    def of(cls, array):
        return array.view(cls)


def _host(shape=(4, 4, 4)):
    return np.zeros(shape, "uint8")


# --------------------------------------------------------------------------- #
# dispatch — no GPU needed
# --------------------------------------------------------------------------- #
def test_dispatch_reads_the_type_and_imports_nothing():
    """Importing cupy to ask whether an array is numpy would make every host-only install
    pay for a CUDA stack, and on a machine without one it would fail outright."""
    import subprocess
    import sys

    assert not backend.is_device_array(_host())
    assert backend.is_device_array(FakeDeviceArray.of(_host()))

    out = subprocess.run(
        [sys.executable, "-c",
         "import numpy as np\n"
         "from neu_proc.ops import backend\n"
         "backend.is_device_array(np.zeros((2, 2, 2)))\n"
         "backend.array_module(np.zeros((2, 2, 2)))\n"
         "import sys; assert 'cupy' not in sys.modules, sorted(sys.modules)[:5]\n"
         "print('ok')"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_a_host_array_gets_scipy():
    ndi, arr = backend.ndimage_for(_host())
    assert ndi.__name__ == "scipy.ndimage"
    assert arr is not None and not backend.is_device_array(arr)
    assert backend.array_module(_host()).__name__ == "numpy"


def test_to_host_is_a_no_op_for_a_host_array():
    a = _host()
    assert backend.to_host(a) is a


def test_the_switch_governs_TRANSFERS_not_dispatch(monkeypatch):
    """A flag that disabled dispatch would, handed a device array anyway, run the host code
    on device memory and raise somewhere less obvious. So `enabled` only stops `to_device`
    from moving anything; the namespace still follows the array."""
    monkeypatch.setattr(backend, "enabled", False)
    a = _host()
    assert backend.to_device(a) is a, "disabled: nothing moves"
    # ...while dispatch is unchanged, because it is a fact about the array
    fake = FakeDeviceArray.of(a)
    assert backend.is_device_array(fake)


def test_to_device_is_a_no_op_without_a_gpu(monkeypatch):
    """One piece of calling code has to run on a host-only machine, so this returns the array
    rather than raising."""
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    a = _host()
    assert backend.to_device(a) is a


def test_the_env_var_is_read_at_import(monkeypatch):
    """`NEU_PROC_GPU=0` turns transfers off for a process without touching code, which is
    what a benchmark or a bisect wants."""
    import importlib

    monkeypatch.setenv("NEU_PROC_GPU", "0")
    reloaded = importlib.reload(backend)
    try:
        assert reloaded.enabled is False
    finally:
        monkeypatch.setenv("NEU_PROC_GPU", "1")
        importlib.reload(backend)


def test_gpu_available_is_false_rather_than_raising_when_cupy_is_broken(monkeypatch):
    """cupy installs fine with no driver, no device or a CUDA mismatch, and then fails at the
    first allocation rather than at import — so both halves are probed and neither raises."""
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
# the real device
# --------------------------------------------------------------------------- #
#: Gated on :func:`backend.gpu_available` rather than ``importorskip("cupy")``, because that
#: function is the one that answers the whole question: cupy installs fine with no driver, no
#: device or a CUDA mismatch, and then fails at the first *allocation* — which importorskip
#: would wave through into a confusing failure. It never raises, so collection is
#: unconditional either way.
needs_gpu = pytest.mark.skipif(
    not backend.gpu_available(),
    reason="no GPU backend; the host paths above cover the dispatch logic")


@needs_gpu
def test_a_device_array_gets_cupyx():
    dev = backend.to_device(_host())
    assert backend.is_device_array(dev)
    ndi, arr = backend.ndimage_for(dev, "grey_dilation")
    assert ndi.__name__ == "cupyx.scipy.ndimage"
    assert arr is dev, "no move when cupyx has the function"
    assert backend.array_module(dev).__name__ == "cupy"


@needs_gpu
def test_a_missing_cupyx_function_falls_back_LOUDLY():
    """Measured against cupy 14.2 / scipy 1.18, cupyx lacks four real functions:
    distance_transform_bf, distance_transform_cdt, geometric_transform, watershed_ift.
    Silence here would turn "I am using the GPU" into the slowest path available."""
    backend._warned.discard("watershed_ift")
    dev = backend.to_device(_host())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ndi, arr = backend.ndimage_for(dev, "watershed_ift")
    assert ndi.__name__ == "scipy.ndimage"
    assert not backend.is_device_array(arr), "moved to the host so scipy can run"
    assert caught and "runs on the HOST" in str(caught[0].message)


@needs_gpu
def test_the_fallback_warns_once_per_function():
    """A filter in a loop would otherwise bury everything else in the log."""
    backend._warned.discard("geometric_transform")
    dev = backend.to_device(_host())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            backend.ndimage_for(dev, "geometric_transform")
    assert len(caught) == 1


@needs_gpu
def test_a_round_trip_preserves_the_data():
    a = np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4)
    np.testing.assert_array_equal(backend.to_host(backend.to_device(a)), a)
    dev = backend.to_device(a)
    assert backend.to_device(dev) is dev, "already there, so nothing moves"


@needs_gpu
def test_a_piece_holds_a_device_array_and_stays_on_the_device():
    """The reason `Piece.__post_init__` stopped calling np.asanyarray unconditionally: cupy
    makes __array__ raise on purpose, so coercing every input meant a device array could not
    be held at all."""
    from neu_lib import Frame, Piece

    dev = backend.to_device(np.arange(4 * 4 * 4, dtype="uint8").reshape(4, 4, 4))
    piece = Piece(dev, Frame(voxel_size_nm=(8, 8, 8), origin_nm=(80, 0, 0)),
                  kind="segmentation")
    assert backend.is_device_array(piece.array)
    assert piece.bbox.lo == (10, 0, 0)
    for derived in (piece.copy(), piece.crop(((11, 1, 1), (13, 3, 3))),
                    piece.apply(lambda a: a * 2)):
        assert backend.is_device_array(derived.array), "never silently moved to the host"


@needs_gpu
def test_a_pieces_copy_is_made_on_the_device():
    from neu_lib import Frame, Piece

    dev = backend.to_device(np.zeros((4, 4, 4), "uint8"))
    piece = Piece(dev, Frame(voxel_size_nm=(8, 8, 8)))
    copy = piece.copy()
    copy.array[0, 0, 0] = 99
    assert int(piece.array[0, 0, 0]) == 0, "the original must not move"
    assert int(copy.array[0, 0, 0]) == 99


# --------------------------------------------------------------------------- #
# stage / like — an op may take a host array to the device and back
# --------------------------------------------------------------------------- #
def test_stage_declines_when_the_op_says_it_is_not_worth_it():
    """`prefer_gpu` is the op's call because arithmetic intensity is a property of the op:
    a dilation earns the round trip (192^3: 811 ms on scipy against 16 ms with both
    transfers) while a threshold is bandwidth-bound and can only lose it."""
    a = _host()
    assert backend.stage(a, prefer_gpu=False) is a


def test_stage_declines_when_transfers_are_disabled(monkeypatch):
    monkeypatch.setattr(backend, "enabled", False)
    a = _host()
    assert backend.stage(a) is a


def test_stage_declines_without_a_device(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    a = _host()
    assert backend.stage(a) is a


def test_like_returns_a_result_to_a_host_reference():
    a, result = _host(), _host()
    assert backend.like(result, a) is result, "already home, so nothing moves"


@needs_gpu
def test_stage_moves_a_host_array_to_the_device():
    a = _host()
    staged = backend.stage(a)
    assert backend.is_device_array(staged)
    dev = backend.to_device(a)
    assert backend.stage(dev) is dev, "already there: a chain pays nothing per step"


@needs_gpu
def test_like_gives_back_what_was_handed_in():
    """So an op returns numpy for a numpy input and nothing silently leaves a caller's data
    on a GPU they did not ask to use — which is what makes `stage` safe by default."""
    host, dev = _host(), backend.to_device(_host())
    assert not backend.is_device_array(backend.like(backend.to_device(_host()), host))
    assert backend.is_device_array(backend.like(_host(), dev))


@needs_gpu
def test_the_op_round_trip_agrees_with_the_host_path():
    """The whole point of allowing the automatic transfer: same answer, much faster. A
    difference here would mean the two backends disagree, which no timing makes acceptable."""
    import neu_proc

    seg = np.zeros((32,) * 3, "uint32")
    seg[15:17, 15:17, 15:17] = 1
    seg[4:6, 20:22, 8:10] = 2
    on_gpu = neu_proc.dilate(seg, max_distance=6, gpu=True)
    on_host = neu_proc.dilate(seg, max_distance=6, gpu=False)
    assert not backend.is_device_array(on_gpu), "numpy in, numpy out"
    np.testing.assert_array_equal(on_gpu, on_host)
