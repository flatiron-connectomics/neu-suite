"""The CPU and GPU backends give the same table, and the CPU factorization is sound.

The GPU tests skip without a GPU. ``_factorize`` uses ``fastremap.renumber`` on the CPU and
``unique`` on the GPU, different algorithms whose dense numbering is arbitrary, so the CPU
test checks the numbering is a valid factorization rather than a particular one.
"""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval import overlap
from neu_eval.overlap import contingency


def _pair(shape=(10, 9, 8), n_a=6, n_b=9, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_a, size=shape, dtype=np.uint64),
            rng.integers(0, n_b, size=shape, dtype=np.uint64))


# -- the CPU factorization -------------------------------------------------------------

def test_the_factorization_gives_one_dense_index_per_distinct_label():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 40, size=5000, dtype=np.uint64) * np.uint64(7919)
    ids, dense = overlap._factorize(np, arr)
    # Injective on the labels that occur, and the round trip is exact.
    used = np.unique(dense)
    assert np.unique(ids[used]).size == used.size
    assert np.array_equal(ids[dense], arr)


def test_huge_ids_reach_the_table_intact():
    """Nothing may index by label value: real ids reach 2**63."""
    a = np.array([[[2**63 + 5, 2**63 + 5]]], dtype=np.uint64)
    b = np.array([[[10**18, 10**18]]], dtype=np.uint64)
    c = contingency(a, b)
    assert int(c.a_ids[0]) == 2**63 + 5
    assert int(c.b_ids[0]) == 10**18


# -- the GPU path --------------------------------------------------------------------

def _gpu():
    from neu_proc.ops.backend import gpu_available

    return gpu_available()


gpu = pytest.mark.skipif(not _gpu(), reason="no GPU available")


def _gpu_array(arr):
    """``arr`` on the GPU via cupy directly. Not ``to_gpu``, which under ``NEU_PROC_GPU=0``
    returns the array unchanged, so these tests would compare the CPU with itself."""
    import cupy

    return cupy.asarray(arr)


@gpu
@pytest.mark.parametrize("options", [
    {},
    {"mask": "first-five-planes"},
    {"ignore_a": (0, 1), "ignore_b": (2,)},
], ids=["plain", "mask", "ignore"])
def test_the_gpu_gives_the_same_table_as_the_cpu(options):
    a, b = _pair()
    options = dict(options)
    if options.get("mask"):
        mask = np.zeros(a.shape, dtype=bool)
        mask[:5] = True
        options["mask"] = mask
    gpu_options = {k: (_gpu_array(v) if k == "mask" else v) for k, v in options.items()}
    assert (contingency(_gpu_array(a), _gpu_array(b), **gpu_options)
            == contingency(a, b, **options))


@gpu
def test_the_gpu_gives_the_same_table_on_the_sorted_key_path(monkeypatch):
    a, b = _pair()
    expected = contingency(a, b)
    monkeypatch.setattr(overlap, "DENSE_CELL_LIMIT", 0)
    assert contingency(_gpu_array(a), _gpu_array(b)) == expected


@gpu
def test_the_table_comes_back_in_cpu_memory_whatever_went_in():
    """No consumer of a table has to know about the GPU."""
    c = contingency(*(_gpu_array(x) for x in _pair()))
    for name in ("a_ids", "b_ids", "counts"):
        assert isinstance(getattr(c, name), np.ndarray)


@gpu
def test_mixing_a_cpu_and_a_gpu_labeling_is_refused():
    a, b = _pair()
    with pytest.raises(TypeError, match="same place"):
        contingency(_gpu_array(a), b)
