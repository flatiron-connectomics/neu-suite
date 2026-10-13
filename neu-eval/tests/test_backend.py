"""The two backends must produce the same table, and the CPU factorization must be sound.

The GPU half skips without a GPU, so this file is mostly inert in CI — which is exactly
why the CPU half is here too: ``_factorize`` uses ``fastremap.renumber`` on the CPU and
``unique`` on the GPU, and those are *different algorithms*, not one function with a
flag. The property that makes swapping them safe is that a factorization's dense numbering
is arbitrary, so the CPU test pins that the numbering is a valid one rather than assuming
it.
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

def test_the_factorization_recovers_the_original_ids():
    arr = np.array([7, 7, 900, 0, 900, 12], dtype=np.uint64)
    ids, dense = overlap._factorize(np, arr)
    assert np.array_equal(ids[dense], arr)


def test_the_factorization_gives_one_dense_index_per_distinct_label():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 40, size=5000, dtype=np.uint64) * np.uint64(7919)
    ids, dense = overlap._factorize(np, arr)
    # A valid factorization: the map is injective on the labels that occur, and the round
    # trip is exact. The particular numbering is not part of the contract.
    used = np.unique(dense)
    assert np.unique(ids[used]).size == used.size
    assert np.array_equal(ids[dense], arr)


def test_a_huge_sparse_id_survives_the_factorization():
    """Real GT carries ids like 3,371,243 for 475 labels; nothing may index by value."""
    arr = np.array([2**63 + 5, 1, 2**63 + 5], dtype=np.uint64)
    ids, dense = overlap._factorize(np, arr)
    assert np.array_equal(ids[dense], arr)


def test_huge_ids_reach_the_table_intact():
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
    """``arr`` on the GPU, via cupy directly rather than through ``to_gpu``.

    Deliberately bypassing the backend helper. ``to_gpu`` honours ``NEU_PROC_GPU=0`` and
    returns the array untouched, which is right for production code and *wrong* for a test
    of the GPU path: under that switch every assertion below would compare the CPU
    against itself and pass for the wrong reason. Caught by running the suite with the
    switch off, which is the only reason it was visible at all.
    """
    import cupy

    return cupy.asarray(arr)


@gpu
def test_the_gpu_gives_the_same_table_as_the_cpu():
    a, b = _pair()
    assert contingency(_gpu_array(a), _gpu_array(b)) == contingency(a, b)


@gpu
def test_the_gpu_gives_the_same_table_on_the_sorted_key_path(monkeypatch):
    a, b = _pair()
    expected = contingency(a, b)
    monkeypatch.setattr(overlap, "DENSE_CELL_LIMIT", 0)
    assert contingency(_gpu_array(a), _gpu_array(b)) == expected


@gpu
def test_the_table_comes_back_in_cpu_memory_whatever_went_in():
    """The reduction leaves the GPU, so no consumer has to know about one."""
    c = contingency(*(_gpu_array(x) for x in _pair()))
    for name in ("a_ids", "b_ids", "counts"):
        assert isinstance(getattr(c, name), np.ndarray)


@gpu
def test_a_mask_works_on_the_gpu_too():
    a, b = _pair()
    mask = np.zeros(a.shape, dtype=bool)
    mask[:5] = True
    assert (contingency(_gpu_array(a), _gpu_array(b), mask=_gpu_array(mask))
            == contingency(a, b, mask=mask))


@gpu
def test_ignored_labels_are_dropped_on_the_gpu_too():
    a, b = _pair()
    assert (contingency(_gpu_array(a), _gpu_array(b), ignore_a=(0, 1), ignore_b=(2,))
            == contingency(a, b, ignore_a=(0, 1), ignore_b=(2,)))


@gpu
def test_mixing_a_cpu_and_a_gpu_labeling_is_refused():
    a, b = _pair()
    with pytest.raises(TypeError, match="same place"):
        contingency(_gpu_array(a), b)
