"""The two backends must produce the same table, and the host factorization must be sound.

The device half skips without a GPU, so this file is mostly inert in CI — which is exactly
why the host half is here too: ``_factorize`` uses ``fastremap.renumber`` on the host and
``unique`` on the device, and those are *different algorithms*, not one function with a
flag. The property that makes swapping them safe is that a factorization's dense numbering
is arbitrary, so the host test pins that the numbering is a valid one rather than assuming
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


# -- the host factorization -------------------------------------------------------------

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


# -- the device path --------------------------------------------------------------------

def _gpu():
    from neu_proc.ops.backend import gpu_available

    return gpu_available()


gpu = pytest.mark.skipif(not _gpu(), reason="no GPU available")


@gpu
def test_the_device_gives_the_same_table_as_the_host():
    from neu_proc.ops.backend import to_device

    a, b = _pair()
    assert contingency(to_device(a), to_device(b)) == contingency(a, b)


@gpu
def test_the_device_gives_the_same_table_on_the_sorted_key_path(monkeypatch):
    from neu_proc.ops.backend import to_device

    a, b = _pair()
    expected = contingency(a, b)
    monkeypatch.setattr(overlap, "DENSE_CELL_LIMIT", 0)
    assert contingency(to_device(a), to_device(b)) == expected


@gpu
def test_the_table_comes_back_on_the_host_whatever_went_in():
    """The reduction leaves the device, so no consumer has to know about one."""
    from neu_proc.ops.backend import to_device

    c = contingency(*(to_device(x) for x in _pair()))
    for name in ("a_ids", "b_ids", "counts"):
        assert isinstance(getattr(c, name), np.ndarray)


@gpu
def test_a_mask_works_on_the_device_too():
    from neu_proc.ops.backend import to_device

    a, b = _pair()
    mask = np.zeros(a.shape, dtype=bool)
    mask[:5] = True
    assert (contingency(to_device(a), to_device(b), mask=to_device(mask))
            == contingency(a, b, mask=mask))


@gpu
def test_mixing_a_host_and_a_device_labeling_is_refused():
    from neu_proc.ops.backend import to_device

    a, b = _pair()
    with pytest.raises(TypeError, match="same place"):
        contingency(to_device(a), b)
