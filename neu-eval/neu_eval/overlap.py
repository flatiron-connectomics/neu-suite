"""The contingency table between two labelings, and the one pass over the voxels.

The only module that reads an array, so the only place a GPU or block boundary shows. A
:class:`Contingency` keeps its pairs sorted by ``(a_id, b_id)`` with no zero counts and no
duplicates, which is what makes ``c1 + c2 == c3`` an exact test of additivity.

**Ignored labels are asymmetric by default**, because 0 means different things: on side
``a``, a partly annotated reference, 0 is "not annotated" and leaves the denominator; on
side ``b``, a dense segmentation, 0 inside an annotated region is an error. But many
pipelines label **membrane** 0. Scored with the default ``ignore_b=()``, that membrane
becomes one huge segment touching every body, a false merge that inflated VOI 2-7x on real
vendor segmentations. Nothing in the array tells the two conventions apart, so pass
``ignore_b=(0,)`` when 0 is membrane, and read ``frac_scored`` beside every score.

``contingency(a, b).T`` is not ``contingency(b, a)``: the second applies the default
ignore sets to the swapped sides.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

#: Above this many ``n_a * n_b`` cells, count pairs by sorting packed keys rather than
#: with a dense ``bincount``, which allocates every cell (2**22 int64 cells is 32 MB).
DENSE_CELL_LIMIT = 1 << 22


@dataclass(frozen=True, eq=False)
class Contingency:
    """Voxel counts for every ``(a_id, b_id)`` pair that co-occurs.

    ``a_ids``, ``b_ids`` and ``counts`` are parallel arrays sorted by ``(a_id, b_id)``,
    with only nonzero pairs, always in CPU memory. ``labels`` names the two sides; it is
    part of equality and addition, so tables with swapped sides cannot be summed.
    """

    a_ids: np.ndarray
    b_ids: np.ndarray
    counts: np.ndarray
    n_ignored: int = 0
    ignore_a: frozenset = frozenset()
    ignore_b: frozenset = frozenset()
    labels: tuple[str, str] = ("a", "b")

    def __post_init__(self) -> None:
        n = self.a_ids.shape[0]
        if self.b_ids.shape[0] != n or self.counts.shape[0] != n:
            raise ValueError(
                f"a_ids, b_ids and counts must be parallel: got {n}, "
                f"{self.b_ids.shape[0]}, {self.counts.shape[0]}")
        if len(self.labels) != 2:
            raise ValueError(f"labels must be a pair of names, got {self.labels!r}")

    # -- basic shape ------------------------------------------------------------------

    @property
    def n_pairs(self) -> int:
        """How many ``(a_id, b_id)`` pairs co-occur."""
        return int(self.a_ids.shape[0])

    @property
    def n_scored(self) -> int:
        """Voxels this table counts. Excludes everything ``ignore_*`` or a mask dropped."""
        return int(self.counts.sum())

    @property
    def n_a(self) -> int:
        """Distinct labels on side ``a`` that survived the ignore sets."""
        return int(np.unique(self.a_ids).size)

    @property
    def n_b(self) -> int:
        return int(np.unique(self.b_ids).size)

    def a_totals(self) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, voxels)`` for side ``a``, ids ascending. The row marginal."""
        return _group_sum(self.a_ids, self.counts)

    def b_totals(self) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, voxels)`` for side ``b``, ids ascending. The column marginal."""
        return _group_sum(self.b_ids, self.counts)

    # -- algebra ----------------------------------------------------------------------

    @property
    def T(self) -> "Contingency":
        """The same table with the sides exchanged (not the same as re-running
        :func:`contingency` with the arguments swapped; see the module docstring)."""
        return _canonical(
            self.b_ids, self.a_ids, self.counts,
            n_ignored=self.n_ignored,
            ignore_a=self.ignore_b, ignore_b=self.ignore_a,
            labels=(self.labels[1], self.labels[0]))

    def __add__(self, other: Any) -> "Contingency":
        """The table for the union of two **disjoint** regions.

        Exact, so summing block tables gives the volume's table. Disjointness is the
        caller's responsibility and cannot be detected; matching ``labels`` and ignore sets
        are checked.
        """
        if other is None or (isinstance(other, int) and other == 0):
            # So that `sum(tables)`, which starts from 0, works.
            return self
        if not isinstance(other, Contingency):
            return NotImplemented
        if self.labels != other.labels:
            raise ValueError(
                f"cannot add tables with different sides: {self.labels!r} vs "
                f"{other.labels!r}. Transpose one, or relabel it deliberately.")
        if self.ignore_a != other.ignore_a or self.ignore_b != other.ignore_b:
            raise ValueError(
                f"cannot add tables that dropped different labels: "
                f"ignore_a {sorted(self.ignore_a)} vs {sorted(other.ignore_a)}, "
                f"ignore_b {sorted(self.ignore_b)} vs {sorted(other.ignore_b)}")
        return _canonical(
            np.concatenate([self.a_ids, other.a_ids]),
            np.concatenate([self.b_ids, other.b_ids]),
            np.concatenate([self.counts, other.counts]),
            n_ignored=self.n_ignored + other.n_ignored,
            ignore_a=self.ignore_a, ignore_b=self.ignore_b, labels=self.labels)

    __radd__ = __add__

    def __eq__(self, other: Any) -> bool:
        """Exact equality of the canonical form, labels and ignore sets included."""
        if not isinstance(other, Contingency):
            return NotImplemented
        return (
            self.labels == other.labels
            and self.ignore_a == other.ignore_a
            and self.ignore_b == other.ignore_b
            and self.n_ignored == other.n_ignored
            and np.array_equal(self.a_ids, other.a_ids)
            and np.array_equal(self.b_ids, other.b_ids)
            and np.array_equal(self.counts, other.counts))

    # -- selection --------------------------------------------------------------------

    def keep_pairs(self, keep: np.ndarray) -> "Contingency":
        """A table holding only the pairs where ``keep`` is True.

        Dropped voxels move into ``n_ignored``, so ``n_scored + n_ignored`` stays the size of
        the region and ``frac_scored`` stays honest.
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.n_pairs,):
            raise ValueError(f"keep must be one flag per pair ({self.n_pairs}), got {keep.shape}")
        dropped = int(self.counts[~keep].sum())
        return replace(
            self,
            a_ids=self.a_ids[keep], b_ids=self.b_ids[keep], counts=self.counts[keep],
            n_ignored=self.n_ignored + dropped)

    def restrict(self, *, a_ids: Iterable | None = None,
                 b_ids: Iterable | None = None) -> "Contingency":
        """A table restricted to the given labels on either side, or both.

        For example, the labels :func:`neu_eval.assign.completeness` keeps.
        """
        keep = np.ones(self.n_pairs, dtype=bool)
        if a_ids is not None:
            keep &= np.isin(self.a_ids, np.asarray(list(a_ids), dtype=self.a_ids.dtype))
        if b_ids is not None:
            keep &= np.isin(self.b_ids, np.asarray(list(b_ids), dtype=self.b_ids.dtype))
        return self.keep_pairs(keep)

    def relabel_sides(self, labels: Sequence[str]) -> "Contingency":
        """The same counts, with the two sides named differently."""
        return replace(self, labels=(str(labels[0]), str(labels[1])))

    # -- output -----------------------------------------------------------------------

    def to_frame(self):
        """A DataFrame of the pairs, columns named after the two sides. Needs pandas
        (``neu-eval[report]``)."""
        import pandas as pd

        a, b = self.labels
        return pd.DataFrame({
            f"{a}_id": self.a_ids,
            f"{b}_id": self.b_ids,
            "n_voxels": self.counts,
        })

    def __repr__(self) -> str:
        a, b = self.labels
        return (f"Contingency({a}={self.n_a}, {b}={self.n_b}, pairs={self.n_pairs}, "
                f"scored={self.n_scored}, ignored={self.n_ignored})")


def pool(tables: Sequence[Contingency]) -> Contingency:
    """Combine tables from regions whose label spaces are **independent**.

    Use ``+`` for regions of one labeling, where a shared id is one body. Use this for
    separately annotated crops, where each numbers its bodies from 1 and a shared id means
    nothing: adding those tables would fuse unrelated cells. Each table's ids are renumbered
    and offset past the previous table's, as ``neu-vol relabel`` does to a volume.

    Pooled ids do not name bodies, so the result is for scalar metrics only.
    """
    tables = list(tables)
    if not tables:
        raise ValueError("pool() needs at least one table")
    first = tables[0]
    for other in tables[1:]:
        if other.labels != first.labels:
            raise ValueError(
                f"cannot pool tables with different sides: {first.labels!r} vs "
                f"{other.labels!r}")
        if other.ignore_a != first.ignore_a or other.ignore_b != first.ignore_b:
            raise ValueError("cannot pool tables that dropped different labels")

    a_parts, b_parts, count_parts = [], [], []
    a_offset = b_offset = np.uint64(1)          # 1-based, so 0 stays free of meaning
    for table in tables:
        if table.n_pairs == 0:
            continue
        a_uniq, a_inv = np.unique(table.a_ids, return_inverse=True)
        b_uniq, b_inv = np.unique(table.b_ids, return_inverse=True)
        a_parts.append(a_inv.reshape(-1).astype(np.uint64) + a_offset)
        b_parts.append(b_inv.reshape(-1).astype(np.uint64) + b_offset)
        count_parts.append(table.counts)
        a_offset += np.uint64(a_uniq.size)
        b_offset += np.uint64(b_uniq.size)

    if not count_parts:
        return _canonical(
            np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros(0, np.uint64),
            n_ignored=sum(t.n_ignored for t in tables),
            ignore_a=first.ignore_a, ignore_b=first.ignore_b, labels=first.labels)

    return _canonical(
        np.concatenate(a_parts), np.concatenate(b_parts), np.concatenate(count_parts),
        n_ignored=sum(t.n_ignored for t in tables),
        ignore_a=first.ignore_a, ignore_b=first.ignore_b, labels=first.labels)


def contingency(a: Any, b: Any, *,
                ignore_a: Iterable[int] = (0,),
                ignore_b: Iterable[int] = (),
                mask: Any = None,
                labels: Sequence[str] = ("a", "b")) -> Contingency:
    """Count the voxels shared by every pair of labels in ``a`` and ``b``.

    The arrays must have the same shape and already be registered, e.g. read with
    ``read_piece(seg, level=..., crop=gt_piece)``. ``ignore_a`` / ``ignore_b`` drop those
    label values and ``mask`` restricts to where it is True; dropped voxels are counted in
    ``n_ignored``. See the module docstring for the asymmetric defaults, and especially
    for segmentations that label membrane 0.

    Runs where the arrays live, CPU or GPU (both must be on the same one); the table comes
    back in CPU memory.
    """
    from neu_proc.ops.backend import array_module, to_cpu

    a = _as_labels(a, "a")
    b = _as_labels(b, "b")
    if a.shape != b.shape:
        raise ValueError(
            f"the two labelings must be the same shape: {a.shape} vs {b.shape}. Two crops "
            f"of one physical box come from read_piece(..., crop=piece), which registers "
            f"them by construction.")

    xp = array_module(a)
    if type(a).__module__.split(".")[0] != type(b).__module__.split(".")[0]:
        raise TypeError(
            "both labelings must live in the same place (both CPU, or both GPU): got "
            f"{type(a).__module__.split('.')[0]} and {type(b).__module__.split('.')[0]}. "
            "Move one with neu_proc.ops.backend.to_gpu / to_cpu.")

    n_total = int(a.size)
    af = a.reshape(-1)
    bf = b.reshape(-1)

    ignore_a = frozenset(int(v) for v in ignore_a)
    ignore_b = frozenset(int(v) for v in ignore_b)

    # A mask is applied to the voxels; ignored labels are applied to the table, below.
    if mask is not None:
        mask = xp.asarray(mask)
        if mask.shape != a.shape:
            raise ValueError(f"mask must match the labelings: {mask.shape} vs {a.shape}")
        keep = mask.reshape(-1).astype(bool)
        af = af[keep]
        bf = bf[keep]
    n_counted = int(af.shape[0])

    a_ids, b_ids, counts = _count_pairs(xp, af, bf)
    a_ids, b_ids, counts = to_cpu(a_ids), to_cpu(b_ids), to_cpu(counts)

    # Dropping whole rows and columns of the table is exact, and O(pairs) rather than a
    # pass over the voxels, so the default `ignore_a=(0,)` costs nothing.
    n_dropped = 0
    if (ignore_a or ignore_b) and counts.size:
        drop = np.zeros(counts.shape, dtype=bool)
        if ignore_a:
            drop |= np.isin(a_ids, np.fromiter(ignore_a, dtype=np.uint64, count=len(ignore_a)))
        if ignore_b:
            drop |= np.isin(b_ids, np.fromiter(ignore_b, dtype=np.uint64, count=len(ignore_b)))
        if drop.any():
            n_dropped = int(counts[drop].sum())
            keep_rows = ~drop
            a_ids, b_ids, counts = a_ids[keep_rows], b_ids[keep_rows], counts[keep_rows]

    return _canonical(
        a_ids, b_ids, counts,
        n_ignored=n_total - n_counted + n_dropped,
        ignore_a=ignore_a, ignore_b=ignore_b,
        labels=(str(labels[0]), str(labels[1])))


# -- internals ------------------------------------------------------------------------

def _as_labels(arr: Any, which: str):
    """``arr`` as an integer label array; refuses floats and negative ids."""
    if arr is None:
        raise TypeError(f"labeling {which} is None")
    if not hasattr(arr, "dtype") or not hasattr(arr, "shape"):
        arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        raise TypeError(
            f"labeling {which} has dtype {arr.dtype}: a float label array is either an "
            f"image read as a segmentation or ids that have already lost precision. Read "
            f"it with kind='segmentation' and an integer dtype.")
    if arr.dtype.kind == "b":
        return arr.astype(np.uint8)
    if arr.dtype.kind not in "ui":
        raise TypeError(f"labeling {which} has dtype {arr.dtype}, which is not an integer")
    if arr.dtype.kind == "i":
        # A negative id would wrap to a huge uint64 and still be counted.
        if arr.size and int(arr.min()) < 0:
            raise ValueError(
                f"labeling {which} holds negative values, which are not label ids")
    return arr


def _factorize(xp, arr):
    """``(ids, dense)``: the distinct labels, and each voxel's index into them.

    On the CPU this uses ``fastremap.renumber`` (hash-based, about 30x faster than
    ``np.unique`` here); on the GPU, cupy's ``unique``. The two number the dense indices
    differently, which nothing downstream depends on.
    """
    from neu_proc.ops.backend import is_gpu_array

    if is_gpu_array(arr):
        ids, dense = xp.unique(arr, return_inverse=True)
        return ids, dense.reshape(-1)

    import fastremap

    dense, mapping = fastremap.renumber(np.ascontiguousarray(arr), in_place=False)
    # Invert renumber's original -> new mapping. It always maps 0 -> 0, which may leave
    # index 0 unused.
    ids = np.zeros(int(dense.max()) + 1, dtype=np.uint64)
    for original, new in mapping.items():
        if new < ids.size:
            ids[new] = original
    return ids, dense


def _count_pairs(xp, af, bf):
    """``(a_ids, b_ids, counts)`` for the co-occurring pairs, wherever they live.

    Each side is renumbered densely, and each pair packed into one integer and counted.
    """
    if af.shape[0] == 0:
        empty_id = xp.zeros(0, dtype=xp.uint64)
        return empty_id, empty_id, xp.zeros(0, dtype=xp.uint64)

    ua, ia = _factorize(xp, af)
    ub, ib = _factorize(xp, bf)
    n_a = int(ua.shape[0])
    n_b = int(ub.shape[0])
    if n_a > 2**32 or n_b > 2**32:
        raise ValueError(
            f"{n_a} labels on side a and {n_b} on side b: more than 2**32 distinct labels "
            f"on one side does not fit the packed key. Score a region at a time.")

    if n_a * n_b <= DENSE_CELL_LIMIT:
        flat = ia.astype(xp.int64) * n_b + ib.astype(xp.int64)
        hist = xp.bincount(flat, minlength=n_a * n_b)
        nz = xp.nonzero(hist)[0]
        counts = hist[nz]
        ai = nz // n_b
        bi = nz - ai * n_b
    else:
        packed = (ia.astype(xp.uint64) << xp.uint64(32)) | ib.astype(xp.uint64)
        keys, counts = xp.unique(packed, return_counts=True)
        ai = (keys >> xp.uint64(32)).astype(xp.int64)
        bi = (keys & xp.uint64(0xFFFFFFFF)).astype(xp.int64)

    return ua[ai].astype(xp.uint64), ub[bi].astype(xp.uint64), counts.astype(xp.uint64)


def _canonical(a_ids, b_ids, counts, **kwargs) -> Contingency:
    """Build a :class:`Contingency` in canonical form: grouped, ordered, no empty pairs."""
    a_ids = np.asarray(a_ids, dtype=np.uint64).reshape(-1)
    b_ids = np.asarray(b_ids, dtype=np.uint64).reshape(-1)
    counts = np.asarray(counts, dtype=np.uint64).reshape(-1)

    if a_ids.size:
        order = np.lexsort((b_ids, a_ids))
        a_ids, b_ids, counts = a_ids[order], b_ids[order], counts[order]
        # Sum duplicates, which addition creates wherever two regions saw the same pair.
        new = np.empty(a_ids.shape[0], dtype=bool)
        new[0] = True
        np.not_equal(a_ids[1:], a_ids[:-1], out=new[1:])
        new[1:] |= b_ids[1:] != b_ids[:-1]
        if not new.all():
            groups = np.cumsum(new) - 1
            counts = np.bincount(groups, weights=counts.astype(np.float64))
            # float64 weights are exact up to 2**53 voxels; bincount has no integer weights.
            counts = counts.astype(np.uint64)
            a_ids = a_ids[new]
            b_ids = b_ids[new]
        nonzero = counts > 0
        if not nonzero.all():
            a_ids, b_ids, counts = a_ids[nonzero], b_ids[nonzero], counts[nonzero]

    return Contingency(a_ids=a_ids, b_ids=b_ids, counts=counts, **kwargs)


def _group_sum(ids: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unique ids ascending, summed counts)``."""
    if ids.size == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)
    uniq, inverse = np.unique(ids, return_inverse=True)
    totals = np.bincount(inverse.reshape(-1), weights=counts.astype(np.float64))
    return uniq, totals.astype(np.uint64)


def boundary_band(labels, width: int = 2, *, ignore: Iterable[int] = (0,)):
    """Boolean mask of voxels within ``width`` voxels of a boundary in ``labels``.

    For a boundary-tolerant score, pass ``mask=~boundary_band(gt)`` to
    :func:`contingency`, so small differences in where a boundary is drawn are not counted.
    Report it beside the ordinary score, not instead of it: a wide band also removes thin
    processes entirely. A boundary is any 6-neighbour pair of different labels, ignored
    values included.
    """
    from scipy import ndimage as ndi

    arr = np.asarray(labels)
    edge = np.zeros(arr.shape, bool)
    for ax in range(arr.ndim):
        lo = [slice(None)] * arr.ndim; hi = [slice(None)] * arr.ndim
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        diff = arr[tuple(lo)] != arr[tuple(hi)]
        edge[tuple(lo)] |= diff
        edge[tuple(hi)] |= diff
    if width > 1:
        edge = ndi.binary_dilation(edge, iterations=width - 1)
    return edge
