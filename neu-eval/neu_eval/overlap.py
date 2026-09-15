"""The contingency table between two labelings, and the one pass over the voxels.

This is the only module in the package that reads an array. Everything else operates on
what it returns, which is why it is also the only place a GPU or a block boundary is
visible.

**Canonical form is what makes the algebra work.** A :class:`Contingency` holds its pairs
sorted by ``(a_id, b_id)``, with no zero counts and no duplicates. Without that,
``c1 + c2 == c3`` would be a statement about insertion order rather than about the
partition, and the additivity test — the property the blockwise sweep is built on — could
pass on a wrong implementation and fail on a right one. It is the same argument
``NEU-PROC-PLAN``'s canonical-id decision makes: a test that only holds for one traversal
order is a test that gets weakened the first time it fails.

**Ignored labels are asymmetric on purpose.** The default drops label 0 on side ``a`` and
nothing on side ``b``, and that asymmetry is about what 0 *means* on each side, not about
which side is authoritative. In the common call side ``a`` is a partially-annotated
reference, where 0 is "nobody looked here" and must leave the denominator; side ``b`` is a
dense segmentation, where 0 is "assigned to no body" inside a region somebody did annotate
— an error, not an absence. Scoring the second as if it were the first is the ordinary way
these numbers come out flattering.

**But "0 means membrane" is a common convention, and then the default is badly wrong.**
Several pipelines label membranes 0 and cell interiors with body ids. Scored with
``ignore_b=()`` that invents one enormous segment threading between every cell in the
volume, so it touches nearly every reference body at once and registers as a colossal false
merge. Measured on two vendor segmentations of the same specimen, ``ignore_b=(0,)`` against
``ignore_b=()``: VOI **0.63 vs 1.37**, **0.51 vs 2.01**, **0.96 vs 2.16**, **0.29 vs 1.98**
— a factor of 2.2 to 6.9, all of it artefact. Nothing in the array distinguishes the two
conventions, so **the caller has to know and say**; there is no safe guess, which is why
there is no attempt at one.

The guard against the obvious abuse — declaring everything membrane so almost nothing is
scored — is ``n_ignored``, and therefore ``frac_scored`` in every summary. On those same two
vendors it reads 93.6% for one and 84.5–87.1% for the other, which is a real difference in
how much each declined to label and belongs beside their scores rather than buried.

One consequence worth stating because it looks like a bug: ``contingency(a, b).T`` is not
``contingency(b, a)``. Transposing relabels the axes of a table already built;
re-running the kernel applies the *defaults* to the new side ``a``. Both are right, and if
you want the second, say so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

#: Above this many cells, count the packed keys with a sort instead of a dense histogram.
#: `bincount` is O(voxels) and beats a sort handily, but it allocates `n_a * n_b` int64
#: cells whether or not they are occupied — so the choice is between paying for the empty
#: cells and paying for the sort. 2**22 cells is 32 MB, which a GT crop (hundreds of
#: reference bodies against thousands of segments) stays comfortably under, and a
#: whole-volume pair with millions of ids on both sides never would.
DENSE_CELL_LIMIT = 1 << 22


@dataclass(frozen=True, eq=False)
class Contingency:
    """Voxel counts for every ``(a_id, b_id)`` pair that co-occurs.

    Sparse and canonical: ``a_ids``, ``b_ids`` and ``counts`` are parallel arrays sorted by
    ``(a_id, b_id)``, carrying only pairs with a nonzero count. Always in **CPU memory** —
    the table is ``O(pairs)`` where the input is ``O(voxels)``, so there is nothing to gain
    by leaving it on the GPU and a great deal to lose in every consumer having to care.

    ``labels`` names the two sides for reporting. It takes part in equality and in addition
    because two tables built with the sides swapped must not silently sum.
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
        """The same table with the two sides exchanged.

        Not the same thing as re-running :func:`contingency` with the arguments swapped —
        see the module docstring.
        """
        return _canonical(
            self.b_ids, self.a_ids, self.counts,
            n_ignored=self.n_ignored,
            ignore_a=self.ignore_b, ignore_b=self.ignore_a,
            labels=(self.labels[1], self.labels[0]))

    def __add__(self, other: Any) -> "Contingency":
        """The table for the union of two **disjoint** regions.

        Exact, and the property the blockwise sweep rests on: summing the tables of the
        blocks of a volume gives the table of the volume, so no metric code has to know
        that blocks exist.

        Disjointness is the caller's to guarantee — two tables over overlapping regions add
        to something with no meaning, and there is nothing in a reduced table to detect it
        with. What *is* checked is the framing: both sides must name their labels the same
        way and have dropped the same labels, because adding tables that disagree about
        either is the failure that produces a plausible number from mismatched halves.
        """
        if other is None or (isinstance(other, int) and other == 0):
            # `sum(tables)` starts from the integer 0, and summing a list of block tables
            # is the natural way to write the driver-side reduction.
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
        """Exact equality of the canonical form, framing included.

        Spelled out because the default dataclass ``__eq__`` compares numpy arrays with
        ``==`` and lands on the ambiguous-truth-value error — and because this is the
        assertion the additivity test makes, so it had better mean what it says.
        """
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

        The dropped voxels move into ``n_ignored``, so ``n_scored + n_ignored`` stays the
        size of the region the table was built over however it is whittled down. That is
        what lets a report say what fraction was actually scored — which is the number an
        adjudicated metric has to disclose.
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

        What ``thresholded_fragmentation`` needs: take the largest segments covering some
        fraction of each side, then measure only those.
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
        """A DataFrame of the pairs, columns named after the two sides.

        pandas is imported here rather than at module scope: it arrives with
        ``neu-eval[report]``, and nothing that computes a metric needs it.
        """
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

    Not the same operation as ``+``, and confusing the two is the chimera bug in table form.
    ``c1 + c2`` is for two regions of *one* labeling, where a shared id means one body.
    :func:`pool` is for separately annotated crops, where every crop numbers its bodies from
    1 and a shared id means **nothing** — so adding those tables directly would fuse
    unrelated cells exactly the way scoring an un-relabelled multi-crop volume does, and
    would produce a plausible pooled number from a partition that does not exist.

    So each table's labels are renumbered densely and offset past the previous table's
    before the sum. The result is a single honest partition over all the regions, which is
    what a pooled VOI needs, and it is the same thing ``neu-vol relabel`` does to a volume —
    done here in the table, since the arrays are already reduced.

    **Pooled ids do not name bodies.** They are positions in a concatenation, so the pooled
    table is for scalars only; per-crop rows keep the real ids and are what you look things
    up by.
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

    The two arrays must be the same shape and are taken to be **already registered** —
    reading them as two crops of one physical box is ``neu_vol.read_piece``'s job
    (``read_piece(seg, level=…, crop=gt_piece)``), and doing it any other way is how a
    half-voxel shift turns into a metric.

    ``ignore_a`` / ``ignore_b`` drop those label values from the count entirely; ``mask``
    restricts to where it is True. Everything dropped is tallied in ``n_ignored`` rather
    than silently vanishing. See the module docstring for why the defaults are asymmetric.

    Runs wherever the arrays already live, CPU or GPU, via :mod:`neu_proc.ops.backend` —
    dispatch is on the array, so one body serves both — and the table comes back in CPU
    memory either way.
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

    # A spatial mask has to be applied to the voxels; there is no other place it exists.
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

    # An **ignored label**, by contrast, is applied to the finished table. Dropping the rows
    # that name it gives exactly the table of the voxels that survive it — the ignore is a
    # union of whole rows and columns — and that is O(pairs) rather than a boolean pass plus
    # a fancy-index compaction of both labelings, which measured 142 ms on a 21.7 Mvoxel
    # crop *whose side a contained no 0 at all*. The default `ignore_a=(0,)` is therefore
    # free, which matters because it is on every call.
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
    """``arr`` as a label array, refusing the things that would wrap silently."""
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
        # A negative id would wrap to an enormous uint64 and still count, giving a table
        # nobody can trace back to a body. Cheaper to refuse it than to explain it later.
        if arr.size and int(arr.min()) < 0:
            raise ValueError(
                f"labeling {which} holds negative values, which are not label ids")
    return arr


def _factorize(xp, arr):
    """``(ids, dense)`` — the distinct labels, and each voxel's index into them.

    Two implementations, because the obvious one is slow where it matters.
    ``unique(return_inverse=True)`` is a sort, and **on the CPU that was most of the cost of
    the whole pass**: measured on a 21.7 Mvoxel uint64 crop with 1,316 labels, 3.43 s
    against **116 ms** for ``fastremap.renumber``, which is hash-based and linear. It took
    the whole CPU pass from 7.26 s to 0.41 s — a larger win than the GPU's, and one that
    CI and every GPU-less worker gets.

    So the CPU uses fastremap (already a neu-proc dependency) and the GPU uses cupy's
    ``unique``, where the sort is parallel and the question does not arise. Both return the
    same thing up to the arbitrary numbering of the dense indices, which nothing downstream
    can see — and ``test_backend`` pins that rather than assuming it, because these are
    genuinely different algorithms and not one function with a flag.
    """
    from neu_proc.ops.backend import is_gpu_array

    # `is_gpu_array` is neu-proc's name for "lives in GPU memory".
    if is_gpu_array(arr):
        ids, dense = xp.unique(arr, return_inverse=True)
        return ids, dense.reshape(-1)

    import fastremap

    dense, mapping = fastremap.renumber(np.ascontiguousarray(arr), in_place=False)
    # `renumber` hands back original -> new; the table needs new -> original to name a body.
    # It always includes 0 -> 0 whether or not 0 occurs, which leaves index 0 unused and
    # costs one empty row.
    ids = np.zeros(int(dense.max()) + 1, dtype=np.uint64)
    for original, new in mapping.items():
        if new < ids.size:
            ids[new] = original
    return ids, dense


def _count_pairs(xp, af, bf):
    """``(a_ids, b_ids, counts)`` for the co-occurring pairs, wherever they live.

    Dense-renumber each side, pack the pair into one integer, and count that — the trick
    ``neu_morpho.measure.compartments.joint_counts`` uses, with its 16-labels-on-one-side
    cap removed by renumbering first instead of assuming a small label set.
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
            # float64 is exact to 2**53 voxels, ~9e15 — a volume that size is not
            # addressable, so the round trip cannot lose a count. bincount has no integer
            # weight path, and np.add.at is an order of magnitude slower.
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
