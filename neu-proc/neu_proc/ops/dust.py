"""Remove small disconnected specks: :func:`dust`.

**Component-wise, not per-label, and the difference is large on real data.** ``cc3d.dust``
finds connected components of the *labelled* image and drops the small ones, so a speck
carrying the same id as a big body somewhere else is removed while the body stays. Measured
on a fixture: label 7 with a 64-voxel blob plus a 1-voxel speck elsewhere, ``min_voxels=10``
leaves 64.

That is the behaviour you want here, because bodies in this dataset are genuinely
fragmented rather than artefactually so — one body measured 68 components at scale 2 and
**344 at scale 1**, most of them single-voxel specks carrying the label, with only 7 of 21
holding 10 or more voxels. Per-label totals would be a *different* operation: it would keep
every speck of a large body and delete small bodies whole. If that is ever wanted it should
be a separately named function, because the two produce identically-shaped output and
wildly different content.

**There is no GPU path, and that is a correctness decision rather than an omission.** The
obvious cupy version — ``cupyx.scipy.ndimage.label`` on the mask, ``bincount``, then drop the
small components — measured 4-5x faster than cc3d even including both transfers, and 10-98x
on an array already resident. It is also **wrong for a multi-label array**: ``label`` works on
a *binary* mask, so two different bodies that touch become one component. Measured on a
fixture, a 4-voxel label adjacent to a 125-voxel one was kept at ``min_voxels=10`` where
cc3d removes it. ``cupyx`` has no multi-label connected-components, so there is nothing to
dispatch to.

A binary input has no such problem — one label, so the mask *is* the labelling — and cc3d
takes ``binary_image=True`` for it. That is where a device path could be added correctly if
the 4-5x is ever worth the second implementation.

**``exclude_boundary`` spares a COMPONENT that reaches a face, and per-component is the
whole point.** A fragment poking into a crop from a body that mostly lives outside it is
indistinguishable, within the crop, from dust: its observed size is a lower bound on its
real one, so no threshold can judge it. Keying that on the *label* instead spares every
speck sharing an id with something on a face, wherever it sits — measured on one 364^3
ground-truth crop at ``min_voxels=1000``: 72,706 voxels of interior dust across 254 labels
survived that way, **34% of the interior dust that was meant to go**. It also cost a
full-array pass per face label, 8.3 s against 0.42 s for the component test below.
"""

from __future__ import annotations

import cc3d
import numpy as np

from .backend import host_only


def dust(arr, min_voxels: int | None = None, *, max_voxels: int | None = None,
         connectivity: int = 6, invert: bool = False, exclude_boundary: bool = False, **kwargs):
    """Drop connected components outside the given voxel-count range.

        dust(labels, 10)                    # remove anything under 10 voxels
        dust(labels, 10, invert=True)       # keep ONLY those — what you are throwing away
        dust(labels, 5, max_voxels=50)      # keep the band, dropping both tails

    ``min_voxels`` keeps components of **at least** that many voxels. With ``max_voxels`` it
    becomes a band and components larger than it are dropped as well, which is no longer
    dusting — worth saying, because "remove the big things too" is rarely what anyone means.
    ``invert`` swaps which side survives, and is the useful way to *look* at the dust before
    committing to losing it.

    Voxel counts, not physical volume, deliberately: the conversion needs a
    :class:`~neu_lib.Frame` and belongs one layer up, where a future ``Dust(min_volume_nm3=)``
    binds to one. Note it is a **volume**, so the conversion divides by the *product* of the
    three voxel sizes rather than scaling per axis —
    :func:`~neu_proc.ops.kernels.volume_to_voxels`.

    ``connectivity`` is 6 by default, counting only face-sharing neighbours. 26 makes
    corner-touching voxels one component, which is the loosest choice and so the most
    conservative for dusting — it merges specks into neighbours where it can and removes
    fewer of them.

    ``exclude_boundary`` spares any component **reaching a face of the array**, however
    small. Within a crop, such a fragment may be part of a body that mostly lives outside
    it, so its observed voxel count is only a lower bound and the threshold cannot judge it;
    a component wholly inside is judged normally, including one that merely comes close to a
    face. The test is per component and not per label — see the module docstring for what the
    per-label version cost on real data.

    Runs on the host; see the module docstring on why there is no device path. A device array
    is copied off and back with a warning rather than failing.
    """
    if min_voxels is None and max_voxels is None:
        raise ValueError("give min_voxels, max_voxels, or both")
    if max_voxels is None:
        threshold: int | tuple[int, int] = int(min_voxels)          # type: ignore[arg-type]
    else:
        # cc3d reads a pair as the range to KEEP, so an absent lower bound is 1 rather than 0
        # — a component of 0 voxels does not exist, and 0 as a bound would read as "no bound".
        threshold = (int(min_voxels) if min_voxels is not None else 1, int(max_voxels))
        if threshold[0] > threshold[1]:
            raise ValueError(
                f"min_voxels {threshold[0]} is above max_voxels {threshold[1]}, which keeps "
                f"nothing")
    host = host_only(arr, "cc3d.dust")
    if exclude_boundary:
        # This path runs its own labelling instead of `cc3d.dust`, so a passthrough keyword
        # would be silently dropped rather than applied. Only `binary_image` means the same
        # thing to both.
        unsupported = sorted(set(kwargs) - {"binary_image"})
        if unsupported:
            raise TypeError(
                f"exclude_boundary=True does not pass {'=, '.join(unsupported)}= through to "
                f"cc3d.dust — it decides per component itself. Drop one or the other")
        return _dust_interior(host, threshold, connectivity=connectivity, invert=invert,
                              binary_image=kwargs.get("binary_image", False))
    return cc3d.dust(host, threshold=threshold, connectivity=connectivity, invert=invert,
                     **kwargs)


def _dust_interior(arr, threshold, *, connectivity: int, invert: bool,
                   binary_image: bool = False):
    """:func:`dust`, sparing every component that reaches a face of the array.

    One connected-components pass and one ``cc3d.statistics``, then a lookup table over
    component ids — so the cost is independent of how many components or labels there are.
    ``cc3d.dust`` cannot express this: it decides per component but reports only the
    surviving *labels*, and recovering "which component was that" from the result means
    scanning per label, which is what made the first version 20x slower as well as wrong.

    **A component reaches a face iff its bounding box does**, which ``statistics`` already
    computed — equivalent to testing the six face slices for its id, without touching the
    voxels.
    """
    cc = cc3d.connected_components(arr, connectivity=connectivity,
                                   binary_image=binary_image)
    stats = cc3d.statistics(cc)
    counts = np.asarray(stats["voxel_counts"])
    small = _within(counts, threshold)
    touches = np.array(
        [any(s.start == 0 or s.stop == extent for s, extent in zip(box, arr.shape))
         for box in stats["bounding_boxes"]], dtype=bool)

    # Component 0 is the background, which is neither dust nor a body. Excluding it here is
    # what keeps the `drop[cc]` lookup below a single expression.
    drop = small & ~touches
    drop[0] = False
    if invert:
        # What WOULD be removed, for looking at the dust before committing to losing it —
        # the same meaning `invert=True` has without exclude_boundary, so a face-touching
        # speck is absent from both answers rather than appearing in each.
        dust_only = np.zeros_like(arr)
        keep = drop[cc]
        dust_only[keep] = arr[keep]
        return dust_only
    dusted = arr.copy()
    dusted[drop[cc]] = 0
    return dusted


def _within(counts, threshold) -> "np.ndarray":
    """Which component sizes fall **outside** the range cc3d would keep, i.e. are dust.

    cc3d reads a scalar threshold as "keep at least this many" and a pair as the range to
    keep; this is the complement of that, expressed once so the two paths cannot disagree
    about the boundary cases.
    """
    if isinstance(threshold, tuple):
        return (counts < threshold[0]) | (counts > threshold[1])
    return counts < threshold
