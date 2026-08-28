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
"""

from __future__ import annotations

import cc3d

from .backend import host_only


def dust(arr, min_voxels: int | None = None, *, max_voxels: int | None = None,
         connectivity: int = 26, invert: bool = False, **kwargs):
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

    ``connectivity`` is 26 by default, i.e. corner-touching voxels are one component. That
    is the loosest choice and so the most conservative for dusting: it merges specks into
    neighbours where it can, removing fewer of them. 6 counts only face-sharing neighbours
    and will find — and delete — more.

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
    return cc3d.dust(host_only(arr, "cc3d.dust"), threshold=threshold,
                     connectivity=connectivity, invert=invert, **kwargs)
