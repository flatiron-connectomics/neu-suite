"""Remove small disconnected specks: :func:`dust`.

Dusting is per connected component, not per label: a 1-voxel speck carrying the same id
as a large body elsewhere is removed while the body stays. That suits segmentations whose
bodies are genuinely fragmented (one measured 344 components at scale 1, most of them
single-voxel specks). Dropping small *labels* would be a different operation, and should
get a different name.

There is no GPU path: ``cupyx.scipy.ndimage.label`` works on a binary mask, so touching
bodies would merge into one component and a small body next to a large one would survive.
For a binary input the mask is the labelling, so a GPU version would be correct there.

``exclude_boundary`` spares a component that touches a face of the array, because within
a crop its size is only a lower bound. The test is per component: keying it on the label
instead would also spare every interior speck sharing an id with something on a face
(34% of the interior dust on one 364^3 ground-truth crop).
"""

from __future__ import annotations

import cc3d
import numpy as np

from .backend import cpu_only


def dust(arr, min_voxels: int | None = None, *, max_voxels: int | None = None,
         connectivity: int = 6, invert: bool = False, exclude_boundary: bool = False, **kwargs):
    """Drop connected components outside the given voxel-count range.

        dust(labels, 10)                    # remove anything under 10 voxels
        dust(labels, 10, invert=True)       # keep ONLY those — what you are throwing away
        dust(labels, 5, max_voxels=50)      # keep the band, dropping both tails

    ``min_voxels`` keeps components of at least that many voxels. With ``max_voxels``,
    larger components are dropped too, keeping only the band between. ``invert`` keeps only
    what would be removed, to inspect it first.

    Thresholds are voxel counts. To start from a physical volume, convert with
    :func:`~neu_proc.ops.kernels.volume_to_voxels`, which divides by the product of the
    voxel sizes.

    ``connectivity`` is 6 (face neighbours) by default; 26 joins corner-touching voxels, so
    fewer specks count as separate and fewer are removed.

    ``exclude_boundary`` spares every component that touches a face of the array, however
    small; interior components are judged normally.

    Runs on the CPU; a GPU array is copied off and back with a warning.
    """
    if min_voxels is None and max_voxels is None:
        raise ValueError("give min_voxels, max_voxels, or both")
    if max_voxels is None:
        threshold: int | tuple[int, int] = int(min_voxels)          # type: ignore[arg-type]
    else:
        # cc3d reads a pair as the range to keep; with no lower bound, use 1.
        threshold = (int(min_voxels) if min_voxels is not None else 1, int(max_voxels))
        if threshold[0] > threshold[1]:
            raise ValueError(
                f"min_voxels {threshold[0]} is above max_voxels {threshold[1]}, which keeps "
                f"nothing")
    CPU = cpu_only(arr, "cc3d.dust")
    if exclude_boundary:
        # This path labels components itself, so cc3d.dust keywords other than binary_image
        # would be silently ignored; refuse them.
        unsupported = sorted(set(kwargs) - {"binary_image"})
        if unsupported:
            raise TypeError(
                f"exclude_boundary=True does not pass {'=, '.join(unsupported)}= through to "
                f"cc3d.dust — it decides per component itself. Drop one or the other")
        return _dust_interior(CPU, threshold, connectivity=connectivity, invert=invert,
                              binary_image=kwargs.get("binary_image", False))
    return cc3d.dust(CPU, threshold=threshold, connectivity=connectivity, invert=invert,
                     **kwargs)


def _dust_interior(arr, threshold, *, connectivity: int, invert: bool,
                   binary_image: bool = False):
    """:func:`dust`, sparing components that reach a face of the array. One labelling pass,
    one ``cc3d.statistics``; a component reaches a face exactly when its bounding box does."""
    cc = cc3d.connected_components(arr, connectivity=connectivity,
                                   binary_image=binary_image)
    stats = cc3d.statistics(cc)
    counts = np.asarray(stats["voxel_counts"])
    small = _within(counts, threshold)
    touches = np.array(
        [any(s.start == 0 or s.stop == extent for s, extent in zip(box, arr.shape))
         for box in stats["bounding_boxes"]], dtype=bool)

    # Component 0 is background.
    drop = small & ~touches
    drop[0] = False
    if invert:
        # What would be removed, as invert=True means without exclude_boundary.
        dust_only = np.zeros_like(arr)
        keep = drop[cc]
        dust_only[keep] = arr[keep]
        return dust_only
    dusted = arr.copy()
    dusted[drop[cc]] = 0
    return dusted


def _within(counts, threshold) -> "np.ndarray":
    """Which component sizes are dust: outside the range cc3d would keep (a scalar means
    "at least this many", a pair the range)."""
    if isinstance(threshold, tuple):
        return (counts < threshold[0]) | (counts > threshold[1])
    return counts < threshold
