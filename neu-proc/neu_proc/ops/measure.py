"""Readings over a label array: voxel counts, component counts.

**Label 0 is background, always, and is never reported.** These functions used to take a
`background=` naming any label, and it was the wrong place for the question twice over:
`neu-vol convert --background` already normalises a source's background to 0 *at ingest*
— which it must, since an all-background block of 1s is not all-fill and would be stored,
costing the volume its sparsity — and `cc3d.connected_components` hardcodes 0 as
background regardless of what is passed here, so `component_sizes` could never have
honoured another value without a defensive copy of the whole array. It also generated
three of the five bugs this module has had. See `NOTES-neu-proc.md`.

The background voxel count is still exactly derivable, so nothing was lost:
``arr.size - sum(sizes(arr).values())``.
"""

from operator import itemgetter

import cc3d
import fastremap
import numpy as np


def sizes(
        arr,
        *,
        exclude_margin: int = 0,
        sort: bool = False,
        descending=True,
        skip_relabel: bool = False,
) -> dict[int, int]:
    """``{label: voxel count}`` for every label present except 0."""
    if skip_relabel:
        relabeled = arr
        rev_map_fun = lambda i: i
    else:
        relabeled, label_map = fastremap.renumber(arr)
        rev_map = dict((v, k) for k, v in label_map.items())
        rev_map_fun = lambda i: rev_map[i]
    size_vec = cc3d.statistics(relabeled)['voxel_counts']
    # Filtered on the ORIGINAL label, not on the index into `size_vec`. The two differ
    # whenever `renumber` ran, and testing the index is only accidentally right here
    # because `renumber` preserves `0 -> 0`; resolving first holds however it renumbers.
    #
    # `size > 0` drops labels that are not in the array at all: `cc3d.statistics` returns
    # a count per index up to the maximum value present, so a gappy label set reported
    # every absent value in between — `skip_relabel=True` on {0, 7, 1, 9} returned
    # `{1: 16, 2: 0, ..., 8: 0, 9: 16}`, making the flag a semantic change rather than the
    # optimisation it is meant to be, and making the two branches disagree.
    label_size_iter = ((label, int(size)) for label, size in
                       ((int(rev_map_fun(i)), size) for i, size in enumerate(size_vec))
                       if size > 0 and label != 0)
    if sort:
        label_size_iter = sorted(label_size_iter, key=itemgetter(1), reverse=descending)
    label_sizes = dict(label_size_iter)
    if exclude_margin:
        # A VIEW of the interior, not a mask plus a masked copy: the old form allocated
        # two arrays the size of the input to ask about a sub-box.
        #
        # This filters WHICH labels are reported; it does not re-count them inside the
        # box, so a label straddling the boundary keeps its full voxel count.
        interior = arr[(slice(exclude_margin, -exclude_margin),) * arr.ndim]
        inside_labels = {int(v) for v in np.unique(interior)}
        label_sizes = {label: size for label, size in label_sizes.items() if label in inside_labels}
    return label_sizes


def component_sizes(
        arr,
        connectivity=26,
        *,
        sort: bool = False,
        descending=True,
) -> dict[int, np.ndarray]:
    """``{label: sizes of its connected components, largest first}``, excluding label 0.

    **One scatter, not a full-array mask per label.** ``np.unique(arr_cc[arr == label])``
    walks the whole array once for every label, making the cost O(voxels x labels). A
    connected component lies entirely inside one label, so writing ``arr`` into a table
    indexed by component id reads every component's label in a single pass, and the
    grouping after it is O(components).

    What that buys is the removal of the label-count term, not a constant factor. Measured
    on a 160^3 blob field, identical output throughout:

    ==========  ========  ========  =======
    labels      before    after     speedup
    ==========  ========  ========  =======
    50          0.26 s    0.10 s    2.7x
    200         0.78 s    0.11 s    7.0x
    800         2.79 s    0.10 s    27x
    ==========  ========  ========  =======

    Measure this on blobs, not on a uniform random field: random labels put nearly every
    voxel in its own component, and ``cc3d.connected_components`` then dominates so
    completely that the same change reads as 1.2x.
    """
    arr_cc = cc3d.connected_components(arr, connectivity=connectivity)
    # `arr_cc` holds COMPONENT IDS, whose background is cc3d's own marker 0 — which is
    # also `sizes`' background, so the two agree without anything being passed across.
    cc_sizes = sizes(arr_cc, sort=False, skip_relabel=True)

    # Duplicate indices are safe precisely because of that containment: every voxel of a
    # component writes the same label, so which write lands last does not matter.
    comp_to_label = np.zeros(int(arr_cc.max()) + 1, dtype=arr.dtype)
    comp_to_label[arr_cc.ravel()] = arr.ravel()

    by_label: dict[int, list[int]] = {}
    for comp_id, size in cc_sizes.items():
        by_label.setdefault(int(comp_to_label[comp_id]), []).append(size)

    comp_sizes = {}
    for label in np.unique(arr):
        if label == 0:
            continue
        comp_sizes[int(label)] = np.array(sorted(by_label.get(int(label), []), reverse=True))
    if sort:
        comp_sizes = dict(sorted(comp_sizes.items(), key=lambda tup: len(tup[1]), reverse=descending))
    return comp_sizes


def flat_component_sizes(
        arr,
        connectivity=26,
        *,
        sort: bool = False,
        descending=True,
) -> np.ndarray:
    """Every component size in the volume, pooled across labels."""
    sizes_by_label = component_sizes(
        arr,
        connectivity=connectivity,
        sort=False,
    )
    # Not named `sizes`: that shadowed the module-level function of the same name, which
    # this file would then be unable to call below the assignment.
    flat = []
    for szs in sizes_by_label.values():
        flat.extend(szs)
    if sort:
        return np.array(sorted(flat, reverse=descending))
    else:
        return np.array(flat)
