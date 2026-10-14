"""Readings over a label array: voxel counts, component counts.

Label 0 is background, always, and is never reported (``neu-vol convert --background``
normalises a source's background to 0, and cc3d treats 0 as background regardless). The
background count is ``arr.size - sum(sizes(arr).values())``.
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
    """``{label: voxel count}`` for every label present except 0.

    ``exclude_margin`` reports only labels present in the interior, ``exclude_margin``
    voxels in from every face, with their full counts. ``skip_relabel`` skips the
    ``fastremap.renumber`` step, for arrays whose labels are already small integers.
    """
    if skip_relabel:
        relabeled = arr
        rev_map_fun = lambda i: i
    else:
        relabeled, label_map = fastremap.renumber(arr)
        rev_map = dict((v, k) for k, v in label_map.items())
        rev_map_fun = lambda i: rev_map[i]
    size_vec = cc3d.statistics(relabeled)['voxel_counts']
    # Map indices back to labels before filtering. `size > 0` drops values absent from the
    # array, which cc3d.statistics reports for every index up to the maximum.
    label_size_iter = ((label, int(size)) for label, size in
                       ((int(rev_map_fun(i)), size) for i, size in enumerate(size_vec))
                       if size > 0 and label != 0)
    if sort:
        label_size_iter = sorted(label_size_iter, key=itemgetter(1), reverse=descending)
    label_sizes = dict(label_size_iter)
    if exclude_margin:
        # Filters which labels are reported; a label straddling the margin keeps its full count.
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

    Each component lies inside one label, so a single scatter of ``arr`` into a table indexed
    by component id finds every component's label; the cost does not grow with the number of
    labels.
    """
    arr_cc = cc3d.connected_components(arr, connectivity=connectivity)
    # Component ids; cc3d marks background as 0, as `sizes` expects.
    cc_sizes = sizes(arr_cc, sort=False, skip_relabel=True)

    # Duplicate indices are safe: every voxel of a component writes the same label.
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
    flat = []
    for szs in sizes_by_label.values():
        flat.extend(szs)
    if sort:
        return np.array(sorted(flat, reverse=descending))
    else:
        return np.array(flat)
