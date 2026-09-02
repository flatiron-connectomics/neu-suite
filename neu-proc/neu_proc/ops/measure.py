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
        background=0,
        skip_relabel: bool = False,
) -> dict[int, int]:
    """Returns a dict with the voxel size count of each non-background segment."""
    if skip_relabel:
        relabeled = arr
        rev_map_fun = lambda i: i
    else:
        relabeled, label_map = fastremap.renumber(arr)
        rev_map = dict((v, k) for k, v in label_map.items())
        rev_map_fun = lambda i: rev_map[i]
    size_vec = cc3d.statistics(relabeled)['voxel_counts']
    label_size_iter = ((int(rev_map_fun(i)), int(size)) for i, size in enumerate(size_vec)
                       if (background is None or i != background))
    if sort:
        label_size_iter = sorted(label_size_iter, key=itemgetter(1), reverse=descending)
    label_sizes = dict(label_size_iter)
    if exclude_margin:
        inside_margin = np.ones_like(arr, dtype=bool)
        inside_margin[:exclude_margin, :, :] = False
        inside_margin[-exclude_margin:, :, :] = False
        inside_margin[:, :exclude_margin, :] = False
        inside_margin[:, -exclude_margin:, :] = False
        inside_margin[:, :, :exclude_margin] = False
        inside_margin[:, :, -exclude_margin:] = False
        inside_arr = arr.copy()
        inside_arr[~inside_margin] = 0
        inside_labels = set(np.unique(inside_arr))
        label_sizes = {label: size for label, size in label_sizes.items() if label in inside_labels}
    return label_sizes


def component_sizes(
        arr,
        connectivity=26,
        *,
        background=0,
        sort: bool = False,
        descending=True,
) -> dict[int, np.ndarray]:
    """Returns a dict with the component sizes of each non-background segment."""
    arr_cc = cc3d.connected_components(arr, connectivity=connectivity)
    cc_sizes = sizes(arr_cc, background=background, sort=False, skip_relabel=True)
    ids = np.unique(arr)
    comp_sizes = {}
    for label in ids:
        if label == background:
            continue
        label_comps = np.unique(arr_cc[arr == label])
        comp_sizes[label] = np.array(
            sorted((cc_sizes[i] for i in label_comps if background is None or i != background), reverse=True))
    if sort:
        comp_sizes = dict(sorted(comp_sizes.items(), key=lambda tup: len(tup[1]), reverse=descending))
    return comp_sizes


def flat_component_sizes(
        arr,
        connectivity=26,
        *,
        background=0,
        sort: bool = False,
        descending=True,
) -> np.ndarray:
    sizes_by_label = component_sizes(
        arr,
        connectivity=connectivity,
        background=background,
        sort=False,
    )
    sizes = []
    for szs in sizes_by_label.values():
        sizes.extend(szs)
    if sort:
        return np.array(sorted(sizes, reverse=descending))
    else:
        return np.array(sizes)
