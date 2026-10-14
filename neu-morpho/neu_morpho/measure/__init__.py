"""Per-body morphology measurement, over PUBLISHED output.

An analysis layer, not a pipeline stage: it reads volumes, meshes and skeletons that are
already published (often not made by this pipeline) and turns them into tables.

Cohorts may control for a compartment differently (a semantic mask on one, a cohort
selection on another), so every row carries an explicit ``variant`` and tables are keyed
``(dataset, body_id, variant)``.

Reading is `neu_vol` (`neu_vol.readback` for meshes and skeletons); the region predicates
that turn a compartment into an ``inside()`` are `neu_lib`.
"""

from .compartments import (joint_counts, semantic_label_names,
                           somatic_labels, sweep_compartments,
                           verify_compartments)
from .driver import resolve_keep, sweep_volumes
from .skeletons import skeleton_columns, sweep_skeletons
from .paths import (attach_points, bridge_gaps, compartment_graph, dvid_roi_region,
                    near_points_region,
                    path_distances, synapse_distance_table, synapse_distances,
                    volume_roi_region)
from .metrics import (cable_length_nm, diameter_stats, frustum_area_nm2,
                      frustum_volume_nm3, max_radius_nm, measure_skeleton, topology,
                      weighted_quantile)
from .tables import (compare, load_bodies, load_segment_properties,
                     log_histogram, summarize, write_table)
from .sweep import (DEFAULT_BLOCK, DEFAULT_VOXEL_NM, SweepTotals, bin_to_nodes,
                    blob_signal, blocks_from_mask, cable_shares, count_labels,
                    log_bin_edges, mean_cross_section, node_radii, roi_block_mask,
                    weighted_histogram)

__all__ = [
    # per-skeleton metrics
    "cable_length_nm",
    "diameter_stats",
    "frustum_area_nm2",
    "frustum_volume_nm3",
    "max_radius_nm",
    "measure_skeleton",
    "topology",
    "weighted_quantile",
    # path length from a reference region, for synapses on a skeleton
    "attach_points",
    "bridge_gaps",
    "compartment_graph",
    "dvid_roi_region",
    "near_points_region",
    "path_distances",
    "synapse_distance_table",
    "synapse_distances",
    "volume_roi_region",
    # the block-mapped driver (the only part that opens a store)
    "resolve_keep",
    "sweep_volumes",
    "skeleton_columns",
    "sweep_skeletons",
    "sweep_compartments",
    "verify_compartments",
    "joint_counts",
    # tidy tables for cohort selection (pandas, from the `measure` extra)
    "load_bodies",
    "load_segment_properties",
    "summarize",
    "compare",
    "log_histogram",
    "write_table",
    "semantic_label_names",
    "somatic_labels",
    # the voxel-counting sweep
    "DEFAULT_BLOCK",
    "DEFAULT_VOXEL_NM",
    "SweepTotals",
    "bin_to_nodes",
    "blob_signal",
    "blocks_from_mask",
    "cable_shares",
    "count_labels",
    "log_bin_edges",
    "mean_cross_section",
    "node_radii",
    "roi_block_mask",
    "weighted_histogram",
]
