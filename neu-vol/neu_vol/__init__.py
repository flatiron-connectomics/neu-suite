"""neu-vol: chunked I/O, conversion, and multiscale generation for
large 3D EM volumes, orchestrated with dask (local workstation or SLURM cluster).

See the README for what it does, and blockrun's docs/dask-slurm.md for the
orchestration background.
"""

__version__ = "0.1.0"

# `BBox` and grid arithmetic live in `neu_lib`, and `block_map` / `start_dask` in
# `blockrun`; they are not re-exported here, so each name has one import path.
#
# The top level is the user-facing API. Everything else stays importable from its module:
# the backend registry (`neu_vol.backends.base`), storage profiles (`neu_vol.profiles`),
# and source inspection (`neu_vol.source_metadata`).
from . import location
# Byte and JSON I/O that works the same on local paths and object stores.
from .location import (exists, is_local, local_path, read_bytes, read_json, to_kvstore,
                       write_bytes, write_json)
from .backends.base import open_backend
# HDF5 gets a top-level opener because a file is a container: "open this path" is
# ambiguous for it in a way it is not for a volume.
from .backends.hdf5 import open_hdf5, sole_dataset
from .logs import quiet_store_logs
from .ops import (convert, create_volume, extract_roi, ingest_image_stack,
                  mask_values, pack_hdf5, plan_subvolume_write, plan_volume,
                  rebuild_pyramid, write_subvolume, write_subvolumes)
from .ops.occupancy import labeled_regions
from .ops.subresources import link_subresources
from .piece import read_piece, write_piece
from . import readback
from .readback import (UnsupportedSubresource, mesh_body_ids, mesh_transform,
                       read_body_mesh, read_body_skeleton, skeleton_transform)
from .retry import is_transient, with_retry
from .scales import describe_scales, read_scales, scale_spec
from .source_metadata import describe

__all__ = [
    "__version__",
    # convert, copy, downsample, mask
    "convert",
    "extract_roi",
    "ingest_image_stack",
    "rebuild_pyramid",
    "mask_values",
    # create a volume and write into it
    "create_volume",
    "plan_volume",
    "write_subvolume",
    "write_subvolumes",
    "plan_subvolume_write",
    # pieces and HDF5
    "read_piece",
    "write_piece",
    "pack_hdf5",
    "open_hdf5",
    "sole_dataset",
    # inspect and read
    "describe",
    "describe_scales",
    "read_scales",
    "scale_spec",
    "open_backend",
    "labeled_regions",
    # a body's published mesh and skeleton
    "readback",
    "read_body_mesh",
    "read_body_skeleton",
    "mesh_body_ids",
    "mesh_transform",
    "skeleton_transform",
    "UnsupportedSubresource",
    # stores, retry and logs
    "location",
    "exists",
    "is_local",
    "local_path",
    "read_bytes",
    "read_json",
    "to_kvstore",
    "write_bytes",
    "write_json",
    "with_retry",
    "is_transient",
    "link_subresources",
    "quiet_store_logs",
]


def __dir__() -> list[str]:
    return sorted(__all__)
