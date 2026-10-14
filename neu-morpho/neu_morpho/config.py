"""Configuration for the block-first meshing pipeline and its output locations.

Meshing is **block-first, two-stage**: stage 1 meshes each
non-empty block (all/allowlisted labels at once) into per-(body, block) fragments;
stage 2 concatenates + stitches each body's fragments into a multi-resolution
mesh. No whole-object binary mask is ever built.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeshConfig:
    """Tunables for block-first meshing."""

    # Segmentation scale to mesh at (0 = full detail). The caller passes its voxel size
    # (nm), so coordinates stay physical whatever the pyramid's factors.
    mesh_scale: int = 2
    block_shape: tuple[int, int, int] = (256, 256, 256)   # block size at mesh_scale

    # Per-block simplification in stage 1 (block boundaries are kept fixed).
    decimation_fraction: float = 0.1
    smoothing_iterations: int = 0

    # Multi-resolution output.
    num_lods: int = 3
    lod_decimation_factor: float = 2.0       # face reduction per coarser LOD
    # Draco spreads 2**bits steps across each octree cell, which doubles in size per
    # LOD; near the voxel size, triangles collapse and the body fails to encode. 16 bits
    # (the spec allows 10 or 16); precomputed.check_quantization checks before a run.
    draco_quantization_bits: int = 16        # must be 10 or 16 (neuroglancer spec)
    sharded: bool = False

    min_segment_voxels: int = 0             # skip bodies smaller than this (if known)
    fragment_format: str = "drc"            # per-fragment on-disk format (Draco)


@dataclass
class OutputConfig:
    """Where results go — **data** and **bookkeeping** are separate destinations.

    ``dst`` is the published precomputed volume and may be an ``s3://`` URL;
    ``work_dir`` must be a filesystem path: it holds a sqlite DB, appended JSONL
    manifests and the stage-1 fragment stores.

    A manifest can therefore outlive the data it describes, and a resumed run would
    skip every task having written nothing. Every stage refuses that
    (``ops.manifest.check_manifest_matches_output``); anything else writing to ``dst``
    under a manifest must make the same check.
    """

    dst: str = ""            # the precomputed VOLUME root: local path or s3:// URL
    work_dir: str = ""       # run dir: fragments, manifests, DB, logs — POSIX only

    # Meshes and skeletons go inside the volume (the `mesh` / `skeletons` info keys name
    # subdirectories), so one neuroglancer layer carries all three.
    mesh_dir: str = "mesh"                   # subdirectory of the VOLUME
    skeleton_dir: str = "skeleton"           # subdirectory of the VOLUME

    chunked_dir: str = ""                    # stage-1 fragments; default work_dir+"/chunked"
    skel_chunked_dir: str = ""               # stage-1 skeleton fragments; default work_dir+"/skel_chunked"
    progress_path: str | None = None         # default work_dir/progress.{mesh,skel}.jsonl
    extra: dict = field(default_factory=dict)

    def volume_dir(self) -> str:
        """The precomputed volume root — what you point neuroglancer at."""
        return self.dst.rstrip("/")

    def mesh_out(self) -> str:
        """Where mesh fragments go: a subdirectory of the volume."""
        return self.volume_dir() + "/" + self.mesh_dir.strip("/")

    def skeleton_out(self) -> str:
        """Where skeleton blobs go: a subdirectory of the volume."""
        return self.volume_dir() + "/" + self.skeleton_dir.strip("/")

    def work(self, *parts: str) -> str:
        """Join ``parts`` onto the POSIX work directory."""
        segs = [self.work_dir.rstrip("/")] + [p.strip("/") for p in parts if p]
        return "/".join(s for s in segs if s)

    def check_work_dir_is_local(self) -> None:
        """Fail early if ``work_dir`` is remote — sqlite and append cannot use it."""
        from neu_vol import is_local

        if not self.work_dir:
            raise ValueError("work_dir is required: it holds the fragments, "
                             "manifests and metrics DB")
        if not is_local(self.work_dir):
            raise ValueError(
                f"work_dir must be a filesystem path, got {self.work_dir!r}. It "
                "holds a sqlite DB, appended JSONL manifests and the stage-1 "
                "fragment stores, none of which work over an object store. Only "
                "--dst may be remote.")
