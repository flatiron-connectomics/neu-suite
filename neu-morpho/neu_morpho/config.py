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

    # Segmentation scale to read/mesh at. 0 = full detail (biggest/slowest); a
    # coarser scale is smaller and cuts per-block mask memory ~8×/level. The
    # caller passes the meshing scale's voxel size (nm) so coordinates stay in
    # physical world space — no assumed 2**scale pyramid factor (see coords.py).
    mesh_scale: int = 2
    block_shape: tuple[int, int, int] = (256, 256, 256)   # block size at mesh_scale

    # Per-block simplification during stage 1 (fixed-edge preserves block
    # boundaries so stage-2 assembly works on already-reduced meshes).
    decimation_fraction: float = 0.1
    smoothing_iterations: int = 0

    # Multi-resolution output.
    num_lods: int = 3
    lod_decimation_factor: float = 2.0       # face reduction per coarser LOD
    # Draco quantizes vertex positions into 2**bits steps ACROSS EACH OCTREE CELL,
    # and a LOD-l cell is chunk * 2**l wide — so the effective step COARSENS by 2x
    # per LOD. Once it approaches the voxel size, decimated triangles collapse to
    # zero area and Draco rejects the fragment ("All triangles are degenerate"),
    # failing the whole body. Measured on real data at scale 2 (32 nm voxels,
    # 256-block => 8192 nm cells): 10 bits gives a 32 nm step at LOD 2 — exactly
    # the voxel size — and 29% of bodies failed to encode. 16 bits gives 0.5 nm.
    # The neuroglancer spec allows only 10 or 16, so 16 is the answer; see
    # precomputed.check_quantization, which computes this before a run.
    draco_quantization_bits: int = 16        # must be 10 or 16 (neuroglancer spec)
    sharded: bool = False

    min_segment_voxels: int = 0             # skip bodies smaller than this (if known)
    fragment_format: str = "drc"            # per-fragment on-disk format (Draco)


@dataclass
class OutputConfig:
    """Where results go — **data** and **bookkeeping** are separate destinations.

    ``dst`` is the published precomputed volume and may be an ``s3://`` URL;
    ``work_dir`` is the run directory and must be an ordinary filesystem path,
    because it holds a sqlite DB, appended JSONL manifests and the stage-1
    fragment stores, none of which work over an object store.

    **The split costs a safety property, deliberately.** Bookkeeping used to live
    inside the output, so ``rm -rf`` took the record and the data together and
    the next run correctly started over. Now a manifest can outlive the data it
    describes, and a resumed run would skip every task and report success having
    written nothing. The CLI guards this by refusing to resume when a manifest
    claims completed work but ``dst`` has no ``info`` — see
    ``neu_morpho/cli.py``. Anything writing to ``dst`` outside that entry point
    has to make the same check.
    """

    dst: str = ""            # the precomputed VOLUME root: local path or s3:// URL
    work_dir: str = ""       # run dir: fragments, manifests, DB, logs — POSIX only

    # Meshes and skeletons go INSIDE the volume: the precomputed spec's `mesh` /
    # `skeletons` info keys name subdirectories of the volume root, so a single
    # neuroglancer layer carries labels, meshes and skeletons together.
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
