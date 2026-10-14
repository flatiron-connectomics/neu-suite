"""End to end: skel-chunk -> skel-fuse -> precomputed skeletons. A body spanning several
blocks must weld into one component covering its full extent."""

import os
from dataclasses import replace

import numpy as np
import pytest

from neu_morpho import fragments as F
from neu_morpho.config import OutputConfig, SkeletonConfig
from neu_morpho.ops.skeletonize_segments import skeletonize_segments
from neu_morpho.skeleton import fuse_body, skeletonize_block


def _write_seg_zarr(path, vol):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _load(out_dir, body):
    from osteoid import Skeleton
    with open(os.path.join(out_dir, str(body)), "rb") as f:
        return Skeleton.from_precomputed(f.read(), segid=body)


VS = (8.0, 8.0, 8.0)


def _volume():
    # 96 x 32 x 32 at 8 nm, blocks of 32 along z -> 3 blocks
    vol = np.zeros((96, 32, 32), np.uint64)
    vol[:, 14:18, 14:18] = 7          # rod spanning all 3 blocks
    vol[40:48, 4:9, 4:9] = 9          # blob wholly inside block 1
    return vol


def _cfg():
    return SkeletonConfig(anisotropy=VS, skeleton_scale=0, block_shape=(32, 32, 32),
                          const=30.0, dust_threshold=0,
                          postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)


def test_skeletonize_end_to_end(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")

    summary = skeletonize_segments(src, out, _cfg(), db_path=db_path, client=None)
    out_dir = summary["out_dir"]

    assert os.path.exists(os.path.join(out_dir, "info"))
    assert summary["num_blocks"] == 3
    assert summary["status_counts"].get("written") == 2

    # the spanning body produced one fragment per block...
    assert len(os.listdir(F.body_dir(summary["chunked_dir"], 7))) == 3
    assert len(os.listdir(F.body_dir(summary["chunked_dir"], 9))) == 1

    # ...and fused into a single connected skeleton over the full z extent.
    rod = _load(out_dir, 7)
    assert len(rod.components()) == 1
    z = rod.vertices[:, 2]                      # stored xyz -> z is column 2
    assert z.min() <= 8 and z.max() >= 744      # 95 voxels * 8 nm = 760
    assert rod.cable_length() > 700

    blob = _load(out_dir, 9)
    bz = blob.vertices[:, 2]
    assert 300 <= bz.min() and bz.max() <= 400  # block 1: z voxels 40..48 -> 320..376 nm


def test_ticks_are_removed_after_the_seam_join_and_metrics_land_in_db(tmp_path):
    """With tick removal on, the rod is still one component over all three blocks.

    `fix_borders` reaches each seam through a short spur, which is exactly what a tick
    filter removes. So the seam join must run before tick removal; the other way round,
    the body fuses into block-length stubs while every status reads "written". The
    spurs are then cleaned up, so the DB sees an unbranched rod.
    """
    from neu_morpho.metrics_db import MetricsDB

    cfg = replace(_cfg(), postprocess_tick_nm=500.0)
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")
    summary = skeletonize_segments(src, out, cfg, db_path=db_path, client=None)

    rod = _load(summary["out_dir"], 7)
    assert len(rod.components()) == 1, "tick removal broke the seam"
    z = rod.vertices[:, 2]
    assert z.min() <= 8 and z.max() >= 744      # still spans all three blocks

    db = MetricsDB(db_path)
    row = db.con.execute(
        "SELECT cable_length_nm, n_branches, n_tips FROM bodies WHERE body_id=7").fetchone()
    db.close()
    assert row is not None
    cable, branches, tips = row
    assert cable > 700                          # the FUSED length, not one block's
    assert branches == 0 and tips == 2          # an unbranched rod


def test_allowlist_restricts_bodies(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    summary = skeletonize_segments(src, out, _cfg(), allowlist=[7], client=None)

    assert os.path.exists(os.path.join(summary["out_dir"], "7"))
    assert not os.path.exists(os.path.join(summary["out_dir"], "9"))
    assert not os.path.exists(F.body_dir(summary["chunked_dir"], 9))


def test_resume_skips_done_and_stage2_reuses_fragments(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    first = skeletonize_segments(src, out, _cfg(), stages=("skel-chunk",), client=None)
    assert first["chunk_counts"].get("written") == 3
    assert not os.path.exists(os.path.join(first["out_dir"], "7"))

    # stage 2 alone reuses the fragments already on disk
    second = skeletonize_segments(src, out, _cfg(), stages=("skel-fuse",), client=None)
    assert second["num_bodies_fused"] == 2
    assert os.path.exists(os.path.join(second["out_dir"], "7"))

    # a third full run has nothing left to do
    third = skeletonize_segments(src, out, _cfg(), client=None)
    assert third["num_bodies_fused"] == 0


def test_fuse_body_welds_seams_not_just_concatenates():
    """Fragments become one tree, with the seam edges added."""
    cfg = _cfg()
    vol = _volume()
    frags = [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in (0, 32, 64)]
    assert len(frags) == 3

    fused = fuse_body(frags, cfg, body_id=7)
    assert fused.id == 7
    assert len(fused.components()) == 1
    # welding adds the seam edges, so the fused cable exceeds the sum of the parts
    assert fused.cable_length() > sum(f.cable_length() for f in frags)


def _split_body():
    """One label, two blobs ~400 nm apart — a segmentation split, not a seam."""
    vol = np.zeros((32, 32, 96), np.uint64)
    vol[8:24, 8:24, 4:20] = 7
    vol[8:24, 8:24, 70:86] = 7
    return vol


def test_default_join_does_not_bridge_a_segmentation_split():
    """The default join radius is seam-scale: distant pieces of a split stay apart and
    present. An unbounded join would invent a straight edge's worth of cable."""
    from neu_morpho.skeleton import join_radius_nm

    cfg = _cfg()
    assert join_radius_nm(cfg) == 16.0                  # 2 x 8 nm voxel
    frags = list(skeletonize_block(_split_body(), (0, 0, 0), cfg).values())

    fused = fuse_body(frags, cfg, body_id=7)
    assert len(fused.components()) == 2                 # kept, just disconnected
    bounded_cable = fused.cable_length()

    unbounded = fuse_body(frags, replace(cfg, join_radius_nm=float("inf")), body_id=7)
    assert len(unbounded.components()) == 1

    # the invented cable is exactly the nearest-pair gap between the two pieces
    comps = fused.components()
    ca, cb = (np.asarray(c.vertices, float) for c in comps)
    gap = float(np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1).min())
    assert unbounded.cable_length() == pytest.approx(bounded_cable + gap, rel=0.05)


def test_join_radius_zero_defers_to_postprocess():
    """radius=0 skips the explicit join; postprocess's own join still welds seams."""
    cfg = replace(_cfg(), join_radius_nm=0)
    vol = _volume()
    frags = [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in (0, 32, 64)]

    fused = fuse_body(frags, cfg, body_id=7)
    assert len(fused.components()) == 1                 # seams still welded
    assert fused.cable_length() > 700
    # ...but it still refuses to bridge a genuine split
    split = list(skeletonize_block(_split_body(), (0, 0, 0), cfg).values())
    assert len(fuse_body(split, cfg, body_id=7).components()) == 2


def _arbor_with_twigs():
    """A trunk with short lateral twigs and a detached speck of the same label."""
    vol = np.zeros((96, 48, 48), np.uint64)
    vol[4:92, 22:26, 22:26] = 7                      # trunk
    for z in (20, 40, 60, 80):                       # short twigs -> tick fodder
        vol[z:z + 2, 26:34, 23:25] = 7
    vol[50:52, 40:42, 40:42] = 7                     # detached speck -> dust fodder
    return vol


def _arbor_frags(cfg):
    vol = _arbor_with_twigs()
    return [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in range(0, 96, 32)]


def test_default_tick_threshold_runs():
    """The shipped (non-zero) tick default runs through postprocess, whose compiled
    remove_ticks rejects int64 edges and float64 vertices."""
    # 32 nm voxels so the ~88-voxel trunk (2816 nm) clears any sane dust default
    cfg = SkeletonConfig(anisotropy=(32.0, 32.0, 32.0), block_shape=(32, 32, 32),
                         const=30.0, dust_threshold=0)
    assert cfg.postprocess_tick_nm > 0          # zero would skip remove_ticks entirely

    fused = fuse_body(_arbor_frags(cfg), cfg, body_id=7)      # must not raise
    assert fused is not None and len(fused.vertices) > 0
    assert fused.edges.dtype == np.uint32 and fused.vertices.dtype == np.float32


def test_the_fusion_chain_keeps_uint32_edges_and_float32_vertices():
    """Fragments and fused skeletons keep the dtypes the compiled postprocess and the
    precomputed encoder require (needs osteoid >= 0.7.3, which stopped promoting edges
    to int64 in `consolidate`)."""
    from neu_morpho.skeleton import join_radius_nm, normalize_dtypes
    import kimimaro

    cfg = replace(_cfg(), postprocess_dust_nm=0.0, postprocess_tick_nm=1000.0)
    frags = [skeletonize_block(_volume()[z0:z0 + 32], (z0, 0, 0), cfg)[7]
             for z0 in range(0, 96, 32)]
    assert all(frag.edges.dtype == np.uint32 for frag in frags)

    clean = normalize_dtypes(
        kimimaro.join_close_components(frags, radius=join_radius_nm(cfg)).consolidate())
    assert clean.edges.dtype == np.uint32 and clean.vertices.dtype == np.float32

    fused = fuse_body(frags, cfg, body_id=7)
    assert fused.edges.dtype == np.uint32 and fused.vertices.dtype == np.float32


@pytest.mark.parametrize("tick_nm", [0.0, 1000.0])
def test_fuse_body_matches_kimimaro_postprocess(tick_nm):
    """`fuse_body` inlines postprocess's steps (for the seam join and per-step stats),
    so it must give the same skeleton as stock `kimimaro.postprocess`."""
    import kimimaro

    from neu_morpho.skeleton import join_radius_nm, normalize_dtypes

    cfg = replace(_cfg(), postprocess_dust_nm=200.0, postprocess_tick_nm=tick_nm)
    frags = _arbor_frags(cfg)

    ours = fuse_body(frags, cfg, body_id=7)

    joined = kimimaro.join_close_components(frags, radius=join_radius_nm(cfg))
    theirs = kimimaro.postprocess(normalize_dtypes(joined.consolidate()),
                                  dust_threshold=cfg.postprocess_dust_nm,
                                  tick_threshold=cfg.postprocess_tick_nm)

    assert len(ours.vertices) == len(theirs.vertices)
    assert ours.cable_length() == pytest.approx(theirs.cable_length(), rel=1e-6)


def test_tick_removal_prunes_twigs():
    """Tick removal gives fewer tips and less cable."""
    cfg = replace(_cfg(), postprocess_dust_nm=0.0)
    frags = _arbor_frags(cfg)

    keep = fuse_body(frags, replace(cfg, postprocess_tick_nm=0.0), body_id=7)
    prune = fuse_body(frags, replace(cfg, postprocess_tick_nm=1000.0), body_id=7)

    assert len(prune.terminals()) < len(keep.terminals())
    assert prune.cable_length() < keep.cable_length()


def test_fusion_stats_attribute_what_was_dropped():
    from neu_morpho.skeleton import fusion_stats_summary

    cfg = replace(_cfg(), postprocess_dust_nm=200.0, postprocess_tick_nm=1000.0)
    frags = _arbor_frags(cfg)

    stats = {}
    fuse_body(frags, cfg, body_id=7, stats=stats)

    assert stats["n_fragments"] == 3
    assert stats["seam_join_comps_merged"] >= 2          # 3 block fragments -> 1 trunk
    assert stats["dust_comps_dropped"] >= 1              # the detached speck
    assert stats["dust_cable_dropped_nm"] > 0
    assert stats["tick_branches_removed"] >= 1           # the lateral twigs
    assert stats["tick_cable_removed_nm"] > 0
    assert not stats["deleted"]

    summary = fusion_stats_summary([stats])
    assert summary["n_bodies"] == 1 and summary["bodies_deleted"] == 0
    assert 0 < summary["dropped_cable_fraction"] < 1
    # nothing dropped when both thresholds are off
    off = {}
    fuse_body(frags, replace(cfg, postprocess_dust_nm=0.0, postprocess_tick_nm=0.0),
              body_id=7, stats=off)
    assert off["dust_comps_dropped"] == 0 and off["tick_cable_removed_nm"] == 0


def test_op_reports_fusion_stats(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _arbor_with_twigs())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    stats_path = str(tmp_path / "fusion.jsonl")

    cfg = replace(_cfg(), block_shape=(32, 32, 32), postprocess_dust_nm=200.0,
                  postprocess_tick_nm=1000.0)
    summary = skeletonize_segments(src, out, cfg, fusion_stats_path=stats_path, client=None)

    fs = summary["fusion_stats"]
    assert fs["n_bodies"] == 1
    assert fs["dust_comps_dropped"] >= 1 and fs["tick_branches_removed"] >= 1
    assert fs["inferred_cable_fraction"] > 0        # the seam edges are inferred

    import json
    rows = [json.loads(ln) for ln in open(stats_path)]
    assert len(rows) == 1 and rows[0]["body_id"] == 7


def test_dust_threshold_deletion_is_reported_not_silent(tmp_path):
    """A body shorter than postprocess_dust_nm is deleted and reported as "dust"."""
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[8:24, 14:18, 14:18] = 7          # ~128 nm of cable, well under the threshold
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), vol)
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    cfg = replace(_cfg(), postprocess_dust_nm=1500.0)
    summary = skeletonize_segments(src, out, cfg, client=None)

    assert summary["status_counts"].get("dust") == 1
    assert summary["status_counts"].get("written", 0) == 0
    assert not os.path.exists(os.path.join(summary["out_dir"], "7"))

    # with the threshold off, the same body survives
    out2 = OutputConfig(dst=str(tmp_path / "out2" / "segmentation"), work_dir=str(tmp_path / "out2"))
    s2 = skeletonize_segments(src, out2, _cfg(), client=None)
    assert s2["status_counts"].get("written") == 1


# --------------------------------------------------------------------------- #
# The neutu tracer through the same pipeline
# --------------------------------------------------------------------------- #
def test_neutu_tracer_end_to_end_welds_a_spanning_body_with_nm_radii(tmp_path):
    """Through the real two-stage op, neutu fragments meet at the seams and weld into
    one component, and radii come out in nm. `min_length` stays at its default, since
    border targets are exempt from it."""
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"),
                       work_dir=str(tmp_path / "out"))
    cfg = replace(_cfg(), tracer="neutu")

    summary = skeletonize_segments(src, out, cfg, db_path=None, client=None)
    out_dir = summary["out_dir"]
    assert summary["num_blocks"] == 3
    assert summary["status_counts"].get("written") == 2
    assert len(os.listdir(F.body_dir(summary["chunked_dir"], 7))) == 3

    rod = _load(out_dir, 7)
    assert len(rod.components()) == 1, (
        f"neutu fragments did not weld: {len(rod.components())} components")
    z = rod.vertices[:, 2]
    assert z.min() <= 16 and z.max() >= 736, f"z extent {z.min()}..{z.max()}"
    assert rod.cable_length() > 700

    r = np.asarray(rod.radius, float)
    # the rod is 4 voxels across at 8 nm/voxel, so radii are ~8-16 nm, never ~1-2
    assert r.max() >= 6.0, f"radii look like voxels, max {r.max():.2f}"
    assert r.max() <= 8.0 * 6
