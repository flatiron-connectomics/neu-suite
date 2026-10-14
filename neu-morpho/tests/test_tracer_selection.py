"""Stage-1 tracer selection: units, parity of shape, and the isotropy guard."""
from __future__ import annotations

import os

import numpy as np
import pytest

from neu_morpho import skeleton
from neu_morpho.config import SkeletonConfig


def test_edge_memory_guard_raises_instead_of_oom():
    """The 'edge' cost's per-component graph is capped, and raises rather than letting a
    dask worker die of OOM. The shipped cap cannot fire at a 256^3 block."""
    from neu_morpho import neutu_trace

    m = _two_body_block(n=32)
    tiny = SkeletonConfig(anisotropy=(32.0,) * 3, dust_threshold=0, tracer="neutu",
                          neutu_cost="edge", neutu_edge_max_gb=1e-9)
    with pytest.raises(MemoryError, match="neutu_edge_max_gb|edge"):
        skeleton.skeletonize_block(m, (0, 0, 0), tiny)

    assert 256 ** 3 * neutu_trace.EDGE_BYTES_PER_VOXEL / 1e9 < SkeletonConfig().neutu_edge_max_gb


def _driver():
    from neu_morpho import cli

    return cli


BASE_ARGV = ["run", "--src", "x", "--dst", "y", "--work-dir", "z"]


@pytest.mark.parametrize("extra, ok", [
    (["--roi", "0,0,0,64,64,64"], False),        # an ROI must say which scale it is in
    (["--roi-scale", "2"], False),               # a scale with no ROI is a mistake
    (["--roi", "0,0,0,64,64,64", "--roi-scale", "2"], True),
    ([], True),
], ids=["roi-without-scale", "scale-without-roi", "both", "neither"])
def test_roi_and_roi_scale_go_together(extra, ok):
    """An ROI read at the wrong scale processes a fraction of the volume and still
    succeeds, so --roi and --roi-scale are only accepted together."""
    parse = _driver()._parse_args
    if not ok:
        with pytest.raises(SystemExit):
            parse(BASE_ARGV + extra)
        return
    args = parse(BASE_ARGV + extra)
    assert (args.roi is None) == (args.roi_scale is None)


def test_default_tracer_is_neutu_in_the_config_and_the_cli():
    """The default tracer is a decision, and the CLI passes --tracer explicitly, so the
    two defaults could drift into producing different skeletons."""
    args = _driver()._parse_args(BASE_ARGV)
    assert args.tracer == SkeletonConfig().tracer == "neutu"
    assert args.neutu_cost == SkeletonConfig().neutu_cost == "edge"


def _two_body_block(n=48, r=4):
    """One block holding two labelled tubes."""
    m = np.zeros((n, n, n), dtype=np.uint64)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= 4) & (zz < n - 4) & ((yy - 12) ** 2 + (xx - 12) ** 2 <= r * r)] = 7
    m[(zz >= 4) & (zz < n - 4) & ((yy - 34) ** 2 + (xx - 34) ** 2 <= r * r)] = 9
    return np.asfortranarray(m)


def _cfg(**kw):
    return SkeletonConfig(anisotropy=(32.0, 32.0, 32.0), dust_threshold=0, **kw)


@pytest.mark.parametrize("tracer", ["kimimaro", "neutu"])
def test_both_tracers_return_both_bodies_in_nm(tracer):
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
    assert set(out) == {7, 9}
    for body_id, skel in out.items():
        v = np.asarray(skel.vertices, float)
        # nm, not voxels: a 48-voxel block at 32 nm spans ~1536 nm
        assert v[:, 0].max() > 200.0, f"{tracer}: vertices look like voxels, not nm"
        assert v.max() <= 48 * 32.0 + 1e-3


def test_neutu_radii_are_in_nm_not_voxels():
    """neutu_trace works in voxels, so radii must be scaled to nm; positions alone would
    not reveal radii 32x too small."""
    m = _two_body_block(r=4)
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu"))
    radii = np.concatenate([np.asarray(s.radii, float) for s in out.values()])
    # a radius-4-voxel tube at 32 nm/voxel is ~128 nm, nowhere near 4
    assert radii.max() > 60.0, f"radii look like voxels: max {radii.max():.2f}"
    assert radii.max() < 32.0 * 8


@pytest.mark.parametrize("tracer", ["kimimaro", "neutu"])
def test_block_origin_is_applied(tracer):
    m = _two_body_block()
    origin = (10, 20, 30)
    at0 = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
    off = skeleton.skeletonize_block(m, origin, _cfg(tracer=tracer))
    d0 = np.asarray(at0[7].vertices, float).min(axis=0)
    d1 = np.asarray(off[7].vertices, float).min(axis=0)
    assert np.allclose(d1 - d0, np.asarray(origin, float) * 32.0, atol=1e-3)


def test_neutu_tracer_rejects_anisotropic_voxels():
    m = _two_body_block()
    cfg = SkeletonConfig(tracer="neutu", anisotropy=(40.0, 32.0, 32.0))
    with pytest.raises(ValueError, match="isotropic"):
        skeleton.skeletonize_block(m, (0, 0, 0), cfg)


def test_unknown_tracer_raises():
    m = _two_body_block()
    with pytest.raises(ValueError, match="unknown tracer"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="nope"))


@pytest.mark.parametrize("cost", ["voxel", "edge"])
def test_neutu_cost_mode_reaches_the_tracer(cost):
    """Both cost modes reach the tracer and trace."""
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                        neutu_cost=cost))
    assert set(out) == {7, 9}
    for skel in out.values():
        v = np.asarray(skel.vertices, float)
        assert len(v) and v[:, 0].max() > 200.0        # still nm, still traced


def test_neutu_cost_is_validated_not_silently_ignored():
    m = _two_body_block()
    with pytest.raises(ValueError, match="voxel|edge"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                      neutu_cost="nonsense"))


# Config resolution itself is tested in blockrun; here, --config reaches it and fails early.
def test_config_defaults_to_the_bundled_local_template_and_is_repeatable(tmp_path):
    cli = _driver()
    assert cli._configs(cli._parse_args(BASE_ARGV)) == ["dask-local"]
    # a real file, because every --config value is resolved at parse time
    site = tmp_path / "site.yaml"
    site.write_text("jobqueue:\n  slurm:\n    account: your-account\n")
    args = cli._parse_args(BASE_ARGV + ["--config", "dask-slurm-example",
                                        "--config", str(site)])
    assert cli._configs(args) == ["dask-slurm-example", str(site)]


def test_bad_config_fails_at_parse_time_not_at_cluster_start():
    """A --config typo fails at parse time, before any network read of --src."""
    cli = _driver()
    with pytest.raises((SystemExit, FileNotFoundError), match="dask-local"):
        cli._parse_args(BASE_ARGV + ["--config", "definitely-not-a-config"])


def test_serial_mode_does_not_require_a_config():
    """--serial runs in-process with no cluster, so a config is meaningless there."""
    cli = _driver()
    args = cli._parse_args(BASE_ARGV + ["--serial", "--config", "not-a-real-config"])
    assert args.serial


def test_a_subcommand_is_required():
    """A bare `neu-morpho` must not silently mean `run`."""
    cli = _driver()
    with pytest.raises(SystemExit):
        cli._parse_args([])
