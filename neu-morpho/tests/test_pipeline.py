"""`pipeline.plan` / `pipeline.run`: the `neu-morpho run` command as a library call."""

import json
import os

import numpy as np
import pytest

from neu_morpho import pipeline


def _source(tmp_path):
    """A two-level precomputed segmentation: a rod across blocks and a blob."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    vol = np.zeros((64, 64, 64), np.uint64)
    vol[4:60, 30:36, 30:36] = 7
    vol[10:22, 8:20, 8:20] = 9
    raw = str(tmp_path / "raw.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", raw, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    src = str(tmp_path / "seg")
    convert(raw, src, voxel_size=(8, 8, 8), kind="segmentation",
            profile="local-neuroglancer", chunk=(16, 16, 16), factors=[(2, 2, 2)],
            min_dim=8, delete_existing=True)
    return src


@pytest.mark.parametrize("kwargs, match", [
    ({"roi": (0, 0, 0, 8, 8, 8)}, "roi_scale is required"),
    ({"roi_scale": 0}, "no effect without roi"),
    ({"stages": ["index"], "metrics_db": False}, "incompatible with the index stage"),
    ({"stages": ["nope"]}, "unknown stage"),
    ({"extra_stages": {"mesh": 0}}, "shadow built-in stages"),
])
def test_plan_refuses_contradictory_arguments(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        pipeline.plan("/no/source", str(tmp_path / "out"), **kwargs)


def test_a_remote_destination_needs_a_work_dir():
    with pytest.raises(ValueError, match="work_dir is required"):
        pipeline.plan("/no/source", "s3://my-bucket/volume")


def test_plan_resolves_scales_and_counts_blocks(tmp_path):
    src = _source(tmp_path)
    p = pipeline.plan(src, str(tmp_path / "out" / "seg"), index_scale=0, mesh_scale=0,
                      stages=("index", "mesh", "extra"), extra_stages={"extra": 1},
                      block=(32, 32, 32), occupancy_scale=-1,
                      roi=(0, 0, 0, 32, 32, 32), roi_scale=1)
    assert p.work_dir == str(tmp_path / "out")
    assert p.scales["extra"].voxel_size == (16.0, 16.0, 16.0)
    assert p.rois["mesh"] == (0, 0, 0, 64, 64, 64)       # the same box, in level-0 voxels
    assert p.planned_blocks == {"index": 8, "mesh": 8, "extra": 1}
    assert not os.path.exists(p.work_dir)                 # planning writes nothing


def test_plan_resolves_only_the_stages_it_runs(tmp_path):
    src = _source(tmp_path)                               # two levels
    p = pipeline.plan(src, str(tmp_path / "out" / "seg"), stages=("extra",),
                      index_scale=7, mesh_scale=7, extra_stages={"extra": 1},
                      block=(32, 32, 32), occupancy_scale=-1)
    assert set(p.scales) == set(p.planned_blocks) == {"extra"}


@pytest.mark.parametrize("kwargs, match", [
    ({}, "metrics.db does not exist"),
    ({"metrics_db": False}, "need metrics_db"),
])
def test_a_size_filter_without_its_counts_is_refused(tmp_path, kwargs, match):
    """Left alone it would select no bodies, or silently select all of them."""
    src = _source(tmp_path)
    with pytest.raises(ValueError, match=match):
        pipeline.plan(src, str(tmp_path / "out" / "seg"), stages=("extra",),
                      extra_stages={"extra": 1}, min_voxels=10, occupancy_scale=-1,
                      **kwargs)


def test_runs_sharing_a_work_dir_accumulate_its_records(tmp_path):
    """`neu-morpho run --stages index`, then `neu-skel run`: both stay on record."""
    src = _source(tmp_path)
    dst, work = str(tmp_path / "out" / "seg"), tmp_path / "out"
    common = dict(index_scale=0, extra_stages={"extra": 1}, block=(32, 32, 32),
                  occupancy_scale=-1)
    pipeline.run(pipeline.plan(src, dst, stages=("index",), **common), command="first")
    p = pipeline.plan(src, dst, stages=("extra",), min_voxels=1, extra={"tool": "x"},
                      **common)
    summary = pipeline.run(p, runners={"extra": lambda p, c, a: {"failed_bodies": [3]}},
                           command="second")

    assert set(summary["stages"]) == {"extra"}           # the return is this run's
    plan = json.loads((work / "run_plan.json").read_text())
    assert plan["stages"] == ["index", "extra"]
    assert plan["commands"] == {"index": "first", "extra": "second"}
    assert set(plan["planned_blocks"]) == {"index", "extra"} and plan["tool"] == "x"
    on_disk = json.loads((work / "run_summary.json").read_text())
    assert list(on_disk["stages"]) == ["index", "extra"]
    assert on_disk["n_failed_bodies"] == 1


def test_run_produces_one_linked_volume(tmp_path):
    pytest.importorskip("vol2mesh")
    src = _source(tmp_path)
    work = tmp_path / "out"
    p = pipeline.plan(src, str(work / "seg"), stages=("seg", "index", "mesh"),
                      index_scale=0, mesh_scale=0, block=(32, 32, 32), occupancy_scale=-1)
    summary = pipeline.run(p, command="test")

    assert summary["n_failed_bodies"] == 0
    assert set(summary["stages"]) == {"seg", "index", "mesh"}
    assert set(summary["linked_subresources"]) == {"mesh"}
    assert "mesh" in json.loads((work / "seg" / "info").read_text())
    assert json.loads((work / "run_summary.json").read_text())["command"] == "test"
    assert json.loads((work / "run_plan.json").read_text())["planned_blocks"]["mesh"] == 8

    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(work / "metrics.db"), read_only=True)
    bodies = dict(db.con.execute("SELECT body_id, voxel_count FROM bodies"))
    db.close()
    assert bodies == {7: 56 * 36, 9: 12 * 144}


def test_an_extra_stage_runs_after_the_built_in_ones_on_the_same_plan(tmp_path):
    """How neu-skel adds its stage: planned like a built-in one, run by its runner."""
    src = _source(tmp_path)
    seen = {}

    def runner(p, client, allow):
        seen.update(scale=p.scales["extra"].index, roi=p.rois["extra"], client=client,
                    index_done=os.path.exists(p.db_path))
        return {"failed_bodies": [3]}

    p = pipeline.plan(src, str(tmp_path / "out" / "seg"), stages=("index", "extra"),
                      index_scale=0, mesh_scale=0, extra_stages={"extra": 1}, block=(32, 32, 32),
                      occupancy_scale=-1, extra={"tool": "x"})
    summary = pipeline.run(p, runners={"extra": runner})
    assert seen == {"scale": 1, "roi": None, "client": None, "index_done": True}
    assert summary["stages"]["extra"] == {"failed_bodies": [3]}
    assert summary["n_failed_bodies"] == 1                  # its failures count too
    assert json.loads((tmp_path / "out" / "run_plan.json").read_text())["tool"] == "x"
    with pytest.raises(ValueError, match="no runner for extra stage"):
        pipeline.run(p)

