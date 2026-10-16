"""A resume may not reuse completed stage-1 blocks on a different block grid."""

import pytest

from blockrun import Manifest

from neu_morpho.ops.manifest import GridChanged, check_grid, grid_record

GRID = grid_record((256, 256, 256), (32, 32, 32))


def _manifest(tmp_path, grid, complete=2):
    m = Manifest(str(tmp_path / "progress.jsonl"))
    m.load()
    if grid is not None:
        check_grid(m, "chunk", grid, stage="mesh", progress_path="p", record=True)
    m.record("chunk", [((i, 0, 0), "written") for i in range(complete)])
    return Manifest(str(tmp_path / "progress.jsonl")).load()


def _check(m, grid, record=False):
    check_grid(m, "chunk", grid, stage="mesh", progress_path="p", record=record)


@pytest.mark.parametrize("grid, match", [
    (grid_record((512, 512, 512), (32, 32, 32)), r"\[256, 256, 256\] voxels of .* uses \[512"),
    (grid_record((256, 256, 256), (16, 16, 16)), r"of \[16.0, 16.0, 16.0\] nm"),
])
def test_a_different_block_shape_or_scale_is_refused(tmp_path, grid, match):
    with pytest.raises(GridChanged, match=match):
        _check(_manifest(tmp_path, GRID), grid)


def test_the_same_grid_resumes_and_a_fresh_start_records_its_own(tmp_path):
    _check(_manifest(tmp_path, GRID), GRID)
    m = _manifest(tmp_path / "fresh", GRID, complete=0)
    other = grid_record((512, 512, 512), (32, 32, 32))
    _check(m, other, record=True)
    assert Manifest(str(tmp_path / "fresh" / "progress.jsonl")).load().meta("chunk")[
        "grid"] == other


def test_a_manifest_from_before_grids_were_recorded_is_not_checked(tmp_path):
    _check(_manifest(tmp_path, None), grid_record((512, 512, 512), (8, 8, 8)))


def _source(tmp_path):
    from test_pipeline import _source as source

    return source(tmp_path)


def test_meshing_resumed_on_another_block_shape_is_refused(tmp_path):
    pytest.importorskip("vol2mesh")
    from neu_morpho import pipeline

    src = _source(tmp_path)
    common = dict(stages=("mesh",), mesh_scale=0, occupancy_scale=-1, metrics_db=False,
                  work_dir=str(tmp_path / "work"))
    pipeline.run(pipeline.plan(src, str(tmp_path / "out" / "seg"), block=(32, 32, 32),
                               **common))
    with pytest.raises(GridChanged, match="Use a new --work-dir"):
        pipeline.run(pipeline.plan(src, str(tmp_path / "out" / "seg"), block=(64, 64, 64),
                                   **common))
    pipeline.run(pipeline.plan(src, str(tmp_path / "out" / "seg"), block=(64, 64, 64),
                               resume=False, **common))


def test_indexing_resumed_on_another_scale_is_refused(tmp_path):
    from neu_morpho import pipeline

    src = _source(tmp_path)
    common = dict(stages=("index",), block=(32, 32, 32), occupancy_scale=-1,
                  work_dir=str(tmp_path / "work"))
    pipeline.run(pipeline.plan(src, str(tmp_path / "out" / "seg"), index_scale=0, **common))
    with pytest.raises(GridChanged, match="index stage-1 blocks"):
        pipeline.run(pipeline.plan(src, str(tmp_path / "out" / "seg"), index_scale=1,
                                   **common))
