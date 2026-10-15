"""A resume may not ask stage 1 for bodies its completed blocks were not run for."""

import json

import pytest

from blockrun import Manifest

from neu_morpho.ops.manifest import SelectionChanged, check_selection


def _manifest(tmp_path, selection, complete=2):
    m = Manifest(str(tmp_path / "progress.jsonl"))
    m.load()
    check_selection(m, "chunk", selection, stage="mesh", progress_path="p", record=True)
    m.record("chunk", [((i, 0, 0), "written") for i in range(complete)])
    return Manifest(str(tmp_path / "progress.jsonl")).load()      # as a resume reads it


def _check(m, allow, record=False):
    check_selection(m, "chunk", allow, stage="mesh", progress_path="p", record=record)


def test_new_bodies_after_stage_1_completed_blocks_are_refused(tmp_path):
    m = _manifest(tmp_path, {1, 2, 3})
    with pytest.raises(SelectionChanged, match=r"2 mesh stage-1 blocks .* 1 bodies .* 9"):
        _check(m, {1, 9})


def test_every_label_after_a_selection_is_refused(tmp_path):
    with pytest.raises(SelectionChanged, match="every label, where they were run for 3"):
        _check(_manifest(tmp_path, {1, 2, 3}), None)


@pytest.mark.parametrize("before, allow", [({1, 2, 3}, {1, 2, 3}), ({1, 2, 3}, {2}),
                                           (None, {5}), (None, None)])
def test_the_same_selection_or_a_subset_resumes(tmp_path, before, allow):
    _check(_manifest(tmp_path, before), allow)


def test_a_subset_resume_narrows_the_record_for_the_blocks_it_runs(tmp_path):
    """The remaining blocks see only the subset, so a later run may not widen it again."""
    m = _manifest(tmp_path, {1, 2, 3})
    _check(m, {1, 2}, record=True)
    with pytest.raises(SelectionChanged, match="e.g. 3"):
        _check(Manifest(str(tmp_path / "progress.jsonl")).load(), {1, 2, 3})


def test_with_no_complete_block_any_selection_starts_over(tmp_path):
    m = _manifest(tmp_path, {1}, complete=0)
    _check(m, {1, 9}, record=True)
    assert Manifest(str(tmp_path / "progress.jsonl")).load().meta("chunk")[
        "selection"] == [1, 9]


def test_a_manifest_from_before_selections_were_kept_is_not_checked(tmp_path):
    m = Manifest(str(tmp_path / "progress.jsonl"))
    m.load()
    m.record("chunk", [((0, 0, 0), "written")])
    _check(Manifest(str(tmp_path / "progress.jsonl")).load(), {9})


def test_meshing_new_bodies_in_a_finished_work_dir_is_refused(tmp_path):
    """The run that reported success having made nothing, end to end."""
    pytest.importorskip("vol2mesh")
    from neu_morpho import pipeline

    from test_pipeline import _source

    src = _source(tmp_path)                                # bodies 7 and 9
    dst = str(tmp_path / "out" / "seg")
    only = tmp_path / "only.csv"
    common = dict(stages=("mesh",), mesh_scale=0, block=(32, 32, 32), occupancy_scale=-1,
                  metrics_db=False)
    only.write_text("7\n")
    pipeline.run(pipeline.plan(src, dst, allowlist=str(only), **common), command="a")
    only.write_text("7\n9\n")
    with pytest.raises(SelectionChanged, match="e.g. 9"):
        pipeline.run(pipeline.plan(src, dst, allowlist=str(only), **common), command="b")
    view = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert view["runs"][-1]["status"] == "aborted"
