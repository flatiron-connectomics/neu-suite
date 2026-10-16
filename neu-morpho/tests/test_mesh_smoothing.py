"""Mesh smoothing: before decimation, recorded in the plan, warned of across blocks."""

import json

import numpy as np
import pytest

pytest.importorskip("vol2mesh")

from neu_morpho import pipeline  # noqa: E402
from neu_morpho.config import MeshConfig  # noqa: E402
from neu_morpho.coords import physical_box  # noqa: E402
from neu_morpho.mesh import mesh_block  # noqa: E402


def _ball():
    z, y, x = np.ogrid[:32, :32, :32]
    return (((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2) < 10 ** 2).astype(np.uint64)


def test_smoothing_runs_before_decimation(monkeypatch):
    from vol2mesh import Mesh

    calls = []
    smooth, simplify = Mesh.laplacian_smooth, Mesh.simplify
    monkeypatch.setattr(Mesh, "laplacian_smooth",
                        lambda self, n, **kw: (calls.append("smooth"), smooth(self, n, **kw))[1])
    monkeypatch.setattr(Mesh, "simplify",
                        lambda self, f, **kw: (calls.append("simplify"), simplify(self, f, **kw))[1])
    region = (slice(0, 32),) * 3
    mesh_block(_ball(), physical_box(region, (8, 8, 8)),
               MeshConfig(mesh_scale=0, block_shape=(32, 32, 32), smoothing_iterations=3))
    assert calls == ["smooth", "simplify"]


def _roughness(mesh) -> float:
    """Mean distance of a vertex from the average of its neighbours (nm)."""
    v, f = np.asarray(mesh.vertices_zyx, dtype=np.float64), np.asarray(mesh.faces)
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                              axis=1), axis=0)
    total, count = np.zeros_like(v), np.zeros(len(v))
    for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
        np.add.at(total, a, v[b])
        np.add.at(count, a, 1)
    used = count > 0
    return float(np.linalg.norm(v[used] - total[used] / count[used, None], axis=1).mean())


def test_smoothing_takes_the_steps_out_of_a_blocky_surface():
    region = (slice(0, 32),) * 3
    rough = {n: _roughness(mesh_block(
        _ball(), physical_box(region, (8, 8, 8)),
        MeshConfig(mesh_scale=0, block_shape=(32, 32, 32), decimation_fraction=1.0,
                   smoothing_iterations=n))[1]) for n in (0, 10)}
    assert rough[10] < 0.5 * rough[0]


def _volume(tmp_path):
    from neu_lib import Frame, Piece
    from neu_vol import convert, write_piece

    write_piece(Piece(_ball(), Frame((8.0,) * 3), kind="segmentation", name="seg"),
                str(tmp_path / "seg.h5"))
    src = str(tmp_path / "seg")
    convert(str(tmp_path / "seg.h5"), src, kind="segmentation", profile="local-neuroglancer",
            chunk=(16, 16, 16), multiscale=False, progress_path=str(tmp_path / "p.jsonl"))
    return src


def test_the_plan_carries_smoothing_and_warns_across_blocks(tmp_path, caplog):
    src = _volume(tmp_path)
    common = dict(work_dir=str(tmp_path / "work"), stages=("mesh",), mesh_scale=0,
                  occupancy_scale=-1, metrics_db=False, mesh_smoothing=5)
    with caplog.at_level("WARNING", logger="neu-morpho"):
        one = pipeline.plan(src, src, block=(32, 32, 32), **common)
    assert one.mesh_smoothing == 5 and "moves the seams" not in caplog.text
    with caplog.at_level("WARNING", logger="neu-morpho"):
        pipeline.plan(src, src, block=(16, 16, 16), **common)
    assert "mesh smoothing on 8 blocks" in caplog.text

    pipeline.run(one)
    plan = json.loads((tmp_path / "work" / "run_plan.json").read_text())
    assert plan["stages"]["mesh"]["mesh_smoothing"] == 5


def test_the_cli_passes_smoothing_to_the_plan(tmp_path, monkeypatch):
    from neu_morpho import cli

    seen = {}
    monkeypatch.setattr(pipeline, "run", lambda p, **kw: seen.update(s=p.mesh_smoothing) or
                        {"n_failed_bodies": 0})
    src = _volume(tmp_path)
    assert cli.main(["run", "--src", src, "--dst", src, "--work-dir", str(tmp_path / "w"),
                     "--stages", "mesh", "--mesh-scale", "0", "--block", "32,32,32",
                     "--occupancy-scale", "-1", "--no-metrics-db", "--serial",
                     "--smoothing", "4"]) == 0
    assert seen == {"s": 4}
