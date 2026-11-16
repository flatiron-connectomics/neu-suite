"""Mesh names, cache tags, OBJ writing, smoothing and the export path, on small meshes."""

import io

import numpy as np
import pytest

from neu_movie import meshes

# A tetrahedron and, apart from it, a single triangle.
TET_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
TET_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
TWO_V = np.vstack([TET_V, TET_V[:3] + 10])
TWO_F = np.vstack([TET_F, [[4, 5, 6]]])


@pytest.mark.parametrize("label, safe", [("LA(L)", "LA_L"), ("LAL", "LAL"),
                                         ("chain-1", "chain1"), ("short PRs", "shortPRs"),
                                         ("CBL_EB", "CBL_EB")])
def test_safe_name(label, safe):
    assert meshes.safe_name(label) == safe


def test_volume_tag_is_stable_and_ignores_a_trailing_slash():
    a = meshes.volume_tag("s3://my-bucket/v1.0/seg")
    assert a == meshes.volume_tag("s3://my-bucket/v1.0/seg/")
    assert a.startswith("seg_") and a != meshes.volume_tag("s3://my-bucket/v0.8/seg")


def test_vertex_normals_point_outward():
    n = meshes.vertex_normals(TET_V, TET_F)
    centre = TET_V.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", n, TET_V - centre) > 0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1)


def grid(n=11):
    """An n x n triangulated sheet in the xy plane, its middle vertex raised by 1."""
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(n * n)])
    v[(n * n) // 2, 2] = 1.0
    k = np.arange(n * n).reshape(n, n)[:-1, :-1].ravel()
    f = np.vstack([np.column_stack([k, k + 1, k + n]), np.column_stack([k + 1, k + n + 1, k + n])])
    return v, f


def test_taubin_flattens_a_spike_without_shrinking_the_sheet():
    v, f = grid()
    assert meshes.taubin(v, f, 0) is v
    smoothed = meshes.taubin(v, f, 10)
    assert smoothed[:, 2].max() < 0.3
    width = np.ptp(v[:, 0])
    assert np.ptp(smoothed[:, 0]) == pytest.approx(width, rel=0.1)


def test_largest_component_drops_stray_pieces():
    v, f = meshes.largest_component(TWO_V, TWO_F)
    assert len(f) == 4 and len(v) == 4
    assert np.array_equal(v[f], TWO_V[TWO_F[:4]])


def test_write_obj(tmp_path):
    path = str(tmp_path / "t.obj")
    meshes.write_obj(path, TET_V, TET_F)
    lines = open(path).read().splitlines()
    assert sum(line.startswith("v ") for line in lines) == 4
    assert sum(line.startswith("vn ") for line in lines) == 4
    assert "f 1//1 3//3 2//2" in lines
    assert not (tmp_path / "t.obj.part").exists()


@pytest.fixture
def fake_store(monkeypatch):
    pytest.importorskip("neu_vol")
    import neu_vol.readback as readback
    reads = []

    def read_body_mesh(volume, sid, lod=None):
        reads.append(sid)
        return TWO_V * 8.0, TWO_F, 0
    monkeypatch.setattr(readback, "read_body_mesh", read_body_mesh)
    return reads


def test_export_converts_units_keeps_one_piece_and_writes_an_offset_copy(tmp_path, fake_store):
    path, copy = str(tmp_path / "a" / "95.obj"), str(tmp_path / "b" / "shell.obj")
    log = io.StringIO()
    assert meshes.export("vol", 95, path, 8.0, 0, one_piece=True, offset_copy=(copy, 2.0),
                         log=log) == 4
    assert "1 in stray pieces dropped" in log.getvalue()
    v = np.array([[float(x) for x in line.split()[1:]]
                  for line in open(path) if line.startswith("v ")])
    assert np.allclose(v, TET_V, atol=1e-3)                      # nm / 8 nm voxels
    shell = np.array([[float(x) for x in line.split()[1:]]
                      for line in open(copy) if line.startswith("v ")])
    assert np.all(np.linalg.norm(shell - v, axis=1) == pytest.approx(2.0, abs=1e-2))
    assert meshes.export("vol", 95, path, 8.0, 0, offset_copy=(copy, 2.0)) is None
    assert fake_store == [95]


def test_export_bodies_skips_bodies_in_the_cache(tmp_path, fake_store):
    (tmp_path / "1.obj").write_text("")
    log = io.StringIO()
    meshes.export_bodies("vol", ["1", "2", "3"], str(tmp_path), 8.0, 0, smooth=0, log=log)
    assert sorted(fake_store) == [2, 3]
    assert "exporting 2 of 3" in log.getvalue()
    meshes.export_bodies("vol", ["1"], str(tmp_path), 8.0, 0, smooth=0, force=True, log=log)
    assert sorted(fake_store) == [1, 2, 3]
