"""`readback.mesh_body_ids`: which bodies have a mesh, from one directory listing."""

import json

import pytest

from neu_vol.readback import UnsupportedSubresource, mesh_body_ids


def _mesh_dir(tmp_path, info, names):
    mesh = tmp_path / "vol" / "mesh"
    mesh.mkdir(parents=True)
    if info is not None:
        (mesh / "info").write_text(json.dumps(info))
    for name in names:
        (mesh / name).write_bytes(b"x")
    return str(tmp_path / "vol")


def test_a_multires_mesh_counts_once_its_manifest_exists(tmp_path):
    vol = _mesh_dir(tmp_path, {"@type": "neuroglancer_multilod_draco"},
                    ["12", "12.index", "5", "5.index", "7", "3:0"])
    assert mesh_body_ids(vol) == [5, 12]                 # 7 has no manifest yet


def test_a_legacy_mesh_counts_by_its_manifest_not_its_fragments(tmp_path):
    vol = _mesh_dir(tmp_path, {"@type": "neuroglancer_legacy_mesh"},
                    ["40:0", "40:0:frag-1", "3:0", "3:0:frag-1", "9.index"])
    assert mesh_body_ids(vol) == [3, 40]


def test_an_untyped_mesh_dir_accepts_either_layout(tmp_path):
    vol = _mesh_dir(tmp_path, None, ["2.index", "8:0", "info.bak"])
    assert mesh_body_ids(vol) == [2, 8]


def test_no_mesh_dir_is_no_bodies(tmp_path):
    assert mesh_body_ids(str(tmp_path / "vol")) == []


def test_a_sharded_mesh_is_refused_not_enumerated(tmp_path):
    vol = _mesh_dir(tmp_path, {"@type": "neuroglancer_multilod_draco",
                               "sharding": {"@type": "neuroglancer_uint64_sharded_v1"}},
                    ["0.shard"])
    with pytest.raises(UnsupportedSubresource, match="sharded"):
        mesh_body_ids(vol)


def test_an_unknown_mesh_type_is_refused(tmp_path):
    vol = _mesh_dir(tmp_path, {"@type": "something_else"}, ["1.index"])
    with pytest.raises(UnsupportedSubresource):
        mesh_body_ids(vol)


def test_the_mesh_dir_is_the_one_the_volume_names(tmp_path):
    mesh = tmp_path / "vol" / "meshes-v2"
    mesh.mkdir(parents=True)
    (mesh / "4.index").write_bytes(b"x")
    (tmp_path / "vol" / "info").write_text(json.dumps({"mesh": "meshes-v2"}))
    assert mesh_body_ids(str(tmp_path / "vol")) == [4]
