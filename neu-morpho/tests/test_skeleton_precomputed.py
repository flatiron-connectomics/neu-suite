"""The precomputed skeleton writer: format, and the zyx->xyz flip (getting it wrong
mirrors skeletons through the z=x diagonal relative to their meshes)."""

import os

import numpy as np
import pytest

from neu_morpho.precomputed import encode_skeleton, write_body_skeleton


def _skel(vertices_zyx, edges, radius=None):
    from osteoid import Skeleton
    s = Skeleton(vertices=np.asarray(vertices_zyx, np.float32),
                 edges=np.asarray(edges, np.uint32), segid=42)
    if radius is not None:
        s.radius = np.asarray(radius, np.float32)
    return s


def test_encode_flips_zyx_to_xyz():
    s = _skel([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], [[0, 1]])
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(encode_skeleton(s))
    # stored coordinates are xyz: the zyx (1,2,3) must come back as (3,2,1)
    np.testing.assert_array_equal(back.vertices, [[3, 2, 1], [30, 20, 10]])
    np.testing.assert_array_equal(back.edges, [[0, 1]])


def test_encode_preserves_radius():
    s = _skel([[0.0, 0, 0], [1, 1, 1]], [[0, 1]], radius=[5.0, 7.0])
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(encode_skeleton(s))
    np.testing.assert_array_equal(back.radius, [5.0, 7.0])


def test_write_skeleton_info_rejects_non_float32_attributes(tmp_path):
    """The spec allows uint8, but neuroglancer rejects the whole layer for it."""
    from neu_morpho.precomputed import write_skeleton_info as w

    with pytest.raises(ValueError, match="must be float32"):
        w(str(tmp_path / "s"),
          vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1},
                             {"id": "vertex_types", "data_type": "uint8", "num_components": 1}])


def test_encode_drops_osteoid_default_vertex_types():
    """osteoid's default uint8 vertex_types must not reach the file.

    Header 8 + vertices 24 + edge 8 + radii 8 = 48 bytes; a uint8 attribute would add 2.
    """
    s = _skel([[0.0, 0, 0], [1, 1, 1]], [[0, 1]], radius=[5.0, 7.0])
    s.vertex_types = np.array([2, 3], np.uint8)          # producer attached one
    assert len(encode_skeleton(s)) == 48


def test_write_body_skeleton_roundtrip(tmp_path):
    out = str(tmp_path / "skeleton")
    s = _skel([[0.0, 0, 0], [8, 16, 24]], [[0, 1]])
    assert write_body_skeleton(out, 42, s) == 2
    assert write_body_skeleton(out, 43, None) == 0

    path = os.path.join(out, "42")
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(open(path, "rb").read(), segid=42)
    np.testing.assert_array_equal(back.vertices, [[0, 0, 0], [24, 16, 8]])   # xyz
    assert not os.path.exists(os.path.join(out, "43"))


def test_mesh_and_skeleton_agree_after_encoding():
    """After encoding to stored xyz, a skeleton still lands inside its mesh bbox."""
    import kimimaro
    from vol2mesh import Mesh
    from osteoid import Skeleton

    from neu_morpho.coords import physical_box
    from neu_morpho.config import SkeletonConfig
    from neu_morpho.skeleton import skeletonize_block

    vs = (8.0, 8.0, 8.0)
    # asymmetric placement, so a z<->x swap cannot pass by accident
    block = np.zeros((24, 16, 40), np.uint64)
    block[4:20, 6:10, 30:34] = 7

    box = physical_box((slice(0, 24), slice(0, 16), slice(0, 40)), vs)
    mesh = Mesh.from_label_volume(block, box, labels=[7], ensure_halo=True, progress=False)[7]
    m_lo_xyz = mesh.vertices_zyx.min(0)[::-1]
    m_hi_xyz = mesh.vertices_zyx.max(0)[::-1]

    cfg = SkeletonConfig(anisotropy=vs, const=30.0, dust_threshold=0)
    skel = skeletonize_block(block, (0, 0, 0), cfg)[7]
    sv = Skeleton.from_precomputed(encode_skeleton(skel)).vertices

    assert (sv.min(0) >= m_lo_xyz - 8).all() and (sv.max(0) <= m_hi_xyz + 8).all(), \
        (m_lo_xyz, m_hi_xyz, sv.min(0), sv.max(0))
    # x is the narrow axis (~240..272 nm) and z the long one (~32..160): a flip bug swaps them
    assert sv[:, 0].min() > 200 and sv[:, 2].max() < 200


# --------------------------------------------------------------------------- #
# Destination-agnostic writes
# --------------------------------------------------------------------------- #
def test_precomputed_module_uses_no_direct_file_io():
    """A static check, kept because the failure is silent: any direct file write in
    `precomputed.py` (open(), os, multires.write_*) writes nothing to an s3 destination
    and passes every local test."""
    import ast
    import inspect

    from neu_morpho import precomputed

    tree = ast.parse(inspect.getsource(precomputed))

    imported = {alias.name.split(".")[0]
                for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert "os" not in imported, "precomputed.py imports os; writes must go through the kvstore"

    banned_calls = {("multires", "write_info"), ("multires", "write_object_mesh")}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "open":
            found.append("open()")
        elif (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and (fn.value.id, fn.attr) in banned_calls):
            found.append(f"{fn.value.id}.{fn.attr}()")
    assert not found, f"{found} in precomputed.py writes to the filesystem, breaking s3"


def test_skeleton_and_mesh_info_land_via_kvstore(tmp_path):
    """Both info files, written through the kvstore: identity transform, and only
    float32 vertex attributes (neuroglancer rejects the layer otherwise)."""
    from neu_vol import read_json

    from neu_morpho.config import MeshConfig
    from neu_morpho.precomputed import (volume_exists, write_mesh_info,
                                           write_skeleton_info)

    mesh_dir, skel_dir = str(tmp_path / "vol" / "mesh"), str(tmp_path / "vol" / "skeleton")
    assert not volume_exists(mesh_dir)                 # nothing written yet

    write_mesh_info(mesh_dir, MeshConfig())
    write_skeleton_info(skel_dir)
    assert volume_exists(mesh_dir) and volume_exists(skel_dir)
    assert read_json(mesh_dir, "info")["@type"] == "neuroglancer_multilod_draco"
    assert read_json(skel_dir, "info")["@type"] == "neuroglancer_skeletons"
    skel_info = read_json(skel_dir, "info")
    assert skel_info["transform"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert skel_info["vertex_attributes"]
    assert all(a["data_type"] == "float32" for a in skel_info["vertex_attributes"])
