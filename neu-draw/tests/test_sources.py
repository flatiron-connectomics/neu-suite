"""Reading a real precomputed volume, built by the production writers.

The fixture writes its skeletons with `em_seg_morpho.precomputed.write_body_skeleton`
and its `info` with `write_skeleton_info` — the same code that produces the volumes this
reads in anger. Hand-rolling the bytes would prove the reader agrees with the test's
idea of the format, which is the mistake em-libraries invariant 9 is about.

The fixture pyramid is deliberately **anisotropic** (`(1, 2, 2)` — halve x and y, leave
z), because an isotropic one cannot tell a real voxel size from `2 ** level`.
"""

import json

import numpy as np
import pytest

osteoid = pytest.importorskip("osteoid", reason="conda-only; absent in CI")
precomputed = pytest.importorskip("em_seg_morpho.precomputed",
                                  reason="conda-only; absent in CI")

from em_viz import Frame, Mesh, Skeleton                                  # noqa: E402
from em_viz import cache as cache_mod                                     # noqa: E402
from em_viz import sources                                                # noqa: E402
from em_viz.geometry import BBox                                          # noqa: E402

BODIES = (11, 22)


def _osteoid_skeleton(offset_nm, segid):
    """A 3-vertex zyx skeleton with radii — what `encode_skeleton` expects."""
    verts = np.array([[0.0, 0, 0], [0.0, 0, 100.0], [0.0, 80.0, 100.0]]) + offset_nm
    skel = osteoid.Skeleton(vertices=verts.astype(np.float32),
                            edges=np.array([[0, 1], [1, 2]], dtype=np.uint32),
                            segid=segid)
    skel.radius = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    return skel


@pytest.fixture
def volume(tmp_path):
    """A precomputed volume with an anisotropic pyramid and two skeletons."""
    root = tmp_path / "vol"
    root.mkdir()
    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            # xyz, as the format stores. z stays 40 nm while x and y double.
            {"key": "8_8_40", "resolution": [8, 8, 40], "size": [400, 400, 100],
             "chunk_sizes": [[64, 64, 64]], "encoding": "raw", "voxel_offset": [0, 0, 0]},
            {"key": "16_16_40", "resolution": [16, 16, 40], "size": [200, 200, 100],
             "chunk_sizes": [[64, 64, 64]], "encoding": "raw", "voxel_offset": [0, 0, 0]},
        ],
        # The trap this fixture exists to reproduce: the KEY is plural, the DIRECTORY
        # is not, exactly as sample3's own info has it.
        "mesh": "mesh",
        "skeletons": "skeleton",
    }
    (root / "info").write_text(json.dumps(info))
    skel_dir = str(root / "skeleton")
    precomputed.write_skeleton_info(skel_dir)
    for i, body in enumerate(BODIES):
        precomputed.write_body_skeleton(skel_dir, body,
                                        _osteoid_skeleton(i * 1000.0, body))
    return str(root)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def test_the_subresource_directory_comes_from_info_not_a_default():
    """sample3's info reads `{"skeletons": "skeleton"}` — plural key, singular
    directory. A hardcoded "skeletons" finds nothing, and finding nothing is
    indistinguishable from a body simply having no skeleton."""
    assert sources.subresource_dir("x", "skeletons",
                                   info={"skeletons": "skeleton"}) == "skeleton"
    assert sources.subresource_dir("x", "mesh", info={"mesh": "mesh"}) == "mesh"


def test_a_volume_declaring_no_skeletons_says_so():
    with pytest.raises(sources.MissingSubresource, match="declares no 'skeletons'"):
        sources.subresource_dir("x", "skeletons", info={"mesh": "mesh"})


def test_volume_frame_uses_real_voxel_sizes_not_two_to_the_level(volume):
    """The invariant. Level 1 of this pyramid is (40, 16, 16) nm in zyx — z did not
    change — whereas `2 ** 1` would say (16, 16, 16) and put everything at the wrong
    depth."""
    assert sources.volume_frame(volume, 0).voxel_size_nm == (40.0, 8.0, 8.0)
    assert sources.volume_frame(volume, 1).voxel_size_nm == (40.0, 16.0, 16.0)


def test_asking_for_a_level_that_is_not_there_says_which_are(volume):
    with pytest.raises(IndexError, match=r"\[0, 1\]"):
        sources.volume_frame(volume, 7)


# --------------------------------------------------------------------------- #
# skeletons
# --------------------------------------------------------------------------- #

def test_a_skeleton_reads_back_through_the_real_writer(volume):
    skel = sources.body_skeleton(volume, BODIES[0])
    assert isinstance(skel, Skeleton)
    assert len(skel.vertices_zyx_nm) == 3 and len(skel.edges) == 2
    assert skel.radii_nm.tolist() == [10.0, 20.0, 30.0]


def test_vertices_come_back_zyx(volume):
    """The writer flips zyx->xyz, the reader flips back. If either side changed, the
    skeleton would be mirrored through the z=x diagonal and still look like a neuron."""
    skel = sources.body_skeleton(volume, BODIES[0])
    assert skel.vertices_zyx_nm[2].tolist() == [0.0, 80.0, 100.0]


def test_an_absent_body_is_none_not_an_error(volume):
    assert sources.body_skeleton(volume, 999) is None


def test_many_bodies_come_back_keyed_by_id(volume):
    out = sources.body_skeletons(volume, BODIES)
    assert set(out) == set(BODIES)
    assert all(isinstance(s, Skeleton) for s in out.values())


def test_missing_bodies_are_skipped_by_default_and_can_raise(volume):
    assert set(sources.body_skeletons(volume, [*BODIES, 999])) == set(BODIES)
    with pytest.raises(KeyError, match="999"):
        sources.body_skeletons(volume, [*BODIES, 999], skip_missing=False)


def test_names_label_the_result_without_touching_the_cache(volume):
    """A cached object is shared, so labelling one caller's copy must not rename
    everyone else's."""
    store = cache_mod.MemoryCache()
    first = sources.body_skeleton(volume, BODIES[0], cache=store, name="KC-1")
    second = sources.body_skeleton(volume, BODIES[0], cache=store)
    assert first.name == "KC-1"
    assert second.name == str(BODIES[0])


def test_a_second_read_comes_from_the_cache(volume, monkeypatch):
    store = cache_mod.MemoryCache()
    sources.body_skeleton(volume, BODIES[0], cache=store)

    import em_seg_morpho.readback as readback

    def explode(*args, **kwargs):
        raise AssertionError("re-read a body that was already cached")

    monkeypatch.setattr(readback, "read_body_skeleton", explode)
    assert sources.body_skeleton(volume, BODIES[0], cache=store) is not None


def test_the_info_is_read_once_for_a_whole_batch(volume, monkeypatch):
    """Otherwise every body costs an extra round trip for a file that cannot have
    changed mid-call — which on S3 is the difference between one request and N."""
    calls = []
    real = sources.volume_info
    monkeypatch.setattr(sources, "volume_info",
                        lambda v: (calls.append(v), real(v))[1])
    sources.body_skeletons(volume, BODIES)
    assert len(calls) == 1


def test_fetching_nothing_is_an_empty_dict(volume):
    assert sources.body_skeletons(volume, []) == {}


# --------------------------------------------------------------------------- #
# meshes
# --------------------------------------------------------------------------- #

@pytest.fixture
def volume_with_mesh(volume):
    """Adds a real multi-LOD Draco mesh, written by the production writer."""
    vol2mesh = pytest.importorskip("vol2mesh", reason="conda-only; absent in CI")
    from em_seg_morpho.config import MeshConfig

    # A tetrahedron, in nm, well inside one octree cell.
    verts_zyx = np.array([[0.0, 0, 0], [0.0, 0, 400.0], [0.0, 400.0, 0],
                          [400.0, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    mesh = vol2mesh.Mesh(verts_zyx, faces)

    cfg = MeshConfig(num_lods=1)
    mesh_dir = str(__import__("pathlib").Path(volume) / "mesh")
    precomputed.write_mesh_info(mesh_dir, cfg)
    written = precomputed.write_body_multires(
        mesh_dir, BODIES[0], mesh, cfg,
        chunk_shape_xyz=[2048.0, 2048.0, 2048.0], grid_origin_xyz=[0.0, 0.0, 0.0])
    if not written:
        pytest.skip("the mesh writer produced no fragments")
    return volume


def test_a_mesh_reads_back_through_the_real_writer(volume_with_mesh):
    mesh = sources.body_mesh(volume_with_mesh, BODIES[0])
    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices_zyx_nm) > 0 and len(mesh.faces) > 0


def test_the_mesh_lands_in_the_same_frame_as_the_skeleton(volume_with_mesh):
    """The cross-check that catches a one-sided flip: both come from the same volume in
    nm, so their boxes must overlap. Mirrored, they would not."""
    mesh = sources.body_mesh(volume_with_mesh, BODIES[0])
    skel = sources.body_skeleton(volume_with_mesh, BODIES[0])
    assert not mesh.bbox.intersect(skel.bbox).is_empty()


def test_an_absent_mesh_is_none(volume_with_mesh):
    assert sources.body_mesh(volume_with_mesh, 999) is None


def test_a_volume_with_no_mesh_key_raises_rather_than_returning_none(volume):
    """`None` means "this body has none"; a missing declaration is a different thing,
    and conflating them hides a misconfigured volume behind empty output."""
    import json as _json
    import pathlib

    root = pathlib.Path(volume)
    info = _json.loads((root / "info").read_text())
    del info["mesh"]
    (root / "info").write_text(_json.dumps(info))
    with pytest.raises(sources.MissingSubresource):
        sources.body_mesh(volume, BODIES[0])


# --------------------------------------------------------------------------- #
# tubes
# --------------------------------------------------------------------------- #

def test_a_tube_wraps_the_skeleton_and_keeps_its_frame(volume):
    skel = sources.body_skeleton(volume, BODIES[0])
    tube = sources.skeleton_tube(skel, sides=8)
    assert isinstance(tube, Mesh) and len(tube.faces) > 0
    # The tube surrounds the centreline, so its box contains the skeleton's.
    assert tube.bbox.union(skel.bbox) == tube.bbox


def test_a_skeleton_without_radii_has_no_tube(volume):
    bare = Skeleton(np.zeros((2, 3)), np.array([[0, 1]]), name="bare")
    with pytest.raises(ValueError, match="carries no radii"):
        sources.skeleton_tube(bare)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def test_columns_are_read_in_the_order_named():
    table = {"z": [1, 4], "y": [2, 5], "x": [3, 6]}
    assert sources.points_from_table(table).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_a_voxel_table_is_scaled_by_the_frame():
    """DVID's coordinates are voxel indices; a precomputed annotation source's are nm.
    Which one you have is the caller's to say, so `frame` is explicit."""
    table = {"z": [1], "y": [1], "x": [1]}
    out = sources.points_from_table(table, frame=Frame((40.0, 8.0, 8.0)))
    assert out.tolist() == [[40.0, 8.0, 8.0]]


def test_a_missing_coordinate_column_names_what_it_looked_for():
    with pytest.raises(KeyError, match="to_z"):
        sources.points_from_table({"z": [1], "y": [1], "x": [1]}, prefix="to_")


def test_synapses_split_by_kind():
    table = {"z": [0, 1, 2], "y": [0, 0, 0], "x": [0, 0, 0],
             "kind": ["PreSyn", "PostSyn", "PreSyn"]}
    out = sources.synapse_points(table)
    assert set(out) == {"PreSyn", "PostSyn"}
    assert out["PreSyn"].shape == (2, 3) and out["PostSyn"].shape == (1, 3)


def test_a_table_with_no_kind_column_is_one_point_set():
    out = sources.synapse_points({"z": [0], "y": [0], "x": [0]})
    assert set(out) == {"points"}


# --------------------------------------------------------------------------- #
# region predicates
# --------------------------------------------------------------------------- #

def test_a_box_predicate_feeds_straight_into_crop():
    skel = Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 1000.0]]), np.array([[0, 1]]))
    inside = sources.box_predicate(BBox((-10, -10, -10), (10, 10, 500)))
    out = skel.crop(inside, tolerance_nm=1.0)
    assert len(out.vertices_zyx_nm) == 2                 # one kept + one boundary
    assert out.vertices_zyx_nm[1][2] == pytest.approx(500.0, abs=2.0)


def test_a_mask_predicate_resolves_points_through_its_own_frame():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    inside = sources.mask_predicate(mask, Frame((100.0, 100.0, 100.0)))
    assert inside(np.array([[50.0, 50.0, 50.0]])).tolist() == [True]
    assert inside(np.array([[150.0, 50.0, 50.0]])).tolist() == [False]


def test_points_outside_the_mask_array_are_outside_the_region():
    """A skeleton normally leaves any one ROI, so this is the question being asked —
    not an out-of-bounds error."""
    inside = sources.mask_predicate(np.ones((2, 2, 2), bool), Frame((10.0, 10.0, 10.0)))
    assert inside(np.array([[-100.0, 0, 0], [1e6, 0, 0]])).tolist() == [False, False]


def test_a_mask_must_be_three_dimensional():
    with pytest.raises(ValueError, match="3-D zyx"):
        sources.mask_predicate(np.ones((4, 4), bool), Frame.identity())


# --------------------------------------------------------------------------- #
# cache plumbing
# --------------------------------------------------------------------------- #

def test_a_plain_dict_is_a_cache():
    assert isinstance({}, cache_mod.Cache)
    assert cache_mod.resolve({}) is not None


def test_caching_can_be_turned_off_explicitly():
    off = cache_mod.resolve(False)
    off["k"] = 1
    assert "k" not in off


def test_something_that_is_not_a_cache_says_what_is_missing():
    with pytest.raises(TypeError, match="__contains__"):
        cache_mod.resolve(object())
