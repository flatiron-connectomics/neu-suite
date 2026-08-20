"""The scene is a value, so everything a figure would contain is assertable here.

This is the whole point of the seam: the predecessor could only answer "what will this
draw?" by drawing it on a GPU.
"""

import numpy as np
import pytest

from em_viz.geometry import BBox, Mesh, Skeleton
from em_viz.scene import (LinesDrawable, MeshDrawable, PointsDrawable, Scene,
                          VolumeDrawable, build_scene)


def _mesh(name="body-1", offset=0.0):
    verts = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.0, 10, 0], [0.0, 0, 10]]) + offset
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


def _skeleton(name="skel-1"):
    verts = np.array([[0.0, 0, 0], [0.0, 0, 50.0]])
    return Skeleton(verts, np.array([[0, 1]]), name=name)


def test_a_scene_is_inspectable_without_a_renderer():
    scene = Scene().add_mesh(_mesh()).add_skeleton(_skeleton())
    assert len(scene) == 2
    assert scene.names == ["body-1", "skel-1"]
    assert isinstance(scene.get("body-1"), MeshDrawable)
    assert isinstance(scene.get("skel-1"), LinesDrawable)


def test_the_bbox_is_the_union_over_visible_drawables():
    scene = Scene().add_mesh(_mesh()).add_points(np.array([[0.0, 0, 100.0]]),
                                                 name="synapses")
    assert scene.bbox == BBox((0, 0, 0), (11, 11, 101))


def test_a_hidden_drawable_does_not_drag_the_camera():
    """Framing follows what is shown. A hidden layer stretching the box would zoom the
    view out for something nobody can see."""
    scene = Scene().add_mesh(_mesh())
    scene.add_points(np.array([[0.0, 0, 10_000.0]]), name="far", visible=False)
    assert scene.bbox == BBox((0, 0, 0), (11, 11, 11))


def test_an_empty_scene_has_an_empty_box_rather_than_raising():
    assert Scene().bbox.is_empty()


def test_a_duplicate_name_is_a_collision_not_a_second_layer():
    """Names key the legend and every later reference, so two sharing one is an error."""
    scene = Scene().add_mesh(_mesh("a"))
    with pytest.raises(ValueError, match="already here"):
        scene.add_mesh(_mesh("a"))


def test_rename_makes_room_instead_of_raising():
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("a"), rename=True)
    assert scene.names == ["a", "a (2)"]


def test_geometry_names_carry_into_the_scene_by_default():
    assert Scene().add_mesh(_mesh("KC-7")).names == ["KC-7"]


def test_an_unknown_marker_is_caught_when_the_scene_is_built():
    """pygfx would reject it at draw time, by which point the scene already exists."""
    with pytest.raises(ValueError, match="unknown marker"):
        PointsDrawable(np.zeros((1, 3)), marker="hexagram")


def test_the_volume_slot_is_reserved_and_says_so():
    with pytest.raises(NotImplementedError, match="EM-VIZ-PLAN"):
        VolumeDrawable()


# --------------------------------------------------------------------------- #
# colouring
# --------------------------------------------------------------------------- #

def test_recolor_assigns_over_the_whole_set_at_once():
    """Assigning on `add` cannot know what is coming, so it cannot guarantee distinct
    neighbours. Colouring the finished set can."""
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("b")).recolor()
    assert scene.get("a").color != scene.get("b").color


def test_recolor_honours_an_explicit_mapping():
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("b"))
    scene.recolor({"b": "r"})
    assert scene.get("b").color == (1.0, 0.0, 0.0, 1.0)


def test_recolor_leaves_the_geometry_alone():
    mesh = _mesh("a")
    scene = Scene().add_mesh(mesh).recolor()
    assert scene.get("a").mesh is mesh


def test_a_colour_passed_to_add_is_normalised_immediately():
    assert Scene().add_mesh(_mesh(), color="r").get("body-1").color == (1.0, 0, 0, 1.0)


# --------------------------------------------------------------------------- #
# build_scene
# --------------------------------------------------------------------------- #

def test_build_scene_makes_one_body_opaque_and_several_translucent():
    """The predecessor's rule, kept: overlapping opaque surfaces hide each other."""
    assert build_scene(meshes=[_mesh("a")]).get("a").alpha == 1.0
    several = build_scene(meshes=[_mesh("a"), _mesh("b", 5.0)])
    assert several.get("a").alpha == 0.8


def test_build_scene_colours_bodies_distinctly_and_takes_overrides():
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 5.0)], colors={"a": "k"})
    assert scene.get("a").color == (0.0, 0.0, 0.0, 1.0)
    assert scene.get("b").color != scene.get("a").color


def test_build_scene_takes_named_point_sets():
    scene = build_scene(skeletons=[_skeleton()],
                        points={"presyn": np.array([[0.0, 0, 25.0]])})
    assert isinstance(scene.get("presyn"), PointsDrawable)
    assert scene.bbox == BBox((0, 0, 0), (1, 1, 51))


def test_build_scene_of_nothing_is_an_empty_scene():
    scene = build_scene()
    assert len(scene) == 0 and scene.bbox.is_empty()


def test_a_bodys_mesh_and_skeleton_are_labelled_by_representation():
    """The most ordinary request there is, and both are named after the body — so they
    collide. Found against the real sample3 volume, not in a test."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")])
    assert scene.names == ["1401 mesh", "1401 skeleton"]


def test_the_suffix_goes_on_BOTH_so_a_name_does_not_depend_on_ordering():
    """Suffixing only the second would make "1401" mean the mesh here and the skeleton
    in a scene assembled the other way round."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")])
    assert "1401" not in scene.names


def test_an_unshared_name_is_left_alone():
    scene = build_scene(meshes=[_mesh("a")], skeletons=[_skeleton("b")])
    assert scene.names == ["a", "b"]


def test_a_point_set_sharing_a_bodys_name_is_disambiguated_too():
    scene = build_scene(skeletons=[_skeleton("x")],
                        points={"x": np.array([[0.0, 0, 0]])})
    assert set(scene.names) == {"x skeleton", "x"}
