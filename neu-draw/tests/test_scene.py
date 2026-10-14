"""The scene is a plain value: everything a figure would contain is assertable without a GPU."""

import numpy as np
import pytest

from neu_lib import BBox, Mesh, Skeleton
from neu_draw.scene import (LinesDrawable, MeshDrawable, PointsDrawable, Scene,
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
    """Framing follows what is shown."""
    scene = Scene().add_mesh(_mesh())
    scene.add_points(np.array([[0.0, 0, 10_000.0]]), name="far", visible=False)
    assert scene.bbox == BBox((0, 0, 0), (11, 11, 11))


def test_a_duplicate_name_is_a_collision_not_a_second_layer():
    """Names are identities; sharing a legend row is what `label` is for."""
    scene = Scene().add_mesh(_mesh("a"))
    with pytest.raises(ValueError, match="already here"):
        scene.add_mesh(_mesh("a"))


def test_rename_makes_room_instead_of_raising():
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("a"), rename=True)
    assert scene.names == ["a", "a (2)"]


def test_an_unknown_marker_is_caught_when_the_scene_is_built():
    """Caught here rather than by pygfx at draw time."""
    with pytest.raises(ValueError, match="unknown marker"):
        PointsDrawable(np.zeros((1, 3)), marker="hexagram")


def test_the_volume_slot_is_reserved_and_says_so():
    with pytest.raises(NotImplementedError, match="reserved for scalar fields"):
        VolumeDrawable()


# --------------------------------------------------------------------------- #
# colouring
# --------------------------------------------------------------------------- #

def test_recolor_assigns_over_the_whole_set_at_once():
    """Colouring the finished set can guarantee distinct neighbours; `add` cannot."""
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


def test_set_color_changes_one_drawable_and_no_other():
    """Unlike `recolor`, which re-assigns the whole set from the palette."""
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("b")).recolor()
    others = scene.get("b").color

    scene.set_color("a", "tab:pink")
    assert scene.get("a").color == pytest.approx((0.8902, 0.4667, 0.7608, 1.0), abs=1e-3)
    assert scene.get("b").color == others


@pytest.mark.parametrize("call", [lambda s: s.set_color("b", "r"),
                                  lambda s: s.rename("z", "y")])
def test_an_unknown_name_says_what_there_is(call):
    with pytest.raises(KeyError, match="no drawable named"):
        call(Scene().add_mesh(_mesh("a")))


# --------------------------------------------------------------------------- #
# renaming
# --------------------------------------------------------------------------- #

def test_rename_changes_the_name_a_caller_refers_to_the_layer_by():
    """`rename` changes identity (what `get` answers to); `relabel` changes a row's text."""
    scene = Scene().add_mesh(_mesh("1401")).add_mesh(_mesh("1402"))
    scene.rename("1401", "MeCN-01 (L)")
    assert scene.names == ["MeCN-01 (L)", "1402"]
    assert scene.get("MeCN-01 (L)").color == scene.drawables[0].color


def test_renaming_onto_a_name_in_use_is_a_collision():
    scene = Scene().add_mesh(_mesh("a")).add_mesh(_mesh("b"))
    with pytest.raises(ValueError, match="collision"):
        scene.rename("a", "b")


def test_renaming_a_drawable_to_its_own_name_is_allowed():
    """A no-op, not a collision."""
    scene = Scene().add_mesh(_mesh("a"))
    assert scene.rename("a", "a").names == ["a"]


# --------------------------------------------------------------------------- #
# legend intent
# --------------------------------------------------------------------------- #

def test_a_legend_is_wanted_by_default():
    assert Scene().legend.visible is True


def test_only_the_two_vertical_legend_locations_exist():
    from neu_draw.scene import Legend

    assert Legend(location="left").location == "left"
    with pytest.raises(ValueError, match="unknown legend location"):
        Legend(location="bottom")


def test_a_nonsense_font_size_is_refused_where_it_is_written():
    from neu_draw.scene import Legend

    with pytest.raises(ValueError, match="font_size must be positive"):
        Legend(font_size=0)


# --------------------------------------------------------------------------- #
# build_scene
# --------------------------------------------------------------------------- #

def test_build_scene_makes_one_body_opaque_and_several_translucent():
    """Overlapping opaque surfaces would hide each other."""
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


def test_an_empty_scene_has_an_empty_box_rather_than_raising():
    scene = build_scene()
    assert len(scene) == 0 and scene.bbox.is_empty()
    assert Scene().bbox.is_empty()


def test_a_bodys_mesh_and_skeleton_are_labelled_by_representation():
    """Both are suffixed, so neither name depends on the order they were added."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")])
    assert scene.names == ["1401 mesh", "1401 skeleton"]


def test_an_unshared_name_is_left_alone():
    scene = build_scene(meshes=[_mesh("a")], skeletons=[_skeleton("b")])
    assert scene.names == ["a", "b"]


def test_a_point_set_sharing_a_bodys_name_is_disambiguated_too():
    scene = build_scene(skeletons=[_skeleton("x")],
                        points={"x": np.array([[0.0, 0, 0]])})
    assert set(scene.names) == {"x skeleton", "x"}
