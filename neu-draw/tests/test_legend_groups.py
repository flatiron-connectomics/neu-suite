"""Legend GROUP rows, and opacity by kind.

A group row sits on top of the per-label rows: one click for many drawables, while each
member keeps a row of its own. The interesting failures are all about the overlap — one
drawable on two rows — so most of these put a drawable on both and check that neither row
lies about it.
"""

import numpy as np
import pytest

from neu_lib import Mesh, Skeleton
from neu_draw.scene import Legend, Scene, build_scene


_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_VERTS = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 100.0, 0], [0.0, 0, 100.0]])


def _mesh(name, shift=0.0):
    return Mesh(_VERTS + shift, _FACES, name=name)


def _skeleton(name):
    return Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 300.0]]), np.array([[0, 1]]),
                    name=name)


def _cells():
    """Two cells, each with a mesh and a skeleton, and one point set per cell and type."""
    points = {f"{c} : {t}": np.array([[i * 10.0, 0, 0]])
              for c in ("A", "B") for i, t in enumerate(("Mi1", "Mi2"))}
    return build_scene(meshes=[_mesh("A"), _mesh("B", 300.0)],
                       skeletons=[_skeleton("A"), _skeleton("B")], points=points)


# --------------------------------------------------------------------------- #
# the declaration — no GPU
# --------------------------------------------------------------------------- #

def test_groups_are_normalised_to_tuples_of_names():
    legend = Legend(groups={"Mi1": ["A : Mi1", "B : Mi1"]})
    assert legend.groups == {"Mi1": ("A : Mi1", "B : Mi1")}


def test_a_bare_string_of_members_is_refused_rather_than_split_into_letters():
    with pytest.raises(TypeError, match="list of names"):
        Legend(groups={"Mi1": "A : Mi1"})


def test_a_group_from_a_predicate_is_resolved_to_names_once():
    scene = _cells().group("Mi1", lambda d: d.name.endswith(": Mi1"))
    assert scene.legend.groups == {"Mi1": ("A : Mi1", "B : Mi1")}


def test_a_group_naming_a_missing_drawable_says_what_there_is():
    with pytest.raises(KeyError, match="not here"):
        _cells().group("Mi1", ["A : Mi1", "nope"])


def test_a_group_matching_nothing_raises():
    with pytest.raises(ValueError, match="matches no drawables"):
        _cells().group("Mi9", lambda d: False)


def test_a_rename_carries_group_membership_along():
    scene = _cells().group("A", ["A mesh", "A skeleton"])
    scene.rename("A mesh", "A surface")
    assert scene.legend.groups["A"] == ("A surface", "A skeleton")


def test_set_alpha_by_kind_touches_only_that_kind():
    scene = _cells().set_alpha(0.2, kind="mesh")
    assert {d.alpha for d in scene.of_kind("mesh")} == {0.2}
    assert 0.2 not in {d.alpha for d in scene.of_kind("skeleton")}


def test_set_alpha_by_name():
    scene = _cells().set_alpha(0.5, names=["A : Mi1"])
    assert scene.get("A : Mi1").alpha == 0.5
    assert scene.get("B : Mi1").alpha != 0.5


def test_set_alpha_refuses_what_it_cannot_mean():
    with pytest.raises(ValueError, match="kind is one of"):
        _cells().set_alpha(0.5, kind="meshes")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _cells().set_alpha(1.5)
    with pytest.raises(KeyError):
        _cells().set_alpha(0.5, names=["nope"])


# --------------------------------------------------------------------------- #
# the rows — a real render
# --------------------------------------------------------------------------- #

# Imported inside the fixture rather than skipped at module level, so the declaration
# tests above still run where the render extra is absent — which is CI.
backend = None


@pytest.fixture
def has_gpu():
    global backend
    pytest.importorskip("pygfx", reason="the render extra is not installed")
    from neu_draw.backends import pygfx as backend
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


@pytest.fixture
def view(has_gpu):
    scene = (_cells()
             .group("Mi1", lambda d: d.name.endswith(": Mi1"))
             .group("cell A", lambda d: d.name.startswith("A")))
    view = backend.show(scene, size=(600, 400), canvas="offscreen", pixel_ratio=1.0)
    yield view
    view.close()


def _click(view, x, y, button=1):
    def event(kind, stamp):
        return dict(event_type=kind, x=float(x), y=float(y), button=button,
                    buttons=(button,), modifiers=(), pointer_id=1, ntouches=0,
                    touches={}, time_stamp=stamp)

    view.renderer.convert_event(event("pointer_down", 1.0))
    view.renderer.convert_event(event("pointer_up", 1.05))


def _objects(view, *names):
    index = {d.name: i for i, d in enumerate(view.scene_data.drawables)}
    return [view.group.children[index[n]] for n in names]


def test_group_rows_come_first_and_every_member_keeps_its_own(view):
    labels = view.legend.labels
    assert labels[:2] == ["Mi1", "cell A"]
    assert set(view.scene_data.names) <= set(labels[2:])
    assert [e.is_group for e in view.legend][:3] == [True, True, False]


def test_clicking_a_group_row_hides_its_members_and_their_own_rows_say_so(view):
    view.snapshot()
    strip = view.legend.rects_for(view.logical_size())[1]
    _click(view, strip[0] + strip[2] / 2, view.legend.row_height * 0.5)   # "Mi1"

    scene = view.scene_data
    assert not scene.get("A : Mi1").visible and not scene.get("B : Mi1").visible
    assert scene.get("A : Mi2").visible
    assert view.legend["A : Mi1"].visibility == "none"
    assert all(not o.visible for o in _objects(view, "A : Mi1", "B : Mi1"))


def test_hiding_one_member_from_its_own_row_leaves_the_group_partly_shown(view):
    before = tuple(view.legend["cell A"].plate.material.color)
    view.legend.toggle("A mesh")
    assert view.legend["cell A"].visibility == "some"
    assert tuple(view.legend["cell A"].plate.material.color) != before


def test_a_member_lit_by_its_own_row_stays_lit_when_its_group_is_dropped(view):
    """The overlap bug groups invite: each row painting its members from its own flag, so
    whichever refreshed last wins and un-lights a body another row had lit."""
    white = pytest.approx(Legend().highlight_color[:3], abs=1e-3)
    view.legend.highlight("A : Mi1")
    view.legend.highlight("Mi1")
    view.legend.unhighlight("Mi1")

    lit, unlit = _objects(view, "A : Mi1", "B : Mi1")
    assert tuple(lit.material.color)[:3] == white
    assert tuple(unlit.material.color)[:3] != white


def test_a_group_plate_reads_as_a_header(view):
    group = tuple(view.legend["Mi1"].plate.material.color)
    row = tuple(view.legend["A : Mi1"].plate.material.color)
    assert group != row


def test_a_relabel_keeps_the_groups_and_their_highlight(view):
    view.legend.highlight("Mi1")
    view.legend.relabel("A mesh", "cell A surface")
    assert view.legend["Mi1"].highlighted
    assert view.legend["cell A"].names == [
        "A mesh", "A skeleton", "A : Mi1", "A : Mi2"]


def test_a_group_added_after_show_appears_on_the_next_frame(view):
    view.scene_data.group("Mi2", lambda d: d.name.endswith(": Mi2"))
    view.snapshot()
    assert "Mi2" in view.legend.labels


def test_a_group_naming_a_missing_drawable_refuses_to_show(has_gpu):
    scene = _cells()
    scene.legend = Legend(groups={"Mi1": ["A : Mi1", "nope"]})
    with pytest.raises(ValueError, match="not in the scene"):
        backend.show(scene, size=(300, 200), canvas="offscreen")


def test_a_group_whose_text_is_also_a_label_refuses_to_show(has_gpu):
    """Every legend method is keyed on row text, so two rows sharing one is ambiguous."""
    scene = _cells()
    scene.legend = Legend(groups={"A mesh": ["A skeleton"]})
    with pytest.raises(ValueError, match="also a label row"):
        backend.show(scene, size=(300, 200), canvas="offscreen")


# --------------------------------------------------------------------------- #
# opacity
# --------------------------------------------------------------------------- #

def test_set_alpha_reaches_the_built_objects(view):
    view.set_alpha(0.25, kind="mesh")
    for obj in _objects(view, "A mesh", "B mesh"):
        assert tuple(obj.material.color)[3] == pytest.approx(0.25)


def test_set_alpha_keeps_a_highlight(view):
    view.legend.highlight("A mesh")
    view.set_alpha(0.25, kind="mesh")
    (obj,) = _objects(view, "A mesh")
    assert tuple(obj.material.color)[:3] == pytest.approx(
        Legend().highlight_color[:3], abs=1e-3)
    assert tuple(obj.material.color)[3] == pytest.approx(0.25)


def test_an_alpha_set_directly_lands_on_the_next_frame_even_without_a_legend(has_gpu):
    scene = _cells()
    view = backend.show(scene, size=(300, 200), canvas="offscreen", legend=False)
    try:
        scene.get("A mesh").alpha = 0.1
        view.snapshot()
        (obj,) = _objects(view, "A mesh")
        assert tuple(obj.material.color)[3] == pytest.approx(0.1)
    finally:
        view.close()
