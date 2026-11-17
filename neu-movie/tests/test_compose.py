"""The neuVid input and the labels, from a spec and a scene; no store, no Blender."""

import numpy as np
import pytest

from conftest import make_spec
from neu_movie import compose as compose_mod
from neu_movie import labels, schedule
from neu_movie.palette import linear_hex

VIEW = {"position": [1, 2, 3], "projectionOrientation": [0, 0, 0, 1], "projectionScale": 1000}


def scene(spec, *, head=False, regions=()):
    ids = {n: ["1"] for n in spec["sets"]}
    return {"view": VIEW, "ids": ids, "colours": {n: "#ff0000" for n in ids},
            "neuron_dir": "/m/neurons",
            "roi_dir": "/m/rois" if (spec["neuropils"] or head) else None,   # as gather does
            "region_colours": {r: "#00ff00" for r in regions},
            "head": {"dir": "/m/head", "roi": "headroi"} if head else None}


def at(animation):
    """``[(time, command, args)]`` with advanceTime folded into the time."""
    now, out = 0.0, []
    for cmd, args in animation:
        if cmd == "advanceTime":
            now = round(now + args["by"], 4)
        else:
            out.append((now, cmd, args))
    return out, now


def test_orbit_replace_movie(replace_spec):
    doc, labels_doc, info = compose_mod.compose(replace_spec, scene(replace_spec,
                                                                    regions=["LA(L)"]))
    assert doc["neurons"] == {"source": "/m/neurons", "A": [1], "B": [1], "C": [1]}
    assert doc["rois"]["C"] == ["LA_L"] and doc["rois"]["c_LA_L"] == ["LA_L"]
    events, end = at(doc["animation"])
    # Content ends at 15 (C's beat at 11 + fade + hold + fade); the tail is 1.5.
    assert end == info["seconds"] == 16.5 and info["frames"] == round(16.5 * 30)
    assert info["smooth"] is None
    fades = [(t, a["meshes"], a["endingAlpha"]) for t, c, a in events if c == "fade"]
    assert fades == [(1.0, "neurons.A", 1.0), (6.0, "neurons.B", 1.0),
                     (9.0, "neurons.A", 0), (9.0, "neurons.B", 0),
                     (11.0, "neurons.C", 1.0), (11.0, "rois.C", 0.08),
                     (14.0, "neurons.C", 0), (14.0, "rois.C", 0)]
    orbit = next((t, a) for t, c, a in events if c == "orbitCamera")
    assert orbit == (1.0, {"localAxis": "y", "endingRelativeAngle": 360.0, "duration": 14.0})
    pose = next(a for t, c, a in events if c == "poseCamera")
    assert pose["distance"] == 1000 * compose_mod.DISTANCE_SCALE
    colour = next(a for t, c, a in events if c == "setValue" and "color" in a
                  and a["meshes"] == "neurons.A")
    assert colour["color"] == linear_hex("#ff0000")
    assert labels_doc["fps"] == 30


def test_follow_camera_frames_a_window_then_pulls_back_and_orbits(accumulate_spec):
    doc, _, info = compose_mod.compose(accumulate_spec, scene(accumulate_spec))
    events, end = at(doc["animation"])
    frames = [(t, a["bound"]) for t, c, a in events if c == "frameCamera"]
    assert frames == [(0.5, "neurons.A"), (2.5, "neurons.A + neurons.B"),
                      (4.5, "neurons.B + neurons.C")]
    settle = 6.5                                                      # a beat after the last
    assert [(t, c) for t, c, a in events if c in ("poseCamera", "orbitCamera")][1:] == [
        (settle, "poseCamera"), (settle + 4.0, "orbitCamera")]
    assert end == settle + 4.0 + 12.0 + 1.5
    assert info["smooth"] == (round(0.5 * 30) + 1, round((settle + 4.0) * 30) + 1)
    assert not [a for t, c, a in events if c == "fade" and a["endingAlpha"] == 0]


def test_head_shell_and_intro():
    spec = make_spec([{"name": "A", "ids": [1]}],
                     head={"library": "lib/v1", "id": 95, "intro_seconds": 2.0})
    doc, _, _ = compose_mod.compose(spec, scene(spec, head=True))
    assert doc["rois"]["head"] == ["headroi"]
    assert doc["neurons"]["source"] == ["/m/neurons", "/m/head"]
    assert doc["neurons"]["head"] == {"ids": [95], "sourceIndex": 1}
    events, _ = at(doc["animation"])
    assert next(a for t, c, a in events if c == "frameCamera")["bound"] == "rois.head"
    intro = [(t, a["meshes"]) for t, c, a in events if c == "fade" and "head" in a["meshes"]]
    assert intro == [(1.0, "neurons.head"), (1.0, "rois.head")]


def test_beats_with_nothing_resolved_are_dropped(replace_spec, capsys):
    s = scene(replace_spec)
    del s["ids"]["C"]
    doc, labels_doc, info = compose_mod.compose(replace_spec, s)
    assert "C" not in doc["neurons"]
    assert "beats with nothing to show, dropped: three" in capsys.readouterr().err
    assert info["seconds"] == schedule.beat_end(replace_spec, 1) + 1.5


def test_timeline_keeps_order_within_a_moment():
    anim = compose_mod.timeline([(2.0, ["b"]), (0.0, ["a"]), (2.0, ["c"])], end=3.0)
    assert anim == [["a"], ["advanceTime", {"by": 2.0}], ["b"], ["c"],
                    ["advanceTime", {"by": 1.0}]]


# -- labels ---------------------------------------------------------------------------- #

def label_rows(doc):
    events, _ = at(doc["animation"])
    return [(t, a["text"], a["position"], a["duration"]) for t, c, a in events]


def test_replace_labels_cross_fade_with_their_beats(replace_spec):
    spec = dict(replace_spec, labels=dict(replace_spec["labels"], legend_header="Types:"))
    ids = {"A": ["1"], "B": ["2"], "C": ["3"]}
    doc = labels.compose(spec, ids, {n: "#ff0000" for n in ids}, end=16.5)
    rows = label_rows(doc)
    # A legend sits on the bottom margin, so a row's height depends on the beat's row count.
    assert (1.0, "one", "top", 4.0) in rows
    assert (1.0, "Types:", [0.04, 0.125], 4.0) in rows
    assert (1.0, "A", [0.04, 0.06], 4.0) in rows
    assert (6.0, "A", [0.04, 0.125], 4.0) in rows
    assert (6.0, "B", [0.04, 0.06], 4.0) in rows
    assert (11.0, "C", [0.04, 0.06], 4.0) in rows
    assert all(a["fade"] == 1.0 for _, c, a in at(doc["animation"])[0])


def test_identical_overlapping_labels_merge():
    out = labels.commands([
        labels._label(0.0, 4.0, "A", [0, 0], 0.1, "#fff", 1.0),
        labels._label(3.0, 9.0, "A", [0, 0], 0.1, "#fff", 1.0),
        labels._label(3.0, 9.0, "A", [0, 1], 0.1, "#fff", 1.0),
    ])
    assert out == [
        ["label", {"text": "A", "position": [0, 0], "size": 0.1, "color": "#fff",
                   "duration": 9.0, "fade": 1.0}],
        ["advanceTime", {"by": 3.0}],
        ["label", {"text": "A", "position": [0, 1], "size": 0.1, "color": "#fff",
                   "duration": 6.0, "fade": 1.0}],
    ]


def test_accumulate_legend_grows_and_stays(accumulate_spec):
    ids = {"A": ["1"], "B": ["2"], "C": ["3"]}
    doc = labels.compose(accumulate_spec, ids, {n: "#00ff00" for n in ids}, end=20.0)
    rows = label_rows(doc)
    gone = 20.0 + 1.5
    assert (0.5, "first", "top", 2.5 + 1.5 - 0.5) in rows             # until beat 2 is in
    assert (4.5, "third", "top", gone - 4.5) in rows
    legend = [(t, text, pos) for t, text, pos, _ in rows if pos != "top"]
    assert legend == [(0.5, "A", [0.04, 0.19]), (2.5, "B", [0.04, 0.125]),
                      (4.5, "C", [0.04, 0.06])]
    colours = {a["color"] for _, c, a in at(doc["animation"])[0] if a["text"] in ids}
    assert colours == {linear_hex("#00ff00")}


@pytest.mark.parametrize("off", [{"titles": False}, {"legend": False}])
def test_titles_and_legend_can_be_turned_off(replace_spec, off):
    spec = dict(replace_spec, labels=dict(replace_spec["labels"], **off))
    ids = {"A": ["1"], "B": ["2"], "C": ["3"]}
    texts = {r[1] for r in label_rows(labels.compose(spec, ids, dict.fromkeys(ids, "#f00000"),
                                                     16.5))}
    assert ("one" in texts) == off.get("titles", True)
    assert ("A" in texts) == off.get("legend", True)


def test_a_looping_orbit_turns_through_the_whole_movie(replace_spec):
    spec = dict(replace_spec, camera=dict(replace_spec["camera"], loop=True))
    doc, _, info = compose_mod.compose(spec, scene(spec))
    events, end = at(doc["animation"])
    orbit = next((t, a) for t, c, a in events if c == "orbitCamera")
    # From the first frame to one past the last, so the frame after the last is the first.
    assert orbit == (0.0, {"localAxis": "y", "endingRelativeAngle": 360.0, "duration": end})
    assert end == info["seconds"] == 16.5 and info["orbit"] and info["loop"]
    info = compose_mod.compose(replace_spec, scene(replace_spec))[2]
    assert info["orbit"] and not info["loop"]
    still = dict(spec, camera=dict(spec["camera"], orbit_degrees=0))
    info = compose_mod.compose(still, scene(still))[2]
    assert not info["orbit"] and not info["loop"]


def test_a_repeated_sequence_runs_its_beats_again_on_the_same_clock():
    spec = make_spec([{"title": "one", "name": "A", "ids": [1]},
                      {"title": "two", "name": "B", "ids": [2]}],
                     timing={"start": 1.0, "every": 5.0, "fade": 1.0, "hold": 2.0,
                             "repeat": 2})
    doc, _, info = compose_mod.compose(spec, scene(spec))
    events, _ = at(doc["animation"])
    fade_ins = [(t, a["meshes"]) for t, c, a in events if c == "fade" and a["endingAlpha"]]
    # A pass is B's beat (6) + fade + hold + fade + tail = 11.5, and the second starts there,
    # its first beat a lead-in (1) later, as the first pass's did.
    assert fade_ins == [(1.0, "neurons.A"), (6.0, "neurons.B"),
                        (12.5, "neurons.A"), (17.5, "neurons.B")]
    assert info["seconds"] == 2 * 11.5


# Two real neuroglancer orientations: an upright frontal view, and one tilted ~23 degrees
# from it that a follow camera opens on.
FRONT = [-0.12595, 0.67477, 0.72233, 0.08407]
TILTED = [0.10992, 0.44167, 0.85068, 0.26305]


def test_quaternions_round_trip():
    rng = np.random.default_rng(0)
    for q in rng.normal(size=(50, 4)):
        m = compose_mod._matrix(q)
        assert np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1)
        assert np.allclose(compose_mod._matrix(compose_mod._quaternion(m)), m)


def test_level_orientation_keeps_the_direction_around_the_vertical():
    m = compose_mod._matrix(compose_mod.level_orientation(TILTED, FRONT))
    front, tilted = compose_mod._matrix(FRONT), compose_mod._matrix(TILTED)
    up = front[:, 1]
    assert np.isclose(m[:, 1] @ up, 1)                 # upright in the reference
    assert np.isclose(m[:, 2] @ up, 0)                 # looking level
    flat = tilted[:, 2] - (tilted[:, 2] @ up) * up     # the same way round the vertical
    assert np.allclose(m[:, 2], flat / np.linalg.norm(flat))
    assert np.allclose(compose_mod._matrix(compose_mod.level_orientation(FRONT, FRONT)), front)


def test_level_orientation_refuses_a_view_along_the_vertical():
    front = compose_mod._matrix(FRONT)
    down = compose_mod._quaternion(np.column_stack([front[:, 0], front[:, 2], -front[:, 1]]))
    with pytest.raises(ValueError, match="straight along"):
        compose_mod.level_orientation(down, FRONT)


def test_a_levelled_follow_camera_pulls_back_upright(accumulate_spec):
    view = dict(VIEW, projectionOrientation=TILTED)
    sc = dict(scene(accumulate_spec), view=view, level=FRONT)
    events, _ = at(compose_mod.compose(accumulate_spec, sc)[0]["animation"])
    opening, back = [a for t, c, a in events if c == "poseCamera"]
    assert opening["orientation"] == TILTED
    assert back["orientation"] == compose_mod.level_orientation(TILTED, FRONT)
    assert back["target"] == opening["target"] and back["distance"] == opening["distance"]
    events, _ = at(compose_mod.compose(accumulate_spec, dict(sc, level=None))[0]["animation"])
    assert [a for t, c, a in events if c == "poseCamera"][1]["orientation"] == TILTED
