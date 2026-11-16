"""The neuVid input and the labels, from a spec and a scene; no store, no Blender."""

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
    assert orbit == (1.0, {"localAxis": "y", "endingRelativeAngle": -360.0, "duration": 14.0})
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
    settle = schedule.beat_start(accumulate_spec, 3)                  # 6.5
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
