"""The neuVid input and the labels for a spec. Pure: everything read from a store arrives
in `scene` (see `build.gather`), so this runs in milliseconds and can be rerun from a
build's own records.

neuVid's unit is the neuron volume's voxel (`scene["unit"]` nm), its camera distances are
scaled to match neuroglancer's, and
its commands run on a clock that only moves forward, so commands are collected as
``(time, command)`` and emitted in time order with `advanceTime` between them.
"""

import sys

import numpy as np

from neu_movie import labels as labels_mod
from neu_movie import schedule
from neu_movie.meshes import safe_name
from neu_movie.palette import linear_hex, mute

#: neuVid's camera sits at 1.3 x the view's height with a 50 degree HORIZONTAL field of
#: view, which frames ~1.45x tighter than neuroglancer at 16:9 (measured on the head's width
#: in the opening frame of both renders). Applied to every `poseCamera` distance.
DISTANCE_SCALE = 1.45


def compose(spec, scene):
    """``(neuvid doc, labels doc, info)``. `info` has the duration, frame count, for a
    follow camera the frame range `smooth_camera` joins into one path, whether there is
    an orbit and whether it loops (both for `blender/orbit.py`), and the fill light's
    strength (for `blender/fill_light.py`)."""
    t, cam, head = spec["timing"], spec["camera"], spec["head"]
    ids, colours = scene["ids"], scene["colours"]
    shown = [b for b in spec["beats"] if any(n in ids for n in b["sets"])]
    if len(shown) != len(spec["beats"]):
        empty = [b["title"] or ", ".join(b["sets"]) for b in spec["beats"] if b not in shown]
        print(f"  beats with nothing to show, dropped: {'; '.join(empty)}", file=sys.stderr)
        spec = dict(spec, beats=shown)

    # -- what neuVid loads ------------------------------------------------------ #
    neurons = {"source": scene["neuron_dir"]}
    rois = {}
    if scene["roi_dir"]:
        rois["source"] = scene["roi_dir"]
        rois["exponents"] = {}
    if spec["neuropils"]:
        rois["exponents"]["*"] = spec["neuropils"]["exponent"]
    intro = bool(head and head["intro_seconds"] > 0)
    if head:
        rois["head"] = [scene["head"]["roi"]]
        rois["exponents"]["head"] = head["exponent"]
        if intro:
            # The lit head is a second neuron source, so its id cannot collide with a body's.
            neurons["source"] = [scene["neuron_dir"], scene["head"]["dir"]]
            neurons["head"] = {"ids": [int(head["id"])], "sourceIndex": 1}
    expr = {}                                         # set -> its mesh expression
    for name in spec["sets"]:
        if name not in ids:
            continue
        g = safe_name(name)
        neurons[g] = [int(b) for b in ids[name]]
        parts = [f"neurons.{g}"]
        shown = spec["sets"][name]["neuropils"]
        if shown:
            rois[g] = [safe_name(r) for r in shown]
            parts.append(f"rois.{g}")
        expr[name] = " + ".join(parts)

    # -- the opening -------------------------------------------------------------- #
    events = []
    view = {"target": [v / scene["unit"] for v in spec["view"]["position"]],
            "orientation": opening_orientation(spec),
            "distance": spec["view"]["height"] / scene["unit"] * DISTANCE_SCALE}
    level = cam["up"]
    events.append((0.0, ["poseCamera", dict(view)]))
    if cam["style"] == "orbit":
        bound = ("rois.head" if head and cam["frame"] == "head"
                 else " + ".join(expr.values()))
        if not bound:
            raise ValueError("an orbit camera frames the head or the sets, and there are neither")
        events.append((0.0, ["frameCamera", {"bound": bound, "scale": cam["scale"],
                                             "duration": 0}]))
    for name, e in expr.items():
        events.append((0.0, ["setValue", {"meshes": e, "alpha": 0}]))
        events.append((0.0, ["setValue", {"meshes": f"neurons.{safe_name(name)}",
                                          "color": linear_hex(colours[name])}]))
    if spec["neuropils"]:
        for neuropil, colour in scene["neuropil_colours"].items():
            rois[f"c_{safe_name(neuropil)}"] = [safe_name(neuropil)]
            events.append((0.0, ["setValue", {
                "meshes": f"rois.c_{safe_name(neuropil)}",
                "color": linear_hex(mute(colour, spec["neuropils"]["mute"]))}]))
    if head:
        # A faint shell throughout, or, with an intro, the solid lit head cross-fading into it.
        events.append((0.0, ["setValue", {"meshes": "rois.head", "color": head["color"]}]))
        if intro:
            events.append((0.0, ["setValue", {"meshes": "neurons.head",
                                              "color": head["lit_color"]}]))
            events.append((0.0, ["setValue", {"meshes": "rois.head", "alpha": 0}]))
            for mesh, a, b in (("neurons.head", 1, 0), ("rois.head", 0, head["alpha"])):
                events.append((head["intro_at"], ["fade", {
                    "meshes": mesh, "startingAlpha": a, "endingAlpha": b,
                    "duration": head["intro_seconds"]}]))
        else:
            events.append((0.0, ["setValue", {"meshes": "rois.head", "alpha": head["alpha"]}]))

    # -- the sets ------------------------------------------------------------------ #
    np_alpha = spec["neuropils"]["alpha"] if spec["neuropils"] else 0
    for name, spans in schedule.intervals(spec).items():
        if name not in expr:
            continue
        g = safe_name(name)
        meshes = [(f"neurons.{g}", t["alpha"])]
        if spec["sets"][name]["neuropils"]:
            meshes.append((f"rois.{g}", np_alpha))
        for start, out in spans:
            for mesh, alpha in meshes:
                events.append((start, ["fade", {"meshes": mesh, "startingAlpha": 0,
                                                "endingAlpha": alpha, "duration": t["fade"]}]))
            if out is not None:
                for mesh, alpha in meshes:
                    events.append((out, ["fade", {"meshes": mesh, "startingAlpha": alpha,
                                                  "endingAlpha": 0, "duration": t["fade"]}]))

    # -- the camera ---------------------------------------------------------------- #
    content_end = schedule.content_end(spec)
    smooth = None
    beat_expr = [" + ".join(expr[n] for n in b["sets"] if n in expr) for b in spec["beats"]]
    if cam["style"] == "orbit":
        # A loop turns over the whole movie: keyed at frames 1 and frames + 1, so the frame
        # after the last is the first again.
        orbit_at, orbit_end = orbit_span(spec)
        end = max(content_end, orbit_end) + (0 if cam["loop"] else t["tail"])
        if cam["orbit_degrees"]:
            events.append((orbit_at, ["orbitCamera", {
                "localAxis": "y", "endingRelativeAngle": cam["orbit_degrees"],
                "duration": round(orbit_end - orbit_at, 3)}]))
    elif cam["style"] == "glide":
        # One move, eased in and out, from the first beat's framing to everything's: it
        # starts as the first beat does and ends as the last has faded in. The orientation
        # never changes, so the orbit after it turns about the opening view's vertical.
        last = len(spec["beats"]) - 1
        glide_at, orbit_at = schedule.beat_start(spec, 0), schedule.beat_end(spec, last)
        events.append((0.0, ["frameCamera", {"bound": beat_expr[0], "scale": cam["scale"],
                                             "duration": 0}]))
        events.append((glide_at, ["frameCamera", {
            "bound": " + ".join(beat_expr), "scale": cam["scale"],
            "duration": round(orbit_at - glide_at, 3)}]))
        events.append((orbit_at, ["orbitCamera", {
            "localAxis": "y", "endingRelativeAngle": cam["orbit_degrees"],
            "duration": cam["orbit_seconds"]}]))
        end = orbit_at + cam["orbit_seconds"] + t["tail"]
    else:
        # Each beat frames its sets with the previous ones in the window. Moves run back to
        # back; neuVid keys each as its own ease-in/ease-out, and smooth_camera (run after
        # addAnimation) joins them into one continuous path.
        move = round(t["every"] if cam["move_seconds"] is None else cam["move_seconds"], 3)
        for i in range(len(spec["beats"])):
            window = " + ".join(beat_expr[max(0, i + 1 - cam["window"]):i + 1])
            events.append((schedule.beat_start(spec, i), ["frameCamera", {
                "bound": window, "scale": cam["scale"], "duration": move}]))
        settle = schedule.beat_start(spec, len(spec["beats"]) - 1) + t["every"]
        back = dict(view)
        if level:
            back["orientation"] = level_orientation(view["orientation"], level)
        events.append((settle, ["poseCamera", dict(back, duration=cam["pullback_seconds"])]))
        orbit_at = settle + cam["pullback_seconds"]
        events.append((orbit_at, ["orbitCamera", {
            "localAxis": "y", "endingRelativeAngle": cam["orbit_degrees"],
            "duration": cam["orbit_seconds"]}]))
        end = orbit_at + cam["orbit_seconds"] + t["tail"]
        # neuVid's frame numbering: frame(t) = round(t * fps) + 1.
        smooth = (round(t["start"] * spec["fps"]) + 1, round(orbit_at * spec["fps"]) + 1)

    doc = {"fps": spec["fps"], "neurons": neurons, "rois": rois,
           "animation": timeline(events, end)}
    orbit = any(cmd[0] == "orbitCamera" for _, cmd in events)
    info = {"seconds": end, "frames": round(end * spec["fps"]), "smooth": smooth,
            "orbit": orbit, "loop": orbit and cam["style"] == "orbit" and cam["loop"],
            "fill": spec["light"]["fill"]}
    return doc, labels_mod.compose(spec, ids, colours, end, scene["neuropil_colours"]), info


def opening_orientation(spec):
    """The movie's first orientation: the view's, levelled about `[camera] up` for an orbit
    or a glide (a follow camera levels only its return)."""
    q, up = spec["view"]["orientation"], spec["camera"]["up"]
    return level_orientation(q, up) if up and spec["camera"]["style"] in ("orbit", "glide") \
        else q


def orbit_degrees_at(spec, when):
    """How far an orbit camera has turned at `when` (seconds), or None for another camera.
    A loop turns at constant speed over the whole movie; otherwise neuVid eases in and out,
    approximated by a smoothstep."""
    cam = spec["camera"]
    if cam["style"] != "orbit":
        return None
    start, stop = orbit_span(spec)
    if cam["loop"]:
        return cam["orbit_degrees"] * when / stop
    f = min(max((when - start) / (stop - start), 0.0), 1.0) if stop > start else 1.0
    return cam["orbit_degrees"] * f * f * (3 - 2 * f)


def orbit_span(spec):
    """``(start, end)`` of an orbit camera's turn, in seconds: a loop's is the whole movie;
    otherwise from `orbit_at` for `orbit_seconds`, or until the last fade ends."""
    cam, t = spec["camera"], spec["timing"]
    content_end = schedule.content_end(spec)
    if cam["loop"]:
        return 0.0, content_end + t["tail"]
    if cam["orbit_seconds"] is not None:
        return cam["orbit_at"], cam["orbit_at"] + cam["orbit_seconds"]
    return cam["orbit_at"], content_end


def turned(q, degrees):
    """Orientation `q` after the camera has orbited `degrees` about its own vertical, as
    neuVid's orbitCamera turns it (positive: the camera moves to its right). neuroglancer's
    view y points down the screen, so the camera's up is minus its y column."""
    m = _matrix(q)
    axis = -m[:, 1]
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    return _quaternion(rotation @ m)


def view_from_state(state):
    """A `[view]` (position and height in nm, orientation) and the `up` of a neuroglancer
    state, for pasting into a spec."""
    dims = state["dimensions"]
    nm = [float(dims[a][0]) * {"m": 1e9, "nm": 1.0}[dims[a][1]] for a in "xyz"]
    if len(set(nm)) != 1:
        raise ValueError(f"anisotropic dimensions {dims}: the zoom has no single unit")
    q = state.get("projectionOrientation", [0, 0, 0, 1])
    up = -_matrix(q)[:, 1]
    return {"position": [round(p * n, 1) for p, n in zip(state["position"], nm)],
            "orientation": [round(float(v), 6) for v in q],
            "height": round(float(state.get("projectionScale", 1)) * nm[0], 1),
            "up": [round(float(v), 6) + 0.0 for v in up]}      # + 0.0: no "-0"


def _matrix(q):
    """The rotation matrix of a neuroglancer quaternion ``[x, y, z, w]``; its columns are
    the view's own axes (x right, y the screen's vertical, z the view direction)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quaternion(m):
    """``[x, y, z, w]`` for a rotation matrix, from whichever of the four standard forms
    has the largest (best-conditioned) leading term."""
    t = np.trace(m)
    forms = [
        (t, [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], 1 + t]),
        (m[0, 0], [1 + 2 * m[0, 0] - t, m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]]),
        (m[1, 1], [m[0, 1] + m[1, 0], 1 + 2 * m[1, 1] - t, m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]]),
        (m[2, 2], [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], 1 + 2 * m[2, 2] - t, m[1, 0] - m[0, 1]])]
    q = np.array(max(forms, key=lambda f: f[0])[1])
    return [float(v) for v in q / np.linalg.norm(q)]


def level_orientation(q, up):
    """`q` turned upright about the direction `up`: the same view direction swung into the
    plane square to `up`, with `up` as its vertical. An orbit about the result's vertical
    (neuVid's local y) then turns about `up`."""
    r = _matrix(q)
    up = np.asarray(up, float) / np.linalg.norm(up)
    if r[:, 1] @ up < 0:
        up = -up
    d = r[:, 2] - (r[:, 2] @ up) * up
    if np.linalg.norm(d) < 1e-6:
        raise ValueError("the view looks straight along `up`")
    d /= np.linalg.norm(d)
    return _quaternion(np.column_stack([np.cross(up, d), up, d]))


def timeline(events, end):
    """neuVid commands in time order, `advanceTime` between. The sort is stable, so commands
    at one moment keep the order they were added in."""
    anim, now = [], 0.0
    for when, cmd in sorted(events, key=lambda e: e[0]) + [(end, None)]:
        if when > now + 1e-9:
            anim.append(["advanceTime", {"by": round(when - now, 4)}])
            now = when
        if cmd:
            anim.append(cmd)
    return anim
