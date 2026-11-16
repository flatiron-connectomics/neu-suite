"""The neuVid input and the labels for a spec. Pure: everything read from a store or the
view arrives in `scene` (see `build.gather`), so this runs in milliseconds and can be rerun
from a build's own records.

neuVid's units are 8 nm voxels, its camera distances are scaled to match neuroglancer's, and
its commands run on a clock that only moves forward, so commands are collected as
``(time, command)`` and emitted in time order with `advanceTime` between them.
"""

import sys

from neu_movie import labels as labels_mod
from neu_movie import schedule
from neu_movie.meshes import safe_name
from neu_movie.palette import linear_hex, mute

#: neuVid's camera sits at 1.3 x `projectionScale` with a 50 degree HORIZONTAL field of view,
#: which frames ~1.45x tighter than neuroglancer at 16:9 (measured on the head's width in the
#: opening frame of both renders). Applied to every `poseCamera` distance.
DISTANCE_SCALE = 1.45


def compose(spec, scene):
    """``(neuvid doc, labels doc, info)``. `info` has the duration, frame count and, for a
    follow camera, the frame range `smooth_camera` joins into one path."""
    t, cam, head = spec["timing"], spec["camera"], spec["head"]
    view_state = scene["view"]
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
        regions = spec["sets"][name]["regions"]
        if regions:
            rois[g] = [safe_name(r) for r in regions]
            parts.append(f"rois.{g}")
        expr[name] = " + ".join(parts)

    # -- the opening -------------------------------------------------------------- #
    events = []
    view = {"target": view_state["position"],
            "orientation": view_state.get("projectionOrientation", [0, 0, 0, 1]),
            "distance": view_state.get("projectionScale", 12000) * DISTANCE_SCALE}
    events.append((0.0, ["poseCamera", dict(view)]))
    if cam["style"] == "orbit":
        bound = "rois.head" if head else " + ".join(expr.values())
        if not bound:
            raise ValueError("an orbit camera frames the head or the sets, and there are neither")
        events.append((0.0, ["frameCamera", {"bound": bound, "scale": cam["scale"],
                                             "duration": 0}]))
    for name, e in expr.items():
        events.append((0.0, ["setValue", {"meshes": e, "alpha": 0}]))
        events.append((0.0, ["setValue", {"meshes": f"neurons.{safe_name(name)}",
                                          "color": linear_hex(colours[name])}]))
    if spec["neuropils"]:
        for region, colour in scene["region_colours"].items():
            rois[f"c_{safe_name(region)}"] = [safe_name(region)]
            events.append((0.0, ["setValue", {
                "meshes": f"rois.c_{safe_name(region)}",
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
        if spec["sets"][name]["regions"]:
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
    if cam["style"] == "orbit":
        if cam["orbit_degrees"]:
            events.append((cam["orbit_at"], ["orbitCamera", {
                "localAxis": "y", "endingRelativeAngle": -cam["orbit_degrees"],
                "duration": round(content_end - cam["orbit_at"], 3)}]))
        end = content_end + t["tail"]
    else:
        # Each beat frames its sets with the previous ones in the window. Moves run back to
        # back; neuVid keys each as its own ease-in/ease-out, and smooth_camera (run after
        # addAnimation) joins them into one continuous path.
        move = round(t["every"] if cam["move_seconds"] is None else cam["move_seconds"], 3)
        beat_expr = [" + ".join(expr[n] for n in b["sets"] if n in expr)
                     for b in spec["beats"]]
        for i in range(len(spec["beats"])):
            window = " + ".join(beat_expr[max(0, i + 1 - cam["window"]):i + 1])
            events.append((schedule.beat_start(spec, i), ["frameCamera", {
                "bound": window, "scale": cam["scale"], "duration": move}]))
        settle = schedule.beat_start(spec, len(spec["beats"]))
        events.append((settle, ["poseCamera", dict(view, duration=cam["pullback_seconds"])]))
        orbit_at = settle + cam["pullback_seconds"]
        events.append((orbit_at, ["orbitCamera", {
            "localAxis": "y", "endingRelativeAngle": -cam["orbit_degrees"],
            "duration": cam["orbit_seconds"]}]))
        end = orbit_at + cam["orbit_seconds"] + t["tail"]
        # neuVid's frame numbering: frame(t) = round(t * fps) + 1.
        smooth = (round(t["start"] * spec["fps"]) + 1, round(orbit_at * spec["fps"]) + 1)

    doc = {"fps": spec["fps"], "neurons": neurons, "rois": rois,
           "animation": timeline(events, end)}
    info = {"seconds": end, "frames": round(end * spec["fps"]), "smooth": smooth}
    return doc, labels_mod.compose(spec, ids, colours, end), info


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
