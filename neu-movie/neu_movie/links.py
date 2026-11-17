"""Neuroglancer links to inspect a movie before rendering: one of everything the movie shows,
then one per beat of what is on screen during it, and nothing else. Each set is a layer in
its colour, the neuropils wear their colours muted as the movie draws them, and the head is
a faint shell. The camera is the movie's opening one; for an orbit, a beat's link is turned
as far as the movie has turned by the middle of the beat (about the view's position, where
the movie turns about what it frames).
"""

from neu_movie import compose, schedule
from neu_movie.palette import mute

#: Opacities in the viewer, where a mesh has no rim lighting: the movie's own (0.08, 0.035)
#: would all but hide them.
NEUROPIL_ALPHA = 0.25
HEAD_ALPHA = 0.1


def _source(volume):
    return volume if volume.startswith("precomputed://") else f"precomputed://{volume}"


def _layers(spec, world, names):
    """The layers showing sets `names`, their neuropils and the head."""
    layers = [{"type": "segmentation", "name": n, "source": _source(world["volume"]),
               "segments": [str(b) for b in world["ids"][n]],
               "segmentDefaultColor": world["colours"][n]} for n in names]
    regions = [r for r in dict.fromkeys(r for n in names for r in spec["sets"][n]["regions"])
               if r in world["regions"]]
    if regions:
        amount = spec["neuropils"]["mute"]
        layers.append({
            "type": "segmentation", "name": "neuropils", "source": _source(world["roi_volume"]),
            "segments": [world["regions"][r]["id"] for r in regions],
            "segmentColors": {world["regions"][r]["id"]: mute(world["regions"][r]["color"],
                                                              amount) for r in regions},
            "objectAlpha": NEUROPIL_ALPHA})
    if spec["head"]:
        layers.append({"type": "segmentation", "name": "head",
                       "source": _source(world["head_volume"]),
                       "segments": [str(spec["head"]["id"])],
                       "segmentDefaultColor": spec["head"]["color"], "objectAlpha": HEAD_ALPHA})
    return layers


def _state(spec, world, names, orientation):
    from neu_glance.state import build_state
    unit, view = world["unit"], spec["view"]
    state, _ = build_state(_layers(spec, world, names), voxel_size_zyx=(unit,) * 3, units="nm",
                           position_zyx=[p / unit for p in view["position"]][::-1],
                           projection_scale=view["height"] / unit,
                           projection_orientation=orientation, layout="3d", show_slices=False)
    return state


def states(spec, world):
    """``[(title, neuroglancer state), ...]``: everything, then each beat."""
    ids = world["ids"]
    # As compose drops them, so the beats keep the movie's clock.
    beats = [b for b in spec["beats"] if any(n in ids for n in b["sets"])]
    spec = dict(spec, beats=beats)
    opening = compose.opening_orientation(spec)
    out = [("everything", _state(spec, world, [n for n in spec["sets"] if n in ids], opening))]
    seen = []
    for i, beat in enumerate(beats):
        if beat in seen:                       # a repeated pass shows the same beats
            continue
        seen.append(beat)
        if spec["mode"] == "replace":
            names = [n for n in beat["sets"] if n in ids]
        else:
            names = [n for b in beats[:i + 1] for n in b["sets"] if n in ids]
        start = schedule.beat_start(spec, i)
        end = (schedule.beat_start(spec, i + 1) if i + 1 < len(beats)
               else schedule.content_end(spec))
        degrees = compose.orbit_degrees_at(spec, (start + end) / 2)
        orientation = compose.turned(opening, degrees) if degrees else opening
        title = beat["title"] or ", ".join(n for n in beat["sets"] if n in ids)
        out.append((title, _state(spec, world, names, orientation)))
    return out


def links(spec, world):
    from neu_glance.state import state_url
    return [(title, state_url(state)) for title, state in states(spec, world)]
