"""A neuroglancer link per beat, to inspect what a beat shows before rendering anything: the
view with the beat's sets as coloured layers in place of the segmentation they came from."""

import copy

from neu_movie.palette import normalize


def beat_states(spec, view, ids, colours):
    """``[(beat title or set names, neuroglancer state), ...]``."""
    from neu_glance.state import subset_layers
    out, seen = [], []
    for beat in spec["beats"]:
        shown = {n: ids[n] for n in beat["sets"] if n in ids}
        if not shown or beat in seen:            # a repeated pass shows the same beats
            continue
        seen.append(beat)
        state, _, _ = subset_layers(copy.deepcopy(view), spec["neurons"]["layer"], shown,
                                    colors={n: normalize(colours[n]) for n in shown})
        out.append((beat["title"] or ", ".join(shown), state))
    return out


def beat_links(spec, view, ids, colours):
    from neu_glance.state import state_url
    return [(title, state_url(state)) for title, state in beat_states(spec, view, ids, colours)]
