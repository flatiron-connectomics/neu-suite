"""Sets of cells: their body ids, from a list or a query, and their colours."""

import sys

from neu_movie import palette, schedule


def resolve(spec, volume, *, properties=None, limit=None, log=sys.stderr):
    """``{set: [body id, ...]}``, every set's ids or query run once.

    A query keeps only bodies whose mesh is in the store (a body without one cannot be
    drawn). `exclude` drops bodies after the query and before its `limit`, so a limited
    test set never holds an excluded body. `limit` (for quick tests) caps every set.
    """
    from neu_glance.sources import segments_with_meshes, select_segments
    from neu_vol.segment_properties import read_segment_properties
    out = {}
    for name, s in spec["sets"].items():
        exclude = set(s["exclude"])
        if s["ids"] is not None:
            found = list(s["ids"])
            missing = exclude - set(found)
            cap = limit
        else:
            if properties is None:
                properties = read_segment_properties(volume)
            query = dict(s["query"])
            cap = query.pop("limit", None)
            if limit is not None:
                cap = min(limit, cap if cap is not None else limit)
            found = select_segments(volume, properties=properties, **query)
            missing = exclude - set(found)
            if cap is not None:
                # The largest `cap` once excluded bodies are gone, not before.
                found = select_segments(volume, properties=properties, **query,
                                        limit=cap + len(exclude))
        for b in sorted(missing, key=int):
            print(f"  ! {name!r}: exclude {b} is not in the set", file=log)
        found = [b for b in found if b not in exclude][:cap]
        lost = 0
        if s["query"] is not None and found:
            drawable = segments_with_meshes(volume, found)
            lost = len(found) - len(drawable)
            found = drawable
        out[name] = found
        print(f"  {name:22s} {len(found):5d} bodies"
              + (f"   ({lost} with no mesh, dropped)" if lost else "")
              + ("" if found else "   ...no matches, skipped"), file=log)
    return {name: ids for name, ids in out.items() if ids}


def colours(spec):
    """``{set: "#rrggbb"}``: each set's `color`, else the next palette colour in order of
    first appearance. Repeats come round only after the whole palette."""
    out, k = {}, 0
    for beat in spec["beats"]:
        for name in beat["sets"]:
            if name in out:
                continue
            if spec["sets"][name]["color"]:
                out[name] = spec["sets"][name]["color"]
            else:
                out[name] = palette.SETS[k % len(palette.SETS)]
                k += 1
    return out


def clashes(spec, colours, ids=None):
    """Pairs of sets on screen at the same time, fades included, that share a colour."""
    fade = spec["timing"]["fade"]
    end = schedule.content_end(spec) + 1.0
    spans = {name: [(s, None if e is None else e + fade) for s, e in iv]
             for name, iv in schedule.intervals(spec).items()
             if ids is None or name in ids}
    names = list(spans)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if colours[a] == colours[b] and schedule.overlapping(spans[a], spans[b], end):
                out.append((a, b, colours[a]))
    return out
