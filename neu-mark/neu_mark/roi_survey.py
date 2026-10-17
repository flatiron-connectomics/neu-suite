"""What a node's DVID ROIs are, measured: sizes, the containment tree, overlaps.

The input to a :mod:`neu_mark.roi_layers` spec, and the check when a node's ROIs change.
Every ROI's **parent** is the smallest larger ROI holding more than half of it, which
recovers the tree from the blocks alone; ``inside`` is the share within the parent, below
``contained`` when the child sticks out. For each ROI with children, ``coverage`` is the
share of it that some child covers, and ``outside`` the share of the children's union lying
outside it. **Siblings** are overlapping ROIs neither of which is the other's ancestor;
past ``max_overlap`` of the smaller, ``roi-layers`` would refuse them in one layer.

Measurement works on DVID's block ranges and needs no DVID; :func:`fetch_all` fetches them.
"""

from __future__ import annotations

import logging
import time
from typing import Mapping, Sequence

import numpy as np

from .roi_layers import (DEFAULT_MAX_OVERLAP, Region, _blocks, _overlap, open_node,
                         region_from_ranges, union)

logger = logging.getLogger(__name__)

#: A child at least this far inside its parent is not reported as sticking out.
DEFAULT_CONTAINED = 0.95


def fetch_all(src: str, *, prefer_locked: bool = False, processes: int = 8) -> dict:
    """Every ROI on the node: ``{"provenance", "sizes", "regions", "seconds"}``.
    ``regions`` omits the empty ROIs, which ``sizes`` gives as 0."""
    from neu_vol import dvid

    t0 = time.monotonic()
    node_spec, provenance = open_node(src, prefer_locked=prefer_locked)
    names = dvid.available_rois(node_spec)
    if not names:
        raise ValueError(f"no roi instances on {provenance['url']}")
    _, ranges, _ = dvid.fetch_roi_ranges(node_spec, names, processes=processes)
    sizes = {n: _blocks(ranges[n]) if len(ranges[n]) else 0 for n in names}
    regions = {n: region_from_ranges(ranges[n]) for n in names if sizes[n]}
    return {"provenance": provenance, "sizes": sizes, "regions": regions,
            "seconds": round(time.monotonic() - t0, 1)}


def _frac(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def overlaps(regions: Mapping[str, Region]) -> list[dict]:
    """Every intersecting pair, ``a`` before ``b`` by name, with each one's share."""
    names = sorted(regions)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = _overlap(regions[a], regions[b])
            if n:
                out.append({"a": a, "b": b, "blocks": n,
                            "of_a": _frac(n, regions[a].size),
                            "of_b": _frac(n, regions[b].size)})
    return out


def containment(regions: Mapping[str, Region], pairs: Sequence[dict]) -> dict[str, dict]:
    """``{roi: {"parent", "inside"}}``: the smallest larger ROI holding most of it."""
    size = {n: r.size for n, r in regions.items()}
    best: dict[str, tuple[int, str, float]] = {}
    for p in pairs:
        for child, parent, share in ((p["a"], p["b"], p["of_a"]),
                                     (p["b"], p["a"], p["of_b"])):
            if size[parent] <= size[child] or share <= 0.5:
                continue
            key = (size[parent], parent, share)
            if child not in best or key[:2] < best[child][:2]:
                best[child] = key
    return {n: {"parent": best[n][1] if n in best else None,
                "inside": best[n][2] if n in best else None} for n in regions}


def _ancestors(name: str, tree: Mapping[str, dict]) -> list[str]:
    out = []
    while (name := tree[name]["parent"]) is not None:
        out.append(name)
    return out


def composites(regions: Mapping[str, Region], tree: Mapping[str, dict]) -> dict[str, dict]:
    """For each ROI with children: how well they cover it, and how far they stick out."""
    children: dict[str, list[str]] = {}
    for n, t in tree.items():
        if t["parent"] is not None:
            children.setdefault(t["parent"], []).append(n)
    out = {}
    for parent, kids in sorted(children.items()):
        kids = sorted(kids)
        u = union([regions[k] for k in kids])
        covered = _overlap(u, regions[parent])
        out[parent] = {"children": kids,
                       "coverage": _frac(covered, regions[parent].size),
                       "gap_blocks": regions[parent].size - covered,
                       "outside": _frac(u.size - covered, u.size),
                       "outside_blocks": u.size - covered}
    return out


def siblings(pairs: Sequence[dict], tree: Mapping[str, dict], *, max_overlap: float,
             twins: Sequence[Sequence[str]] = ()) -> list[dict]:
    """Overlapping pairs that are not ancestor and descendant, largest share first.
    ``twins`` are groups of identical ROIs, which count as one and are not listed."""
    alias = {n: g[0] for g in twins for n in g}

    def lineage(n: str) -> set[str]:
        return {alias.get(m, m) for m in [n, *_ancestors(n, tree)]}

    out = []
    for p in pairs:
        a, b = p["a"], p["b"]
        la, lb = lineage(a), lineage(b)
        if alias.get(a, a) in lb or alias.get(b, b) in la:
            continue
        share = max(p["of_a"], p["of_b"])
        out.append({**p, "of_smaller": share, "too_much": share > max_overlap})
    return sorted(out, key=lambda s: (-s["of_smaller"], s["a"], s["b"]))


def identical(regions: Mapping[str, Region], pairs: Sequence[dict]) -> list[list[str]]:
    """Groups of ROIs holding exactly the same blocks."""
    groups: dict[str, set[str]] = {}
    for p in pairs:
        a, b = p["a"], p["b"]
        if regions[a].size == regions[b].size == p["blocks"]:
            g = groups.get(a, {a}) | groups.get(b, {b})
            for n in g:
                groups[n] = g
    return [list(g) for g in sorted({tuple(sorted(g)) for g in groups.values()})]


def past_extent(regions: Mapping[str, Region], extent: Sequence[int]) -> dict[str, dict]:
    """ROIs reaching outside ``[0, extent)`` blocks (zyx), by how many on each side."""
    ext = np.asarray(extent)
    out = {}
    for n, r in sorted(regions.items()):
        below, above = np.maximum(-r.lo, 0), np.maximum(r.hi - ext, 0)
        if below.any() or above.any():
            out[n] = {"below_zyx": below.tolist(), "above_zyx": above.tolist()}
    return out


def compare(old: Mapping[str, Region], new: Mapping[str, Region],
            old_sizes: Mapping[str, int], new_sizes: Mapping[str, int]) -> dict:
    """What changed between two nodes' ROIs, block for block."""
    changed = []
    for n in sorted(set(old_sizes) & set(new_sizes)):
        a, b = old.get(n), new.get(n)
        kept = _overlap(a, b) if a is not None and b is not None else 0
        before, after = old_sizes[n], new_sizes[n]
        if kept == before == after:
            continue
        changed.append({"roi": n, "before": before, "after": after,
                        "gained": after - kept, "lost": before - kept,
                        "jaccard": _frac(kept, before + after - kept)})
    changed.sort(key=lambda c: (c["jaccard"], c["roi"]))
    return {"added": sorted(set(new_sizes) - set(old_sizes)),
            "removed": sorted(set(old_sizes) - set(new_sizes)),
            "changed": changed,
            "unchanged": len(set(old_sizes) & set(new_sizes)) - len(changed)}


def check_spec(spec_path: str, regions: Mapping[str, Region], sizes: Mapping[str, int],
               *, contained: float) -> dict:
    """The spec against this node: placement problems, each part's share inside its
    top-level neuropil's own ROI, and each layer's overlaps or why it would be refused."""
    from . import roi_layers as rl

    spec = rl.load_spec(spec_path)
    problems = rl.check_against_node(spec, list(sizes), sizes)
    loose = []
    for top, parts in spec.neuropils.items():
        if top not in regions:
            continue
        for p in parts:
            if p in regions:
                share = _frac(_overlap(regions[p], regions[top]), regions[p].size)
                if share < contained:
                    loose.append({"part": p, "top": top, "inside": share})
    layers: dict[str, dict] = {}
    if not problems:
        for lay in spec.layers:
            try:
                _grid, out = rl.resolve(spec, regions, only=[lay.name])
                layers[lay.name] = {"overlaps": [list(p) for p in out[lay.name]["overlaps"]]}
            except (rl.NestedRois, ValueError) as exc:
                layers[lay.name] = {"refused": str(exc)}
    return {"path": spec_path, "problems": problems, "parts_outside_top": loose,
            "layers": layers}


def survey(regions: Mapping[str, Region], sizes: Mapping[str, int], *,
           contained: float = DEFAULT_CONTAINED,
           max_overlap: float = DEFAULT_MAX_OVERLAP,
           extent: Sequence[int] | None = None) -> dict:
    """The measurements, as one JSON-ready record."""
    pairs = overlaps(regions)
    tree = containment(regions, pairs)
    twins = identical(regions, pairs)
    rois = {}
    for n in sorted(sizes):
        row: dict = {"blocks": sizes[n]}
        if n in regions:
            row |= {"box_lo_zyx": regions[n].lo.tolist(), "box_hi_zyx": regions[n].hi.tolist(),
                    **tree[n]}
        rois[n] = row
    record = {"thresholds": {"contained": contained, "max_overlap": max_overlap},
              "rois": rois,
              "empty": sorted(n for n, s in sizes.items() if not s),
              "identical": twins,
              "composites": composites(regions, tree),
              "siblings": siblings(pairs, tree, max_overlap=max_overlap, twins=twins),
              "overlaps": pairs}
    if extent is not None:
        record["extent_blocks_zyx"] = [int(v) for v in extent]
        record["past_extent"] = past_extent(regions, extent)
    return record


def like_extent(like: str) -> list[int]:
    """``like``'s level-0 extent in ROI blocks (zyx)."""
    from neu_vol import read_scales
    from neu_vol.dvid import ROI_BLOCK_VOXELS

    shape = np.asarray(read_scales(like)[0].shape[-3:])
    return np.ceil(shape / ROI_BLOCK_VOXELS).astype(int).tolist()


def run(src: str, *, like: str | None = None, spec: str | None = None,
        compare_to: str | None = None, contained: float = DEFAULT_CONTAINED,
        max_overlap: float = DEFAULT_MAX_OVERLAP, prefer_locked: bool = False) -> dict:
    """Fetch the node's ROIs and survey them; optionally against a segmentation's extent,
    a layer spec, and another node."""
    got = fetch_all(src, prefer_locked=prefer_locked)
    logger.info("%d ROIs fetched in %.1f s", len(got["sizes"]), got["seconds"])
    record = {"source": got["provenance"], "fetch_seconds": got["seconds"],
              **survey(got["regions"], got["sizes"], contained=contained,
                       max_overlap=max_overlap,
                       extent=like_extent(like) if like else None)}
    if like:
        record["like"] = like
    if spec:
        record["spec"] = check_spec(spec, got["regions"], got["sizes"], contained=contained)
    if compare_to:
        other = fetch_all(compare_to, prefer_locked=prefer_locked)
        record["compare"] = {"source": other["provenance"],
                             **compare(other["regions"], got["regions"], other["sizes"],
                                       got["sizes"])}
    return record


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    if 0 < x < 0.001:
        return "<0.1%"
    return f"{100 * x:.1f}%" if x < 0.995 else "100%" if x >= 1 else ">99.5%"


def format_report(record: Mapping) -> str:
    """The record as text: tree first, then everything that looks wrong."""
    src, rois = record["source"], record["rois"]
    th = record["thresholds"]
    ref = f" ({src['ref']})" if src.get("ref") and src.get("ref") != src.get("uuid") else ""
    lines = [f"node {src['uuid']}{ref}: {len(rois)} ROIs"
             + (f", fetched in {record['fetch_seconds']:g} s" if "fetch_seconds" in record
                else "")]
    if record["empty"]:
        lines.append(f"empty: {', '.join(record['empty'])}")
    for g in record["identical"]:
        lines.append(f"identical: {' = '.join(g)}")

    comps = record["composites"]
    kids: dict[str | None, list[str]] = {}
    for n, r in rois.items():
        if r["blocks"]:
            kids.setdefault(r["parent"], []).append(n)
    lines += ["", f"containment (parent = smallest larger ROI holding most of it; "
                  f"'inside' shown below {_pct(th['contained'])}):"]

    def walk(name: str, depth: int) -> None:
        r = rois[name]
        note = []
        if r["inside"] is not None and r["inside"] < th["contained"]:
            note.append(f"{_pct(r['inside'])} inside")
        if name in comps:
            c = comps[name]
            note.append("children cover all of it" if not c["gap_blocks"] else
                        f"children cover {_pct(min(c['coverage'], 0.9999))}")
            if c["outside_blocks"]:
                note.append(f"{c['outside_blocks']:,} of their blocks outside it")
        lines.append(f"{'  ' * depth}{name:<{max(1, 24 - 2 * depth)}} {r['blocks']:>10,}"
                     + (f"   {'; '.join(note)}" if note else ""))
        for k in sorted(kids.get(name, ())):
            walk(k, depth + 1)

    for root in sorted(kids.get(None, ()), key=lambda n: (-rois[n]["blocks"], n)):
        walk(root, 1)

    sib = record["siblings"]
    lines += ["", f"sibling overlaps: {len(sib)} pair(s)"
                  + (f", {sum(s['too_much'] for s in sib)} past max_overlap "
                     f"{_pct(th['max_overlap'])} (roi-layers refuses those in one layer)"
                     if any(s["too_much"] for s in sib) else "")]
    for s in sib:
        smaller = s["a"] if s["of_a"] >= s["of_b"] else s["b"]
        lines.append(f"  {s['a']} / {s['b']}: {s['blocks']:,} blocks, "
                     f"{_pct(s['of_smaller'])} of {smaller}" + ("  <-- nested?" if s["too_much"]
                                                                 else ""))

    if "past_extent" in record:
        ext = record["extent_blocks_zyx"]
        past = record["past_extent"]
        lines += ["", f"past {record.get('like', 'the segmentation')}'s extent "
                      f"({ext} blocks zyx): {len(past) or 'none'}"]
        for n, p in past.items():
            lines.append(f"  {n}: below {p['below_zyx']}, above {p['above_zyx']}")

    if "spec" in record:
        sp = record["spec"]
        lines += ["", f"spec {sp['path']}:"]
        lines += [f"  {p}" for p in sp["problems"]] or ["  every ROI on the node is placed"]
        for p in sp["parts_outside_top"]:
            lines.append(f"  part {p['part']} is {_pct(p['inside'])} inside {p['top']}")
        for name, lay in sp["layers"].items():
            if "refused" in lay:
                lines.append(f"  layer {name}: REFUSED: {lay['refused']}")
            elif lay["overlaps"]:
                lines.append(f"  layer {name}: {len(lay['overlaps'])} overlapping pair(s), "
                             f"e.g. {lay['overlaps'][0]}")

    if "compare" in record:
        c = record["compare"]
        lines += ["", f"against node {c['source']['uuid']}: {len(c['changed'])} changed, "
                      f"{c['unchanged']} unchanged"]
        if c["added"]:
            lines.append(f"  added: {', '.join(c['added'])}")
        if c["removed"]:
            lines.append(f"  removed: {', '.join(c['removed'])}")
        for ch in c["changed"]:
            lines.append(f"  {ch['roi']:<20} {ch['before']:>10,} -> {ch['after']:>10,}   "
                         f"+{ch['gained']:,} -{ch['lost']:,}   jaccard {ch['jaccard']:.3f}")
    return "\n".join(lines)
