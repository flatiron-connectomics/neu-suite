"""DVID ROIs as precomputed segmentation layers, one label per voxel in each.

A DVID ``roi`` instance is a set of 32-voxel blocks of the scale-0 grid, and a node's ROIs
overlap: some nest (ME(L) in OL(L)), some are siblings drawn imprecisely (GL21(L) bleeding
into VMNP(L)). A segmentation holds one label per voxel, so a spec (:func:`load_spec`)
sorts them into **layers**, each written as its own volume with every ROI's id taken from
one id table shared by all layers:

- ``rois = [...]``: those ROIs, overlaps settled by the ``overlap`` policy.
- ``tree = "top"``: the top-level neuropils of the spec's ``[neuropils]`` tree, each the
  union of its parts and its own ROI where the node has one.
- ``tree = "parts"``: each top-level neuropil's region divided among its parts; with
  ``fill_gaps`` the blocks no part claims keep the top-level id. It is cut from the
  ``top`` layer's regions, so the two nest exactly.
- ``from = LAYER``: the union of another layer as one label, closed by ``close_blocks``.

``overlap = "split"`` gives a contested block to the ROI it lies deepest inside (a distance
transform of each ROI's own mask, ties to the ROI listed first); ``"order"`` to the ROI
listed first; ``"error"`` refuses. Siblings overlapping by more than ``max_overlap`` of the
smaller are nested, not imprecise, and are refused under every policy.

Resolution works on DVID's block ranges and needs no DVID; :func:`build` fetches them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

POLICIES = ("split", "order", "error")

#: Siblings overlapping by more than this fraction of the smaller are nested.
DEFAULT_MAX_OVERLAP = 0.2

#: The layer volumes' chunk, in ROI blocks.
CHUNK = (64, 64, 64)

_SPEC_KEYS = {"build", "layer", "neuropils", "ignore", "names", "ids"}
_LAYER_KEYS = {"name", "rois", "tree", "from", "label", "close_blocks", "fill_gaps",
               "overlap"}
_SIDE = re.compile(r"^(.*)\((L|R)\)$")


class SpecError(ValueError):
    """The ROI layer spec is malformed, or disagrees with the node."""


class NestedRois(ValueError):
    """ROIs placed as siblings overlap too much to be imprecise neighbours."""


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layer:
    name: str
    rois: tuple[str, ...] = ()
    tree: str | None = None
    source: str | None = None
    label: str | None = None
    close_blocks: int = 0
    fill_gaps: bool = False
    overlap: str = "split"

    @property
    def kind(self) -> str:
        return ("rois" if self.rois else f"tree:{self.tree}" if self.tree
                else f"from:{self.source}")


@dataclass(frozen=True)
class Spec:
    layers: tuple[Layer, ...]
    neuropils: dict[str, tuple[str, ...]]
    ids: dict[str, int]
    names: dict[str, str] = field(default_factory=dict)
    ignore: dict[str, str] = field(default_factory=dict)
    max_overlap: float = DEFAULT_MAX_OVERLAP

    def layer(self, name: str) -> Layer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise SpecError(f"no layer {name!r}; the spec has "
                       f"{', '.join(lay.name for lay in self.layers)}")

    def used_rois(self) -> set[str]:
        """Every DVID ROI the layers read. A top-level neuropil with parts is optional."""
        out = {r for lay in self.layers for r in lay.rois}
        if any(lay.tree for lay in self.layers):
            for top, parts in self.neuropils.items():
                out |= set(parts) or {top}
        return out

    def optional_rois(self) -> set[str]:
        """Top-level neuropils with parts: read where the node has them, else built."""
        if not any(lay.tree for lay in self.layers):
            return set()
        return {top for top, parts in self.neuropils.items() if parts}


def layer_rois(spec: Spec, name: str) -> list[str]:
    """The DVID ROIs a layer's segments are, in its priority order, for labelling synapses
    with the same set (``neu-mark points --rois SPEC.toml:LAYER``)."""
    lay = spec.layer(name)
    if lay.rois:
        return list(lay.rois)
    if lay.tree == "parts":
        return [p for top, parts in spec.neuropils.items() for p in (parts or (top,))]
    raise SpecError(f"layer {name!r} ({lay.kind}) is built from other ROIs, so no list of "
                    f"DVID ROIs labels synapses the same way; name a rois or a tree = "
                    f"'parts' layer")


def top_level(spec: Spec) -> dict[str, str]:
    """``{part: its top-level neuropil}``, a top-level neuropil without parts its own."""
    return {p: top for top, parts in spec.neuropils.items() for p in (parts or (top,))}


def load_spec(path: str) -> Spec:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_spec(raw, where=path)


def parse_spec(raw: Mapping[str, Any], *, where: str = "spec") -> Spec:
    unknown = set(raw) - _SPEC_KEYS
    if unknown:
        raise SpecError(f"{where}: unknown section(s) {sorted(unknown)}; expected "
                        f"{sorted(_SPEC_KEYS)}")
    build = dict(raw.get("build") or {})
    default_overlap = build.pop("overlap", "split")
    max_overlap = float(build.pop("max_overlap", DEFAULT_MAX_OVERLAP))
    if build:
        raise SpecError(f"{where}: unknown [build] key(s) {sorted(build)}")

    neuropils = {str(k): tuple(map(str, v)) for k, v in (raw.get("neuropils") or {}).items()}
    layers = []
    for i, entry in enumerate(raw.get("layer") or []):
        bad = set(entry) - _LAYER_KEYS
        if bad:
            raise SpecError(f"{where}: layer {i + 1}: unknown key(s) {sorted(bad)}")
        name = entry.get("name")
        if not name:
            raise SpecError(f"{where}: layer {i + 1} has no name")
        kinds = [k for k in ("rois", "tree", "from") if entry.get(k)]
        if len(kinds) != 1:
            raise SpecError(f"{where}: layer {name!r} needs exactly one of rois, tree or "
                            f"from; it has {kinds or 'none'}")
        lay = Layer(name=str(name), rois=tuple(map(str, entry.get("rois") or ())),
                    tree=entry.get("tree"), source=entry.get("from"),
                    label=entry.get("label"), close_blocks=int(entry.get("close_blocks", 0)),
                    fill_gaps=bool(entry.get("fill_gaps", False)),
                    overlap=entry.get("overlap", default_overlap))
        if lay.overlap not in POLICIES:
            raise SpecError(f"{where}: layer {name!r}: overlap must be one of {POLICIES}, "
                            f"not {lay.overlap!r}")
        if lay.tree is not None and lay.tree not in ("top", "parts"):
            raise SpecError(f"{where}: layer {name!r}: tree must be 'top' or 'parts'")
        if lay.tree and not neuropils:
            raise SpecError(f"{where}: layer {name!r} cuts the [neuropils] tree, which is "
                            f"empty")
        if lay.source is not None and not lay.label:
            raise SpecError(f"{where}: layer {name!r} is built from {lay.source!r} and "
                            f"needs a label for its one segment")
        if lay.fill_gaps and lay.tree != "parts":
            raise SpecError(f"{where}: layer {name!r}: fill_gaps applies to a tree = "
                            f"'parts' layer only")
        layers.append(lay)
    names_seen = [lay.name for lay in layers]
    dup = sorted({n for n in names_seen if names_seen.count(n) > 1})
    if dup:
        raise SpecError(f"{where}: layer name(s) repeated: {dup}")
    for lay in layers:
        if lay.source is not None and lay.source not in names_seen:
            raise SpecError(f"{where}: layer {lay.name!r} is built from {lay.source!r}, "
                            f"which is not a layer")
        if lay.source == lay.name:
            raise SpecError(f"{where}: layer {lay.name!r} is built from itself")

    ids = {str(k): int(v) for k, v in (raw.get("ids") or {}).items()}
    spec = Spec(layers=tuple(layers), neuropils=neuropils, ids=ids,
                names={str(k): str(v) for k, v in (raw.get("names") or {}).items()},
                ignore={str(k): str(v) for k, v in (raw.get("ignore") or {}).items()},
                max_overlap=max_overlap)
    labelled = set(spec.used_rois()) | {lay.label for lay in layers if lay.label}
    if any(lay.tree for lay in layers):
        labelled |= set(neuropils)
    missing = sorted(labelled - set(ids))
    if missing:
        raise SpecError(f"{where}: no id for {missing}; every label needs one in [ids]")
    values = list(ids.values())
    if any(v <= 0 for v in values):
        raise SpecError(f"{where}: ids must be positive (0 is background)")
    clash = sorted({v for v in values if values.count(v) > 1})
    if clash:
        raise SpecError(f"{where}: id(s) used twice: {clash}")
    return spec


def check_against_node(spec: Spec, available: Sequence[str],
                       sizes: Mapping[str, int] | None = None) -> list[str]:
    """Problems between the spec and the node's ROIs; empty if none.

    ``available`` names the node's ROIs and ``sizes`` (``{name: blocks}``) the fetched
    ones. Every ROI on the node must be used or ignored, so a new one is never dropped
    unseen.
    """
    used, optional = spec.used_rois(), spec.optional_rois()
    problems = []
    absent = sorted(r for r in used if r not in available)
    if absent:
        problems.append(f"not on the node: {', '.join(absent)}")
    empty = sorted(r for r in used if (sizes or {}).get(r) == 0)
    if empty:
        problems.append(f"empty on the node: {', '.join(empty)}")
    placed = used | optional | set(spec.ignore)
    unplaced = sorted(r for r in available if r not in placed)
    if unplaced:
        problems.append(f"on the node but in no layer and not in [ignore]: "
                        f"{', '.join(unplaced)}")
    return problems


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A boolean mask whose element ``(0, 0, 0)`` is block ``lo`` (zyx)."""

    mask: np.ndarray
    lo: np.ndarray

    @property
    def hi(self) -> np.ndarray:
        return self.lo + np.asarray(self.mask.shape)

    @property
    def size(self) -> int:
        return int(self.mask.sum())


def region_from_ranges(ranges: np.ndarray) -> Region:
    """DVID's ``[z, y, x0, x1]`` ranges (``x1`` inclusive) as a :class:`Region`."""
    r = np.asarray(ranges, dtype=np.int64).reshape(-1, 4)
    if not len(r):
        raise ValueError("an empty ROI has no region")
    lo = np.array([r[:, 0].min(), r[:, 1].min(), r[:, 2].min()])
    hi = np.array([r[:, 0].max(), r[:, 1].max(), r[:, 3].max()]) + 1
    shape = tuple(int(v) for v in hi - lo)
    # Run starts +1 and run ends -1 along x, then a running sum.
    edges = np.zeros((shape[0], shape[1], shape[2] + 1), dtype=np.int32)
    z, y = r[:, 0] - lo[0], r[:, 1] - lo[1]
    np.add.at(edges, (z, y, r[:, 2] - lo[2]), 1)
    np.add.at(edges, (z, y, r[:, 3] + 1 - lo[2]), -1)
    return Region(np.cumsum(edges, axis=2)[..., :-1] > 0, lo)


def union(regions: Sequence[Region]) -> Region:
    lo = np.min([r.lo for r in regions], axis=0)
    hi = np.max([r.hi for r in regions], axis=0)
    mask = np.zeros(tuple(int(v) for v in hi - lo), dtype=bool)
    for r in regions:
        mask[_slices(r.lo - lo, r.mask.shape)] |= r.mask
    return Region(mask, lo)


def _slices(offset, shape) -> tuple[slice, ...]:
    return tuple(slice(int(o), int(o) + int(s)) for o, s in zip(offset, shape))


def _overlap(a: Region, b: Region) -> int:
    lo, hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)
    if np.any(hi <= lo):
        return 0
    shape = hi - lo
    return int((a.mask[_slices(lo - a.lo, shape)] & b.mask[_slices(lo - b.lo, shape)]).sum())


def _depth(region: Region) -> np.ndarray:
    """Distance from each block of the region to its outside, in blocks."""
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(region.mask, 1)
    return distance_transform_edt(padded).astype(np.float32)[1:-1, 1:-1, 1:-1]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    """The block grid every layer shares: ``shape`` blocks from block ``lo``."""

    lo: np.ndarray
    shape: tuple[int, int, int]

    def view(self, arr: np.ndarray, region: Region) -> np.ndarray:
        return arr[_slices(region.lo - self.lo, region.mask.shape)]


def paint(grid: Grid, members: Sequence[tuple[str, int, Region]], *, policy: str,
          max_overlap: float, within: np.ndarray | None = None,
          what: str = "layer") -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Label the grid with ``members`` (``(name, id, region)``, in priority order).

    ``within`` limits every member to a boolean region of the grid. Returns the labels and
    the overlapping pairs ``(earlier, later, blocks)``. Raises :class:`NestedRois` for a
    pair overlapping by more than ``max_overlap`` of the smaller, and ``ValueError`` for
    any overlap under ``policy="error"``.
    """
    if policy not in POLICIES:
        raise ValueError(f"overlap policy must be one of {POLICIES}, not {policy!r}")
    regions = []
    for name, ident, reg in members:
        if within is not None:
            reg = Region(reg.mask & grid.view(within, reg), reg.lo)
        regions.append((name, ident, reg))

    pairs = []
    for i, (a, _ia, ra) in enumerate(regions):
        for b, _ib, rb in regions[i + 1:]:
            n = _overlap(ra, rb)
            if not n:
                continue
            pairs.append((a, b, n))
            smaller = min(ra.size, rb.size)
            if smaller and n / smaller > max_overlap:
                raise NestedRois(
                    f"{what}: {a} and {b} overlap by {n} blocks, {n / smaller:.0%} of the "
                    f"smaller. ROIs that nest belong in different layers, or in the "
                    f"[neuropils] tree as a neuropil and its part.")
    if pairs and policy == "error":
        raise ValueError(f"{what}: {len(pairs)} overlapping pair(s) under overlap = "
                         f"'error', e.g. {pairs[:3]}")

    labels = np.zeros(grid.shape, dtype=np.uint32)
    claims = np.zeros(grid.shape, dtype=np.uint8)
    for _name, _ident, reg in regions:
        grid.view(claims, reg)[...] += reg.mask
    for _name, ident, reg in regions:                       # the uncontested blocks
        sole = reg.mask & (grid.view(claims, reg) == 1)
        grid.view(labels, reg)[sole] = ident
    if not pairs:
        return labels, pairs

    contested = claims > 1
    if policy == "order":
        for _name, ident, reg in regions:
            view = grid.view(labels, reg)
            view[reg.mask & grid.view(contested, reg) & (view == 0)] = ident
        return labels, pairs

    best = np.full(grid.shape, -1.0, dtype=np.float32)
    for _name, ident, reg in regions:
        here = reg.mask & grid.view(contested, reg)
        if not here.any():
            continue
        depth = _depth(reg)
        bview = grid.view(best, reg)
        win = here & (depth > bview)
        bview[win] = depth[win]
        grid.view(labels, reg)[win] = ident
    return labels, pairs


def resolve(spec: Spec, regions: Mapping[str, Region],
            only: Sequence[str] | None = None) -> tuple[Grid, dict[str, dict]]:
    """Every layer (or those named in ``only``, plus what they are built from) as label
    arrays on one grid. Returns ``(grid, {layer: {"labels", "overlaps"}})``."""
    used = [regions[r] for r in spec.used_rois() | spec.optional_rois() if r in regions]
    whole = union(used)
    grid = Grid(whole.lo, tuple(int(s) for s in whole.mask.shape))
    out: dict[str, dict] = {}

    def tops() -> dict[str, Region]:
        built = {}
        for top, parts in spec.neuropils.items():
            members = [regions[p] for p in parts]
            if top in regions:
                members.append(regions[top])
            built[top] = union(members)
        return built

    def layer(name: str) -> dict:
        if name in out:
            return out[name]
        lay = spec.layer(name)
        if lay.rois:
            labels, pairs = paint(grid, [(r, spec.ids[r], regions[r]) for r in lay.rois],
                                  policy=lay.overlap, max_overlap=spec.max_overlap,
                                  what=f"layer {name!r}")
            out[name] = {"labels": labels, "overlaps": pairs}
        elif lay.tree == "top":
            labels, pairs = paint(grid, [(t, spec.ids[t], r) for t, r in tops().items()],
                                  policy=lay.overlap, max_overlap=spec.max_overlap,
                                  what=f"layer {name!r}")
            out[name] = {"labels": labels, "overlaps": pairs}
        elif lay.tree == "parts":
            top_name = next((lay2.name for lay2 in spec.layers if lay2.tree == "top"), None)
            top_labels = (layer(top_name)["labels"] if top_name else paint(
                grid, [(t, spec.ids[t], r) for t, r in tops().items()],
                policy=lay.overlap, max_overlap=spec.max_overlap,
                what=f"layer {name!r} (top level)")[0])
            labels = np.zeros(grid.shape, dtype=np.uint32)
            pairs: list = []
            for top, parts in spec.neuropils.items():
                within = top_labels == spec.ids[top]
                if not within.any():
                    continue
                members = [(p, spec.ids[p], regions[p]) for p in parts] or \
                    [(top, spec.ids[top], regions[top])]
                sub, sub_pairs = paint(grid, members, policy=lay.overlap,
                                       max_overlap=spec.max_overlap, within=within,
                                       what=f"layer {name!r}, parts of {top}")
                if lay.fill_gaps:
                    sub[within & (sub == 0)] = spec.ids[top]
                labels[within] = sub[within]
                pairs += sub_pairs
            out[name] = {"labels": labels, "overlaps": pairs}
        else:
            src = layer(lay.source)["labels"] != 0
            if lay.close_blocks > 0:
                src = _close(src, lay.close_blocks)
            labels = np.where(src, np.uint32(spec.ids[lay.label]), np.uint32(0))
            out[name] = {"labels": labels, "overlaps": []}
        return out[name]

    for lay in spec.layers:
        if only is None or lay.name in only:
            layer(lay.name)
    return grid, {n: v for n, v in out.items() if only is None or n in only}


def _close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing by a ball, padded so the grid's edge does not erode it."""
    from scipy.ndimage import binary_closing

    r = int(radius)
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (z * z + y * y + x * x) <= r * r
    closed = binary_closing(np.pad(mask, r), structure=ball)
    return closed[r:-r, r:-r, r:-r] | mask


# --------------------------------------------------------------------------- #
# Segment properties
# --------------------------------------------------------------------------- #
def describe(name: str, names: Mapping[str, str]) -> str:
    """``names``' description for an ROI, its side and exclusion suffixes added."""
    side = ""
    m = _SIDE.match(name)
    core = name
    if m:
        core, side = m.group(1), {"L": " (left)", "R": " (right)"}[m.group(2)]
    if core in names:
        return names[core] + side
    ex = re.fullmatch(r"(.+)\(-(.+)\)", core)
    if ex and ex.group(1) in names:
        return f"{names[ex.group(1)]}, without {names.get(ex.group(2), ex.group(2))}{side}"
    return ""


def properties(spec: Spec, layer_name: str, labels: np.ndarray,
               block_nm: Sequence[float]) -> dict:
    """The layer's ``segment_properties`` document: name, description, tags, volume."""
    from .segprops import AT_TYPE, normalize_tag

    lay = spec.layer(layer_name)
    by_id = {v: k for k, v in spec.ids.items()}
    present, counts = np.unique(labels[labels != 0], return_counts=True)
    parent_of = {p: top for top, parts in spec.neuropils.items() for p in parts}
    rows = []
    for ident, n in zip(present.tolist(), counts.tolist()):
        name = by_id[ident]
        tags = [f"layer:{layer_name}"]
        m = _SIDE.match(name)
        if m:
            tags.append(f"side:{m.group(2)}")
        if lay.tree == "parts" and name in parent_of:
            tags.append(f"parent:{parent_of[name]}")
        rows.append((ident, name, describe(name, spec.names),
                     [t for t in (normalize_tag(t) for t in tags) if t], n))
    vocabulary = sorted({t for *_, tags, _n in rows for t in tags})
    index = {t: i for i, t in enumerate(vocabulary)}
    um3 = float(np.prod(block_nm)) / 1e9
    return {"@type": AT_TYPE, "inline": {
        "ids": [str(r[0]) for r in rows],
        "properties": [
            {"id": "label", "type": "label", "values": [r[1] for r in rows]},
            {"id": "description", "type": "description", "values": [r[2] for r in rows]},
            {"id": "tags", "type": "tags", "tags": vocabulary,
             "values": [sorted(index[t] for t in r[3]) for r in rows]},
            {"id": "volume_um3", "type": "number", "data_type": "float32",
             "description": "volume in cubic micrometres",
             "values": [round(r[4] * um3, 3) for r in rows]},
        ]}}


# --------------------------------------------------------------------------- #
# Fetch, resolve, write
# --------------------------------------------------------------------------- #
def parse_node_url(url: str) -> dict[str, str]:
    """``dvid://SERVER/UUID`` as a spec for :mod:`neu_vol.dvid`, which addresses a node's
    ROIs through any instance name."""
    from neu_vol import dvid

    if not dvid.is_url(url):
        raise ValueError(f"not a DVID URL: {url!r} (expected dvid://SERVER/UUID)")
    scheme = "dvid+https://" if url.startswith("dvid+https://") else "dvid://"
    parts = [p for p in url[len(scheme):].split("/") if p]
    if len(parts) == 3:
        raise ValueError(
            f"{url} names an instance, but ROIs are read from a whole NODE: drop the "
            f"instance and pass dvid://{parts[0]}/{parts[1]}")
    if len(parts) != 2:
        raise ValueError(f"malformed DVID node URL {url!r}: expected dvid://SERVER/UUID")
    return {**dvid.parse_url(f"{scheme}{parts[0]}/{parts[1]}/roi"), "instance": "roi"}


def _blocks(ranges: np.ndarray) -> int:
    r = np.asarray(ranges, dtype=np.int64).reshape(-1, 4)
    return int((r[:, 3] - r[:, 2] + 1).sum())


def fetch(src: str, spec: Spec, *, prefer_locked: bool = False,
          processes: int = 8) -> dict:
    """The regions of every ROI the spec reads, after checking the spec against the
    node's list of ROIs. Raises :class:`SpecError` listing every disagreement."""
    from neu_vol import dvid

    node_spec = parse_node_url(src)
    node = dvid.resolve_node(node_spec, prefer_locked=prefer_locked)
    node_spec = {**node_spec, "uuid": node["uuid"]}
    available = dvid.available_rois(node_spec)
    problems = check_against_node(spec, available)
    if problems:
        raise SpecError("the spec does not match the node:\n  " + "\n  ".join(problems))
    wanted = sorted((spec.used_rois() | spec.optional_rois()) & set(available))
    _, ranges, _ = dvid.fetch_roi_ranges(node_spec, wanted, processes=processes)
    sizes = {n: _blocks(r) if len(r) else 0 for n, r in ranges.items()}
    problems = check_against_node(spec, available, sizes)
    if problems:
        raise SpecError("the spec does not match the node:\n  " + "\n  ".join(problems))
    provenance = dvid.node_provenance(node_spec, node)
    provenance.pop("instance", None)
    provenance["url"] = dvid.spec_url(node_spec).rsplit("/", 1)[0]
    return {"sizes": sizes, "provenance": provenance,
            "regions": {n: region_from_ranges(ranges[n]) for n in wanted if sizes[n]}}


def grid_frame(like: str, grid: Grid):
    """The :class:`neu_lib.Frame` of the grid: blocks of 32 voxels of ``like``'s level 0."""
    from neu_lib import Frame
    from neu_vol import read_scales
    from neu_vol.dvid import ROI_BLOCK_VOXELS

    level0 = read_scales(like)[0]
    block = np.asarray(level0.voxel_size, dtype=np.float64) * ROI_BLOCK_VOXELS
    origin = np.asarray(level0.frame.origin_nm, dtype=np.float64) + grid.lo * block
    extent = np.ceil(np.asarray(level0.shape[-3:]) / ROI_BLOCK_VOXELS).astype(int)
    return Frame(tuple(block), tuple(origin)), extent


def write_layer(dst: str, labels: np.ndarray, frame, *, overwrite: bool,
                tmp: str) -> dict:
    """``labels`` as a precomputed segmentation at ``dst``, with a mode pyramid."""
    from neu_lib import Piece
    from neu_vol import convert, write_piece
    from neu_vol.ops.create import profile_for

    name = dst.rstrip("/").rsplit("/", 1)[-1]
    h5 = os.path.join(tmp, f"{name}.h5")
    write_piece(Piece(labels.astype(np.uint64), frame, kind="segmentation", name=name), h5,
                overwrite=True)
    return convert(h5, dst, kind="segmentation", profile=profile_for("precomputed", None, dst),
                   chunk=CHUNK, min_dim=32, delete_existing=overwrite,
                   progress_path=os.path.join(tmp, f"{name}.progress.jsonl"))


def build(src: str, spec_path: str, like: str, dst: str, *, layers: Sequence[str] | None = None,
          overwrite: bool = False, dry_run: bool = False, prefer_locked: bool = False) -> dict:
    """Fetch the node's ROIs, resolve the spec's layers and write them under ``dst``.

    Each layer is ``<dst>/<layer>``, a precomputed segmentation linked to its
    ``segment_properties``; ``<dst>/rois.json`` records the ids, the overlaps and the
    node. ``dry_run`` resolves everything and writes nothing.
    """
    import tempfile

    from neu_vol import exists, link_subresources, location

    from . import segprops

    spec = load_spec(spec_path)
    if layers:
        unknown = sorted(set(layers) - {lay.name for lay in spec.layers})
        if unknown:
            raise SpecError(f"no layer(s) {unknown} in {spec_path}")
    got = fetch(src, spec, prefer_locked=prefer_locked)
    grid, resolved = resolve(spec, got["regions"], only=layers)
    frame, extent = grid_frame(like, grid)

    dst = dst.rstrip("/")
    if not dry_run and not overwrite:
        taken = [n for n in resolved if exists(f"{dst}/{n}", "info")]
        if taken:
            raise FileExistsError(
                f"layer(s) {taken} already exist under {dst}; pass --overwrite to replace "
                f"them, which also deletes their meshes")

    outside = np.zeros(3, dtype=int)
    hi = grid.lo + np.asarray(grid.shape)
    report = {"layers": {}, "grid": {"lo_blocks": grid.lo.tolist(), "shape": list(grid.shape),
                                     "voxel_size_nm": list(frame.voxel_size_nm),
                                     "origin_nm": list(frame.origin_nm)}}
    if np.any(grid.lo < 0) or np.any(hi > extent):
        outside = hi - extent
        logger.warning("the ROIs reach past %s's extent (%s blocks) to %s; the layers "
                       "keep their whole extent", like, extent.tolist(), hi.tolist())
    report["grid"]["past_like_extent_blocks"] = np.maximum(outside, 0).tolist()

    with tempfile.TemporaryDirectory() as tmp:
        for name, res in resolved.items():
            labels = res["labels"]
            info = properties(spec, name, labels, frame.voxel_size_nm)
            ids = [int(i) for i in info["inline"]["ids"]]
            report["layers"][name] = {
                "kind": spec.layer(name).kind, "segments": len(ids), "ids": ids,
                "overlaps": [list(p) for p in res["overlaps"]]}
            logger.info("%-16s %3d segments, %d overlapping pair(s)", name, len(ids),
                        len(res["overlaps"]))
            if dry_run:
                continue
            write_layer(f"{dst}/{name}", labels, frame, overwrite=overwrite, tmp=tmp)
            segprops.write(f"{dst}/{name}", info)
            link_subresources(f"{dst}/{name}", segment_properties=segprops.SUBDIR)

    with open(spec_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    record = {"source": got["provenance"],
              "spec": {"path": os.path.abspath(spec_path), "sha256": digest},
              "like": like, **report,
              "ids": spec.ids, "ignored": spec.ignore}
    if not dry_run:
        location.write_json(dst, record, "rois.json")
    return record
