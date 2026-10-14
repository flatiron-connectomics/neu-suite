"""The NeuTu-style tracer.

`test_per_voxel_weights_match_neutu_edge_cost` checks that dijkstra3d's per-voxel weights
stand in for NeuTu's symmetric edge cost ``d·[f(v1) + f(v2)]``. The two agree for uniform
steps, but 3D moves mix step lengths (1, √2, √3), so this is checked against an
independent reference implementation.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from neu_morpho import neutu_trace


def _tube(length=40, r=3, pad=4, bend=False):
    """A cylinder along z, optionally bent, as a Fortran-order uint8 mask."""
    s = (length + 2 * pad, 2 * (r + pad) + (length // 2 if bend else 0), 2 * (r + pad))
    zz, yy, xx = np.indices(s)
    cy, cx = r + pad, r + pad
    off = np.clip(zz - pad, 0, None) // 2 if bend else 0
    m = ((zz >= pad) & (zz < pad + length) &
         ((yy - cy - off) ** 2 + (xx - cx) ** 2 <= r * r))
    return np.asfortranarray(m.astype(np.uint8))


def _y_body():
    """A Y: a stem along z splitting into two arms."""
    s = (60, 60, 24)
    zz, yy, xx = np.indices(s)
    stem = (zz >= 6) & (zz < 30) & ((yy - 30) ** 2 + (xx - 12) ** 2 <= 9)
    t = np.clip(zz - 30, 0, None)
    arm1 = (zz >= 30) & (zz < 54) & ((yy - 30 - t) ** 2 + (xx - 12) ** 2 <= 9)
    arm2 = (zz >= 30) & (zz < 54) & ((yy - 30 + t) ** 2 + (xx - 12) ** 2 <= 9)
    return np.asfortranarray((stem | arm1 | arm2).astype(np.uint8))


def _weights(mask):
    import edt
    import kimimaro.skeletontricks

    dbf = edt.edt(mask, anisotropy=(1, 1, 1), black_border=False, order="F")
    return neutu_trace.neutu_pdrf(kimimaro.skeletontricks.zero2inf(dbf))


def test_neutu_pdrf_values():
    dbf = np.asfortranarray(np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert p[0, 0, 0] == pytest.approx(0.5)          # 1/(1+1)
    assert p[0, 0, 1] == pytest.approx(0.2)          # 1/(1+4)
    assert p[0, 0, 2] == pytest.approx(0.1)          # 1/(1+9)
    assert p.dtype == np.float32


def test_neutu_pdrf_forbids_background():
    """Background stays inf: 1/(1+inf²) would be 0, making empty space free to cross."""
    dbf = np.asfortranarray(np.array([[[1.0, np.inf]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert np.isinf(p[0, 0, 1])
    assert p[0, 0, 0] < p[0, 0, 1]


def _neutu_edge_cost_path(weights, src, dst):
    """Reference Dijkstra with NeuTu's edge cost ``d·[f(u)+f(v)]``, 26-connected. Plain
    and independent of the code under test on purpose."""
    shape = weights.shape
    offs = [(dz, dy, dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    step = {o: float(np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)) for o in offs}

    best = {src: 0.0}
    prev: dict = {src: None}
    pq = [(0.0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        fu = weights[u]
        for o in offs:
            v = (u[0] + o[0], u[1] + o[1], u[2] + o[2])
            if not all(0 <= v[k] < shape[k] for k in range(3)):
                continue
            fv = weights[v]
            if not np.isfinite(fv):             # background is forbidden
                continue
            nd = d + step[o] * (fu + fv)
            if nd < best.get(v, float("inf")) - 1e-12:
                best[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return np.array(path[::-1], dtype=int)


@pytest.mark.parametrize("bend", [False, True])
def test_per_voxel_weights_match_neutu_edge_cost(bend):
    """The per-voxel route is no worse under NeuTu's own cost than NeuTu's optimum (ties
    may break differently, so identical paths are not required)."""
    import dijkstra3d

    m = _tube(length=30, r=3, bend=bend)
    w = _weights(m)
    inside = np.argwhere(m > 0)
    src = tuple(int(v) for v in inside[0])
    dst = tuple(int(v) for v in inside[-1])

    ours = np.asarray(dijkstra3d.path_from_parents(dijkstra3d.parental_field(w, src), dst))
    ref = _neutu_edge_cost_path(w, src, dst)
    assert ref is not None and len(ours) > 0

    def neutu_cost(path):
        d = np.linalg.norm(np.diff(path, axis=0), axis=1)
        f = w[path[:, 0], path[:, 1], path[:, 2]]
        return float(np.sum(d * (f[:-1] + f[1:])))

    c_ours, c_ref = neutu_cost(ours), neutu_cost(ref)
    assert c_ref <= c_ours + 1e-9                      # ref is the true optimum
    assert c_ours <= c_ref * 1.02, (
        f"per-voxel weighting chose a materially worse path under NeuTu's own "
        f"cost: {c_ours:.6g} vs optimum {c_ref:.6g}")


def test_skeleton_stays_inside_the_tube():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    assert v[:, 0].max() - v[:, 0].min() >= 35              # spans the tube
    assert m[v[:, 0], v[:, 1], v[:, 2]].all()               # never leaves it


def test_radii_are_inscribed_not_inflated():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_empty_mask_returns_empty_skeleton():
    skel = neutu_trace.skeletonize(np.zeros((8, 8, 8), dtype=np.uint8))
    assert len(np.asarray(skel.vertices)) == 0


def test_default_invalidation_is_neutus_own_value():
    """NeuTu's own EDT+2. A raised value once masked two bugs and deleted real cable, so
    if this ever needs changing, look for a reintroduced bug first."""
    assert neutu_trace.NEUTU_CONST == 2.0
    assert neutu_trace.INVALIDATION_CONST == neutu_trace.NEUTU_CONST


def test_tighter_invalidation_yields_more_vertices():
    """A smaller invalidation ball leaves more to cover, so more paths."""
    m = _tube(length=40, r=3, bend=True)
    tight = neutu_trace.skeletonize(m, scale=1.0, const=2.0)
    looser = neutu_trace.skeletonize(m, scale=2.0, const=8.0)
    assert len(np.asarray(tight.vertices)) >= len(np.asarray(looser.vertices))


def test_all_connected_components_are_traced():
    """Every component is traced. A single-root trace covers only the root's component,
    and the root can land on a speck first in memory order (on a real body, a trace
    covering 3% of it). Here a small tube comes first and a large one after."""
    m = np.zeros((60, 40, 20), dtype=np.uint8)
    m[2:8, 3:9, 3:9] = 1                                    # small, first in z
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 20) & (zz < 55) & ((yy - 25) ** 2 + (xx - 10) ** 2 <= 16))] = 1
    m = np.asfortranarray(m)

    # min_length=0, or the 6-voxel cube is legitimately dropped as too short
    skel = neutu_trace.skeletonize(m, min_length=0.0)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    in_small = ((v[:, 0] < 15)).sum()
    in_large = ((v[:, 0] >= 15)).sum()
    assert in_small > 0, "the small leading component was not traced"
    assert in_large > 0, "the large component was not traced (single-root bug)"


def test_component_crop_does_not_inflate_radii():
    """A component flush against the array edge: the crop must pad with background, or
    the EDT at that face is too large."""
    m = np.zeros((30, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 0) & (zz < 30) & ((yy - 10) ** 2 + (xx - 10) ** 2 <= 9))] = 1
    skel = neutu_trace.skeletonize(np.asfortranarray(m))
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_min_length_rejects_short_spurs_but_keeps_the_trunk():
    """NeuTu's `minimalLength` rejects a branch by the new territory it covers: a tiny
    bump on a tube is dropped and the trunk survives."""
    m = _tube(length=50, r=3, pad=5)
    m = np.asfortranarray(m.copy())
    m[28:30, 12:14, 4] = 1                                  # a tiny nub

    strict = neutu_trace.skeletonize(m, min_length=10.0)
    loose = neutu_trace.skeletonize(m, min_length=0.0)
    assert len(np.asarray(strict.vertices)) <= len(np.asarray(loose.vertices))
    v = np.asarray(strict.vertices)
    assert v[:, 0].max() - v[:, 0].min() >= 40, "the trunk was truncated too"


def test_min_length_is_measured_on_uninvalidated_length_not_geometry():
    """Only the trailing run of still-valid voxels counts."""
    path = np.array([[i, 0, 0] for i in range(10)])
    labels = np.ones((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels) == pytest.approx(9.0)

    labels_covered = labels.copy()
    labels_covered[:8] = 0                      # only the last 2 are new
    assert neutu_trace._uninvalidated_length(path, labels_covered) == pytest.approx(1.0)

    labels_all = np.zeros((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels_all) == 0.0


def test_uninvalidated_length_survives_uint32_paths():
    """dijkstra3d returns uint32 paths, and np.diff on a decreasing coordinate wraps
    to ~2**32, which silently disables branch rejection."""
    labels = np.ones((12, 12, 12), dtype=np.uint8)
    down = np.array([[9, 5, 5], [8, 5, 5], [7, 5, 5], [6, 5, 5]], dtype=np.uint32)
    assert neutu_trace._uninvalidated_length(down, labels) == pytest.approx(3.0)

    diag = np.array([[9, 9, 5], [8, 8, 5], [7, 7, 5]], dtype=np.uint32)
    assert neutu_trace._uninvalidated_length(diag, labels) == pytest.approx(
        2 * np.sqrt(2))


def test_branches_are_traced():
    """A Y gives a branch point (a degree-3 vertex)."""
    skel = neutu_trace.skeletonize(_y_body())
    e = np.asarray(skel.edges)
    degree = np.bincount(e.ravel(), minlength=len(np.asarray(skel.vertices)))
    assert (degree >= 3).any(), "no branch point found on a Y-shaped body"


def test_root_seeds_from_the_thickest_voxel_not_memory_order():
    """Seed at the thickest voxel (NeuTu's rebase), so the tree does not depend on
    memory order. The thickest part here is deliberately not first."""
    import edt

    thin_z0, thin_z1, blob_z, blob_r = 2, 40, 48, 8
    m = np.zeros((60, 30, 30), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= thin_z0) & (zz < thin_z1)
      & ((yy - 15) ** 2 + (xx - 15) ** 2 <= 4)] = 1                        # thin, first
    m[((zz - blob_z) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2)
      <= blob_r ** 2] = 1                                                  # thick, later
    m = np.asfortranarray(m)
    dbf = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")

    # the sphere's centre is its deepest point, by construction
    thickest = np.unravel_index(int(np.argmax(dbf)), dbf.shape)
    assert thickest[0] == blob_z, "test body is not built as intended"

    from_thick = neutu_trace._find_root(m, dbf)
    from_order = neutu_trace._find_root(m, None)
    assert from_thick is not None and from_order is not None
    # the root is the voxel farthest from the seed: the thin cylinder's far tip
    assert from_thick[0] == thin_z0


def test_find_root_ignores_zero2inf_background():
    """The raw EDT must be passed: after zero2inf the argmax is background."""
    import edt
    import kimimaro.skeletontricks as skt

    m = _tube(length=30, r=3)
    raw = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")
    poisoned = skt.zero2inf(raw.copy())
    root = neutu_trace._find_root(m, poisoned)
    assert root is not None
    assert m[tuple(int(v) for v in root)], "root fell outside the object"


@pytest.mark.parametrize("attach", [False, True])
def test_attach_at_skeleton_keeps_the_tree_connected(attach):
    """The cut at the path mask is usually not an existing vertex, so a branch needs an
    anchor or it comes back as its own component."""
    skel = neutu_trace.skeletonize(_y_body(), attach_at_skeleton=attach)
    v = np.asarray(skel.vertices)
    parent = list(range(len(v)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.asarray(skel.edges):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    assert len({find(i) for i in range(len(v))}) == 1


def test_path_mask_radius_is_capped():
    """The skeleton tube is min(EDT + 1, SKELETON_RADIUS), not the full EDT."""
    import edt

    m = _tube(length=30, r=8)          # fat tube: EDT well above the cap
    dbf = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")
    pm = np.zeros(m.shape, dtype=bool)
    centre = np.array([[15, 12, 12]], dtype=np.uint32)
    neutu_trace._stamp_path_mask(pm, centre, dbf)
    # a cap of 3 cannot reach 6 voxels away even though the EDT there is large
    assert not pm[15, 12, 12 + 6]
    assert pm[15, 12, 12]


TUBE_RADIUS = 4          # the fixture's radius; bounds below are derived from it


@pytest.mark.parametrize("bend", [False, True])
def test_fix_borders_makes_adjacent_blocks_meet_at_the_face(bend):
    """Two halves of a tube, skeletonized independently, meet at the same point on the
    shared face, which is what lets stage 2 weld them with a seam-scale join.

    - each endpoint lies on its block's face, within one voxel;
    - a straight tube gives exactly zero offset (both faces expose the same disc);
    - a bent tube's offset stays within the cross-section, ``2 * r``, since adjacent
      blocks do not share a plane;
    - ``fix_borders`` never makes the seam worse than without it.
    """
    r = TUBE_RADIUS
    m = _tube(length=60, r=r, pad=4, bend=bend)
    cut = m.shape[0] // 2
    left = np.asfortranarray(m[:cut].copy())
    right = np.asfortranarray(m[cut:].copy())

    def endpoint(mask, face, fix):
        skel = neutu_trace.skeletonize(mask, fix_borders=fix)
        v = np.asarray(skel.vertices, float)
        assert len(v)
        j = int(np.argmin(np.abs(v[:, 0] - face)))
        return v[j], abs(v[j, 0] - face)

    lp, ld = endpoint(left, left.shape[0] - 1, True)
    rp, rd = endpoint(right, 0, True)
    assert ld <= 1.0 and rd <= 1.0, "a fragment did not reach the face"
    off_fix = float(np.linalg.norm(lp[1:] - rp[1:]))

    lp0, _ = endpoint(left, left.shape[0] - 1, False)
    rp0, _ = endpoint(right, 0, False)
    off_no = float(np.linalg.norm(lp0[1:] - rp0[1:]))

    if not bend:
        assert off_fix == 0.0, (
            f"identical contact discs must give the same target, got {off_fix:.2f}")
    assert off_fix <= 2 * r, (
        f"seam offset {off_fix:.2f} exceeds the tube cross-section {2 * r}")
    assert off_fix <= off_no, (
        f"fix_borders made the seam worse: {off_no:.2f} -> {off_fix:.2f}")


def test_border_targets_sit_on_the_faces_and_inside_the_body():
    """Targets must be on a face of the array and inside the object."""
    import cc3d

    m = _tube(length=30, r=4, pad=0)          # pad=0 so the tube touches z faces
    cc = cc3d.connected_components(np.asfortranarray(m), connectivity=26)
    bt = neutu_trace.border_targets(cc)
    assert bt, "no border targets found on a tube that touches the faces"
    for label, pts in bt.items():
        for p in pts:
            z, y, x = (int(v) for v in p)
            assert m[z, y, x], "border target is outside the object"
            on_face = (z in (0, m.shape[0] - 1) or y in (0, m.shape[1] - 1)
                       or x in (0, m.shape[2] - 1))
            assert on_face, f"border target {(z, y, x)} is not on a face"


def test_short_boundary_crossing_stub_survives_min_length():
    """A branch crossing a block face gets a mandatory border target, exempt from
    min_length, so per-block rejection cannot delete long-range cable. Here a ~4-voxel
    spur through the y-max face survives min_length=10 only with fix_borders."""
    m = np.zeros((40, 30, 30), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= 5) & (zz < 35) & ((yy - 25) ** 2 + (xx - 15) ** 2 <= 9)] = 1   # trunk
    m[(np.abs(zz - 20) <= 1) & (yy >= 25) & ((xx - 15) ** 2 <= 4)] = 1      # spur
    m = np.asfortranarray(m)
    assert m[:, -1, :].any(), "spur does not reach the y-max face"

    def reaches_face(fix):
        skel = neutu_trace.skeletonize(m, fix_borders=fix, min_length=10.0)
        v = np.asarray(skel.vertices, float)
        return bool(len(v)) and bool((v[:, 1] >= m.shape[1] - 2).any())

    assert reaches_face(True), (
        "a boundary-crossing spur was dropped even with fix_borders; per-block "
        "min_length would then delete long-range cable")
    assert not reaches_face(False), (
        "test is vacuous — the spur survives without fix_borders, so it does not "
        "demonstrate the exemption")


def test_keep_single_object_prevents_silent_body_loss():
    """NeuTu's keepingSingleObject: if min_length rejects every branch, keep the longest,
    so a small body is not lost from stage 1 without a record."""
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 9] = 1   # tiny blob
    m = np.asfortranarray(m)

    kept = neutu_trace.skeletonize(m, min_length=50.0, keep_single_object=True)
    dropped = neutu_trace.skeletonize(m, min_length=50.0, keep_single_object=False)
    assert len(np.asarray(kept.vertices)) > 0, "body lost despite keep_single_object"
    assert len(np.asarray(dropped.vertices)) == 0, (
        "test is vacuous — the body survives even with the guard off, so it does "
        "not demonstrate anything")
