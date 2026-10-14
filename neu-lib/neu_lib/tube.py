"""A skeleton as a solid tube: one truncated cone per edge, from its radii.

Unlike lines, a tube shows calibre, and a radius too large for the body shows up as the
tube breaking through the mesh surface. The tube is built in xyz and flipped back, so its
triangle winding matches a precomputed mesh; built directly in zyx, the normals would
face inward and the surface would render dark, with no error.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .frame import to_xyz
from .mesh import Mesh
from .skeleton import Skeleton


def skeleton_tube(skeleton: Skeleton, sides: int = 8,
                  name: Optional[str] = None) -> Mesh:
    """A :class:`Skeleton` as a solid tube :class:`Mesh` (zyx), from its radii. Raises if
    the skeleton has none."""
    if skeleton.radii_nm is None:
        raise ValueError(
            f"skeleton {skeleton.name!r} carries no radii, so it has no tube. "
            f"Draw it as lines, or read a skeleton whose info declares a 'radius' "
            f"vertex attribute.")
    vertices, faces = frustum_mesh(to_xyz(skeleton.vertices_zyx_nm), skeleton.edges,
                                   skeleton.radii_nm, sides=sides)
    return Mesh.from_precomputed(vertices, faces,
                                 name=name if name is not None else skeleton.name)


def frustum_mesh(vertices_xyz, edges, radii, sides: int = 8):
    """One truncated cone per edge, in xyz; returns ``(vertices, faces)``.

    Each edge gets a ring of ``sides`` points at each end, at that vertex's radius,
    joined by a triangle strip, so the tube tapers linearly between nodes. Rings are
    perpendicular to the edge. Zero-length edges are dropped. Cones are not stitched at
    junctions; they overlap, which renders correctly as a solid surface.
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.asarray(radii, dtype=float)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not len(e):
        return empty

    p0, p1 = v[e[:, 0]], v[e[:, 1]]
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 0
    p0, p1, r0, r1, d, length = (a[keep] for a in (p0, p1, r0, r1, d, length))
    if not len(d):
        return empty

    axis = d / length[:, None]
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(axis), 1))
    ref[np.abs(axis[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    a_hat = np.cross(axis, ref)
    a_hat /= np.linalg.norm(a_hat, axis=1)[:, None]
    b_hat = np.cross(axis, a_hat)

    theta = 2.0 * np.pi * np.arange(sides) / sides
    cos_t, sin_t = np.cos(theta)[None, :, None], np.sin(theta)[None, :, None]
    disc = cos_t * a_hat[:, None, :] + sin_t * b_hat[:, None, :]      # (E, sides, 3)
    ring0 = p0[:, None, :] + r0[:, None, None] * disc
    ring1 = p1[:, None, :] + r1[:, None, None] * disc
    verts = np.concatenate([ring0, ring1], axis=1).reshape(-1, 3)

    k = np.arange(sides)
    kn = (k + 1) % sides
    tri = np.concatenate([np.stack([k, kn, sides + k], axis=1),
                          np.stack([kn, sides + kn, sides + k], axis=1)], axis=0)
    base = (np.arange(len(p0)) * 2 * sides)[:, None, None]
    return verts, (tri[None, :, :] + base).reshape(-1, 3)
