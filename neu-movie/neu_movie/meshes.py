"""Meshes for neuVid: OBJ files in the view's voxel units, smoothed, with vertex normals.

They are cached by source volume, smoothing and level of detail, and reused by every build.
A volume rewritten at the SAME location needs `force` (the cache cannot tell).
"""

import hashlib
import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def safe_name(label):
    """A file and neuVid-name safe form: `LA(L)` -> `LA_L`, `chain-1` -> `chain1`.

    neuVid's own cleaning (`LA(L)` -> `LAL`) would collide with the real LAL, and its mesh
    expressions use `-` as set difference, so names may not contain one.
    """
    return re.sub(r"[^A-Za-z0-9_]", "", label.replace("(", "_"))


def volume_tag(url):
    """A cache-directory tag for a volume: its last path part plus a digest of the whole URL."""
    digest = hashlib.sha1(url.rstrip("/").encode()).hexdigest()[:8]
    return f"{safe_name(url.rstrip('/').rsplit('/', 1)[-1])}_{digest}"


def vertex_normals(vertices, faces):
    """Area-weighted vertex normals. Without them Blender shades every face flat."""
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(n, faces[:, k], face_n)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def taubin(vertices, faces, iterations, lam=0.5, mu=-0.53):
    """Taubin smoothing (uniform Laplacian, shrink then inflate), as neuVid's fetcher does."""
    if not iterations:
        return vertices
    from scipy.sparse import coo_matrix, diags
    n = len(vertices)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0                               # duplicate edges summed; count once
    walk = diags(1.0 / np.maximum(adj.sum(axis=1).A1, 1)) @ adj
    v = vertices.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (walk @ v - v)
        v = v + mu * (walk @ v - v)
    return v


def write_obj(path, vertices, faces):
    """Wavefront OBJ with vertex normals, 1-based faces, written to a temporary name first so
    a killed run leaves no partial file."""
    tmp = path + ".part"
    with open(tmp, "w") as f:
        np.savetxt(f, vertices, fmt="v %.3f %.3f %.3f")
        np.savetxt(f, vertex_normals(vertices, faces), fmt="vn %.4f %.4f %.4f")
        np.savetxt(f, np.repeat(faces + 1, 2, axis=1), fmt="f %d//%d %d//%d %d//%d")
    os.replace(tmp, path)


def largest_component(vertices, faces):
    """The mesh's largest connected piece (by faces), vertices reindexed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(vertices)
    edges = coo_matrix((np.ones(faces.size), (faces.ravel(), np.roll(faces, 1, 1).ravel())),
                       shape=(n, n))
    _, labels = connected_components(edges, directed=False)
    face_labels = labels[faces[:, 0]]
    keep = faces[face_labels == np.bincount(face_labels).argmax()]
    used, inverse = np.unique(keep, return_inverse=True)
    return vertices[used], inverse.reshape(keep.shape)


def export(volume, sid, path, unit_nm, lod, *, force=False, one_piece=False, smooth=0,
           offset_copy=None, log=sys.stderr):
    """One segment's mesh to `path`, in units of `unit_nm`. Returns its face count, or None
    when it was already there.

    `offset_copy=(path2, distance)` also writes the mesh moved `distance` units along its
    vertex normals. Two coincident copies of a surface (a lit head and its silhouette,
    cross-faded) make ray hits ambiguous, and the render then differs by GPU, run and frame:
    a horizontal cut that moves between renders of one frame, and flicker.
    """
    from neu_vol.readback import read_body_mesh
    targets = [path] + ([offset_copy[0]] if offset_copy else [])
    if all(os.path.exists(p) for p in targets) and not force:
        return None
    got = read_body_mesh(volume, int(sid), lod=lod)
    if got is None:
        raise RuntimeError(f"{volume}: no mesh for segment {sid}")
    vertices_nm, faces, used = got
    vertices_nm, faces = np.asarray(vertices_nm), np.asarray(faces)
    total = len(faces)
    if one_piece:
        vertices_nm, faces = largest_component(vertices_nm, faces)
    vertices_nm = taubin(vertices_nm, faces, smooth)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vertices = vertices_nm / unit_nm
    write_obj(path, vertices, faces)
    if offset_copy:
        copy_path, distance = offset_copy
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        write_obj(copy_path, vertices + distance * vertex_normals(vertices, faces), faces)
    dropped = f", {total - len(faces):,} in stray pieces dropped" if total > len(faces) else ""
    print(f"  {os.path.basename(path)}: {len(faces):,} faces (LOD {used}{dropped})", file=log)
    return len(faces)


def export_bodies(volume, ids, directory, unit_nm, lod, *, smooth, force=False, workers=16,
                  log=sys.stderr):
    """Every body in `ids` to `directory/<id>.obj`, in parallel, skipping ones present."""
    os.makedirs(directory, exist_ok=True)
    todo = [b for b in ids
            if force or not os.path.exists(os.path.join(directory, f"{b}.obj"))]
    print(f"exporting {len(todo)} of {len(ids)} neuron meshes to {directory}", file=log)

    def one(b):
        # One line per body is noise here. Not redirect_stderr: that swaps sys.stderr for
        # the whole process, and threads restoring it out of order can leave it swallowed.
        return export(volume, b, os.path.join(directory, f"{b}.obj"), unit_nm, lod,
                      force=True, smooth=smooth, log=io.StringIO())

    faces = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, n in enumerate(pool.map(one, todo), 1):
            faces += n or 0
            if i % 250 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ({faces:,} faces so far)", file=log)
