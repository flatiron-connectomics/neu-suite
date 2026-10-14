"""The multi-LOD Draco mesh reader, against an object built here from the format spec.

No meshing stack: the fragments are integer positions Draco-encoded directly, as the
format requires, and the manifest is written field by field. Round trips through the
real writer, and parity with vol2mesh's own parser, are in neu-morpho's tests.
"""

import json

import numpy as np
import pytest

DracoPy = pytest.importorskip("DracoPy")

from neu_vol.readback import read_body_mesh  # noqa: E402

CHUNK = np.array([10.0, 20.0, 40.0])          # xyz, a LOD-0 cell
ORIGIN = np.array([100.0, 200.0, 300.0])
OFFSETS = np.array([[0.0, 0.0, 0.0], [1.5, -2.0, 0.25]])   # per-LOD vertex offset
BITS = 16
TETRA_Q = np.array([[0, 0, 0], [65535, 0, 0], [0, 65535, 0], [0, 0, 65535]], np.uint16)
TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.uint32)

#: LOD -> fragment positions (xyz cells); LOD 0 has two, LOD 1 one.
FRAGMENTS = {0: [(0, 0, 0), (1, 0, 0)], 1: [(0, 0, 0)]}


def _expected(lod, position):
    cell = CHUNK * 2 ** lod
    return ORIGIN + np.asarray(position) * cell + OFFSETS[lod] + cell * (TETRA_Q / 65535.0)


def _triangles(vertices, faces):
    """Each triangle as a sorted tuple of its corners: Draco may reorder and split vertices,
    so geometry, not indexing, is what survives encoding."""
    return sorted(tuple(sorted(tuple(np.round(c, 9)) for c in tri)) for tri in vertices[faces])


def _write(tmp_path, *, transform=None, truncate=0):
    blobs, manifest = [], [CHUNK.astype("<f4").tobytes(), ORIGIN.astype("<f4").tobytes(),
                           np.uint32(len(FRAGMENTS)).tobytes(),
                           np.array([1, 2], "<f4").tobytes(), OFFSETS.astype("<f4").tobytes(),
                           np.array([len(FRAGMENTS[l]) for l in FRAGMENTS], "<u4").tobytes()]
    for lod, positions in FRAGMENTS.items():
        sizes = []
        for _ in positions:
            blob = DracoPy.encode(points=TETRA_Q, faces=TETRA_F)
            blobs.append(blob)
            sizes.append(len(blob))
        manifest.append(np.asarray(positions, "<u4").T.tobytes())   # [3][n], C order
        manifest.append(np.asarray(sizes, "<u4").tobytes())
    mesh = tmp_path / "vol" / "mesh"
    mesh.mkdir(parents=True)
    info = {"@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": BITS,
            "lod_scale_multiplier": 1.0,
            "transform": transform or [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}
    (mesh / "info").write_text(json.dumps(info))
    data = b"".join(blobs)
    (mesh / "5").write_bytes(data[:len(data) - truncate])
    (mesh / "5.index").write_bytes(b"".join(manifest))
    return str(tmp_path / "vol")


def test_each_lod_is_dequantized_against_its_own_cell(tmp_path):
    vol = _write(tmp_path)
    v, f, lod = read_body_mesh(vol, 5, lod=0)
    assert lod == 0
    # both fragments, each in its own cell: faces past the first are rebased correctly
    assert _triangles(v, f) == sorted(
        t for p in FRAGMENTS[0] for t in _triangles(_expected(0, p), TETRA_F))

    v, f, lod = read_body_mesh(vol, 5, lod=1)
    assert _triangles(v, f) == _triangles(_expected(1, (0, 0, 0)), TETRA_F)


def test_the_default_is_the_coarsest_lod_and_a_missing_one_raises(tmp_path):
    vol = _write(tmp_path)
    assert read_body_mesh(vol, 5)[2] == 1
    with pytest.raises(ValueError, match=r"LOD 4 not present; have \[0, 1\]"):
        read_body_mesh(vol, 5, lod=4)
    assert read_body_mesh(vol, 6) is None


def test_the_info_transform_maps_stored_coordinates_to_nm(tmp_path):
    vol = _write(tmp_path, transform=[2, 0, 0, 10, 0, 2, 0, 0, 0, 0, 2, 0])
    v, f, _ = read_body_mesh(vol, 5, lod=1)
    assert _triangles(v, f) == _triangles(_expected(1, (0, 0, 0)) * 2 + [10, 0, 0], TETRA_F)


def test_fragment_data_shorter_than_its_manifest_is_refused(tmp_path):
    vol = _write(tmp_path, truncate=3)
    with pytest.raises(ValueError, match="3 bytes short"):
        read_body_mesh(vol, 5, lod=1)
