"""Write neuroglancer precomputed meshes and skeletons (stage-2 output).

Meshes: :func:`write_mesh_info` (the ``info``, once) and :func:`write_body_multires`
(per-LOD octree fragments), encoded by ``vol2mesh.multires``. Skeletons: the
``neuroglancer_skeletons`` format via osteoid's ``Skeleton.to_precomputed`` —
:func:`write_skeleton_info` and :func:`write_body_skeleton`.

**Axis order.** Model space is physical nm, held *zyx* in memory; both formats store
*xyz*. vol2mesh flips meshes internally and :func:`encode_skeleton` flips skeletons;
getting it wrong mirrors skeletons through the z=x diagonal relative to their meshes.

**Every write goes through neu-vol's kvstore helpers**, never ``open()``, so
``output_dir`` may be a local path or an ``s3://`` URL. Hence ``multires.build_info`` /
``encode_multilod_object`` (a dict and bytes), never ``write_info`` /
``write_object_mesh``, which write to the local filesystem.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from neu_vol import exists, read_json, write_bytes, write_json

# Re-exported: linking the subresources after the seg stage is this package's job, and
# neu-vol, which owns a volume's `info`, has the one implementation.
from neu_vol.ops.subresources import link_subresources  # noqa: F401

from .config import MeshConfig

logger = logging.getLogger(__name__)
# Declared in the skeleton ``info``; every blob carries exactly these, in this order.
# float32 only (see :func:`_check_vertex_attributes`), so osteoid's default uint8
# ``vertex_types`` is not emitted.
SKELETON_VERTEX_ATTRIBUTES = [
    {"id": "radius", "data_type": "float32", "num_components": 1},
]

# The spec permits integer types, but neuroglancer uploads skeleton attributes as WebGL
# vertex attributes and rejects the whole layer for anything but float32.
WEBGL_SKELETON_ATTRIBUTE_DTYPES = frozenset({"float32"})

IDENTITY_TRANSFORM = [1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0]


VALID_QUANTIZATION_BITS = (10, 16)      # neuroglancer multiresolution mesh spec


def quantization_step_nm(cfg: MeshConfig, chunk_shape_xyz: Sequence[float]) -> float:
    """Position resolution (nm) at the COARSEST LOD — the one that bites.

    Draco spreads ``2**bits`` steps across each octree cell, and a LOD-l cell is
    ``chunk * 2**l`` wide, so the step doubles per LOD.
    """
    coarsest_cell = max(chunk_shape_xyz) * (2 ** max(0, cfg.num_lods - 1))
    return coarsest_cell / (2 ** cfg.draco_quantization_bits)


def check_quantization(cfg: MeshConfig, chunk_shape_xyz: Sequence[float],
                       voxel_size_zyx: Sequence[float], *, min_ratio: float = 4.0) -> None:
    """Fail fast if Draco quantization would collapse triangles at a coarse LOD.

    When the coarsest LOD's step approaches the voxel size, decimated triangles quantize
    to zero area and Draco rejects the whole body ("All triangles are degenerate"); at
    10 bits and scale 2 that was 29% of bodies.
    """
    if cfg.draco_quantization_bits not in VALID_QUANTIZATION_BITS:
        raise ValueError(
            f"draco_quantization_bits must be one of {VALID_QUANTIZATION_BITS} "
            f"(neuroglancer spec), got {cfg.draco_quantization_bits}. Intermediate "
            "values encode fine but neuroglancer will not read them.")

    step = quantization_step_nm(cfg, chunk_shape_xyz)
    finest = min(voxel_size_zyx)
    if step * min_ratio > finest:
        raise ValueError(
            f"Draco quantization step at the coarsest LOD is {step:.3g} nm, too close to "
            f"the {finest:.3g} nm voxel — triangles will collapse and bodies will fail to "
            f"encode. Raise draco_quantization_bits (currently "
            f"{cfg.draco_quantization_bits}; 16 is the max the spec allows), or reduce "
            f"num_lods ({cfg.num_lods}) or block_shape ({tuple(cfg.block_shape)}).")


def write_mesh_info(output_dir: str, cfg: MeshConfig, *, transform=None,
                    lod_scale_multiplier: float = 1.0) -> None:
    """Write the multi-resolution mesh ``info`` (once per output volume)."""
    from vol2mesh import multires

    info = multires.build_info(vertex_quantization_bits=cfg.draco_quantization_bits,
                               transform=transform,
                               lod_scale_multiplier=lod_scale_multiplier)
    write_json(output_dir, info, "info")


def _encode_cell(cell, position, lod: int, cfg: MeshConfig,
                 chunk_shape_xyz: Sequence[float], grid_origin_xyz: Sequence[float]) -> bytes | None:
    """One octree cell's Draco bytes, or ``None`` if it has nothing to draw.

    Cutting a body at the cell planes can leave a sliver under one quantization step
    across alone in a cell. Every triangle of it quantizes to zero area, and DracoPy 2
    raises ("Invalid mesh") where vol2mesh expects an empty result, failing the body.
    Such a cell is dropped; any other encoding failure still raises.
    """
    from vol2mesh import multires

    vertices_xyz = np.asarray(cell.vertices_zyx, dtype=np.float64)[:, ::-1]
    faces = np.asarray(cell.faces, dtype=np.int64)
    if not len(faces):
        return None
    q = multires.quantize_fragment_vertices(
        vertices_xyz, position, chunk_shape_xyz, grid_origin_xyz, lod,
        cfg.draco_quantization_bits).astype(np.int64)
    corners = q[faces]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    if not area.any():
        return None
    return multires.encode_fragment(
        cell, position, chunk_shape_xyz, grid_origin_xyz, lod=lod,
        vertex_quantization_bits=cfg.draco_quantization_bits)


def write_body_multires(output_dir: str, body_id: int, mesh, cfg: MeshConfig,
                        *, chunk_shape_xyz: Sequence[float], grid_origin_xyz: Sequence[float]) -> int:
    """Write one body's multi-resolution mesh; returns fragments written (0 if empty).

    LOD 0 is the assembled mesh; each coarser LOD decimates by
    ``cfg.lod_decimation_factor`` and is octree-partitioned. Writes ``<body>`` and
    ``<body>.index`` (unsharded multires). Vertices are nm; the ``info`` transform is
    identity. Cells count from ``grid_origin_xyz``, which must lie below every vertex.
    """
    from vol2mesh import multires

    # vol2mesh splits on a grid from 0 nm, so split relative to the grid origin and move
    # each cell back before encoding, which quantizes against that origin.
    origin_zyx = np.asarray(grid_origin_xyz, dtype=np.float64)[::-1]
    mesh.vertices_zyx = mesh.vertices_zyx - origin_zyx
    fragments_by_lod: dict[int, dict] = {}
    lod_mesh = mesh
    for lod in range(cfg.num_lods):
        if lod > 0:
            try:
                lod_mesh.simplify(1.0 / cfg.lod_decimation_factor)   # coarsen (mutates)
            except Exception as exc:                               # noqa: BLE001
                logger.warning("body %d: simplifying for LOD %d failed (%s: %s); LOD %d "
                               "keeps LOD %d's detail", body_id, lod, type(exc).__name__,
                               exc, lod, lod - 1)
        cells = multires.split_mesh_for_lod(lod_mesh, list(chunk_shape_xyz), lod)
        for cell in cells.values():
            cell.vertices_zyx = cell.vertices_zyx + origin_zyx
        fragments_by_lod[lod] = {
            pos: enc for pos, cell in cells.items()
            if (enc := _encode_cell(cell, pos, lod, cfg, chunk_shape_xyz, grid_origin_xyz))}

    data_bytes, index_bytes, counts = multires.encode_multilod_object(
        fragments_by_lod, list(chunk_shape_xyz), list(grid_origin_xyz),
        vertex_quantization_bits=cfg.draco_quantization_bits)
    if not data_bytes:
        return 0

    # Data before index: a torn write then leaves an unreferenced blob, not an index
    # pointing at missing data.
    write_bytes(output_dir, data_bytes, str(int(body_id)))
    write_bytes(output_dir, index_bytes, f"{int(body_id)}.index")
    return sum(counts)


# --------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------- #
def volume_exists(volume_dir: str) -> bool:
    """True if a precomputed ``info`` is present at ``volume_dir`` (one request; works on
    object stores, where there is no directory to stat)."""
    return exists(volume_dir, "info")


def _check_vertex_attributes(attrs: Sequence[dict]) -> None:
    """Reject attribute dtypes neuroglancer cannot render (spec-legal, but the viewer fails
    the whole layer with "Data type not supported by WebGL")."""
    bad = [a for a in attrs if a.get("data_type") not in WEBGL_SKELETON_ATTRIBUTE_DTYPES]
    if bad:
        names = ", ".join(f"{a.get('id')!r}={a.get('data_type')!r}" for a in bad)
        raise ValueError(
            f"skeleton vertex attributes must be float32 for neuroglancer to render "
            f"them; got {names}. The precomputed spec allows integer types, but the "
            f"viewer uploads these as WebGL vertex attributes and rejects the layer "
            f'with "Data type not supported by WebGL". Cast the attribute to float32 '
            f"or drop it.")


def write_skeleton_info(output_dir: str, *, transform: Sequence[float] | None = None,
                        vertex_attributes: Sequence[dict] | None = None) -> None:
    """Write the ``neuroglancer_skeletons`` ``info`` (once per output volume).

    The transform is **identity**: vertices are already physical nm in the same
    model space as the meshes, so neuroglancer must not re-scale them.
    """
    attrs = list(vertex_attributes or SKELETON_VERTEX_ATTRIBUTES)
    _check_vertex_attributes(attrs)
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": list(transform if transform is not None else IDENTITY_TRANSFORM),
        "vertex_attributes": attrs,
    }
    write_json(output_dir, info, "info")


def encode_skeleton(skeleton) -> bytes:
    """Encode a global-nm **zyx** skeleton as precomputed **xyz** bytes.

    The vertex attributes are normalized to :data:`SKELETON_VERTEX_ATTRIBUTES`, so the
    blob matches the ``info``. Unknown radii stay ``-1``.
    """
    from osteoid import Skeleton

    _check_vertex_attributes(SKELETON_VERTEX_ATTRIBUTES)
    verts_zyx = np.asarray(skeleton.vertices, dtype=np.float32).reshape(-1, 3)
    verts_xyz = np.ascontiguousarray(verts_zyx[:, ::-1])          # <-- the flip
    out = Skeleton(vertices=verts_xyz,
                   edges=np.asarray(skeleton.edges, dtype=np.uint32).reshape(-1, 2),
                   segid=getattr(skeleton, "id", None))
    n = len(verts_xyz)
    radius = getattr(skeleton, "radius", None)
    if radius is not None and len(radius) == n:
        out.radius = np.asarray(radius, dtype=np.float32)
    # exactly what info declares, in that order — to_precomputed walks this list
    out.extra_attributes = [dict(a) for a in SKELETON_VERTEX_ATTRIBUTES]
    return out.to_precomputed()


def write_body_skeleton(output_dir: str, body_id: int, skeleton) -> int:
    """Write one body's skeleton blob; returns the vertex count (0 if empty)."""
    if skeleton is None or len(skeleton.vertices) == 0:
        return 0
    data = encode_skeleton(skeleton)
    write_bytes(output_dir, data, str(int(body_id)))
    return len(skeleton.vertices)
