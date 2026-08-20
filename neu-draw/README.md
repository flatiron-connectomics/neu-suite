# em-viz

Interactive 3D rendering of EM segment morphology in Jupyter — body **meshes** and
**skeletons**, synapse points, and ROI meshes — drawn locally with
[pygfx](https://github.com/pygfx/pygfx).

Its sibling [em-ngl](https://github.com/flatiron-connectomics/em-ngl) builds
neuroglancer states for a *remote* viewer. The axis that separates the two is **where
the rendering happens**, not what is rendered.

```python
import em_viz
from em_viz import Mesh, Skeleton, build_scene

skel = Skeleton.from_precomputed(*read_body_skeleton(volume, body_id), name="KC-1")
skel = skel.crop(inside_the_lobe)          # branches end at the region surface

scene = build_scene(meshes=[soma], skeletons=[skel], points={"presyn": coords_zyx_nm})
view = em_viz.show(scene)                  # displays itself in a notebook
view.save("kc1.png")                       # or render offscreen anywhere
```

## Status

Early, but it draws. Geometry, colours, scene assembly and the pygfx backend are in
place; the source adapters and the legend are not. See `EM-VIZ-PLAN.md` in the
`em-libraries` root.

## Layout

| module | what it is |
|---|---|
| `em_viz.geometry` | `Frame`, `Mesh`, `Skeleton`, `BBox`. nm, zyx, arrays. No I/O, no renderer. |
| `em_viz.colors` | palette and colour assignment. Pure — no matplotlib. |
| `em_viz.scene` | drawables, colours, camera intent. Pure, so a figure is assertable without a GPU. |
| `em_viz.backends.pygfx` | the only renderer-aware module. |
| `em_viz.sources` | the only module that reads anything. *(not yet)* |

## Skeletons are edges, not polylines

A skeleton is vertices plus an edge list, and it renders as **one object per body**
whatever its topology — cycles, self-loops and thousands of disconnected components
included — because pygfx's `LineSegmentMaterial` consumes an edge list directly. The
predecessor decomposed each skeleton into branch polylines and emitted one graphic per
branch, purely because `fastplotlib.add_line_collection` wanted a list of them.

## Two conventions, both load-bearing

**One model space: physical nanometres.** Geometry carries a `Frame` — real per-axis
voxel sizes — never an integer scale, and never a factor derived from `2 ** scale`. Real
pyramids are anisotropic. Resolving a scale index to a `Frame` is the `sources` layer's
job.

**zyx in memory, xyz at the boundary.** Every array here is zyx; `to_xyz` is the single
conversion, called once per drawable in the backend. Getting this wrong mirrors
everything through the z=x diagonal, which looks like plausible data in the wrong place
rather than like an error.

## Install

Part of the `em-libraries` suite, whose packages live as siblings and are installed into
one conda environment:

```bash
conda activate em-lib
pip install --no-deps -e ./em-viz
```

`--no-deps` is load-bearing across this suite: numpy and tensorstore come from
conda-forge, and letting pip re-resolve them invites an ABI mismatch.

Extras: `render` (pygfx, rendercanvas, jupyter-rfb, cmap), `sources` (em-seg-morpho,
pandas), `dvid` (neuclease), `cache` (yes3), `dev` (pytest).

**Importing `em_viz` does not import a renderer**, and must not — a headless machine has
no canvas backend, and importing one there raises rather than degrading. A test asserts
this.

## Tests

```bash
python -m pytest -q
```
