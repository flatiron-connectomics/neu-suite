# em-viz

Interactive 3D rendering of EM segment morphology in Jupyter — body **meshes** and
**skeletons**, synapse points, and ROI meshes — drawn locally with
[pygfx](https://github.com/pygfx/pygfx).

Its sibling [em-ngl](https://github.com/flatiron-connectomics/em-ngl) builds
neuroglancer states for a *remote* viewer. The axis that separates the two is **where
the rendering happens**, not what is rendered.

```python
from em_viz import Skeleton, Frame

skel = Skeleton.from_precomputed(*read_body_skeleton(volume, body_id), name="KC-1")
skel = skel.crop(inside_the_lobe)      # branches terminate at the region surface
```

## Status

Early. The geometry layer is in place; the scene layer, the pygfx backend and the source
adapters are not yet. See `EM-VIZ-PLAN.md` in the `em-libraries` root.

## Layout

| module | what it is |
|---|---|
| `em_viz.geometry` | `Frame`, `Mesh`, `Skeleton`, `BBox`. nm, zyx, arrays. No I/O, no renderer. |
| `em_viz.scene` | drawables, colours, camera intent. Pure. *(not yet)* |
| `em_viz.sources` | the only module that reads anything. *(not yet)* |
| `em_viz.backends` | pygfx. The renderer seam. *(not yet)* |

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
