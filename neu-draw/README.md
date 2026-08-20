# neu-draw

Interactive 3D rendering of EM segment morphology in Jupyter — body **meshes** and
**skeletons**, synapse points, and ROI meshes — drawn locally with
[pygfx](https://github.com/pygfx/pygfx).

Its sibling [neu-glance](https://github.com/flatiron-connectomics/neu-glance) builds
neuroglancer states for a *remote* viewer. The axis that separates the two is **where
the rendering happens**, not what is rendered.

```python
import neu_draw
from neu_draw import build_scene, sources
from neu_draw.cache import MemoryCache

volume = "s3://my-bucket/my-dataset/segmentation…"
cache = MemoryCache()

skels = sources.body_skeletons(volume, [10014014, 10017394], cache=cache)
meshes = sources.body_meshes(volume, [10014014], cache=cache)

scene = build_scene(meshes=list(meshes.values()), skeletons=list(skels.values()))
view = neu_draw.show(scene)                  # displays itself in a notebook
view.save("bodies.png")                    # or renders offscreen anywhere
```

Cropping a skeleton to a region, and drawing it with its real calibre:

```python
skel = skels[10014014].crop(sources.box_predicate(some_box))   # ends at the surface
tube = sources.skeleton_tube(skel)                             # radii as a solid tube
```

## Status

Early, but it works end to end against real precomputed volumes. Geometry, colours,
scene assembly, the pygfx backend and the source readers are in place; the legend, DVID
sources and 2D projections are not. See `NEU-DRAW-PLAN.md` in the `neu-suite` root.

## Layout

| module | what it is |
|---|---|
| `neu_draw.geometry` | `Frame`, `Mesh`, `Skeleton`, `BBox`. nm, zyx, arrays. No I/O, no renderer. |
| `neu_draw.colors` | palette and colour assignment. Pure — no matplotlib. |
| `neu_draw.scene` | drawables, colours, camera intent. Pure, so a figure is assertable without a GPU. |
| `neu_draw.sources` | the only module that reads anything. |
| `neu_draw.cache` | a three-method protocol; in-memory by default, `yes3` optional. |
| `neu_draw.backends.pygfx` | the only renderer-aware module. |

## Those `AuthCredentialsProvider` lines are not errors

Reading from S3 prints two ERROR-severity lines the first time each prefix is opened —
tensorstore reporting the credential providers it could not build before falling through
to the one that works. **The marker of a real problem is `PERMISSION_DENIED` or
`AccessDenied`**; without one, the read succeeded.

```python
neu_draw.install_quiet_stores()          # for the session; remove_quiet_stores() to undo
with neu_draw.quiet_stores(): ...        # or scoped
```

A deny-list of known-benign strings, not a severity filter: anything unrecognised passes
through, and a genuine failure prints even though it looks identical. Nothing in neu-draw
installs it implicitly — filtering fd 2 is process-wide, and a library that reassigns
its caller's stderr is how an unrelated traceback goes missing.

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

Part of the `neu-suite` suite, whose packages live as siblings and are installed into
one conda environment:

```bash
conda activate neu-env
pip install --no-deps -e ./neu-draw
```

`--no-deps` is load-bearing across this suite: numpy and tensorstore come from
conda-forge, and letting pip re-resolve them invites an ABI mismatch.

Extras: `render` (pygfx, rendercanvas, jupyter-rfb, cmap), `sources` (neu-morpho,
pandas), `dvid` (neuclease), `cache` (yes3), `dev` (pytest).

**Importing `neu_draw` does not import a renderer**, and must not — a headless machine has
no canvas backend, and importing one there raises rather than degrading. A test asserts
this.

## Tests

```bash
python -m pytest -q
```
