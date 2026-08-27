# neu-draw

Interactive 3D rendering of segment morphology in Jupyter — body **meshes** and
**skeletons**, synapse points, and ROI meshes — drawn locally with
[pygfx](https://github.com/pygfx/pygfx).

Its sibling [neu-glance](https://github.com/flatiron-connectomics/neu-glance) builds
neuroglancer states for a *remote* viewer. The axis that separates the two is **where
the rendering happens**, not what is rendered.

```python
import neu_draw
from neu_draw import build_scene, sources
from neu_draw.cache import MemoryCache

volume = "s3://my-bucket/my-dataset/segmentation"   # or a local precomputed path
cache = MemoryCache()

skels = sources.body_skeletons(volume, [10014014, 10017394], cache=cache)
meshes = sources.body_meshes(volume, [10014014], cache=cache)

scene = build_scene(meshes=list(meshes.values()), skeletons=list(skels.values()))
view = neu_draw.show(scene)                  # displays itself in a notebook
view.save("bodies.png")                      # or renders offscreen anywhere
```

Cropping a skeleton to a region, and drawing it with its real calibre:

```python
skel = skels[10014014].crop(neu_draw.box_predicate(some_box))  # ends at the surface
tube = neu_draw.skeleton_tube(skel)                            # radii as a solid tube
```

A skeleton is vertices plus an edge list, and renders as **one object per body**
whatever its topology — cycles, self-loops and thousands of disconnected components
included.

## Coordinates

Two conventions, and getting either wrong puts your data somewhere plausible rather
than raising:

- **Physical nanometres.** Geometry carries a `Frame` of real per-axis voxel sizes,
  never an integer scale and never `2 ** scale` — real pyramids are anisotropic.
  Resolving a scale index to a `Frame` is the `sources` layer's job.
- **zyx in memory, xyz at the boundary.** Every array here is zyx. `to_xyz` is the
  single conversion, called once per drawable in the backend.

`Vec3` is the coordinate type those conventions apply to — three floats in zyx that do
arithmetic (`b.bbox.center - a.bbox.center`) and are accepted anywhere a sequence is, so
nothing that already took a tuple had to change.

## Placing and arranging

Cells from different datasets have real, unrelated coordinates. Lining them up is a
property of the **drawable**, not of the geometry:

```python
scene.get("cellB").aligned_to(scene.get("cellA"))     # centres coincide
scene.get("cellB").offset_by((0, 0, 5_000))           # nudge, in nm
```

**The offset is a display transform and the vertices are never touched.** Physical
nanometres are the one model space, so moving the data would make `mesh.bbox` report where
a thing is *drawn* rather than where the tissue is — and two datasets in one scene is
exactly when the real coordinates still need to be real. It is also free: re-placing a
million-vertex mesh copies nothing, because the backend sets the object's transform.

A whole set is laid out with two methods, which are the same operation — put each object's
anchor on a target point — differing only in how the targets are generated:

```python
scene.superimpose(anchor="min")             # every object's corner on one point
scene.arrange(along="x")                    # a row, packed by each object's own extent
scene.arrange(along="x", spacing=20_000)    # a row on a fixed 20 µm pitch
scene.arrange(along="x", wrap=5, down="z")  # a 5-wide grid
```

They compose, and that is the useful part:

```python
scene.superimpose(axes="z").arrange(along="x", align_cross=False)
```

Align on depth, spread horizontally, leave y at its true coordinates. **A layout that
regularises all three axes throws away whatever the axes meant** — if soma depth or layer
position carries information, tiling in every direction destroys it.

`anchor` is `center`, `min` or `max` throughout, so `arrange(anchor="min")` bottom-aligns a
row. `key=` sorts the layout without reordering the scene, so a legend built from
`scene.names` still matches. Hidden drawables reserve no slot. `Scene.bake()` folds the
offsets into the geometry, for when the arrangement is the output rather than the view.

## Install

Part of the [neu-suite](https://github.com/flatiron-connectomics/neu-suite) suite, whose
packages live as siblings in one conda environment:

```bash
conda activate neu-env
pip install --no-deps -e ./neu-draw
```

`--no-deps` is load-bearing across this suite: numpy and tensorstore come from
conda-forge, and letting pip re-resolve them invites an ABI mismatch.

Extras: `render` (pygfx, rendercanvas, jupyter-rfb, cmap), `sources` (neu-morpho,
pandas), `dvid` (neuclease), `cache` (yes3), `dev` (pytest).

**Importing `neu_draw` does not import a renderer**, so it is safe on a headless
machine; only `show()` and the pygfx backend need one.

## Store logging

Reading from S3 makes tensorstore print two ERROR-severity
`AuthCredentialsProvider` lines the first time each prefix is opened — the credential
providers it could not build before falling through to the one that works. They are
noise, not failures.

**Every `sources` function already filters them**, so you should not normally see them.
Two knobs if you need them:

```python
neu_draw.logs.enabled = False             # see raw store logging, benign lines included
neu_draw.install_quiet_stores()           # filter for the whole session, covering reads
                                          # made outside sources; remove_quiet_stores()
                                          # to undo, or `with neu_draw.quiet_stores():`
```

The filter is a deny-list of known-benign strings rather than a severity filter, so
anything unrecognised still prints. **The marker of a real problem is
`PERMISSION_DENIED` or `AccessDenied`**; without one, the read succeeded.

## Status

Early, but it works end to end against real precomputed volumes. Geometry, colours,
scene assembly, the pygfx backend and the source readers are in place; the legend, DVID
sources and 2D projections are not.

| module | what it is |
|---|---|
| (from `neu_lib`) | `BBox`, `Frame`, `Mesh`, `Skeleton`, the region predicates, `skeleton_tube`. nm, zyx, arrays; re-exported here so a notebook can say `neu_draw.Skeleton` |
| `neu_draw.colors` | palette and colour assignment |
| `neu_draw.scene` | drawables, colours, camera intent — no renderer |
| `neu_draw.sources` | the only module that reads anything |
| `neu_draw.cache` | a three-method protocol; in-memory by default, `yes3` optional |
| `neu_draw.backends.pygfx` | the only renderer-aware module |

## Tests

```bash
python -m pytest -q
```
