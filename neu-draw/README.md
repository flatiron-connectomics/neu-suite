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

## The legend

Each row is a glyph saying what kind of thing it holds, and a label.

- **Left-click a row to hide it**, click again to bring it back. Hidden rows are dimmed
  rather than removed, since a hidden drawable with no row is one nobody can turn back on.
- **Right-click a row to highlight it** — those bodies turn white in the scene and the row
  lights up, until you right-click again.

Those are the two questions a crowded scene raises: *what does it look like without this*,
and *which one of these is this*.

It is on by default and docked to the right. From a notebook:

```python
view.legend.relabel("1401 mesh", "MeCN-01 (L)")  # change what a row says
view.legend.relabel({"a": "Tm2", "b": "Tm2"})    # a mapping — and this MERGES the two rows
view.legend.recolor("Tm2", "tab:pink")           # every body on the row, and its swatch
view.legend.toggle("presyn")                     # what left-clicking the row does
view.legend.set_visible("presyn", False)

view.legend.highlight("Tm2")                     # what right-clicking does
view.legend.highlight("LC6", exclusive=True)     # …and drop the others
view.legend.highlighted                          # ['LC6']
view.legend.clear_highlights()

view.center()                                    # re-fit to what is left showing
```

**A highlight is a display override and never touches the drawable's colour** — the same
rule the placement offsets follow. `scene.get(name).color` still says what colour the body
is, so recolouring a highlighted entry is meaningful and shows up when the highlight comes
off, and a saved scene never has a temporary highlight baked into it. (The fastplotlib
predecessor swapped the real colour and stashed the original, which holds right up until
something else reads it.) The drawable's alpha is preserved too: highlighting a translucent
surface must not turn it opaque, since being translucent is often why you could not find it.

`Legend(highlight_color=...)` is the knob for a figure whose own palette collides with
white — a body that is already white lights up only its row.

### Names and labels

**A row is a group, and several drawables can share one.** That needs two ideas, because
one cannot do both jobs:

| | what it is | unique? | addressed by |
|---|---|---|---|
| `name` | a drawable's **identity** | yes — a duplicate raises | `scene.get(name)`, `scene.set_color` |
| `label` | its **legend row**, defaulting to the name | no | `legend[label]`, `scene.by_label` |

So forty bodies of a cell type are one row, one colour and one click, and are still forty
individually addressable drawables underneath. Grouping could not be done by relaxing the
name rule: a many-to-one relation cannot be an identity.

```python
scene = build_scene(meshes=meshes, skeletons=skels,
                    labels={10014014: "Tm2", 10017394: "Tm2", 10022881: "LC6"})
# → two rows reading "Tm2 (4)" and "LC6 (2)", not six
```

`labels` is keyed on each item's **own** name — the body id, before `build_scene`'s
`mesh`/`skeleton` suffix — so one entry covers every representation of that body, which is
what "label this body's geometry as Tm2" has to mean. Keys match by value or by `str()`,
since a body id is an `int` as often as a string. Unlisted items keep a row of their own.

**Passing `labels` also switches colouring to one colour per label**, because forty palette
colours behind a swatch that can show one is most of the value of grouping thrown away.
`build_scene(..., color_by="name")` opts out; `Scene.recolor(by="label")` does it after the
fact.

Afterwards, from either side:

```python
scene.relabel({"1401 mesh": "Tm2"})      # keyed on drawable NAMES, for bulk assignment
view.legend.relabel({"Tm2": "Tm2 (L)"})  # keyed on the row text you can see
scene.rename({"a": "b", "b": "a"})       # identity, still unique; a mapping applies as a batch
```

`relabel` sets the label and leaves names alone, so `scene.get("1401 mesh")` keeps working.
It replaced a `rename` method on the legend: once a row can hold several drawables its text
is a label and not a name, and renaming one member would not change the row at all.

### Edits take effect where you make them

**Nothing needs re-running `show()`, and in the usual cases nothing needs a refresh either.**
Three routes, in the order you are likely to meet them:

| what you did | when it shows |
|---|---|
| `view.legend.relabel(...)`, `.recolor(...)`, `.toggle(...)` | immediately |
| `scene.relabel(...)`, `.rename(...)`, `.set_color(...)`, `.recolor(...)`, `.add(...)` | immediately — the scene asks for a repaint |
| `scene.get("a").label = "Tm2"` / `.visible = False` / `.color = ...` | on the next frame drawn, or now with `view.refresh()` / the **Refresh** button |

The last row is the one that cannot be automatic: a `Scene` is a plain mutable dataclass, so
assigning a field cannot notify anybody, and making every field observable would mean
properties on all of them. Half-automatic notification would be worse than none — you would
learn to rely on it and then meet the case it misses. So the backend closes the gap from the
other side: **every frame re-reads the scene** and rebuilds the rows if the labels changed,
or reassigns materials if only colours and visibility did. That is two cheap tuple
comparisons per frame, and it means no edit can stay invisible for longer than one repaint —
including in `snapshot()`, so `view.save("f.png")` straight after a relabel writes the new
text.

`view.refresh()` (and the **Refresh** button) forces it now, for when you have set fields
directly and nothing is going to draw a frame on its own.

A group's swatch is the **first member's** colour, since one swatch cannot show forty — and
where the point of grouping is that a type shares a colour, they agree anyway. A group of
mixed kinds (a type group normally holds meshes *and* skeletons) gets a plain square rather
than a line or a marker claiming to speak for all of it. A **partly** hidden group gets a
third row appearance, distinct from both on and off, and left-clicking it hides the rest.

**The legend is drawn in the canvas, not beside it, and that is the whole design
constraint.** A figure without its legend is not the figure, so it has to be part of what
`snapshot()` and `save()` produce — which rules out widgets and settles everything else:
pygfx objects rendered in a second pass into a strip, clicks resolved by the renderer's own
pick buffer, and the camera controller registered on a *viewport* limited to the scene's
rect so that clicking a row does not also spin the camera.

Sizes are logical pixels, so the panel does not grow or shrink with the camera:

```python
neu_draw.show(scene, legend=False)                          # off entirely
Scene(legend=Legend(location="left", font_size=16))
Scene(legend=Legend(width=200, panel_color="w", text_color="k"))   # for a light figure
```

Entries wrap into columns when a single column would not fit, and the whole legend shrinks
if even that is not enough — it is never clipped, because in a figure there is nobody to
scroll it. The strip never takes more than 45% of the canvas.

## The toolbar, and saved viewpoints

In a notebook `show()` puts eight buttons above the canvas — no argument needed, and
nothing to remember after a kernel restart:

| button | what it does |
|---|---|
| **Center** | re-fit the camera to everything *currently visible* |
| **Reset** | back to the view this figure opened with — everything shown, no highlights |
| **Refresh** | re-read the scene, for a field you set directly on a drawable |
| **Save** | remember this camera as `views["saved"]`, for *any* later figure |
| **Restore** | go to `views["saved"]` |
| **Last** | go to `views["last"]` — where the last **closed** figure was |
| **Capture** | write the PNG named in the box below the buttons |
| **Close** | close the canvas, leaving the image it last showed in its place |

Every one is also a method, so a notebook or a script can do the same:

```python
view.center()
view.reset()
view.refresh()
view.save_view()                 # or save_view("dorsal"), for as many slots as you like
view.restore_view("dorsal")      # returns None, not an error, if nothing is there
view.restore_view("last")
view.save("figure.png")
```

**A viewpoint is kept outside the view, in `neu_draw.views`, and that is the point** — you
save an angle in order to use it in the *next* figure, by which time the one you saved it
from is gone. It records the canvas size along with the camera, because a perspective
camera's field of view follows the rect's aspect ratio, so the same camera in a differently
shaped canvas is a different picture.

**`View.close()` writes `views["last"]` on the way out**, so wanting the angle back after a
figure is gone needs no foresight — and it is in `close()` rather than in the Close button,
so a `view.close()` from a cell counts too. Two slots and two buttons, deliberately: they
answer different questions ("the angle I chose" against "wherever I happened to be"), and
one button picking between them would leave it unclear which you got.

To skip the button entirely — re-run a cell and come back on the same angle:

```python
view = neu_draw.show(scene, viewpoint="last")     # or "saved", or a ViewState you held
```

An empty slot just means the figure keeps its own framing, so that is safe in the first
cell of a session.

**Reset is not Center.** `center()` fits *what is visible now*, so after hiding half the
bodies it frames the remainder; `reset()` un-hides whatever you hid, drops every highlight,
and returns to the camera the figure opened at — including a `viewpoint=` it was opened
with. It deliberately leaves **colours** alone: hiding and highlighting are transient
exploration, but `legend.recolor` is an authored change, and a button that silently reverted
it would be destroying work rather than tidying up.

`toolbar=False` gives the bare canvas; `toolbar=True` insists and raises if ipywidgets is
missing or the canvas is not a widget. Offscreen and desktop renders quietly get no
toolbar, which is why the buttons never appear in a `snapshot()` — the legend is drawn in
the canvas instead, precisely so that it does.

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
scene assembly, the pygfx backend, the source readers, the clickable legend and the
notebook toolbar are in place; DVID sources and 2D projections are not.

| module | what it is |
|---|---|
| (from `neu_lib`) | `BBox`, `Frame`, `Mesh`, `Skeleton`, the region predicates, `skeleton_tube`. nm, zyx, arrays; re-exported here so a notebook can say `neu_draw.Skeleton` |
| `neu_draw.colors` | palette and colour assignment |
| `neu_draw.scene` | drawables, colours, camera intent — no renderer |
| `neu_draw.sources` | the only module that reads anything |
| `neu_draw.cache` | a three-method protocol; in-memory by default, `yes3` optional |
| `neu_draw.viewstate` | saved camera viewpoints, outliving the view they came from |
| `neu_draw.backends.pygfx` | the renderer seam — build, camera, canvas, snapshots |
| `neu_draw.backends.legend` | the clickable legend, drawn in the canvas |
| `neu_draw.toolbar` | the notebook buttons; the only module that needs ipywidgets |

## Tests

```bash
python -m pytest -q
```
