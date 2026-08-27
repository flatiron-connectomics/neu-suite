# Rendering locally in a notebook

Everything on this page is `neu-draw`, the only package in the suite that draws pixels. It
has no command: it is a library you call from a notebook, and it renders on your own GPU
through pygfx.

**It is not a rival to `neu-glance`, and the axis between them is where the rendering
happens.** `neu-glance` composes a state for a remote neuroglancer; `neu-draw` puts the
image in the cell. Reach for `neu-glance` to share a view or to proofread against the raw
labels, and for `neu-draw` when you want a figure, or to look at forty cells at once
without a browser in the way.

```bash
conda activate neu-env
pip install --no-deps -e ./neu-draw      # extras: render, sources, dvid, cache
```

## Read some bodies, show them

`neu_draw.sources` reads a published precomputed volume — the same one `neu-morpho`
wrote, or a foreign one — and hands back `neu-lib` geometry.

```python
import neu_draw

VOL = "s3://.../gt_v2"                              # or a local path
bodies = [1401, 1402, 1403]

meshes = neu_draw.body_meshes(VOL, bodies)
skels  = neu_draw.body_skeletons(VOL, bodies)

scene = neu_draw.build_scene(meshes=meshes.values(), skeletons=skels.values())
view  = neu_draw.show(scene)
```

`show()` returns a `View`, and in a notebook it also puts a canvas with eight buttons and a
legend in the cell. Nothing has to be switched on.

Three things the readers do that are worth knowing before you trust a picture:

- **They apply the source's own `transform`.** Identity is what this suite *writes*, not
  what it reads — a published FlyEM mesh source declares `diag(16)`, and ignoring it
  returns meshes 16× too small and 16× out of register with the skeletons beside them,
  with nothing to signal it. There is no opt-out, because the return is named in
  nanometres and a flag making that sometimes false is the failure the rule exists to
  prevent.
- **They handle a sharded subresource.** Published sources shard their meshes, so there is
  no per-body object to fetch at all, and an unsharded read reports every body absent —
  indistinguishable from a volume that holds none.
- **`skip_missing` is about bodies, not about the source.** A body with no mesh is
  skipped; a structural problem — a subresource that is not there, a format the reader
  cannot decode, a keyword the reader does not take — is raised. Every one of those looks
  exactly like "no meshes here" if it is swallowed.

Fetching is threaded and shard readers are reused across a batch, so ask for the whole
list in one call rather than looping. Pass `cache=` to keep the bytes between kernel
restarts.

## Lining up cells that have no reason to line up

Cells from different datasets have real, unrelated coordinates. Arranging them is a
property of the drawable, and **the vertices are never touched** — physical nanometres
stay the one model space, so `mesh.bbox` keeps reporting where the tissue is rather than
where the thing is drawn. It also copies nothing, however large the mesh.

```python
scene.arrange(along="x")                        # a row, packed by each cell's own extent
scene.arrange(along="x", wrap=5, down="z")      # a 5-wide grid
scene.superimpose(axes="z").arrange(along="x", align_cross=False)
```

That last line is the one worth stealing: align on depth, spread horizontally, and leave y
where it really is. **A layout that regularises all three axes throws away whatever the
axes meant** — if soma depth or layer position carries information, tiling in every
direction destroys it.

## One row per cell type, not per body

A `name` is identity — unique, what `scene.get` and `set_color` resolve. A `label` is the
legend row, and it does **not** have to be unique. That is what makes forty bodies of one
type one row, one colour and one click, while still leaving forty addressable drawables.

```python
labels = {1401: "Tm2", 1402: "Tm2", 1403: "LC6"}      # keyed on the BODY, so one entry
scene = neu_draw.build_scene(meshes=meshes.values(),  # covers its mesh and its skeleton
                             skeletons=skels.values(),
                             labels=labels)
# → two rows, "Tm2 (2)" and "LC6", not six
```

Passing `labels` also switches colouring to **one colour per label**. Without that the
palette hands out forty colours behind a swatch that can only show one, which is most of
the value of grouping thrown away; `color_by="name"` opts out.

In the canvas: **left-click a row to hide it, right-click to highlight it.** A highlight is
a display override, so the body's real colour is untouched underneath and reappears when
the highlight comes off. A group that is only partly hidden shows a third appearance,
because a row claiming either extreme would be lying about half its members.

## Edits land without re-running `show()`

```python
scene.relabel({"1401 mesh": "Tm2"})   # a Scene method → repaints on its own
view.legend.recolor("Tm2", "orange")  # a legend method → immediate
scene.get("1403 mesh").visible = False
view.refresh()                        # a field set directly → nothing knew, so ask
```

The first two need nothing further. The third cannot be automatic: a `Scene` is a plain
mutable dataclass, and `drawable.visible = False` has nobody to notify. **Half-automatic
notification would be worse than none** — you would learn to rely on it and then meet the
case it misses — so the methods notify honestly, every frame re-reads the scene anyway,
and `view.refresh()` (or the **Refresh** button) covers the case where nothing is going to
draw a frame.

## Getting a figure out

```python
view.save("figure.png")           # supersampled; the legend is in it, the buttons are not
img = view.snapshot()             # the same pixels as an array
```

The dividing line between the legend and the buttons is exactly this: **the legend is
drawn in the canvas because it belongs in the PNG, and the buttons are widgets above it
because they do not.** A figure without its legend is not the figure.

Camera angles are kept in `neu_draw.views`, outside any view, because the point of saving
one is to use it in the *next* figure:

```python
view.save_view("dorsal")
...
neu_draw.show(scene, viewpoint="dorsal")   # or "last" — where the last CLOSED figure was
```

`View.close()` writes `views["last"]` on the way out, so wanting an angle back after a
figure is gone needs no foresight. An empty slot restores nothing and does not raise,
which is what makes `viewpoint="last"` safe in a session's first cell.

## Synapses

`neu-draw` takes synapses as a **table**, and never imports `neu-mark` — so anything that
produces a DataFrame of positions works, `neu-mark points` included.

```python
points = neu_draw.synapse_points(conns, frame=neu_draw.volume_frame(VOL, 0))
scene  = neu_draw.build_scene(meshes=meshes.values(), points=points)
```

The frame is what turns voxel coordinates into the nanometres everything else is in. Take
it from the volume rather than assuming a factor: `volume_frame` reads the level's real
per-axis voxel size, and real pyramids are anisotropic.

## Where the detail is

The [`neu-draw` README](../_generated/neu-draw-readme.md) is the full reference for the
legend, the toolbar and the layout methods, and the
[API reference](../_generated/api-index.md) has every signature.
