# Viewing the results in neuroglancer

## Finding the data in a sparse volume

A volume holding a few labelled boxes inside a large empty frame is hard to look at — the
boxes are needles. `em-vol bboxes-json` emits an annotation layer with one bounding box
per written region, giving a clickable list that jumps between them.

```bash
em-vol bboxes-json s3://.../gt_v2 --label gt              # layer JSON to stdout
em-vol bboxes-json s3://.../gt_v2 --out layer.json        # local path or s3://...
em-vol bboxes-json s3://.../gt_v2 --state --out state.json  # a whole loadable state
```

Paste the layer object into the `layers` array via neuroglancer's `{}` (Edit JSON state)
button. Clicking a row jumps to and selects that region; `[` and `]` step through them.

`--tighten-level` defaults to `--level`, so the boxes are exact in the level-0 voxels they
are reported in. Raise it if the occupied footprint is large enough that reading it at
full resolution is slow — each level is a factor cheaper, at the price of quantizing every
bound to one voxel there.

## Annotating coordinates you already have

`bboxes-json` asks the volume where its data is. When you already know where to look — a
synapse table, a list of ROIs, points from another tool — `em-vol annotate-json` puts
those in the same kind of layer.

```bash
em-vol annotate-json --volume s3://.../seg --points synapses.csv --out syn.json
em-vol annotate-json --volume s3://.../seg --boxes rois.csv --lines pre_to_post.csv
em-vol annotate-json --volume s3://.../seg --point 5700,4500,6800 --name spot
cat table.csv | em-vol annotate-json --volume s3://.../seg --points -
```

Points, boxes, lines and ellipsoids, from CSV files or inline flags, and one layer may
mix them. CSV columns are read **by name**, so a table with its own column order and
extra columns needs no preparation:

| kind | columns |
| --- | --- |
| `--points` | `z,y,x` |
| `--boxes`, `--lines` | `z0,y0,x0,z1,y1,x1` |
| `--ellipsoids` | `z,y,x,rz,ry,rx` |

Any of them may also carry `id`, `description` and `segments`. `segments` is the useful
one: whitespace- or comma-separated body ids, and clicking the annotation then selects
those bodies. Ids are kept as strings, because a 19-digit body id does not survive a JSON
number — `annotate-json` refuses a `segments` value that arrives as `1.23e+18`, which is
what a spreadsheet does to one.

```{warning}
**Coordinates are level-0 voxels unless you say otherwise**, and this is the mistake the
command exists to catch. Coordinates in the wrong unit are still *valid* annotations —
they are simply somewhere else in the volume, and nothing renders as an error.

`--scale N` converts from scale-N voxels, using the volume's own recorded per-level voxel
sizes rather than an assumed `2**N`: on a `(1, 2, 2)` pyramid, z does not convert like y
and x, so the assumption moves annotations into the wrong plane. `--nm` converts from
physical nanometres. Whichever applies is echoed back, and any annotation falling outside
the volume's extent is reported with both flags named.
```

`--volume` is read for its voxel size, units and extent only — never its voxels. Without
one, pass `--voxel-size`, or the layer is unitless and will not line up with anything.

## Local annotations, or the precomputed annotation format

Both commands emit **local** annotations, carried inline in the viewer state. That is a
real choice with a real ceiling, and the alternative is a genuine on-store data format —
`neuroglancer_annotations_v1` — not merely a different way of writing the same thing.

| | local, inline in the state | `neuroglancer_annotations_v1` |
| --- | --- | --- |
| listed in the Annotations tab | **yes**, clickable, `[` / `]` step | **no** — zero rows |
| how many | bounded by state size: thousands | millions; spatially indexed |
| kinds per layer | any mix | **exactly one** `annotation_type` |
| filter by selected segments | renders all of them | **yes**, fetched by segment id |
| where it lives | the state you distribute | objects on the store |

The empty list is not a bug to be worked around. `AnnotationLayerView.updateView` builds
the list with `Array.from(source)` for local and multiscale sources alike, and
`MultiscaleAnnotationSource` — behind every precomputed annotation source — defines
`[Symbol.iterator]` as an empty generator. There is also no reverse panel: nothing lists
"annotations related to segment X" when X is selected, only the other direction. None of
this is in the format specification, because it is a property of the frontend class.

What the precomputed format buys instead is scale and the relationship index. Its
`spatial` levels store a probabilistic subsample (`limit / maxCount`) per grid cell, so a
zoomed-out view draws a representative subset rather than everything; and a
`relationships` index keyed on uint64 segment id is used at render time
(`segmentFilteredSources`, `forEachVisibleSegment`, `getObjectKey`), so "show only the
synapses on this body" is a keyed fetch rather than a scan.

So for a whole volume's worth of synapses the two are complementary rather than
alternatives: precomputed for the full set with a relationship index on the pre- and
post-synaptic bodies, and a small local layer for whatever subset you want to click
through. `em-annot annotation-source` writes the precomputed form — see below.

## A synapse layer for a whole dataset

`em-annot annotation-source` writes a `neuroglancer_annotations_v1` source holding one
**line** per synaptic connection, from the presynaptic site to the postsynaptic one. Lines
rather than points because a connectome is edges: one endpoint *pair* is one annotation, so
a T-bar with five partners contributes five lines and each carries its own partner body.

```bash
em-annot annotation-source --tables syn/ \
    --dst s3://bucket/sample3/seg_v1/synapses_v1 \
    --voxel-size 8,8,8 --verify
```

`--tables` is a directory an earlier `em-annot points` run wrote. Those tables are the
durable artifact, so rebuilding the source with different bounds, a different `--per-cell`
or a corrected confidence join costs no refetch. Pass `--src` and `--bodies` instead to
fetch first.

Nesting the source inside the segmentation directory, as above, is organisational only. An
annotation layer is resolved from its own URL, so unlike `mesh` and `skeletons` it is never
named in the volume's `info` and a viewer always adds it as a separate layer.

### What the properties are for

Four properties per line, in the descending-size order the format requires:
`conf_pre` and `conf_post` (float32), then `body_pre_u32` and `body_post_u32`. The body ids
are **truncated to 32 bits** because the format permits no uint64 property — they are for
shader arithmetic only. The authoritative, full-width ids live in the two relationships,
`body_pre` and `body_post`, which is what "show only this body's synapses" actually reads.

A confidence the tables do not have is written as **NaN, not 0**, and the reason is about the
data rather than the encoding: a synapse carrying no confidence value was usually **added by
hand by a human proofreader**, so it is among the *most* trustworthy annotations in the set —
its confidence was simply never quantified. Reading those as low, or dropping them, throws away
the best data. NaN gets it right mechanically, because NaN fails every comparison and so
survives a threshold test.

That puts a constraint on any shader written against these properties: express the threshold as
**discard-if-below**, never as keep-if-at-or-above. The two are equivalent for real numbers and
opposite for NaN — one leaves hand-annotated synapses visible at every setting, the other hides
them at all of them.

`annotation-source` refuses to run if the confidence columns are missing altogether, rather than
filling the property with zeros — a file that says every synapse has confidence 0, with nothing
to indicate otherwise, is the failure this guards.

### Lines with one endpoint's body unresolved

Partner bodies come from a position self-join on the fetched points, so a synapse whose
partner belongs to a body outside `--bodies` has a known *position* and an unknown *body*.
Those lines are **kept and not marked** by default: the synapse is real and the line is
where it is. The unknown side declares a zero-length relationship, so the line appears under
the body that is known and under no other. `--drop-partial` excludes them if you want only
fully-resolved connections.

### Spatial levels

The spatial index is a subsample per grid cell: zoomed out, a viewer fetches one cell and
draws a scattering rather than every synapse in the volume. `--per-cell` sets the target, and
the schedule **deepens until nothing is left over** rather than stopping at a fixed depth —
synapses are not spread evenly, and with the depth fixed in advance the finest level absorbs
every overflow and its declared `limit` balloons far past the setting, which is the number a
viewer actually downloads.

The levels **partition** the annotations — each one is emitted at exactly one level — so the
depth costs index overhead rather than duplicated data. How a level chooses what to emit is
the subtle part, and it is worth stating because getting it wrong produces a file that is
valid and renders nothing:

> `maxCount(level)` is the largest number of *remaining* annotations in any one cell.
> Each remaining annotation is emitted, independently, with probability
> `min(1, limit / maxCount(level))`.

One probability per level, applied to **every** cell. Capping each cell at `limit` instead
looks equivalent and is not: a cell holding fewer than `limit` annotations is drained
*completely* at that level and contributes nothing to any finer one, so sparse regions stop
having cells partway down the pyramid. The occupied-cell count then peaks and *falls* toward
the finest level, and a viewer zoomed in far enough to be reading those levels finds nothing
there. Measured on this dataset: the buggy rule gave level 11 **124** cells against level
10's 192; the correct rule gives **867**, rising monotonically. Nothing about the file looks
wrong either way — the tell is the falling cell count.

### Checking that a viewer will see it

`--verify N` reads `N` annotations back **through the source's own `info`** and compares
endpoints, confidences and relationships with the table, then checks one body's relationship
index in full. Everything before the write can be checked in memory; only a read-back proves
the *keys* are right, and a wrong key leaves a viewer with nothing while every byte on the
store is correct.

### Adding it to a link

`em-vol ng-url-gen --annotations` takes a precomputed annotation source and adds it as its own
layer, with a shader and — the load-bearing part — the relationships **bound** to the
segmentation layer:

```bash
em-vol ng-url-gen --seg s3://.../seg_v1 \
    --annotations s3://.../seg_v1/synapses_v1 \
    --segments 61189731 --annotation-split
```

The source keys its relationships on segment id, but neuroglancer only consults that index
once each relationship is bound to a layer whose selection it can read
(`linkedSegmentationLayer`, a map from relationship name to layer name). Without the binding
the layer draws everything and "this body's synapses" is not available at all.

`ng-url-gen` binds every relationship and sets `filterBySegmentation` **on by default**, so the
annotations track whatever you select. A link with no `--segments` therefore opens showing no
annotations until you click a body — that is the filter working, and the command says so on
stderr because an empty viewport is otherwise indistinguishable from a broken layer.
`--no-filter-by-segmentation` turns it off; in the viewer, the per-relationship checkbox is in
the layer's **Annotations** tab.

**`--annotation-split` adds two layers on the one source**, filtering one on `body_pre` and
the other on `body_post`. That is the selected body's outputs and its inputs, which a single
layer filtered on both conflates; and each half shows only its own endpoint marker, which
matters because the two overlap at any zoom showing more than a few synapses.

### The shader

Annotation shaders live in the viewer state, not in the source, so a link is the only place
one can be shipped. `--annotation-shader` takes a built-in name, a file, or `none`; the
default picks a built-in whose properties the source declares — a shader naming a `prop_` that
is absent does not degrade, it fails to compile and the layer draws **nothing**, with the
error only in the layer's shader tab.

```glsl
#uicontrol bool show_pre checkbox(default=true)
#uicontrol bool show_post checkbox(default=true)
#uicontrol float pre_size slider(min=0.0, max=20.0, default=6.0)
#uicontrol float post_size slider(min=0.0, max=20.0, default=4.0)
#uicontrol vec3 pre_color color(default="#ff2000")
#uicontrol vec3 post_color color(default="#00c0ff")
#uicontrol vec3 line_color color(default="#ffffff")
#uicontrol float min_conf slider(min=0.0, max=1.0, default=0.0, step=0.01)

void main() {
  // NaN fails every comparison, so an unknown confidence is never hidden here.
  // Unknown is not the same as low, and this is where that distinction is kept.
  if (prop_conf_pre() < min_conf || prop_conf_post() < min_conf) discard;
  setEndpointMarkerColor(vec4(pre_color, 1.0), vec4(post_color, 1.0));
  setLineColor(line_color);
  if (show_pre && show_post) {
    setLineWidth(1.0);
    setEndpointMarkerSize(pre_size, post_size);
  } else if (show_pre) {
    setLineWidth(0.0);
    setEndpointMarkerSize(pre_size, 0.0);
  } else if (show_post) {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, post_size);
  } else {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, 0.0);
  }
}
```

Three things worth knowing about this shader language:

- `discard` hides an annotation, and it works inside a helper function, so a filter can be
  factored out.
- `setLineWidth(0.0)` with a non-zero `setEndpointMarkerSize` renders a LINE source as points,
  so choosing lines at write time does not commit a viewer to drawing lines.
- **Draw the line only when both endpoints are shown.** A synapse is a few hundred nanometres
  long, so at any zoom showing more than one the line is sub-pixel — and a line coloured as a
  blend of the two endpoint colours then swamps the markers and the whole layer reads as one
  flat colour. This was a real mistake: `mix(#ff9000, #00c0ff, 0.5)` is a light green, and a
  field of orange and blue synapses came out uniformly green.

## Sharing a view as a link

```bash
em-vol ng-url-gen --image s3://.../em --seg s3://.../gt_v2 \
    --layer layer.json --segments 1,2,3 --layout xy-3d --select-last
```

The URL goes to stdout. `--layer` takes what `bboxes-json` wrote — either the bare layer
or a whole state, it uses the layers either way — so the two commands compose without
knowing about each other.

`--position` is zyx like every coordinate in these packages. Pass `--position-order xyz`
to use numbers copied straight out of the viewer, since xyz is what neuroglancer
displays. Whichever you use is echoed back on every run.

`--hide-slices` sets `showSlices: false`, hiding the cross-section planes **inside the 3D
panel** — usually what you want when the link is about meshes or skeletons, which the
slices otherwise sit across. It leaves the 2D panels alone; `--layout 3d` is what removes
those. Like `--position` and the two zooms, the key is written only when asked for, so a
link without it opens the way the viewer normally would.

```{note}
Everything after `#!` is a URL fragment and never reaches a server, so a link carries no
data anywhere. It does mean the entire state travels in the URL: a dozen inline bounding
boxes is a few thousand characters, which is fine, but a large annotation set makes a URL
that some mail and chat clients will wrap or truncate. `--state-out` writes the JSON
alongside for those cases.
```

Annotations cannot be named from a volume's own `info` the way `mesh` and
`skeletons` are, so a viewer always adds them as a separate source. A saved state is the
distribution unit either way.

## Verifying a change to `info`

```{warning}
**Check in a private/incognito window, not a hard reload.**

Two caches stack. S3 sends no `Cache-Control`, so the browser may treat an old `info` as
fresh; and neuroglancer memoises the resolved datasource per URL for the lifetime of the
page. A hard reload is not reliably enough.

The symptom mimics a data bug exactly: the labels render, there is no `mesh` row in the
source tab, and no error appears anywhere. Before investigating the store, confirm in a
private window.
```

The decisive check is devtools → Network filtered on `mesh`. If `mesh/info` is never
*requested*, the layer resolved from stale metadata and nothing on the store is wrong.
If it is requested and returns 200 but no `<id>.index` follows, neuroglancer is rejecting
the mesh metadata and the console will say why.

## What a working setup looks like

One precomputed volume whose `info` names its subdirectories, so a single layer carries
labels, meshes and skeletons together:

```text
{"type": "segmentation", "data_type": "uint64", "mesh": "mesh",
 "skeletons": "skeleton", "scales": [ ... ]}
```

Meshes are multi-LOD Draco with vertices in **physical nanometres** and an identity
transform. That is correct rather than a convention worth changing: neuroglancer converts
into the volume's voxel space itself, applying `1 / resolution[i]` to the mesh subsource.
