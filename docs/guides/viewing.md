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
through. Writing the precomputed form is a separate piece of software — see the
`em-annotate` note in `NOTES-TODO.md`.

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
