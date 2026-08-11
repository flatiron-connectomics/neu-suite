# Viewing the results in neuroglancer

## Finding the data in a sparse volume

A volume holding a few labelled boxes inside a large empty frame is hard to look at — the
boxes are needles. `em-vol annotations` emits an annotation layer with one bounding box
per written region, giving a clickable list that jumps between them.

```bash
em-vol annotations s3://.../gt_v2 --label gt              # layer JSON to stdout
em-vol annotations s3://.../gt_v2 --out layer.json        # paste into `layers`
em-vol annotations s3://.../gt_v2 --state --out state.json  # a whole loadable state
```

Paste the layer object into the `layers` array via neuroglancer's `{}` (Edit JSON state)
button. Clicking a row jumps to and selects that region; `[` and `]` step through them.

```{note}
The annotations are **local** — inline in the state — not a precomputed annotation layer,
and that is forced rather than chosen. Neuroglancer builds its annotation list by
iterating the layer's source, and `MultiscaleAnnotationSource`, the class behind every
precomputed annotation source, defines `[Symbol.iterator]` as an empty generator. A
precomputed annotation layer renders in the viewport but contributes **no rows** to the
Annotations tab, so there is nothing to click through. This is not in the format's
specification, because it is a property of the frontend class rather than the format.
```

Annotations also cannot be named from a volume's own `info` the way `mesh` and
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
