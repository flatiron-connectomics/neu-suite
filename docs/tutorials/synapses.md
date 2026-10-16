# Synapses in neuroglancer, pre and post in two colours

From a DVID synapse instance to a link that shows the selected neuron's output synapses in
one colour and its inputs in another. Four commands; the
[viewing guide](../guides/viewing.md#a-synapse-layer-for-a-whole-dataset) explains each
choice in depth.

You need a DVID node with a synapse annotation instance and a precomputed segmentation
exported from the same node, so body ids agree. The paths below are placeholders.

## 1. Choose the bodies

DVID cannot cheaply list every body worth fetching, so the list comes from its synapse index:

```bash
neu-mark select-bodies --src dvid://dvid.example.org/abc123/synapses \
    --min-synapses 10 --out bodies.csv
```

## 2. Fetch their synapses into tables

```bash
neu-mark points --src dvid://dvid.example.org/abc123/synapses \
    --bodies bodies.csv --out syn/ --connections
```

`syn/` gets a `points` table (one row per T-bar or PSD) and, with `--connections`, a
`connections` table pairing each T-bar with its partners. The tables are the durable
record: the next step can be rerun with other settings without refetching.

## 3. Write the annotation source

```bash
neu-mark annotation-source --tables syn/ \
    --dst s3://my-bucket/seg/synapses --voxel-size 8,8,8 --verify 100
```

One **line** per connection, from the presynaptic site to the postsynaptic one, indexed by
`body_pre` and `body_post` so a viewer can fetch one body's synapses directly.
`--voxel-size` is the segmentation's, so the lines land on it. `--verify 100` reads 100
annotations back through the source's own `info`: a wrong key would otherwise leave the
viewer empty while every file looks correct.

## 4. Make the link

```bash
neu-glance gen --seg s3://my-bucket/seg --annotations s3://my-bucket/seg/synapses \
    --annotation-split --annotation-shader synapse --segments 12345 --format url
```

`--annotation-split` adds the source as two layers, each showing only the selected bodies'
synapses: one filtered on `body_pre` (its outputs, drawn as presynaptic markers) and one on
`body_post` (its inputs, drawn as postsynaptic markers). The `synapse` shader colours
presynaptic sites red-orange and postsynaptic sites blue.

Neither split layer draws the connecting lines: the shader draws a line only when both of
its ends are shown, and at most zooms a line is sub-pixel and swamps the markers. Tick the
other end's checkbox (`show_pre` or `show_post`) in a layer to get its lines, in white.
Without `--annotation-split` there is one layer with both ends and the lines, showing the
selected bodies' inputs and outputs together.

## In the viewer

- **Nothing shows until a body is selected.** The layers follow the segmentation's
  selection; `--segments` selects some up front. To show everything, untick the filter in
  a layer's **Annotations** tab, or pass `--no-filter-by-segmentation`.
- **The colours, marker sizes and a confidence threshold** are controls in each layer's
  **Rendering** tab (`pre_color`, `post_color`, `pre_size`, `post_size`, `min_conf`).
- **Synapses with no confidence value stay visible at every threshold.** They are mostly
  added by hand, so unknown confidence is not treated as low.
- **Some lines have a partner outside `bodies.csv`.** They are kept, since the synapse is
  real; `annotation-source --drop-partial` leaves them out.

A link carries its whole state, so it can get long. `--format state --out state.json` writes
the state instead, and neuroglancer opens one stored at a URL given after `#!`
(`…/#!s3://my-bucket/seg/state.json`) if the object is readable without credentials and its
bucket allows CORS.
