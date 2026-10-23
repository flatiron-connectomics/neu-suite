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
    --annotation-shader synapse --segments 12345 --format url
```

One layer shows the selected bodies' synapses, inputs and outputs together. The `synapse`
shader colours presynaptic sites red-orange and postsynaptic sites blue, with a yellow line
between them. Its `show_pre` / `show_post` boxes show either end alone (without the line),
and the layer's **Annotations** tab filters on each relationship separately: `body_pre`
alone gives the selected bodies' outputs, `body_post` alone their inputs.

With `annotation-source --roi-spec SPEC.toml` (and points fetched with `--rois`), the source
also carries each synapse's ROI and top-level neuropil, and the shader gains
`color_by_neuropil` and `color_by_neuropil_part`; `color_by_neuron_id` and
`color_by_tbar_fanout` need nothing extra.

## In the viewer

- **Nothing shows until a body is selected.** The layers follow the segmentation's
  selection; `--segments` selects some up front. To show everything, untick the filter in
  a layer's **Annotations** tab, or pass `--no-filter-by-segmentation`.
- **The colours, marker sizes and a confidence threshold** are controls in the layer's
  **Rendering** tab (`pre_color`, `post_color`, `line_color`, `pre_size`, `post_size`,
  `min_conf`), next to the `color_by_*` boxes.
- **Synapses with no confidence value stay visible at every threshold.** They are mostly
  added by hand, so unknown confidence is not treated as low.
- **Some lines have a partner outside `bodies.csv`.** They are kept, since the synapse is
  real; `annotation-source --drop-partial` leaves them out.

A link carries its whole state, so it can get long. `--format state --out state.json` writes
the state instead, and neuroglancer opens one stored at a URL given after `#!`
(`…/#!s3://my-bucket/seg/state.json`) if the object is readable without credentials and its
bucket allows CORS.
