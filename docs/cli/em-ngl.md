# `em-ngl`

From `em-ngl`. Neuroglancer viewer states, annotation layers and shareable links —
everything a *viewer* consumes, and nothing that produces data.

`python -m em_ngl.cli` is equivalent.

The three producers (`gen`, `annotate`, `bboxes`) share one output stage:
`--format {layer,state,url}` chooses the serialization, `--out` writes it somewhere instead
of stdout, and `--into` merges the new layers into an existing state given as a URL or a JSON
file — keeping that state's own `dimensions`, position and zoom, so adding a layer does not
move your view.

`gen`, `annotate` and `bboxes` were `em-vol ng-url-gen`, `em-vol annotate-json` and
`em-vol bboxes-json`. A clean break, with no aliases: the old invocations fail loudly rather
than quietly doing something slightly different.

```{argparse}
:module: clitools
:func: em_ngl_parser
:prog: em-ngl
```
