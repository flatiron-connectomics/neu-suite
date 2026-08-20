# `neu-glance`

From `neu-glance`. Neuroglancer viewer states, annotation layers and shareable links —
everything a *viewer* consumes, and nothing that produces data.

`python -m neu_glance.cli` is equivalent.

The three producers (`gen`, `annotate`, `bboxes`) share one output stage:
`--format {layer,state,url}` chooses the serialization, `--out` writes it somewhere instead
of stdout, and `--into` merges the new layers into an existing state given as a URL or a JSON
file — keeping that state's own `dimensions`, position and zoom, so adding a layer does not
move your view.

`gen`, `annotate` and `bboxes` were `neu-vol ng-url-gen`, `neu-vol annotate-json` and
`neu-vol bboxes-json`. A clean break, with no aliases: the old invocations fail loudly rather
than quietly doing something slightly different.

```{argparse}
:module: clitools
:func: neu_glance_parser
:prog: neu-glance
```
