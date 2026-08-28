# `neu-glance`

From `neu-glance`. Neuroglancer viewer states, annotation layers and shareable links —
everything a *viewer* consumes, and nothing that produces data.

`python -m neu_glance.cli` is equivalent.

The three producers (`gen`, `annotate`, `bboxes`) share one output stage:
`--format {layer,state,url}` chooses the serialization, `--out` writes it somewhere instead
of stdout, and `--into` merges the new layers into an existing state given as a URL or a JSON
file — keeping that state's own `dimensions`, position and zoom, so adding a layer does not
move your view.

`serve` takes none of that stage, because what it produces is a *running server* rather than
a serialization: it hosts arrays from its own process and prints a link. A served array is
addressed `python://volume/<token>`, scoped to that process, so there is nothing to save or
share — see [the guide](../guides/viewing.md). It needs the optional `serve` extra.

`gen`, `annotate` and `bboxes` were `neu-vol ng-url-gen`, `neu-vol annotate-json` and
`neu-vol bboxes-json`. A clean break, with no aliases: the old invocations fail loudly rather
than quietly doing something slightly different.

```{argparse}
:module: clitools
:func: neu_glance_parser
:prog: neu-glance
```
