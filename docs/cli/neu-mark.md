# `neu-mark`

From `neu-mark`. DVID annotations into columnar tables, and on into neuroglancer:
synapses and other point annotations, and the per-body records that carry a neuron's name
and status.

`python -m neu_mark` is equivalent.

Every fetch takes an explicit `--bodies` list, because DVID has no cheap way to enumerate
the bodies worth asking about — `select-bodies` is how you build one, from the synapse-count
index DVID already maintains.

```{argparse}
:module: clitools
:func: em_annot_parser
:prog: neu-mark
```
