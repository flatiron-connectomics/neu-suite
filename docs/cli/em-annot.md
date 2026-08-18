# `em-annot`

From `em-annotation`. DVID annotations into columnar tables, and on into neuroglancer:
synapses and other point annotations, and the per-body records that carry a neuron's name
and status.

`python -m em_annotation` is equivalent.

Every fetch takes an explicit `--bodies` list, because DVID has no cheap way to enumerate
the bodies worth asking about — `select-bodies` is how you build one, from the synapse-count
index DVID already maintains.

```{argparse}
:module: clitools
:func: em_annot_parser
:prog: em-annot
```
