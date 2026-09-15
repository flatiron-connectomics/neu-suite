# `neu-eval`

From `neu-eval`. Comparison metrics between a segmentation and a reference labeling, and a
ranked list of *where* the two disagree.

`python -m neu_eval.cli` is equivalent.

`compare` reads both sides as two registered pieces — the segmentation is read with
`crop=<the reference piece>`, so the reference's physical box is converted using the
segmentation level's own voxel size and origin rather than an assumed `2**level` factor —
scores them, and writes four things:

| file | |
|---|---|
| `summary.parquet` | the metrics, one row per piece (plus a pooled row under `--all-datasets`) |
| `per_label.parquet` | each reference body's share of the merge error and each segment's share of the split error |
| `disagreements.csv` | ranked worst-first, with a `verdict` column to fill in, and a `.README.txt` legend beside it |
| `annotations.csv` | the same points as a coordinate list for [`neu-glance annotate`](neu-glance.md) |

**The disagreement table round-trips.** A reference is not always right: a manual annotation
and an algorithm disagree both because the algorithm is wrong and because the annotation is.
So you fill in `verdict` while looking at the voxels and pass the file back with
`--adjudicated`, which reports a second score excluding the pairs you judged to be reference
defects — *alongside* the raw one, together with how much of the region each was computed
over. Only `<segmentation>_correct` leaves the denominator; `ambiguous` is still scored,
because "I could not tell" is not evidence the reference was wrong.

**The points become local annotations, and that is deliberate.** A ranked worst-first list is
something you step through, and a precomputed annotation source renders in the viewport but
puts zero rows in the Annotations tab — so `[` and `]` do not step. `neu-glance annotate`
inlines them in the state instead. See [the viewing guide](../guides/viewing.md).

**Run [`neu-vol relabel`](neu-vol.md) on a multi-crop reference first.** Every annotated
region numbers its bodies from 1, so the same integer names a different cell in each crop;
untreated, a low-numbered "body" is a chimera of unrelated cells and every metric is
meaningless with nothing to show it. `compare` warns when reference labels are scattered far
beyond their own size, which is what that looks like. For the same reason `--all-datasets`
*pools* rather than sums — each crop's labels are renumbered into their own range first, so
pooled ids are positions in a concatenation and name no bodies.

```{argparse}
:module: clitools
:func: neu_eval_parser
:prog: neu-eval
```
