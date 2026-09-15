# neu-eval

Comparison metrics between a segmentation and a reference labeling — and a ranked, located
list of *where* they disagree, because a score tells you how much and not what to look at.

```bash
neu-eval compare --reference gt.h5:/crop_01 --segmentation SEG_URL --level 0 --out cmp/
```

```text
== gt.h5/crop_01
gt vs seg: VOI 0.1400 (split 0.0661, merge 0.0739), Rand err 0.0086, covering 0.9862
  | 6,870,532 voxels scored (85.9%) | 1 split, 11 merge, 3 tangle

wrote:
  cmp/summary.parquet        the metrics, one row
  cmp/per_label.parquet      each body's and each segment's share of the blame
  cmp/disagreements.csv      ranked worst-first, with a `verdict` column to fill in
  cmp/annotations.csv        the same points, for `neu-glance annotate`
```

Part of [neu-suite](https://github.com/flatiron-connectomics/neu-suite). Sits at the top
tier with [neu-draw](https://github.com/flatiron-connectomics/neu-draw); depends on
[neu-vol](https://github.com/flatiron-connectomics/neu-vol),
[neu-proc](https://github.com/flatiron-connectomics/neu-proc) and
[neu-lib](https://github.com/flatiron-connectomics/neu-lib).

## The idea

Everything hangs off **one object**: a sparse contingency table of
`(reference_id, segment_id) -> voxels`. Every voxel metric here is a pure function of that
table rather than of the arrays, which buys three things.

- **It is additive.** `c(region1) + c(region2) == c(region1 | region2)`, exactly. A
  whole-volume sweep is a block map with a driver-side sum, and no metric code changes to
  get there.
- **The expensive pass happens once.** Variation of information, adapted Rand, per-body
  decompositions, matching and completeness are then cheap and recomputable in a notebook.
  Which matters, because the literature's own advice is not to trust any single metric —
  Plaza & Funke's survey declines to recommend one and recommends reading several together.
- **It is symmetric.** Nothing below the reporting layer calls either side truth.

## What it measures

| | |
|---|---|
| `voi`, `voi_split`, `voi_merge` | variation of information, decomposed. `voi_merge = H(ref\|seg)` counts the segmentation fusing distinct bodies; `voi_split = H(seg\|ref)` counts it fragmenting one |
| `rand_precision`, `rand_recall`, `rand_f_score`, `rand_error` | adapted Rand, from pair counts |
| `covering` | size-weighted mean, over reference bodies, of the best single segment's IoU |
| `fragmentation` | extra segments, as a parameter-free edit-distance estimate |
| `n_ref_at90`, `n_seg_at90`, `fragmentation_at90` | how many labels each side needs to account for 90% of the volume |
| `n_matched_iou50` | bodies with a one-to-one segment match at IoU ≥ 0.5 |
| `frac_scored` | **read this next to every other number** — a VOI over 3% of a box is not comparable to one over 95% |

Per-label blame is in `per_label.parquet`: each reference body's share of the merge error
and each segment's share of the split error, which is what localizes a score to a body.

Not yet: expected run length and the synapse-based connectivity metrics. Both need a
skeleton or synapse reference rather than voxels.

## The disagreement table

The point of the package. One row per structurally significant `(body, segment)` pair,
ranked worst-first, each with a world coordinate you can paste into a viewer.

`kind` is `split` (one body across several segments), `merge` (one segment over several
bodies), or `tangle` (both at once — the case a proofreader cannot fix with one action).

Two details that took measuring:

**The coordinate is the deepest interior voxel of the overlap**, not its centroid. A
centroid can fall outside a curved region, and one that lands on a boundary puts you on the
disagreement's edge rather than in it.

**Specks are filtered asymmetrically.** Bodies in these volumes are genuinely fragmented —
one measured body had 344 connected components, only 7 of them ten voxels or more — so
whether a body is *split* is judged on fractions **of that body**, and whether a segment is
a *merge* on fractions **of that segment**. The filter that suggests itself, "a real
fraction of either side", does not work: a one-voxel stray segment inside a body is 100% *of
itself*, so it passes and every body reads as split twenty-one ways.

**Ranking has a knob because the default has a known bias.** `--rank severity` (default)
orders by VOI contribution — what the metric says the disagreement costs. That inherits
VOI's size weighting, which is the standard criticism of it: measured, a 600-voxel body cut
clean in half scores 0.003 while a 100,000-voxel body losing a 6% sliver scores 0.242. So
`--rank fraction` orders by how much of a body or segment the pair accounts for, which is
what you want when hunting broken bodies rather than moving a number. Both columns are in
every row.

## Adjudication: the reference is not always right

A manual annotation and an algorithm disagree for two different reasons, and only one of
them is a segmentation error. Scoring against an annotation known to be wrong in specific
places charges the algorithm for being right.

So the disagreement table round-trips. It goes out with a blank `verdict` column; you fill
it in while looking at the voxels; passing it back reports a second score.

```bash
# the coordinates are nanometres, so annotate needs a frame -- the volume is the best one
neu-glance annotate --points cmp/annotations.csv --nm --volume REF_URL \
    --name disagreements --format layer --out layer.json
neu-glance gen --image IMAGE_URL --seg REF_URL --seg SEG_URL --layer layer.json \
    --format url            # then step the list with [ and ]
# ... fill in cmp/disagreements.csv ...
neu-eval compare --reference gt.h5:/crop_01 --segmentation SEG_URL \
    --adjudicated cmp/disagreements.csv --out cmp/
```

```text
gt vs seg: VOI 0.1400, Rand err 0.0086, 85.9% of the region scored
adjudicated: VOI 0.1403, Rand err 0.0083, 81.0% scored  [2 seg_correct, 1 ambiguous, 12 unreviewed]
```

Verdicts are `<reference>_correct`, `<segmentation>_correct`, `ambiguous` and `skip`. Only
`<segmentation>_correct` — "the reference is defective here" — leaves the denominator.

- **Both scores are always reported.** An adjudicated score alone is unfalsifiable: it is
  what you get after deciding which disagreements not to count. It means something next to
  the raw one and next to `frac_scored`.
- **`ambiguous` is still scored,** deliberately. "I could not tell" is not evidence the
  reference was wrong, and excluding it is how an adjudicated score drifts upward one
  judgement call at a time.
- **Adjudication is not monotonic.** In the run above VOI rose slightly while Rand error
  fell — removing a large well-matched pair changes the entropy normalisation. That is
  correct, and worth knowing before reading it as a regression.
- Excluding is not *accepting*: this declines to score where the reference is bad, it does
  not fold the segmentation's answer in as a corrected reference. That needs the voxels and
  would produce a new annotation volume.

## Three ways a comparison is quietly meaningless

**Unregistered pieces.** Read the segmentation with `crop=<the reference piece>`, which
converts the reference's physical box using the target level's own voxel size and origin —
never an assumed `2**level` factor, because real pyramids are anisotropic. `compare` checks
the frames agree and refuses when they do not; a half-voxel shift otherwise becomes a
metric nobody can explain.

**A `0` that means something other than what you assumed.** The reference side ignores 0 by
default (unannotated); the segmentation side does not (a 0 there is a body the pipeline
failed to assign). **But several pipelines label membranes 0**, and then the default invents
one enormous segment threading between every cell — a colossal false merge touching nearly
every body. Measured on two vendor segmentations of one specimen, `--ignore-segmentation 0`
against the default: VOI 0.63 vs 1.37, 0.51 vs 2.01, 0.96 vs 2.16, 0.29 vs 1.98 — a factor
of 2.2 to 6.9, all artefact. Nothing in the array tells the two conventions apart, so you
have to. `frac_scored` is the guard against the opposite abuse (declare everything membrane
and almost nothing gets scored), which is why it sits in every summary.

**An un-relabelled multi-crop reference.** Every annotated region numbers its bodies from
1, so the same integer names a different cell in each crop — measured on one dataset, 3,637
label-instances over 12 regions but only 1,824 distinct ids, 496 of them shared. Score that
untreated and a low-numbered "body" is a chimera of a dozen unrelated cells; every metric
comes out meaningless and nothing looks wrong. **Run `neu-vol relabel` first.** `compare`
warns when reference labels are scattered far beyond their own size, which is what this
looks like — a warning and not a refusal, since a genuinely diffuse annotation is
indistinguishable from the voxels alone.

Relatedly, `--all-datasets` **pools** rather than sums: each crop's labels are renumbered
into their own range before combining, for exactly the same reason. Pooled ids are positions
in a concatenation and do not name bodies, so the pooled row carries scalars only.

## CPU and GPU

The voxel pass runs wherever the arrays already are, dispatching through
`neu_proc.ops.backend` — on the array, never on a flag. Because a contingency table is
`O(pairs)` and the input is `O(voxels)`, the result comes back to CPU memory at the boundary
and every metric downstream is plain numpy with no backend awareness.

Measured on a 21.7 Mvoxel crop with 1,316 × 296 labels: **414 ms on the CPU, 35 ms on a
GPU-resident pair** (11.9×), 116 ms including the transfers. At a 512³-equivalent block with
10.5k × 2.4k labels, 5.03 s against 300 ms (16.8×). `NEU_PROC_GPU=0` keeps everything on the
CPU.

Worth knowing that the larger win was on the CPU side: factorizing with
`fastremap.renumber` instead of `np.unique(return_inverse=True)` took that path from 7.26 s
to 0.41 s on the same crop, which beats the GPU's margin and benefits every worker without
one.

## Install

Python 3.12. The core is numpy + scipy, and keeping it that way is deliberate: every
dependency is pip-installable, so CI runs every test rather than skipping conda-only paths.

```bash
conda activate neu-env
pip install --no-deps -e ../neu-lib -e ../neu-proc -e ../neu-vol -e .
python -m pytest -q
```

`--no-deps` is load-bearing: without it pip re-resolves conda-provided binaries
(tensorstore, h5py) from PyPI and invites an ABI mismatch. The packages depend on each other
by relative `../sibling` path, so they must stay siblings.

Extras: `report` (pandas, pyarrow — imported inside functions, so `import neu_eval` and
`--help` do not pay for them), `link` (neu-glance), `erl` (neu-morpho, for the skeleton
metrics when they land).

## Reading

- Plaza & Funke, *Analyzing Image Segmentation for Connectomics*, Front. Neural Circuits
  2018 — the metric survey this follows, including the "use several, trust none alone"
  recommendation and the advice to inspect the regions of largest difference.
- Funke, *Uncertainty Quantification for Connectomics*, Nat. Methods 2025 — the argument for
  quantifying uncertainty rather than any specific metric; it shapes the roadmap here.

## License

Apache-2.0. ©2026 The Simons Foundation, Inc.
