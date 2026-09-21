# How `neu-eval` was calibrated

Every default in this package that could have gone another way, with the measurement that
decided it. Written so that nobody has to re-derive these, and so that a number quoted from
`neu-eval` can be traced to why it means what it says.

Measurements come from two sources: a wasp specimen's ground-truth crops (13 annotated
regions, ~250³ voxels each at 8 nm) compared against a cleaned version of themselves, and
two externally produced segmentations of a second specimen compared against its annotation.
Dataset locations are deliberately absent — get them from the site config.

---

## The numbers are right

Before any calibration question, the metrics were checked against an independent route:
`pandas.crosstab` for the table, `scipy.stats.entropy` for the entropies, and an explicit
per-body loop for covering. On one real annotated region, agreement to **nine decimal
places** on `voi_split`, `voi_merge`, Rand precision, Rand recall and covering.

Deliberately *not* checked against an existing segmentation-metrics library: such a library
shares the one mistake this code is most likely to make — getting the VOI direction
backwards — and would agree enthusiastically about it. The direction is instead pinned by
hand-built pure-split and pure-merge cases where the answer is one bit by construction.

**Additivity** — `c(A) + c(B) == c(A ∪ B)` for disjoint regions — was checked on all 13
real annotated regions as well as on random arrays at several split points and on all three
axes. It is exact, and the blockwise sweep will rest on it.

---

## The biggest calibration decision is what `0` means

**Nothing in an array tells you whether `0` is "unlabelled" or "membrane", and the two
answers differ by up to a factor of 7.**

The defaults assume the reference's `0` is unannotated (dropped) and the segmentation's `0`
is a body the pipeline failed to assign (kept, because it is an error and not an absence).
That is right for a dense segmentation. It is badly wrong for the common convention where
membranes are labelled `0` and cell interiors carry body ids: scored as a body, that `0`
becomes one enormous segment threading between every cell in the volume, touching nearly
every reference body at once, and registering as a colossal false merge.

Measured on two externally produced segmentations, `ignore_b=(0,)` against `ignore_b=()`:

| region | correct | membrane scored as a body | inflation |
|---|---|---|---|
| A | 0.629 | 1.375 | 2.2× |
| B | 0.512 | 2.009 | 3.9× |
| C | 0.962 | 2.164 | 2.3× |
| D | 0.288 | 1.985 | 6.9× |

All of it artefact. The caller has to know and say; there is no safe guess and no attempt
at one.

**The guard against the opposite abuse is `frac_scored`,** which is why it appears in every
summary. A pipeline could otherwise "win" by marking most of the volume membrane. On those
same two segmentations it reads 93.6% for one and 84.5–87.1% for the other — a real
difference in how much each declined to label, and one that belongs beside their scores
rather than buried. **A VOI over 85% of a region is not comparable to a VOI over 94% of
it.**

---

## Ranking: severity by default, and what it is biased toward

Rows are ranked by their contribution to VOI. That is what the metric itself says a
disagreement costs, so it is the right answer to "what should I fix to move the number".

It is *not* the right answer to "which body is most broken", and the gap is large. Measured
on a constructed pair where both failures are present:

| failure | voxels | severity |
|---|---|---|
| a 600-voxel body cut clean in half | 300 + 300 | 0.003 |
| a 100,000-voxel body losing a 6% sliver | 6,000 | **0.242** |

Eighty times the score for the less serious failure, because VOI weights by size — the
standard criticism of the metric, inherited in full. `rank="fraction"` orders by how much
of a body or segment the pair accounts for instead. On real data the two orderings share
only **3 of their top 10**, so the option earns its place.

Severity stays the default because it is the honest answer to what the score is made of,
and because a third tool is better for "which bodies are worst": `per_label.parquet` ranks
bodies by their share of the blame directly, rather than re-sorting pairs.

---

## The speck filter has to be asymmetric

Bodies in these volumes are genuinely fragmented — one measured body had **344 connected
components** at one scale, only 7 of them ten voxels or larger. So an unfiltered "is this
body split?" answers yes for every body, and the disagreement table is useless.

The filter that suggests itself — "the pair is a real fraction of *either* side" — does not
work. A one-voxel stray segment sitting inside a body is 0.1% of the body but **100% of
itself**, so it passes on its own fraction. Measured on that exact case (one body, one good
segment, twenty single-voxel strays): the symmetric filter reports **21 rows**, the correct
filter reports **none**.

So whether a body is *split* is judged on fractions **of that body**, and whether a segment
is a *merge* on fractions **of that segment**. Default threshold 5%.

---

## Points go on the seam, not in the overlap

A located disagreement first pointed at the deepest interior voxel of `a ∩ b`. That is the
middle of the part the two labelings **agree** about — arguably the least informative voxel
in the pair.

Points now go where the two part company: a `split` row's point goes on the false *cut*
(the overlap's boundary inside the reference body), a `merge` row's on the false *join*
(its boundary inside the segment), and a `tangle` has both with the wider contact winning.
Of the seam's voxels it returns the one deepest inside the containing object, so the point
lands mid-cut rather than where the cut grazes the object's own surface — a one-voxel-thick
contact sheet has no interior of its own to discriminate by.

**Two things the seam depends on, both found by tests rather than by reasoning:**

The other side must be another labelled thing **in the same container**. Requiring only
"another segment" made a fragment separated by a gap in the body report a seam against
something outside the body entirely — not a false cut, and not where anyone should be sent.

Ignored labels cannot be the other side. Where `0` is membrane and ignored, counting it
would put every split point on an ordinary cell boundary: correct, and useless.

Verified on **600 real disagreements** — every point inside its pair *and* adjacent to the
other side of the seam it claims:

| kind | seam chosen | n |
|---|---|---|
| merge | `seam-merge` | 313 |
| split | `seam-split` | 167 |
| tangle | `seam-merge` | 73 |
| tangle | `seam-split` | 21 |
| any | `overlap` fallback | 26 |

The 26 fallbacks (4.3%) are pairs whose pieces genuinely never touch — disconnected
fragments, which have no seam. They report `point_at: overlap` rather than returning a
seam-shaped answer, and all 26 are still inside their pair.

Of the 94 tangles, the wider-contact tiebreak chose the merge seam 73 times: when a pair
fails both ways on this data, the false join is usually the bigger surface.

---

## Adjudication is not monotonic

Excluding a disagreement the reviewer judged to be a *reference* defect does not simply
improve the score. Measured on one region, excluding two pairs:

```
raw:          VOI 0.1400   Rand err 0.0086   85.9% scored
adjudicated:  VOI 0.1403   Rand err 0.0083   81.0% scored
```

VOI rose slightly while Rand error fell. Removing a large well-matched pair changes the
entropy normalisation, so this is correct rather than a bug — worth knowing before reading
a small rise as a regression.

Both scores are always reported, because an adjudicated score alone is unfalsifiable: it is
what you get after deciding which disagreements not to count. `ambiguous` is deliberately
still scored — "I could not tell" is not evidence the reference was wrong, and excluding it
is how an adjudicated score drifts upward one judgement call at a time.

---

## Making the one pass fast

Everything is a pure function of one contingency table, so only the pass over the voxels
costs anything. Three changes, on a 21.7 Mvoxel uint64 crop with 1,316 × 296 labels:

**Factorizing dominated the CPU path.** `np.unique(return_inverse=True)` is a sort:
**3,434 ms**. `fastremap.renumber` is hash-based and linear: **116 ms**. Whole CPU pass
7.26 s → 0.41 s, a bigger win than the GPU's and one that CI and every GPU-less worker
gets. The GPU still uses `cupy.unique`, where the sort is parallel and the question does
not arise; the two agree up to the arbitrary numbering of the dense indices, which a test
pins rather than assumes.

**A label ignore applies to the finished table, not to the voxels.** An ignore is a union of
whole rows and columns, so dropping those rows from the table gives exactly the table of the
surviving voxels — `O(pairs)` instead of a boolean pass plus a fancy-index compaction of
both labelings. That compaction measured **142 ms on a crop whose reference side contained
no `0` at all**. The default `ignore_a=(0,)` is now free.

**Counting picks a strategy by size.** `bincount` over packed keys is `O(voxels)` but
allocates `n_a × n_b` cells occupied or not; above 2²² cells (32 MB) it switches to sorting
the packed keys. A test runs both paths over the same input and asserts the tables are
identical, because the choice is invisible at the call site.

**The GPU is a workstation lever, not a cluster one:**

| problem | CPU | GPU resident | + transfers |
|---|---|---|---|
| 21.7 Mvox, 1.3k × 0.3k labels | 414 ms | **35 ms** (11.9×) | 116 ms (3.6×) |
| 173 Mvox, 10.5k × 2.4k labels | 5.03 s | **300 ms** (16.8×) | 948 ms (5.3×) |

Both scale linearly, and the sort path favours the GPU more. It is kept because it costs
one branch in one function and the interactive loop is score → look → re-score, where 12×
on a sub-second operation is the difference between fluid and not. It is *not* the answer
for a whole-volume sweep, which is I/O bound and parallelises across CPU workers, where
there is one GPU per workstation.

Because a contingency table is `O(pairs)` and the input is `O(voxels)`, the result returns
to CPU memory at the boundary. Every metric downstream is plain numpy with no backend
awareness at all.

---

## Open

**The severity/fraction split is a workaround, not a resolution.** Both are defensible and
they disagree on most of the top ten. The literature's own advice is to read several
measures rather than pick one, which is an argument for the current arrangement — but
nobody has checked which ordering a proofreader actually finds more useful on a real
worklist.

**`min_frac=0.05` is a default, not a finding.** It was chosen to make the speck case come
out right and has never been swept. The measurement that would settle it is the same shape
as the skeleton tick-threshold sweep neu-morpho still owes.

**Expected run length and connectivity correctness are not implemented.** Both are
connectomics-specific and both need a reference that is not voxels — skeletons for ERL,
synapse tables for connectivity. The survey these metrics come from argues they track
downstream utility better than anything here does, so the voxel metrics should be read as
necessary rather than sufficient.
