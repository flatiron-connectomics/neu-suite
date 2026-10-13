# Skeletonization: the NeuTu-style tracer, and why it is set up this way

`neu_morpho.neutu_trace` is the default tracer. It reproduces NeuTu's skeletons
in Python rather than shelling out to NeuTu, and almost every constant in it was
chosen against a measurement. This document records those conclusions — it is
what the code's comments point at when they say "see the skeletonization notes".

Everything here was established on one specimen at scale 2 (32 nm/voxel) over a
12-body benchmark, deliberately weighted toward thick, problematic bodies. That
makes it a set of **regression fixtures**, not a sample: treat the numbers as
"what this pipeline does on hard cases", not as dataset statistics. The full
measurement rounds, the withdrawn conclusions and the figures are archived
outside the repository.

## The two tracers

| | `--tracer neutu` (default) | `--tracer kimimaro` |
|---|---|---|
| voxels | **isotropic only** | any |
| median nodes vs kimimaro | ~2.6× fewer | — |
| dropped cable, production run | 0.033 | 0.091 |

Both are TEASAR. The measured quality difference comes from a few specific
choices, not a different algorithm — which is what made reimplementing it
tractable. **Use `kimimaro` on an anisotropic pyramid**: `neutu_trace` requires
isotropic voxels, and NeuTu's own EDT is not anisotropy-aware either (anisotropy
enters only through Dijkstra step lengths).

Node economy is the large, consistent win, and it scales with thickness — from
1.3× on thin bodies to 12.7× on the thickest. Thin bodies barely differ.

### Where NeuTu differs, mechanically

All references are into the NeuTu tree; entry point
`neurolabi/gui/zstackskeletonizer.cpp:490`.

1. **Path cost is local, not normalized.** `Stack_Voxel_Weight_I`
   (`c/tz_stack_graph.c:205`) is `d/(1+r₁²) + d/(1+r₂²)` on the *squared* EDT,
   against kimimaro's globally-normalized `compute_pdrf`. Probably irrelevant
   here — the effect needs `dbf_max ≫ neurite radius`, and these masks span
   only ~4.8×.
2. **Radius-adaptive node placement** — `createSwcByRegionSampling`
   (`gui/zswcgenerator.cpp:196`). Ported as `swc_simplify.region_sample`.
3. **Radius-aware simplification to a fixpoint** — `ZSwcResampler::optimalDownsample`
   (`gui/swc/zswcresampler.cpp:90`), a radius-aware Douglas–Peucker. Ported as
   `swc_simplify.optimal_downsample`.
4. **Tighter invalidation, and length-based branch rejection.** Invalidation
   radius `EDT + 2` voxels (`gui/zspgrowparser.cpp:344`) against kimimaro's
   `1.5·DBF + 4.7`, and branches rejected by *un-invalidated geodesic length*
   rather than by post-hoc tick removal. This early termination is why NeuTu
   takes seconds where kimimaro at comparable invalidation takes minutes.
5. **Radius convention** — `AdjustedDistanceWeight`
   (`gui/zstackskeletonizer.cpp:82`), `max(0.1, √v − 0.5)`. **Deliberately not
   ported**; see below.

## How closely the port matches

It matches NeuTu on every structural property measured, using **NeuTu's own
settings** (`scale=1, const=2, minimalLength=10`) — no tuned constant:

| | port vs NeuTu |
|---|---|
| tips | 1.01× median (0.87–1.16× per body) |
| cable | 1.04–1.07× |
| branch points / mean degree / max | 66 / 3.02 / 4 — identical |
| centreline distance outside bulbs | sub-voxel, often coincident |
| radii | exact inscribed, 0.00 voxel median error |

What remains is route choice through segmentation noise inside bulbs. Neither is
closer to a real structure there, because in those regions the mask does not
contain one. **There is no measurement left that ranks them.**

## Settings, and why they are what they are

### `radii = DBF[vertex]` — do not port NeuTu's `−0.5` correction

Radii are the exact inscribed radius, as kimimaro reports them. NeuTu's
`AdjustedDistanceWeight` subtracts half a voxel, which is why its *median* radius
error is **−0.50 voxels** — it reports radii **smaller** than the inscribed
sphere. At these radii the correction removes 42–58% of ball volume and accounts
for ~9 points of fill deficit. It is established harmful here.

The inscribed radius also has the advantage of meaning something — it is the
largest sphere that fits — so it survives being reused for measurement. Both
simplification passes were checked for silently drifting away from it: after
`region_sample` and `optimal_downsample`, median error is still 0.00 voxels with
0% of nodes over by more than 2, so the published radii are exact inscribed
radii.

Note NeuTu *does* over-report a minority tail on thick bodies — up to 2.4× the
largest radius the mask can support, on 7–27% of nodes. Where that enters was
never identified; it is not in the two passes above, and we do not reproduce it.

### `COST = "edge"`

`"edge"` is NeuTu's own symmetric `d·[f(u)+f(v)]`, built explicitly and solved
with scipy. `"voxel"` uses dijkstra3d's per-voxel field, whose effective edge
cost is `d·f(destination)`. They agree only for uniform step lengths: measured
inside real bulbs, per-voxel routes cost **~10% more under NeuTu's own cost and
never matched it (0/16)**. Over the benchmark `"edge"` is closer on every axis
(cable 1.00× vs 1.07×, centreline 0.66 vs 0.83 voxels).

Its cost is small, measured **in block mode** on one real dense 256³ block with
1,170 allowlisted labels: **+3% peak memory, +5% wall** (1.62 GB / 224.9 s versus
1.57 GB / 214.4 s). An older whole-body figure claiming `"edge"` doubled memory
was wrong, and kept a measured improvement out of production for a full run.

`"edge"` is the only part of the tracer whose memory scales with the *component*,
at ~630 B/voxel — and real components are tiny next to a block (largest in that
test: 112k voxels, 0.08 GB). `SkeletonConfig.neutu_edge_max_gb` (default 16 GB)
raises rather than OOMs above that, because an OOM on a dask worker reads as
infrastructure trouble rather than a sizing problem. **The cap cannot fire at
256³**; it exists for larger blocks, where a full-block component would need
~94 GB at 512³.

### `min_length = 10` — not a tuning knob

Branch rejection by un-invalidated length is **the single largest determinant of
node count**, and it must be applied *in extraction*: pruning short twigs
geometrically afterwards is strictly dominated (swept to comparable node counts,
in-extraction reached 72–80% fill against the pruner's 45–65%). The test is on
*new territory reached*, not geometric length — a short branch into unclaimed
volume survives, a long one shadowing covered ground does not — and length alone
cannot express that. The post-hoc pruner that lost has been removed.

Un-invalidated length is bimodal, so 5 / 10 / 20 / 40 give near-identical output.
NeuTu's default of 10 is fine; **do not sweep it.**

### `const = 2` (NeuTu's `maskExpansionRadius`)

NeuTu's own value reproduces NeuTu (tip ratio 1.01× median). An earlier revision
shipped `const=8` as compensation for "weaker target selection", with a 12-body
validation behind it — **that was an artefact of two silent bugs**, and with them
fixed `const=8` over-prunes and deletes real cable (tip 0.40–0.50×). The bugs:

1. `_uninvalidated_length` measured `uint32` paths with `np.diff`, which
   underflows on any decreasing coordinate and returned ~2.7e11, so every
   `>= min_length` test passed and **branch rejection never ran at all**. The
   unit test missed it by using an int64, monotonically-increasing path.
2. The extraction loop had no progress guarantee — invalidation never erases a
   path's own voxels and `CachedTargetFinder` does not remember what it
   returned, so the same path was re-extracted until `max_paths`. kimimaro
   escapes this only via `fix_branching=True`; a `parental_field` port must
   retire the path explicitly.

**If `const` ever seems to need raising, look for a reintroduced bug first.** The
generalisable lesson: a parameter compensating for a mechanism you have not
directly verified is usually masking a defect — this one survived several rounds
of scrutiny and a figure, because every measurement was downstream of the same
two defects.

### `PATIENCE = None` (global stop off)

NeuTu stops on the first short branch, because `extractLongestPath` guarantees
the best remaining branch is the one just tested. We select by max-DAF
(kimimaro's `CachedTargetFinder`), which carries no such guarantee, so stopping
early truncates live arbor — B→A p90 went 2.40 → 74.71 on one body. Reject and
continue instead.

NeuTu's actual max-un-invalidated-length selector was implemented and removed: it
cut NeuTu-cable-absent from 13.6% to 2.8%, but at 31.5% of its own cable added,
overwhelmingly boundary convolution inside bulbs. It was not choosing branches
better, it was tracing nearly everything — including the third of its cable NeuTu
deliberately rejects.

### `fix_borders=True`, and the seam join

Block-first tracing needs fragments from adjacent blocks to meet, so
`border_targets` makes the **centre of each face-contact area** a mandatory
target, exempt from `min_length`. It delegates to
`kimimaro.intake.compute_border_targets` rather than reimplementing the rule,
because stage-2 fusion was built against that rule and a divergence would break
the seam join silently.

**Adjacent blocks do not share a plane**, so on curved structure the two contact
centres genuinely differ. On a split synthetic tube, seam offset without → with:
straight 5.10 → **0.00** voxels, bent 4.12 → **2.83**. The bent residual *exceeds*
the auto `join_radius_nm` (2 voxels = 64 nm at scale 2), so do not assume zero
offset from the straight case. On real bodies, though, seam gaps are
**1.00–1.41 voxels, already inside the default join**, and a production run's
39,167 seam joins added only 0.34% cable — so the default is adequate as
measured. A 16-voxel halo was built to improve this, measured to make it slightly
*worse* (1.73), and removed.

**`fix_borders` costs one spur per block face, per body**, because TEASAR's own
extremum on a face need not be the contact centre. That spur is exactly what a
tick filter targets, so **`fuse_body` joins seams BEFORE removing ticks** — the
order is load-bearing, and reversing it would amputate every reaching branch and
fuse a body into disconnected block-length stubs while every block still reported
`written`. A large `tick_branches_removed` is therefore expected, not alarming.

## Things established as wrong — do not re-derive

- **"NeuTu's larger radii are what fills better" is false.** Its median radius
  error is negative, and the port reaches 92% fill using exact inscribed radii.
  Fill is driven by path coverage and node placement, not radius. Do not tune
  radius to chase fill.
- **A single-root TEASAR trace covers exactly one connected component**, and
  these bodies are genuinely fragmented. Both the parent field and the
  rolling-ball invalidation are confined to the root's component, and the root
  comes from `first_label` — whichever voxel is first in memory order. On body
  6308993 that landed on a component holding **3.06%** of the voxels (of 7
  components, the largest holding 96.9%) and the trace reported 3% coverage. It
  fails as a plausible-looking low number, not as an error.
  `neutu_trace.skeletonize` iterates components and
  `test_all_connected_components_are_traced` holds it to that.
- **Set background to `inf` in any new cost function.** `dijkstra3d` takes a
  weight field with no separate mask, and `1/(1 + inf²)` is **0** — which makes
  empty space the cheapest thing in the volume and sends paths straight through
  it. NeuTu never puts background in its graph; kimimaro is safe only
  incidentally.
- **`anisotropy=(8,8,8)` in `SkeletonConfig` is a placeholder**, and `const` is
  in nm. The CLI overrides it from the source metadata, so published radii are
  correct, but the default alone is wrong by 4× at scale 2 (32 nm/voxel).
- **The mask is not ground truth.** Check the thickness map before blaming the
  skeletonizer for a bad radius: these segmentations carry no soma-scale thick
  core at all (max inscribed radius 345 nm on a nucleated body), so no
  skeletonizer can recover one. That is a statement about segmentation quality,
  not anatomy.
- **Skeletonizing at a coarser scale to save memory is dead.** Scale 3 (64 nm) is
  ~8× cheaper but **destroys the mask** before any skeletonization: component
  counts go 2 → 16, 9 → 94, 13 → 207, and the largest component's share collapses
  (96.3% → 41.2% on one body). Median process radius is ~72 nm, about 1.1 voxels
  at 64 nm, so thin neurites sit at the resolution limit and shatter. A property
  of the data, so no tracer change recovers it — block-first is the route.
