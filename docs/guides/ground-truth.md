# Ground truth: from annotated chunks to meshes

The full path for manually annotated chunks assembled into one volume, meshed, and
published for viewing.

```bash
em-vol create  s3://.../gt_v1 --like s3://.../image --dtype uint64 --kind segmentation
em-vol write   s3://.../gt_v1 --src chunk1.h5 --src chunk2.h5 ... [--background 1]
em-vol downsample s3://.../gt_v1 --start-level 0 --config ... --workers 24

em-vol relabel s3://.../gt_v1 --out s3://.../gt_v2       # ← do not skip
em-vol downsample s3://.../gt_v2 --start-level 0 --config ... --workers 24

em-morpho run --src s3://.../gt_v2 --dst s3://.../gt_v2 \
    --work-dir /mnt/ceph/users/<you>/gt-meshing --stages mesh --mesh-scale 0 \
    --config ... --workers 48

em-ngl bboxes s3://.../gt_v2 --label gt --out gt_layer.json
em-ngl gen --seg s3://.../gt_v2 --layer gt_layer.json --select-last
```

## When background is not 0

Some annotation tools number labels from 0, which makes **background 1**. Pass
`--background 1` to `write` (or `to-hdf5`, or `convert`) and it becomes 0 as the source is
read. Do it there rather than afterwards, because of what happens otherwise:

```{warning}
**An all-background block of 1s is not all-fill, so it gets stored.** The volume then has a
chunk object everywhere data was written, whether or not it holds anything — and "which
chunk objects exist" stops answering "where is the data". That is precisely the question
`em-ngl bboxes`, `relabel`, `downsample --sparse` and em-seg-morpho's occupancy filter all
ask, so they all quietly start answering "everywhere". Background also becomes a body when
meshed, and an enormous one.

Measured on a 16×32×32 test volume with background 1: **32 stored chunks, of which 4 held
any label.**
```

For data that has already landed, `em-vol mask-by-value` repairs it:

```bash
em-vol mask-by-value s3://.../gt_v1 --values 1 --out s3://.../gt_v1_fixed
em-vol downsample s3://.../gt_v1_fixed --start-level 0        # single-scale, like relabel
```

Either destination restores the sparsity — writing zeros over a stored chunk removes the
object, on both formats — so `--out` is preferred for the ordinary reason instead: a sparse
copy is cheap, and the original stays as the record of what was annotated. It reports how
many voxels it replaced, and warns if **none** matched, which almost always means the
background value was not what you thought.

## Why `relabel` is not optional

Annotation tools number each chunk from 1. Assembled into one volume, the same integer
names a different cell in every chunk — and nothing downstream can tell. Meshing produces
one body whose components are scattered metres apart in model space: correct for the
label, useless as ground truth.

Measured on a real 12-chunk ground truth volume: **3,832 label-instances, 1,901 distinct
ids, 508 of them used by more than one chunk.** A body numbered 1 was a chimera of a
dozen unrelated cells spanning 57 × 29 × 33 µm, where a single chunk is about 2 µm.

`relabel` walks the occupied regions in order and gives each its own range. It finds the
regions the same way `em-ngl bboxes` does — from which chunk objects exist — so they are
pairwise disjoint and chunk-aligned, and it is serial by construction because each range
begins where the last ended.

```{important}
The old→new mapping is written to `<destination>.relabel-<level>.json` and is the **only**
route from a new id back to the region and original label it came from. Keep it with the
volume. Prefer `--out` over `--in-place` for the same reason: a sparse copy is nearly
free, and the original stays as the record of the raw annotation.
```

`--block-size N` numbers region *k* from `N*k+1` instead of consecutively, so the chunk a
label came from is readable straight off the id.

Because `relabel` is single-scale, the levels above it hold the old ids until
`downsample` re-runs. It says so on every run.

## Meshing

`--stages mesh` if you do not want skeletons; without it you get both.

Fault policy is asymmetric on purpose. Per-**body** tasks isolate failures — recorded with
a traceback, retried on the next run — while per-**block** tasks fail fast, because stage 2
aggregates across blocks and a silently skipped block truncates every body passing through
it while the output still looks complete. Expect roughly 0.5% of bodies to fail; they are
highly fragmented ones whose small components collapse under LOD decimation.

```{warning}
If you delete a subresource directory afterwards, remove its key from the volume's `info`
too. An `info` naming a `skeletons` directory that no longer exists is a volume describing
something that is not there.
```

## Watching it

```bash
em-morpho progress   <work-dir>      # live, both stages have real denominators
em-morpho run-report <work-dir>      # self-contained HTML, works on an in-flight run
```
