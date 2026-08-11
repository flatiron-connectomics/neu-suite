# Converting a source into a multiscale volume

`em-vol convert` builds a new pyramid from a source; `em-vol copy` does the same while
keeping everything the source already decided; `em-vol downsample` rebuilds the levels
above one you already trust. All three are block-mapped over dask and resumable.

```bash
em-vol info /path/to/source                 # what is it, which levels exist
em-vol convert --src /path/to/source --dst s3://bucket/out \
    --kind segmentation --config dask-slurm-example --workers 48
em-vol progress s3://bucket/out             # how far along
```

## Copying a volume, whole or in part

`copy` is `convert` with one difference: the output's format, chunking, voxel size and
image/segmentation type come **from the source**, and a source that records none of them
is an error rather than a guess.

```bash
em-vol copy --src <volume> --dst <destination>                    # the whole thing
em-vol copy --src <volume> --dst <destination> \
    --crop-bbox 5632,4480,6784,5760,4736,7040                     # one box
em-vol copy --src <volume> --dst <destination> --crop-bbox ... --dry-run
```

Use `convert` when you are *changing* something — the format, the chunking, the pyramid
schedule — and `copy` when you are not. The distinction is worth having because
`convert`'s defaults are not the source's: it defaults to precomputed, 128³ chunks and
`--kind image`, so copying a segmentation with it and forgetting `--kind segmentation`
averages label ids and publishes ids that were never in the data. Nothing raises; the
volume simply becomes wrong at every level above 0. `copy` reads `"type":
"segmentation"` out of the source's own `info` and cannot make that mistake.

`--crop-bbox z0,y0,x0,z1,y1,x1` restricts either command to one box. It is half-open, in
voxels, and clipped to the volume — `convert` and `copy` trim rather than pad, so a box
that overshoots costs nothing and never invents voxels. (The library's
`extract_roi()` is the crop-*and*-pad variant, for when you want the margin filled.)

**The output keeps the source's coordinate frame.** Its physical offset shifts by the
crop origin, which for precomputed means a `voxel_offset` equal to the crop start, so
loading the crop beside the original in neuroglancer lands it exactly over the region it
came from rather than at the origin.

```{tip}
`--crop-scale N` lets you give the box in scale-N voxels — usually the level you were
browsing when you picked it. The conversion uses the source's **own recorded per-level
voxel sizes**, never an assumed `2**N`: real pyramids are anisotropic, and with factors
`(1,2,2)` the same six numbers name a box of a different shape at every level. The
resolved level-0 box is logged before anything runs, and `--dry-run` shows it along with
the level shapes and the byte count.
```

### The pyramid is rebuilt, not copied

Both commands derive level *N* from level *N-1* of the **output**; the source's own
coarse levels are never read. For a crop that is what you want, since a slice of the
source's coarse level is not the reduction of the crop. It does mean a whole-volume copy
recomputes a pyramid that already exists — that cost is real, and `--single-level`
skips it if the levels are going to be rebuilt later anyway.

One consequence to know about: the output's reduction windows start at the **crop
origin**. If that origin is not a multiple of the coarsest cumulative factor, the
output's coarse voxels straddle the source's differently and each level's `voxel_offset`
rounds to its own grid. Level 0 is exact either way. Both commands warn and print an
aligned origin to use instead.

## Choose the level you trust

Downsampling cascades, so a bad level poisons every level above it. `--start-level N`
seeds from level N and rebuilds upward; `--dry-run` prints the schedule beside what is on
disk and refuses if they disagree, rather than leaving the pyramid half-consistent.

`--start-level 0` is ordinary and means "keep level 0, rebuild everything above it".

## Resume, and what the numbers mean

An interrupted run continues where it stopped. `convert` resumes by default (pass
`--fresh` to start over); `downsample` does not (pass `--resume`), because its usual
reason for existing is that you want the levels rebuilt.

Progress can be counted two ways and they legitimately disagree on sparse data:

`em-vol progress <volume>`
: counts from the run manifest — the tasks the run dispatched. Cheap, and it is what
  answers "how far along is my run".

`em-vol progress <volume> --storage`
: counts stored chunk objects. Authoritative about what exists, but it lists the store,
  and on sparse data it will read as far behind: **an all-fill chunk is never written**,
  so a block that processed correctly may leave no object at all.

```{warning}
A run that reports many blocks processed and **none written** means every block read as
the fill value. Usually the source could not be read at all — a precomputed volume
written by CloudVolume stores `.gz`-suffixed chunk keys that tensorstore requests without
the suffix and reads as zeros. `progress` says so explicitly rather than reporting 100%.
```

## Transient failures

Object stores fail occasionally, and at tens of thousands of tasks one bad connection is
close to certain. Every per-block worker retries with bounded backoff, classifying by the
error text because tensorstore maps both `PERMISSION_DENIED` and `UNAVAILABLE` onto
`ValueError`. Permanent errors are checked first and are **not** retried, so a
misconfigured run fails in seconds rather than burning the backoff budget on every task.

This does not soften fail-fast: an error that persists still ends the run. It only
removes the ones that would have fixed themselves.
