# Converting a source into a multiscale volume

`em-vol convert` builds a new pyramid from a source; `em-vol downsample` rebuilds the
levels above one you already trust. Both are block-mapped over dask and resumable.

```bash
em-vol info /path/to/source                 # what is it, which levels exist
em-vol convert --src /path/to/source --dst s3://bucket/out \
    --kind segmentation --config dask-slurm-example --workers 48
em-vol progress s3://bucket/out             # how far along
```

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
