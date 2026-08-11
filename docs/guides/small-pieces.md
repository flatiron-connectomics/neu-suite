# Several small pieces into one frame

`convert` assumes one source materialised wholesale. The other real workflow is a handful
of small subvolumes — image stacks, HDF5 files — that belong at known positions inside one
larger volume and arrive at different times. That is `create` then `write`.

Both run in the calling process. No dask, no manifest, no cluster.

```bash
em-vol create /abs/gt.precomputed --like s3://.../image.precomputed \
    --dtype uint64 --kind segmentation          # empty, in the image's exact frame

em-vol write /abs/gt.precomputed --src a.h5 --src b.h5 --src c.h5
```

## `create` lays out an empty volume

Every level exists and no chunk data does, which is nearly free: an unwritten chunk reads
back as the fill value, so an empty pyramid is a few JSON documents regardless of its
nominal size.

`--like <reference>` copies the reference's frame **verbatim** — level shapes included,
not recomputed — so a voxel index means the same thing in both volumes. Recomputing would
usually agree, and the once it did not you would be a voxel apart partway up the pyramid
with nothing to tell you. It copies the reference's *format* too, unless `--format` says
otherwise.

```{warning}
Creating a volume where one of the *other* format already lives is refused, and
`--overwrite` will not resolve it. Both markers would sit in one directory, and `info` is
checked before `zarr.json`, so the older volume becomes unreachable while its chunks
still occupy the store. Delete the destination first.
```

## `write` places one piece at an offset

`--src` is repeatable, and the whole batch is checked — offsets, bounds, dtype — before
any of it is written, so a mistyped offset in the last file is caught while the volume is
still clean.

`--offset` is optional: with none given, each source is asked for its own. HDF5 files
routinely record one, and `--offset-field` names it (default `voxel_offset`).

```{warning}
**The axis order is asked for, never guessed.** `voxel_offset` is precomputed's field
name, and precomputed means xyz — while everything in these packages is zyx. Reversed,
the piece lands mirrored through the z=x diagonal, and nothing downstream can tell.
`--offset-order xyz` if the stored numbers are xyz. The provenance and any reversal are
printed on every run.
```

## Partial chunks are preserved, but only one writer at a time

A region that does not start and end on the destination's chunk grid makes tensorstore
read-modify-write the boundary chunks: it fetches what is stored, overlays the new data,
and writes the whole chunk back. A partially-covered chunk therefore **keeps** what was
already in it, including data an earlier `write` put there. Both commands report whether
your region is aligned.

Serially that is entirely safe. It becomes a lost update the moment two pieces sharing a
boundary chunk are written **concurrently**, with nothing left behind to detect it — the
piece you just wrote looks perfect and its neighbour is gone. Since these ops run in the
calling process, that hazard only exists across invocations you launch yourself.

## Then a pyramid

`write` is single-scale on purpose: how a patch should look when coarsened is a separate
decision, and averaging label ids invents ids. Run `em-vol downsample` afterwards.
