# Several small pieces into one frame

`convert` assumes one source materialised wholesale. The other real workflow is a handful
of small subvolumes — image stacks, HDF5 files — that belong at known positions inside one
larger volume and arrive at different times. That is `create` then `write`.

All three run in the calling process. No dask, no manifest, no cluster.

```bash
neu-vol to-hdf5 --src slices/ --out a.h5 \
    --voxel-size 8,8,8 --offset 24,128,256      # a piece that knows where it goes

neu-vol create /abs/gt.precomputed --like s3://.../image.precomputed \
    --dtype uint64 --kind segmentation          # empty, in the image's exact frame

neu-vol write /abs/gt.precomputed --src a.h5 --src b.h5 --src c.h5
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
**The axis order is never guessed.** `voxel_offset` is precomputed's field name, and
precomputed means xyz — while everything in these packages is zyx. Reversed, the piece
lands mirrored through the z=x diagonal, and nothing downstream can tell.

So it is either *recorded* or *asked for*: a source may state its order in an `axes`
attribute, which is what `neu-vol to-hdf5` writes, and then `write` reads it rather than
assuming. Failing that it falls back to zyx, and `--offset-order xyz` overrides both. The
provenance, which of the three applied, and any reversal are printed on every run.
```

## `to-hdf5` makes a piece worth placing

The inverse of `write`. An image stack off a microscope or an annotation tool is a
directory of PNGs with no coordinates attached; `neu-vol to-hdf5` packs it into one HDF5
file **with** its frame and position, so placing it later needs no arguments at all:

```bash
neu-vol to-hdf5 --src slices/ --out piece.h5 --voxel-size 40,8,8 --offset 24,128,256
neu-vol write <volume> --src piece.h5        # no --offset, no --offset-order
```

What it records: `voxel_offset` in whole voxels on the dataset — the field `write` already
looks for — plus `voxel_size`, `offset` (the same place in physical units), `units` and
`axes` in this package's own vocabulary, on the root *and* the dataset, so either the file
or the array alone is self-describing.

The dataset defaults to `/data`, which is also what the reader assumes when it is not told,
so a file packed with no arguments reads with none. `--dataset` names another.

An existing file is **added to** when its recorded frame matches — several pieces of one
volume in one file is a legitimate arrangement, each keeping its own `voxel_offset` — and
refused when it does not, since one file describing two coordinate systems is not worth
allowing. A dataset name already in use needs `--dataset` or `--overwrite`.

```{tip}
A file with more than one volumetric dataset can no longer be read without naming one, so
`to-hdf5` says so when it creates that situation rather than leaving you to meet it later.
```

Reads are blocked, so a "small" volume that turns out not to be still packs rather than
filling memory. `--chunk` sets the HDF5 storage chunk, which is what governs partial reads
when the piece is written back.

### When background is not 0

`--background 1` on `write`, `to-hdf5` or `convert` replaces those values with 0 **as the
source is read**. That timing is the point, not a convenience: an all-background block of 1s
is not all-fill, so without it every such block is *stored*, and the volume stops answering
"where is the data" by which chunks exist. `neu-vol mask-by-value` repairs data that has
already landed — see [the ground-truth guide](ground-truth.md).

### Bringing a foreign HDF5 file into this layout

A file written elsewhere often already describes itself — data in `main`, with
`voxel_offset` and `voxel_size` beside it, as attributes *or* as top-level datasets. Point
`to-hdf5` at it and pass nothing:

```bash
neu-vol to-hdf5 --src theirs.h5 --out canonical.h5
```

Everything it records becomes the default: the offset, the voxel size, and a recorded
`axes` if it has one. `main` needs no `--src-dataset` either — it is the file's only 3D
dataset, so it is found on its own. Add `--crop-bbox` and the piece lands at the **sum** of
what the file said and where the box started inside it, since a box out of a piece that
knows its position belongs there rather than at the box's own offset.

### When another tool named the fields differently

`--voxel-size-field` (default `voxel_size`) sets the attribute the scale is written under
**and** read from, so a file keeps whatever spelling its siblings use and repacking one
never asks you to retype a scale it already carries. `--offset-field` does the same for
`voxel_offset` — change it on `neu-vol write` too, or `write` will look for the old name.

`neu-vol write --voxel-size-field` uses it for one thing only: **checking** the piece against
the level it is going into. A region extracted at level 1 and written to level 0 fits,
places cleanly, and is at the wrong resolution — the shapes, dtype and bounds are all
consistent, so nothing else here would notice. It warns rather than refuses, since writing
a deliberately coarser piece is legitimate.

### Taking a box out of a volume

`--src` is any readable source, a volume included — so the same command extracts a region
for annotation or inspection and hands it back afterwards:

```bash
neu-vol to-hdf5 --src <volume> --out region.h5 --level 1 --crop-bbox 2,2,2,10,10,10
#   ... annotate region.h5 ...
neu-vol write <volume> --src region.h5 --level 1        # straight back where it came from
```

`--level` picks which level to read (default 0) and `--crop-bbox` a box within it, in **that
level's voxels**. Two defaults make the round trip argument-free: the level's own recorded
voxel size becomes the frame — never `2**level`, so an anisotropic pyramid is handled — and
the crop origin becomes the recorded `voxel_offset`, which is the one number nobody should
have to type twice. `--offset` overrides it, for extracting from one volume to place into
another somewhere else.

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
decision, and averaging label ids invents ids. Run `neu-vol downsample` afterwards.
