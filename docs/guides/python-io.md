# Reading and writing from Python

The CLI works in whole volumes: convert one, create one, place a piece into one. In a
notebook or a cleaning script the unit is smaller and the loop is different — read a box,
transform it, write the result — and the thing that has to survive that loop is not the
array. It is **where the array is**.

Two functions do it, and they are inverses:

```python
import neu_vol

piece   = neu_vol.read_piece("gt_v1_eval.h5:/vol_03700")
cleaned = piece.apply(neu_proc.dust, min_voxels=1000)
neu_vol.write_piece(cleaned, "gt_v1_eval_cleaned.h5")     # -> /vol_03700, same frame
```

Nothing in that snippet names a voxel size, an offset, an axis order or a dataset. All four
came out of the file, rode along through the transform, and went back in.

## A piece is an array that knows where it is

`neu_lib.Piece` pairs an array with a `Frame` — a voxel size and an origin, both in
nanometres — plus what the voxels *mean* (`kind`) and what to call them (`name`). The
pairing is the point: an array on its own forgets where it came from after the first
operation, and the failure that causes is silent. Data that lands at nanometre zero looks
exactly like data that belongs there.

```python
piece.shape          # (256, 256, 256)
piece.voxel_size_nm  # (8.0, 8.0, 8.0)
piece.origin_nm      # (19200.0, 45600.0, 22160.0)
piece.origin_voxel   # (2400, 5700, 2770)  — whole voxels, or it raises
piece.bbox           # BBox(lo=(2400, 5700, 2770), hi=(2656, 5956, 3026)) in FRAME voxels
piece.bounds_nm      # the same box in nm — what transfers between two sources
piece.kind           # 'segmentation'
```

`origin_voxel` **raises** rather than rounding when the origin is not a whole number of
voxels, because rounding shifts the piece by up to half a voxel against whatever it is meant
to line up with, and nothing downstream can detect that. `bounds_nm` is always available.

```{note}
Everything here is zyx in memory and nanometres in the world — invariants the whole suite
depends on. Both precomputed mesh/skeleton files and precomputed `voxel_offset` are *xyz*,
which is why the axis order is something the file states and the reader reads, never
something anybody guesses.
```

## `read_piece` — the voxels, and everything known about where they are

```python
read_piece("piece.h5")                        # one dataset in the file: found
read_piece("gt_v1_eval.h5:/vol_03700")        # a container: name the array
read_piece(volume, level=2, crop=((0, 0, 0), (256, 256, 256)))
read_piece(volume, level=2, crop=piece)       # the same PHYSICAL box as `piece`
```

`src` is `PATH` or `PATH:/DATASET`; only a *leading slash* makes the tail a dataset, so a
scheme's own colon (`s3://…`) is left alone. A spec mapping or an already-open `backend=`
works too.

**The crop may be given three ways and the physical one is what earns its keep.** A voxel
box means nothing outside its own frame — two levels of one volume have different voxel
sizes, a crop and its parent have different origins — so nanometres are the only currency
that transfers. Handing `read_piece` another piece is how "show me the EM image under this
ground-truth crop" becomes one call:

```python
gt    = read_piece("gt_v1_eval.h5:/vol_03700")
image = read_piece(image_volume, level=0, crop=gt)      # registered, by construction
```

A physical box is converted with the target level's *own* voxel size and origin — read from
its metadata, never assumed to be `2**level`, because real pyramids are anisotropic — and
grown outward, so the read contains what was asked for rather than dropping a face where the
levels do not divide evenly. A box that mostly misses is reported: losing most of a box
usually means the two sources are not the same volume.

Three things it will not do for you:

`kind` **is not guessed.** A uint8 label array is indistinguishable from an image by dtype,
and getting it wrong later authorises averaging label ids into ids that were never in the
data. Where the source records nothing, `kind` stays `None` and you pass it:
`read_piece(path, kind="segmentation")`.

`voxel_size` **is required for a source that records none** — a slice stack, or a plain HDF5
file that carries no `voxel_size` attribute. There is nothing to read, so there is nothing
to infer.

**A cast on the way in is allowed, and a narrowing one is called out.** `dtype=` casts
after the read, so a piece arrives in the dtype it will be used in — a set of crops exported
by different tools comes as uint8/16/32/64, and a consumer that has to handle all four is
what this avoids:

```python
piece = read_piece(gt_file, dataset=name, kind="segmentation", dtype="uint64")
```

Widening among unsigned ints loses nothing. Narrowing is warned about rather than refused,
since only you know the range your labels use — but it wraps a label id into another
plausible label id, and a segmentation has no invalid values for anything downstream to
catch.

**A read too large to finish is refused, not attempted.** A whole level 0 of a production
volume is terabytes; `read_piece(volume)` on one does not fail, it *hangs*, which in a
notebook is indistinguishable from a wedged kernel with every later cell pending behind it.
The cap is 4 GiB, the error names the size and a coarser level that would fit, and
`max_bytes=None` lifts it for a caller who means it.

## Transforming without losing the frame

```python
cleaned = piece.apply(neu_proc.opening, radius=2).apply(neu_proc.dilate)
edited  = piece.copy()          # the array is copied too
edited.array[edited.array == 668] = 650
```

`apply` keeps the frame, the name and the kind across a chain, and guards the two cases
where it cannot: a transform that **changes the spatial shape** must supply the new `frame`
(a 2x downsample halves the shape *and* doubles the voxel size), and one that **changes the
dtype** must say what the result is (`kind=...`, or `kind="same"`). Both guards exist because
the alternative is metadata that quietly stops describing the array.

`copy()` copies the array, which `dataclasses.replace` does not — so an in-place edit on what
looks like a copy cannot reach back into the original, or into whatever the original was a
view of.

## `write_piece` — back out to a file

```python
plan = neu_vol.write_piece(cleaned, "gt_v1_eval_cleaned.h5")
plan["dataset"], plan["voxel_offset"], plan["kind"]
# ('/vol_03700', (2400, 5700, 2770), 'segmentation')
```

**Four arguments `to-hdf5` has to ask for are absent here, because the piece answers them:**
`voxel_size` is the frame's, `voxel_offset` is `piece.origin_voxel`, `axes` is always `zyx`
and `units` always `nm`. So this path has no axis-order question at all — the file records
`axes="zyx"` and `neu-vol write` reads it rather than assuming.

`kind` travels too, and that is what makes the round trip lossless: a cleaned segmentation
read back is still a segmentation. Use `piece.with_kind(...)` to change it.

**The dataset name comes from the piece.** `read_piece` names a piece after its source —
`stem/inner` for an array inside a container — so the last component is reused, and a bag of
crops round-trips through a whole cleaning pass with no arguments:

```python
for name in neu_vol.describe(gt_file)["datasets"]:
    piece = neu_vol.read_piece(gt_file, dataset=name, kind="segmentation")
    neu_vol.write_piece(clean(piece), out_file)         # one file, same names
```

A piece named after a *volume* has no such component and gets `/data`, which is also what
the reader assumes when it is not told. `dataset=` overrides both, and is the way to write a
nested path.

An existing file is **added to** when its recorded frame matches — several pieces of one
volume in one file is the ordinary arrangement here, each keeping its own `voxel_offset`.
Two things are refused rather than guessed: a frame that disagrees with the file's, since
one file describing two coordinate systems has no correct reading, and a dataset name
already in use, unless `overwrite=True`.

What lands in the file, and why in two places:

| attribute | where | what it says |
|---|---|---|
| `voxel_size`, `units`, `axes` | root **and** dataset | the coordinate system, so either the file or the array alone is self-describing |
| `voxel_offset` | dataset | where this piece starts, in whole voxels |
| `offset` | dataset | the same place in `units` |
| `kind` | dataset | image, probability or segmentation |

Writing is **blocked**, so a piece holding a lazy array — a dask or zarr array, an open h5py
dataset — streams rather than being materialised whole. `dry_run=True` returns the same plan
dict without touching the file.

```{warning}
`out` must be an ordinary filesystem path ending in `.h5` / `.hdf5` / `.hdf` / `.he5`. h5py
has no object-store driver, and the extension is not cosmetic: HDF5 has no marker object, so
`describe`, `neu-vol info` and `neu-vol write` recognise a container **by name**. A file
written under any other name would be unreadable by everything that has to recognise it.
```

```{note}
Reading a file and then writing to it in the same process works, including a re-run of the
same cell. That needs saying because it once did not: a backend caches its open `h5py.File`,
and HDF5 refuses to open a file for writing while any handle holds it read-only, so the
write failed with an error naming the file rather than the reader still holding it. The
write path now releases those handles first. `neu_vol.release_backends(path)` is the same
thing by hand, if something outside this package is holding one.
```

## Then placing it back into a volume

`write_piece` writes a file, not a volume — and the file is exactly what `neu-vol write`
wants, so the two compose:

```python
neu_vol.write_piece(cleaned, "piece.h5")
neu_vol.write_subvolume(volume, "piece.h5")      # no offset: the file recorded one
```

```bash
neu-vol write <volume> --src piece.h5            # the same thing from the shell
```

That indirection is deliberate. Writing into a volume is a *chunked* operation with a
concurrency hazard — a region that does not land on the chunk grid makes the store
read-modify-write the boundary chunks, so two pieces sharing one written at the same time
lose an update with nothing left behind to detect it. Going through a file keeps the
placement in one serial, checked step. `write` is also single-scale on purpose: run `neu-vol
downsample` afterwards, since how labels should look when coarsened is a separate decision.

## Looking at one on the way past

A piece is a layer input, so a crop can go to a viewer without being written anywhere
first — `neu_glance.serve` and `srv.add_layer` both take one, and it arrives already named
and knowing its kind:

```python
neu_glance.serve([piece]).add_layer(cleaned)      # before and after, side by side
```

## Checking what you wrote

```python
neu_vol.describe("cleaned.h5")["datasets"]        # every array, with its own offset
```

```bash
neu-vol info cleaned.h5                           # the same table
neu-vol info cleaned.h5 --dataset /vol_03700      # the full report on one
```

If a file records no frame, the report says **which attribute names it searched**, because
"records no scale" and "spells it differently" look identical otherwise —
`voxel_size_field` / `offset_field` are parameters for exactly that reason, on both the
reading and the writing side, so a file keeps whatever spelling its siblings use.

## What is deliberately not here

**There is no region-by-region HDF5 write.** `HDF5Backend.write_region` raises and says so:
an HDF5 file is produced in one call, by `write_piece` for an array in memory or
`pack_hdf5` / `neu-vol to-hdf5` to stream one out of another source. Both go through the same
layout code, so either file is placeable the same way.

**There is no remote HDF5**, in either direction. Copy the file, or write locally and upload.

**`write_piece` does not take a volume as its destination.** See above — that is
`write_subvolume`, and it reads from a file.

See also [several small pieces into one frame](small-pieces.md) for the same workflow driven
entirely from the CLI, and [ground truth to meshes](ground-truth.md) for what happens to a
cleaned GT volume next.
