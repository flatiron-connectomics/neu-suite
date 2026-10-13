# neu-lib

The vocabulary the [neu-suite](https://github.com/flatiron-connectomics/neu-suite)
packages share: boxes and grid arithmetic, the voxel-to-nanometre frame, and the two
geometry types meshes and skeletons are carried in.

**numpy only.** Nothing here reads a store, opens a canvas, or imports anything else —
which is the whole reason it is its own distribution rather than part of `neu-vol`. Every
tier of the suite can name these types without inheriting an I/O stack, and this builds
on Python 3.11 as well as 3.12 (the rest of the suite is pinned to 3.12 by `vol2mesh` and
`dvidutils`, both conda-only). **If something here ever needs to open a store, it belongs
in `neu-vol` instead.**

```python
from neu_lib import (BBox, Frame, Mesh, Piece, ScaleInfo, Skeleton, align_box,
                     box_predicate, mask_predicate, skeleton_tube, to_xyz)
```

| name | what it is |
|---|---|
| `BBox` | a half-open box `[lo, hi)`, rank-agnostic, immutable and hashable |
| `align_box` / `clamp_box` / `misaligned_axes` / `lcm_grid` | the same arithmetic on plain tuples, per axis |
| `Frame` | voxel indices → physical nm, per axis, with an origin |
| `Piece` | an array **and** the frame that says where it is |
| `to_xyz` | the single zyx→xyz conversion, for a renderer |
| `Mesh` | vertices and faces, nm, zyx |
| `Skeleton` | vertices and an **edge list** (not polylines), nm, zyx, with `crop` / `exclude` |
| `ScaleInfo` | one pyramid level: index, shape, key, and the `Frame` that places its voxels |
| `box_predicate` / `mask_predicate` / `union` | build the `inside()` that `Skeleton.crop` and `.exclude` take |
| `skeleton_tube` / `frustum_mesh` | a skeleton as a solid tube, one truncated cone per edge |

## `Piece`: an array and where it is

`ScaleInfo` pairs a *shape* with a `Frame` and answers "what does a voxel at this level
mean". `Piece` pairs the *data* with one and answers "where is this box, and what is in
it":

```python
piece.shape          # (364, 244, 244)
piece.bbox           # BBox in the frame's voxels — where it sits in its parent
piece.bounds_nm      # the same box in nm, as floats
piece.crop(box)      # a sub-piece, its origin shifted to match
```

It also carries `kind` (what the voxels mean — `image` / `probability` /
`segmentation`, which decides whether coarsening may average or must take a mode) and
`name` (what it is called; `neu_vol.read_piece` sets both from the source).

It exists because every place that read a crop used to carry the array here and the voxel
size and origin there, and each such pair is a chance to drop the origin — which puts the
data at nm zero instead of on top of what it came from, silently.

### Transforming one

```python
out = piece.apply(gaussian_filter, sigma=2)                   # frame, kind, name survive
out = piece.apply(threshold, kind="segmentation")             # ...unless you say otherwise
out = piece.copy(); out.array[mask] = 0                       # or edit a copy in place
```

`apply` calls `fn(array, *args, **kwargs)` and returns a new piece. Two guards, and they
are the same shape — each is an observable signal that a transform did something the
metadata cannot follow:

- **A changed spatial shape needs `frame=`.** Shape and frame are two halves of one
  statement about where the voxels are: a 2× downsample halves the shape *and* doubles the
  voxel size, so carrying the old frame through would place the result at half its real
  size. The factor is not guessed — real pyramids are anisotropic, and a crop changes the
  origin instead of the size.
- **A changed dtype needs `kind=`**, one of the three, or `"same"` if the meaning is
  unchanged, or `None` if it is no longer known. Nothing can *detect* a change of meaning —
  `(a > 0.5).astype("float32")` changes neither shape nor dtype — but a dtype change is the
  signal that is available, and it catches the two that occur: a probability map thresholded
  to a mask, and a mask labelled. Full dtype equality, not `dtype.kind`, because `uint8 →
  uint32` is exactly the connected-components case.

`copy(**changes)` is the same thing without a function, and it **copies the array** —
`dataclasses.replace` would re-pair the same one, so an in-place edit on what looks like a
copy would reach into the original and into whatever that was a view of.

**One coordinate space, and it is the frame's voxels.** `bbox` and `crop` both speak it,
not array indices: a piece read from voxel 7819 of a level reports `lo = 7819`, because
that is the number that places it back or compares it with a sibling. Array indices are
always `0..shape` and answer nothing.

**`bbox` is integers, `bounds_nm` is floats, and they are named apart on purpose.** `BBox`
has integer bounds and every grid operation on it assumes that, so a box in nanometres
cannot be one without truncating.

`Frame.voxel_box(bounds_nm)` is the conversion between frames, and the reason nanometres
are the shared model space — two levels of one volume, or a crop and its parent, have
different voxel sizes *and* different origins, so no voxel box transfers between them:

```python
gt.bbox.lo                                    # (7819, 8979, 3479) at 8 nm
level2.voxel_box(gt.bounds_nm).lo             # (1954, 2244, 869)  at 32 nm
```

It grows the box **outward** by default, so the result contains what was asked for rather
than dropping a face when the levels do not divide evenly.

## Two conventions, both load-bearing

Both fail *silently* when broken, which is why they are stated here rather than left to
each caller.

**Physical nanometres are the one model space, expressed per axis.** A geometry object
never carries an integer level, and a factor is never derived from `2 ** level`. Real
pyramids are anisotropic — halving x and y while leaving z alone is ordinary — so a level
index means nothing without the source's own voxel sizes. Reading those belongs to
whatever layer can open a store: `neu_vol.read_scales` returns a `ScaleInfo` per level,
each already carrying its `Frame`, so nothing downstream rebuilds a transform from a
bare voxel size — which is how a level's origin got dropped for as long as it did.

**Boxes are half-open, `[lo, hi)`.** A `hi` already sitting on a block boundary must not
move; ceiling it anyway grows every correctly-sized box by a whole block. The same rule is
why `BBox.from_points` adds one to its maximum — a box built from points must *contain*
them, and getting it wrong drops the far face on every axis while looking fine.

**zyx in memory, xyz at the boundary.** Every array here is zyx, and `to_xyz` is the only
conversion. Reversed, geometry mirrors through the z=x diagonal, which reads as plausible
data in the wrong place rather than as an error.

## Regions

`Skeleton.crop(inside)` and `.exclude(inside)` take a **callable** — given an `(N, 3)`
array of zyx nm points, which are in the region — and that is the whole interface. Keeping
it a plain function is what lets a caller supply a region this package has never heard of:
a mesh hull, a distance field, a polygon.

`box_predicate` and `mask_predicate` cover the two ordinary cases, and `union` composes
them. `union` matters more than it looks: excluding a soma *and* a nucleus is two masks,
and merging them into one array would force them to share a resolution and an origin,
which they need not — a compartment read in a small box around the thing that localises it
has its own frame.

**A point outside a region's own extent is outside the region, not an error.** A skeleton
normally runs past any one ROI or compartment mask, and that is the question being asked.

## Which grid?

`align_box` takes a block shape and does integer arithmetic per axis. It deliberately does
**not** know which grid you mean, and there are several — the write unit (the chunk, or the
*shard* where a level is sharded), the pyramid's cumulative factor, or the per-axis LCM of
a source and destination chunking. Resolving that needs a volume, so it lives in `neu-vol`;
see `neu-vol align-bbox`.

## Install

```bash
pip install -e ./neu-lib
```

Within the suite's shared conda environment use `--no-deps`, so pip does not re-resolve the
conda-provided numpy:

```bash
pip install --no-deps -e ./neu-lib
```

## Tests

```bash
python -m pytest -q
```

No fixtures, no store, no temp directories — every test is arrays and tuples, which is the
property worth keeping.

## License

©2026 The Simons Foundation, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE). See
[CONTRIBUTING.md](CONTRIBUTING.md) for how contributions are licensed.
