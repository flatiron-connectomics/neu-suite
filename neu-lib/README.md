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
from neu_lib import BBox, Frame, Mesh, ScaleInfo, Skeleton, align_box, to_xyz
```

| name | what it is |
|---|---|
| `BBox` | a half-open box `[lo, hi)`, rank-agnostic, immutable and hashable |
| `align_box` / `clamp_box` / `misaligned_axes` / `lcm_grid` | the same arithmetic on plain tuples, per axis |
| `Frame` | voxel indices → physical nm, per axis, with an origin |
| `to_xyz` | the single zyx→xyz conversion, for a renderer |
| `Mesh` | vertices and faces, nm, zyx |
| `Skeleton` | vertices and an **edge list** (not polylines), nm, zyx, with `crop` / `exclude` |
| `ScaleInfo` | one pyramid level: index, shape, key, and the `Frame` that places its voxels |

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
