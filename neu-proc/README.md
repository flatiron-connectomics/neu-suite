# neu-proc

Blockwise array processing for large 3D EM volumes: scipy / scikit-image filters and
Seung-lab label tools (cc3d, fastremap, fastmorph, edt, fill_voids), applied to whole arrays
now and across a volume later.

**Only `neu_proc.ops` exists so far, and it is deliberately the pure half.** The array
functions are the part that can be written, read and tested on their own; the halo, seam and
reconcile machinery that makes them correct across block boundaries is designed in
`NEU-PROC-PLAN.md` and not built. That is a phasing choice, not a stub: an op that works on
one array is the *reference* the distributed runner gets checked against.

```python
import neu_proc

grown  = neu_proc.dilate(labels, max_distance=3)      # grow labels into background
cleaned = neu_proc.dust(labels, min_voxels=50)         # drop small components

piece.apply(neu_proc.dust, min_voxels=50)              # or through a Piece
```


## What exists so far

Four operations, a measurement module, and the two support modules under them. Top-level
names resolve lazily (PEP 562), so `import neu_proc` pays for neither scipy nor cupy.

### `dilate(arr, method="edt", *, max_distance=None, gpu=None, **kwargs)`

Grow the labels in an array, three ways. `edt` assigns every background voxel the label of
the nearest foreground voxel; `fastmorph` and `spherical` call fastmorph.

**`edt` is unbounded unless you bound it.** With no `max_distance` the labels grow until they
meet and no background is left — a Voronoi partition of the array rather than a dilation by a
radius. Measured on a 16³ array with two 2³ seeds: unbounded `edt` fills all 4096 voxels,
fastmorph 128, spherical 64. Useful, and not what "dilate" usually implies.

`max_distance`'s units follow `sampling`. With `sampling=(z, y, x)` in nm the bound is in nm;
without it both are counted in voxels — and on an anisotropic volume voxel-counted growth is
lopsided in tissue while looking symmetric in the array. Measured on a 32³ array with one 2³
seed: `max_distance=5` (voxels) grows it to 792 voxels, while `sampling=(40, 8, 8),
max_distance=80` (nm) gives 1280. Same request, different answers — which is the whole reason
the `Op` layer declares lengths in nm.

The three methods take **disjoint** parameters (`edt` has `sampling`, `spherical` has
`radius`/`anisotropy`, `fastmorph` has `iterations`/`mode`), so they are less interchangeable
than one entry point suggests. A parameter on the wrong method is refused by name, with a
pointer to the one that takes it; an unknown method name is refused with the list.

### `dust(arr, min_voxels=None, *, max_voxels=None, connectivity=6, invert=False, exclude_boundary=False)`

Drop small connected components. **Component-wise, not per-label**, and the difference is
large here: a speck carrying the same id as a big body somewhere else is removed while the
body stays. That is the right behaviour for this data, where bodies are genuinely fragmented
rather than artefactually so — one body measured 68 components at scale 2 and **344 at scale
1**, most of them single-voxel specks carrying the label.

It is **purely subtractive**: label ids are preserved exactly (7 stays 7, nothing is
renumbered), and a label's two disconnected surviving components both keep the same id — dust
never splits anything. Nothing is filled in behind what it removes.

```python
dust(labels, 10)                  # remove anything under 10 voxels
dust(labels, 10, invert=True)     # keep ONLY those — what you are throwing away
dust(labels, 5, max_voxels=50)    # keep the band, dropping both tails
```

`connectivity=6` is the default, counting only face-sharing neighbours. 26 is the loosest
connectivity and so the *conservative* choice for dusting — it merges specks into neighbours
where it can and removes fewer of them.

`exclude_boundary=True` spares any component **reaching a face of the array**, however
small. Inside a crop such a fragment may belong to a body that mostly lives outside it, so
its voxel count is only a lower bound and the threshold cannot judge it; a component wholly
inside is judged normally, including one that merely comes close to a face.

**Per component, not per label**, and that distinction is the whole operation: keyed on the
label instead, every speck sharing an id with something on a face survives wherever it sits.
Measured on one 364³ ground-truth crop at `min_voxels=1000`, that spared 72,706 voxels of
interior dust across 254 labels — 34% of what was meant to go — while also costing a
full-array pass per face label (8.3 s against 0.42 s). The test is a bounding box per
component, which `cc3d.statistics` has already computed and which is equivalent to scanning
the six face slices for its id.

Voxel counts, not physical volume, deliberately: the conversion needs a `Frame` and belongs
one layer up. Note it is a **volume**, so `kernels.volume_to_voxels` divides by the *product*
of the three voxel sizes rather than scaling per axis — a 40×8×8 nm voxel is 2560 nm³, not
40, so a 1 µm³ threshold is 390 voxels and not 25,000.

### `opening(arr, radius, *, anisotropy=None)`

An erosion then a dilation, label-aware, via `fastmorph.spherical_open` — the way to remove
a thin protrusion or a one-voxel bridge without shrinking what survives.

**It is not anti-extensive, and a mathematical opening is.** `γ(X) ⊆ X` is what the name
promises, and this does not have it, because fastmorph's erosion and dilation read `radius`
differently. Measured on a 10³ box: at `radius=1` the erosion removes *nothing* while the
dilation still grows by a 6-neighbourhood, so the call is a pure dilation and returns 1600
voxels; at `radius=2` a one-voxel shell does come off — a thin spike is genuinely removed —
but the survivor comes back 1384 voxels, 38% larger than it went in. Neither is a bug to fix
here; what would be a bug is a caller assuming sizes are comparable across the operation. A
radius whose erosion is inert warns rather than passing silently.

`radius` follows `anisotropy` exactly as `dilate`'s `max_distance` follows `sampling`, and
is counted in voxels without it.

### `offset(arr, offset, fill_method="extend", fill_const=0)`

Shift by whole voxels, **keeping the shape** — so the far side is filled, either by repeating
the edge plane (`extend`) or with a constant (`const`). Nothing wraps: a body leaving one
face does not reappear on the opposite one, which `np.roll` would do and which produces a
well-formed array with tissue teleported across the volume.

The frame does not come along. Shifting the voxels by one is the same statement as moving the
origin one voxel the other way, so a physical shift is `piece.apply(..., frame=...)` one
layer up; this layer is arrays in, arrays out.

### `ops/measure.py` — component and label sizes

`sizes`, `component_sizes`, `flat_component_sizes`: voxel counts per label, per component,
and flattened across labels. Exposed as a module (`from neu_proc import measure`) rather
than as loose top-level names, since it is a namespace of related readings rather than one
transform.

### `ops/kernels.py` — the one place a length becomes voxels

`per_axis`, `to_voxels` (a length, per axis), `volume_to_voxels` (a volume, by the product),
`gaussian_halo`, `ball` (an *ellipsoid* whenever the radii differ, which is the anisotropic
case), `box` (odd sizes, so a window stays centred). Imports no scipy: converting a length
needs arithmetic and a frame, and keeping it scipy-free means the conversion can be tested
without the filters and reasoned about beside the halo it produces.

### `ops/backend.py` — scipy on the host, cupyx on the device

**Dispatch is on the array, never on a flag**, since `cupyx.scipy.ndimage` requires a
`cupy.ndarray` and the module cannot be swapped without moving the data:

```python
ndi, array = backend.ndimage_for(array, "grey_dilation")
return ndi.grey_dilation(array, footprint=kernels.ball(radius_vox))
```

One body, both backends. Three properties follow from dispatching on the array:

- **No hidden state.** A global `use_gpu` is what makes one call behave differently for two
  people. Here the answer is a property of the array in your hand.
- **Free when cupy is absent.** `is_device_array` reads `type(a).__module__`, so nothing
  imports cupy to discover that it is not needed.
- **`NEU_PROC_GPU=0` governs transfers, not dispatch.** Handed a device array anyway, the
  right thing is still cupyx; a flag that ran host code on device memory would just raise
  somewhere less obvious.

**A host array may still go to the device and come back, and for an expensive op that wins.**
Measured on an RTX A6000 for the EDT dilation — `round trip` includes both transfers:

| size | host (scipy) | round trip | already on device |
|---|---|---|---|
| 96³ | 36.4 ms | 2.7 ms (13.6×) | 0.7 ms (52×) |
| 144³ | 140.7 ms | 6.3 ms (22.3×) | 1.3 ms (112×) |
| 192³ | 811.2 ms | 16.4 ms (49.5×) | 2.0 ms (401×) |

The margin *grows*, because scipy's EDT is superlinear in the voxel count while a transfer is
linear — so the break-even is below any size worth processing, for this op. None of that
generalises to a cheap one: a threshold is bandwidth-bound and can only lose the transfer.
So `stage(value, prefer_gpu=)` is the **op's** call, not the caller's, and `like(result,
reference)` returns the answer to wherever the input lived — `dilate(numpy_array)` gives back
numpy, and nothing silently leaves your data on a GPU. For a *chain*, move once yourself with
`backend.to_device` and every step stays there, which is where the extra ~8× is.

Two host-only escapes, both **loud**: `ndimage_for` falls back to scipy with a warning for the
four functions cupyx lacks (`distance_transform_bf`, `distance_transform_cdt`,
`geometric_transform`, `watershed_ift`), and `host_only` does the same for libraries with no
device build at all — fastmorph, cc3d, fill_voids, edt. Loud because the array came to the
device to be fast and a copy back is the slowest thing that can happen to it; not fatal
because the op still works and refusing would let one function poison a chain.

**There is no GPU `dust`, and that is a correctness decision.** The obvious cupy version —
`cupyx.scipy.ndimage.label` on the mask, then drop the small components — measured 4-5× faster
than cc3d even including both transfers, and it is **wrong for a multi-label array**: `label`
works on a *binary* mask, so two different bodies that touch become one component. Measured, a
4-voxel label adjacent to a 125-voxel one survives that way where cc3d removes it. cupyx has
no multi-label connected components, so there is nothing to dispatch to.

### cupy is optional and deliberately undeclared

`pyproject.toml` names no cupy, because the wheel is CUDA-version-specific
(`cupy-cuda12x`, `cupy-cuda11x`, ...) and pinning one would break the install on any other
machine. Install it yourself if you want the device path; everything works without it.

## Three layers, and only the middle one is nanometres

| layer | example | units | can run? |
|---|---|---|---|
| `Op` | `Gaussian(sigma_nm=80)` | **nanometres** | no |
| `BoundOp` | `op.bind(frame)` | voxels, **plus a halo** | yes |
| a function | `gaussian(a, sigma_vox=…)` | **voxels** | — |

The declaration is physical because a voxel count means different things at different pyramid
levels and along different axes of an anisotropic volume — and real pyramids are anisotropic,
`(1, 2, 2)` being common. The execution is in voxels because that is what scipy takes.

The conversion happens in exactly one place, `ops/kernels.py`, reached through `bind` — never
inside a function, and never declared beside a halo. **A halo derived from anything other
than the converted parameter is wrong:** a Gaussian whose halo assumes `truncate=3.0` while
scipy's default is `4.0` is wrong in a shell around every block seam, which shows up as faint
block edges in the output rather than as an error.

Only `Op` and `BoundOp` are not written yet. Functions written voxel-native today need no
change when they arrive, since a `BoundOp` is just a callable.

## Conventions in `ops/`

- **zyx**, like everything else in the suite.
- **Per-axis lengths, or a scalar that broadcasts.** `sigma_vox=(1, 4, 4)` is what `bind`
  will hand down on an anisotropic volume, so a scalar-only signature would have to change.
- **Array in, array out, and dumb about metadata.** `Piece.apply` is where the discipline
  lives: it refuses a shape change without a new `frame=` and a dtype change without a
  `kind=`, so a function here carries no frame awareness at all.

## Why not neu-lib

`neu-lib` is **numpy and nothing else**, test-pinned — that is what keeps it installable on
3.11 while the rest of the suite is pinned to 3.12, and what lets every tier name its types
without inheriting a stack. The first `scipy.ndimage` import would break it.

## Every dependency is pip-installable

Worth protecting: it makes this the first package in the suite whose CI can run *every* test
rather than skipping the conda-only paths (tensorstore, vol2mesh, dvidutils, neuclease,
kimimaro). `neu-lib` is numpy-only, so depending on it drags nothing in — it is what lets
`ops/` name a `Frame` while staying pure.

```bash
conda activate neu-env
pip install --no-deps -e ../neu-lib -e .
python -m pytest -q
```

## License

Apache-2.0. Copyright the Simons Foundation.
