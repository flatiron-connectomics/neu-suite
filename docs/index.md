# neu-suite

```{figure} _static/pathway-R7p-MeTu-TuBu-TL1a.jpg
:alt: Four reconstructed neurons forming a pathway from the medulla to the ellipsoid body, inside a translucent rendering of the brain
:width: 100%

A four-neuron pathway carrying visual information from the eye to the central complex:
**R7p** (yellow) in the medulla, **MeTu_MC1b** (blue) to the anterior optic tubercle,
**TuBu_BUs1** (magenta) to the bulb, and **TL1a** (green) to the ellipsoid body — meshes
and skeletons built with these packages from a proofread segmentation.
```

Python packages for working with large EM volumes, locally or on a SLURM cluster. They are separate
repositories with a strict one-way dependency, and the split is deliberate: the substrate
that knows how to run work on a cluster knows nothing about electron microscopy, and the
layer that knows about volumes knows nothing about meshes or biology.

```text
neu-lib   blockrun  vocabulary (BBox, Frame, Mesh, Skeleton — numpy only)
    ↑         ↑        and the dask/SLURM substrate. Neither knows about the other.
neu-vol             volume I/O: tensorstore backends, storage profiles,
    ↑                  convert / create / write / relabel, source metadata
    ├─ neu-morpho      per-body meshes (vol2mesh) and skeletons (TEASAR)
    ├─ neu-mark        DVID annotations into tables, and on into neuroglancer
    ├─ neu-glance      viewer states, layers, links and shaders
    └─ neu-proc        array processing: filters, label tools, distance transforms
           ↑
        neu-draw      local 3D rendering in a notebook, on pygfx
```

Nothing lower may import from anything higher, and consumers **at the same tier** do not
import each other.

## Which one do I want?

You almost certainly want a **command**, and there are four:

::::{grid} 2
:::{grid-item-card} `neu-vol`
The volumes themselves — inspect one, convert a source into a multiscale pyramid,
create an empty volume in a known frame and write pieces into it, renumber labels,
watch a run.
:::
:::{grid-item-card} `neu-morpho`
Meshes and skeletons from a segmentation, published into the same precomputed volume
so one neuroglancer layer shows labels, meshes and skeletons together.
:::
:::{grid-item-card} `neu-mark`
Annotations out of DVID — synapses and per-body records — into columnar tables, then on
into neuroglancer. Choose the bodies by synapse count, and label each synapse with the
neuropil it sits in.
:::
:::{grid-item-card} `neu-glance`
Anything a **viewer** consumes: a shareable link or a state, a layer of your own
coordinates, a layer of boxes showing where a sparse volume's data is. The other three
write data and know nothing about neuroglancer.
:::
::::

Four packages have no command. `neu-lib` holds the types they all share and `blockrun`
is the substrate they all run on; `neu-draw` is a notebook library — it draws meshes,
skeletons and synapse points locally with pygfx, where `neu-glance` sends a state to a
remote neuroglancer. `neu-proc` is the newest and the least finished: array-in, array-out
processing functions, with the blockwise machinery that will make them correct across
block boundaries designed but not built.

**Start at the [cheat sheet](_generated/cheatsheet.md)** — every subcommand on one page
with its synopsis. The [CLI reference](cli/index.md) has the full flag-by-flag detail,
generated from the parsers themselves so it always matches `--help`.

Calling them from Python instead — from a notebook, or from your own script — is the
[API reference](_generated/api-index.md): every module of all eight packages, parsed
from the source.

```{toctree}
:maxdepth: 2
:caption: Using them

getting-started
_generated/cheatsheet
cli/index
```

```{toctree}
:maxdepth: 2
:caption: API

_generated/api-index
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/converting
guides/small-pieces
guides/ground-truth
guides/viewing
guides/rendering
```

```{toctree}
:maxdepth: 1
:caption: Reference

_generated/neu-lib-readme
_generated/blockrun-readme
_generated/dask-slurm
_generated/neu-vol-readme
_generated/neu-morpho-readme
_generated/neu-morpho-skeletonization
_generated/neu-morpho-measure-calibration
_generated/neu-mark-readme
_generated/neu-glance-readme
_generated/neu-draw-readme
_generated/neu-proc-readme
```
