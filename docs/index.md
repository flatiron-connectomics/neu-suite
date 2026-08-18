# em-libraries

Three Python packages for working with large EM volumes on Flatiron's clusters. They are
separate repositories with a strict one-way dependency, and the split is deliberate: the
substrate that knows how to run work on a cluster knows nothing about electron
microscopy, and the layer that knows about volumes knows nothing about meshes.

```text
em-blockrun        dask/SLURM substrate. No EM knowledge.
    ↑              Manifest, block_map, iter_blocks, start_dask, dask configs
em-volume-tools    volume I/O: tensorstore backends, storage profiles,
    ↑              convert / create / write / relabel, source metadata
em-seg-morpho      per-body meshes (vol2mesh) and skeletons (kimimaro)
```

Nothing lower may import from anything higher.

## Which one do I want?

You almost certainly want a **command**, and there are two:

::::{grid} 2
:::{grid-item-card} `em-vol`
The volumes themselves — inspect one, convert a source into a multiscale pyramid,
create an empty volume in a known frame and write pieces into it, renumber labels,
watch a run, or emit a viewer layer showing where the data is.
:::
:::{grid-item-card} `em-morpho`
Meshes and skeletons from a segmentation, published into the same precomputed volume
so one neuroglancer layer shows labels, meshes and skeletons together.
:::
::::

`em-blockrun` has no command. It is the library both of them run on.

**Start at the [cheat sheet](_generated/cheatsheet.md)** — every subcommand on one page
with its synopsis. The [CLI reference](cli/index.md) has the full flag-by-flag detail,
generated from the parsers themselves so it always matches `--help`.

```{toctree}
:maxdepth: 2
:caption: Using them

getting-started
_generated/cheatsheet
cli/index
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/converting
guides/small-pieces
guides/ground-truth
guides/viewing
```

```{toctree}
:maxdepth: 1
:caption: Reference

_generated/em-blockrun-readme
_generated/em-volume-tools-readme
_generated/em-volume-tools-design
_generated/em-seg-morpho-readme
_generated/em-seg-morpho-design
_generated/em-annotation-readme
_generated/dask-slurm-rusty
```
