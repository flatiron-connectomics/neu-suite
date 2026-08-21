# neu-suite

Documentation and the shared environment for a suite of EM volume packages developed at
the Flatiron Institute's Center for Computational Neuroscience.

**📖 [Documentation](https://flatiron-connectomics.github.io/neu-suite/)** — start at
the cheat sheet if you just need to remember a flag.

The packages live in their own repositories and are meant to sit here as siblings:

| repository | what it is |
| --- | --- |
| [blockrun](https://github.com/flatiron-connectomics/blockrun) | dask/SLURM substrate. No EM knowledge. Library only, no command. |
| [neu-vol](https://github.com/flatiron-connectomics/neu-vol) | volume I/O and the `neu-vol` command |
| [neu-morpho](https://github.com/flatiron-connectomics/neu-morpho) | meshes and skeletons, and the `neu-morpho` command |
| [neu-mark](https://github.com/flatiron-connectomics/neu-mark) | DVID annotations into tables and precomputed sources, and the `neu-mark` command |
| [neu-glance](https://github.com/flatiron-connectomics/neu-glance) | neuroglancer states, layers and links, and the `neu-glance` command |
| [neu-draw](https://github.com/flatiron-connectomics/neu-draw) | local 3D rendering in Jupyter, on pygfx. Library only, no command. |

The dependency order is one-way — `blockrun ← neu-vol ←
{neu-morpho, neu-mark, neu-glance} ← neu-draw` — and they depend on each other by
relative `../sibling` path, so the layout matters:

```text
neu-suite/          ← this repository
├── blockrun/       ← cloned separately
├── neu-vol/
├── neu-morpho/
├── neu-mark/
├── neu-glance/
└── neu-draw/
```

Consumers **at the same tier** do not import each other. In particular **neu-glance does
not import neu-mark**: it reads a precomputed annotation source's `info` like any other
store object. That is what keeps the viewer knowledge — shaders, states, links — out of
the packages that write data.

**neu-glance and neu-draw are not rivals**; the axis between them is *where the rendering
happens*. neu-glance emits a state for a remote neuroglancer, neu-draw draws pixels in
the notebook.

## The environment

One conda environment covers every package, installed editable. **Python 3.12 is
required**, and it is `vol2mesh` and `dvidutils` that force it — both are py312-only
*and* conda-only on the `flyem-forge` channel, with no PyPI equivalent, so pip can
never resolve them and a conda environment has to provide them.

```bash
conda create -n neu-env -c flyem-forge -c conda-forge python=3.12 \
    vol2mesh dvidutils kimimaro tensorstore zarr dask distributed dask-jobqueue \
    numpy scipy h5py tifffile imageio pandas pyarrow ngff-zarr jsonschema pyyaml
conda activate neu-env
pip install --no-deps -e ./blockrun -e ./neu-vol -e ./neu-morpho \
                      -e ./neu-mark -e ./neu-glance
```

`--no-deps` is load-bearing: the `pyproject.toml` files declare real runtime deps, and
without it pip re-resolves conda-provided binaries (tensorstore, h5py) from PyPI and
invites an ABI mismatch.

Add `neuclease` from `flyem-forge` for DVID sources — it needs `libdvid-cpp`, `vigra`
and `dvidutils`, none of which exist on PyPI, which is why there is no `dvid` pip extra
and cannot be one. `cloud-volume` is deliberately **not** in this environment: it pins
`DracoPy<2` and would downgrade the DracoPy behind neu-morpho's mesh serialisation, so
it belongs in a separate one.

The exact pinned specs for the environment used to develop this are not published —
they describe one machine's environment rather than a supported matrix.

## Building the docs locally

The site does **not** need the `neu-env` environment. Importing the two CLI modules pulls
in only dask, distributed and numpy, so PyPI is enough:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html -W docs docs/_build/html
```

The CLI reference is rendered from the packages' actual `ArgumentParser` objects and the
cheat sheet is generated from them at build time, so neither can drift away from
`--help`. Nothing under `docs/_generated/` is committed.

CI publishes to GitHub Pages on every push to `main`, and on a `repository_dispatch` of
type `package-updated` from any of the package repositories.
