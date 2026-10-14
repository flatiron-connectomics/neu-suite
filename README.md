# neu-suite

A suite of EM volume packages developed at the Flatiron Institute's Center for
Computational Neuroscience. They are developed together in this repository and installed
separately: each top-level directory is its own Python distribution.

**📖 [Documentation](https://flatiron-connectomics.github.io/neu-suite/)** — start at
the cheat sheet if you just need to remember a flag.

| package | what it is |
| --- | --- |
| [neu-lib](neu-lib/) | the shared vocabulary: boxes, grid arithmetic, frames, meshes, skeletons. **numpy only.** Library, no command. |
| [blockrun](blockrun/) | dask/SLURM substrate. No EM knowledge. Library only, no command. |
| [neu-vol](neu-vol/) | volume I/O and the `neu-vol` command |
| [neu-morpho](neu-morpho/) | meshes and skeletons, and the `neu-morpho` command |
| [neu-mark](neu-mark/) | DVID annotations into tables and precomputed sources, and the `neu-mark` command |
| [neu-glance](neu-glance/) | neuroglancer states, layers and links, and the `neu-glance` command |
| [neu-proc](neu-proc/) | array processing: filters, label tools, distance transforms. Library only, no command. |
| [neu-draw](neu-draw/) | local 3D rendering in Jupyter, on pygfx. Library only, no command. |
| [neu-eval](neu-eval/) | comparison metrics against a reference labeling, and the `neu-eval` command |

The dependency order is one-way — `{neu-lib, blockrun} ← neu-vol ←
{neu-morpho, neu-mark, neu-glance, neu-proc} ← {neu-draw, neu-eval}` — and they depend on
each other by relative `../sibling` path:

```text
neu-suite/
├── neu-lib/
├── blockrun/
├── neu-vol/
├── neu-morpho/
├── neu-mark/
├── neu-glance/
├── neu-proc/
├── neu-draw/
└── neu-eval/
```

`neu-lib` and `blockrun` sit side by side at the bottom and know nothing about each
other: one is types, the other is orchestration. **`neu-lib` depends on numpy alone**,
which is why the shared types are reachable from every tier without dragging an I/O
stack down with them — and why it is the one package that still builds on Python 3.11.

Consumers **at the same tier** do not import each other. In particular **neu-glance does
not import neu-mark**: it reads a precomputed annotation source's `info` like any other
store object. That is what keeps the viewer knowledge — shaders, states, links — out of
the packages that write data.

**neu-glance and neu-draw are not rivals**; the axis between them is *where the rendering
happens*. neu-glance emits a state for a remote neuroglancer, neu-draw draws pixels in
the notebook.

**neu-proc is the newest and the least finished.** Only the pure half exists — array in,
array out — and that is a phasing choice rather than a stub: an op that works on one array
is the reference the blockwise runner will be checked against. It is also the one package
whose every dependency is pip-installable, so its CI runs every test rather than skipping
the conda-only paths.

## The environment

One conda environment covers every package, installed editable. **Python 3.12 is
required**, and it is `vol2mesh` and `dvidutils` that force it — both are py312-only
*and* conda-only on the `flyem-forge` channel, with no PyPI equivalent, so pip can
never resolve them and a conda environment has to provide them.

```bash
git clone https://github.com/flatiron-connectomics/neu-suite.git
cd neu-suite
mamba env create -n neu-env -f environment.yml    # or micromamba; plain conda is slow
conda activate neu-env
make install                                     # every package, editable, --no-deps
make test                                        # every suite
```

`environment.yml` is the one environment spec, used by CI as well. The optional GPU path
in neu-proc and neu-eval needs the cupy wheel matching your CUDA, installed by hand
(see the comment at the top of `environment.yml`).

`--no-deps` is load-bearing: the `pyproject.toml` files declare real runtime deps, and
without it pip re-resolves conda-provided binaries (tensorstore, h5py) from PyPI and
invites an ABI mismatch.

**DVID sources are optional.** They need `neuclease`, also conda-only on `flyem-forge`
(it needs `libdvid-cpp`, `vigra` and `dvidutils`, none on PyPI), so there is no `dvid` pip
extra. Add it by layering `environment-dvid.yml` when you create the environment:
`mamba create -n neu-env -f environment.yml -f environment-dvid.yml`. `cloud-volume` is deliberately **not** in this environment: it pins
`DracoPy<2` and would downgrade the DracoPy behind neu-morpho's mesh serialisation, so
it belongs in a separate one.

Every CI run uploads the exact environment it tested (`conda` explicit list and
`pip freeze`) as a build artifact.

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

CI builds the site with `-W` on every push and pull request, and publishes it to GitHub
Pages from `main` once every package's tests pass.

## License

neu-morpho is **GPL-3.0-or-later**, because its skeletonization uses `kimimaro` and
`dijkstra3d`. Every other package is **Apache-2.0**. Each package directory carries its
own `LICENSE`.
