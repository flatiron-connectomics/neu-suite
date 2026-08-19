# em-libraries

Documentation and the shared environment for three EM volume packages developed at the
Flatiron Institute's Center for Computational Neuroscience.

**📖 [Documentation](https://flatiron-connectomics.github.io/em-libraries/)** — start at
the cheat sheet if you just need to remember a flag.

The packages live in their own repositories and are meant to sit here as siblings:

| repository | what it is |
| --- | --- |
| [em-blockrun](https://github.com/flatiron-connectomics/em-blockrun) | dask/SLURM substrate. No EM knowledge. Library only, no command. |
| [em-volume-tools](https://github.com/flatiron-connectomics/em-volume-tools) | volume I/O and the `em-vol` command |
| [em-seg-morpho](https://github.com/flatiron-connectomics/em-seg-morpho) | meshes and skeletons, and the `em-morpho` command |
| [em-annotation](https://github.com/flatiron-connectomics/em-annotation) | DVID annotations into tables and precomputed sources, and the `em-annot` command |
| [em-ngl](https://github.com/flatiron-connectomics/em-ngl) | neuroglancer states, layers and links, and the `em-ngl` command |

The dependency order is one-way — `em-blockrun ← em-volume-tools ←
{em-seg-morpho, em-annotation, em-ngl}` — and they depend on each other by relative
`../sibling` path, so the layout matters:

```text
em-libraries/          ← this repository
├── em-blockrun/       ← cloned separately
├── em-volume-tools/
├── em-seg-morpho/
├── em-annotation/
└── em-ngl/
```

The three consumers do not import each other. In particular **em-ngl does not import
em-annotation**: it reads a precomputed annotation source's `info` like any other store
object. That is what keeps the viewer knowledge — shaders, states, links — out of the
packages that write data.

## The environment

One conda environment, `em-lib`, covers all five, installed editable.

```bash
conda env create -n em-lib -f environment.yml
conda activate em-lib
pip install -r pypi_requirements.txt
pip install --no-deps -e ./em-blockrun -e ./em-volume-tools -e ./em-seg-morpho \
                      -e ./em-annotation -e ./em-ngl
```

`--no-deps` is load-bearing: without it pip re-resolves conda-provided binaries
(tensorstore, h5py) from PyPI and invites an ABI mismatch. Python 3.12 is required, and
it is `vol2mesh` and `dvidutils` that force it — both are py312-only and conda-only on
flyem-forge.

## Building the docs locally

The site does **not** need the `em-lib` environment. Importing the two CLI modules pulls
in only dask, distributed and numpy, so PyPI is enough:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html -W docs docs/_build/html
```

The CLI reference is rendered from the packages' actual `ArgumentParser` objects and the
cheat sheet is generated from them at build time, so neither can drift away from
`--help`. Nothing under `docs/_generated/` is committed.

CI publishes to GitHub Pages on every push to `main`, and on a `repository_dispatch` of
type `package-updated` from any of the three package repositories.
