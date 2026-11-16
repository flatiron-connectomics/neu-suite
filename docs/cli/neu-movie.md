# `neu-movie`

From `neu-movie`. Turns a movie spec — a TOML file naming a neuroglancer view, the sets of
cells to show in order, the camera and the labels — into a BUILD folder of meshes and
Blender files, renders it with [neuVid](https://github.com/connectome-neuprint/neuVid)
(usually on a Slurm GPU node), then labels the frames and encodes the movie. The spec
format, the build folder and the site config are described in the
[neu-movie README](../_generated/neu-movie-readme.md).

`python -m neu_movie` is equivalent.

```{argparse}
:module: clitools
:func: neu_movie_parser
:prog: neu-movie
```
