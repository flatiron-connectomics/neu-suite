# neu-movie

Animations of EM neurons and neuropils: a TOML spec in, a rendered movie out. Rendering is
[neuVid](https://github.com/connectome-neuprint/neuVid)'s, in Blender, usually on a Slurm GPU
node.

```
neu-movie check   SPEC [--links]        sets, counts, schedule, colour clashes; no Blender
neu-movie build   SPEC                  meshes and Blender files -> a new build folder
neu-movie stills  BUILD -f 300 450      render a few frames here, labelled
neu-movie submit  BUILD                 render on Slurm; the job then labels and encodes
neu-movie animate SPEC                  build, then submit
neu-movie relabel BUILD                 rewrite labels from the build's spec.toml ...
neu-movie finish  BUILD                 ... then label the frames and encode the movie
```

- **Spec:** one TOML file per animation. It names a neuroglancer view (the opening camera
  and the volumes), the sets of cells to show in order (`[[beat]]`), the camera, the timing
  and the labels. The format is documented in `neu_movie/spec.py`.
- **Build:** the folder one `build` makes, `renders/<spec name>_<timestamp>/` beside the
  spec. It holds what the build was made from, the resolved body ids, the neuVid input,
  the Blender files, the frames and the movie. Everything after `build` reads only the
  build.
- **Site config:** `configs/neu-movie.toml`, found above the spec or named by
  `$NEU_MOVIE_CONFIG`. It gives the locations of Blender, neuVid and the mesh cache, and
  the Slurm options. See `neu_movie/site.py`.

Install in the suite's environment with `pip install --no-deps -e ./neu-movie`.
