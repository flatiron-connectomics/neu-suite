# neu-movie

Animations of EM neurons and neuropils: a TOML spec in, a rendered movie out. A spec names
a neuroglancer view, the sets of cells to show beat by beat, how the camera moves and what
text goes on screen. `neu-movie` turns it into meshes and Blender files, renders them with
[neuVid](https://github.com/connectome-neuprint/neuVid) (usually on a Slurm GPU node), puts
the labels on the frames and encodes the video.

```
neu-movie check   SPEC [--links]        sets, counts, schedule, colour clashes; no Blender
neu-movie build   SPEC                  meshes and Blender files -> a new build folder
neu-movie stills  BUILD -f 300 450      render a few frames here, labelled
neu-movie submit  BUILD                 render on Slurm; the job then labels and encodes
neu-movie animate SPEC                  build, then submit
neu-movie relabel BUILD                 rewrite labels from the build's spec.toml ...
neu-movie finish  BUILD                 ... then label the frames and encode the movie
```

## Installing

In the suite's environment, `pip install --no-deps -e ./neu-movie`. It needs neu-vol (mesh
readback), neu-glance (view states, segment queries, links) and neu-recipe (volume names
from a library). Three programs that are not Python packages come from the site config:
**Blender** (4.2 or later; used with 4.5), **neuVid** (a checkout; neu-movie runs its
`importMeshes.py`, `addAnimation.py`, `render.py` and `compLabels.py`) and **ffmpeg**.

## A typical session

```bash
neu-movie check movies/pathway.toml --links     # what each beat shows; a link per beat
neu-movie build movies/pathway.toml             # -> movies/renders/pathway_20261116T1413/
neu-movie stills movies/renders/pathway_20261116T1413 -f 300 900
neu-movie submit movies/renders/pathway_20261116T1413 --time 01:00:00
```

`check` reads only segment properties, so it is the quick loop for a spec: the bodies in
each set, when each beat starts, how long the movie runs, and any two sets that share a
colour while on screen together. `--links` prints a neuroglancer link per beat with that
beat's sets as coloured layers, to inspect before anything is rendered. `--limit N` caps
every set at N bodies, as does `build --limit` for a fast test movie.

`submit` writes `render.sbatch` into the build and submits it. The job runs one Blender per
GPU, each on its own run of frames, then calls `neu-movie finish` to label and encode.
Resubmitting resumes: rendered frames are skipped. If the job ends before finishing, run
`neu-movie finish BUILD` anywhere Blender runs. `animate` is `build` then `submit`.

To change only the text, edit the build's own `spec.toml` (`[labels]`, beat titles), then
`neu-movie relabel BUILD` and `neu-movie finish BUILD`. Labels are composited onto the
rendered frames, so nothing is rendered again.

## The spec

One TOML file per movie. Every key is checked, so a misspelled one is an error naming the
keys that exist, never a silent default.

```toml
view = "pathway_view.json"   # a neuroglancer state: the opening camera, and the layers
mode = "accumulate"          # or "replace"

[neurons]
layer = "segmentation"       # the view's layer the sets come from
lod = 0
query = { without = ["fragment"] }   # defaults merged into every query set

[neuropils]                  # optional: regions shown with a set (its `regions`)
layer = "neuropils"

[head]                       # optional: the animal's outline, a faint shell
library = "my-specimen/rois-v1"      # a volume library name, or volume = "<url>"
id = 95

[timing]
start = 0.5
every = 2.2
fade = 2.4

[camera]
style = "follow"

[[beat]]                     # one set: its keys given on the beat
name = "R7p"
ids = [27332882]
regions = ["LA(L)", "ME(L)"]

[[beat]]                     # several sets
title = "optic lobe"
sets = [
    { name = "Mi", label = "^Mi" },
    { name = "Tm", label = "^Tm", exclude = [82159813] },
]
```

**Sets.** A set is either `ids = [...]` or a query on the segmentation's
`segment_properties`: `label` (a regular expression, searched), `tags` (all of),
`any_tags` (one of), `without` (none of), `limit` (the largest N). Query sets keep only
bodies whose mesh is in the store. `exclude = [...]` drops bodies either way, before the
limit is applied. `color = "#rrggbb"` overrides the palette; otherwise sets take tab10 then
Dark2 in order of first appearance, without the colours too dim to see on black.
`regions` names neuropils, by their segment names in the `[neuropils]` layer, that fade in
with the set. A set named in several beats is one set: define it once, then give only its
`name`.

**Modes.** In `replace`, a beat's sets fade in at its start and out `hold` seconds after
they have faded in, unless the next beat names them too, in which case they stay on. In
`accumulate`, every set stays once shown, so a set may appear in only one beat.

**Camera.** `orbit` frames the head (or, without one, every set) and turns once about the
screen's vertical from `orbit_at` until the last fade ends. `follow` moves to frame each
beat's sets together with the previous `window - 1` beats', in one continuous path, then
pulls back to the opening view and orbits for `orbit_seconds`.

**Labels.** Each beat's `title` shows at the top. A legend of the sets on screen, each name
in its set's colour, sits at the bottom left, with an optional `legend_header` line above
it. Labels fade with their cells. In `replace` the legend is per beat; in `accumulate` it
grows a row per set.

Every key, with its default:

| section | keys |
| --- | --- |
| top level | `view` (required), `name` (the file's stem), `fps = 30`, `mode = "replace"` |
| `[neurons]` | `layer` (required), `lod = 0`, `query = {}`, `workers = 16` |
| `[neuropils]` | `layer`, `alpha = 0.08`, `exponent = 7.0`, `mute = 0.5` (how far region colours move toward grey) |
| `[head]` | `library` or `volume`, `id`, `path = "rois/head"`, `lod = 0`, `alpha = 0.035`, `color = "#808080"`, `exponent = 7.0`, `offset = 4.0`, `intro_at = 1.0`, `intro_seconds = 0.0` (above 0, the head starts solid in `lit_color` and fades to the shell), `lit_color = "#d9d9d9"` |
| `[meshes]` | `smooth = 10` (Taubin iterations) |
| `[timing]` | `start = 0.5`, `every = 5.5`, `fade = 1.5`, `hold = 4.0`, `alpha = 1.0`, `tail = 1.5` (seconds after the content ends) |
| `[camera]` | `style = "orbit"`, `scale = 1.0` (framing margin), `orbit_degrees = 360.0`, `orbit_at = 1.0`, `window = 2`, `move_seconds` (defaults to `every`), `pullback_seconds = 4.0`, `orbit_seconds = 12.0` |
| `[labels]` | `titles = true`, `legend = true`, `legend_header = ""`, `title_size = 0.06`, `legend_size = 0.045`, `legend_x = 0.04`, `legend_bottom = 0.06`, `legend_step = 0.065` (sizes and positions are fractions of the frame height) |

## The build

`build` makes `renders/<spec name>_<YYYYmmddTHHMM>/` beside the spec. Everything after it
reads only the build, so `relabel`, `finish` and a resubmitted job work from its records
even if the spec or the segmentation has changed since.

| file | what it is |
| --- | --- |
| `spec.toml`, `<view>.json` | copies of what the build was made from |
| `sets.json` | each set's body ids and colour, as resolved |
| `build.json` | duration, frame count, the follow camera's smoothing range |
| `<name>.json`, `<name>_labels.json` | neuVid's input, and the labels |
| `<name>.blend`, `<name>Anim.blend` | the Blender files neuVid made from it |
| `links.txt` | a neuroglancer link per beat |
| `build-*.log`, `render-*.log`, `labels-*.log`, `slurm-*.out` | each step's output |
| `frames/`, `frames-labeled/`, `<name>.mp4` | the result |

Meshes are not in the build. They are exported once to the site's mesh cache, by source
volume, smoothing and level of detail, and shared by every build. A volume rewritten at the
same location looks unchanged to the cache, so pass `build --force` to export it again.

## The site config

`configs/neu-movie.toml`, found by walking up from the spec (or the build) and then the
working directory, or named by `$NEU_MOVIE_CONFIG`. Relative paths in it are relative to
the file.

```toml
blender = "/path/to/blender"             # the Blender executable
neuvid = "/path/to/neuVid/neuVid"        # neuVid's script directory
library = "library"                      # the volume library, for [head] library names
meshes = "/path/to/mesh-cache"           # exported meshes, shared by every build
ffmpeg_setup = "module load ffmpeg"      # run first when ffmpeg is not on PATH

[slurm]
options = ["--partition=gpu", "--gpus-per-node=4", "--cpus-per-task=64", "--mem=64G"]
time = "03:00:00"                        # default; `submit --time` overrides
```

Blender, neuVid and the Python running neu-movie must all be visible from the compute
nodes, since the render job calls back into `neu-movie finish`.

## Colours, and two neuVid details

neuVid reads a hex colour straight into a material as a linear value, and Blender's display
transform then brightens it, so sRGB colours would come out pale. neu-movie converts every
colour so a set renders in the colour the spec, `check` and `links.txt` give it, and its
legend text matches.

Labels are kept in their own file rather than in neuVid's input, because neuVid's
`addAnimation` stops on a label argument it does not know (`fade`). They are drawn by
`neu_movie/blender/labels.py`, which is neuVid's `compLabels.py` with each label's fade
matched to its cells and a cheaper renderer for flat text.
