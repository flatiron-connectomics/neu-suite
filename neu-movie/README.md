# neu-movie

Animations of EM neurons and neuropils: a TOML spec in, a rendered movie out. A spec names
the volumes, the opening camera, the sets of cells to show beat by beat, how the camera
moves and what text goes on screen. `neu-movie` turns it into meshes and Blender files, renders them with
[neuVid](https://github.com/connectome-neuprint/neuVid) (usually on a Slurm GPU node), puts
the labels on the frames and encodes the video.

```
neu-movie camera  STATE                 a spec's [view] from a neuroglancer state or link
neu-movie check   SPEC [--links]        sets, counts, schedule, colour clashes; no Blender
neu-movie build   SPEC                  meshes and Blender files -> a new build folder
neu-movie stills  BUILD -f 300 450      render a few frames here, labelled
neu-movie submit  BUILD                 render on Slurm; the job then labels and encodes
neu-movie animate SPEC                  build, then submit
neu-movie relabel BUILD                 rewrite labels from the build's spec.toml ...
neu-movie finish  BUILD                 ... then encode the movie, and a labelled one
```

## Installing

In the suite's environment, `pip install --no-deps -e ./neu-movie`. It needs neu-vol (mesh
readback), neu-glance (segment queries, links) and neu-recipe (volume names from a
library). Three programs that are not Python packages come from the site config:
**Blender** (4.2 or later; used with 4.5), **neuVid** (a checkout; neu-movie runs its
`importMeshes.py`, `addAnimation.py`, `render.py` and `compLabels.py`) and **ffmpeg**.

## A typical session

```bash
neu-movie camera framed.json                    # paste the [view] it prints into the spec
neu-movie check movies/pathway.toml --links     # what the movie shows; links to inspect it
neu-movie build movies/pathway.toml             # -> movies/renders/pathway_20261116T1413/
neu-movie stills movies/renders/pathway_20261116T1413 -f 300 900
neu-movie submit movies/renders/pathway_20261116T1413 --time 01:00:00
```

`camera` reads a neuroglancer state, a JSON file or a link, framed the way the movie should
open, and prints the spec's `[view]` from it, with the `up` that would level an orbit about
that view's vertical.

`check` reads only metadata and segment properties, so it is the quick loop for a spec: the
bodies in each set, each neuropil's colour, when each beat starts, how long the movie runs,
and any two sets that share a colour while on screen together. `--links` prints
neuroglancer links showing what the movie shows and nothing else: first everything, then
each beat's sets as they are on screen during it (in `accumulate`, everything so far), each
set a layer in its colour, its neuropils in their colours muted as drawn, and the head's
shell. They open on the movie's camera; for an orbit, a beat's link is turned as far as the
movie has turned by the middle of the beat, though about the view's position rather than
the centre the movie frames. `build` writes the same links to the build's `links.txt`.
`--limit N` caps every set at N bodies, as does `build --limit` for a fast test movie.

`submit` writes `render.sbatch` into the build and submits it. The job runs one Blender per
GPU, each on its own run of frames, then calls `neu-movie finish`, which encodes the frames
as `<name>.mp4`, labels them and encodes `<name>_labeled.mp4` (`--no-labels`: only the
first).
Resubmitting resumes: rendered frames are skipped. If the job ends before finishing, run
`neu-movie finish BUILD` anywhere Blender runs. `animate` is `build` then `submit`.

To change only the text, edit the build's own `spec.toml` (`[labels]`, beat titles), then
`neu-movie relabel BUILD` and `neu-movie finish BUILD`. Labels are composited onto the
rendered frames, so nothing is rendered again.

## The spec

One TOML file per movie. Every key is checked, so a misspelled one is an error naming the
keys that exist, never a silent default.

```toml
mode = "accumulate"          # or "replace"

[view]                       # the opening camera, as `neu-movie camera` prints it
position = [49596, 34908, 46028]     # what it looks at, nm
orientation = [0.1099, 0.4417, 0.8507, 0.2630]   # neuroglancer's projectionOrientation
height = 96000               # nm spanned top to bottom at `position`

[neurons]                    # the segmentation the sets come from
library = "my-specimen/v1"   # a volume library name, or volume = "<url>"
lod = 0
query = { without = ["fragment"] }   # defaults merged into every query set

[neuropils]                  # optional: shown with a set (its `neuropils`)
library = "my-specimen/v1"
path = "rois/neuropils"      # the neuropil layer under the library volume
colors = { "ME(L)" = "#e6ab02" }     # optional: chosen colours, kept as they are

[head]                       # optional: the animal's outline, a faint shell
library = "my-specimen/v1"   # path defaults to "rois/head"
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
neuropils = ["LA(L)", "ME(L)"]

[[beat]]                     # several sets
title = "optic lobe"
sets = [
    { name = "Mi", label = "^Mi" },
    { name = "Tm", label = "^Tm", exclude = [82159813] },
]
```

**Sources.** `[neurons]`, `[neuropils]` and `[head]` each name a volume, by its name in
the volume library (`library`) or its URL (`volume`), and a `path` under it. neuVid's unit
is the neuron volume's finest voxel, read from its metadata, so the volumes must share a
coordinate space.

**Sets.** A set is either `ids = [...]` or a query on the segmentation's
`segment_properties`: `label` (a regular expression, searched), `tags` (all of),
`any_tags` (one of), `without` (none of), `limit` (the largest N). Query sets keep only
bodies whose mesh is in the store. `exclude = [...]` drops bodies either way, before the
limit is applied. `color = "#rrggbb"` overrides the palette; otherwise sets take tab10 then
Dark2 in order of first appearance, without the colours too dim to see on black.
`neuropils` names neuropils, by their segment names in the `[neuropils]` volume, that fade in
with the set. A neuropil takes its colour in `[neuropils] colors`, else one of tab10, Dark2
and Set2 that stays a colour once muted, so it never reads as the head's grey, and is
at least 90 degrees of hue from the sets shown with it, so the neurons inside it stand out.
A chosen colour also keeps its hue from the other neuropils of the same sets, which those
neurons join, and is used once before any is used twice; one far from every set's is
preferred. `build` prints each neuropil's colour. A set named in several beats is one set:
define it once, then give only its `name`.

**Modes.** In `replace`, a beat's sets fade in at its start and out `hold` seconds after
they have faded in, unless the next beat names them too, in which case they stay on. In
`accumulate`, every set stays once shown, so a set may appear in only one beat.

`[timing] repeat = N` plays the beats N times over, in `replace` mode, each pass with its
own `start` and `tail`. With a looping orbit of one turn, `repeat = 2` shows every set once
from the front and once from behind.

**Camera.** `[view]` is the opening camera. `orbit` frames the head (with
`frame = "sets"`, or without a head, every set) and turns once about the screen's vertical
from `orbit_at`, for `orbit_seconds` or else until the last fade ends; a longer orbit runs
on after the beats, and the movie with it. With `loop = true` it turns at constant speed
from the first frame to the last instead, and the frame after the last is the first, so
the movie plays on a loop without a seam. `follow` moves to frame each beat's sets together
with the previous `window - 1` beats', in one continuous path, then pulls back to the
opening view and orbits for `orbit_seconds` (default 12). `glide` opens framed on the first
beat and makes one slow move, from the first beat's start until the last beat has faded in,
to frame every set, then orbits for `orbit_seconds` (default 12). A positive
`orbit_degrees` moves the camera to its right, so the scene turns to the left. When the opening view is tilted, the
orbit is too; `up = [x, y, z]` names the vertical to turn about instead (`neu-movie camera`
prints a view's). `follow` pulls back to the opening direction levelled to it; `orbit` and
`glide` are levelled throughout.

**Light.** neuVid lights the scene with lamps fixed in the world, aimed for the opening view,
and no ambient light, so a camera that orbits round to the back sees black surfaces.
`[light] fill` adds a lamp that moves with the camera and shines along its view, in W/m²
(each of neuVid's lamps gives about 4; `0` for none).

**Labels.** Text drawn over the frames, three parts each with its own switch in `[labels]`.
`beat_titles`: each beat's `title` at the top. `neuron_legend`: the sets on screen at the
bottom left, each name in its set's colour, under an optional `neuron_legend_header`.
`neuropil_legend`: the neuropils at the bottom right, right-aligned, each name in the colour
its neuropil is drawn in, under an optional `neuropil_legend_header`. Labels fade with
their cells. In `replace` the legends are per beat; in `accumulate` they grow a row per set
and per neuropil. With all three off, `finish` makes only the unlabelled movie.

Every key, with its default:

| section | keys |
| --- | --- |
| top level | `name` (the file's stem), `fps = 30`, `mode = "replace"` |
| `[view]` | `position` (nm, required), `orientation = [0, 0, 0, 1]`, `height` (nm, required) |
| `[neurons]` | `library` or `volume`, `path = ""`, `lod = 0`, `query = {}`, `workers = 16` |
| `[neuropils]` | `library` or `volume`, `path` (required with `library`), `colors = {}`, `alpha = 0.08`, `exponent = 7.0`, `mute = 0.5` (how far neuropil colours move toward grey) |
| `[head]` | `library` or `volume`, `id`, `path = "rois/head"`, `lod = 0`, `alpha = 0.035`, `color = "#808080"`, `exponent = 7.0`, `offset = 4.0`, `intro_at = 1.0`, `intro_seconds = 0.0` (above 0, the head starts solid in `lit_color` and fades to the shell), `lit_color = "#d9d9d9"` |
| `[meshes]` | `smooth = 10` (Taubin iterations) |
| `[timing]` | `start = 0.5`, `every = 5.5`, `fade = 1.5`, `hold = 4.0`, `alpha = 1.0`, `tail = 1.5` (seconds after the content ends), `repeat = 1` |
| `[camera]` | `style = "orbit"`, `scale = 1.0` (framing margin), `orbit_degrees = 360.0`, `orbit_at = 1.0`, `loop = false`, `window = 2`, `move_seconds` (defaults to `every`), `pullback_seconds = 4.0`, `orbit_seconds` (orbit: until the last fade ends; follow and glide: 12), `up` (a vertical), `frame = "head"` (or `"sets"`; orbit) |
| `[light]` | `fill = 2.0` |
| `[labels]` | `beat_titles = true`, `neuron_legend = true`, `neuron_legend_header = ""`, `neuropil_legend = true`, `neuropil_legend_header = ""`, `title_size = 0.06`, `legend_size = 0.045`, `legend_x = 0.04`, `legend_bottom = 0.06`, `legend_step = 0.065` (sizes and positions are fractions of the frame height) |

## The build

`build` makes `renders/<spec name>_<YYYYmmddTHHMM>/` beside the spec. Everything after it
reads only the build, so `relabel`, `finish` and a resubmitted job work from its records
even if the spec or the segmentation has changed since.

| file | what it is |
| --- | --- |
| `spec.toml` | a copy of the spec the build was made from |
| `sets.json` | the volumes and neuVid's unit, each set's body ids and colour, each neuropil's segment id and colour, as resolved |
| `build.json` | duration, frame count, the follow camera's smoothing range, whether the orbit loops, the fill light |
| `<name>.json`, `<name>_labels.json` | neuVid's input, and the labels |
| `<name>.blend`, `<name>Anim.blend` | the Blender files neuVid made from it |
| `links.txt` | neuroglancer links: everything the movie shows, then each beat |
| `build-*.log`, `render-*.log`, `labels-*.log`, `slurm-*.out` | each step's output |
| `frames/`, `frames-labeled/` | the rendered frames, and the same with their labels |
| `<name>.mp4`, `<name>_labeled.mp4` | the movie, and the movie with its labels |

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

## Colours, and four neuVid details

neuVid reads a hex colour straight into a material as a linear value, and Blender's display
transform then brightens it, so sRGB colours would come out pale. neu-movie converts every
colour so a set renders in the colour the spec, `check` and `links.txt` give it, and its
legend text matches.

Labels are kept in their own file rather than in neuVid's input, because neuVid's
`addAnimation` stops on a label argument it does not know (`fade`). They are drawn by
`neu_movie/blender/labels.py`, which is neuVid's `compLabels.py` with each label's fade
matched to its cells and a cheaper renderer for flat text.

neuVid's `orbitCamera` turns about the vertical of a camera matrix nothing has updated since
the file was saved, which is Blender's default camera rather than the view, so its orbits
come out tilted. `neu_movie/blender/orbit.py`, run after `addAnimation` whenever there is an
orbit, turns each one about the camera's own vertical where it starts, and for `loop` makes
the turn linear.

neuVid's `poseCamera` keys Euler angles, and Blender blends each angle separately. Of the two
Euler triples for an orientation, it can key the far one, so a small turn becomes a swing of
hundreds of degrees. `neu_movie/blender/smooth_camera.py` rewrites each rotation key as the
triple nearest the key before it. The fill light is added last, by
`neu_movie/blender/fill_light.py`, as a sun lamp parented to the camera.
