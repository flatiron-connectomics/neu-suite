"""Builds: a folder per invocation of a spec, `renders/<spec name>_<YYYYmmddTHHMM>/` beside
the spec. It holds::

    spec.toml, <view>.json     copies of what it was made from (and of a `level` view)
    sets.json                  each set's body ids and colour, as resolved
    build.json                 duration, frame count, camera smoothing range, orbit
    <name>.json                neuVid's input; <name>.blend, <name>Anim.blend from it
    <name>_labels.json         the labels
    links.txt                  a neuroglancer link per beat
    build-*.log                Blender's output for each step
    frames/, frames-labeled/   rendered, then labelled
    <name>.mp4

Everything after the build reads only the build, so `relabel` and `finish` work from its
records even if the spec or the segmentation has changed since.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from neu_movie import compose as compose_mod
from neu_movie import labels as labels_mod
from neu_movie import links, meshes, sets, site as site_mod, spec as spec_mod
from neu_movie.meshes import safe_name, volume_tag

BLENDER_SCRIPTS = Path(__file__).parent / "blender"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# -- reading the world --------------------------------------------------------------- #

def _layer(state, name):
    for lyr in state["layers"]:
        if lyr.get("name") == name:
            return lyr
    raise spec_mod.SpecError(f"the view has no layer {name!r}; it has "
                             f"{', '.join(repr(lyr.get('name')) for lyr in state['layers'])}")


def unit_nm(state):
    dims = state["dimensions"]
    sizes = {float(dims[a][0]) for a in "xyz"}
    if len(sizes) != 1:
        raise spec_mod.SpecError(f"anisotropic display dimensions {dims}: neuVid units need "
                                 f"one voxel size")
    return sizes.pop() * 1e9


def head_volume(spec, site):
    head = spec["head"]
    if head["volume"]:
        base = head["volume"]
    else:
        from neu_recipe import library
        base = library.location(head["library"], directory=site.library)
    return base.rstrip("/") + "/" + head["path"].strip("/")


def level_reference(spec):
    """The orientation of the spec's `[camera] level` view, or None."""
    if not spec["camera"]["level"]:
        return None
    with open(spec["camera"]["level"]) as f:
        return json.load(f).get("projectionOrientation", [0, 0, 0, 1])


def resolve(spec, *, limit=None):
    """The view, the neuron volume, each set's ids and colour. Reads segment properties,
    nothing else."""
    from neu_glance.sources import layer_volume
    from neu_glance.state import load_state
    view = load_state(spec["view"])
    volume = layer_volume(_layer(view, spec["neurons"]["layer"]))
    ids = sets.resolve(spec, volume, limit=limit)
    if not ids:
        raise spec_mod.SpecError("no set matched anything")
    colours = sets.colours(spec)
    for a, b, c in sets.clashes(spec, colours, ids):
        log(f"  ! {a!r} and {b!r} are on screen together and share {c}")
    return view, volume, ids, colours


def gather(spec, site, view, volume, ids, colours, *, force=False):
    """Export every mesh the movie needs and return the `scene` `compose` takes."""
    from neu_glance.sources import layer_volume, segment_ids
    unit = unit_nm(view)
    smooth = spec["meshes"]["smooth"]
    cache = site.meshes / f"smooth{smooth}"
    lod = spec["neurons"]["lod"]
    neuron_dir = cache / f"neurons-{volume_tag(volume)}-lod{lod}"
    bodies = sorted({b for members in ids.values() for b in members}, key=int)
    meshes.export_bodies(volume, bodies, str(neuron_dir), unit, lod, smooth=smooth,
                         force=force, workers=spec["neurons"]["workers"])

    scene = {"view": view, "ids": ids, "colours": colours, "neuron_dir": str(neuron_dir),
             "roi_dir": None, "region_colours": {}, "head": None,
             "level": level_reference(spec)}
    regions = list(dict.fromkeys(r for name in ids for r in spec["sets"][name]["regions"]))
    if regions and not spec["neuropils"]:
        raise spec_mod.SpecError("sets name `regions` but the spec has no [neuropils] layer")
    if spec["neuropils"]:
        np_layer = _layer(view, spec["neuropils"]["layer"])
        roi_vol = layer_volume(np_layer)
        scene["roi_dir"] = str(cache / f"rois-{volume_tag(roi_vol)}")
        region_ids = dict(zip(regions, segment_ids(roi_vol, regions))) if regions else {}
        for name, sid in region_ids.items():
            meshes.export(roi_vol, sid, os.path.join(scene["roi_dir"], f"{safe_name(name)}.obj"),
                          unit, None, force=force, smooth=smooth)
        # A region's colour: the view's, else the next neuropil palette colour unused there.
        from neu_movie.palette import NEUROPILS
        view_colours = np_layer.get("segmentColors", {})
        unused = iter(c for c in NEUROPILS if c not in view_colours.values())
        scene["region_colours"] = {name: view_colours.get(region_ids[name]) or
                                   next(unused, "#ffffff") for name in regions}
    head = spec["head"]
    if head:
        if not scene["roi_dir"]:
            scene["roi_dir"] = str(cache / "rois-head")
        head_vol = head_volume(spec, site)
        tag = volume_tag(head_vol)
        head_dir = cache / f"head-{tag}-lod{head['lod']}"
        # The shell is the same mesh as an ROI, moved off the lit head (see meshes.export).
        roi = f"head{tag[-8:]}id{head['id']}lod{head['lod']}off{head['offset']:g}"
        meshes.export(head_vol, head["id"], str(head_dir / f"{head['id']}.obj"), unit,
                      head["lod"], force=force, one_piece=True, smooth=smooth,
                      offset_copy=(os.path.join(scene["roi_dir"], f"{roi}.obj"),
                                   head["offset"]))
        scene["head"] = {"dir": str(head_dir), "roi": roi}
    return scene


# -- builds -------------------------------------------------------------------------- #

def renders_dir(spec):
    return Path(spec["path"]).parent / "renders"


def new_build(spec):
    d = renders_dir(spec) / f"{spec['name']}_{time.strftime('%Y%m%dT%H%M')}"
    if d.exists():
        raise FileExistsError(f"{d} already exists; wait a minute or remove it")
    d.mkdir(parents=True)
    return d


def open_build(path):
    """``(build dir, spec, records)`` for an existing build."""
    d = Path(path).resolve()
    if d.is_file():
        d = d.parent
    if not (d / "build.json").exists():
        raise FileNotFoundError(f"{d} is not a build (no build.json)")
    # The build's own copy of the view, wherever the original spec found it.
    import tomllib
    with open(d / "spec.toml", "rb") as f:
        raw = tomllib.load(f)
    raw["view"] = os.path.basename(raw.get("view", ""))
    if raw.get("camera", {}).get("level"):
        raw["camera"]["level"] = os.path.basename(raw["camera"]["level"])
    spec = spec_mod.parse(raw, str(d / "spec.toml"))
    with open(d / "build.json") as f:
        info = json.load(f)
    with open(d / "sets.json") as f:
        recorded = json.load(f)
    return d, spec, info, recorded


def write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def blender(site, script, *args, log_path=None, env=None):
    # Blender reports a script it cannot open and still exits 0.
    if not Path(script).is_file():
        raise FileNotFoundError(f"no Blender script {script}")
    cmd = [str(site.blender), "--background", "--python", str(script), "--", *map(str, args)]
    with open(log_path, "w") if log_path else open(os.devnull, "w") as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})})
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(shlex.quote, cmd))} exited {result.returncode}"
                           + (f"; see {log_path}" if log_path else ""))


def build(spec_path, *, limit=None, force=False, blend=True):
    """Make a build from the spec at `spec_path`. Returns its folder."""
    spec = spec_mod.load(spec_path)
    site = site_mod.load(Path(spec_path).parent)
    d = new_build(spec)
    log(f"== build {d}")
    shutil.copy(spec_path, d / "spec.toml")
    for view_file in (spec["view"], spec["camera"]["level"]):
        if view_file:
            shutil.copy(view_file, d / os.path.basename(view_file))

    log("== sets")
    view, volume, ids, colours = resolve(spec, limit=limit)
    write_json(d / "sets.json", {"volume": volume, "limit": limit,
                                 "sets": {n: {"color": colours[n], "ids": ids[n]} for n in ids}})
    with open(d / "links.txt", "w") as f:
        for title, url in links.beat_links(spec, view, ids, colours):
            f.write(f"# {title}\n{url}\n\n")

    log("== meshes")
    scene = gather(spec, site, view, volume, ids, colours, force=force)
    doc, labels, info = compose_mod.compose(spec, scene)
    name = spec["name"]
    info["name"] = name
    write_json(d / f"{name}.json", doc)
    write_json(d / f"{name}_labels.json", labels)
    write_json(d / "build.json", info)
    log(f"  {info['seconds']:g} s at {spec['fps']} fps = {info['frames']} frames, "
        f"{sum(len(v) for v in ids.values())} bodies")
    if blend:
        make_blend(d, site, info)
    return d


def make_blend(d, site, info):
    name = info["name"]
    json_path = d / f"{name}.json"
    steps = [("importMeshes", site.neuvid / "importMeshes.py", ["-i", json_path]),
             ("addAnimation", site.neuvid / "addAnimation.py", ["-i", json_path])]
    if info["smooth"]:
        s, e = info["smooth"]
        steps.append(("smooth_camera", BLENDER_SCRIPTS / "smooth_camera.py",
                      ["-i", json_path, "--start", s, "--end", e]))
    if info.get("orbit"):
        steps.append(("orbit", BLENDER_SCRIPTS / "orbit.py",
                      ["-i", json_path] + (["--loop"] if info.get("loop") else [])))
    for step, script, args in steps:
        log(f"== {step} (log: {d / f'build-{step}.log'})")
        blender(site, script, *args, log_path=d / f"build-{step}.log")
    # neuVid reports its own errors and still exits 0, so check that the files were made.
    for blend_file in (d / f"{name}.blend", d / f"{name}Anim.blend"):
        if not blend_file.is_file() or blend_file.stat().st_size == 0:
            raise RuntimeError(f"no {blend_file}; see {d}/build-*.log")


# -- after the build ------------------------------------------------------------------- #

def relabel(path):
    """Rewrite a build's labels from its own spec.toml and records. Edit the build's
    spec.toml ([labels], beat titles) first; the cells' colours and timing stay as built."""
    d, spec, info, recorded = open_build(path)
    ids = {n: s["ids"] for n, s in recorded["sets"].items()}
    colours = {n: s["color"] for n, s in recorded["sets"].items()}
    spec = dict(spec, beats=[b for b in spec["beats"] if any(n in ids for n in b["sets"])])
    write_json(d / f"{info['name']}_labels.json",
               labels_mod.compose(spec, ids, colours, info["seconds"]))
    log(f"wrote {d / (info['name'] + '_labels.json')}")
    return d


def frame_numbers(directory):
    return sorted(int(p.stem) for p in Path(directory).glob("*.png") if p.stem.isdigit())


def label_frames(d, site, info, frames_dir, out_dir, processes=1, frames=None):
    """Composite the build's labels onto `frames_dir` into `out_dir`, in `processes` passes
    side by side, each on its own run of frames."""
    have = frame_numbers(frames_dir) if frames is None else frames
    if not have:
        raise RuntimeError(f"no frames in {frames_dir}")
    out_dir.mkdir(exist_ok=True)
    labels_json = d / f"{info['name']}_labels.json"
    per = -(-len(have) // max(1, processes))
    chunks = [have[i:i + per] for i in range(0, len(have), per)]
    procs = []
    for k, chunk in enumerate(chunks):
        log_path = d / f"labels-{k}.log"
        log(f"labels: frames {chunk[0]}-{chunk[-1]} -> {log_path}")
        cmd = [str(site.blender), "--background", "--python",
               str(BLENDER_SCRIPTS / "labels.py"), "--", "-i", str(labels_json),
               "-if", str(frames_dir), "-o", str(out_dir), "-s", str(chunk[0]),
               "-e", str(chunk[-1])]
        procs.append((subprocess.Popen(cmd, stdout=open(log_path, "w"),
                                       stderr=subprocess.STDOUT,
                                       env={**os.environ, "NEUVID": str(site.neuvid)}),
                      log_path))
    failed = [str(p) for proc, p in procs if proc.wait() != 0]
    if failed:
        raise RuntimeError(f"label pass failed; see {', '.join(failed)}")


def encode(d, site, info, frames_dir, output):
    first = frame_numbers(frames_dir)[0]
    with open(d / f"{info['name']}.json") as f:
        fps = json.load(f)["fps"]
    cmd = shlex.join(["ffmpeg", "-loglevel", "error", "-y", "-framerate", f"{fps:g}",
                      "-start_number", str(first), "-i", str(Path(frames_dir) / "%04d.png"),
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)])
    if not shutil.which("ffmpeg") and site.ffmpeg_setup:
        cmd = f"{site.ffmpeg_setup} && {cmd}"
    result = subprocess.run(["bash", "-lc", cmd])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {cmd}")
    log(f"wrote {output}")


def finish(path, processes=1):
    """Label a build's frames and encode the movie."""
    d, spec, info, _ = open_build(path)
    site = site_mod.load(d)
    frames = d / "frames"
    src = frames
    if (d / f"{info['name']}_labels.json").exists() and (spec["labels"]["titles"] or
                                                         spec["labels"]["legend"]):
        src = d / "frames-labeled"
        label_frames(d, site, info, frames, src, processes)
    encode(d, site, info, src, d / f"{info['name']}.mp4")
    return d


def stills(path, frames, processes=1):
    """Render a few frames on this machine, into `<build>/stills/`, then label them into
    `<build>/stills-labeled/`."""
    d, spec, info, _ = open_build(path)
    site = site_mod.load(d)
    out = d / "stills"
    out.mkdir(exist_ok=True)
    for f in frames:
        log(f"== frame {f}")
        blender(site, site.neuvid / "render.py", "-i", d / f"{info['name']}.json", "--optix",
                "--nocomp", "-s", f, "-e", f, "-o", out, log_path=d / f"stills-{f}.log")
    label_frames(d, site, info, out, d / "stills-labeled", processes, frames=sorted(frames))
    return d / "stills-labeled"


# -- the cluster --------------------------------------------------------------------- #

JOB = """#!/bin/bash
#SBATCH --job-name=neu-movie
{options}
#SBATCH --time={time}
#SBATCH --output={build}/slurm-%j.out
# Renders frames {first}-{last} of {build}, one Blender per GPU, each on a contiguous range,
# then (once every frame is there) labels and encodes the movie. Resubmitting resumes:
# frames already rendered are skipped.
set -euo pipefail
export NEU_MOVIE_CONFIG={site}
first={first}; last={last}; total={total}
out={build}/frames
mkdir -p "$out"

IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
n=${{#gpus[@]}}
threads=$(( SLURM_CPUS_PER_TASK / n ))
per=$(( (last - first + n) / n ))
echo "$(hostname): frames $first-$last on $n GPU(s) [${{CUDA_VISIBLE_DEVICES}}], $per each"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

pids=()
for i in "${{!gpus[@]}}"; do
    s=$(( first + i * per )); e=$(( s + per - 1 ))
    (( s > last )) && break
    (( e > last )) && e=$last
    log={build}/render-$SLURM_JOB_ID-gpu$i.log
    echo "gpu ${{gpus[$i]}}: frames $s-$e -> $log"
    CUDA_VISIBLE_DEVICES=${{gpus[$i]}} {blender} --background --python {render} -- \\
        -i {json} --optix --nocomp --skipExisting --threads "$threads" \\
        -s "$s" -e "$e" -o "$out" > "$log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${{pids[@]}}"; do wait "$p" || status=$?; done
done=$(ls "$out" | grep -c '\\.png$' || true)
echo "rendered (status $status): $done of $total frames in $out"
(( status != 0 )) && exit $status

if (( done >= total )); then
    {python} -m neu_movie finish {build} --processes "$n"
fi
"""


def job_script(path, *, time_limit=None, first=1, last=None):
    d, spec, info, _ = open_build(path)
    site = site_mod.load(d)
    q = shlex.quote
    options = "\n".join(f"#SBATCH {o}" for o in site.slurm.get("options", []))
    text = JOB.format(
        options=options, time=time_limit or site.slurm.get("time", "01:00:00"),
        build=q(str(d)), site=q(str(site.path)), first=first, last=last or info["frames"],
        total=info["frames"], blender=q(str(site.blender)),
        render=q(str(site.neuvid / "render.py")), json=q(str(d / f"{info['name']}.json")),
        python=q(sys.executable))
    job = d / "render.sbatch"
    job.write_text(text)
    return job


def submit(path, *, time_limit=None, first=1, last=None, dry_run=False):
    job = job_script(path, time_limit=time_limit, first=first, last=last)
    if dry_run:
        log(f"wrote {job}; submit it with:\nsbatch {job}")
        return None
    result = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    log(result.stdout.strip())
    d, total = job.parent, open_build(job.parent)[2]["frames"]
    jobid = result.stdout.split()[-1] if result.stdout.split() else "<jobid>"
    log(f"follow it with:\n  squeue -u $USER\n  tail -f {d}/slurm-{jobid}.out\n"
        f"  echo \"$(ls {d}/frames | wc -l) of {total} frames\"")
    return result.stdout.strip()
