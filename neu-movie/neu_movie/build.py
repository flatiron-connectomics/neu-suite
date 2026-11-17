"""Builds: a folder per invocation of a spec, `renders/<spec name>_<YYYYmmddTHHMM>/` beside
the spec. It holds::

    spec.toml                  a copy of the spec it was made from
    sets.json                  the volumes, each set's body ids and colour, each neuropil's
                               segment id and colour, as resolved
    build.json                 duration, frame count, camera smoothing range, orbit, fill light
    <name>.json                neuVid's input; <name>.blend, <name>Anim.blend from it
    <name>_labels.json         the labels
    links.txt                  neuroglancer links: everything, then each beat
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

def location(section, site):
    """The URL of a `[neurons]`, `[neuropils]` or `[head]` volume: its library volume or
    `volume`, with its `path` under it."""
    if section["volume"]:
        base = section["volume"]
    else:
        from neu_recipe import library
        base = library.location(section["library"], directory=site.library)
    path = (section["path"] or "").strip("/")
    return base.rstrip("/") + (f"/{path}" if path else "")


def unit_nm(volume):
    """neuVid's unit, in nm: the neuron volume's finest voxel."""
    from neu_vol import read_scales
    size = read_scales(volume)[0].frame.voxel_size_nm
    if len(set(size)) != 1:
        raise spec_mod.SpecError(f"{volume} has anisotropic voxels {size}; neuVid units need "
                                 f"one voxel size")
    return float(size[0])


def resolve(spec, site, *, limit=None):
    """What the movie shows, as `world`: the volumes, neuVid's unit, each set's ids and
    colour, each neuropil's segment id and colour. Reads metadata and segment properties,
    no meshes."""
    volume = location(spec["neurons"], site)
    ids = sets.resolve(spec, volume, limit=limit)
    if not ids:
        raise spec_mod.SpecError("no set matched anything")
    colours = sets.colours(spec)
    for a, b, c in sets.clashes(spec, colours, ids):
        log(f"  ! {a!r} and {b!r} are on screen together and share {c}")
    world = {"volume": volume, "unit": unit_nm(volume), "ids": ids, "colours": colours,
             "roi_volume": None, "regions": {},
             "head_volume": location(spec["head"], site) if spec["head"] else None}
    regions = list(dict.fromkeys(r for name in ids for r in spec["sets"][name]["regions"]))
    if regions:
        from neu_glance.sources import segment_ids

        from neu_movie.palette import neuropil_colours
        nps = spec["neuropils"]
        world["roi_volume"] = location(nps, site)
        own = {r: [colours[n] for n in ids if r in spec["sets"][n]["regions"]] for r in regions}
        chosen = neuropil_colours(regions, nps["colors"], colours.values(), own, nps["mute"])
        for name, sid in zip(regions, segment_ids(world["roi_volume"], regions)):
            world["regions"][name] = {"id": str(sid), "color": chosen[name]}
            log(f"  neuropil {name}: {chosen[name]}"
                + (" (the spec's)" if name in nps["colors"] else ""))
    return world


def gather(spec, site, world, *, force=False):
    """Export every mesh the movie needs and return the `scene` `compose` takes."""
    unit, smooth = world["unit"], spec["meshes"]["smooth"]
    cache = site.meshes / f"smooth{smooth}"
    lod = spec["neurons"]["lod"]
    neuron_dir = cache / f"neurons-{volume_tag(world['volume'])}-lod{lod}"
    bodies = sorted({b for members in world["ids"].values() for b in members}, key=int)
    meshes.export_bodies(world["volume"], bodies, str(neuron_dir), unit, lod, smooth=smooth,
                         force=force, workers=spec["neurons"]["workers"])

    scene = {"unit": unit, "ids": world["ids"], "colours": world["colours"],
             "neuron_dir": str(neuron_dir), "roi_dir": None, "head": None,
             "region_colours": {r: v["color"] for r, v in world["regions"].items()}}
    if world["roi_volume"]:
        scene["roi_dir"] = str(cache / f"rois-{volume_tag(world['roi_volume'])}")
        for name, region in world["regions"].items():
            meshes.export(world["roi_volume"], int(region["id"]),
                          os.path.join(scene["roi_dir"], f"{safe_name(name)}.obj"),
                          unit, None, force=force, smooth=smooth)
    head = spec["head"]
    if head:
        if not scene["roi_dir"]:
            scene["roi_dir"] = str(cache / "rois-head")
        tag = volume_tag(world["head_volume"])
        head_dir = cache / f"head-{tag}-lod{head['lod']}"
        # The shell is the same mesh as an ROI, moved off the lit head (see meshes.export).
        roi = f"head{tag[-8:]}id{head['id']}lod{head['lod']}off{head['offset']:g}"
        meshes.export(world["head_volume"], head["id"], str(head_dir / f"{head['id']}.obj"),
                      unit, head["lod"], force=force, one_piece=True, smooth=smooth,
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
    spec = spec_mod.load(d / "spec.toml")
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

    log("== sets")
    world = resolve(spec, site, limit=limit)
    ids, colours = world["ids"], world["colours"]
    write_json(d / "sets.json", {
        "volume": world["volume"], "unit_nm": world["unit"], "limit": limit,
        "sets": {n: {"color": colours[n], "ids": ids[n]} for n in ids},
        "neuropils": world["roi_volume"], "regions": world["regions"],
        "head": world["head_volume"]})
    with open(d / "links.txt", "w") as f:
        for title, url in links.links(spec, world):
            f.write(f"# {title}\n{url}\n\n")

    log("== meshes")
    scene = gather(spec, site, world, force=force)
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
    if info.get("fill"):
        steps.append(("fill_light", BLENDER_SCRIPTS / "fill_light.py",
                      ["-i", json_path, "--strength", info["fill"]]))
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
    regions = {r: v["color"] for r, v in recorded.get("regions", {}).items()}
    write_json(d / f"{info['name']}_labels.json",
               labels_mod.compose(spec, ids, colours, info["seconds"], regions))
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
    lab = spec["labels"]
    if (d / f"{info['name']}_labels.json").exists() and (
            lab["titles"] or lab["legend"] or lab["neuropil_legend"]):
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
