"""Composite a build's labels onto its rendered frames. A Blender script, run by
`neu-movie finish` with $NEUVID naming neuVid's script directory:

    blender --background --python labels.py -- \\
        -i <build>/<name>_labels.json -if <build>/frames -o <build>/frames-labeled [-s N -e M]

neuVid's `compLabels.py`, with two of its functions replaced rather than forked:

- a label's `fade` (seconds) sets its fade in and out, so it can match its cells'; compLabels
  fixes every fade at 4 frames;
- label materials BLEND rather than dither, so a half-faded label is clean at a few EEVEE
  samples. compLabels renders at Blender's default 64, ~10x slower for flat text.

Labels live in `<name>_labels.json` beside the build's neuVid input, never inside it:
addAnimation exits on a `label` argument it does not know, such as `fade`.
"""

import argparse
import os
import sys

import bpy

sys.path.insert(0, os.environ["NEUVID"])

import compLabels as cl  # noqa: E402
from utilsMaterials import insertMaterialKeyframe, setMaterialValue  # noqa: E402

_process_label = cl.process_label


def process_label(cmd, time, fps, image_size):
    label = _process_label(cmd, time, fps, image_size)
    if label is not None:
        label["fade_frames"] = cmd[1].get("fade", 4 / fps) * fps
    return label


def setup_label_animation(obj, label):
    mat = bpy.data.materials[f"Material.{obj.name}"]
    mat.surface_render_method = "BLENDED"
    start, end = label["start_frame"], label["end_frame"]
    fade = min(label["fade_frames"], (end - start) / 2)
    for frame, alpha in ((start, 0), (start + fade, 1), (end - fade, 1), (end, 0)):
        setMaterialValue(mat, "alpha", alpha)
        insertMaterialKeyframe(mat, "alpha", frame)


cl.process_label = process_label
cl.setup_label_animation = setup_label_animation


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="labels.py")
    ap.add_argument("-i", dest="labels", required=True, help="<build>/<name>_labels.json")
    ap.add_argument("-if", dest="frames", required=True, help="the rendered frames")
    ap.add_argument("-o", dest="output", required=True, help="where labelled frames go")
    ap.add_argument("-s", dest="start", type=int, default=None, help="first frame")
    ap.add_argument("-e", dest="end", type=int, default=None, help="last frame")
    ap.add_argument("--samples", type=int, default=4, help="EEVEE samples per frame")
    args = ap.parse_args(argv)

    image_size = cl.get_image_size(args.frames)
    labels = cl.parse_labels(args.labels, image_size)
    labels_scene = cl.setup_labels_scene(labels)
    try:
        labels_scene.render.engine = "BLENDER_EEVEE_NEXT"     # Blender 4.2 to 4.5
    except TypeError:
        labels_scene.render.engine = "BLENDER_EEVEE"          # other versions' name for it
    labels_scene.eevee.taa_render_samples = args.samples
    comp_scene = cl.setup_comp_scene(os.path.join(args.output, ""))
    cl.comp_labels(labels_scene, comp_scene, args.frames, image_size, args.start, args.end)


main()
