"""Turn neuVid's orbits about the camera's own vertical, and make a looping one steady. A
Blender script, run by `neu-movie build` after addAnimation (and smooth_camera) whenever the
movie orbits:

    blender --background --python orbit.py -- -i <build>/<name>.json [--loop]

neuVid's `orbitCamera` with a `localAxis` copies `camera.matrix_world` to orient the
"Orbiter" it turns about, but nothing has evaluated the scene since the camera was posed, so
that matrix is the one saved with the file: Blender's default camera. Every orbit therefore
turned about the default camera's vertical, whatever the view. This re-orients each Orbiter
to the camera's own pose where the orbit starts, re-derives the constraint's inverse to
match, and re-keys where the camera is left when the orbit ends.

`--loop`: neuVid keys the orbit with Blender's default interpolation, which eases in and out,
so a looped movie would stall at the seam. This makes the turn linear. The orbit is keyed at
frames 1 and N + 1, so frame N + 1, the first one not rendered, is frame 1 again.

Edits `<build>/<name>Anim.blend` in place. Imports bpy, so it runs only inside Blender.
"""

import argparse
import os
import re
import sys

import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="orbit.py")
ap.add_argument("-i", dest="json", required=True, help="the build's neuVid JSON")
ap.add_argument("--loop", action="store_true", help="turn at constant speed")
args = ap.parse_args(argv)

blend = os.path.splitext(os.path.abspath(args.json))[0] + "Anim.blend"
bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
camera = bpy.data.objects["Camera"]
UP = mathutils.Vector((0, 1, 0))

orbiters = [o for o in bpy.data.objects if re.fullmatch(r"Orbiter\.\d+-\d+", o.name)]
if not orbiters:
    raise SystemExit("no orbit found in the .blend")
for orbiter in orbiters:
    start, end = (int(x) for x in orbiter.name.split(".")[1].split("-"))
    pivots = [c for c in orbiter.children if c.name.startswith("Pivot")]
    if not pivots:
        print(f"{orbiter.name}: a world-axis orbit, left as neuVid made it")
        continue
    pivot = pivots[0]
    constraint = camera.constraints[f"Orbit.{start}-{end}"]

    # The camera's own pose (its keys, before the constraint) where the orbit starts.
    scene.frame_set(start)
    pose = camera.matrix_basis.copy()
    was = (orbiter.matrix_world.to_quaternion() @ UP).normalized()
    centre = orbiter.matrix_world.translation.copy()
    orbiter.matrix_world = mathutils.Matrix.Translation(centre) @ \
        pose.to_quaternion().to_matrix().to_4x4()
    bpy.context.view_layer.update()
    constraint.inverse_matrix = orbiter.matrix_world.inverted()

    # Where the turn leaves the camera, keyed at the end frame, when the constraint lets go.
    scene.frame_set(end)
    left = pivot.matrix_world @ constraint.inverse_matrix @ pose
    euler = left.to_euler("XYZ", camera.rotation_euler)
    camera.location = left.translation
    camera.rotation_euler = euler
    camera.keyframe_insert("location", frame=end)
    camera.keyframe_insert("rotation_euler", frame=end)

    if args.loop:
        for fc in pivot.animation_data.action.fcurves:
            if fc.data_path == "rotation_euler":
                for k in fc.keyframe_points:
                    k.interpolation = "LINEAR"
                fc.update()

    # The checks that matter: the camera circles in a plane square to its vertical, and lands
    # where the turn left it. Sampled sparsely, since every frame_set re-evaluates every fade.
    samples = sorted(set(np.linspace(start, end - 1, 13).round().astype(int)))
    points = []
    for f in samples:
        scene.frame_set(f)
        points.append(np.array(camera.matrix_world.translation))
    points = np.array(points) - np.mean(points, axis=0)
    normal = mathutils.Vector(np.linalg.svd(points)[2][-1])
    up = (pose.to_quaternion() @ UP).normalized()
    tilt = np.degrees(normal.angle(up))
    tilt = min(tilt, 180 - tilt)
    def step(f):
        scene.frame_set(f - 1)
        before = camera.matrix_world.translation.copy()
        scene.frame_set(f)
        return (camera.matrix_world.translation - before).length

    print(f"{orbiter.name}: axis moved {np.degrees(was.angle(up)):.1f} deg onto the camera's "
          f"vertical; the orbit's plane is {tilt:.2f} deg from square to it; step into the "
          f"end frame {step(end):.4g}, mid-orbit {step((start + end) // 2):.4g}"
          + ("; linear" if args.loop else ""))

bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=blend)
