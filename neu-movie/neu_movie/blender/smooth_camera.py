"""Join neuVid's back-to-back camera moves into one continuous path. A Blender script, run by
`neu-movie build` after addAnimation for a follow camera:

    blender --background --python smooth_camera.py -- -i <build>/<name>.json --start S --end E

neuVid keys each `frameCamera` as its own move: a key where it starts (a copy of where the
previous move ended, a frame later) and a key where it ends, eased in and out. A sequence of
moves therefore stops at every key — the step-and-hold look. Between frames S and E this
deletes every key that only repeats the position before it, and gives the remaining keys
automatic (unclamped) Bezier handles, so the camera passes through each framing without
stopping. The key at S keeps its handles, so the path still eases out of rest there, and keys
from E on are untouched (the orbit is driven by a constraint over keys that must stay put).

It also turns each camera rotation the short way. `poseCamera` keys Euler angles and Blender
blends each angle on its own, so of the two Euler triples for one orientation, neuVid's
`make_compatible` can leave the far one: a 25 degree turn keyed as (54, 177, -33) ->
(-104, 11, 141) swung the camera through several hundred degrees. Each rotation key is
rewritten as whichever triple is nearer the key before it.

Edits `<build>/<name>Anim.blend` in place, which is what render.py reads. Imports bpy, so it
runs only inside Blender.
"""

import argparse
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="smooth_camera.py")
ap.add_argument("-i", dest="json", required=True, help="the build's neuVid JSON")
ap.add_argument("--start", type=int, required=True, help="first frame of the joined path")
ap.add_argument("--end", type=int, required=True, help="frame the joined path ends at")
args = ap.parse_args(argv)

blend = os.path.splitext(os.path.abspath(args.json))[0] + "Anim.blend"
bpy.ops.wm.open_mainfile(filepath=blend)
camera = bpy.data.objects["Camera"]
curves = [fc for fc in camera.animation_data.action.fcurves if fc.data_path == "location"]
if len(curves) != 3:
    raise SystemExit(f"expected 3 camera location curves, found {len(curves)}")


def keys_at(fc):
    return {int(round(k.co[0])): k for k in fc.keyframe_points}


# A frame is a duplicate when every channel's key there repeats the previous key's value.
frames = sorted(set.intersection(*(set(keys_at(fc)) for fc in curves)))
dropped = []
for prev, f in zip(frames, frames[1:]):
    if args.start < f <= args.end and all(
            abs(keys_at(fc)[f].co[1] - keys_at(fc)[prev].co[1]) < 1e-6 for fc in curves):
        dropped.append(f)
for fc in curves:
    for f in dropped:
        fc.keyframe_points.remove(keys_at(fc)[f])
    for f, k in keys_at(fc).items():
        # Interpolation belongs to the segment AFTER a key, so the start key needs it too —
        # neuVid's ease-in-out there would brake to a stop at the first framing. Its handles
        # stay as they are (flat beside the identical key before it), easing out of rest.
        if args.start <= f < args.end:
            k.interpolation = "BEZIER"
        if args.start < f < args.end:
            k.handle_left_type = k.handle_right_type = "AUTO"
    fc.update()



def nearest_euler(e, prev):
    """Of the two XYZ Euler triples for `e`'s orientation, each made compatible with
    `prev`, the one nearer `prev`."""
    alt = mathutils.Euler((e.x + math.pi, math.pi - e.y, e.z + math.pi), "XYZ")
    best = []
    for c in (e.copy(), alt):
        c.make_compatible(prev)
        best.append((sum(abs(a - b) for a, b in zip(c, prev)), c))
    return min(best, key=lambda dc: dc[0])[1]


turns = [fc for fc in camera.animation_data.action.fcurves if fc.data_path == "rotation_euler"]
if turns:
    by_frame = [keys_at(fc) for fc in turns]
    frames_r = sorted(set.intersection(*(set(k) for k in by_frame)))
    for prev_f, f in zip(frames_r, frames_r[1:]):
        prev = mathutils.Euler([k[prev_f].co[1] for k in by_frame], "XYZ")
        e = mathutils.Euler([k[f].co[1] for k in by_frame], "XYZ")
        n = nearest_euler(e, prev)
        if max(abs(a - b) for a, b in zip(n, e)) > 1e-6:
            turn = math.degrees(prev.to_quaternion().rotation_difference(e.to_quaternion()).angle)
            print(f"rotation key at frame {f}: {[round(math.degrees(v)) for v in e]} -> "
                  f"{[round(math.degrees(v)) for v in n]} deg, a turn of "
                  f"{min(turn, 360 - turn):.1f} deg from the key before")
        for k, v in zip(by_frame, n):
            k[f].co[1] = v
    for fc in turns:
        fc.update()

kept = [f for f in sorted(keys_at(curves[0])) if args.start <= f <= args.end]
print(f"smoothed camera between frames {args.start} and {args.end}: dropped {len(dropped)} "
      f"repeated keys, path now through {kept}")

# The check that matters: speed along the path. A stop shows as a speed near zero inside it.
scene = bpy.context.scene
speeds = []
for f in range(args.start, args.end + 1):
    scene.frame_set(f)
    p = camera.matrix_world.translation.copy()
    if f > args.start:
        speeds.append((f, (p - last).length))
    last = p
peak = max(s for _, s in speeds) or 1.0
inner = [(f, s) for f, s in speeds if args.start + 10 < f < args.end - 10]
slowest = min(inner, key=lambda fs: fs[1]) if inner else None
if slowest:
    print(f"camera speed: peak {peak:.3g}/frame; slowest inside the path {slowest[1] / peak:.0%} "
          f"of peak, at frame {slowest[0]}")
# No `.blend1` backup of the file being replaced: it is as big as the file (GBs for
# thousands of bodies), and the build can always make it again.
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=blend)
