"""Light the scene from the camera as well, so a side turned away from neuVid's lights is
never black. A Blender script, run by `neu-movie build` last:

    blender --background --python fill_light.py -- -i <build>/<name>.json --strength S

neuVid's render.py adds its lights when it renders, fixed in the world and aimed for the
opening view, and gives Cycles no ambient light, so an orbit that comes round behind the
scene shows surfaces lit by nothing. This adds a sun lamp parented to the camera and shining
along its view: it moves with every camera move and orbit, and a sun has no falloff, so its
brightness does not change with zoom. `--strength` is its irradiance in W/m^2; each of
neuVid's lamps gives roughly 4 at the scene's centre. render.py leaves the lamp alone.

Edits `<build>/<name>Anim.blend` in place. Imports bpy, so it runs only inside Blender.
"""

import argparse
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="fill_light.py")
ap.add_argument("-i", dest="json", required=True, help="the build's neuVid JSON")
ap.add_argument("--strength", type=float, required=True, help="irradiance, W/m^2")
args = ap.parse_args(argv)

blend = os.path.splitext(os.path.abspath(args.json))[0] + "Anim.blend"
bpy.ops.wm.open_mainfile(filepath=blend)
camera = bpy.data.objects["Camera"]

light = bpy.data.lights.new("Fill", type="SUN")
light.energy = args.strength
light.angle = 0.5                   # radians: soft, since it is a fill and not a key
light.use_shadow = False            # anything it would shadow is hidden from the camera
fill = bpy.data.objects.new("Fill", light)
bpy.context.scene.collection.objects.link(fill)
# A sun shines along its own -z, as a camera looks: with the camera as parent and no
# transform of its own, it shines wherever the camera looks.
fill.parent = camera
fill.matrix_parent_inverse = mathutils.Matrix.Identity(4)
fill.matrix_basis = mathutils.Matrix.Identity(4)

bpy.context.view_layer.update()
aim = (fill.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))).normalized()
view = (camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))).normalized()
print(f"fill light: sun of {args.strength:g} W/m^2 on the camera, "
      f"{aim.angle(view) * 57.2958:.2f} deg off its view")

bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=blend)
