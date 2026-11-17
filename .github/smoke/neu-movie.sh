#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-movie --help > /dev/null
neu-movie submit --help > /dev/null
python -c "
import neu_movie.build as b, sys
for script in ('labels.py', 'smooth_camera.py', 'orbit.py', 'fill_light.py'):
    assert (b.BLENDER_SCRIPTS / script).is_file(), f'the wheel has no blender/{script}'
assert 'neu_glance' not in sys.modules, 'importing neu_movie.build pulled in neu_glance'
print('blender scripts installed; version', __import__('neu_movie').__version__)
"
