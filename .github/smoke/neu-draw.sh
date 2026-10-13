#!/usr/bin/env bash
# The pip job installs no renderer, so the tests that check neu_draw imports without one
# only prove something if it really is absent.
set -euo pipefail
python -c "
import importlib.util
for mod in ('pygfx', 'wgpu', 'rendercanvas'):
    assert importlib.util.find_spec(mod) is None, f'{mod} is installed; the isolation test proves nothing here'
import neu_draw
print('neu_draw imports with no renderer available:', neu_draw.__version__)
"
