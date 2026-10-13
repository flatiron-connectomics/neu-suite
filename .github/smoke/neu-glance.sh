#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-glance --help > /dev/null
neu-glance gen --help > /dev/null
neu-glance shaders > /dev/null
python -c "
import neu_glance, sys
assert 'neu_vol' not in sys.modules, 'importing neu_glance pulled in neu_vol'
assert hasattr(neu_glance, 'build_state')
print('lazy exports resolve; version', neu_glance.__version__)
"
