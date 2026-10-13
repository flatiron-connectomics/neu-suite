#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
python -c "
import sys, neu_proc
for heavy in ('scipy', 'cc3d', 'fastmorph', 'skimage'):
    assert heavy not in sys.modules, f'importing neu_proc pulled in {heavy}'
assert callable(neu_proc.dust) and callable(neu_proc.dilate)
print('lazy exports resolve; version', neu_proc.__version__)
"
