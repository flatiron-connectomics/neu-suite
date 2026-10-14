#!/usr/bin/env bash
# Runs in the full-environment job: the conda-only pieces are present, and the installed
# console script runs outside a checkout. dvidutils ships no .dist-info, so only an
# import can show it is there.
set -euo pipefail
python -c "
import importlib
for m in ('vol2mesh', 'dvidutils', 'tensorstore', 'neu_morpho'):
    importlib.import_module(m); print(f'  ok  {m}')
"
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-morpho --help > /dev/null
