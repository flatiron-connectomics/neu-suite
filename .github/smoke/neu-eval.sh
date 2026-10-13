#!/usr/bin/env bash
set -euo pipefail
NEU_PROC_GPU=0 python -m pytest -q          # again, with GPU transfers disabled
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-eval --help > /dev/null
neu-eval compare --help > /dev/null
neu-eval --version
python -c "
import neu_eval, sys
for name in ('neu_vol', 'tensorstore', 'pandas', 'dask', 'scipy'):
    assert name not in sys.modules, f'importing neu_eval pulled in {name}'
assert callable(neu_eval.contingency) and callable(neu_eval.compare)
print('lazy exports resolve; version', neu_eval.__version__)
"
