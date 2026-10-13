#!/usr/bin/env bash
# The installed console script runs outside a checkout, and blockrun's templates resolve.
set -euo pipefail
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-vol --help > /dev/null
python -c "
from blockrun import bundled_configs, resolve_config
assert 'dask-local' in bundled_configs(), bundled_configs()
print('bundled dask templates resolve:', resolve_config('dask-local'))
"
