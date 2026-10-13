#!/usr/bin/env bash
# The bundled dask templates must ship inside the wheel.
set -euo pipefail
python -m pip install -q build
python -m build -q --wheel -o "${RUNNER_TEMP:-/tmp}/dist-blockrun" .
python - <<'PY'
import glob, os, sys, zipfile
wheel = glob.glob(os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "dist-blockrun", "*.whl"))[0]
names = zipfile.ZipFile(wheel).namelist()
want = {"blockrun/configs/dask-local.yaml", "blockrun/configs/dask-slurm-example.yaml"}
missing = want - set(names)
sys.exit(f"missing from wheel: {sorted(missing)}" if missing else 0)
PY
