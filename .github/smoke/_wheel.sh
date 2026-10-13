# Sourced by the smoke scripts: build this package's wheel and reinstall from it, so the
# checks after it exercise the installed package rather than the source tree.
TMP="${RUNNER_TEMP:-/tmp}"
python -m pip install -q build
python -m build -q --wheel -o "$TMP/dist-$(basename "$PWD")" .
python -m pip install -q --force-reinstall --no-deps "$TMP/dist-$(basename "$PWD")"/*.whl
