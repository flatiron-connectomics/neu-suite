#!/usr/bin/env bash
# The DVID tests must have SKIPPED here (no neuclease): zero skips means either neuclease
# got installed or those tests stopped being collected, and pytest reports neither.
set -euo pipefail
grep -qE '[0-9]+ skipped' "$PYTEST_OUT" || {
  echo "::error::no tests were skipped; the DVID suite is expected to skip without neuclease"
  exit 1
}
source "$(dirname "$0")/_wheel.sh"
cd "$TMP"
neu-mark --help > /dev/null
neu-mark select-bodies --help > /dev/null
python -c "
import neu_mark, sys
assert 'pandas' not in sys.modules, 'importing neu_mark pulled in pandas'
assert hasattr(neu_mark, 'select_bodies')
print('lazy exports resolve; version', neu_mark.__version__)
"
