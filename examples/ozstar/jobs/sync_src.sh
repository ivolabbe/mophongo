#!/bin/bash
# Pull the latest mophongo into the run tree, leaving the venv alone.
#
# mophongo is installed editable, so the venv imports straight from
# $SRC and a code change needs no reinstall. Use this rather than
# setup_env.sh REBUILD=1 while jobs are queued: already-running jobs keep the
# code they imported, and queued ones pick this up when they start - which is
# worth knowing before pulling mid-campaign.
set -euo pipefail
: "${BASE:?BASE not set}" "${RUN:?RUN not set}" "${CFGDIR:?CFGDIR not set}" "${VENV:?VENV not set}" "${SRC:?SRC not set}"
# Lmod here is hierarchical: python/3.12.3 is only visible once its gcccore
# parent is loaded. Unquoted on purpose - it is a list, not one name.
PYMODULES=${PYMODULES:-"gcccore/13.3.0 python/3.12.3"}
BRANCH=${BRANCH:-main}

module purge
module load $PYMODULES
unset PYTHONPATH   # keep the EasyBuild shim off sys.path; see setup_env.sh

git -C "$SRC" fetch --quiet origin "$BRANCH"
git -C "$SRC" checkout --quiet "$BRANCH"
git -C "$SRC" reset --hard "origin/$BRANCH"
echo "mophongo $(git -C "$SRC" log --oneline -1)"

"$VENV/bin/python" -c "
from mophongo.pipeline import RunConfig
from mophongo.utils import as_label_array
print('imports ok')
"
echo UPDATE_DONE
