#!/bin/bash
# Replace the mophongo source in place, leaving the venv alone.
#
# mophongo is installed editable, so the venv imports straight from
# $RUN/mophongo and a code change needs no reinstall. setup_env.sh deletes and
# rebuilds the venv, which would break any run using it, so use this to ship a
# fix while other jobs are in flight. Already-running jobs keep the code they
# imported; only later ones pick this up.
set -euo pipefail
: "${RUN:?RUN not set}"
tar -xzf $RUN/mophongo_src.tgz -C $RUN/mophongo
echo "source updated: $(find $RUN/mophongo -name '*.py' | wc -l) files"
$RUN/venv/bin/python -c "
from mophongo.pipeline import RunConfig
from mophongo.utils import as_label_array
print('imports ok')
"
echo UPDATE_DONE
