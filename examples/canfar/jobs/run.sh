#!/bin/bash
# Run one config. $CFG names the rewritten config, e.g. uds_f770w.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
export MPLCONFIGDIR=$RUN/.mplconfig
export MPLBACKEND=Agg
mkdir -p $RUN/out
cd $RUN/out
echo "=== $CFG on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
time $RUN/venv/bin/python -m mophongo.pipeline $RUN/${CFG}_canfar.json all
echo "=== outputs:"; ls -lh $RUN/out/$CFG | head -25
echo RUN_DONE
