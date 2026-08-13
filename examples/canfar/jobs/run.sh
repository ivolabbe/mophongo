#!/bin/bash
# Run one config. $CFG names the rewritten config, e.g. uds_f770w.
#
# $STEP is the pipeline step, default `all` (the whole fit). A campaign uses
# `prep` first, once per field: that builds the shared F444W ePSF grids and
# runs the saturation repair into a cache the field's bands then hit, instead
# of every band redoing both and writing the same cache file at once.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
STEP=${STEP:-all}
export MPLCONFIGDIR=$RUN/.mplconfig
export MPLBACKEND=Agg
mkdir -p $RUN/out
cd $RUN/out
echo "=== $CFG [$STEP] on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
# Which mophongo this is. Without it a run that silently used stale source -
# push writes the tarball, but only setup_env.sh and update_src.sh unpack it -
# looks exactly like one that picked the change up.
echo "=== mophongo: $(cat $RUN/SRC_VERSION 2>/dev/null || echo 'SRC_VERSION missing')"
time $RUN/venv/bin/python -m mophongo.pipeline $RUN/${CFG}_canfar.json $STEP
echo "=== outputs:"; ls -lh $RUN/out/$CFG | head -25
echo RUN_DONE
