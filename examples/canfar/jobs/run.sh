#!/bin/bash
# Run one config. $CFG names the rewritten config, e.g. uds_f770w.
#
# $STEP is the pipeline step, default `all` (the whole fit). A campaign uses
# `prep` first, once per field: that builds the shared F444W ePSF grids and
# runs the saturation repair into a cache the field's bands then hit, instead
# of every band redoing both and writing the same cache file at once.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
RUNNUM=${RUNNUM:-1}
# A run pins one mophongo version: its source and venv live beside its configs
# in run<N>/config, not at the tree root, so two runs can differ.
CFGDIR=$RUN/run$RUNNUM/config
STEP=${STEP:-all}
export MPLCONFIGDIR=$RUN/.mplconfig
# Same shared OPD cache the psf step fills; a fit that has to autobuild an
# epoch reads it rather than fetching into the home quota.
export STPSF_PATH=${STPSF_PATH:-$RUN/stpsf-data}
export MPLBACKEND=Agg
mkdir -p $RUN
cd $RUN
echo "=== $CFG [$STEP] on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
# Which mophongo this is. Without it a run that silently used stale source -
# push writes the tarball, but only setup_env.sh and update_src.sh unpack it -
# looks exactly like one that picked the change up.
echo "=== mophongo: $(cat $CFGDIR/SRC_VERSION 2>/dev/null || echo 'SRC_VERSION missing')"
time $CFGDIR/venv/bin/python -m mophongo.pipeline $CFGDIR/${CFG}_canfar.json $STEP
# out_dir is absolute in the config: run<N>/<field>/<band>
echo "=== outputs:"; ls -lh $RUN/run*/*/$CFG 2>/dev/null | head -25
echo RUN_DONE
