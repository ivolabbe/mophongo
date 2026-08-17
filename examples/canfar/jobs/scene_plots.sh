#!/bin/bash
# Redraw one run's scene figures from products it already wrote.
#
# For the bands that fitted successfully and then died rendering the figures:
# load_fit restores the post-run state and rebuilds the stamps file when it is
# missing or truncated, then scene_plots.py draws the figures. The residual,
# fit table and template table are not rewritten.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
RUNNUM=${RUNNUM:-1}
# A run pins one mophongo version: its source and venv live beside its configs
# in run<N>/config, not at the tree root, so two runs can differ.
CFGDIR=$RUN/run$RUNNUM/config
export MPLCONFIGDIR=$RUN/.mplconfig
export MPLBACKEND=Agg
mkdir -p $RUN
cd $RUN
echo "=== $CFG [scene plots] on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
echo "=== mophongo: $(git -C $CFGDIR/mophongo log --oneline -1 2>/dev/null || echo 'no checkout')"
time $CFGDIR/venv/bin/python $RUN/jobs/scene_plots.py $CFGDIR/${CFG}_canfar.json
echo "=== scene figures: $(ls $RUN/run*/*/$CFG/scenes 2>/dev/null | wc -l)"
echo RUN_DONE
