#!/bin/bash
# Redraw one run's scene figures from products it already wrote.
#
# For the bands that fitted successfully and then died rendering the figures:
# load_fit restores the post-run state and rebuilds the stamps file when it is
# missing or truncated, then scene_plots.py draws the figures. The residual,
# fit table and template table are not rewritten.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
export MPLCONFIGDIR=$RUN/.mplconfig
export MPLBACKEND=Agg
mkdir -p $RUN
cd $RUN
echo "=== $CFG [scene plots] on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
echo "=== mophongo: $(cat $RUN/setup/SRC_VERSION 2>/dev/null || echo 'SRC_VERSION missing')"
time $RUN/setup/venv/bin/python $RUN/jobs/scene_plots.py $RUN/setup/${CFG}_canfar.json
echo "=== scene figures: $(ls $RUN/run*/*/$CFG/scenes 2>/dev/null | wc -l)"
echo RUN_DONE
