#!/bin/bash
# Build one shard of the campaign's ePSF grids, in parallel.
#
# One shard of the campaign's whole grid list. Every job is given every
# config and derives the same deduplicated (pattern, epoch) list from them,
# then takes the entries $SHARD selects -- so the shards are disjoint without
# talking to each other, and the fan-out is a free choice rather than one job
# per field.
#
# $CFGS is a comma-separated list of run names (uds_f770w,uds_f1280w,...);
# skaha splits a job's `args` on whitespace, so lists travel as environment
# variables and avoid spaces.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFGS:?CFGS not set}"
RUNNUM=${RUNNUM:-1}
CFGDIR=$RUN/run$RUNNUM/config
export MPLCONFIGDIR=$RUN/.mplconfig
mkdir -p "$MPLCONFIGDIR"

# stpsf caches the MAST wavefront files under $STPSF_PATH/MAST_JWST_WSS_OPDs.
# Unset, it resolves to /arc/home/<user>/data/stpsf-data: on the home quota
# this tree exists to stay off, and shared by every container either way. Name
# it in the run tree instead, so the cache is a release-level product like the
# grids beside it.
export STPSF_PATH=${STPSF_PATH:-$RUN/stpsf-data}
mkdir -p "$STPSF_PATH"

echo "=== psf shard ${SHARD:-1/1} on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
echo "=== STPSF_PATH=$STPSF_PATH"
echo "=== mophongo: $(cat $CFGDIR/SRC_VERSION 2>/dev/null || echo 'SRC_VERSION missing')"
echo "=== grids before: $(ls $RUN/PSF | wc -l)"

cfgs=()
IFS=',' read -ra names <<< "$CFGS"
for name in "${names[@]}"; do cfgs+=("$CFGDIR/${name}_canfar.json"); done
echo "=== configs: ${#cfgs[@]}"

time $CFGDIR/venv/bin/python $RUN/jobs/build_psfs.py \
    --workers "${WORKERS:-0}" --date-mode "${DATE_MODE:-all}" \
    --shard "${SHARD:-1/1}" ${PREWARM:+--prewarm-only} "${cfgs[@]}"

echo "=== grids after: $(ls $RUN/PSF | wc -l)"
echo PSF_DONE
