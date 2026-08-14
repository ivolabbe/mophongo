#!/bin/bash
# Build every ePSF grid this field's configs will ask for, in parallel.
#
# One job per field, because the grids a field's bands share -- the F444W
# photometry set and the 30" halo set -- are one work list, and build_psfs.py
# deduplicates it. Fields share nothing, so their jobs run concurrently.
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

echo "=== psf build on $(hostname): $(nproc) cores, $(free -g | awk '/Mem/{print $2}')GB"
echo "=== STPSF_PATH=${STPSF_PATH:-<unset: stpsf will use the image default>}"
echo "=== mophongo: $(cat $CFGDIR/SRC_VERSION 2>/dev/null || echo 'SRC_VERSION missing')"
echo "=== grids before: $(ls $RUN/PSF | wc -l)"

cfgs=()
IFS=',' read -ra names <<< "$CFGS"
for name in "${names[@]}"; do cfgs+=("$CFGDIR/${name}_canfar.json"); done
echo "=== configs: ${#cfgs[@]}"

time $CFGDIR/venv/bin/python $RUN/jobs/build_psfs.py \
    --workers "${WORKERS:-0}" --date-mode "${DATE_MODE:-all}" "${cfgs[@]}"

echo "=== grids after: $(ls $RUN/PSF | wc -l)"
echo PSF_DONE
