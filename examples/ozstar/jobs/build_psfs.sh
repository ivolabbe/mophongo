#!/bin/bash
# Build a config's ePSF grids into $PSFDIR, on the login node, detached.
#
# This cannot be a SLURM job. PSFFactory generates MJD-tagged grids, and stpsf
# resolves each exposure's date to a wavefront OPD by querying MAST
# (load_wss_opd_by_date -> mast_wss_opds_around_date_query). There is no offline
# path for that query, and compute nodes have neither DNS nor a route: the run
# dies in NameResolutionError seconds after starting. The datamover nodes do
# have internet but no /apps module tree, so mophongo cannot run there either.
# The login node is the only machine with both.
#
# Detached with setsid+nohup, because this runs for hours and the first attempt
# was killed partway through when the driving ssh connection dropped and the
# remote bash took the SIGHUP with it. The caller gets a log path and polls it.
#
# `nice` because this is real CPU on a shared login node. Building the grids on
# a laptop and shipping them with `push --psf` is quicker when they exist there.
#
# $CFGS is a space-separated list of config names.
set -euo pipefail
: "${BASE:?BASE not set}" "${RUN:?RUN not set}" "${CFGDIR:?CFGDIR not set}" "${CFGS:?CFGS not set}"
# Lmod here is hierarchical: python/3.12.3 is only visible once its gcccore
# parent is loaded. Unquoted on purpose - it is a list, not one name.
PYMODULES=${PYMODULES:-"gcccore/13.3.0 python/3.12.3"}
LOG=${LOG:-$RUN/logs/psfbuild-$(date +%Y%m%d-%H%M%S).log}

module purge
module load $PYMODULES
unset PYTHONPATH   # keep the EasyBuild shim off sys.path; see setup_env.sh

export MPLCONFIGDIR=$BASE/.mplconfig
export MPLBACKEND=Agg
export STPSF_PATH=${STPSF:-$BASE/stpsf-data}
mkdir -p "$MPLCONFIGDIR" "$PSFDIR" "$RUN/logs"
cd "$RUN"

configs=()
for cfg in $CFGS; do
    configs+=("$CFGDIR/${cfg}_ozstar.json")
done

# Bands of a field share the hi-res grids, so the driver walks them one at a
# time: concurrent builds would write the same filenames into one psf_dir.
setsid nohup nice -n 10 "$CFGDIR/venv/bin/python" "$BIN/build_psfs.py" \
    --date-mode "${DATE_MODE:-all}" "${configs[@]}" > "$LOG" 2>&1 < /dev/null &

echo "PSF build detached, pid $!"
echo "LOG=$LOG"
