#!/bin/bash
#SBATCH --job-name=mophongo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#
# Fit one config. $CFG names the rewritten config, e.g. uds_f770w.
#
# $STEP is the pipeline step, default `all` (the whole fit). A campaign runs
# `repair` first, once per field: that runs the saturation repair into a cache
# the field's bands then hit, instead of every band re-running it and several
# writing the same cache file at once. The ePSF grids that step needs are built
# separately, on the login node, because compute nodes cannot reach MAST
# (jobs/build_psfs.sh).
#
# 16 cores and 64 GB is the standard request. The fitting path itself is
# largely serial, so the cores buy threaded BLAS and FFTs rather than a linear
# speed-up; the memory is what actually matters, and a run that exceeds it is
# killed by SLURM with no Python traceback. No --partition is set, so the
# scheduler picks: both skylake (36 cores / 191 GB) and milan (64 / 256) hold
# this comfortably, and constraining the choice only lengthens the queue.
set -euo pipefail
: "${BASE:?BASE not set}" "${RUN:?RUN not set}" "${CFGDIR:?CFGDIR not set}" "${VENV:?VENV not set}" "${SRC:?SRC not set}" "${CFG:?CFG not set}"
STEP=${STEP:-all}
# Lmod here is hierarchical: python/3.12.3 is only visible once its gcccore
# parent is loaded. Unquoted on purpose - it is a list, not one name.
PYMODULES=${PYMODULES:-"gcccore/13.3.0 python/3.12.3"}

module purge
module load $PYMODULES
# The python module puts an EasyBuild shim ahead of the venv on sys.path, which
# on some nodes resolves shared packages to a build for another python. The venv
# is self-contained; drop the shim.
unset PYTHONPATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MPLCONFIGDIR=$BASE/.mplconfig
export MPLBACKEND=Agg
# Compute nodes have no internet: stpsf cannot fetch its reference data here,
# so it has to be on /fred already (setup_env.sh checks).
export STPSF_PATH=${STPSF:-$BASE/stpsf-data}

# out_dir is <run>/<field>/<name>, created by the pipeline; cd to the run
# so any stray relative path lands there rather than in $HOME
mkdir -p "$MPLCONFIGDIR" "$RUN"
cd "$RUN"

echo "=== $CFG [$STEP] on $(hostname): ${SLURM_CPUS_PER_TASK:-?} cores, ${SLURM_MEM_PER_NODE:-?} MB"
echo "=== mophongo $(git -C "$SRC" rev-parse --short HEAD)"
time "$VENV/bin/python" -m mophongo.pipeline "$CFGDIR/${CFG}_ozstar.json" "$STEP"

echo "=== outputs:"; ls -lh "$RUN/${CFG%%_*}/$CFG" | head -25
echo "=== peak memory:"; sacct -j "${SLURM_JOB_ID}" --format=JobID,MaxRSS,Elapsed --noheader 2>/dev/null | head -3
echo RUN_DONE
