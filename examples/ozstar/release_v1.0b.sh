#!/usr/bin/env bash
# MINERVA mophongo release v1.0b on OzStar: every field, every MIRI band, full
# field (no trial patch).
#
# COSMOS   f770w f1000w f1280w f1500w f1800w f2100w
# EGS      f560w f770w f1000w f1280w f1500w f1800w f2100w
# UDS      f770w f1280w f1500w f1800w
#
# 17 fits, each 16 cores and 64 GB, writing to $OZSTAR_RUN/out/<field>_<band>_v1.0b.
# The whole thing is one SLURM dependency graph: each field's fits wait on that
# field's staging, and a field whose PSF grids do not exist yet sends one band
# ahead of the rest to build them without racing on psf_dir.
#
#   ./release_v1.0b.sh              # stage and run
#   ./release_v1.0b.sh --skip-stage # inputs already on /fred
#   ./release_v1.0b.sh --dry-run    # print the plan
#
# Any other argument is passed through to campaign.py, and MEM overrides the
# request, so a field that needs more memory than the rest is a second call:
#
#   ./release_v1.0b.sh --skip-stage --fields uds cosmos
#   MEM=96 ./release_v1.0b.sh --skip-stage --fields egs
#
# EGS is the one to watch: its detection grid is 1221 Mpx against UDS's 876, and
# UDS full field measures a 46.5 GB footprint, so EGS is the band group that can
# exceed a 64 GB request. A run that does is killed with no Python traceback.
#
# Watch it with:  python submit.py status
set -euo pipefail

SUFFIX=${SUFFIX:-_v1.0b}
CORES=${CORES:-16}
MEM=${MEM:-64}
WALLTIME=${WALLTIME:-24:00:00}
# ozify.py needs the vos client, which lives in this venv rather than on PATH.
PYTHON=${PYTHON:-$HOME/.venvs/canfar/bin/python}

cd "$(dirname "${BASH_SOURCE[0]}")"

: "${OZSTAR_USER:?set OZSTAR_USER to your cluster username}"
[[ -s $HOME/.ssl/cadcproxy.pem ]] || {
    echo "no CADC certificate; run ../../scratch/canfar/canfar-cert.sh" >&2
    exit 1
}

args=(--r-trial 0 --suffix "$SUFFIX" --cores "$CORES" --mem "$MEM" --time "$WALLTIME")
for opt in "$@"; do
    case "$opt" in
        --skip-stage) args+=(--skip stage) ;;
        --dry-run)    args+=(--dry-run) ;;
        *)            args+=("$opt") ;;
    esac
done

echo "release $SUFFIX: all fields, all MIRI bands, full field"
exec "$PYTHON" campaign.py "${args[@]}"
