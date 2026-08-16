#!/usr/bin/env bash
# A MINERVA mophongo release on OzStar: every field, every MIRI band, full
# field (no trial patch).
#
# COSMOS   f770w f1000w f1280w f1500w f1800w f2100w
# EGS      f560w f770w f1000w f1280w f1500w f1800w f2100w
# UDS      f770w f1280w f1500w f1800w
#
# THE VERSION IS THE RUN DIRECTORY, NOT A NAME SUFFIX. Outputs land in
# $OZSTAR_RUN/<field>/<field>_<band>/ -- run2/uds/uds_f770w, not
# run2/uds/uds_f770w_v2. Bump OZSTAR_RUN for the next attempt and the previous
# one is untouched, while data/, PSF/ and bin/ above it are reused. Putting a
# version in the run *name* too would repeat it in every path and every output
# filename, and make two attempts at the same release impossible to compare
# without renaming files.
#
#   ./release.sh                       # into $OZSTAR_RUN (default run2)
#   OZSTAR_RUN=run3 ./release.sh       # the next attempt
#   ./release.sh --skip-stage          # inputs already on /fred
#   ./release.sh --dry-run             # print the plan
#
# Any other argument is passed through to campaign.py -- `--skip psf` when the
# grids are already on /fred, `--note` to say what this attempt changes. Memory
# is 96 GB for every field (submit.DEFAULT_MEM_GB); setting MEM overrides it.
#
# Measured on the v1.0b UDS bands: 53.3, 55.6, 57.4 GB peak RSS, so 72 runs at
# about 80% with 15 GB spare. A run that exceeds its request is killed with no
# Python traceback, which reads as a silent failure - if a band dies without
# one, that is the first thing to check.
#
# Watch it with:  python submit.py status
set -euo pipefail

CORES=${CORES:-32}
MEM=${MEM:-}          # empty = campaign.py's per-field defaults
WALLTIME=${WALLTIME:-24:00:00}
# ozify.py needs the vos client, which lives in this venv rather than on PATH.
PYTHON=${PYTHON:-$HOME/.venvs/canfar/bin/python}

cd "$(dirname "${BASH_SOURCE[0]}")"

: "${OZSTAR_USER:?set OZSTAR_USER to your cluster username}"
[[ -s $HOME/.ssl/cadcproxy.pem ]] || {
    echo "no CADC certificate; run ~/bin/remote/canfar-cert.sh" >&2
    exit 1
}

args=(--r-trial 0 --cores "$CORES" --time "$WALLTIME")
[[ -n $MEM ]] && args+=(--mem "$MEM")
for opt in "$@"; do
    case "$opt" in
        --skip-stage) args+=(--skip stage) ;;
        *)            args+=("$opt") ;;
    esac
done

echo "release into ${OZSTAR_RUN:-run2}: all fields, all MIRI bands, full field"
exec "$PYTHON" campaign.py "${args[@]}"
