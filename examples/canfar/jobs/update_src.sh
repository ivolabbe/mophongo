#!/bin/bash
# Pull this run's mophongo checkout, leaving the venv alone.
#
# mophongo is installed editable, so the venv imports straight from
# $CFGDIR/mophongo and a code change needs no reinstall. setup_env.sh deletes
# and rebuilds the venv, which would break any run using it, so use this to
# ship a fix while other jobs are in flight. Already-running jobs keep the code
# they imported; only later ones pick this up.
#
# Scope note: this moves the version a run is pinned to. Two bands of the same
# run fitted either side of it used different code, which is exactly what
# SRC_VERSION records so the difference is visible afterwards.
set -euo pipefail
: "${RUN:?RUN not set}"
RUNNUM=${RUNNUM:-1}
CFGDIR=$RUN/run$RUNNUM/config
BRANCH=${BRANCH:-main}

[ -d "$CFGDIR/mophongo/.git" ] || {
    echo "no checkout at $CFGDIR/mophongo; run setup first" >&2; exit 1; }
git -C "$CFGDIR/mophongo" fetch -q origin "$BRANCH"
git -C "$CFGDIR/mophongo" checkout -q "$BRANCH"
git -C "$CFGDIR/mophongo" reset -q --hard "origin/$BRANCH"
# The commit IS the version; nothing to promote from a separate upload.
git -C "$CFGDIR/mophongo" rev-parse --short HEAD > "$CFGDIR/SRC_VERSION"
echo "source updated: $(find "$CFGDIR/mophongo" -name '*.py' | wc -l) files"
echo "mophongo: $(cat "$CFGDIR/SRC_VERSION")"
echo SYNC_DONE
