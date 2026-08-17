#!/bin/bash
# Build the run tree on /fred: clone mophongo from GitHub and make the venvs.
#
# Runs on the login node, which is the only place with both internet and the
# module system. Idempotent: re-running updates the clone in place and leaves
# the venvs alone unless REBUILD=1.
#
# Two venvs, because the two kinds of node do not share a software stack:
#
#   venv      mophongo, built with the module python. Used by the fits, which
#             run on ordinary compute nodes. Its bin/python is a symlink into
#             the module tree, so the same modules must be loaded to use it -
#             which is why every script here loads the same PYMODULES, and why
#             changing them means rebuilding.
#   venv-vos  the CADC transfer tools, built with /usr/bin/python3. Used by the
#             staging job, which runs on the datamover partition - and those
#             nodes have no /apps module tree at all, so the module python is
#             simply not there. The OS python is on every node.
set -euo pipefail
: "${BASE:?BASE not set}" "${RUN:?RUN not set}" "${CFGDIR:?CFGDIR not set}" "${VENV:?VENV not set}" "${SRC:?SRC not set}"
# Lmod here is hierarchical: python/3.12.3 is only visible once its gcccore
# parent is loaded. Unquoted on purpose - it is a list, not one name.
PYMODULES=${PYMODULES:-"gcccore/13.3.0 python/3.12.3"}
REPO_URL=${REPO_URL:-https://github.com/ivolabbe/mophongo.git}
BRANCH=${BRANCH:-main}
STPSF=${STPSF:-}

module purge
module load $PYMODULES
# The python module puts an EasyBuild shim ahead of the venv on sys.path, which
# on some nodes resolves shared packages (cryptography, and so anything using
# TLS) to a build for another python and fails on a missing libssl. The venv is
# self-contained; drop the shim.
unset PYTHONPATH

mkdir -p "$BASE"/{PSF,data,bin} "$RUN"/logs "$CFGDIR"

echo "=== source"
if [[ -d $SRC/.git ]]; then
    git -C "$SRC" fetch --quiet origin "$BRANCH"
    git -C "$SRC" checkout --quiet "$BRANCH"
    git -C "$SRC" reset --hard "origin/$BRANCH"
else
    # A directory that exists but holds no .git is an unpacked tarball from the
    # older ship-the-source convention. `git clone` refuses a non-empty target,
    # so neither branch above repairs it and setup fails on every retry until
    # someone clears it by hand -- which is exactly what happened on CANFAR.
    # Move it aside and clone: reversible, and the run gets a real checkout.
    if [[ -e $SRC ]]; then
        stale="$SRC.stale-$(date +%Y%m%dT%H%M%S)"
        echo "no .git in $SRC; moving it to $stale"
        mv "$SRC" "$stale"
    fi
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$SRC"
fi
# The commit is the version. Stamped beside the configs so a finished run's
# outputs, its configs and its source are one directory and cannot drift apart.
git -C "$SRC" rev-parse --short HEAD > "$CFGDIR/SRC_VERSION"
echo "mophongo $(cat "$CFGDIR/SRC_VERSION") on $BRANCH"

echo "=== venv (mophongo, module python, compute nodes)"
if [[ ${REBUILD:-0} == 1 ]]; then
    # Only this run's venv. `$VOS` is the CADC tool venv under `bin/`, shared by
    # every run and by the datamover jobs, so rebuilding one run's environment
    # must not take it out from under the others.
    rm -rf "$VENV"
fi
if [[ ! -x $VENV/bin/python ]]; then
    python -m venv "$VENV"
    "$VENV/bin/pip" -q install -U pip
fi
# Editable, so `sync_src.sh` (a git pull) is enough to ship a code change.
"$VENV/bin/pip" install -q -e "$SRC"

echo "=== venv-vos (CADC tools, OS python, datamover nodes)"
if [[ ! -x $VOS/bin/python ]]; then
    /usr/bin/python3 -m venv "$VOS"
    "$VOS/bin/pip" -q install -U pip
fi
"$VOS/bin/pip" install -q vos cadcutils
"$VOS/bin/vcp" --help > /dev/null && \
    echo "$("$VOS/bin/python" -c 'import vos; print(vos.version.version)') ok"

echo "=== versions"
"$VENV/bin/python" -c "
import astropy, numpy, photutils, psutil, drizzlepac
print('astropy', astropy.__version__, 'numpy', numpy.__version__,
      'photutils', photutils.__version__, 'psutil', psutil.__version__,
      'drizzlepac', drizzlepac.__version__)
from mophongo.pipeline import RunConfig
print('mophongo imports ok')
"

# STPSF reference data (~250 MB). Compute nodes have no internet, so a run that
# has to build a PSF grid cannot fetch this itself: it must be here first.
if [[ -n $STPSF ]]; then
    if [[ -d $STPSF && -d $STPSF/NIRCam ]]; then
        echo "stpsf data: $STPSF ($(du -sh "$STPSF" | cut -f1))"
    else
        echo "WARNING: no STPSF data at $STPSF."
        echo "  Runs that build a PSF grid will fail on the compute nodes,"
        echo "  which have no internet. Fetch it once on the login node:"
        echo "    curl -LO https://stsci.box.com/shared/static/stpsf-data-LATEST.tar.gz"
        echo "    mkdir -p $STPSF && tar xzf stpsf-data-LATEST.tar.gz -C $(dirname "$STPSF")"
    fi
fi

echo ENV_DONE
