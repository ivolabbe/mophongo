#!/bin/bash
# Build one run's mophongo checkout and venv, inside that run's config dir.
#
# A run pins one mophongo version. Keeping the source and the environment in
# run<N>/config rather than at the tree root is what makes that reproducible:
# run1 and run2 can be built from different commits and neither disturbs the
# other. Inputs and ePSF grids stay shared at the root, because they do not
# change between runs.
#
# The source is cloned here rather than shipped from the laptop. CANFAR
# containers have outbound internet - `pip install` reaches PyPI from the same
# shell - so the repo is public and one clone is faster and more honest about
# what it built than a tarball whose upload and unpack are separate steps.
#
# The stock image's astropy/numpy are too old, so this is a clean venv rather
# than --system-site-packages.
set -euo pipefail
: "${RUN:?RUN not set}"
RUNNUM=${RUNNUM:-1}
CFGDIR=$RUN/run$RUNNUM/config
REPO=${REPO:-https://github.com/ivolabbe/mophongo.git}
BRANCH=${BRANCH:-main}
export MPLCONFIGDIR=$RUN/.mplconfig
mkdir -p "$MPLCONFIGDIR" "$CFGDIR" "$RUN/PSF"

echo "=== source: $REPO@$BRANCH"
if [ -d "$CFGDIR/mophongo/.git" ]; then
    git -C "$CFGDIR/mophongo" fetch -q origin "$BRANCH"
    git -C "$CFGDIR/mophongo" checkout -q "$BRANCH"
    git -C "$CFGDIR/mophongo" reset -q --hard "origin/$BRANCH"
else
    # A directory that exists but holds no .git is an unpacked tarball from the
    # older ship-the-source convention. `git clone` refuses a non-empty target,
    # so neither branch repairs it and setup fails on every retry until someone
    # clears it by hand -- which is what run1 did. Move it aside and clone.
    if [ -e "$CFGDIR/mophongo" ]; then
        stale="$CFGDIR/mophongo.stale-$(date +%Y%m%dT%H%M%S)"
        echo "no .git in $CFGDIR/mophongo; moving it to $stale"
        mv "$CFGDIR/mophongo" "$stale"
    fi
    git clone -q --branch "$BRANCH" "$REPO" "$CFGDIR/mophongo"
fi
# Stamped because the laptop cannot run git on arc -- there is no ssh, only file
# transfer -- so `submit._arc_src_version()` reads this file to refuse a launch
# against stale source. The OzStar toolkit deliberately has no equivalent: it
# asks git over ssh, where a stamp would only be a second copy able to drift.
git -C "$CFGDIR/mophongo" rev-parse --short HEAD > "$CFGDIR/SRC_VERSION"
echo "mophongo: $(cat "$CFGDIR/SRC_VERSION")"

echo "=== venv"
# Rebuilt only when asked. The install is editable, so a code change needs a
# git pull and nothing else, and `pip install -e` still runs below to pick up a
# dependency change -- rebuilding from scratch every time cost seven minutes
# per setup and bought nothing. REBUILD=1 forces it.
if [ "${REBUILD:-0}" = 1 ]; then
    rm -rf "$CFGDIR/venv"
fi
if [ ! -x "$CFGDIR/venv/bin/python" ]; then
    python -m venv "$CFGDIR/venv"
    "$CFGDIR/venv/bin/pip" -q install -U pip
fi
"$CFGDIR/venv/bin/pip" install -e "$CFGDIR/mophongo" 2>&1 | tail -3

echo "=== versions"
"$CFGDIR/venv/bin/python" -c "
import astropy, numpy, photutils, psutil, drizzlepac
print('astropy', astropy.__version__, 'numpy', numpy.__version__,
      'photutils', photutils.__version__, 'psutil', psutil.__version__,
      'drizzlepac', drizzlepac.__version__)
from mophongo.pipeline import RunConfig
print('mophongo imports ok')
"
echo "psf grids: $(ls "$RUN/PSF" | wc -l)"
echo ENV_DONE
