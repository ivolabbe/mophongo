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
    git clone -q --branch "$BRANCH" "$REPO" "$CFGDIR/mophongo"
fi
# The commit IS the version: nothing to stamp, nothing to keep in step with a
# separate upload, and a run that used stale source cannot look current.
git -C "$CFGDIR/mophongo" rev-parse --short HEAD > "$CFGDIR/SRC_VERSION"
echo "mophongo: $(cat "$CFGDIR/SRC_VERSION")"

echo "=== venv"
rm -rf "$CFGDIR/venv"
python -m venv "$CFGDIR/venv"
"$CFGDIR/venv/bin/pip" -q install -U pip
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
