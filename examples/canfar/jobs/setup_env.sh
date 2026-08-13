#!/bin/bash
# Build the mophongo venv on /arc. One-time; the venv persists between jobs.
# The stock image's astropy/numpy are too old, so this is a clean venv rather
# than --system-site-packages.
set -euo pipefail
: "${RUN:?RUN not set}"
export MPLCONFIGDIR=$RUN/.mplconfig
mkdir -p $MPLCONFIGDIR $RUN/mophongo $RUN/PSF

echo "=== unpack"
tar -xzf $RUN/mophongo_src.tgz -C $RUN/mophongo
tar -xf  $RUN/psf.tar          -C $RUN/PSF
echo "psf grids: $(ls $RUN/PSF | wc -l)"
# See update_src.sh: SRC_VERSION records what is unpacked, not what was uploaded.
cp -f $RUN/SRC_VERSION.pending $RUN/SRC_VERSION
echo "mophongo: $(cat $RUN/SRC_VERSION)"

echo "=== venv"
rm -rf $RUN/venv
python -m venv $RUN/venv
$RUN/venv/bin/pip -q install -U pip
$RUN/venv/bin/pip install -e $RUN/mophongo 2>&1 | tail -3

echo "=== versions"
$RUN/venv/bin/python -c "
import astropy, numpy, photutils, psutil, drizzlepac
print('astropy', astropy.__version__, 'numpy', numpy.__version__,
      'photutils', photutils.__version__, 'psutil', psutil.__version__,
      'drizzlepac', drizzlepac.__version__)
from mophongo.pipeline import RunConfig
print('mophongo imports ok')
"
echo ENV_DONE
