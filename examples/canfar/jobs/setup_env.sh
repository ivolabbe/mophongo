#!/bin/bash
# Build the mophongo venv on /arc. One-time; the venv persists between jobs.
# The stock image's astropy/numpy are too old, so this is a clean venv rather
# than --system-site-packages.
set -euo pipefail
: "${RUN:?RUN not set}"
export MPLCONFIGDIR=$RUN/.mplconfig
mkdir -p $MPLCONFIGDIR $RUN/setup/mophongo $RUN/PSF

echo "=== unpack"
tar -xzf $RUN/setup/mophongo_src.tgz -C $RUN/setup/mophongo
tar -xf  $RUN/setup/psf.tar          -C $RUN/PSF
echo "psf grids: $(ls $RUN/PSF | wc -l)"
# See update_src.sh: SRC_VERSION records what is unpacked, not what was uploaded.
cp -f $RUN/setup/SRC_VERSION.pending $RUN/setup/SRC_VERSION
echo "mophongo: $(cat $RUN/setup/SRC_VERSION)"

echo "=== venv"
rm -rf $RUN/setup/venv
python -m venv $RUN/setup/venv
$RUN/setup/venv/bin/pip -q install -U pip
$RUN/setup/venv/bin/pip install -e $RUN/setup/mophongo 2>&1 | tail -3

echo "=== versions"
$RUN/setup/venv/bin/python -c "
import astropy, numpy, photutils, psutil, drizzlepac
print('astropy', astropy.__version__, 'numpy', numpy.__version__,
      'photutils', photutils.__version__, 'psutil', psutil.__version__,
      'drizzlepac', drizzlepac.__version__)
from mophongo.pipeline import RunConfig
print('mophongo imports ok')
"
echo ENV_DONE
