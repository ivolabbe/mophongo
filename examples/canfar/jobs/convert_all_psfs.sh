#!/bin/bash
# Convert every ePSF grid in $PSF to the FOV naming and stamp its provenance.
#
# One pass per grid family, because the provenance records the exposure list
# the grids were built from and each family has its own: the detection grids
# of a field, that field's 30" halo grids, and each band's MIRI grids. Using
# one CSV for all of them would stamp hashes that no run will ever match,
# which is worse than not stamping at all - the grids would look permanently
# stale and be rebuilt every time.
#
# The CSVs are the ones the configs name (run1/config/*_canfar.json). Content
# is what gets hashed, not the path, so a copy under data/ is as good as the
# original as long as it is the same file.
#
# Dry run by default; pass --apply to make the changes.
#
#     bash bin/convert_all_psfs.sh            # show what would happen
#     bash bin/convert_all_psfs.sh --apply
set -uo pipefail

ROOT=${ROOT:-/arc/projects/minerva/ifl/mophongo}
PSF=${PSF:-$ROOT/PSF}
DATA=${DATA:-$ROOT/data}
OLD=${OLD:-/arc/home/ilabbe/run/data}          # where the MIRI lists used to be
PY=${PY:-python3}
CONV=${CONV:-$ROOT/bin/convert_psf_names.py}
APPLY=${1:-}

MOSAICS=${MOSAICS:-/arc/projects/minerva}
# <FIELD> <csv path>, from the configs' csv_hi. A plain table rather than an
# associative array: those need bash 4, and this has to run wherever it is
# pasted, including a macOS shell.
NIRCAM="
UDS    $MOSAICS/uds/mosaics/nircam/n3.0/grizli/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv
COSMOS $MOSAICS/cosmos/mosaics/nircam/n3.0/grizli/cosmos-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv
EGS    $MOSAICS/egs/mosaics/nircam/n2.0/grizli/egs-grizli-v8.0-minerva-v2.0-40mas-f444w-clear_wcs.csv
"
# <FIELD> <BAND> <csv basename>, from the configs' csv_lo
MIRI="
UDS    F770W   uds-v3.1_f770w_wcs.csv
UDS    F1280W  uds-v3.1_f1280w_wcs.csv
UDS    F1500W  uds-v3.1_f1500w_wcs.csv
UDS    F1800W  uds-v3.1_f1800w_wcs.csv
COSMOS F770W   cosmos-v3.0_f770w_wcs.csv
COSMOS F1000W  cosmos-v3.0_f1000w_wcs.csv
COSMOS F1280W  cosmos-v3.0_f1280w_wcs.csv
COSMOS F1500W  cosmos-v3.0_f1500w_wcs.csv
COSMOS F1800W  cosmos-v3.0_f1800w_wcs.csv
COSMOS F2100W  cosmos-v3.0_f2100w_wcs.csv
EGS    F560W   egs-m2.1_f560w_wcs.csv
EGS    F770W   egs-m2.1_f770w_wcs.csv
EGS    F1000W  egs-m2.1_f1000w_wcs.csv
EGS    F1280W  egs-m2.1_f1280w_wcs.csv
EGS    F1500W  egs-m2.1_f1500w_wcs.csv
EGS    F1800W  egs-m2.1_f1800w_wcs.csv
EGS    F2100W  egs-m2.1_f2100w_wcs.csv
"

find_csv() {   # data/ first, then the old home tree
    for d in "$DATA" "$OLD"; do
        [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }
    done
    return 1
}

convert() {    # <csv> <pattern> [--fov N]
    local csv=$1 pattern=$2; shift 2
    $PY "$CONV" "$PSF" --csv "$csv" --date-mode all --pattern "$pattern" \
        "$@" $APPLY
}

echo "### PSF dir: $PSF"
echo "### mode   : ${APPLY:-dry run}"

echo "$NIRCAM" | while read -r field csv; do
    [ -n "${field:-}" ] || continue
    if [ ! -f "$csv" ]; then
        echo "!! $field: no NIRCam exposure list at $csv - skipped"
        continue
    fi
    echo; echo "== $field detection grids"
    convert "$csv" "^${field}_NRC.._F444W_MJD[0-9]+(_FOV[0-9]+)?_GRID25_OS4"
    echo "== $field 30\" halo grids"
    # the halo pattern names its FOV, so the run asks for fov=30 and the cards
    # have to carry it or they will not match
    convert "$csv" "^${field}_NRC.._F444W_MJD[0-9]+_FOV30_GRID1_OS4" --fov 30
done

echo "$MIRI" | while read -r field band name; do
    [ -n "${field:-}" ] || continue
    if ! csv=$(find_csv "$name"); then
        echo "!! $field $band: $name not found in $DATA or $OLD - skipped"
        continue
    fi
    echo; echo "== $field $band  ($csv)"
    convert "$csv" "^${field}_MIRI_${band}_MJD[0-9]+(_FOV[0-9]+)?_GRID9_OS4"
done

echo; echo "### done"
