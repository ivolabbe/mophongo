#!/bin/bash
# Copy cached PSF and kernel maps between run directories.
#
# The maps are keyed on psf_size/blur and kernel method/reg, not on the trial
# patch, so a full-field run can reuse what a patch run of the same band already
# built - about half an hour per band. They live in out_dir and are named after
# the run, so seeding a new run means copying with the prefix rewritten.
#
# PAIRS is a comma-separated list of src:dst run names.
set -euo pipefail
: "${RUN:?RUN not set}" "${PAIRS:?PAIRS not set}"

copied=0
IFS=',' read -ra items <<< "$PAIRS"
for pair in "${items[@]}"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    [[ -d "$RUN/out/$src" ]] || { echo "skip $src: no source dir"; continue; }
    mkdir -p "$RUN/out/$dst"
    for suffix in psf_hi.fits psf_hi.geojson psf_lo.fits psf_lo.geojson \
                  kernel.fits kernel.geojson; do
        s="$RUN/out/$src/${src}_${suffix}"
        d="$RUN/out/$dst/${dst}_${suffix}"
        [[ -s "$s" ]] || continue
        [[ -e "$d" ]] && continue          # already seeded
        # symlink, not copy: these are read-only inputs and the kernel maps run
        # to hundreds of MB each, which is what exhausted the home quota once
        ln -s "$s" "$d" && copied=$((copied+1))
    done
    echo "seeded $dst from $src"
done
echo "linked $copied files"
echo SEED_DONE
