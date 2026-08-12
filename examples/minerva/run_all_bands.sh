#!/bin/bash
# Run every generated MINERVA config sequentially (one python at a time: a
# single r<3' band peaks at tens of GB, so no parallelism). Regenerates the
# configs first, then UDS -> COSMOS -> EGS, then the UDS-vs-IDL comparison.
# EGS has no ePSF grids yet; its first band triggers psf_autobuild and is slow.
set -u
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python"   # absolute: survives the cd below

( cd .. && "$PY" make_minerva_configs.py ) \
    > /Users/ivo/Astro/PROJECTS/MINERVA/data/stage/configs.log 2>&1
grep -q '"r_trial": 3.0' uds_f770w.json || { echo "config regen failed"; exit 1; }

for cfg in uds_f770w uds_f1280w uds_f1500w uds_f1800w \
           cosmos_f770w cosmos_f1000w cosmos_f1280w cosmos_f1500w cosmos_f1800w cosmos_f2100w \
           egs_f560w egs_f770w egs_f1000w egs_f1280w egs_f1500w egs_f1800w egs_f2100w; do
    echo "=== START $cfg $(date +%H:%M:%S)"
    "$PY" -m mophongo.pipeline "$cfg.json" > "run_$cfg.log" 2>&1
    echo "=== EXIT $cfg rc=$? $(date +%H:%M:%S)"
done

( cd ../.. && .venv/bin/python scratch/wren/make_compare_idl_python.py \
    > scratch/wren/compare_r3.log 2>&1 )
echo ALL_BANDS_DONE
