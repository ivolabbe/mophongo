# Verification v8 — the 2026-08-12 P1 fixes

Run 2026-08-12, commit `a06cc56`. Two passes:

- all four bands on the P1-01..P1-05 fixes
  (`run_verification_v2.py --version v8 --scheme psf_wings --psf-dir data/PSF8
  --psf-size 8.0 --r-trial 1.5`),
- **F770W re-run afterwards** on the weight-calibration fix, the stamp-footprint
  convolution and the aperture centring (`... fits f770w --version v8 ...`).

Only F770W carries the final code. The other three bands' numbers below are from
the first pass and are superseded; `verification_summary.json` holds the F770W
re-run, and `verification_summary_allbands_preshift.json` the first pass.

## What changed against v7

v7 reused v6's real-data fits and only reworked the mock leg. v8 re-runs the
fits, because P1-03/P1-04 change the inverse-variance calibration and P1-05
changes which PSF region each source reads. Every fix is listed in `STATUS.md`;
the three that move these numbers are the exact astrometric blocks (P1-01), the
final flux-only solve (P1-02) and the weight calibration.

### The weight calibration bug, found during this version

The first v8 F770W pass was invalid and is worth recording. `get_bg_and_ivar`
took its robust baseline over *all* coarse blocks, including the zero-filled
pixels outside the mosaic footprint. On a field that is ~50% uncovered — the
normal case for MIRI — that dragged `sigma0` from 449 down to 12, which
collapsed the detection threshold, which flagged the whole field as source,
which left the background fit interpolating the data. `sigma_true` went to
1.4e-4 and the inverse variance was inflated by 5e7.

The visible symptoms were a good lesson in not trusting a pipeline's own
diagnostics: every source read SNR ~1e7, so every scene reported "bright"
anchors and the log line `0 scene(s) without bright members` was true and
meaningless. The scene partition fragmented from 15 scenes to 207, and each
fitted its own order-0 shift from noise — those shifts were spatially
uncorrelated (|dshift| 0.207 px between neighbours at 0-20" against 0.204 px
for random pairs).

The baseline is now taken over valid blocks only. Two different patches of the
same mosaic now return `sigma_true` 3295 and 3291, agreeing to 0.1%, where
before they differed by a factor 2.3e7.

## Results

F770W, final code:

| quantity | v7 | v8 |
|---|---|---|
| mock, recovered/true total | 0.9533 | 0.9607 |
| IDL est1 (no EE), py - IDL mag | +0.020 | +0.023 |
| IDL psfcor | 0.985 | 0.975 |
| `sigma_true` | — | 3295 |
| scenes / median size | — | 15 / 250 |
| sources above SNR 15 | — | 120 |

Bands not yet re-run on the final code (first pass, superseded):

| band | mock med | est1 (no EE) | psfcor |
|---|---|---|---|
| F1280W | 0.9645 | +0.04 | 0.985 |
| F1500W | 0.9673 | +0.05 | 0.972 |
| F1800W | 0.9688 | +0.06 | 0.958 |

## Reading

The mock leg moved +0.7% toward unity in F770W, so the `psf_wings`
extended-source deficit is 3.9% rather than v7's 4.7%. The band trend cannot be
read from this version until the other three bands are re-run.

Five fixes landed together in the first pass and three more in the F770W
re-run, so none of these numbers attributes an effect to a single change. A
per-fix run is needed for that.

## Open

- Re-run F1280W/F1500W/F1800W on the final code so the band trend is
  self-consistent.
- The `totcor`/`stampcor` offset against IDL is analysed in
  `scratch/wren/flux_estimator_comparison.tex`; most of it is aperture
  placement, the residual 1-2% is the low-SNR template prior
  (`tmpl_snrlo` 15 in subphot against `psf_wings_snrlo` 5 here).
