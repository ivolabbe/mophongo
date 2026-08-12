# Verification v4 — `psf_wings` from main, est1 corrected by `ee_psf_lo`

Run 2026-08-11/12. Driver:
`examples/minerva/run_verification_v2.py fits idl mock --version v4`
(no `--scheme`: the config default `psf_wings` applies). Identical inputs
and settings to v2; the differences are code-side, all on main:

- `ap_flux_total_<i> = ap_flux_corr_<i> / ee_psf_lo` — the aperture
  estimator now applies the recorded per-source box encircled energy, the
  same factor `flux_<i>_total` divides by. Verified in the run:
  `ap_flux_total_1/ap_flux_corr_1` median 1.0864 = 1/0.9205.
- The IDL comparison's Estimator-1 panel reads `ap_flux_total_1`
  (fallback `ap_flux_corr_1` for older tables).
- Scene reporting is post-merge only: one INFO line per band, e.g. F770W
  "62 scenes for 17790 templates: sizes 31-1058 (median 229), 0 scene(s)
  without bright members".

## Results

IDL leg (est1 = `ap_flux_total_1` vs IDL `flux_F`, median py-IDL mag):

| band | totcor ratio | est1 (mag<24) | est1 SNR>25 | v2 SNR>25 (uncorrected) |
|---|---|---|---|---|
| F770W | 0.950 | -0.03±0.05 | **-0.03**±0.05 | +0.06 |
| F1280W | 0.964 | -0.04±0.20 | **-0.05**±0.04 | +0.05 |
| F1500W | 0.967 | -0.05±0.45 | **-0.07**±0.07 | +0.04 |
| F1800W | 0.942 | +0.21±0.68* | **-0.07**±0.09 | +0.06 |

(*magnitude-cut artefact; read the SNR>25 column.)

Mock leg: identical to v2 to four decimals (same scheme and seed — a
determinism check; the est1 correction does not touch fit totals):
med_lo 0.9755/0.9776/0.9788/0.9810, resid/noise 0.799, pull -2.7..-2.2.

## Reading

The `ee_psf_lo` division flips the est1 offset from +0.04..+0.06 to
-0.03..-0.07: python totals are now *brighter* than IDL `flux_F` by 3-7%.
The division (1/0.92 = +8.6%) overshoots the +5% gap because IDL's
`totcor` is not a full-total convention — it carries most, not all, of
the box-EE factor, and IDL's separate `psfcor` (~1.26) never multiplies
`flux_F`. `ap_flux_total` is a genuine total-flux estimate;
`flux_F` stops at IDL's model support. The residual -3..-7% therefore
measures a *convention* difference, not an error in either code; the
truth reference for the estimator chain is the mock, where the fit totals
recover injected flux at 0.9755-0.9810 (the psf_wings extended-source
shape term, band trend tracking resolution).

Follow-up recorded in TODO.md: carry `ap_flux_total` into the
verification recovery table so the aperture estimator is also checked
against injected truth directly.
