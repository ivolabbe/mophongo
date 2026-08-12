# Verification v6 — full-support stamps, production repair, IDL-parity columns

Run 2026-08-12. Driver:
`examples/minerva/run_verification_v2.py fits idl --version v6 --scheme
psf_wings --psf-dir data/PSF8 --psf-size 8.0 --r-trial 1.5`.

The first fully like-for-like IDL comparison:

- **8" PSF support** (`data/PSF8` F444W grids at 8" FOV, all epochs;
  MIRI grids natively 8.1"), matching the measured 7.8" support of the
  IDL monu run — the support-parity check passes for every band (no
  warnings).
- **Saturated-star repair ON** (30" FOV halo grids autobuilt and cached;
  saturated sources flagged, repaired and isolated into their own
  scenes — the earlier versions ran without it).
- **IDL-parity columns**: `tot_stamp` (=1/ap_lo, IDL's released totcor
  convention), `psfcor` (=ap_hi/ap_lo), `totcor` (=1/(ap_lo*ee_psf_lo),
  always EE-inclusive), est1 = `ap_flux_corr` (genuine totals).
- r < 1.5' trial patch (~4-5k matched sources per band), ~11 min per
  band with globally seeded PSF/kernel caches; single-print logging.

## Results (vs IDL monu)

| band | raw aperture | tot_stamp / IDL totcor | est1 SNR>25 | n |
|---|---|---|---|---|
| F770W | +0.01 | 0.987 | -0.01±0.06 | 4522 |
| F1280W | +0.02 | 0.991 | -0.03±0.32 | 4998 |
| F1500W | +0.03 | 0.975 | -0.03±0.11 | 3979 |
| F1800W | +0.34* | 0.955 | -0.02±0.04 | 4427 |

(*the usual magnitude-cut noise artefact; read the SNR>25 column.)

F770W medians: `tot_stamp` 1.432 vs IDL totcor 1.450 (1.3%), `psfcor`
1.240 vs IDL psfcor 1.261 (1.7%), `ee_psf_lo` at 8" = 0.973.

## Reading

With support matched, the two codes agree on the support-only
aperture-to-total to 1-2% in the blue/mid bands (F1800W 4.5%, broadest
kernel — the remaining stamp-edge term), and est1 sits at -0.01..-0.03
mag: our totals are brighter than IDL `flux_F` by almost exactly the
beyond-8" EE (~3%), which our `totcor` includes by convention and IDL's
does not. The residual per-band scatter is measurement noise. The
support chain, the EE bookkeeping, and the shape correction all now
reconcile against the reference implementation at the percent level.

History of the attribution (v2 -> v6): the original +0.05 mag est1
offset decomposed into (a) the aperture estimator never applying the
recorded `ee_psf_lo` (fixed: `totcor` convention), and (b) the 4" vs
7.8" PSF-support difference (fixed: 8" grids). The extended-source
template *shape* term (mock: 0.975 recovery) is orthogonal and remains
the open scheme question (TODO: aperture-floor support).
