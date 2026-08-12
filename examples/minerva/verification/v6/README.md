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
- **IDL-parity columns**: `stampcor` (was `tot_stamp`; =1/ap_lo, IDL's released totcor
  convention), `psfcor` (=ap_hi/ap_lo), `totcor` (=1/(ap_lo*ee_psf_lo),
  always EE-inclusive), est1 = `ap_flux_corr` (genuine totals).
- r < 1.5' trial patch (~4-5k matched sources per band), ~11 min per
  band with globally seeded PSF/kernel caches; single-print logging.

## Layout

- `runs/uds_<band>/` — real-data fits, configs in `runs/*.json`.
- `uds_monu/` — IDL leg: `compare_idl_vs_python_<band>.png`,
  `idl_summary.json`, `idl_compare.log`.
- `uds_sims/` — mock leg: per-band `uds_<band>/`, `mock_flux_ratio_<band>
  .png`, `mock_wiener_<band>.png`, `summary_all.json`, `mock.log`.
- `verification_summary.json`, `driver.log`.

The mock leg was run separately after the fits and IDL leg (driver
invocation `... idl --version v6 --psf-dir data/PSF8`), on the same
commit, and used the same `data/PSF8` grids as the real-data leg. Its
source population is the v2-v4 one: sigma log-uniform over 1-5 pixels on
the 40 mas grid (0.04-0.20" sigma), 10% point sources.

## Results (vs IDL monu)

| band | raw aperture | stampcor / IDL totcor | est1 SNR>25 | n |
|---|---|---|---|---|
| F770W | +0.01 | 0.987 | -0.01±0.06 | 4522 |
| F1280W | +0.02 | 0.991 | -0.03±0.32 | 4998 |
| F1500W | +0.03 | 0.975 | -0.03±0.11 | 3979 |
| F1800W | +0.34* | 0.955 | -0.02±0.04 | 4427 |

(*the usual magnitude-cut noise artefact; read the SNR>25 column.)

F770W medians: `stampcor` (then `tot_stamp`) 1.432 vs IDL totcor 1.450 (1.3%), `psfcor`
1.240 vs IDL psfcor 1.261 (1.7%), `ee_psf_lo` at 8" = 0.973.

Mock leg (recovered/true total flux, 696 fitted sources per band):

| band | med | p16-p84 | med hi self-fit | pull med | resid/noise | v4 med (4" F444W) |
|---|---|---|---|---|---|---|
| F770W | 0.9669 | 0.898-1.009 | 0.9658 | -3.22 | 0.799 | 0.9755 |
| F1280W | 0.9694 | 0.913-1.016 | 0.9681 | -2.76 | 0.799 | 0.9776 |
| F1500W | 0.9715 | 0.923-1.019 | 0.9698 | -2.65 | 0.799 | 0.9788 |
| F1800W | 0.9740 | 0.931-1.021 | 0.9720 | -2.49 | 0.799 | 0.9810 |

Widening the detection PSF from 4" to 8" costs a further 0.7-0.9% of
recovered flux in every band. The `psf_wings` composite is built on the
F444W stamp, so a larger stamp puts more wing flux inside the template
normalization; the fitted amplitude drops correspondingly while the
injected truth does not move. The band trend (bluest worst) is unchanged.
The 8" detection stamp also raises the recorded F444W box EE from 0.96317
to 0.98522, which is the same effect seen from the normalization side.

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
