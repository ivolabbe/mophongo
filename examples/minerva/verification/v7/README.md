# Verification v7 — production-sized segments, resolved source population

Run 2026-08-12, commit `a4ff14b`. Driver:
`examples/minerva/run_verification_v2.py idl --version v7 --scheme psf_wings
--psf-dir data/PSF8 --psf-size 8.0 --r-trial 1.5`.

v7 reuses v6's real-data fits unchanged — `runs/` holds copies of v6's four
configs and fit tables, and the IDL leg is a rerun against them, reproducing
v6's numbers exactly. Everything that differs is in the mock leg, which
existed to answer one question: how much of the `psf_wings` extended-source
deficit reported through v6 was real, and how much was the mock flattering
itself.

## What changed against v6

- **Segments are production-sized.** The mock segmap was dilated *twice*:
  once by `Catalog.detect`'s own `dilate_segmap` disk(2) — the step that
  stands in for however the released MINERVA segmap was built — and again by
  `mock_dilate_segmap`, which had no counterpart anywhere in a real run
  (`Pipeline._load_data` reads the released segmap off disk and applies no
  dilation at all; `FitConfig.template_dilate_segmap` is 0). A 5x5 source
  grew 25 -> 69 -> 129 pixels, so templates were taking roughly twice the
  production footprint of each source directly from the data instead of
  relying on the PSF wings. `mock_dilate_segmap` now defaults to 0.
- **Sources reach resolved sizes.** Sigma is log-uniform over 1-10 pixels on
  the 40 mas grid (0.04-0.40", FWHM 0.09-0.94") rather than 1-5. Through v6
  the median injected source had FWHM 0.21", *smaller than the F770W beam*,
  so the population barely probed the effect being measured. Painting stamps
  stay at the builder default (F444W 4", F770W 8"): a 0.4" sigma source sits
  inside a 4" box to better than 1e-4.
- **Diagnostics** gained a running median of the unblended sources, an
  `SNR>20 median +/- standard error` band, a green point-source pull
  histogram, and a fifth panel of fractional residual against injected size.

## Results

Mock leg (recovered/true total flux, 696 fitted sources per band):

| band | med | p16-p84 | med hi self-fit | pull med | v6 med | change |
|---|---|---|---|---|---|---|
| F770W | 0.9533 | 0.833-1.000 | 0.9532 | -6.01 | 0.9669 | -1.4% |
| F1280W | 0.9573 | 0.851-1.003 | 0.9554 | -5.03 | 0.9694 | -1.2% |
| F1500W | 0.9606 | 0.871-1.010 | 0.9581 | -4.48 | 0.9715 | -1.1% |
| F1800W | 0.9637 | 0.884-1.013 | 0.9599 | -3.86 | 0.9740 | -1.0% |

IDL leg. The fits are v6's, but the comparison itself was reworked after
the mock leg ran, so these supersede v6's numbers rather than reproducing
them (py - IDL magnitudes, SNR>20 throughout — the old mag<24 cut mixed in
sources that are noise-dominated in the fitted band, which is what produced
the spurious F1800W "offset"):

| band | n | (a) raw aperture | (b) stampcor | (c) est1, no EE | (e) psfcor |
|---|---|---|---|---|---|
| F770W | 4522 | +0.007 | 0.986 | +0.020 | 0.985 |
| F1280W | 4998 | +0.013 | 0.992 | +0.021 | 0.986 |
| F1500W | 3979 | +0.009 | 0.978 | +0.034 | 0.973 |
| F1800W | 4427 | +0.008 | 0.964 | +0.052 | 0.959 |

Estimator 1 is now compared with **no encircled-energy term on either
side**: IDL's `flux_F` applies `totcor = 1/ap_lo` and carries no EE, so the
python side uses `ap_flux * stampcor` (written as `tot_stamp_<i>` in the
v7 tables, renamed afterwards) rather than `ap_flux_corr`, which
divides by `ee_psf_lo` as well. That single change flips the sign of the
comparison — python was reading 0.012 mag *brighter* than IDL and now reads
0.020 mag *fainter*. The earlier offset was the convention, not the code.

## Reading

The deficit is **4-5%, not the 2-3% carried since v2**, and the band trend
(bluest worst) survives. Both corrections push the same way and neither was
a modelling choice: one removed a dilation that no production run performs,
the other stopped measuring an extended-source effect on a population that
was not extended.

The F770W diagnostic separates the two error budgets cleanly. Point sources
fit a pull of mu=+0.30, sigma=0.91 — unbiased, with a correctly calibrated
error model. The full SNR>20 sample fits mu=-3.35, sigma=2.81. The entire
bias therefore sits in the extended sources, and the fifth panel shows it
turning on immediately: flat at zero for sigma=0, falling to about -0.07 by
sigma 4-6 pixels, then flattening. The SNR>20 median is 0.9553 +/- 0.0022
(n=392), so this is a ~20-sigma effect, not a scatter artefact.

One caveat before these numbers decide the aperture-floor question in
`TODO.md`: every injected source is a **pure circular Gaussian**
(`mock_mosaic.py:1260` convolves the PSF stamp with `gaussian_filter`).
Outside 3 half-light radii a Gaussian retains 0.2% of its flux against 3.9%
for an exponential disc and 21% for a de Vaucouleurs profile. Since the
deficit arises precisely from PSF-shaped template wings failing to follow a
real outer profile, 4-5% remains a floor. A Sersic injector is the next
change worth making.

A second caveat applies to reading *across* bands: each MINERVA config picks
its own `trial_center` (the deepest fully covered patch of that band's MIRI
weight map), and at `r_trial=1.5` two of the six UDS band pairs are
completely disjoint. The per-band values are sound; the band-to-band trend
partly conflates filter with sky patch.

## Layout

- `runs/` — copies of v6's configs and fit tables (3.1 MB; the fits
  themselves are untouched by any mock-side change).
- `uds_monu/` — IDL leg: `compare_idl_vs_python_<band>.png`,
  `idl_summary.json`, `idl_compare.log`.
- `uds_sims/` — mock leg: per-band `uds_<band>/`,
  `mock_flux_ratio_<band>.png`, `mock_wiener_<band>.png`,
  `summary_all.json`, `mock.log`.
- `verification_summary.json`, `driver.log`.
