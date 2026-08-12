# Verification v2 — least-squares `psf_wings` scheme (post template merge)

Run 2026-08-11, commit `a2fecb1` (template branch merged: scheme-based
builds, `extend_mode` selector, ee_psf_lo resampling fix, extend_mode
naming). Driver: `examples/minerva/run_verification_v2.py fits idl mock`.

## What changed against v1

v1's real-data leg ran the pre-merge code, whose `psf_wings` was the
**convolution-fill** extension (template ⊗ detection PSF outside the
segment; now `extend_mode="psf_convolution"`). v2 runs the merged default:
the **least-squares scaled-PSF wings** build scheme
(`template_schemes.composite_psf_wings`), the same mechanism as IDL
classic's build. Everything else is production settings, unchanged from v1:
IDL apertures (0.70/1.20/1.20/1.50"), per-band Gaussian blur, wiener
kernels with scanned regularization, scene limits 800/1000, r<3' trial
patch. New in v2: the detection weight map is wired in (`wht_hi`, required
by the scheme), and the per-source `ee_psf_lo` chain is active — 0 of
~17.5k templates per band fell back to the filter-mean EE (v1: 100%
fallback by the resampling bug).

## Layout

- `runs/uds_<band>/` — real-data fits (fit table, residual, stamps,
  templates, per-run log, config snapshot). Configs in `runs/*.json`.
- `uds_monu/` — IDL leg: `compare_idl_vs_python_<band>.png` (2x2 panels;
  the b2 shape-correction blank was dropped), `idl_summary.json`,
  `idl_compare.log`.
- `uds_sims/` — mock leg: per-band `uds_<band>/` outputs,
  `mock_flux_ratio_<band>.png`, `mock_wiener_<band>.png`,
  `summary_all.json`, `mock.log`.
- `verification_summary.json` — combined, with commit hash.
- `driver.log` — full driver log.

## Results

IDL leg (est1 = python `ap_flux_corr_1` vs IDL `flux_F`, median py-IDL mag):

| band | N | raw aperture | ap→total | est1 SNR>25 | v1 est1 SNR>25 |
|---|---|---|---|---|---|
| F770W | 17565 | +0.01±0.04 | 0.950 | **+0.06±0.05** | +0.02±0.06 |
| F1280W | 16911 | +0.02±0.18 | 0.964 | **+0.05±0.04** | +0.01±0.05 |
| F1500W | 17431 | +0.03±0.37 | 0.967 | **+0.04±0.07** | +0.01±0.05 |
| F1800W | 17265 | +0.27±0.67 | 0.942 | **+0.06±0.09** | +0.03±0.09 |

(F1800W magnitude-cut column is the known noise-selection artefact; read
the SNR>25 column.)

Mock leg (recovered/true total flux, 696 fitted sources per band):

| band | med | p16–p84 | med hi self-fit | pull med | resid/noise | v1 med |
|---|---|---|---|---|---|---|
| F770W | 0.9755 | 0.907–1.010 | 0.9745 | −2.68 | 0.799 | 0.9998 |
| F1280W | 0.9776 | 0.916–1.015 | 0.9772 | −2.47 | 0.799 | 0.9983 |
| F1500W | 0.9788 | 0.920–1.017 | 0.9782 | −2.32 | 0.799 | 0.9982 |
| F1800W | 0.9810 | 0.927–1.017 | 0.9807 | −2.16 | 0.799 | 0.9983 |

## Finding

The least-squares `psf_wings` scheme biases total fluxes ~2–2.5% low on
injected truth, where the v1 convolution-fill scheme was exact to <=0.2%.
The F770W slice localizes it: point sources recover at 1.008, extended
sources at 0.971, isolated-extended equally low — so it is not a
blending/neighbour-drop effect. Mechanism: PSF-shaped wings cannot
represent extended outer profiles, so segment truncation is partially
reintroduced for resolved sources. IDL classic shares the mechanism (and
subphot's own source comment names the compactness problem), which is
consistent with the IDL leg moving by roughly the same amount
(+0.04..+0.06 mag) — the two codes agree with each other better than
either recovers truth for extended sources. Residual/noise is unchanged at
0.80: the fit is clean, the template *shape* under-fills extended wings.

Follow-up: verification **v3** runs the `wren` scheme (competitive-dilation
ownership — data trusted out to aperture scales, the SNR-graded version of
an aperture-floor segment support) on the same two legs. Decision recorded
in TODO.md: adopt wren, add a hard aperture-floor variant, or revert the
default to `psf_convolution`.
