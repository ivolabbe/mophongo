# Verification v3 — `wren` build scheme

Run 2026-08-11/12, same commit family as v2 (post template merge; fits on
`a2fecb1`+). Driver:
`examples/minerva/run_verification_v2.py fits idl mock --version v3 --scheme wren`.
Identical inputs and settings to v2 (r<3' patch, IDL apertures, per-band
blur, wiener kernels, scenes 800/1000); the only change is
`fit.extend_mode = "wren"` in both legs. Layout as v2 (`runs/`, `uds_monu/`,
`uds_sims/`).

Motivation: wren's competitive-dilation ownership trusts *data* out to
aperture scales (fill radius `max(R_EE95, r_aper + kernel_hw)` = 34-52 px
across the bands), the SNR-graded version of an aperture-floor segment
support — the candidate fix for v2's extended-source deficit.

## Results

IDL leg (vs `uds_monu`, ~17k matched per band):

| band | totcor ratio | est1 SNR>25 | v2 | v1 |
|---|---|---|---|---|
| F770W | 0.949 | +0.07±0.07 | +0.06±0.05 | +0.02±0.06 |
| F1280W | 0.959 | +0.05±0.06 | +0.05±0.04 | +0.01±0.05 |
| F1500W | 0.959 | +0.04±0.06 | +0.04±0.07 | +0.01±0.05 |
| F1800W | 0.890 | +0.07±0.09 | +0.06±0.09 | +0.03±0.09 |

Mock leg (recovered/true totals, 696 sources per band):

| band | med | med hi self-fit | pull med | resid/noise | v2 med | v1 med |
|---|---|---|---|---|---|---|
| F770W | 0.9635 | 0.9718 | −3.48 | 0.800 | 0.9755 | 0.9998 |
| F1280W | 0.9633 | 0.9726 | −3.44 | 0.801 | 0.9776 | 0.9983 |
| F1500W | 0.9632 | 0.9711 | −3.62 | 0.800 | 0.9788 | 0.9982 |
| F1800W | 0.9633 | 0.9707 | −4.23 | 0.800 | 0.9810 | 0.9983 |

F770W morphology slice: point sources 0.968 (v2: 1.008), extended 0.972
(v2: 0.971).

## Findings

1. Wren does **not** fix the extended-source deficit (0.972, unchanged) —
   at MIRI-band SNR the outer annuli are graded toward the PSF model
   anyway — and it **adds** a ~3% deficit for point sources that psf_wings
   recovered exactly, consistent with its positive-clipped noisy-data wings
   carrying spurious template flux. Net: 3.7% low across all bands,
   uniformly. Residual/noise unchanged at 0.80.
2. Against IDL, wren lands on psf_wings' numbers (+0.04..+0.07 at SNR>25):
   expected once the dominant term was identified as the missing
   `ee_psf_lo` application in the aperture estimator, common to every
   scheme. That is fixed post-run: `ap_flux_total_<i> =
   ap_flux_corr_<i>/ee_psf_lo` (measured on F770W: IDL totcor 1.450, bare
   ap_corr 1.358, ap_corr/ee 1.480 — within 2% of IDL).
3. Scene structure: wren's broader support merges more sources — 4867
   scenes (F1500W) vs psf_wings' 6031, with a handful of 500-800-template
   scenes from bisecting one giant coupled component. Fits cost ~10-16
   min/band (the single 310-min F1500W wall time shows normal per-scene
   progress and spans a likely machine sleep; F1800W with a broader kernel
   took 10.1 min).

## Scheme scoreboard (injected truth, F770W med_lo)

| scheme | med | points | extended |
|---|---|---|---|
| `psf_convolution` (v1, conv-fill) | **0.9998** | ~1.007 | ~1.0 |
| `psf_wings` (v2, least-squares PSF wings) | 0.9755 | 1.008 | 0.971 |
| `wren` (v3) | 0.9635 | 0.968 | 0.972 |

The mock verdict favors reverting the default `extend_mode` to
`psf_convolution`; the decision item lives in TODO.md (aperture-floor
template support), together with the remaining idea of a hard aperture
floor on *trusted data* combined with convolution-fill wings.
