# Aperture corrections in the mophongo Python port: diagnosis and design

**MINERVA UDS / mophongo IDL-to-Python port — 2026-07-15 (rev 3)**

This document covers (a) how the Python code currently builds templates and computes
aperture corrections, (b) the reference behavior — from the IDL source, M. Sharma's reading
of the original production code, the flux-estimator design document, and the published
literature — and (c) the design changes required to make the Python outputs consistent with
IDL and, separately, with the MINERVA NIRCam catalog totals.

All numbers are measured on current products: the F1500W fit tables in `uds_1500/`
(latest: `test_extendtemps_july2`), the IDL catalog
`cat_radec_uds-sbkgsub-v3.1-40mas-f1500w.fits`, the `_psf3as` PSF geojsons, and the 8"
STPSF grids in `data/PSF/`. **None of the changes in §5 are implemented yet.**

Summary of the four verified root causes and the design that addresses them:

| # | Root cause (§4) | Symptom | Design fix (§5) |
|---|---|---|---|
| 1 | PSF encircled-energy curves normalized to finite 3" stamps | corrections ~9% low (the Fig 1b "line" at 1.233 vs IDL 1.357) | true-total normalization, auto-computed (§5.2) |
| 2 | `ap_model` mixes PSF-stamp EE fractions with template-fitted amplitudes (Phase A) | raw-flux offset vs IDL appeared between June and July runs (+0.01 → −0.075 mag) | aperture bookkeeping always from the fitted template (§5.3) |
| 3 | extended-branch correction models contain raw noise-dominated halo data | the Fig 1b "cloud": totcor1 1.8–2.7 for marginal sources where IDL says 1.36 | one unified data/PSF-blended template for fit *and* corrections (§5.1) |
| 4 | catalog-tied `tcor_H` divides by measured aperture flux / imports catalog scatter | unbounded low-SNR tail ("the band"); also wrongly compared against IDL | two-step tie: internal Kron-style total, then one per-source catalog scale (§5.4) |

---

## 1. Current state vs IDL

The comparison (`compare_ap_mophongo.py`, F1500W, `use_phot & usef1500w==1` selection)
disagrees with IDL in three distinct ways.

### 1.1 Raw aperture flux (Fig 1a): an offset that was not there before

Median `ap_flux_1` / (IDL `flux_Ff1500w/totcorf1500w`), IDL raw mag < 23.5:

| Run | ratio | mag offset |
|---|---|---|
| `test_extendtemps_june25_2` | 0.989 | +0.01 |
| `test_extendtemps_july2` | 1.072 | **−0.075** |

The change is fully attributable to bookkeeping, not to the fit. Decomposition for bright
sources common to both runs, split by the july2 correction path (PSF-path vs template-path,
identified per §1.2):

| Bright sources, july2/june25_2 | amplitude `flux_1` | `ap_model_1` | `res_sum_1`/flux | `ap_flux_1` |
|---|---|---|---|---|
| PSF-path | 1.0000 | **1.1067** | ~0.6%, unchanged | 1.108 |
| template-path | 1.0000 | 1.0000 | ~0.3%, unchanged | 1.0000 |

The fitted amplitudes are identical; only `ap_model = flux_1 × apB_frac` moved, and only for
the sources whose `apB_frac` was switched (commit 62673f9, "Phase A") from the fitted
template's own aperture fraction to a PSF-stamp encircled-energy value. This is root
cause 2 (§4.2).

### 1.2 Internal aperture-to-total (Fig 1b): two per-source-uncorrelated populations

Fig 1(b) plots IDL `totcor` against Python `totcor1_1`. It is not an offset; it is two
populations, neither of which is correlated with IDL per source:

| Population | Fraction | Python `totcor1_1` | IDL `totcor` (same sources) |
|---|---|---|---|
| "line" — PSF-path | 95.0% | 1.233, constant | 1.358 [1.342, 1.503] |
| "cloud" — template-path | 5.0% | 2.20 [1.65, 3.03], max 265 | 1.364 [1.320, 1.524] |

The branch is not stored in the fit table; the split is inferred from discreteness: PSF-path
`totcor1 = 1/EE_band(Rphi)` can only take the PSF-region-map values, which span 1.230–1.236
across all 3,725 regions, so anything off that narrow line is necessarily template-path
(the bright+extended branch plus PSF-lookup/extension-failure fallbacks, which also return
data-extended stamps). Cross-check: `apcor1_1` flips branches with the same switch, and zero
sources are PSF-discrete in `totcor1_1` without being PSF-discrete in `apcor1_1`.

Per-source agreement, july2:

| Subset (`totcor1_1` vs IDL `totcor`) | N | med(py/IDL) | within 10% | Spearman r |
|---|---|---|---|---|
| all | 103,828 | 0.910 | 54% | −0.01 |
| F444W < 24.4 mag | 6,359 | 0.877 | 26% | −0.20 |
| IDL `totcor` > 1.45 (clearly non-point-source) | 25,523 | 0.811 | 1% | 0.00 |

- The **line** carries no per-source information (a constant), and its level is 9% below the
  IDL faint-bulk locus — root cause 1 (§4.1). 7.1% of line sources have IDL `totcor` > 1.6
  while Python pins them at 1.233.
- The **cloud** blows up in Python, not IDL, and the blow-up anti-correlates with F444W
  brightness: F444W < 23 mag → 1.44 [1.35, 1.62] (compatible with IDL's own extended range);
  23–25 → 1.80 [1.55, 2.19]; > 25 → 2.66 [2.10, 3.28] — root cause 3 (§4.3). The same
  sources' `apcor1_1` stays at 1.155 [1.12, 1.18]: halo noise suppresses Σ(H) and Σ(H·K)
  together, cancelling in the apF/apB ratio but not in 1/apB, which is the fingerprint of a
  noise-diluted normalization.

The corresponding targets differ by regime. For the faint bulk, IDL's own `totcor` is the
per-source PSF encircled energy (few-% spread), so per-source correlation is neither
achievable nor meaningful there — the target is the *level* (~1.35) with few-% scatter. For
bright/extended sources IDL's `totcor` is a genuine per-source morphology measurement, and
the target is per-source correlation (§1.3 sets the bar).

### 1.3 Per-source agreement across code versions

The full correction compared per-source against IDL `totcor`, same selection, for the three
code generations that produced comparable columns ("bright" = F444W < 24.4 mag):

| Version | Construction | med(py/IDL) all | within 10% all | r all | bright: within 10% | bright: r |
|---|---|---|---|---|---|---|
| `chimean_yoshi_diam_kronv7` (May 22) | internal Kron-based | 1.004 | 25% | 0.04 | 67% | 0.49 |
| `test_extendtemps_june25_2` | apcor1 × tcor_H(measured) | 1.029 | 24% | 0.00 | 70% | 0.56 |
| `test_extendtemps_july2` | `totcor1_1` (PSF curve of growth) | 0.910 | 54% | −0.01 | 26% | −0.20 |

Two observations that shape the design:

1. The faint bulk has never matched per source in any version (r ≈ 0 throughout); the
   May/June medians landing on IDL's 1.357 were population-level coincidences. (The June
   construction centers correctly because dividing by a measured total-flux proxy implicitly
   supplies the true-total normalization that stamp-normalized factors lack — §4.1.)
2. For bright sources, both the May Kron-based and the June measured constructions genuinely
   tracked IDL per source (r ≈ 0.5, ~70% within 10%): a measured flux ratio, an internal
   Kron model, and IDL's composite model all encode the same source morphology. The july2
   construction lost this (r = −0.20). Recovering it — without the June faint-end noise —
   is the quantitative bar for §5.

### 1.4 The catalog-tied correction (`tcor_1`, `apcor_1`)

`tcor_1` (median 1.18 for MIRI-detected, 1.69 [1.24, 2.81] with an unbounded tail for
MIRI-undetected sources) belongs to a different total-flux convention (§3.3) and has no IDL
counterpart — the IDL catalogs contain no tcor_H of any kind. Its remaining defects are
root cause 4 (§4.4). It must only ever be compared against the NIRCam super catalog (Fig 2),
never against IDL.

---

## 2. Current implementation

### 2.1 Template construction (`src/mophongo/templates.py`)

`Templates.extract_templates` builds one composite template `H` per source from the F444W
image via `_extended_composite` (templates.py:1158–1274), a decision tree driven by two
per-source SNRs: `snr_seg` (integrated over the source's segment; stored on the template at
:1209–1210) and `snr_wings` (over the owned background halo inside the measurement
aperture).

1. **Faint** (`0 < snr_seg < 1.5·fit_snrlo_psf` = 15): the in-segment core is blended in
   quadrature with the per-source detection PSF, `H_core = sqrt(data² +
   (e_seg·fit_snrlo_psf·PSF)²)` (:1242–1247; the port of IDL `mophongo__define.pro:327-328`),
   then flux-renormalized so the core sum equals the real in-segment flux (floored at the
   in-segment noise). PSF wings are added outside the segment to the 95% PSF-EE radius,
   anchored on the positive in-segment flux (:1264–1273; IDL :330-331).
2. **Bright + compact** (`snr_wings ≤ wings_snr_psf` = 3): real data in the segment, PSF
   wings outside; no core blend.
3. **Bright + extended** (`snr_wings > 3`): raw data over the source's area-weighted
   ownership territory (:1220–1222); no PSF anywhere. The data extension beyond the segment
   is deliberate: real flux exists outside the segmap and, for extended sources, the
   ownership halo captures it.

Branches 1–2 set `apcor_from_psf = True` (:1232). Ownership territories are disjoint
(area-weighted `kseg > knn` convolution; port of IDL :300-303). One known port difference:
IDL computes the segment SNR over positive pixels after rejecting negatives (:316-318), so
its faint criterion always fires for non-detections; the Python condition requires
`snr_seg > 0`, so negative-sum segments skip the core blend.

Knob inventory (all in `FitConfig`, `src/mophongo/fit.py`) — the template-side and
correction-side knobs are disjoint sets:

| Knob | Side | Used at | Role |
|---|---|---|---|
| `fit_snrlo_psf` (10.0) | template | templates.py:1215, 1246 | faint threshold (×1.5) and PSF-prior amplitude |
| `wings_snr_psf` (3.0) | template | templates.py:1218 | extended-vs-compact wing routing |
| `extend_template_ee` (0.95) | template | pipeline sizing | PSF-wing reach / template size cap |
| `tcor_lowsnr_psf` (False) | correction | pipeline.py:696, 899 | gates the tcor_H denominator blend |
| `tcor_blend_center` (1.5), `tcor_blend_width` (0.30) | correction | pipeline.py:916–920 | logistic weight of that blend (center = 1.5·`fit_snrlo_psf`, which couples the *threshold value* but not the machinery) |

### 2.2 Aperture corrections (`src/mophongo/pipeline.py:_add_aperture_photometry`, :633–986)

Templates are unit-sum normalized over their own cutouts; `template_norm` holds the
pre-normalization F444W sum. With `Rphi` the photometric aperture radius (0.6" for F1500W):

| Symbol | Current computation |
|---|---|
| `apB_frac` | PSF-path sources: `EE_band(Rphi)` from the band PSF stamp at the source position (:816–833). Template-path: aperture sum of the unit-sum convolved template `H·K` (:834–835). |
| `apF_frac` | Same switch on the F444W side (:861–867). |
| `apcor1` | `apF_frac / apB_frac` (:872). |
| `totcor1` | `1 / apB_frac` (:875). |
| `ap_model` | `flux_1 × apB_frac` (:848) — **uses the same switched `apB_frac`**, which is root cause 2. |
| `apf_data` | measured neighbour-subtracted F444W flux in the Rphi aperture (:881–883). |
| `aper_rphi` | logistic blend of `apf_data` with a catalog-anchored prediction (:885–926). |
| `tcor_H` | `f_f444w / aper_rphi` (:928–931). |
| `res_sum` | band residual over the aperture disk, other sources' segment pixels masked (:933–953). |

Output estimators:

```
ap_flux      = ap_model + res_sum
ap_flux_est2 = ap_model · totcor1 + res_sum
ap_flux_corr = ap_model · apcor1 · tcor_H + res_sum
```

All encircled-energy fractions are normalized by the finite stamp sums: the 3"-wide
`_psf3as` PSF stamps for the PSF path (`utils.psf_ee_at_radius` divides by `psf.sum()`),
the template cutout for the template path.

---

## 3. Reference behavior

### 3.1 The IDL recipe

From the IDL source in `legacy/autopilot/` and M. Sharma's reading of the production code
(July 2026, paraphrased):

- **The flux is measured on the neighbour-subtracted image.** IDL subtracts all *neighbour*
  templates and measures the target's aperture flux on what remains (`phot = org −
  model_nn`, design doc Eq. 2). That measurement contains the target's model flux *plus all
  residual light in the aperture* — flux the template did not capture. Equivalently,
  `aper(phot) = ap_model + res_in_aperture` (design doc Eq. 3). No *additional* residual
  term (e.g. a segment sum) is added beyond what the aperture already contains — the
  residual aperture sum computed in `solve()` (`mophongo__define.pro:1135`) feeds the
  regularization loop only.
- **The corrections are always evaluated on a model, never on data.** Above segment SNR 15
  the model is the composite (in-segment data + noiseless PSF wings); below it the
  correction routine substitutes the pure per-source PSF. The written corrections are
  `psfcor = apF/apB` and `totcor = 1/apB` on that model (design doc Eqs. 5–7).
- **The final flux is Estimator 1**: `flux_F = aper(phot) × totcor`
  (`old/dophot.pro:820-824`).
- **The external catalog is never used** (positions and IDs only). IDL's "total" is a purely
  internal, model-defined convention, and **no tcor_H exists anywhere in the IDL outputs** —
  the design document's tcor_H (Eq. 8) is a separate reconstruction that was never applied
  to the catalogs being compared against.
- Caveats: the exact correction routine (`subphot.pro`) is not in the repository, and
  poorly-fit sources carry a fit-quality inflation of `totcor` (catalog max 62) that cannot
  be reproduced.

### 3.2 The published recipe

- **Wuyts et al. 2008** (FIREWORKS, arXiv:0804.0615, Eq. 6): the low-res aperture flux is
  scaled by the ratio of the high-res color-aperture flux to the convolved high-res model's
  flux in the same low-res aperture — the `apcor1` construction.
- **Skelton et al. 2014** (3D-HST, ApJS 214, 24, §3.2, Eq. 1): aperture-to-total =
  `FLUX_AUTO/FLUX_APER` × the inverse PSF growth-curve fraction at the circularized Kron
  radius (growth curve normalized at r = 2"). The Kron radius is floored at the
  color-aperture radius, so for small/faint sources the total correction degenerates to the
  point-source growth curve — a geometric floor, not an SNR cut. Because the Kron aperture
  scales with the source, the measured Kron flux remains well-defined at all brightnesses.
- **Weibel et al. 2024** (arXiv:2403.08872): the same recipe on the F444W PSF, with a
  Kron-area floor.

In every published version, the aperture-to-total correction of faint sources comes from a
model growth curve, never from flux measured in a large aperture on the science data.

### 3.3 The two total-flux conventions and the valid comparisons

Two different, both legitimate definitions of "total flux" are in play and must never be
mixed on one axis:

1. **mophongo-internal totals** (IDL convention): the aperture flux extrapolated over the
   source's own model, `totcor1 = 1/apB`. This is what IDL ships.
2. **Catalog-tied totals** (release convention): fluxes on the system of the MINERVA
   catalog's `f_f444w`, so that a source with a flat F444W→MIRI color reproduces the catalog
   total. The two systems genuinely differ per source: matching IDL's own internal F444W
   total (`flux_Ff444w`) to the catalog `f_f444w` gives a median ratio 0.96 overall but
   **0.83 [0.65, 0.88] for bright (f_f444w > 50) sources** — a real, morphology-dependent,
   per-source difference (Kron totals vs model totals), not an error in either system.

The IDL catalog columns are
`id_irac, xdet, ydet, flux_Ff444w, flux_Ff1500w, eflux_Ff1500w, flux_contamf1500w, shx/shy,
chi/chi_half/chi_ann, rbg_ann, contam, snr_nn, psfcorf1500w, totcorf1500w, whtf1500w,
usef1500w, original_id`, and the valid pairings are:

| IDL column | Meaning | Python counterpart |
|---|---|---|
| `flux_Ff1500w / totcorf1500w` | aperture flux on the neighbour-subtracted image (contains residual light) | `ap_flux_1` (= ap_model + res_sum) |
| `psfcorf1500w` | apcor1 = apF/apB (model) | `apcor1_1` |
| `totcorf1500w` | totcor1 = 1/apB (model) | `totcor1_1` |
| `flux_Ff1500w` | Estimator 1 = aperture flux × totcor | `ap_flux_est1_1` (§5.4) |
| `flux_Ff444w` | IDL's internal F444W total | mophongo internal F444W total (§5.4) |

Anything built with the catalog tie (`tcor_*`, `apcor_*`, the est3 columns) compares only
against the super catalog. One small known asymmetry in the raw-flux pairing: the Python
`res_sum` masks pixels of other sources' segments inside the aperture; the IDL aperture does
not.

---

## 4. Root causes (all verified on run products)

### 4.1 Finite-stamp normalization of the PSF encircled-energy curves

`utils.psf_ee_at_radius` normalizes by `psf.sum()` over the stamp. The run's PSF geojsons
use 3"-wide stamps (`psf_size = 3.0` in `run_770.py`), and the stamps do not contain the
full PSF flux:

| Measurement | Value |
|---|---|
| EE_F1500W(0.6") normalized to the 3" geojson stamp | 0.8115 → totcor1 = 1.2323 — reproduces the fit table's 1.233 exactly |
| Fraction of true F1500W PSF flux inside the 3" stamp (8" STPSF grid `UDS_MIRI_F1500W_OS4_GRID9.fits`; inscribed-disk fraction — the drizzled stamps are circularly apodized, corner pixels identically zero) | **0.9192** |
| EE_F1500W(0.6") normalized to the 8" grid | 0.7427 → totcor1 = **1.3465** — matches IDL's 1.357 to <1% |
| EE_F444W(0.6"), 3"-stamp vs 8"-grid | 0.9428 vs 0.9037 (stamp holds 96.2%) |
| resulting true apcor1 | 1.217 (IDL `psfcor` 1.255; residual ~3% — §7) |

Reproduction (mophongo conda env, repo root):

```python
from mophongo.psf_map import PSFRegionMap
from mophongo import utils
import numpy as np
from astropy.io import fits

prm = PSFRegionMap.from_geojson(".../uds-sbkgsub-v3.0-80mas-f1500w_psf_psf3as.geojson")
psfs = [p for p in np.asarray(prm.psfs) if np.isfinite(p).all() and p.sum() > 0]
print(1 / np.median([utils.psf_ee_at_radius(p, 7.5) for p in psfs]))   # 1.2323

grid = fits.getdata("data/PSF/UDS_MIRI_F1500W_OS4_GRID9.fits")          # OS4, 0.0275"/px
print(1 / np.median([utils.psf_ee_at_radius(np.asarray(p, float), 0.6 / 0.0275)
                     for p in grid]))                                   # 1.3465
```

The template stamps are much less affected: template cutouts extend to
`r_fill = max(R95, Rphi + kernel half-width)` ≈ 2.1" radius for F1500W, and the implied
convolved-template aperture fraction for bright compact sources is ≈ 0.733 (from the §4.2
decomposition) — within ~1.5% of the true 0.7427. The truncation problem is specific to the
3" PSF stamps used by the PSF path.

Two consequences: (i) every PSF-path correction is low by the stamp capture fraction
(−8.1% at F1500W; worse at F1800W); (ii) any construction that divides by a *measured*
total-flux proxy (the June-era tcor_H) inherits the correct normalization implicitly —
which is why those eras centered on the right median (§1.3) despite their noise.

### 4.2 `ap_model` mixes normalizations (the Fig 1a offset)

The fitted amplitude `flux_1` is defined against the unit-sum *fitted template*: it is the
model flux within the template's own footprint. `ap_model = flux_1 × apB_frac` is therefore
only the model's aperture flux if `apB_frac` is measured on that same template. Phase A
(commit 62673f9) replaced `apB_frac` with the 3"-stamp PSF EE for PSF-path sources while the
amplitudes remained template-fitted, which multiplies `ap_model` by the ratio of the two
normalizations — measured at ×1.1067 for bright PSF-path sources (§1.1), with template-path
sources bit-identical (×1.0000) and amplitudes bit-identical (×1.0000). This single change
produced the −0.075 mag raw-flux offset against IDL; the June runs, whose `ap_model` used
template fractions throughout, sat at +0.01 mag.

The invariant to restore: **aperture bookkeeping (`ap_model`, `ap_flux`) always uses the
fitted template's own aperture fraction. Curve-of-growth substitutions belong only in the
correction factors.** (§5.1's unified template makes this automatic, because the fitted
model and the correction model become the same object.)

### 4.3 Extended-branch correction models contain raw halo data (the Fig 1b cloud)

For template-path sources, `totcor1 = 1/apB_frac` with
`apB_frac = aper(H·K, Rphi) / Σ_stamp(H·K)`, and the bright+extended branch's `H` is raw
data over the whole ownership territory (§2.1, branch 3). The branch is entered exactly when
the owned halo sums positive at ≥3σ (`snr_wings > 3`) — so, by selection, a positive halo
flux (real low-surface-brightness light, neighbour contamination, or a noise fluctuation)
enters `Σ(H)` while lying mostly outside the 0.6" aperture. `apB_frac` shrinks with the
halo-to-core flux ratio, and `1/apB` grows without bound as the core gets fainter.

The measurements (§1.2) bear this out: the blow-up is absent for genuinely bright cloud
members (F444W < 23: totcor1 1.44, compatible with real extended morphology) and grows
toward faint ones (F444W > 25: 2.66); and `apcor1` stays near the PSF value throughout,
because the halo dilution cancels in the apF/apB ratio but not in 1/apB.

In deep NIRCam data `snr_seg > 15` holds down to F444W ~27 mag, so the faint/PSF branch
never intercepts these marginal sources; the routing between "compact" and "extended" is
carried entirely by the 3σ wing test, and a hard threshold on a marginal quantity puts
thousands of sources on the noisy side of the switch.

### 4.4 The catalog-tied `tcor_H` divides by noise (the band)

`tcor_H = f_f444w / aper_rphi`, where `aper_rphi` blends the *measured* neighbour-subtracted
F444W flux in the Rphi aperture (~707 pixels at 40 mas) with a catalog-anchored prediction,
weighted by a logistic in `snr_seg` (center 15). Two remaining failure modes: (i) the weight
is keyed to the F444W *segment* SNR, which stays high for compact sources whose 707-pixel
aperture sum is nonetheless sky-noise dominated, so measured noise still enters (12.3% of
raw denominators are negative for MIRI-undetected sources); (ii) even at zero weight, the
prediction imports the catalog `tot_cor` per-source scatter. Result: MIRI-undetected
`tcor_1` = 1.69 [1.24, 2.81] with an unbounded tail. Both the IDL recipe (§3.1) and the
published recipe (§3.2) avoid this by construction: corrections come from models, and the
only measured quantities are fluxes in apertures matched to the source size.

---

## 5. Design changes (none implemented yet)

### 5.1 One unified template per source, used for the fit and all corrections

Replace the three-branch decision tree (§2.1) with a single construction that blends data
and PSF model smoothly, per region, according to what the data can support:

- **Core (in-segment)**: the existing quadrature blend with the PSF prior
  (IDL `mophongo__define.pro:327-328` port) plus flux-preserving renormalization — bright
  cores stay data, noise cores converge to the PSF, continuously.
- **Halo (owned territory outside the segment)**: the same idea applied to the wings — blend
  owned halo data with PSF wings scaled to the (positive) in-segment flux, weighted by the
  halo's own measured SNR, rather than switching all-or-nothing at `snr_wings = 3`. A bright
  extended galaxy keeps its real measured wings (the reason the data extension exists);
  a marginal halo degrades gracefully toward the noiseless PSF wings instead of injecting
  noise into the model.
- **One model per source**: this same template is used for the fit, for `ap_model`, and for
  every correction factor.

What this buys, relative to patching each symptom separately:

- Root cause 3 disappears (noisy halos no longer dilute `Σ(H)`), while true extended
  morphology is preserved exactly where the data measures it — including the per-source
  bright-end totcor spread that IDL shows and that the current PSF path erases.
- Root cause 2 *cannot recur*: with a single model there is no second normalization to mix
  into `ap_model`.
- The estimator algebra (`aper(phot) = ap_model + res_sum`, `est2 = ap_model·totcor1 +
  res_sum`) is exact, since amplitude, aperture fraction, and corrections refer to one
  model.
- No hard faint/compact/extended discontinuities in any output quantity.

The faint limit of the unified template is the PSF (as in IDL); the bright-compact limit is
segment data + PSF wings (as in IDL); the bright-extended limit is data throughout (the
deliberate improvement over IDL). The SNR-15 core threshold and the wing blending scale are
the existing `fit_snrlo_psf`/`wings_snr_psf` knobs, reinterpreted as blend scales rather
than branch switches. The IDL positive-pixel treatment of the segment SNR (§2.1) should be
adopted at the same time so non-detections blend fully to the PSF.

### 5.2 True-total normalization of every curve of growth, computed automatically

- `PSFRegionMap` gains a per-region **`containment`** — the fraction of the PSF's total
  flux contained within the stored stamp (standard PSF "containment" in the STPSF/JWST
  sense) — computed where the PSFs are built: the geojson region PSFs are drizzled from
  STPSF grid parents with 8" support, so the constructor can record the contained fraction.
  The drizzled stamps are circularly apodized (corner pixels identically zero; support
  radius ≈ inscribed radius), so the correct definition is the **inscribed-disk** flux
  fraction at r = stamp width / 2, not a square-box fraction (which would over-count ~1%
  corner flux that is not in the stamp sum). Serialized in the geojson; defaults to 1.0 for
  plain arrays and old files. Measured values for the current setup: F444W ≈ 0.962 (r =
  1.50"), F1500W ≈ 0.919 (r = 1.56") (per band; values depend on `psf_size`). The residual
  ~0.4% between this and the empirically-required factor (0.9152 at F1500W) is a genuine
  drizzle-vs-parent core-shape difference, filed with the §7 EE-residual open item.
- `EE_true(r) = EE_stamp(r) × containment` wherever a PSF curve of growth is used in
  correction factors. Template-based fractions get the analogous stamp-edge extrapolation
  via the band PSF's `containment` at the template's outer radius (a ≲1.5% effect for the
  current `r_fill`, per §4.1).
- No user-supplied numbers. Operationally, the cached `_psf3as` geojson/fits pairs predate
  `containment` and must be regenerated once per band; the loader must **warn loudly** when
  a map without `containment` is used (corrections silently stay stamp-normalized
  otherwise).

Expected direct effect: the PSF-limit totcor1 moves 1.233 → 1.347, onto IDL's 1.357 (<1%).

### 5.3 Aperture bookkeeping from the fitted template (immediate fix)

Independent of §5.1's timeline: revert `ap_model` (and hence `ap_flux`) to the fitted
template's own aperture fraction for every source, undoing the Phase-A substitution for
PSF-path sources. Restores Fig 1(a) to the June state (+0.01 mag). Curve-of-growth values
remain available for `totcor1`/`apcor1` until §5.1 lands, at which point fitted template and
correction model coincide.

### 5.4 The estimator suite and the two-step catalog tie

Replace the blended `tcor_H` (§4.4) — including `tcor_lowsnr_psf`,
`tcor_blend_center/width`, `_tcor_blend_weight`, the clamp, and the `tcor_w_*`/`aper_pred_*`
columns (all correction-side; the template knobs are untouched, §2.1 table) — with two
separately-motivated, separately-named steps, and carry **four** flux estimators with
explicit conventions:

| Column | Definition | Convention | Compare against |
|---|---|---|---|
| `ap_flux_est1` | `(ap_model + res_sum) · totcor1` | internal (IDL-exact: Estimator 1) | IDL `flux_F*` |
| `ap_flux_est2` | `ap_model · totcor1 + res_sum` | internal (design-doc Estimator 2: residual unscaled) | IDL `flux_F*` (secondary) |
| `ap_flux_est3int` | `ap_model · apcor1 · tcor_int + res_sum` | internal, Kron-convention total | (internal; sanity vs IDL `flux_Ff444w` on the F444W side) |
| `ap_flux_est3cat` | `ap_model · apcor1 · tcor_int · s_cat + res_sum` | catalog-tied (release) | super catalog (Fig 2) |

**Step 1 — `tcor_int`: internal Kron-style aperture-to-total on the F444W model** (design
doc Eq. 8, evaluated on the unified template rather than on data or catalog columns):

```
tcor_int = [ FLUX_AUTO(model) / FLUX_APER(model, Rphi) ] × 1 / EE_PSF444(k·R_circ)
```

Where and how the Kron quantities are computed (all on the model, none on data or catalog
columns):

- The measurement runs per source on the **unified F444W template stamp** (unit-sum template
  × `template_norm`), using the source's own segment from the segmap stamp, via photutils
  `SourceCatalog` — the same machinery and Kron conventions already used by
  `src/mophongo/catalog.py` (`kron_radius`, `kron_flux` are in its default column set).
- `FLUX_AUTO(model)` = the photutils `kron_flux` of the model stamp (elliptical Kron
  aperture, standard k = 2.5), with the circularized Kron radius
  `r_kron = 2.5·kron_radius·√(a·b)` **floored at the color-aperture radius**
  (`use_aper/2`, Skelton-style) and capped at the template stamp reach.
- `FLUX_APER(model, Rphi)` = `template_norm × apF_frac(Rphi)` — already computed.
- The Kron→total factor is `1/EE_PSF444_true(r_kron)` from the true-normalized F444W PSF
  growth curve (§5.2).
- The same machinery applied to the F444W side gives the internal F444W total for step 2:
  `F444W_total(mophongo) = kron_flux(model) / EE_PSF444_true(r_kron)`.

This is a genuinely different construction from `totcor1 = 1/apB` for extended sources — the
two are kept as separate, separately-documented columns: `totcor1` is the IDL-comparison
quantity; `tcor_int` is the internal Kron-convention total correction. Because the Kron
aperture scales with the source (small apertures for faint/compact sources, where the floor
takes over) and the model is noise-free by §5.1, `tcor_int` is well-defined and bounded at
every SNR. The May `kronv7` implementation and IDL's `totalphot`
(`mophongo__define.pro:1293-1303`, via `detect_kron`) are prior art for exactly this
construction — and the kronv7 bright-end per-source correlation with IDL (r = 0.49, §1.3) is
the floor to beat.

**Step 2 — `s_cat`: one per-source scale from the mophongo total system to the catalog
system**:

```
s_cat = f_f444w(catalog) / F444W_total(mophongo)
```

where `F444W_total(mophongo)` is the internal Kron-convention F444W total from the same
step-1 machinery applied to the F444W side. Applied multiplicatively to the model-shape flux
of every band (residual unscaled, per the design document's separation of model-shape and
residual flux), it puts all released fluxes on the catalog total system, with the F444W tie
exact by construction. The §3.3 measurement (IDL-internal vs catalog totals = 0.83 ± 0.13
for bright sources) shows this factor is real, per-source, and morphology-dependent — a
global scalar would not do. Both `tcor_int` and `s_cat` (and their product) are written as
columns so the internal and catalog conventions remain separable in every downstream
comparison.

### 5.5 Comparison-script changes (`compare_ap_mophongo.py`)

- Fig 1 (IDL axis): panel (b) `totcorf*` vs `totcor1_1` plus a new `psfcorf*` vs `apcor1_1`
  panel; panels (c)/(d) use `ap_flux_est1_1` (exact parity), optionally est2 alongside.
  Optional new panel: IDL `flux_Ff444w` vs the mophongo internal F444W total (a direct check
  of the internal total systems, independent of MIRI).
- Fig 2 (catalog axis): `ap_flux_est3cat_1` and its correction factors only.
- No catalog-tied quantity on an IDL axis anywhere.

### 5.6 Tests

- Unified template: faint limit = PSF (bit-level, given a pure-noise segment); bright-compact
  limit = segment data + PSF wings; halo blend monotonic in halo SNR; flux renormalization
  preserved.
- `containment`: geojson round-trip; `totcor1 == 1/(EE_stamp·containment)`; default 1.0
  when absent reproduces current behavior bit-for-bit, with a warning emitted.
- Bookkeeping invariant: `ap_model`/`ap_flux` from the fitted template's fraction;
  regression vs a pre-Phase-A reference.
- Estimators: `ap_flux_est1 == (ap_model + res_sum)·totcor1`; `tcor_int` positive/bounded
  on a noise-only source; `s_cat` reproduces a known injected total ratio.
- Validation: `pytest tests --ignore=tests/test_benchmark.py --ignore=tests/test_jwst_psf.py
  --ignore=tests/test_astro_fit.py`.

---

## 6. Acceptance criteria (F1500W re-run + `compare_ap_mophongo.py`)

1. **Fig 1(a)**: raw-flux offset back to |μ| ≤ 0.02 mag (from −0.075), σ unchanged or
   better.
2. **Fig 1(b)**, per regime: faint bulk med(py/IDL) = 1.00 ± 0.02 with few-% scatter (level
   target; correlation is not meaningful there); bright (F444W < 24.4) Spearman r ≥ 0.5 with
   ≳70% within 10% (the May/June bar; currently −0.20); IDL-extended subset
   (`totcor` > 1.45) med(py/IDL) → ~1 (from 0.81); no template-path tail above ~2 for
   F444W > 25 sources (cloud gone).
3. **Fig 1(c,d)** with `ap_flux_est1_1`: residual median |μ| ≤ 0.02 mag for mag < 24.
4. **Catalog side**: `tcor_int` finite and bounded at every SNR; `s_cat > 0` enforced (a
   negative catalog `f_f444w` cannot define a total-flux system — those sources get
   bad_value in the catalog-tied columns only). The large-positive `s_cat` tail for
   F444W-marginal sources is a ratio diagnostic that cancels algebraically in `est3cat`
   (closure: est3cat = f_f444w·B + res_sum) — report its size, don't clip it. Fig 2(b,d)
   residual median → 0.
5. **Regression**: fitted amplitudes `flux_1` unchanged where the template construction is
   unchanged; every change in `ap_*` columns attributable to a named fix.

## 7. Open items

- **F444W EE residual (~3%)**: true-normalized apcor1 1.217 vs IDL `psfcor` 1.255; IDL's
  implied EE_444(0.6") is 0.925 vs 0.904 from the STPSF grids. Candidate causes: empirical
  vs model PSF wings, the `f444w-matched` mosaic PSF, aperture-convention differences in the
  IDL run. Independent of the root causes above; needs its own check.
- **IDL fit-quality inflation**: `subphot.pro` (missing from the repo) inflates `totcor` for
  poorly-fit sources (catalog max 62); unreproducible — expect large-deviation outliers in
  any IDL comparison.
- **Per-source PSF details**: IDL reconstructs the PSF per source but ignores sub-pixel
  shifts (`mophongo__define.pro:322-326`); few-% per-source scatter is inherent.
- **IDL totcor floor**: the catalog shows `totcor ≥ 1.0` exactly (a clip; not reproduced).
- **`res_sum` segment masking**: Python masks other sources' segment pixels in the aperture
  residual; the IDL aperture does not. Small; revisit if Fig 1(a/c) scatter stays high after
  the fixes.
- **Unified-template blend details** (halo blend weight functional form, whether the wing
  prior scales with local or global SNR): to be settled at implementation time with tests
  against the §6 criteria.
- **Estimator-3 catalog tie still on the masked template (Stage 4c, scoped)**: Stage 4b
  moved the corrections (`totcor1`/`apcor1`) onto the partially-unmasked model, but the
  Estimator-3 tie (`tcor_int`, `f444w_ktot`, `template_norm`) is still measured on the
  ownership-masked fit template, so `est3cat` re-inherits the crowding artifact 4b removed
  (`est3cat/est1` ≈ 1.49× at <0.6″ vs isolated, matching the pre-4b masked-`totcor1`
  signature). Fix: use the unmasked-model F444W total (`template_norm + flux_beyond_stamp`)
  in the tie denominator and unmask `f444w_ktot` (scalar top-up). Full diagnosis, rulings,
  and implementer brief in [`docs/stage4c_scope_and_brief.md`](stage4c_scope_and_brief.md).
  Underneath it, a separate ~10–16% catalog-vs-IDL F444W total-definition offset remains —
  a science choice about which total system to anchor to, not a code bug.

## References

- Design document: *Total-flux estimators for mophongo template-fit low-res band photometry*
  (flux_estimator_comparison.pdf), Eqs. 1–17.
- Labbé et al. 2006, ApJ 649, L67 — original template-fit photometry.
- Wuyts et al. 2008, ApJ 682, 985 (arXiv:0804.0615) — Eq. 6, the EE-ratio correction.
- Whitaker et al. 2011, ApJ 735, 86 (arXiv:1105.4609) — point-source growth-curve totals.
- Skelton et al. 2014, ApJS 214, 24 (arXiv:1403.3689) — §3, Eq. 1; PSF-growth-curve
  Kron-to-total; minimum-radius floor for faint sources.
- Labbé et al. 2015, ApJS 221, 23 (arXiv:1507.08313) — spatially varying IRAC PSF maps.
- Weibel et al. 2024, MNRAS 533, 1808 (arXiv:2403.08872) — F444W PSF growth-curve totals.
- IDL source: `legacy/autopilot/mophongo__define.pro` (:253-378 templates; :1093-1154
  solve/regularization; :1276-1315 totalphot Kron path), `old/dophot.pro` (:782-869 catalog
  assembly); `subphot.pro` not in repo.
- M. Sharma, private communication (Slack, July 2026): IDL catalogs are raw Estimator 1,
  internal-only, no tcor_H applied; correction models are composite above segment SNR 15 and
  pure PSF below; per-source PSF EE scatter expected at the few-percent level.
