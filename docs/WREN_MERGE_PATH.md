# Merge path for the dev-wren aperture-correction and total-flux system

Companion to `docs/FORK_AUDIT_WREN.md`, which inventories what `wrensuess/mophongo`
`dev-wren` has that `main` does not. This document decides what of that inventory is
still wanted, restates the estimator algebra on `main`'s encircled-energy convention,
and sequences the work.

Three things changed on `main` after the audit was written, and each of them retires
part of it:

1. **The encircled-energy chain is settled** (`docs/ENCIRCLED_ENERGY.pdf`, commits
   `490e13c`..`dd9f5a5`). One scalar converts a fitted amplitude to a total flux:
   `ee_psf_lo`, the absolute encircled energy of the low-resolution PSF stamp at the
   source position, measured per region by `PSFRegionMap.refresh_ee` and recorded per
   template by `Templates.convolve_templates`. For a point source `A = f · S_lo`
   exactly, for any weight map and independently of how the high-resolution stamp is
   truncated.
2. **The template builder was rewritten** on the `template` branch
   (`src/mophongo/template_schemes.py`, 948 lines): `FitConfig.extend_mode` dispatches
   into `composite_psf_wings` (default), `composite_classic` and `composite_wren`, with
   `template_norm`, `wing_frac_lost`, `snr_seg`, `w_core` and `fpsf` recorded per source.
3. **Curve-of-growth helpers already exist** in `psf.py` (`_encircled_energy`,
   `stamp_encircled_energy`, `DrizzlePSF._ee_fraction_to_arcsec`), in `utils.py`
   (`CircularApertureProfile.cog`) and, in pixel-domain array-in/float-out form, in
   `template_schemes.psf_ee_radius_pix`.

None of that changes the case for the model-Kron path. `ee_psf_lo` answers "what is
this source's total flux". The Kron chain answers a different question — "what would
the reference catalogue have measured, if it had measured our model" — and the two are
kept side by side.

---

## 1. What the audit's tiers reduce to

| audit item | status here |
|---|---|
| Tier 1 #1 `utils` EE helpers | **retire.** `template_schemes.psf_ee_radius_pix` is the pixel-domain helper; `psf.py` has the rest. `psf_ee_area_pix` had no caller on wren either. |
| Tier 1 #4 `PSFRegionMap.resolve_key` | **take.** Small, and it removes a real risk now that `get_psf`, `get_ee_box` and a per-region growth curve are all queried per source. |
| Tier 4 #22 `PSFRegionMap.containment` | **do not port — replace with a growth curve.** wren needs `containment` because its stored stamps are unit-normalised, so `EE_true(r) = EE_stamp(r)·containment`. `main`'s stamps are absolutely calibrated (`NORMALIZ = first` grids, `in_units="cps"` drizzle), so `EE_true(r) = aper(stamp, r)` directly. What `main` *does* lack is EE at an arbitrary radius: `refresh_ee` caches two scalars per region. §2.6 adds the per-region curve, which is what the Kron path needs. |
| Tier 4 #23–25, the four estimators | **port in full**, with the columns renamed `est3int → est3` and `est3cat → est4`, and the algebra translated from `c_det`/`c_b` to `S_hi`/`S_lo` (§2.2). `est3` is not a fallback for truncated templates — it is a noise-free run of the catalogue's own Kron recipe on the model. See §2.3. |
| §1.10 `f444w_totcor_col` "declared and never read" | **reinstated, live.** It is the `tot_cor` column, and §2.4 needs it to recover `ee_kron_cat`. |
| Tier 3 #16–19 blend weight, ownership, `template_norm`, composite schemes | **already done** on the `template` branch. |
| §2.8 flag-bit collision `0x40`/`0x80` | **already resolved** on the `template` branch: `FLAG_DEBLENDED`/`FLAG_SATURATED` keep `0x40`/`0x80`, extension moved to `FLAG_PSF_EXTENDED = 0x100` and `FLAG_EXTEND_FAILED = 0x200`. Needs a regression test, nothing more. |
| §2.7 wren's `convolve_cutout` | **rejected.** `main`'s origin-parity/even-alignment convolution stands unchanged. wren's post-convolution unit-sum renormalisation is also rejected: `build_kernels` already renormalises the *kernel* to unit sum, which removes the same residual DC without touching the template. |
| §6.1(2) flux-block ridge | **not solved.** See §5. The `flux-bug` fix removed a different, larger term. |
| §1.3 truncation term / §1.7 crowding delta | **wanted, recast.** `trunc` collapses into `S_hi`; the crowding delta becomes the neighbour-blanking terms `apF_blank`/`apB_blank` (§2.2). |
| §2.4–2.5 performance cuts | **take both** — see §4. |

---

## 2. The merged design: four estimators, translated

### 2.1 The four columns

Adopted as named. All four share one model term shape and one residual term, and differ
only in how the aperture-to-total correction is built.

| column | definition | convention | compare against |
|---|---|---|---|
| `ap_flux_est1_<i>` | `(ap_model + res_sum)·totcor1` | internal, IDL-exact | IDL `flux_F*` |
| `ap_flux_est2_<i>` | `ap_model·totcor1 + res_sum` | internal, residual unscaled | IDL `flux_F*` (secondary) |
| `ap_flux_est3_<i>` | `ap_model·apcor1·tcor_int + res_sum` | internal, Kron-convention total | internal only (IDL `flux_Ff444w` is a raw 0.35" aperture flux, not a total — no cross-check possible) |
| `ap_flux_est4_<i>` | `ap_model·apcor1·tcor_int·s_cat + res_sum` | catalogue-tied release flux | SUPER catalogue |

`est1` is retained for IDL comparison only. Its residual term is multiplied by
`totcor1 ≈ 3–4`, which inflates the median by 8–13 % and the residual's variance
contribution by `totcor1² ≈ 10–16`; any quoted error that is not also scaled by
`totcor1` is overconfident by that factor.

### 2.2 Translating wren's algebra onto absolutely calibrated stamps

**The formulae cannot be transcribed verbatim.** wren's stored PSF stamps are
unit-normalised, so its chain carries the finite-stamp truncation through
`containment` (`c_det` on the detection side, `c_b` on the band side). `main`'s stamps
are absolutely calibrated, and the same physics is already carried by `S_hi` and `S_lo`.
Copying wren's expressions on top of `main`'s stamps double-corrects by `S_hi/S_lo`,
which is 4.6 % on the UDS products.

The substitution is `c_det → S_hi`, `c_b → S_lo`, with `S_hi = prm_hi.get_ee_box(ra, dec)`
and `S_lo = ee_psf_lo` (already recorded per template). Writing `A = flux_<i>`,
`R_cat = use_aper/2`, `R_img` the per-band radius, and `T`/`T_conv` the unit-sum
reference and band templates:

```
ap_model  = A · apB_book                       apB_book = aper(T_conv, R_img)
                                               apF_book = aper(T,      R_cat)

apF_corr  = (apF_book + apF_blank) · S_hi      apcor1  = apF_corr / apB_corr
apB_corr  = (apB_book + apB_blank) · S_lo      totcor1 = 1 / apB_corr
```

`apF_corr` and `apB_corr` are, by construction, the fraction of the source's **true
total** that the model puts inside `R_cat` at detection resolution and inside `R_img` at
band resolution. `S_hi/S_lo` therefore survives in `apcor1` as a genuine resolution
effect — wren's own ruling — and `totcor1` is a true aperture-to-total.

Three checks that a port must pin:

- **Point-source identity.** For an isolated point source under `psf_wings`,
  `apB_book = EE_lo(R_img)/S_lo`, so `ap_model·totcor1 = A/S_lo` exactly. That is
  `flux_<i>_total`. Hence

  ```
  ap_flux_est2_<i> == flux_<i>_total + res_sum        (exact, isolated point source)
  ```

  which is the bridge between the two chains and the single most valuable regression
  test in the set. `est2` therefore needs no aperture photometry: its model term is the
  amplitude the fit already wrote.
- **`est3` reduces to the same value.** `ap_model·apcor1 = (A/S_lo)·apF_book·S_hi`, which
  is the aperture-matched model flux on an absolute scale — for a point source
  `f·EE_hi(R_cat)`. The step-1 denominator `template_norm·apF_book` is *already* that same
  absolute aperture flux, so `tcor_int` carries **no** `S_hi` of its own and reduces to
  `1/EE_hi(R_cat)`, giving `est3 = A/S_lo`. Getting this wrong by an `S_hi` is the easiest
  mistake in the translation: `S_hi` belongs in `apcor1` (through `apF_corr`), not in
  `tcor_int`. Divergence between `est2` and `est3` is the resolved-source signal, not a
  bug.
- **The blanking terms are live under `psf_wings`.** `composite_psf_wings` normalises
  over the whole stamp and *then* zeroes neighbour-owned pixels, so a neighbour segment
  crossing the aperture removes model flux from it. `apF_blank`/`apB_blank` are the
  unit-sum-template equivalents of wren's `flux_beyond_aper/trunc_denom`: exact, not
  approximate, because the template is identically zero there. Keep wren's engineering —
  a bounding-box pre-check that skips the isolated common case, and
  `CircularAperture.to_mask('exact').multiply()` rather than `aperture_photometry`
  (~20× faster, identical sum), stored as a scalar and never as an array at 345k sources.

`trunc = template_norm/model_total` cancels in `apcor1` and survives in `totcor1`; wren
pins both properties in one test at `rel=1e-10`. Under `psf_wings` with the cutout
floored at the PSF stamp size, `model_total = template_norm/S_hi` and the free parameter
collapses into `S_hi`. Guard rather than assume: assert that the resampled unit-sum PSF
retains `> 0.999` of its sum on the cutout, and warn otherwise.

### 2.3 Why the model-Kron path earns its place

The MINERVA SUPER catalogue builds its totals as

```
f_band_cat = faper_band(use_aper) · tot_cor
tot_cor    = (fauto_KRON / faper_KRON) · 1/EE_kron_cat
```

Both factors are measured **on the data**, and the first is noise-dominated exactly where
template fitting is supposed to earn its keep. Measured on
`MINERVA-UDS_n3.0_m3.1_v1.2.1_..._SUPER_CATALOG_wMIRI.fits` (345,792 rows):

| `f_f444w/e_f444w` | N | `tot_cor` | NMAD | `fauto/faper` | NMAD | med `r_kron_circ` | no `r_kron` |
|---|---|---|---|---|---|---|---|
| 0–3 | ~105k | 3.51 | **1.67 (48 %)** | 3.09 | **1.37 (44 %)** | 0.296″ | **18.5 %** |
| 3–5 | 40,081 | 3.77 | 1.29 | 3.14 | 1.16 (37 %) | 0.373″ | 2.2 % |
| 5–10 | 34,617 | 3.09 | 1.15 | 2.59 | 0.96 (37 %) | 0.438″ | 0.4 % |
| 10–30 | 31,584 | 2.06 | 0.53 | 1.78 | 0.46 (26 %) | 0.506″ | 0.2 % |
| 30–100 | 16,261 | 1.54 | 0.27 | 1.36 | 0.24 (18 %) | 0.587″ | 0.5 % |
| > 100 | 10,026 | 1.28 | 0.18 (14 %) | 1.17 | 0.18 (15 %) | 0.781″ | 7.7 % |

57,622 rows (17 % overall) have no `kron_radius` at all, concentrated in the faintest
bin. The 7.7 % at `SNR > 100` is a separate population and is unexplained — probably
saturation flags, worth identifying before the comparison is trusted at the bright end.

The trend in `fauto/faper` — 1.17 at the bright end rising to 3.09 at the faint end — has
two contributions and a port must not conflate them. `use_aper` takes six discrete values
(0.2, 0.32, 0.48, 0.7, 1.0, 1.4″) and is a size proxy strongly correlated with brightness:
median SNR runs 1.67, 8.4, 19.4, 42.3, 94.6, 335 across those six classes, with 72.4 % of
the catalogue sitting on the 0.2″ floor. So the raw trend is mostly a *size* trend.

Holding `use_aper` fixed isolates the noise term, and it is large:

| `use_aper` | SNR | N | `fauto/faper` | `r_kron_circ` |
|---|---|---|---|---|
| 0.48″ | 0–3 | 62 | 4.24 | 0.951″ |
| 0.48″ | 3–10 | 2,857 | 2.02 | 0.657″ |
| 0.48″ | 10–30 | 12,627 | 1.62 | 0.555″ |
| 0.48″ | > 30 | 5,189 | 1.41 | 0.492″ |
| 0.70″ | 3–10 | 245 | 2.49 | 1.039″ |
| 0.70″ | 10–30 | 3,251 | 1.54 | 0.740″ |
| 0.70″ | > 30 | 7,692 | 1.27 | 0.604″ |
| 1.00″ | 10–30 | 511 | 1.60 | 1.053″ |
| 1.00″ | > 30 | 5,566 | 1.17 | 0.740″ |

Within one size class the measured Kron radius nearly doubles as SNR falls, and
`fauto/faper` follows it. That is Kron-radius inflation by noise, and it is exactly what a
Kron radius measured on the model removes. The 0.2″ floor class runs the other way
(2.82 → 3.56 with rising SNR) because that class is not a size class at all — it is
everything too faint or too compact to have earned a larger aperture.

The missing-`kron_radius` fraction is 13.2 % in the 0.2″ class and 10.8 % in the 1.4″
class, so it is bimodal: faint failures at one end and a distinct bright population at the
other, probably saturation or Kron failures on very extended objects. Identify the bright
group before trusting any bright-end comparison.

Running the same recipe on the *extended model template* — smooth, positive and defined
everywhere — gives all three per-source inputs with no measurement noise, for every
source including the 57,622 with no catalogue Kron radius at all. That is what `est3` is
for, and neither `A/ee_psf_lo` nor the catalogue's own `tot_cor` column supplies it.

### 2.4 `tot_cor` has three per-source inputs and one global curve

Read the recipe as four inputs, three of them per source and measured on data:

| input | per source? | from | degrades at faint SNR? |
|---|---|---|---|
| `fauto_KRON` | yes | Kron photometry on the detection image | yes |
| `faper_KRON` | yes | `use_aper` aperture on the detection image | yes |
| `r_kron_circ` | yes | Kron moments on the detection image | yes, and 18.5 % are missing entirely |
| `EE(r)` | **no** | one assumed F444W growth curve | no |

That split is the whole design. `_model_kron` replaces the three per-source inputs with
model-derived ones — `kron_flux_model`, `template_norm·apF_book`, `r_kron_model` — all
from a noiseless profile. The curve is the one piece we can either share with the
catalogue or take from our own PSF map, and §2.5a says which.

The catalogue does not publish the curve, but it is exactly invertible:

```
ee_kron_cat = (fauto_KRON / faper_KRON) / tot_cor
```

Measured on 288,153 rows with all three columns finite and positive: median 0.835, 1st
to 99th percentile 0.631 to 0.941, and only 0.13 % exceed 1. Binned by
`kron_radius_circ` it is monotone and tight:

| `r_kron_circ` (″) | N | `ee_kron_cat` | NMAD | same, `f/e > 20` |
|---|---|---|---|---|
| 0.10–0.20 | 21,732 | 0.664 | 0.026 | 0.571 |
| 0.20–0.25 | 34,134 | 0.720 | 0.021 | 0.659 |
| 0.25–0.30 | 36,273 | 0.777 | 0.021 | 0.740 |
| 0.30–0.35 | 38,346 | 0.818 | 0.011 | 0.817 |
| 0.35–0.40 | 36,403 | 0.838 | 0.005 | 0.838 |
| 0.40–0.50 | 55,856 | 0.849 | 0.005 | 0.851 |
| 0.50–0.70 | 44,131 | 0.873 | 0.015 | 0.876 |
| 0.70–1.00 | 14,961 | 0.905 | 0.008 | 0.905 |
| 1.00–2.00 | 5,351 | 0.930 | 0.012 | 0.930 |

That is an F444W encircled-energy curve, not a per-source nuisance. Two things follow.

**(a) A 288k-row acceptance gate for `get_ee_at`, valid for `r_kron ≳ 0.32″`.** Our
per-region growth curve (§2.6) must reproduce this, per source, from the drizzled
high-resolution PSF map. It is a far harder test than anything in wren's suite, and it is
free. Below 0.32″ the recovered curve is not SNR-independent (§2.5a) and is excluded
until that is understood. A systematic shortfall would
be the same wing deficit already in `TODO.md` (azimuthal model/star 0.83–0.95 at F444W),
now measurable against 288k independent points instead of a few dozen stars.

**(b) `s_cat` is not a flux-scale tie; it is two ratios.** With
`f_f444w = fauto_KRON/ee_kron_cat` and `ktot = kron_flux_model/ee_kron_int`,

```
s_cat = f_f444w / ktot = (fauto_KRON / kron_flux_model) · (ee_kron_int / ee_kron_cat)
                          \_____ Kron flux, model vs data ____/   \__ PSF EE, ours vs theirs __/
```

Write both factors as columns. wren's single `s_cat` conflates them, and its measured
bright-source value, `0.83 [0.65, 0.88]`, sits on top of the median `ee_kron_cat` of
0.835 with a 5th–95th range of 0.675–0.905. That coincidence is worth testing directly:
it suggests the "morphology-dependent per-source difference" wren attributed to
`s_cat` was substantially a missing `1/EE_kron` factor, i.e. an internal total that was
a Kron flux never corrected to a total. Test it; do not assume it.

It also gives `est4` a second, more robust route: since `ee_kron_cat` is recoverable
without the catalogue total, `est4` can be formed as `est3 · (ee_kron_int/ee_kron_cat)`
with no division by a noisy `f_f444w`. Compute both, report the ratio.

### 2.5 The model-Kron chain, restated

All of it runs on the fitted reference template in real flux units,
`T[slices_cutout] · template_norm`, never on image data, with the source's own segment.

```
r_floor        = R_cat = use_aper/2                          # Skelton floor, reference pixels
kron_flux_model,
r_kron_model   = model_kron(T·template_norm, seg,
                            kron_params=(2.5, 1.4, r_floor))  # photutils SourceCatalog
                 r_kron_model = max(2.5·kron_radius·sqrt(a·b), r_floor),
                                capped at 0.5·min(stamp.shape),
                                quantised to a 0.25 px grid
ee_kron_int    = EE_psf_hi(r_kron_model)                      # get_ee_at, absolute
ktot           = kron_flux_model / ee_kron_int                # model "fauto -> total"
tcor_int       = ktot / (template_norm · apF_book)            # model aperture -> total
ee_kron_cat    = (fauto_KRON/faper_KRON) / tot_cor            # catalogue, per source
s_cat          = f_f444w / ktot                               # positive-guarded
```

`tcor_int` is the catalogue's `tot_cor`, computed on the model. `s_cat` is F444W-side and
band-independent, so like `tot_cor` it is one factor per source applied to every band and
cancels exactly in colour. It is guarded on a positive catalogue total and deliberately
not clipped at the top: the large-positive tail cancels algebraically in `est4`.

Four notes a port must keep:

- **`model_total` replaces `trunc_denom`.** wren wrote
  `trunc_denom = template_norm + flux_beyond_stamp` with `flux_beyond_stamp` a
  `1/c_det`-anchored extrapolation. Under `psf_wings` the halo is `fpsf·P` with `P` the
  unit-sum resampled high-resolution stamp, so `model_total = template_norm/S_hi`:
  measured, not extrapolated. This needs the high-resolution PSF map wired into the
  config path (`psfs=[None, prm_lo]` today, already in `TODO.md`).
- **Stage 4c: the tie denominator must be unmasked.** wren's `est3cat` re-inherited the
  crowding artifact because the denominator was the ownership-masked `template_norm`.
  Under `psf_wings`, `template_norm` is the whole-stamp sum *before* neighbour blanking,
  so it is already unmasked — confirm in a test rather than assuming.
- **The `apcor_from_psf` shortcut is a performance gate, not a physics branch.** When the
  composite has converged to a bare point source (`w_core → 0`, `FLAG_PSF_EXTENDED`), the
  model Kron *is* the PSF Kron, so `ktot = model_total` and `tcor_int = 1/(apF_book·S_hi)`
  with no photutils call and no EE lookup. Pin that the shortcut and the full path agree
  on that population.
- **On cap, radii must not be mixed.** When the `0.5·min(stamp.shape)` cap engages,
  photutils' elliptical (edge-truncated) Kron flux is replaced by the circular flux at
  the same capped radius, so `kron_flux_model/ee_kron_int` never mixes radii. Degenerate
  moments, an empty segment or any photutils exception fall back to a circular aperture
  at the quantised floor radius scaled by `template_norm`.

### 2.5a `EE_cat(r)` is one global curve, and that decides how `s_cat` is built

Step 1 of §5.4 needs `EE_PSF444_true(r_kron)`. The catalogue's own version of that curve
is recoverable from §2.4. Regressing `ee_kron_cat` on `kron_radius_circ` in 200 quantile
bins leaves a residual of NMAD `4.3e-4` (0.05 % fractional), and the residual medians
over a 4x4 sky-quartile grid span `-6e-5` to `+1e-5`. The catalogue used **one** F444W
growth curve for the whole field, looked up at `kron_radius_circ`.

The *curve* being global does not make the *correction* global: it is evaluated at a
per-source radius, so `1/EE(r_kron_circ)` is per-source information — it just carries no
per-source information beyond the radius. That is also why the recovery is tight even
including SNR < 3 sources. A faint source has a noisy radius, but wherever that radius
lands the lookup is still exact, so the noise goes into *where* a source sits on the
curve, never into the curve itself.

Recovered, `flag_kron == 0`, radius in arcsec:

| r | EE | 1/EE | | r | EE | 1/EE |
|---|---|---|---|---|---|---|
| 0.10–0.15 | 0.583 | 1.714 | | 0.55–0.60 | 0.875 | 1.143 |
| 0.15–0.20 | 0.668 | 1.498 | | 0.65–0.70 | 0.894 | 1.119 |
| 0.20–0.25 | 0.720 | 1.388 | | 0.75–0.80 | 0.904 | 1.106 |
| 0.25–0.30 | 0.777 | 1.288 | | 0.85–0.90 | 0.911 | 1.097 |
| 0.30–0.35 | 0.818 | 1.222 | | 0.95–1.00 | 0.918 | 1.089 |
| 0.35–0.40 | 0.838 | 1.194 | | 1.10–1.15 | 0.927 | 1.079 |
| 0.40–0.45 | 0.847 | 1.181 | | 1.25–1.30 | 0.934 | 1.070 |
| 0.45–0.50 | 0.854 | 1.171 | | 1.35–1.40 | 0.939 | 1.065 |

The shape is a NIRCam F444W growth curve: steep rise to the first Airy minimum near
0.375″, a shoulder, then a slow climb. It does **not** reach 1 — 0.939 at 1.4″ — so it
carries its own normalisation convention, which is precisely what `s_cat` must absorb.

**The curve is SNR-independent only above `r ≈ 0.32″`.** At fixed fine radius bins,
comparing `SNR < 5` against `SNR > 20`:

| r (″) | N faint | EE | N bright | EE | difference |
|---|---|---|---|---|---|
| 0.16–0.18 | 6,906 | 0.6545 | 31 | 0.5725 | −0.0820 |
| 0.20–0.22 | 11,028 | 0.7031 | 79 | 0.6410 | −0.0621 |
| 0.26–0.28 | 12,313 | 0.7709 | 102 | 0.7337 | −0.0371 |
| 0.32–0.34 | 11,645 | 0.8213 | 420 | 0.8193 | −0.0020 |
| 0.36–0.38 | 9,733 | 0.8367 | 499 | 0.8365 | −0.0002 |
| 0.42–0.44 | 6,219 | 0.8475 | 1,048 | 0.8474 | −0.0001 |
| 0.52–0.56 | 3,286 | 0.8670 | 3,350 | 0.8673 | +0.0003 |
| 0.72–0.78 | 503 | 0.9017 | 2,448 | 0.9019 | +0.0002 |
| 1.00–1.10 | 103 | 0.9222 | 1,123 | 0.9222 | −0.0001 |

Above 0.32″ the two agree to `2e-4`. Below it they diverge to 8 percentage points. The
curve's slope there is ~0.9 per arcsec, so `-0.082` corresponds to about `0.07″` of
radius — roughly one 40 mas detection pixel — which points at `r_kron_circ` as stored
not being exactly the radius the EE was evaluated at, showing up only where the curve is
steep. The bright samples in those bins are 31 to 102 objects and are effectively stars,
so this is a small, specific population. **Restrict the §2.4a acceptance gate to
`r_kron ≳ 0.32″` until it is explained.**

Two further open items: 3.4 % of sources sit 2 % or more below the curve, unexplained;
and `flag_kron` is 0 for every row in this release, so it cannot be used to filter them.

Do not ship the table. Derive it at run time from the configured catalogue columns — it
is exact, free, and it tracks whatever release is actually in use.

**Consequence for step 1.** Compute `ktot` twice, from the same `kron_flux_model`:

```
ktot     = kron_flux_model / ee_kron_int    # ours:   get_ee_at(ra, dec, r_kron), per region
ktot_cat = kron_flux_model / ee_kron_cat    # theirs: EE_cat(r_kron), one global curve
```

`tcor_int` and `est3` use `ktot` — `est3` is the *internal* Kron-convention total and must
be on our own PSF system. `s_cat = f_f444w/ktot` then factorises exactly:

```
s_cat = (fauto_KRON / kron_flux_model) · (ee_kron_int / ee_kron_cat)
```

with `f_f444w/ktot_cat = fauto_KRON/kron_flux_model` isolating the Kron-flux term — the
model-versus-data comparison we actually want — and `ee_kron_int/ee_kron_cat` isolating
the PSF-model term. That second factor splits again, into a global curve ratio and the
per-region structure the catalogue does not have (`ee_box` spans 0.947–0.966 across the
1694 F444W regions, ±1 %). Write all three as columns.

This also gives `est4` a route that never divides by a noisy `f_f444w`:
`est4 = est3 · (ee_kron_int/ee_kron_cat) · (fauto_KRON/kron_flux_model)`. Compute both
and report the ratio.

### 2.6 The per-region growth curve (replaces `containment`)

`model_kron` needs `EE_psf_hi(r)` at an arbitrary radius, and `refresh_ee` caches only two
per-region arrays (`ee_box`, `ee_rlim`) plus the scalar `r_lim` — which is in units of
`PSFRegionMap.pscale`, and `pscale` defaults to 1.0 and is never assigned by any caller, so
`r_lim` is in pixels. Add to `PSFRegionMap`:

```
refresh_cog(dr=0.25)     -> per-region cumulative curve on a fixed radius grid out to r_lim,
                            measured on the stored (absolute) stamp; no normalisation
get_ee_at(ra, dec, r)    -> resolve_key + linear interpolation
```

Because the stamps are absolute, this *is* `EE_true(r)`; wren needs `containment` only
because its stamps are not. Two properties to preserve from wren's version:

- **Key on the region, never on the PSF array.** wren fixed a real collision here:
  `get_psf` returns a fresh ndarray per call and CPython reuses freed ids, so an
  `id(psf)`-keyed cache silently served another region's growth curve. Indexing a
  per-region array by `resolve_key` removes the class of bug rather than patching it.
- **Radius quantisation to 0.25 px** so the cache hits — hours saved on a 340k-source run
  for a < 0.2 % EE effect.

The radius arrives in *reference* pixels while the stamp is on its own native grid, so
`r_stamp = r_ref · pscale_ref/pscale_stamp`. In upsample mode `wcs[i] = wcs[0]`, so the
native scales must be captured before the fit loop — the same trap as the `PSFSZ<i>` /
`RCIRC<i>` bug already in `TODO.md`.

### 2.7 The EE bounds

No real source is more concentrated than a point source, so

```
apF_corr <- min(apF_corr, EE_psf_hi(R_cat))
apB_corr <- min(apB_corr, EE_psf_lo(R_img))
```

must hold. A truncated compact template has `apF_book = 1`, which the bound catches
immediately; it removes the observed 40× correction tail without touching any template.
Both numbers come from `get_ee_at`. Write the clip rate as a column — it is a direct
template-quality diagnostic.

### 2.8 Residual region

All four estimators take `res_sum = res_ap_<i>`, the residual in `disk(R_img)` with other
sources' segment pixels zeroed (`_other_source_mask`, OR-reduced to the residual
resolution for `k > 1`). This is wren's definition and it is the right one for `est4`,
whose target is an aperture-matched catalogue quantity.

Also record `res_seg_<i>` (over the dilated segment actually used for extraction, not the
raw catalogue segment) and `res_omega_<i>` (their union), so the bias-against-variance
trade stays measurable downstream instead of baked in. A total wants `Ω` or larger; an
aperture-matched quantity wants the disk; using `Ω` for both silently makes the matched
quantity not matched.

### 2.9 Column contract

Per band `i`, unless marked band-independent:

| column | meaning | status |
|---|---|---|
| `flux_<i>`, `err_<i>`, `err_pred_<i>` | fitted amplitude and errors | exists |
| `flux_<i>_total`, `err_<i>_total`, `err_pred_<i>_total` | `A/S_lo`, summed per template | exists |
| `throughput_<i>` | the *filter-level* mean stamp sum, constant per row — **not** the per-source `S_lo` the `_total` columns used | exists |
| `ap_model_<i>`, `ap_flux_<i>`, `ap_flux_corr_<i>` | `A·ap_lo`; the neighbour-subtracted aperture sum; and their product with `ap_corr_<i>`, which is Estimator 1 | exists |
| `apcor1_<i>`, `totcor1_<i>` | `apF_corr/apB_corr`, `1/apB_corr` | replaces `ap_corr_<i>` |
| `apF_book`, `apB_book_<i>`, `apF_blank`, `apB_blank_<i>` | raw and blanking terms | new |
| `clip_apF`, `clip_apB_<i>` | EE bound engaged | new |
| `res_ap_<i>`, `res_seg_<i>`, `res_omega_<i>`, `err_res_ap_<i>` | residual regions | new |
| `ee_tmpl_<i>` | `1 − wing_frac_lost` | exists on `template` |
| `template_norm`, `model_total`, `fpsf`, `snr_seg`, `w_core`, `ee_psf_hi` *(band-indep.)* | template bookkeeping | partly on `template` |
| `kron_flux_model`, `r_kron_model`, `ee_kron_int`, `ee_kron_cat`, `ktot`, `tcor_int`, `s_cat` *(band-indep.)* | model-Kron chain | new |
| `ap_flux_est1_<i>` … `ap_flux_est4_<i>` and `err_ap_flux_est1_<i>` … | the four estimators | new |

Errors follow wren's rule, which is worth keeping and stating: the fractional profile-fit
error `err_<i>/flux_<i>` is applied to the corrected **model** term only. The
multiplicative corrections are treated as noiseless and the residual pixel noise is
deliberately excluded, because it is correlated inside the aperture and drizzle
correlation between native pixels is not captured by per-pixel propagation. So
`err_est1 = err_est2 = |totcor1·ap_model|·frac_err`, and analogously for `est3`/`est4`.
Record the residual's formal error separately (`err_res_ap_<i>`) as a documented lower
bound. If `est1` is reported at all, its error must also be scaled by `totcor1`.

`apF_book`, `ktot`, `tcor_int` and `s_cat` are F444W-side and band-independent. wren
recomputed them per band; compute once.

---

### 2.10 Bookkeeping invariant and comparison axes

Two rules from wren's §5.3 and §5.5 that constrain the port and the diagnostics.

**`ap_model` comes from the fitted template, for every source.** No substitution of a
curve-of-growth or PSF-path value for the PSF-converged population — the aperture
fraction must be the one the fit actually used, so `ap_flux = ap_model + res_sum` holds
by linearity. `main` already satisfies this: `ap_model = fl · den` with
`den = _aperture_sum_on_template(tmpl, r_img_pix)` on the fitted convolved template
(`pipeline.py:1464-1466`), and the audit confirms this invariant was never broken here.
The only defect is the frame mismatch of §4.4. Curve-of-growth values stay available for
`totcor1`/`apcor1`; they do not enter `ap_model`.

**No catalogue-tied quantity on an IDL axis.** `est1` is the exact-parity IDL comparison,
`est2` the secondary; `est4` appears **only** against the SUPER catalogue. Do **not**
compare anything against IDL `flux_Ff444w`: that column is the raw 0.35"-radius aperture
flux on the detection image, zeropoint-scaled only (`dophot.pro:733,737-738,773`), so a
total-vs-`flux_Ff444w` panel is a convention mismatch by construction — measured
`f_f444w/flux_Ff444w = tot_cor·aper(use_aper/2)/aper(0.35")` per source, 0.992±0.017.
Omitted from the comparison set.
Mixing them is how a catalogue total-flux convention silently becomes an IDL residual.
`main`'s comparison scripts are `scratch/wren/make_compare.py` and
`make_compare_subphot.py` (wren's `compare_ap_mophongo.py` does not exist here); the
panels to add are `totcorf* vs totcor1_<i>`, a new `psfcorf* vs apcor1_<i>`, `est1` (and
optionally `est2`) on the flux panels. The previously planned `flux_Ff444w`-vs-`ktot`
panel is dropped: `flux_Ff444w` is not a total (see above).

---

## 3. Sequencing

### PR-0 — land the `template` branch (prerequisite)

Everything downstream assumes `extend_mode`, `template_norm`, `fpsf` and the renumbered
flag bits. `git merge-tree main template` conflicts in only two paths, `STATUS.md` and
`src/mophongo/templates.py`; `pipeline.py`, `utils.py`, `psf_map.py` and
`verification.py` auto-merge.

Auto-merging is not the same as merging correctly. Verify by hand:

- `Pipeline.run` must call `extract_templates` with `extend_mode`/`detection_psf`/
  `detection_weight` (from `template`) **and** `convolve_templates(..., psf_lo=prm_lo)`
  (from `main`, `pipeline.py:1697`). `template` predates the `psf_lo` argument.
- `psf_map.py`: `refresh_ee`/`ee_box`/`ee_rlim`/`get_ee_box`/`get_ee_rlim` are
  `main`-only (86 lines) and `template` deletes nothing there; confirm they survive.
- `templates.py`: keep `main`'s `convolve_cutout`, keep `template`'s
  `extract_templates`, and carry `ee_tmpl`, `ee_psf_lo` **and** `template_norm` through
  `convolve_cutout`/`downsample`.
- `ee_tmpl` and `wing_frac_lost` measure the same thing from opposite ends
  (`ee_tmpl = 1 − wing_frac_lost` for the prenormalised schemes). Keep `ee_tmpl` and have
  `composite_psf_wings` populate it.
- Wire the high-resolution PSF map into the config path. With `psfs=[None, prm_lo]` the
  inline extension branch at `pipeline.py:1588` falls back to `psfs[1]` — the
  *low-resolution* map — to build high-resolution wings. (There is no function named
  `_psf_for_template_extension` on main; the logic is inline.) §2.2 and §2.5 both need
  `S_hi` per region, and `self.prm_hi` is `None` on any cached rerun because `_ensure_maps`
  reloads only the low-resolution map.

Gate: full `poetry run pytest`, with `test_template_schemes.py` and
`test_template_convolution.py` green together.

### PR-0b — blockers the audit exposed

An exhaustive audit of `main` (`docs`/`scratch/wren/flux_estimator_comparison.pdf` §7,
50 agents, 751 claims re-checked at their cited `file:line`) found six preconditions that
must hold before any estimator column is meaningful. None was visible from the fork audit.

| # | blocker | site |
|---|---|---|
| 1 | **`ee_psf_lo` is destroyed in the default multi-resolution path.** `multi_resolution_method` defaults to `"upsample"`; at `k>1` `Pipeline.run` calls `convolve_templates` (which sets `ee_psf_lo`) and *then* rebuilds every template through `project_to_block_replicated_grid`, which copies only `flag`, `deblend_parent_label`, `deblend_nchildren`. Confirmed by execution: `0.917` in, `nan` out. Every source falls back to the filter-level mean, so `flux_<i>_total` silently reverts to pre-encircled-energy behaviour. MINERVA (40 mas ref, 80 mas MIRI, `k=2`) is exactly this case. | `templates.py:746-748`, `pipeline.py:1700-1704` |
| 2 | **Templates have no wings in a config-driven run.** `extend_templates` is a `Pipeline.__init__` keyword defaulting to `None`, is not a `RunConfig` or `FitConfig` field, and `load_data` never passes it. `template_norm`, `ap_hi` and `model_total` have no meaning until this is wired. | `pipeline.py:448` |
| 3 | **`S_hi` is unreachable.** `self.prm_hi` is `None` and `_ensure_maps` reloads only the low-resolution map. | `pipeline.py:831` |
| 4 | **The catalogue columns are dropped before fitting.** `run()` keeps `id`, `x`, `y` plus `is_deblended`/`deblend_*`, `FLAG_SATURATED_*`, and the one column named by `aperture_catalog`. `fauto_KRON`, `faper_KRON`, `tot_cor`, `kron_radius_circ`, `use_aper`, `f_f444w`, `ra`, `dec` never reach the estimator code, and `RunConfig` has no `f444w_*` fields. | `pipeline.py:1537-1547` |
| 5 | **`R_cat` is dead code.** `_resolve_catalog_ap_radius_pix` has zero callers in `src/`, `tests/` or `examples/`; `aperture_catalog` only names a column to copy through. | `pipeline.py:1338` |
| 6 | **The aperture and total families are on different absolute scales.** `ap_flux_corr_<i>` is never divided by `ee_psf_lo`, so for an isolated point source it converges to `flux_<i>`, not `flux_<i>_total` — they differ by exactly `1/S_lo`. | `pipeline.py:1475` |

Two further defects worth folding in here because they are cheap:

- **Operator-precedence bug in the weight mask.** `fit.py:269` and `scene.py:1192` both write
  `model[self.weights <= 0 | np.isnan(self.weights)] = 0.0`. `|` binds tighter than `<=`, so
  this evaluates as `weights <= (0 | isnan(weights))` and NaN-weight pixels are never masked.
- **`generate_scenes` cannot be called with its own defaults.** `minimum_bright` defaults to
  `None` and reaches a `<` comparison, raising `TypeError`. Only `Pipeline.run`, which always
  passes a value, works.

Also corrected against this audit: `main` now implements Estimator 1 *in full* —
`ap_corr_<i> = sum(T_conv)/aper(T_conv, R_img) = 1/ap_lo`, which is `totcor1`. The
`apF`-for-post-conv-total substitution that caused a 1.2 mag offset is gone, so the earlier
reading that `main` "stopped one factor short of a total" no longer holds.

### PR-1 — free fixes, no dependencies

| # | change | site |
|---|---|---|
| 1 | Relative flux ridge (§5) | `scene_fitter.py:178-181` |
| 2 | `np.searchsorted` for the segment bbox lookup (§4.2) | `templates.py:1395` (`template`: `:1538`) |
| 3 | Aperture geometry consistency (§4.4) | `pipeline.py:1392-1398` |
| 4 | `PSFRegionMap.resolve_key`, and route `get_psf`/`get_ee_box`/`get_ee_rlim` through it | `psf_map.py` |
| 5 | Delete `utils.rebin_wcs` (`NameError` on any call: `factor = 2**n`, `n` unbound) and `catalog.calibrate_ivar_with_bg_median` (`UnboundLocalError`: `bgmask` read at `:271`, bound at `:307`; sole call site `:673` already commented out) | `utils.py:123`, `catalog.py:221` |
| 6 | pytest marker taxonomy (`slow`, `network`, `needs_data`, `benchmark`) with `addopts` deselecting all four, so a bare `pytest` runs only fast offline tests | `pyproject.toml` |
| 7 | Flag-bit regression test: all eight bits distinct, `FLAG_DEBLENDED == 0x40`, `FLAG_SATURATED == 0x80` | `tests/test_template_schemes.py` |
| 8 | Capture native pixel scales before the fit loop, fixing `PSFSZ<i>`/`RCIRC<i>` in upsample mode — the same trap §2.6 must avoid | `pipeline.py` |

### PR-2 — test harvest (§6)

### PR-3 — performance (§4.1, §4.2)

### PR-4 — growth curve, residual regions, `est1` and `est2`

1. `PSFRegionMap.refresh_cog`/`get_ee_at` (§2.6), and the 288k-row `ee_kron_cat`
   validation of it (§2.4a). This gate runs before anything depends on the curve.
2. `_other_source_mask`: zero other sources' segment pixels in the residual patch,
   OR-reduced to the residual resolution when `k > 1`. Write `res_ap_<i>`,
   `res_seg_<i>`, `res_omega_<i>`, `err_res_ap_<i>`.
3. `apF_book`/`apB_book_<i>` and the blanking terms `apF_blank`/`apB_blank_<i>`; then
   `apcor1_<i>`/`totcor1_<i>` with the `S_hi`/`S_lo` translation and the EE bounds
   (§2.2, §2.7), with clip-rate columns. `ap_corr_<i>` retires in favour of
   `totcor1_<i>`.
4. `ap_flux_est1_<i>` and `ap_flux_est2_<i>` with their errors.
5. **The bridge test**: `ap_flux_est2_<i> == flux_<i>_total + res_ap_<i>` for an isolated
   point source under `psf_wings`. If this does not hold to ~1e-6, the `S_hi`/`S_lo`
   translation is wrong and nothing downstream is trustworthy.

### PR-5 — the model-Kron chain, `est3`

1. `model_kron` on the reference model stamp, with the invariants of §2.5: circular flux
   replaces the elliptical one when the cap engages, 0.25 px radius quantisation,
   and the documented fallbacks for degenerate moments, an empty segment or any
   photutils exception.
2. `model_total`, `ee_kron_int`, `ktot = kron_flux_model/ee_kron_int`,
   `tcor_int = ktot/(template_norm·apF_book)` — **no `S_hi`**, it is already in the
   denominator (§2.2) — and `ap_flux_est3_<i>`.
3. The `apcor_from_psf` shortcut as a performance gate, pinned against the full path.
4. Config surface: `f444w_aper_col` (colour aperture, drives `r_floor`).
5. Loud-failure discipline. Keep the shape of wren's three named guards: a per-band
   `aperture_diam` array, a PSF map with no growth curve, and no usable
   `f444w_aper_col`. Degrading to `tcor_int = 1/(apF_book·S_hi)` is acceptable;
   degrading silently is not.

### PR-6 — the catalogue tie, `est4`

1. `ee_kron_cat = (fauto_KRON/faper_KRON)/tot_cor` per source, guarded on all three
   columns positive and finite. Fit it once per run as a monotone function of
   `kron_radius_circ` and keep the residual as a data-quality column — it is a global
   curve to 0.05 % (§2.5a), so a source far off it is telling you something.
2. `ktot_cat = kron_flux_model/ee_kron_cat` alongside `ktot`, so `s_cat` factorises
   without a second Kron measurement.
3. `s_cat = f_f444w/ktot`, positive-guarded, not top-clipped, plus its factors
   `fauto_KRON/kron_flux_model` (= `f_f444w/ktot_cat`) and `ee_kron_int/ee_kron_cat` as
   separate columns (§2.4b, §2.5a).
4. `ap_flux_est4_<i>` and its error, by both routes — through `s_cat`, and through
   `est3·(ee_kron_int/ee_kron_cat)·(fauto_KRON/kron_flux_model)`, which never divides by
   a noisy catalogue total. Report the ratio.
5. Config surface: `f444w_col` (catalogue total), `f444w_auto_col`, `f444w_aperkron_col`,
   `f444w_totcor_col`. Note this reinstates `f444w_totcor_col`, which the audit lists as
   dead on wren — here it is live, because `ee_kron_cat` needs it.
6. Assert the aperpy `KERNEL` metadata matches the high-resolution filter. `apcor1_<i>`
   is only valid because `KERNEL == F444W == sci_hi` today.
7. Move the band-independent quantities into the catalogue-augmenting preprocessing step
   already scoped in `docs/PHOTOMETRY_APERTURES.md` §7.

### PR-7 — low-SNR robustness and acceptance

`composite_psf_wings` already blends toward the scaled PSF with
`w = blend_weight(snr_seg, snrlo_psf, blend_p)`. Outstanding: decide `S0` and the exponent
on injected-truth mocks rather than inheriting IDL's 10, promote `snr_seg` and `w_core` to
catalogue columns, and confirm the engagement profile (a quadratic weight still carries a
20 % PSF fraction at `S = 2·S0`).

---

## 4. Performance on 345k-source fields

### 4.1 `_sources_with_coverage` — pre-extraction cut

Before extraction, keep only catalogue rows whose template footprint can touch positive
weight in **any** fitted band. Measured on UDS: 196–203k of 345,792 sources never built
(~57 %, ceiling 67 % at 33 % MIRI coverage).

Implementation constraints that make it correct rather than merely fast:

- The footprint is the segment bbox floored at `min_size//2`, **per source**. A single
  global radius is wrong: one 3829-px star halo in UDS dilates the coverage mask until
  nothing is cut.
- Coverage is tested in each band's frame after a WCS round-trip with a +1 px margin. The
  common case is one lookup into `maximum_filter(w > 0, size=2·r0+1)`; only the rare
  oversized segments get their own box tested.
- It must mirror `Templates.prune_outside_weight`, which still does the exact per-band cut
  afterwards, so the surviving set is unchanged. This is a rough cut, not a new selection
  rule.
- The catalogue is **not** filtered. Cut sources keep `bad_value` rows.

Port target: a module-level helper in `pipeline.py`, called immediately before
`Templates.from_image`. Source: `wren:pipeline.py:57-148`, call site `:1499-1507`.

### 4.2 `np.searchsorted` for the segment bbox lookup

`SegmentationImage.get_index(label)` calls `check_labels`, which scans every label on
every call: 3.894 → 0.053 ms/source at 345k sources, i.e. ~22 minutes of pure lookup.
`get_index` itself is already a `searchsorted`; the cost is entirely the validation. Hoist
`segm.labels` and `segm.bbox` out of the sizing loop and index directly.

### 4.3 ROI-restricted ownership

Already on the `template` branch as `template_schemes.cutout_roi` +
`build_ownership(roi=..., roi_step=8)`, and only active for `extend_mode='wren'`
(`psf_wings` masks by segment ownership, not by an ownership map). The restricted result
is identical inside the ROI rather than approximate, because a label writes only inside
its bbox-padded-by-radius window. No further work.

### 4.4 Aperture geometry consistency

`_aperture_sum_on_template` (`pipeline.py:1392-1398`) integrates the **whole** `tmpl.data`
with the cutout-frame shift commented out, while the residual patch in
`_add_aperture_photometry` (`:1442-1447`) uses `tmpl.data[tmpl.slices_cutout]` with the
shift applied. Each is self-consistent in its own frame, but for an edge-clipped template
`den` includes model flux outside the image that the residual aperture cannot see, so
`ap_flux = ap_model + res_sum` stops holding by linearity. Measure on
`tmpl.data[tmpl.slices_cutout]` with the position shifted into that frame.

---

## 5. The flux-block ridge is still live

The audit records this as a defect documented nowhere on `main`. It is still accurate:
`main` has not fixed it, and the `flux-bug` work fixed a different, larger term.

`docs/FLUXBUG.md` records that `SceneFitter.solve` was adding `config.reg_astrom`
(default `1e-4`) to the photometric normal matrix. That was removed. What remains at
`scene_fitter.py:178-181` is

```python
scale_A = _positive_diagonal_scale(A)          # median of the positive diagonal
lam_A = reg_flux if reg_flux > 0 else 1e-6 * scale_A
Areg = A + sp.eye(A.shape[0], format="csr") * lam_A
```

which is exactly the configuration wren measured: one absolute value per scene applied to
every source, so recovered flux goes as `d_i/(d_i + λ)` — −0.05 % at `d_i/median = 1e-3`,
−0.5 % at `1e-4`, −4.8 % at `1e-5`, −33 % at `1e-6`. The ridge is added *before* whitening
(`solve_flux` computes `d = sqrt(diag(Areg))` at `:207`), so whitening does not undo it.

It is quiet today because truncated templates keep `ΣT²` concentrated. It stops being
quiet at exactly the point this merge path reaches: extended composites spread `ΣT²` thin
enough to reach those ratios. **This must land before PR-4.**

Fix: make the ridge relative per column, which is the same thing as adding it in the
whitened space.

```python
d2 = np.asarray(A.diagonal(), dtype=float)
d2 = np.where(np.isfinite(d2) & (d2 > 0), d2, _positive_diagonal_scale(A))
lam_rel = reg_flux if reg_flux > 0 else 1e-6
Areg = A + sp.diags(lam_rel * d2, format="csr")
```

For an isolated source this shrinks the flux by `1/(1+λ)` — a global 1e-6 scale identical
for every source, not a differential bias — and it removes the dependence on `d_i/median`
entirely. `reg_flux` changes meaning from absolute to relative; its docstring and
`FitConfig` comment must say so.

Add the guard that protected against a real past bias (commit `9d2ed2d`): sweep
`d_i/median` from `1e-3` to `1e-7` and assert recovery within `1e-4`. Note `tests/test_fit.py`
does still exist on `main` — it holds a single test, `test_flux_and_rms_estimation`, and no
diagonal-scale sweep — so this is a new test, not a restoration.

The other `CHECKLIST.md` finding — run-to-run non-reproducibility of scene partitioning,
`5/5/6` scenes from identical inputs, ~6 % of SNR>10 sources shifting by more than 1σ — is
also live and is already carried in `TODO.md`. Suspected mechanism:
`scene_coupling_thresh` is a hard cut on ATA couplings, so roundoff-level variation from
multithreaded FFT/BLAS reduction order flips couplings near the cut.

---

## 6. What to take from the 3100 lines of wren tests and docs

### `tests/test_pipeline_aperture.py` (1476 lines, 29 tests)

With the Kron chain retained, most of this file ports. Adapt rather than copy: the
symbols change (`model_total` for `trunc_denom`, `get_ee_at` for the containment-scaled
lookup) but the assertions do not.

| wren test | adapted to |
|---|---|
| `test_ap_flux_est1` | `ap_flux_corr_<i>` + its new error column |
| `test_truncation_cancels_in_apcor1_survives_in_totcor1` | `trunc = template_norm/model_total`, `rel=1e-10` both ways |
| `test_bookkeeping_invariant_to_truncation_term` | `template_norm·H == composite` exactly |
| `test_totcor1_faint_limit_matches_true_total_psf_ee` (1- and 2-band) | faint-limit true-total identity with a real Tukey-windowed matching kernel |
| `test_tcor_int_kron_construction_with_catalog_aperture` | `model_kron` with `r_floor = use_aper/2` |
| `test_tcor_int_kron_cap_shares_radius` | on cap, circular flux replaces elliptical at the same radius |
| `test_tcor_int_apcor_from_psf_shortcut_uses_floor_circle` | shortcut agrees with the full path on the converged population |
| `test_tcor_int_noisy_faint_stamp_finite_positive` | finite positive `tcor_int` on a noisy faint stamp |
| `test_tcor_int_fallback_no_catalog_aperture_column` | degrades to `tcor_int = 1/apF_book`, warned once |
| `test_s_cat_requires_positive_catalog_total` | positive-guard on `f_f444w` |
| `test_est3cat_crowding_flatness`, `test_est3int_and_f444w_ktot_crowding_flatness`, `test_crowding_regression_totcor1_matches_isolated` | `est4`, `est3`/`ktot` and `totcor1` flat across `[<0.6″ \| 0.6–1.2″ \| 1.2–2.4″ \| isolated]` |
| `test_est3_isolated_invariance`, `test_est3_faint_limit_identity`, `test_est1_est2_unchanged_by_stage4c` | unchanged in intent, renamed to `est3`/`est4` |
| `test_asrc_invariance_of_totcor1_with_data_core_and_crowded_wings` | `template_norm` invariance under crowded wings |
| `test_residual_segmap_sum_same_res`, `..._multi_res` | `res_seg_<i>` at `k=1` and `k>1` |
| `test_estimator3_uses_aperture_residual_not_segmap` | all four estimators use `res_ap_<i>`; `res_seg_<i>`/`res_omega_<i>` are recorded, not consumed (§2.8) |
| `test_psf_ee_cache_keys_on_region_not_psf_id` | keyed on `resolve_key`, not on `id(psf)` |
| `test_band_ee_uses_native_pixel_scale_not_fit_grid` | §2.6 native-scale trap / PR-1 #8 |
| `test_apcor1_gains_band_containment_ratio` | **drop.** wren's `c_det/c_b` band ratio exists because its stamps are unit-normalised; `main`'s are absolute. |
| `test_fit_invariant_to_containment_perturbation` | **drop** with `containment` |
| `test_aperture_photometry_with_tcor`, `test_tcor_lowsnr_psf_kwarg_removed` | **drop** — `tcor_H` machinery, removed on wren too |

Several are written as revert-verify regressions with measured acceptance numbers — they
must fail against the pre-fix code. Keep that property when adapting.

### `tests/test_template_extension.py` (585 lines, 26 tests)

Largely superseded by `template`'s `tests/test_template_schemes.py` (605 lines). Diff the
two and take what is missing; at minimum the flag-bit distinctness test, `n_pix` survival
through convolution including `FLAG_SUM_ZERO`, `min_size` semantics, and the
NaN-pixels-take-the-model test.

### wren's own §5.6 acceptance list, translated

| wren test | on `main` |
|---|---|
| unified template: faint limit is the PSF bit-for-bit; bright-compact limit is segment data + PSF wings; halo blend monotone in halo SNR; renormalisation preserved | already covered by `template`'s `tests/test_template_schemes.py`; check coverage rather than re-porting |
| `containment` geojson round-trip; `totcor1 == 1/(EE_stamp·containment)`; absent column defaults to 1.0 bit-for-bit with a warning | becomes `totcor1 == 1/(apB_book·S_lo)`, with `S_lo = ee_psf_lo`; plus the `refresh_cog`/`get_ee_at` round-trip and its missing-data default |
| bookkeeping invariant: `ap_model`/`ap_flux` from the fitted template's own fraction, regression against a pre-Phase-A reference | `main` already satisfies it (§2.10); the regression becomes `ap_flux == ap_model + res_ap` to machine precision, including edge-clipped templates once §4.4 lands |
| `ap_flux_est1 == (ap_model + res_sum)·totcor1` | verbatim |
| `tcor_int` positive and bounded on a noise-only source | verbatim |
| `s_cat` reproduces a known injected total ratio | verbatim, plus the three-way factorisation of §2.5a |

Add one wren did not have, and it is the most valuable in the set: the bridge identity
`ap_flux_est2_<i> == flux_<i>_total + res_ap_<i>` for an isolated point source under
`psf_wings` (§2.2).

### Take as-is

- `tests/test_downsample_flux.py` (49 lines) — `main` has both targets and **zero**
  downsample coverage anywhere in `tests/`.
- `tests/test_utils.py`, the `bin_factor_from_wcs` half — exact integers for 2×/4×,
  `ValueError` on 1.5×. Silent mis-binning corrupts every multi-resolution flux, and
  `main` has the function in `src` and no test.
- `tests/test_scene.py` partition test. **Drop its star test**: wren hard-codes
  `& ~is_star`, `main` makes it opt-in via `astrom_exclude_stars` and default-off, so the
  test asserts the opposite of `main`'s default.
- `tests/test_psf_map.py`: the `resolve_key` test, and the three base tests `main` dropped
  (region count, no tiny regions, `from_file`). The containment tests become
  `refresh_ee`/`refresh_cog` round-trip and missing-column-default tests.
- The `tests/test_pipeline.py` keystone assertions: background (`segmap == 0`) chi² per
  pixel between 0.85 and 1.15 against the known ivar maps, and fewer than 10 % of segments
  with `|residual sum|/σ > 5` in the independent low-resolution band. `main` currently
  asserts only `mean(ratio_err) ≈ 1 ± 3`.
- The end-to-end astrometry wiring test: inject a known (0.6, −0.5) px offset, run with
  `niter=2` order-0, and assert on the accumulated `Template.shifted` rather than on the
  residual — a broad low-resolution PSF makes the residual nearly insensitive to a
  sub-pixel shift, so only the accumulated shift proves the pipeline both computed *and*
  applied it.

### Docs

`docs/aperture_corrections_wren.md` (602 lines) is already in the tree and is now the
reference for the retained Kron chain, not a superseded design. Fold its IDL reference
recipe, its published-recipe citations (Wuyts 2008 Eq. 6, Skelton 2014 Eq. 1, Weibel
2024) and its acceptance criteria into `docs/FLUX_ESTIMATORS.md`. `main`'s
`PHOTOMETRY_APERTURES.md` §4 is stale against `main`'s own code (it says
`num = aperture_sum(T_ref, r_cat)`; `pipeline.py:1461` uses `tmpl.data.sum()`) and needs
the same pass. `docs/stage4c_scope_and_brief.md` is worth porting as a format: a
cold-start implementer brief with the algebra, numbered rulings, the exact code change,
and acceptance tests each required to fail against the unfixed code.

---

## 7. Acceptance gates

From `docs/aperture_corrections_wren.md` §1.11, on the real F1500W UDS run. They were
measured against wren's chain, so they are targets to reproduce, not invariants:

- PSF-limit aperture-to-total: 1.233 → 1.347, against IDL's 1.357 (< 1 %).
- Raw-flux offset −0.075 mag → target |μ| ≤ 0.02.
- `totcor1` crowding bins: 1.40/1.24/1.07/1.0 → 0.99/0.99/1.00/1.0.
- `est4/est1` crowding bins (wren's `est3cat/est1`): 1.49/1.19/1.07/1.0 → flat.
- Known open, left visible on purpose: ~3 % F444W EE residual; ~10–16 % catalogue-vs-IDL
  F444W total-definition offset.

New gates specific to the model-Kron path:

- `tcor_int` finite and positive for **every** source, including the 57,622 UDS rows with
  no catalogue `kron_radius`.
- `tcor_int` versus catalogue `tot_cor`: agreement improving monotonically with
  `f_f444w/e_f444w`, and `NMAD(tcor_int)` in the SNR 0–3 bin well below the catalogue's
  measured 1.67.

Per project policy every one of these is validated on injected-truth mocks first. The
real-photometry comparison is a separate exercise and does not gate this work.

---

## 8. Explicitly not ported

wren's diagonal-only flux errors (`main`'s marginalised `sqrt(diag(A⁻¹))` with
off-diagonal covariance is better); wren's `Scene._overlaps` deletion (`main`'s
`create_scene_graph` calls it); wren's `convolve_cutout`; `PSFRegionMap.containment` and
the `c_det/c_b` band ratio that goes with it (both are consequences of unit-normalised
stamps); the `tcor_H` measured-blend machinery (removed on wren too, and its ruling —
corrections come from models only, the only measured quantities are fluxes in apertures
matched to the source size — is adopted); and wren's dead ends: `_intersect_slices`,
`_get_representative_kernel`, `f444w_totcor_col`, and the unconditional CWD-relative
`f444w_template_residual.fits` write.
