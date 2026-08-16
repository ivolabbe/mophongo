# Fork comparison: `wrensuess/mophongo@dev-wren` vs `ivolabbe/mophongo@flux-bug`

Scope: **active code paths only.** Two entry points define "active":

- **A (ours)** — `examples/run_uds_770_dr0.1.py` + `examples/uds_770_dr0.1.json`
  → `Pipeline.from_config` → `build_psfs` → `build_kernels` → `run` → `write_outputs`.
  (`cosmos_770_dr0.1.json` and `uds_770_dr0.json` are the same code path with
  different inputs; `uds_770_dr0.json` additionally uses saturation-repaired
  products.)
- **B (wren)** — `examples/run_uds_770_wren.py` (a hand-rolled script that lives
  in our `examples/`, but resolves its `mophongo` imports against the fork).

Merge base `5ee1f4f`. Wren's branch also contains our `main` through
`80cad37`, so the divergence is `5ee1f4f..dev-wren` (25 commits) against
`5ee1f4f..flux-bug` (7 commits).

Line counts, active modules only:

| module | ours | wren |
|---|---|---|
| `pipeline.py` | 2134 | 1771 |
| `templates.py` | 1625 | 1456 |
| `psf.py` | 3144 | 1177 |
| `utils.py` | 2835 | 1859 |
| `fit.py` | 1675 | **135** |
| `scene.py` / `scene_fitter.py` | 1114 / 300 | 1070 / 260 |
| `psf_map.py` | 567 | 618 |

Modules that exist only in ours and are **not** on either active path:
`mock_mosaic.py`, `saturate.py`, `verification.py`, `psf_factory.py`,
`astro_fit.py`, `_mock_sip.py`. Wren deleted them (commit `8192f91`,
`846a022`); that deletion has no effect on path B.

---

## 1. Orchestration

Wren has **no config machinery**. There is no `RunConfig`, no `from_config`,
no `build_psfs`/`build_kernels`/`load_data`/`write_outputs`. The script does
all of it by hand: two `DrizzlePSF`, `PSFRegionMap.from_footprints(...)
.overlay_with(...)`, `get_psf_radec(pos, size=3.0)`, `psf_stamp_containment`,
`matching_kernel(..., recenter=True)`, `to_file`/`from_geojson`,
`get_bg_and_ivar`, `Pipeline(...).run(config)`, then `fits.writeto` /
`table.write` / a scene CSV.

Differences inside the equivalent operations:

| | ours | wren |
|---|---|---|
| PSF stamp size | `"psf_size": null` → full native stamp | `size=3.0` arcsec |
| PSF grids | MJD-tagged `UDS_*_MJD*_GRID*_OS4`, nearest-MJD per exposure | 2 static single-epoch grids (`UDS_NRCA5/B5_F444W_OS4_GRID25`, `UDS_MIRI_F770W_OS4_GRID9`) |
| lo-res PSF blur | `_drizzle_lo_blurred`, Gaussian FWHM 0.08″ (`DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC["f770w"]`) | none |
| kernel recentring | `matching_kernel(...)`, `recenter` defaults **False** | `recenter=True` explicitly |
| region maps | each map drizzled at **its own** centroids; kernel map re-drizzled at `prm_kern` centroids | all three cubes drizzled at `prm_kern` centroids, then assigned to maps that index by their **own** `psf_key` |
| template side | raw F444W mosaic | aperpy `_sci_f444w-matched` image |

**As checked in, `run_uds_770_wren.py:13` sets `miri_filt = "1800"`**, so its
`aperture_diam = 1.5″` and `scene_coupling_thresh = 0.02`. For F770W the same
script gives `0.7″` and `1e-3`. Ours uses `aperture_diam = 0.5″`,
`scene_coupling_thresh = 1e-3`.

---

## 2. `Pipeline.run()` — where the two sequences diverge

| step | ours | wren |
|---|---|---|
| load | `load_data()`: footprint cut, `r_trial=0.5′` patch, `get_bg_and_ivar(σ=64)`, NaN guard | script does it |
| native pixel scales | — | `self._native_pscale` captured **before** upsample clobbers `wcs[idx]` |
| PSF growth curve | — | widest-region `rep_psf`, `ee_radii_pix{0.5,0.95,0.99}`, `r_fill`, `min_size` |
| templates | `Templates()` (`min_size=8`), `extract_templates(dilate_segmap=2)` | `Templates(min_size≈62)`, `extract_templates(extend_mode='auto', …)` |
| tagging | `is_deblended`, `deblend_parent_label`, `deblend_nchildren`, `is_saturated` | `is_star` from catalog `flag_star` |
| extension | dispatch present but **not taken** (§4) | already done inside `extract_templates` |
| upsample ivar | `block_replicate(conserve_sum=False) * k²` | `block_replicate(w, k) * k²` with astropy's default `conserve_sum=True` |
| grid projection | `project_to_block_replicated_grid(k)` per template | **absent** |
| scenes | `generate_scenes(coupling=1e-3, minimum_bright=10)` | `generate_scenes(coupling=0.02, minimum_bright=5)` |
| astrometry | up to 5 passes, early stop at `max_step < 0.05` px | exactly 2 passes, no convergence test |
| totals | `_filter_psf_throughput` → `flux_<i>_total`; `_record_psf_ee` → `cat.meta` | none; instead the `est1/est2/est3int/est3cat` column family |
| apertures | 4 columns | 19 columns |

Two numerical consequences worth naming.

**(a) ivar normalization differs by `k² = 4`.** `astropy.nddata.block_replicate`
defaults to `conserve_sum=True`, so wren's `block_replicate(w, k) * k**2`
returns per-subpixel ivar equal to the native ivar, while ours returns `k²`
times that. Fitted fluxes are invariant to a global weight scale, but
`err_<i>` and `err_pred_<i>` in wren come out **2× larger**, and the scene
`snr_proxy = b/sqrt(diag A)` scales as `√w`, so at the same
`astrom_minimum_snr=15` wren flags **half** the SNR and therefore far fewer
`is_bright` templates. Different scene partition, different astrometry sample.

**(b) no block projection in wren.** Ours re-integrates each convolved template
over native `k×k` blocks and re-replicates, so the model has the same
piecewise-constant structure as the upsampled data. Wren fits an un-projected
fine-grid model against a piecewise-constant image; the residual then depends
on where the source sits inside the 80 mas pixel — a phase-dependent bias,
largest for compact sources.

---

## 3. Solver

Both trees converge on `Scene.solve` → `SceneFitter.solve`; ours still
*imports* `SparseFitter` but `run_scene_solver=True` means it is never
instantiated on path A. Both are on `spsolve` (wren `38c9917`, ours
`6ed937b`), and both carry the absolute-regularization fix (wren `9d2ed2d`).

Real differences:

| | ours | wren |
|---|---|---|
| `fit_astrometry_niter` | 5 (+ `astrom_shift_tol=0.05` early stop) | 2, fixed |
| `fit_astrometry_joint` | exists, set `true` | **removed** — `niter` is the only knob |
| astrometry source cut | `snr_proxy > 15` only | `snr_proxy > 15` **and** `not_star` **and** `_astrom_isolation_mask(≥0.7)` |
| `_flux_errors` | `sqrt(diag(A⁻¹))` — full covariance | `1/sqrt(diag(A))` — diagonal only |
| joint-branch error | `sqrt(diag(S⁻¹))` of the Schur complement | `1/sqrt(diag(S))` |
| flux ridge | `1e-6 · median(**positive** diag)` | `1e-6 · median(**all** diag)` |
| shift application | re-resamples from a pristine `_data_unshifted` with the accumulated shift | re-shifts the already-shifted array each pass (cubic smoothing compounds) |
| saturated isolation | singleton scene per `is_saturated` template | none |

Because `diag(A_w) ≡ 1` after whitening, wren's `_flux_errors` collapses to
`1/d` — the single-template error with all neighbour covariance discarded.
Ours is always ≥ wren's: for a two-source blend with normalized coupling ρ the
ratio is `1/√(1−ρ²)` — 1.40 at ρ=0.7, 2.29 at ρ=0.9. On upsampled MIRI
templates ρ is large for most sources, so the `err_<i>` columns differ by tens
of percent even where the fluxes agree.

Wren's `is_star` cut is **inert**: `pipeline.py:1414-1419` sets `is_star` on
the unconvolved templates, but `convolve_cutout` does not propagate it
(`wren templates.py:581-587`), and `scene.py:545,661` reads it on the
convolved list. `not_star` is all-True in the active path.

---

## 4. Template extension — the largest single difference

### Ours: there is none on path A

`load_data` constructs with `psfs=[None, prm_lo]` and never passes
`extend_templates`, which defaults to `None`; `RunConfig` has no field for it
and `from_json` rejects unknown keys. So the dispatch at
`pipeline.py:1125/1142/1157` falls through, and
`skip_template_extension_for_deblended` / `extend_wings_background_only` are
inert. The only live shaping knob is `template_dilate_segmap = 2`:
`skimage.morphology.disk(2)` (13 px in a 5×5) binary dilation of every segment
**into original background only**, before any cutout is taken. Contested
background goes to whichever label is processed last, i.e. the **highest
catalog id** (`catalog.py:372`).

Two extension routines exist but are unreachable:

- `extend_with_psf_wings` — convolves the *segment-masked template itself*
  with the local PSF and pastes into zero pixels. Self-convolution, not a PSF
  model: too broad for any resolved source, no SNR gate, no radial cap.
- `extend_with_psf_model` — grid search over 7 Gaussian widths (0–6 px), least
  squares on segment pixels, best-χ² model pasted outside. Morphology-aware,
  closest analogue to wren's `M`.

Even if enabled, `psfs[0] is None` would make `_psf_for_template_extension`
fall back to `psfs[1]` — the **F770W** PSF — and smear F444W templates with the
low-resolution PSF.

### Wren: on by default, `H = W·data + (1−W)·M`

`template_extend_mode='data'` is rewritten to `'auto'` by `__post_init__`; the
data/psf/hybrid three-way choice no longer exists.

Sizing, in `pipeline.py:1271-1363`:

```python
rep_psf  = widest region PSF by psf_ee_radius_pix(p, 0.95)     # f919f69 fix
r_fill   = max(R95(rep_psf), r_aperture + R95(|kernel|))
min_size = max(8, 2*ceil(r_fill) + 2)                          # always even
```

`psf_ee_radius_pix` normalizes by the **stamp's** sum, and it is handed the
drizzled 3″ region-map stamp, so `R95` is 16.5 px @40 mas, not the 21.6 px R95
of the parent grid. Measured kernel half-widths give `min_size ≈ 58 px` for
F770W and **≈96 px** for F1800W (the checked-in filter) — against ours' 8, a
pixel cost ratio of ~50× to ~144× per source.

Per source, in `templates.py:1052-1258`:

```python
w_core = blend_weight(snr_seg, 1.5*fit_snrlo_psf, p)      # onset at SNR 15
A_src  = Σ max(data,0)[own] / Σ psf_cut[own]              # point-source amplitude
M      = A_src * psf_cut                                   # unit-sum PSF, bilinear, sub-pixel centred
w_k    = blend_weight(snr_k, wings_snr_psf, p)             # per 0.15″ annulus, onset SNR 3
w_k    = np.minimum.accumulate([w_core, *w_k])[1:]         # monotone, seeded at w_core
H      = np.where(ext_psf, W*data + (1-W)*M, 0)
```

with `blend_weight(snr, thresh, p) = min(1, (max(snr,0)/thresh)**p)`, `p=2`.

Verified against the code, three docstring claims are **wrong**:

1. `ee_reach ≤ max_radius_pix` **always**, because `r_fill = max(R95, …) ≥ R95
   = ee_reach`. So `ext_psf ⊆ ext_data`, the composite support is `ext_psf`
   (≈16.6 px = 0.66″), and `ext_data` appears only in the two failure returns.
   The docstring's "beyond `max_radius_pix` but within the PSF reach, only the
   PSF model contributes" describes a regime that never occurs. The
   `r_orig + kernel_hw` margin only inflates `min_size` and the ownership disk.
2. `extend_template_ee` is documented as a "max template-size cap"; it is a
   **floor** on `min_size`. What it caps is the support of `H`.
3. `extend_template_segmap` and `extend_template_min_size_margin` are
   documented as live and are **read by nothing**. `min_size_from_aperture`
   (commit `276e52c`) has no caller — the sizing is inlined in `pipeline.py`.

Neighbours: extension never writes into another segment's pixels
(`own = seg==label`, `bg_owned = owned & (seg==0)`). Contested background is
partitioned by an area-weighted `owner_map` (fftconvolve with a disk of radius
`r_fill`, lowest label wins ties) — geometric, unlike ours' id-ordered
tie-break. But the halo is **raw mosaic data**, so a neighbour's PSF wings
inside this source's territory enter the template at `w ≈ 1`.

Failure path: `return data_f * ext_data` — raw data extended to `r_fill`
(≈2× the normal support), unblended, with `flux_beyond_* = 0`. A source that
fails the PSF lookup gets a **larger, noisier** template and **no** truncation
correction — the opposite of a conservative fallback.

`M` is always a point source. For a resolved galaxy `f_own_psf` (the PSF's
in-segment fraction) exceeds the galaxy's own, so `A_src` under-estimates the
total, and both the halo amplitude and `flux_beyond_stamp`/`flux_beyond_aper`
are too small.

Four things that shape the comparison and are easy to miss:

- **Wren applies no segmap dilation at all** — `extract_templates` has no such
  argument, and `own = seg_stamp == label` is the raw segmap. So `snr_seg`,
  `w_core`, `A_src`, `f_own_psf` and `n_pix` are all measured on the
  **undilated** segment, while ours' equivalent support is `disk(2)`-dilated.
- **The positivity clip sits before `template_norm`** (`templates.py:1366`,
  then `:1374`), so `template_norm = Σ max(H,0)`, not `Σ(H)`, and the
  `template_norm · H == composite` invariant holds only against the *clipped*
  composite. Wherever `W > 0` — bright cores and every annulus above SNR 3 —
  negative sky pixels are rectified over a support of π·16.5² ≈ 855 px, so
  `template_norm` is biased high and that propagates into `trunc` and
  `ap_b_corr`/`ap_f_corr`.
- **No detection-side background subtraction.** The script passes the raw
  F444W mosaic (only MIRI gets `sci_miri - bg_miri`), and the blend runs on
  raw `data_f` over that ~855 px halo. Neither `_region_snr` nor `A_src`
  removes a pedestal, so a sky offset enters `template_norm` and the template
  shape linearly with halo area. Ours' segment-only template (tens of px) does
  not incur this.
- **`bg_rms` is inactive in this run** — it is computed only when
  `detection_weight is None`, and the script passes `wht_444`. A region with no
  positive ivar therefore gets `noise = 0 → snr = 0 → w = 0`: full PSF model,
  no flag, no fallback.

### Knobs

| ours (`fit.py`) | default | live on A |
|---|---|---|
| `template_dilate_segmap` | 2 | **yes** |
| `skip_template_extension_for_deblended` | False | no |
| `extend_wings_background_only` | True | no |
| `Templates.min_size` (class attr) | 8 | yes |

| wren (`fit.py`) | default | live on B |
|---|---|---|
| `template_extend_mode` | `"auto"` | yes |
| `fit_snrlo_psf` | 10.0 (core onset = 15) | yes |
| `wings_snr_psf` | 3.0 | yes |
| `template_blend_p` | 2.0 | yes |
| `template_blend_annulus` | 0.15″ → 3.75 px | yes |
| `extend_template_ee` | 0.95 | yes |
| `extend_template_segmap` | False | **dead** |
| `extend_template_min_size_margin` | 1.5 | **dead** |

### Expected effect on the fitted flux

F444W true-total encircled energy at 40 mas: EE(3 px)=0.61, EE(5 px)=0.74,
EE(10 px)=0.89, EE(16.6 px)=0.93, EE(21.6 px)=0.95.

- **Bright isolated point source** — ours: dilated 2σ segment, r≈5 px, E≈0.74,
  fitted flux ≈8 % low. Wren: `w_core=1`, support to R95, E≈0.93–0.95, plus the
  `trunc` correction. Wren ≈6–8 % higher.
- **Faint source (SNR≈5)** — ours: identical treatment, plus a dilated ring of
  pure sky; with no positivity clip, negative pixels shrink the normalizer and
  rescale the template up. Wren: `w_core = (5/15)² = 0.111`, so the template is
  89 % PSF model — clean shape, small scatter. Wren higher and far less
  scattered; the cost is that a genuinely resolved faint source is replaced by
  a PSF.
- **Bright resolved galaxy** — agree to a few percent; ours' best case.
- **Blend** — ours: contested dilation strip goes to the higher catalog id (an
  id-ordered flux-ratio bias). Wren: territory goes by area, the halo absorbs
  the neighbour's real wings (both templates contain some of the same photons),
  and the ownership truncation inside the aperture is corrected explicitly by
  `flux_beyond_aper`. Ours has no crowding correction at all.

---

## 5. PSF encircled-energy bookkeeping

### The two chains

**Ours.** ePSF grid (absolute, `NORMALIZ='first'`) → load-time cosine edge
taper 4 native px, *not* renormalized → `eval_ePSF` → `adrizzle.do_driz`
(`wcslin_pscale = psf_wcs.pscale`), **no taper, no renormalization** → the
stamp sum *is* the finite-support throughput. `stamp_encircled_energy`
measures `ee_box`/`ee_circ`/`r_circ`/`r_ee` on the delivered cube;
`_record_realized_ee` caches them on `DrizzlePSF`; `_filter_psf_throughput`
takes the mean stamp sum of the lo-res cube; `flux_<i>_total = flux_<i> /
throughput`; `_record_psf_ee` writes `EEBOX<i>`/`EECIRC<i>`/`RCIRC<i>`/
`PSFSZ<i>` into `cat.meta`. One scalar per filter, per `GUIDE.md`.

**Wren.** Same grid → `eval_ePSF` → `do_driz` with **`wcslin_pscale = 1.0`**
(the drop scale is then in arcsec instead of the dimensionless ratio; the raw
output is low by `1/pscale²` ≈ 82 for MIRI) → radial `TukeyWindow(alpha=0.05)`
built on `int(sqrt((outwht>0).sum()))` and padded → **`scale = psf.sum() /
outsci.sum()`**, which pins the returned stamp to the sum of the *last*
contributing frame's native-sampled ePSF over its `(2·npix+1)²` window. The
`wcslin_pscale` error is entirely hidden by that renormalization. Separately,
`psf_stamp_containment` measures the parent grid's inscribed-disk fraction and
stores it per region on `PSFRegionMap.containment`, round-tripped through the
geojson. `_psf_ee` returns `EE_stamp(r) × containment`; `_psf_containment`
feeds `c_det`/`c_b`; `templates.py` uses `c_det` for `flux_beyond_stamp`.

### Measured, on the real UDS n3.0/m3.0 data, one position

Both trees run with `size=3.0″`:

| | ours | wren |
|---|---|---|
| F444W stamp | 76×76 (3.04″), full square, corners nonzero | 75×75 (3.00″), **apodized disk**, 65.9 % nonzero, support r = 1.369″ |
| F444W stamp sum | 0.95221 | 0.95506 |
| F770W stamp | 38×38 (3.04″), full square | 39×39 (3.12″), disk, 63.6 % nonzero, support r = 1.449″ |
| F770W stamp sum | 0.89538 | 0.89879 |

Truth from the parent OS4 grids (absolute, sum/16):

| | true flux in that support | stamp / truth |
|---|---|---|
| ours F444W 3.04″ box | 0.94621 | **1.0063** |
| ours F770W 3.04″ box | 0.90346 | **0.9911** (−0.41 % of which is the deliberate load-time taper) |
| wren F444W disk r=1.369″ | 0.93620 | **1.0201** |
| wren F770W disk r=1.449″ | 0.89274 | **1.0068** |

Wren's stamps are too bright because the renormalization sets an **apodized
disk** to the sum of a **square** native window.

### Wren's `containment` is measured against the wrong denominator

`psf_ee_at_radius` normalizes by `psf.sum()` of the array it is handed, and
`psf_stamp_containment` hands it the **parent grid**. So `containment` is the
fraction of the *parent stamp's* flux inside `r = W/2`, not the fraction of the
true total. The parent grids are themselves finite: F444W 260 px = 4.09″,
absolute EE 0.9569; F770W 292 px = 8.07″, absolute EE 0.9852.

Second error: the radius is `W/2`, but the Tukey window zeroes everything
beyond `≈0.95·(n−1)/2`. F444W: 1.50″ assumed vs 1.369″ actual.

Net multiplier applied to the EE that feeds every correction,
`containment / stamp_sum`:

| | containment as coded | stamp sum | multiplier vs true absolute EE |
|---|---|---|---|
| F444W (detection, `c_det`) | 0.9830 | 0.95506 | **+2.92 %** |
| F770W (band, `c_b`) | 0.9122 | 0.89879 | **+1.49 %** |

Note the grid provenance: wren's script loads the **static**
`UDS_*_OS4_GRID25/9` grids, which is what the containment numbers above are
measured from; ours loads the **MJD-tagged** grids. The two agree to 1e-4
(0.956872 vs 0.956940 F444W, 0.985221 vs 0.985238 F770W), so the comparison is
unaffected.

Propagating through the point-source algebra — exact only in the
`apcor_from_psf` / faint limit, where `Σ(H) = A_src·f_cut` so
`trunc_denom = A_src/c_det`. For a data-dominated template (`w_core → 1`)
`Σ(H)` is data-driven and the identity does not hold:

```
trunc_denom = template_norm + flux_beyond_stamp = A_src / c_det
apF_corr    = c_det · EE_det_stamp(r)     -> +2.9 %
apB_corr    = c_b   · EE_band_stamp(r)    -> +1.5 %
apcor1      = apF_corr / apB_corr         -> +1.4 %
totcor1     = 1 / apB_corr                -> -1.5 %   (est1, est2 read low)
tcor_int    = 1 / apF_corr                -> -2.9 %
```

The structure is right — containment enters **exactly once per side**, and
`trunc` cancels in `apcor1` by construction. The error is in the measurement,
not the algebra. The doc's own open item Sec 7 ("F444W EE residual ~3 %:
true-normalized `apcor1` 1.217 vs IDL `psfcor` 1.255") is the same order as the
parent-grid truncation this convention leaves in.

### Ours divides `flux_<i>` by the wrong stamp sum

Ours' `matching_kernel` deliberately does not normalize, so

```
Σk = S_lo / S_hi
```

(verified on the cached run: kernel median sum 1.00747, `S_lo/S_hi` = 1.00672).
For a point source the unit-sum detection template convolves to `psf_lo / S_hi`,
so the least-squares amplitude against absolutely calibrated data is

```
flux_<i> = A_true · S_hi          # the DETECTION stamp sum
```

Verified numerically on the cached region-0 stamps: fitted `a = 0.9557` against
`S_hi = 0.9606` (0.5 %, the residual being window/shape mismatch).

But `pipeline.py` divides by `throughput = mean(S_lo)`:

```
flux_<i>_total = flux_<i> / S_lo = A_true · S_hi / S_lo
```

The low-res throughput has **already cancelled inside the kernel DC**; dividing
by it again is a double correction, and the detection-side truncation `S_hi` is
never removed. The right divisor is `S_hi`.

The bias is small on path A only because `psf_size: null` leaves both stamps
near-native with similar EE (`S_hi = 0.9613`, `S_lo = 0.9678` → **−0.7 %**). It
is not small at other sizes:

| `psf_size` | S_hi (F444W) | S_lo (F770W) | `flux_total` bias |
|---|---|---|---|
| 2.0″ | 0.9242 | 0.8691 | **+6.3 %** |
| 3.0″ | 0.9458 | 0.9019 | **+4.9 %** |
| **4.0″ (RunConfig default)** | 0.9564 | 0.9275 | **+3.1 %** |
| 6.0″ | 0.9569 | 0.9627 | −0.6 % |
| 8.0″ | 0.9569 | 0.9850 | −2.9 % |

Two caveats on those numbers. They are the **point-source limit**: they assume
`T = P_hi/S_hi`, whereas path A's templates are segmap-truncated and
unit-normalized with no extension, so the amplitude is not `A_true·S_hi` for a
real source and the bias is source-dependent. And they are the *bookkeeping*
error alone — see the next section.

### `ee_box` is not a clean absolute EE

The premise that the drizzled stamp sum is directly the fraction of true total
captured does not survive measurement. Comparing each drizzled cube against the
**tapered** parent grid it was built from (the taper is applied at load,
`edge_taper_pixels=4.0`, and is deliberately not renormalized):

| | tapered parent total | drizzled stamp | ratio |
|---|---|---|---|
| F444W, 102² @0.04″ = 4.08″ | 0.95488 | 0.96131 | **1.0067** |
| F770W, 101² @0.08″ = 8.08″ | 0.98125 | 0.97008 (pre-blur) | **0.9886** |

The F444W stamp holds **more flux than the entire parent grid** while being
circularly apodized (11.8 % of `psf_hi[0]` is exactly zero), so the excess is
larger than 0.67 % against the support it actually covers. `DrizzlePSF.get_psf`
performs no renormalization, so this is a drizzle/resampling normalization
error of order ±1 % per band — the **same order as the 0.64 % mis-banding
above**, and it does not have a consistent sign. This is the open
`wcslin_pscale` item already in `TODO.md`; the two must be settled together,
because `EEBOX<i>` inherits it.

Wren does not have this problem: photutils' `create_matching_kernel`
normalizes both PSFs and the kernel, so `Σk = 1` and the convolved template
carries the **hi-res** stamp EE, which the `c_det`/`c_b` chain then corrects
explicitly.

### `PSFSZ<i>` and `RCIRC<i>` are half their true value

`pipeline.py:1207` does `wcs[ifilt] = wcs[0]` in the upsample branch, and
`wcs` is an alias of `self.wcs` (`:1053`). `_record_psf_ee` is called
afterwards (`:1384-1392`) with `self._pixel_scale_arcsec(self.wcs[ifilt])`,
which is now the 0.04″ reference grid, not the PSF stamp's native 0.08″.

Confirmed on the cached run: `PSFSZ1 = 4.04` and `RCIRC1 = 2.02` for a
101 px F770W stamp that is physically 8.08″ across with a 4.04″ inscribed
radius. `EEBOX1` and `EECIRC1` are unaffected — `i_circ` in
`stamp_encircled_energy` is scale-free, so `ee_circ` is identical either way.

Wren avoids the same trap by snapshotting `self._native_pscale` **before** the
fit loop (`pipeline.py:1260-1263`).

### Wren's floored `tcor_int` branch drops a factor of `trunc`

```python
# fallback (no r_floor)                       pipeline.py:1030
tcor_int = 1.0 / apF_corr                   # = 1/(apF_book · trunc)   for an isolated source
# floored + apcor_from_psf  (the production branch — the script sets
# f444w_aper_col='use_aper', and apcor_from_psf is True for snr_seg < 10)
denom      = trunc_denom * apF_book         # pipeline.py:1045
f444w_ktot = trunc_denom                    # pipeline.py:1061
tcor_int   = f444w_ktot / denom             # = 1/apF_book
```

The two differ by exactly `trunc`, though the code comment claims they are
"the same identity". The correct aperture-to-total is
`trunc_denom / (template_norm · apF_book) = 1/(trunc · apF_book)` — the
*fallback* form; the floored branch's denominator multiplies a true-total flux
by a stamp-and-ownership-normalized fraction. With `f_cut ≈ 0.95` and
`c_det = 0.983`, `trunc ≈ 0.934`, so `est3int` reads **≈6.6 % low** for the
floored (faint) population. `est3cat` is unaffected — `f444w_ktot` cancels in
`tcor_int · s_cat`.

### Side-by-side

| quantity | ours | wren |
|---|---|---|
| where measured | delivered drizzled stamp | parent ePSF grid |
| normalization | absolute (entrance-pupil unit) | relative to the parent grid's own sum |
| granularity | one scalar per filter | one scalar per PSF region, serialized to geojson |
| stored on | `DrizzlePSF.ee_box/ee_circ/r_circ/r_ee/psf_size`, `cat.meta` | `PSFRegionMap.containment` |
| applied to | `flux_<i>_total = flux_<i>/throughput` | `EE_true(r) = EE_stamp(r)·containment` inside `_psf_ee`; `c_b/c_det` on `apB_corr`; `1/c_det` in `flux_beyond_stamp` |
| detection-side truncation | **not corrected** | corrected per source via `flux_beyond_stamp` |
| crowding truncation | **not corrected** | corrected per source via `flux_beyond_aper` |
| kernel DC | carries `S_lo/S_hi` | forced to 1 |

---

## 6. Output columns

| ours only | wren only | shared |
|---|---|---|
| `flux_1_total`, `err_1_total`, `err_pred_1_total`, `throughput_1`, `ap_corr_1`, `ap_flux_corr_1`, meta `EEBOX1`/`PSFSZ1`/`EECIRC1`/`RCIRC1` | `apcor1_1`, `totcor1_1`, `apf_data_1`, `tcor_int_1`, `s_cat_1`, `f444w_ktot_1`, `res_sum_1`, `res_seg_1`, `apcor_1`, `ap_flux_est{1,2,3int,3cat}_1` + 4 matching `err_*` | `id`, `x`, `y`, `aper_1`, `flux_1`, `err_1`, `err_pred_1`, `ap_model_1`, `ap_flux_1` |

Names that collide but differ:

- `ap_flux_1` — ours: `aper(residual + model, r_img)`. Wren: `ap_model +
  res_sum` with other sources' segment pixels masked out of `res_sum`.
- `ap_model_1` — ours: the last component visited. Wren: summed over all
  components sharing a parent id.
- `ap_corr_1` (ours, a shape ratio `EE_F444W/EE_conv`) vs `apcor_1` (wren,
  the full catalog-tied `apcor1·tcor_int·s_cat`).
- `err_1` — same expression, 2× larger in wren (§2a).

---

## 7. Defects found in the fork's active path

1. **Kernel region lookup uses the wrong sky position.**
   `wren templates.py:1422-1428` feeds `tmpl.position_original` (mosaic frame)
   to `tmpl.wcs` (the `Cutout2D` CRPIX-shifted cutout frame). Measured on a
   4000×4000 40 mas mosaic with the source at (3000.3, 2500.7): correct
   `(34.377677, −5.123311)`, actual `(34.344313, −5.095631)` — **≈2′ off**,
   growing linearly with distance from CRPIX. On the real UDS mosaic that is
   tens of arcmin: outer-field sources fall outside every region and
   `resolve_key` silently returns index 0. The spatially varying kernel map
   effectively degenerates. Inherited from the merge base; the extension code
   knows about this trap and works around it (`templates.py:1035-1039`), but
   `convolve_templates` was never fixed. Ours fixed it with `wcs_original`.

2. **`wcslin_pscale=1.0`** in `do_driz` — hidden by `renormalize=True`, but it
   means the drizzled stamp carries no drizzle flux scale at all; the stamp sum
   is set by the last contributing frame's native window.

3. **`eval_ePSF` mis-centres even grids.** `x0 = ((size*2) − 1)//2` with
   `size = (sh[0]−1)//4` gives 128/144 for the 260/292 grids whose true centre
   is 129.5/145.5 — off by 1.5 oversampled px = **0.375 native px** per axis.
   Measured centroid offset (+0.342, +0.321) native px vs ours (+0.007,
   −0.016). The differential F444W↔F770W is ≈0.8 MIRI pixels; `recenter=True`
   on the kernel discards it.

4. **MIRI grid knots hard-coded wrong.** `[1, 358, 1032]` vs the header's
   `[1, 513, 1024]`; and `nx = clip(int(rx),0,2)` then `epsf[:,:,nx+1+(ny+1)*3]`
   indexes up to 12 on a 9-plane cube.

5. **`is_star` never survives convolution** (§3), so the astrometry star cut is
   a no-op.

6. **`from_geojson` raises `NameError`** when the sibling `.fits` is missing
   (the `psfs = None` initialization was deleted).

7. **Two dead config knobs shipped as live documentation**, plus the inverted
   "max template-size cap" wording (§4).

8. **Wren's floored `tcor_int` drops `trunc`** relative to its own fallback
   branch (§5) — `est3int` ≈6.6 % low for the floored (faint,
   `snr_seg < fit_snrlo_psf`) population. Bright/extended sources take the
   photutils Kron branch and are unaffected.

9. **The per-region PSF cubes are index-misaligned with their region maps.**
   `run_uds_770_wren.py:66` takes centroids of **`prm_kern`** and assigns the
   resulting cubes to `prm_444.psfs` / `prm_miri.psfs` (`:73-74`), but
   `prm_kern = prm_444.overlay_with(prm_miri)` (`:55`) is a strictly finer
   partition. `PSFRegionMap.get_psf` and `_psf_ee` then index
   `psfs[resolve_key(ra,dec)]` with the **band map's own** key, so a region
   generally gets the stamp drizzled at a different sky position. Nothing
   raises, because `n_444 ≤ n_kern`. For scale, our equivalent maps have
   1694 / 294 / 2911 regions. `containment` is unaffected (one scalar per
   band), but every per-region PSF and EE lookup is. `to_file`'s containment
   broadcast is separately sized off the region table
   (`n = keys.max()+1`) rather than the PSF cube, and nothing checks the two
   agree. Ours drizzles each map at its own centroids.

Defects on our side found in the same pass:

1. **`flux_<i>_total` divides by the low-res stamp sum** when the algebra says
   the detection-side sum (§5). −0.7 % at `psf_size: null`, +3.1 % at the
   `RunConfig` default 4.0″.
2. **`PSFSZ<i>`/`RCIRC<i>` are half their true value** whenever
   `bin_factor > 1` in upsample mode (§5). Blast radius is confined to
   `_record_psf_ee`: the same aliased WCS is read at `pipeline.py:1408-1412`
   to write `aper_<i>`, and there it is correct, because the fit image really
   is on the 0.04″ grid after upsampling (`aper_1 = 0.5` exactly).
3. **No positivity clip before normalization** (`templates.py:1544-1551`), so
   negative pixels in the dilated ring shrink the normalizer and scale the
   template up.
4. **Dilation tie-break is catalog-id ordered**, not geometric
   (`catalog.py:372`).
5. **Kernel is not recentred on path A** (`recenter` defaults `False`); ≈0.5
   hi-res px = 20 mas of the FFT-origin offset survives.
6. **No detection-side truncation correction at all** — no `template_norm`, so
   nothing can reconstruct the F444W flux a template implies.
   `_add_aperture_photometry` does write a per-source `ap_corr_<i>`, but it is
   `apF_book/apB_book` on two unit-sum templates: it corrects convolution
   loss, not truncation, and carries no EE.

---

## 8. Cross-tree API incompatibilities

Neither script runs against the other tree:

- `FitConfig(fit_astrometry_joint=…)` → `TypeError` in wren (removed in
  `846a022`).
- `FitConfig(template_extend_mode=…, f444w_col=…, f444w_aper_col=…,
  fit_snrlo_psf=…, wings_snr_psf=…, template_blend_p=…,
  template_blend_annulus=…, extend_template_ee=…, astrom_isolation_thresh=…)`
  → `TypeError` in ours (fields do not exist).
- `Pipeline(wht_images=…, psf_throughputs=…, templates=…)` → `TypeError` in
  wren.
- `get_psf_radec(ee_fraction=…, size_quantum_arcsec=…, parity=…)` →
  `TypeError` in wren.
- `PSFRegionMap.containment` / `get_containment` / `resolve_key` do not exist
  in ours; `lookup_key_slow` does not exist in wren.
- `utils.psf_stamp_containment`, `psf_ee_at_radius`, `psf_ee_radius_pix` are
  wren-only. `psf.stamp_encircled_energy`, `DrizzlePSF._record_realized_ee` are
  ours-only.
