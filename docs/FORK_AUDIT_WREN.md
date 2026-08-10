# What wren implemented in `dev-wren` that `ivo:main` does not have

Audit of `wrensuess/mophongo` `dev-wren` @ `512db4b` against `ivolabbe/mophongo` `main` @ `15e0b4a`,
merge-base `80cad37` (2026-05-01). Every claim below was read out of the code by one agent and
re-checked against both trees by a second. Claims that failed re-check are listed at the end
rather than deleted.

Line references are `file:line` at the stated ref. `main` means `main` HEAD unless noted;
where the local `template` branch (a descendant of main) already carries a port, that is called out.

---

## Executive summary

- The bulk of the fork is a **rebuilt aperture-correction / total-flux system**: four named
  estimators (`est1`, `est2`, `est3int`, `est3cat`) with per-estimator errors, ~20 new catalog
  columns, and a two-step tie (model-Kron internal total `tcor_int`, then a per-source catalog
  scale `s_cat`). `main`'s `_add_aperture_photometry` is still byte-identical to the merge base.
- Underneath it sits a **rewritten template builder**, `Templates._extended_composite`: one
  SNR-graded linear blend `H = W·data + (1−W)·M` between real detection pixels and a
  data-anchored PSF model, over a globally **disjoint area-weighted ownership map**. This
  replaces the old faint/bright branch switch. `main` has two unrelated PSF-wing extension
  routines instead.
- **`PSFRegionMap.containment`**: a per-region factor converting stamp-normalised encircled
  energy to (near-)absolute EE, serialised through the geojson. `main` solves the same physics
  at filter level with `_filter_psf_throughput` + `flux_<i>_total`. Two incompatible answers to
  "what is a total flux"; this is the one merge decision that cannot be made mechanically.
- Four reusable **PSF curve-of-growth helpers in `utils`** (`psf_ee_radius_pix`,
  `psf_ee_at_radius`, `psf_ee_area_pix`, `psf_stamp_containment`). `main` has EE machinery only
  inside `psf.py`/`DrizzlePSF`, in arcsec and tied to drizzled stamps. `main`'s own
  `examples/run_uds_770_wren.py:88` already calls `utils.psf_stamp_containment`, which does not
  exist on main — that script is broken as committed.
- **Two performance cuts on 345k-source fields**: `_sources_with_coverage` pre-extraction rough
  cut (~57% of UDS sources never built) and ROI-restricted ownership (~50% of labels skipped).
  Plus `np.searchsorted` replacing `SegmentationImage.get_index` (3.9 → 0.05 ms/source).
- **Complete retirement of `SparseFitter`** (fit.py 1657 → 135 lines, `FitConfig` only). Both
  forks deleted ~1250 of the same lines; the wren-only delta is the assembly/estimator shell that
  `main` kept, plus five config fields.
- **~3100 lines of tests and design docs** that main has no equivalent of:
  `tests/test_pipeline_aperture.py` (1476 lines, 29 tests, several written as revert-verify
  regressions with measured acceptance numbers), `tests/test_template_extension.py` (585 lines),
  and `docs/aperture_corrections.md` (602 lines) + `docs/stage4c_scope_and_brief.md`.
- Two documented-but-unfixed defects logged in wren's `CHECKLIST.md` that are **live in main and
  documented nowhere in main**: run-to-run non-reproducibility of scene partitioning, and the
  flux-block ridge biasing faint sources low (−33% at `d_i/median = 1e-6`).
- **Hard merge hazard**: `Template.FLAG_*` bits `0x40`/`0x80` mean `DEBLENDED`/`SATURATED` on main
  and `PSF_EXTENDED`/`EXTEND_FAILED` on wren. Neither exists at the merge base; both forks claimed
  the same two bits.
- A plain merge produces 30 conflicted paths and at least one **silent** semantic auto-merge
  (`Template.convolve_cutout` interleaves main's origin-parity fix with wren's unit-sum
  renormalisation, and the merged body then reads wren-only attributes). Do not merge; port.

---

## 1. Aperture corrections and total-flux estimators

This is the fork. `main`'s `Pipeline._add_aperture_photometry` (`main:src/mophongo/pipeline.py:1062-1137`)
is byte-identical to `80cad37:src/mophongo/pipeline.py:436-511` and writes four columns:
`ap_model_{idx}`, `ap_flux_{idx}`, `ap_corr_{idx} = tmpl.data.sum()/aperture_sum(tmpl)`,
`ap_flux_corr_{idx}`. No error column, no per-source curve-of-growth, no total-flux convention.
Everything in this section is new since the split, on the wren side only.

### 1.1 Four-estimator suite with per-estimator errors

Wren writes four separately named totals with stated conventions, never algebraically
pre-collapsed:

| column | formula |
|---|---|
| `ap_flux` | `ap_model + res_sum` |
| `ap_flux_est1` | `(ap_model + res_sum)·totcor1` (IDL-exact, residual scaled) |
| `ap_flux_est2` | `ap_model·totcor1 + res_sum` (residual unscaled) |
| `ap_flux_est3int` | `ap_model·apcor1·tcor_int + res_sum` (internal Kron total) |
| `ap_flux_est3cat` | `ap_model·apcor1·tcor_int·s_cat + res_sum` (catalog-tied release flux) |

Errors take the fractional profile-fit error `frac_err = err_{idx}/flux_{idx}` from the sparse
solve and apply it to the corrected **model** part only; the multiplicative corrections are
treated as noiseless and the `res_sum` pixel noise is deliberately excluded.
`err_est1 = err_est2 = |totcor1·ap_model|·frac_err`, and analogously for est3int/est3cat.

Symbols: `Pipeline._add_aperture_photometry` `wren:pipeline.py:707-1303`; writes at
`:1266`, `:1270`, `:1283`, `:1296`; errors `:1277-1279`, `:1285-1287`, `:1298-1300`;
`frac_err` `:1245-1250`.

**main status**: absent. `ap_flux_corr_{idx}` is a crude est1-shaped analogue with an
uncorrected `1/apB` and no error column. **Port difficulty**: entangled — see the dependency
chain in §1.9.

### 1.2 New column schema

Twenty per-band columns wren writes and main does not, created at `wren:pipeline.py:863-885`:

| column | meaning |
|---|---|
| `apcor1_{idx}` | `apF_corr/apB_corr` — hi-res/lo-res model aperture-flux ratio (Wuyts 2008 shape correction) |
| `totcor1_{idx}` | `1/apB_corr` — internal aperture-to-total, the IDL `totcor` analogue |
| `apcor_{idx}` | **repurposed** to the full released correction `apcor1·tcor_int·s_cat` |
| `tcor_int_{idx}` | `f444w_ktot / ((template_norm + flux_beyond_stamp)·apF_book)` |
| `s_cat_{idx}` | catalog `f_f444w / f444w_ktot` |
| `f444w_ktot_{idx}` | internal mophongo F444W total |
| `apf_data_{idx}` | real neighbour-subtracted F444W aperture flux (diagnostic only) |
| `res_sum_{idx}` | residual in the aperture disk, other sources' segments zeroed |
| `res_seg_{idx}` | residual over the source's own segmap footprint (diagnostic) |
| `ap_flux_est*`, `err_ap_flux_est*` | the eight estimator/error columns above |

Removed relative to main/base: `ap_corr_{idx}`, `ap_flux_corr_{idx}`.

### 1.3 Truncation term and aperture crowding delta (Stage 4 / 4b)

The single algorithmic idea replacing the old branch-switched correction path. Per source,

```
trunc_denom = template_norm + flux_beyond_stamp
trunc       = template_norm / trunc_denom
apF_corr      = apF_book·trunc      + flux_beyond_aper      / trunc_denom
apB_corr_book = apB_book·trunc      + flux_beyond_aper_band / trunc_denom
```

`flux_beyond_stamp = max(A_src·(1/c_det − f_cut), 0)` is the PSF-extrapolated, core-anchored
estimate of source flux outside the model support, with `c_det` the detection-PSF containment at
the source sky position and `f_cut` the PSF fraction inside the actual support `ext_psf`
(deliberately not the whole cutout, so the faint limit `sum(H) == A_src·f_cut` holds exactly).
`flux_beyond_aper` is the source's own PSF-model flux inside the measurement aperture but outside
the fit support — exact, not approximate, because `H` is identically zero there, so
`H_corr − H == A_src·psf_cut`. Stored as a scalar (never the array; ~340k sources) computed with
`CircularAperture.to_mask('exact').multiply()` rather than `aperture_photometry` (~20× faster,
identical sum), behind a bounding-box pre-check that skips the isolated common case.

`trunc` cancels exactly in `apcor1` (a shape ratio) and survives in `totcor1` (aperture-to-total).
Both properties are pinned in one test so they cannot be conflated
(`wren:tests/test_pipeline_aperture.py:445`, asserts at `:489` and `:491`, rel=1e-10).

Measured payoff, `totcor1` vs nearest-neighbour distance bins `[<0.6″ | 0.6–1.2″ | 1.2–2.4″ | isolated]`:
`1.40 / 1.24 / 1.07 / 1.0` (masked) → `0.99 / 0.99 / 1.00 / 1.0` (Stage 4b).

Symbols: `wren:pipeline.py:1037-1039`, `:1052`, `:1066-1067`;
`Template.flux_beyond_stamp` `wren:templates.py:445` set at `:1260-1270`;
`Template.flux_beyond_aper` `:453` set at `:1293-1302`.

### 1.4 Two-step catalog tie: `tcor_int` then `s_cat`

Step 1 is evaluated entirely on models. For the floored / PSF-converged population
(`Template.apcor_from_psf` True), `f444w_ktot` is set directly to `trunc_denom`, which is exactly
`A_src/c_det` — the containment-corrected point-source total — with no photutils call and no EE
lookup (`wren:pipeline.py:1156-1163`). For the bright/extended population,
`f444w_ktot = kron_flux_model / EE_true_444(r_kron_circ)` from `_model_kron`
(`:1164-1185`). With no usable catalog colour-aperture column the whole thing degrades to the
true-normalised point-source form `tcor_int = 1/apF_corr`, warned once (`:1110-1130`).

Rationale (docs §3.2/5.4, Skelton 2014 Eq. 1 / Weibel 2024): a Kron aperture scales with the
source so the total stays bounded at all SNR, unlike the earlier measured-flux denominator whose
faint tail was unbounded.

Step 2 is `s_cat = f_f444w(catalog) / f444w_ktot`, applied to the model-shape flux of every band
(residual unscaled) so `est3cat` sits on the catalog's total-flux system with the F444W tie exact
by construction. Guarded on a **positive** catalog total; deliberately not clipped at the top
because the large-positive tail cancels algebraically in `est3cat`
(`wren:pipeline.py:1187-1195`, `:1293-1300`). Justified by a measured `0.83 [0.65, 0.88]`
bright-source ratio between IDL-internal and catalog totals (docs §3.3) — a real,
morphology-dependent per-source difference a global scalar cannot absorb.

**main status**: absent. No `f444w_col`/`f444w_aper_col`, no external catalog tie, no Kron on a
model stamp. `main`'s only total convention is `flux_{idx}_total = flux/throughput`.

### 1.5 `_model_kron`

Runs `photutils.SourceCatalog` on the fitted F444W template's own **model** stamp
(`orig_t.data[slices_cutout] · template_norm`, real flux units, never image data) with the
source's own segment and `kron_params=(2.5, 1.4, r_floor_pix)` — photutils' own
minimum-circular-radius mechanism supplies the Skelton floor, taken from the catalog
colour-aperture diameter through the reference pixel scale. Returns
`(kron_flux_model, r_kron_circ)` with `r_kron_circ = max(2.5·kron_radius·sqrt(a·b), r_floor_pix)`
capped at `0.5·min(stamp.shape)`.

Two engineering invariants worth keeping in any port: when the cap engages, photutils' elliptical
(edge-truncated) Kron flux is **replaced** by the circular flux at the same capped radius so
`kron_flux/EE(r_kron)` never mixes radii; and the radius is quantised to a 0.25-px grid so the
per-region PSF-EE cache hits (documented as hours saved on a 340k-source run, <0.2% EE effect).
Degenerate moments, an empty segment, or any photutils exception fall back to a circular aperture
at the quantised floor radius scaled by `template_norm`.

`wren:pipeline.py:482-564`; independently recomputed in
`tests/test_pipeline_aperture.py:232`, cap invariant at `:299`.

*Caveat for a port*: the `use_source_catalog=False` shortcut has no `src` caller at HEAD — Stage 4c
made the `apcor_from_psf` branch bypass `_model_kron` entirely. It is exercised only by tests.

### 1.6 Stage 4c: unmasked tie denominator

After 4b fixed `totcor1`/`apcor1`, the est3 tie denominator was still the ownership-**masked**
`template_norm`, so `est3cat` re-inherited the crowding artifact (`est3cat/est1` =
`1.49 / 1.19 / 1.07 / 1.0` across the same bins). Fix: substitute `trunc_denom` for
`template_norm` in `denom = trunc_denom · apF_book` (`wren:pipeline.py:1141`) and unmask
`f444w_ktot` for the floored population. The non-floored bright/extended population keeps the
masked photutils Kron, on the grounds that its owned support is large so the clipped fraction is
small and the masked Kron preserves real extended structure a PSF-total would discard.
Acceptance tests `:1076`, `:1095`, `:1163`.

Note the implementation deviates from the brief: `docs/stage4c_scope_and_brief.md:88` ruling D2
specifies a scalar Kron top-up; the code substitutes the exact point-source total instead, which
the brief explicitly permitted as the cheaper consistent route.

### 1.7 Supporting mechanics worth porting independently

| item | what | symbols | main |
|---|---|---|---|
| Band-vs-detection containment ratio | matching kernel maps the `c_det`-truncated detection PSF onto the `c_b`-truncated band PSF, so the model's band-aperture flux is high by `c_det/c_b`; applied on the band side only, and `apcor1` is ruled to keep the resulting `c_det/c_b` factor as a genuine shape effect | `wren:pipeline.py:1053-1055`, `:1073` | absent |
| Band-native EE radius | in upsample mode `wcs[idx]=wcs[0]`, so `r_img_pix` is in fine reference pixels while `psfs[idx]` is on its coarse native grid; `self._native_pscale` is captured before the fit loop and `r_band = r_orig·pscale_ref/pscale_band` | `wren:pipeline.py:1356-1359`, `:831-838` | absent (main samples no band PSF, but inherits the trap on any port) |
| Aperture geometry consistency | measure the template aperture sum on `tmpl.data[tmpl.slices_cutout]` with the position shifted into that frame, matching the residual patch geometry, so `ap_flux = ap_model + res_sum` is exact by linearity | `wren:pipeline.py:453-465` | **main is wrong here**: `main:pipeline.py:1056-1059` sums the full `tmpl.data` with the shift commented out, so edge-clipped templates include out-of-image model flux the residual aperture cannot see. Trivial standalone fix. |
| Residual accounting | `res_sum` zeroes other sources' segment pixels first (`_other_source_mask`, OR-reduced to the residual resolution for `k>1`); `res_seg` sums over the source's own footprint with the same binning | `wren:pipeline.py:649-675`, `:597-647`, `:1197-1220` | absent |
| Neighbour-subtracted F444W residual | `images[0] − Σ_j template_norm_j·H_j` built once per run and reused across bands; `apf_data = template_norm·apF_book + aper(residual)` adds the source's own model back by linearity | `wren:pipeline.py:566-595`, `:1091-1100` | absent |
| Per-region EE cache | keyed on `(id(psfmap), region_key, radius, with_containment)`, fixing a real collision — `get_psf` returns a fresh ndarray per call and CPython reuses freed ids, so an `id(psf)`-keyed cache silently served another region's growth curve | `wren:pipeline.py:922-956` | absent (no per-source EE lookup) |
| Loud-failure discipline | three named guards where a missing input would otherwise silently revert a correction: per-band `aperture_diam` array → `flux_beyond_aper == 0` for every source; geojson without a containment column; no usable `f444w_aper_col` | `wren:pipeline.py:1424-1431`, `:855-861`; `psf_map.py:281-287` | absent |

### 1.8 Removals wren made in this area (decisions main has not taken)

- **The measured/catalog-anchored `tcor_H` blend.** `tcor_H = f_f444w / aper_rphi` with
  `aper_rphi` a logistic blend of the *measured* neighbour-subtracted F444W aperture flux with a
  catalog-anchored prediction. Removed with `FitConfig.tcor_lowsnr_psf`/`tcor_blend_center`/
  `tcor_blend_width`, `_tcor_blend_weight`, and the `tcor_`/`aper_rphi_`/`ap_flux_corr_` columns.
  Rationale, verified on real products: dividing by a measured 707-pixel aperture sum imported sky
  noise (12.3% of raw denominators negative for MIRI-undetected sources) giving
  `tcor_1 = 1.69 [1.24, 2.81]` with an unbounded tail. **Ruling: corrections come from models
  only; the only measured quantities are fluxes in apertures matched to the source size.**
  main never had this machinery, but the ruling rules out a whole family of estimators.
- **The Phase-A PSF-stamp-EE correction branch.** Corrections used to branch per source
  (point-source-like → PSF-stamp EE; bright/extended → convolved-template aperture fraction),
  which created two uncorrelated populations: a constant "line" at `totcor1 = 1.233` carrying no
  per-source information, and a "cloud" of template-path sources up to `totcor1 = 265` because raw
  noise-dominated halo data diluted `Σ(H)`. `Template.apcor_from_psf` survives, downgraded to a
  pure performance gate.

### 1.9 Dependency chain (why most of this is entangled)

The estimators consume four per-`Template` quantities that do not exist anywhere in main's `src`
(`git grep -n 'template_norm\|flux_beyond_stamp\|flux_beyond_aper\|apcor_from_psf\|snr_seg' main -- src/`
returns nothing), all produced inside `Templates._extended_composite`:

`template_norm`, `flux_beyond_stamp`, `flux_beyond_aper`, `apcor_from_psf`/`snr_seg`.

main's nearest analogue is `Template.extension_pre_norm_sum` (`main:templates.py:1261`), set only
on the psf-wings branch and never read by the pipeline.

So: **est1/est2 need `template_norm` + `flux_beyond_stamp`. est3 additionally needs the Kron path
and the catalog columns.** A minimal useful subset that is *not* entangled and can land
independently: the four `utils` EE helpers, `PSFRegionMap.containment`/`resolve_key`/
`get_containment`, and the `_aperture_sum_on_template` geometry fix.

### 1.10 Loose ends on wren, do not port

`Pipeline._intersect_slices` (`:677-705`) and `_get_representative_kernel` (`:435-451`) have no
callers; `FitConfig.f444w_totcor_col` is declared and never read; `_build_f444w_residual` writes
`f444w_template_residual.fits` to the CWD unconditionally (`:591`) and holds a full-mosaic float32
copy on `self`; `_add_aperture_photometry` uses `print()` for progress (`:856`, `:908`) contrary to
project style; the method rows on `tmpl.id` while `_update_catalog_with_fluxes` rows on
`id_parent`; `tcor_int`/`f444w_ktot`/`s_cat` are F444W-side and band-independent yet recomputed
per band.

### 1.11 Regression targets if main ports this

All from `docs/aperture_corrections.md`, real F1500W UDS run:

- PSF-limit `totcor1`: 1.233 → 1.347, vs IDL 1.357 (<1%).
- Raw-flux offset −0.075 mag → target |μ| ≤ 0.02.
- `totcor1` crowding bins: 1.40/1.24/1.07/1.0 → 0.99/0.99/1.00/1.0.
- `est3cat/est1` crowding bins: 1.49/1.19/1.07/1.0 → flat (Stage 4c).
- Known open: ~3% F444W EE residual (true-normalised `apcor1` 1.217 vs IDL `psfcor` 1.255);
  ~10–16% catalog-vs-IDL F444W total-definition offset, left visible on purpose.

---

## 2. Template construction

### 2.1 `Templates._extended_composite` — the SNR-graded data/PSF composite

For each source, one composite stamp over the source's owned territory:

```
H     = W·data + (1−W)·M          over ext_psf, 0 elsewhere
M     = A_src · psf_cut           (data-anchored PSF model)
A_src = Σ max(data,0)|segment / Σ psf|segment
```

`psf_cut` is the detection PSF resampled onto the cutout grid with `map_coordinates(order=1)`
after unit-sum normalisation. `W` is a radial weight field: one scalar `w_core` over the segment
from the in-segment SNR, plus one weight per radial annulus over the owned background halo from
that annulus' own SNR. Halo weights are forced monotone non-increasing outward via
`np.minimum.accumulate` seeded at `w_core`, so data trust never increases with radius and a faint
core caps its halo. Support is `own segment | (owned background within the PSF ee_reach radius)`;
the data reach is capped separately at `max_radius_pix`, beyond which only the PSF model
contributes. Non-finite pixels are excluded from the halo statistics and take the model, so one
bad pixel cannot NaN a template.

`wren:templates.py:1098-1304`; `H` at `:1245`; `A_src` at `:1191`; monotone accumulate at `:1232`;
annulus binning (arcsec→px via `proj_plane_pixel_scales(cut.wcs)`) at `:1196-1232`.

**main status**: absent. main has two older unrelated routines, `extend_with_psf_model`
(`main:templates.py:1051`, grid of Gaussian-smoothed PSFs least-squares fit to segment pixels) and
`extend_with_psf_wings` (`main:templates.py:1165`, template convolved with the local PSF, fill only
zero pixels, optionally background-only). Both are **live**, wired into `Pipeline.run` at
`main:pipeline.py:1249-1283` and `:1686-1705` via `Pipeline(extend_templates=...)`. main's
`extract_templates` also does a weak default extension of its own (`dilate_segmap=2`).

**Already ported on the `template` branch**: `template_schemes.composite_wren`
(`template:src/mophongo/template_schemes.py:763`) reproduces this algorithm faithfully — same
ownership mask, same `w_core`/annulus ladder, same `A_src` anchor, same `ext_data`/`ext_psf`
supports, same `flux_beyond_*` bookkeeping. Most of the algorithmic content here is therefore
understood and re-implemented already; what is missing is on `main`.

### 2.2 `blend_weight`

`w(snr, thresh, p) = 1` for `snr ≥ thresh`, else `(snr/thresh)**p`; `0` for NaN SNR, clamped at 0
for negative flux. `thresh` is the **onset** of PSF blending, so the old hard switch is reproduced
in the limit. Applied with `thresh = 1.5·fit_snrlo_psf` for the core and `thresh = wings_snr_psf`
per halo annulus. Deliberately isolated at module level so the functional form can be swapped
without touching call sites. `wren:templates.py:355-370`. **main**: absent, no SNR-dependent
template blending of any kind. **Port**: trivial.

### 2.3 Global area-weighted ownership map

Ports the IDL `kseg > knn` idea and makes it provably disjoint. For every pixel the owner is the
label with the largest area within a disk of radius `max_radius_pix`, computed by `fftconvolve` of
each label's binary mask with a disk kernel against a shared best/owner arbiter. Segment pixels
unconditionally own themselves (`owner` seeded with `segmap`), so only label-0 background is
contested. Areas are `np.rint`'ed because `fftconvolve` of binary arrays carries ~1e-15 noise that
would break exact ties non-deterministically; strict `>` makes the lowest label win ties. Being
area-weighted rather than a distance Voronoi, a large segment wins more inter-source territory
than a small one.

`Templates._build_ownership` `wren:templates.py:983-1048`; `_disk_kernel` `:926-931`; consumed at
`:1132`.

**main status**: partially. main's `catalog.safe_dilate_segmentation` (`main:catalog.py:340-363`),
called by default from `extract_templates`, does produce a disjoint partition — but only out to
~2 px and not area-weighted. main's wings path (`main:templates.py:1226-1230`) only excludes
foreign-segment pixels, so two neighbours can both fill the same background pixel.

### 2.4 ROI-restricted ownership

`_cutout_roi` builds a coarse bool mask (step 8, ~14 MB instead of ~875 MB full-res on MINERVA) of
every pixel a retained cutout will read, from the same resolved sizing list the extraction loop
uses. `_build_ownership` then skips any label whose bbox-padded-by-radius window misses the ROI.
Because a label writes only inside that window the restricted result is **identical** inside the
ROI, not approximate. Measured: ~50% of labels skipped for ~2 s of overhead on UDS.
`wren:templates.py:960-981`, `:1032-1036`, call site `:1380-1386`.

### 2.5 `_sources_with_coverage` — pre-extraction rough cut

Before extraction, keep only catalog rows whose template footprint can touch positive weight in
**any** fitted band. The footprint is the segment bbox floored at `min_size//2`, per source —
deliberately not one global radius, because a single 3829-px star halo in UDS would dilate the
coverage mask until nothing is cut. Coverage is tested in each band's frame after a WCS round-trip
with a +1 px margin; the common case is a single lookup into `maximum_filter(w > 0, size=2·r0+1)`
and only the rare oversized segments get their own box tested. Mirrors `prune_outside_weight`
(which still does the exact per-band cut afterwards) so the surviving set is unchanged; the
catalog is not filtered, cut sources keep `bad_value` rows. Measured: 196k–203k of 345,792 UDS
sources cut (~57%, ceiling 67% at 33% MIRI coverage).

`wren:pipeline.py:57-148`, call site `:1499-1507`. **main**: builds every detection template and
relies solely on `Templates.prune_outside_weight` after the fact.

### 2.6 Sizing and bookkeeping

| item | what | symbols | main |
|---|---|---|---|
| `min_size` as instance attribute | so the pipeline can pre-size every cutout to hold the PSF extension radius before extraction; slice bookkeeping is then correct from birth rather than patched afterwards | `wren:templates.py:665-671` | class attribute `min_size = 8` (`main:templates.py:742`), `__init__` takes no arguments |
| `min_size_from_aperture` | smallest **even** size enclosing a photometry aperture, using the **finer** pixel-scale axis (`np.min(proj_plane_pixel_scales)`) so a square cutout contains the aperture on both axes for non-square pixels, rounding up so the `(min_size//2)*2` floor does not drop a pixel | `wren:templates.py:673-699` | absent |
| pipeline `r_fill` | one representative detection-PSF growth curve per run (widest region if `psfs[0]` is a `PSFRegionMap`), cached as `self.ee_radii_pix`; `r_fill = max(R_ee, aperture_radius_F444W + kernel_half_width)` where `kernel_half_width` is the 95% encircled radius of `abs(K)` across fitted bands (**not** the zero-padded array size, which would inflate template memory); `min_size` floored at `2·ceil(r_fill)+1` rounded to even | `wren:pipeline.py:1361-1461`, `:1509` | absent |
| `template_norm`, `n_pix` | pre-normalisation sum in real image units and the segmap pixel count, captured after the composite and before normalising so `template_norm·H == composite` exactly; both survive `convolve_cutout` and `downsample` | `wren:templates.py:432-433`, `:1399`, `:1434` | absent entirely; the pre-normalisation sum is discarded (`main:templates.py:1399-1403`) |
| `np.searchsorted` for bbox lookup | replaces `SegmentationImage.get_index(label)`, whose `check_labels` scans all labels every call: 3.894 → 0.053 ms/source at 345k sources | `wren:templates.py:1358` | still `segm.get_index(label)` (`main:templates.py:1381`) |

### 2.7 Behaviour changes on the default path

- **Unit-sum renormalisation after convolution.** `convolve_cutout` divides the convolved stamp by
  its own sum, on the argument that a numerically constructed matching kernel does not have sum
  exactly 1 and otherwise biases fitted amplitudes and aperture corrections by `1/kernel.sum()`.
  `template_norm` is propagated unchanged on the argument that PSF matching preserves total flux.
  `wren:templates.py:589-599`. **Entangled**: main's `convolve_cutout` carries a kernel-recentring
  fix wren lacks (main shifts the output origin by `kernel.shape//2` and recomputes
  `input_position_cutout`/`position_cutout`, `main:templates.py:558-591`; wren still builds at
  `position=self.input_position_original`). Any port must graft onto main's recentred version.
- **Universal positivity clip.** `np.clip(cut.data, 0.0, None, out=cut.data)` on every template on
  every path before normalisation, because negative pixels corrupt the unit-sum normalisation, the
  wing-flux anchor and the `apF/apB` ratio. Carries a TODO that zeroing is a placeholder and a
  negative pixel should ideally be replaced by the scaled PSF model value (IDL's `<=0` → PSF-fill).
  `wren:templates.py:1420-1426`. This changes non-extended templates too.

### 2.8 Flag bits — hard collision

`Template.FLAG_PSF_EXTENDED = 0x40` and `FLAG_EXTEND_FAILED = 0x80` (`wren:templates.py:382-383`)
vs `FLAG_DEBLENDED = 0x40` and `FLAG_SATURATED = 0x80` (`main:templates.py:383-384`), with main's
`is_deblended`/`is_saturated` properties driving scene-builder behaviour
(`main:scene.py:615`, `main:pipeline.py:1247`, `:1678`). Neither pair exists at the merge base;
both forks independently claimed the same two bits. A port must renumber one side and audit every
`flag &` site.

### 2.9 Removals

- `Templates.add_component` deleted on wren. main keeps it (`main:templates.py:759-806`) with
  a cosine-similarity 0.999 degenerate-column guard — but with **no callers** anywhere in main's
  `src` or `tests`.
- `AlignedCutout.as_block_reduced` / `as_block_replicated` deleted on wren; present but dead on
  main (`main:templates.py:295`, `:301`).
- **Retained on wren, removed on main**: the quick-flux kernel-cropping path
  (`_prepare_fft_fast`, `_crop_kernel`, `prepare_kernel_info`, `Template.ee_rlim`/`ee_fraction`).
  Important qualification: it is **dead on wren too** — the only caller is
  `wren:tests/test_templates.py:82`, nothing in `src` calls it, and the same was true at the merge
  base. main's deletion removed unreachable code. main gained `_is_identity_kernel` short-circuiting
  instead.

---

## 3. PSF and `PSFRegionMap`

wren made **zero additions** to `src/mophongo/psf.py` — the base→wren diff there is pure deletion
(1421 → 1177 lines) — and `jwst_psf.py` is byte-identical to base. All new PSF-area capability is
in `psf_map.py` and four `utils` helpers.

### 3.1 `PSFRegionMap.containment`

A per-`psf_key` field holding the fraction of PSF flux inside the stored (finite, circularly
apodised) stamp, so `EE_true(r) = EE_stamp(r) · containment`. Serialised as a geojson column
(broadcast from a scalar if uniform), read back by `from_geojson` with a **loud warning and a 1.0
default** when the column is absent, so a pre-Stage-2 cached map reproduces the old
stamp-normalised behaviour bit-for-bit but noisily.

Symbols: field `wren:psf_map.py:57-58`; `from_geojson` block `:265-291`; `get_containment` `:590-599`;
`to_file` column `:601-620`; producer `utils.psf_stamp_containment` `:utils.py:189-227`.

**Physics correction to wren's own docstring**: the value is *not* the fraction of the PSF's true
total. `psf_stamp_containment` normalises by `psf.sum()` of the **parent grid** (via
`psf_ee_at_radius`, `wren:utils.py:161-162`), and the parent 8″ STPSF grid is itself truncated. So
`containment = EE_disk / EE_parentgrid`, systematically too large by the parent grid's own missing
flux. main's `docs/FORK_DIFF_WREN.md:342-362` already records this objection.

The disk-not-box geometry is correct and worth keeping: drizzled region-map stamps are circularly
apodised, their corner pixels are identically zero, so a box fraction over-counts flux the stamp
does not contain.

Measured effect: F1500W `totcor1` 1.233 → ~1.347 vs IDL 1.357 (docs §4.1/5.2); containment
F444W 0.962, F1500W 0.919.

**main status**: absent as a per-region quantity. main solves the same physics per filter:
`stamp_encircled_energy` (`main:psf.py:1866`) measures `ee_box`/`ee_circ`/`r_circ`/`r_ee` on the
drizzled stamp — which is already on an absolute flux scale, so main's denominator is arguably
better grounded — and `_filter_psf_throughput` (`main:pipeline.py:197`) reduces it to one
filter-level scalar dividing fitted amplitudes into `flux_{idx}_total`. wren's advantage is
granularity: per sky region, following the region map's exposure/PA structure, and applied to the
*correction factors* rather than to the amplitude.

**This is the one merge decision that cannot be made mechanically.** Applying both naively
double-corrects the same finite-stamp truncation: main's throughput is the stamp EE, wren's
containment is its complement.

### 3.2 `resolve_key`

Extracts the None/NaN-guarded region lookup out of `get_psf` into a public
`resolve_key(ra, dec) -> int`, so a PSF and its containment can never be taken from different
regions; the docstring instructs callers needing both to resolve once and index directly.
`wren:psf_map.py:571-585`. main keeps the guard inlined in `get_psf` (`main:psf_map.py:531-541`)
and has no `resolve_key`. Note wren **keeps** `lookup_key` — nothing was renamed. Trivial port,
removes a real divergence risk.

### 3.3 `utils` PSF curve-of-growth family

Three photutils `CurveOfGrowth` wrappers with one convention: profile normalised by `psf.sum()`,
centre at `((nx-1)/2, (ny-1)/2)`, radii `np.arange(0.5, min(xc,yc), 0.5)`, linear interpolation.

- `psf_ee_radius_pix(psf, f)` — radius enclosing fraction `f`. `wren:utils.py:90`.
- `psf_ee_at_radius(psf, r)` — exact inverse, round-trip pinned to 1e-6 for Gaussian and Moffat.
  `wren:utils.py:139`.
- `psf_ee_area_pix(psf, f)` — `ceil(pi·r²)`, documented as the segmap-size threshold below which
  a template is too truncated to be a faithful PSF shape. `wren:utils.py:178`. *Caveat*: imported
  by `templates.py:19` but **never called** anywhere in wren's `src`; the role exists in the
  docstring only.

Deliberately no `abs`/`clip` on the input (matched kernels ring negative and clipping would distort
the curve); instead warns when `psf.min() < -1e-6·psf.max()` and raises when `psf.sum() <= 0`.

**main status**: partially. The names are absent, but main is not without EE machinery:
`psf._encircled_energy` (`main:psf.py:161`) builds the same `psf.sum()`-normalised cumulative
curve; `DrizzlePSF._ee_fraction_to_arcsec` (`main:psf.py:2584`) inverts a growth curve to a radius
in arcsec with absolute-EE semantics; `stamp_encircled_energy` measures realised stamp EE and
`r_ee`. What is wren-only is a reusable **pixel-domain** helper over an arbitrary PSF array, plus
`psf_ee_area_pix`, plus their use to size template geometry.

**Standalone port, ~50 lines, zero new dependencies** — main already imports
`photutils.profiles.CurveOfGrowth` at `main:utils.py:20`.

### 3.4 Dead-code removals in `psf.py` (all safe, none conflict)

wren deleted `psf_matching_kernel_basis` (main keeps it at `main:psf.py:1478`, zero callers,
already flagged in `main:TODO.md:25`), `NEffectivePSF` (`main:psf.py:2758`, only a commented
reference), `jwst_header`/`jwst_probe_headers` (`main:psf.py:2884`/`:2900`, self-reference plus a
commented example; removing them also removes an implicit network dependency), and a block of
commented-out NRC bilinear code.

On `NEffectivePSF`: its distinguishing feature — learning PSF grid break-points from `IPSFX##`/
`JPSFY##` headers instead of hard-coding `[0,512,1024,1536,2048]` — was **absorbed into main's live
class** as `EffectivePSF._interp_grid` (`main:psf.py:1779-1801`, driven by `self.epsf_meta`).
Deleting the class on main loses nothing.

---

## 4. Fitting

### 4.1 Complete retirement of `SparseFitter`

`fit.py` went base 1657 → wren **135** lines (`FitConfig` + `__post_init__` only) → main 378.
Both forks independently deleted ~1250 of the same lines: every module-level function in base's
`fit.py`, the `SparseFitter` solve half, nine orphan `FitConfig` fields, the whole `astro_fit.py`
module, and `tests/test_sparse_cholesky.py`.

**The wren-only delta is exactly one thing plus its fallout**: the assembly/estimator shell that
main kept — `__init__`, `_slice_intersection`, `_weighted_norm`, `build_normal`,
`build_normal_tree`, `model_image`, `ata`/`atb`, `add_flux_priors`, `residual`, `quick_flux`,
`predicted_errors`, `flux_and_rms` (`main:fit.py:125-378`). Deleting it forces:
`__init__.py` exporting `FitConfig` instead of `SparseFitter`, removing the dead
`from .fit import SparseFitter` inside `Pipeline.run` (`main:pipeline.py:1151`), deleting
`tests/test_fit.py`, and repointing `tests/test_astrometry.py`.

wren documents that the legacy path was already broken three ways at the split: `is_bright` was
computed from `t.flux/t.err` (both 0 on a fresh fitter) so astrometry silently never fit; the
all-False mask made `merge_small_scenes` collapse every scene into one; and `_flux_errors`
returned exactly zero for isolated sources.

### 4.2 Config deletions wren made that main has not

`reg`, `cg_kwargs`, `normal` (`'loop'|'tree'`), `run_scene_solver`, `fit_astrometry_joint`.
All five are base-inherited fields main **kept**, not fields main invented. Verified dead on main:
`normal` and `cg_kwargs` have zero readers beyond their definitions.

`fit_astrometry_joint` is the interesting one. wren deleted it and gates the shift block purely on
`int(cfg.fit_astrometry_niter) <= 0` (`wren:scene.py:666`), on the argument that the flag named a
"joint vs separate" choice that no longer exists and could only mean on/off, duplicating `niter`.
It also created a trap: pipeline does `max(niter, 1)` and the old gate tested only the flag, so
`niter=0` did **not** disable astrometry. main closed the trap differently, by ANDing the two
(`main:scene.py:735`), keeping the redundant flag. Note main still passes
`fit_astrometry_joint` in live code (`main:verification.py:1000`, three `examples/run_*.py`,
`examples/cosmos_770_dr0.1.json:49`), so removing it is not a no-op edit.

`SceneFitter.solve` also lost the dead `cg_kwargs` parameter, and four docstrings claiming
conjugate gradients were corrected to say sparse direct solve. Three stale strings remain on main:
`main:scene_fitter.py:59` ("Stateless clone of SparseFitter.build_normal_tree" — the referent no
longer exists), `:139`, `:203`.

### 4.3 `scene.py` divergence — wren's sweep is *not* portable as-is

wren deleted `_bbox_overlap` and `Scene._overlaps`. `_bbox_overlap` is genuinely dead on main and
safe to delete. `Scene._overlaps` is **not**: main's `Scene.create_scene_graph`
(`main:scene.py:857`) calls it. And note wren's own tree is broken here — `Scene.create_scene_graph`
was already called from `Scene.overlay_scene_graph` at the merge base with no definition anywhere;
main **added** the missing implementation, while wren deleted `_overlaps` and left the broken call
(`wren:scene.py:788`), so wren's `overlay_scene_graph` now references two nonexistent symbols.

Conversely main deleted `scene.summarize_scenes` and the `Scene.augment_templates` no-op stub,
which wren kept. The two sweeps are complementary, not overlapping.

### 4.4 `FitConfig` surface for the extension/aperture work

Eleven wren-only fields, all absent from main:
`template_extend_mode` (`'none'|'auto'`, with `__post_init__` collapsing legacy `data`/`psf`/
`hybrid` to `'auto'` and raising on typos), `fit_snrlo_psf` (10.0), `wings_snr_psf` (3.0),
`template_blend_p` (2.0), `template_blend_annulus` (0.15″), `extend_template_ee` (0.95),
`extend_template_segmap` (deprecated), `extend_template_min_size_margin` (1.5),
`f444w_col`, `f444w_totcor_col` (dead), `f444w_aper_col`. `wren:fit.py:66-126`.

main's nearest equivalents are a different design: `template_dilate_segmap` (`main:fit.py:92`),
`skip_template_extension_for_deblended` (`:96`), `extend_wings_background_only` (`:100`), with the
mode on the `Pipeline` constructor as `extend_templates`.

The local `template` branch has ivo's reworked naming for the same idea:
`FitConfig.extend_mode = 'psf_wings'` (`template:src/mophongo/fit.py:122`) with `EXTEND_MODES`
dispatch into `template_schemes.py`.

### 4.5 Default divergences to settle on any merge

| field | wren | main |
|---|---|---|
| `fit_astrometry_niter` | 2 | 5 |
| `astrom_isolation_thresh` | 0.5 (= base) | 0.7 |
| star exclusion from the bright/astrometry mask | unconditional (`& ~is_star`, base-inherited) | opt-in via `astrom_exclude_stars`, default False |

### 4.6 Where main is ahead — do not regress

- **Error estimation.** `main:scene_fitter.py:280` returns `sqrt(diag(A⁻¹))` with off-diagonal
  covariance (dense inverse below 500 sources, sparse `splu` back-solve of unit columns above),
  and `_solve_flux_and_shifts` materialises the full Schur complement. wren keeps the
  covariance-free `1/sqrt(max(diag, 1e-12))` (`wren:scene_fitter.py:246-260`) and its own test
  comments concede it is "structurally unable to reflect the off-diagonal covariance".
- Numerical guards `_positive_diagonal_scale`/`_finite_nonnegative`, the empty-shift-block
  fallback, and `FitConfig.reg_flux` as an explicit override of the adaptive ridge.
- `astrom_shift_tol` early-stop on shift convergence.
- Scene partitioning: `max_size` soft cap with local bisection, `isolate_saturated`,
  merge-time `isolation_thresh` against the full-field normal matrix, `to_shift = 0` reset on the
  flux-only fallback.

---

## 5. Tests

Base had 30 test files; wren has 31, main 21. wren's policy was repair-or-park-as-xfail; main's was
delete. main dropped 18 of base's 27 modules and kept 20 total.

| file | size | what it pins | main |
|---|---|---|---|
| `tests/test_pipeline_aperture.py` | 1476 lines, 29 tests (base: 17 lines, 1 test) | every stage of §1, several written as revert-verify regressions that must fail against the pre-fix code: faint-limit true-total identity (single and two-band, with a real Tukey-windowed matching kernel), truncation cancellation, `A_src` invariance, crowding flatness for `totcor1`/`est3cat`/`est3int`/`f444w_ktot`, Kron cap shared-radius, EE-cache region keying, band-native pixel scale, containment perturbation bit-identity | **deleted entirely** |
| `tests/test_template_extension.py` | 585 lines, 26 tests | EE helpers against the analytic Gaussian `1−exp(−r²/2σ²)`, flag-bit distinctness, `n_pix` survival through convolution including `FLAG_SUM_ZERO`, `min_size` semantics, ownership self-consistency and neighbour disjointness, blend-weight onset semantics, monotone halo weights, halo ≤ core, negative annulus → pure PSF, NaN pixels taking the model without NaN-ing the template, the cutout-frame `PSFRegionMap` regression | absent (main tests its own different extension path in `test_template_convolution.py` and `test_pipeline.py`) |
| `tests/test_utils.py` | 85 lines | `bin_factor_from_wcs` exact integers for 2×/4× and `ValueError` on 1.5× (silent mis-binning would corrupt all multi-resolution flux); `psf_stamp_containment` in (0,1] and monotone in stamp width; EE round-trip | absent as a file; main has `bin_factor_from_wcs` in src, untested |
| `tests/test_downsample_flux.py` | 49 lines | `Template.downsample(k)` conserves flux to 1e-10 for k=2,3,4 after checking origin k-alignment; `utils.downsample_psf` likewise. Replaces base's dead `test_downsample.py`, which targeted a `bin2d_mean` that never existed | absent; main deleted `test_downsample.py` with no replacement and has **zero** `downsample` references anywhere in `tests/` |
| `tests/test_scene.py` | 95 lines | `generate_scenes` overlap-vs-isolated partitioning, exactly-one-scene membership, star exclusion from `is_bright` | absent as a file. main does exercise `generate_scenes` directly in `test_scene_saturated.py`. **The star test is not portable**: wren hard-codes `& ~is_star`, main makes it opt-in and default-off, so the test asserts the opposite of main's default behaviour |
| `tests/test_psf_map.py` | +4 tests | containment geojson round-trip, missing-column default **plus** the warning (caplog-asserted), `resolve_key` shared by `get_psf`/`get_containment` with NaN/None falling back to region 0, `psf_stamp_containment` vs the analytic disk-in-Gaussian fraction to 1% | main cut this file to 2 tests, dropping three base tests (region count, no tiny regions, from_file) |
| `tests/test_pipeline.py` keystone | — | per-band `flux_i/flux_true` median within 5% and 16–84 spread under 0.20; bright quartile within 3%; **background (segmap==0) chi² per pixel between 0.85 and 1.15** using the known ivar maps; fewer than 10% of segments with `|residual sum|/σ > 5` in the independent low-res band | main's version asserts only a loose `mean(ratio_err) ≈ 1 ± 3`. No chi² floor, no z-score criterion. (main does have a tighter noiseless-prebuilt-template test at `main:tests/test_pipeline.py:414`, atol 2e-5.) |
| `tests/test_pipeline.py::test_pipeline_astrometry` | — | injects a known (0.6, −0.5) px offset, runs the real pipeline with `niter=2` order-0, asserts on the **accumulated `Template.shifted`** rather than the residual (a broad low-res PSF makes the residual nearly insensitive to a sub-pixel shift) — proving the pipeline both computed *and applied* the shift | base had a `return` stub; main deleted it. main's pipeline-level tests all pass `fit_astrometry_niter=0` |
| `tests/test_astrometry.py` | +68 | `_scene_flux_and_residual` helper solving through `generate_scenes` and accumulating per-scene `model_image()` the way `Pipeline.run` does; reproduces the SparseFitter baseline bit-for-bit | main's still constructs `SparseFitter` for assembly (it does use `SceneFitter.solve` for the actual solve) |
| `tests/test_catalog.py` | 49 lines | rebuilt on synthetic data: `Catalog.run()` produces finite `segment_flux`/`kron_flux`/`kron_radius`/`r50`/`sharpness`/`snr` with `snr == segment_flux/segment_fluxerr` | main rewrote it to a single 28-line deblend-provenance test with no measurement-column assertions |
| `pyproject.toml` | +9 | four markers (`slow`, `network`, `needs_data`, `benchmark`) with `addopts` deselecting all of them by default, and the markers actually applied — so a bare `pytest` runs only fast offline tests | main has **no** `[tool.pytest.ini_options]`, no `conftest.py`, no `pytest.ini`; it deleted the slow files instead |
| `tests/utils.py` | −262 | deleted `make_testdata` (hardcoded `/Users/ivo/Astro/PROJECTS/MINERVA/...` paths) and `check_project` (imports `astropy.nddate`, a typo that can never succeed) | both still present and uncalled at `main:tests/utils.py:991` and `:1212` |

---

## 6. Documentation

| doc | lines | content | main |
|---|---|---|---|
| `docs/aperture_corrections.md` | 602 (rev 3, 2026-07-15) | four root causes verified numerically on real F1500W products with reproduction snippets; the IDL reference recipe read out of `legacy/autopilot/mophongo__define.pro` and `old/dophot.pro`; published recipes (Wuyts 2008 Eq. 6, Skelton 2014 Eq. 1, Weibel 2024); the two total-flux conventions and which Python column may be compared against which IDL column; the estimator table; quantitative acceptance criteria; seven open items | main has `docs/FLUX_ESTIMATORS.md` (445 lines, same `apcor1`/`totcor1` algebra, explicitly notes "totcor1 … not computed") and `docs/PHOTOMETRY_APERTURES.md` (278). Neither tree's docs are a superset. main's are slightly stale against main's own code (`PHOTOMETRY_APERTURES.md` §4 says `num = aperture_sum(T_ref, r_cat)`; `main:pipeline.py:1124` uses `tmpl.data.sum()`) |
| `docs/stage4c_scope_and_brief.md` | 181 | cold-start implementer brief: numeric diagnosis, the algebra showing `f444w_ktot` cancels in `est3cat` so `template_norm` is the leak, four numbered rulings, the exact code change, five acceptance tests each required to fail against the unfixed code, and the single scoped pytest command that is the gate | absent; no equivalent format |
| `docs/dead_code.md` | 698 | tiered removal ledger (A safe / B decide-first / C critical-path / test-only / not-dead-but-flagged) with per-symbol `file:line`, reason, the exact commit to `git show` for recovery, and a "Coverage to restore" section naming three things the deleted tests covered that nothing covers now | main did its own cleanup (2ec4acb) with only a commit message |
| `docs/test_suite_cleanup_plan.md` | 356 | suite audit reorganised in `Pipeline.run` execution order with CRITICAL/CORE/SUPPORTING tags; central finding that effort had pooled at the wrong end (the aperture file was the healthiest and least critical while the keystone flux-recovery test was red because the entry point was broken); 30 failures in four buckets; ordered repair plan | absent |
| `CHECKLIST.md` | +59 | **two unresolved science findings live in main and documented nowhere in main** — see below | main **deleted** the file (−165) |

### 6.1 The two `CHECKLIST.md` findings

**(1) The pipeline is not reproducible.** `run_770.py` run three times with identical code, inputs
and config gives different scene counts and different membership (`5 / 5 / 6`), so all 22 numeric
columns differ. `flux_1` is unbiased in the median (0.12–0.38%) but ~6% of SNR>10 and ~5% of
SNR>50 sources shift by more than 1σ between identical runs — quoted errors therefore understate
the run-to-run spread and output catalogs can never be bit-compared. Suspected mechanism:
`scene_coupling_thresh` is a hard cut on ATA couplings, so roundoff-level variation from
multithreaded FFT/BLAS reduction order flips couplings near the cut.

**(2) The flux-block ridge biases faint sources low.** `SceneFitter.solve` uses
`flux_reg = 1e-6·median(A.diagonal())`, one absolute value per scene applied to every source, so
recovered flux goes as `d_i/(d_i + reg)`: measured **−0.05% at `d_i/median = 1e-3`, −0.5% at 1e-4,
−4.8% at 1e-5, −33% at 1e-6**. Extended Estimator-3 templates spread `Σ T²` thin enough to reach
those ratios. This mechanism is live in main too — main's default `reg_flux = 0.0` falls through
to the same adaptive `1e-6·scale_A` (`main:scene_fitter.py:180`) — main has only added the
robustness guards and a config override, not a fix. The `test_regularization_does_not_bias_flux`
guard that protected against a real past bias (commit 9d2ed2d) was deleted with `test_fit.py`.

### 6.2 `CLAUDE.md` structural divergence

main turned `CLAUDE.md` into a symlink to `AGENTS.md` (mode 120000) and rewrote `AGENTS.md`
(+165/−73) into a process/policy document. wren left `AGENTS.md` byte-identical to base and instead
updated `CLAUDE.md` as an architecture map documenting the post-retirement code and its deliberate
limitations: step 4 is "Scene Solving", `SparseFitter`/`GlobalAstroFitter` documented as retired,
regularisation documented as internal and not configurable, flux errors documented as
`1/sqrt(diag)` of the **whitened** normal matrix so they do not reflect off-diagonal covariance,
multi-component fitting documented as not implemented. A file-vs-symlink type conflict on merge.

---

## 7. Cleanups wren made that main has not (free, zero-risk)

Two functions wren deleted that **main still ships and that cannot run**:

- `main:src/mophongo/utils.py:123` `rebin_wcs(wcs, factor)` — first statement is `factor = 2**n`
  with `n` undefined and no module-level `n` (line 142). `NameError` on any call. Dead in `src`
  (definition only). Note the `tests/utils.py` copy takes `n` as its parameter and is fine.
- `main:src/mophongo/catalog.py:221` `calibrate_ivar_with_bg_median` — uses `bgmask` at line 271
  before its only binding at line 307, so `UnboundLocalError`. Its sole call site
  (`main:catalog.py:673`) is commented out in favour of `get_bg_and_ivar`.

Also worth taking: the `psf.py` dead-code deletions (§3.4), and wren's notebook fixes —
`main:examples/full_pipeline.ipynb:289` still has a live cell
`FitConfig(fit_astrometry_niter=0, solve_method='ata')`, which now raises `TypeError` on main
because both forks deleted `solve_method`. wren patched that cell.

**Divergence main should not follow**: none in this area. (The earlier claim that wren deleted
`catalog._expand_remap` was wrong — see §10.)

---

## 8. What main has that wren does not

Roughly 7400 lines of main-only source that wren cannot conflict with because the files do not
exist there, plus a set of algorithmic improvements inside shared files.

### Main-only modules

| module | lines | what |
|---|---|---|
| `saturate.py` | 1773 | saturated-core PSF repair: interior zero-weight hole finding, donut-ring significance pre-filter, joint STPSF amplitude + sub-pixel shift fit on an annulus by iteratively re-drizzling until `\|dx\|,\|dy\| < shift_tol`, core replacement and weight restoration, flat fit Table + diagnostics. (wren does keep base's `find_saturated_stars` **detector**, but no repair.) |
| `mock_mosaic.py` + `_mock_sip.py` | 1452 + 421 | pysiaf-derived per-detector footprints, grizli-compatible `_wcs.csv`, nested 20/40/80 mas mosaics, physically calibrated noise `σ = K/(p_out·sqrt(t_exp))` × the Fruchter-2011 drizzle correlation factor, injection through `DrizzlePSF.get_psf_radec`, truth tables with per-aperture flux fractions. Replaces base's `sim_data.py`, which wren still ships. |
| `verification.py` | 2061 | injected-truth harness: realistic two-detector NIRCam+MIRI mocks with native phase dithers, `wht_noise_check`, Wiener/Tikhonov regularisation grid search into PSF maps, detection→truth remapping, source-recovery tables, and the standard figures |
| `psf_factory.py` | 402 | telescope-backend registry (`PSFBackend` Protocol + `BACKENDS`), MJD-aware PSF generation, `dates_from_csv` deriving epochs from a grizli wcs.csv (modal MJD or `delta_day` clustering), MJD embedded in filenames, `DrizzlePSF._resolve_epsf_key` nearest-MJD lookup |
| `template_schemes.py` (on `template`, not main HEAD) | — | leaf module with `composite_wren` (faithful port of wren's `_extended_composite`), `composite_classic` (IDL `subphot.pro::build_cube`), `composite_psf_wings` (current default), behind `FitConfig.extend_mode` |

### Main-only capability inside shared files

- **PSF matching-kernel optimisation**: grid search over `SplitCosineBellWindow(alpha,beta)` and
  over scalar regularisation for tikhonov/wiener/forward, scored by squared EE mismatch + squared
  log core-profile mismatch + a kernel-stability penalty (signed-flux cancellation `C(K)²`), with
  a canonical multi-panel diagnostic PNG. `main:psf.py:1026`, `:1175`, `:1340`.
- **Regularised matching-kernel methods** in `utils.matching_kernel(method=...)`: Tikhonov
  (`K = conj(H_hi)H_lo/(|H_hi|² + λ·max|H_hi|²)`, dimensionless reg), Wiener, and ForWaRD
  (Neelamani, Choi & Baraniuk 2004: Tikhonov inverse → redundant stationary wavelet decomposition
  → per-subband noise variance propagated from the Tikhonov impulse response → hard threshold →
  optional wavelet-domain Wiener). `main:utils.py:442`, `:461`, `:516`.
- **PSF throughput / `*_total` columns**: `_filter_psf_throughput`, `_record_psf_ee` writing
  `EEBOX<i>`/`EECIRC<i>`/`RCIRC<i>`/`PSFSZ<i>` metadata, and
  `flux_<i>_total`/`err_<i>_total`/`err_pred_<i>_total`/`throughput_<i>`. **wren never had this** —
  the merge base has no throughput at all — so it is a main-side addition, not a wren deletion.
- **Config-driven runs**: `RunConfig` JSON dataclass (unknown keys raise), `Pipeline.from_config`,
  cached step methods (`build_psfs`/`build_kernels`/`load_data`/`info`/`run`/`write_outputs`/
  `run_all`), and `python -m mophongo.pipeline config.json [steps]`.
- **IDL `subphot` inspection panels** (`plot_subphot`, pixel-for-pixel `bytscl`/`robust_sigma`/
  `fptv` port) and **per-source stage diagnostics** (`diagnose_sources`, rebuilding a source
  through extraction → extension → convolution from scratch with snapshots after each op).
- **Parity-correct convolution**: `convolve_cutout` origin from `_origin_original_true − kernel//2`,
  `project_to_block_replicated_grid`, `downsample` warning on non-k-aligned origins,
  `_is_identity_kernel`, `_copy_template_overlap`, pinned by 10 tests.
- **Marginalised flux errors** (§4.6), **scene `max_size` bisection + `isolate_saturated`**,
  **`astrom_exclude_stars`** and merge-time isolation.
- **`lw_detection_coadd`** (PSF-matched inverse-variance LW detection image) and
  **`reconstruct_wcs`** (rebuild a missing grizli `_wcs.csv` from mosaic provenance by parallel
  S3/MAST header fetches).
- **Deblend provenance**: `_deblend_label_info` mapping each final label to its pre-deblend parent
  by majority pixel vote, `Template.deblend_parent_label`/`deblend_nchildren`.
- **`EffectivePSF._interp_grid`**: header-driven STDPSF grid interpolation replacing base's
  hard-coded knots (wren still hard-codes `[0,512,1024,1536,2048]`).
- **`PSFRegionMap._rebuild_spatial_index`** called from five sites, keeping `_geoms`/`_prepared`/
  `_keys` coherent; and `psfs = None` initialisation in `from_geojson` (`main:psf_map.py:249`),
  which wren lacks — wren's `from_geojson` raises `UnboundLocalError` when the `.fits` sidecar is
  missing.
- **`GaussianFit`/`MoffatFit`/`fit_moffat`/`fit_gaussian`**: main **deleted** these, and wren's
  `CircularApertureProfile.moffat_fit`/`moffat_fwhm` depend on them — taking main's `psf.py`
  wholesale silently breaks wren's `utils.py`.
- Ten main-only test modules (`test_mock_mosaic`, `test_moffat_recovery`, `test_lw_detection_coadd`,
  `test_pipeline_config`, `test_pipeline_inspect`, `test_repair_saturated_catalog`,
  `test_scene_max_size`, `test_scene_saturated`, `test_subphot_diag`, `test_template_convolution`).

---

## 9. Merge conflict risk

`git merge-tree --write-tree main 512db4b` (read-only, result tree `00c3f748`) produces
**30 conflicted paths**: 11 in `src/mophongo`, 13 test modules, plus `CLAUDE.md` as a
symlink-vs-file type conflict. Raw counts understate the problem — see the two notes after the
table.

| file | wren change | main change | risk | note |
|---|---|---|---|---|
| `templates.py` | `_extended_composite` + ownership + blend + `template_norm`/`n_pix` + `FLAG_PSF_EXTENDED`/`FLAG_EXTEND_FAILED` (0x40/0x80) + unit-sum renorm + positivity clip | `extend_with_psf_wings`/`_psf_model`, `FLAG_DEBLENDED`/`FLAG_SATURATED` (0x40/0x80), origin-parity convolution fix, `add_component`, `from_cutout_models`, `project_to_block_replicated_grid` | **catastrophic** | 8 markers, but the real damage is two incompatible construction schemes + the flag-bit collision + a silent auto-merge (below). A textual merge keeps BOTH schemes in one class. |
| `pipeline.py` | 4-estimator aperture system, `_model_kron`, containment plumbing, `_sources_with_coverage`, `r_fill` sizing | `RunConfig`/`from_config`/step methods/CLI, `_filter_psf_throughput` + `*_total`, `plot_subphot`, `diagnose_sources`, upsample ivar helper | **catastrophic** | 9 markers incl. a 450-line and a 253-line region. Both rewrote `run()`, `_add_aperture_photometry`, `_update_catalog_with_fluxes`, `plot_result`. They disagree on the **output contract**. |
| `fit.py` | `SparseFitter` deleted; 11 new config fields; 5 fields removed | `SparseFitter` kept + extended; `reg_flux`, `astrom_shift_tol`, `scene_max_size`, `template_dilate_segmap`, … | **catastrophic in kind, easy in mechanics** | 5 markers, one 260-line region that is simply "class exists" vs "class deleted", plus a 39-line `FitConfig` conflict. Policy call + manual union. |
| `utils.py` | window-only `matching_kernel`; 4 EE helpers; 4 extra kernel bases | `method=` dispatch (tikhonov/wiener/ForWaRD), `pixel_ratio` resampling, `lw_detection_coadd`, `reconstruct_wcs` family | **high** | 2 markers but one is 165 lines; both rewrote `matching_kernel` from the same base. Everything else is disjoint and appends cleanly. Defaults also differ (`recenter` False on main, True on wren). |
| `catalog.py` | ~500 lines deleted (`calibrate_ivar_with_bg_median`, detect/vet helpers, `CatConfig`) | +500 lines (saturation merging, ivar calibration, deblend provenance) | **medium-high** | 4 markers, largest 41 lines, at shared import/detect/run seams. Mostly mechanical. |
| `psf.py` | pure deletion (−250): `psf_matching_kernel_basis`, `NEffectivePSF`, `jwst_header`, NRC comment block | +2279: kernel optimisation, diagnostics, `stamp_encircled_energy`, `_interp_grid`, STDPSF tapering | **medium** | only 2 markers (32 + 84 lines), exactly where wren deleted code main extended. main deleted `GaussianFit`/`MoffatFit`, which wren's `utils.py` needs. |
| `scene_fitter.py` | `cg_kwargs` param removed, docstrings corrected | marginalised `sqrt(diag(A⁻¹))`, Schur complement, `reg_flux` override, numerical guards | **medium** | 2 markers in `solve()`. main's algorithm survives; only the signature/docstring/reg block collides. main's version is strictly better. |
| `psf_map.py` | `containment` field + geojson round-trip + `resolve_key`/`get_containment`; `__post_init__()` in `from_footprints` | `_rebuild_spatial_index()` helper called from five sites; `psfs = None` guard | **medium** | 2 markers; both touched the same five `self.tree = STRtree(...)` lines for different reasons. Both changes are wanted. ~20 min hand-merge. |
| `scene.py` | deleted `_bbox_overlap` and `Scene._overlaps`; `niter>0` astrometry gate | `max_size` bisection, `isolate_saturated`, `astrom_exclude_stars`, merge-time isolation, `create_scene_graph` | **low-medium but booby-trapped** | only 1 marker. Auto-merge produces a **NameError**, not a conflict: main's `create_scene_graph` calls the `_overlaps` wren deleted. |
| `astrometry.py`, `__init__.py` | same dead-code deletions as main; `FitConfig` export | same deletions; `fftconvolve` moved to utils | **low** | single import-block marker. Trivial. |
| `jwst_psf.py` | none (0-line diff vs base) | +618 (`JWSTBackend`, `build_jwst_psf`) | **zero** | clean. |
| `saturate.py`, `mock_mosaic.py`, `_mock_sip.py`, `verification.py`, `psf_factory.py` | files do not exist | main-only | **zero** | ~7400 lines safe under any strategy. |
| `photutils_deblend.py`, `sim_data.py` | modified (−362, −8) | **deleted** | modify/delete | main's deletion is the right call; wren keeps a vendored compact deblender never wired into `__init__`. |
| `tests/test_fit.py` | **deleted** | modified (kept 1 test) | delete/modify | follows the `SparseFitter` decision. |
| 12 more test modules | kept/repaired | deleted | modify/delete | wren repaired what main deleted. |
| `CLAUDE.md` | regular file, architecture map | symlink → `AGENTS.md` | type conflict | |

**Two things make this worse than the counts look.**

1. **Silent auto-merges.** `Template.convolve_cutout` merges with no conflict marker, interleaving
   main's true-origin/parity centring with wren's unit-sum renormalisation; the merged body then
   reads `self.n_pix` / `self.template_norm`, which exist only on wren's `Template.__init__` —
   inside a class whose attribute block *is* conflicted. At file level, the merged `templates.py`
   contains both `extend_with_psf_model`/`extend_with_psf_wings` and
   `_extended_composite`/`_build_ownership`/`blend_weight`. It roughly compiles and is
   scientifically meaningless.
2. **The 0x40/0x80 flag collision** (§2.8) has no mechanical resolution.

---

## 10. Claims that did not survive verification

Dropped or downgraded relative to the reading agents' inventories:

- *"Removal of the legacy module-level `extend_with_psf_wings` / `_sample_psf`"* — **already in
  main**; both forks removed the same base code. Not a differentiator.
- *"Per-source detection-PSF lookup in the cutout frame is a wren-only capability"* — **partial**.
  main's `_psf_for_template` already pairs cutout-frame WCS with cutout-frame pixels; the bug
  wren's commit fixes was wren-specific. Wren-only: the `id(psf_src)` cache, degrading to `None`
  instead of raising, and using the lookup inside extraction.
- *"`utils` EE helpers have no analogue in main"* — **partial**. main has `_encircled_energy`,
  `stamp_encircled_energy` and `_ee_fraction_to_arcsec`; wren-only is the pixel-domain array-in/
  float-out API plus `psf_ee_area_pix`.
- *"Real-flux bookkeeping invariant (`ap_model` from the fitted template)"* — **already in main**.
  `main:pipeline.py:1126-1128` already satisfies it; this was a wren-internal regression fix.
- *"wren dropped main's filter-level throughput and `*_total` columns"* — **direction wrong**.
  The base has no throughput; main *added* it after the split. The merge conflict is real, the
  framing is not.
- *"`docs/aperture_corrections.md` covers ground main has nothing on"* — **partial**. main has
  `FLUX_ESTIMATORS.md` and `PHOTOMETRY_APERTURES.md` over much of the same algebra.
- *"Stale-spatial-index fix in `from_footprints`"* — **already in main**, and more thoroughly
  (`_rebuild_spatial_index` from five sites). The claimed remaining exposure in wren's
  `group_by_pa`/`overlay_with` is also wrong: those construct new objects so `__post_init__` runs.
- *"Removal of duplicate `to_header`/`get_slice_wcs` from psf.py"* — **already in main**.
- *"Removal of the commented NRC bilinear block is cosmetic and main still has it"* —
  **misdescribed**. main removed it *and* replaced both branches with the header-driven
  `_interp_grid`, which is strictly more capable than wren's remaining hard-coded knots.
- *"Deletion of `astro_fit.py`"*, *"the fit.py OBSOLETE tail and CG/Cholesky stack"*,
  *"the SparseFitter solve path"*, *"orphan FitConfig fields"* — all **already in main**
  (convergent deletions, ~1250 shared lines). Only wren's notebook fixes in the last item are
  wren-only.
- *"Removal of the kernel-regularization experiments"*, *"dead PSF/kernel basis functions"*,
  *"duplicate `_mean_downsample`"*, *"`CatConfig`"*, *"dead astrometry helpers"*,
  *"`sim_data.Frame`"* — all **already in main**.
- *"wren deleted `catalog._expand_remap`, which main uses live"* — **false**. `_expand_remap` is
  alive on wren (`wren:catalog.py:308`); what both forks removed was a base **duplicate**
  definition. No divergence to avoid.
- *"Retained-on-wren kernel cropping (`prepare_kernel_info`) is a live capability"* — **downgraded**.
  Dead on wren too: the only caller is a test. main deleted unreachable code.
- *"Pipeline-level template-extension wiring is absent from main"* — **partial**. main dispatches
  `extend_templates` into its own extension routines at `main:pipeline.py:1249-1283` and `:1687`.
  Wren-only: the `FitConfig` mode knob, the cached growth curve, and aperture-derived `min_size`.
  (The claim's supporting citation to `template_schemes.py` on main was also wrong — that file is
  on the `template` branch, not main.)
- *"Extraction-time extension has four modes (data/psf/auto/none)"* — **misdescribed**. Wren
  collapses to one boolean (`extend = extend_mode != "none"`, `wren:templates.py:1366`); `data` and
  `psf` differ only in the SNR thresholds passed.
- *"main has no ownership rule at all"* — **partial**. main restricts wing fill to
  background-or-own-segment pixels (`main:templates.py:1226-1230`) — a membership test, not an
  area-weighted contest, and it does not partition background between neighbours.
- *"`tests/test_scene.py` is the first direct coverage of `generate_scenes`"* — **false**;
  `main:tests/test_scene_saturated.py` calls it directly at four sites. And its star test pins
  behaviour main deliberately made opt-in.
- *"`resolve_key` renames `lookup_key`"* — **false**; wren keeps both.
- *"main has no finite-stamp EE correction"* — **false**; `_filter_psf_throughput` +
  `stamp_encircled_energy` (`ee_circ` uses the same inscribed-disk convention).
- *"wren repaired a broken pipeline entry point that main abandoned"* — **premise inverted**.
  main repaired the wrapper in source (`Pipeline.__init__` accepts `wht_images`,
  `main:pipeline.py:340,351`; the wrapper returns a 3-tuple) and its own tests call it. It is
  **wren** that leaves the wrapper broken and routes tests around it.
- *"xfail-parking documents two live defects"* — **partial**. One (`Scene.create_scene_graph`
  undefined) is fixed on main. The `plot_result`/`self.fit` `IndexError` is real and shared —
  `main:pipeline.py:1499` still has the `self.fit.append(fitter)` commented out.
- *"Scene-solver test hardening brings new API"* — **partial**. `SimpleNamespace` returns,
  `model_image`/`residual`, and `positivity=False` in dense comparisons are all already in main.
  Wren-only: the nsrc 5→20 real-template dense reference and the `matrix_rank` premise guard.
- *"`psf_stamp_containment` measures the fraction of the PSF's true total flux"* — **physics
  wrong**; it normalises by the parent grid sum. It also has **no `src` caller on wren** —
  production populates the field from an external run script, so a port must add the call site or
  the correction silently stays at 1.0.
- *"`tests/utils.py` on main is byte-identical to base"* — **false**; main grew it (+259/−42). It
  simply never removed the two dead helpers.
- *"wren deleted 8 test modules main kept"* — **false**; base−wren is 5 modules and main kept only
  one of them. main is the branch that deleted many test modules wren kept.
- *"main still ships `tests/test_downsample.py`"* — **false**; both forks deleted it.
- *"`_model_kron(use_source_catalog=False)` is the `apcor_from_psf` performance shortcut"* —
  **downgraded**; no `src` caller at HEAD, tests only.
- *"`psf_ee_area_pix` gates template truncation"* — **downgraded**; imported but never called in
  wren's `src`.

---

## 11. Recommendation

Do **not** attempt a symmetric merge. Take `main` as the base — it carries the five conflict-free
modules, the config/CLI layer, the saturation chain, the mock/verification harness, the
marginalised errors and the parity-correct convolution — and land wren's better pieces as ports.

The `template` branch already demonstrates the right pattern: `composite_wren` in a leaf
`template_schemes.py` behind `extend_mode`, so wren's algorithm is adopted without adopting wren's
`templates.py`.

### Tier 1 — free, no dependencies, land immediately (≈1 day total)

| # | item | effort |
|---|---|---|
| 1 | `utils.psf_ee_radius_pix` + `psf_ee_at_radius` (+ `psf_ee_area_pix` if a caller is written) | ~50 lines, 1 h |
| 2 | Delete `utils.rebin_wcs` and `catalog.calibrate_ivar_with_bg_median` — both unrunnable, both dead | 15 min |
| 3 | `_aperture_sum_on_template` geometry fix (measure on `tmpl.data[slices_cutout]` in the shifted frame) — main is currently inconsistent at image edges | 30 min |
| 4 | `PSFRegionMap.resolve_key` refactor | 30 min |
| 5 | pytest marker taxonomy in `pyproject.toml` + apply the four markers | 1 h |
| 6 | `Templates.min_size` as an instance attribute + `min_size_from_aperture` | 1 h |
| 7 | `np.searchsorted` for the segm bbox lookup (3.9 → 0.05 ms/source) | 15 min |
| 8 | Port the two `CHECKLIST.md` findings into main's `TODO.md`/`STATUS.md` — both mechanisms are live in main and documented nowhere | 30 min |
| 9 | `psf.py` dead-code deletions (`psf_matching_kernel_basis`, `NEffectivePSF`, `jwst_header`, `jwst_probe_headers`) and the `full_pipeline.ipynb:289` `solve_method='ata'` fix | 30 min |

### Tier 2 — cheap, high-value tests (≈1 day)

| # | item | effort |
|---|---|---|
| 10 | `tests/test_downsample_flux.py` verbatim — main has both targets and **zero** downsample coverage | 30 min |
| 11 | `tests/test_utils.py` `bin_factor_from_wcs` half + the EE round-trip | 1 h |
| 12 | The four `test_psf_map.py` containment/`resolve_key` tests, plus restoring the three base tests main dropped | 2 h |
| 13 | `tests/test_scene.py` partition test (**drop** the star test — main's default differs) | 1 h |
| 14 | The `test_pipeline.py` keystone assertions: background chi² floor `0.85 < χ² < 1.15` and the <10% per-segment z-score criterion | 2 h |
| 15 | The end-to-end astrometry wiring test (assert on accumulated `Template.shifted`, not the residual) | 2 h |

### Tier 3 — template construction (≈1–2 weeks)

| # | item | effort |
|---|---|---|
| 16 | `blend_weight` + `_region_snr` + `_background_sigma` as leaf functions | 0.5 day |
| 17 | `_build_ownership` + `_disk_kernel` + `_cutout_roi` — the area-weighted disjoint partition. Substantially already written as `template_schemes.build_ownership` on the `template` branch | 2 days |
| 18 | `template_norm` + `n_pix` bookkeeping through `convolve_cutout`/`downsample`, **grafted onto main's recentred `convolve_cutout`, not replacing it** | 2 days |
| 19 | `_extended_composite` as a scheme in `template_schemes.py`, dispatched by `extend_mode` — do not touch main's `extend_with_psf_wings`. **Renumber the flag bits** (main owns 0x40/0x80). | 3 days |
| 20 | `_sources_with_coverage` pre-extraction cut and pipeline `r_fill` sizing | 2 days |
| 21 | `tests/test_template_extension.py`, adapted to main's flag numbering and scheme dispatch | 2 days |

### Tier 4 — total-flux system (decision first, then ≈3–4 weeks)

**Before any code**: settle whether the true-total correction is owned per filter
(main's `_filter_psf_throughput` on absolutely calibrated drizzled stamps) or per PSF region
(wren's `containment` as a parent-grid ratio). Applying both double-corrects. main's denominator is
better grounded; wren's granularity is better. A hybrid — main's absolute stamp measurement stored
per region on `PSFRegionMap` and applied to correction factors rather than to amplitudes — gets
both, and is the option I would take.

| # | item | effort |
|---|---|---|
| 22 | `PSFRegionMap.containment` + `get_containment` + geojson round-trip with the loud default, **keeping main's `psfs = None` guard and `_rebuild_spatial_index`**, and wiring the producer into PSF construction (wren never did — it populates the field from an external script) | 3 days |
| 23 | `flux_beyond_stamp` on `Template` + the `trunc`/`trunc_denom` chain; then `est1`/`est2` with per-estimator errors | 1 week |
| 24 | `flux_beyond_aper` (Stage 4b crowding delta) with the bbox pre-check and the exact-mask fast path | 3 days |
| 25 | `_model_kron` + `tcor_int` + `s_cat` + `est3int`/`est3cat`, plus the `f444w_*` config knobs | 1.5 weeks |
| 26 | `tests/test_pipeline_aperture.py` adapted, and the §1.11 measured numbers used as acceptance gates | 1 week |
| 27 | Port `docs/aperture_corrections.md` and `docs/stage4c_scope_and_brief.md`, reconciled against main's `FLUX_ESTIMATORS.md` | 2 days |

### Tier 5 — policy calls, no urgency

- **Retire `SparseFitter`** on main as wren did. Its assembly half has real users only in
  `tests/test_astrometry.py`, which wren already showed how to repoint via
  `_scene_flux_and_residual`. Removing it also removes the `normal`/`cg_kwargs`/`run_scene_solver`
  dead config. ~1 day, mostly test churn.
- **Delete `fit_astrometry_joint`** and gate on `niter > 0` alone — but note five live call sites
  in main's `examples/` and `verification.py`. ~2 h.
- **Fix the three stale `scene_fitter.py` docstrings** still referencing conjugate gradients and
  the deleted `SparseFitter.build_normal_tree`. 15 min.
- **Decide on `add_component`**: main keeps it with zero callers. Either wire it or drop it.
- **Adopt wren's `docs/dead_code.md` ledger format** for main's own future cleanups.

### Not worth porting

wren's diagonal-only flux errors (main's marginalised version is better), wren's
`Scene._overlaps` deletion (main uses it), wren's `psf.py` `GaussianFit`/`MoffatFit` retention
(main deleted them deliberately — but check `CircularApertureProfile`), the `tcor_H` machinery
(already removed on wren, never on main), and wren's dead ends: `_intersect_slices`,
`_get_representative_kernel`, `f444w_totcor_col`, the CWD-relative `f444w_template_residual.fits`
write.
