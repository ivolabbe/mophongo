# Photometry and Aperture Schemes

How mophongo turns a fit into fluxes, what each output column means, and what it
needs from an external (aperpy) catalog. Companion to
[PSF_SHAPE_THROUGHPUT_CONVENTION.md](./PSF_SHAPE_THROUGHPUT_CONVENTION.md),
which defines the shape/throughput split this document assumes.

Code references are `file:line` against the state of `flux-bug` at the time of
writing.

## 1. Three flux estimates

A run produces three different numbers per source per fitted band. They answer
different questions and are not interchangeable.

| column | what it is | use for |
|---|---|---|
| `flux_<i>` | fitted template amplitude, finite-PSF-support basis | model comparison, χ², internal QA |
| `flux_<i>_total` | `flux_<i>` / PSF throughput | total source flux, compare to truth catalogs |
| `ap_flux_corr_<i>` | aperture flux on (model + residual), corrected to the reference-band PSF | colours and SEDs against a matched-aperture catalog |

`err_<i>` / `err_pred_<i>` accompany the first two. The aperture scheme has **no
error column** — see §6.

Sources with no fitted template keep `FitConfig.bad_value`, which is `NaN`
(`fit.py:54`).

## 2. Grid conventions, and why they matter here

Everything in the aperture code is done in *pixels*, so the grid a quantity
lives on is load-bearing.

Templates are extracted from the high-resolution reference image (`images[0]`,
40 mas for MINERVA) and normalized to unit sum at extraction
(`templates.py:1549-1551`). "Unit sum" means the flux inside the (dilated)
segment; the fitted amplitude is therefore the total flux of that segment's
model, not of an aperture.

For a lower-resolution band the pipeline defaults to
`multi_resolution_method="upsample"` (`fit.py:83`):

1. the band image and its inverse variance are block-replicated onto the
   reference grid, science with `conserve_sum=True` and ivar copied then scaled
   by `k**2` (`pipeline.py:115-132`);
2. **`wcs[ifilt] = wcs[0]`** (`pipeline.py:1207`) — the band's WCS is replaced
   by the reference WCS, because the band image now lives on the reference grid;
3. templates are convolved with the matching kernel and projected onto the same
   block-replicated grid (`pipeline.py:1228-1234`).

Consequence: after step 2, `self._pixel_scale_arcsec(self.wcs[ifilt])` returns
the *reference* pixel scale, and every radius computed from it is in reference
pixels. The image aperture and the catalog aperture end up on the same grid, so
they are directly comparable. In the `downsample` path this rebinding does not
happen and the two are on different grids — see §6.

## 3. Scheme A — template-fit flux

The solver returns one amplitude per template. `_update_catalog_with_fluxes`
(`pipeline.py:733-808`) writes them out, with two behaviours worth knowing:

- **Deblend children are summed onto the parent.** Rows are keyed by
  `tmpl.id_parent` when present, so a catalog deblended into children reports
  one summed flux on the parent id, with errors added in quadrature
  (`pipeline.py:784-796`).
- **Throughput is applied as a division.** `flux_<i>_total = flux_<i> /
  throughput`, and identically for both error columns.

`throughput` is one scalar per filter from `_filter_psf_throughput`
(`pipeline.py:223-252`): the mean sum of the finite PSF stamps the run actually
used, post-drizzle and post-broadening. It is the square-stamp encircled energy
`ee_box`. Callers passing unit-sum shapes must supply `explicit_throughput`,
otherwise the correction silently becomes 1.

`_record_psf_ee` (`pipeline.py:255-298`) writes the reference numbers into
`cat.meta` as eight-character keys so they survive a FITS header:

| key | meaning |
|---|---|
| `EEBOX<i>` | realized EE of the full square stamp = the applied `throughput` |
| `EECIRC<i>` | realized EE inside the stamp's inscribed circle |
| `RCIRC<i>` | that inscribed-circle radius, arcsec |
| `PSFSZ<i>` | delivered PSF stamp side, arcsec |

`ee_box >= ee_circ` always, by the corner flux (`psf.py:2113`).

### Errors

Two flavours:

- `err_<i>` — from the solver, includes template covariance (neighbour
  degeneracy inflates it);
- `err_pred_<i>` — `Templates.predicted_errors` (`templates.py:1430-1445`),
  `1/sqrt(sum(w * T^2))`, ignoring covariance.

Their ratio is the standard calibration diagnostic. On the MINERVA F770W trial
patches it sits at 1.000 (p16/p50/p84 = 0.9997/1.000/1.001), meaning crowding is
not inflating the formal errors there.

## 4. Scheme B — aperture photometry on model + residual

`_add_aperture_photometry` (`pipeline.py:939-1018`) exists because template
fluxes are model-dependent: they assume the reference-band morphology is
correct at the fitted wavelength. An aperture measured on `model + residual`
recovers real flux the template shape did not predict, at the cost of
neighbour contamination.

Per source:

```text
patch     = residual[slices] + flux * template[slices]      # model + residual
ap_raw    = aperture_sum(patch, r_img)                      # ap_flux_<i>
den       = aperture_sum(T_conv, r_img)                     # this band's template
num       = aperture_sum(T_ref,  r_cat)                     # reference-band template
corr      = num / den                                       # ap_corr_<i>
ap_corr   = ap_raw * corr                                   # ap_flux_corr_<i>
```

`ap_model_<i> = flux * den` is written too, as the model-only aperture flux for
comparison.

Read `corr` as: *rescale an aperture flux measured at this band's resolution
into the aperture flux the same source would have at the reference band's
resolution.* Both `T_ref` and `T_conv` are unit-sum, so `num` and `den` are
encircled-energy fractions of the source model, not of a point-source PSF —
morphology is included, which is the point.

Worked numbers from the DR0.1 COSMOS F770W run (`flux_1 > 5`, n=119), recovered
from the output columns:

| quantity | median |
|---|---|
| `den` = `ap_model_1`/`flux_1` (F770W-resolution EE in 0.5″) | 0.468 |
| `num` = `ap_corr_1` × `den` (F444W-resolution EE in 0.5″) | 0.662 |
| `ap_corr_1` | 1.393 |

### The two radii

`r_img` — `_resolve_image_ap_radius_pix` (`pipeline.py:831-875`). One value per
band, from `FitConfig.aperture_diam`:

- scalar → that diameter for all bands;
- `ndarray` of length `len(images)-1` → per band;
- `None` → 1.5 × the Gaussian-equivalent FWHM of that band's PSF, falling back
  to 3.0 px if no PSF is available.

`aperture_units` ("arcsec" or "pix") controls the conversion. The resolved value
is echoed as the column `aper_<i>` in arcsec diameter (`pipeline.py:1407-1413`).

`r_cat` — `_resolve_catalog_ap_radius_pix` (`pipeline.py:877-929`). Per source,
in reference-image pixels, from `FitConfig.aperture_catalog`:

- `str` → a catalog column of per-source **diameters** (this is the aperpy hook);
- `float` → one fixed diameter for all sources;
- `None` → the caller's `r_default`, which `_add_aperture_photometry` sets to
  `r_img` (`pipeline.py:973`).

With `aperture_catalog=None`, `r_cat == r_img` and `corr` reduces to a pure
PSF-matching correction within a single fixed aperture. That is what both
MINERVA runs did. To produce numbers comparable to an adaptive-aperture catalog
you must set `aperture_catalog` explicitly.

## 5. What is needed from an aperpy catalog

Verified against `MINERVA-COSMOS_n3.0_v1.3_LW_Kf444w_SUPER_CATALOG.fits`
(197198 rows, 164 columns).

### Required now

| need | source | DR0.1 value |
|---|---|---|
| join key = segmap label | `id` | — |
| reference-grid positions | `x`, `y` | — |
| sky positions (footprint + trial-patch cuts, `pipeline.py:562-585`) | `ra`, `dec` | — |

### Required to reconstruct matched-aperture photometry

| need | source | DR0.1 value |
|---|---|---|
| aperture actually used, **diameter in arcsec** | `use_aper` | {0.20, 0.32, 0.48, 0.70, 1.00, 1.40} |
| PSF everything is homogenized to | meta `KERNEL` | `f444w` |
| arcsec ↔ pixel tie | meta `PIXSCALE` | 0.04 |
| aperture → total | `tot_cor` | median 2.147 |
| flux scale | meta `PHOT_ZP`, `PHOT_UNIT` | 28.9, `10*nJy` |
| usable subset | `use_phot`, `flag_star`, `flag_artifact`, `flag_kron`, `flag_lowsnr` | — |

Two notes on these:

- `use_aper` is already an arcsec diameter, so `aperture_catalog="use_aper"` with
  `aperture_units="arcsec"` needs no preprocessing. `CatConfig.aperture`
  (`catalog.py:628`) defaults to the string `"use_aper"`, but that dataclass is
  not wired into the pipeline path.
- `tot_cor` must be taken, not recomputed. It is not a plain Kron ratio:
  `fauto_KRON/faper_KRON` has median 1.853 against `tot_cor` median 2.147, a
  ratio of 0.869.

`Table.read` preserves the aperpy `meta`, and `cat = catalog.copy()[keep_cols]`
carries it through, so `KERNEL`/`PHOT_ZP`/`PIXSCALE` already appear in the
output FITS header. Verified on `cosmos_770_fit_table.fits`.

### Not needed

`kron_radius`, `kron_radius_circ`, `a_image`, `b_image`, `iso_area`,
`flux_radius`, `theta_J2000`. Useful as QA or priors, but no correction reads
them.

## 6. Sharp edges

**Optional columns are also consumed.** Beyond `id`/`x`/`y`, `run()` picks up
`is_deblended`, `deblend_parent_label`, `deblend_nchildren`
(`pipeline.py:1076-1080`) and any `FLAG_SATURATED_*` column
(`pipeline.py:1081-1082`), the latter to isolate saturated sources into their
own scenes.

**The output table cannot build an SED on its own.** `run()` cuts to
`keep_cols = ["id","x","y"]` (`pipeline.py:1075-1083`). `ra`, `dec`,
`use_aper`, `tot_cor`, `use_phot` and all quality flags are dropped, so any
downstream colour work has to re-join to the source catalog by `id`.

**`KERNEL` consistency is unchecked.** The `num` term uses the *pre-convolution*
reference template (`pipeline.py:1000`), which equals the aperpy-matched
resolution only because `KERNEL == f444w` and `sci_hi` happens to be F444W.
Using the chi-mean catalog, or a non-F444W `sci_hi`, would silently invalidate
`ap_corr_<i>`. Worth an assertion.

**Mixed radius grids in the `downsample` path.** `_resolve_catalog_ap_radius_pix`
documents its return as reference-image pixels (`pipeline.py:881`), and
`_add_aperture_photometry` defaults it to `r_img`. In the `upsample` path this
is correct because of the WCS rebinding in §2. In the `downsample` path
`wcs[ifilt]` keeps the native band scale while `T_ref` stays on the reference
grid, so the default `r_cat` is wrong by the bin factor. Only the `None` branch
is affected; an explicit `aperture_catalog` converts through the reference pixel
scale and is safe either way.

**The documented `None` fallback for `r_cat` is dead code.** `pipeline.py:886`
promises 1.5 × FWHM of `PSF[0]`, but the only caller always passes a non-`None`
`r_default`, so that branch never runs.

**No aperture errors.** `ap_flux_<i>` and `ap_flux_corr_<i>` have no companion
uncertainty. The residual inside the aperture is correlated (block-replicated
ivar, neighbour models), so `err_pred` is not a valid substitute.

**`num` is recomputed per band.** It depends only on the reference template and
`r_cat`, both band-independent, yet it sits inside the per-band source loop
along with a `ref_tmpls` dict rebuilt each band (`pipeline.py:976, 1000-1004`).

## 7. Suggested refactor

Move the band-independent part into a preprocessing step that augments the
catalog once per `(catalog, segmap, sci_hi)`:

```text
columns: id, x, y, ra, dec,
         use_aper, r_aper_pix,        # arcsec -> reference pixels, once
         f_aper_tmpl,                 # aperture sum of the reference template
         f_tmpl_tot, aper_frac_tmpl,  # normalization + fraction, for QA
         tot_cor, use_phot, flag_*
meta:    aperpy meta verbatim, plus HIRESFLT and an assert KERNEL == HIRESFLT
```

The fit-time correction then collapses to one line per source:

```python
corr = cat["f_aper_tmpl"][row] / den
```

That removes `_resolve_catalog_ap_radius_pix` (53 lines of
`float`/`str`/`None` × arcsec/pix branching), the `ref_tmpls` dict, and the
reference-template aperture call from the hot loop. It also eliminates the
mixed-grid hazard by construction, since the arcsec → pixel conversion happens
once, on the reference grid, where the scale is unambiguous.
`FitConfig.aperture_catalog` and `aperture_units` can then be retired, leaving
`aperture_diam` as the only aperture knob at fit time.

Module boundaries (see `CLAUDE.md`): the aperture-sum primitive belongs in
`templates.py` (e.g. `Templates.aperture_sums(radii_pix)`), with a thin
`prepare_catalog()` writer in `pipeline.py` as a step between `load_data()` and
`run()`. Keeping it out of `catalog.py` avoids preprocessing importing the
photometry side.
