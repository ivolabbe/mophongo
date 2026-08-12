# Preprocessing: saturation repair and astrometry

Two preprocessing concerns are handled outside the photometry pipeline proper:
repairing saturated stellar cores in a drizzled mosaic
(`mophongo.saturate`), and measuring and applying spatially varying
astrometric shifts between images (`mophongo.astrometry`). Both modules
operate on plain arrays and astropy objects, so they can be run before or
independently of the {doc}`pipeline`.

## Saturation repair (`mophongo.saturate`)

For the ready-to-run command-line front end (`mophongo-repair`), which
wraps this module and the catalog flagging step, see {doc}`repair`. This
section describes the underlying image-level algorithm.

Saturated stars appear in drizzled JWST/HST mosaics as interior regions of
zero weight ("holes") surrounded by bright PSF wings. `mophongo.saturate`
detects these holes, fits the local PSF amplitude on a donut-shaped ring
around each hole, and either fills the core with the best-fit PSF model
(`mode="repair"`) or subtracts the scaled PSF from the whole cutout
(`mode="subtract"`).

The module is image-only by design: its inputs are a science/weight pair, a
WCS, and a {class}`mophongo.psf.DrizzlePSF`. It does not import segmentation
maps or catalogs. The output table carries `id, xc, yc, r_equiv` per hole,
so a separate catalog step
({func}`mophongo.catalog.merge_segments_at_holes`) can collect the
segmentation labels around each repaired star and relabel the map
afterwards. See {doc}`catalog` for that step and
{doc}`psf` for building a `DrizzlePSF`.

```python
from astropy.io import fits
from astropy.wcs import WCS
from mophongo.saturate import repair_saturated_holes

sci = fits.getdata("image.fits")
wht = fits.getdata("weight.fits")   # inverse-variance weights
wcs = WCS(fits.getheader("image.fits"))

result = repair_saturated_holes(
    sci, wht, dpsf=dpsf, wcs=wcs,
    output_csv="saturate_fits.csv",
    plot_dir="saturate_png",
)
sci_repaired = result["sci"]
wht_repaired = result["wht"]
fit_table = result["fits"]
```

### How the repair works

For each interior hole with equivalent radius `r_equiv`, two radii are
derived:

- `r_in = r_equiv + buffer`
- `r_out = max(2 * fwhm_pix, factor * r_in)`

`r_out` is the outer radius of the fitting region. Its inner boundary is the
dilated zero-weight footprint rather than the `r_in` circle, so the excluded
core follows the actual saturation shape; `r_in` is used for the recorded
donut `significance` and is reported in the output table. The per-hole
sequence is:

1. **Saturation pre-filter.** The median sky-subtracted flux in the thin
   annulus just outside the hole (the "buffer" pixels created by dilating the
   hole mask) is compared against the global sky noise
   (`mad_std` of valid pixels). Holes below `min_buffer_snr` sigma are
   rejected as cosmic-ray hits or low-coverage bays rather than saturation.
2. **Joint amplitude and sub-pixel shift fit.** The ePSF is drizzled onto the
   cutout WCS at the current position, and a linearised fit of
   `data ≈ A·ψ + B·∂ψ/∂x + C·∂ψ/∂y` on the donut recovers the amplitude and
   a position update `(dx, dy) = (-B/A, -C/A)`. Drizzle and fit iterate
   until the shift is below `shift_tol`, with the cumulative shift capped at
   `max_shift_pix`.
3. **Quality checks.** A residual-fraction test (`Σ|data − A·ψ| / Σ|A·ψ|`
   over the fit ring) rejects fits where the model does not describe the
   data; when the donut flux exceeds the model by more than
   `extended_max_data_to_model`, the fit is repeated with an additive
   pedestal so that an extended host galaxy does not bias the point-source
   amplitude (the pedestal is reported, never subtracted).
4. **Action.** In `repair` mode the dilated hole pixels are replaced with
   `A·ψ` and their weights restored to the median donut weight. In
   `subtract` mode `A·ψ` is subtracted over the full cutout (wings,
   diffraction spikes) and the zero-weight core plus strongly discrepant
   residual pixels are blanked (`sci = wht = 0`) so downstream photometry
   skips them.

```{figure} images/saturation_repair_diagnostic.png
:width: 100%
:alt: Two rows of five panels showing the data, scaled PSF model, residuals, hole and fit-ring overlays, repaired image, radial profile, and polar residual map for one repaired saturated star.

Per-source repair diagnostic for a saturated star in an F444W mosaic, as
rendered by `plot_repair_diagnostic`. Top row: data, shifted PSF model
`A·ψ`, shifted and no-shift residuals, and a radially calibrated SNR map.
Bottom row: hole (red) and fit-ring (cyan) overlays, the repaired image
with the filled region outlined, a 2x zoom, the radial profile of data,
model, and repaired image, and a polar-remapped residual SNR map. The
donut fit reproduces the PSF wings (residual fraction 0.17), and the
repaired core continues the profile smoothly across the filled region.
```

### `repair_saturated_holes`

{func}`mophongo.saturate.repair_saturated_holes` is the main entry point.
Signature:
`repair_saturated_holes(sci, wht, *, dpsf, wcs, holes=None, ...)`.

Positional arguments:

`sci`, `wht` (`np.ndarray`)
: 2D science and weight arrays. Weights are inverse variance; only
  `wht > eps_wht` pixels are considered valid.

Keyword-only arguments:

`dpsf` ({class}`mophongo.psf.DrizzlePSF`)
: PSF model configured with the same drizzle WCS as `sci` and an ePSF
  already loaded. Required.

`wcs` (`astropy.wcs.WCS`)
: WCS of `sci`. Required.

`holes` (`Table | None`, default `None`)
: Precomputed hole table from
  {func}`mophongo.saturate.find_wht_holes`. `None` runs hole detection
  internally.

`buffer` (`float`, default `2.0`)
: Pixels added to `r_equiv` to form the donut inner radius `r_in`.

`factor` (`float`, default `2.5`)
: Multiplier on `r_in` for the donut outer radius `r_out`.

`fwhm_pix` (`float`, default `2.0`)
: PSF FWHM in pixels; sets the floor `r_out >= 2 * fwhm_pix`.

`eps_wht` (`float`, default `0.0`)
: Pixels with `wht <= eps_wht` count as zero-weight.

`return_diagnostics` (`bool`, default `True`)
: Collect per-source `RepairDiagnostic` objects in the return value.

`only_ids` (`list[int] | None`, default `None`)
: Restrict processing to these hole ids.

`fit_shift` (`bool`, default `True`)
: Fit the sub-pixel PSF position jointly with the amplitude. `False` for
  amplitude-only fits.

`max_shift_iter` (`int`, default `5`)
: Maximum drizzle-and-fit iterations for the position refinement.

`shift_tol` (`float`, default `0.05`)
: Convergence tolerance on the per-iteration shift, in pixels.

`merge_radius` (`int`, default `3`)
: Dilation radius (pixels) used during hole detection so disconnected
  fragments of one saturated core share a label.

`sat_significance` (`float`, default `10.0`)
: Minimum median-donut significance in sigma above sky. Currently not
  applied as a filter: the measured significance is recorded in the output
  and the acceptance filter uses `min_buffer_snr` instead.

`hole_dilate` (`int`, default `2`)
: Dilation (pixels) of the zero-weight mask; the dilated footprint defines
  the repair region and the inner boundary of the fit ring.

`max_resid_frac` (`float`, default `1.0`)
: Intended threshold for the residual fraction. Currently not consulted:
  the applied guard is hard-coded to reject fits with residual fraction
  above 1.0 (no action taken on the image).

`min_ring_snr` (`float`, default `5.0`)
: Intended minimum median ring SNR. Currently not applied: the measured
  ring SNR is stored in the diagnostics only.

`min_buffer_snr` (`float`, default `200.0`)
: Minimum median sky-subtracted flux of the buffer pixels, in units of the
  global sky noise, for the hole to be treated as genuine saturation. The
  main pre-filter.

`max_shift_pix` (`float`, default `3.0`)
: Hard cap on the cumulative fitted position shift, in pixels.

`extended_max_data_to_model` (`float`, default `1.15`)
: If `Σ data / Σ (A·ψ)` over the donut exceeds this ratio, refit with an
  additive pedestal to absorb host-galaxy flux.

`mode` (`str`, default `"repair"`)
: `"repair"` fills the saturated core with the model; `"subtract"` removes
  `A·ψ` over the full cutout and blanks the core.

`psf_size_pix_subtract` (`int | None`, default `None`)
: Cutout size in drizzle pixels for subtract mode. `None` uses
  `min(400, native ePSF field of view)`.

`psf_filter`, `psf_pixfrac`, `psf_kernel` (default `None`)
: Forwarded to {meth}`mophongo.psf.DrizzlePSF.get_psf`. `None` falls back
  to the `dpsf` defaults (drizzle-header `PIXFRAC`/`KERNEL`).

`sky_sample` (`int`, default `200000`)
: Number of valid pixels sampled to estimate the global sky level and
  `mad_std` noise. `0` uses all pixels.

`output_csv` (`str | Path | None`, default `None`)
: If given, write the fit table to this CSV path.

`plot_dir` (`str | Path | None`, default `None`)
: If given, write per-source diagnostic PNGs to this directory.

Returns a dict with keys `"sci"` (repaired science image, `float32`),
`"wht"` (repaired weight map), `"fits"` (astropy `Table`),
`"diagnostics"` (list of `RepairDiagnostic`), `"holes"` (hole table),
`"sky"`, and `"sky_noise"`.

The `fits` table has one row per hole, including rejected ones, with
columns `id, yc, xc, r_equiv, r_in, r_out, amplitude, amp_err, chi2_red,
n_pix, n_iter, shift_x, shift_y, significance, buffer_snr, flux_added,
pedestal, fit_mode, data_to_model, amplitude_noshift, chi2_red_noshift,
ok, status`. `ok=False` rows record why a hole was skipped in `status`.

### Helper functions

`repair_saturated_holes` drives these internally; call them directly only
when building a custom repair flow. See the {doc}`api` reference for full
signatures.

{func}`mophongo.saturate.find_wht_holes`
: Locates interior zero-weight regions — connected components of
  `wht <= eps_wht` that do not touch the image border, so chip gaps and
  out-of-field regions are dropped automatically — optionally dilating by
  `merge_radius` so fragments of one saturated core share an id. Returns
  the hole table (`id, yc, xc, area, r_equiv`, bounding box).

{func}`mophongo.saturate.fit_psf_donut`
: Fits `data ≈ A·psf [+ C]` by weighted linear least squares on the ring
  `r_in <= r <= r_out`, returning the amplitude, its error, reduced
  chi-square, optional pedestal, and the shape-mismatch diagnostic
  `rho_psf` (an amplitude-invariant weighted data–PSF correlation over the
  ring).

{func}`mophongo.saturate.fit_amp_and_shift`
: One linearised step of the joint amplitude + sub-pixel shift fit of step
  2 above; the caller applies the recovered shift, re-drizzles the ePSF,
  and calls again until convergence.

{func}`mophongo.saturate.refine_center_from_donut`
: Refines a source center by iterating a flux-weighted centroid of the
  background-subtracted donut; PSF wings are azimuthally symmetric about
  the true center, so this converges even for asymmetric holes.

{func}`mophongo.saturate.plot_repair_diagnostic`
: Renders the two-row, five-panel per-source diagnostic figure shown above
  from one `RepairDiagnostic`; pass `to_file` to save a PNG instead of
  returning the matplotlib figure.

### `RepairDiagnostic` fields

`RepairDiagnostic` is a result container, collected when
`return_diagnostics=True` for holes that reached the fitting stage. Holes
rejected by the buffer-SNR pre-filter or by the residual-fraction guard get a
table row but no diagnostic; a hole whose fit failed gets a stub diagnostic
with `ok=False`. Scalar fields: `id`, `yc`, `xc`, `r_equiv`, `r_in`, `r_out`
(geometry), `amplitude`, `chi2_red`,
`n_pix`, `n_iter`, `shift_total`, `significance`, `center`,
`amplitude_noshift`, `chi2_red_noshift`, `center_noshift` (fit results,
with a no-shift comparison fit at the original hole centroid),
`resid_frac`, `ring_snr`, `buffer_snr`, `pedestal`, `rho_psf`,
`fit_mode` (`"donut"` or `"donut+pedestal"`), `data_to_model`,
`action_mode` (`"repair"` or `"subtract"`), `ok`, `status`. Array fields
hold the cutouts and masks used by the plot: `sci_cut`, `wht_cut`,
`psf_cut_scaled`, `sci_repaired_cut`, `hole_mask`, `dilated_hole_mask`,
`ring_mask`, `repair_mask`, `psf_cut_noshift_scaled`,
`ring_mask_noshift`, `bad_resid_mask`.

## Astrometric corrections (`mophongo.astrometry`)

Ground-truth astrometry between a detection image and a measurement image is
rarely perfect at the milliarcsecond level, and residual distortions vary
across the field. `mophongo.astrometry` measures local shifts at source
positions and fits a smooth 2D shift field — a Chebyshev polynomial or a
Gaussian process — that can be evaluated anywhere in the image. Two facades
exist:

- {class}`mophongo.astrometry.AstroCorrect` measures shifts from the fit
  residual during template fitting and shifts the templates in place. In the
  current scene-based {doc}`pipeline` the per-pass shifts are solved by the
  scene fitter itself (`FitConfig.fit_astrometry_niter` sets the number of
  passes), which uses `AstroCorrect.build_poly_predictor` to evaluate its
  polynomial shift solutions; `AstroCorrect.fit` is a standalone entry
  point. See {doc}`fitting` for the configuration fields.
- {class}`mophongo.astrometry.AstroMap` measures shifts between two images
  directly at catalog positions, independent of any fit.

### How shifts are measured

For each sufficiently bright template, the local model
(`amplitude x template + residual`) is compared against the template
itself in a small cutout. The shift is either the difference of quadratic
centroids (with center-of-mass fallback) or the sub-pixel peak of the
cross-correlation, depending on the chosen method. Each measurement is
weighted by SNR squared. The collected `(position, dx, dy, weight)` samples
are then fit with a smooth field model, and templates are resampled by
cubic-spline interpolation (`scipy.ndimage.shift`) with the applied shift
accumulated in `Template.shifted`.

### `AstroCorrect`

Dataclass with a single constructor field:

`cfg` ({class}`mophongo.fit.FitConfig`)
: Supplies the astrometry options: `astrom_model` (`"poly"` or `"gp"`),
  `astrom_centroid` (`"centroid"` or `"correlation"`),
  `snr_thresh_astrom`, and `astrom_kwargs` (per-model keyword dicts, e.g.
  `{"poly": {"order": 2}, "gp": {"length_scale": 400}}`). These fields are
  documented in {doc}`fitting`.

Methods:

`fit(templates, residual, coeffs)`
: Measure shifts from the current fit state and build the predictor.
  `templates` is a sequence of {class}`mophongo.templates.Template`,
  `residual` the current residual image, and `coeffs` the fitted
  amplitudes. For the polynomial model the basis order comes from
  `astrom_kwargs["poly"]["order"]` (default 2); for the Gaussian process,
  `astrom_kwargs["gp"]["length_scale"]` (default 300 pixels) sets the RBF
  kernel scale. The cutout `box_size` (default 7) is read from the same
  per-model dict. After fitting, a template is shifted in place unless both
  predicted shift components are below 0.01 pixel.

  The defaults above are `AstroCorrect`-internal fallbacks that apply only
  when `astrom_kwargs` omits the key. `FitConfig` itself supplies
  `astrom_kwargs = {"poly": {"order": 0}, "gp": {"length_scale": 400}}` by
  default, so pipeline runs with an unmodified `FitConfig` use order 0 and a
  400-pixel length scale; the internal fallbacks take effect only for a
  config whose `astrom_kwargs` leaves those entries out.

`__call__(x, y=None)`
: Evaluate the fitted shift field. Accepts either an `(N, 2)` array of
  `(x, y)` positions or separate `x` and `y` arrays; returns `(dx, dy)`
  arrays with matching shape. Before `fit` is called, the predictor
  returns zeros.

`build_poly_predictor(coeffs, x_cen, y_cen, order, Sx=1.0, Sy=1.0)`
: Static helper that constructs a shift predictor from flat Chebyshev
  coefficients (first `n_terms(order)` entries for `dx`, next block for
  `dy`), with positions centered on `(x_cen, y_cen)` and scaled by
  `(Sx, Sy)`. Used to evaluate shift solutions produced elsewhere, for
  example the joint astrometric blocks solved during fitting.

### `AstroMap`

{class}`mophongo.astrometry.AstroMap` is a dataclass for image-to-image
shift mapping at catalog positions, independent of any fit:
{meth}`~mophongo.astrometry.AstroMap.fit` measures shifts of `img2` relative
to `img1` at the positions of sufficiently high-SNR catalog sources
(quadratic centroids or cross-correlation, optionally comparing through a
WCS pair) and fits a Chebyshev shift field of the configured `order`;
calling the instance evaluates the field, with the same conventions as
`AstroCorrect.__call__`. See the {doc}`api` reference for the constructor
fields and measurement keywords.

```python
from mophongo.astrometry import AstroMap

amap = AstroMap(order=2, snr_threshold=5.0)
amap.fit(img_ref, img_other, catalog)   # catalog with x, y, snr columns
dx, dy = amap(catalog["x"], catalog["y"])
```

### Lower-level helpers

`measure_template_shifts(templates, coeffs, residual, *, box_size=5,
snr_threshold=7.0, method="quadratic")`
: Returns `(positions, dx, dy, weights)` arrays for templates with
  `flux / err >= snr_threshold`. `method="correlation"` selects
  cross-correlation; any other value selects quadratic centroids. Weights
  are SNR squared.

`fit_polynomial_field(pos, dx, dy, w, *, order, shape)`
: Weighted least-squares fit of 2D Chebyshev shift fields over an image of
  the given `shape`; returns a `predict(positions) -> (dx, dy)` callable.

`cheb_basis(x, y, order)` and `n_terms(order)`
: Chebyshev basis values `T_i(x) T_j(y)` for coordinates scaled to
  `[-1, 1]`, and the number of terms, `(order + 1)(order + 2) / 2`.

See the {doc}`api` reference for full signatures.
