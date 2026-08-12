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

{func}`mophongo.saturate.repair_saturated_holes` is the main entry point,
called as in the example above. Its many knobs group into the donut
geometry (`buffer`, `factor`, `fwhm_pix`), the iterative amplitude and
position fit (`fit_shift`, `shift_tol`, `max_shift_pix`), the acceptance
filtering — the main pre-filter is `min_buffer_snr`, the minimum buffer
significance in units of the global sky noise, default 200 — and the
action `mode` (`"repair"` or `"subtract"`). All parameters and defaults
are documented in the {doc}`api` reference.

The returned dict carries the repaired `"sci"` and `"wht"` arrays, the fit
table `"fits"` (one row per hole, including rejected ones, whose `status`
column records why a hole was skipped), the per-source `"diagnostics"`,
the hole table, and the global sky level and noise; `output_csv` and
`plot_dir` additionally write the fit table as CSV and per-source
diagnostic PNGs.

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

```python
import numpy as np
from mophongo.saturate import find_wht_holes

wht = np.ones((101, 101))
yy, xx = np.indices(wht.shape)
wht[np.hypot(yy - 60, xx - 40) < 6] = 0.0   # saturated core: interior hole
wht[:, :4] = 0.0                            # chip gap touching the border: dropped
holes = find_wht_holes(wht, min_area=5)
print(len(holes), int(holes["area"][0]), round(float(holes["r_equiv"][0]), 2))  # 1 109 5.89
```

{func}`mophongo.saturate.fit_psf_donut`
: Fits `data ≈ A·psf [+ C]` by weighted linear least squares on the ring
  `r_in <= r <= r_out`, returning the amplitude, its error, reduced
  chi-square, optional pedestal, and the shape-mismatch diagnostic
  `rho_psf` (an amplitude-invariant weighted data–PSF correlation over the
  ring).

```python
import numpy as np
from mophongo.psf import PSF
from mophongo.saturate import fit_psf_donut

rng = np.random.default_rng(11)
psf = PSF.gaussian(101, 8.0).array              # unit-sum PSF model
sci = 500.0 * psf + rng.normal(0, 1e-4, psf.shape)
wht = np.full(sci.shape, 1e8)
yy, xx = np.indices(sci.shape)
wht[np.hypot(yy - 50, xx - 50) < 8] = 0.0       # zero-weight saturated core
fit = fit_psf_donut(sci, wht, psf, center=(50.0, 50.0), r_in=9.0, r_out=25.0)
print(round(fit["amplitude"], 2), round(fit["rho_psf"], 3))  # 500.01 1.0 (truth: 500)
```

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

Each hole that reached the fitting stage also yields a
{class}`~mophongo.saturate.RepairDiagnostic` (collected when
`return_diagnostics=True`): a result container holding the fit geometry
and quality metrics as scalars, plus the cutouts and masks that
`plot_repair_diagnostic` renders. Holes rejected by the buffer-SNR
pre-filter or by the residual-fraction guard get a table row but no
diagnostic; a hole whose fit failed gets a stub diagnostic with
`ok=False`.

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

{class}`mophongo.astrometry.AstroCorrect` is constructed from a
{class}`~mophongo.fit.FitConfig`, which supplies all its options: the field
model `astrom_model` (`"poly"` or `"gp"`), the measurement method
`astrom_centroid`, the anchor cut `snr_thresh_astrom`, and the per-model
keyword dicts in `astrom_kwargs` — all documented in {doc}`fitting`.
{meth}`~mophongo.astrometry.AstroCorrect.fit` measures shifts from the
current fit state (templates, residual image, fitted amplitudes), fits the
shift field, and resamples each template in place unless its predicted
shift is below 0.01 pixel in both components; calling the instance
afterwards evaluates the field at any position (zeros before `fit`). The
static helper
{meth}`~mophongo.astrometry.AstroCorrect.build_poly_predictor` constructs
the same kind of predictor from flat Chebyshev coefficients produced
elsewhere — this is how the joint astrometric solutions of the scene
fitter are evaluated.

```python
from mophongo.astrometry import AstroCorrect
from mophongo.fit import FitConfig

acorr = AstroCorrect(FitConfig(astrom_model="poly"))
acorr.fit(templates, residual, coeffs)   # shifts templates in place
dx, dy = acorr(x, y)                     # evaluate the fitted field
```

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

Both facades build on the same primitives:
{func}`~mophongo.astrometry.measure_template_shifts` collects the per-source
`(position, dx, dy, weight)` samples,
{func}`~mophongo.astrometry.fit_polynomial_field` fits the weighted 2D
Chebyshev shift field and returns a predictor, and
{func}`~mophongo.astrometry.cheb_basis` /
{func}`~mophongo.astrometry.n_terms` provide the Chebyshev cross-term basis
for coordinates scaled to `[-1, 1]`. See the {doc}`api` reference for full
signatures.

```python
import numpy as np
from mophongo.astrometry import fit_polynomial_field

rng = np.random.default_rng(11)
pos = rng.uniform(0, 2000, size=(50, 2))        # (x, y) sample positions
dx = 0.10 + 1e-4 * pos[:, 0]                    # a linear shift field ...
dy = -0.05 + rng.normal(0, 0.01, 50)            # ... plus noisy constant dy
predict = fit_polynomial_field(pos, dx, dy, np.ones(50), order=1, shape=(2000, 2000))
px, py = predict(np.array([[1000.0, 1000.0]]))
print(round(float(px[0]), 3), round(float(py[0]), 3))  # 0.2 -0.05
```
