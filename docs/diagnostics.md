# Diagnostics and interactive use

After {meth}`mophongo.pipeline.Pipeline.run` completes (or after
{meth}`~mophongo.pipeline.Pipeline.load_fit` restores a finished run), the
`Pipeline` instance holds everything needed to inspect individual sources:
the fitted templates, the model and residual images, the fit table, and the
band PSFs. This page covers the input quicklook, the per-source inspection
helpers, the stamps file that persists them, run logging, and the standard
PSF/kernel diagnostics in {mod}`mophongo.psf`. See {doc}`outputs` for the
on-disk catalog products and {doc}`psf` for PSF construction itself.

```{figure} images/scene_diagnostic.png
:width: 100%
:alt: Six-panel scene diagnostic: template, image, model with shift arrows, segmentation map, residual, and color composite.

The scene diagnostic written by `write_outputs` shows the products the
per-source helpers on this page cut their stamps from: the high-resolution
template image, the fitted band, the best-fit model with the fitted
astrometric shift drawn as an arrow grid, the segmentation map, the fit
residual, and a color composite. The residual is flat down to the noise
apart from the cores of the few brightest sources.
```

`write_outputs` also writes `<name>_scene_map.png`: the whole field with
every segment colored by the scene that fitted it, which is the partition
the figure above shows one cell of (see {doc}`outputs`).

## Inspecting the inputs

{meth}`mophongo.pipeline.Pipeline.plot_inputs` gives a 2×2 quicklook of the
loaded inputs before any fitting: the high-resolution detection image, the
last low-resolution image, that image's inverse-variance weight map, and the
segmentation map. By default it overlays the catalog positions on the
high-resolution panel; `pipe.plot_inputs(save="inputs.png")` writes the
figure to disk. It requires loaded data (after
{meth}`~mophongo.pipeline.Pipeline.load_data` or the array constructor).

## Per-source products

### Collecting one source's products

{meth}`mophongo.pipeline.Pipeline.source_products` collects everything the
fit produced for one source into a dict. It works on the in-memory state
after `run()` or `load_fit()`; nothing is re-extracted or re-convolved. All
stamps and cutouts for a grid share one source-centered window (set by
`half_size`, defaulting to the template footprint), so they overlay
directly. Image 0 is the high-resolution detection image, so the fitted-band
index `ifilt` is 1-based, as everywhere else in the pipeline.

Every helper on this page works on a small synthetic run — two Gaussian
sources, a `photutils` segmentation map, and an array-based fit as in the
{doc}`quickstart` — so a `pipe` to experiment with costs a few lines (the
later examples reuse it):

```python
import numpy as np
from astropy.table import Table
from photutils.segmentation import detect_sources
from mophongo import pipeline
from mophongo.psf import PSF

rng = np.random.default_rng(11)
psf = PSF.gaussian(41, 3.0).array
img = rng.normal(0, 1e-3, (121, 121))
img[30:71, 25:66] += 50 * psf                # source 1 at (45, 50)
img[44:85, 50:91] += 20 * psf                # source 2 at (70, 64)
segmap = detect_sources(img, 0.01, npixels=5).data
catalog = Table({"id": [1, 2], "x": [45.0, 70.0], "y": [50.0, 64.0]})
wht = np.full_like(img, 1e6)                 # inverse variance of the noise
table, residuals, pipe = pipeline.run(
    [img, img], segmap, catalog=catalog,
    weights=[wht, wht], psfs=[psf / psf.sum()] * 2,
)
p = pipe.source_products(1, ifilt=1)
print(sorted(p)[:6])  # ['ee_psf_lo', 'err', 'err_pred', 'flag', 'flux', 'id']
print(round(p["flux"], 2), round(p["err"], 3))  # 49.99 0.005
fig, axes = pipe.show_sources([1, 2], save="sources.png")
```

The recovered `flux` of `49.99 ± 0.005` matches the injected 50 to within
the error, and `sources.png` holds the quicklook grid described below.

```python
p = pipe.source_products(42, ifilt=1, half_size=30)
p["residual"].std(), p["flux"], p["err"]
```

The returned dict has these keys:

| Key | Contents |
| --- | --- |
| `id`, `ifilt` | The requested source id and band index. |
| `tmpl_hi` | High-resolution template placed on the hi-grid window (2D array). |
| `tmpl_lo` | PSF-matched (convolved) template on the fitting-grid window. |
| `img_hi` | Detection-image cutout on the hi-grid window. |
| `segmap` | Segmentation-map cutout on the same hi-grid window. |
| `img_lo` | Fitted-band image cutout on the lo-grid window. |
| `model` | Best-fit model image cutout (`None` if no model is stored). |
| `residual` | Residual image cutout (`None` if no residual is stored). |
| `psf_hi`, `psf_lo` | PSF stamps at the source position, drawn from the band's static PSF or its {class}`~mophongo.psf_map.PSFRegionMap`. |
| `slices_hi`, `slices_lo` | The `(slice, slice)` windows into the full hi/lo images. |
| `position` | `(x, y)` source position on the reference grid. |
| `flux`, `err` | Fitted template amplitude and its uncertainty. |
| `err_pred` | Predicted (weight-map based) flux error. |
| `ee_psf_lo` | Encircled energy of the low-resolution PSF recorded for this template. |
| `flag` | Per-template fit flag. |
| `shift` | `(dx, dy)` astrometric shift applied to the template. |
| `row` | The matching row of the fit table, or `None`. |

Every hi-grid entry (`tmpl_hi`, `img_hi`, `segmap`, `slices_hi`) is `None`
when the source has no high-resolution template. The `flux` value is the raw
fitted amplitude (`flux_<i>` in the catalog); throughput-corrected totals
live in the `flux_<i>_total` catalog columns described in {doc}`outputs`.

### A quicklook grid of sources

{meth}`mophongo.pipeline.Pipeline.show_sources` renders a quicklook figure
with one row per source and eight columns: the detection-image cutout, the
high-resolution and convolved templates, the fitted-band cutout, the
best-fit model, the residual, and the two band PSFs at the source position.
The image, model, and residual panels share one display scale, so the
quality of the subtraction can be judged by eye; each row is labeled with
the source id and its fitted `flux ± err`.

```python
fig, axes = pipe.show_sources([12, 42, 108], half_size=25, save="sources.png")
```

### Tracing how a template was built

{meth}`mophongo.pipeline.Pipeline.diagnose_sources` traces how each template
was built, stage by stage, with one figure row per source: the
high-resolution image and segmentation map on the extracted-template
footprint, the template as extracted, after profile extension, and after
matching-kernel convolution onto the fitting grid, followed by the
fitted-band stamp, the best-fit model (including neighbors), and the
residual. The extracted and extended panels share one display scale, so the
effect of the extension step is visible directly; the segmentation panel
colors each neighbor distinctly with its label id. It requires a completed
{meth}`~mophongo.pipeline.Pipeline.run`, and re-derives the template stages
for each requested source rather than reading back possibly mutated
collections.

```python
fig, axes = pipe.diagnose_sources([12, 42], half_size=30, save="stages.png")
```

Two thin wrappers cover common cases:
{meth}`~mophongo.pipeline.Pipeline.diagnose_source` takes a single id, and
{meth}`~mophongo.pipeline.Pipeline.diagnose_bright_sources` selects the `n`
brightest sources by a flux column.

### The legacy subphot diagnostic

{meth}`mophongo.pipeline.Pipeline.diagnose_subphot` renders the IDL
`subphot.pro` six-panel diagnostic (`mkdiag`/`fptv`) for one source. It is a
pixel-for-pixel port of the legacy code — same panel layout, byte scalings,
background/rms estimator, circular fit mask, and grayscale segmentation
coloring — so its output compares one-to-one against IDL PNGs of the same
source. The panels are the low-resolution stamp, the high-resolution stamp,
the masked segmentation map, the full best-fit model, the masked
error-normalized residual, and the neighbor-cleaned image. The stamp size
(IDL `tsz`) defaults to the source's template footprint; pass the IDL tile
size via `size` for exact comparisons, and adjust the display stretch with
`nsig` (the default 3 matches the survey-era `subphot_nsigma=3` runs).

```{figure} images/subphot_six_panel.png
:width: 100%
:alt: Six-panel diagnose_subphot diagnostic: low-resolution image, high-resolution stamp, segmentation map, model, residual, and neighbor-cleaned image.

`diagnose_subphot` output for one source. The low-resolution stamp (`img`)
resolves into separate sources in the high-resolution stamp (`tmpl`); the
grayscale segmentation map (`seg`) carries the circular fit mask; the
best-fit model reproduces both sources, leaving a residual (`res`)
consistent with noise; the `clean` panel removes the neighbor models so
only the target source remains.
```

It requires a completed run, and the fitting grid must match the reference
grid (`NotImplementedError` otherwise). A `source_id` absent from the
fitted templates raises `KeyError`; batch scripts looping over ids should
catch it. It returns the rendered RGB image as a `uint8` array whether or
not `save` is given.

```python
rgb = pipe.diagnose_subphot(42, size=101, save="subphot_42.png")
```

## Persisting and restoring a fit

### The stamps file

{meth}`mophongo.pipeline.Pipeline.write_stamps` writes the per-source
template stamps to one FITS file (default
`<out_dir>/<name>_stamps.fits`). Together with the fit table and residual
image written by {meth}`~mophongo.pipeline.Pipeline.write_outputs`, this
file lets {meth}`~mophongo.pipeline.Pipeline.load_fit` restore the post-run
state without refitting.

The file contains a primary HDU and one `SOURCES` binary table. Templates
keep their native per-source sizes: each is stored flattened in a
variable-length array column next to its shape and grid origin, so nothing
is padded to a common size. The primary header records only pointers and the
grid shapes used for staleness checks: `NSRC`, `IFILT`, `RUNNAME` (the run
whose JSON/geojson save data these stamps use), and `NX_HI`/`NY_HI`,
`NX_LO`/`NY_LO`. PSF stamps are not duplicated in the file; each source
carries only its region key into the cached `<name>_psf_*.geojson` region
maps (`0` for a static PSF, `-1` when the band has none).

`SOURCES` columns:

| Column | Contents |
| --- | --- |
| `id`, `x`, `y` | Source id and reference-grid position. |
| `flux`, `err` | Fitted amplitude and uncertainty. |
| `tmpl_hi`, `ny_hi`, `nx_hi` | Hi-res template pixels (flattened; reshape to `(ny, nx)`). |
| `x0_hi`, `y0_hi` | Original-grid pixel of `data[0, 0]` (may be negative for edge-padded cutouts). |
| `xs_hi`, `ys_hi` | Source position on the original grid. |
| `tmpl_lo`, `ny_lo`, `nx_lo`, `x0_lo`, `y0_lo`, `xs_lo`, `ys_lo` | Same fields for the convolved template on the fitting grid. |
| `key_psf_hi`, `key_psf_lo` | Region keys into the band's PSF region map. |
| `flag_hi`, `flag`, `id_parent`, `id_scene`, `ee_psf_lo`, `ee_tmpl`, `err_pred`, `shift_x`, `shift_y` | Per-template fit metadata restored by `load_fit`. |

{meth}`mophongo.pipeline.Pipeline.read_stamps` (a static method taking only
`path`) reads the file back into a list of per-source dicts with the scalar
columns plus the reshaped 2D `tmpl_hi`/`tmpl_lo` arrays.

### Restoring a finished run

{meth}`mophongo.pipeline.Pipeline.load_fit` restores the post-run state from
written outputs without refitting. It is the counterpart of
{meth}`~mophongo.pipeline.Pipeline.load_data`, which restores the pre-run
state, and requires a config-driven pipeline (one built with
{meth}`~mophongo.pipeline.Pipeline.from_config`). Since `run()` writes a
copy of its config to `<out_dir>/<name>.json`, a finished run reopens with:

```python
pipe = Pipeline.from_config("run.json").load_fit()
pipe.show_sources([42])
```

It reads the fit table and residual, rebuilds the fitted templates from the
stamps file (regenerating and rewriting it through the same path `run()`
uses when it is missing), and recreates the derived state, so
`source_products` and `show_sources` work as if `run()` had just completed.
Scenes are not persisted, and regenerated stamps reproduce the fitted
templates exactly only when the run applied no astrometric shifts; see the
{meth}`~mophongo.pipeline.Pipeline.load_fit` API notes for the details.

## Run logging

{meth}`mophongo.pipeline.Pipeline.log_run` is a context manager that
captures everything a run emits into one log file (default
`<out_dir>/<name>.log`, opened in append mode):
{meth}`~mophongo.pipeline.Pipeline.run_all` wraps its whole sequence in it
automatically, so explicit use is only needed when calling the steps
individually:

```python
with pipe.log_run() as log_path:
    pipe.build_psfs()
    pipe.build_kernels()
    pipe.run()
```

Inside the block, records from every logger (mophongo and dependencies such
as astropy, drizzlepac, and stpsf), captured `warnings.warn` messages, and
teed stdout/stderr (bare `print` and `tqdm`) all land in the file while the
console stays unchanged. Each entry starts with a header recording the run
name, timestamp, Python version, platform, and output directory, and ends
with the elapsed time — or a `FAILED after <t>s` line if the block raises.

## From the command line

The `mophongo` console script reaches the same products from a shell:

```bash
# the matching kernel at a sky position, as FITS with a WCS
mophongo psf run/uds_770_kernel.geojson 34.5202 -5.2566 -o kernel.fits

# the same for a band PSF, naming the run instead of the map
mophongo psf run/uds_770.json 34.5202 -5.2566 --map-kind psf_lo

# one source's cutouts, PSFs, and fit row as a multi-extension FITS
mophongo stamps run/uds_770.json 4711 --half-size 40

# the subphot six-panel diagnostic (--style stages for the build stages)
mophongo diag run/uds_770.json 4711 --size 101

mophongo info run/uds_770.json               # summarize a run, no pixels read
mophongo run  run/uds_770.json fit outputs   # pipeline steps
```

`psf` reads only the cached region map (`_psf_hi`, `_psf_lo`, or `_kernel`),
so it costs nothing. The stamp is written centered on the requested
position, inheriting the orientation and pixel scale of the mosaic the map
was drizzled onto; the run config beside the map names that mosaic, and
`--pixel-scale` in arcsec covers the case where there is none (a north-up
tangent plane). The header carries the region key, the stamp's encircled
energy, and the map's provenance — PSF pattern, stamp size, broadening,
kernel method and regularization.

`stamps` and `diag` restore the run with
{meth}`~mophongo.pipeline.Pipeline.load_fit`, so they read the mosaics;
several ids in one invocation share that single load. `stamps` writes the
{meth}`~mophongo.pipeline.Pipeline.source_products` dict as image extensions
(`IMG_HI`, `SEGMAP`, `TMPL_HI`, `IMG_LO`, `TMPL_LO`, `MODEL`, `RESID`,
`PSF_HI`, `PSF_LO`), each with the WCS of its parent grid, the fitted
scalars in the primary header, and the fit-table row as `FITROW`.

Every subcommand is a thin wrapper over a function in {mod}`mophongo.cli`
(`psf_to_fits`, `source_stamps_to_fits`, `source_diagnostic_png`); the last
two take a `Pipeline` that is already loaded.

## PSF and kernel diagnostics

Matching-kernel quality is scored with a common figure of merit built from
the mean squared encircled-energy mismatch between the matched and target
PSFs (growth term), the mean squared log radial-profile mismatch inside a
core radius (core term), an optional image-space L2 term, and a kernel
stability penalty. The stability penalty combines the fraction of kernel
Fourier power near the Nyquist scale (pixel-scale ringing) and the excess
absolute flux from positive/negative kernel oscillations (signed-flux
cancellation `C(K) = sum|K| / |sum K| - 1`). The default stability term is
cancellation only.

Use these scans rather than ad hoc PSF figures; the written
`diagnostic_<method>.png` is the standard product.

### Scanning the regularization

{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization` grid-scans
the scalar regularization parameter for the non-windowed matching methods
(`"tikhonov"`, `"wiener"`, `"forward"`), scoring each candidate kernel with
the figure of merit above and returning the best fit as a
{class}`~mophongo.psf.MatchingKernelRegFit`. When the pair live on
different pixel scales, `pixel_ratio` resamples them onto a common grid
first, so the scan scores the kernel that will actually be built.

```python
from mophongo.psf import PSF

psf_hi = PSF.from_array(hi_stamp)
res = psf_hi.optimize_matching_kernel_regularization(
    lo_stamp, method="tikhonov", diagnostic_path="diag/"
)
kernel = res.kernel  # best kernel, at reg = res.reg
```

The diagnostic figure written by `diagnostic_path` is a 3×3 grid: the figure
of merit versus lambda with the chosen value marked; the target and matched
radial profiles (log-log); the growth-curve ratio `EE(match)/EE(target)`
with ±2% reference bands; five image panels on a shared asinh scale (source
PSF, target PSF, kernel, matched PSF, and matched-minus-target residual);
and a text panel with the score breakdown.

### Optimizing a Fourier window

{meth}`mophongo.psf.PSF.optimize_matching_kernel_window` grid-searches the
`(alpha, beta)` parameters of a split-cosine-bell Fourier window for
`method="window"` matching, over all pairs with `alpha + beta <= 1`, scored
with the same figure of merit; it returns a
{class}`~mophongo.psf.MatchingKernelWindowFit`.

{meth}`mophongo.psf.PSF.auto_matching_kernel_window` is the lightweight
production wrapper around that search. It maps a named figure-of-merit
preset (the default `fom="c2"` scores growth MSE + core MSE +
`1e-3 * C(K)^2`) to the stability weights and returns the optimized
`SplitCosineBellWindow`, ready to pass to
{meth}`mophongo.psf.PSF.matching_kernel`. Passing `diagnostic_path` writes
the window-search diagnostic (`diagnostic_window.png`).

```python
window = psf_hi.auto_matching_kernel_window(lo_stamp, diagnostic_path="diag/")
kernel = psf_hi.matching_kernel(lo_stamp, window=window)
```

### Scan results

Both scans return dataclasses carrying the winner and the full search
history, so the trade-off between fidelity and kernel stability can be
re-examined without re-running. {class}`mophongo.psf.MatchingKernelRegFit`
holds the chosen method and regularization together with the kernel,
matched PSF, 1D scan and metric grids, and the sampled profiles;
{class}`mophongo.psf.MatchingKernelWindowFit` holds the same for the window
search, with `alpha`/`beta` in place of `method`/`reg` and 2D
`(beta, alpha)` grids. Field-level detail is in the {doc}`api` reference.

### Encircled energy of drizzled stamps

{func}`mophongo.psf.stamp_encircled_energy` measures the realized encircled
energy of drizzled PSF stamps. Because absolutely calibrated ePSFs drizzle
onto an absolute flux scale, a stamp sum is itself an encircled energy;
measuring it on the final stamp folds in the drizzle kernel, geometric
distortion, and the exposure stack at the position. It returns a dict with
`ee_box` (the full-stamp sum), `ee_circ` (the sum inside the inscribed
circle), the corresponding radii, and optionally per-stamp arrays for PSF
cubes. `ee_box` is the quantity that converts a fitted amplitude into a
total flux; `ee_circ` is the one to compare against tabulated
encircled-energy curves.

```python
from mophongo.psf import PSF, stamp_encircled_energy

stamp = 0.95 * PSF.gaussian(51, 6.0).array   # stamp with finite throughput
ee = stamp_encircled_energy(stamp, pscale=0.08, ee_fraction=0.5)
print({k: round(v, 4) for k, v in ee.items()})
# {'ee_box': 0.95, 'ee_circ': 0.95, 'r_circ': 2.04, 'r_ee': 0.253}
```

```{figure} images/stamp_encircled_energy.png
:width: 100%
:alt: Cumulative PSF flux versus radius from the stamp centre, with the inscribed-circle radius dotted and the full-stamp sum dashed

Growth curve of a drizzled JWST PSF stamp. The cumulative flux flattens
towards the stamp edge but never reaches unity: the curve at the
inscribed-circle radius (dotted) is `ee_circ`, and the full-stamp sum
(dashed) sits slightly higher because the stamp corners hold flux outside
that radius. Support beyond the square stamp is lost entirely, which is why
`ee_box` stays below one.
```

### Comparing a PSF against a star

{func}`mophongo.utils.compare_psf_to_star` compares a model PSF to a real
star cutout, optionally after convolution with a matching kernel, and
produces a two-row figure (five image panels over three profile panels). It
can clean neighbors from the cutout, register the PSF to the star centroid,
and scale the PSF to the data inside a normalization radius. Its full
signature is in the {doc}`api` reference; it lives in `mophongo.utils`
rather than `mophongo.psf`.

## See also

- {doc}`pipeline` — running the fit and the `Pipeline` constructor.
- {doc}`outputs` — catalog columns, residual images, and throughput-corrected
  total fluxes.
- {doc}`psf` — PSF construction, `DrizzlePSF`, and matching-kernel methods.
- {doc}`psf_maps` — spatially varying PSFs via `PSFRegionMap`.
