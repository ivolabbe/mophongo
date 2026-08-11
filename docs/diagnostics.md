# Diagnostics and interactive use

After {meth}`mophongo.pipeline.Pipeline.run` completes (or after
{meth}`~mophongo.pipeline.Pipeline.load_fit` restores a finished run), the
`Pipeline` instance holds everything needed to inspect individual sources:
the fitted templates, the model and residual images, the fit table, and the
band PSFs. This page covers the input quicklook, the per-source inspection
helpers, the stamps file that persists them, run logging, and the standard
PSF/kernel diagnostics in {mod}`mophongo.psf`. See {doc}`outputs` for the
on-disk catalog products and {doc}`psf` for PSF construction itself.

## Inspecting the inputs

### `Pipeline.plot_inputs`

{meth}`mophongo.pipeline.Pipeline.plot_inputs` gives a 2×2 quicklook of the
loaded inputs before any fitting: the high-resolution detection image
(`images[0]`), the last low-resolution image (`images[-1]`), that image's
inverse-variance weight map, and the segmentation map. It requires loaded
data (after {meth}`~mophongo.pipeline.Pipeline.load_data` or the array
constructor) and raises `RuntimeError` otherwise. The weight panel is left
blank when no weights are loaded; the segmentation panel uses the
`photutils` segmentation colormap and is titled with the maximum label.

Parameters (keyword-only):

`sources` (`bool`, default `True`)
: Overlay catalog `x`/`y` positions as open red circles on the
  high-resolution panel (skipped when no catalog is present).

`save` (`str | os.PathLike | None`, default `None`)
: Optional path to save the figure with `savefig` at 180 dpi.

Returns the created matplotlib figure and its flat array of four axes.

## Per-source products

### `Pipeline.source_products`

{meth}`mophongo.pipeline.Pipeline.source_products` collects everything the
fit produced for one source into a dict. It works on the in-memory state
after `run()` or `load_fit()`; nothing is re-extracted or re-convolved. All
stamps and cutouts for a grid share one source-centered window, so they
overlay directly.

```python
p = pipe.source_products(42, ifilt=1, half_size=30)
p["residual"].std(), p["flux"], p["err"]
```

Parameters:

`source_id` (`int`)
: Catalog id of the source.

`ifilt` (`int`, default `1`)
: Fitted image index. Image 0 is the high-resolution detection image, so
  fitted bands are 1-based, as everywhere else in the pipeline.

`half_size` (`int | None`, default `None`)
: Half-size in pixels of the source-centered window cut on each grid; the
  window is `2 * half_size + 1` pixels on a side, clipped at image edges.
  `None` uses each template's own footprint.

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

### `Pipeline.show_sources`

{meth}`mophongo.pipeline.Pipeline.show_sources` renders a quicklook figure
with one row per source and eight columns:

1. `hi image` — detection-image cutout
2. `tmpl_hi` — high-resolution template
3. `tmpl_lo` — convolved template
4. `lo image` — fitted-band cutout
5. `model` — best-fit model
6. `residual` — data minus model
7. `psf_hi` — hi-band PSF at the source position
8. `psf_lo` — lo-band PSF at the source position

The `lo image`, `model`, and `residual` panels share one display scale — a
median/MAD stretch computed from their combined pixels, clipped at ±5 MAD —
so the quality of the subtraction can be judged by eye. The other panels are
auto-scaled individually. Each row is labeled with the source id and its
fitted `flux ± err`.

```python
fig, axes = pipe.show_sources([12, 42, 108], half_size=25, save="sources.png")
```

Parameters:

`source_ids` (`int | Sequence[int]`)
: One id or a sequence of ids; one figure row per id.

`ifilt` (`int`, default `1`)
: Fitted image index (1-based).

`half_size` (`int | None`, default `None`)
: Window half-size in pixels; `None` uses each template's footprint.

`save` (`str | os.PathLike | None`, default `None`)
: Optional path to save the figure with `savefig` at 150 dpi; the output
  format follows the file extension.

Returns the created matplotlib figure and its `(nsrc, 8)` axes array.

### `Pipeline.diagnose_sources`

{meth}`mophongo.pipeline.Pipeline.diagnose_sources` traces how each template
was built, stage by stage, with one figure row per source and eight columns:

1. `hires image (ref grid)` — detection-image stamp on the
   extracted-template footprint
2. `segmap` — segmentation map over the same footprint
3. `extracted template (ref grid)` — the template as extracted, before
   extension
4. `after extension (ref grid)` — the template after profile extension
5. `after conv/proj (fit grid)` — after matching-kernel convolution and
   projection onto the fitting grid
6. `low-res image (fit grid)` — fitted-band stamp at the same location
7. `best-fit model (fit grid)` — the full model, including neighbors
8. `residual (fit grid)` — data minus model

Columns 3 and 4 share one display scale, so the effect of the extension step
is visible directly. The segmentation panel shows the target source in gray
and each neighbor in a distinct color from a ten-entry palette, with every
label id printed at its segment centroid (the target in bold). It requires a
completed {meth}`~mophongo.pipeline.Pipeline.run`, since it uses the stored
extracted and extended template collections; each row rebuilds the stage
templates for its source (falling back to stored snapshots when the run used
externally supplied templates).

```python
fig, axes = pipe.diagnose_sources([12, 42], half_size=30, save="stages.png")
```

Parameters:

`source_ids` (`Sequence[int]`)
: Source ids; one figure row per id. Must not be empty.

`ifilt` (`int`, keyword-only, default `1`)
: Fitted image index (1-based).

`half_size` (`int | None`, keyword-only, default `None`)
: Window half-size in pixels; `None` uses the extracted template's
  footprint.

`save` (`str | os.PathLike | None`, keyword-only, default `None`)
: Optional path to save the figure with `savefig` at 180 dpi.

Returns the created matplotlib figure and its `(nsrc, 8)` axes array. Two
thin wrappers cover common cases:
{meth}`~mophongo.pipeline.Pipeline.diagnose_source` takes a single id, and
{meth}`~mophongo.pipeline.Pipeline.diagnose_bright_sources` selects the `n`
(default 5) brightest sources by a flux column (default `flux_<ifilt>`).

### `Pipeline.plot_subphot`

{meth}`mophongo.pipeline.Pipeline.plot_subphot` renders the IDL
`subphot.pro` six-panel diagnostic (`mkdiag`/`fptv`) for one source. It is a
pixel-for-pixel port of the legacy code, so its output compares one-to-one
against IDL PNGs of the same source: the same 2×3 panel layout at 2×
nearest-neighbour zoom, the same byte scalings, the same background/rms
estimator (aperture-scale block sums, 2-sigma clipped, `prms = rms / na`),
the same circular `rlim` fit mask on the residual panels, and the same
distance-sorted five-level grayscale segmentation coloring. The panels are:

- `img` — low-resolution stamp, displayed at `±nsig * prms`
- `tmpl` — high-resolution image stamp, at `median ± 8 * robust_sigma`
- `seg` — color-cycled segmentation map, minus `0.1 *` the fit mask
- `model` — full best-fit model, at `±nsig * prms`
- `res` — masked `(img - model) / err`, at `±nsig`
- `clean` — masked image minus neighbor models, at `±nsig * prms`

It requires a completed run, and the fitting grid must match the reference
grid (`NotImplementedError` otherwise).

```python
rgb = pipe.plot_subphot(42, size=101, save="subphot_42.png")
```

Parameters:

`source_id` (`int`)
: Catalog id of the source to center on.

`ifilt` (`int`, keyword-only, default `1`)
: Fitted image index (1-based).

`size` (`int | None`, keyword-only, default `None`)
: Stamp side in fit-grid pixels (IDL `tsz`). Defaults to the source's
  template-footprint size, made odd. Pass the IDL tile size for exact
  comparisons.

`rlim` (`float | None`, keyword-only, default `None`)
: Fit-mask radius in pixels; `None` uses `(size - 1) / 2`.

`nsig` (`float`, keyword-only, default `3.0`)
: Display stretch in sigma. The default 3 matches the survey-era
  `subphot_nsigma=3` runs; the IDL code default was 5.

`sys_err` (`float`, keyword-only, default `0.02`)
: Systematic error fraction in
  `err = sqrt(prms**2 + (sys_err * model)**2)` (the IDL default).

`photbin` (`int`, keyword-only, default `1`)
: Optional SNR-preserving display binning of the photometry-based panels
  (IDL `photbin`).

`raper` (`float | None`, keyword-only, default `None`)
: Aperture radius in pixels setting the rms block size `na`; `None` takes
  it from the fit configuration.

`save` (`str | os.PathLike | None`, keyword-only, default `None`)
: Optional PNG output path.

Returns the rendered RGB image as a `(4*size, 6*size, 3)` `uint8` array
(panel labels burned in), whether or not `save` is given.

## Persisting and restoring a fit

### `Pipeline.write_stamps`

{meth}`mophongo.pipeline.Pipeline.write_stamps` writes the per-source
template stamps to one FITS file. Together with the fit table and residual
image written by {meth}`~mophongo.pipeline.Pipeline.write_outputs`, this
file lets {meth}`~mophongo.pipeline.Pipeline.load_fit` restore the post-run
state without refitting.

Parameters:

`path` (`str | os.PathLike | None`, default `None`)
: Output file. Defaults to `<out_dir>/<name>_stamps.fits` for config-driven
  runs; required otherwise.

`ifilt` (`int`, keyword-only, default `1`)
: Fitted image index (1-based).

Returns the path of the written file.

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

### `Pipeline.load_fit`

{meth}`mophongo.pipeline.Pipeline.load_fit` restores the post-run state from
written outputs without refitting. It is the counterpart of
{meth}`~mophongo.pipeline.Pipeline.load_data`, which restores the pre-run
state, and requires a config-driven pipeline (one built with
{meth}`~mophongo.pipeline.Pipeline.from_config`).

Parameters:

`ifilt` (`int`, default `1`)
: Fitted image index (1-based).

Returns `self`, in the post-run state.

It reads `<name>_fit_table.fits` and `<name>_residual.fits`, rebuilds the
fitted templates from `<name>_stamps.fits`, and recreates the derived state
(grid upsampling, model image). When the stamps file is missing, the
templates are regenerated through the same path `run()` uses, fluxes are
taken from the fit table, and the stamps file is written back to disk.
Afterwards the instance exposes `table`, `residuals`, `model_images`,
`tmpls`, and `all_templates`, so `source_products` and `show_sources` work
as if `run()` had just completed.

Not restored: `all_scenes` (scenes are not persisted), and the
pre-extension pixels of `templates_extracted` when loading from a stamps
file. Regenerated stamps reproduce the fitted templates exactly only when
the run applied no astrometric shifts.

```python
pipe = Pipeline.from_config("run.json").load_fit()
pipe.show_sources([42])
```

## Run logging

### `Pipeline.log_run`

{meth}`mophongo.pipeline.Pipeline.log_run` is a context manager that
captures everything a run emits into one log file;
{meth}`~mophongo.pipeline.Pipeline.run_all` wraps its whole sequence in it
automatically, so explicit use is only needed when calling the steps
individually:

```python
with pipe.log_run() as log_path:
    pipe.build_psfs()
    pipe.build_kernels()
    pipe.run()
```

Parameters:

`path` (`str | Path | None`, default `None`)
: Log file. Defaults to `<out_dir>/<name>.log` for config-driven runs.
  Parent directories are created; the file is opened in append mode, so
  successive runs against one output directory accumulate.

The context manager yields the log path. Inside the block it captures three
output channels:

- **All library loggers.** A file handler is attached to the root logger, so
  records from every logger — mophongo itself, and dependencies such as
  astropy, drizzlepac, and stpsf that log through their own loggers — reach
  the file even when the caller configured logging before the run. The root
  level is raised to `INFO` for the duration if it was unset or higher.
- **Warnings.** `logging.captureWarnings` routes `warnings.warn` messages
  into the same file (the capture state is reset first, so a hook installed
  earlier in the process does not shadow it).
- **stdout/stderr.** Both streams are teed, so bare `print` and `tqdm`
  output land in the file while the console stays unchanged. When the
  `mophongo` logger has no handler of its own, a console handler is added
  for the duration so package records still appear on screen.

Each entry starts with a header recording the run name, timestamp, Python
version, platform, and output directory, and ends with the elapsed time. If
the block raises, a `FAILED after <t>s: <exception>` line is written and the
exception propagates. All logging state, the warning hook, and the streams
are restored on exit.

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

### `PSF.optimize_matching_kernel_regularization`

{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization` grid-scans
the scalar regularization parameter for the non-windowed matching methods
(`"tikhonov"`, `"wiener"`, `"forward"`), scoring each candidate kernel with
the figure of merit above and returning the best fit as a
{class}`~mophongo.psf.MatchingKernelRegFit`.

```python
from mophongo.psf import PSF

psf_hi = PSF.from_array(hi_stamp)
res = psf_hi.optimize_matching_kernel_regularization(
    lo_stamp, method="tikhonov", diagnostic_path="diag/"
)
kernel = res.kernel  # best kernel, at reg = res.reg
```

Parameters:

`other` (`PSF | np.ndarray`)
: Target PSF that `self` should be convolved into.

`method` (`str`, default `"tikhonov"`)
: Matching method: `"tikhonov"`, `"wiener"`, or `"forward"`.

`reg_grid` (`np.ndarray | None`, default `None`)
: Regularization values to scan; `None` uses `np.logspace(-6, -1, 21)`,
  the standard scan range.

`pixel_ratio` (`float`, default `1.0`)
: Pixel-scale ratio between the pair. Values above 1 resample the target
  onto the source grid (flux-conserving cubic), values below 1 resample the
  source, so the scan scores the kernel that will actually be built.

`core_radius` (`float | None`, default `None`)
: Maximum radius in pixels for the core profile term; `None` uses one
  quarter of the PSF size.

`growth_weight`, `core_weight`, `l2_weight` (`float`, defaults `1.0`, `1.0`, `0.0`)
: Weights of the growth-curve, radial-core, and image-space mean-square
  terms.

`kernel_regularization_weight` (`float`, default `1e-3`)
: Overall weight of the kernel stability term.

`kernel_high_frequency_radius` (`float`, default `0.7`)
: Fourier radius, in Nyquist units, above which kernel power counts as
  pixel-scale ringing.

`kernel_high_frequency_weight`, `kernel_cancellation_weight` (`float`, defaults `0.0`, `1.0`)
: Relative weights of high-frequency power and flux cancellation inside the
  stability term.

`recenter` (`bool`, default `False`)
: Passed to the kernel builder; `False` keeps already-centered PSFs fixed
  during scoring.

`wavelet` (`str`, default `"db4"`), `levels` (`int`, default `3`), `threshold_factor` (`float`, default `3.0`), `noise_sigma` (`float | None`, default `None`), `forward_wavelet_wiener` (`bool`, default `True`)
: Options for `method="forward"` (ForWaRD Fourier+wavelet deconvolution).
  `threshold_factor` sets the hard threshold on detail coefficients in units
  of the estimated per-subband noise.

`signal_psd` (`np.ndarray | None`, default `None`)
: Signal power spectral density for `method="wiener"`.

`diagnostic_path` (`str | Path | None`, default `None`)
: Write the standard diagnostic figure. Passing a directory writes
  `diagnostic_<method>.png` inside it.

`source_label` (`str`, default `"source PSF"`), `target_label` (`str`, default `"target PSF"`), `diagnostic_title` (`str | None`, default `None`), `aperture_radius` (`float | None`, default `None`), `diagnostic_note` (`str | None`, default `None`)
: Labels used only in the diagnostic figure; `aperture_radius` draws a
  vertical marker on the growth-ratio panel, `diagnostic_note` is appended
  to the info panel.

The diagnostic figure is a 3×3 grid: the figure of merit versus lambda with
the chosen value marked; the target and matched radial profiles (log-log);
the growth-curve ratio `EE(match)/EE(target)` with ±2% reference bands;
five image panels on a shared asinh scale (source PSF, target PSF, kernel,
matched PSF, and matched-minus-target residual); and a text panel with the
score breakdown (growth MSE, core log MSE, L2 MSE, RMS residual, and the
kernel high-frequency and cancellation metrics).

### `PSF.optimize_matching_kernel_window`

{meth}`mophongo.psf.PSF.optimize_matching_kernel_window` grid-searches the
`(alpha, beta)` parameters of a split-cosine-bell Fourier window for
`method="window"` matching, over all pairs with `alpha + beta <= 1`. The
score is the same figure of merit as above. It returns a
{class}`~mophongo.psf.MatchingKernelWindowFit`.

Parameters:

`other` (`PSF | np.ndarray`)
: Target PSF.

`alpha_grid`, `beta_grid` (`np.ndarray | None`, defaults `None`)
: Explicit window-parameter grids; the defaults span alpha 0.02–0.90 and
  beta 0.05–0.95, avoiding the singular endpoints.

`grid_oversample` (`int`, default `2`)
: Refinement factor for the default grids; ignored when explicit grids are
  given.

`core_radius`, `growth_weight`, `core_weight`, `l2_weight`, `kernel_regularization_weight`, `kernel_high_frequency_radius`, `kernel_high_frequency_weight`, `kernel_cancellation_weight`, `recenter`
: Same meaning and defaults as in
  `optimize_matching_kernel_regularization`.

### `PSF.auto_matching_kernel_window`

{meth}`mophongo.psf.PSF.auto_matching_kernel_window` is the lightweight
production wrapper around the window search. It maps a named figure-of-merit
preset to the stability weights and returns the optimized
`SplitCosineBellWindow`, ready to pass to
{meth}`mophongo.psf.PSF.matching_kernel`.

```python
window = psf_hi.auto_matching_kernel_window(lo_stamp, diagnostic_path="diag/")
kernel = psf_hi.matching_kernel(lo_stamp, window=window)
```

Parameters:

`other` (`PSF | np.ndarray`)
: Target PSF.

`fom` (`str`, default `"c2"`)
: Figure-of-merit preset. `"c2"` is an alias for `"growth_core_cancel"`
  (growth MSE + core MSE + `1e-3 * C(K)^2`). Other presets:
  `"growth_core_only"`, `"growth_core_hf"`, `"growth_core_hf_cancel"`.

`alpha_grid`, `beta_grid`, `grid_oversample`, `core_radius`, `growth_weight`, `core_weight`, `l2_weight`, `kernel_high_frequency_radius`, `recenter`
: Passed through to `optimize_matching_kernel_window`, same defaults.

`reg_lambda` (`float`, default `1e-3`)
: Overall weight of the kernel stability term (the
  `kernel_regularization_weight` of the underlying search).

`diagnostic_path` (`str | Path | None`, default `None`)
: Write the window-search diagnostic (score grid over alpha/beta, profiles,
  growth ratio, kernel, matched PSF, residual). A directory path writes
  `diagnostic_window.png`.

`source_label`, `target_label`, `diagnostic_title`, `aperture_radius`
: Figure labels only, as above.

`return_result` (`bool`, default `False`)
: If `True`, return `(window, result)` with the full
  `MatchingKernelWindowFit`.

### Scan results

Both scans return dataclasses carrying the winner and the full search
history, so the trade-off between fidelity and kernel stability can be
re-examined without re-running:

{class}`mophongo.psf.MatchingKernelRegFit`
: `method` and `reg` (the chosen regularization), `score`, `kernel`,
  `matched_psf`, the 1D scan arrays `reg_grid`, `score_grid`,
  `growth_error_grid`, `core_error_grid`, `l2_error_grid`,
  `kernel_regularization_grid`, `kernel_high_frequency_grid`,
  `kernel_cancellation_grid`, the profile sampling `radii` with
  `target_growth`, `matched_growth`, `target_profile`, `matched_profile`,
  and `extra` (a dict of the scan settings).

{class}`mophongo.psf.MatchingKernelWindowFit`
: The same fields (except `extra`) with `alpha`/`beta` in place of
  `method`/`reg`, 2D `(beta, alpha)` score and metric grids, and the
  `alpha_grid`/`beta_grid` axes.

### Encircled energy of drizzled stamps

{func}`mophongo.psf.stamp_encircled_energy` measures the realized encircled
energy of drizzled PSF stamps. Because absolutely calibrated ePSFs drizzle
onto an absolute flux scale, a stamp sum is itself an encircled energy;
measuring it on the final stamp folds in the drizzle kernel, geometric
distortion, and the exposure stack at the position. `ee_box` (the full-stamp
sum) is the quantity that converts a fitted amplitude into a total flux;
`ee_circ` is the one to compare against tabulated encircled-energy curves.

Parameters:

`psf` (`np.ndarray`)
: One stamp `(ny, nx)` or a cube `(..., ny, nx)`; non-finite pixels count
  as zero. Cubes are averaged, so a spatially varying PSF map returns its
  mean behavior.

`pscale` (`float`)
: Stamp pixel scale in arcsec.

`ee_fraction` (`float | None`, keyword-only, default `None`)
: If given, also return the radius enclosing this absolute fraction.

`per_stamp` (`bool`, keyword-only, default `False`)
: Return per-stamp arrays in cube order instead of scalar means.

Returns a dict with `ee_box`, `ee_circ`, `r_circ` (the inscribed-circle
radius in arcsec), and `r_ee` (the radius enclosing `ee_fraction`, `nan`
when not reached).

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
