# Simulation and verification

Mophongo verifies its fitting scheme on synthetic mosaics with injected
truth: every source has a known position and flux, so recovered fluxes,
errors, and residuals can be compared against exact answers.
{class}`mophongo.mock_mosaic.MockMosaic` generates JWST-like NIRCam and MIRI
mosaics whose WCS, PSF, noise, and drizzle properties match real reductions
closely enough that the products drop into the {doc}`pipeline` unchanged.
`mophongo.verification` supplies the reusable bookkeeping around such runs:
standard mock configurations, PSF/kernel map construction, source-recovery
tables, and diagnostic figures.

## Synthetic mosaics: MockMosaic

`MockMosaic` is a dataclass that acts as both configuration and factory. A
typical run chains four steps, or calls
{meth}`~mophongo.mock_mosaic.MockMosaic.build` which does all four:

```python
from mophongo.mock_mosaic import MockMosaic, Pointing

mock = MockMosaic(
    out_dir="mock_out",
    center_radec=(34.5, -5.2),
    nircam_lw_frames={"f444w": [Pointing(34.5, -5.2, pa=0.0)]},
    miri_frames={"f770w": [Pointing(34.5, -5.2, pa=0.0)]},
    stpsf_dir="data/PSF",
    noise_seed=42,
)
paths = mock.write()                       # per-filter wcs.csv + FITS stubs
noise_info = mock.inject_noise_all(paths)  # noise images + wht maps
dpsfs = mock.load_drizzle_psfs(paths)      # DrizzlePSF per filter
truth = mock.inject_point_sources(paths, dpsfs, n=200)
```

Products, per filter `<f>`, in `out_dir`:

- `mock_<f>_wcs.csv` — one row per detector exposure with the SIP WCS,
  `exptime`, and `mjd-avg` columns consumed by
  {class}`mophongo.psf.DrizzlePSF` and
  {class}`mophongo.psf_map.PSFRegionMap`.
- `mock_<f>_sci.fits` — mosaic image (noise plus injected sources), with
  `KERNEL`/`PIXFRAC` drizzle keywords and `BUNIT` in the header.
- `mock_<f>_wht.fits` — weight map (see convention below), with `WHTTYPE`,
  `RNOISE`, `NOISEK`, `INPSCALE`, and `OUTPSCAL` provenance keywords.
- `mock_<f>_truth.fits` — noiseless image of the injected sources only.
- `mock_truth.ecsv` — truth catalog (written by `build`; call
  `truth.write(...)` yourself when running the steps individually).

Mosaic grids for the three detector families are nested at 20/40/80 mas by
integer factors of two, aligned by the half-pixel CRPIX rule
({func}`mophongo.mock_mosaic.nested_crpix`), so block-binning a fine grid
reproduces the coarse grid exactly. This matches the multi-resolution
convention used by the pipeline ({doc}`templates`).

### Noise model and weight convention

Noise follows the count-rate drizzle convention. With per-pixel total
exposure time `t_exp(x, y)` (sum of frame `EXPTIME` over covering
footprints), output pixel scale `p_out`, and a per-filter calibration
constant `K`:

```
sigma_nominal(x, y) = K / (p_out * sqrt(t_exp(x, y)))
sigma_pix(x, y)     = R(pixfrac, p_in, p_out) * sigma_nominal(x, y)
wht(x, y)           = 1 / sigma_pix(x, y)**2
```

`R` is the Fruchter (2011) square-kernel drizzle correlation factor,
computed by {func}`mophongo.mock_mosaic.drizzle_correlation_factor`. The
written `wht` map is therefore actual per-pixel inverse variance:
`1/sqrt(wht)` is the RMS of the noise actually injected into the science
image, including exposure-count, output-pixel-area, and drizzle-correlation
factors. This is the weight convention the whole package assumes
({doc}`fitting`). For real mosaics whose weights use a different convention,
`mophongo.mock_mosaic.DEFAULT_WHT_CALIB` provides empirically calibrated
per-filter scalars that convert those weights to actual inverse variance
(`wht_real * wht_calib = 1/sigma_pix**2`); a value of 1.0 means the real
weight already is inverse variance.

Module constants:

- `NATIVE_PSCALE` — native detector pixel scales in arcsec: `nircam_sw`
  0.031, `nircam_lw` 0.063, `miri` 0.110.
- `DEFAULT_OUTPUT_PSCALE` — mosaic output scales: `nircam_sw` 0.020,
  `nircam_lw` 0.040, `miri` 0.080.
- `DEFAULT_NOISE_K` — per-filter `K` values (units `BUNIT * arcsec *
  sqrt(s)`) calibrated against real deep JWST reductions; entries exist for
  `f444w`, `f770w`, `f1280w`, `f1500w`, and `f1800w`.
- `DEFAULT_WHT_CALIB` — per-filter weight calibration scalars for the same
  reductions.
- `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC` — extra Gaussian broadening of the
  effective in-mosaic PSF over the drizzled STPSF model, per MIRI filter
  (FWHM in arcsec, e.g. 0.08 for F770W), derived from star tests (some
  filters interpolated or extrapolated along the measured trend). The mock
  injects this blur; any model-PSF chain fitting such data must apply the
  same broadening before kernel construction ({doc}`psf`).
- `PSF_BLUR_FWHM_PER_SIGMA` — FWHM/sigma conversion (2.355) shared by every
  blur path so all apply the identical operator.

### Pointing

`Pointing` is a dataclass describing a single JWST pointing:

`ra` : `float`, required
: Right ascension of the aperture reference point, degrees.

`dec` : `float`, required
: Declination, degrees.

`pa` : `float`, required
: Position angle of the V3 axis, degrees.

### MockMosaic fields

`out_dir` : `Path`, required
: Output directory for all products.

`center_radec` : `tuple[float, float]`, default `(34.5, -5.2)`
: Sky center (RA, Dec) in degrees; used as CRVAL unless `mosaic_crval` is
  set.

`nircam_sw_frames`, `nircam_lw_frames`, `miri_frames` : `dict[str, list[Pointing]]`, default `{}`
: Filter name to list of pointings, per detector family. Each pointing
  expands to one WCS row per detector in the family (8 SW detectors, 2 LW,
  1 MIRI) unless restricted by `detectors`.

`mosaic_pscale` : `str`, default `"nircam_lw"`
: Family key of `DEFAULT_OUTPUT_PSCALE` defining the reference grid; the
  other families nest from it via the half-pixel CRPIX rule.

`mosaic_npix` : `tuple[int, int] | None`, default `None`
: Reference-grid mosaic size (nx, ny). `None` auto-fits the union of all
  configured detector footprints.

`mosaic_crval` : `tuple[float, float] | None`, default `None`
: Explicit CRVAL; `None` uses `center_radec`.

`mosaic_crpix` : `tuple[float, float] | None`, default `None`
: Explicit reference-grid CRPIX; `None` centers on the footprint union,
  snapped so all nested scales land on half-integer CRPIX values.

`mjd_avg` : `float`, default `59960.26`
: `MJD-AVG` written to every WCS row (drives MJD-aware PSF selection).

`exptime` : `float | dict[str, float]`, default `418.734`
: Per-frame exposure time in seconds; scalar or per-filter dict.

`noise_K` : `dict[str, float]`, default `{}`
: Per-filter noise constants overriding `DEFAULT_NOISE_K`.

`pixfrac` : `float | dict[str, float]`, default `0.75`
: Drizzle pixfrac; scalar, or dict keyed by filter or family.

`noise_seed` : `int | None`, default `None`
: Random seed for noise and source injection.

`stpsf_dir` : `Path | None`, default `None`
: Directory holding STPSF ePSF grid files (falls back to `data/PSF`).

`stpsf_patterns` : `dict[str, str]`, default `{}`
: Per-filter ePSF filename patterns overriding
  {meth}`~mophongo.mock_mosaic.MockMosaic.default_stpsf_pattern`.

`detectors` : `dict[str, tuple[str, ...]]`, default `{}`
: Optional detector-key restriction keyed by filter or family, e.g.
  `{"f444w": ("NRCA5",)}` for a single-detector frame.

`snr_range` : `tuple[float, float]`, default `(5.0, 5000.0)`
: Log-uniform matched-filter SNR range for injected sources.

`apertures_arcsec` : `tuple[float, ...]`, default `(0.32, 0.7)`
: Circular aperture diameters for truth aperture-flux columns.

`psf_size_arcsec` : `float | dict[str, float]`, default `2.0`
: PSF stamp size in arcsec; must be large enough to hold the full ePSF
  footprint (8 arcsec is safe for NIRCam LW and MIRI grids).

`source_sigma_pix` : `float | tuple[float, float] | None`, default `None`
: Intrinsic circular Gaussian source sigma, in pixels on
  `source_sigma_pscale`. `None`/0 injects pure point sources; a two-value
  tuple draws log-uniform sizes between the bounds.

`source_sigma_pscale` : `float`, default `0.040`
: Pixel scale (arcsec) on which `source_sigma_pix` is defined.

`point_source_fraction` : `float`, default `0.0`
: Fraction of sources forced to be point sources when `source_sigma_pix`
  requests extended profiles.

`source_psf_normalization` : `str`, default `"native"`
: `"native"` preserves the finite-stamp PSF integral returned by
  `DrizzlePSF` (so the image contains `flux_true * psf_throughput` within
  the stamp, matching the package shape/throughput convention);
  `"unit"` is an explicit legacy convention that renormalizes stamps to
  unit sum.

`psf_gaussian_fwhm_arcsec` : `float | dict[str, float] | None`, default copy of `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC`
: Extra Gaussian PSF broadening per filter, FWHM in arcsec. Pass `0.0` or
  `{}` to disable.

`psf_gaussian_fwhm_pix` : `float | dict[str, float] | None`, default `None`
: Legacy broadening override in output pixels; takes precedence over the
  arcsec form when set. Prefer `psf_gaussian_fwhm_arcsec`.

`bunit` : `str`, default `"10.0*nanoJansky"`
: `BUNIT` written to the science image headers.

### Methods

`build(n_sources=200, psf_dir=None, ref_filter="f444w",
sample_filters=None)` — the driver: chains `write`, `inject_noise_all`,
`load_drizzle_psfs`, and `inject_point_sources`, writes `mock_truth.ecsv`,
and returns `(paths, noise_info, dpsfs, truth)`.

The step methods it chains, and the remaining utilities, are summarized
below; follow the links for full signatures.

{meth}`~mophongo.mock_mosaic.MockMosaic.write` — emits the per-filter
`wcs.csv` and empty mosaic FITS stubs, and returns the per-filter `paths`
dict that all later steps consume.

{meth}`~mophongo.mock_mosaic.MockMosaic.inject_noise` — rasterizes the
exposure-time map from the detector footprints, draws Gaussian noise
following the convention above, overwrites `_sci.fits`, and writes
`_wht.fits`; returns a per-filter noise-info dict.
{meth}`~mophongo.mock_mosaic.MockMosaic.inject_noise_all` runs it for every
filter with a known `K`.

{meth}`~mophongo.mock_mosaic.MockMosaic.load_drizzle_psfs` — returns a
per-filter {class}`~mophongo.psf.DrizzlePSF` with its STPSF ePSF grid
loaded.

{meth}`~mophongo.mock_mosaic.MockMosaic.sample_positions` —
rejection-samples (RA, Dec) positions uniformly inside the valid
drizzled-PSF coverage.

{meth}`~mophongo.mock_mosaic.MockMosaic.inject_point_sources` — injects `n`
sources and returns the truth table. Per-source flux is set from a
log-uniform target SNR via the matched filter on the reference-filter
weight map, and the same true flux is painted in every filter through
{meth}`~mophongo.mock_mosaic.MockMosaic.get_filter_psf_radec`; keyword
options select the sampling footprint, explicit positions, per-filter
position offsets (for astrometric-recovery tests), and source-profile
overrides. Truth columns record `id`/`ra`/`dec`, source-profile metadata,
`snr_<ref>`, and per filter `x_<f>`, `y_<f>`, `flux_<f>`, aperture fluxes,
blur bookkeeping, and `valid_<f>`.

{meth}`~mophongo.mock_mosaic.MockMosaic.source_model_templates` — returns
the exact unit-flux source models painted into the mock as a
{class}`mophongo.templates.Templates` collection, useful for separating
template-extraction errors from the linear flux solve.

{meth}`~mophongo.mock_mosaic.MockMosaic.report` — logs per-filter mosaic
shape, coverage, and valid-source counts.
{meth}`~mophongo.mock_mosaic.MockMosaic.plot` — diagnostic figure with the
science mosaics, detector footprints, and truth sources
(`mock_diagnostic.png`).

{meth}`~mophongo.mock_mosaic.MockMosaic.blur_filter_psf` — applies the
configured extra Gaussian blur to a PSF stamp or cube on its own grid. The
operator is an exact analytic Gaussian transfer function in Fourier space
({func}`mophongo.mock_mosaic.gaussian_blur_fourier`), so it is
grid-independent and exact for sub-pixel sigmas.
{meth}`~mophongo.mock_mosaic.MockMosaic.get_filter_psf_radec` — the mock's
PSF creation hook: delegates to
{meth}`mophongo.psf.DrizzlePSF.get_psf_radec`, then applies
`blur_filter_psf`.

{meth}`~mophongo.mock_mosaic.MockMosaic.default_stpsf_pattern` (static) —
default ePSF filename pattern per filter.

`to_dict()` / `from_dict(d)` / `to_json(path)` / `from_json(path)` — full
config round-trip; JSON-loaded lists are coerced back to tuples and
`Pointing` objects.

### Module functions

- `drizzle_correlation_factor(pixfrac, input_pscale, output_pscale)` —
  square-kernel pixel-to-resel RMS ratio `R` in (0, 1].
- `nested_crpix(crpix_ref, ratio)` — CRPIX on a grid `ratio` times finer,
  via `crpix_fine = ratio * crpix_ref - (ratio - 1)/2`.
- `gaussian_blur_fourier(arr, sigma_pix)` — flux-conserving Gaussian blur by
  the exact analytic transfer function over the last two axes.
- `gaussian_blur_psf(psf, fwhm_arcsec, pscale)` — the shared angular-blur
  operator: converts FWHM to grid sigma and calls `gaussian_blur_fourier`.

## Verification framework

`mophongo.verification` encodes reusable bookkeeping for injected-truth
runs, independent of any particular survey layout. The standard scenario:
build a mock, build PSF and kernel region maps consistent with it, run the
{doc}`pipeline` against the truth-labelled segmentation, and reduce the
outputs to recovery tables and diagnostic figures.

```python
from mophongo import verification as ver

mock, paths, noise_info, dpsfs, truth = ver.build_realistic_two_detector_mock(
    "verify_out", psf_dir="data/PSF", nsrc=300, seed=42,
)
psf_maps = ver.build_wiener_psf_maps(
    mock, paths, dpsfs, "verify_out", psf_dir="data/PSF",
)
result = ver.run_pipeline_extension_scenario(
    "psf_convolution", out_dir="verify_out", paths=paths, noise_info=noise_info,
    truth=truth, psf_maps=psf_maps,
)
```

```{figure} images/mock_flux_recovery.png
:width: 100%
:alt: Four-panel flux-recovery diagnostic comparing recovered and true fluxes on an injected-truth mock

Flux-recovery diagnostic for an F770W fit on the standard two-detector mock,
as written by `save_flux_recovery_plot`: recovered versus true flux, the
recovered/true ratio against true flux with a matched-filter SNR axis, the
distribution of error-normalized residuals with MAD Gaussian fits split at
SNR 20, and residuals against recovered flux. Recovery tracks the one-to-one
line across the full flux range, with scatter growing toward low SNR as the
orange predicted-error envelope indicates.
```

### Result dataclasses

All four are frozen dataclasses; follow the links for field-level detail.

- {class}`~mophongo.verification.PSFShape` — a unit-sum PSF `shape` with
  its finite-stamp `throughput`, encoding the package convention that
  fitting uses unit-sum shapes while the stamp sum is filter-level
  throughput metadata.
- {class}`~mophongo.verification.WHTNoiseCheck` — summary statistics of
  `(sci - truth) * sqrt(wht)` for one filter.
- {class}`~mophongo.verification.WienerPSFMaps` — the source, target, and
  kernel {class}`mophongo.psf_map.PSFRegionMap` trio with the chosen Wiener
  lambda and per-region throughputs.
- {class}`~mophongo.verification.PipelineScenarioResult` — one scenario
  run: the pipeline, fit and source-recovery tables, residual and model
  images, output directory, and `summary`. The `summary` dict carries the
  recovery statistics downstream scripts consume: `med_hi`/`med_lo` and
  `p16_lo`/`p84_lo` (recovered/true flux medians and quantiles),
  `pull_lo_median`/`pull_lo_std` (error-normalized pulls),
  `resid_std_over_noise`, plus run bookkeeping (`template_extension`,
  `n_fit`, `n_position_mismatched`, `wiener_lambda`,
  `n_source_diagnostics`).

### PSF shape and throughput helpers

- {func}`~mophongo.verification.prepare_psf_shape` — wraps a stamp as a
  `PSFShape` without renormalizing the native stamp in place.
- {func}`~mophongo.verification.filter_average_throughput` — mean of the
  finite positive stamp sums; feeds `Pipeline(psf_throughputs=...)`.
- {func}`~mophongo.verification.parse_regularization_grid` — parses a
  string or sequence into a validated positive float array.
- {func}`~mophongo.verification.psf_centroid_info` — peak and
  center-of-mass centroid of a stamp; measurement only.
- {func}`~mophongo.verification.apply_mock_filter_blur_on_grid` — applies
  `MockMosaic.blur_filter_psf` on the given grid so PSF/kernel maps receive
  the identical blur operator as the painted sources.

### Pointing generators

- {func}`~mophongo.verification.offset_pointing` — `Pointing` offset from a
  center by small sky offsets.
- {func}`~mophongo.verification.native_phase_dither_pointings` —
  deterministic dithers sampling native-pixel sub-pixel phases for a
  detector family; {func}`~mophongo.verification.nircam_lw_phase_pointings`
  is the NIRCam LW specialization.
- {func}`~mophongo.verification.miri_center_for_nircam_detector` — MIRI
  pointing center whose footprint centroid matches one named NIRCam LW
  detector.
- {func}`~mophongo.verification.miri_two_macro_phase_pointings` — two MIRI
  macro positions aligned to the LW detectors, each with deterministic
  phase dithers.
- {func}`~mophongo.verification.write_pointing_summary` — writes
  `wcs_products.csv` listing the WCS CSV and mosaic FITS paths, frame
  counts, and pixel scales.

### Weight sanity checks

- {func}`~mophongo.verification.wht_noise_check` — verifies that a weight
  map is actual inverse variance: over valid pixels,
  `(sci - truth) * sqrt(wht)` should have unit scatter. Returns a
  `WHTNoiseCheck`; {func}`~mophongo.verification.wht_noise_check_from_fits`
  is the FITS-loading wrapper.
- {func}`~mophongo.verification.actual_inverse_variance` — inverse variance
  rebuilt from the mock's stored `sigma_pix`, falling back to the given
  weight map.

### The standard two-detector mock

`build_realistic_two_detector_mock(out_dir, *, psf_dir, ...)` constructs the
standard realistic verification setup — two NIRCam LW detectors with six
phase dithers in F444W, and two MIRI macro pointings (aligned to the LW
detectors) with eight phase dithers each in F770W — then writes, injects
noise and sources, saves `mock_config.json`, `wcs_products.csv`,
`mock_truth.ecsv`, and `mock_mosaic.png`, and returns
`(mock, paths, noise_info, dpsfs, truth)`. Sources are calibrated on F770W
(`ref_filter="f770w"`) and sampled inside the F444W/F770W footprint
intersection, so every source has a template in both bands.

Keyword parameters (see
{func}`~mophongo.verification.build_realistic_two_detector_mock` for the
full list and defaults) control the source count and seed, SNR and
intrinsic-size ranges and point-source fraction, field center and mosaic
size, ePSF filename patterns and detector restrictions, an optional
deliberate F770W source-position shift for astrometric-recovery tests, and
the extra PSF broadening.

### PSF and kernel maps

`build_wiener_psf_maps(mock, paths, dpsfs, out_dir, *, psf_dir, ...)` builds
{class}`mophongo.psf_map.PSFRegionMap` maps for a source and target filter
plus their overlay kernel map, evaluates a drizzled PSF at each region
centroid on the source-filter grid, applies the mock's blur, and optimizes a
single Wiener regularization with
{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization`. Native
finite-stamp sums are preserved as throughput metadata; the maps carry only
unit-sum shapes and matching kernels. It writes `diagnostic_wiener.png`,
`psf_kernel_wiener_lambda_scan.csv`, `psf_kernel_wiener_results.csv`, and
three GeoJSON region maps, and returns a `WienerPSFMaps`.

```{figure} images/mock_kernel_diagnostic.png
:width: 100%
:alt: Wiener matching-kernel diagnostic with regularization scan, radial profiles, and PSF, kernel, and residual images

The `diagnostic_wiener.png` product for the F444W-to-F770W kernel of the
two-detector mock: figure-of-merit scan over the Wiener regularization with
the selected lambda marked, radial profiles and encircled-energy ratio of the
matched versus target PSF, and images of the source PSF, target PSF, kernel,
convolved source, and convolution residual. The matched growth curve stays
within the 2% tolerance band at all radii.
```

Keyword parameters: `reg_grid` (`Sequence[float]`,
`DEFAULT_WIENER_REG_GRID`) — regularization scan grid; `kernel_grid_nside`
(`int`, 1) — kept for older callers, must be 1; `source_pattern` /
`target_pattern` (`str`, the F444W/F770W defaults above); `source_filter`
(`str`, `"f444w"`); `target_filter` (`str`, `"f770w"`); `psf_size_arcsec`
(`float`, 8.0) — PSF stamp size; `target_label` (`str | None`, `None`) —
display name of the target band used in figure captions and labels,
defaulting to the internal band name.

### Truth matching and recovery tables

- {func}`~mophongo.verification.remap_detection_to_truth` — runs
  {class}`mophongo.catalog.Catalog` detection on the high-resolution mock
  image and relabels the segmentation map with truth-source ids; a blended
  segment is assigned to its brightest truth member, and other members and
  undetected sources get a 3x3 stamp.
- {func}`~mophongo.verification.build_source_recovery_table` — joins a
  pipeline catalog ({doc}`outputs`) to the truth table by id, keeping raw
  amplitudes as `flux_<i>_model` and totals as `flux_<i>_total`, with
  ratio and pull columns (`ratio_<i>`, `pull_<i>_pred`, `pull_<i>_cov`)
  comparing totals to `flux_true`.
- {func}`~mophongo.verification.segment_weighted_positions` — flux-weighted
  mean (x, y) per segment; used to flag truth/segment mismatches.
- {func}`~mophongo.verification.residual_to_native` — block-sums an
  upsampled residual back to the native low-resolution grid.

### Scenario runner

`run_pipeline_extension_scenario(scenario, *, out_dir, paths, noise_info,
truth, psf_maps, ...)` runs one complete verification pipeline pass into
`out_dir/template_extension_<scenario>/` and returns a
`PipelineScenarioResult`. It builds the truth-labelled segmentation, fits
F444W and F770W with {meth}`mophongo.pipeline.Pipeline.run`, and writes the
model/residual FITS, the source-recovery CSV, flux-recovery figures for both
bands, per-source stage diagnostics
({meth}`mophongo.pipeline.Pipeline.diagnose_sources`), and scene
diagnostics. The `summary` dict includes median flux ratios, pull statistics,
and (when a position shift was injected) recovered-vs-expected shift fields.

Parameters: `scenario` (`str`, required) — template build scheme passed to
`Pipeline(extend_mode=...)`: `"psf_convolution"`, `"psf_wings"`, or
`"psf_model"`, with `"none"` disabling extension;
`out_dir`, `paths`, `noise_info`, `truth`, `psf_maps` (required,
keyword-only) — products of the builders above. Optional keywords:
`mock_dilate_segmap` (`int`, 2) — dilation when building the truth
segmentation; `template_dilate_segmap` (`int`, 4) — pipeline
`FitConfig.template_dilate_segmap`; `fit_astrometry_niter` (`int`, 2);
`fit_background` (`bool`, `False`) — subtract a fitted background from
F770W; `source_diagnostic_count` (`int`, 10) — number of bright sources in
the stage-diagnostic figure; `full_diagnostic_highres_size` (`int | None`,
3000) — crop size of the full-image diagnostic, `None`/0 skips it;
`scene_diagnostic_count` (`int | None`, 12) — number of scenes plotted;
`f770w_position_shift_xy` (`tuple | None`, `None`) — the injected shift, for
recovery bookkeeping; `nsrc` (`int | None`, `None`) and `sigma_range`
(`tuple`, `(1.0, 5.0)`) and `point_source_fraction` (`float`, 0.10) —
caption metadata only; `max_match_offset_pix` (`float`, 3.0) — maximum
segment-centroid offset before a row is flagged position-mismatched and
excluded from the recovery plots; `fit_overrides` (`dict | None`, `None`) —
extra `FitConfig` keywords merged over the scenario defaults, e.g. a
per-band `aperture_diam` matching a production run; `target_label` (`str`,
`"F770W"`) — display name of the fitted band used in captions and axis
labels, so the F770W-keyed mock slots can carry another band's name.

### Diagnostic figure helpers

These emit the standard verification figures; see {doc}`diagnostics` for the
pipeline-side diagnostics they build on, and the API links for full
signatures.

- {func}`~mophongo.verification.diagnostic_note` — compact caption string
  summarizing a scenario.
- {func}`~mophongo.verification.diagnostic_lupton_norm` — the shared asinh
  display normalization.
- {func}`~mophongo.verification.label_segmap` — overlays catalog positions
  as index labels.
- {func}`~mophongo.verification.save_diagnostic_image` — the standard
  truth/hires/segmap/lowres/model/residual panel figure, with an optional
  residual-pull stamp grid.
- {func}`~mophongo.verification.bright_source_residual_stamp_grid` —
  residual/noise stamps of the brightest sources, annotated to mark
  deblended detections.
- {func}`~mophongo.verification.crop_from_origin` and
  {func}`~mophongo.verification.segmap_lower_left_origin` — cropping
  utilities used by
  {func}`~mophongo.verification.save_realistic_full_diagnostic`, which
  crops all panels to the lower-left covered tile and calls
  `save_diagnostic_image`.
- {func}`~mophongo.verification.save_scene_diagnostics` — writes
  `scene_catalog.csv`, a `scene_overview.png` via
  {func}`~mophongo.verification.save_scene_overview`, and per-scene
  `Scene.plot` figures for the largest fitted scenes.
- {func}`~mophongo.verification.save_flux_recovery_plot` — the standard
  four-panel recovery figure: flux vs truth, ratio vs truth with a SNR
  axis, pull histogram with MAD Gaussian fits, and pull vs recovered flux.
- {func}`~mophongo.verification.plot_saturated_catalog_repair` —
  before/after cutout panels for
  {func}`mophongo.catalog.repair_saturated_catalog` merges.

## Relation to real-data runs

Verification of the fitting scheme happens on these mocks, where truth is
known exactly. The same conventions the mock enforces — inverse-variance
weights, unit-sum PSF shapes with throughput carried as metadata
({doc}`psf`), flux-conserving block replication with `factor**2` weight
scaling across resolutions ({doc}`templates`) — are the ones the pipeline
assumes on real data, so a configuration validated here transfers directly.
