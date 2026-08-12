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

```python
from mophongo.mock_mosaic import nested_crpix

crpix_80 = (1024.5, 1024.5)            # 80 mas reference grid
crpix_40 = nested_crpix(crpix_80, 2)   # 40 mas, 2x finer
crpix_20 = nested_crpix(crpix_80, 4)   # 20 mas, 4x finer
print(crpix_40, crpix_20)              # (2048.5, 2048.5) (4096.5, 4096.5)
print(nested_crpix(crpix_40, 2) == crpix_20)  # True: nesting composes
```

Most of the ~25 config fields keep their defaults; the ones users actually
set are visible in the example above: `out_dir`, the per-family frame dicts
(`nircam_sw_frames`, `nircam_lw_frames`, `miri_frames`, each mapping a
filter name to a list of {class}`~mophongo.mock_mosaic.Pointing` — RA, Dec,
and V3 position angle in degrees), `stpsf_dir` for the STPSF ePSF grids, and
`noise_seed` for reproducibility. Beyond those, `exptime` and `pixfrac`
(scalars or per-filter dicts) are the common knobs for matching a real
reduction's depth and drizzle setup. The full field list — mosaic grid
overrides, source-injection defaults, PSF broadening, detector restrictions
— is documented on the {class}`~mophongo.mock_mosaic.MockMosaic` API page,
and `to_json`/`from_json` round-trip the entire configuration.

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

```python
from mophongo.mock_mosaic import drizzle_correlation_factor

# R = sigma_pix / sigma_nominal for square-kernel drizzling
for pixfrac in (1.0, 0.75):
    for p_in, p_out in [(0.063, 0.040), (0.110, 0.080)]:
        R = drizzle_correlation_factor(pixfrac, p_in, p_out)
        print(f"pixfrac={pixfrac:.2f}  {p_in:.3f}\" -> {p_out:.3f}\"  R={R:.3f}")
```

Drizzling NIRCam LW onto the 40 mas grid at `pixfrac=1` gives `R=0.501`, and
MIRI onto 80 mas gives `R=0.551`; shrinking `pixfrac` to 0.75 raises them to
0.608 and 0.656, moving the per-pixel RMS closer to the nominal uncorrelated
value.

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

### Building, injecting, and inspecting

{meth}`~mophongo.mock_mosaic.MockMosaic.build` is the driver: it chains
{meth}`~mophongo.mock_mosaic.MockMosaic.write` (per-filter `wcs.csv` and
mosaic FITS stubs), {meth}`~mophongo.mock_mosaic.MockMosaic.inject_noise_all`
(exposure-time maps and Gaussian noise following the convention above, via
{meth}`~mophongo.mock_mosaic.MockMosaic.inject_noise` per filter),
{meth}`~mophongo.mock_mosaic.MockMosaic.load_drizzle_psfs` (a
{class}`~mophongo.psf.DrizzlePSF` per filter), and
{meth}`~mophongo.mock_mosaic.MockMosaic.inject_point_sources`, writes
`mock_truth.ecsv`, and returns `(paths, noise_info, dpsfs, truth)`.

{meth}`~mophongo.mock_mosaic.MockMosaic.inject_point_sources` injects `n`
sources and returns the truth table. Per-source flux is set from a
log-uniform target SNR via the matched filter on the reference-filter
weight map, and the same true flux is painted in every filter through
{meth}`~mophongo.mock_mosaic.MockMosaic.get_filter_psf_radec`. Positions are
rejection-sampled inside the valid drizzled-PSF coverage
({meth}`~mophongo.mock_mosaic.MockMosaic.sample_positions`); keyword options
select the sampling footprint, explicit positions, per-filter position
offsets (for astrometric-recovery tests), and source-profile overrides.
Truth columns record `id`/`ra`/`dec`, source-profile metadata, `snr_<ref>`,
and per filter `x_<f>`, `y_<f>`, `flux_<f>`, aperture fluxes, blur
bookkeeping, and `valid_<f>`. The exact unit-flux source models painted into
the mock are available afterwards as a
{class}`mophongo.templates.Templates` collection via
{meth}`~mophongo.mock_mosaic.MockMosaic.source_model_templates`, useful for
separating template-extraction errors from the linear flux solve.

For inspection, {meth}`~mophongo.mock_mosaic.MockMosaic.report` logs
per-filter mosaic shape, coverage, and valid-source counts, and
{meth}`~mophongo.mock_mosaic.MockMosaic.plot` writes a diagnostic figure
with the science mosaics, detector footprints, and truth sources
(`mock_diagnostic.png`).

The configured extra Gaussian blur is applied through
{meth}`~mophongo.mock_mosaic.MockMosaic.blur_filter_psf`, an exact analytic
Gaussian transfer function in Fourier space
({func}`mophongo.mock_mosaic.gaussian_blur_fourier`), so it is
grid-independent and exact for sub-pixel sigmas;
{meth}`~mophongo.mock_mosaic.MockMosaic.get_filter_psf_radec` delegates to
{meth}`mophongo.psf.DrizzlePSF.get_psf_radec` and then applies it. The same
operator is available standalone, driven by the per-filter FWHM lookup:

```python
from mophongo.psf import PSF
from mophongo.mock_mosaic import DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC, gaussian_blur_psf

psf = PSF.gaussian(101, 3.0).array           # model PSF on an 80 mas grid
fwhm = DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC["f770w"]
blurred = gaussian_blur_psf(psf, fwhm, pscale=0.080)
print(fwhm, round(psf.sum(), 6), round(blurred.sum(), 6))  # 0.08 1.0 1.0
print(round(blurred.max() / psf.max(), 3))   # 0.9: the core is broadened
```

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

{func}`~mophongo.verification.build_wiener_psf_maps` builds
{class}`mophongo.psf_map.PSFRegionMap` maps for a source and target filter
(defaulting to F444W and F770W) plus their overlay kernel map, evaluates a
drizzled PSF at each region centroid on the source-filter grid, applies the
mock's blur, and optimizes a single Wiener regularization with
{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization`. Native
finite-stamp sums are preserved as throughput metadata; the maps carry only
unit-sum shapes and matching kernels. It writes `diagnostic_wiener.png`,
`psf_kernel_wiener_lambda_scan.csv`, `psf_kernel_wiener_results.csv`, and
three GeoJSON region maps, and returns a `WienerPSFMaps`:

```python
psf_maps = ver.build_wiener_psf_maps(
    mock, paths, dpsfs, "verify_out", psf_dir="data/PSF",
)
```

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

{func}`~mophongo.verification.run_pipeline_extension_scenario` runs one
complete verification pipeline pass into
`out_dir/template_extension_<scenario>/` and returns a
`PipelineScenarioResult`. The `scenario` argument is the template build
scheme passed to `Pipeline(extend_mode=...)`: `"psf_convolution"`,
`"psf_wings"`, or `"psf_model"`, with `"none"` disabling extension. It
builds the truth-labelled segmentation, fits F444W and F770W with
{meth}`mophongo.pipeline.Pipeline.run`, and writes the model/residual FITS,
the source-recovery CSV, flux-recovery figures for both bands, per-source
stage diagnostics ({meth}`mophongo.pipeline.Pipeline.diagnose_sources`), and
scene diagnostics. The `summary` dict includes median flux ratios, pull
statistics, and (when a position shift was injected) recovered-vs-expected
shift fields.

```python
result = ver.run_pipeline_extension_scenario(
    "psf_convolution", out_dir="verify_out", paths=paths, noise_info=noise_info,
    truth=truth, psf_maps=psf_maps,
    fit_overrides={"aperture_diam": 0.7},  # extra FitConfig keywords
)
```

Keyword options (see the API reference for the full list) control the
segmentation dilation, astrometric iterations, background fitting, the
diagnostic figure counts and crop sizes, the truth-matching offset cut, and
`fit_overrides` — extra {class}`~mophongo.fit.FitConfig` keywords merged
over the scenario defaults, e.g. a per-band `aperture_diam` matching a
production run.

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
