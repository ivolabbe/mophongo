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

`write()` — emits the per-filter `wcs.csv` and empty mosaic FITS stubs, and
returns the `paths` dict (per filter: `csv`, `fits`, `wcs`, `n_rows`,
`crpix`, `size`, `pscale`, `family`) that all later steps consume.

`inject_noise(filter_name, paths, K=None, pixfrac=None, seed=None,
bunit=None, dpsf=None)` — rasterizes the exposure-time map from detector
footprints, draws Gaussian noise, overwrites `_sci.fits`, and writes
`_wht.fits`. Parameters override the corresponding config fields; `dpsf`
supplies a pre-built {class}`mophongo.psf.DrizzlePSF`. Returns a dict with
`sci`, `wht`, `texp`, `sigma_nom`, `sigma_pix`, `R`, `K`, `p_in`, `p_out`,
`pixfrac`. `inject_noise_all(paths, **kwargs)` runs it for every filter with
a known `K` (skipping the rest) and forwards `kwargs`.

`load_drizzle_psfs(paths, psf_dir=None, stpsf_patterns=None)` — returns a
per-filter `DrizzlePSF` with its STPSF ePSF grid loaded from `psf_dir`
(default: `stpsf_dir` or `data/PSF`); `stpsf_patterns` overrides filename
patterns per filter.

`sample_positions(n, dpsf, seed=None, oversample=4, coverage=None)` —
rejection-samples `n` (RA, Dec) positions uniformly inside the valid
drizzled-PSF coverage of `dpsf` (or an explicit shapely `coverage`
geometry); `oversample` controls the candidate batch size.

`inject_point_sources(paths, dpsfs, n=100, snr_range=None,
ref_filter="f444w", apertures_arcsec=None, psf_size_arcsec=None,
source_sigma_pix=None, source_sigma_pscale=None,
point_source_fraction=None, source_psf_normalization=None,
sample_filters=None, positions_radec=None,
filter_position_offsets_pix=None, seed=None)` — injects `n` sources and
returns the truth table. `None` for any config-mirroring parameter falls
back to the corresponding field. Positions are sampled inside the
`ref_filter` coverage, or inside the intersection of the filters named in
`sample_filters` so every source has a template in each; `positions_radec`
supplies explicit position arrays instead. Per-source flux is set from the
target SNR via the matched filter on the `ref_filter` weight map
(`SNR = F * sqrt(sum(P**2 * wht))`), and the same true flux is painted in
every filter through {meth}`~mophongo.mock_mosaic.MockMosaic.get_filter_psf_radec`.
`filter_position_offsets_pix` shifts painted positions in named filters by
(dx, dy) mosaic pixels, for astrometric-recovery tests. `source_sigma_pix`
may also be a length-`n` array of per-source sigmas. Truth columns: `id`,
`ra`, `dec`, `source_sigma_pix`, `source_sigma_arcsec`, `is_point_source`,
`source_psf_normalization`, `snr_<ref>`, and per filter `x_<f>`, `y_<f>`,
`flux_<f>`, `flux_aper_D<ddd>_<f>` (one per aperture diameter),
`psf_gaussian_fwhm_pix_<f>`, `psf_gaussian_fwhm_arcsec_<f>`, `valid_<f>`.

`source_model_templates(filter_name, *, paths=None, ids=None,
normalize=False)` — returns the exact unit-flux source models painted into
the mock as a {class}`mophongo.templates.Templates` collection, useful for
separating template-extraction errors from the linear flux solve. `ids`
restricts to a source-id subset; `normalize` forwards to
`Templates.from_cutout_models`.

`build(n_sources=200, psf_dir=None, ref_filter="f444w",
sample_filters=None)` — chains `write`, `inject_noise_all`,
`load_drizzle_psfs`, and `inject_point_sources`, writes `mock_truth.ecsv`,
and returns `(paths, noise_info, dpsfs, truth)`.

`report()` — logs per-filter mosaic shape, coverage, and valid-source
counts. `plot(save=True, figsize=None, ref_snr="snr_f444w",
stretch_sigma=2.0, dpi=900)` — diagnostic figure with the science mosaics,
detector footprints, and truth sources; `save=True` writes
`mock_diagnostic.png` in `out_dir`, a path saves there instead.

`blur_filter_psf(filter_name, psf, *, pscale=None)` — applies the configured
extra Gaussian blur to a PSF stamp or cube on its own grid (`pscale` in
arcsec; `None` uses the filter's output scale). The operator is an exact
analytic Gaussian transfer function in Fourier space
({func}`mophongo.mock_mosaic.gaussian_blur_fourier`), so it is
grid-independent and exact for sub-pixel sigmas.
`get_filter_psf_radec(filter_name, dpsf, positions, *, filter_pattern=None,
size=None, verbose=False)` — the mock's PSF creation hook: delegates to
{meth}`mophongo.psf.DrizzlePSF.get_psf_radec`, then applies
`blur_filter_psf`.

`default_stpsf_pattern(filter_name)` (static) — default ePSF filename
pattern; as of this writing the literal values are
`"UDS_MIRI_<FILTER>_OS4_GRID1"` for MIRI and
`"UDS_NRC.._<FILTER>_OS4_GRID1"` for NIRCam.

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
    "psf", out_dir="verify_out", paths=paths, noise_info=noise_info,
    truth=truth, psf_maps=psf_maps,
)
```

### Result dataclasses

All four are frozen dataclasses.

- `PSFShape` — `shape` (unit-sum PSF array) and `throughput` (finite-stamp
  sum). Encodes the package convention that fitting uses unit-sum shapes
  while the finite stamp sum is filter-level throughput metadata.
- `WHTNoiseCheck` — summary of `(sci - truth) * sqrt(wht)`: `filter_name`,
  `n_pix`, `std`, `mad`, `median`, plus `wht_type` and `rnoise` header
  values (both default `None`).
- `WienerPSFMaps` — `source_map`, `target_map`, `kernel_map`
  ({class}`mophongo.psf_map.PSFRegionMap` instances), `wiener_lambda`, and
  `source_throughputs`/`target_throughputs` arrays.
- `PipelineScenarioResult` — `scenario`, `pipeline`, `table`,
  `source_table`, `residuals`, `residual_native`, `model_native`,
  `output_dir`, `summary`.

### PSF shape and throughput helpers

- `prepare_psf_shape(psf, label="PSF")` — returns a `PSFShape` without
  renormalizing the native stamp in place; raises on a non-positive sum.
- `filter_average_throughput(values)` — mean of the finite positive stamp
  sums; 1.0 when none are usable. Feeds `Pipeline(psf_throughputs=...)`.
- `parse_regularization_grid(value)` — parses a comma-separated string or
  sequence into a validated positive float array
  (`DEFAULT_WIENER_REG_GRID` spans 1e-6 to 0.1).
- `psf_centroid_info(psf, prefix)` — peak and center-of-mass centroid of a
  stamp, returned as `<prefix>_*` dict entries; measurement only.
- `apply_mock_filter_blur_on_grid(mock, filter_name, psf, *, grid_pscale)` —
  applies `MockMosaic.blur_filter_psf` on the given grid so PSF/kernel maps
  receive the identical blur operator as the painted sources.

### Pointing generators

- `offset_pointing(center, *, dx_arcsec, dy_arcsec, pa)` — `Pointing`
  offset from `center` by small sky offsets.
- `native_phase_dither_pointings(center, *, family, n_dither, pa)` — up to 8
  deterministic dithers sampling native-pixel sub-pixel phases for the given
  detector family; `n_dither <= 1` returns the center pointing.
- `nircam_lw_phase_pointings(center, *, pa=0.0, n_dither=6)` — the NIRCam LW
  specialization.
- `miri_center_for_nircam_detector(center, *, detector, pa=0.0,
  miri_detector=("MIRIM",))` — MIRI pointing center whose footprint centroid
  matches one named NIRCam LW detector.
- `miri_two_macro_phase_pointings(center, *, pa=0.0,
  nircam_detectors=("NRCA5", "NRCB5"), miri_detector=("MIRIM",),
  n_dither=8)` — two MIRI macro positions aligned to the LW detectors, each
  with deterministic phase dithers.
- `write_pointing_summary(paths, out_dir)` — writes `wcs_products.csv`
  listing the WCS CSV and mosaic FITS paths, frame counts, and pixel scales.

### Weight sanity checks

- `wht_noise_check(sci, truth, wht, *, filter_name="", header=None)` —
  verifies that a weight map is actual inverse variance: over valid pixels,
  `(sci - truth) * sqrt(wht)` should have unit scatter. Returns a
  `WHTNoiseCheck`. `wht_noise_check_from_fits(sci_path, truth_path,
  wht_path, *, filter_name="")` is the FITS-loading wrapper.
- `actual_inverse_variance(noise_info, filter_name, fallback_wht)` —
  inverse variance rebuilt from the mock's stored `sigma_pix`, falling back
  to the given weight map.

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

`out_dir` (path, required); `psf_dir` (path, required, keyword-only) — STPSF
grid directory. Remaining keyword parameters:

- `nsrc` (`int`, 300) — number of injected sources.
- `center` (`tuple`, `(34.50, -5.20)`) — field center (RA, Dec) degrees.
- `pa` (`float`, 0.0) — position angle.
- `snr_range` (`tuple`, `(1.0, 500.0)`) — log-uniform SNR range.
- `sigma_range` (`tuple`, `(1.0, 5.0)`) — intrinsic source sigma range in
  40 mas pixels.
- `point_source_fraction` (`float`, 0.10) — fraction forced to point
  sources.
- `seed` (`int`, 42) — random seed.
- `image_size` (`int | None`, `None`) — square mosaic size on the reference
  grid; `None` auto-fits the footprints.
- `source_pattern` / `target_pattern` (`str`, `DEFAULT_F444W_PATTERN` /
  `DEFAULT_F770W_PATTERN`) — ePSF filename patterns for F444W and F770W.
- `nircam_detectors` (`Sequence[str]`, `("NRCA5", "NRCB5")`) and
  `miri_detector` (`Sequence[str]`, `("MIRIM",)`) — detector restrictions.
- `f770w_position_shift_xy` (`tuple | None`, `None`) — deliberate F770W
  source-position shift in F770W mosaic pixels (limited to ±1 per axis), for
  astrometric-recovery tests.
- `psf_gaussian_fwhm_arcsec` (`float | dict | None`, `None`) — extra PSF
  broadening; `None` keeps the `MockMosaic` default, `0.0` or `{}` disables.

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

Keyword parameters: `reg_grid` (`Sequence[float]`,
`DEFAULT_WIENER_REG_GRID`) — regularization scan grid; `kernel_grid_nside`
(`int`, 1) — kept for older callers, must be 1; `source_pattern` /
`target_pattern` (`str`, the F444W/F770W defaults above); `source_filter`
(`str`, `"f444w"`); `target_filter` (`str`, `"f770w"`); `psf_size_arcsec`
(`float`, 8.0) — PSF stamp size.

### Truth matching and recovery tables

- `remap_detection_to_truth(sci_path, wht_path, truth, *, ndilate=0)` —
  runs {class}`mophongo.catalog.Catalog` detection on the high-resolution
  mock image and relabels the segmentation map with truth-source ids
  (`ndilate` optionally dilates segments first). A blended segment
  containing several truth sources is assigned to its brightest member;
  other members and undetected sources get a 3x3 stamp. Returns
  `(segmap, catalog)` with catalog deblend metadata (`is_deblended`,
  `deblend_parent_label`, `deblend_nchildren`) propagated.
- `build_source_recovery_table(fit_table, truth, *, true_flux_col,
  fitted_indices, snr_col=None, sigma_col="source_sigma_pix",
  point_source_col="is_point_source", template_extension=None)` — joins a
  pipeline catalog ({doc}`outputs`) to the truth table by id. For each
  fitted image index `i`, raw amplitudes are kept as `flux_<i>_model`;
  `flux_<i>_total` is read from the catalog when present, otherwise computed
  from `throughput_<i>`; ratio and pull columns (`ratio_<i>`,
  `pull_<i>_pred`, `pull_<i>_cov`) compare totals to `flux_true`. The
  optional `snr_col`/`sigma_col`/`point_source_col` columns are copied when
  present in `truth`, the deblend metadata columns when present in
  `fit_table`; `template_extension` stamps a constant label column.
- `segment_weighted_positions(image, segmap, ids)` — flux-weighted mean
  (x, y) per segment; NaN for segments without positive flux. Used to flag
  truth/segment mismatches.
- `residual_to_native(residual, native_shape)` — block-sums an upsampled
  residual back to the native low-resolution grid (inverse of the
  flux-conserving block replication used by multi-resolution fitting).

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

Parameters: `scenario` (`str`, required) — template-extension mode passed to
`Pipeline(extend_templates=...)`: `"psf"`, `"psf_wings"`, or `"psf_model"`,
with `"none"` disabling extension;
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
per-band `aperture_diam` matching a production run.

### Diagnostic figure helpers

These emit the standard verification figures; see {doc}`diagnostics` for the
pipeline-side diagnostics they build on.

- `diagnostic_note(*, nsrc, sigma_range, point_source_fraction,
  template_dilate_segmap, wiener_lambda=None, template_extension=None,
  f770w_position_shift_xy=None)` — compact caption string summarizing a
  scenario.
- `diagnostic_lupton_norm(img)` — the shared asinh display normalization.
- `label_segmap(ax, segmap, catalog, fontsize=10)` — overlays catalog
  positions as index labels; no-op without `x`/`y` columns.
- `save_diagnostic_image(filename, truth, hires, lowres, model, residual,
  *, segmap=None, catalog=None, caption=None, stamp_grid=None,
  stamp_grid_title="bright-source residual pulls",
  stamp_grid_labels=None, dpi=None)` — the standard
  truth/hires/segmap/lowres/model/residual panel figure, with an optional
  residual-pull stamp grid; `dpi=None` picks a DPI that samples every image
  pixel.
- `bright_source_residual_stamp_grid(residual, ivar, ids, truth_by_id,
  is_deblended_by_id, *, filt="f770w", n_stamps=16, half_size=6)` —
  residual/noise stamps of the brightest sources plus label annotations
  marking deblended detections; returns `(grid, labels)`.
- `crop_from_origin(arr, shape, origin_yx)` and
  `segmap_lower_left_origin(segmap)` — cropping utilities used by
  `save_realistic_full_diagnostic(path, *, paths, img_444, img_770_bgsub,
  model_770, resid_770, segmap, caption, stamp_grid, stamp_grid_labels,
  highres_size)`, which crops all panels to the lower-left covered tile of
  size `highres_size` (high-resolution pixels) and calls
  `save_diagnostic_image`.
- `save_scene_diagnostics(pipe, tmpl_image, segmap, out_dir, *,
  scene_collection_index=-1, max_scenes=12, display_sig=5.0)` — writes
  `scene_catalog.csv` (per-scene template counts, bounding boxes, median
  fitted astrometric shifts), a `scene_overview.png` via
  `save_scene_overview(image, segmap, scenes, filename, *, alpha=0.42)`,
  and per-scene `Scene.plot` figures for the `max_scenes` largest scenes of
  the selected fitted-scene collection.
- `save_flux_recovery_plot(filename, truth, recovered, *, error=None,
  label="Recovered Flux", xlabel="True Flux", ylabel="Recovered Flux",
  snr_values=None, point_source_mask=None, deblended_mask=None,
  error_label="Error", systematic_error_fraction=0.0, caption=None)` — the
  standard four-panel recovery figure: flux vs truth, ratio vs truth with a
  SNR axis, pull histogram with MAD Gaussian fits, and pull vs recovered
  flux. `systematic_error_fraction` adds a fractional error floor in
  quadrature; the masks highlight point sources and deblended children.
- `plot_saturated_catalog_repair(sci, seg_before, seg_after, cat_before,
  cat_after, merge_log, *, out_path, n_sources=4, half_size=None,
  pad_factor=1.8, id_col="id", x_col="x", y_col="y", select_ids=None,
  asinh_a=0.1, sci_percentiles=(1.0, 99.5))` — before/after cutout panels
  for {func}`mophongo.catalog.repair_saturated_catalog` merges: science
  image, oversplit children, and merged parent per selected `merge_log` row
  (`select_ids` picks explicit parents, otherwise the `n_sources` rows with
  most children); `half_size=None` sizes cutouts from the parent segment
  area scaled by `pad_factor`; `asinh_a` and `sci_percentiles` set the
  stretch.

## Relation to real-data runs

Verification of the fitting scheme happens on these mocks, where truth is
known exactly. The same conventions the mock enforces — inverse-variance
weights, unit-sum PSF shapes with throughput carried as metadata
({doc}`psf`), flux-conserving block replication with `factor**2` weight
scaling across resolutions ({doc}`templates`) — are the ones the pipeline
assumes on real data, so a configuration validated here transfers directly.
