# The pipeline

{class}`mophongo.pipeline.Pipeline` orchestrates a full template-fitting
photometry run: it extracts per-source templates from a high-resolution
detection image, PSF-matches them to each low-resolution band, solves for the
template amplitudes scene by scene, and writes a flux catalog with residual
images. The module-level {func}`mophongo.pipeline.run` function is a
backward-compatible wrapper around the class.

There are two ways to drive it:

- **In memory**: construct a `Pipeline` from arrays you have already loaded
  (images, segmentation map, catalog, PSFs, kernels) and call
  {meth}`~mophongo.pipeline.Pipeline.run`.
- **Config-driven**: describe one hi-res + one lo-res band pair in a JSON file
  ({class}`~mophongo.pipeline.RunConfig`), then
  `Pipeline.from_config("run.json").run_all()`, or run individual steps. The
  same flow is available from the command line:

```
python -m mophongo.pipeline run.json [steps]
```

where `steps` is any of `psfs`, `kernels`, `load`, `loadfit`, `info`, `fit`,
`outputs`, `all` (default `all`).

## Processing flow

`Pipeline.run()` executes the following stages. Image index 0 is always the
high-resolution reference (detection) image; indices 1 and up are the bands to
fit.

1. **Catalog skeleton.** The output table starts from the input catalog's
   `id`, `x`, `y` columns, plus deblending provenance
   (`is_deblended`, `deblend_parent_label`, `deblend_nchildren`) and any
   `FLAG_SATURATED_*` columns when present. Flux columns are appended per
   fitted band.

2. **Template extraction.** {class}`mophongo.templates.Templates` cuts one
   template per source from `images[0]` using the segmentation map: pixels
   inside a source's segment (optionally dilated by
   `FitConfig.template_dilate_segmap`) form its template. Sources flagged as
   stars (`flag_star` catalog column) or saturated are marked on their
   templates; saturated sources are later isolated into their own scenes. A
   prebuilt `Templates` collection can be passed instead via the `templates`
   argument, in which case extraction and extension are skipped. See
   {doc}`templates`.

3. **Template extension** (optional, controlled by `extend_templates`).
   Segment-truncated templates miss PSF-wing flux, which biases total fluxes
   low, most severely for faint sources.

   - `"psf_wings"` (the config-driven default): add the high-resolution PSF
     wings beyond the segmentation footprint
     ({meth}`mophongo.templates.Templates.extend_with_psf_wings`). By default
     only background pixels are filled, so blended neighbours keep ownership
     of their own segment pixels (`FitConfig.extend_wings_background_only`).
   - `"psf_model"`: replace the template entirely by the PSF
     ({meth}`mophongo.templates.Templates.extend_with_psf_model`).
   - `None`: leave templates truncated at the segment boundary.

   Both modes require a high-resolution PSF in `psfs[0]` (an array or a
   {class}`mophongo.psf_map.PSFRegionMap`). Extension is a shape operation:
   PSF stamps are normalized to unit sum internally and the native
   finite-support sum is kept only as throughput metadata (see below).

4. **Per-band loop.** For each fitted image `i >= 1`:

   - **Convolution.** Templates are convolved with the matching kernel
     `kernels[i]` (a static array or a `PSFRegionMap` for spatially varying
     kernels) to match the band's PSF.
   - **Multi-resolution binning.** The integer bin factor between the
     reference and band grids is derived from the two WCS objects
     ({func}`mophongo.utils.bin_factor_from_wcs`). With the default
     `multi_resolution_method="upsample"`, the band's science image is
     block-replicated onto the reference grid with flux conservation, and its
     inverse-variance map is copied to the subpixels and multiplied by
     `factor**2` (this preserves the native chi-square; flux-conserving
     replication must not be used for weights). Convolved templates are then
     projected onto the block-replicated grid. With `"downsample"`, templates
     and kernels are instead downsampled to the band's native grid.
   - **Scene generation.** Templates are grouped into scenes of coupled
     sources ({func}`mophongo.scene.generate_scenes`), controlled by
     `FitConfig.scene_coupling_thresh`, `scene_max_size`,
     `scene_minimum_bright`, and `scene_max_merge_radius`. See {doc}`fitting`.
   - **Fitting.** Each scene is solved for template amplitudes, iterating up
     to `FitConfig.fit_astrometry_niter` passes with per-template astrometric
     shifts applied between passes, stopping early once the largest shift
     increment falls below `FitConfig.astrom_shift_tol` (fit-grid pixels).
   - **Residual and model.** The summed scene models are subtracted from the
     band image to form the residual; the model image is kept as well.
   - **Flux columns.** Fitted amplitudes, uncertainties, predicted errors,
     the filter throughput, and throughput-corrected totals are written into
     the catalog (see {doc}`outputs`). Aperture photometry on model+residual
     follows, using the configured aperture or its FWHM-based default.

5. **Return.** `Pipeline.run()` returns `(table, residuals)`; the module-level
   {func}`mophongo.pipeline.run` returns `(table, residuals, pipeline)`. The
   fitted state stays on the instance: `table`, `residuals`, `model_images`,
   `all_templates`, `all_scenes`, `tmpls` (hi-res templates),
   `templates_extracted`, and `templates_extended`.

### Conventions

- **Weights are inverse variance.** `1/sqrt(wht)` is the per-pixel RMS.
- **PSF shape vs throughput.** Fitting uses unit-sum PSF shapes; the finite
  stamp sum is filter-level throughput metadata. `flux_<i>` holds the raw
  fitted template amplitude; `flux_<i>_total` divides by the encircled energy
  of the low-resolution PSF stamp at each source position (per-source
  `ee_psf_lo` where available, else the filter-mean stamp sum). Native PSF
  stamps are never silently renormalized in a way that loses the throughput.
- **Coordinates.** Catalog `x`, `y` are pixels on the high-resolution grid;
  templates carry both original-image and cutout coordinate systems.

## `pipeline.run()` and the `Pipeline` constructor

{func}`mophongo.pipeline.run` and `Pipeline.__init__` take the same
arguments; the wrapper constructs a `Pipeline`, calls its
{meth}`~mophongo.pipeline.Pipeline.run` method, and additionally returns the
pipeline instance.

```python
import mophongo

table, residuals, pipe = mophongo.pipeline.run(
    [hires, lowres],          # images: index 0 = detection, 1.. = fitted bands
    segmap,                   # segmentation map on the hi-res grid
    catalog=cat,              # astropy Table with id, x, y
    psfs=[psf_hi, psf_lo],    # per-image PSFs (arrays or PSFRegionMaps)
    weights=[None, ivar_lo],  # inverse-variance maps
    kernels=[None, kernel],   # matching kernels for images[1:]
    wcs=[wcs_hi, wcs_lo],
    extend_templates="psf_wings",
    config=mophongo.fit.FitConfig(),
)
```

All arguments after `images` and `segmap` are keyword-only:

- `images` (*Sequence[np.ndarray]*, required) — image list; `images[0]` is
  the high-resolution reference, `images[1:]` are fitted.
- `segmap` (*np.ndarray*, required) — segmentation map on the reference grid;
  pixel labels are catalog ids.
- `catalog` (*astropy.table.Table | None*, default `None`) — source catalog
  with at least `id`, `x`, `y` (reference-grid pixels). Required in practice:
  catalog generation from the segmentation map is not implemented as of this
  writing.
- `psfs` (*Sequence[np.ndarray | PSFRegionMap] | None*, default `None`) — one
  PSF per image; `psfs[0]` is the high-resolution PSF used for template
  extension. Must match `len(images)` when given.
- `weights` (*Sequence[np.ndarray] | None*, default `None`) — inverse-variance
  maps, one per image (`None` entries allowed). Must match `len(images)`.
- `wht_images` (*Sequence[np.ndarray] | None*, default `None`) — alias for
  `weights`, used only when `weights` is not given.
- `kernels` (*Sequence[np.ndarray | PSFRegionMap] | None*, default `None`) —
  PSF-matching kernels per image (`kernels[0]` unused). A
  {class}`~mophongo.psf_map.PSFRegionMap` gives spatially varying kernels.
- `psf_throughputs` (*Sequence[float] | None*, default `None`) — explicit
  filter-level finite-support PSF sums, one per image, used for
  `flux_<i>_total` when callers pass unit-sum PSF shapes in `psfs`. Must
  match `len(images)` when given. When `None`, the throughput is measured as
  the mean stamp sum of `psfs[i]`.
- `wcs` (*Sequence[astropy.wcs.WCS] | None*, default `None`) — one WCS per
  image; needed for multi-resolution bin factors, PSF-map lookups by sky
  position, and arcsec apertures.
- `window` (default `None`) — accepted and stored, but not used anywhere in
  the pipeline as of this writing.
- `extend_templates` (*str | None*, default `None`) — `"psf_wings"`,
  `"psf_model"`, or `None` (see the flow above; `"psf"` is accepted as a
  synonym for `"psf_wings"`).
- `templates` (*Templates | Sequence[Template] | None*, default `None`) —
  prebuilt templates; skips extraction and extension when given.
- `config` (*FitConfig | None*, default `None`) — fitting configuration; a
  default {class}`mophongo.fit.FitConfig` is created when `None`.

{meth}`Pipeline.run <mophongo.pipeline.Pipeline.run>` itself takes one
optional argument:

- `config` (*FitConfig | None*, default `None`) — when given, replaces the
  stored configuration (it stays on the instance for later calls).

## `RunConfig`: config-driven runs

{class}`mophongo.pipeline.RunConfig` holds the inputs for one filter fit (one
hi-res + one lo-res band). It loads from JSON with
{meth}`~mophongo.pipeline.RunConfig.from_json` (lines starting with `#` are
stripped, so the file can carry comments); unknown keys raise an error.
{meth}`Pipeline.from_config <mophongo.pipeline.Pipeline.from_config>` accepts
either a path or a `RunConfig` instance and defers data loading until
{meth}`~mophongo.pipeline.Pipeline.run` or
{meth}`~mophongo.pipeline.Pipeline.load_data`.

```python
from mophongo.pipeline import Pipeline

pipe = Pipeline.from_config("run.json")
pipe.info()        # inspect inputs without loading pixel data
pipe.run_all()     # build_psfs + build_kernels + run + write_outputs, logged
```

Required fields:

- `name` (*str*) — run label; prefixes every output file.
- `out_dir` (*str*) — output directory for products and the geojson PSF and
  kernel caches (never inputs).
- `sci_hi` (*str*) — high-resolution template (detection) image, FITS.
- `segmap` (*str*) — segmentation map on the hi-res grid; labels are catalog
  ids.
- `catalog` (*str*) — source catalog with `id`, `x`, `y` (hi-res pixels),
  `ra`, `dec`.
- `sci_lo` (*str*) — low-resolution science mosaic to fit.
- `wht_lo` (*str*) — low-resolution weight (inverse-variance) map.
- `csv_hi`, `csv_lo` (*str*) — per-frame WCS csv of the hi-res and lo-res
  mosaics, used by {class}`mophongo.psf.DrizzlePSF` to drizzle position-
  dependent PSFs. See "Per-frame WCS CSVs" below for the file contents and
  how to generate them.

Optional fields:

- `driz_hi` (*str | None*, default `None`) — mosaic used for `DrizzlePSF`
  footprints of the hi-res side; defaults to `sci_hi`. Set it when `sci_hi`
  is a derived template image.
- `psf_dir` (*str*, default `"data/PSF"`) — directory of STDPSF grid files.
- `pattern_hi`, `pattern_lo` (*str*, default `""`) — STDPSF filename regex
  for each band, of the form
  `{prefix}_{DET}_{FILT}[_MJD..]_GRID{N}_{OS4|DET}`.
- `filter_lo` (*str*, default `""`) — lo-res filter name (e.g. `"f770w"`),
  used for the default Gaussian-blur lookup.
- `psf_size` (*float | None*, default `4.0`) — PSF stamp size in arcsec;
  `None` keeps the full native ePSF stamp.
- `psf_autobuild` (*bool*, default `True`) — generate missing PSF grids with
  {class}`mophongo.psf_factory.PSFFactory` (see caching below).
- `psf_fov_arcsec` (*float | None*, default `None`) — PSFFactory field of
  view; `None` uses the backend default.
- `psf_blur_fwhm` (*float | str | None*, default `"default"`) — extra
  Gaussian broadening of the lo-res model PSF (FWHM, arcsec). `"default"`
  looks up a per-filter value keyed by `filter_lo`; a number applies that
  value; `None` applies no broadening.
- `expect_frames` (*list[int] | None*, default `None`) — optional
  `[n_frames_hi, n_frames_lo]` sanity check against the WCS csvs; a mismatch
  raises.
- `extend_templates` (*str | None*, default `"psf_wings"`) — template
  extension mode, as above.
- `bg_filter_sigma` (*float*, default `64.0`) — background filter scale
  passed to {func}`mophongo.catalog.get_bg_and_ivar`; the fitted image is the
  lo-res science image minus this background.
- `footprint_filter` (*bool*, default `True`) — keep only catalog sources
  where `wht_lo > 0`.
- `r_trial` (*float*, default `0.0`) — trial-patch radius in arcmin; `0`
  fits the full catalog. Requires `trial_center` when positive.
- `trial_center` (*list[float] | None*, default `None`) — `[ra, dec]` in
  degrees of the trial patch.
- `fit` (*dict*, default `{}`) — keyword arguments forwarded to
  {class}`mophongo.fit.FitConfig`.
- `scene_plots` (*bool*, default `True`) — write per-scene diagnostic PNGs
  in {meth}`~mophongo.pipeline.Pipeline.write_outputs`.
- `save_stamps` (*bool*, default `True`) — write the per-source stamps FITS
  (native-size hi/lo templates plus fit metadata; see
  {meth}`~mophongo.pipeline.Pipeline.write_stamps`).

### Per-frame WCS CSVs

`csv_hi` and `csv_lo` follow grizli's `_wcs.csv` companion-file convention:
one row per exposure that entered the mosaic, holding the frame name and SCI
extension index, the frame's WCS keywords (`crpix*`, `crval*`, the CD
matrix, SIP `a_*`/`b_*` distortion coefficients), array sizes (`naxis*`),
exposure time, and observation dates (`mjd-beg`, `mjd-avg`, `mjd-end`). From
these rows {class}`mophongo.psf.DrizzlePSF` rebuilds each frame's WCS and
sky footprint, so it can drizzle a position-dependent PSF from the exposures
— and observation dates — that cover a given position.

{func}`mophongo.utils.reconstruct_wcs` generates the file from public MAST
data without downloading any frames: per frame it fetches only the header
byte range of the `_cal.fits` file and writes one CSV row.

```python
from mophongo.utils import reconstruct_wcs

reconstruct_wcs("mosaic-f770w_drz_sci.fits", out_csv="frames_lo.csv")
```

Parameters (all but `mosaic` keyword-only):

- `mosaic` (*str | Path*, required) — drizzled mosaic FITS path.
- `out_csv` (*str | Path | None*, default `None`) — output CSV path; `None`
  writes `<stem>_wcs.recon.csv` next to the mosaic.
- `source` (*str*, default `"s3"`) — where headers are fetched from: the
  public MAST S3 bucket (`"s3"`) or the MAST download API (`"mast"`).
- `workers` (*int*, default `32`) — parallel header fetches.
- `filelist` (*str | Path | None*, default `None`) — text file with one
  input frame name per line, overriding the frame-list discovery below.
- `limit` (*int*, default `0`) — when positive, reconstruct only the first
  `limit` frames (debugging).
- `companion_csv` (*str | Path | None*, default `None`) — fallback
  frame-list source (its `file` column) when the mosaic header names no
  frames.

The function returns the table as a `pandas.DataFrame` and writes it to
`out_csv`. The frame list is discovered from, in order: `filelist` when
given; `FLT*` keys in the mosaic primary header; the
`Files used to create mosaic` COMMENT block; the `file` column of
`companion_csv` (or of an existing `<stem>_wcs.csv` next to the mosaic).
When none of these is available the function raises.

{func}`mophongo.utils.write_wcs_csv` builds the same table by reading the
frame headers themselves, for the case where the stage-2 frames are on local
disk:

- `mosaic_or_glob` (*Path | str*, required) — a glob pattern matching local
  stage-2 frames (e.g. `"data/*/F770W/stage2/*_cal.fits"`), or a mosaic
  FITS whose primary-header COMMENT cards name the JWST datasets (the frame
  headers are then read from MAST).
- `out_csv` (*str | None*, default `None`) — output path; `None` derives
  `<stem>_wcs.csv` from the mosaic (or first matched frame) name by
  stripping any `_drz_wht`/`_drz_sci`/`_i2d`/`_wht`/`_sci` suffix.

**Auto-reconstruction.** The CSVs need not exist before a run.
{meth}`mophongo.psf.DrizzlePSF.read_wcs_csv` derives the CSV path from the
mosaic name when none is given (stripping `_drz_sci`/`_drc_sci`/`_sci`
and appending `_wcs.csv`), and when the file is missing and
`auto_reconstruct=True` (the default) it calls
{func}`mophongo.utils.reconstruct_wcs` to create it at that path. In
config-driven runs the pipeline passes `csv_hi`/`csv_lo` explicitly, so a
missing file is reconstructed at the configured path on first use.

### Step methods

Config-driven runs decompose into steps, each returning `self` so they chain;
expensive products are cached in `out_dir`:

- {meth}`~mophongo.pipeline.Pipeline.build_psfs` `(overwrite=False)` — build
  or reload the per-band PSF region maps, with PSFs drizzled at each map's own
  region centroids.
- {meth}`~mophongo.pipeline.Pipeline.build_kernels` `(overwrite=False,
  method="wiener", reg=None)` — build or reload the matching-kernel map on
  the hi/lo overlay. `method` is passed to
  {func}`mophongo.utils.matching_kernel`; when `reg` is `None` and the
  method is regularized, the regularization is optimized once on the median
  PSF shape and reused for every region. Kernels are matched between
  unit-sum PSF shapes and renormalized to unit sum, so the kernel carries no
  flux scale of its own.
- {meth}`~mophongo.pipeline.Pipeline.load_data` `(kernels=True)` — read
  images, segmentation map, and catalog; subtract the background; zero
  non-finite pixels in both image and weight; apply the footprint and trial
  filters. With `kernels=False` the PSF/kernel maps are skipped for quick
  inspection and built later by `run()`.
- {meth}`~mophongo.pipeline.Pipeline.run` — the fit itself (loads data
  and maps first when needed).
- {meth}`~mophongo.pipeline.Pipeline.write_outputs` — write
  `<name>_residual.fits` (on the hi-res reference grid, with the `sci_hi`
  header), `<name>_fit_table.fits`, the stamps file (when `save_stamps`),
  a scene catalog CSV (`id`, `n_templates`, `is_bright`, `ra`, `dec`), and
  optional per-scene PNGs.
- {meth}`~mophongo.pipeline.Pipeline.run_all` — all of the above in
  order, with everything the run emits (logging, `print`, progress bars)
  captured to `<out_dir>/<name>.log`.
- {meth}`~mophongo.pipeline.Pipeline.load_fit` `(ifilt=1)` — restore the
  post-run state from written outputs without refitting: reads the fit
  table and residual, rebuilds fitted templates from the stamps file
  (regenerating and rewriting it through the run's template path when
  missing). Scenes are not persisted and are not restored.
- {meth}`~mophongo.pipeline.Pipeline.info` — print a summary of config,
  inputs, cache state, loaded data, and results at any stage.

Inspection and diagnostic helpers
({meth}`~mophongo.pipeline.Pipeline.plot_inputs`,
{meth}`~mophongo.pipeline.Pipeline.show_sources`,
{meth}`~mophongo.pipeline.Pipeline.diagnose_sources`,
{meth}`~mophongo.pipeline.Pipeline.source_products`,
{meth}`~mophongo.pipeline.Pipeline.plot_subphot`) are described in
{doc}`diagnostics`.

## `FitConfig` reference

{class}`mophongo.fit.FitConfig` configures the solver, astrometry, apertures,
template extension behavior, and scene processing. In config-driven runs its
fields come from the `fit` dict of the JSON config.

Solver:

- `positivity` (*bool*, default `True`) — constrain fitted amplitudes to be
  non-negative.
- `reg_flux` (*float*, default `0.0`) — flux regularization strength.
- `bad_value` (*float*, default `np.nan`) — fill value for catalog entries of
  sources that were not fitted.
- `cg_kwargs` (*dict*, default `{"M": None, "maxiter": 500, "atol": 1e-6}`) —
  keyword arguments for the conjugate-gradient solver.
- `normal` (*str*, default `"tree"`) — normal-matrix assembly strategy,
  `"loop"` or `"tree"`.
- `multi_resolution_method` (*str*, default `"upsample"`) — `"upsample"`
  (block-replicate the lo-res image onto the reference grid) or
  `"downsample"` (downsample templates and kernels to the lo-res grid).

Astrometry:

- `fit_astrometry_niter` (*int*, default `5`) — maximum astrometry
  refinement passes; `0` disables shift fitting (one fitting pass still
  runs).
- `astrom_shift_tol` (*float*, default `0.05`) — stop iterating once the
  largest per-template shift increment of a pass drops below this tolerance
  (fit-grid pixels).
- `fit_astrometry_joint` (*bool*, default `True`) — solve shifts jointly
  with fluxes rather than as a separate step.
- `reg_astrom` (*float*, default `1e-4`) — regularization of the astrometric
  shift solve.
- `snr_thresh_astrom` (*float*, default `15.0`) — minimum SNR for a source
  to constrain astrometry; `0` keeps all sources.
- `astrom_isolation_thresh` (*float*, default `0.7`) — minimum flux
  dominance (0–1) for inclusion in the astrometric fit; `0.0` applies no
  cut.
- `astrom_exclude_stars` (*bool*, default `False`) — exclude sources flagged
  `is_star` from the shift fit. Off by default: unsaturated stars are good
  astrometric anchors, and saturated ones are already isolated into their
  own scenes.
- `astrom_model` (*str*, default `"gp"`) — spatial shift model,
  `"poly"` or `"gp"`; any other value raises.
- `astrom_centroid` (*str*, default `"centroid"`) — shift measurement,
  `"centroid"` or `"correlation"`.
- `astrom_kwargs` (*dict*, default
  `{"poly": {"order": 0}, "gp": {"length_scale": 400}}`) — per-model
  parameters.

Apertures:

- `aperture_diam` (*float | np.ndarray | None*, default `None`) —
  measurement aperture **diameter** on the fitted image: a scalar applies to
  all bands, an array of length `len(images) - 1` gives one per band, and
  `None` falls back to an aperture *radius* of 1.5 times the PSF FWHM in
  pixels (3.0-pixel radius when no PSF is available). Aperture photometry
  runs either way; setting a diameter additionally writes the aperture-size
  column `aper_<i>`.
- `aperture_catalog` (*float | str | None*, default `None`) — catalog
  aperture: a fixed diameter, the name of a catalog column with per-source
  diameters, or `None`.
- `aperture_units` (*str*, default `"arcsec"`) — units of the two aperture
  settings, `"arcsec"` or `"pix"`.

Templates:

- `template_dilate_segmap` (*int*, default `0`) — dilate each segment by
  this disk radius (pixels) before extraction. Off by default: dilation
  mostly adds a ring of sky noise, and wing recovery is the job of template
  extension.
- `skip_template_extension_for_deblended` (*bool*, default `False`) — leave
  catalog deblend children unextended; by default extension applies to every
  source.
- `extend_wings_background_only` (*bool*, default `True`) — PSF-wing
  completion fills only background pixels of the (dilated) segmentation map,
  so blended neighbours keep ownership of their own pixels. `False` fills
  every zero template pixel.

Scenes:

- `run_scene_solver` (*bool*, default `True`) — must stay `True`; the scene
  solver is the only fitting path and `False` raises.
- `scene_coupling_thresh` (*float*, default `1e-3`) — template-coupling
  (leakage) threshold for splitting scenes.
- `scene_max_size` (*int | None*, default `800`) — soft cap on templates per
  scene; oversized components are split by raising the coupling threshold
  locally. `None` disables the cap.
- `scene_max_merge_radius` (*float*, default `1000.0`) — maximum distance
  (pixels) over which underfilled scenes are merged.
- `scene_minimum_bright` (*int | None*, default `5`) — minimum number of
  bright sources per scene; when set to `None` it is derived from the
  astrometric polynomial order as `(order + 1) * (order + 2) + 1`.
- `generate_scene_catalog` (*bool*, default `False`) — write a scene catalog
  (`scene_catalog_<i>.ecsv`) and exit without fitting.

## Caching, provenance, and PSF auto-build

Config-driven runs cache the expensive PSF products as geojson region maps in
`out_dir` (see {doc}`psf_maps`):

- `<name>_psf_hi.geojson`, `<name>_psf_lo.geojson` — per-band
  {class}`~mophongo.psf_map.PSFRegionMap` objects: exposure-overlap regions
  with a PSF drizzled at each region centroid. The lo-res map already
  includes the configured Gaussian broadening.
- `<name>_kernel.geojson` — the matching-kernel map on the hi/lo region
  overlay, with kernels built from PSF pairs drizzled at the overlay
  centroids.

Each cached map carries provenance columns recording what produced it. PSF
maps record the STDPSF `pattern`, `psf_size`, and `blur_fwhm`; the kernel map
records `kernel_method`, `kernel_reg`, and `psf_size`. On reload,
`build_psfs` compares the cached provenance against the current config and
rebuilds when any field disagrees (a map written before provenance existed
also counts as stale). `build_kernels` reuses a cached map only when its
recorded method matches the requested one — the matching method changes the
flux scale at the percent level, so a map built another way is rebuilt rather
than silently reused. Passing `overwrite=True` to either step forces a
rebuild.

**PSF grid auto-build.** Loading the ePSF grids matches files under `psf_dir`
against `pattern_hi`/`pattern_lo`. When no file matches and `psf_autobuild`
is `True` (the default), the pipeline derives the generator settings
(prefix, grid size, oversampling, detector sampling, MJD tagging) from the
pattern itself, runs {class}`mophongo.psf_factory.PSFFactory` over the band's
exposure csv, writes the grids into `psf_dir`, and loads them; deriving the
settings from the search pattern guarantees the generated files are found
again. This step is slow (it generates PSFs with stpsf). With
`psf_autobuild=False` a missing grid raises `FileNotFoundError` instead of
letting the run continue without a PSF. See {doc}`psf`.

Other cached state: the stamps file (`<name>_stamps.fits`) records the grid
shapes it was written for, and {meth}`~mophongo.pipeline.Pipeline.load_fit`
rejects a stamps file whose grids no longer match the loaded images.

For the catalog columns the run produces, see {doc}`outputs`; for the
detection side, see {doc}`catalog`; for building test data with injected
truth, see {doc}`simulation`.
