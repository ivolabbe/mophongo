# Detection and catalogs

The `mophongo.catalog` module owns source detection, segmentation maps, and
source catalogs. Its central object is {class}`mophongo.catalog.Catalog`, a
dataclass that turns a high-resolution science image and weight map into a
segmentation map plus a measurement table. The segmentation map and the
catalog table are the inputs the photometry pipeline needs: segment labels
become source ids, and each labelled region defines the footprint of one
template (see {doc}`templates` and {doc}`pipeline`).

Detection and photometric measurement are built on
[photutils](https://photutils.readthedocs.io/): `detect_sources` for
segmentation, `deblend_sources` for separating blended sources, and
`SourceCatalog` for the measured quantities.

## Conventions

- **Weights are inverse variance.** When neither `estimate_background` nor
  `estimate_ivar` is set (the default), the `wht` array is used directly as
  the per-pixel inverse variance. Set `estimate_ivar=True` to recalibrate it
  from the image itself.
- **Detection runs on a noise-equalised image**, `(sci - background) *
  sqrt(ivar)`, smoothed with a Gaussian kernel. The `detect_threshold`
  parameter is applied directly to this smoothed image. Because smoothing
  suppresses pixel-to-pixel noise, the effective significance per pixel is
  higher than the nominal value; the threshold is not rescaled by the
  post-smoothing noise as of this writing. `sci` here is the attribute, which
  `run()` has already background-subtracted when `estimate_background=True`,
  so as of this writing the fitted background is subtracted a second time
  before detection.
- **Segment labels are catalog ids.** After `run()`, `table["id"]` equals the
  integer labels in `segmap`, which is the contract the pipeline relies on.

## Building a catalog

```python
from mophongo.catalog import Catalog

cat = Catalog.from_fits(
    "image.fits",
    "weight.fits",
    estimate_background=True,
    estimate_ivar=True,
    params={"detect_threshold": 1.5, "kernel_size": 3.0},
)

cat.table       # astropy Table with id, x, y, ra, dec, fluxes, shapes
cat.segmap      # photutils SegmentationImage, labels match table["id"]
```

To reuse an external segmentation map instead of detecting, pass it in;
detection is skipped and only measurement runs:

```python
cat = Catalog.from_fits("image.fits", "weight.fits", segmap="segmap.fits")
```

The resulting `cat.table` (ids, `x`, `y`, `ra`, `dec`) and `cat.segmap.data`
are what {class}`mophongo.pipeline.Pipeline` consumes as its `catalog` and
`segmap` arguments.

## The Catalog class

{class}`mophongo.catalog.Catalog` is a `@dataclass`. Constructor fields:

`sci` : `np.ndarray`
: Science image (2D). When `estimate_background=True`, `run()` rebinds this
  attribute to the background-subtracted image; the caller's array is not
  modified.

`wht` : `np.ndarray`
: Weight map. Interpreted as inverse variance unless `estimate_ivar=True`.

`nbin` : `int`, default `4`
: Binning factor used by the {meth}`~mophongo.catalog.Catalog.plot_bg`
  diagnostic display.

`estimate_background` : `bool`, default `False`
: Fit and subtract a smooth background via
  {func}`mophongo.catalog.get_bg_and_ivar` before detection.

`estimate_ivar` : `bool`, default `False`
: Recalibrate the inverse variance from the measured background-pixel
  scatter (also via {func}`~mophongo.catalog.get_bg_and_ivar`) instead of
  trusting `wht`.

`background` : `float` or `np.ndarray`, default `0.0`
: Background level or map. Filled by `run()` when
  `estimate_background=True`.

`ivar` : `np.ndarray | None`, default `None`
: Inverse-variance map. `run()` sets it to `wht` when no estimation is
  requested, or to the recalibrated map when `estimate_ivar=True`. As of
  this writing, `estimate_background=True` on its own leaves `ivar` unset
  and `run()` fails; enable `estimate_ivar` as well (or pass `ivar`
  yourself).

`segmap` : `SegmentationImage | None`, default `None`
: Segmentation map. If provided, detection is skipped; otherwise filled by
  `run()`.

`parent_segmap` : `SegmentationImage | None`, default `None`
: Copy of the segmentation map before deblending; used to record deblend
  provenance.

`catalog` : `photutils.segmentation.SourceCatalog | None`, default `None`
: The underlying photutils catalog object, filled by `run()`.

`table` : `astropy.table.Table | None`, default `None`
: The measurement table, filled by `run()`.

`det_img` : `np.ndarray | None`, default `None`
: The noise-equalised detection image, filled during detection.

`params` : `dict[str, float | int]`, default `{}`
: Detection and deblending parameters; user-supplied entries are merged
  over the defaults listed below.

`header` : `fits.Header | None`, default `None`
: FITS header; used to construct a WCS when one is not given. Filled
  automatically by `from_fits` when `sci` is a filename.

`wcs` : `astropy.wcs.WCS | None`, default `None`
: World coordinate system for sky positions. Built from `header` if absent.

`default_columns` : `list[str]`
: photutils `SourceCatalog` columns exported to `table`. The default list is
  `label`, `xcentroid`, `ycentroid`, `sky_centroid`, `area`,
  `semimajor_sigma`, `semiminor_sigma`, `kron_radius`, `eccentricity`,
  `orientation`, `min_value`, `max_value`, `local_background`,
  `segment_flux`, `segment_fluxerr`, `kron_flux`, `kron_fluxerr`.

### Detection parameters (`params`)

`kernel_size` : `float`, default `3.5`
: FWHM in pixels of the Gaussian smoothing kernel applied to the detection
  image (the kernel sigma is `kernel_size / 2.355`; the stamp is
  `int(2 * kernel_size) | 1` pixels on a side).

`detect_npixels` : `int`, default `5`
: Minimum number of connected pixels for a detection; also passed to the
  deblender.

`detect_threshold` : `float`, default `2.0`
: Threshold applied to the smoothed, noise-equalised detection image (see
  Conventions above).

`dilate_segmap` : `int`, default `2`
: Radius of the disk used to dilate segments into background via
  {func}`mophongo.catalog.safe_dilate_segmentation`. `0` disables dilation.

`deblend_mode` : `str | None`, default `"exponential"`
: Mode passed to `photutils.segmentation.deblend_sources`. `None` skips
  deblending entirely.

`deblend_nlevels` : `int`, default `32`
: Number of multi-thresholding levels for the deblender.

`deblend_contrast` : `float`, default `1e-4`
: Minimum flux fraction for a deblended child.

`deblend_compactness` : `float`, default `0.0`
: Reserved; as of this writing it is not forwarded to the deblender.

`background_filter_sigma` : `float`, default `64.0`
: Forwarded to {func}`~mophongo.catalog.get_bg_and_ivar` as
  `bg_filter_sigma`; sets the coarse-grid bin factor and the background
  smoothing scale (see below).

### `Catalog.from_fits`

```python
Catalog.from_fits(sci, wht, *, segmap=None, header=None, **kwargs)
```

`sci` : `str | Path | np.ndarray`
: Science image or FITS filename. When a filename, the header is read and
  stored for WCS construction.

`wht` : `str | Path | np.ndarray`
: Weight map or FITS filename.

`segmap` : `str | Path | np.ndarray | SegmentationImage | None`, keyword-only, default `None`
: External segmentation map. When given, detection and deblending are
  skipped and sources are measured within the provided segments.

`header` : `fits.Header | None`, keyword-only, default `None`
: Header to use when `sci` is an array.

`**kwargs`
: Forwarded to the `Catalog` constructor (e.g. `estimate_background`,
  `params`).

`from_fits` constructs the object and immediately calls
{meth}`~mophongo.catalog.Catalog.run`, so the returned instance has `table`
and `segmap` populated.

### `Catalog.run`

Takes no arguments. Steps, in order: optional background/inverse-variance
estimation; detection, dilation, and deblending (only if `segmap` is not
already set); WCS construction from `header`; measurement with
`SourceCatalog(sci, segmap, error=np.sqrt(1.0 / ivar), wcs=wcs)`; and
assembly of `table`.

Columns in the resulting `table`, beyond `default_columns`:

- `id`, `x`, `y` — renamed from `label`, `xcentroid`, `ycentroid`.
- `ra`, `dec` — degrees, extracted from `sky_centroid` when a WCS is
  available (the `sky_centroid` column itself is removed).
- `r50` — radius enclosing half the segment flux (`fluxfrac_radius(0.5)`).
- `sharpness` — `max_value * pi * r50**2 / segment_flux`; near unity for
  point sources.
- `snr` — `segment_flux / segment_fluxerr`.
- `deblend_parent_label`, `deblend_nchildren`, `is_deblended` — deblend
  provenance: the pre-deblend parent label, how many children that parent
  split into, and whether it split at all.

### `Catalog.find_stars`

```python
table, idx_stars = cat.find_stars(psf=None, snr_min=100, r50_max=5,
                                  eccen_max=0.2, sharp_lohi=(0.2, 1.2),
                                  chi2_max=3.0)
```

Selects point-like sources from `cat.table` and optionally fits a PSF stamp
to each. Adds a boolean `point_like` column to `cat.table` as a side effect.
All arguments are keyword-only.

`psf` : `np.ndarray | None`, default `None`
: PSF stamp. When given, each candidate is fit with
  {func}`mophongo.catalog.fit_psf_stamp` and the returned table gains
  `flux_psf` and `chi2_red` columns.

`snr_min` : `float`, default `100`
: Minimum `snr` for a candidate.

`r50_max` : `float`, default `5`
: Maximum half-light radius in pixels.

`eccen_max` : `float`, default `0.2`
: Maximum eccentricity.

`sharp_lohi` : `tuple[float, float]`, default `(0.2, 1.2)`
: Accepted range of the `sharpness` statistic.

`chi2_max` : `float`, default `3.0`
: As of this writing this cut is not applied inside the method; filter on
  the returned `chi2_red` column yourself.

`return_seg` : `bool`, default `False`
: As of this writing this flag has no effect; the method always returns
  `(table, idx_stars)` where `idx_stars` are the row indices of the
  selected sources in `cat.table`.

#### Feeding the star selection into the fit

The pipeline does not read `point_like`. Star handling in
{class}`mophongo.pipeline.Pipeline` keys off an integer `flag_star` column
in the catalog table passed to it: templates whose catalog row has
`flag_star == 1` are marked `is_star` (see {doc}`templates`), which the
`astrom_exclude_stars` option in {doc}`fitting` uses to drop stars from the
astrometric shift fit. To make the `find_stars` selection take effect,
convert the column before running the pipeline:

```python
cat.table["flag_star"] = cat.table["point_like"].astype(int)
```

This flags every point-like source. To apply the additional `snr_min` cut,
flag only the rows `find_stars` returned:

```python
cat.table["flag_star"] = 0
cat.table["flag_star"][idx_stars] = 1
```

Without a `flag_star` column no templates are marked as stars. Marking them
is necessary but not sufficient: `astrom_exclude_stars` is `False` by
default, so the flag changes the astrometry only once that option is on.

### Display diagnostics

{meth}`~mophongo.catalog.Catalog.show_stamp` displays a cutout of one source
by catalog `id` with its segmentation footprint overlaid; pass an existing
axis, since as of this writing a fresh figure is created but left empty when
`ax=None`. {meth}`~mophongo.catalog.Catalog.plot_bg` is a four-panel
diagnostic on a binned grid — image, segmentation overlay, fitted background,
and background-subtracted noise-equalised image; pass `nbin` explicitly, as
of this writing the documented `None` default raises `TypeError`.

## Background and noise estimation

{func}`mophongo.catalog.get_bg_and_ivar` implements the estimator used by
`Catalog.run`:

```python
bg_img, ivar_new = get_bg_and_ivar(sci, wht, bg_filter_sigma=64.0,
                                   detect_thresh=1.0, dilate=3)
```

`sci`, `wht` : `np.ndarray`
: Science image and weight map.

`bg_filter_sigma` : `float`, keyword-only, default `64.0`
: Sets the coarse-grid bin factor (`floor(sqrt(bg_filter_sigma))`) and the
  scale of the mask-aware Gaussian background smoothing.

`detect_thresh` : `float`, keyword-only, default `1.0`
: Detection threshold, in units of the robust sigma of the coarse
  noise-equalised image, for masking sources out of the background fit.

`dilate` : `int`, keyword-only, default `3`
: Disk radius for smoothing the coarse detection image and dilating the
  background mask.

The function bins the image, masks sources with a two-pass detection
(bright, smoothed + faint, per-pixel), fits a mask-aware smoothed background
on the coarse grid, measures the robust scatter of background pixels after
subtraction, and returns the background interpolated back to full resolution
together with the weight map rescaled so it is a calibrated inverse
variance. A related function, `calibrate_ivar_with_bg_median`, appears in
the module but is not called by `Catalog` and, as of this writing, raises a
`NameError` when invoked; use `get_bg_and_ivar`.

## Segmentation-map helpers

{func}`mophongo.catalog.safe_dilate_segmentation` grows each segment into
background pixels only, so neighbouring segments never overwrite each other;
it returns the dilated label array without modifying the input.
{func}`mophongo.catalog.fit_psf_stamp` fits a PSF plus a constant to a small
stamp by weighted least squares and returns `(flux, chi2_red)`; `find_stars`
uses it for its per-candidate PSF fits.

## Saturated stars

{func}`mophongo.catalog.find_saturated_stars` locates bright stars whose
cores have zero weight (the usual signature of saturation) by detecting on a
binned noise-equalised image and testing the minimum weight around each
centroid; it returns a Table of binned and full-resolution centroids with a
`sat_flag` column.

Two further helpers consume the per-star tables produced by the
saturated-pixel repair step described in {doc}`preprocessing`:

{func}`mophongo.catalog.merge_segments_at_holes` is a read-only inspection:
for each repaired hole it lists the segmentation labels within a search
radius and returns `{hole_id: [label, ...]}` without modifying the segmap.

{func}`mophongo.catalog.repair_saturated_catalog` merges the oversplit
segments of a repaired saturated star into a single parent segment and
catalog row, optionally keeping unrelated neighbours by requiring a minimum
PSF-model flux fraction in each candidate child. It returns the merged
catalog (with a `FLAG_SATURATED_<FILTER>` column, which the pipeline reads —
see below), the relabelled segmap with the saturated core filled, and a merge
log.

## Feeding the pipeline

The pipeline contract is: `segmap` labels equal catalog `id` values, and the
catalog provides `id`, `x`, `y` on the high-resolution pixel grid. Those three
columns are all the fit needs; the output catalog carries them forward together
with the deblend and `FLAG_SATURATED_*` provenance columns when present, and
drops everything else, `ra` and `dec` included. `Catalog.run` produces this, so
a typical hand-off is

```python
cat = Catalog.from_fits("image.fits", "weight.fits")
# ... build PSFs and kernels, see the psf page ...
from mophongo.pipeline import Pipeline
pipe = Pipeline(images, cat.segmap.data, catalog=cat.table)
```

### Columns for a user-supplied catalog

The catalog does not have to come from `Catalog`: any table with the right
columns works, for example one converted from a SExtractor run, together with
its matching segmentation map. Required columns:

`id` : `int`
: Must equal the segmentation label at the source position. The pipeline
  extracts each template at `(x, y)`, takes the segment label found there as
  the template id, and matches fitted fluxes back to catalog rows through
  `id` — a mismatch silently attaches fluxes to the wrong rows.

`x`, `y` : `float`
: Source position in **0-indexed** pixel coordinates on the high-resolution
  detection grid, pixel-center convention (integer value = center of that
  pixel). SExtractor's `X_IMAGE`/`Y_IMAGE` follow the 1-based FITS
  convention, so subtract 1 from each. Rows whose position is non-finite,
  outside the image, or lands on segmentation background are skipped
  silently: those sources get no template and their flux columns keep the
  bad value.

Optional columns the pipeline consumes when present:

`flag_star` : `int`
: Rows with `flag_star == 1` mark their templates `is_star` (see the star
  selection above).

`is_deblended`, `deblend_parent_label`, `deblend_nchildren`
: Deblend provenance, copied onto the templates and carried into the output
  catalog; the parent/children columns are read only when `is_deblended` is
  present.

`FLAG_SATURATED_*`
: Any column whose name starts with `FLAG_SATURATED_`; a nonzero value marks
  the template saturated, which isolates it into its own scene.

`ra`, `dec` : `float`, degrees
: Not read by the fit itself. The run-config driver needs them only when the
  `r_trial` patch cut is enabled.

*aperture column*
: When `FitConfig.aperture_catalog` names a column, that column must exist
  and hold per-source aperture **diameters**, in the units set by
  `FitConfig.aperture_units` (arcsec or pixels).

All other columns are dropped from the fit's output catalog.

See {doc}`pipeline` for the full argument list, {doc}`templates` for how
segments become fit templates, and {doc}`outputs` for the flux columns the
fit adds to the catalog.
