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

{class}`mophongo.catalog.Catalog` is a `@dataclass` holding the science
image, the weight map, and the products {meth}`~mophongo.catalog.Catalog.run`
fills in: the segmentation map, the underlying photutils `SourceCatalog`,
and the measurement `table`. Detection and deblending are configured through
the `params` dict, whose user-supplied entries are merged over the defaults;
the two most commonly tuned are `detect_threshold` (default 2.0, applied to
the smoothed noise-equalised detection image — see Conventions above) and
`kernel_size` (default 3.5, the FWHM in pixels of the Gaussian detection
smoothing). Setting `deblend_mode=None` turns deblending off entirely. Every
constructor field and `params` key, with defaults and behavior notes, is
documented on the {class}`~mophongo.catalog.Catalog` API page.

The usual entry point is {meth}`~mophongo.catalog.Catalog.from_fits`, which
accepts arrays or FITS filenames (reading the header for WCS construction),
builds the object, and immediately calls
{meth}`~mophongo.catalog.Catalog.run`, so the returned instance has `table`
and `segmap` populated — as in the examples above. `run()` estimates the
background and inverse variance when requested, detects, dilates, and
deblends (all skipped when a `segmap` was supplied), and measures sources
with `SourceCatalog(sci, segmap, error=np.sqrt(1.0 / ivar), wcs=wcs)`.

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

The whole flow fits in a few lines on a synthetic two-source field; arrays
work in place of filenames, and the weight map doubles as the inverse
variance:

```python
import numpy as np
from mophongo.catalog import Catalog
from mophongo.psf import PSF

rng = np.random.default_rng(11)
sci = rng.normal(0, 1e-3, (101, 101))
sci += 50 * PSF.gaussian(101, 3.0).array                        # source at (50, 50)
sci += 30 * np.roll(PSF.gaussian(101, 3.0).array, 12, axis=1)   # neighbor at (62, 50)
wht = np.full_like(sci, 1e6)                                    # ivar for sigma = 1e-3
cat = Catalog.from_fits(sci, wht)
print(len(cat.table))                       # 2
print(cat.table["id", "x", "y", "snr"])     # one row per detected source
```

### Star selection

{meth}`~mophongo.catalog.Catalog.find_stars` selects point-like sources from
`cat.table` by keyword-only cuts on half-light radius, eccentricity, and the
`sharpness` statistic, and returns `(table, idx_stars)`: the selected rows —
those also passing the `snr_min` cut, default 100 — and their row indices in
`cat.table`. A boolean `point_like` column recording the shape cuts (without
the SNR cut) is added to `cat.table` as a side effect. When a PSF stamp is
passed, each candidate is fit with {func}`mophongo.catalog.fit_psf_stamp`
and the returned table gains `flux_psf` and `chi2_red` columns; as of this
writing the `chi2_max` argument is not applied inside the method, so filter
on `chi2_red` yourself.

```python
table, idx_stars = cat.find_stars(psf=psf_stamp, snr_min=50)
```

On a field with a single bright point source the shape cuts keep exactly that
source, with its `sharpness` in the point-like range:

```python
import numpy as np
from mophongo.catalog import Catalog
from mophongo.psf import PSF

rng = np.random.default_rng(11)
sci = rng.normal(0, 1e-3, (101, 101))
sci += 200 * PSF.gaussian(101, 3.0).array   # one bright point source
wht = np.full_like(sci, 1e6)
cat = Catalog.from_fits(sci, wht)
stars, idx_stars = cat.find_stars(snr_min=50)
print(len(stars), round(float(stars["sharpness"][0]), 2))  # 1 0.7
```

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

{func}`mophongo.catalog.get_bg_and_ivar`, the estimator behind
`estimate_background` and `estimate_ivar`, returns `(bg_img, ivar_new)`. It
bins the image (bin factor `floor(sqrt(bg_filter_sigma))`), masks sources
with a two-pass detection (bright, smoothed + faint, per-pixel), fits a
mask-aware smoothed background on the coarse grid, measures the robust
scatter of background pixels after subtraction, and returns the background
interpolated back to full resolution together with the weight map rescaled
so it is a calibrated inverse variance. A related function,
`calibrate_ivar_with_bg_median`, appears in the module but is not called by
`Catalog` and, as of this writing, raises a `NameError` when invoked; use
`get_bg_and_ivar`.

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
: When `FitConfig.phot.aperture_catalog` names a column, that column must exist
  and hold per-source aperture **diameters**, in the units set by
  `FitConfig.phot.units` (arcsec or pixels).

All other columns are dropped from the fit's output catalog.

See {doc}`pipeline` for the full argument list, {doc}`templates` for how
segments become fit templates, and {doc}`outputs` for the flux columns the
fit adds to the catalog.
