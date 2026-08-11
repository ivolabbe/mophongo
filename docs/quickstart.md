# Quickstart

Mophongo measures fluxes in low-resolution images by fitting per-source
templates extracted from a high-resolution detection image, after convolving
each template with a PSF-matching kernel. This page shows the two ways to run
it: driving a full run from a JSON config with
{meth}`mophongo.pipeline.Pipeline.from_config` — the standard path for real
mosaics — and passing in-memory arrays directly to
{func}`mophongo.pipeline.run`, which suits simulations and experiments. See
{doc}`overview` for the method itself.

## Installation

Install the dependencies with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

or install in editable mode with pip:

```bash
pip install -e .
```

## Choosing an entry point

Both paths share the fitting engine; they differ in what you supply and what
gets built for you.

| | Arrays: {func}`mophongo.pipeline.run` | Config: {meth}`mophongo.pipeline.Pipeline.from_config` |
|---|---|---|
| Inputs | in-memory numpy arrays + `astropy` Table | FITS/CSV paths in a JSON `RunConfig` |
| PSFs | one static stamp per band, supplied by you | drizzled, position-dependent ePSFs built from the per-frame WCS CSVs |
| Kernels | you build them ({func}`mophongo.utils.matching_kernel`), or pass a prebuilt {class}`mophongo.psf_map.PSFRegionMap` | per-region kernel map, built and geojson-cached automatically |
| Template extension | off by default (`extend_templates=None`, templates stay truncated) | `"psf_wings"` by default |
| Preprocessing | none — you supply inverse-variance weights | background/ivar estimation, footprint and trial-patch cuts |
| Products | in-memory `(table, residuals, pipe)` only | residual FITS, fit table, per-source stamps, scene diagnostics, run log; restore later with `load_fit()` |
| Suits | simulations, mocks, method experiments | real mosaics |

## Config-driven runs

For runs on real mosaics — with drizzled, spatially varying PSFs, cached
kernels, and file outputs — describe the run in a JSON file and let the
pipeline handle PSF generation and bookkeeping. A typical session runs the
steps one at a time:

```python
from mophongo.pipeline import Pipeline

pipe = Pipeline.from_config("f770w.json")

pipe.build_psfs()      # per-band PSF region maps (geojson-cached in out_dir)
pipe.build_kernels()   # matching-kernel map (geojson-cached)
pipe.run()             # load data (footprint/trial cuts, bg/ivar) and fit
pipe.write_outputs()   # residual FITS, fit table, stamps, scene diagnostics

pipe.table                 # fitted catalog
pipe.show_sources([1, 2])  # image/model/residual figure per source
```

`pipe.run_all()` performs the same four steps in order and logs everything
to `<out_dir>/<name>.log`. The command line is equivalent:

```bash
python -m mophongo.pipeline f770w.json          # all steps
python -m mophongo.pipeline f770w.json psfs fit # selected steps
```

Valid step names are `psfs`, `kernels`, `load`, `loadfit`, `info`, `fit`,
`outputs`, and `all`. {meth}`mophongo.pipeline.Pipeline.from_config` is
lazy: images are read when `run()` (or `load_data()`) is first called.
{meth}`mophongo.pipeline.Pipeline.write_outputs` writes
`<out_dir>/<name>_residual.fits` (on the detection grid),
`<name>_fit_table.fits`, per-source stamps, and scene diagnostics.

A minimal config:

```json
{
  "name": "myrun",
  "out_dir": "output",
  "sci_hi": "image_hi.fits",
  "segmap": "segmap.fits",
  "catalog": "catalog.fits",
  "sci_lo": "image_lo.fits",
  "wht_lo": "wht_lo.fits",
  "csv_hi": "frames_hi.csv",
  "csv_lo": "frames_lo.csv",
  "pattern_hi": "STDPSF_NRCA._F444W.*fits",
  "pattern_lo": "STDPSF_MIRI_F770W.*fits",
  "filter_lo": "f770w"
}
```

Lines starting with `#` are treated as comments and stripped, so configs can
be annotated. Unknown keys raise an error, so typos fail loudly. A realistic
config for a JWST field — 40 mas F444W detection mosaic, 80 mas MIRI F770W
band, MJD-tagged ePSF grids, a trial patch for testing before the full-field
run — looks like:

```text
{
  "name": "f770w",
  "out_dir": "runs/f770w",

  # high-resolution template side
  "sci_hi": "data/mosaic-40mas-f444w_drc_sci.fits",
  "segmap": "data/segmap_f444w.fits",
  "catalog": "data/catalog.fits",
  "csv_hi": "data/mosaic-40mas-f444w_wcs.csv",

  # low-resolution fit side
  "sci_lo": "data/mosaic-80mas-f770w_drz_sci.fits",
  "wht_lo": "data/mosaic-80mas-f770w_drz_wht.fits",
  "csv_lo": "data/mosaic-80mas-f770w_wcs.csv",

  # MJD-tagged ePSF grids; psf_size in arcsec
  "psf_dir": "data/PSF",
  "pattern_hi": "NRC.._F444W_MJD\\d+_GRID25_OS4",
  "pattern_lo": "MIRI_F770W_MJD\\d+_GRID9_OS4",
  "filter_lo": "f770w",
  "psf_size": 4.0,
  "psf_blur_fwhm": "default",

  # preprocessing: footprint cut + trial patch (r_trial 0 = full mosaic)
  "footprint_filter": true,
  "r_trial": 0.6,
  "trial_center": [34.35, -5.27],
  "bg_filter_sigma": 64.0,

  # FitConfig overrides
  "fit": {"fit_astrometry_joint": true, "aperture_diam": 0.5}
}
```

Start with a small `r_trial` patch to validate PSFs, kernels, and residuals,
then set `r_trial` to `0` for the full field: the cached PSF and kernel maps
in `out_dir` are reused.

### `RunConfig` fields

{class}`mophongo.pipeline.RunConfig` describes one filter fit (one
high-resolution plus one low-resolution band). Fields without a default are
required.

`name` (`str`)
: Run label; prefixes every output file.

`out_dir` (`str`)
: Output directory for products and PSF/kernel caches (never inputs).

`sci_hi` (`str`)
: High-resolution template image (FITS).

`segmap` (`str`)
: Segmentation map on the high-resolution grid; labels equal catalog ids.

`catalog` (`str`)
: Source catalog with `id`, `x`, `y` (high-resolution pixels), `ra`, `dec`.

`sci_lo` (`str`)
: Low-resolution science mosaic to fit.

`wht_lo` (`str`)
: Low-resolution weight map (inverse variance).

`csv_hi`, `csv_lo` (`str`)
: Per-frame WCS CSV files of the high- and low-resolution mosaics, used to
  drizzle position-dependent PSFs. The "Per-frame WCS CSVs" section of
  {doc}`pipeline` describes what they contain and how to generate them with
  {func}`mophongo.utils.reconstruct_wcs`.

`driz_hi` (`str | None`, default `None`)
: Mosaic providing the DrizzlePSF footprints/grid of the high-resolution
  side; defaults to `sci_hi`. Set when `sci_hi` is a derived template
  image.

`psf_dir` (`str`, default `"data/PSF"`)
: Directory holding STDPSF grid files.

`pattern_hi`, `pattern_lo` (`str`, default `""`)
: STDPSF filename regexes selecting the PSF grids for each band.

`filter_lo` (`str`, default `""`)
: Low-resolution filter name (e.g. `"f770w"`), used for the blur lookup.

`psf_size` (`float | None`, default `4.0`)
: PSF stamp size in arcsec; `None` keeps the full native ePSF stamp.

`psf_autobuild` (`bool`, default `True`)
: Generate missing PSF grids with
  {class}`mophongo.psf_factory.PSFFactory` (see {doc}`psf`).

`psf_fov_arcsec` (`float | None`, default `None`)
: PSFFactory field of view; `None` uses the backend default.

`psf_blur_fwhm` (`float | str | None`, default `"default"`)
: Extra Gaussian broadening of the low-resolution model PSF (FWHM,
  arcsec). `"default"` looks up a per-filter value from
  `mophongo.mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC`; a number uses
  that value; `None` applies no broadening.

`expect_frames` (`list[int] | None`, default `None`)
: Optional `[n_frames_hi, n_frames_lo]` sanity assertion on the WCS CSVs.

`extend_templates` (`str | None`, default `"psf_wings"`)
: Template extension mode, as in the array interface below.

`bg_filter_sigma` (`float`, default `64.0`)
: Background filter scale for the background/inverse-variance
  preprocessing step (see {doc}`preprocessing`).

`footprint_filter` (`bool`, default `True`)
: Keep only sources where the low-resolution weight is positive.

`r_trial` (`float`, default `0.0`)
: Trial-patch radius in arcmin; `0` fits the full mosaic.

`trial_center` (`list[float] | None`, default `None`)
: `[ra, dec]` in degrees of the trial patch center.

`fit` (`dict`, default `{}`)
: Keyword arguments forwarded to {class}`mophongo.fit.FitConfig`
  (see {doc}`fitting`).

`scene_plots` (`bool`, default `True`)
: Write per-scene diagnostic PNGs during `write_outputs()`.

`save_stamps` (`bool`, default `True`)
: Write the per-source stamps FITS file (native-size high/low templates
  plus each source's PSF region key; the PSF stamps themselves are not
  duplicated here, they stay in the cached `<name>_psf_*.geojson` maps),
  which later allows restoring a finished run with `Pipeline.load_fit()`.

The full description of the config-driven flow, the step methods, and the
caching behavior is in {doc}`pipeline`.

## An array-level run

The array interface skips configs, file caching, and drizzled PSFs: pass
images and PSF stamps directly. It needs, at minimum:

- a list of science images, with the high-resolution detection image first
  and the band(s) to fit after it;
- a segmentation map on the detection-image grid whose integer labels match
  the catalog `id` column;
- a source catalog (`astropy.table.Table`) with columns `id`, `x`, `y`
  (positions in detection-image pixels);
- per-image weight maps in inverse-variance units;
- one PSF-matching kernel per fitted band (see below).

```python
from astropy.io import fits
from astropy.table import Table

from mophongo import pipeline
from mophongo.utils import matching_kernel

# detection image first, then the band to fit
sci_hi = fits.getdata("image_hi.fits")
sci_lo = fits.getdata("image_lo.fits")
wht_hi = fits.getdata("wht_hi.fits")   # inverse variance
wht_lo = fits.getdata("wht_lo.fits")

segmap = fits.getdata("segmap.fits")   # labels match catalog "id"
catalog = Table.read("catalog.fits")   # columns: id, x, y

psf_hi = fits.getdata("psf_hi.fits")
psf_lo = fits.getdata("psf_lo.fits")

# fit with unit-sum PSF shapes; keep the finite stamp sum as throughput
throughput_lo = float(psf_lo.sum())
psf_hi_shape = psf_hi / psf_hi.sum()
psf_lo_shape = psf_lo / psf_lo.sum()
kernel = matching_kernel(psf_hi_shape, psf_lo_shape)

table, residuals, pipe = pipeline.run(
    [sci_hi, sci_lo],
    segmap,
    catalog=catalog,
    weights=[wht_hi, wht_lo],
    kernels=[None, kernel],
    psfs=[psf_hi_shape, psf_lo_shape],
    psf_throughputs=[1.0, throughput_lo],
)
```

`table` is a new catalog carrying `id`, `x`, `y`, and any
deblending/saturation provenance columns from the input catalog (other input
columns are dropped), with flux columns added; `residuals` is a
list of residual images (one per fitted band, so `residuals[0]` corresponds
to `images[1]`), and `pipe` is the {class}`mophongo.pipeline.Pipeline`
instance holding the templates, scenes, and model images for later
inspection (for example `pipe.show_sources(ids)`; see {doc}`diagnostics`).

Two conventions to keep in mind:

- **Weights are inverse variance.** `1 / sqrt(wht)` is the per-pixel RMS.
- **PSF shape vs throughput.** Fitting uses unit-sum PSF shapes; the finite
  stamp sum of a realistic PSF is filter-level throughput metadata. The raw
  fitted amplitude appears as `flux_<i>` and the throughput-corrected total
  as `flux_<i>_total`. See {doc}`psf` for the full convention.

If a fitted band has a coarser pixel scale than the detection image, pass
per-image WCS objects via `wcs`: the pipeline derives an integer bin factor
from the WCS pair, block-replicates the low-resolution science pixels with
flux conservation onto the reference grid, and copies the inverse variance
to the subpixels multiplied by `factor**2` so the native chi-square is
preserved.

### `pipeline.run()` parameters

{func}`mophongo.pipeline.run` is a thin wrapper that constructs a
{class}`mophongo.pipeline.Pipeline` (same parameters) and calls
{meth}`mophongo.pipeline.Pipeline.run`. All parameters after the first two
are keyword-only.

`images` (`Sequence[np.ndarray]`, required)
: Science images. `images[0]` is the high-resolution detection image from
  which templates are extracted; `images[1:]` are fitted. To also fit the
  detection band, include it twice.

`segmap` (`np.ndarray`, required)
: Integer segmentation map on the `images[0]` grid. Labels correspond to
  catalog `id` values; 0 is background.

`catalog` (`astropy.table.Table | None`, default `None`)
: Source catalog with columns `id`, `x`, `y`. Optional deblending columns
  (`is_deblended`, `deblend_parent_label`, `deblend_nchildren`) are carried
  through when present. As of this writing, passing `None` raises
  `NotImplementedError` — catalog generation from the segmentation map is
  not implemented, so build the catalog first (see {doc}`catalog`).

`psfs` (`Sequence[np.ndarray] | None`, default `None`)
: One PSF stamp per image. `psfs[0]` supplies the shape used for template
  extension; the others provide per-band PSF metadata and, when
  `psf_throughputs` is not given, the fallback throughput from the stamp
  sum. Pass unit-sum shapes for fitting.

`weights` (`Sequence[np.ndarray] | None`, default `None`)
: Inverse-variance maps, one per image.

`wht_images` (`Sequence[np.ndarray] | None`, default `None`)
: Backward-compatible alias for `weights`; used only when `weights` is
  `None`.

`kernels` (`Sequence[np.ndarray | PSFRegionMap] | None`, default `None`)
: One convolution kernel per image, mapping the detection-band PSF to that
  band's PSF. Use `None` for entries needing no convolution (the detection
  image itself). A {class}`mophongo.psf_map.PSFRegionMap` entry applies a
  spatially varying kernel (see {doc}`psf_maps`).

`psf_throughputs` (`Sequence[float] | None`, default `None`)
: Finite-support PSF stamp sums, one per image. When given, these override
  the sums of the `psfs` stamps as the filter-level throughput used for the
  `flux_<i>_total` columns. Use `1.0` for a band fitted with a native-sum
  PSF already summing to one.

`wcs` (`Sequence[astropy.wcs.WCS] | None`, default `None`)
: Per-image WCS. Enables the multi-resolution upsampling path (integer bin
  factors derived from the WCS pair), RA/Dec in diagnostics, and aperture
  radii specified in arcseconds.

`window` (default `None`)
: Accepted and stored on the pipeline for backward compatibility; as of
  this writing it is not consumed by the fitting path.

`extend_templates` (`str | None`, default `None`)
: How to fill each template outside its segment: `"psf_wings"` adds the
  detection-band PSF wings beyond the segmentation footprint, `"psf_model"`
  replaces the template with the PSF, `None` leaves it truncated.
  Truncated templates bias total fluxes low, badly so for faint sources;
  config-driven runs default to `"psf_wings"`.

`templates` (`Templates | Sequence[Template] | None`, default `None`)
: Prebuilt {class}`mophongo.templates.Templates` to use instead of
  extracting from `images[0]` and `segmap` (see {doc}`templates`).

`config` ({class}`mophongo.fit.FitConfig` `| None`, default `None`)
: Fitting configuration (regularization, astrometric iterations, scene
  construction, aperture photometry). Defaults to `FitConfig()`; every
  field is documented in {doc}`fitting`.

### Output columns

For each fitted image `i` (counting from 1, in input order) the returned
table gains `flux_i`, `err_i`, `err_pred_i` (raw fitted amplitude, solver
error, and weight-map-predicted error), `throughput_i`, and the
throughput-corrected totals `flux_i_total`, `err_i_total`,
`err_pred_i_total`. When per-source encircled energies of the
low-resolution PSF are available they are used in place of the filter-level
throughput. Aperture and diagnostic columns are described in
{doc}`outputs`.

### `matching_kernel()` parameters

{func}`mophongo.utils.matching_kernel` computes a kernel `k` such that
`psf_hi * k ≈ psf_lo` under convolution. It preserves the input sums: if
`sum(psf_lo) / sum(psf_hi)` is not one, that ratio propagates into
`sum(k)`, which is why pipeline-facing calls should pass unit-sum shapes.
PSFs of different shapes are zero-padded to a common grid.

`psf_hi_in`, `psf_lo_in` (`np.ndarray`, required)
: High- and low-resolution PSF arrays.

`window` (default `None`)
: Fourier-domain window for `method="window"`; defaults to
  `photutils.psf.matching.SplitCosineBellWindow(alpha=0.4, beta=0.1)`.

`recenter` (`bool`, default `False`)
: Shift the kernel to its measured centroid with bicubic interpolation.

`pixel_ratio` (`float`, default `1.0`)
: Pixel-scale ratio between the two PSF grids; a non-unity value resamples
  one PSF onto the other's grid with flux-conserving cubic interpolation
  before the kernel is computed.

`method` (`str`, default `"window"`)
: Kernel algorithm: `"window"` (Fourier ratio with the window function),
  `"tikhonov"`, `"wiener"`, or `"forward"` (ForWaRD: a regularized Fourier
  inverse followed by wavelet denoising).

`reg` (`float`, default `1e-3`)
: Regularization strength for the `tikhonov`, `wiener`, and `forward`
  methods.

`wavelet` (`str`, default `"db4"`), `levels` (`int`, default `3`),
`threshold_factor` (`float`, default `3.0`), `noise_sigma`
(`float | None`, default `None`), `forward_wavelet_wiener` (`bool`,
default `True`)
: Wavelet-denoising controls for `method="forward"`.

`signal_psd` (`np.ndarray | None`, default `None`)
: Optional signal power spectrum for `method="wiener"`.

See {doc}`psf` for kernel diagnostics and regularization scans.

## Where to go next

- {doc}`pipeline` — the `Pipeline` class, step methods, and run restore.
- {doc}`templates` — template extraction, extension, and coordinate
  conventions.
- {doc}`psf` and {doc}`psf_maps` — PSF generation, matching kernels, and
  spatial PSF variation.
- {doc}`fitting` — `FitConfig`, scenes, astrometric shifts, and error
  estimates.
- {doc}`catalog` — building the detection catalog and segmentation map.
- {doc}`outputs` — every output column and file.
- {doc}`simulation` — synthetic mosaics and injected-truth verification.
- {doc}`api` — full API reference.
