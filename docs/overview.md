# Overview

Mophongo performs template-fitting photometry on multi-band imaging with
heterogeneous angular resolution. A high-resolution detection image defines
the positions and morphologies of all sources; each source becomes a template
that is PSF-matched to a lower-resolution band and fit, simultaneously with
its neighbors, to that band's pixels. The approach follows the
high-resolution-prior tradition of codes such as TFIT, T-PHOT, and the
original MOPHONGO IDL implementation, and is aimed at JWST and HST surveys
where a NIRCam or ACS/WFC3 mosaic serves as the prior for lower-resolution
bands (for example MIRI, or ground-based imaging).

## The method

The measurement proceeds in four stages:

1. **Detection and segmentation.** Sources are detected on a high-resolution
   image and assigned pixels in a segmentation map, either supplied by the
   user or built with {class}`mophongo.catalog.Catalog`
   (see {doc}`catalog`).
2. **Template extraction.** For every source, the pixels inside its segment
   are cut from the detection image to form a template. Because segmentation
   truncates the faint outer profile, templates are by default extended
   beyond the segment with the high-resolution PSF wings; without an
   extension the recovered total flux is biased low
   (see {doc}`templates`).
3. **PSF matching.** Each template is convolved with a convolution kernel
   that transforms the high-resolution PSF into the PSF of the measurement
   band. Kernels may be single arrays or spatially varying
   {class}`mophongo.psf_map.PSFRegionMap` lookups
   (see {doc}`psf` and {doc}`psf_maps`).
4. **Simultaneous fit.** All matched templates in a band are fit to the
   science pixels by sparse weighted linear least squares. Overlapping
   sources are solved together, so blended flux is apportioned by the
   templates rather than by aperture geometry. The fit is partitioned into
   independent "scenes" of coupled sources, and can solve small astrometric
   shifts per scene (see {doc}`fitting`). Outputs are per-source fluxes,
   uncertainties, and a full residual image per band (see {doc}`outputs`).

The {class}`mophongo.pipeline.Pipeline` class orchestrates these stages,
either from in-memory arrays or from a JSON run configuration
(see {doc}`pipeline`); {doc}`quickstart` walks through a minimal run.

## When to use it

Template fitting pays off when aperture or profile-fitting photometry on the
low-resolution image alone breaks down:

- **Blended and crowded fields.** Neighboring sources that merge at the
  low-resolution PSF are deblended using their high-resolution morphologies.
- **Mixed-resolution surveys.** Fluxes measured in every band refer to the
  same high-resolution prior, so colors are consistent across instruments
  and pixel scales.
- **Bands with poorly sampled or spatially varying PSFs.** PSF generation,
  drizzling to the mosaic frame, and per-region kernels are built in
  (see {doc}`psf`).

For isolated sources in a single well-sampled band, standard aperture
photometry with `photutils` is simpler and sufficient.

## Package map

- **Pipeline** — {doc}`pipeline` covers
  {class}`mophongo.pipeline.Pipeline`, the JSON `RunConfig`, and the
  command line entry point `python -m mophongo.pipeline config.json`.
  {doc}`outputs` documents the catalog columns and files a run writes;
  {doc}`diagnostics` covers the standard diagnostic figures.
- **Templates** — {doc}`templates` covers
  {class}`mophongo.templates.Template` and
  {class}`mophongo.templates.Templates`: extraction from the segmentation
  map, PSF-wing extension, convolution, and multi-resolution block
  projection.
- **PSFs** — {doc}`psf` covers {class}`mophongo.psf.PSF` and
  {class}`mophongo.psf.DrizzlePSF` (analytic, empirical, and drizzled JWST
  PSFs, matching-kernel construction and regularization) and the
  `stpsf`-backed `PSFFactory`; {doc}`psf_maps` covers
  {class}`mophongo.psf_map.PSFRegionMap`, which tiles a mosaic into regions
  of constant effective PSF.
- **Fitting** — {doc}`fitting` covers
  {class}`mophongo.fit.SparseFitter`, {class}`mophongo.fit.FitConfig`,
  {class}`mophongo.scene.Scene`, and
  {class}`mophongo.scene_fitter.SceneFitter`: the normal-equation assembly,
  scene partitioning, astrometric shift blocks, and error estimates.
- **Detection and catalogs** — {doc}`catalog` covers
  {class}`mophongo.catalog.Catalog`: source detection, segmentation,
  deblending, and catalog-owned segmentation helpers.
- **Preprocessing** — {doc}`preprocessing` covers saturated-pixel repair
  (`mophongo.saturate`) and spatially varying astrometric corrections
  (`mophongo.astrometry`). Background and inverse-variance estimation
  ({func}`mophongo.catalog.get_bg_and_ivar`) is documented on
  {doc}`catalog`.
- **Utilities** — `mophongo.utils` collects shared helpers; two are
  user-facing. {func}`mophongo.utils.lw_detection_coadd` builds an
  inverse-variance, PSF-matched long-wavelength coadd suitable as the
  high-resolution detection prior, and
  {class}`mophongo.utils.CircularApertureProfile` computes combined radial
  profiles and curves of growth (encircled energy) for PSF and source
  diagnostics. Both are listed on the {doc}`api` page.
- **Simulation and verification** — {doc}`simulation` covers
  `MockMosaic` synthetic JWST-like mosaics and the injected-truth
  verification helpers in `mophongo.verification`.
- **API reference** — {doc}`api` is the autosummary listing of all public
  modules.

## Minimal example

The array-level entry point is {func}`mophongo.pipeline.run`, a thin wrapper
that constructs a {class}`mophongo.pipeline.Pipeline` and calls
{meth}`mophongo.pipeline.Pipeline.run`:

```python
from astropy.io import fits
from astropy.table import Table

from mophongo import pipeline

images = [fits.getdata("detection.fits"), fits.getdata("image.fits")]
segmap = fits.getdata("segmap.fits")
catalog = Table.read("catalog.fits")
psfs = [fits.getdata("psf_hi.fits"), fits.getdata("psf_lo.fits")]
weights = [None, fits.getdata("weight.fits")]

table, residuals, pipe = pipeline.run(
    images, segmap, catalog=catalog, psfs=psfs, weights=weights
)
```

`images[0]` is the high-resolution detection image; every later entry is a
band to measure. The returned table gains `flux_<i>` and `err_<i>` columns
for image index `i`.

### Parameters of `pipeline.run` (and `Pipeline.__init__`)

All arguments after `segmap` are keyword-only. Sequence arguments run
parallel to `images` (index 0 = detection image).

`images` (`Sequence[np.ndarray]`, required)
: Science images. The first is the high-resolution detection image; each
  subsequent image is fit.

`segmap` (`np.ndarray`, required)
: Segmentation map on the detection-image grid; pixel values are catalog ids.

`catalog` (`Table | None`, default `None`)
: Source catalog with `id`, `x`, `y` columns matching the segmentation ids.
  The output table is a trimmed copy (`id`, `x`, `y`, plus any deblending and
  saturation flag columns) with the flux columns added; the input table is
  not modified. Required in practice: `run()` raises `NotImplementedError`
  when no catalog is given.

`psfs` (`Sequence[np.ndarray] | None`, default `None`)
: Per-image PSFs, as arrays or {class}`mophongo.psf_map.PSFRegionMap`
  instances; must match `images` in length when given. The detection-image
  PSF (`psfs[0]`) supplies the wings for template extension.

`weights` (`Sequence[np.ndarray] | None`, default `None`)
: Per-image inverse-variance maps. `wht_images` is an accepted alias. The
  detection-image entry may be `None`, but every fitted image needs a map;
  omitting both arguments is not supported and fails inside
  {meth}`mophongo.pipeline.Pipeline.run`.

`wht_images` (`Sequence[np.ndarray] | None`, default `None`)
: Alias for `weights`, kept for backward compatibility.

`kernels` (`Sequence[np.ndarray | PSFRegionMap] | None`, default `None`)
: Precomputed PSF-matching kernels per image (array or region map). A `None`
  entry fits the templates without further convolution; config-driven runs
  build kernels from the high- and low-resolution PSF maps automatically.

`psf_throughputs` (`Sequence[float] | None`, default `None`)
: Per-filter finite-stamp PSF sums used to convert fitted amplitudes into
  `flux_<i>_total` (see conventions below); must match `images` in length.

`wcs` (`Sequence[WCS] | None`, default `None`)
: Per-image WCS. Required for multi-resolution fitting (bin factors are
  derived from the WCS pixel scales) and for sky coordinates in outputs.

`window` (default `None`)
: Stored on the instance but not consumed by the fit as of this writing; the
  annotated `Window` type is not defined in the package. Leave unset.

`extend_templates` (`str | None`, default `None`)
: Template extension mode: `"psf_wings"` adds the high-resolution PSF wings
  beyond the segment, `"psf_model"` replaces the template with the PSF, and
  `None` leaves templates truncated at the segment boundary. Config-driven
  runs default to `"psf_wings"`.

`templates` (`Templates | Sequence[Template] | None`, default `None`)
: Pre-built templates to reuse instead of extracting them from `segmap`.

`config` (`FitConfig | None`, default `None`)
: Fitting configuration; a default {class}`mophongo.fit.FitConfig` is used
  when omitted. All fields are documented in {doc}`fitting`.

{func}`mophongo.pipeline.run` returns `(table, residuals, pipeline)`;
{meth}`mophongo.pipeline.Pipeline.run` returns `(table, residuals)` and keeps
the fitter, templates, and model images on the instance.

## Conventions

- **Weights are inverse variance.** A weight map `wht` is interpreted as
  `1 / sigma**2` per pixel, so `1 / sqrt(wht)` is the pixel RMS. Weight maps
  in other conventions must be converted before use.
- **PSF shape vs throughput.** Fitting uses unit-sum PSF shapes; templates
  are normalized to unit sum, so fitted amplitudes are modeled-stamp fluxes.
  The finite PSF stamp sum is kept separately as filter-level throughput
  metadata. {meth}`mophongo.pipeline.Pipeline.run` writes the raw fitted
  amplitude as `flux_<i>` and the throughput-corrected total as
  `flux_<i>_total` (with matching `err_<i>_total`, `err_pred_<i>_total`, and
  `throughput_<i>` columns). Native PSF stamps are not silently
  renormalized, since that would discard the throughput information.
- **Multi-resolution handling.** When a measurement band is coarser than the
  detection grid, an integer bin factor is derived from the two WCS. The
  low-resolution science image is block-replicated with flux conservation
  onto the fine grid; the inverse-variance map is copied to the subpixels
  and multiplied by `factor**2`, preserving the native chi-square and flux
  errors.
- **Coordinate systems.** Each {class}`mophongo.templates.Template` carries
  both original-image and cutout pixel coordinates. Positions in catalogs
  refer to the detection-image grid; templates map between the two through
  their stored slices.

## Requirements

Mophongo requires Python >= 3.11, < 3.13. Core dependencies are `numpy`,
`scipy`, `astropy`, `photutils`, and `stpsf` (for JWST/HST PSF generation),
with `drizzlepac` used to drizzle empirical PSFs to mosaic frames.
`shapely` and `geopandas` are also required: they back the
{class}`mophongo.psf_map.PSFRegionMap` region geometry used by every
config-driven run. Install
with Poetry (`poetry install`) or in editable mode with
`pip install -e .`.
