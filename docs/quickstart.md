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

Mophongo is not on PyPI; clone it from GitHub and install from the checkout.
It needs Python >= 3.11, < 3.13.

```bash
git clone https://github.com/ivolabbe/mophongo.git
cd mophongo
poetry install          # resolves and installs into .venv (recommended)
```

or, in an environment you manage yourself:

```bash
pip install -e .        # editable install from the checkout
```

Poetry keeps the environment in `.venv/` inside the checkout and does not put
it on your `PATH`. Either prefix commands with `poetry run`, or activate it:

```bash
poetry run mophongo --help              # one-off
eval $(poetry env activate)             # activate for the session (Poetry >= 2.0)
source .venv/bin/activate               # same thing, by hand
```

The `mophongo` console script and `python -m mophongo.pipeline` both become
available once the environment is active ({doc}`diagnostics` covers the
subcommands).

## Choosing an entry point

Both paths share the fitting engine; they differ in what you supply and what
gets built for you.

| | Arrays: {func}`mophongo.pipeline.run` | Config: {meth}`mophongo.pipeline.Pipeline.from_config` |
|---|---|---|
| Inputs | in-memory numpy arrays + `astropy` Table | FITS/CSV paths in a JSON `RunConfig` |
| PSFs | one static stamp per band, supplied by you | drizzled, position-dependent ePSFs built from the per-frame WCS CSVs |
| Kernels | you build them ({func}`mophongo.utils.matching_kernel`), or pass a prebuilt {class}`mophongo.psf_map.PSFRegionMap` | per-region kernel map, built and geojson-cached automatically |
| Template build scheme | `"psf_wings"` by default, from the PSF stamp you pass as `psfs[0]` | the same default, from the drizzled detection-band PSF map |
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
`outputs`, and `all`. The `mophongo` console script runs the same steps and
reaches the products of a finished run without opening a session:

```bash
mophongo run    f770w.json psfs fit          # same steps as above
mophongo info   runs/f770w                   # summarize a run, no pixels read

# the matching kernel at a sky position, as FITS with a WCS
mophongo psf    runs/f770w 34.5202 -5.2566 --map-kind kernel -o kernel.fits

# one source: cutouts, PSFs, and fit row as a multi-extension FITS, plus the
# subphot six-panel diagnostic
mophongo stamps runs/f770w 4711 --half-size 40
mophongo diag   runs/f770w 4711 --size 101
```

Every subcommand takes the config JSON or the run directory. `psf` reads only
the cached region map, so it is instant; `stamps` and `diag` restore the run
with `load_fit()` and share that single load across the ids given. See
{doc}`diagnostics` for what they write. {meth}`mophongo.pipeline.Pipeline.from_config` is
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
  "wht_hi": "wht_hi.fits",
  "csv_hi": "frames_hi.csv",
  "csv_lo": "frames_lo.csv",
  "filter_lo": "f770w",
  "psf": {
    "pattern_hi": "STDPSF_NRCA._F444W.*fits",
    "pattern_lo": "STDPSF_MIRI_F770W.*fits"
  }
}
```

Lines starting with `#` are treated as comments and stripped, so configs can
be annotated. Unknown keys raise an error, so typos fail loudly.

One input stays implicit above: every run needs a detection-band weight map.
`wht_hi` names it; left unset, the run looks for the standard
`_sci.fits` -> `_wht.fits` sibling of `sci_hi`.

A realistic config for a JWST field — 40 mas F444W detection mosaic, 80 mas
MIRI F770W band, MJD-tagged ePSF grids, a trial patch for testing before the
full-field run — looks like:

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

  # MJD-tagged ePSF grids; psf.size in arcsec
  "filter_lo": "f770w",
  "psf": {
    "dir": "data/PSF",
    "pattern_hi": "NRC.._F444W_MJD\\d+_GRID25_OS4",
    "pattern_lo": "MIRI_F770W_MJD\\d+_GRID9_OS4",
    "size": 4.0,
    "blur_fwhm": "default"
  },

  # preprocessing: footprint cut + trial patch ("trial": null = full mosaic)
  "footprint_filter": true,
  "trial": {"center": [34.35, -5.27], "radius": 0.6},
  "bg_filter_sigma": 64.0,

  # FitConfig overrides
  "fit": {"fit_astrometry_joint": true, "aperture_diam": 0.5}
}
```

Start with a small `trial` patch to validate PSFs, kernels, and residuals,
then set `trial` to `null` for the full field: the cached PSF and kernel maps
in `out_dir` are reused.

### The config fields

{class}`mophongo.pipeline.RunConfig` describes one filter fit (one
high-resolution plus one low-resolution band). The required fields are the
ones in the minimal config above: the run `name` and `out_dir`, the
high-resolution side (`sci_hi`, `segmap`, `catalog`, `csv_hi`), and the band
to fit (`sci_lo`, `wht_lo`, `csv_lo`). Two nested blocks hold the settings
that come in groups: `psf` selects the ePSF grids and the stamp size
({class}`mophongo.pipeline.PsfConfig`), and `fit` forwards keywords to
{class}`mophongo.fit.FitConfig`, including the template build scheme
(`extend_mode`). The remaining top-level fields control preprocessing
(`bg_filter_sigma`, `footprint_filter`, `trial`) and outputs (`scene_plots`,
`save_stamps`).

Every field, its default, and the full description of the config-driven flow
are in {doc}`pipeline`; the `FitConfig` fields are in {doc}`fitting`.

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
- one PSF-matching kernel per fitted band (see below);
- the detection-band PSF as `psfs[0]`, which the default template build
  scheme needs (see below).

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

If a fitted band is coarser than the detection image, pass per-image WCS
objects via `wcs`: the pipeline then derives the integer bin factor between
the two grids and fits on the reference grid ({doc}`pipeline`).

Every argument of {func}`mophongo.pipeline.run` (the `Pipeline` constructor
takes the same ones) is documented in {doc}`pipeline`, the kernel arguments in
{doc}`psf`, and the `FitConfig` fields in {doc}`fitting`.

### Output columns

For each fitted image `i` (counting from 1, in input order) the returned table
gains `flux_i` (the fitted amplitude), `err_i` and `err_pred_i` (solver and
weight-map errors), `throughput_i`, the total-flux versions `flux_i_total`,
`err_i_total`, `err_pred_i_total`, and `scene_i`, the scene the source was
fitted in (`-1` for sources with no template). Aperture and diagnostic columns
are described in {doc}`outputs`.

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
- {doc}`diagnostics` — per-source inspection and the `mophongo` command line.
- {doc}`simulation` — synthetic mosaics and injected-truth verification.
- {doc}`api` — full API reference.
