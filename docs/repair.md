# Repairing saturated stars

Saturated stars leave interior regions of zero weight ("holes") in drizzled
JWST/HST mosaics, surrounded by bright PSF wings. Detection and photometry
software then fragments each star into many spurious sources around an empty
core. `mophongo` ships a standalone repair tool that fixes both the images
and the catalog, without running the photometry pipeline:

1. **Image repair** — detect the holes, fit the local PSF amplitude on a
   ring around each one, and fill the core with the best-fit PSF model.
   The result is a repaired science/weight FITS pair that can be used as
   the detection image for any catalog software.
2. **Catalog repair** (optional) — merge the fragmented segments of each
   repaired star into a single parent source and add a
   `FLAG_SATURATED_<FILTER>` column to the catalog.

## Quick start

```bash
mophongo-repair mosaic-f444w_drc_sci.fits mosaic-f444w_drc_wht.fits
```

or, without the console script installed:

```bash
python -m mophongo.repair mosaic-f444w_drc_sci.fits mosaic-f444w_drc_wht.fits
```

To also repair a catalog and segmentation map built from these images:

```bash
mophongo-repair mosaic-f444w_drc_sci.fits mosaic-f444w_drc_wht.fits \
    --catalog SUPER_CATALOG.fits --segmap SEGMAP.fits
```

The filter is parsed from the filename (`f444w` above); pass `--filter`
when the filename does not contain it.

## What it needs

- **Science + weight pair.** A drizzled science mosaic and its
  inverse-variance weight map. Saturated cores must appear as `wht = 0`
  pixels (the grizli / JWST pipeline convention).
- **Exposure listing.** The PSF model is drizzled from the contributing
  exposures, so the tool needs the grizli `<root>_wcs.csv` next to the
  mosaic (or `--csv`). When the file is missing it is reconstructed from
  public MAST/S3 cal-file headers, which requires network access.
- **PSF grids.** STDPSF FITS grids are loaded from `--psf-dir` (default
  `<sci dir>/PSF`); filenames are matched against `--psf-pattern` (a
  regex, default `NRC.._<FILTER>` for NIRCam and `<FILTER>` otherwise).
  When none match, one is generated per detector at the modal exposure
  epoch with [stpsf](https://stpsf.readthedocs.io), which requires the
  stpsf reference data to be installed. Pass `--no-build-psf` to fail
  instead of generating.

## Outputs

Everything is written next to the science image, or to `--out-dir`. File
names are built from the input stems:

| File | Content |
| --- | --- |
| `<sci>_repaired.fits` | science image, saturated cores replaced by the PSF model |
| `<wht>_repaired.fits` | weight map, repaired pixels restored to the local weight |
| `<sci>_saturate_repair.csv` | per-hole fit table (amplitude, quality metrics, status) |
| `<catalog>_repaired.fits` | catalog with merged parents and `FLAG_SATURATED_<FILTER>` |
| `<segmap>_repaired.fits` | segmentation map with child labels merged per star |
| `<catalog>_mergelog.csv` | one row per merged star: parent id and child labels |

With `--mode subtract` the image outputs are named `<sci>_subtracted.fits`
/ `<wht>_subtracted.fits` and the CSV `<sci>_saturate_subtract.csv`; in
that mode the weight is *not* restored — saturated cores and pixels with
bad residuals are blanked to `sci = wht = 0` so downstream photometry
skips them.

Sources whose fit failed keep `wht = 0` in the repaired weight map and are
recorded in the CSV with a `status` explaining the rejection. Repaired
images carry `SATREPAI`, `SATMODE`, `SATNFIX`, and `SATFILT` (when the
filter is known) header keywords for provenance.

### The saturation flag

`FLAG_SATURATED_<FILTER>` (e.g. `FLAG_SATURATED_F444W`) is `1` for
catalog rows whose photometry rests on a repaired core: the fragmented
child rows are removed and replaced by one parent row at the PSF-fit
position. Downstream users can select or reject these sources with a
single column cut. The parent row is inherited from the largest child
segment, with its `id`, `x` and `y` replaced by the PSF-fit values.

The merge is conservative when it can be: a candidate segment inside the
merge circle is absorbed only if the scaled PSF model accounts for at
least `flux_frac_thresh` (default 0.5) of the science flux in its pixels,
so a neighbour whose own light dominates over the star's wings stays a
separate source (see
{func}`mophongo.catalog.repair_saturated_catalog`). Without that test the
merge is geometric: every segment inside the circle becomes part of the
star.

## Python API

```python
from mophongo.repair import repair_image, flag_catalog

res = repair_image("mosaic-f444w_drc_sci.fits", "mosaic-f444w_drc_wht.fits")
res["fits"]        # per-hole fit table
res["sci_out"]     # path of the repaired science image

flag_catalog(
    "SUPER_CATALOG.fits", "SEGMAP.fits", res["fits"],
    filter_name=res["filter"], fwhm_pix=res["fwhm_pix"],
)
```

Both functions are thin wrappers around
{func}`mophongo.saturate.repair_saturated_holes` and
{func}`mophongo.catalog.repair_saturated_catalog`; keywords of the
underlying functions pass through, except the ones the wrappers set
themselves (`wcs`, `psf_filter`, `output_csv`, `plot_dir` for
`repair_image`). The neighbour-protection flux filter needs both `sci`
and a unit-sum `psf_stamp` passed to `flag_catalog` — the command line
does this automatically.

## Tuning

- `--min-buffer-snr` (default 200) is the saturation pre-filter: the
  median flux just outside the hole, in sigma above sky. The default
  suits bands whose saturated stars have bright halos, such as F444W.
  Bluer bands have fainter halos and need a lower threshold: values of
  order 20 for F356W and 5 for F277W have been used in practice. Holes
  below the threshold are left untouched and recorded with a low
  buffer-SNR status.
- `--fwhm-pix` overrides the PSF FWHM used for the fitting-ring
  geometry. By default it is measured from a drizzled PSF stamp.
- `--merge-radius` (default 3) closes gaps between nearby `wht = 0`
  fragments before hole labelling, so a saturation footprint broken up
  by the dither pattern counts as one star.
- `--n-fwhm` (default 5) sets the catalog merge radius in units of the
  FWHM. Increase it when diffraction spikes fragment the segmentation map
  far from the core.
- `--plots` writes one diagnostic PNG per hole that reached the fitting
  stage — data, fitted model, fitting ring, and the repaired result —
  into `<sci>_saturate_<mode>_png/`.
- `--mode subtract` runs the same fit but removes the full PSF halo
  (wings and diffraction spikes) instead of filling the core — a
  preprocessing step for photometry near bright stars. See
  {doc}`preprocessing` for when to use which.

The algorithm, its acceptance metrics, and the failure modes are
described in {doc}`preprocessing`; the API reference is
{mod}`mophongo.repair`, {mod}`mophongo.saturate`, and
{func}`mophongo.catalog.repair_saturated_catalog`.
