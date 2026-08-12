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
2. **Catalog repair** (optional) — flag the fragmented segments of each
   repaired star in a `FLAG_SATURATED_TMPL` column and clean them out of
   the segmentation map, keeping one segment per star.

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
  When none match, grids are generated with
  [stpsf](https://stpsf.readthedocs.io) — one per detector per
  observation epoch (`date_mode="cluster"`, the same policy as the
  pipeline's PSF generation), and the grid nearest in MJD is used for
  each contributing exposure. Requires the stpsf reference data. Pass
  `--no-build-psf` to fail instead of generating.

## Outputs

Everything is written next to the science image, or to `--out-dir`. File
names are built from the input stems:

| File | Content |
| --- | --- |
| `<sci>_repaired.fits` | science image, saturated cores replaced by the PSF model |
| `<wht>_repaired.fits` | weight map, repaired pixels restored to the local weight |
| `<sci>_saturate_repair.csv` (+ `.fits`) | per-hole fit table (amplitude, quality metrics, status) |
| `<catalog>_flagged.fits` | catalog with `FLAG_SATURATED_TMPL` on star-dominated segments |
| `<segmap>_flagged.fits` | segmentation map with the flagged segments set to 0 |
| `<catalog>_flaglog.csv` | per (star, segment): observed flux, model flux, flag decision |
| `<catalog>_flag_diagnostic.png` | per-star before/after panels (see below) |

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

Catalogs built before the repair split each saturated star into many
spurious sources: an empty core, wedges of PSF wing, and diffraction
spike fragments. The catalog step identifies those segments by comparing
fluxes: the best-fit star model from the image repair (`A · PSF`) is
placed at each star, and a segment is considered saturated when the
model accounts for more than `--flux-frac` (default 0.3) of the observed
flux in its pixels *and* the model flux exceeds a noise floor
(`min_snr x sky_noise x sqrt(n_pix)`, default 5σ — without it every
noise-level segment in the far wings would flag on a near-zero
denominator). A genuine neighbour whose own light dominates over the
star's wings fails the flux test and is kept.

The flag column encodes star membership: all flagged segments of one
star get the same **group id** — the lowest flagged segment id of that
star — in `FLAG_SATURATED_TMPL` (e.g. segments 6, 7, 9, 100 of one
star all get 6). `flag > 0` is the boolean cut; equal values group rows
belonging to the same star. The column name is band-independent
(`TMPL` = the template band the repair ran on) so downstream code does
not change when the detection band does; pass `filter_name` to override.
No rows are dropped or added, so row order and matching to other
versions of the catalog are preserved.

In the output segmentation map the flagged labels are set to 0, and the
star's core — the undetected `seg = 0` region plus the repaired pixels
within the fit radius `r_in` — is set to the group id. The group-id
row therefore keeps a segment covering the PSF-repaired core, and a
mophongo run on the repaired image models the star as a normal source
while the flag column marks its photometry as saturated-repaired.

Every decision is recorded in `<catalog>_flaglog.csv` (observed flux,
model flux, ratio, and group id per segment), and a diagnostic PNG is
written by default with five panels per star: segmap before; science
before the repair with the to-flag segments overlaid; repaired science
with the flagged segments overlaid; segmap after with the whole star
(flagged segments + filled core) in a single color; and the repaired
science with the flagged segments zeroed, so the kept neighbours stand
out.

With `--merge` the flagged children are instead merged destructively
into a single parent row at the PSF-fit position, inherited from the
largest child segment (see
{func}`mophongo.catalog.repair_saturated_catalog`); outputs then use the
`_repaired` suffix and a `_mergelog.csv`.

## Python API

```python
from mophongo.repair import repair_image, flag_catalog, drizzled_psf_stamp

res = repair_image("mosaic-f444w_drc_sci.fits", "mosaic-f444w_drc_wht.fits")
res["fits"]        # per-hole fit table
res["sci_out"]     # path of the repaired science image

flag_catalog(
    "SUPER_CATALOG.fits", "SEGMAP.fits", res["fits"],
    sci=res["sci"],                       # repaired image
    psf_stamp=drizzled_psf_stamp(res["dpsf"], res["psf_pattern"], npix=401),
    flux_frac=0.3,
)
```

Both functions are thin wrappers around
{func}`mophongo.saturate.repair_saturated_holes` and
{func}`mophongo.catalog.flag_saturated_segments` /
{func}`mophongo.catalog.repair_saturated_catalog` (`merge=True`);
keywords of the underlying functions pass through, except the ones the
wrappers set themselves (`wcs`, `psf_filter`, `output_csv`, `plot_dir`
for `repair_image`). Segments are compared only where the model is
non-zero, so the `psf_stamp` support sets how far from each star
segments can be flagged: an ePSF is identically zero beyond its native
field of view. For bright stars whose diffraction spikes fragment the
segmentation map, drizzle the stamp from a large-FOV (~30") ePSF and use
a generous `npix`, as in the example notebook.

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
- `--flux-frac` (default 0.3) is the catalog flag threshold: a segment
  is flagged when the star model exceeds this fraction of its observed
  flux. Lower values flag more aggressively.
- `--merge` switches the catalog step from flag-only to the destructive
  merge; `--n-fwhm` (default 5) then sets the merge radius in units of
  the FWHM. Increase it when diffraction spikes fragment the
  segmentation map far from the core.
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
