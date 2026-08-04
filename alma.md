# Running the MINERVA-UDS F770W photometry (mophongo)

Short instructions for reproducing `examples/run_uds_770_dr0.1.py`: template-fitting
photometry of the MINERVA-UDS MIRI F770W mosaic using the F444W mosaic as the
high-resolution template prior.

Everything below is fetched from public/collaboration archives — no files from
anyone's personal Drive.

## 1. Get the code

```bash
git clone https://github.com/ivolabbe/mophongo.git
cd mophongo
git checkout flux-bug            # the branch this run lives on
poetry install                   # or: pip install -e .
```

Python ≥3.10. PSF generation additionally needs `stpsf` (formerly `webbpsf`) plus its
reference-data files; see https://stpsf.readthedocs.io for the `STPSF_PATH` setup.

## 2. Get the data

Make a data directory outside the repo, e.g. `/path/to/MINERVA/data/UDS/`, with
subdirectories `n3.0/`, `m3.0/`, `n3.0_v1.2_SEC/`, `n3.0_m3.1_v1.2.1/`.

### 2a. NIRCam F444W mosaic — grizli S3 (public, no login)

Index: https://s3.amazonaws.com/grizli-v2/MINERVA/mosaics/uds/40mas-v3.0/index.html

```bash
cd .../UDS/n3.0
B=https://s3.amazonaws.com/grizli-v2/MINERVA/mosaics/uds/40mas-v3.0
curl -O $B/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci.fits.gz   # 2.2 GB
curl -O $B/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_wht.fits.gz   # 2.1 GB
curl -O $B/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv           # frame list
gunzip *.gz
```

The `_wcs.csv` is the per-exposure WCS/MJD listing that drives the drizzled PSFs — it is
required, not optional. Expect 297 frames.

### 2b. MIRI F770W mosaic, segmap, catalog — MINERVA Google Drive

These are collaboration products, so you need MINERVA Drive access (ask Ivo or the WG2
leads to share `MINERVA > Data > UDS` with your Google account). Download through the
Drive web UI.

| File | Drive location | Put it in |
|---|---|---|
| `uds-sbkgsub-v3.0-80mas-f770w_drz_sci_extrabkg.fits.gz` | `UDS/Images/MIRI/v3.0/` | `m3.0/` |
| `uds-sbkgsub-v3.0-80mas-f770w_drz_wht.fits.gz` | `UDS/Images/MIRI/v3.0/` | `m3.0/` |
| `uds-v3.0_f770_wcs.csv` | `UDS/Images/MIRI/v3.0/` | `m3.0/` |
| `MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits.gz` | `UDS/Catalogs/All Versions/n3.0_v1.2/ACS+WEBB Chi-Mean Detection/Ancillary/` | `n3.0_v1.2_SEC/` |
| `MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits` | `UDS/Catalogs/All Versions/n3.0_m3.1_v1.2.1/ACS+Webb Chi-Mean Detection/` | `n3.0_m3.1_v1.2.1/` |

Direct folder links:

- UDS root: https://drive.google.com/drive/folders/19tucvAe52nMUQ6qS5hqTcD5kDkiv7YX6
- MIRI v3.0: https://drive.google.com/drive/folders/1ysCwj26OmXDvk_m5oPAehPlfUKVEi-uh
- n3.0_v1.2 Ancillary (segmap): https://drive.google.com/drive/folders/1pMteyjfSw1I6JNHNqMICp4YO_sYXmdrv
- n3.0_m3.1_v1.2.1 catalog: https://drive.google.com/drive/folders/1f_jIJ3TI0C-Kx648hTTMTSBIcZMzYes3

`gunzip` everything, then add the symlink the filter parser expects (the shipped MIRI
csv has no trailing `w` in the filter token):

```bash
cd .../UDS/m3.0
gunzip *.gz
ln -sf uds-v3.0_f770_wcs.csv uds-v3.0_f770w_wcs.csv
```

Expect 229 F770W frames.

### 2c. PSF grids — generate them locally

The MJD-tagged ePSF grids are not in the repo (too large). Build them from the two
`_wcs.csv` files with `stpsf` — one grid per detector/epoch cluster:

```python
from mophongo.psf_factory import PSFFactory

nrc = PSFFactory(prefix="UDS", outdir="data/PSF", date_mode="cluster", delta_day=2.0,
                 num_psfs=25, oversample=4, fov_arcsec=4.0)
nrc.from_csv(".../UDS/n3.0/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv")

miri = PSFFactory(prefix="UDS", outdir="data/PSF", date_mode="cluster", delta_day=2.0,
                  num_psfs=9, oversample=4, fov_arcsec=8.0)
miri.from_csv(".../UDS/m3.0/uds-v3.0_f770_wcs.csv")
```

This writes files named `UDS_NRCA5_F444W_MJD<mjd>_GRID25_OS4.fits`,
`UDS_NRCB5_F444W_MJD<mjd>_GRID25_OS4.fits` and `UDS_MIRI_F770W_MJD<mjd>_GRID9_OS4.fits`
into `data/PSF/`, which is exactly what the config's `pattern_hi` / `pattern_lo`
regexes look for. Takes a while (tens of minutes); it only has to be done once.

## 3. Point the config at your paths

Edit the five absolute paths at the top of `examples/uds_770_dr0.1.json`
(`sci_hi`, `segmap`, `catalog`, `csv_hi`, `sci_lo`, `wht_lo`, `csv_lo`) to your data
directory. Leave the rest alone — `#` comment lines in that file explain each setting.

Useful knobs:

- `r_trial`: radius in arcmin of a small trial patch. `0.5` (the default in the file) is
  a quick ~10-minute test run; set it to `0` for the full mosaic (hours, large RAM).
- `trial_center`: `[ra, dec]` of the patch, currently the deepest fully F770W-covered
  0.5' region.
- `out_dir`: where products are written (relative to `examples/`).

## 4. Run

```bash
cd examples
python run_uds_770_dr0.1.py
```

or equivalently, straight from the config:

```bash
python -m mophongo.pipeline uds_770_dr0.1.json
```

The script is three steps and can be run cell-by-cell in an IDE:

```python
pipe = Pipeline.from_config("uds_770_dr0.1.json")
pipe.build_psfs()      # per-band PSF region maps  -> *_psf_hi/lo.geojson (cached)
pipe.build_kernels()   # F444W->F770W matching kernels -> *_kernel.geojson (cached)
pipe.run()             # load data, cut to the patch, fit
pipe.write_outputs()   # residual FITS, fit table, scene PNGs, scene catalog
```

The geojson caches mean the PSF/kernel steps are only paid once; re-running with a
different `r_trial` reuses them.

## 5. Outputs

In `examples/uds_770_dr0.1/`:

| File | What |
|---|---|
| `uds_770_fit_table.fits` | fitted catalog. `flux_1` = raw fitted F770W template amplitude, `flux_1_total` = throughput-corrected total, plus errors, chi², astrometric shifts |
| `uds_770_residual.fits` | F444W-grid residual image |
| `uds_770_psf_hi.geojson`, `uds_770_psf_lo.geojson`, `uds_770_kernel.geojson` | PSF/kernel region maps (+ matching `.fits` cubes) |
| `uds_770_scene_*.png`, `uds_770_scene_catalog.csv` | per-scene diagnostic plots and their positions |

First things to check: the residual image around bright sources, `err_1/err_pred_1`
≈ 1 (error calibration), and the astrometry log lines reporting the bulk MIRI-vs-NIRCam
offset convergence.

## Notes / gotchas

- The template is the **raw** F444W mosaic, not the aperpy `_sci_f444w-matched` image
  used by the older `examples/run_uds_770_wren.py`; mophongo drizzles its own ePSFs, so
  it needs the unmatched mosaic.
- No saturated-star repair has been run on the v3.0 UDS mosaic; bright stars will show
  residuals.
- `expect_frames: [297, 229]` in the config is a sanity assert on the two `_wcs.csv`
  files. If it trips, you have a different mosaic version than this config expects.
- Background context lives in `GUIDE.md`, `STATUS.md` (recent work) and `TODO.md` (open
  items) in the repo root.
