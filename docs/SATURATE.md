# `mophongo.saturate` — saturated-star repair and subtraction

Two-pass preprocessing for drizzled JWST mosaics:

1. **Repair pass** — find interior `wht=0` "saturation holes", fit the local
   STPSF to a donut around each one, and replace the saturated core with the
   best-fit model. Output: a science/weight image with the cores filled in,
   suitable as the **detection / segmentation image** for a photometry pipeline.
2. **Subtract pass** — using the *same* fit machinery on the *original*
   science image, subtract a much larger PSF (full diffraction halo + spikes,
   ~30″) from each star's footprint and mask any residual contamination.
   Output: a science/weight image cleaned of bright-star halos, suitable for
   **template photometry of nearby galaxies**.

Both passes share one entry point: `repair_saturated_holes(...)` with
`mode='repair'` or `mode='subtract'`.

---

## Pipeline at a glance

```
                 sci, wht (drizzled mosaic)
                          │
                          ▼
            ┌─────────────────────────────┐
            │  find_wht_holes()           │  detect interior wht=0 blobs
            └─────────────────────────────┘
                          │ holes table (xc, yc, r_equiv, area)
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │  repair_saturated_holes(mode='repair', dpsf=std)    │
   │    • per-source donut fit of A·ψ                    │
   │    • iterative shift via WCS re-drizzle              │
   │    • optional pedestal switch when host-embedded    │
   │    • replace dilated_mask with A·ψ                  │
   └─────────────────────────────────────────────────────┘
                          │ saturate_*.csv  +  repair_*.png
                          │ sci_repaired, wht_repaired
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │  repair_saturated_holes(mode='subtract',            │
   │                         dpsf=large, sci, wht)       │  ← original image
   │    • same fit on the same donut                     │
   │    • SUBTRACT A·ψ_large from full cutout            │
   │    • mask bad-residual pixels (sci=wht=0)           │
   └─────────────────────────────────────────────────────┘
                          │ subtract_*.csv  +  subtract_*.png
                          ▼
                   sci_subtracted, wht_subtracted
                  (clean of bright-star halos)
```

---

## Step-by-step algorithm

### 1. Hole detection — `find_wht_holes(wht, min_area=1, merge_radius=3)`

* **Interior** wht=0 blobs only — pixels in the wht map that are zero AND
  not connected to the image border (computed via `binary_fill_holes`).
* `min_area=1` because JWST saturation can punch only 1–3 px holes.
* Optional `merge_radius` morphologically closes nearby fragments so a
  diffraction-spike-broken hole counts as one source.
* Returns a Table with `id, yc, xc, area, r_equiv = √(area/π)`.

### 2. Per-source loop (inside `repair_saturated_holes`)

For each detected hole:

1. **Cutout** sized for the donut (repair) or to the large-PSF FOV
   (subtract), clipped to image bounds.
2. **Restrict the hole** to its connected component containing the source
   centre — large cutouts contain many unrelated wht=0 pixels (CRs,
   neighbouring stars, chip edges) that would otherwise pollute the fit.
3. **Dilate** by `hole_dilate=2` pixels → `dilated_mask`; the
   2-px-thick annulus `dilated_mask & ~hole_mask` is the *buffer ring*.
4. **Saturation pre-filter**: median (data − sky) over the buffer ring,
   in σ. If `< min_buffer_snr` (default 200), this is not a saturation
   core; skip with status `"low buffer SNR"`.

### 3. Joint amplitude + sub-pixel shift fit

* **Drizzle** the STPSF onto the cutout WCS at the current `(RA, Dec)`.
* **Linearised LSQ** on the donut `[r_in, r_out]` with `bad_mask =
  dilated_mask`, solving
  `data ≈ A·ψ + B·∂ψ/∂x + C·∂ψ/∂y` →  `dx = -B/A, dy = -C/A`.
* Update `(cx, cy) += (dx, dy)`, **re-drizzle** the PSF at the new
  world coordinates (no analytic shift), iterate until
  `|dx|, |dy| < shift_tol = 0.05 px` or `max_shift_iter = 5`.
* **Cumulative shift cap**: `|shift_total| ≤ max_shift_pix = 3` px to
  prevent the model walking off onto a neighbour.
* Final amplitude-only fit at the converged position records the
  `ring_mask` actually used.

### 4. Decision FOM and pedestal switch

* `data_to_model = Σ data / Σ (A·ψ)` over the donut ring.
  * `≈ 1` → clean star.
  * `> extended_max_data_to_model = 1.15` → the data has more flux
    than the model can account for → **embedded source** (e.g. AGN
    inside a galaxy or a star on top of one).
* When triggered, refit on the *same* donut with `fit_pedestal=True`:
  `data ≈ A·ψ + C` (2-column LSQ). The constant `C` absorbs the host's
  smooth surface brightness so `A` reflects the point-source flux only.
  `fit_mode = "donut+pedestal"`. **Only `A·ψ` is replaced/subtracted —
  the pedestal stays in the image.**

### 5. Bad-fit guard

If `Σ|residual| / Σ|A·ψ|` over the donut exceeds `max_resid_frac = 1.0`,
the model can't describe the data — leaving the image untouched is safer
than corrupting it. Row written with `ok=False, status="bad fit … no action"`.

### 6. Action

`mode='repair'`
:   Replace `dilated_mask` pixels with `A·ψ`. Restore `wht` on those
    pixels to the median wht of the donut so the photometry pipeline
    treats them as ordinary high-S/N pixels. Source brightness is
    correctly modelled, position-dependent.

`mode='subtract'`
:   Subtract `A·ψ` from the entire cutout. Then **mask** pixels where:

      * default mode  → `r ≤ r_out` AND `A·ψ > 1e-4 × peak(A·ψ)`,
      * pedestal mode → just `dilated_mask`,

    AND `|residual| > 1.5 × sky_noise`. Set both `sci` and `wht` to 0
    at masked pixels — downstream photometry skips them via `wht=0`.

### 7. Diagnostics (8-panel PNG per source, optional)

```
+-----------+-----------+-----------+-----------+
| data / A  |   A·ψ     |  resid    |  resid    |
| (log)     | (log)     |  shifted  |  no-shift |
|           |           | (RdBu)    | (RdBu)    |
+-----------+-----------+-----------+-----------+
| data +    | result    | result    | radial    |
| masks     | (log /    | (zoom 2×) | profile   |
|           |  linear)  |           |           |
+-----------+-----------+-----------+-----------+
```

* **(0,0)–(0,1)**: log10 grayscale, fixed stretch `[-5.3, -1.5]` in
  units of `sci / A`.
* **(0,2)–(0,3)**: residual `(data − A·ψ) / A`, RdBu_r,
  ±3 × MAD over all valid pixels of the residual.
* **(1,0)**: data with `hole` (red) and `fit ring` (cyan) contours.
* **(1,1)–(1,2)**:
  * `mode='repair'` → log10 grayscale of repaired `sci / A`, with
    lime contour around the filled footprint.
  * `mode='subtract'` → **linear** grayscale of `sci_sub / A`,
    ±3 × MAD, with a semi-transparent lime `r_out` circle. Masked
    pixels (`sci = 0`) appear as the colormap's mid-grey.
* **(1,3)**: median radial profile of data, A·ψ, and result.

Suptitle reports: `id`, position, `A`, `r_eq`, `r_out`, buffer SNR,
shift, `n_iter`, **decision FOM `Σdata/Σ(A·ψ)` with → PSF only or
PSF + pedestal**.

---

## API cheatsheet

```python
from mophongo.saturate import find_wht_holes, repair_saturated_holes
from mophongo.psf import DrizzlePSF
from astropy.io import fits
from astropy.wcs import WCS

sci_hdu = fits.open("sci.fits")[0]
wcs     = WCS(sci_hdu.header)
sci     = sci_hdu.data.astype("float32")
wht     = fits.getdata("wht.fits").astype("float32")

# Standard STDPSF for the repair pass (built from psf_grid_from_csv,
# small ~4" FOV, 5x5 grid across the detector).
dpsf = DrizzlePSF(driz_image="sci.fits", csv_file="exposures_wcs.csv")
dpsf.epsf_obj.load_jwst_stdpsf(local_dir="data/PSF",
    filter_pattern="UDS_NRC.._F444W_OS4_GRID25")

# 1. detect holes
holes = find_wht_holes(wht, merge_radius=3, min_area=1)

# 2. repair pass
res = repair_saturated_holes(
    sci, wht,
    dpsf=dpsf, wcs=wcs, holes=holes,
    psf_filter="UDS_NRC.._F444W_OS4_GRID25",
    output_csv="saturate_fits.csv",
    plot_dir="diagnostics/",
)
fits.writeto("sci_repaired.fits", res["sci"], sci_hdu.header, overwrite=True)
fits.writeto("wht_repaired.fits", res["wht"], overwrite=True)

# 3. (optional) subtract pass with the LARGE STPSF (~30" FOV, single
#    position at detector centre, built once via scratch/build_large_psf.py).
dpsf_lg = DrizzlePSF(driz_image="sci.fits", csv_file="exposures_wcs.csv")
dpsf_lg.epsf_obj.load_jwst_stdpsf(local_dir="data/PSF",
    filter_pattern=r"UDS_NRC.._F444W_MJD\d+_FOV30_GRID1_OS4")

ok = res["fits"][res["fits"]["ok"]]
sub = repair_saturated_holes(
    sci, wht,                       # ORIGINAL — not the repaired image
    dpsf=dpsf_lg, wcs=wcs, holes=ok,
    psf_filter=r"UDS_NRC.._F444W_MJD\d+_FOV30_GRID1_OS4",
    mode="subtract",
    output_csv="subtract_fits.csv",
    plot_dir="diagnostics/",
)
fits.writeto("sci_subtracted.fits", sub["sci"], sci_hdu.header, overwrite=True)
fits.writeto("wht_subtracted.fits", sub["wht"], overwrite=True)
```

The two passes are *independent in principle* — the repair pass alone is
useful as a detection-image preprocessor; the subtract pass alone is
useful for cleaning halos from a sky image. They share the same fit
geometry so the same OK list naturally feeds from one to the other.

---

## Building the large PSF

The standard MAST STDPSF cubes have a ~4″ native FOV — fine for the
repair-stage donut fit but far too small to model JWST's diffraction
spikes and halo. Build a single-position, 30″-FOV PSF once per filter
via `mophongo.jwst_psf.psf_grid_from_csv`:

```python
psf_grid_from_csv(
    "exposures_wcs.csv",
    num_psfs=1,             # central position only
    oversample=4,
    fov_arcsec=30.0,
    prefix="UDS",
    postfix="FOV30_GRID1_OS4",  # NIRCam convention (30" halo grids)
    outdir="data/PSF",
    save=True,
)
```

This:

* picks the modal MJD of the input exposures and loads the matching
  measured OPD via `inst.load_wss_opd_by_date(...)` — guarantees the
  spike orientation matches the data;
* uses `inst.options['parity'] = 'odd'` so the PSF peak lands on a
  pixel centre (mophongo convention);
* writes a STDPSF-format cube the existing loader reads, so subsequent
  `DrizzlePSF.get_psf` calls work unchanged.

NIRCam → `FOV30_GRID1_OS4` (formerly misnamed `OS4_GRID5`), MIRI → `OS4_GRID3`. See
`scratch/build_large_psf.py` for the reference build script.

---

## Configurable parameters

| parameter | default | meaning |
|---|---|---|
| `min_area` | `1` | minimum hole area in `find_wht_holes` |
| `merge_radius` | `3` | morphological closing of nearby holes |
| `hole_dilate` | `2` | buffer ring thickness (pix) |
| `buffer` | `2.0` | added to `r_equiv` to get `r_in` of donut |
| `factor` | `2.5` | `r_out = factor × r_in` |
| `fwhm_pix` | `2.0` | floor on `r_out` (= `2 × fwhm_pix`) |
| `min_buffer_snr` | `200.0` | sat-vs-not-sat pre-filter |
| `max_shift_pix` | `3.0` | hard cap on cumulative iterative shift |
| `extended_max_data_to_model` | `1.15` | pedestal-switch threshold |
| `max_resid_frac` | `1.0` | bad-fit guard (no action above this) |
| `mode` | `"repair"` | or `"subtract"` |
| `psf_size_pix_subtract` | `min(200, native/2)` | subtract-cutout half (pix) |

---

## Output products

* `saturate_*.csv` (repair) / `subtract_*.csv` (subtract) — fit table:
  `id, yc, xc, r_equiv, r_in, r_out, amplitude, amp_err, chi2_red,
  n_pix, n_iter, shift_x, shift_y, significance, buffer_snr,
  flux_added, pedestal, fit_mode, data_to_model, ok, status`.
* `sci_repaired.fits` / `sci_subtracted.fits` — the modified science image.
* `wht_repaired.fits` / `wht_subtracted.fits` — the modified weight map.
* `repair_NNNN.png` / `subtract_NNNN.png` — per-source diagnostics
  (one for every source with `ok=True` plus a stub for fit-attempt
  failures).
