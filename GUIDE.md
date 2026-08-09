# Implementation Guide

This document provides guidelines for developing the Standalone Photometry Pipeline.

## General Rules
- use poetry to maintain environment, keep pyproject.toml current, never directly edit poetry.lock
- Prefer using the following packages:
   - **numpy**, **scipy**, for general numerical and scientific 
   - **astropy** and **photutils** for astronomical operations. Specifically use advanced photutils photometry and segmentation functionality where beneficial (e.g. , units). Dont reimplement unless necessary
- Structure  project in a modular way to promote reusability and clear separation of concerns
- Organize code under the `dotfit` package with clear module boundaries.
- Use `@dataclass` for  structured data
- use object oriented design where abstraction is appropriate and makes the code easier to maintain / extend 
- Keep functions pure when reasonable and document expected input shapes.
- Write unit tests alongside new functionality using `pytest` and produce an insightful diagnostic image / figure

## Setup
- [x] Initialize repository with `pyproject.toml` and Poetry
- [x] Create base package structure under `src/`
- [x] Add basic test suite with `pytest`

## Dependencies
- [x] Add `numpy`, `scipy`, and `astropy` to project dependencies
- [x] Run `poetry install` to install all dependencies
- [x] Added `nbformat` for generating example notebooks

## assumptions input data
- [x] input data are images + wcs, and weights that are proportional to variance
- [x] input catalog is catalog of sources positions: id, ra, dec
- [x] detection image, and associated segmentation map image, where each pixel can only belong to a source of a certain id.

## PSF Shape vs Throughput Convention

Mophongo separates PSF morphology from finite-stamp throughput. A realistic
pixelated PSF stamp should be allowed to sum below one because flux exists
outside the modeled support. That finite-stamp sum is the PSF throughput
metadata, not part of the fitting shape basis.

- Use unit-sum `psf_shape = psf_native / psf_native.sum()` for template
  extension, PSF matching kernels, and template convolution.
- Preserve `psf_throughput = psf_native.sum()` as metadata for flux reporting
  and diagnostics.
- The template-fitting pipeline normalizes extracted/extended templates to
  unit sum. Fitted amplitudes are therefore modeled-stamp fluxes. Convert to a
  total-flux estimate with `flux_total = flux_model / psf_throughput` when the
  science image or truth catalog uses native finite-support PSFs.
- `Pipeline.run()` keeps the raw fitted model amplitudes in `flux_<i>` and now
  also writes `throughput_<i>`, `flux_<i>_total`, `err_<i>_total`, and
  `err_pred_<i>_total`. Use one average throughput correction per filter: PSF
  cores may vary spatially, but the missing far-wing flux is treated as a
  filter-level correction. Pass these values to `Pipeline(...,
  psf_throughputs=[...])`; do not store or apply region-dependent throughput
  corrections.
- `psf_wings` and `psf_model` template extension apply to catalog deblend
  children by default. If a validation run needs the older behavior of leaving
  deblended child templates unextended, set
  `FitConfig(skip_template_extension_for_deblended=True)`.
- The low-level `mophongo.utils.matching_kernel` function preserves whatever
  input sums it is given. Pipeline-facing callers should pass unit-sum PSF
  shapes unless they intentionally want a throughput-carrying diagnostic
  kernel.
- When scanning scalar PSF-matching regularization (`wiener`, `tikhonov`,
  `forward`), use `PSF.optimize_matching_kernel_regularization(...,
  diagnostic_path=...)` for the standard `diagnostic_<method>.png` output
  rather than creating ad hoc PSF diagnostic figures in scripts. The standard
  scalar scan range is `lambda=1e-6..0.1`.

See `PSF_SHAPE_THROUGHPUT_CONVENTION.md` for the full convention.

## MockMosaic Weight Convention

MockMosaic `_wht.fits` products must be actual per-pixel inverse variance
maps by default: `1 / sqrt(wht)` is the RMS of the pixel noise injected into
the corresponding `_sci.fits` image. The weight generation includes exposure
count, output-pixel area, and square-kernel drizzle correction factors. Real
mosaic `wht` maps that use another convention should be converted with the
per-filter calibrated `DEFAULT_WHT_CALIB` scalars before being treated as
inverse variance.

When the pipeline upsamples a lower-resolution image onto a finer reference
grid, the science image is block-replicated with flux conservation. The inverse
variance map must not use flux-conserving replication: each native weight must
be copied to the subpixels and then multiplied by `factor**2` so the native
chi-square and WHT-derived template flux errors are preserved.

## Known upstream bugs / workarounds

### `photutils.centroids.centroid_quadratic` — numerical conditioning on large mosaics

`centroid_quadratic(data, xpeak=..., ypeak=...)` builds its 2-D quadratic
design matrix using **absolute** pixel indices (photutils ≤ 3.0,
`photutils/centroids/core.py` ~L245). For a source at `(x, y) ≈ (1e4, 1e4)`
the matrix gets an `x²` column of order `10⁸` alongside a column of ones —
condition number blows up, `np.linalg.lstsq` silently returns garbage, the
downstream `det = 4·c20·c02 − c11²` check fails, and the function returns
`[nan, nan]` on perfectly clean, bright stars.

**Symptom:** NaN centroid on full UDS / MINERVA mosaics (~17280×12672) even
though the same pixel values fit fine when passed as a small cropped patch.

**Fix applied in `DrizzlePSF.get_driz_cutout`:** before calling
`centroid_quadratic`, crop a local neighborhood
(`half = max(search_boxsize, fit_boxsize) + 2`) around the WCS-predicted
peak and pass coordinates relative to the crop, then add the offset back.
On NaN return (e.g. masked-pixel patch), fall back to the WCS position
with a logged warning.

**Upstream status:** still present on `photutils@main` as of 2026-04; not
fixed by upgrading. The bug is independent of photutils version.

### photutils version in environment vs `pyproject.toml`

`pyproject.toml` requests `photutils = "^2.2.0"`, but the active
environment has `photutils 1.12.0` and upstream is `3.0.0`. Declared
transitive pins are loose — `drizzlepac 3.9.1`, `jwst 1.16.1`, `stpsf`
all just ask for `photutils >= 1.10` (see the poetry.lock `drizzlepac`
entry) — but the *real* ceiling is `drizzlepac`, not numpy or jwst.

`drizzlepac` relies on several **private** photutils internals. This is
self-documented in the package itself:

> `/opt/miniconda3/lib/python3.12/site-packages/drizzlepac/haputils/_detection_utils.py`:
> *"This is a copy of photutils private functions, which have been
> refactored. Use of these tools can likely be removed after drizzlepac
> requires photutils >= 1.2.0."*

When photutils refactors its private detection/background helpers
(anything underscore-prefixed and any reshuffling in `photutils.detection`
/ `photutils.background`), drizzlepac's import chain snaps. That's why
upgrading photutils past ~1.13 in practice breaks `from drizzlepac import
adrizzle` at import time, even though the solver resolves the version
graph. The photutils APIs *we* use (`centroid_quadratic`,
`create_matching_kernel`, `STDPSFGrid`, `SplitCosineBellWindow`,
`EPSFModel`) are unchanged 1.x → 3.x, so staying on 1.12 is not a
functional limitation for mophongo — only an upstream compatibility trap.

The `jwst`/`numpy<2` and `drizzlepac`/`photutils-private` pins form the
two pinch points; either must be resolved upstream before the env can
move forward.

## Future upgrades

### Environment upgrade: photutils 1.12 → 2.2, drizzlepac 3.9.1 → 3.10.0, jwst 1.16.1 → ≥1.18

Both pinch points described above are resolved in published releases as of
2026-04. When next touching the env, upgrade the four packages together —
individually is not safe (the solver lets drizzlepac 3.9.1 coexist with
newer photutils but runtime imports fail).

**Target versions** (all stable on PyPI):

| package | from | to | why |
|---|---|---|---|
| `drizzlepac` | 3.9.1 | **3.10.0** (2025-07-14) | declares `photutils>=2.0.0`, `numpy>2.0`. PR #1934 ("Photutils deprecation updates Round 2") did the real rework. The `_detection_utils.py` header still has the stale "copy of private functions" comment, but its imports are now all public APIs. |
| `jwst` | 1.16.1 | **1.20.2** *or* **2.0.0** (2026-04-13) | jwst **1.18.0** is the first release to drop `numpy<2`. 2.0.0 also caps `photutils<3`, which fits our target. |
| `photutils` | 1.12.0 | **2.2.x** | satisfies `pyproject.toml`'s existing `^2.2.0`; stays below the `<3` cap from jwst 2.0.0. |
| `numpy` | 2.2.6 | keep | unblocked by jwst ≥1.18. |
| Python | 3.12 | 3.12 (3.11–3.13 OK) | drizzlepac 3.10.0 dropped 3.10. |

**How to execute** (do this in a **throw-away venv / branch first**, not the
active env):

```bash
# new venv
python3.12 -m venv /tmp/mophongo-upgrade && source /tmp/mophongo-upgrade/bin/activate
pip install 'drizzlepac>=3.10.0' 'jwst>=1.20,<3' 'photutils>=2.2,<3' \
            'numpy>=2.2,<3' 'stpsf>=2.1,<3' -e /path/to/mophongo
# smoke tests
python -c "from drizzlepac import adrizzle; import photutils, jwst, numpy; \
           print(drizzlepac.__version__, photutils.__version__, jwst.__version__, numpy.__version__)"
python -m pytest tests/test_mock_mosaic.py -q
python -m pytest tests/test_psf.py -q        # if exists
```

**Mophongo code risk — low but with one audit item.** Public photutils APIs we
use (`centroid_quadratic`, `create_matching_kernel`, `STDPSFGrid`,
`GriddedPSFModel`, `SegmentationImage`, `SourceCatalog`, `MADStdBackgroundRMS`,
`deblend_sources`, `detect_sources`, `aperture_photometry`, `TukeyWindow`,
`SplitCosineBellWindow`, `EPSFModel`, `RadialProfile`, `CurveOfGrowth`) are
stable 1.x → 2.x. The main 1.x → 2.x break was removal of the old
`BasicPSFPhotometry` / `IterativelySubtractedPSFPhotometry` classes, which
mophongo does **not** use.

**Fragile spots to smoke-test** (these use `_private` photutils APIs — valid
on 2.x *main* today but with no stability guarantee across point releases):

| file | line | import |
|---|---|---|
| `src/mophongo/catalog.py` | 22 | `photutils.segmentation.catalog.DEFAULT_COLUMNS` |

(The three private-API imports in `photutils_deblend.py` are gone: that module
was deleted in the 2026-08 cleanup, leaving `DEFAULT_COLUMNS` as the only
private photutils dependency.)

Worth a smoke test right after upgrade; if any break, either vendor the
private helper (copy it into `mophongo` with a note) or rewrite against the
nearest public API. Do NOT ship the upgrade if these imports fail silently
at import time.

**Things unchanged by the upgrade:**

- The `centroid_quadratic` absolute-pixel-indexing bug (previous section) is
  still present on photutils main. Our `DrizzlePSF.get_driz_cutout`
  crop-first workaround stays in place.
- The `DrizzlePSF` hard-coded `oversample=4` assumption (see
  `make_stpsfs.ipynb` caveat) is orthogonal to this upgrade.

**Upstream issues to revisit** when acting on this:

- drizzlepac — confirm 3.11.0 has released (2026-02 was still rc).
- jwst — confirm stpsf's photutils bound still permits 2.2 on the target
  jwst release (stpsf 2.x currently requires photutils, no upper cap).
- grizli (if added as a dep later) — check its photutils bound.
