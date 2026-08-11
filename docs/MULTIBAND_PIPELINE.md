# Multi-band fitting: current support and upgrade path

Status note, 2026-08-11. Records what survives of sequential multi-image
fitting after the 2026-08 pipeline updates (template-prep refactor, stamps
output, `load_fit` save/restore), and what a full multi-band pipeline stream
would take.

## Current support

| Layer | Multi-band? | Notes |
|---|---|---|
| `Pipeline.run()` (API path) | yes | `for ifilt in range(1, len(images))` fits each band sequentially: shared hi-res templates built once (`_prepare_hi_templates`), then per band `_convolved_templates(ifilt)`, scene solve, catalog columns `flux_<i>/err_<i>`. Exercised by `tests/test_pipeline.py::test_pipeline_flux_recovery` (ref + 2 fitted images). |
| `RunConfig` / JSON stream | no (by design) | One `sci_lo/wht_lo/pattern_lo/filter_lo` slot per config. Multi-band = one config and one run per band (the 17 MINERVA configs). |
| `write_outputs` | band 1 only | Writes `residuals[0]` only (pre-existing) and `write_stamps()` at default `ifilt=1`. A second band would collide on `<name>_stamps.fits`. |
| `write_stamps(ifilt=k)` | per band, manual | Works for any band if given a distinct `path`; header records `IFILT`. Not called automatically for `k > 1`. |
| `load_fit` | band 1 only | Hard-codes single-slot lists: `all_templates = [templates]`, `residuals = [residual]`, `model_images = [...]` at index 0. Restoring `ifilt=2` would land that band at slot 0, breaking `source_products(ifilt=2)`. Only one residual file exists to read anyway. Internally consistent because `load_fit` requires a config-driven (single-band) run, but its `ifilt` parameter promises more than it delivers. |
| `source_products` / `show_sources` | yes post-`run()`, band 1 post-`load_fit` | Index `all_templates[ifilt-1]`, `model_images[ifilt-1]`; correct whenever those lists are filled per band. |

Per-band state the pipeline already keeps correctly in a live run:
`all_templates`, `all_scenes`, `residuals`, `model_images`,
`fit_bin_factors`, and the `flux_<i>` catalog columns. The upsample path
mutates `images[ifilt]`/`wcs[ifilt]` in place per band; memory therefore
scales with the number of bands held on the reference grid.

## Upgrade path to a full multi-band stream

Ordered so each step is independently shippable.

1. **Config**: give `RunConfig` a `bands` list of per-band dicts
   (`sci_lo, wht_lo, csv_lo, pattern_lo, filter_lo, psf_blur_fwhm,
   aperture_diam`). Scalar top-level fields remain valid and mean one band,
   so every existing config keeps working (normalize to a one-entry `bands`
   list in `__post_init__`).
2. **Caches**: suffix the per-band map paths with the filter name —
   `f_psf_lo(band)` → `<name>_psf_lo_<filter>.geojson`, same for
   `f_kernel`. Filter names are stable when the band list changes; indices
   are not. `build_psfs`/`build_kernels` loop over bands; the hi-res map
   stays shared.
3. **`load_data`**: read all lo images/weights, finish construction with
   `images=[hi, lo1, lo2, ...]`, `psfs=[prm_hi, prm_lo1, ...]`,
   `kernels=[None, kern1, ...]`. `run()` needs no change — the loop already
   iterates bands.
4. **Outputs**: `write_outputs` loops `ifilt`, writing
   `<name>_residual_<filter>.fits` and `<name>_stamps_<filter>.fits`.
   Catalog columns stay index-based (`flux_<i>`), so record the
   index-to-filter mapping in the fit-table header (`FILT<i> = <filter>`)
   to keep files and columns cross-referenced.
5. **`load_fit`**: fill band slots instead of overwriting slot 0 —
   pre-size `all_templates`/`residuals`/`model_images`/`fit_bin_factors`
   to `len(images) - 1` and assign at `ifilt - 1`; loop all bands by
   default, or restore a subset on request. Validate each stamps file's
   `IFILT`/filter against the config before accepting it.
6. **Tests**: extend `test_load_fit_restores_post_run_state` to a
   three-image run, asserting slot alignment (`source_products(ifilt=2)`
   after restore) and per-band stamp round-trips.

Open design question: whether one multi-band run beats N single-band runs
in practice. The bands share only the hi-res template extraction (~seconds
against minutes of drizzling/solving), while one config per band
parallelizes trivially across jobs and keeps failure domains small — the
CANFAR and MINERVA runs use exactly that pattern. The upgrade is therefore
about convenience and consistency of the saved state, not speed.
