# Astropy Ecosystem Bug Ledger

Date opened: 2026-05-03

This file tracks suspected or confirmed upstream issues in Astropy or closely
related Astropy ecosystem packages that affect Mophongo. These notes are for
later upstream reports. Each item should eventually get a minimal reproducer,
version matrix, expected behavior, observed behavior, and local workaround.

## 1. Photutils Centroid Failures On Large Images

Status: confirmed for `photutils.centroids.centroid_quadratic`; possible but
not yet isolated for Gaussian centroiding/fitting paths.

Observed behavior:

- `centroid_quadratic` can fail or return invalid centroids when fit directly
  on a large mosaic.
- The same source can centroid correctly when first cut out into a small local
  stamp and then centroided in local pixel coordinates.
- The suspected mechanism is numerical conditioning from fitting with large
  absolute pixel indices. This is already documented in `GUIDE.md`.
- A similar failure may affect Gaussian centroid/fitting paths when applied
  directly to large images, but this still needs a minimized reproducer before
  reporting.

Mophongo workaround:

- Do not run these centroid routines directly on full mosaics.
- Cut out a local stamp around the source first.
- Measure the centroid in stamp-local coordinates.
- Add the cutout origin back to recover global pixel coordinates.

Evidence to collect before reporting:

- Package versions: `astropy`, `photutils`, `numpy`, `scipy`, Python.
- Minimal synthetic image where full-image centroid fails but local-cutout
  centroid succeeds.
- Separate tests for `centroid_quadratic` and Gaussian centroid/fitting APIs.
- Failure mode: NaN, exception, biased centroid, or bad covariance/fit status.

## 2. `astropy.nddata.Cutout2D.shape_input` Is Not Reliable

Status: suspected upstream bug or API inconsistency; local workaround present.

Observed behavior:

- Mophongo's `Template` class, which subclasses `astropy.nddata.Cutout2D`,
  cannot rely on `Cutout2D.shape_input` being set to the parent input image
  shape in the way the pipeline needs.
- The current local workaround is in `src/mophongo/templates.py`:

```python
# @@@ bug in Cutout2D: shape_input is not set correctly
self.shape_input = data.shape
```

Expected behavior:

- For a cutout created from an input image `data`, `shape_input` should
  reliably refer to the full parent image shape, not the cutout shape or a
  stale/inconsistent value.

Why Mophongo cares:

- `Template.shape_input` is used to reconstruct full-size model images and to
  place convolved or transformed templates back onto the parent image grid.
- If this metadata is wrong, later template placement can become ambiguous or
  silently incorrect.

Evidence to collect before reporting:

- Minimal `Cutout2D` example showing `shape_input` differs from `data.shape`
  or changes unexpectedly under the relevant mode/size/WCS/subclass path.
- Test both odd and even cutout sizes.
- Test `mode="partial"` because that is the production template path.
- Test with and without WCS.
- Confirm behavior on current Astropy release and Astropy main.

Current local rule:

- Keep explicitly setting `self.shape_input = data.shape` in `Template` until a
  minimal upstream reproducer proves the issue fixed or clarifies the intended
  API contract.

