# MINERVA all-field broadband-SED stack

This directory holds the generated products from
`examples/minerva/plot_uds_sed_stack.py`. The large FITS and rendered figures
are intentionally ignored by Git; this README and `.gitignore` are tracked.

Run from the repository root:

```bash
poetry run python examples/minerva/plot_uds_sed_stack.py
```

The default field manifest resolves one distinct current SUPER catalog for
COSMOS (`n3.0_m3.0_v1.0.1`), EGS (`n2.0_m2.1_v1.3.1`), and UDS
(`n3.0_m3.1_v1.2.1`). Their union contains 41 HST/JWST bands, including all
seven available MIRI bands from F560W through F2100W. The script requires the
exact matching EAzY `zout.fits` for every field; it verifies row count, IDs,
and sky positions before using `z_phot`, with positive catalog `z_spec` values
taking precedence. Per-filter run configs that share a catalog do not duplicate
galaxies.

## Construction

- In each field, start with `use_phot > 0`, finite positive redshift, and valid per-band
  flux/error/coverage. There is no per-band flux or S/N cut: finite zero and
  negative fluxes enter both the stack sum and its valid-pixel count. MIRI
  persistence, edge, and high-scale-factor flags are rejected per measurement.
- Convert catalog F-nu to a quantity proportional to F-lambda using
  `F_lambda ∝ F_nu / lambda_pivot^2`. These are the released catalog fluxes;
  EAzY fitting zeropoint adjustments are not applied.
- Normalize each galaxy at rest 5000 Angstrom with an inverse-variance
  weighted local linear fit of F-lambda versus log wavelength. The fit uses
  the nearest three valid bands and must bracket 5000 Angstrom; when only two
  are available it reduces exactly to interpolation. The default requires
  only this per-galaxy divisor—not the other SED pixels—to be positive and
  measured at S/N >= 5, because division by a noise-consistent-with-zero
  normalization creates an unstable flux ratio.
- Paint each measurement as a top hat over the actual main-lobe
  half-maximum interval in `uds_sed_filters.ecsv`. Uncovered pixels stay NaN;
  overlapping filters are averaged within a galaxy.
- Build a separate first-order cell reconstruction for display and analysis.
  Each galaxy is linear in signed F-lambda versus log wavelength between valid
  pivots only when their half-maximum supports form one connected coverage
  component. Component ends are held constant only out to the physical filter
  edge; real gaps and missing-band breaks remain NaN. Nearly coincident pivots
  (within five percent of the narrower filter width) are arithmetic-averaged
  once per galaxy. This is not post-stack image smoothing.
- Pad absent field-specific filters as invalid, concatenate individual galaxies,
  bin at constant `Delta log(1+z) = log(1.05)`, and take an
  unweighted nanmean in linear normalized F-lambda. Pixels need at least five
  galaxies and one percent of the galaxies in their redshift bin. The combined
  image is therefore galaxy-weighted, not an equal-weight mean of field means.

The rest-frame stack uses 1,992 fixed 100 Angstrom bins from 0.08 to 20 micron.
The observed-frame stack instead uses the actual blue and red half-maximum
boundaries of the 41 filters, so each isolated pixel spans a complete filter
width. Where filters overlap, a boundary splits the shared interval so the
valid fluxes can still be averaged once per galaxy. The plots use the exact
physical bin edges on a logarithmic display axis. An exact filter-interval
sweep avoids allocating a source-by-wavelength cube. Products include a
comparison PNG/PDF, separate wide rest- and observed-frame PNGs, a
sampling-depth PNG, an interpolated comparison PNG/PDF, a machine-readable
FITS file with combined and per-field
mean/count images, field/filter provenance tables, all coordinate tables, and
a JSON run summary. In the observed-frame plot the filter footprints remain
fixed while rest-frame feature guides move diagonally with redshift.

The interpolated product retains the 100-Angstrom rest grid and evaluates the
observed model on a log-wavelength grid no coarser than 0.0025 dex, augmented
with every filter edge and pivot. Its third panel is explicitly display-only:
it subtracts a masked 0.12-dex continuum from an asinh-transformed observed
stack and uses light, gap-preserving interpolation between redshift-bin
centers. This contrast view makes broad feature tracks easier to see without
altering the mean/count arrays or filling true wavelength gaps.

These are broadband glyphs, not recovered spectra. Apparent narrow
features can be caused by filter boundaries, photo-z structure, selection,
or small-number statistics; consult the contributor-count image and FITS
count planes before interpreting them astrophysically.
