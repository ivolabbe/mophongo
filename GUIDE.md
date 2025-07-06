# Implementation Guide

This document provides guidelines for developing the Standalone Photometry Pipeline.

## General Rules
- use poetry to maintain environment, keep pyproject.toml current, never directly edit poetry.lock
- Prefer using **numpy**, **scipy**, for numerical and scientific, and **astropy** and **photutils** for astronomical operations. Specifically use advanced photutils photometry and segmentation functionality where beneficial (e.g. photometry, psf matching, segmentation, wcs, cutouts, profiles). Dont reimplement unless necessary.
- Structure  project in a modular way to promote reusability and clear separation of concerns
- Organize code under the `mophongo` package with clear module boundaries.
- Use `@dataclass` for all structured data (e.g. `PSF`, `Template`)
- use object oriented design where abstraction is appropriate and makes the code easier to maintain / extend (e.g. for PSF)
- Keep functions pure when reasonable and document expected input shapes.
- Write unit tests alongside new functionality using `pytest`.

## Recommended Dataclasses
- `PSF`: parameters such as `size`, `fwhm_x`, `fwhm_y`, `beta`, and `theta`.
- `Template`: stores the PSF-matched cutout array and its bounding box.
- `Fit`: holds solver options like regularization strength and positivity enforcement.

Using dataclasses clarifies data flow and helps keep functions stateless.

