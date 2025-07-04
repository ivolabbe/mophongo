# Implementation Guide

This document provides guidelines for developing the Standalone Photometry Pipeline.

## General Rules
- Prefer using **numpy**, **scipy**, and **astropy** for numerical and astronomical operations.
- Organize code under the `mophongo` package with clear module boundaries.
- Keep functions pure when reasonable and document expected input shapes.
- Write unit tests alongside new functionality using `pytest`.

## Recommended Dataclasses
- `PSF`: parameters such as `size`, `fwhm_x`, `fwhm_y`, `beta`, and `theta`.
- `Template`: stores the PSF-matched cutout array and its bounding box.
- `FitConfig`: holds solver options like regularization strength and positivity enforcement.

Using dataclasses clarifies data flow and helps keep functions stateless.

