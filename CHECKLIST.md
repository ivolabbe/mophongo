# Project Checklist

This checklist tracks tasks for building the Standalone Photometry Pipeline using Poetry and pytest.

## Setup
- [x] Initialize repository with `pyproject.toml` and Poetry
- [x] Create base package structure under `src/`
- [x] Add basic test suite with `pytest`

## Dependencies
- [x] Add `numpy`, `scipy`, and `astropy` to project dependencies
- [x] Run `poetry install` to install all dependencies

## assumptions input data
- [ ] input data are images + wcs that are multiextention fits files with first extention the SCI image, and second extention the ERR image
- [ ] input catalog is catalog of sources positions: id, ra, dec
- [ ] detection image, and associated segmentation map image, where each pixel can only belong to a source of a certain id.

## Core Modules
- [x] **PSF utilities** (`src/mophongo/psf.py`)
  - `moffat_psf` Generate Moffat PSF images (ellipticity/FWHM/beta parameters).
  - `psf_matching_kernel` to Compute convolution kernels to transform the high‑resolution PSF into the low‑resolution PSF (Fourier domain or direct numerical solution)
  - [x] Add methods to fit Moffat and Gaussian profiles to existing PSF arrays
- [x] **Template builder** (`src/mophongo/templates.py`)
  - `extract_templates` to create PSF-matched templates
  - Extract per-object cutouts from the high‑res image using the detection segmentation.
  - Normalize cutouts to unit flux and convolve each with the PSF kernel to produce a template in the low‑res pixel grid.
  - Store bounding box coordinates for later overlap calculations.
- [x] **Sparse fitter** (`src/mophongo/fit.py`)
  - Build sparse normal matrix AᵀA and vector Aᵀb using the templates and low‑res image (weights from inverse variance).
  - Solve for fluxes with scipy.sparse.linalg.cg (plus optional positivity and residual regularization).
  - Create the modeled low‑res image and residual map.
 - [x] **Pipeline orchestrator** (`src/mophongo/pipeline.py`)
  - `run_photometry` to tie all pieces together
  - don't implement source detection just yet: assume detection + segmentation image + catalog are available.
  - Load or receive arrays for the images, catalog, and PSFs.
  - Call template builder, construct sparse system, solve for fluxes, and return a table of measurements plus residuals.
  - [x] Propagate RMS images as weights to compute flux uncertainties
 - [x] **Simulation utilities for tests** (`tests/utils.py`)
  - [x] Create fake catalogs and images with Moffat sources of varying size and ellipticity. positions are ra,dec
  - Produce matching high‑res and low‑res PSFs, with low res PSF at least 5x high res PSF.
  - max 20 sources, max 400 x 400 pixel high resolution image
  - Convolve with a kernel derived from different PSFs to obtain the low‑resolution image and add Gaussian noise.
  - Run the pipeline with the known PSFs and verify recovered fluxes agree with input fluxes within ≈5%.
  - Check that the residual image contains only noise (no strong artifacts).
  - Test failure modes (e.g., negative flux regularization) on a subset of sources.
    
## Testing
 - [x] Add simulated data utilities in `tests/utils.py`
- [x] Create end-to-end tests in `tests/test_pipeline.py`
- [x] Run `pytest` to ensure all tests pass
- [x] Save diagnostic plot during pipeline test
- [x] Save diagnostic plots for PSF, fitter and template tests
- [x] Save output catalog to disk during pipeline test

- [x] Implement template extension methods (Moffat fit and PSF dilation)
- [x] Added selectable template extension method in `run_photometry` and consolidated analytic profiles
- [x] Benchmarked key pipeline steps in `tests/test_benchmark.py`
- [x] Introduced Cutout2D-based template extraction and normal matrix helpers
- [x] Renamed TemplateNew to Template and updated extraction defaults
