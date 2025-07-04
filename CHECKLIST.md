# Project Checklist

This checklist tracks tasks for building the Standalone Photometry Pipeline using Poetry and pytest.

## Setup
- [x] Initialize repository with `pyproject.toml` and Poetry
- [x] Create base package structure under `src/`
- [x] Add basic test suite with `pytest`

## Dependencies
- [ ] Add `numpy`, `scipy`, and `astropy` to project dependencies
- [ ] Add `scikit-image` for segmentation utilities
- [ ] Run `poetry install` to install all dependencies

## Core Modules
- [ ] **PSF utilities** (`src/mophongo/psf.py`)
  - `moffat_psf` to generate Moffat PSFs
  - `psf_matching_kernel` to compute convolution kernels
- [ ] **Template builder** (`src/mophongo/templates.py`)
  - `extract_templates` to create PSF-matched templates
- [ ] **Sparse fitter** (`src/mophongo/fit.py`)
  - Build sparse matrices and solve for fluxes
- [ ] **Pipeline orchestrator** (`src/mophongo/pipeline.py`)
  - `run_photometry` to tie all pieces together

## Testing
- [ ] Add simulated data utilities in `tests/utils.py`
- [ ] Create end-to-end tests in `tests/test_pipeline.py`
- [ ] Run `pytest` to ensure all tests pass

