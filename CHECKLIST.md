# Project Checklist

This checklist tracks tasks for building the photometry pipeline using Poetry and pytest.

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

## Core Modules
- [x] **PSF utilities** (`src/mophongo/psf.py`)
  - [x] `moffat_psf` Generate Moffat PSF images (ellipticity/FWHM/beta parameters).
  - [x] `matching_kernel` to Compute convolution kernels to transform the high‑resolution PSF into the low‑resolution PSF (Fourier domain or direct numerical solution)
  - [x] Added `recenter` option to `psf_matching_kernel` to shift kernels to their centroid
  - [x] Add methods to fit Moffat and Gaussian profiles to existing PSF arrays
  - [x] Added `PSF.delta` for symmetric delta-function PSFs
  - [x] Added `PSF.from_star` constructor for extracting PSFs from images
  - [x] Added `PSF.gaussian_matching_kernel` and `DrizzlePSF.register`
  - [x] Added `matching_kernel_basis` with Gauss–Hermite and multi-Gaussian basis sets
  - [x] Added `CircularApertureProfile` utility for radial profile and curve of growth
  - [x] Implement JWST STDPSF extension utility for STPSF / Webb PSF
  - [x] Implement drizzling PSF
  - [x] Build PSF region map from exposure footprints
  - [x] Add PA-based coarsening option to PSFRegionMap
  - [x] Added spatially varying kernel support in `run` and template convolution
  - [x] Implemented basic `Catalog` for source detection
  - [x] Added configurable detection parameters in `Catalog`
  - [x] Implemented star finder in Catalog
- [x] **Template builder** (`src/mophongo/templates.py`)
  - [x] `extract_templates` to create PSF-matched templates
  - [x] Extract per-object cutouts from the high‑res image using the detection segmentation.
  - [x] Normalize cutouts to unit flux and convolve each with the PSF kernel to produce a template in the low‑res pixel grid.
  - [x] Store bounding box coordinates for later overlap calculations.
  - [x] Introduced Cutout2D-based template extraction and normal matrix helpers
- [x] **Sparse fitter** (`src/mophongo/fit.py`)
  - Build sparse normal matrix AᵀA and vector Aᵀb using the templates and low‑res image (weights from inverse variance).
  - Solve for fluxes with scipy.sparse.linalg.cg (plus optional positivity and residual regularization).
  - Create the modeled low‑res image and residual map.
  - [x] Added GlobalAstroFitter for astrometric correction
  - [x] Added polynomial-based local astrometric correction
  - [x] Added safeguards against singular normal matrices
- [x] Added Gaussian-process-based local astrometric correction
- [x] Introduced `AstroCorrect` for pluggable local astrometry models
- [x] Merged astrometry modules and added `AstroMap` for image-to-image shift mapping
- [x] Removed deprecated `fit_astrometry` flag in `FitConfig`; use `fit_astrometry_niter` only
- [x] Added ILU preconditioner and SuperLU-based flux error estimation with Hutchinson fallback
- [x] Added LSQR-based matrix-free solver (`solve_lo`)
- [ ] Deduplicate templates using weighted overlap cosine similarity
- [x] Consolidated flux and RMS estimation into parent `SparseFitter`
- [x] Added STRtree-based normal matrix builder (`build_normal_tree`)
 - [x] Removed deprecated `fit_astrometry` flag in `FitConfig`; use `fit_astrometry_niter` only
  - [x] Added ILU preconditioner and SuperLU-based flux error estimation with Hutchinson fallback
  - [x] Added LSQR-based matrix-free solver (`solve_lo`)
  - [ ] Deduplicate templates using weighted overlap cosine similarity
  - [x] Consolidated flux and RMS estimation into parent `SparseFitter`
  - [x] Added STRtree-based normal matrix builder (`build_normal_tree`)
  - [x] Added component-wise CG solver using STRtree groups
  - [x] Added component-wise solver with shift blocks
  - [x] Whitened component solver with sparse Cholesky preconditioner
- [x] **Pipeline orchestrator** (`src/mophongo/pipeline.py`)
  - [x] `run` to tie all pieces together
  - [x] don't implement source detection just yet: assume detection + segmentation image + catalog are available.
  - [x] Load or receive arrays for the images, catalog, and PSFs.
  - [x] Call template builder, construct sparse system, solve for fluxes, and return a table of measurements plus residuals.
  - [x] Propagate RMS images as weights to compute flux uncertainties
  - [x] Prune templates lacking weight overlap before convolution
  - [x] Enabled template deduplication after extraction
  - [x] Added multi-template second pass for poor-fit sources
  - [x] Added integer-factor multi-resolution support with template and kernel downsampling
  - [x] Block templates and PSFs before convolution with `block_reduce` and centroid-preserving PSF shifts
  - [x] Downsample templates and kernels in the pipeline prior to convolution to avoid per-source PSF rebinning
  - [x] Introduced `Pipeline` class to persist images and fit results
- [x] **Simulation utilities for tests** (`tests/utils.py`)
  - [x] Create fake catalogs and images with Moffat sources of varying size and ellipticity. positions are ra,dec
  - [x] Produce matching high‑res and low‑res PSFs, with low res PSF at least 5x high res PSF.
  - [x] max 50 sources, max 300 x 300 pixel high resolution image
  - [x] Convolve with a kernel derived from different PSFs to obtain the low‑resolution image and add Gaussian noise.
  - [x] Run the pipeline with the known PSFs and verify recovered fluxes agree with input fluxes within ≈5%.
  - [x] Check that the residual image contains only noise (no strong artifacts).
  - [x] Test failure modes (e.g., negative flux regularization) on a subset of sources.
  - [x] Add simulated data utilities in `tests/utils.py`  
  - [x] Create end-to-end tests in `tests/test_pipeline.py`
    
## Testing
- [x] Run `pytest` to ensure all tests pass
- [x] Save diagnostic plot during pipeline test
- [x] Save diagnostic plots for PSF, fitter and template tests
- [x] Save output catalog to disk during pipeline test
- [x] Benchmarked key pipeline steps in `tests/test_benchmark.py`

## TODO
- [ ] scan for bug fixes / robustness improvements
  - [x] align PSF components to fractional template centers
  - [ ] automated way of determining optimal convolution kernels for PSF  
- [ ] storage
  - [ ] best way to store intermediate results
  - [ ] "drop" image
- [ ] templates
  - [ ] test and validate fitting in downsampled space
  - [ ] profiles of low SNR objects -> asymptotically to psf
- [ ] background options
  - [ ] global background fit
  - [ ] background per stamp
- [ ] diagnostics
  - [ ] standard diagnostic view of fit result for object
- [ ] validate output catalogs on MIRI data
  - [ ] color color, color mag
  - [ ] SEDs of stars, photo-z
  - [ ] add in residuals in core for improved flux measurements (shift / psf errors)
- [ ] investigate blending in detection image
- [ ] Investigate template extension methods (Moffat fit and PSF dilation)
- [ ] End-to-end test with realistic mosaic data using `make_mosaic_dataset`
- [ ] Profiling speed + memory usage
- [ ] optimizations
  - [x] adaptive kernel size depending on SNR
  - [x] preconditioning matrix
- [ ] strong residuals
  - [ ] handle saturated stars in 444 -> catalog pre pass detection
  - [ ] fit as PSF both 444, 770, fit for centroid, mask center
- [ ]  wavelength dependent morphology: only where residuals are significant.
  - [ ] Add point source, if PSF not given start with marginally sampled Gaussian?  
  - [ ] add second bluer band
- [ ] refactoring for readibility and modularity
  - [ ] split off PSF map / drizzle PSF / PSFs module, make submodule
  - [ ] split off real data as submodule?
  - [ ] other code review, misc refactoring, consolidation
  - [ ] remove unused modules, orphan code
