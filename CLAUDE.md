# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- **Install dependencies**: `poetry install`
- **Alternative installation**: `pip install -e .`
- **Run tests**: `pytest` (standard pytest runner)
- **Individual tests**: `pytest tests/test_<module>.py` or `pytest tests/test_<module>.py::test_function`

### Project Structure
This is a Python package for astronomical photometry processing, organized as:
- `src/mophongo/` - Main package source code
- `tests/` - Test suite (pytest-based)
- `examples/` - Jupyter notebooks demonstrating usage
- `data/` - Test data and PSF files for astronomical processing
- `legacy/` - Legacy IDL/Pro code for reference

## Architecture Overview

### Core Pipeline Flow
The photometry pipeline follows this sequence:
1. **Template Extraction** (`templates.py`) - Extract source templates from high-resolution detection image
2. **PSF Handling** (`psf.py`, `psf_map.py`) - Manage point spread functions and spatially-varying PSF maps
3. **Convolution/Matching** - Match PSFs between images using kernels
4. **Sparse Fitting** (`fit.py`) - Solve for source fluxes using sparse matrix methods
5. **Astrometric Correction** (`local_astrometry.py`) - Optional astrometric refinement

### Key Classes and Components

**Templates System** (`templates.py`):
- `Template` - Individual source template (extends `astropy.nddata.Cutout2D`)
- `Templates` - Collection manager with extraction, convolution, and downsampling

**Fitting Framework** (`fit.py`):
- `SparseFitter` - Main sparse matrix solver with configurable methods ('ata' or 'lo')
- `FitConfig` - Configuration dataclass controlling fitting behavior
- `GlobalAstroFitter` - Joint astrometry and photometry fitting

**PSF Management** (`psf.py`, `psf_map.py`):
- `PSF` - Point spread function with analytic and array-based creation
- `PSFRegionMap` - Spatially-varying PSF lookup system
- `DrizzlePSF` - JWST/HST-specific PSF handling with coordinate mapping

**Pipeline Orchestrator** (`pipeline.py`):
- `run()` - Main entry point taking images, segmap, catalog, PSFs
- Handles multi-resolution processing and template matching
- Memory-efficient processing with progress tracking

### Important Implementation Details

**Multi-Resolution Support**:
- Automatic binning factor detection from WCS (`utils.bin_factor_from_wcs`)
- Template and PSF downsampling for resolution matching
- Upsampling of lower-resolution images when needed

**Sparse Matrix Optimization**:
- Two solving methods: 'ata' (normal equations) and 'lo' (linear operator with iterative solvers)
- Configurable regularization and positivity constraints
- Fast FFT-based convolution for large templates (`config.fft_fast`)

**Astrometric Refinement**:
- Iterative astrometry fitting with configurable passes (`fit_astrometry_niter`)
- Joint or separate astrometry solving modes
- SNR-based source selection for astrometry

**Memory Management**:
- Template pruning based on weight maps
- In-place operations where possible
- Memory usage tracking throughout pipeline

### Testing Patterns
- Tests use `utils.make_simple_data()` for synthetic datasets
- Visual diagnostics saved to tmp directories for debugging
- Pytest with module-level skipping for data-dependent tests
- Integration tests with full pipeline workflows

### Data Handling
- FITS-based I/O with Astropy integration
- Segmentation maps for source identification
- Coordinate transformations via WCS
- Support for both synthetic and real astronomical data

## Development Notes

**Template System**: Templates maintain both original and cutout coordinate systems with careful slice bookkeeping for sparse matrix construction.

**PSF Matching**: Kernel computation uses Fourier-domain matching with Tukey windowing for stable deconvolution.

**Flux Errors**: Two error modes - simple diagonal errors and full covariance-based errors (`config.fit_covariances`).

**Multi-Component Fitting**: Support for adding PSF cores and color components for poorly-fit sources based on chi-squared thresholds.