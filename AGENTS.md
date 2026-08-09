# AGENTS.md - Coding Agent Instructions

This file is the single source of agent-facing instructions for Mophongo.
`CLAUDE.md` should be a symlink to this file.

Mophongo is a scientific Python package for PSF modeling, image registration,
image simulation, and template-fitting photometry.

## Development Commands

- Install dependencies: `poetry install`
- Alternative editable install: `pip install -e .`
- Run tests: `poetry run pytest`
- Run one test file: `poetry run pytest tests/test_<module>.py`
- Run one test: `poetry run pytest tests/test_<module>.py::test_function`
- Add dependencies only with Poetry, for example `poetry add <package>`
- Never edit `poetry.lock` directly

## Required Reading

Before writing or modifying code:

1. Read [GUIDE.md](./GUIDE.md).
2. Check [STATUS.md](./STATUS.md) for recent/current implementation context.
3. Check [TODO.md](./TODO.md) for open future work that may overlap the task.

## Documentation Tracking

`STATUS.md` and `TODO.md` must be kept current. Every implementation turn
that changes behavior, layout, validation state, or future work should update
one or both files before finishing.

- Update `STATUS.md` when a feature, fix, validation run, or behavior/layout
  change is completed.
- Update the `Current Work` section in `STATUS.md` while actively working on a
  multi-step change.
- Update `TODO.md` when future desired features, checks, or investigations are
  added, completed, or clarified.

## Project Layout

- `src/mophongo/` - main package source code
- `tests/` - pytest-based test suite
- `examples/` - notebooks and runnable examples
- `data/` - test data and PSF files
- `docs/` - longer-form documentation
- `legacy/` - legacy IDL/Pro reference code
- `scratch/` - exploratory scripts and one-off diagnostics

## Implementation Workflow

- Add reusable code inside `src/mophongo/`.
- Keep scratch code in `scratch/` only when it is exploratory or not reusable.
- Avoid project-wide refactors unless specifically requested.
- Prefer existing package patterns and helper APIs over new ad hoc utilities.
- Use type hints and Google-style docstrings for new public functions/classes.
- Prefer pure functions where practical.
- Use `logging` for runtime reporting. Do not add new `print` calls in package
  code unless preserving an existing interface.
- Use pytest tests for added functionality.
- Put tests under `tests/`.
- Use temporary files/mocks for FITS I/O in tests when possible.
- Avoid network access or external resources in tests.

## Architecture Overview

The main photometry flow is:

1. Template extraction from a high-resolution detection image
   (`templates.py`).
2. PSF handling and spatial PSF lookup (`psf.py`, `psf_map.py`).
3. PSF matching / convolution kernels (`utils.py`, `psf.py`).
4. Sparse or scene-based fitting (`fit.py`, `scene.py`, `scene_fitter.py`).
5. Pipeline orchestration and diagnostics (`pipeline.py`).

Key components:

- `Template` and `Templates`: source cutout geometry, extraction,
  convolution, block projection, and template collections.
- `SparseFitter`, `Scene`, and `SceneFitter`: flux solving, scene grouping,
  optional astrometric shift blocks, and error estimates.
- `FitConfig`: fitting configuration.
- `PSF`, `DrizzlePSF`, and `PSFRegionMap`: analytic/array PSFs, drizzled JWST
  PSFs, and spatially varying PSF maps.
- `Catalog`: detection, segmentation maps, source catalogs, and catalog-owned
  segmentation helpers.
- `Pipeline`: main entry point for image/template fitting.
- `MockMosaic`: synthetic JWST-like mosaic generation.
- `PSFFactory` (`psf_factory.py`): telescope-backend registry and MJD-aware
  PSF generation.
- `saturate.py`: saturated-pixel repair (see Module Boundaries).
- `astrometry.py`, `jwst_psf.py`, `deblender.py`, `astro_fit.py`: astrometric
  corrections, JWST PSF utilities, deblending, and astrometry-aware fitting.

## Module Boundaries

Mophongo is being prepared for public release. Keep modules reusable and
separable. Each module should own its own data type and interact with other
modules through numpy arrays, astropy objects, Tables, dicts, or small
dataclasses.

Guiding rules:

- `saturate.py` is pure image-pixel repair. Inputs are `sci`, `wht`, `wcs`,
  and a `DrizzlePSF`. Outputs are a repaired image and flat fit table. It must
  not import segmentation maps, catalogs, or photometry concepts.
- `psf.py` / `DrizzlePSF` own drizzling, ePSFs, PSF profiles, and kernel
  diagnostics. Do not add wrappers that import fitting, saturation, or catalog
  domains.
- `catalog.py` owns segmentation maps, source catalogs, and helpers that act on
  segmentation/catalog state. It can consume flat tables from preprocessing,
  but preprocessing must not import catalog.
- `fit.py`, `templates.py`, `scene.py`, `scene_fitter.py`, and `pipeline.py`
  own the photometry pipeline and should not reach into preprocessing
  utilities.
- `verification.py` owns reusable validation and diagnostic helpers. Keep
  survey/instrument-specific orchestration in examples or scripts.

Concrete rule: if module A needs information that module B owns, prefer passing
the result through a flat structure B already exposes rather than importing A
into B.

## Implementation Details To Preserve

- Templates maintain original-image and cutout coordinate systems. Be careful
  with slices and origin conventions.
- Multi-resolution fitting uses WCS-derived bin factors. When upsampling a
  lower-resolution image, science pixels are flux-conserving block-replicated
  and inverse variance is copied then multiplied by `factor**2`.
- Pipeline-facing PSF matching uses unit-sum PSF shapes. Finite PSF stamp sums
  are filter-level throughput metadata for total-flux reporting; do not
  silently renormalize native PSF stamps and lose that information.
- `Pipeline.run()` writes raw fitted template amplitudes as `flux_<i>` and
  throughput-corrected totals as `flux_<i>_total`.
- Use standard PSF diagnostic helpers, especially
  `PSF.optimize_matching_kernel_regularization(..., diagnostic_path=...)`,
  instead of reinventing PSF diagnostic figures.

## Testing And Verification

- Run focused tests for the modules you touched.
- For shared pipeline behavior, run at least the relevant `tests/test_pipeline.py`
  tests.
- For PR preparation or large changes, run `poetry run pytest`.
- When adding diagnostics, make outputs reproducible and write tests for the
  data products or helper behavior, not just the image file existence.
- Long-form reports (`.tex`/`.md` compiled to PDF): plain academic prose, no
  AI narrative tics; run a humanizer pass before finalizing/compiling.

## Pull Request Preparation

Before preparing a PR:

- Ensure relevant tests pass; use full `poetry run pytest` when feasible.
- Update `STATUS.md` and `TODO.md` as appropriate.
- Verify changes are scoped to relevant modules.
- Do not refactor unrelated modules unless requested.

PR body should include:

- Summary of logic
- Modules modified or added
- Validation/tests run
- Links or references to relevant status/design notes
