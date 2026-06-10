# Status

This file records completed implementations, validation runs, and the current work state.

## Current Work
- [x] Consolidated agent-facing instructions into `AGENTS.md` and split
  `CHECKLIST.md` into `STATUS.md` and `TODO.md`.
- [x] Consolidate reusable verification logic from
  `scratch/run_realistic_mosaic_two_detector_extensions.py` into package code
  and validate `examples/verify_pipeline.ipynb`.
- [x] Updated template extension so `psf_wings`/`psf_model` extend deblended
  child templates by default, with
  `FitConfig(skip_template_extension_for_deblended=True)` as the opt-in
  non-extension behavior.
- [x] Added and ran a single-source high-SNR PSF/template mismatch diagnostic
  in `scratch/debug_single_source_psf_mismatch.py`. The isolated F444W/F770W
  run shows the central residual is present in `truth - model`, persists with
  noiseless truth-image inputs, and is dominated by a small effective centroid
  mismatch between the F444W-derived model and the injected F770W truth rather
  than by noise or the 0.08 arcsec F770W Gaussian blur term.
- [x] Removed the centroid-based `ndimage.shift` recentering from
  `MockMosaic.inject_point_sources`; source phases now come only from the
  requested sky position and the DrizzlePSF cutout WCS. Also removed the
  artificial verification `kernel_grid_nside` subdivision path because
  `PSFRegionMap` already encodes overlap/rotation regions.
- [x] Found and definitively tested the root cause of the bright-source F770W
  residual dipoles in `examples/verify_pipeline_realistic_out`: the stored
  mock predates the recentering removal, and the old COM recentering placed
  painted truth sources at a filter-dependent COM phase while the
  template/kernel model carries the natural drizzled WCS phase (coherent
  sub-pixel offset -> dipoles). Full diagnosis chain and metrics in
  `scratch/dipole_rootcause/README.md`.
- [x] Fixed a second injection bug introduced with even-parity PSF stamps:
  `inject_point_sources` pasted at `round(x)-size//2` while the
  `get_psf_radec` Cutout2D origin is `ceil(x-size/2)`, shifting half the
  painted sources by exactly 1 pixel per axis. Paste origins now use the
  Cutout2D convention. Fresh end-to-end verification: bright-source residual
  effective shifts drop from 0.05-0.6 to <=0.002 native pixels, residuals
  noise-like with no dipole term. `examples/verify_pipeline.ipynb` outputs
  must be regenerated.
- [x] Diagnosed the scene-19/39 residual dipoles in the rerun verification:
  per-scene shifts are constant (order-0) and consistent across scenes except
  scenes that had not converged — the linearized astrometric shift solve only
  captures part of a ~2.5 fit-pixel offset per pass and the loop ran a fixed
  `fit_astrometry_niter=2` passes. Solution: convergence-based early stopping
  in the scene iteration loop; `fit_astrometry_niter` is now the maximum
  number of passes (default 5) and `FitConfig.astrom_shift_tol` (default
  0.05 fit pix) stops the loop once the largest per-template shift increment
  of a pass falls below it. With this, 37 of 40 verification scenes recover
  the injected F770W shift to within ±0.03 fit pixels.
- [ ] Remaining: 3 of 40 verification scenes fit shifts 0.1-0.34 fit pixels
  off the injected value (scene ids 16/17/38 in the current run). Explored
  and ruled out, with diagnostic scripts in `scratch/dipole_rootcause/`:
  - iteration count: increments decay geometrically to a stationary point at
    the biased value (`scene_shift_introspection.py`, 8 passes);
  - faint templates biasing the shift blocks: `snr_thresh_astrom=15` gives
    identical outliers;
  - cross-scene flux contamination: chi2 scan on cleaned data (all other
    scene models subtracted) shows the same preference
    (`scene38_chi2_scan.py`);
  - kernel-region differences: outlier and converged scenes share regions;
  - local mock painting errors: painted truth positions in the outlier scene
    area are exact to <=0.05 pix;
  - cumulative spline smoothing from per-pass re-shifting: fixed in
    `apply_template_shifts` (shift-from-original), outliers unchanged.
  Findings: scene 17's bias comes from a blend flux swap (ids 503/614); the
  others are dominated by a single bright extended source (id 402) whose
  exact chi2 prefers an even larger shift than injected
  (`scene38_member_scan.py`) — template-shape/shift coupling, open issue
  tracked in TODO.md.
- [x] Added a truth-match position guard to the verification scenario:
  `segment_weighted_positions` measures the flux-weighted position of each
  source's segment on the detection image; rows with offset to the truth
  position above `max_match_offset_pix` (default 3 fit pix) get
  `position_matched=False` in the source table and are excluded from the
  flux-recovery plots and summary statistics
  (`n_position_mismatched` reported in the summary). Catches truth ids whose
  segment actually contains a different source.
- [x] Silenced per-stamp drizzlepac/stwcs INFO and "points were outside the
  output image" log spam during `DrizzlePSF.get_psf` via a scoped
  `_quiet_drizzle` context manager; the messages are expected per-frame
  chatter because the evaluated ePSF input grid is larger than the output
  cutout.
- [x] Changed `remap_detection_to_truth` to assign blended (multi-truth)
  detection labels to the brightest truth member instead of catalog order;
  the non-owning members keep the 3x3 fallback stamp. Fixes flux swaps where
  a faint neighbour's template inherited a bright source's blended segment
  (e.g. ids 670/622: ratio 0.12/38 -> 0.98/0.13).
- [x] Changed `Templates.apply_template_shifts` to interpolate from the stored
  original template data with the accumulated total shift instead of
  re-shifting already-shifted data (one cubic-spline interpolation regardless
  of astrometry pass count).
- [x] Made scene diagnostic PNGs show the global residual: `Scene.plot` takes
  `residual_image` (full-frame fit-grid residual with all scene models
  subtracted) and `save_scene_diagnostics` passes `pipe.residuals`. The old
  panel used `Scene.residual()` (own-scene model only, neighbor segments
  masked), which left other scenes' sources in the image — measured rms 0.5-0.7
  vs 0.08-0.09 for the true global residual, explaining why scene PNGs looked
  different from `f770w_residual.fits`. Scene shift annotation now reports
  median (dx, dy) instead of max |dr|.
- [x] Made PSF-wing template extension background-aware by default:
  `extract_templates` now records the dilated segmap on `Templates`, and
  `extend_with_psf_wings` fills only zero template pixels that are segmap
  background (or own-segment), leaving pixels owned by blended neighbours at
  zero. New `FitConfig.extend_wings_background_only=True` flag (set False for
  the old fill-everything behavior); per-template `extension_blocked_sum`
  metadata; regression test in `tests/test_template_convolution.py`.


## Past Work
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
- [x] Added static utilities in `AstroCorrect` for applying stored template shifts and building polynomial predictors
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
- [x] Renamed component terminology to scene and centralized whitening in scene solver
- [x] Introduced stateless `SceneFitter` and `Scene` utilities
- [x] Fixed alpha0 scaling and Cholesky whitening in scene solver
- [x] Fixed scene-solver flux-only path and `fit_astrometry_niter=0` handling
- [x] Documented scene-solver flux regularization bug in `FLUXBUG.md`
- [x] Reran mock validation with explicit `reg_flux=0.0` and `reg_astrom=0.0`
- [x] Hardened scene-solver flux and astrometric regularization against non-finite diagonals
- [x] Renamed photometric regularization config from `reg` to `reg_flux`
- [x] Removed stale tests targeting retired SparseFitter and GlobalAstroFitter APIs
- [x] Removed stale tests targeting retired pipeline, plotting, benchmark, and template helper APIs
- [x] Fixed PSFRegionMap lookup cache rebuilds after region replacement
- [x] Reduced pytest suite to compact current-code smoke and regression coverage
- [x] Added `Scene.plot` for scene-level diagnostics
- [x] Adjusted Chebyshev basis to accept [-1,1] inputs and added edge tests
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
- [x] Consolidated catalog matching and flux extraction into helper methods
- [x] Added aperture photometry on model+residual with PSF correction
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
- [x] Rewrote `examples/verify_pipeline.ipynb` to run the realistic
  two-detector F444W/F770W MockMosaic setup with 600 sources, real PSFs,
  standard Wiener PSF diagnostics, full-image diagnostics, and standard
  pipeline source diagnostics.
- [x] Added a fixed F770W-vs-F444W image shift to the verification notebook
  using public `(x, y)` pixel order at F770W truth/source placement time
  (`F770W_FIXED_SHIFT_XY=(-0.80, 0.95)`) and emitted the fit's existing
  `Scene.plot` diagnostics plus scene-overview diagnostics for selected scenes.
- [x] Reran `examples/verify_pipeline.ipynb` with 1000 input sources and
  `snr_range=(10, 5000)`; fitted scene shifts recovered the injected F770W
  offset on the upsampled fit grid (`dx=-1.6`, `dy=1.9`) to better than 0.02
  pixels for both no-extension and `psf_wings`.
- [x] Increased full-image and scene diagnostic save DPI from panel pixel
  dimensions so individual pixels are sampled in the output PNGs rather than
  compressed below native resolution.
- [x] Investigated bright non-deblended F770W residual-pull dipoles in the
  realistic verification output. Standard source diagnostics were saved for
  sources 160, 235, 402, 602, 774, and 854, and a no-F770W-shift control run
  showed comparable dipole moments; the pattern is therefore not caused by the
  injected fixed F770W offset.
- [x] Executed `examples/verify_pipeline.ipynb` with `jupyter execute`; the
  run wrote products under `examples/verify_pipeline_realistic_out`.
- [x] Ran `pytest tests/test_pipeline.py tests/test_psf.py tests/test_mock_mosaic.py -q`
  after the verification refactor (`37 passed`).
- [x] Ran `pytest tests/test_pipeline.py tests/test_psf.py tests/test_mock_mosaic.py tests/test_template_convolution.py -q`
  after the deblended-extension default change (`47 passed`).
- [x] Run `pytest` to ensure all tests pass
- [x] Save diagnostic plot during pipeline test
- [x] Save diagnostic plots for PSF, fitter and template tests
- [x] Save output catalog to disk during pipeline test
- [x] Benchmarked key pipeline steps in `tests/test_benchmark.py`
- [x] Added mock star centroid astrometry validation to `examples/mock_test.ipynb`
- [x] Restored legacy `pipeline.run(...)` wrapper compatibility for tests
- [x] Added rerunnable Moffat control script using `tests/utils.py` mock generation
- [x] Added scratch scans for Moffat growth-curve controls and intrinsic source sampling limits
- [x] Aligned Moffat mock source detection with `Catalog` smoothed S/N-image defaults
- [x] Added PSF split-cosine-bell kernel-window optimizer and scratch diagnostics
- [x] Renamed scratch kernel optimization note to `scratch/codex_kernel_optimization.md`
- [x] Added DrizzlePSF-based real F444W/F770W kernel-window comparison script and PNG diagnostics
- [x] Updated real F444W/F770W kernel-window diagnostics with full 2D panel comparisons and scaled cancellation penalty
- [x] Reran kernel diagnostics with full EE PSFs while investigating taper placement
- [x] Moved PSF edge smoothing to native-pixel ePSF load taper without flux renormalization
- [x] Simplified real F444W/F770W diagnostic WCS CSVs to MockMosaic-style one-row headers
- [x] Standardized all 2D diagnostic panels to a single shared contrast scale
- [x] Increased shared symmetric 2D diagnostic contrast by 20x
- [x] Switched kernel-window score-grid diagnostics to log10(FOM)
- [x] Removed misleading global best diagnostic from kernel-window comparison outputs
- [x] Updated kernel regularizer to q0=0.7 Nyquist with `lambda=1e-3` and default `C^2`
- [x] Updated individual kernel diagnostics with symlog radius axes and 0.16" aperture marker
- [x] Removed sqrt(2) corner-radius inflation from full-EE DrizzlePSF sizing
- [x] Enabled native DrizzlePSF stamp sizing with `ee_fraction=None` in real PSF diagnostics
- [x] Switched real F770W diagnostic resampling from 2x block replication to zoom interpolation
- [x] Added real-star Gaussian broadening scan and PNG diagnostics for F444W/F770W PSFs
- [x] Added unit-sum Gaussian target softening option to real F444W/F770W kernel diagnostics
- [x] Added `DrizzlePSF.load_jwst_stdpsf` with default 4-native-pixel ePSF edge taper
- [x] Made 2-pixel unit-sum Gaussian F770W target softening the default kernel diagnostic
- [x] Documented real PSF kernel optimization conclusions in `scratch/codex_kernel_optimization.md`
- [x] Increased kernel cancellation regularizer prefactor to `0.1*C^2`
- [x] Moved kernel high-frequency regularizer threshold to `0.7` Nyquist
- [x] Reduced kernel high-frequency regularizer prefactor to `0.1*HF`
- [x] Simplified kernel regularizer defaults to cancellation-only `1e-3*C^2`
- [x] Documented available photutils kernel-window families and their split-cosine-bell relationships
- [x] Expanded kernel optimization note with final regularization and windowing recommendation
- [x] Added `PSF.auto_matching_kernel_window` for named-FOM window optimization and diagnostic PNG output
- [x] Exposed `reg_lambda=1e-3` on `PSF.auto_matching_kernel_window`
- [x] Doubled default kernel-window alpha/beta sampling with `grid_oversample`
- [x] Updated kernel-window diagnostics with physical-value colorbars and fixed radial/growth ticks
- [x] Fixed even-sized kernel-window scoring with explicit centered full-convolution crops
- [x] Added shared `utils.fftconvolve` utility to protect same-sized even-kernel convolutions
- [x] Documented template-convolution alignment risk for even PSF-matching kernels
- [x] Added scratch benchmark comparing block replication, SciPy zoom, and fast_interp grid alignment
- [x] Updated resampling benchmark with exact method definitions, cubic interpolation, and edge-safe isolated sources
- [x] Expanded resampling benchmark to 100 random subpixel-phase scenes with mean/std statistics
- [x] Added OpenCV headless cubic resize to resampling benchmark and reran smaller statistics batch
- [x] Implemented OpenCV `INTER_CUBIC` flux-conserving PSF matching resize with no kernel renormalization
- [x] Updated Moffat scratch flux validation to use scene-solver `reg_flux` fix and emit best kernel-window diagnostic PNG
- [x] Added best kernel-window diagnostic PNG output to the standard Moffat flux recovery validation
- [x] Expanded standard Moffat flux recovery validation to include dilation radii 1, 2, and 3
- [x] Compared HF-inclusive Moffat kernel-window FOM and kept cancellation-only default
- [x] Standardized Moffat dilation validation scenarios to explicit radii 0, 1, 2, and 3
- [x] Updated Moffat validation truth-flux definition to sampled injected flux and added stress settings
- [x] Changed Moffat validation intrinsic Gaussian sizes to compact-weighted 0.4-8 pix distribution
- [x] Expanded Moffat validation image to 2x size with half as many sources for low-crowding residual checks
- [x] Routed package and validation FFT convolution through `mophongo.utils.fftconvolve`
- [x] Added fixed-source-size Moffat sweep diagnostic for sigma 5, 4, 3, 2, 1, and 0.5 pix
- [x] Normalized Moffat mock sources to sampled integrated flux after assigning intrinsic size
- [x] Replaced native-grid Moffat Gaussian rendering with oversampled flux-conserving stamp insertion
- [x] Recalibrated Moffat validation noise to a fixed minimum integrated-flux SNR after flux normalization
- [x] Updated fixed-size Moffat sweep to 1356x1356 images, 300 sources, and uniform true SNR 1-1000
- [x] Updated Moffat mock source scaling to use per-object effective fitted-flux SNR and F444W-like FWHM=2 pix
- [x] Parameterized fixed-size Moffat scratch sweep and reran sigma 0.5-5 with effective SNR 1-500, 1000x1000 image, and 500 sources
- [x] Reran fixed-size Moffat sweep in effective SNR 1-10 regime with per-source flux-pull tables and astrometry-on/off control
- [x] Reran fixed-size Moffat sweep with log-uniform effective SNR 1-500 and enforced F444W-like SNR >= 7
- [x] Added Moffat mock option to enforce high F444W-like SNR by lowering only high-res noise
- [x] Added Gaussian intrinsic source sizes to `MockMosaic` and ran realistic F444W/F770W Gaussian-source recovery sweeps with PSF-matching diagnostics
- [x] Limited PSF matching kernel diagnostic growth-curve ratio plots to radii > 0.7 pixel
- [x] Scaled source-stage template diagnostic panels to +/-5 times each panel's pixel MAD
- [x] Simplified the realistic Gaussian-source sweep to single-frame F444W/F770W mocks, optimized kernel-window diagnostics, and split-SNR residual-error histograms
- [x] Added default F770W Gaussian PSF broadening to the `MockMosaic` PSF creation hook with FWHM=0.08 arcsec, applied on the 40 mas reference grid
- [x] Moved realistic split-SNR residual-error histograms into the existing 4-panel flux-ratio diagnostic
- [x] Added noiseless mock truth FITS products and Moffat-style image diagnostics to the realistic Gaussian-source sweep
- [x] Added native-PSF mock source injection, 20% point-source markers, and true-template recovery diagnostics to the realistic Gaussian-source sweep
- [x] Added optional prebuilt-template input for `Pipeline` to separate template-extraction effects from flux-solver effects
- [x] Removed hidden PSF/kernel renormalization from the realistic mock PSF path and Fourier-basis kernel fits
- [x] Changed `PSF.from_array` to preserve native PSF sums for kernel fitting and diagnostics
- [x] Offset the realistic single-frame F770W mock footprint onto the single-detector F444W footprint and set the default source count to 300
- [x] Fixed realistic Gaussian-source kernel convention for unit-normalized extracted templates and added exact F770W true-template recovery A/B diagnostics
- [x] Restored MockMosaic DrizzlePSF finite-PSF integral scaling while keeping matching kernels unnormalized
- [x] Removed redundant `psf_matching_diagnostic.png` output from the realistic Gaussian-source sweep
- [x] Restored SNR-split residual histograms in the true-template flux-ratio diagnostic
- [x] Made SNR-split residual histograms the default in the realistic Gaussian-source flux-ratio diagnostics
- [x] Added F444W template-support fractions to the realistic Gaussian-source source table and summary
- [x] Wrote definitive realistic PSF flux-recovery report with tested root cause and solution options
- [x] Added regression tests for native-sum true-template scalar recovery and unit-template PSF kernel normalization
- [x] Removed the public `DrizzlePSF.get_psf*` `renormalize` keyword and removed post-drizzle PSF rescaling
- [x] Added DrizzlePSF full-stamp and partial-stamp flux-conservation regression coverage
- [x] Rejected and removed the DrizzlePSF-side aligned-stamp fix; kept template/kernel alignment ownership out of DrizzlePSF
- [x] Added detection/fitting metadata captions to realistic flux, residual, summary, and PSF-kernel diagnostic images
- [x] Unified legacy direct `utils.convolve2d` with the shared `fftconvolve` convention and added asymmetric/even-kernel regression tests
- [x] Fixed `Template.convolve_cutout` placement for odd template dimensions and added parity regression tests
- [x] Reran the realistic sigma=2 validation with 300 input sources and source-location residual pulls
- [x] Recorded that the exact true-template control passes residuals while current extracted templates fail the residual hard requirement
- [x] Updated the realistic PSF flux-recovery report with the residual hard-fail and three non-implemented template-extension options
- [x] Removed the pipeline's dummy one-pixel identity kernel; `None` now reaches the template no-convolution path
- [x] Added block-basis projection for upsampled low-resolution templates so model templates match block-replicated image pixels
- [x] Changed realistic F444W/F770W kernel diagnostics to drizzle the F770W target directly onto the F444W WCS grid
- [x] Reran the 300-source realistic sigma=2 validation and confirmed all extracted-template model/truth integer shifts are zero
- [x] Documented that the offset is fixed while extracted-template source residuals still fail because of incomplete template support
- [x] Added F770W-reference bright-source stamp residual metrics to the realistic 300-source diagnostic source table and summary
- [x] Added bright-source residual-pull stamp mosaics to the existing realistic image diagnostic panel
- [x] Reran the 300-source realistic sigma=2 F770W-reference validation and confirmed the exact F770W true-template residual path is noise-like
- [x] Updated the realistic PSF flux-recovery report to use the F770W true-template residual as the decisive reference control
- [x] Added production-path Cutout2D/template-convolution alignment regression covering all lower-left origin parities
- [x] Reran the 300-source no-extension realistic sigma=2 baseline and recorded that forced aligned cutouts do not change the F770W residual failure
- [x] Updated realistic diagnostics so kernel PNGs omit pipeline metadata and flux-ratio plot titles identify the template path
- [x] Added true-template image diagnostics and fitted-model centroid-shift plots to the 300-source realistic sweep
- [x] Verified bright fitted model-template centroids are not scattered at the pixel scale relative to exact injected F770W templates
- [x] Wrote dated flux-recovery debug synthesis report covering solver, PSF flux conservation, kernels, centroiding, convolution, template support, and recommendations
- [x] Added `ASTROPY_BUGS.md` to track upstream centroid and `Cutout2D.shape_input` issues for later reporting
- [x] Implemented `psf_wings` template completion by filling zero-valued extracted-template pixels from the local high-resolution unit PSF-shape convolution, with native PSF sums retained as throughput metadata
- [x] Reran the 300-source realistic sigma=2 validation with `template_dilate_segmap=0` and `template_extension=psf_wings`; median F770W flux scale is near unity but bright-source residuals still fail the noise-like requirement
- [x] Corrected the realistic F770W mock PSF convention so the true F770W PSF is the F770W drizzle/STPSF response convolved with 0.08 arcsec FWHM Gaussian broadening, not two native F770W pixels
- [x] Reran the 300-source isolated `psf_wings` validation with the corrected F770W 0.08 arcsec PSF convention; flux scale remains near unity but bright-source residuals still fail the noise-like requirement
- [x] Fixed the source-stage template diagnostic to place extracted, extended, convolved, model, and residual stamps on one common fitting-grid footprint with uniform inverted MAD scaling
- [x] Updated source-stage diagnostics with a standalone segmentation-map column, full-template-footprint default stamps, and shared ref-grid scaling for hires/extracted/extended panels
- [x] Refined source-stage diagnostics to default to the extracted-template tile, use template-only scaling for extracted/extended panels, and include the low-resolution image stamp before the model
- [x] Rendered source-stage segmentation panels as explicit RGBA label maps with black background, gray target source, and colored neighboring segments
- [x] Replaced segmap diagnostic neighbor colors with a bright fixed categorical palette and labels for every visible segment
- [x] Added an 8-point native-pixel phase dither A/B to the realistic mock validation and reran the 300-source isolated `psf_wings` test; phase sampling reduces but does not eliminate the F444W-through-kernel source residual excess
- [x] Added separate two-detector realistic diagnostic driver with NRCA5+NRCB5 F444W, two 8-subpixel MIRI macro pointings, global Wiener-kernel lambda selection, 300 sources by default, larger full-image diagnostics, and side-by-side `none`/`psf_wings` template-extension runs
- [x] Replaced the ambiguous PSF normalization policy with an explicit shape-vs-throughput convention: unit-sum PSF shapes for fitting operations, native finite-stamp sums as throughput metadata and final flux corrections
- [x] Made the two-detector realistic driver self-contained from other scratch scripts, switched mock PSF stamps to F444W=4 arcsec and F770W=8 arcsec, removed the bespoke PSF diagnostic PNG, and marked F444W-blended detections in full-image residual-stamp diagnostics
- [x] Added catalog deblend provenance flags, standard PSF-class regularization diagnostics (`diagnostic_<method>.png`), and open-circle markers for hires-deblended sources in flux-ratio diagnostics
- [x] Omitted the low-information template-extension summary PNG while retaining the machine-readable summary CSV
- [x] Made flux-ratio diagnostics draw deblend-flagged sources as colored open markers instead of filled markers with black overlays, and fit residual-error Gaussians separately for the SNR histogram groups
- [x] Standardized scalar PSF-kernel lambda optimization to scan `1e-6..0.1`
- [x] Changed MockMosaic default weight products to actual pixel inverse variance, including pixel-scale, exposure-depth, and drizzle corrections
- [x] Switched two-detector flux-ratio residual pulls to use WHT-derived predicted template errors while preserving scene-covariance errors in the source tables
- [x] Recomputed PSF growth-ratio diagnostic samples densely from `r>0.7` pixel and anchored full-image diagnostics to the lower-left covered tile
- [x] Fixed low-resolution inverse-variance upsampling so 80 mas -> 40 mas fitting preserves native chi-square and predicted flux errors
- [x] Propagated catalog `is_deblended` provenance onto template flags and skipped template extension for deblended children
- [x] Added a 1% flux systematic floor to flux-ratio residual/predicted-error diagnostic plots
- [x] Added filter-level throughput-corrected total-flux columns to the `Pipeline.run()` output catalog
