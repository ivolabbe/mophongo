# TODO

This file tracks future desired features, checks, and investigations.

- [ ] scan for bug fixes / robustness improvements
- [ ] rerun `examples/verify_pipeline.ipynb` to regenerate the stale
  `verify_pipeline_realistic_out` products with the fixed
  `inject_point_sources` (see `scratch/dipole_rootcause/README.md`)
- [ ] 3 of 40 verification scenes still fit astrometric shifts 0.1-0.34 fit-pix
  off the injected value. Ruled out: iteration count (converges to a stationary
  biased point), faint templates in the shift blocks (`snr_thresh_astrom=15`
  identical), cross-scene contamination (clean-data chi2 scan identical),
  kernel regions, and local mock painting errors. One case is a blend flux
  swap (ids 503/614); the others are dominated by a single bright extended
  source (id 402) whose exact chi2 prefers an even larger shift than injected
  — template-shape/shift coupling. Diagnostic scripts in
  `scratch/dipole_rootcause/scene38_*`.
- [ ] investigate the remaining small symmetric bright-source core residual
  (~1.2-1.7x noise rms in stamp, no dipole) — kernel/core mismatch, separate
  from the fixed injection-phase dipoles
- [ ] storage
  - [ ] best way to store intermediate results
  - [ ] "drop" image
- [ ] templates
  - [ ] test and validate fitting in downsampled space
  - [ ] profiles of low SNR objects -> asymptotically to psf
- [ ] background options
  - [ ] global background fit
  - [ ] background per stamp
- [ ] validate output catalogs on MIRI data
  - [ ] color color, color mag
  - [ ] SEDs of stars, photo-z
  - [ ] add in residuals in core for improved flux measurements (shift / psf errors)
- [ ] investigate blending in detection image
- [ ] Investigate template extension methods (Moffat fit and PSF dilation)
- [ ] End-to-end test with realistic mosaic data using `make_mosaic_dataset`
- [ ] Maintain an executable `examples/verify_pipeline.ipynb` notebook that
  demonstrates a pipeline setup from scratch and regenerates standard
  diagnostics
- [ ] Profiling speed + memory usage
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
