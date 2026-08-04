# TODO

This file tracks future desired features, checks, and investigations.

- [ ] scan for bug fixes / robustness improvements
- [ ] drizzled PSF stamps gain ~0.93% flux relative to the ePSF that went in.
  UDS DR0 F770W: native `EE_stamp` = 0.9619 and the detector-sampled `eval_ePSF`
  sum matches it to 0.9620 at both 0 and 0.5 pixel phase, but the drizzled sum
  asymptotes to 0.9708 (7.2" -> 0.9662, 8.16" -> 0.9698, 12" -> 0.9708, 16" ->
  0.9707), so it is a flux scale, not a stamp-boundary effect. 1.0092 in area is
  0.46% in linear scale — suspect `wcslin_pscale=psf_wcs.pscale` (one global
  `get_wcs_pscale`) against MIRI's real local pixel area, unproven.
  Consequences: (a) if grizli drizzled the mosaic with the same convention the
  bias cancels and `EEBOX<i>` is right, which is the assumption the pipeline now
  runs on; (b) if not, `flux_<i>_total` is ~0.9% low. Settle it on
  injected-truth mocks, not by arguing about drizzle internals.
- [ ] `examples/run_mock.py:73` still asks for `ee_fraction=0.95`, which under the
  absolute semantics now gives 7.200"/90 pix instead of the old 4.731"/60 pix.
  Decide whether to pin `size=4.731` or accept the bigger stamp.
- [ ] `_ee_fraction_to_arcsec` is non-monotonic near the ceiling: `ee_fraction=0.96`
  gives 8.64" but `1.0` gives 8.16", because the `>=1.0` branch uses the ePSF side
  length while the sub-1.0 branch uses `2*r`. Either clamp sub-1.0 sizes to the side
  length or leave it and document.
- [ ] MINERVA DR0.1 (COSMOS) follow-ups, now that
  `examples/cosmos_770_dr0.1.json` runs end to end:
  - [ ] run the saturated-star repair pass (`saturate.py`) on the COSMOS F444W
    mosaic + LW segmap/catalog, so DR0.1 matches DR0's repaired template side
  - [ ] full-field run (`r_trial` 0) once the trial patch is signed off
  - [ ] the other DR0.1 MIRI bands are already unpacked (F1000W, F1280W,
    F1500W, F1800W, F2100W) — configs are a copy of the F770W json with the
    filter fields changed, but F1500W/F1800W stay gated on kernel-window
    vetting per the audit plan, and F1000W/F2100W have no
    `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC` entry yet
  - [ ] decide whether to also fit the chi-mean detection (the DR0.1 README
    recommends it; the LW flavour was used here to match DR0)
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
