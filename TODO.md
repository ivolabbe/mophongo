# TODO

This file tracks future desired features, checks, and investigations.

- [x] ~~`scene_max_size` = 500 costs F1800W its photometry~~ -- tested and
  rejected (2026-08-11). Reran all four UDS bands with `scene_max_size` = 800,
  `scene_max_merge_radius` = 1000 px: every comparison number unchanged to the
  second decimal, even though the F1800W bisection threshold *rose* to 0.381
  (the whole r < 1' patch is one coupled component at the 1e-3 floor). The
  apparent red-band disagreement with IDL was a selection artefact instead:
  cutting at IDL mag < 24 in a shallow band selects SNR-of-a-few sources, and
  the mean over that asymmetric tail inflates (F1800W raw-aperture mean +0.48,
  median +0.25, and +0.02 at SNR > 25). At SNR > 25 all four bands agree with
  IDL to 1-3%. `make_compare_idl_python.py` now quotes medians and adds an
  SNR > 25 line. Still open, moved to its own items: `psf_size` = 4" at 18 um,
  and whether the generated configs should keep 800/1000 (photometry is
  insensitive; solve time was not).
- [ ] Confirm the photometric aperture for F560W, F1000W and F2100W. The
  generated MINERVA configs take `aperture_diam` from the classic IDL subphot
  values (0.70, 1.20, 1.20, 1.50" for F770W/F1280W/F1500W/F1800W, from
  `examples/run_uds_770_wren.py`), which is what makes the raw aperture fluxes
  comparable between the two codes. The other three bands have no IDL
  counterpart and are interpolated on the same trend (0.60, 0.90, 1.70"); ask
  what COSMOS/EGS actually use.
- [ ] Measure the extra Gaussian broadening at F2100W. The COSMOS and EGS
  releases have F2100W, but the MINERVA-UDS star test only reached F1800W, so
  `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC["f2100w"] = 0.30"` is an extrapolation
  along the F1280W-F1800W trend. Repeat the star test in COSMOS or EGS.
- [ ] The generated MINERVA configs (`examples/make_minerva_configs.py`) fix
  `psf_size` at 4.0" for every MIRI band because the same value sets the
  hi-res support and the F444W grids are only 4.09" across. At F1800W/F2100W
  the MIRI PSF FWHM is ~0.6-0.7", so 4" is only ~6 FWHM and the wings are cut
  well inside where they still carry flux. Regenerate the NIRCam grids at a
  larger FOV before running the reddest bands, or land the decoupling of PSF
  support from kernel support below.

- [ ] **`ee_psf_lo` never survives to the catalogue in the default path.**
  `multi_resolution_method` defaults to `upsample`; at `k>1` `Pipeline.run` calls
  `convolve_templates` (which sets `ee_psf_lo`) and then rebuilds every template through
  `project_to_block_replicated_grid`, which copies only `flag`, `deblend_parent_label`
  and `deblend_nchildren` (`templates.py:746-748`). Confirmed by execution: 0.917 in,
  nan out; `Template.downsample` loses it too. So every source falls back to the
  filter-level mean and `flux_<i>_total` silently reverts to the pre-encircled-energy
  behaviour. MINERVA (40 mas ref, 80 mas MIRI, k=2) is exactly this case, so the whole
  encircled-energy chain is inactive on those runs. Fix: propagate `ee_psf_lo`/`ee_tmpl`
  in `project_to_block_replicated_grid` and `downsample`, and add a test that a projected
  template keeps them. Found by the 2026-08-10 main audit (`scratch/wren/flux_estimator_comparison.pdf` §7.6).
- [ ] Validate the `ee_psf_lo` divisor on a real rerun. `flux_<i>_total` now
  divides by the per-source `ee_psf_lo` recorded by `convolve_templates`,
  falling back to the filter mean only where it is missing. Against the old
  filter-level mean this moves fluxes by -1.0% to +0.6% across the UDS field,
  in a pattern that tracks the exposure layout, so DR0.1 outputs need a rerun
  before they are compared with anything. Requires `build_kernels(overwrite=
  True)` as well, since cached kernel maps do not record the method that built
  them.
- [ ] `ee_tmpl` is recorded per template and is deliberately not applied as a
  correction: the fitted amplitude does not scale with the blanked wing flux,
  it scales with the leverage of the blanked pixels, and dividing by `ee_tmpl`
  over-corrects in every configuration tested (isolated, 0.2% error becomes
  0.9%; blended pair at 8 px, 3.3% becomes 6.7%). Keep it as a trust
  diagnostic. It is 1.0 on the current default path and only becomes
  informative once `composite_psf_wings` (normalise over the whole stamp, then
  blank neighbours) is ported from the `template` branch.
- [ ] Check the two anomalies in the PSF grid EEs (measured with
  `sum(plane)/oversampling**2`): F444W spans 0.9179-0.9809 across the 25 grid
  points of one MJD file and F770W 0.9659-1.0030 across 9, a +-3% spatial
  spread that now propagates straight into photometry; and F770W's 1.0030 is
  unphysical by 0.3%. Also confirm that `normalize='first'` loses flux only to
  FOV truncation, by generating one large-FOV (~20") grid and checking that
  `sum/oversampling**2` approaches 1.0 - anything it saturates below would be
  optical loss already carried by the mosaic zeropoint, and correcting for it
  would double-count.
- [ ] The F444W PSF grids have a 4.09" FOV against F770W's 8.10", so the
  *detection* PSF model is the more truncated one: 4.3% of F444W light is
  outside the model entirely versus 1.5% for F770W. Enlarging `psf_size` past
  ~4" for F444W measures the grid edge rather than the PSF. Regenerate the
  NIRCam grids at a larger FOV if the hi-side wings start to matter.
- [ ] Decouple PSF support from kernel support. `psf_size` currently sets
  both, so shrinking the stamp to save time also renormalizes the PSFs the
  kernel is derived from, which biases the kernel core by `S_hi/S_lo`:
  +6.3% at 2", +3.1% at the 4" default, -0.7% on the full parent grid
  (`scratch/wren/template_comparison.tex` rec 4). Derive kernels on the
  largest available grid and crop the resulting kernel afterwards; do not
  renormalize the crop.
- [ ] Replace or tune the default `matching_kernel` window: it costs 1-9% in
  flux scale and is the largest remaining term in the EE chain.
  Measured on truncated Moffat pairs at fixed R with `S_hi` = 0.9820, as
  `A/S_lo` for an injected point source (1.0 is exact):

  | `S_hi/S_lo` | window | tikhonov 1e-3 | tikhonov 1e-6 |
  |---|---|---|---|
  | 1.3289 | 1.01255 | 1.00178 | 1.00000 |
  | 1.0244 | 1.01643 | 1.00213 | 1.00000 |
  | 0.9822 | 1.02159 | 1.00252 | 1.00000 |
  | 1.0000 (lo == hi) | 1.09053 | 1.01561 | 1.00007 |

  The error is worst when the two PSFs are similar, i.e. when the true kernel
  approaches a delta and the SplitCosineBell low-pass damages it most, so
  adjacent-band matching suffers more than NIRCam-to-MIRI. On the real UDS
  F444W/F770W pair the window gives `A/S_lo` = 0.9688, the opposite sign, so
  this has to be measured per band pair rather than assumed.
  With a faithful kernel (`tikhonov` at `reg` = 1e-6, match residual 2e-5 to
  2e-4 of peak) the recovery is exact at every truncation and every EE
  mismatch, which is what establishes that `S_lo` alone is the complete PSF-side
  correction. `reg` = 1e-6 is on noise-free analytic PSFs; real drizzled ePSFs
  need a bias-variance compromise, so tune with
  `PSF.optimize_matching_kernel_regularization(..., diagnostic_path=...)` and
  add `A/S_lo` on an injected point source to its criteria.
- [ ] Scene partitioning is not reproducible run to run (from wren's
  `CHECKLIST.md`; the mechanism is live here and recorded nowhere else).
- [ ] The flux-block ridge biases faint sources low: -33% at
  `d_i/median = 1e-6` (also from wren's `CHECKLIST.md`). **Confirmed still
  live** (2026-08-10): the `flux-bug` fix removed `reg_astrom=1e-4` leaking
  into the photometric normal matrix, a different and larger term. What
  remains at `scene_fitter.py:178-181` is `lam_A = 1e-6 * median(diag(A))`,
  one absolute value per scene added *before* whitening, which is exactly the
  configuration wren measured. Quiet today because truncated templates keep
  `sum(T^2)` concentrated; extended composites reach those ratios. Fix and
  regression test in `docs/WREN_MERGE_PATH.md` §5 — must land before the
  estimator work.
- [ ] Wing deficit in the drizzled PSF, found by `examples/check_psf.ipynb`:
  the azimuthally averaged model/star ratio sits at 0.83-0.95 (F444W, F770W)
  and 0.72-0.76 (F1500W) outside the core, so the drizzled ePSF is missing
  wing flux. Candidates: the 4-native-pixel edge taper applied at load time,
  the finite STDPSF stamp, and the MIRI extended-PSF handling. Until it is
  understood, any full-stamp kernel fit trades core against wings.
- [ ] F444W central blur is censored: 82 % of stars land on the narrowest
  Gaussian basis element (0.4 drizzle pixels), so the fit only bounds the blur
  from above. Either extend the basis below 0.4 px or read the blur off the
  basis-free width comparison (`blur_quad`, 0.037" for F444W).
- [ ] The catalog-free finder in `examples/check_psf.ipynb` (peaks plus an
  aperture-ratio locus) is worth promoting into `catalog.py` if it holds up: it
  finds 37 usable F1500W stars where `flag_star` yields 3, and needs neither a
  background model nor inverse variance. It currently lives in the notebook.
- [ ] Fetch the m3.1 F1500W mosaic. The catalog's MIRI photometry is m3.1 but
  the only local F1500W image is DR0 v2.4, so catalog positions land on
  zero-weight pixels. Less critical now that selection no longer needs the
  catalog, but the two versions should not be mixed.
- [ ] scan for bug fixes / robustness improvements
- [ ] `tests/test_pipeline_multitemplate.py::test_pipeline_multitemplate_pass`
  no longer exercises a multi-template pass: `_add_templates_for_bad_fits` and
  the `multi_tmpl_*` config knobs were removed in the 2026-08 cleanup (the
  only call site had been commented out long before). Either reinstate the
  feature or rename the test to what it now covers, a plain pipeline run.
- [ ] `SparseFitter` no longer solves: after the 2026-08 cleanup it builds the
  normal matrix and returns model/residual images and covariance-free
  estimators, while `SceneFitter` owns all flux solving. Decide whether the
  remaining `SparseFitter` surface (`build_normal_tree`, `add_flux_priors`,
  `quick_flux`, `predicted_errors`, `flux_and_rms`, `model_image`,
  `residual`) should fold into `scene_fitter.py` or stay a separate class.
- [ ] Public API exercised only from `scratch/` and notebooks has no test
  coverage: `utils.compare_psf_to_star`, `utils.clean_stamp`,
  `utils.write_wcs_csv`, `utils.retile_blocked`, `utils.gauss_hermite_basis`,
  `verification.run_pipeline_extension_scenario`,
  `verification.build_wiener_psf_maps`,
  `verification.build_realistic_two_detector_mock`, `Catalog.show_stamp`,
  `Catalog.plot_bg`, `Catalog.find_stars`, `catalog.find_saturated_stars`,
  `DrizzlePSF.register`, `PSF.from_data`, `PSF.matching_kernel_basis`,
  `psf.psf_matching_kernel_basis`, `MockMosaic.source_model_templates`,
  `Scene.overlay_scene_graph`, `SparseFitter.add_flux_priors`,
  `Pipeline.plot_result`. Either cover them or demote them to `scratch/`.
- [ ] `saturate.py`, `psf_factory.py` and `jwst_psf.py` are driven only from
  `scratch/` scripts and have no direct test of their own.
- [ ] `AlignedCutout.as_block_reduced` / `.as_block_replicated` are kept as
  public wrappers over `_block_reduce`/`_block_replicate` but have no caller.
- [ ] Point-in-time reports under `docs/` (`FLUXBUG.md`, `LWBUG_ANALYSIS.md`,
  `FORK_DIFF_WREN.md`, `FLUX_RECOVERY_DEBUG_SYNTHESIS_2026-05-03.md`,
  `REALISTIC_PSF_FLUX_RECOVERY_REPORT.md`) reference modules deleted in the
  2026-08 cleanup (`astro_fit.py`, `deblender.py`, `photutils_deblend.py`,
  `sim_data.py`). They are kept as history; decide whether to archive them.
- [ ] persist `id_scene` as a fit-table column in `Pipeline.write_outputs` —
  scene membership currently lives only on `Template.id_scene` during the run,
  so partition diagnostics (compactness, nesting, per-scene flux stats) cannot
  be computed from the outputs.
- [ ] Per-band `scene_coupling_thresh` ladder (wren: 1e-3 F770W ... 0.02
  F1800W) collapses to one constant 0.030 +/- 0.005 after dividing by median
  SNR and PSF area (`docs/SCENE_PARTITION.md`) — a noise-relative,
  area-normalized coupling score would remove the per-band tuning without any
  new machinery. Template support (`psf_size`, currently null = 8" stamps) is
  the real lever for scene size: wren's 3" stamps ran the full field at a
  fixed 1e-3.
- [ ] `PSFSZ<i>` and `RCIRC<i>` in `cat.meta` are half their true value in
  upsample mode. `pipeline.py:1207` sets `wcs[ifilt] = wcs[0]` and `wcs` is an
  alias of `self.wcs` (`:1053`), so `_record_psf_ee` (`:1384`) reads the 0.04"
  reference scale instead of the PSF stamp's native 0.08". Cached run shows
  `PSFSZ1 = 4.04` / `RCIRC1 = 2.02` for a 101-px 8.08" stamp. `EEBOX1` and
  `EECIRC1` are correct (`i_circ` is scale-free). Fix: capture the native
  pixel scales before the fit loop.
- [ ] flux-estimator work from `docs/FLUX_ESTIMATORS.md` and
  `scratch/wren/flux_estimator_comparison_v2.pdf` (ordered by impact).
  Sequenced against the dev-wren fork in `docs/WREN_MERGE_PATH.md`, which
  decides which of `docs/FORK_AUDIT_WREN.md` is still wanted now that the
  encircled-energy chain is settled. All four estimators are kept, renamed
  `est3int -> est3` and `est3cat -> est4`, with the algebra translated from
  wren's `c_det`/`c_b` to `S_hi`/`S_lo` (transcribing it verbatim
  double-corrects by 4.6% on UDS). `PSFRegionMap.containment` is dropped in
  favour of a per-region curve of growth (`refresh_cog`/`get_ee_at`), since our
  stamps are absolutely calibrated; the catalogue's own
  `ee_kron_cat = (fauto_KRON/faper_KRON)/tot_cor` is a 288k-row per-source
  acceptance gate for it:
  - [ ] bound the composite EEs by the PSF EEs: `ap_F <= EE_psf_hi(R_cat)`,
    `ap_B <= EE_psf_lo(R_img)`. Two numbers per band, kills the 40x correction
    tail without touching templates; report the clip rate as a template-quality
    diagnostic
  - [ ] report the aperture-matched quantity `A*ap_F(R_cat) + sum_disk(res)` as
    primary, with `R_cat` per source from `use_aper`, and form totals with the
    catalogue's own `tot_cor` — the total correction cancels in colour only if
    the same factor is used for every band, so mixing mophongo `totcor1` with
    catalogue `tot_cor` injects the full 10%/9% discrepancy into colours
  - [ ] add a low-SNR shrinkage prior, linear rather than IDL's quadrature:
    `H = w*H_data + (1-w)*PSF` with `w = S^2/(S^2+S0^2)`; `fit_snrlo_psf <= 0`
    disables (no PSF lookup, w=1, templates untouched). Blend *after*
    extension; needs hi-res segment SNR (`wht_hi` in `RunConfig`, or global
    nmad fallback) and the hi-res PSF wired into the config path
    (`psfs=[None, prm_lo]` today); centre the PSF on the flux-weighted segment
    centroid, not the brightest pixel. Engagement: quadratic w has 20% PSF at
    S=2*S0 (IDL's quadrature adds ~12% there) — if onset must be negligible by
    2*S0, steepen to `w=S^4/(S^4+S0^4)` (6%) or lower S0; decide S0/exponent on
    injected-truth mocks (IDL default `fit_snrlo_psf=10`); a skip at w>0.99
    avoids any step. Record `snr_template` and `psf_shrink_weight` as columns
  - [ ] decontaminate template cores at extraction (v3 pdf, cross-contamination
    section): cores are raw detection data inside the segment and carry
    neighbour wing light, so both wing placements fit a wrong profile — bridge
    residuals + biased amplitudes, worst for faint-beside-bright
    (`f_nn*c/(f*e) > 1`; `kseg > knn` cannot see smooth unsegmented wings).
    Pass 1: re-extract cores from `D - sum_{j!=i} W_j` (the other sources'
    analytic wing models), renormalise wings to the decontaminated segment
    flux, then switch wing fill from background-only to the full complement
    (IDL placement: every stamp pixel outside the source's own positive
    segment, never its own data — correct once cores are clean). One
    iteration converges;
    validate on mock pairs (separation x flux ratio)
  - [ ] adopt IDL's competitive dilation (`kseg > knn`) in place of the
    order-dependent background-only `safe_dilate_segmentation`, and consider
    excluding non-positive segment pixels from the template
    (2026-08-08: `template_dilate_segmap` default set to 0 — the run-path IDL
    (`subphot.pro`) never dilated either; competitive dilation only exists in
    the never-run `mophongo__define.pro` rewrite. This item is now about the
    extension-mask arbiter, not the default extraction)
  - [ ] add `extend_templates` to `RunConfig` and pass it through `load_data` —
    it is currently unreachable from a run config, so config-driven runs use
    truncated templates and every downstream correction inherits the low bias.
    Wing construction needs deciding too (`docs/FLUX_ESTIMATORS.md`, "Wing
    construction" section): IDL tile (exact compact, wrong resolved), ifl
    `psf_wings` self-convolution (data ⊛ PSF — over-broadens as PSF^2, worst
    for compact, exact only far-field), wren radial blend `W·data+(1−W)·tile`
    (right structure; point-source `M` underestimates resolved halos, kinked
    weight, never enters neighbour segments, raw-data halo re-admits neighbour
    wings). Compare all three + decontaminated full-complement on the mock
    pair grid. Also: `_psf_for_template_extension` falls back to the *lo-res*
    PSF when `psfs[0] is None` — wire the hi-res PSF in first
  - [ ] write Estimator 2 as a column: `flux_<i>/throughput_<i> + sum_Omega(res)`
    (no aperture photometry needed for the first term — it collapses to `A`)
  - [ ] record the residual sum over `disk(R_phi)`, `seg_H` and their union
    separately, so the bias/variance trade is measurable instead of baked in
    (deferred 2026-08: matched quantity uses the disk residual only — not
    testing colour gradients now, and the catalogue's adaptive `use_aper`
    already grows with source size; keep for the totals path)
  - [ ] propagate errors on the residual term; if any Estimator-1 style column
    is kept, scale its error by `totcor1` too
  - [ ] test on injected-truth mocks whether template extension closes the ~10%
    `totcor1` vs `apcor1 x tcor_H` gap (mocks only, per project policy)
- [ ] aperture/photometry cleanups identified in `docs/PHOTOMETRY_APERTURES.md`:
  - [ ] move the band-independent `num` (reference-template aperture sum) into a
    preprocessing step that augments the catalog with `r_aper_pix`,
    `f_aper_tmpl`, `tot_cor`, flags; collapses the fit-time correction to
    `corr = cat["f_aper_tmpl"][row] / den` and lets `aperture_catalog` /
    `aperture_units` retire (section 7)
  - [ ] assert aperpy meta `KERNEL` matches the hi-res filter — `ap_corr_<i>` is
    only valid because `KERNEL == f444w == sci_hi` filter today
  - [ ] fix the mixed-grid `r_cat` default in the `downsample` path, or refuse
    the `None` branch there; also remove the dead 1.5xFWHM fallback doc'd at
    `pipeline.py:886`
  - [ ] carry `ra`/`dec` (and optionally `use_aper`/`tot_cor`/flags) into the
    fit table so outputs are usable without re-joining the source catalog
  - [ ] decide whether `ap_flux_<i>` needs an uncertainty; residual inside the
    aperture is correlated, so `err_pred` is not a valid substitute
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
  New evidence (2026-08, `docs/FORK_DIFF_WREN.md`): measured against the
  *tapered* parent grid the drizzle error does not have a consistent sign —
  F444W 102 px/4.08" stamp = 0.96131 vs tapered parent 0.95488 (**+0.67%**,
  and the stamp is circularly apodized with 11.8% exactly-zero pixels, so the
  excess over its actual support is larger), while F770W 101 px/8.08" =
  0.97008 vs tapered parent 0.98125 (**-1.14%**). So `EEBOX<i>` is not a clean
  absolute EE, and this must be settled together with the `flux_<i>_total`
  divisor item above — same order of magnitude.
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
- [x] packaging / CLI defects found installing on a clean environment (CANFAR,
  2026-08-10; fixed same day, see `scratch/canfar/RUNNING_ON_CANFAR.md`)
  - [x] `psutil`, `photutils`, `matplotlib`, `pysiaf` and `pillow` are all
    imported directly but were declared nowhere in `pyproject.toml`. Four
    arrived transitively, which hid the problem until pip resolved `photutils`
    to 3.0.0 and broke `drizzlepac` 3.9.1 (`IntegratedGaussianPRF` removed).
    Added with `poetry add`, `photutils` bounded `>=2.2.0,<3.0.0`.
  - [x] `python -m mophongo.pipeline <cfg>` with no steps failed with
    `invalid choice: []`. `nargs="*"` makes argparse check the collected list
    against `choices` as one value, so neither an empty default nor `["all"]`
    passes; dropped `choices` and validate the steps explicitly instead.
  - Verified by building a fresh venv on CANFAR from `pyproject.toml` alone,
    with no manual pins: resolves photutils 2.3.0 / psutil 7.2.2, all pipeline
    imports succeed, and the bare CLI runs.
- [ ] refactoring for readibility and modularity
  - [ ] split off PSF map / drizzle PSF / PSFs module, make submodule
  - [ ] split off real data as submodule?
  - [ ] other code review, misc refactoring, consolidation
  - [ ] remove unused modules, orphan code
