# TODO

This file tracks future desired features, checks, and investigations.

- [ ] scan for bug fixes / robustness improvements
- [ ] Run the four `extend_mode` schemes head to head on injected-truth mocks
  (`verification.py`, not ad hoc comparisons) and pick a default. The synthesis
  `template_comparison.tex` Sec. 8.3 recommends: wren's blended extension but
  with the halo running to the full PSF support instead of stopping at R95, a
  background subtraction before blending raw data over an ~855 px halo, and
  IDL's least-squares halo amplitude in place of wren's flux-ratio anchor.
  Each of those is a small edit inside `template_schemes.py`.
- [ ] `extend_mode='classic'` does not reproduce IDL step 7 (`subphot.pro:324`:
  normalise the *convolved* plane, then apply a circular `apermask` of radius
  `ceil(ksz/2)` centred on the source, so the fitted template sums to slightly
  less than 1 by a different amount per source). It sits in the convolution
  stage, which is scheme-agnostic. Decide whether the 1-1 comparison needs it;
  if so it wants a scheme hook in `convolve_templates`, not an inline branch.
- [ ] `extend_mode='wren'` carries `flux_beyond_stamp`/`flux_beyond_aper` in
  `Template.extend_info` but nothing consumes them. wren feeds them into its
  aperture-correction chain (`trunc = norm / (norm + flux_beyond_stamp)`).
  Wire them up, or drop them, once the scheme comparison settles.
- [ ] `uds_770_dr0.json` gets no detection ivar: its `sci_hi` is the
  saturated-star-repaired mosaic in `repair_saturate_out/`, and the weight map
  sits with the original under `MINERVA/data/UDS/DR0/`. `RunConfig.wht_hi`
  auto-derivation tries `sci_hi` then `driz_hi` (`_sci.fits` -> `_wht.fits`)
  and neither resolves, so a `wren`/`classic` run there warns and falls back
  to the scalar sky sigma. Set `wht_hi` (or `driz_hi`) explicitly in that
  config before using it for a scheme comparison. `cosmos_770_dr0.1.json`
  resolves automatically (verified: 32768x18944, 99.8% covered).
- [ ] The detection background is measured in `_load_detection_ivar` and
  logged but not subtracted (COSMOS median -7.2e-4), because subtracting it
  would change `default` templates too. `template_comparison.tex` Sec. 8.1
  lists "raw detection image, no background subtraction" as a wren defect —
  a sky pedestal enters the halo linearly with its area, over ~855 px. Decide
  whether the extended schemes should subtract it.
- [ ] `extend_mode='wren'`'s `WrenParams.containment` (wren's
  `PSFRegionMap.containment`, the detection-PSF stamp containment `c_det` in
  `flux_beyond_stamp`) defaults to 1.0 because this tree has no equivalent.
  `psf.stamp_encircled_energy` / `DrizzlePSF.ee_box` measure the same thing;
  wire one in if the wren truncation bookkeeping is kept. Note
  `template_comparison.tex` Sec. 8.1 flags wren's own `containment` as
  normalised to the parent grid and therefore ~+2.9% (F444W) high.
- [ ] `fit.py:1134` (`SparseFitter.solve` path) calls its local
  `build_scene_tree_from_normal` with `coupling_thresh=1e-4` hardcoded,
  ignoring `cfg.scene_coupling_thresh`. Scene partitions from that path
  silently disagree with the config. Either route it through
  `scene.generate_scenes` or pass the config through. Note `fit.py` carries
  its own duplicate `merge_small_scenes` (`:389`) and
  `build_scene_tree_from_normal` — the duplication is the underlying problem.
  See `docs/SCENE_PARTITION.md`.
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
- [ ] `flux_<i>_total` divides by the wrong stamp sum. `matching_kernel`
  deliberately does not normalize, so `sum(k) = S_lo/S_hi` (verified: cached
  kernel median 1.00747 vs `S_lo/S_hi` 1.00672). For a point source the
  unit-sum detection template convolves to `psf_lo/S_hi`, so the fitted
  amplitude is `flux_<i> = A_true * S_hi` — the *detection* stamp sum
  (verified numerically: a = 0.9557 vs S_hi = 0.9606). `pipeline.py:790` then
  divides by `throughput = mean(S_lo)`, but `S_lo` already cancelled inside
  the kernel DC, so `flux_<i>_total = A_true * S_hi/S_lo`. The right divisor
  is `S_hi`. Bias is only −0.7% on `uds_770_dr0.1.json` because
  `"psf_size": null` leaves both stamps near-native (0.9613 vs 0.9678); at the
  `RunConfig` default 4.0" it is +3.1%, at 3.0" +4.9%, at 2.0" +6.3%.
  Either divide by the detection-side sum, or normalize the kernel and pass
  unit-sum shapes as `utils.matching_kernel`'s own docstring recommends.
  Caveat: this is the point-source limit (`T = P_hi/S_hi`); path A's templates
  are segmap-truncated with no extension, so the real bias is source-dependent.
  Settle it together with the drizzle-normalization item below — that is the
  same order of magnitude. See `docs/FORK_DIFF_WREN.md` Sec 5.
- [ ] `PSFSZ<i>` and `RCIRC<i>` in `cat.meta` are half their true value in
  upsample mode. `pipeline.py:1207` sets `wcs[ifilt] = wcs[0]` and `wcs` is an
  alias of `self.wcs` (`:1053`), so `_record_psf_ee` (`:1384`) reads the 0.04"
  reference scale instead of the PSF stamp's native 0.08". Cached run shows
  `PSFSZ1 = 4.04` / `RCIRC1 = 2.02` for a 101-px 8.08" stamp. `EEBOX1` and
  `EECIRC1` are correct (`i_circ` is scale-free). Fix: capture the native
  pixel scales before the fit loop.
- [ ] flux-estimator work from `docs/FLUX_ESTIMATORS.md` and
  `scratch/wren/flux_estimator_comparison_v2.pdf` (ordered by impact):
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
- [ ] refactoring for readibility and modularity
  - [ ] split off PSF map / drizzle PSF / PSFs module, make submodule
  - [ ] split off real data as submodule?
  - [ ] other code review, misc refactoring, consolidation
  - [ ] remove unused modules, orphan code
