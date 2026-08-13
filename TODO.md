# TODO

This file tracks future desired features, checks, and investigations.

- [ ] `AlignedCutout.downsample`/`upsample` pass `self.wcs` (this cutout's
  WCS) as the parent WCS of a new cutout built on a full-shape dummy, the
  same pattern fixed in `Template.convolve_cutout` and
  `project_to_block_replicated_grid` under P1-05 (2026-08-12). Left alone
  because `AlignedCutout` has no `wcs_original` to pass, so fixing it means
  either carrying the parent WCS on the base class or taking it as an
  argument the way `Template.downsample` already does. No caller currently
  reads the resulting WCS for a sky lookup, which is why it is not a P1.

- [ ] Re-tune the background source mask against real mosaics. P1-03
  (2026-08-12) set `detect_thresh=2.5` and `faint_thresh=4.0` from injected
  sources on synthetic correlated noise (`tests/test_background_masking.py`),
  which fixes the coupled polarity/threshold defects but calibrates the
  thresholds on a source population whose realism is unverified. Check the
  mask occupancy and recovered `sigma_true` on a real UDS/COSMOS mosaic
  before the release; the acceptance metrics are already in the test file.
  Related: `dilate` sets both the detection smoothing radius and the
  dilation radius, so the mask is not nested in `dilate` -- worth splitting
  into two parameters if the tuning turns out to be sensitive to it.

- [ ] Finish scoping a `trial` run to its patch. The *read* is scoped
  (2026-08-12): `_read_image(path, box)` pulls only the patch off disk via
  `hdu.section` into a full-shape array, so pixel coordinates, slices and WCS
  are untouched, and the background/ivar and repair passes take the box. On
  MINERVA UDS F770W at radius 0.5', reads went 12.24 GB -> 0.88 GB and
  `load_data` to 8 s. Peak RSS only fell 22.3 -> 15.5 GB, because several
  full-grid operations downstream still touch every page and fault the whole
  mosaic back in — chiefly `sci_fit = sci_lo - bg` and the non-finite guard
  in `load_data`, the full-shape `ivar_hi`, and the lo->hi upsample in `run`
  (`pipeline.py:3357`, 876 Mpx = 3.5 GB per array). Scoping those to the box
  (in global coordinates, no coordinate changes needed) should get a trial
  run to ~1-2 GB and make it laptop-runnable. Measured breakdown in the
  2026-08-12 session notes.

- [ ] Aperture placement for `ap_flux`/`stampcor`: python measures at the
  catalog position while sources sit med ~1.3 px (40 mas) off it after
  astrometric shifts; measuring at the shifted/fitted position (as subphot
  does for its raw flux) is worth +~1% on ap_flux and +0.8% on stampcor
  coherence (2026-08-12 QA measurement). Decide convention and document.

- [ ] Saturated-star fragment stamps: `write_stamps` rows for spike
  fragments (QA patch, contiguous id blocks) hold smeared star structure
  while the fit table's stampcor for the same ids is sane — stamps-file
  content vs fit-time templates disagree for those sources. Not touched by
  the 2026-08-12 convolution change; inspect write path.

- [ ] Robust astrometry option: IRLS across scene anchors. Extended sources
  with asymmetric colour gradients produce a residual dipole formally
  identical to a shift, so no per-source test on residual size or shape
  separates them; they pull the scene's shift field the wrong way. The
  discriminator is coherence: a real offset is smooth in position (the
  premise of the GP/poly field), while morphology-driven pseudo-shifts are
  random per source. So reweight on *disagreement with the neighbours*, not
  on residual size. Per pass: solve, compute each anchor's implied shift
  `dx_i = -<Gx,w,r> / (a_i <Gx,w,Gx>)`, take the robust scatter of `dx_i`
  about the fitted field, apply a Tukey/Huber weight to that anchor's
  contribution to `AB`/`BB`/`bB`, re-solve. A genuinely offset region keeps
  full weight because its neighbours agree with it; one galaxy disagreeing
  with twenty anchors gets cut. Costs one extra assembly per pass. This is
  the case `astrom_leverage_cap` (2026-08-12) cannot cover: the cap bounds
  the influence of the brightest anchors without knowing which one is
  wrong, and does nothing in a scene where the offender is the only bright
  member.

- [ ] Scene shift iteration does not converge on r < 3' patches (found in
  verification v9, 2026-08-13; see `examples/minerva/verification/v9/README.md`
  and STATUS.md). Three of four UDS bands leave scenes moving after the 5
  allowed passes: F1800W scene 27 at 3.12 px, F1280W scene 12 at 1.44 px,
  F770W scene 66 at 0.61 px. It is a walk, not an oscillation about a
  stationary point (F1800W scene 27: (-1.27, 0.28) -> (-2.00, 0.88) ->
  (-1.88, 2.47) -> (0.86, 4.63) -> (2.38, 3.90) px), so it is a different
  failure from the biased-but-stationary 0.1-0.34 px cases below. At r < 1.5'
  F770W converged in one pass. Suspect the scene/model scale mismatch: scenes
  get fewer and larger toward longer wavelength (F1800W 31 scenes, sizes
  2-1964, median 471, spanning the whole 6' patch) while `astrom_model='gp'`
  keeps `length_scale` 400 px = 32" at 80 mas. `scene_max_size` (800) is
  already exceeded at both radii — v8's r < 1.5' F770W run produced a
  1061-template scene — so the cap is not the trigger. Candidates: make the
  size cap bind, scale `length_scale` with the scene, or raise
  `fit_astrometry_niter` / lower `astrom_damping` (currently 5 and 0.8).
  Check the IRLS item above first: a walk driven by a few high-leverage
  extended anchors would be cured by the same reweighting.

- [ ] Resolve the 2026-08-12 deep-review release gates in
  `docs/CODE_REVIEW_2026-08-12.md`. P1-01 (exact astrometric block system),
  P1-02 (final flux-only solve), P1-03 (background/IVAR source masking),
  P1-04 (non-finite preprocessing input) and P1-05 (template WCS provenance)
  are done (2026-08-12, see STATUS.md); next
  enforce template/WCS and catalog/segmentation invariants, remove OS4/cache
  assumptions, repair mock-validation independence, and reconcile the public
  docs only after the numerical gates pass. Each P1 fix needs the focused
  regression named in the report; do not close this umbrella item from a
  passing legacy suite alone.
- [ ] Footprint-limited saturation repair. On a trial-patch run the repair
  still fits every saturated star in the full mosaic; restrict the
  candidate list to the run footprint (r_trial + the halo-model reach,
  ~30", so stars just outside the patch whose halos reach in are still
  repaired and flagged). Complements the shared `repair_cache_path`
  (2026-08-12), which already makes bands 2..N of a campaign reload band
  1's repair; the footprint cut makes band 1 itself cheap on small
  patches.
- [x] `make_minerva_configs.py` writes `wht_hi` (verified 2026-08-13): the
  scheme-based template builds require the detection weight map, and the
  automatic `_sci`->`_wht` substitution cannot see through the `_bkgsub`
  suffix of the MINERVA `sci_hi` names. The generator emits it directly and
  all 53 CANFAR configs on disk carry it, so the verification driver's
  per-config stopgap is no longer load-bearing.
- [ ] F1280W scene 1: convolved templates lose their aperture flux. In the
  v7 run 612 of the 618 sources in scene 1 (a compact block at x 10982-12254,
  y 3712-5157) have `tot_stamp_1` ~ 7.96, i.e. only 12.6% of the convolved
  template's flux lands inside the 1.2" aperture against ~76% everywhere
  else, while the pre-convolution composite still holds ~92%. `psfcor_1`
  follows to ~7.3 (the plume above the 1:1 line in panel (e)) and `totcor_1`
  to ~8.4, so every aperture-derived column is wrong for those sources.
  Confined to one scene, so it is the kernel or low-res PSF assigned to that
  region, not a per-source property — segment areas there are if anything
  *larger* than normal (median 17 vs 12 px), which rules out tiny segments.
  Inspect that scene's matching kernel before trusting F1280W corrections.
- [ ] One `trial_center` per field, not per band. `make_minerva_configs.py`
  picks each band's own deepest fully covered MIRI patch, so the trial
  patches sit on different sky per filter: UDS centers span 3.53', COSMOS
  6.01' (EGS is uniform). At the `r_trial=1.5` used by v6/v7 that leaves
  2 of 6 UDS band pairs and 8 of 15 COSMOS pairs *completely disjoint* —
  F1280W and F1800W share no sources at all. Per-band numbers are sound,
  but any band-to-band trend (e.g. the bluest-worst ordering of the mock
  deficit) partly conflates filter with sky patch. Fix: choose the center
  from the intersection of all bands' MIRI coverage, then regenerate the
  17 configs. Note this invalidates the v6/v7 patch definition.
- [ ] Carry `ap_flux_total_<i>` into the verification recovery table
  (`build_source_recovery_table` keeps only flux/err columns), so the
  aperture estimator is checked against injected truth directly — the v4
  IDL leg shows it 3-7% brighter than IDL `flux_F`, attributed to IDL's
  partial-total convention, and the mock should confirm the absolute side.
- [ ] Aperture-floor template support. The v2 injected-truth mocks show the
  `psf_wings` scheme recovers point sources at 1.008 but extended sources at
  0.971: PSF-shaped wings cannot represent extended outer profiles, so
  segment truncation is partially reintroduced for anything resolved (IDL
  classic shares the mechanism and the bias — subphot's own comment says
  "the segmentation map does not contain all the flux"). Proposed cheap
  scheme variant: trust the *data* out to at least the catalog aperture
  radius (segment ∪ background-owned pixels within r_aper, neighbours
  nulled), PSF wings only beyond. Wren's competitive-dilation ownership is
  the SNR-graded version of the same idea; verification v3 (wren, all four
  bands, real + mock legs) measures whether it already removes the deficit.
  Decided 2026-08-12: default STAYS `psf_wings` (v3 showed wren adds a
  ~3% point-source deficit and does not cure the extended one). Still
  open: the hard aperture-floor variant, and whether the extended-source
  2-2.5% mock deficit warrants a scheme change later.
  Superseded 2026-08-12 by v7: the deficit is **4-5%**, not 2-2.5%. The
  earlier number was flattered by two things, both fixed — sizes were
  log-uniform 1-5 px (median FWHM 0.21", smaller than the F770W beam) and
  the mock segmap was dilated twice. v7 gives 0.9533/0.9573/0.9606/0.9637,
  with point sources unbiased (pull mu=+0.30, sigma=0.91) and the entire
  bias in the extended population. Decide the scheme question on these
  numbers, and re-check once Sersic injection lands (next item), since
  4-5% is still a Gaussian-profile floor.
- [ ] Inject Sersic sources, not Gaussians. `MockMosaic` paints every
  extended source as PSF (x) circular Gaussian
  (`mock_mosaic.py:1260`), which has almost no outer wings: outside 3
  half-light radii a Gaussian keeps 0.2% of its flux against 3.9% for an
  exponential disc and 21% for a de Vaucouleurs profile. The
  extended-source deficit comes precisely from PSF-shaped template wings
  failing to follow a real outer profile, so a Gaussian population is
  close to the best case and cannot size the problem. Add a Sersic option
  (n as a per-source truth column) before deciding whether the
  aperture-floor variant is worth it.

- [ ] Public-release cleanup: strip MINERVA-internal material from the public
  repo before advertising it. The repo is public and currently carries
  collaboration-internal content that Read the Docs does not publish but
  GitHub does: `examples/canfar/` (Science Platform manual with arc paths and
  collaborator handles), `examples/minerva/` (field configs and internal
  release layout), `docs/WREN_MERGE_PATH.md` and the other dev notes in
  `docs/`. Decide per item: move to a private repo, keep as sanitized
  examples, or drop. Removing them from history needs a scrub
  (`git filter-repo`), not just a delete commit. No credentials are in any of
  it (checked 2026-08-11); this is about internal data layout, not secrets.
  Also in code: `Pipeline.write_outputs` writes a `minerva_link` column into
  the scene-catalog CSV. Made configurable 2026-08-12 — `RunConfig
  .minerva_viewer` (`<field>/<release>`, default derived from the run name
  plus `minerva_release`, `""` drops the column); the URL now carries the
  field/release path, e.g. `.../uds/DR0/?ra=...&dec=...&zoom=7`. Still
  decide before release whether a public build should emit it at all.
  (The `matching_kernel.recenter` and `Pipeline.run` docstring staleness
  noted here earlier was fixed on 2026-08-11.)
- [ ] Code findings from the 2026-08-11 full docs verification (12 agents
  checked every docs page against source; these are source-side, docs already
  describe actual behavior). The seven actionable ones were FIXED on
  2026-08-11 (regression tests in `tests/test_verification_fixes.py`):
  * [x] ~~Operator precedence bug~~: `(weights <= 0) | np.isnan(...)` now in
    `fit.py` and `scene.py`; NaN-weight pixels zeroed again, fitting.md
    wording restored.
  * [x] ~~Double background subtraction~~: `_detect` no longer re-subtracts
    when `estimate_background=True` (run() already rebinds `self.sci`);
    user-supplied background levels still come off.
  * [x] ~~`PSF.gaussian` UnboundLocalError~~: `fwhm` now required
    (ValueError), second positional argument is `fwhm_y` (matches the
    existing `PSF.gaussian(n, fx, fy)` call sites, which previously passed
    `fy` into `theta`).
  * [x] ~~`PSFRegionMap` unpicklable~~: `__getstate__` drops `_prepared`/
    `_geoms` too; pickle and deepcopy round-trip. Nothing in the package
    crosses a process boundary with one, so this only mattered for user-side
    multiprocessing.
  * [x] ~~`AstroCorrect.fit` mutating caller config~~: pops now act on a
    copy; the dead `astro = AstroCorrect(config)` in `Pipeline.run` removed
    (pipeline shifts come from the scene solver, never this class).
  * [x] ~~`PSFSZ<i>`/`RCIRC<i>` wrong pixel scale~~: native lo-band scale is
    recorded before the upsample path rebinds `wcs[ifilt]`. Metadata-only
    bug — nothing in the package reads these keys back.
  * [x] ~~`id_scene` never assigned~~: `generate_scenes` stamps each
    template with its scene id; stamps column and `plot_subphot`/
    `plot_result` scene loops now see real scene membership.
  * `FLAG_HAS_NAN`/`FLAG_OUTSIDE_WEIGHT` declared, never set.
  * `pipeline.py:2275` `astrom_shift_tol` getattr fallback 0.02 disagrees
    with the FitConfig default 0.05.
  * [x] ~~Stale docstrings/comments~~ all fixed 2026-08-11: `ee_tmpl`
    comment, shift-direction comment, `FitConfig.astrom_model` comment,
    `saturate.py` module header (`r_out`, joint amp+shift fit), `rho_psf`
    polarity, ten-panel diagnostic layout, blank-to-zero comment,
    `read_wcs_csv` docstring path, `matching_kernel.recenter` default,
    `Pipeline.run` return arity, `RunConfig.save_stamps` "PSF cubes"
    comment, `verification.py` mosaic-pixel error message.
  * Possibly dead: `refine_center_from_donut` (never called),
    `getattr(tmpl, "parent_id", ...)` branches (no Template has parent_id).
- [ ] `utils.write_wcs_csv` is dead code: a leftover debug `continue` at
  `utils.py:1983` skips the row-building loop, so it writes a header-only CSV.
  Remove the `continue` (or the function; `reconstruct_wcs` is the working
  path and the one `DrizzlePSF.read_wcs_csv` auto-calls). (The related stale
  refs — `astrom_model` comment, `read_wcs_csv` docstring path — were fixed
  on 2026-08-11.)

- [ ] From the 2026-08-11 wren cross-reference
  (`scratch/wren/CROSSREF_2026-08-11.md`): (a) [x] ~~reopen from out_dir~~
  fixed — `from_config` accepts a directory with one `*.json`, and `run()`
  now stamps the executed config to `<out_dir>/<name>.json`; (b) relative
  config paths resolve against the process CWD, not the config file's
  directory, forcing scripts to `os.chdir` first — consider anchoring to
  the config's parent; (c) [x] ~~plot_subphot KeyError~~ documented in
  diagnostics.md; (d) document `astrom_damping` (0.8) in fitting.md WHEN
  the template branch merges — checked 2026-08-11: `git log main..template`
  still shows unmerged commits and `astrom_damping` is absent at main HEAD,
  so nothing to document yet.
- [ ] Post-merge follow-ups from the 2026-08-11 docs update of the template
  merge (`6e6cec6`): (a) `RunConfig.driz_hi` looks accidentally dropped by
  the merge — `6e6cec6^1` had it and passed `driz_image=str(cfg.driz_hi or
  cfg.sci_hi)` in `_ensure_dpsfs`; merged HEAD passes `str(cfg.sci_hi)` and
  the name is gone from `src/`. Restore or confirm intentional (docs
  document HEAD, no `driz_hi` entry). (b) `scene_<i>` fit-table fallback is
  `-1` for templates without a scene while `_template_fit_table` writes `0`
  — unify. (c) `ClassicParams.force_psf` and `WrenParams.containment` have
  no FitConfig knob and stay at defaults on every pipeline run — wire
  through or drop. (d) `Templates.from_image` accepts an `extension`
  argument it never uses. (e) `mophongo-repair` console script missing from
  the current env until the next `poetry install`.
- [ ] Multi-band pipeline stream. `run()` still fits multiple images
  sequentially (shared hi-res templates, per-band convolution/solve,
  `flux_<i>` columns), but the config/save/restore stream is one band per
  run: `RunConfig` has a single lo slot, `write_outputs` writes only band
  1's residual and stamps, and `load_fit` hard-codes band slot 0 (its
  `ifilt` parameter promises more than it delivers). Status matrix and a
  six-step upgrade path (per-band `bands` config list, filter-suffixed
  caches/outputs, slot-filling `load_fit`, three-image restore test) in
  `docs/MULTIBAND_PIPELINE.md`. Open question recorded there: one
  multi-band run buys convenience, not speed, over N single-band runs.

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

- [x] ~~`ee_psf_lo` never survives to the catalogue in the default path~~ —
  fixed 2026-08-11. `project_to_block_replicated_grid` and `Template.downsample`
  now carry `ee_psf_lo`/`ee_tmpl`/`template_norm` (plus `id_parent`/`id_scene`/
  `name`) across resampling, so at `k>1` the per-source encircled energy set by
  `convolve_templates` reaches `_update_catalog_with_fluxes` and
  `flux_<i>_total` divides by the per-source value instead of the filter mean.
  Regression test `tests/test_template_convolution.py::
  test_resampling_preserves_ee_metadata`. Found by the 2026-08-10 main audit
  (`scratch/wren/flux_estimator_comparison.pdf` §7.6); real-data validation of
  the divisor is the item below.
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
- [ ] `extend_mode='wren'`: `Templates.extract_templates` calls
  `schemes.wren_fill_radius` without `kernel_half_width` and with
  `WrenParams.aperture_radius_pix` at its `None` default, so out of the box
  `r_fill` collapses to `R_95` (20.6 px on UDS 40 mas) instead of the fork's
  `max(R_95, r_ap + kernel_hw)` = 29.0 px. That shrinks the ownership-contest
  disk, the halo-annulus reach and the cutout floor relative to the code being
  ported. The helper already takes both arguments; decide whether the pipeline
  should supply them (aperture from the low-res band, half-width from the
  matching kernel) or whether the collapse is intended, and say so in
  `template_schemes`. Flagged while rebuilding `template_comparison.tex` Fig. 6,
  which documents the fork's chain and now carries the discrepancy as a note.
- [ ] `positivity=True` puts 45% of the UDS DR0.1 F770W trial sources
  (1015/2242) at exactly flux 0, so the non-detection population has no
  negative fluxes and cannot be stacked or averaged without bias. Decide
  whether the default should be an unconstrained solve with the positivity
  clip reported as a flag, or a per-source switch above/below some SNR.
- [ ] Bright resolved sources in the upsample path leave sub-block
  checkerboard residuals: the model is smooth while the block-replicated
  80mas data is piecewise constant over each 2x2 cell, so on MIN_UDS48823 a
  2x2 block sum cuts the residual std to 0.60 of its per-pixel value (0.46 on
  MIN_UDS38103). It is a resolution-mismatch display/chi2 artefact rather
  than a flux error, but it inflates per-pixel chi2 on the brightest sources
  and makes residual panels look worse than the equivalent native-40mas monu
  stamps. Check whether the chi2/error estimate should be evaluated on the
  native low-res grid.
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
- [ ] The detection background is measured in `_load_detection_ivar` (by the
  same `get_bg_and_ivar` used on the lo-res side) and logged but **discarded**
  — it is the only `get_bg_and_ivar` call in the tree whose `bg` is not
  subtracted; the lo-res one feeds `sci_fit = sci_lo - bg`. Subtracting it
  would change `default` templates too, so it is left alone for now, but
  `template_comparison.tex` Sec. 8.1 lists "raw detection image, no
  background subtraction" as a wren defect: a sky pedestal enters the halo
  linearly with its area, over ~855 px. Decide whether the extended schemes
  should subtract it (COSMOS DR0.1 median bg -7.2e-4).
- [ ] `extend_mode='wren'`'s `WrenParams.containment` (wren's
  `PSFRegionMap.containment`, the detection-PSF stamp containment `c_det` in
  `flux_beyond_stamp`) defaults to 1.0 because this tree has no equivalent.
  `psf.stamp_encircled_energy` / `DrizzlePSF.ee_box` measure the same thing;
  wire one in if the wren truncation bookkeeping is kept. Note
  `template_comparison.tex` Sec. 8.1 flags wren's own `containment` as
  normalised to the parent grid and therefore ~+2.9% (F444W) high.
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
  off the injected value. **Recheck against the P1-01 fix (2026-08-12) before
  investigating further**: both surviving cases are blends or bright-extended
  sources, exactly where the old per-anchor blocks were biased, and the
  measured spurious order-1 field (0.05 px rms, 0.16 px peak on a synthetic
  blend) is the right order of magnitude for the smaller offsets here.
  Ruled out before that fix: iteration count (converges to a stationary
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
  - [ ] SED validation
    - [ ] SEDs of stars
    - [x] All-field COSMOS+EGS+UDS photo-z broadband stack in rest and observed
      wavelength; exact current EAzY pairing, per-field/filter provenance,
      filter-count planes, raw filter-footprint and connected pivot-cell
      reconstructions, a continuum-residual contrast view, and machine-readable
      stack products are implemented in
      `examples/minerva/plot_uds_sed_stack.py`
  - [ ] add in residuals in core for improved flux measurements (shift / psf errors)
- [ ] investigate blending in detection image
- [ ] Investigate template extension methods (Moffat fit and PSF dilation)
- [ ] End-to-end test with realistic mosaic data using `make_mosaic_dataset`
- [ ] Maintain an executable `examples/verify_pipeline.ipynb` notebook that
  demonstrates a pipeline setup from scratch and regenerates standard
  diagnostics
- [ ] Profiling speed + memory usage
  - [x] Full-field memory pass (2026-08-13): redundant template snapshots,
    the stored model image, the detection weight map, and the float64
    intermediates in `get_bg_and_ivar` / the upsample. See STATUS.md.
  - [ ] Saturation repair sets the peak on a trial patch and was not
    touched; profile `repair_in_memory` next.
  - [ ] `scene_fitter.build_normal` assembles a 138k x 138k `lil_matrix`
    entry by entry from Python. ~400 MB of Python objects for ~5M stored
    values, and the insertion loop is the slow part of scene generation;
    accumulate COO index/value arrays and build the CSR in one call.
  - [ ] `run()`'s finiteness guard on the images is behind
    `if images[i] is None`, so it never runs. Decide whether the check is
    wanted (it costs a full-field boolean pass per image) before fixing the
    condition -- turning it on may fail runs that pass today.
  - [ ] Finish the float32 sweep in `template_schemes.py`. The composite
    builders still upcast per source in about 15 places -- `stamp`,
    `ivar_stamp`, `psf_cut`, `D`, `P`, the `W` weight image -- so every one
    of the 138,610 extractions runs its wing fit at float64 and casts the
    composite back on the way out (`cut.data[sl_c] = comp.astype(...)`).
    Left out of the 2026-08-13 pass because these are transient per source
    rather than held, and each one feeds a small least-squares solve, which
    is the side of the rule where precision is wanted: the policy is float32
    for what is *stored*, float64 for what is *solved* (see
    `docs/precision.md`, and the PSF-matching case for the same split done
    deliberately). Decide per site which of the two a buffer is, rather than
    narrowing the file wholesale. Worth measuring the extraction peak first:
    the win is transient working set, not resident, so it may not move the
    ceiling at all.
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
