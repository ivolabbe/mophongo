# TODO

This file tracks future desired features, checks, and investigations.

- [ ] Carry the shift covariance through the fit table (2026-08-17). The live
  run now reports a finite `sigma_shift`, but `restore_scene_fit` restores
  only the shift coefficients and their normalisation, so a scene catalog
  re-emitted from a reloaded fit (`examples/canfar/jobs/scene_plots.py`, the
  recovery path) still reports NaN -- and `write_scene_catalog` promises the
  same file either way. Either add the covariance to the `SCENES` extension
  (flattened and padded like `shift_coeff`, 2x2 at the default order 0) or
  store the derived scalar and have `shift_error` fall back to it.

- [ ] Say why `astrom_floor` is NaN rather than only that it is (2026-08-17).
  It is NaN whenever the robust anchor pass declined to judge, which is not a
  failure but is indistinguishable in the catalog from one. On
  `examples/minerva/cosmos_f770w-robust` that was 792 of 1434 scenes: the gate
  is `max(scene_minimum_anchors, 2p)` = 7 there, and every scene with fewer
  usable anchors was refused (NaN for all n_anchor <= 7, finite from 8 up).
  Two things would settle it in the file itself: `n_anchor` counts
  `is_bright`, not the anchors that came through `measure_anchor_shifts` with
  finite information, so it does not match the gate the pass applied; and
  `AnchorWeights.reason`/`n_eff` are recorded on the scene and thrown away.
  `Scene.n_anchors()` now puts the same `is_bright` count on the residual
  panel of every scene PNG, so whatever this settles on should move both.

- [ ] Bring the verification scene loop's dpi down with the pipeline's
  (2026-08-17). `verification.py:1722` asks for `oversample=2.0` between
  `min_dpi=400` and `max_dpi=2400`, so its floor alone is a 24 Mpx canvas at
  3-7 s per scene, and a 2000 px scene asks for 1000 dpi -- 150 Mpx, a 600 MB
  RGBA buffer. `bbox_inches="tight"` renders it twice. It is capped at
  `scene_diagnostic_count` scenes so it does not dominate a mock run, but the
  settings are past what any viewer resolves. Left alone here because
  lowering them changes the resolution of an existing diagnostic.

- [ ] Clear the 21 stale configs in CANFAR `run1/config` (2026-08-16). They
  are older runs -- the whole `_v1.0` set for the three fields, plus
  `uds_770_dr0.1`, `uds_f770w_test`, `uds_f1800w_test`, `cosmos_f1000w_full` --
  and they still name `UDS_*`/`COSMOS_*` PSF patterns. The grids were renamed
  to `STDPSF_*` under them, so launching any of those names now matches zero
  grids and falls through to `autobuild`. They are distinct runs rather than
  superseded copies of the current 17, so regenerate them from their source
  configs if they are still wanted, or move them aside the way
  `mophongo.stale-archive` was; leaving them is a trap for whoever launches one
  next.

- [ ] Remove the two one-off job scripts left on CANFAR (2026-08-16):
  `jobs/rename_psfs.sh` (the grid rename) and `jobs/verify_cfg.sh` (reads
  deployed state from inside a container, which is how you check a config that
  `vcp` refuses to download after an overwrite). Harmless, but `jobs/` is
  meant to hold the campaign's scripts, not maintenance leftovers.

- [ ] Decide whether `PhotConfig.aperture_diam` should accept a column name
  (2026-08-16). Its type hint and comment say `str` names an input-catalog
  column of per-source sizes, but `Pipeline._resolve_image_ap_radius_pix`
  (`pipeline.py:4404`) tests only `int`/`float`/`np.ndarray`, so a string
  leaves `diam` as `None` and the aperture is silently sized by
  `aperture_ee` instead. Either implement the lookup (`aperture_catalog`
  already has one in `_resolve_catalog_ap_radius_pix`) or drop `str` from the
  annotation and raise on it. `docs/catalog.md` and `docs/pipeline.md`
  currently document the implemented behaviour, not the annotation.

- [ ] Key ePSF grid identity on the OPD file, not the MJD (2026-08-16).
  `stpsf`'s `load_wss_opd_by_date(..., choice="closest")` snaps a date to the
  nearest measured wavefront-sensing OPD, and JWST measures roughly every two
  days, so several MJDs resolve to one OPD and produce byte-identical grids.
  Measured on the 443-file `data/PSF` set: of 356 dated grids, 68
  (config, OPD) groups hold more than one MJD, so **75 grids duplicate another
  grid's content** -- 1.17 GiB of 3.31 GiB, 35%. Verified directly, e.g.
  MJD59945/59946/59947 of `MIRI_F770W_FOV8_GRID9_OS4` are all
  `R2023010203-NRCA3_FP1-1.fits` and compare equal to 0.

  The identity key is already on disk: every grid header carries `OPDFILE`.
  Have `PSFFactory` resolve the date to its OPD first and skip the build when a
  grid with that (detector, filter, OPD, FOV, GRID, OS) already exists, then
  point the extra MJDs at it. What that saves is CPU and disk, not bandwidth:
  `prewarm_opds` already fetches each OPD once into `STPSF_PATH` and the
  parallel phase is local. Worth checking whether the *date to OPD resolution*
  still needs MAST when the file is cached -- if it does, this cuts lookups
  too, which matters on OzStar where only the login node has internet.

  Two grids in the scan report the same MJD twice under one OPD
  (`MIRI_F770W` at 60086 and 59940). That is a straight duplicate rather than
  an OPD coincidence and wants identifying separately.

  While in there: the `.gitignore` negations for the PSF directory
  (`!data/PSF/*OS4_GRID25*`, `!data/PSF/*OS4_GRID9*`) use the pre-rename token
  order and match nothing now.

- [ ] Give badly determined scenes the shift their neighbours define, in a
  robust pass after all scenes are fit (2026-08-16). Interim for the global
  field below, and worth having on its own.

  Three ways a scene's shift comes out untrustworthy today, and none of them
  is visible in the catalog as such:

  * **Starved.** `merge_small_scenes` chases `scene_minimum_anchors` but
    `max_size` and `max_merge_radius` win over it (`scene.py:326-343`), so a
    scene that cannot merge without breaching them is left short of anchors.
    It then either solves a shift from one or two anchors, or -- with none --
    falls through to flux-only and its templates are never shifted at all
    (`scene.py:1552`), which biases its fluxes by whatever the local field was.
  * **Unconverged.** `flag_astrom = 1`: still moving when the pass budget ran
    out, so the stored shift is the last iterate.
  * **Internally inconsistent.** A high `astrom_floor`: the anchors disagree
    with each other beyond their formal errors, so the fitted shift is better
    determined on paper than in fact.

  `flag_astrom` is `-1` only when no shift was fitted at all, so a scene that
  solved from two anchors converges and reports `0` like any other.

  Do **not** gate on the failure modes separately. Unconverged is not the same
  as wrong: the linearized solve captures only part of a large offset per pass
  (`fit.py:48-57`), so a scene still moving at the budget may be tracking a
  genuinely large *real* shift, and overwriting it with its neighbours' would
  destroy a correct measurement. The question that decides the fill is whether
  a scene's shift disagrees with its neighbours by more than both can explain.
  That one test covers all three modes and spares the slow-but-right scene.

  Fold the internal inconsistency into the scene's error instead of gating on
  it. `astrom_floor` is per-anchor scatter *about the fitted field*, hence
  incoherent by construction -- a coherent error would have been absorbed into
  the field -- so it averages down over the anchors:

      sigma_eff^2 = sigma_shift^2 + astrom_floor^2 / astrom_neff

  All three are already in the `SCENES` extension. `sigma_eff` is then both the
  honest uncertainty on the scene's shift and the right inverse-variance weight
  for it as a donor.

  Then run :func:`mophongo.astrom_robust.robust_anchor_weights` one level up,
  with scenes as the anchors. It takes plain arrays --
  ``(eps, info, basis, chi2_red=..., min_anchors=5)`` -- so the scene table
  maps straight onto it: `eps` the per-scene `(dx, dy)`, `info` `1/sigma_eff^2`,
  `basis` the shift-field basis evaluated at the scene centroids, `chi2_red`
  the scene `chi2_dof`. One call returns the fitted field (`coeff`, and `field`
  evaluated at every scene), the rejections, a field-level `sys_floor` and
  `n_eff`. At anchor level `coeff` is documented as diagnostic only because the
  joint solve refits with the weights; at scene level there is no refit, so
  `field` *is* the fill value. `AGENTS.md` keeps the module a leaf taking plain
  arrays precisely so the non-joint path can reuse it -- this is that reuse.

  A robust fit also covers what a plain nearest-N mean does not: a donor whose
  own shift is wrong. The redescending Tukey step rejects it; an average has no
  defense. `min_anchors=5` already defaults to the natural neighbour count.

  Run it either per bad scene on its local donor set (order 0 = robust weighted
  mean, order 1 where the donors support it) or globally at low order. Local is
  safer on a mosaic, where one low-order polynomial across the whole field is
  too rigid; if run locally, weight donors by `exp(-d^2 / 2 L^2)` with
  `L = astrom_kwargs["gp"]["length_scale"]` so the fill is evaluated at the
  recipient's position rather than at the donors' centroid. Guard on the
  nearest donor: past roughly `2L` the fill is extrapolation, and leaving the
  scene unshifted beats moving it by a field never measured near it.

  Then apply the shift and re-solve that scene flux-only, exactly as
  `_refine_scene_astrometry` closes out a normal scene -- otherwise the fluxes
  belong to the unshifted templates. One pass is enough: the donors are frozen
  by the time the fill runs, so there is nothing to iterate against.

  Unlike a global field this costs no barrier. It runs over the finished scene
  list, after every scene has been solved and dropped, so the per-scene
  independence that `docs/SCALING_FIXED_MEMORY.md` depends on survives.

  Needs a new column rather than a new `flag_astrom` value: a scene can be both
  starved and unconverged, and `flag_astrom` is a `0`/`1`/`-1` sentinel whose
  meaning is already documented and inherited per source as `flag_astrom_<i>`.
  Add `flag_shift_source` (`0` measured, `1` inherited from neighbours, `2` no
  shift available) and inherit it the same way. A borrowed shift must not read
  as a measured one.

  With a global basis at scene centroids this interim *is* a global polynomial
  shift field, so it is a step toward the Schur-complement solve below rather
  than scaffolding to throw away.

- [ ] Fit one global astrometric shift field across all scenes, by summing
  per-scene Schur complements (2026-08-16). The field the joint path fits
  today is scene-local: `make_scene_basis` (`scene.py:480`) centers and scales
  a Chebyshev basis on the scene's own bright members, so every scene gets an
  independent order-0 offset and scene boundaries acquire a statistical
  meaning they should not have. `astrom_model` is inert here -- the joint path
  reads `astrom_kwargs["poly"]["order"]` directly (`scene.py:1475`) and never
  branches on the model -- so a GP is reachable only through the non-joint
  `AstroCorrect` path, which measures shifts from residual centroids rather
  than from the fit itself.

  A global field does not require solving every scene at once. Fluxes are
  block-disjoint across scenes, so each scene's flux block can be eliminated
  on its own and the reduced systems added:

      S_s   = BB_s - AB_s^T A_s^-1 AB_s
      rhs_s = bB_s - AB_s^T A_s^-1 b_s
      (sum_s S_s + K^-1) beta = sum_s rhs_s

  `beta` holds the global field's coefficients and `K^-1` the GP prior
  precision (identity times `astrom_reg` recovers the polynomial case). This
  is exactly the joint solution over all scenes and one shared field, not an
  approximation of it. Fluxes come back per scene by back-substitution, which
  is the flux-only solve that already closes out each scene.

  Work involved:

  * `make_scene_basis` must evaluate global features at absolute positions
    instead of scene-local Chebyshev rows, or scenes are not in a common
    coordinate system. Fixed RBF centers (or random Fourier features) at
    `astrom_kwargs["gp"]["length_scale"]` spacing; the existing polynomial
    stays available as a global basis with a single center and scale.
  * `assemble_scene_system_AB` (`scene.py:818`) sizes the shift block from
    `len(cheb_basis(0, 0, order))` (`scene.py:947`); it must take `p` directly.
  * New per-scene reduction: `A_s^-1 AB_s`, one solve with `2p` right-hand
    sides, then accumulate into the global system. Use a compactly supported
    kernel, or drop centers beyond a few length scales of the scene, so each
    scene touches only a handful of features and the accumulation stays banded
    -- otherwise `p` is mosaic-wide and the per-scene cost is prohibitive.
  * `SceneFitter.solve` regularizes the shift block with scalar `astrom_reg`;
    a GP prior needs a matrix precision.
  * `AstroCorrect.build_poly_predictor` (`scene.py:1516`) needs a kernel
    counterpart to evaluate the field at faint members.
  * `_shift_amplitude` recovers the order by inverting `n_terms(order)`
    (`scene.py:1724`), and `scene_minimum_anchors` is derived from the same
    order (`fit.py:218`). Both break under a non-polynomial basis.
  * Wire `astrom_model` into the joint path so 'gp' selects the kernel basis
    rather than being silently ignored.

  The structural cost is the loop nesting. Today the astrometric iteration is
  entirely inside one scene -- `for scn in scenes: _refine_scene_astrometry(...)`
  at `pipeline.py:5128` runs a scene's passes, its convergence test and its
  closing flux solve as one self-contained unit, and the comment at
  `pipeline.py:5110` records that the barrier was removed on purpose so a
  scene can go to a worker process. A global field inverts that: assemble and
  reduce all scenes, solve `beta`, apply to all scenes, test convergence,
  repeat. Each pass is still parallel across scenes, but there is a sync point
  per pass, and templates must stay resident across passes instead of being
  dropped as each scene finishes -- which is the streaming property
  `docs/SCALING_FIXED_MEMORY.md` depends on. Convergence also becomes global:
  `Scene.astrom_converged` and `astrom_niter` stop meaning anything per scene,
  and every scene pays the worst scene's pass count. Partly self-correcting,
  since a sparse scene inherits its shift from its neighbours instead of
  grinding out its own, but the memory argument does not recover.

  A cheaper variant reaches the same fixed point by block coordinate descent:
  pool the per-anchor implied shifts and Fisher information that
  `measure_anchor_shifts` (`scene.py:593`) already computes from the flux-only
  residual, fit the field to that global table, then apply and re-solve. The
  anchor table is a diagonal approximation to `S_s` -- per-anchor 3x3 systems,
  neighbour-conditioned but without the full scene coupling -- so it converges
  to the joint answer without matching it pass for pass. It carries the same
  barrier, so the only thing it buys is avoiding the Schur reduction. Worth
  measuring before committing to either; `chi2_red` should gate entry to the
  table in both, or one wrong template pollutes the field out to a length
  scale.

  Both variants still linearize `T(x + d) ~ T + d . grad T`, so the outer
  `fit_astrometry_niter` loop stays.

- [ ] Take the flux-flux block of the anchor local systems from `A` and `b`
  (2026-08-18). `measure_anchor_shifts` now assembles each anchor's local
  system from slice intersections rather than from padded buffers, which is
  7x. Most of what is left is still the flux-flux block, and it does not need
  computing at all: `<T_j, W, T_k>` *is* `A[j, k]`, and `<T_j, W, resid>` *is*
  `(b - A @ flux0)_j`, both already built for the scene's flux solve. In a
  crowded scene the flux columns are ~95% of a local system -- 134 of 138
  columns on the 920-template benchmark -- so the remaining integrals would be
  the gradient rows only. Measured at 40-129x against the padded version
  (`scratch/bench_anchor_fast.py::measure_anchor_shifts_fastest`), against 7x
  for intersections alone.

  Two things to weigh before doing it. The signature grows `A`, `b`, `flux0`,
  and with them an invariant that is currently implicit: the matrix passed
  must be the one the residual was built from. Both come out of
  `Scene._robust_anchor_weights` three lines apart, so it holds today, but it
  would break silently rather than loudly. And `build_normal` forms
  `cut * w * cut` in float32 before widening the accumulator, so reusing `A`
  moves the flux-flux block by ~1e-7 relative where the intersection version
  matches the padded one to ~1e-14; `test_local_systems_match_the_padded_reference`
  would need a tolerance rather than staying exact.

  Only worth it if the astrometry phase is still the run's bottleneck after
  the three changes already made. On the 920-template benchmark it would take
  the robust pass from ~74 ms to ~50 ms against a scene solve of ~440 ms.

- [ ] Accumulate the shift coefficients across astrometric passes
  (2026-08-16). `Scene.solve` overwrites `self.shifts` on every pass
  (`scene.py:1512`), so a scene that took two passes keeps only the second,
  and the `SCENES` extension inherits that. The total offset survives only as
  the per-template `dx`, `dy`, which is why `_scene_shift_samples` refits
  rather than evaluating the stored field. Accumulating would make the stored
  coefficients the whole solution and let the catalog's `dx`, `dy` come from
  it directly. Check first whether `make_scene_basis` returns the same
  `(x0, y0, Sx, Sy)` on every pass -- it is built from `position_original`,
  which does not move, so the coefficients should be addable, but the damping
  factor has to go in on the way.

- [ ] Point the MINERVA configs at their FITSMap roots (2026-08-16).
  `RunConfig.minerva_viewer` takes a full URL and no longer guesses, so a
  config that leaves it unset now writes no `minerva_link` column at all.
  COSMOS serves from `/cosmos` and UDS from `/uds/DR0`; the 17 configs under
  `examples/minerva` need the value filled in to get their links back.

- [ ] Add direct tests of the `PSFFactory` public entry points (2026-08-14,
  from the Aperpy clean-implementation audit): `build()` called on its own
  rather than through `from_csv()`, and `filename()` across OS1/2/4/8 -- it
  writes `OS4` for every oversampled grid. The provenance half of that audit
  item is gone rather than fixed: grids no longer carry an exposure hash,
  date mode or FOV card, so there is nothing left to keep in step (2026-08-15).

- [ ] If the clean Aperpy-style pipeline is approved, implement it as a sibling
  `AperturePipeline` following
  `docs/APERPY2_CLEAN_IMPLEMENTATION_REPORT.md` (2026-08-14), with
  `DrizzlePSF`/`PSFRegionMap` as the strict JWST default. Phase zero must add a
  provider contract, versioned/content-validated map persistence, strict
  no-coverage behavior, and absolute arbitrary-radius `ee_at`/`radius_at_ee`;
  then recover SEP aperture/Kron semantics in a one-region fixture before
  enabling spatial matching. Treat diagonal `K**2` variance as an
  approximation and calibrate release errors with empty apertures by
  depth/kernel class. Keep HST empirical PSFs, survey flags, dust/Gaia, and
  EAZY as explicit fallbacks or downstream adapters.

- [ ] Homogenize the ePSF grids on one field of view (2026-08-14). 118 of the
  369 science grids are the older, smaller build -- UDS entirely (both F444W
  detectors, all four MIRI bands), plus COSMOS's 22+22 NIRCam and 16 F770W
  epochs -- at FOV4/FOV8 against FOV6/FOV11 elsewhere; EGS is already uniform.
  Nothing is wrong today: the epochs of a family are disjoint, so no MJD is
  served by two grids, and the fit crops to `psf.size` either way (a 101-px
  build agrees with a 65-px one to 1.1e-4 of peak over the shared region).
  Rebuild them at the default so the set is one thing rather than two. The 124
  FOV30 halo grids stay as they are.

- [ ] Turn the theoretical-target deconvolution experiment into a calibrated
  science product only if a use case justifies its noise/ringing cost
  (2026-08-13). The UDS F444W patch proves the mechanics and the tradeoff:
  lambda=1e-3 gives 0.139" / 2.22x source-masked field scatter / 1.87x
  empty-aperture RMS, and lambda=1e-4 gives 0.126" / 4.36x / 2.33x. The
  closest support-safe scan point is 0.116" at 9.69x / 3.85x, while the
  nominal 0.109/0.112" endpoints are rejected by absolute edge L1. Both
  retained solutions have about 0.28 integrated negative response. These
  noise statistics include correlated sky and residual ringing rather than
  representing propagated instrumental noise. Before release:
  estimate a data-informed signal/noise PSD rather than the current flat-prior
  Wiener (= Tikhonov), propagate or characterize the full correlated-noise
  covariance rather than inventing a diagonal WHT, run injected-source and
  reconvolution closure tests, and test soft partition-of-unity blending if a
  larger mosaic shows PSF-region seams. Also settle the existing F444W wing
  deficit/grid-support TODOs first: inverse filtering amplifies PSF-model error
  along with the sky. The reusable API and real-patch driver are complete;
  these are science-calibration gates, not missing mechanics.
  The ringing scan adds one candidate if a product is pursued: expose an
  explicit smooth post-Wiener transfer taper (generalized Gaussian with a
  documented half-power frequency), and select it jointly on realized width,
  response negativity, correlated-noise gain, L1, and edge support.  Do not
  present this as an exact Gaussian target: even the best compact spatial
  taper retains a roughly -4.6% trough, while the nearly ring-free Fourier
  setting costs 1.65x white-noise gain and 256-pixel support.  A nonlinear
  positivity-constrained reconstruction would be a separate, prior-dependent
  image solver and must be reconvolved and closed against the native image.

- [ ] Parallelise the PSF build across *patterns*, not only within one
  (2026-08-13). `PSFFactory(workers=N)` now fans out over `(detector, date)`,
  which is safe because each job writes a uniquely named file, and the OPD
  fetches are pre-warmed serially so the pool does not race MAST. What is
  still serial is bands of a *field*: they all derive the same `psf.pattern_hi`,
  so `uds_f770w` and `uds_f1000w` build identical F444W filenames into one
  `psf.dir` and would tear each other's files. Prep sidesteps it by building
  one band per field first. Doing better means a lock or a claim file per
  target path, at which point every band could build concurrently and prep
  would only need to exist for the repair.
  * The per-band `pattern_lo` grids have distinct names and are already safe
    to build concurrently across bands.
  * Measured 2026-08-13 on UDS F444W (25-PSF grids, 10-core laptop): serial
    34.2 s/grid (137 s for 4 grids, peak RSS 2.84 GB); four workers 18.9 s/grid
    (75.8 s, peak 2.88 GB), a 1.81x speedup. Not 4x: pinning each worker to one
    thread roughly halves a single grid's own speed, since an unpinned build
    already takes about 1.6 cores. Memory is a non-issue. Extrapolating, the
    ~416-grid release is about 4 hours serial and 2 hours on four workers --
    both well under the 9 hours estimated before measuring.
  * `psf.workers` defaults to 1 everywhere. 4 is the number to set: it matches
    a CANFAR container's `cores_for` of 4, and is modest enough for an OzStar
    login node, where the build runs `nice`d on a shared machine. 16 is wrong
    on both -- 4x oversubscription on CANFAR, antisocial on a login node.
  * Not yet measured: whether pinning is the right trade. An unpinned pool
    would demand ~1.6 cores per worker, which suits a 10-core laptop and not a
    4-core container. The shipped default should suit the container.

- [x] Missing ePSF epochs are built; nothing else is (2026-08-15). The grids
  already on disk (and on arc, and on `/fred`) were built in `cluster` mode --
  UDS F770W has 9 where `all` wants 17, COSMOS F444W 44 against 78 -- and
  `_load_epsf` used to autobuild only when *nothing* matched the pattern, so
  those bands could never gain the missing dates on their own.
  `Pipeline._missing_psf_dates` now asks the only question that matters:
  `dates_from_csv(csv, psf.date_mode)` minus the `_MJD` tokens on disk. What
  is missing is built, one factory call per epoch, and what is there is left
  alone. A band holding 99 of 100 epochs costs one grid.
  * This also retires the provenance experiment. Grids carry no
    `HIERARCH MPH` cards, no exposure-list hash and no date mode: a grid is
    its detector, filter, epoch and field of view, all of them already in its
    filename, and none of them a function of the rest of the exposure list.
    Stamping the list meant that adding one frame invalidated every grid built
    from it, which is the opposite of what the file records. FOV is not
    compared either -- a drizzled stamp smaller than its grid is fine, and
    `_check_psf_size_fits_grids` warns when it is not.
  * Still worth measuring: whether `cluster` and `all` differ detectably in
    the photometry. If the wavefront drift between adjacent cluster dates is
    small, `cluster` is the cheaper convention and the default is the thing to
    reconsider, not the grids.

- [ ] Measure the OzStar side of the campaign (2026-08-13). `examples/ozstar/`
  is written and running but has no measured numbers in it yet: staging wall
  time and volume per field, peak RSS per full-field band from
  `sacct --format=MaxRSS`, and whether 16 cores buys anything over 4 now that
  the inputs sit on Lustre rather than an NFS-mounted `/arc` (the CANFAR
  measurement of 0.2 of a core was dominated by waiting on `/arc`). Those
  belong in `examples/ozstar/README.md` next to the request defaults.

- [ ] `examples/ozstar/` has no test coverage, for the same reasons and with
  the same one worth testing as the CANFAR toolkit: `has_shared_grids` is pure
  given a name list. It is now duplicated in both campaign scripts because the
  OzStar one must ignore local grids (nothing uploads them there) - if a test
  is written, that is the moment to decide whether the two should share an
  implementation.

- [x] Per-field prep step, both toolkits (2026-08-13). `STEPS["prep"]` and
  `STEPS["repair"]`, `$STEP` in both job scripts, `submit.py run --step`, and a
  `prep` phase in both campaigns. CANFAR waits on it laptop-side; OzStar makes
  each band job depend on its field's repair. See `docs/campaigns.md`.
- [ ] Decide whether `scene_plots` can stay on for a full field (2026-08-13).
  It is what killed all ten cosmos/uds bands of the CANFAR v1.0 campaign:
  each wrote its fit table, residual and stamps, then died rendering Lupton
  RGB composites for several hundred scenes on top of ~35 GB resident
  (cosmos_f1280w got 120 of 380 PNGs out). `scene_plots=false` plus the
  per-field RAM below gives a clean run - egs_f560w completed at 29.7 GB
  peak. Worth re-testing rather than leaving off permanently: `7784f99`
  added `_residual_memmap`, which removes the 3.5-4.5 GB anonymous residual
  the campaign's commit still held, and the plot loop reads only a bbox
  slice of `self.images[0]` and `self.segmap` while both stay fully
  resident. If it still does not fit, that loop is the place to read those
  two memmapped, since it never needs more than the scene's own box.
  The 2026-08-17 cap changes what is being re-tested: `scene_plots_max=200`
  puts ~200 figures through that loop instead of ~1600, at a canvas sampled
  to the scene rather than a fixed 4500x3000, so a band now spends ~5 min
  there rather than ~55. The peak resident is unchanged -- it is one figure
  at a time either way -- but the window in which a band can die shrinks by
  10x.
  Per-field RAM itself is done: `submit.py::ram_for` gives 64 GB standard
  and 82 for EGS, `--ram` overrides, and `campaign.py` passes nothing unless
  asked so each field takes its own size. Still to record: the measured peak
  per field in `examples/canfar/README.md` next to the UDS number, once
  enough EGS bands have finished to have one.

- [ ] `examples/canfar/` has no test coverage. The 2026-08-13 changes
  (`kill`, `push --src-only`, arc-aware `has_shared_grids`, `--skip`) were
  verified by invoking them, not by tests. `has_shared_grids` is the one
  worth a real test: it is pure given a name list, so it needs no network,
  and getting it wrong either serialises a field for hours or lets several
  bands race on one `psf.dir`.

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
  sources on synthetic correlated noise (`tests/test_catalog.py`),
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

- [ ] `write_stamps` corrupts the tail of a full-field stamps file. The
  2026-08-13 UDS F770W full-field round trip wrote 138,610 sources / 11.7 GiB
  of template pixels and astropy raised
  `RuntimeWarning: overflow encountered in scalar add` from
  `io/fits/column.py:2274` (`_offset += descr_output[idx, 0] * _nbytes`),
  which computes heap offsets for the variable-length `tmpl_hi`/`tmpl_lo`
  columns. The file reads back and most rows are fine, but 64 of the last 200
  rows come back all-zero against 1 in 140 sampled across the whole file --
  a 45x tail enrichment -- and none of those 64 has `FLAG_SUM_ZERO` set, so
  the pipeline believed their templates were non-zero when it wrote them.
  Their fitted fluxes are in the fit table, so only the stamps are affected.
  Treat any full-field stamps file written before this is fixed as suspect in
  its tail. Likely a 32-bit heap offset in the `P` descriptor: `Q` descriptors
  (64-bit) exist for exactly this, or the file needs splitting per band/chunk.
  Reproduce with a full field and `save_stamps: true` (the default).

- [ ] Saturated-star fragment stamps: `write_stamps` rows for spike
  fragments (QA patch, contiguous id blocks) hold smeared star structure
  while the fit table's stampcor for the same ids is sane — stamps-file
  content vs fit-time templates disagree for those sources. Not touched by
  the 2026-08-12 convolution change; inspect write path.

- [x] Robust astrometry option: IRLS across scene anchors. Implemented
  2026-08-16 as `FitConfig.astrom_robust` and `mophongo.astrom_robust`; see
  STATUS.md. Three things the original note got wrong, worth keeping:
  * "no per-source test on residual size or shape separates them" is only
    half right. Residual *size* does not separate them; residual size **after
    the anchor's own displacement is projected out** does, because a pure
    displacement lives entirely in the span of the two gradient columns.
    That is `chi2_red` from `scene.measure_anchor_shifts`, and it is the
    second of the two weighting layers.
  * "Costs one extra assembly per pass" -- it costs none. Eliminating the
    fluxes gives `(B'WB - B'WA(A'WA)^-1 A'WB) beta = B'W r0`, so the shift
    block is driven by the *flux-only* residual and the weights can be
    measured before the joint system is assembled. One extra flux-only solve,
    no loop around the joint one.
  * A plain IRLS as described would not have worked: started from the
    information-weighted least-squares fit, the reweighting masks, because
    the dominant anchor *is* that fit and the honest anchors carry the large
    residuals. It needs a leverage-blind high-breakdown start first.

- [ ] `astrom_robust` stays off by default until it is shown to help on real
  scenes. First real-data test (2026-08-16, COSMOS F770W scene 16, refit
  through `Pipeline.refit_scene`, 236 templates both sides) says it hurt:

      run       (-0.017, +0.002) px, 1 pass
      baseline  (+0.221, -0.088) px, 2 passes, converged
      robust    (+0.879, -0.870) px, 3 passes, converged
      chi2 182175 -> 295274   (dchi2 +113099)
      11 anchors, 0 rejected, systematic floor 0.61 px
      flux: median +0.000 sigma, one source 48 sigma

  The mechanism is the one measured synthetically: 11 anchors scattering
  0.61 px about the field is broad disagreement, not one liar, so the floor
  absorbs it, flattens the weights and rejects nothing -- diluting information
  rather than protecting against an outlier. 0.61 px is also six times
  `astrom_shift_tol`, so this scene's anchors do not agree at the level the
  loop is trying to converge to, which is worth understanding on its own.

  An earlier version of this measurement was void: the band weight was on the
  wrong grid (see STATUS), which corrupted the baseline astrometry to
  (+0.03, -0.92) px and pruned 48 covered templates.

  What the test cannot settle: one scene with 11 anchors is not a sample, and
  24 of the run's 260 sources are unavailable to any refit (`ff3b8d4` version
  skew). Needs a clean run on current code, several scenes, and at least one
  scene of the shape the scheme is for -- one dominant bright anchor against
  several agreeing ones.

- [ ] Reduce the shrinkage bias on a blended anchor's implied shift.
  `measure_anchor_shifts` fits each anchor conditionally on its neighbourhood
  -- a flux column per overlapping template, a free displacement per
  overlapping anchor -- but the local system still stops at one level of
  overlap and uses the union footprint in place of the global flux
  constraint. What survives is a *shrinkage toward zero*: at 4 px separation
  (1.6 sigma) with dominance 0.65, blended anchors read 0.12-0.145 px against
  a true 0.20, in both axes, whatever the pair's orientation.
  Shrinkage being coherent is what makes it dangerous, and it is why
  `astrom_isolation_thresh` cannot be retired in favour of the robust pass
  (2026-08-16 measurement, see STATUS.md): every blended anchor is biased the
  same way, so they agree with each other, and robust weighting is majority
  rule. With 6 blended against 3 clean anchors the fit follows the blended
  ones and the systematic floor moves weight further toward them (share
  0.627 -> 0.690); beta error 0.052 -> 0.086 px with the robust pass on,
  against 0.0017 px when the cut excludes them. Extending the local system to
  two levels of overlap is the direct attack; the bias is a bias, not a
  variance, so no amount of information weighting removes it.

- [ ] Veto a scene's shift when its only anchor cannot be trusted. Robust
  weighting cannot reach this case: with one anchor the weight is a global
  scale, and a global scale cannot move where the field lands, so a scene
  whose sole bright member is a colour-gradient galaxy still gets that
  galaxy's pseudo-shift. The information is available -- `chi2_red` from
  `scene.measure_anchor_shifts` is per-anchor and needs no neighbours -- but
  it has to drive a decision rather than a weight: either fit no shift for
  that scene, or shrink it toward a field-level prior taken from the
  neighbouring scenes. The second is the better answer and needs cross-scene
  state that `Scene.solve` does not currently have.

- [ ] The `lev_w` two-power split is inconsistent under the flux
  marginalization. `BB` scales as `sqrt(c_i c_j)` and the RHS correction
  `AB' A^-1 b` as `c_i`, both correct, but the LHS correction `AB' A^-1 AB`
  then scales as `c_i c_j` where the information it corrects scales as
  `sqrt(c_i c_j)`. One `AB` matrix cannot satisfy both requirements. Measured
  (2026-08-16, `scratch/astrom_robust/lev_sweep.py`): beta departs from a genuine
  downweight by ~10% of the capped anchor's contribution at the shipped
  `astrom_leverage_cap=0.9`, rising to ~36% at `c=0.5`, when a neighbour sits
  about one FWHM from the anchor. Zero when nothing overlaps, zero at
  `c_i = 0`, and bounded by the flux-shift degeneracy fraction as `c -> 0`.
  The fix is two `AB` matrices (`sqrt(c_i)` for the LHS, `c_i` for the RHS)
  and explicit Schur elimination in `SceneFitter._solve_flux_and_shifts`
  instead of the single `K = sp.bmat(...)` solve -- roughly `nB + 2` sparse
  back-solves, so not obviously more expensive. Worth doing if soft weights
  are ever pushed well below 0.9; hard rejection is exact and sidesteps it.

- [ ] `alpha0` for the derivative columns is the diagonal-only flux `b/d`, so
  for a blended anchor it reads high by whatever leaks into its footprint and
  the fitted shift comes back low in proportion. Measured 2026-08-16
  (`scratch/astrom_robust/verify_claims.py`): 2.7% seed error, 2.5% shift
  error, flat in shift magnitude. Feeding the joint solve's own fluxes back
  and re-solving gives 0.02%, at the cost of a second joint solve. The
  flux-only solve is *not* the fix -- it is worse above ~0.1 px (10.7% at
  0.5 px), because the unmodelled dipole gets absorbed into the blended
  neighbour's flux. Kept rather than dropped because it bites in two places
  the damped iteration does not cover: a run with `fit_astrometry_niter=1`
  takes the full 2.5%, and the error varies per anchor with contamination, so
  it mis-weights anchors against each other by ~5% in information. Otherwise
  low priority -- it is 2.5% of the *step*, and the fixed point is unaffected.

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
  Try `astrom_robust=True` first (2026-08-16): a walk driven by a few
  high-leverage extended anchors is exactly what it cures, and the
  per-anchor table it logs (implied shift, information, misfit, weight) says
  whether that is the cause before anything else is changed. Note the seed
  bias measured below is *not* a candidate -- it under-estimates each step by
  a fixed 2.5%, which slows convergence rather than driving a walk.
  `scene_max_merge_radius` (2026-08-13) is now the direct instrument for the
  scale-mismatch hypothesis: it bounds the scene's longer side rather than its
  template count, which is the quantity the GP length scale should be compared
  against. It is already on at 1500 px (~4x `length_scale`), so re-run v9 and
  read off whether the walk survives before reaching for the damping.

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
  SNR > 25 line. Still open, moved to its own items: `psf.size` = 4" at 18 um,
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
  `psf.size` at 4.0" for every MIRI band because the same value sets the
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
  outside the model entirely versus 1.5% for F770W. Enlarging `psf.size` past
  ~4" for F444W measures the grid edge rather than the PSF. Regenerate the
  NIRCam grids at a larger FOV if the hi-side wings start to matter.
- [ ] Decouple PSF support from kernel support. `psf.size` currently sets
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
  **F356W-to-F444W cross-check (2026-08-14):** colocated real UDS model stamps
  confirm that this adjacent-band direction is smoothing. Wiener at
  `reg=1e-3` matches the 0.166x0.160" F444W target to 0.167x0.162", has
  `sqrt(sum(K**2))=0.313`, normalized maximum Fourier gain 1.000, and point
  projection 1.005. The stock SplitCosineBell broadens to about
  0.215x0.213" and biases the projection by 11%; an optimized window
  (`alpha` about 0.16, `beta` about 0.525) is faithful. Do not build a
  production F356W region map from the current two MJD 59967 GRID1 files:
  the science WCS table spans 23 date groups and needs date-aware grids first.
- [ ] Scene partitioning is not reproducible run to run (from wren's
  `CHECKLIST.md`; the mechanism is live here and recorded nowhere else).
- [ ] The flux-block ridge biases faint sources low: -33% at
  `d_i/median = 1e-6` (also from wren's `CHECKLIST.md`). **Confirmed still
  live** (2026-08-10): the `flux-bug` fix removed `astrom_reg=1e-4` leaking
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
- [ ] `tests/test_pipeline.py::test_pipeline_multitemplate_pass` no longer
  exercises a multi-template pass: `_add_templates_for_bad_fits` and the
  `multi_tmpl_*` config knobs were removed in the 2026-08 cleanup (the only
  call site had been commented out long before). Either reinstate the feature
  or rename the test to what it now covers, a plain pipeline run.
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
  new machinery. Template support (`psf.size`, currently null = 8" stamps) is
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
  biased point), faint templates in the shift blocks (`astrom_minimum_snr=15`
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
    - [x] Add MIRI-specific visibility/QC for F560W, F770W, and F1000W with a
      shared local display stretch, empirical SEM/counts, gentle field-local
      winsorization, population-scatter-regularized weighting, and explicit
      raw/capped inverse-variance failure diagnostics
    - [x] Compare 0.05/0.035/0.025 fractional redshift bins and repeated
      split-half equal/winsor/scatter-weighted/IVW estimators; keep 0.035 as an
      experimental companion, 0.5% winsorization as the conservative robust
      view, and formal IVW only as a demonstrated failure control
    - [ ] Add whole-galaxy field-stratified bootstrap uncertainty (500 final
      draws), delete-one-field checks, and spatial 4x4 tile jackknives; preserve
      each galaxy's correlated wavelength cells during every resample
    - [ ] Add labeled redshift-quality sensitivity stacks (spec-z or narrow
      EAzY posterior), and use full P(z) multiple imputation only when the exact
      EAzY posterior files are staged; do not weight the primary stack by
      photo-z risk or pretend quantiles are Gaussian PDFs
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
- [ ] Fixed memory budget for arbitrarily large fields, and parallel scene
  solves. Design note: `docs/SCALING_FIXED_MEMORY.md` (2026-08-13). Only the
  loop inversion below has been built. Peak today is 46.5 GB on a full UDS field and
  ~72 GB on COSMOS, both linear in field area; the proposal is tiled template
  build + global scene partition on `ATA` alone + scene-streaming solve, which
  bounds the peak by the largest scene rather than by the image (~2-5 GB at
  eight workers). Threads are measured to be the wrong tool for the scene loop
  (`build_normal`-shaped work runs 0.47-0.66x with 4-8 threads); processes over
  memory-mapped inputs are. Prerequisites, each worth doing alone and each
  blocking part of the above:
  - [x] Astrometric loop inverted to scene -> pass (2026-08-13): one scene is
    now refined to convergence before the next starts, so a scene is a
    self-contained unit of work with no barrier between scenes. Bit-identical
    against HEAD in three configurations; see STATUS.md.
  - [ ] Chunk `assemble_scene_system_AB` over row bands. `Bq`/`Bl` (0.61 GB
    worst case) and the `Bq[k] * wbuf` temporaries are the only per-scene term
    that scales with spatial extent, so they multiply by the worker count.
    Blocks parallelism.
  - [ ] `Scene.model_image` allocates float64 over the scene bbox (150 MB for
    the widest); float32, or accumulate straight into the residual.
  - [x] Bound scene *extent*, not only `scene_max_size` (2026-08-13):
    `scene_max_merge_radius` is now one knob read three ways -- split a scene
    wider than it, search no further than it for a merge partner, refuse a
    merge that would exceed it. Default unchanged at 1000 px.
  - [ ] Validate `scene_max_merge_radius=1500` as an extent limit on a full
    field. It is now on for every run, and it changes fluxes: the split cuts
    couplings the threshold pass chose to keep, so this needs the same
    before/after comparison `scene_max_size` got, not just a memory number.
    Measure scene count, the widest bbox, peak memory, and whether the v9
    non-converging scenes converge. The motivating case is the 25-template
    scene spanning 4300 px, which set both the 0.61 GB `Bq`/`Bl` buffer and,
    most likely, the r < 3' astrometric walk (the GP keeps `length_scale`
    400 px while the scene spans ten times that); at 1500 px it becomes 4
    scenes with a worst bbox of 0.36 Mpx. If 1500 px still leaves the walk,
    1000 px (2.5x the length scale, 5 scenes, 0.21 Mpx) is the next value
    down.
  - [x] `write_stamps` no longer builds the complete `vla` list before writing
    (2026-08-13) -- that was a full extra copy of every stamp, 12 GB
    full-field, at the end of the run. Offsets come from the shapes recorded in
    the first pass and each stamp is written into its slot. What remains for
    the streaming solve is appending as *tiles* complete rather than as
    templates complete, which needs the tiled build to exist first.
  - [ ] Upsampled band on demand instead of materialised: `_upsample_boxed`
    holds 7 GB for the sci/ivar pair and is where COSMOS OOMs. The array is a
    pure function of the lo-res pixels and is only ever read as
    `image[slices_original]`.
  - [x] Residual as a `np.memmap` over the output file rather than 3.5 GB of
    dirty anonymous pages (2026-08-13; `Pipeline._residual_memmap`).
  - [x] Release the band weight map once nothing reads it (2026-08-13).
    `weights_i` (3.5 GB upsampled ivar) was last read by
    `Templates.predicted_errors` but held by every `Scene`, so it sat through
    the stamp write. `Pipeline._release_scene_weights` clears it; `run` calls
    it after `predicted_errors` when no scene figures are drawn, and
    `write_outputs` after the figures. See `docs/MEMORY_LIFETIMES.md`.
  - [x] `write_outputs` writes the stamps last (2026-08-13), after the scene
    figures rather than before, so the weights can go before the run's other
    memory peak and a run that dies in the stamp write keeps its figures.
  - [x] Residual accumulates into its own output file (2026-08-13).
    `_residual_memmap` writes the header, extends the file sparsely and maps
    the data section; `write_outputs` flushes instead of writing. Falls back to
    anonymous memory for API-driven runs and on any mapping error.
  - [x] Repair replays its patch table onto a fresh copy-on-write map
    (2026-08-13) instead of holding the two full-field mosaics
    `repair_saturated_holes` returns (`saturate.py:733`). Fresh and cache-reuse
    paths now share `_apply_repair_patches`.
  - [x] `write_stamps` streams (2026-08-13): offsets from the recorded shapes,
    datasets created at final size, each stamp written into its slot. Removes
    the 12 GB `vla`-plus-concatenate copy at the end of the run.
  - [ ] Avoid the byte-order copies in `get_bg_and_ivar`. FITS is big-endian,
    so a memory-mapped float32 image arrives as `>f4` and
    `np.asarray(x, dtype=np.float32)` (`catalog.py:267-268`) copies rather than
    views. The detection-band call transiently allocates ~12 GB on a full field
    (`s`, `w`, two bool masks, `ivar_new`), undoing every memmap saving
    upstream. The full-resolution arrays are only used by `_valid_block_means`,
    a strided median sample and the final `w * scale`, all of which work on
    `>f4` directly.
  - [ ] ivar as `(memmapped wht, scale, invalid mask)` rather than a 3.5 GB
    anonymous array: the calibration is one scalar plus a mask
    (`catalog.py:369-372`). Composes with the on-demand upsampled band above --
    with both, neither ivar array exists.
  - [ ] Drop `weights[1]` after the upsample, which needs `source_products`
    pointed at the reference-grid array rather than the native one.
  - [ ] Timers inside `Scene.solve`. `56530f2` added the per-section wall-time
    breakdown for `run()`, which covers the outer split (templates, convolve,
    generate scenes, astrometry passes, final flux solve, residual). Still
    missing is the split *within* a scene solve -- `build_normal` against
    `assemble_scene_system_AB` against the factorisation -- which is what says
    how much of the scene loop is GIL-bound assembly and therefore what a
    worker pool would actually buy. Do this before building any of the above.
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

- [ ] Decide where the MINERVA SED estimator experiment belongs (2026-08-16).
  `tests/test_sed_estimator_experiment.py` imported
  `scratch.minerva_sed_estimator_experiment`, and `scratch/` is gitignored, so
  the module was absent from every fresh checkout. That broke collection and
  with it the whole CI run -- the suite reported one error and never ran, on
  `main` as well as on branches. Guarding the import with
  `pytest.importorskip` stopped it taking the run down but left the test dead
  everywhere except a working tree that happened to have the experiment, and
  the test file is now deleted (2026-08-16 suite shrink). What remains open is
  the estimator itself: promote it into `src/mophongo/` with a real test if it
  has become reusable, or leave it in `scratch/` as a one-off. `CLAUDE.md` is
  explicit that tests are tracked and must not depend on scratch work.

- [ ] `src/mophongo/sed_stack.py` has no tests (2026-08-16). Its 27-test file
  was removed in the suite shrink. The module is a leaf -- nothing in the
  package imports it -- so nothing in the pipeline notices, but its callers
  are the redshift-stacking analysis and the tests were its only written
  specification of the rasterization and filter-overlap conventions. If the
  module stays, give it a small test covering the top-hat filter interval,
  the once-per-galaxy overlap averaging, and the rest-frame `1+z` scaling; if
  it is analysis code rather than package code, move it out of
  `src/mophongo/`.

