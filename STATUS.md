# Status

This file records completed implementations, validation runs, and the current work state.

## Current Work
- [x] `scene_minimum_anchors` is a flat 10, and the campaign configs stopped
  setting it (2026-08-18). The default derived the floor from the astrometric
  polynomial order, `(order+1)(order+2)+1`, which is 3 at the default order 0
  -- an algebraic count of the field's free parameters, not a number of
  anchors a scatter can be measured from. Every campaign config overrode it by
  hand anyway (5 on CANFAR and OzStar, 7 and 10 in the MINERVA runs), so the
  derivation only ever described runs nobody launched.

  It is now `scene_minimum_anchors: int = 10` with the `__post_init__`
  derivation removed, and the hand-set values are gone from
  `examples/minerva`, `examples/canfar` and `examples/ozstar` (and from
  `make_minerva_configs.py`, which writes them) so those runs take the
  default. A wider astrometric basis still raises the floor where it needs to:
  `astrom_robust.anchor_gate` takes `max(scene_minimum_anchors, 2p)`.

  `_beta_err` in `tests/test_scene_astrometry_robust.py` pins the floor at 3:
  its nine-template field is about the weighting, not the gate.

- [x] Anchor local systems assembled from slice intersections (2026-08-18).
  The rest of the `astrom_robust` slowdown. `measure_anchor_shifts` gave every
  anchor a local least-squares system over the union footprint of its
  neighbours, with each column zero-padded onto that footprint and the Gram
  formed there: `ncol^2 / 2` products over ~80k pixels per anchor, for columns
  supported on one ~10k-pixel stamp each. Both factors grow with crowding,
  since `ncol` counts every template whose bbox touches the anchor.

  Columns are now keys `(template, kind)` carrying no padding, and every inner
  product runs over the intersection of the two templates' slices -- zero, and
  untouched, where they do not overlap. The entries are cached across anchors
  as well: `<c_p, W, c_q>` depends on the two columns and the weight map, not
  on whose system it appears in, and neighbouring anchors share most of their
  neighbourhoods. The `chi2_red` model is built on the anchor's own stamp,
  which is the only place it was ever read.

  Neither is an approximation, and the numbers say so: against the padded
  version on synthetic scenes at MINERVA density, `max |d eps| <= 4e-17 px`,
  `max rel d info <= 1e-14`, `d chi2_red` exactly zero, and the NaN/zero masks
  identical. 1456 -> 184 ms at 320 templates, 1308 -> 197 ms at 920, 13249 ->
  1965 ms at 920 with 151-px stamps: 6.7-7.9x throughout.
  `test_local_systems_match_the_padded_reference` keeps a padded reference in
  the test file and checks the two agree on overlapping templates of different
  sizes, one clipped by the frame, over a weight map with a dead strip.

  With the two changes below, the robust pass on the 920-template benchmark
  goes from 1489 ms to ~74 ms, i.e. ~5% of what it cost. TODO carries the one
  remaining step: the flux-flux block is `A` and its RHS is `b - A @ flux0`,
  so ~95% of a crowded local system need not be integrated at all.

- [x] The robust anchor pass stops paying for work it discards (2026-08-18).
  `astrom_robust` (on by default since `6e0f27a`) took the astrometry phase of
  a cosmos_f770w run from 6m23s to 29m02s. Profiling a synthetic scene at
  MINERVA density (`scratch/bench_anchor_shifts.py`) puts almost all of it in
  `scene.measure_anchor_shifts`, with two smaller pieces around it that are
  pure waste:

  * The gate is now checked before the measurement. `robust_anchor_weights`
    refuses any scene with fewer usable anchors than
    `max(scene_minimum_anchors, 2p)`, but `Scene._robust_anchor_weights` ran
    the whole per-anchor measurement first and only then found out. Three of
    the six scenes in the UDS DR0.1 trial patch (5, 7 and 7 anchors against a
    gate of 10) paid in full for a verdict fixed in advance. The gate itself
    moved to `astrom_robust.anchor_gate` so the two sites cannot drift apart,
    and the inactive verdict to `astrom_robust.inactive_anchor_weights` so the
    early return reports exactly what the late one did.
  * The flux-only solve that seeds the residual no longer computes errors it
    throws away. `SceneFitter.solve`/`solve_flux` take `errors=False`;
    `sqrt(diag(A^-1))` is a factorization plus one back-solve per template --
    125 ms of a 150 ms solve at 920 templates, 569 ms of 671 ms at 1718 --
    and `_robust_anchor_weights` uses only `.flux`. Skipped errors come back
    NaN, not zero, so they cannot be mistaken for measured ones.

  Together these are ~30-40% of the added time. The remainder is
  `measure_anchor_shifts` itself, which zero-pads every column of every
  anchor's local system to the union footprint of its neighbours and forms the
  Gram there: `ncol^2/2 * footprint_pixels` where the columns are supported on
  a stamp each. That rewrite is separate work.

- [x] The wall-clock breakdown covers the whole run, not only the solve
  (2026-08-18). A cosmos_f770w run reported `all 2h17m58s` against a fit whose
  phases summed to 40m03s, and the missing 1h38m was invisible: the second
  table was normalised to the sum of its own rows, so its percentages added to
  100 however little of the run they covered.

  Both tables now report against the whole invocation, which makes the
  `(other)` row the genuinely untimed remainder. Phases were added for the
  work that was outside the fit or inside it but untimed: `epsf grids`,
  `psf maps`, `matching kernels`, `read inputs`, `saturation repair`,
  `background + ivar`, `convolve templates`, `catalog update`,
  `write residual + tables`, `scene figures`, `field figures`,
  `write stamps`. Phases must not nest, so `build_kernels` starts its own
  after the `build_psfs` call it may make.

  `_phase` yields a `_PhaseTag` whose `from_disk()` retags the phase --
  `psf maps (from disk)`, `matching kernels (from disk)`,
  `saturation repair (from disk)` -- so a cache hit is never reported under
  the name of the work it stood in for. `run_all` now holds the report until
  the outputs are written (`_defer_timings`, renamed from `_cli_stepping`),
  which is where a quarter of the run had been going unreported.

- [x] Scene PNGs sampled to the scene, and capped at `scene_plots_max`
  (2026-08-17). `write_outputs` drew every scene at a fixed `dpi=300`, i.e.
  onto a 4500x3000 canvas whatever the scene's size. Measured on the six-panel
  figure, `savefig` is 90-97% of `Scene.plot` (~60% Agg draw, ~40% PNG
  encoding) and both halves scale with the canvas, not with the data: a
  200 px scene and a 2000 px one cost 0.7 s and 2.8 s only because the larger
  one compresses worse. `make_lupton_rgb`, `SegmentationImage.cmap`,
  `model_image` and the six `imshow` calls together are under 5%.

  Two changes. The dpi now comes from `verification.diagnostic_pixel_sampling_dpi`
  (made public for this) at one output pixel per scene pixel, clipped to
  100-300. And `RunConfig.scene_plots_max` (default 200) keeps only the scenes
  worth opening: half by worst `chi2_dof`, half by largest `astrom_floor`,
  with the floor half topping up from the chi2 ranking because the floor is
  NaN wherever the robust pass declined. `_scenes_to_plot` logs what it kept
  and what it dropped.

  On `examples/minerva/cosmos_f770w` (1643 scenes, median extent 770 px, all
  written at 4500x3000 for 6.9 GB): ~55 min and 6.9 GB before, ~23 min and
  2.9 GB from the dpi alone, ~5 min and 0.6 GB with the cap. Threads over
  figures were measured at 1.3x (Agg holds the GIL) and processes are out --
  scene plotting is already where campaign bands died on memory.

- [x] `sigma_shift` was NaN in every scene catalog ever written (2026-08-17).
  `_refine_scene_astrometry` closes each scene with a flux-only solve on the
  final templates (`fit_astrometry_niter=0`). That solve returns
  `shift_cov=None`, and `Scene.solve` assigns `self.solution = sol`
  unconditionally, so the joint solve's covariance was overwritten on every
  scene of every run; `Scene.shift_error()` read it off `solution` and
  returned NaN. `self.shifts` survived because the joint branch alone writes
  it, which is why `dx`/`dy` looked fine and only their scale was missing.
  The covariance is now kept on `Scene.shift_cov` beside `shifts`, where the
  closing solve does not reach, and `shift_error()` reads it there.

  Not fixed: `restore_scene_fit` does not put it back, so a catalog re-emitted
  from a reloaded fit still reports NaN. See TODO.

- [x] `shift_rms` reported ~1e-15 px rather than zero (2026-08-17). At
  `astrom_kwargs={'poly': {'order': 0}}` the field is constant, so every
  template in a scene receives a bitwise-identical shift -- verified across
  all 1404 multi-template scenes of the cosmos_f770w run. `shift_scatter`
  subtracted the mean anyway and reported the ulp that leaves behind (max
  1.6e-14 px on shifts up to 10 px). It now returns an exact 0.0 when the
  shifts are identical. The docstring's claim that a non-zero value at order 0
  means the scene walked was wrong and is gone: at order 0 a walk moves every
  template equally, so the scatter stays zero however badly the scene walks.

- [x] `build_kernels` indexes the band PSF maps instead of re-drizzling them
  (2026-08-17). A run drizzled four PSF cubes, not two: `build_psfs` at the
  hi and lo region centroids (4621 + 944 stamps on the COSMOS F444W/F770W
  pair), then `build_kernels` rebuilt both region maps and drizzled both bands
  again at the 7368 hi-x-lo overlay centroids. 20301 stamps, ~26 minutes.

  An overlay region lies inside exactly one hi region and one lo region, and a
  region map defines the PSF as constant within a region -- that is what
  `get_psf(ra, dec)` returns. `overlay_with` already records the parents as
  `psf_key_1`/`psf_key_2`, and `psf_key` is the dense row index, so the pair
  for overlay row *i* is a gather: `prm_hi.psfs[k1]`, `prm_lo.psfs[k2]`. The
  method now takes `self.prm_hi`/`self.prm_lo` (building them if a caller
  entered here first), overlays those, and indexes. 5565 stamps, ~8 minutes,
  and no second `_region_maps()` build.

  Correctness moves with it. Re-drizzling at the overlay centroid produced a
  stamp at a different field point inside the same exposure set, so every
  kernel was matched against a PSF pair the fitter never looks up; the gather
  matches the exact pair `prm_hi.get_psf()` returns. `_ensure_dpsfs` no longer
  loads the ePSF grids here -- only the two `driz_pscale` values are needed.

  Cached kernel maps carry a `kernel_psf_source` stamp and are reused only
  when it matches, so a map written by the old path is rebuilt rather than
  silently kept. The older standalone scripts (`examples/run_uds_770_wren.py`
  and friends) drizzled once per band at the overlay centroids and assigned
  those cubes to the *band* maps, leaving `prm_444.psfs` with 7368 planes
  against 4621 regions; `PSFRegionMap._dense_psf_keys` rejects that mismatch
  today.

- [x] Scene astrometry logs one line per scene per iteration (2026-08-17). The
  robust-anchor report was a second INFO record alongside the shift line, so a
  200-scene run emitted ~1000 lines. Its numbers now ride on the shift line
  (`; 0/11 anchor(s) rejected, floor 0.540 px, 10.7 effective`) and the
  standalone record is DEBUG. Both the shift line and the no-anchors warning
  now name the scene's template count, and the warning stays the only case
  where a scene emits a second record.

- [x] The run log names the source, and CANFAR asks git instead of a stamp
  (2026-08-17). Two halves of the same gap: a finished run could not say which
  mophongo produced it.

  `pipeline.source_version()` returns `<package version> (<git sha>)`, read
  from the checkout the installed package lives in -- an editable install from
  `run<N>/config/mophongo` reports that clone's HEAD -- and `log_run` writes it
  into every run's header beside the python and platform lines. `+dirty` marks
  an uncommitted tree. Falls back to the package version where the source is
  not a checkout, rather than failing a run over provenance.

  This is what a config cannot record. The outputs already carry the full
  config, but a config does not pin behavior: a *default* can move under a file
  that never named the field, which is exactly how `fit_method` gave a clipped
  solve before today and an NNLS one after, with nothing in the config to tell
  the two runs apart. Four defaults moved in this session alone.

  On CANFAR, `check_src_current` now asks git in the checkout from a one-CPU
  4 GB container (`jobs/src_version.sh`) instead of reading a stamped
  `SRC_VERSION`. It schedules immediately and takes about 27 s, against a
  campaign of hours, and it is the same question OzStar answers over ssh. The
  stamp is gone from every writer and reader -- `setup_env.sh`,
  `update_src.sh`, and the log lines in `build_psfs.sh`, `run.sh` and
  `scene_plots.sh` all ask git now -- so the repository is the only record of
  what is installed and cannot disagree with itself. Exercised live: it read
  `2d2d014` off arc and correctly refused a launch against local `95e6113`.

- [x] `PSFRegionMap.to_file` writes a projection (2026-08-17). `from_wcs_list`
  sets `EPSG:4326`, but `overlay_with` rebuilt its frame with
  `gpd.GeoDataFrame(overlays)` and `group_by_pa` its empty frames with
  `gpd.GeoDataFrame()`, both dropping the CRS, so a map that had been through
  either was written as a projectionless GeoJSON and pyogrio warned on every
  write. Those three constructions now carry the source CRS, and `to_file`
  falls back to `EPSG:4326` when a map still arrives without one: the polygons
  are sky footprints in degrees, so that is what they are. The fallback applies
  to a copy, so the in-memory object is unchanged.

  **No lookup was ever wrong.** `lookup_key` builds a `Point(ra, dec)` and runs
  point-in-polygon against the raw shapely geometries, which carry no
  projection; the written coordinates were identical either way; and GeoJSON
  implies WGS84 by spec, so GDAL assigns `EPSG:4326` on read regardless.
  Checked by round-tripping a map both ways: identical coordinates, identical
  keys. The value of the fix is narrower than the warning suggests --
  `to_file` takes any geopandas driver, and for a format with no implied CRS
  (GPKG, shapefile) a missing one is genuinely lost. Verified under
  `-W error::UserWarning`.

- [x] OzStar pins its source and venv per run, like CANFAR (2026-08-17).
  `ozroot.src_dir()` and `venv_dir()` moved from `base/mophongo` and
  `bin/venv` to `run<N>/config/{mophongo,venv}`. A run pins one mophongo
  version, and a shared clone meant a `sync` for the run being worked on
  silently changed the code under every other run: a finished run's outputs
  could not be tied to the source that produced them, and its config -- which
  does live per run -- could disagree with the code that read it. Every job
  script already took `$SRC` and `$VENV` from the environment, so the two
  functions moved everything. `venv-vos` stays shared under `bin/`: the CADC
  tools run on datamover nodes that have no module tree, and they are not
  pinned to a mophongo version.

  `campaign.py` already runs `setup` as its third step, so a new `run<N>` gets
  its own checkout with nobody remembering to ask, and `write_run_readme`
  (`campaign.py:241`) already writes `run<N>/README.md` over ssh with the
  commit, the bands, the release-version table and a diff against the previous
  run -- both were in place, only the paths were not.

  Harmonized the two setups while there:

  * Both now move a source directory aside when it exists without a `.git` and
    clone fresh. That state is an unpacked tarball from the older
    ship-the-source convention, `git clone` refuses a non-empty target, and
    neither branch of the old conditional repaired it -- so setup failed on
    every retry until someone cleared it by hand, which is exactly what CANFAR
    `run1` did earlier today.
  * `SRC_VERSION` stays a CANFAR-only file, and the asymmetry is now stated in
    both scripts so a later harmonization pass does not delete it. The laptop
    cannot run git on arc -- there is no ssh, only file transfer -- so
    `submit._arc_src_version()` reads that stamp to refuse a launch against
    stale source. OzStar takes ssh, and `submit.src_version()` already asks git
    in the clone, where a stamp would be a second copy of the same fact able to
    drift. Briefly added to OzStar for symmetry and removed once it was clear
    nothing read it.
  * Both rebuild the venv only under `REBUILD=1`. CANFAR was deleting and
    rebuilding on every setup, seven minutes each time, for no gain: the
    install is editable and `pip install -e` still runs to pick up dependency
    changes.
  * OzStar's `REBUILD=1` no longer deletes `$VOS`, the shared CADC venv --
    rebuilding one run's environment must not take it from the others.

  Verified on OzStar: `run2/config/{mophongo,venv}` in place, `SRC_VERSION`
  `4e76e5c` matching the clone, and the per-run venv importing mophongo from
  the per-run clone. The old `base/mophongo` and `bin/venv` are now orphaned
  and were left rather than deleted. The empty `csv/` directory is gone; the
  wcs csvs were already in `data/`, which is where the configs point.

- [x] OzStar and CANFAR redeployed on `2d2d014` (2026-08-16). Both clusters
  now run the same source as `main`, with grids and configs that agree:

  | | OzStar | CANFAR |
  |---|---|---|
  | source | `2d2d014` | `2d2d014` (SRC_VERSION == HEAD) |
  | grids | 487 STDPSF, 0 field-prefixed | 487 STDPSF, 0 field-prefixed |
  | configs | 17, all STDPSF + `aperture_ee` | 17, all STDPSF + `aperture_ee` |

  Two things needed fixing on the way, and both are worth knowing before the
  next deploy:

  * **CANFAR `setup` failed the first time.** `run1/config/mophongo` held an
    unpacked source tree with no `.git` -- a leftover from the older
    ship-a-tarball convention -- so `setup_env.sh` took its `git clone` branch
    and refused a non-empty directory. Renamed to `mophongo.stale-archive`
    (not deleted) and re-run, it cloned cleanly. `setup_env.sh` pulls when
    `.git` exists and clones when it does not; neither path repairs a tree that
    is neither.
  * **OzStar configs live in `run2/config`**, which `ozroot.config_dir()`
    resolves; a check against `run3` reads nothing and looks like a failed
    push.

  Cluster configs are not tracked -- `examples/canfar/*` and the generated
  `*_ozstar.json` are gitignored -- so they are regenerated by
  `arcify.py`/`ozify.py` from `examples/minerva/` before each deploy rather
  than committed. `vcp` reports an md5 mismatch when reading a file it has just
  overwritten, so verifying deployed configs means reading them from inside a
  container rather than pulling them back down.

- [x] `fitsmap_url` replaces `minerva_viewer` + `minerva_release` (2026-08-16).
  `minerva_release` was declared at `pipeline.py:293` and read nowhere, not
  even by the code that builds the viewer link -- the release is already part
  of the URL path when a field wants one, so a second field could only ever
  disagree with the first.

  `RunConfig.from_json` rejects unknown keys, so a plain rename would stop
  every config in an existing run tree from loading. It migrates instead:
  `minerva_viewer` is carried over to `fitsmap_url`, and `minerva_release` is
  dropped. Tested both directions.

- [x] Shrank the test suite to the contracts a future change can break
  (2026-08-16). 41 files / 14,088 lines / 72 s became **33 files / 11,407
  lines / 46 s**, and the suite is green: 471 passed, 0 failed. Nothing that
  pins a behaviour was dropped without its coverage landing somewhere else.

  Deleted outright: `test_scene_astrometry.py` (an empty file), and
  `test_sed_estimator_experiment.py`, whose module lives in the gitignored
  `scratch/` tree so it could only ever skip. `test_sed_stack.py` went too --
  `sed_stack.py` is a leaf nothing in the package imports.

  Merged into the module that owns them: `test_fit.py` into
  `test_scene_fitter.py`; `test_pipeline_dedup.py` and
  `test_pipeline_multitemplate.py` into `test_pipeline.py`;
  `test_background_masking.py` and the old `test_catalog.py` into one
  `test_catalog.py`. Cross-file duplicates went with them: `get_bg_and_ivar`'s
  masking was asserted in three files and is now in `test_catalog.py` alone,
  and `as_label_array` was tested weakly in `test_templates.py` and
  thoroughly in `test_memory_footprint.py` -- the thorough four moved to
  `test_templates.py` and the weak one is gone.

  `test_moffat_recovery.py` became `examples/verify_moffat_recovery.py`. It
  was 24 s of the suite -- a third of it -- to produce diagnostic PNGs and
  assert only that the median flux ratio is somewhere near unity. It is a
  validation run, not a unit test, and now runs on demand
  (`python examples/verify_moffat_recovery.py [outdir]`); verified end to end,
  all five scenarios, median ratio 0.9998-1.0007.

  The astrometry files were the exception to the plan. Four files, 2,199
  lines, were the obvious consolidation target, but `test_astrom_robust.py`
  (leaf statistics), `test_scene_astrometry_blocks.py` (blocks against a dense
  design) and `test_scene_astrometry_robust.py` (per-anchor measurement) are
  three layers of one argument rather than three copies of it -- collapsing
  them would have lost the layer that localizes a failure. The redundancy was
  in `test_astrometry.py` instead: four end-to-end scene tests asserting a
  0.6 px shift to `atol=0.3`, where the scene files assert the same properties
  to 1e-9. Those four went, `test_astrometry.py` is now the non-joint
  `AstroCorrect`/`AstroMap` path plus the alpha0 scaling nothing else covers,
  and its write into `Path("../tmp")` -- outside the repository -- went with
  them.

  Left alone deliberately: `tests/test_nnls.py`, live work on this branch.

- [x] `fit_method` replaces `positivity`, and NNLS is the default (2026-08-16).
  One switch with three values: `"lls"` keeps negative fluxes, `"clip"` clamps
  them after an unconstrained solve, `"nnls"` solves under the constraint.
  `positivity` is gone rather than deprecated -- `RunConfig.from_json` rejects
  unknown keys, so an old config naming it now fails loudly instead of being
  silently ignored.

  Clipping was never the constrained optimum. It pins a template at zero and
  leaves its neighbours holding flux they took only because the negative one
  was there; the survivors are never re-solved. `SceneFitter.fnnls` is the
  Bro & de Jong (1997) rearrangement of Lawson-Hanson, which consumes the Gram
  matrix and `A^T d` directly -- exactly what a scene holds, so no design
  matrix is reconstructed. The `lsq_linear` alternative was rejected on
  evidence: the Cholesky at `scene_fitter.py:370` is of the *shift* block `BB`,
  not of `A`, and scipy has no sparse Cholesky, so that route needs a
  factorization this code does not have.

  The joint flux+shift path uses the same routine with the shift coefficients
  held free (`fnnls(..., free=...)`): a shift has no sign to respect, only the
  fluxes do. Without that, `nnls` would have clipped on every scene, since
  `fit_astrometry_joint` is on by default.

  Verified on injected truth through `Pipeline`. Noiseless recovery is exact to
  4e-16 for all three methods and for the joint path. Over 30 shared noise
  realizations on a blend of 100/3/200:

      lls   bias [-0.43 +0.23 -0.24]  rms [2.32 3.34 2.71]
      clip  bias [-0.43 +0.54 -0.24]  rms [2.32 2.88 2.71]
      nnls  bias [-0.43 +0.54 -0.24]  rms [2.32 2.88 2.71]

  The bright sources are untouched by the constraint; the SNR~1 source is
  biased high by both constrained modes, which is the truncation effect and is
  the reason `info["flux_uncon"]` now always carries the unconstrained fluxes.
  `clip` and `nnls` agree on this weakly blended mock and diverge in strong
  blends, which is what `tests/test_nnls.py` covers with an 85%-correlated
  pair.

  Lawson-Hanson starts at `x = 0` and admits one component per iteration, so a
  scene whose solution is entirely positive -- the common case -- would pay `n`
  growing solves to reach what one solve gives: 4.0 s against 0.077 s for
  `spsolve` at 1000 templates. `fnnls` now tries the unconstrained answer
  first and keeps it when it satisfies the KKT conditions, which is a proof of
  optimality rather than a guess, so the shortcut cannot return a different
  solution from the long route. That case is now 0.012 s, faster than
  `spsolve`. A bounded refinement (drop what came out negative, re-solve,
  re-test) handles partly pinned scenes, and anything that fails the test falls
  back to the exact loop.

  On realistic scene coupling -- a template overlapping only its neighbours --
  the worst measured case is 0.074 s at 500 templates with 30% pinned. The
  alarming ratios in a dense random benchmark come from every template
  overlapping every other, which no real scene does.

  `info["at_bound"]` marks components sitting on the bound. Their `err` is the
  unconstrained `sqrt(diag(A^-1))`, which is not a symmetric 1-sigma interval
  for a parameter at a constraint -- flagged rather than silently reported as
  if it were.

- [x] Dropped `cg_kwargs`, made `filter_lo` a fallback (2026-08-16).
  `FitConfig.cg_kwargs` was dead: nothing passed it and nothing read it, and
  the solver is a direct sparse factorization (`spsolve`/`splu`), so
  `maxiter`, `atol` and `M` described an iterative solver that is not there.
  Gone, along with the `SceneFitter.solve` parameter and the doc-table rows.
  The returned `{"cg_info": ...}` is now `{"solver": "spsolve"}` -- it was the
  last cg-flavoured name, and a direct solve has no iteration count to report
  -- and the two dead `info = 0` assignments went with it.

  `Pipeline._filter_lo()` reads the `FILTER` card of `sci_lo` and falls back to
  `RunConfig.filter_lo`, which is now documented as the fallback it is: the
  mosaic header cannot disagree with the pixels being fitted, so it is the
  better source, but a mosaic carrying no filter still needs the config value.
  All three readers go through it -- blur lookup and two labels.

  Deliberately not cached. A cached version broke
  `test_blur_resolution_modes`, which edits `run_config.filter_lo` between
  calls; caching makes an edited config silently inert, and interactive
  sessions do exactly that. `fits.getheader` reads header blocks, not pixels.

- [x] Example configs measure at `aperture_ee = 0.7` (2026-08-16). The five
  `examples/*.json` and `make_minerva_configs.py` now carry
  `phot: {aperture_ee: 0.7, aperture_catalog: null}` instead of a fixed
  `aperture_diam`. `APERTURE_DIAM_ARCSEC` stays in the generator as the record
  of the IDL-matched diameters, for runs that tie to Y. Asada's catalog with an
  explicit `aperture_diam`.

- [x] ePSF grids renamed on both clusters (2026-08-16). OzStar
  (`/fred/oz030/ilabbe/mophongo/PSF`) and CANFAR
  (`/arc/projects/minerva/ifl/mophongo/PSF`) both went 493 -> 487 grids, zero
  field-prefixed, zero without a FOV token. CANFAR ran as a headless skaha job
  (`j03kabyb`); OzStar over ssh.

  OzStar needed the FOV token inserted for **369** of its 493 grids -- legacy
  names from before that token existed -- and two of its collisions were
  genuinely different grids (FOV4 against FOV6) that a plain `mv` would have
  destroyed. The FOV came from each grid's own `FOV` header keyword, which
  separated them before anything moved.

  Six collisions on each cluster were verified bit-identical in data before
  anything was parked, and the survivors went to `PSF/_superseded/` rather than
  being deleted. Those six are the COSMOS/EGS pairs, and they are the direct
  confirmation of the field-independence claim on real data -- the local set
  has no cross-field pairs to test it with.

  Note `examples/canfar/*` and the generated `*_ozstar.json` are gitignored:
  cluster configs are local artifacts regenerated by `arcify.py`/`ozify.py`
  before each deploy, not committed. Both were regenerated here and now carry
  the `STDPSF` patterns and `aperture_ee`.

- [x] Documented the external input catalog contract (2026-08-16).
  `docs/catalog.md` "Running with an external input catalog" now lists every
  column the run reads and the name it must carry: the required `id`/`x`/`y`,
  the optional `flag_star`, deblend trio, `FLAG_SATURATED_*` and `ra`/`dec`,
  and the three configuration-named ones (`phot.aperture_catalog`, and
  `phot.kron_flux_col` / `aper_flux_col` / `kron_radius_col` behind
  `totcor_cat`), plus which of them survive into the output table and that
  `Table.meta` round-trips as HIERARCH cards. Two corrections along the way:
  `FLAG_SATURATED_*` values are star *group* ids (rows sharing one are fitted
  in a single scene), not a plain saturated flag, and the `phot` block in
  `docs/pipeline.md` still described the pre-`aperture_ee` defaults and was
  missing the Kron column knobs. Note for later: `PhotConfig.aperture_diam`
  is typed and commented as accepting a column name, but
  `Pipeline._resolve_image_ap_radius_pix` handles only float/array/None, so a
  string silently falls through to the encircled-energy path; the docs
  describe the implemented behaviour.
- [x] ePSF grids are named `STDPSF_*`, with no field (2026-08-16). The field
  was never a physical input. `build_jwst_psf` (`jwst_psf.py:106`) takes
  instrument, filter, detector, date, grid geometry and parity, and its body
  sets `inst.filter`, `inst.detector`, `inst.options["parity"]` and calls
  `load_wss_opd_by_date`. Sky position is never passed and stpsf has nowhere to
  take it. The field dependence that does exist is *detector* position, which
  is what `GRID<N>` samples. So a grid is a property of the instrument and the
  epoch, and one file serves every field observed then.

  Renamed all 443 files in `data/PSF` from `<FIELD>_` to `STDPSF_`; verified
  collision-free first (every stem is unique once the prefix is stripped) and
  the count is unchanged. Git is unaffected -- the `.gitignore` negations for
  that directory use the pre-rename token order and match nothing.

  `PsfConfig.pattern_hi`/`pattern_lo` were empty strings and now default to the
  MINERVA pair, `STDPSF_NRC.._F444W_MJD\d+_GRID25_OS4` against
  `STDPSF_MIRI_F770W_MJD\d+_GRID9_OS4`. The five example configs and
  `make_minerva_configs.py` follow, as do `docs/campaigns.md`, `docs/pipeline.md`,
  `docs/SATURATE.md` and the `verification.py` defaults; the historical reports
  keep their original names as the record of what those runs used.

  The payoff is pooling. Matched the way the loader does it
  (`fov_agnostic_pattern`), the F444W pattern now finds 160 grids and the F770W
  pattern 45, where a `UDS_`-prefixed pattern saw only that field's. A UDS run
  can now use a COSMOS-built grid of the right epoch, because they are the same
  grid.

  `PSFFactory.prefix` already defaulted to `STDPSF` and the pipeline's autobuild
  passes no prefix, so autobuilt grids were landing as `STDPSF_*` while
  hand-built ones carried fields -- the rename removes that split.
  `examples/repair_saturate.py` no longer forces `prefix="UDS"`. TODO carries
  the follow-up: grid identity belongs on the OPD file, not the MJD, which
  would retire 75 duplicate grids.

- [x] `mophongo config <out.json>` writes a default run config (2026-08-16).
  A new run had no starting point but an existing config from another field,
  which carries that field's paths and whatever settings it happened to pin.
  The command writes every `RunConfig`, `PsfConfig` and `FitConfig` setting at
  its default, so the file doubles as a listing of what exists without reading
  the source.

  The `fit` block is a plain dict on `RunConfig` and defaults to `{}`, so a
  naive `asdict` dump would say nothing at all about the fit; it is expanded to
  the full `FitConfig` defaults (41 keys). The nine fields with no default are
  input paths and a run name, written as `<angle bracket>` placeholders rather
  than empty strings -- an empty string reads as a legitimate value and fails
  deep in a run, a placeholder cannot. The file parses through `from_json` but
  does not run until they are filled in.

  Refuses to overwrite without `--force`, since configs are hand-edited after
  generation, and reports that as a clean argparse error rather than a
  traceback. `tests/test_cli_config.py` (7 tests) checks the round trip, that
  every `FitConfig` field is present, that the recently changed defaults
  (`astrom_model="poly"`, `astrom_robust=True`, `phot.aperture_ee=0.70`) are
  what lands in the file, and the clobber guard. Full suite: 500 passed.

- [x] Estimator 3 is written out, in the form the report defines it
  (2026-08-16). `ap_flux_est3_<i> = ap_model * psfcor * totcor_cat + ap_res`,
  Eq. 12 of `scratch/wren/flux_estimator_comparison.pdf`. The aperture-to-total
  is built as `psfcor * totcor_cat` rather than from a bare `1/ap_lo`, which
  sidesteps the name `totcor` -- that name has meant both the with-EE and the
  without-EE convention in different codebases -- and it is a total in the full
  sense because `totcor_cat` carries the encircled-energy term.

  The correction multiplies the model term **only**; the residual is added
  unscaled. Nothing in the catalog did this before: `ap_flux_cat_<i>` scales
  `ap_flux_<i>`, which already contains the residual, so it corrects the
  residual along with the model. The two differ by
  `ap_res * (psfcor * totcor_cat - 1)` and coincide only where the residual
  vanishes -- they are different estimators, and the existing one was left
  alone rather than redefined.

  `ap_res_<i>` is the new `sum_Omega(res)` column: the residual over
  `disk(aper_<i>/2)` with **other sources' segment pixels zeroed**. The
  neighbours are already subtracted in the model, but their residuals are not
  this source's to claim, and inside a shared aperture they would be.

  Three tests in `tests/test_aperture_ee.py`. Worth noting the pipeline fixture
  alone cannot distinguish the two estimators: with the template extracted from
  the same image the residual is ~0 and the forms coincide, so the distinction
  is pinned arithmetically instead. Full suite: 493 passed.

- [x] The measurement aperture is the larger of the catalog and encircled-energy
  apertures (2026-08-16). Per source, `R = max(R_cat, R_ee)`. The two ends of
  the catalog fail in opposite directions and one rule covers both: a bright
  extended source has a wide catalog aperture, and measuring the band in that
  same aperture leaves `totcor_cat` needing no aperture-size adjustment at all,
  since it was defined at that radius on the detection band -- `psfcor` is left
  doing pure PSF work. A faint source sits on the catalog's aperture floor
  (72.4% of the MINERVA SUPER catalog is at 0.2"), which at MIRI resolution is
  well inside the PSF core; there the encircled-energy aperture takes over near
  maximum SNR.

  Both raw sums are written whichever way the rule falls -- `ap_flux_catap_<i>`
  and `ap_flux_ee_<i>` -- so the trade is auditable per source and either can be
  recovered without a rerun. `aper_<i>` (used diameter) is now per source rather
  than a band-wide scalar, and `aper_ee_<i>` records the encircled-energy
  aperture it falls back to. The band-wide `aper_<i>` write in `run` was removed;
  it would have overwritten the per-source values.

  `docs/outputs.md` carries the estimator in the form it is actually defined,
  `aper(model - model_nn, R_phi) * psfcor * totcor_cat + sum_Omega(res)` with
  `psfcor = ap_hi/ap_lo` and `Omega = disk(R_phi)`, and why the max rule makes
  those terms compose. Two tests in `tests/test_aperture_ee.py` drive a
  two-source run across both regimes. Full suite: 490 passed.

- [x] The measurement aperture can be set by encircled energy (2026-08-16).
  `FitConfig.phot.aperture_ee`, default `0.70`, sizes the aperture from the
  diameter enclosing that fraction of the band's *model* PSF -- the drizzled
  stamp after the Gaussian diffusion blur, which is the PSF the matching kernel
  was built against, so the model/empirical mismatch is already folded in and
  needs no separate correction. Replaces the old `None` fallback of 1.5x FWHM,
  which survives as the last resort when there is no usable PSF or the stamp
  never reaches the requested fraction inside its inscribed circle (warned, not
  extrapolated). An explicit `aperture_diam` still wins, so runs tied to an
  external catalog's aperture are never silently resized.

  Fixed EE, not fixed angle, is what makes a colour aperture-correction-free to
  first order: the estimator's model term depends on the aperture only through
  `ap_lo/ap_hi`, and at fixed EE `ap_lo` is that fraction by construction for a
  point source, the same number in every band, so the band-to-band part
  cancels. A fixed 0.5" samples a different part of the growth curve at F770W
  than at F1800W and the difference goes into the colour. 0.70 also sits near
  the SNR optimum: `SNR ~ EE(r)/r` peaks for a Gaussian at `r = 1.585 sigma`,
  i.e. 1.35x FWHM in diameter, ~71% enclosed -- pinned in the tests against the
  analytic optimum rather than against a recorded number.

  `aper_<i>` (realized diameter, arcsec) is now written unconditionally. It
  used to appear only when `aperture_diam` was set, which is exactly backwards:
  under `aperture_ee` the diameter is derived rather than configured, so the
  column is the only record of what was used.

  New `Pipeline._ee_ap_diam_arcsec` over the existing
  `psf.stamp_encircled_energy`; `tests/test_aperture_ee.py` (8 tests) covers
  the growth-curve match, the SNR optimum, PSF-width tracking, explicit-diameter
  precedence and the fallback. `docs/outputs.md` gains the estimator
  decomposition `(A * ap_lo/ap_hi) * totcor_cat + res(R_measure)` and where the
  aperture enters each term. Full suite: 488 passed.

- [x] Hand-written campaign configs use their band's standard aperture
  (2026-08-16). `examples/make_minerva_configs.py` has carried the per-filter
  table (`APERTURE_DIAM_ARCSEC`) matched to classic IDL `subphot` on Y. Asada's
  MIRI catalog since it was written, so *generated* configs were always right.
  Every hand-written config had a flat `aperture_diam = 0.5` instead, which is
  no band's standard: F770W wants 0.70, F1280W and F1500W 1.20, F1800W 1.50.
  `uds_770_dr0.1.json` documents itself as the modern-pipeline reproduction of
  `run_uds_770_wren.py`, whose own `aps` dict already used 0.70 for F770W, so
  the aperture silently broke the comparison the file exists to make. F1800W
  was the worst of them -- a 0.5" aperture on a PSF whose standard is three
  times wider.

  Fixed in the five F770W configs (`uds_770_dr0.json`, `uds_770_dr0.1.json`
  and its `_head`/`_ps3` variants, `cosmos_770_dr0.1.json`) and in
  `run_1280.py`, `run_1500.py`, `run_1800.py`, each with a comment pointing at
  the canonical table. Mock scripts are left alone: a synthetic mosaic has no
  Asada catalog to tie to. `docs/campaigns.md` gains the table and its
  provenance, including which values are measured and which interpolated.

  This changes `flux_<i>_aper` and anything derived from it in reruns of those
  configs. Verified all five configs still parse and report 0.70.

- [x] Config vocabulary and the `phot` block (2026-08-16). Names now say what
  the thing is, in the lingo the rest of the astrometry code uses:
  * `scene_minimum_bright` -> `scene_minimum_anchors` (the count is templates
    passing the *anchor* cuts, and isolation is not brightness),
    `snr_thresh_astrom` -> `astrom_minimum_snr`, `reg_astrom` -> `astrom_reg`.
    252 occurrences across 105 files, including every `examples/**.json`.
  * Aperture photometry moved out of `FitConfig`'s flat namespace into a
    `PhotConfig` block, `fit.phot`, coerced from JSON in `__post_init__` with
    unknown keys raising, exactly as `RunConfig.psf` does. Nothing in the
    linear solve reads any of it, which is the reason it sits apart. The four
    `cat_*_col` knobs drop the prefix -- `phot.kron_flux_col` -- since the
    block already says they name input-catalog columns, and
    `aperture_units` -> `phot.units`.
  * All 61 `RunConfig` JSONs under `examples/` upgraded and verified to load,
    which also fixed the ones the `psf` block had already broken: the
    schema moved without migrating the configs, so every canfar/ozstar config
    and both saved run snapshots were unloadable. `make_minerva_configs.py`
    emits the new shape. Files under `examples/**/verification/v*` are run
    outputs and were left as they are.
  * No compatibility shim: old configs raise with the mapping in the message.
- [x] `FitConfig.astrom_robust` defaults to `True` (2026-08-16). The failure it
  prevents is one-sided: anchor leverage grows as flux squared, so a single
  bright extended source with an asymmetric colour gradient produces a
  residual dipole indistinguishable from a shift and drags its scene's field.
  The estimator costs a few percent of efficiency on clean anchors, which is
  the cheaper of the two errors. Still gated on `scene_minimum_anchors`, so
  scenes too small to judge fall back to `astrom_leverage_cap` as before, and
  `astrom_isolation_thresh` is still needed ahead of it -- a blended anchor's
  implied shift is shrunk coherently, so blended anchors agree with each other
  and majority rule follows them. `astrom_robust=False` recovers the old
  behavior, and `SCENES` carries `astrom_robust`/`astrom_nreject`/`astrom_neff`
  for the A/B comparison. Docs updated in `docs/fitting.md`, `docs/pipeline.md`.

- [x] `FitConfig.astrom_model` defaults to `"poly"` (2026-08-16). It was
  `"gp"`, which never described what the default run does: the joint path is
  on by default and takes its basis order straight from
  `astrom_kwargs["poly"]["order"]` (`scene.py:1475`) without branching on the
  model, so `"gp"` was reachable only by also setting
  `fit_astrometry_joint=False`. A provenance dump therefore reported a model
  the run had not used. No behavior change -- the joint path reads the same
  order either way -- only the recorded and non-joint defaults now agree.
  Docstrings in `fit.py` and the tables in `docs/fitting.md` and
  `docs/pipeline.md` say which path reads the field. Making the joint path
  honor `"gp"` needs a global cross-scene field; the design is in `TODO.md`.

- [x] Scene catalogs say how well the shift was measured, not just what it was
  (2026-08-16). `dx`/`dy` alone cannot be read: a 0.2 px shift means nothing
  until you know whether it was measured to 0.02 px or to 0.5 px.
  * `SceneFitter._shift_covariance` recovers the shift block's covariance,
    which `_solve_flux_and_shifts` was computing (the Cholesky of `BB`) and
    discarding. In whitened variables the shift block of the joint inverse is
    `(I - AB_w' A_w^-1 AB_w)^-1`, i.e. after the fluxes absorb what they can;
    unwhitened by `Linv' cov Linv`. Costs one sparse factorization plus `nB`
    back-solves, and `nB` is 2 at order 0. Taking `BB^-1` instead would be
    free but wrong by the flux-shift degeneracy, measured at tens of percent
    when a neighbour sits about a FWHM from an anchor.
  * `Scene.shift_error()` evaluates it at the basis origin, averaged over the
    axes -> `sigma_shift`. It is the *last pass's* number: passes re-measure
    the same pixels, so at convergence the accumulated shift is determined
    about as well as any one pass determines it. Tests pin that it scales
    linearly with the noise and brackets the true error (7 of 8 realizations
    inside 3 sigma).
  * `Scene.chi2_dof()` -> `chi2_dof`, reduced chi-square over the scene bbox
    against the residual with *every* scene's model subtracted (`Scene.residual`
    keeps the neighbours' light and would charge it to this scene), with one
    free parameter per template plus the shift coefficients. Sorting on it is
    the direct way to find the scenes worth looking at.
  * `Scene.shift_scatter()` -> `shift_rms`, the spread of applied shifts about
    their mean. At order 0 every template gets the same offset, so non-zero
    means the field moved between passes -- the scene walked rather than
    settled, which is a direct detector for the non-convergence in TODO.
  * `is_bright` -> `n_anchor` in the CSV and `n_bright` -> `n_anchor` in the
    SCENES extension: the count is templates passing the *anchor* cuts (SNR,
    isolation, star exclusion), and isolation is not brightness. It is the
    count before robust rejection; `astrom_neff` is after. `verification.py`
    and `examples/compare_dr0_dr0.1.py` follow.
- [x] Refit validated against the run on COSMOS F770W scene 1313 (2026-08-16):
  107 of 107 templates rebuilt and the baseline within 0.035 px of the run's
  own shift, so the frozen-membership refit path reproduces a normal scene.
  Scene 16's 24-source shortfall is the `ff3b8d4` version skew, not a refit
  defect. The robust pass declined 1313 (6 anchors, `n_eff` 4.9 below the
  `scene_minimum_anchors` gate of 5) and was then an exact no-op: `dchi2` 0,
  zero flux change on all 107 sources. Note the interaction -- a scene with
  exactly `scene_minimum_anchors` anchors nearly always declines, because any
  real disagreement pulls `n_eff` below the count.
- [x] An upsampled band keeps its weight on the same grid as its image
  (2026-08-16). `_convolved_templates` upsampled `images[ifilt]` onto the
  reference grid and set `wcs[ifilt] = wcs[0]`, but the upsampled weight only
  ever lived in a local: the instance was left holding a reference-grid image
  beside a native-grid weight, and the collapsed WCS hid it, because the next
  call computes `k = 1` and upsamples neither. `load_fit`'s replay of the same
  transform had the same omission.
  * A second pass over one band -- exactly what `_solve_frozen_scene` does --
    then sliced an 80 mas weight map with 40 mas coordinates. Numpy clips
    out-of-range slices rather than raising, so the read silently returned the
    wrong region of sky.
  * Symptoms on COSMOS F770W scene 16: 48 templates over covered sky pruned as
    uncovered (236 -> 188), the scene residual masked over valid pixels with
    the mask tracing a footprint offset from the data, and the fitted shift
    corrupted to (+0.03, -0.92) px against the run's (-0.02, +0.00). After the
    fix all 236 available templates survive, and the baseline lands at
    (+0.22, -0.09) px. Found from the residual figure, not from a test.
  * Both call sites now write the upsampled weight back, and
    `_convolved_templates` raises when the weight and image shapes disagree
    rather than trusting the pairing. Tests
    `test_upsampled_band_keeps_its_weight_on_the_same_grid` and
    `test_convolved_templates_rejects_a_weight_on_the_wrong_grid`.
  * Every refit measurement taken before this is void.
- [x] PSF settings moved into a `psf` config block, `expect_frames` removed
  (2026-08-16). `RunConfig` carried ten flat `psf_*`/`pattern_*` keys; they
  are now a nested `PsfConfig` (`dir`, `pattern_hi`, `pattern_lo`, `size`,
  `autobuild`, `provenance`, `workers`, `fov_arcsec`, `date_mode`,
  `blur_fwhm`), reached as `cfg.psf.<field>` and written as a JSON object the
  way `fit` already was. The redundant `psf_` prefix is dropped inside the
  block. `filter_lo` stays top-level: it names the band for figure labels as
  well as for the blur lookup. A JSON dict is coerced by `__post_init__`,
  unknown keys inside the block raise, and `from_json` raises on the old flat
  keys naming their replacements rather than ignoring them.
  * `expect_frames` is gone. It asserted the row counts of the same two WCS
    csvs the run reads, so it could only fire when the exposure list changed
    -- which is when a run wants the new frames, not a raise. The stale
    COSMOS value (518 against the 288 frames now in the table) is what it
    produced in practice. `_ensure_dpsfs` logs the two counts instead.
  * Migrated: the five hand-written `examples/*.json`, the 18 generated
    `examples/minerva/*.json`, `make_minerva_configs.py` (which no longer
    counts csv rows), the ozstar/canfar `build_psfs.py`, `campaign.py`,
    `ozify.py` and `arcify.py`, and the docs (`pipeline.md` gained a
    "The `psf` block" reference section; `quickstart.md`, `psf.md`,
    `repair.md`, `campaigns.md`, `psf_maps.md` and `TODO.md` follow the new
    spelling). Full suite: 473 passed.
  * `docs/conf.py` include_patterns gained `precision.md`, which was already
    in the index toctree and had been warning on every build. `campaigns.md`
    went the other way: it documents the MINERVA run campaigns, not mophongo,
    so it is out of the index toctree and out of the build (the file stays in
    `docs/` alongside `examples/ozstar` and `examples/canfar`, which live in
    the repo for convenience and will move out later).
- [x] `docs/quickstart.md` installation section now starts from the GitHub
  checkout (clone -> `poetry install` / `pip install -e .`) and states how to
  reach the `mophongo` console script (`poetry run`, `eval $(poetry env
  activate)`, or sourcing `.venv/bin/activate`), since Poetry 2.x's
  `poetry env activate` only prints the command instead of running it.
- [x] `_solve_frozen_scene` refuses to rebuild a scene it cannot rebuild whole
  (2026-08-16). Extraction is positions-driven, so a frozen source id that is
  not a row of `self.catalog` contributes nothing: the scene came back short,
  or -- when nothing matched -- empty, and the failure surfaced far downstream
  as `ValueError: No templates to convolve`. It now raises where the mismatch
  happens, naming the counts and the first missing ids. The usual cause is a
  config whose footprint or trial cut differs from the run's, which is easy to
  arrange when two configs share one `out_dir`: `examples/minerva`
  `cosmos_f770w.json` (3' trial) and `cosmos_f770w_full.json` (full field)
  both write to `cosmos_f770w/` under the name `cosmos_f770w`, so whichever
  ran last owns the outputs and reloading with the other one silently
  describes a different source set.
- [x] A refit is refined exactly as the run refined it, and a scene's reported
  shift is the shift it was actually given (2026-08-16).
  * `Pipeline._refine_scene_astrometry` is new and shared: up to
    `fit_astrometry_niter` solve/apply passes at `astrom_damping`, stopping at
    `astrom_shift_tol`, then the closing flux-only solve. `run()` and
    `_solve_frozen_scene` both go through it, where `_solve_frozen_scene` used
    to take a single `scene.solve()`. A one-pass refit is not comparable with
    anything the run wrote -- the scene catalog and scene figures report the
    accumulated shift after the loop -- so a refit read as a large
    disagreement when it was only an unfinished one. `run()`'s inline loop is
    gone rather than duplicated, which is what let the two drift.
  * `Scene.mean_shift()` is the scene's one reported shift: the plain mean of
    `Template.shifted` over its members, at the centroid of those members.
    The scene catalog's `dx`/`dy` and the annotation on the scene figure's
    model panel both use it. Previously both refit a Chebyshev field of the
    scene's order to the accumulated shifts and evaluated it at the centre.
    That is an approximation of something already known exactly: accumulated
    shifts are a sum of damped increments, each fitted at whatever the
    previous pass left behind, so at order >= 1 the total is not in general
    representable by the functional form of any single pass. The two can
    differ by ~0.1 px. `_scene_shift_samples` keeps the field fit, which is
    the right tool for the shift-field figure's spatial structure.
- [x] `Templates.prune_outside_weight` no longer depends on the set it is
  handed (2026-08-16). The drop threshold was `rtol * median(wnorm)` over the
  templates being pruned, so a template's verdict depended on the company it
  kept. Found through `Pipeline.refit_scene`: refitting scene 16 of the COSMOS
  F770W run rebuilt 184 templates where the run recorded 260. The scene's
  members are brighter than the field median, so pruning them on their own
  raised the threshold and took 76 faint edge members with it -- and
  `refit_scene`'s whole premise is that extracting a subset gives the same
  pixels as extracting it alongside everything else.
  * The test is now per-template and dimensionless: the fraction of a
    template's own squared flux landing on positive weight, against `rtol`.
    That is what the docstring always claimed the function did.
  * `rtol` default 1e-8 -> 0.0, since it now means a coverage fraction rather
    than a fraction of a population median; the two are not comparable. 0.0 is
    the documented rule exactly -- a single usable pixel keeps the template,
    and the sum is identically zero when there is none, so no tolerance is
    needed. An intermediate 1e-3 was tried first on the reasoning that it
    excluded mosaic-edge slivers "permissively"; measured on COSMOS F770W
    scene 16 it cut 56 of 236 templates, because an edge scene's members
    genuinely do have under a thousandth of themselves on usable pixels.
    Raising it is a real cut, not a numerical guard.
  * Tests `test_prune_outside_weight_is_subset_invariant` and
    `test_prune_outside_weight_drops_templates_off_the_weight_map`.
  * Anything that re-solved a subset was affected, not just refits.
- [x] The solved shift field survives the figure loop (2026-08-16). A band that
  dies drawing scenes has already solved its astrometry, and until now that
  solution died with it: only the per-template applied shifts were persisted,
  in `<name>_templates.fits`. `write_outputs` now also writes a `SCENES`
  extension of `<name>_fit_table.fits`, and writes it *before* the first
  figure, which is the point -- the figure loop is where three of the
  seventeen bands of the overnight campaign stopped.
  * `Pipeline._scene_fit_table` records `shift_coeff`, `shift_order` and the
    `(x0, y0, sx, sy)` normalisation, which are exactly the arguments of
    `AstroCorrect.build_poly_predictor`, plus `astrom_damping`, the per-scene
    astrometry counters and the robust-pass verdict. Read it with
    `Table.read(f_fit_table, hdu="SCENES")`.
  * The coefficients are stored rather than recovered because the two are not
    the same field. `_scene_shift_samples` refits a polynomial to the applied
    shifts of every template; the solution was fitted on the scene's anchors.
    Close enough to plot an arrow, not close enough to rebuild from.
  * Two limits of what that field *is*, both now stated in the docstring.
    `Scene.solve` overwrites `shifts` on every astrometric pass
    (`scene.py:1512`, driven by the loop at `pipeline.py:4847`), so a scene
    with `astrom_niter` above 1 kept only the final pass -- the accumulated
    offset lives in the per-template `dx`, `dy` instead. And the coefficients
    are undamped: the applied shift is `astrom_damping` times the field
    (default 0.8), which is why the factor is stored beside them rather than
    left to whichever config is loaded later.
  * Order is per scene, not per run -- a saturated scene is forced to order 0
    so its fragments move rigidly -- so `shift_coeff` is padded with NaN to
    the widest order present and cut back with `n_coeff`. `shift_order` is -1
    for a scene that solved no shifts.
  * `_scene_catalog.csv` is trimmed to `id, n_templates, is_bright, ra, dec,
    dx, dy, flag_astrom, minerva_link`. It is the file for reading; the
    counters it lost are in the extension, which is the file for machines.
  * The read side is wired. `load_outputs` picks up the extension as
    `Pipeline.scene_fit`, `restore_scene_fit(scenes)` puts the coefficients,
    basis and counters back onto scenes regrouped from `id_scene`, and
    `write_scene_catalog` is now a public method both `write_outputs` and
    `examples/canfar/jobs/scene_plots.py` call, so the recovered file is the
    same file. The replot re-emits `_scene_catalog.csv` and `_shift_field.png`
    when the extension is there, and skips both with a warning when it is not
    -- which is the case for anything written before today, run2's three
    replotted bands included.
  * `_scene_catalog.csv` is trimmed to `id, n_templates, is_bright, ra, dec,
    dx, dy, flag_astrom, minerva_link`. It is the file for reading; the
    counters it lost are in the extension, which is the file for machines.
  * `RunConfig.minerva_viewer` takes a full FITSMap URL, resolved against
    `FITSMAP_URL` when given as a bare `<field>/<release>`. The fields do not
    agree on whether the release belongs in the path -- COSMOS serves from
    `/cosmos`, UDS from `/uds/DR0` -- so the derived guess is gone: None or ""
    drops the column rather than writing a link that does not resolve.
  * Log lines name the band. `upsampling image 1 by factor 2` was the only
    place a run log identified an image by index; `_band_label` makes it
    `upsampling f770w (cosmos-...-f770w_drz_sci.fits) by factor 2`, and falls
    back to the index for a pipeline driven without a run config.
  * Cost: a bare `Table.read(<name>_fit_table.fits)` now warns that multiple
    tables are present before returning the fit table, which is still HDU 1.
    `load_outputs` passes `hdu=1`; external readers should too.
  * `poetry run pytest`: 466 passed. `tests/test_scene_fit_table.py` (17)
    covers the order inversion, NaN padding across mixed orders, the
    unsolved-scene case, the damping factor, the restore path and its no-op on
    an older run; two tests in `tests/test_pipeline.py` drive the real
    `write_outputs`, one checking that the stored coefficients reproduce the
    live scene predictor exactly and one covering the viewer-URL spellings
    including both drop cases. A manual round trip confirmed the re-emitted
    scene catalog is byte-identical to the one the run wrote.
- [x] Docs restructured after a read-through of the RTD pages (2026-08-16).
  Reference material that had been duplicated in the getting-started pages now
  lives with the component it belongs to: the `RunConfig` fields and the
  `pipeline.run()` / `Pipeline.__init__` arguments are in `docs/pipeline.md`
  only, and the `matching_kernel()` parameter list moved to `docs/psf.md`.
  `docs/quickstart.md` and the overview keep the examples and a pointer.
  `docs/templates.md` gained the general statement behind the normalization
  order -- a correction supplying support outside a segment must not be
  applied over a neighbouring segment, because segmentation does not deblend
  and the neighbour's own template already models those pixels -- and a new
  *Encircled energy of a template* subsection: `EE_tmpl = EE_psf` for a point
  source, `EE_tmpl < EE_psf` for an extended one after convolution, with the
  deficit measurable as `EE_tmpl_hi / EE_tmpl_lo` (the `psfcor_<i>` column).
  The `flux_<i>` entry in `docs/outputs.md` now states what the column is and
  refers to those sections instead of restating the normalization.
- [x] run2 on CANFAR is complete at 17 of 17 bands, and the replot that
  recovers a dead one works for the first time (2026-08-16). Three bands of
  the overnight campaign -- `cosmos_f770w`, `cosmos_f1280w`, `egs_f2100w` --
  fitted and then stopped partway through their scene figures with no
  traceback, having written 196, 577 and 37 of them. All three now carry the
  full set, along with `_stamps.h5`, `_scene_map.png` and `_scene_blobs.png`.
  * `examples/canfar/jobs/scene_plots.py` had never drawn a figure.
    `Scene.model_image` raises `RuntimeError("No solution available")` when
    `solution` is None, and a scene regrouped from `id_scene` has never been
    solved, so the first attempt spent 2h12m to report "wrote 0 of N" three
    times over. `solution_from()` now carries the loaded per-source fluxes
    onto each rebuilt scene. That guard is the only thing on this path that
    reads `solution` -- the model accumulates from `t.flux` and `t.data`,
    both of which `load_fit` restores -- so the fit's own numbers satisfy it
    rather than a weakened check. The script's docstring claimed it populated
    everything `Scene.plot` reads "because nothing is solved", which was the
    false premise; it now says otherwise.
  * The script also draws the two full-field partition views, from the same
    `save_scene_overview` and `save_scene_blobs` helpers and under the same
    names `write_outputs` uses, so a recovered band is not missing products a
    band that finished on its own has.
  * `_scene_catalog.csv` and `_shift_field.png` are deliberately not rebuilt
    and stay missing on those three. Both carry per-scene astrometry -- `dx`,
    `dy`, `astrom_niter`, `flag_astrom` -- recorded during the solve and not
    persisted, so anything written would be a plausible file with invented
    columns.
  * CANFAR defaults are 16 cores and 96 GB, from 8 and 64 (82 for EGS).
    `FIELD_RAM` is empty: EGS had more on the argument that its 1221 Mpx
    detection grid needed it, but OzStar measured EGS at 29-59 GB, below the
    other fields, and `egs_f2100w` died holding 82 anyway. 16 cores costs
    little here because a skaha session is a Kubernetes pod rather than a
    share of a node, so the request cannot change co-tenancy the way it does
    in `examples/ozstar`; it only lengthens the queue.
  * What killed the three bands in the first place is still unknown. The
    replot reruns completed at 96 GB, but on a different code path, so that is
    not evidence memory was the cause. `free -g` inside a pod reports the
    node's memory -- 755 GB in these logs -- not the pod's limit, so the job
    banner is not a headroom measurement. `GET /session/<id>` distinguishes
    `OOMKilled` from `Evicted`, and is worth reading before the record ages
    out the next time a band dies silently.
- [x] The OzStar campaign takes the CANFAR shape (2026-08-16). The two
  platforms now expose the same steps, filters and flags, so one campaign
  reads the same way on either; what stays different is who enforces the order
  between phases, which is SLURM here and the laptop there.
  * `examples/ozstar/campaign.py` steps are
    `ozify push setup seed psf stage repair run`. `prep` was renamed `repair`
    to match CANFAR and the pipeline step it actually runs, with `prep`
    accepted as an alias by `--from`/`--skip`. `--bands`, `--note`, `--ref`,
    `--force-stale`, `--push-psf` and a `--ram` spelling of `--mem` are new;
    `--prep-time` is now `--repair-time` (old spelling still accepted).
    `--branch` and `--ref` are one flag: the ref `setup` clones and the ref the
    jobs are checked against have to agree, and two flags that could disagree
    would mean refusing to submit the branch just cloned.
  * `psf` is a campaign phase rather than something done beforehand. It lists
    `$OZSTAR_BASE/PSF`, skips when every family the configs name is present,
    and otherwise starts the login-node build and blocks on it. It is the one
    phase that must block: the build queries MAST for each exposure's OPD, so
    it cannot be a SLURM job and nothing can depend on it.
    `submit.wait_for_psf_build` polls for `PSF_BUILD_DONE` and judges liveness
    by the log growing, since `nt.swin.edu.au` round-robins over four login
    nodes and the build's pid means nothing on the others.
  * The grid check gained `pattern_lo` and the fov-agnostic `_GRID` rewrite
    that `mophongo.jwst_psf.fov_agnostic_pattern` does, which the OzStar copy
    was missing; `has_grids(..., shared_only=True)` keeps the old hi+halo
    question for the leader fallback.
  * `submit.check_src_current` refuses to submit against a clone that is not
    at `--ref`. The cluster pulls GitHub, so an unpushed local commit is simply
    absent from the run while every output looks normal. The campaign pulls
    once up front and passes `sync=False` to each dispatch, so a commit landing
    mid-submission can no longer split a campaign across two versions.
  * Each campaign writes `$OZSTAR_RUN/README.md` before submitting: mophongo
    commit, the release versions each field is pinned to (via
    `arcify.config_versions`), `--note`, and a diff against the previous run
    directory.
  * `examples/canfar/campaign.py` gained `--bands` and a `--mem` spelling of
    `--ram` for the same symmetry.
  * `ozify.py --check-versions` reports configs pinned to an older release than
    arc now holds, reusing `arcify.check_release_versions` the way the rest of
    the arc index is reused. It rewrites nothing, and it runs on the laptop:
    listing arc needs only the CADC certificate, and the datamover partition
    exists to copy inputs to `/fred` rather than to see them. The advice it
    prints is the OzStar one - a new release means a new `$OZSTAR_RUN` *and* a
    re-stage, since `/fred` holds a copy of each input. Both flags are now
    documented in the two READMEs; neither was before.
  * Both `campaign.py` take `--check-versions` as a pre-flight: it reports what
    is behind, then carries on with the pinned versions rather than refusing.
    Moving onto a new release changes the photometry, so it is a deliberate act
    belonging to a new run with a note saying why - not something a launch
    should do, or block on, because a directory appeared upstream. It runs
    under `--dry-run` too, which is when the answer is most useful.
  * Each campaign accepts the other platform's name for the config-rewrite
    step, so `--from arcify` works on OzStar and `--from ozify` on CANFAR, and
    the CANFAR campaign gained `--ref`/`--force-stale` (passed to every
    `submit.py run` dispatch, so one campaign checks one ref). What is left
    platform-specific is only what has no counterpart: the OzStar walltimes
    (`--time`, `--repair-time`, `--stage-time`) and `--push-psf`.
- [x] Robust astrometric anchor weighting, `FitConfig.astrom_robust`
  (2026-08-16, branch `shift-robust`, off by default). The shift field is
  fitted by least squares, so an anchor's pull scales as its Fisher
  information `I_i = alpha_i^2 <grad T_i, w, grad T_i>` -- as the *square* of
  its flux. One bright extended source with an asymmetric colour gradient
  therefore carries a scene: its residual is a dipole aligned with the
  template gradient, formally indistinguishable from a displacement.
  * `astrom_robust.py` (new leaf module) takes an anchor table -- implied
    shift, information, basis row, misfit -- and returns a per-anchor weight.
    It is an MM-estimator: a leverage-blind high-breakdown start
    (`_robust_start`), then redescending Tukey steps that put the information
    back. Both halves are needed, and this was found by test rather than by
    reasoning: the first implementation started from the
    information-weighted least-squares fit with a Huber warm-up, and a
    400x-leverage liar defeated it outright. Huber's weights decay only as
    `1/u`, so a tenfold downweight still left the liar holding 83% of the vote
    and the fit walked back onto it. See
    `test_scene_astrometry_robust.py::test_a_dominant_liar_pulls_the_scene_and_robust_weighting_stops_it`.
  * The weight has two parts. A **systematic floor** `s`, estimated by robust
    moment matching on the anchors' scatter about the fitted field, makes an
    anchor's weight saturate at `1/s^2` however bright it is -- which is what
    `astrom_leverage_cap` approximates with a quantile, now set by the data
    instead. And **Tukey rejection** on disagreement with the field, the only
    discriminator for a morphology-driven pseudo-shift: a real offset is
    smooth in position, a pseudo-shift is random per source.
  * `scene.measure_anchor_shifts` produces the table: one 3x3 fit per anchor
    on `(T_i, -grad_x T_i, -grad_y T_i)` against the **flux-only** residual,
    giving the implied shift, its flux-marginalized information, and
    `chi2_red` -- what is left once a displacement is projected out. That last
    quantity discriminates a source that moved from one whose template is
    wrong, and it is why "residual size does not discriminate" (TODO, since
    revised) was only half true: residual size *after removing the shift*
    does.
  * Measuring against the flux-only residual is exact, not an approximation.
    Eliminating the fluxes from the joint system gives
    `(B'WB - B'WA(A'WA)^-1 A'WB) beta = B'W r0`, so the flux-only residual is
    precisely what the shift block responds to. Verified to 1.7e-15 at orders
    0 and 1, with and without noise. That is what removes the need for an IRLS
    loop around the joint solve: the weights are known before the joint system
    is assembled. Cost is one extra flux-only solve and one union-bbox
    residual buffer per scene per pass.
  * `assemble_scene_system_AB` gained `anchor_weights`, multiplied into the
    existing `lev_w`, and drops zero-weight anchors from the derivative
    columns entirely rather than carrying them at zero, where they would still
    enlarge the dense buffers. Hard rejection is also the one exactly
    consistent case of the `lev_w` two-power split (see TODO).
  * Works at any polynomial order; order enters only through the basis width.
    Gated on `max(scene_minimum_anchors, 2 * n_terms)` anchors -- the same
    number `merge_small_scenes` merges scenes up to, so a scene built to
    support the shift model is by construction big enough to be judged.
  * Where it applies it **supersedes** `astrom_leverage_cap` for that scene,
    rather than composing with it: both bound one anchor's pull, but the cap
    does it blind (it clips whichever anchors carry the most information,
    usually the best ones) while the floor sets the same ceiling from the
    anchors' measured scatter. The cap stays on wherever the robust pass
    declines.
  * `scene_minimum_anchors` default 5 -> None, i.e. derived from the
    astrometric order as `(order+1)(order+2)+1`: 3 at order 0, 7 at order 1.
    A hand-set 5 did not track the order it was supposed to support. At the
    shipped order 0 this lowers the merge floor from 5 to 3, so scenes merge
    less; `docs/fitting.md` now quotes 0.10 fit pixels for the weakest
    admissible scene's centroid rather than 0.08.
  * `astrom_isolation_thresh` stays 0.7 -- see the separate entry below. The
    robust pass does *not* make it redundant: the two cover different
    failures, and a blended majority defeats robustness by construction.
  * The per-anchor measurement is **conditional on the anchor's
    neighbourhood**, not marginal. Written the marginal way first, it was
    badly wrong for exactly the anchors the isolation cut admits, in two
    separate ways, both found by measurement rather than by reasoning:
    - *A neighbour's free flux.* The residual arrives with fluxes already
      fitted, and a neighbour to one side absorbs part of a dipole by
      adjusting its brightness. A bright anchor with a faint blended
      neighbour read 0.095 px against a true 0.20 -- half its shift, in the
      blend direction, with `chi2_red ~ 0` to show for it -- while its
      isolated twin read 0.20 exactly. Fixed by a flux column for every
      overlapping template, fitted over their union footprint: 0.105 ->
      0.013 px.
    - *A neighbouring anchor's free shift.* Two overlapping anchors leave
      overlapping dipoles; holding one at zero split them badly. A pair at
      6 px read 0.089 and 0.039 px against 0.20 -- both low, both agreeing
      with each other, and both carrying more information than the honest
      anchors they disagreed with, which is the configuration most likely to
      capture a robust fit. With each overlapping anchor given its own free
      displacement: 0.171 and 0.151.
    The numerator `<G_i, w, r0>` was right in every version; it is the
    information divided by that has to be marginalized over everything local
    and free. What remains (neighbours of neighbours, the union footprint
    standing in for the global flux constraint) is smaller than the anchor's
    own reported uncertainty in every case tested -- the property that
    matters, since a bias inside the error bar cannot make a blend look like
    a liar. Tests
    `test_a_blended_anchor_is_measured_conditionally_on_its_neighbour`,
    `test_two_blended_anchors_are_measured_against_each_other`,
    `test_an_unmodelled_neighbour_shift_shows_up_as_misfit`,
    `test_a_degenerate_blend_reports_its_own_uncertainty`.
  * `chi2_red` is evaluated as an explicit residual rather than by the
    cancellation `q - s'theta`, which in a well-fitting scene left roundoff.
    Since the misfit inflation is a *ratio* to the scene median, roundoff
    ratios were being amplified into real downweights; `CHI2_FLOOR = 1e-2`
    now stands the inflation down when a scene already fits orders of
    magnitude inside its own noise.
  * Nothing here can help a scene whose only bright member is the offender:
    with one anchor the weight is a global scale, and a global scale cannot
    move the field. Only a veto could. Recorded in TODO.
  * Built for A/B: one flag, off by default, and `False` leaves every code
    path it touches untouched. The diagnostics are written either way so a
    pair of runs differs in values rather than in schema -- fit table gains
    `astrom_weight_<i>` (per source, the weakest weight among its templates),
    the scene catalog gains `astrom_robust`, `astrom_nreject`,
    `astrom_floor` and `astrom_neff`, and the stamps table carries
    `astrom_weight` per template so `load_fit` restores it.
  * Tests: `tests/test_astrom_robust.py` (23, synthetic anchor tables at
    orders 0/1/2) and `tests/test_scene_astrometry_robust.py` (25, scene and
    pipeline level, noiseless and noisy, including the end-to-end A/B
    check). Full suite 445 passed.
- [x] `FitConfig.astrom_isolation_thresh` stays at 0.7, now on measured
  grounds rather than inheritance (2026-08-16). It was loosened to 0.6 and
  then 0.5 on the reasoning that the cross-anchor terms in
  `assemble_scene_system_AB` already handle blending and that `astrom_robust`
  would catch whatever leaked through. A threshold scan (4 blended pairs at
  3/5/7/9 px separation, sigma = 2.5 px, plus 4 isolated anchors, 6 seeds)
  says otherwise -- the robust pass flips from harmful to helpful between
  0.55 and 0.60:

  | threshold | anchors | robust off | robust on |
  | --- | --- | --- | --- |
  | 0.50 | 12.0 | 0.0344 | 0.0714 |
  | 0.55 | 12.0 | 0.0344 | 0.0714 |
  | 0.60 | 10.2 | 0.0207 | 0.0171 |
  | 0.70 | 10.0 | 0.0177 | 0.0139 |
  | 0.90 |  6.0 | 0.0019 | 0.0024 |

  (median |beta error| in fit pixels.) The cut is in effect a separation cut
  -- dominance 0.54 at 0.8 PSF sigma, 0.59 at 1.2, 0.66 at 1.6, 0.73 at 2.0,
  0.97 at 3.6 -- so 0.6 admits blends down to ~1.2 sigma and 0.7 to ~2. Same
  shape at 5x the noise. The scan cannot set the upper end, which is fixed by
  anchor availability on a real field rather than by this synthetic; 0.7 is
  the shipped value and the one `examples/run_uds_770_wren.py` already passes
  explicitly.
- [x] The remote-access shell toolkit left the repo (2026-08-15).
  `canfar-cert.sh`, `canfar-common.sh`, `canfar-mount.sh`, `canfar-umount.sh`,
  `canfar-sync.sh`, `canfar.conf` and `ozstar-mount.sh` now live in
  `~/bin/remote/`. They mount `/arc` and `/fred/oz030` over sshfs and fetch the
  CADC certificate, which is machine setup rather than mophongo: four
  `com.ivo.*-mount` LaunchAgents run them at login, and a `RunAtLoad` path
  pointing into a gitignored `scratch/` of a repo being prepared for release
  is the wrong thing to depend on. `scratch/canfar/` keeps what is mophongo's:
  `jobs/`, `README.md`, `RUNNING_ON_CANFAR.md`.
  * `runroot.canfar_user()` reads `$CANFAR_CONF`, defaulting to
    `~/bin/remote/canfar.conf`, and no longer takes a `repo` argument. It had
    no callers, so the signature change is contained.
  * The cert-script references in `examples/canfar` and `examples/ozstar`
    (READMEs, `MANUAL.md`, `release.sh`, and the `SystemExit` messages in
    `submit.py`, `arcify.py`, `ozify.py`) name `~/bin/remote/canfar-cert.sh`
    instead of a repo-relative path.
  * Both mount scripts drop `ServerAliveCountMax` from 10 to 3. FUSE-T serves
    these mounts to macOS as a loopback NFS server, so when the ssh link dies
    the NFS client declares the volume down after
    `vfs.generic.nfs.client.initialdowndelay` and `NetAuthAgent` raises
    "Server connections interrupted". 45s of keepalive fits inside that
    window; 150s did not.
- [x] The repair cache stores the flagged ids, not the whole catalog
  (2026-08-15). `Pipeline._save_repair_cache` wrote a FLAGS row for every
  source, so COSMOS run3 held 294126 rows of which 596 were nonzero, and
  every band's reload walked all of them in a python `zip` loop, once per
  flag column. It now keeps only the rows a flag column marks, and
  `_load_repair_cache` maps the ids through one vectorised assignment.
  * Behaviour is unchanged: the loader already zero-filled a missing column,
    and now rebuilds each column from zero rather than merging, so an id the
    cache omits reads as "not flagged" exactly as before. An id absent from
    this band's catalog is still skipped.
  * The write log line stops disagreeing with the reload line -- both now
    report the flagged count.
  * Old dense caches still load correctly; the format is a superset.
  * `webbpsf` is gone from the dev group. It has not been imported since the
    move to `stpsf`, nothing sets `WEBBPSF_PATH`, and webbpsf 2.x is only a
    deprecation shim over stpsf. Removed with `poetry remove --group dev`.
  * `poetry run pytest tests/test_pipeline_config.py tests/test_pipeline.py
    tests/test_repair.py tests/test_repair_saturated_catalog.py
    tests/test_pipeline_provenance.py tests/test_scene_saturated.py`:
    111 passed, including a new round-trip test.
- [x] ePSF completeness is a question about dates (2026-08-15).
  `Pipeline._missing_psf_dates` returns `dates_from_csv(csv, psf_date_mode)`
  minus the `_MJD` tokens of the files matching the pattern, and `_load_epsf`
  builds exactly those, one `PSFFactory(date_mode=<mjd>)` call each. A band
  holding 99 of 100 epochs costs one grid. It also catches what a match count
  cannot: a set built under the old `modal` default holds one grid for a band
  spanning years, matches the pattern, loads, and looks healthy -- here it is
  one epoch present and the rest missing.
  * Grids carry no provenance any more. `grid_provenance`, `csv_fingerprint`,
    `read_stdpsf_provenance` and `write_stdpsf(provenance=...)` are gone, with
    the `HIERARCH MPH` cards they wrote. A grid is its detector, filter, epoch
    and field of view, all of them already in the filename, and none of them a
    function of the rest of the exposure list -- so stamping the list meant
    that adding one frame invalidated every grid built from it.
  * FOV is not compared either: a drizzled stamp smaller than the grid it
    comes from is fine, and `_check_psf_size_fits_grids` warns when it is not.
  * `RunConfig.psf_provenance` keeps its name and now decides only what
    happens when epochs are missing and `psf_autobuild` is off: `"warn"`
    (default) names them and fits with the rest, `"error"` refuses, `"off"`
    skips the check.
  * Removed with it: `examples/canfar/jobs/{convert_psf_names.py,
    restamp_psfs.py,convert_all_psfs.sh}` and
    `tests/test_convert_psf_names.py`, whose purpose was stamping those cards.
    The FOV renaming they also did is already applied to all 493 grids on arc.
  * `poetry run pytest`: 391 passed.
- [x] Deep audit of `astrowhit/aperpy@aperpy-2` and clean-slate aperture
  photometry design (2026-08-14). The upstream branch is pinned at
  `dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038`; the completed report is
  `docs/APERPY2_CLEAN_IMPLEMENTATION_REPORT.md`.
  * Decision: build a sibling Mophongo `AperturePipeline`, not an upstream
    wrapper/direct port and not an expansion of the template-fit `Pipeline`.
    `DrizzlePSF`-derived `PSFRegionMap` products should be the strict JWST
    default, with an explicit empirical/static fallback for unsupported or
    unprovenanced inputs.
  * The PSF work is not the dominant implementation cost: local PSF/kernel
    maps and region-wise whole-image convolution already exist. The main gaps
    are arbitrary-radius absolute COG lookups, mask/variance/covariance
    treatment, direct SEP aperture/Kron semantics, empirical empty-aperture
    calibration, artifact provenance, and validation fixtures absent upstream.
  * The readiness audit found three PSFFactory regressions: public `build()`
    raises `NameError` on undefined `csv_path`/`mode`, `from_csv()` does not set
    the provenance its writers consume, and filenames hard-code `OS4`.
    `PSFRegionMap` persistence/cache identity and strict no-coverage behavior
    also need hardening before it is a production default; tracked in TODO.
  * Validation: focused existing convolution/EE tests passed (20 passed,
    14 deselected). The `PSFFactory.build()` failure was reproduced directly.
    No full suite was run for this documentation-only audit.
- [x] JWST ePSF grids build at a stated 101 native pixels (2026-08-14).
  `jwst_psf.DEFAULT_FOV_PIXELS = 101` is passed as `fov_pixels` whenever a
  config sets no `psf_fov_arcsec`, instead of omitting the key and inheriting
  whatever `stpsf.gridded_library.CreatePSFLibrary` falls back to. Same grid
  as before -- that fallback is also 101 -- but the size now lives in this
  repo and cannot change under a dependency bump.
  * `default_fov_arcsec` returns `101 x pixelscale`, read from the live stpsf
    instrument rather than tabulated, so it is per *detector*: 6.354" on the
    NIRCam long-wave SCAs, 3.11" short-wave, 11.203" MIRI. It previously
    returned 4.09/8.10, which are what `fov_arcsec=4`/`8` produce after the
    odd-parity bump -- the values the older grids were built with, not any
    default. A grid built with no FOV set was therefore *named* FOV4 while
    holding 6.35", the ambiguity the token exists to remove.
  * `psf_size` is unchanged and unrelated: it is the drizzled stamp side,
    applied to both bands, and the grid FOV only has to be at least that.
    `Pipeline._check_psf_size_fits_grids` now warns when it is not, naming the
    smallest offending grid -- `eval_ePSF` returns zero outside the grid, so
    the stamp would otherwise be silently zero-padded and the PSF wings lost.
  * Loading stays FOV-agnostic (`fov_agnostic_pattern`), so the FOV-less
    patterns every config carries keep finding both generations of grid.
    Verified against the 493 grids on arc: a `..._MJD\d+_GRID9_OS4` pattern
    loads the nine `FOV8`-named UDS F770W grids and reads their FOV back.
  * The 493 grids were audited for a `(stem, MJD, GRID, sampling)` key served
    at more than one FOV: none. Three families hold two FOVs over disjoint
    epochs (see TODO.md).
  * `poetry run pytest`: 396 passed.
- [x] PSF grid filenames always carry the field of view (2026-08-14).
  `PSFFactory.include_fov` now defaults True, so a grid is named
  `..._MJD60308_FOV4_GRID25_OS4.fits` and FOV4, FOV8 and FOV30 sets of the
  same GRID/OS layout are distinct files that can share one directory instead
  of needing one each.
  * When `fov_arcsec` is unset the token still appears, resolved from stpsf's
    own per-instrument default (`jwst_psf.default_fov_arcsec`: NIRCam 4.09,
    MIRI 8.10). A grid built without an explicit FOV had one all along; it
    just was not written down.
  * Reading is backwards compatible in both directions.
    `jwst_psf.fov_agnostic_pattern` inserts an optional `(?:_FOV\d+)?` before
    `_GRID` when a pattern names no FOV, so every config written so far finds
    both the old grids on disk and the new ones. A pattern that does name an
    FOV is left alone -- the 30" halo patterns depend on being specific.
    Applied at both match sites: `EffectivePSF.load_jwst_stdpsf` and
    `Pipeline._stale_psf_grids`, so the staleness check sees exactly the files
    the loader will load.
  * `include_fov=False` reproduces pre-2026-08 names exactly.
  * The relaxation is deliberately loose and can match a grid built at another
    field of view with the same GRID/OS layout -- the ambiguity the token
    exists to remove. New configs should name the FOV they want; this is for
    reading what is already on disk.
  * `poetry run pytest`: 389 passed.
- [x] CANFAR run trees restructured around a release (2026-08-13). The root
  defaults to `/arc/projects/minerva/ifl/release_v1.0` and a tree under
  `/arc/home` is refused: the v1.0 campaign had put 200 GB there -- 132 GB of
  outputs across 15 bands, 67 GB of staged inputs -- against a home quota of a
  few hundred GB for everything a user owns.
  * Layout: `setup/` (source, venv, configs, tarballs), `data/`, `PSF/`,
    `run<N>/<field>/<band>/`, `run<N>/<field>/<field>_repair_cache.fits`.
    `$CANFAR_RUNNUM` picks the run number, so a re-run bumps a number rather
    than inventing a name suffix.
  * `runroot.run_number()`, `arcify.py` (out_dir, repair cache name),
    `jobs/*.sh` (setup/ paths, run<N> globs, seed_cache finds a band under any
    run/field), `submit.py` (uploads to `setup/`, `fetch` reads
    `run<N>/<field>/<band>/`).
  * The orphaned `out/repair_cache.fits` in home is EGS's -- its
    `REPAIR SCI_HI` names the EGS detection image -- so all three fields had
    been overwriting one file in turn. The `<field>_` prefix is what prevents
    that; a trial patch keeps its geometry in the name as well, since a patch
    and a full field repair different pixels.
  * Done through the mount already: `ifl/out -> ifl/release_test`,
    `ifl/data -> release_v1.0/data`, `ifl/PSF -> release_v1.0/PSF`. Renames
    inside projects are server-side and instant.
  * NOT done: moving the 200 GB out of `/arc/home`. sftp refuses cross-tree
    renames ("Operation not permitted"), so it cannot go through the laptop
    mount without streaming every byte down and back up. The script is written
    and staged at `/arc/projects/minerva/ifl/jobs/move_to_release.sh`; it needs
    a CANFAR container to run it, and `skaha` is not installed on this laptop.
  * `poetry run pytest`: 387 passed.
- [x] Spatially varying Wiener deconvolution toward theoretical PSFs, plus a
  real MINERVA UDS F444W patch experiment (2026-08-13).
  * `PSFRegionMap.gaussian_psf_map` defines one noise-free, discrete-unit-sum
    Gaussian per source-PSF region. It preserves the region's fitted subpixel
    core phase (otherwise a fixed array-centre Gaussian encodes up to a
    half-pixel astrometric jump) and accepts larger padded support for an
    inverse kernel. `PSF.gaussian` now exposes the existing `x0`/`y0` support
    of `utils.gaussian`; `psf_core_centroid` and `psf_core_fwhm` measure the
    phase and the direct central-line width of a non-Gaussian/ringing response.
  * `PSFRegionMap.matching_kernel_map` passes each normalized source/target
    pair through the existing `utils.matching_kernel`, requires a strictly
    positive explicit regularization (the smoothing-kernel auto-FOM is not a sharpening
    selector), restores unit DC after regularization, and writes per-region
    white-noise gain, L1 cancellation, edge support, realized FWHM, target
    peak, negative response flux, normalized L2 residual, and centroid-shift
    metrics into the
    returned map, and rejects a target map on a different pixel scale. Unit
    DC is rechecked after float32 map storage so pathological cancellation
    fails loudly rather than changing the flux scale. With no `signal_psd`,
    the existing Wiener path remains the flat-prior/Tikhonov solution.
  * `PSFRegionMap.convolve_image` now promotes integer input to floating point
    and maps NaN/+inf/-inf explicitly to zero. Previously integer results were
    truncated and `np.nan_to_num` turned infinities into enormous finite
    values, especially destructive under a signed inverse kernel.
  * `examples/run_uds_f444w_deconvolution.py` reads only a 1024x1024 science
    patch plus a 258-pixel halo at (34.3413274, -5.2615397), clips the current
    1694-region production F444W map to the six regions needed by the halo
    (three cross the delivered patch), builds 512-pixel phase-matched
    targets/kernels, scans lambda=1e-6..1e-2, and writes selected FITS/map
    products plus comparison/tradeoff figures and a CSV under
    `scratch/uds_f444w_deconvolution/`. The adjacent input WHT is explicitly
    labelled as native-only; no false diagonal weight is made for the signed,
    correlated output.
  * The real result is a useful negative result for a literal 0.1" product.
    The center-region native central-line FWHM is 0.161" and its sampled
    2.5-pixel target is
    0.105". Lambda=1e-3 realizes 0.139" at 2.22x source-masked field scatter,
    1.87x empty-aperture RMS, and 0.277 integrated negative response;
    lambda=1e-4 realizes 0.126" at 4.36x / 2.33x and 0.285 negative response.
    The closest support-safe scan point is only 0.116" (lambda=1e-5) and costs
    9.69x field scatter / 3.85x aperture RMS; the nominal 0.109/0.112" points
    fail the absolute edge-L1 guardrail and are greyed out. The field metric
    uses the release segmap (54% of the patch retained) and is stable to
    12/16/24-pixel dilation; the aperture metric uses 1,908 fixed empty
    positions. Both include correlated background and residual ringing rather
    than representing propagated instrumental noise. The retained lambda
    values are unchanged between 256- and 512-pixel support, large-aperture
    star flux stays within 0.4%, and residual shifts are <=0.061 pixel. The
    output therefore has to be called "regularized toward 0.1\"", not a
    realized clean Gaussian.
  * The standard `PSF.optimize_matching_kernel_regularization` Wiener scan was
    also run directly on the three noise-free PSFRegionMap models under the
    delivered patch (parent keys 564/574/575) and their phase-matched 0.1"
    targets, plus their median. On the actual 512-pixel kernel grid every scan
    selects lambda=1e-2: the realized central-line FWHM is 0.156-0.157",
    white-noise gain is 0.808-0.810, target-peak recovery is 0.566-0.585, and
    integrated negative response is 0.201-0.205. This is essentially the
    native PSF, because the stock PSF-matching FOM includes
    `1e-3 * cancellation**2`; it is a useful stability baseline but not an
    automatic science-image deconvolution selector. The optimizer is also
    support/FOM dependent: the same region on an unpadded 100-pixel grid
    selects lambda=0.0562 and broadens to 0.179". Standard diagnostics and
    scan CSVs are under `scratch/uds_f444w_deconvolution/optimizer/`.
  * Analytic-target/support follow-up (2026-08-14):
    `PSFRegionMap.moffat_psf_map` now builds phase-matched, finite-stamp
    discrete-unit-sum circular Moffat targets, records beta and the raw
    discrete sum, and uses the same target-generic `matching_kernel_map` path.
    `PSF.moffat` exposes subpixel `x0`/`y0` and validates positive widths and
    beta > 1. The UDS driver accepts `--target-model gaussian|moffat` and
    `--target-beta`, with target model/beta in its FITS provenance.
  * The real three-region source-model scan shows that 512 pixels was inverse
    kernel padding, not a physical PSF extent. A 160-pixel kernel is a robust
    common support for the useful compromises (maximum outer-edge absolute
    L1 < 0.01 and fractional L1 < 1e-3). Gaussian 0.14" at lambda=1e-4
    realizes 0.153" with white-noise gain 1.01 and negative response 0.041;
    the real patch has 1.43x source-masked field scatter. Gaussian 0.08" is
    worse: at lambda=1e-3 it realizes 0.130" with gain 2.25 and negative
    response 0.466, and approaching the requested width is not credible.
  * The strongest winged sensitivity target is Moffat FWHM 0.10", beta=2.5,
    lambda=2e-4 on 160-pixel support. It realizes 0.139" (the same core width
    as the retained 0.10" Gaussian/lambda=1e-3 result), white-noise gain 1.50,
    2.03x source-masked field scatter, 1.52x empty-aperture RMS, and 0.074
    integrated negative response. The comparable Gaussian has gain 1.56,
    2.22x / 1.87x scatter and 0.277 negative response. The Moffat therefore
    rings less in the effective PSF but has a larger kernel L1 (22.1 versus
    18.5) and is more sensitive to PSF mismatch; it is not a physical JWST
    target or a clean 0.1" reconstruction. The requested stock Wiener
    optimizer was run on this Moffat pair as well: all three regions select
    lambda=0.0316 and broaden to about 0.182", so it remains a conservative
    PSF-only stability reference rather than the selector for this experiment.
    If compact storage is the priority, the 128-pixel Moffat/lambda=3e-4 run
    closes numerically against 160 pixels and realizes 0.142" at 1.80x / 1.47x
    field/aperture scatter and 0.070 negative response. Its maximum edge L1 is
    0.0091, just inside the 0.01 guardrail, so 160 remains the margin-safe run.
  * Ringing-mitigation follow-up used the same three noise-free region-map
    PSFs and the real patch at matched realized width.  A post-Wiener spatial
    taper `exp(-(r/6 pix)**2)` on the 0.10" Gaussian/lambda=1e-3 kernels is
    the best compact fixed-kernel compromise tested: it realizes 0.145",
    lowers integrated negative response from 0.277 to 0.116, kernel L1 from
    18.52 to 6.07, source-masked field scatter from 2.22x to 1.41x, and empty-
    aperture scatter from 1.87x to 1.36x.  The broad ring train disappears,
    but a compact -4.6% trough remains, so it is an apodized effective target,
    not a clean Gaussian.  A generalized-Gaussian Fourier taper can reduce
    the trough to -0.7% and negative response to 0.055 at the same 0.145"
    width, but raises white-noise gain to 1.65, kernel L1 to 27.7, and needs
    256-pixel support; on the patch its field/aperture scatter is 2.18x/1.66x
    and a low-contrast concentric train remains visible.  SplitCosineBell and
    ForWaRD did not improve the matched-width Pareto front.  Signal PSDs
    estimated from 1,807 empty
    patches, Matérn priors, and target-OTF priors likewise did not beat flat
    Wiener at matched resolution; their apparent gain at fixed lambda was
    resolution loss.  A 0.12"/0.48" two-Gaussian core+halo target nearly
    removes the model-response negative bowl (0.015 at 0.143"), but moves 40%
    of the flux into the halo and still produces visible real-star ripples.
    Exact ring-free matching to a narrower Gaussian is impossible for one
    linear kernel because the Gaussian OTF remains nonzero where F444W's OTF
    vanishes; smooth tapering can only trade the missing modes for a broader,
    non-Gaussian response.
  * An actual UDS F356W-to-F444W PSF test is forward smoothing, not the same
    inverse problem. On native 100-pixel stamps the stock Wiener optimizer
    selects lambda=1e-3 and realizes 0.167x0.162" against the
    0.166x0.160" target, with white-noise gain 0.31 and maximum normalized
    Fourier gain 1.0. Its signed lobes correct diffraction/drizzle structure.
    At 512-pixel padding the stock objective instead oversmooths by selecting
    lambda=0.0178, another reminder to fix the scoring aperture/support or use
    the native-grid result. The reverse F444W-to-F356W direction is genuine
    deconvolution. A production F356W region map was not built because the
    checkout has only single-epoch F356W grids while the mosaic spans 23 date
    groups.
  * Focused regressions cover phase matching, padding, unit DC,
    sharpening and diagnostics, numeric provenance round-trip, malformed-map
    validation, and convolution dtype/non-finite handling. Focused PSF/map
    validation: 42 passed. Full repository validation: 389 passed (388 in the
    workspace sandbox plus the legacy astrometry diagnostic-write test rerun
    with permission for its `../tmp` output).
- [x] Full-field scene-blob diagnostic (2026-08-13).
  `verification.save_scene_blobs` draws each scene as the convex hull of its
  template positions, filled in its own colour and labelled with its scene id
  in grey, and `write_outputs` writes it as `<name>_scene_blobs.png` next to
  the existing `_scene_map.png`. The two answer different questions: the map
  colours segments (which source went where) and reads the mosaic to do it;
  the blobs show where the scenes are and how big they got, as pure vector --
  no raster, no decimation, cost set by the scene count rather than the field.
  * Scenes narrower than `label_min_pix` (60 px) are drawn as circles rather
    than hulls and left unlabelled: a hull over three sources is a sliver that
    reads as noise at field scale, and on a real partition most scenes are
    small, so numbering all of them buries the figure. Rendering 195 synthetic
    scenes labelled 24.
  * Drawn largest first, so a compact scene inside a sprawling one is not
    buried by it.
  * `poetry run pytest`: 381 passed.
- [x] PSF grids carry provenance and are rebuilt when it disagrees
  (2026-08-13). A grid filename records detector, filter, MJD, grid size and
  oversampling -- but not the exposure list it came from, nor the `date_mode`
  that chose its MJDs. Those are exactly what changed under the `modal` ->
  `all` default, so a one-epoch-per-band set is indistinguishable on disk from
  a per-epoch one, and `_load_epsf` autobuilds only when *nothing* matches the
  pattern: the wrong set loads and is never corrected.
  * `psf_factory.grid_provenance()` returns the exposure-list content hash
    (content, not path or mtime), the date mode and the FOV;
    `PSFFactory.from_csv` stamps it and `write_stdpsf(provenance=...)` writes
    it as `HIERARCH MPH` cards. `jwst_psf.read_stdpsf_provenance` reads it.
  * `Pipeline._stale_psf_grids` compares the cards of every file matching the
    band's pattern against what the run wants. `RunConfig.psf_provenance`
    decides what follows: `"warn"` (default) reuses them and says so loudly,
    `"rebuild"` deletes them and builds the missing MJDs, `"error"` refuses to
    run, `"off"` skips the check. Grids written before provenance existed
    carry no cards and count as stale: they cannot be shown to agree.
  * `"rebuild"` deletes rather than overwrites, and does not force
    `PSFFactory.overwrite`. Overwriting in place does not converge: a
    `cluster` grid is named for a cluster midpoint and an `all` grid for a
    rounded MJD, so the new set never regenerates the old filenames -- they
    would survive, keep matching the pattern, keep being loaded alongside the
    fresh ones, and be re-flagged on every later run. With the delete, an
    existing grid whose encoded MJD is already in the target set is skipped
    (`psf_factory.py:407`), so only the missing dates are built.
  * `"warn"` for now because the release holds ~416 `cluster`-mode grids that
    would otherwise all rebuild on first contact. Flipping the default to
    `"rebuild"` once that cost is affordable is filed in `TODO.md`; the generated
    MINERVA configs write `psf_provenance` explicitly so the switch is
    visible in each of them.
  * Not covered: an *incomplete* grid set. One grid and seventeen still both
    match, provided both were built the same way; comparing loaded dates
    against `dates_from_csv` would close that.
  * `poetry run pytest`: 375 passed.
- [x] `scene_max_merge_radius` now bounds a scene's shape, not only its
  merging (2026-08-13). `scene_max_size` caps a scene's template count and
  leaves its geometry free, and connected components near the percolation
  threshold are dendritic. The measured case is the 25-template scene whose 14
  anchors spanned 18.8 Mpx -- 25 sources strung across ~4300 px. That costs
  twice: the astrometric shift field is fitted over an extent an order of
  magnitude larger than the correlation length the model assumes
  (`astrom_kwargs['gp']['length_scale']`, 400 px), and the derivative columns
  in `assemble_scene_system_AB` are dense planes over the anchors' bounding
  box, so a scene's memory follows its bbox area rather than its member count
  -- the 0.61 GB buffer chased in the full-field memory work.
  One knob, read three ways, all the same limit from different sides of the
  partition: a scene wider than `scene_max_merge_radius` is median-bisected
  along its longer axis (`scene._split_oversized_spatial`), an underfilled
  scene looks no further than it for a merge partner (unchanged), and a merge
  that would leave the scene wider than it is refused. The default is
  1500 px; at MINERVA's 0.08"/px that is 120" = 2', and ~4x the GP length
  scale (400 px). `np.inf` restores the old behaviour on all three.
  Bisecting the *longer* axis is what removes elongation specifically: halving
  the long axis of a thin scene squares it up, and each split strictly reduces
  membership so the recursion terminates. The merge veto tests the union
  bounding box rather than the centroid separation, because a distance bound
  alone does not give an extent bound -- merging a scene already 3000 px wide
  with a neighbour 900 px away leaves a scene wider still. It has the same
  precedence over `minimum_bright` that `scene_max_size` has. Positions are why
  the split lives in `generate_scenes` rather than
  `build_scene_tree_from_normal`, which sees only `ATA`/`ATb`.
  Provenance: this is the approach a 2026-08-08 session wrote and then parked
  in a stash. Its docstring recorded that local threshold-raising had been
  tried first and rejected as producing ragged shapes -- and threshold-raising
  is what shipped on 2026-08-13 for the size cap, without that note in view.
  The 18.8 Mpx scene is that predicted failure, measured.
  What the default does to that case: the 25-template scene spanning 4300 px
  becomes 4 scenes, longest side 1412 px, and the bbox area driving the
  `Bq`/`Bl` buffers falls 18.8 -> 0.36 Mpx, a factor 52. At 1000 px it
  would be 5 scenes, 883 px, 0.21 Mpx. A crowded
  but compact scene is untouched (600 templates inside 1200 px stays one
  scene). End to end on a 120-source field the limit binds and holds at every
  value tried: 13 scenes / longest side 207 px at `inf`, 14 / 121 px at 150,
  22 / 77 px at 80.
  **This changes fluxes on real fields**, and the standard fixture cannot show
  it -- a 301 px test field never reaches 1500 px, so the fixture reproduces
  `7784f99` bit for bit and proves only that the plumbing is inert when the
  limit does not bind. A full-field before/after is still owed: scene count,
  widest bbox, peak memory, whether the v9 non-converging scenes converge, and
  the flux comparison `scene_max_size` got. The split cuts couplings the
  threshold pass chose to keep, so this is not a memory-only change. Every run
  that already sets `scene_max_merge_radius` (all of them, via the default)
  now gets the split as well. See TODO.md.
  Thirteen tests in `tests/test_scene_max_size.py`.
  They cover the longer-side bound, the elongation fix (45:1 in, within 4:1
  out), compact scenes left alone, termination on coincident positions, the
  merge veto, and `np.inf` as a no-op.
- [x] Two-phase campaigns: a per-field prep step, both toolkits (2026-08-13).
  A field's bands share its F444W and 30" halo ePSF grids and its saturation
  repair -- `_repair_provenance` keys on the detection band and the trial box,
  nothing that varies between bands -- so submitting them together made every
  band rebuild the same grids, re-run the same repair, and write the same cache
  file at once. Campaigns now submit prep per field first, then fan out.
  * `Pipeline.prep()` = `build_psfs()` + `build_repair_cache()`;
    `build_repair_cache()` = `load_data(kernels=False)`, which runs the repair
    and writes the cache without building a per-band kernel map. Registered as
    `STEPS["prep"]` and `STEPS["repair"]`; `repair` is a no-op when
    `repair_saturated` is off.
  * The two exist separately because of OzStar: grid building needs MAST and
    the module stack, which only the login node has, so there the halves run on
    different machines (`jobs/build_psfs.sh`, then `repair` under SLURM).
    CANFAR compute has internet, so it uses the combined `prep`.
  * `jobs/run.sh` and `jobs/run.slurm` take `$STEP` (default `all`);
    `submit.py run --step` on both sides, with the step in the session/job name
    so `status`/`squeue` distinguishes prep from the fits. CANFAR waits on
    prep laptop-side; OzStar expresses it as `--dependency=afterok:<repair>`.
  * `docs/campaigns.md` (new) explains both, including where the platform
    differences force the design.
  * `examples/canfar/` and `examples/ozstar/` are now tracked in full --
    scripts, docs, generated configs and staging lists, 202 files and under a
    megabyte of text. They are staging only: `submit.py fetch` and the upload
    scratch write to `scratch/<toolkit>/` instead, and `out/`/`_upload/` stay
    ignored inside them so an output cannot creep back in.
  * `poetry run pytest`: 367 passed.
- [x] Saturation-repair cache: per-field naming and a readable-file guard
  (2026-08-13). Found while checking the generated CANFAR configs against the
  current `RunConfig`. They validate (all 53 load, no unknown keys, all carry
  `psf_date_mode`), but every one left `repair_cache_path` unset, so the
  default `'..'` resolved to a single `<run>/out/repair_cache.fits` for the
  whole tree. `_repair_provenance` keys on the detection image and the trial
  box, so those 53 fell into 16 mutually-invalidating groups -- three fields
  and a dozen patch geometries -- each recomputing the repair and overwriting
  the others' cache.
  * `arcify.py` now writes `repair_cache_path` next to `out_dir`, named by
    field and geometry (`uds_full_repair_cache.fits`,
    `uds_r3_34.38792-5.30102_repair_cache.fits`). Bands of a field with the
    same geometry share one cache, which is what the default was reaching for;
    nothing else collides. Fixed in the generator, not the configs: they are
    gitignored and regenerated by `campaign.py`'s `arcify` step.
  * `Pipeline._load_repair_cache` treats an unreadable cache as absent
    (`OSError`/`KeyError`/`ValueError` -> re-run the repair). A campaign
    submits a field's bands together, so one job can read the file while
    another is writing it; the cache is recomputable and a lost fit is not.
  * Follow-up in `TODO.md`: a per-field `prep` step so the repair runs once
    before the bands are fired off, rather than concurrently in each.
  * `poetry run pytest`: 365 passed.
- [x] Array-lifetime audit and five memory fixes, `docs/MEMORY_LIFETIMES.md`
  (2026-08-13). Companion to `docs/SCALING_FIXED_MEMORY.md`, which proposes the
  decomposition; this one inventories every full-field array a run allocates,
  records where each is born and last read, and separates what is still needed
  from what is merely still referenced. Changes made, in the order they run:
  * The band weight map is released once nothing reads it.
    `Pipeline._release_scene_weights` clears `Scene.weights` across a band;
    `run` calls it after `predicted_errors` when the run draws no scene
    figures, `write_outputs` after the figures. `weights_i` was 3.5 GB of dead
    weights held by every `Scene` through the stamp write -- the stage where
    the unexplained full-field failures occur. `Scene.residual`/`Scene.plot`
    mask on the weights only when they are still attached; `Scene.solve` still
    refuses to run without them.
  * `write_outputs` writes the stamps last, after the scene figures rather than
    before. The products are independent, and a run that dies in the stamp
    write now keeps its figures.
  * The residual accumulates straight into its own output file
    (`_residual_memmap`): header written, file extended sparsely with
    `truncate`, data section mapped big-endian. `write_outputs` flushes instead
    of writing. Falls back to anonymous memory for API-driven runs and on any
    mapping error.
  * The repair replays its patch table onto a fresh copy-on-write map instead
    of holding the two full-field mosaics `repair_saturated_holes` returns
    (`saturate.py:733`). Fresh and cache-reuse paths now share
    `_apply_repair_patches`; astropy maps a read-only HDU copy-on-write, so the
    input mosaic on disk is untouched.
  * `write_stamps` streams: offsets from the shapes recorded in the first pass,
    datasets created at final size, each stamp written into its slot. Removes a
    full extra copy of every stamp (12 GB full-field) at the end of the run.
  * Also: the `isfinite` sweep at the top of `run` is gone (inverted, dead
    image branch; weight branch re-checked a guard `load_data` had applied).
  * `poetry run pytest`: 362 passed. New tests cover the residual memmap
    round-trip end to end, the API fallback, copy-on-write patch replay, and
    stamp pixel equality after a round trip.
  * Still open, in `TODO.md`: the byte-order copies in `get_bg_and_ivar` (FITS
    is big-endian, so `np.asarray(x, dtype=np.float32)` copies rather than
    views -- ~12 GB transient on the detection band); ivar as
    `(memmapped wht, scale, mask)`; `weights[1]` after the upsample.
- [x] `PSFFactory.date_mode` now defaults to `"all"`, and configs can set it
  (2026-08-13). The default was `"modal"` -- the centre of the densest 5-day
  window, i.e. exactly *one* date per (detector, filter). Since the grids are
  MJD-tagged and looked up by nearest date, autobuilding a band whose
  exposures span years produced a single epoch's wavefront for all of them,
  silently. Found on OzStar, where nothing had been pre-built: MINERVA
  exposure lists span up to 1460 days over 4-18 epochs, and every autobuilt
  band got one grid while the laptop-built UDS F770W/F1800W had 9 (those were
  made in `cluster` mode). A release built that way would have mixed two PSF
  conventions across its bands.
  * `psf_factory.PSFFactory.date_mode` default `"modal"` -> `"all"`, with the
    docstring saying why the collapsing modes are the wrong default.
  * New `RunConfig.psf_date_mode` (default `"all"`), threaded into the
    `PSFFactory` call in `Pipeline._load_epsf`. The mode was previously
    unreachable from a config at all.
  * `examples/make_minerva_configs.py` writes `psf_date_mode` explicitly, so
    every generated MINERVA config states it rather than inheriting it; the
    53 CANFAR and 35 OzStar configs derived from them carry it too.
  * Grid counts this implies, from the real exposure lists: 416 across the
    release in `all` mode against 333 in `cluster` (UDS F770W 17 vs 9,
    COSMOS F444W 78 vs 44, EGS F444W 42 vs 2).
  * `poetry run pytest`: 358 passed.
- [ ] MINERVA v1.0b full-field campaign on OzStar (2026-08-13/14, running).
  Measured, which is what the OzStar README was missing:
  * UDS full field F770W: **57.4 GB MaxRSS**, 56m19s, 26 GB of outputs,
    138,634 sources (8,827 at SNR>5). A 64 GB request would have peaked at
    90%; 96 GB is the right ask for UDS/COSMOS and EGS gets 128 GB, since its
    detection grid is ~1.4x UDS's.
  * 1.5' trial patch: 20 GB, 9m51s, 4,607 sources. Full field is ~3x the patch
    in memory.
  * CPU is **6.1% of 16 cores, i.e. about one core**. The fit is serial; cores
    buy threaded BLAS in a few phases and queue time everywhere else. Memory
    is the only resource that kills a run. (CANFAR measured 0.2 of a core for
    the same work, dominated by waiting on NFS `/arc`; Lustre removes that.)
  * Staging: 66 files, 64 GB, 15-21 min per field on three datamover nodes.
  * OzStar took a ~30 min unannounced maintenance outage mid-campaign. Queued
    jobs survived it; `uds_f770w_v1.0b` had already completed.
  Done autonomously while the user was away, and worth reviewing: the
  COSMOS/EGS grids are being rebuilt with `date_mode="all"` passed explicitly
  by `jobs/build_psfs.py`, which calls `PSFFactory` directly rather than going
  through the pipeline's autobuild. That sidesteps needing the `psf_date_mode`
  change deployed (the run tree pulls from GitHub `main`, and the change is
  local and uncommitted) and sidesteps the autobuild's fire-only-when-nothing-
  matches rule. COSMOS F444W resolves to 39 dates in `all` mode against 22 in
  `cluster` and 1 in `modal`. UDS grids are deliberately untouched: its bands
  are in flight and adding dates mid-release would leave `f770w` fitted
  against 9 grids and later bands against 17.
- [ ] OzStar campaign toolkit, `examples/ozstar/` (2026-08-13). The CANFAR
  toolkit's counterpart for Swinburne's Ngarrgu Tindebeek. Two differences
  drive the design: OzStar has no view of the MINERVA release, so every input a
  config names is copied from CANFAR arc onto `/fred` before anything runs; and
  compute is ssh + SLURM rather than a REST API, so a campaign is one
  dependency graph submitted in a single command instead of a laptop-side
  process blocking on each stage.
  * `ozroot.py` (run tree and ssh target from `$OZSTAR_*`), `ozify.py` (local
    RunConfig -> `<name>_ozstar.json` + `<name>_stage.tsv`), `submit.py`
    (cert/setup/sync/push/stage/run/status/logs/fetch/cancel/seed),
    `campaign.py`, `release_v1.0b.sh`, and `jobs/` (setup_env.sh, stage.sh,
    run.slurm, sync_src.sh, seed_cache.sh).
  * `ozify.py` imports `roots_for`/`arc_index`/`resolve` from
    `examples/canfar/arcify.py` rather than copying them - finding a file on
    arc is the same problem on both platforms. That needed one change there:
    the module-level `RUN` is now a lazy `arc_run()`, so importing the helpers
    no longer requires a CANFAR run root. Behaviour is unchanged.
  * Staging is one datamover job per *field* (bands share the F444W mosaic,
    weight and segmap), fits wait on it with `--dependency=afterok`, and a
    field with no PSF grids sends one band ahead of the rest so concurrent
    bands do not race on one `psf_dir`.
  * Four platform traps, all found by running into them and all documented in
    `README.md`/`MANUAL.md`: Lmod is hierarchical, so `python/3.12.3` needs
    `gcccore/13.3.0` loaded first; the python module puts an EasyBuild shim
    ahead of the venv on `sys.path`, which on some nodes resolved
    `cryptography` to a build for another python and killed every `vcp` with a
    missing `libssl.so.1.1` (job scripts now `unset PYTHONPATH`); a site plugin
    reassigns the partition of a job that names `datamover` in a `#SBATCH`
    directive alone, so it has to be on the sbatch command line or the transfer
    lands on a node with no internet; and datamover nodes have no `/apps`, so
    the module python is absent there and the CADC tools need their own
    `venv-vos` built from `/usr/bin/python3`.
  * Standard request 16 cores / 64 GB per fit, no partition (the scheduler
    picks between skylake and milan).
  * State: all three fields staging from arc (17 configs, 8 inputs each);
    `uds_f770w_trial` (1.5' patch) queued behind UDS staging as the smoke test;
    the v1.0b full-field release goes in after it passes.
- [x] Controlled MINERVA SED co-add experiment (2026-08-13). Rebuilt the
  underlying stack at `Delta z=0.035(1+z)` with 2x rendering interpolation and
  restricted the continuum-residual panel to observed 0.35--5 micron, where
  connected filter coverage makes that residual meaningful. Relative to 0.05,
  the experiment adds 42% more measured redshift rows below z=8 while retaining
  a median 5,292 galaxies/bin; 0.025 nearly doubles the rows but raises residual
  noise 16% and adds horizontal striping. OIII coherence improves modestly,
  H-alpha does not benefit from 0.025, and no coherent Pa-beta/Pa-alpha ridge is
  detected. The original 0.05 equal-mean product remains primary.
  Added `scratch/minerva_sed_estimator_experiment.py`: 64 exact complementary
  split halves over 4,204 field/native-band/redshift cells independently
  recompute every estimator. Raw IVW is 3.66x less repeatable than the equal
  mean with median Neff/N=0.0215; Q95 IVW is 1.62x worse with 0.185. The
  population-scatter weight improves repeatability 16% overall but moves MIRI
  by a median 0.31 MAD. Winsorizing 0.5% per tail is the conservative companion
  (1.7% precision gain, 0.008-MAD median shift); 1% is the empirical knee
  (3.5%, 0.013 MAD). Matched-three-band, `use_phot_miri`, high-quality-z, and
  spec-only EGS splits yield no stable feature-S/N gain; triple matching keeps
  only 3,229/126,655 normalized galaxies. Corrected H-alpha and OIII guides to
  their vacuum wavelengths. Focused SED tests: 30 passed.
- [x] MINERVA all-field SED-stack MIRI visibility and quality-control pass
  (2026-08-13). Regenerated every ignored FITS/PNG/PDF product for 357,044
  normalized COSMOS+EGS+UDS galaxies. Replaced the 1.6-micron continuum-bump
  guide with vacuum Pa-beta/Pa-alpha, and added a distinct magenta rest-5000-A
  normalization guide. The new MIRI PNG uses a shared tight signed-asinh
  stretch plus independently scaled signed profiles, empirical standard
  errors, N, and effective N; the values are not renormalized per filter.
  Sampling explains much of the visibility difference: F560W has 5,299
  normalized measurements and is EGS-only, F770W has 169,862 across all three
  fields, and F1000W has 94,566 across COSMOS+EGS.
  Added machine-readable MIRI ECSV/FITS QC with equal means, medians, empirical
  SEM, 0.5--99.5% field-local winsorized means, population-scatter-regularized
  weights, effective counts, and raw/Q95 IVW failure columns. An actual
  4,204-cell all-band audit rejects ordinary IVW: median Neff/N=0.0215 and
  empirical error 3.2x the equal mean; Q95 capping still gives 0.185 and 1.47x.
  The equal-galaxy signed mean therefore remains primary and measured S/N is
  never a weight. Common-galaxy checks find no local 2.5--3-micron F277W
  excess: F277W is 2--4% below the F250M--F300M interpolation in every field;
  the visual block is the declining F-lambda continuum plus the real
  2.226--2.412-micron gap, with the 5000-A anchor crossing it at z=4--5.
  `tests/test_sed_stack.py`: 27 passed.
- [ ] Scene solve cost and the size cap (2026-08-13). Two defects found while
  chasing the full-field memory peak; both are about the cost of a scene
  growing with the number of templates in it.
  * `SceneFitter._flux_errors` chose its branch on size alone. Above
    `dense_threshold=500` it built `sp.csc_matrix(A)` and factored with
    `splu`, then back-solved one unit column at a time. That is right for the
    flux-only path, where the whitened normal matrix really is sparse, and
    backwards for the joint flux+shift path: it is handed
    `S_w = A_w - AB_w AB_w^T`, and the outer product is fully populated, so
    `S_w` arrives dense. A 1718-template scene therefore paid 118 MB and 5.1
    Gflop where one LAPACK inversion costs 47 MB and 1.7 Gflop. The dispatch
    is now on whether the matrix *is* sparse. Dense input matches the exact
    inverse bit-for-bit (it previously differed at ~1e-15, the splu path's
    error); a genuinely sparse matrix still takes the sparse branch and agrees
    to 3e-15.
  * `scene_max_size` never bound. `build_scene_tree_from_normal` splits
    oversized components, and then `merge_small_scenes` -- which was never
    given the cap -- merged them straight back chasing `minimum_bright`. A run
    configured with 800 produced a 1718-template scene. The cap is now
    threaded through and tested against the scene as it grows within a merge
    round; it wins over `minimum_bright`, so a scene that cannot merge without
    breaching it is left short of anchors and logged rather than grown without
    bound. Four tests in `tests/test_scene_max_size.py`.
  * Defaults: `scene_max_size` 800 -> 1000. The shift-basis order was already
    0 in `FitConfig.astrom_kwargs`, but two fallbacks disagreed with it --
    `__post_init__` assumed 1 when deriving `scene_minimum_anchors` and
    `AstroCorrect` assumed 2 for the polynomial field (its own docstring said
    "an unmodified FitConfig supplies order 0"). Both now read 0, so a config
    that omits the `poly` key derives `scene_minimum_anchors` 3 rather than 7.
  Order 0 means nB = 2: one rigid (dx, dy) per scene. That matters for memory
  because `assemble_scene_system_AB` holds nB float64 planes over the bright
  anchors' bounding box, doubled when the leverage cap clips -- 0.61 GB for
  the widest full-field scene at nB=2, and linear in nB above that (order 1
  would be 1.8 GB, order 2 3.6 GB). Note the size cap does *not* bound this:
  the widest buffer came from a 25-template scene whose 14 anchors spanned
  18.8 Mpx. Chunk those columns over row bands before raising the order.
  Measured on the UDS F770W 3' trial (17,791 templates, config cap 800), against
  the same trial before these two fixes:
  * Partition: 70 scenes (sizes 2-1019, median 199) -> 74 scenes (sizes 2-779,
    median 199). The cap binds -- the largest scene was over it by 27% and is
    now under -- at the cost of four extra scenes, and *no* scene lost its
    anchors: both runs report 0 scenes without bright members, so preferring
    the cap over `minimum_bright` cost nothing here.
  * Fluxes: 1,592 of 17,796 sources changed at all (9%). Median ratio
    after/before is 1.000000 with 16-84% both 1.000000. Restricted to the 713
    sources at SNR > 10 the worst moves 8.2%; the large excursions (up to 40x)
    are all faint sources that changed scene. Errors move more often (2,682
    rows) but by less: median 2.5e-3, worst 20%. `stampcor`/`totcor`/`psfcor`
    move at most 1.9e-3. The residual differs in 0.42% of pixels, max |delta|
    1.913, rms of the changed pixels 0.0105.
  * Memory from that pair is NOT comparable: the second run shared the machine
    with ~30 subagent processes, and macOS phys_footprint counts compressed
    pages, so it read 30.3 GB against 20.0 GB for reasons that have nothing to
    do with the code. Re-measure on an idle machine.
  Still to do: a full-field run on an idle machine for the peak against the
  recorded 46.5 GB, and the same partition/flux comparison at full-field scale,
  where scenes are larger and the cap bites harder.
- [ ] CANFAR v1.0 campaign, **full field** (2026-08-13). v1.0 is now a
  whole-field run per field-band, not the 1.5' patch the entry below
  describes: the memory work merged in `9fc52a6` brought UDS full field to
  46.5 GB, under the 48 GB request. 17 configs (uds/cosmos/egs x all staged
  MIRI bands) re-arcified with `--r-trial 0 --suffix _v1.0`, so `trial` is
  null and outputs still go to
  `/arc/home/ilabbe/run/out/<field>_<band>_v1.0`. Jobs request 4 cores and
  48 GB. EGS is 1221 Mpx against UDS's 876 and extrapolates over 48 GB, so
  its bands are the ones expected to OOM; the agreed response is to resubmit
  those at `--ram 64` rather than raise the request everywhere, since 48 GB
  and up have queued for hours when the platform is busy.
  The earlier patch submission was stopped first. That took two rounds: six
  sessions were listed and destroyed, and ten more appeared minutes later,
  because skaha does not list a session until the service registers it. They
  carried the same run names as the new campaign and would have written into
  the same `out/` directories. `submit.py kill` now sweeps for this.
  **Result so far: the fits succeed, the stamps write does not.** Every band
  that has failed wrote a complete fit table first -- egs_f1000w 142,323 rows
  (142,299 finite `flux_1`), uds_f1800w 137,609, uds_f1280w 140,972, all 22
  columns -- plus its residual, kernel, PSFs and templates, then died in the
  stamps write with no `RUN_DONE` and no traceback. Not a memory ceiling:
  egs_f1000w reached `Pipeline (end) memory: 49.4 GB` against a 48 GB request,
  but uds_f1800w reached only 41.6 GB and died the same way, and the stamps
  files it left are truncated at 31-32 MB where a full field should be
  multi-GB. This is the area `3c29ddd`, `8ca21f5` and `b7bec1e` address on
  main; the campaign predates them and stays pinned to `9fc52a6a6` through
  `submit.py run --ref`, so v1.0 is one code version rather than two. Open
  decision: accept v1.0 as the fit tables from that commit and rebuild stamps
  separately, or rerun the campaign on current main, which also brings the
  scene-by-scene astrometric loop (`1ab936e`) and the float32 narrowing into
  the release. EGS's seven bands are at `--ram 64` since it is the hottest
  field, though that alone will not produce `RUN_DONE`.
  Four toolkit defects fixed to get this far, all committed:
  * `campaign.py` globbed every JSON in `examples/minerva`, including
    `minerva_sed_fields.json`, which is not a RunConfig and died inside
    arcify. Restricted to `<field>_f<band>w.json`.
  * skaha session names accept only alphanumerics and `-`; the `_v1.0`
    suffix put a dot in them and every job 400'd. `submit.py::session_name`
    now normalises once instead of replacing underscores per call site.
  * `arcify --r-trial 0` wrote `radius: 0`; it now writes `trial: null`.
  * the 48 GB default is a patch-run number (`MANUAL.md`'s own table is
    patch measurements) and does not cover a full field.
  Sizing, measured rather than assumed. `Pipeline` logs its own peak: a
  1.5' patch peaks at 27.3 GB (v8 UDS F770W). Full field needs ~72 GB on
  COSMOS (151,778 templates; OOM'd at 48 GB right at the upsample) and
  ~110-130 GB extrapolated on EGS, which is 1.8x COSMOS (1221 Mpx,
  520,875 catalogue sources). The platform reports
  `memoryGB.defaultLimit = 32`; 8 GB jobs schedule instantly, 48 GB
  scheduled once while the platform was quiet, and 48/64/128/192 GB have
  all queued for hours since. Those allocation figures are **pre-`9fc52a6`**
  and no longer describe the pipeline: cutting the redundant template sets
  and banding the whole-array passes brought UDS full field to 46.5 GB, so
  the conclusion recorded here -- that full field was not runnable on this
  platform, and that the only ways forward were
  `multi_resolution_method: downsample` or tiling a field across jobs --
  is superseded. Full field per field-band is what v1.0 now runs; neither
  fallback was needed. EGS remains the field where 48 GB may not be enough.
  Note the stamp-footprint fix is what makes even this feasible: at the old
  402^2 support, COSMOS's convolved templates alone would need 98 GB.
- [x] Astrometric loop inverted to scene -> pass (2026-08-13). `run()` ran
  `for pass in 1..niter: for scene in pending:`, synchronising every scene at
  each pass and carrying a `pending` list so a converged scene dropped out. It
  now refines one scene to convergence -- its passes, its convergence test, and
  the flux-only pass that closes it out -- before starting the next. Scenes are
  independent *across* passes and not only within one: `solve()` reads the
  scene's own templates and read-only slices of the shared image and weights,
  and writes only to itself and to those templates, so nothing couples one
  scene's iterate to another's. What goes is the barrier, which is what stops a
  scene from being handed to a worker process (`docs/SCALING_FIXED_MEMORY.md`).
  Results are unchanged, checked rather than argued: the same fixture run
  through HEAD and through the working tree gives bit-identical fluxes, errors,
  per-scene shift coefficients, per-template accumulated shifts, pass counts,
  convergence flags and residual, in three configurations -- 3 scenes
  converging uniformly, 10 scenes at a tolerance that makes the pass count vary
  between scenes (median 4, max 5, which is where the two orderings genuinely
  interleave differently), and the flux-only path (`fit_astrometry_niter=0`,
  one solve per scene, no closing re-solve).
  Behaviour preserved in the corners: a scene whose shift block was never built
  (flux-only run, or fewer than two bright anchors) keeps `astrom_converged`
  None and stops after one pass, exactly as dropping out of `pending` did.
  Logging changes shape, since the per-pass lines no longer mean anything: the
  loop now logs its budget up front, each scene at DEBUG, and one summary line
  (scenes converged, median and maximum passes run). The warning naming the
  worst non-converged scenes is unchanged.
  `tests/test_pipeline.py::test_scene_results_do_not_depend_on_scene_order`
  pins the invariant by handing the loop its scenes back to front and requiring
  identical fluxes, errors, shifts, pass counts and residual.
- [x] HIERARCH card warnings silenced outside `log_run` (2026-08-13). Long
  keywords carried on the input catalog's `Table.meta` -- `PHOT_UNIT`,
  `WEBBSTARFILT`, `HSTSTARFILT`, `APER_DIAM`, `SHRINK_FACTOR` -- round-trip
  into the fit table's header as HIERARCH cards by design, and astropy warns
  once per card, twice per card once its own warning logging has a handler.
  `log_run` already filtered them, but at the time it only wrapped `run_all`,
  and the steps are normally run one at a time (`python -m mophongo.pipeline
  config.json psfs kernels load fit outputs`, which is what the campaign and
  every validation run use), so that path never entered the block. `01dd473`
  has since put every CLI invocation inside `log_run`, which closes the same
  hole from the other end; a notebook or script driving `Pipeline` directly
  still does not go through it, and the filter belongs with the writes that
  provoke the warning in any case. It is now a module
  helper applied at the write path itself -- `write_outputs`, around the
  residual, fit table and template table -- with `log_run` calling the same
  helper instead of repeating the filter inline. `write_stamps` needs no filter
  since `8ca21f5` moved it to HDF5.
  It is scoped to the write: a caller's own filters are
  untouched afterwards. Nothing about what is written changes; the keywords
  still land as HIERARCH cards.
  `tests/test_pipeline.py::test_write_outputs_silences_hierarch_card_warnings`
  checks all three: no `VerifyWarning` escapes `write_outputs`, the keywords
  are readable back off the fit table, and a bare header assignment still
  warns afterwards.
- [x] CANFAR toolkit: lessons from starting the full-field campaign
  (2026-08-13). Nine changes in `examples/canfar/`, each from something that
  cost time in this run:
  * **A dropped submission is no longer silent.** Submitting seven EGS bands,
    the service returned HTTP 500 with
    `JedisDataException: ERR max number of clients reached` for three of
    them; skaha swallows that and returns an empty list, so `launch` logged
    `FAILED` and the batch carried on with three bands that never existed.
    `launch` now retries three times with a 10 s pause, and `do_run` exits
    non-zero naming any band that still did not start, rather than returning
    quietly under `--no-wait`.
  * **`wait` no longer calls a job dead on an empty `info`.** A two-minute
    DNS outage on the laptop (09:05-09:07) made three consecutive `info`
    calls return `[]` -- skaha swallows network errors and answers with an
    empty list, so an unreachable service is indistinguishable from a reaped
    session -- and `wait` declared the *running* EGS leader `Gone`. The
    campaign then moved on and tried to launch the six bands that were
    waiting on the grids that leader was still building, which is the
    `psf_dir` race the leader-first ordering exists to prevent. Only the
    `SRC_VERSION` guard stopped it, and only by accident: the same outage
    made the version unreadable, so `run` refused. `Gone` now requires
    `still_listed()` to come back negative from a *non-empty* session
    listing; an empty or failed listing means "unknown" and waiting
    continues.
  * **`sync` goes through the /arc mount, not a container.** Queue latency
    dominates small work: the 1-core sync job sat `Pending` for 28 minutes,
    twice, to do seconds of copying, while the sshfs mount turns out to be
    writable and does the same unpack in 19 s. `run_root_local()` finds the
    mounted run tree (`$CANFAR_RUN_LOCAL`, else `~/canfar_home`) and
    `do_sync` unpacks there in the same order as `update_src.sh` - version
    promoted only after the tar - with `--job` to force a container. Verified
    byte-identical: the arc copies of `sed_stack.py`, `scene.py` and
    `catalog.py` sha-match `git show main:<path>`. The hazard is unchanged
    from the container path, since both rewrite source under running jobs.
  * **CANFAR runs a commit, not a working tree.** `do_push` tarred `src/`
    off disk, so the 06:05 push carried another session's uncommitted
    `sed_stack.py` and `scene.py` and would have shipped them to 17
    full-field jobs. `push` now ships `git archive` of `main` (`--ref` for
    another commit, `--worktree` to opt back in, loudly). Provenance runs
    end to end: `push` uploads `SRC_VERSION.pending`, the unpack step
    promotes it to `SRC_VERSION` -- deliberately after untarring, so the
    file means "installed" and not "uploaded" -- `run.sh` prints it at the
    top of every job log, and `do_run` refuses to submit when it does not
    match the local ref. That last check is the one with teeth: `push`
    uploads but does not unpack, so a push without a sync leaves every job
    importing the previous campaign's code with entirely normal-looking
    outputs. Verified here: the arc source was four hours older than the
    memory fix the full-field runs depend on.
  * `submit.py kill` -- destroying every session took two rounds, because
    skaha does not list a session until the service registers it and a
    `--no-wait` campaign registers over several minutes. `kill` sweeps until
    N consecutive passes come back empty (`--sweeps`, default 3) and spares
    the `sync` job by default. There was no stop command at all before.
  * `push --src-only`. Only `setup_env.sh` unpacks `psf.tar`, so a `push`
    before a `sync` uploads several hundred MB (732 here) that never reach
    `$RUN/PSF`. Nothing is lost by skipping it: `PSFFactory` skips grid files
    that already exist unless `overwrite` is set.
  * `campaign.has_shared_grids` counted only local grids, but the run reads
    `$RUN/PSF` on arc. COSMOS has its 30" halo pair on arc and none locally,
    so the check said "missing" and would have serialised one full-field band
    ahead of the other five for nothing. It now takes the arc listing too
    (`arc_psf_names`, falling back to local if the listing fails). Verified:
    cosmos False -> True, egs correctly still False, uds unchanged.
  * `campaign.py --skip` -- `--from` can only drop a prefix of the chain, so
    there was no way to re-arcify without also re-staging.
  * cores default 2 -> 4 (`cores_for`), and the memory guidance in
    `README.md`/`MANUAL.md` replaced with the measured full-field numbers.
  Also documented: skaha appends a `-1` replica index to session names, so
  `mophongo-uds-f770w-v1-0` lists as `mophongo-uds-f770w-v1-0-1`.
  No tests: `examples/canfar/` is not covered by the suite. Verified by
  invoking each changed path (`kill --help`, `push --help`, a `--skip`
  dry-run, and `arc_psf_names` against the live listing).
- [x] Verification v9: all four UDS MIRI bands at r < 3' (2026-08-13,
  commit `ff1447a`, `examples/minerva/verification/v9/`). Same code and PSF
  grids as v8's F770W re-run, applied to every band, at the trial radius the
  band configs already carry (3' against v8's 1.5'), so the band trend is
  finally read off one code version. ~17.5k matched sources per band against
  v8's ~4.6k. IDL est1 (no EE) +0.024/+0.022/+0.034/+0.053 mag and psfcor
  0.976/0.983/0.973/0.957 for F770W/F1280W/F1500W/F1800W; mock
  recovered/true 0.9607/0.9646/0.9673/0.9688 with resid/noise 0.799
  throughout. F770W is the only band with the final code at both radii and
  moves +0.001 mag between them, so doubling the radius is worth ~0.001 mag
  and the larger shifts in the other three bands (est1 -0.013 to -0.006
  against v8's first pass) are the late fixes, not the patch. The mock leg
  does not depend on the trial radius and reproduces v8 to <= 0.001 in
  `med_lo` in every band, so v8's caveat about superseded mock numbers turns
  out not to matter. Weight calibration holds at twice the radius: F770W
  `sigma_true` 3405 here against 3295 on v8's patch.
- [ ] Scene astrometric shifts do not converge at r < 3' (2026-08-13, found
  in v9). Three of four bands leave scenes moving after the 5 allowed passes
  (tol 0.1 px = 4 mas): F1800W scene 27 at 3.12 px, F1280W scene 12 at
  1.44 px, F770W scene 66 at 0.61 px; F1500W converged in 4. The failure is a
  walk, not an oscillation -- F1800W scene 27 runs (-1.27, 0.28) ->
  (-2.00, 0.88) -> (-1.88, 2.47) -> (0.86, 4.63) -> (2.38, 3.90) px. At
  r < 1.5' F770W converged in one pass. Scenes get fewer and larger toward
  longer wavelength (F770W 70 scenes, sizes 2-1003, median 226; F1800W 31
  scenes, sizes 2-1964, median 471) while the GP astrometric model keeps
  `length_scale` 400 px (32" at 80 mas), so a 6'-wide scene is described by a
  correlation length a tenth its size. The partition exceeded
  `scene_max_size` (800) at both radii -- v8's r < 1.5' F770W run already
  produced a 1061-template scene -- so the cap is not what changed. Needs
  either a size cap that holds or a change to `fit_astrometry_niter` (5) and
  `astrom_damping` (0.8).
- [ ] Memory pressure on full-field runs (2026-08-13). Full-field MINERVA
  runs were running out of memory; the target is a peak under 48 GB. Nothing
  here changes what is fitted -- every product is bit-identical -- so the
  changes are listed by what they stop allocating.
  Scale of a full-field UDS F770W run, for the numbers below: the detection
  grid is 34560x25344 (876 Mpx, 3.5 GB per float32 array, the same for the
  int32 segmap and for the F770W mosaic once it is upsampled onto that grid),
  and 138,609 of the 345,792 catalog sources fall inside the MIRI footprint.
  Their stamps have a 100 px floor (the detection PSF stamp) and a mean side
  of 104 px, so one full set of template pixels is 5.95 GB.
  * Build-stage template snapshots. `_prepare_hi_templates` kept
    `templates_extracted` and `templates_extended` as two `deepcopy`s of
    `tmpls`, and `_convolved_templates` took a third for `tmpls_lo`. Each
    `deepcopy` of a `Templates` also duplicates its `segmap`, so a snapshot
    cost the stamps *plus* the whole detection-grid segmentation map: about
    10.4 GB each at full-field scale, ~31 GB for the three.
    The two build-stage names now alias `tmpls` except where the two stages
    genuinely differ -- only `extend_mode` `'psf_convolution'`/`'psf_model'`
    run a post-extraction pass that rewrites pixels; the default
    `'psf_wings'` and the other build-time schemes compose inside
    `extract_templates`, so the snapshots were already pixel-for-pixel
    identical. `tmpls_lo` is now a shallow container over the same `Template`
    objects: `prune_outside_weight` only drops list entries, and
    `convolve_templates(inplace=False)` copies each stamp as it goes.
  * Projection to the block-replicated grid built a second full list before
    dropping the first; it now projects in place, one stamp at a time.
  * `model_images` stored `image - residual` per band, a third full-field
    array derived from two the run already holds. It is now `_ModelImages`,
    a sequence that subtracts on access and caches the band asked for last.
    Nothing in the fit reads it; only the diagnostics do.
  * The detection-band inverse variance is read only while templates are
    built (the build schemes grade data against the PSF by SNR). `run()`
    releases it afterwards and records that it went, so a second `run()` on
    the same instance raises instead of quietly rebuilding weightless
    templates.
  * `get_bg_and_ivar` masked `sci` and `wht` into full-resolution copies, and
    `np.where(mask, float32, 0.0)` promotes to float64 on the way, so three
    coarse arrays of 200 kB cost ~21 GB to produce. `_valid_block_means` now
    reduces one band of coarse rows at a time; the three coarse outputs are
    bit-identical on real UDS data. The median weight in its log line is
    taken on an 8x8-strided subsample (boolean-indexing the full mask copies
    the mosaic, and `np.median` partitions a second copy). `need_bg=False`
    skips the full-resolution background entirely, which is what the
    detection-ivar path wanted: it used it for one median that
    `get_bg_and_ivar` already logs.
  * `_upsample_flux_conserving_image_and_ivar` used
    `block_replicate(conserve_sum=True)`, which divides by `k**2` in float64
    -- a 7 GB intermediate for a 3.5 GB float32 result. Replicating without
    it and dividing in place stays in float32; dividing by an integer square
    is exact, and the output is bit-identical for k=2,3,4.
  * `as_label_array` validated a float segmap whole-array:
    `arr[np.isfinite(arr)]` copies every finite pixel, `np.rint` copies it
    again and `np.nan_to_num` a third time. COSMOS is the field that ships
    float64 labels (BITPIX -64, 621 Mpx = 4.97 GB stored, 2.48 GB as int32);
    UDS and EGS ship int32. Validation and cast now run in ~16 Mpx bands:
    peak RSS for the COSMOS map goes 15.8 -> 8.0 GB for the same result, and
    non-finite pixels now become background rather than being handed to
    `astype(int32)` as an overflowing infinity. Integer segmaps are still
    returned untouched whatever their width or byte order (UDS and EGS
    arrive as `>i4`): the full-field read hands over a memmap view, and
    narrowing it to native int32 would convert file-backed pages into
    anonymous memory.
  * Every shifted template kept its pre-shift pixels (`_data_unshifted`) so
    that each astrometric pass resamples the original rather than compounding
    the cubic smoothing -- correct, but the copy was retained for the whole
    run, a second full set of stamps (~6 GB) still held while the residual
    allocated its 3.5 GB. `run()` now releases them once the shifts are
    settled, and `apply_template_shifts` raises rather than shift a released
    template, so the smoothing-compound failure cannot return silently. The
    `.copy()` also went: `tmpl.data` is rebound to the shifted array on the
    next line, so holding the original reference is all that was needed.
  * `_save_repair_cache` found its changed pixels with
    `(sci != sci0) | (wht != wht0) | (seg != seg0)`, three full-field boolean
    arrays plus the temporaries of the two ORs (4.4 GB) at the one moment
    both the pre- and post-repair mosaics are in memory. It now scans in
    bands of ~4 Mpx and concatenates the indices.
  * The residual is now formed in place (`np.subtract(..., out=res)`), the
    pre-repair `sci`/`segmap` snapshots and the raw hi-res weight map are
    freed once the cache is written, and `Template.__deepcopy__` shares the
    parent-image WCS (one object per template set, read-only) instead of
    duplicating it per template.
  Separately, a speed defect that made a full field impractical regardless:
  `extract_templates` resolved each source's segment with
  `SegmentationImage.get_index`, which validates the label with
  `np.setdiff1d` against the full label list -- one sort of all 345,792
  labels per source, 138,610 times. The label comes straight out of
  `segm.data` and is nonzero, so it is valid by construction;
  `np.searchsorted` on the (sorted) label array is the whole lookup. A
  full-field extraction went from no progress in 100 minutes to 2.5 minutes
  at ~950 sources/s. `plot_result`'s scene map had the same call in a
  per-template loop and now builds one label->index dict.
  Validation: UDS F770W 3' trial, `psfs kernels load fit outputs`, HEAD
  against the working tree on the same machine. Peak physical footprint
  37.1 -> 20.0 GB. The fit table is identical in all 17,796 rows and every
  column, and the residual image is identical pixel for pixel. The trial
  understates the saving: its template set is 1/8 of a full field, and its
  mosaic-sized arrays are full-shape but only the patch is ever touched --
  except in a `deepcopy`, which writes every page. `get_bg_and_ivar` was
  checked separately against HEAD on two 3000x3000 UDS patches (`bg` and
  `ivar` bit-identical), and the upsample for k=2,3,4.
  `tests/test_memory_footprint.py` pins the equivalences.
  Full field (UDS F770W, `trial: null`, 138,610 templates, 591 scenes, repair
  reloaded from cache, `load fit`): completed in 87 min at **46.5 GB peak
  physical footprint** (32.2 GB max RSS). Checkpoints: `(start)` 10.4 GB,
  `(templates)` 28.2, weight-map release -> 24.7, `(convolved)` 27.7,
  `(end)` 11.0. `load` alone peaks at 22.2 GB, so the fit contributes the
  rest. A 15 s RSS sampler over the same run topped out at 29.1 GB, so the
  peak is a spike shorter than that and does not sit at any checkpoint --
  under the 48 GB ceiling but without much margin. Being chased; the
  candidates are the two buffers in the astrometric solve that scale with a
  scene's *spatial* extent rather than its template count
  (`assemble_scene_system_AB`'s `Bq`/`Bl`, nB float64 planes over the bright
  anchors' bounding box, doubled when the leverage cap clips, and
  `Scene.model_image`'s float64 plane over the full scene bbox).
  Left alone and worth knowing: the saturation repair sets the peak on a
  trial patch, and `run()`'s finiteness guard on `images[i]` is behind
  `if images[i] is None`, so it never runs. Both are in TODO.md.
- [x] Weight calibration on partially covered fields (2026-08-12). The
  robust baseline in `get_bg_and_ivar` (`med0`/`nmad0`) was taken over all
  coarse blocks, including the zero-filled pixels outside the mosaic
  footprint -- the normal case for MIRI, where a square box around any patch
  is roughly half uncovered. Those zeros dragged `sigma0` from 449 to 12,
  which collapsed the detection threshold, which flagged the whole field as
  source, after which the background fit interpolated the data and the
  residual scatter went to zero: `sigma_true` 1.4e-4 and the inverse
  variance inflated by 5e7.
  Symptoms were entirely downstream and the pipeline's own diagnostics read
  clean: every source came out at SNR ~1e7, so every scene reported bright
  anchors and `0 scene(s) without bright members` was true and meaningless;
  the partition fragmented 15 -> 207 scenes and each fitted an order-0 shift
  from noise (spatially uncorrelated: |dshift| 0.207 px between neighbours
  at 0-20" against 0.204 px for random pairs). Fitted fluxes moved 13%.
  The baseline is now taken over valid blocks only, and `MIN_BG_FRACTION`
  (0.02) makes the estimator warn and leave the weight map unscaled rather
  than calibrate on a field the source mask has eaten. Two different patches
  of the same mosaic now return `sigma_true` 3295 and 3291 (0.1%), where
  they previously differed by 2.3e7. The full-field runs were affected too,
  so v6's 3518 and the 3518 -> 3921 shift previously attributed to the P1-03
  retune were partly this bias.
  Also corrected: the 207-scene fragmentation was this bug, not the
  convolution crop as first reported. With the calibration fixed the
  partition is 15 scenes (median 250) against v6's 13 (median 346), and 120
  sources exceed SNR 15.
- [x] Added a science-safe interpolated companion to the all-field MINERVA SED
  stack (2026-08-12). Interpolate each galaxy only between adjacent valid
  field bands in linear normalized F-lambda versus log wavelength, retain
  signed fluxes, keep broken/missing-band intervals uncovered, and preserve
  the equal-galaxy redshift-bin denominator. The existing half-maximum
  filter-footprint stack remains the raw measurement view. The final cell
  reconstruction uses each galaxy's connected union of valid half-maximum
  supports, holds component endpoints only to their physical edges, and
  arithmetic-averages effectively coincident pivots. Its observed evaluation
  grid is <=0.0025 dex and includes every filter edge/pivot; rest stays at the
  requested 100 A. Added raw interpolated mean/count FITS extensions and a
  full set of per-field interpolated mean/count cubes with explicit frame,
  support-model, coincident-pivot, and wavelength-table provenance, plus a
  separate contrast panel that continuum-subtracts only for rendering, with
  masked 4x redshift interpolation that never crosses a finite-support gap.
  The full COSMOS+EGS+UDS render completed for 357,044 normalized galaxies;
  `tests/test_sed_stack.py`: 23 passed, including nested broad-filter union
  connectivity and full component endpoint support.
- [x] Band convolution back on the stamp footprint (2026-08-12).
  `convolve_templates` had switched to `convolve_cutout`, growing every
  convolved template to full linear support (8" stamp + 8" kernel = 16",
  402 px at 40 mas) — outer half a truncated-PSF x truncated-kernel outer
  product (0.08% of a point source's flux), 6.7 GB stamps files, inflated
  scene overlaps. Restored the shared `_convolve2d` (mode="same")
  convention: footprint unchanged, truncated sum kept (no renormalisation
  hides the boundary; `stampcor` relies on it), totals mean "flux within
  the stamp box" corrected by `ee_psf_lo`, matching the PSF-throughput
  convention. `convolve_cutout` itself is untouched (the `extend_with_psf`
  smearing path needs full support). Tests updated to the contract:
  footprint + truncated-sum checks, interior-alignment check for the
  parity test, changed-pixels+flag check for the inplace test. 320 passed.
  Predicted effect on the faint-template aperture correction: 1.4389 ->
  1.4377 (matched-support pure-PSF ladder).
- [x] stampcor-vs-IDL offset attributed by measurement (2026-08-12).
  Panel (b)'s ~1.3% (F770W) / ~3.7% (F1800W) offset and the IDL fan are
  NOT the PSF (matched-support 1/EE ours/theirs = 1.0043), NOT support
  (P_det x kernel puts 0.08% outside the 8" box), NOT the mosaics (same-sky
  raw apertures on 40 mas native vs 80 mas replicated agree to 0.2-0.3%).
  Measured on 735 QA sources with `legacy` subphot fully reproduced
  (totcor re-derived from their saved model fits to 0.02%): subphot
  computes totcor with the aperture at the UNSHIFTED catalog position on
  the SHIFTED best-fit model (their own `@@@` comment fixes this for flux
  but totcor is computed 25 lines earlier) — inflation med +1.2%, scaling
  with |fitted shift|: +0.2% (<1 px) to +42% (>4 px) = the fan. At model
  centroids IDL totcor = 1.4400 ~= its pure-PSF value 1.4373. Python's
  centered measurement on near-PSF faint templates = the flat pinned band
  (its own centering term +0.8%). Matched pairs (n=12, QA-cluster trial
  run): released py/IDL 0.9746 -> 0.9819 with both apertures at
  centroids; faint subset 0.905 -> 0.982. Residual ~1.5-2%: python blends
  data cores down to segment SNR 5 where subphot switches to pure PSF at
  15 (`tmpl_snrlo`), and segment-truncated cores are sharper than the PSF.
  The ~2% raw-flux offset is the same centering story: subphot measures at
  the fitted position (`xaper = xc - p`), python at the catalog position.
  Analysis scripts + decomposition table in the session scratchpad
  (`decompose_stampcor.py`, `qa_run/decomposition.ecsv`).
- [x] Astrometric-anchor messages corrected (2026-08-12, wording only).
  Three of them described filters that are not what the code does:
  * `Scene.solve`'s skip warning said "no bright **non-star** isolated
    sources", naming a cut that is off by default (`astrom_exclude_stars`
    is False — unsaturated stars are the best anchors). It now states the
    real condition and the thresholds in force: "fewer than 2 sources pass
    the astrometric anchor cuts (SNR > 15, isolation >= 0.7)", with
    ", stars excluded" appended only when that option is on.
  * Its TODO claimed isolation filtering "can't be applied at merge time";
    `generate_scenes` has applied it there since (`scene.py:751-757`). The
    note now says what is actually true — merge-time cuts run against the
    full-field normal matrix, where a source competes with neighbours
    outside its scene as well, so a scene can pass there and still fall
    short at solve time.
  * `_prepare_hi_templates` logged "Marked N templates as stars (excluded
    from astrometry)" unconditionally; it now reports "excluded" or "kept
    as anchors" from `config.astrom_exclude_stars`.
  No behavior change; full suite 312 passed.
- [x] MINERVA UDS "mother of all SEDs" diagnostic (2026-08-12). Added the
  reusable NumPy raster/stack helpers in `src/mophongo/sed_stack.py` and the
  survey-specific `examples/minerva/plot_uds_sed_stack.py`. The current
  n3.0_m3.1_v1.2.1 catalog is paired with its exact 345,792-row EAzY table
  only after row-ID and sky-position assertions; positive `z_spec` replaces
  `z_phot`. All 33 HST/JWST filters are included, including F770W/F1280W/
  F1500W/F1800W. Raw F-nu is converted to relative F-lambda, normalized by
  bracketed interpolation at rest 5000 A, painted over each filter's actual
  half-maximum interval, and unweighted-nanmeaned in bins with
  `Delta log(1+z)=log(1.05)`. Negative measurements survive and uncovered
  pixels remain NaN. The default S/N >= 5 normalization sample contains
  98,752 galaxies. Generated rest/observed comparison PNG+PDF, count PNG,
  summary JSON, and a FITS product carrying both mean/count images and all
  redshift, wavelength, and filter tables. `tests/test_sed_stack.py` covers
  conversion, normalization/bracketing, zero/negative valid-flux averaging,
  overlap semantics, rest-frame support, nanmean/chunk invariance, and count
  thresholds. The
  science raster now uses fixed 100 A wavelength bins for the rest frame
  (1,992 pixels over the default range), with H-alpha and [O III] 5007 guides
  in both frames and separate wide rest- and observed-frame exports. The
  observed frame uses cells bounded by the filters' actual half-maximum edges:
  isolated cells span one complete band, while overlaps are split only at a
  physical filter edge so valid measurements are still averaged once per
  galaxy. Fixed filter footprints remain vertical while rest-frame features
  trace diagonals with redshift. All wavelength cells are drawn with their true
  edges on the log-wavelength display axis.
  `stack_filter_seds` now sweeps
  exact per-galaxy filter interval events into redshift-bin difference arrays,
  preserving the original overlap average and valid-galaxy denominator
  without allocating a source-by-wavelength cube. The rest-5000 A divisor is
  now an inverse-variance weighted local linear fit using the nearest three
  valid bands with mandatory bracketing (two-band interpolation fallback),
  increasing the default normalized sample from 98,752 to 113,356 galaxies.
  The PDF embeds the raster at the requested output DPI.
  Generalized the diagnostic to the three distinct current MINERVA fields:
  COSMOS n3.0_m3.0_v1.0.1 (38 bands), EGS n2.0_m2.1_v1.3.1 (37), and UDS
  n3.0_m3.1_v1.2.1 (33). Their exact EAzY tables are required and fully
  row/ID/sky validated. Each field is normalized independently, absent bands
  are padded invalid on a 41-filter union, and 357,044 selected galaxies are
  concatenated before stacking, so the combined mean is galaxy-weighted rather
  than an equal-field mean. The all-field product includes combined and
  per-field mean/count planes plus FIELDS and FIELD_FILTERS provenance tables.
  The observed union grid has 79 physical filter-boundary cells; the rest grid
  remains fixed at 100 A.
- [x] `trial` patch replaces `r_trial`/`trial_center`, and a trial run now
  reads only its patch (2026-08-12). `RunConfig.trial` is
  `{"center": [ra, dec], "radius": <arcmin>}` with an optional `"margin"`
  (arcsec, default 60) for the PSF support, template stamps and convolution
  wings that reach outside the patch; `None` is a full-field run. The old
  pair is retired outright — `from_json` raises and names the replacement
  rather than ignoring them. `RunConfig.trial_geometry()` validates and
  returns `((ra, dec), radius, margin)`.
  Previously `r_trial` only trimmed the *catalog*: the full mosaic was read,
  background-fitted and saturation-repaired regardless, which is why
  `examples/canfar/submit.py` told users to keep `--ram 48` "even for a small
  patch". On UDS F770W the repair scanned 22455 holes and repaired 129 stars
  across the field so that 4 saturated templates could reach an
  `r_trial=1.5'` fit.
  `_read_image(path, box)` now pulls just the patch off disk with
  `hdu.section` and places it in a full-shape array. Untouched pages of that
  array are never faulted in (measured: `np.zeros(876 Mpx)` costs 0.026 GB
  RSS), so **no pixel coordinate, slice, catalog x/y or WCS changes** — the
  whole point of doing it this way. The background/ivar estimate takes the
  same box (`Pipeline._bg_and_ivar_boxed`) because running it on a
  full-shape array would both fault in the mosaic and measure the noise of a
  field of zeros. The trial box is hashed into `_repair_provenance`, so a
  trial cache can no longer satisfy a full-field run.
  Measured on MINERVA UDS F770W, radius 0.5': file reads 12.24 GB -> 0.88 GB,
  `load_data` 8.1 s, reading 2.3% of the hi-res mosaic. Peak RSS fell 22.3 ->
  15.5 GB only, because full-grid arithmetic downstream still touches every
  page; the remaining offenders are listed in TODO.md.
  A trial run is deliberately *not* a subset of a production run: the
  background and the ivar calibration are measured on the patch, so
  `sigma_true` and the flux errors differ. `load_data` logs that as a warning
  on every trial run.
  Migrated 58 configs (`examples/canfar/*.json`, `examples/minerva/*.json`,
  the DR0/DR0.1 examples) with a line-based script, so the `#` comment lines
  a JSON round-trip would have dropped survive. Also updated
  `make_minerva_configs.py`, `canfar/arcify.py` (`--r-trial` now overrides
  only the radius and keeps the config's centre), `canfar/submit.py`,
  `canfar/MANUAL.md`, `compare_dr0_dr0.1.py` and
  `minerva/run_verification_v2.py`.
- [x] Scene diagnostic: the colour panel no longer inherits the saturated
  nulling (2026-08-12). `Scene.plot` overwrote `img_cut` in place, so the
  colour composite's red and green channels were built from the nulled image
  while the blue (template) channel was not. The composite now uses the raw
  image and is byte-identical with and without `null_segments`.
- [x] Verification v8 (2026-08-12): first run on the P1-01..P1-05 fixes.
  `run_verification_v2.py --version v8 --scheme psf_wings --psf-dir data/PSF8
  --psf-size 8.0 --r-trial 1.5`, all three stages (~48 min). Unlike v7, the
  real-data fits were re-run rather than reused, because P1-03/P1-04 change
  the ivar calibration and P1-05 changes which PSF region each source reads.
  Mock leg moved +0.5 to +0.8% toward unity in every band (F770W 0.9533 ->
  0.9608, F1280W 0.9573 -> 0.9645, F1500W 0.9606 -> 0.9673, F1800W 0.9637 ->
  0.9688), so the `psf_wings` extended-source deficit is 3.1-3.9% rather than
  v7's 3.6-4.7%; the bluest-worst band trend survives. IDL leg is stable
  (psfcor within 0.003 of v7). Five fixes landed together, so this does not
  attribute the improvement — a per-fix run is needed for that.
  Also noted: the fitted-band `sigma_true` moved 3518 -> 3921 (+11.5%, so
  ~20% lower ivar and ~11% larger flux errors) and the detection band 1.047
  -> 1.013. That is the P1-03 retune landing on real data; the detection
  band moved toward the honest value of 1.0, but the MIRI band's arbitrary
  drizzle normalisation makes its number uncheckable on its own. This is the
  "validate on a real mosaic" item already in TODO.md.
- [x] Full-field scene map as a run output (2026-08-12).
  `write_outputs` now writes `<name>_scene_map.png` next to the scene
  catalog, under the same `scene_plots` switch: every segment colored by the
  scene that fitted it, each scene's bbox drawn over it. It reuses
  `verification.save_scene_overview`, which was written for mock fields and
  did not scale — it built the scene map with one `np.isin` over the full
  segmap *per scene* (3000 scenes x 219 Mpx on a MINERVA mosaic). The new
  `verification.scene_label_map` does it as a single label lookup-table pass,
  and fields wider than `max_side=4000` are reduced by block *maximum* so a
  3-pixel segment survives the decimation instead of falling between
  samples. Measured at UDS scale (17280x12672, 50k segments, 3000 scenes):
  0.41 s, peak RSS set by the segmap itself. Bounding boxes are drawn up to
  `max_boxes=250` scenes, past which they overlap into noise.
- [x] Scene diagnostics keep foreign saturated stars in the image panel
  (2026-08-12). `Scene.plot(null_segments=...)` zeroed those pixels in the
  image panel; it now draws them and only excludes them from that panel's
  grayscale stretch, which is what the nulling was protecting. The residual
  panel still nulls them (the fit residual under a saturated core is
  meaningless). `tests/test_scene_saturated.py` checks the new contract:
  image pixels identical with and without `null_segments`, stretch strictly
  narrower with it.
- [x] `mophongo` console script: command-line access to a finished run's
  products (2026-08-12, new `src/mophongo/cli.py`). Five subcommands, each a
  wrapper over an existing method, with no algorithmic logic of its own:
  * `psf <map.geojson|run.json> RA DEC` writes the PSF or matching kernel of
    the region containing that position to FITS. It reads only the cached
    region map, so it is instant. The stamp gets a WCS centered on the
    requested position that inherits the CD matrix of the mosaic the map was
    drizzled onto — `psf_hi` and `kernel` are on the detection grid, `psf_lo`
    on the band's own — resolved through the run config beside the map;
    `--pixel-scale` builds a north-up tangent plane when there is none. The
    header carries the region key, the stamp's encircled energy
    (`EE_BOX`/`EE_RLIM`/`R_LIM`), and the map's provenance columns.
  * `stamps <run.json> ID...` writes one source's
    `Pipeline.source_products` dict as a multi-extension FITS: `IMG_HI`,
    `SEGMAP`, `TMPL_HI`, `IMG_LO`, `TMPL_LO`, `MODEL`, `RESID`, `PSF_HI`,
    `PSF_LO`, each carrying the sliced WCS of its parent grid, the fitted
    scalars in the primary header, and the fit-table row as `FITROW`.
  * `diag <run.json> ID...` writes the subphot six-panel PNG
    (`--style stages` for the template-construction row).
  * `info` and `run` delegate to `Pipeline.info` and `pipeline.main`.
  `stamps`/`diag` restore the run through `load_fit` once and loop over the
  ids given. Non-finite floats (an unmeasured `ee_psf_lo`, an unregularized
  kernel) are written as undefined FITS cards rather than dropped, since
  `nan` is illegal in a header but is a real product value.
  A config argument may be the JSON or the run directory, as for
  `Pipeline.from_config`; when a relative `out_dir` does not hold the map
  (it resolves against the process CWD, as everywhere else) the directory
  the config sits in is used instead, so a finished run opens from anywhere.
  New `tests/test_cli.py` (10 tests) covers region lookup by position, both
  WCS paths, the provenance header, the stamps layout and its cutout WCS,
  both diagnostic styles, and an end-to-end config -> `load_fit` -> files
  run. Smoke-tested against the real `examples/uds_770_dr0.1` products:
  80 mas `psf_lo`, 40 mas `kernel`, both centered on the requested position.
  Full suite 270 passed; the one failure, `tests/test_background_masking.py
  ::test_dilation_grows_the_exclusion`, is in concurrent untracked work and
  is untouched by this change.
- [x] P1-05 fixed: template geometry ops kept the parent WCS (2026-08-12).
  `Template.convolve_cutout` and `project_to_block_replicated_grid` built
  the new `Template` on a parent-sized image but passed `self.wcs` -- this
  cutout's WCS, i.e. the parent with CRPIX shifted by the stamp origin -- as
  that image's WCS. The new template's `wcs_original` was therefore off by
  the stamp origin, and the next operation shifted it again. Both now pass
  `self.wcs_original`; the `# note wcs origin is wrong` comment is resolved
  rather than annotated. Measured on a 1 arcsec/px TAN WCS, a source at
  (260, 190) in a 41x41 stamp moved (-240, +170) arcsec through either
  operation and is now invariant to 1e-6 arcsec.
  Not cosmetic: `Templates.convolve_templates` converts `position_original`
  through `wcs_original` to choose the `PSFRegionMap` region and the
  encircled-energy correction, so a spatially varying PSF was being read
  from the wrong part of the mosaic.
  Checked the neighbours: `Template.downsample` takes an explicit `wcs_lo`
  from its caller and is correct. `AlignedCutout.downsample`/`upsample`
  share the pattern but return `AlignedCutout`, which has no `wcs_original`
  to pass -- left alone and recorded in TODO.md.
  New `tests/test_template_wcs_provenance.py` (13 tests) asserts world
  coordinates survive convolution, block projection, padding and repeated
  operations, across translated, rotated, fine-scale and SIP WCSs, and that
  `wcs` and `wcs_original` keep agreeing about where the source is. 11 of
  the 13 failed before the fix.
- [x] P1-04 fixed: non-finite science no longer poisons preprocessing
  (2026-08-12). `get_bg_and_ivar` masked only the weight map, so a single
  non-finite science pixel spread over its whole block in the coarse mean,
  through the median and MAD, and made every statistic NaN: one NaN (or
  inf) anywhere returned a background and an inverse variance that were
  **0% finite**, and a four-row NaN border left the background 19.5%
  finite. There is now one common `valid = isfinite(sci) & isfinite(wht) &
  (wht > 0)` mask applied before binning, block means taken over each
  block's valid pixels (`vfrac`) so a bad pixel costs sample size instead of
  poisoning the block, the sigma measured only on blocks that are >90%
  valid, and a warning plus an unscaled weight map when no usable background
  sample exists at all. `bg_gaussian_normalized` now drops non-finite
  samples from the mask and replaces them explicitly: `NaN * 0` is `NaN`, so
  a pixel the mask already excluded still spread across the smoothing
  footprint (15% of the output). A single NaN now changes the recovered
  calibration by 0.1% (1.012 vs 1.013). The `0 + 0` scalar `seg_all` path
  went away with the P1-03 rewrite, which allocates a full-shape mask.
- [x] P1-03 fixed: background source mask had coupled polarity and
  threshold errors (2026-08-12). Three defects that partly cancelled:
  (a) the bright pass convolved with an *unnormalised* 29-pixel disk, whose
  white-noise RMS is `sqrt(N) sigma`, but scaled the threshold by the
  normalised kernel's `1/sqrt(N)` -- a factor of 29 too low, flagging 47% of
  a pure noise field; (b) `binary_dilation(seg_all == 0)` dilated the
  *background* mask, growing background into the sources and re-admitting
  100% of an r<=2 source and 59% of an r=8 source; (c) the faint pass ran at
  1 sigma with `npixels=1`, which alone flags 16% of pure noise. Together
  they produced a mask that looked reasonable (3.9% of pure noise excluded)
  by accident, with no predictable dependence on `detect_thresh` or
  `dilate` -- mask extent was not even monotone in `dilate`
  (`{1: 547, 2: 527, 3: 555, 5: 1099}`).
  The mask construction moved into `catalog.coarse_source_mask`, which
  normalises the smoothing kernel, dilates the *source* mask, and runs the
  faint pass at `faint_thresh` (new parameter, default 4.0) with
  `npixels=3`. `detect_thresh` now means "sigma of the smoothed image" and
  its default moved 1.0 -> 2.5. Chosen against measurements, not by
  inspection: on injected sources over correlated noise at two depths, the
  chosen setting recovers `sigma_true` to 1.007 (shipped: 1.016), masks 96%
  of injected source flux (shipped: 88%), and flags 0.0% of a pure-noise
  field. The `mask_src0` line, computed and never used, is gone.
  New `tests/test_background_masking.py` (20 tests) covers background bias,
  recovered variance, mask occupancy, source coverage, `dilate` monotonicity,
  depth dependence, correlated noise, and the non-finite cases above. Two of
  them fail on the shipped logic; the compact-source polarity test passes
  either way (the bright pass's smoothing halo absorbed the inverted
  dilation) and is labelled a regression guard, not a reproduction.
- [x] P1-02 fixed: final flux-only solve on the shifted templates
  (2026-08-12). Each astrometry pass solved fluxes and shifts together and
  *then* resampled the templates, so the fluxes it produced belonged to the
  pre-shift basis and the last applied shift was never accounted for. The
  pipeline then built the model, residual and stamps from the shifted
  templates using those stale fluxes. `Pipeline.run` now runs one flux-only
  pass per scene after the loop (`replace(config, fit_astrometry_niter=0)`),
  for every scene regardless of the convergence verdict; fitted shifts are
  untouched. On the `offset=(0.6, -0.4)` mock the fluxes moved by 0.18% max
  and 0.027% median, and scene chi2 dropped 22105.05 -> 22101.18. The
  correction is smaller than the review measured because the P1-01 fix lands
  a more accurate step and leaves less residual shift at stop; it does not
  remove it.
  New `test_final_fluxes_are_stationary_on_the_shifted_templates` re-solves
  each scene from `build_normal` on the final templates and requires the
  stored fluxes and errors to match, the normal equations to be stationary,
  and `model_image` to be built from those same fluxes and stamps. Verified
  to fail with the pipeline change reverted.
  `test_astrometry_passes_skip_converged_scenes` counts `Scene.solve` calls
  and now expects `astrom_niter + 1`; the loop still drops converged scenes.
  Full suite 254 passed.
- [x] P1-01 fixed: exact joint astrometric blocks (2026-08-12).
  `assemble_scene_system_AB` accumulated only each template's own gradient
  products, so `AB`, `BB` omitted every cross-template term and the x-y
  block; `bB` was already scene-wide, which made the system inconsistent.
  It now forms the scene-wide derivative columns
  `B_k = sum_i -alpha_i phi_k(u_i,v_i) grad(T_i)` over the union footprint
  of the bright anchors and contracts them, so the blocks are the exact
  normal equations of the design in `docs/fitting.md`. The `continue` that
  dropped faint flux rows from `AB` is gone -- a faint row is simply its
  overlap with the anchors' columns -- and `ab_from_bright_only` went with
  it (it only selected the defect; three call sites).
  Measured against a dense reference design: the old blocks were exact for
  isolated anchors and wrong only in blends. At order 0 the error cancelled
  by symmetry (`B_x = -d(model)/dx`), so the fixed point was accidentally
  right but the step was undersized -- a 0.30 px offset came back as
  0.18 px at 6 px separation, and with `astrom_shift_tol=0.1` the loop
  tol-stopped 0.05 px short. At order 1 (the default) the cancellation dies
  and a perfectly aligned 12-source blend fitted a spurious shift field of
  0.054 px rms, 0.16 px peak; it is now machine zero (1e-15 px unregularized,
  2.5e-6 px under the default `reg_flux` ridge). Single-pass flux error in a
  4 px blend: 11.6% -> 0.34%. Flux errors now inherit the shift covariance
  through `S_w = A_w - AB_w AB_w^T`, which the near-zero `AB` had suppressed
  by up to 40%.
  `leverage_cap` keeps its meaning: it is a weight on the shift equations,
  entering `AB`/`bB` linearly as `wl_i` and `BB` as `sqrt(wl_i wl_j)` so the
  diagonal stays `wl_i` and the implied shift `dx_i` is untouched. It
  reduces exactly to the old per-anchor form when nothing overlaps.
  New `tests/test_scene_astrometry_blocks.py` (14 tests) checks block
  equivalence against an independent dense design (orders 0/1/2, both axes,
  bright+faint, non-uniform weights), zero spurious shift on an aligned
  blend, separation-independent recovery, error inflation, and both cap
  properties. 11 of the 14 failed before the fix.
  `tests/test_pipeline.py::test_shift_field_arrows_track_applied_template_shifts`
  had been passing *on* the bug: it ran with no injected offset and needed
  the spurious shifts to have anything to track; it now injects
  `offset=(0.6, -0.4)`. Full suite 253 passed.
- [x] `FitConfig.astrom_leverage_cap` (2026-08-12, default `0.9`).
  Anchor leverage in the shift block goes as flux squared
  (`I_i = a_i^2 <Gx,w,Gx>`), so one bright source can carry a scene's
  astrometry -- and when that source is extended with an asymmetric colour
  gradient, its residual dipole is formally a shift and drags the field.
  The cap scales anchors above the quantile by `I_cap / I_i` in AB/BB/bB,
  which bounds influence while leaving the shift the anchor measures
  unchanged; the flux block is untouched so photometry does not move.
  `I_i` does not depend on the residual, so it is applied during assembly
  with no extra pass. What it cannot do -- identify *which* anchor is
  wrong, or help when the offender is the only bright member -- is the
  cross-anchor IRLS option now recorded in TODO.md.
- [x] Relentless cross-source code and documentation review completed
  (2026-08-12). `docs/CODE_REVIEW_2026-08-12.md` records the audit of package
  source, tests, public docs, deployed Read the Docs, code comments, and every
  top-level `scratch/wren` TeX/PDF report. It separates 23 release-blocking P1
  findings, 40 conditional/API P2 findings, conceptual risks, document drift,
  positive checks, and a gated remediation/acceptance plan. Validation included
  the full suite (237 passed; one test-only diagnostic-path failure), focused
  suites, a strict Sphinx build, package/dependency checks, PDF renders, and
  numerical reproductions of the highest-impact solver, background, WCS,
  config, repair, PSF, and mock-verification defects. This turn changed no
  package behavior; existing concurrent verification work was preserved.
- [x] Verification v7 (2026-08-12): v6's fits and IDL leg with a reworked
  mock leg. **The `psf_wings` extended-source deficit is 4-5%, not the
  2-3% carried since v2**: recovery 0.9533/0.9573/0.9606/0.9637 across the
  four bands against v6's 0.9669/0.9694/0.9715/0.9740. The F770W
  diagnostic splits the budget cleanly — point sources fit a pull of
  mu=+0.30, sigma=0.91 (unbiased, error model calibrated) while the full
  SNR>20 sample fits mu=-3.35, sigma=2.81, so the whole bias lives in the
  extended sources; the size panel shows it reaching -0.07 by sigma 4-6
  pixels. SNR>20 median 0.9553 +/- 0.0022 (n=392), a ~20-sigma effect.
  Still a floor: sources are pure Gaussians (see TODO). Two changes:
  * Source sizes sigma **log-uniform over 1-10 pixels** on the 40 mas grid
    (0.04-0.40" sigma, 0.09-0.94" FWHM) instead of 1-5, reaching resolved
    galaxies while staying weighted to the small sizes that dominate a real
    catalogue. Painting stamps stay at the builder default (F444W 4",
    F770W 8"): a 0.4" sigma source is inside a 4" box to better than 1e-4,
    so no truncation term enters. Past ~12 pixels it would, and the new
    `psf_size_arcsec` argument on `build_realistic_two_detector_mock` is
    how to raise it.
  * `mock_dilate_segmap` default 2 -> 0 (see below).
  `runs/` seeded with copies of v6's four configs and fit tables (3.1 MB);
  the fits are untouched by any mock-side change, so the IDL leg is a
  rerun of v6's and only the mock leg differs.
- [x] Correction nomenclature fixed across doc and code (2026-08-12). One
  rule: **a name may carry "tot" only if it includes the encircled-energy
  term** — a correction that stops at the edge of the model's own finite
  support is not a total, whatever the code calls it. Three codebases had
  used `totcor` for two different quantities, which is the origin of most
  of their apparent disagreement.
  * `psfcor` = ap_hi/ap_lo, the shape/resolution correction (IDL `apcor1`).
  * `stampcor` = 1/ap_lo, aperture to the support total, **no EE**. Renamed
    from `tot_stamp_<i>`, which broke the rule. IDL's released `totcor<f>`
    is this quantity and is likewise misnamed.
  * `totcor` = 1/(ap_lo*ee_psf_lo) keeps the name because it does include
    the EE.
  Estimators are now written factored as `psfcor * totcor_cat` rather than
  with a bare `totcor`, since the factors state the convention and the bare
  name does not. `flux_estimator_comparison.tex` gained a "Naming" section
  and Estimator 1 is written both ways — `aper(_phot,Rphi)*stampcor` and
  `[aper(_model - _model_nn,Rphi) + aper(_res,Rphi)]*stampcor` — making
  plain that `_phot` already contains the residual and that what separates
  Estimator 1 from 2/3 is that it *scales* the residual. Readers accept the
  old `tot_stamp_<i>` so v6/v7 tables still load; PDF recompiles clean.
- [x] IDL comparison reworked to be like-for-like (2026-08-12),
  `scratch/wren/make_compare_idl_python.py`. Estimator 1 now carries **no
  encircled-energy term on either side** — IDL's `flux_F` applies
  `totcor = 1/ap_lo` and has none, so the python side uses
  `ap_flux * tot_stamp` (derived in `matched()`) instead of `ap_flux_corr`,
  which also divides by `ee_psf_lo`. The comparison flips sign: python read
  0.012 mag brighter than IDL, now 0.020 fainter, so the old offset was the
  convention rather than the code. Also: magnitude panels are drawn as
  `IDL - python` vs IDL magnitude over (-1, 1) (the recorded JSON keeps the
  `py - IDL` sign every earlier version used); every quoted statistic is
  SNR>20 alone, replacing the mag<24 cut that had been producing a spurious
  F1800W offset; the PSF support of both codes is printed on every figure;
  and a new panel (e) compares `psfcor = ap_hi/ap_lo` directly. v7's
  `uds_monu/` figures and JSON were regenerated on this basis and supersede
  v6's IDL numbers.
- [x] Mock segmaps no longer dilated twice (2026-08-12).
  `remap_detection_to_truth` builds its segmap with `Catalog.from_fits`,
  which already applies `Catalog`'s own `dilate_segmap` disk(2) — the same
  step a production catalog run gets — and then dilated a second time with
  `ndilate=mock_dilate_segmap=2`. Segments came out about twice the
  production area (a 5x5 source grows 25 -> 69 -> 129 px, equivalent
  radius 2.8 -> 4.7 -> 6.4 px), letting templates take more of each source
  from the data than a real run would and flattering the extended-source
  recovery. Default is now 0: the catalog step's dilation stands alone and
  nothing dilates again inside the mophongo run (`template_dilate_segmap`
  was already 0, matching `FitConfig`).
- [x] Both verification compare legs are mandatory (2026-08-12). The
  driver honored `fits idl` literally, so v5 and v6 shipped with a
  `uds_monu/` and no `uds_sims/`. Step arguments now select only whether
  the expensive `fits` stage re-runs; `idl` and `mock` always both run,
  and the driver exits non-zero if either leg returns nothing.
  `--psf-dir` now also reaches the mock leg (it read a hardcoded
  `data/PSF`, so v6's mock would have verified 4" support while its fits
  leg used 8"). Confirmed by the F444W box EE moving 0.96317 -> 0.98522
  while the MIRI band held at 0.96864 — the MIRI grids are byte-identical
  between `data/PSF` and `data/PSF8`.
- [x] `data/PSF8` gained the F444W `OS4_GRID1` pair (2026-08-12). The 8"
  set had only GRID25 and the 30" halo layouts, so the mock leg's default
  `UDS_NRC.._F444W_OS4_GRID1` pattern matched nothing and every band died
  with a `no stpsf grid loaded` KeyError. Built at 8" FOV, same epoch
  (MJD 59967.188) and OPD as the 4" originals, 508x508. Rebuild script:
  `scratch/build_psf8_f444w_grid1.py` (`data/` is gitignored).
- [x] Flux-recovery diagnostic extended (2026-08-12), in
  `verification.save_flux_recovery_plot`: running median of the unblended
  sources and a `SNR>20 median +/- MAD-based standard error` band in the
  flux-ratio panel; a fifth panel of fractional residual vs injected
  source size for SNR>20, which is what makes the extended-source shape
  term directly readable; and a green point-source histogram (any SNR,
  any size) in the residual-pull panel, separating the error model from
  the shape term. Equal-count binning is shared through
  `verification._running_median`. Recovered flux stays `flux_<i>_total`
  = fitted amplitude / `ee_psf_lo` (pipeline.py:2497).
- [x] Verification v6 (2026-08-12): the first fully like-for-like IDL
  comparison — 8" PSF support (matching monu's measured 7.8"; parity
  check passes), saturated-star repair active (30" halo grids, flagged +
  isolated scenes), IDL-parity columns (`tot_stamp` = 1/ap_lo, `psfcor`
  = ap_hi/ap_lo, EE-inclusive `totcor`), r<1.5'. Agreement: tot_stamp/
  IDL-totcor 0.987/0.991/0.975/0.955 across the four bands, psfcor
  1.240 vs 1.261, est1 -0.01..-0.03 mag at SNR>25 — percent-level
  reconciliation, with the est1 offset being exactly the beyond-8" EE
  our totcor includes and IDL's does not. Session infrastructure landed
  along the way: shared `repair_cache_path` (band 1 fits, bands 2..N
  reload), global provenance-guarded PSF/kernel cache seeding in the
  driver, single-print logging (module handlers removed), HIERARCH
  VerifyWarning silenced in log_run, weight-calibration log names its
  band and verdict, template panel keeps saturated stars visible.
  `examples/minerva/verification/v6/` has README + figures + json +
  logs. The mock leg was backfilled on 2026-08-12 (it had been skipped):
  recovery 0.9669/0.9694/0.9715/0.9740 across the four bands, 0.7-0.9%
  below v4's 4"-detection-PSF numbers because the `psf_wings` composite
  is built on the F444W stamp, so more wing flux inside the template
  normalization means a smaller fitted amplitude against unchanged truth.
- [x] Astrometry convergence: tolerance 0.1, honest verdicts (2026-08-12).
  Three changes on top of the per-scene freeze loop:
  * `astrom_shift_tol` default 0.05 -> 0.1 fit-grid pixels. The MINERVA logs
    show the increment falling ~10x per pass, so 0.05 buys one extra pass
    and no precision: on a synthetic field with an injected (1.5, -0.8) px
    offset, 0.05 converges in 4 passes and 0.1 in 3, both recovering
    (1.487, -0.791). 0.1 px also sits just above the ~0.08 px statistical
    floor of the weakest scene the anchor cuts admit (5 anchors at
    `astrom_minimum_snr` = 15) and well below PSF-matching centroid
    systematics, which are a bias no tolerance iterates away.
  * `flag_astrom` = 0 now means solved-and-converged, not never-moved. A
    flux-only run, and scenes with too few bright anchors to carry a shift
    block, never move and previously came out flagged converged; they now
    keep `astrom_converged = None` and flag -1, as `docs/outputs.md` already
    specified. Test
    `test_astrometry_verdict_is_none_where_no_shift_was_fitted`.
  * The tolerance is logged in mas as well as pixels: on the upsample path
    the fit grid is the hi-res grid, not the grid of the band being fitted,
    so "0.1 pix" alone is ambiguous (4 mas at 40 mas/pix hi-res, not at the
    80 mas MIRI scale).
  Also confirmed by test that the increment the loop stops on has already
  been applied — `Scene.solve(apply_shifts=True)` applies before the caller
  measures `Template.shifted`
  (`test_final_sub_tolerance_shift_is_applied_to_the_templates`). Increments
  below 0.01 px in both axes are never applied at all, which is the floor
  the regenerated `docs/images/shift_iteration_damping.png` now shows,
  alongside the flux error already sitting at its floor by pass two.
  Full suite 238 passed.
- [x] Saturated stars: held out of scene building, core-named groups,
  spike flagging (2026-08-12). Three defects found from the uds_f770w v5
  scene plots:
  1. Saturated templates were part of the coupling graph and merge, then
     relabelled afterwards, so a star's wings glued its neighbours into
     one scene that then lost the member that shaped it.
     `generate_scenes` now partitions the non-saturated templates on
     their own (submatrix of ATA/ATb) and adds the saturated scenes
     after.
  2. The group id was the *lowest* flagged segment id. On UDS star
     1017146 that named a spike fragment 291 px (11.6") from the centre,
     whose catalog row has `wht_lo = 0` and was cut by
     `footprint_filter`; the filled core carried that label, so nothing
     modelled the core (the dipole in scene 60's residual) and a fragment
     absorbed by the fill lost its segment entirely. The group id is now
     the flagged segment reaching closest to the fitted centre (lowest id
     among equals, within 1 px + 5 %).
  3. The 30 % ratio test misses bright diffraction spikes: a saturated
     star's real spikes run far above its ePSF (spike segments at
     frac 0.03-0.29 where flagged segments sit at 0.85-1.30; smoothing
     the model does not recover them, so it is not misregistration).
     Added `halo_nsigma` (default 5): a segment is also flagged when the
     model's mean surface brightness over it exceeds
     `halo_nsigma x sky_noise` per pixel. UDS: 555 -> 568 segments
     (128 stars), the 13 new ones at median frac 0.12 / halo 13.5 sigma.
     The flag log gained `npix` and `halo_sig`.
  Also: the repair diagnostic's residual panel is back to its own linear
  MAD stretch (the data's log stretch made a few-percent residual a flat
  gray field), and `flag_astrom_<i>` is int16 -- astropy writes an int8
  column as a FITS logical and -1 came back as True.
- [x] CANFAR staged configs carry the repair settings (2026-08-12).
  All 36 `examples/canfar/*_canfar.json` (17 patch + 17 `_full` + 2
  `_test`) regenerated with `arcify.py` from the updated MINERVA
  sources, so they now carry explicit `wht_hi` (the bkgsub `sci_hi`
  breaks the `_sci`→`_wht` guess and `resolve_wht_hi` would raise),
  `repair_saturated: true` and `repair_kwargs: {min_buffer_snr: 200}`.
  No explicit `repair_psf_pattern`: the halo pattern is derived from
  `pattern_hi`, so the canonical `_FOV30_GRID1_OS4` spelling follows the
  code. `submit.py push` now also ships those halo grids
  (`*_NRC*_F444W_MJD*_FOV30_GRID1_OS4.fits`), and
  `campaign.has_shared_grids` counts them when `repair_saturated` is set,
  so a field without them (COSMOS, EGS) still runs one band alone first
  rather than racing several node-side builds into one `psf_dir`.
- [x] Region-wise image convolution (2026-08-12).
  `PSFRegionMap.convolve_image(image, wcs, buffer=None, fill_value=0.0)`
  applies a kernel map to a whole mosaic: each region is cut with a
  convolution border (default half the largest stamp), convolved with its
  own stamp, and only the pixels inside the region polygon are written
  back, so cutouts overlap but kept pixels never do. Polygons are mapped
  into pixel space once (`shapely.ops.transform` + `contains_xy`) rather
  than running the WCS over every pixel. Levels stay separate: the method is
  array-in/array-out, and module-level `convolve_fits(sci, region_map,
  out_path)` does the file work, taking the map as an object or as the
  GeoJSON a run left behind and writing the input header plus
  CONVMAP/CONVNREG. Fixed `from_geojson` along the way: it called
  `.replace()` on its argument, so a `Path` hit `Path.replace` (rename) with
  a TypeError instead of finding the stamp sidecar.
  Lives in `psf_map.py` because that module owns both the geometry and
  the stamps; it borrows only `utils.fftconvolve`. Validated on the real
  uds_f770w kernel map (2911 regions): a 2000^2 cutout convolves in 1.3 s,
  89% covered (the rest outside the MIRI footprint), and inside a region
  it matches a plain single-kernel convolution to 6e-10. Documented in
  docs/psf_maps.md; tests in tests/test_psf_map_convolve.py.
- [x] Per-scene astrometric convergence (2026-08-12). The refinement
  loop in `Pipeline.run` iterated *every* scene until the global maximum
  increment fell under `astrom_shift_tol`, so converged scenes were
  re-solved for as long as the slowest one kept moving. Scenes are
  independent (`Scene.solve` reads only its own templates/image/weights),
  so each now drops out of the pass list as it converges. `Scene` gained
  `astrom_step`, `astrom_niter`, `astrom_converged`; scenes still moving
  when the budget runs out are logged as a warning with the five worst.
  Every source inherits its scene's verdict as `flag_astrom_<i>` in the fit
  table (0 converged / 1 still moving / -1 no template), and the scene
  catalog gained `astrom_niter`, `astrom_step`, `flag_astrom`.
- [x] Run-path `print` calls converted to `logger.info` (2026-08-12), so
  the memory/config/upsampling/kernel lines carry the timestamp and level
  of the rest of the log instead of arriving bare on stdout
  (`pipeline.py` × 12, `templates.py` template pruning). The interactive
  `summary()` display print stays a print.
- [x] Interactive single-star repair + diagnostic stretch (2026-08-12).
  New `repair.repair_star(sci, wht, ra=/dec= or x=/y=, ...)`: cuts a box
  around the coordinate, finds the nearest interior wht=0 hole (clear
  error beyond `search_radius`), runs the donut fit + core fill, and
  returns `{fit, diagnostic, fig, sci, wht, slices}` — the ten-panel
  diagnostic displays live in a notebook or saves via `to_file`.
  Verified on UDS star 15871 (A=7.97e5, shift +2.9 px, matching the
  full-mosaic run). The diagnostic's residual panels (shifted and
  no-shift) now use the same 2-dex log stretch as the data / A·psi
  panels (negatives clip to sky tone) instead of their own MAD
  grayscale. Documented in docs/repair.md ("Repairing a single star
  interactively") and demoed in the example notebook (section 3).
  Cross-session hardening: the halo-grid loader falls back to
  pattern_hi with a warning on ValueError too (unparseable/legacy
  pattern spellings no longer crash load_data — hit by the verification
  session that copied configs carrying the interim explicit
  `repair_psf_pattern`).
- [x] Halo grids: canonical naming, autobuild, hybrid flag model
  (2026-08-12). The `UDS_*_OS4_GRID5` files were a misnomer (num_psfs=1,
  fov 30") — renamed on disk to
  `UDS_{det}_F444W_MJD59967_FOV30_GRID1_OS4.fits`; `PSFFactory.filename`
  gained an `include_fov` `_FOV{int}` token (parsed back by
  `_psf_factory_kwargs`/`_PSF_PATTERN_RE`) so large-FOV grids can't
  collide with the standard 4" GRID1 epoch files. stpsf has no
  "automatic maximum" FOV (backend default is 5") — 30" is explicit.
  The pipeline now auto-derives the halo pattern from `pattern_hi`
  (`_repair_halo_pattern`) and builds missing grids by default through
  the existing `psf_autobuild` path (one-off ~minutes per detector,
  cached; the log says so), falling back to `pattern_hi` with a warning
  only when autobuild is off. The flag model is now a hybrid
  (`repair.hybrid_psf_stamp`, via `jwst_psf.blend_psf`): the MJD-matched
  `pattern_hi` PSF verbatim inside its support, the 30" halo model
  grafted outside (rescaled over the seam annulus), unit-sum — and the
  star's fitted centre incl. sub-pixel shift positions the whole model.
  All GRID5 references updated (configs regenerated without an explicit
  `repair_psf_pattern`, notebook, docs, SATURATE.md, v1 README). Tests:
  FOV-token round-trip, halo-pattern derivation, hybrid-stamp shape and
  core-fidelity, hybrid trigger in `repair_in_memory`.
- [x] Scene catalog carries the total shift (2026-08-12).
  `<name>_scene_catalog.csv` gained `dx`, `dy`: the accumulated astrometric
  shift at the scene center in reference-grid pixels, NaN where the scene
  solved no astrometry. Same quantity as the per-template table's `dx`, `dy`
  (`Template.shifted`), evaluated through the refit total field, so it is
  not the last iteration's increment — `Scene.shifts` was and stays an
  in-memory attribute that is never written. Test
  `test_scene_catalog_carries_total_shift` asserts the value matches the
  applied shifts and differs from the last increment.
- [x] Shift field is a standard output (2026-08-12).
  `write_outputs` now writes `<name>_shift_field.png` whenever at least one
  scene solved for astrometry, from the new `Pipeline.plot_shift_field()`.
  Each solved scene contributes `2**order` arrows sampled over its own
  extent (order 0: one at the scene centre, order 1: two along the longer
  axis, order 2: 2x2), drawn from the template position toward where the
  source is measured in the fitted band, with the scene id in light gray
  next to the first arrow. Positions and arrows are in RA/Dec degrees at
  aspect `1/cos(dec)` with RA increasing left; arrows carry a common
  magnification set from the 90th percentile of their length, and the
  legend arrow gives that length in pixels and arcsec.
  `Scene.shifts` holds only the last astrometric iteration, so
  `Pipeline._scene_shift_samples` refits the same-order Chebyshev field to
  the accumulated `Template.shifted` values (the total applied offsets)
  before sampling. Order is recovered per scene by inverting `n_terms` on
  the coefficient count, so a saturated-star scene (forced to order 0)
  plots correctly. Four tests in `test_pipeline.py`: sample counts per
  order, sampled field consistent with the applied shifts, arrow sign
  against an injected image offset, and the `write_outputs` product with
  its label and arrow counts. Full suite 221 passed.
- [x] Large-FOV flag model for the in-pipeline repair (2026-08-12).
  `RunConfig.repair_psf_pattern` loads a second STDPSF set (the 30"
  `..._OS4_GRID5` grids) into a dedicated DrizzlePSF sharing the hi
  band's exposure info; `repair_in_memory` gained `stamp_dpsf` /
  `stamp_pattern` so the flag model (halo + spikes) comes from it while
  the core fit keeps the MJD-matched `pattern_hi` — the same two-PSF
  split as the original standalone repair flow
  (`scratch/run_saturate.py`). Default stamp size = the stamp PSF's
  native FOV. `make_minerva_configs.py` emits the pattern when the
  large grids are staged (UDS yes, COSMOS/EGS not yet). Clarified:
  `RunConfig.psf_size` never clipped the repair (it only trims the
  photometry region-map stamps — the repair drizzles onto its own
  cutouts at full native ePSF support), and `EffectivePSF.extended_epsf`
  is vestigial (no loader fills it; `get_extended=True` is a no-op).
- [x] Concept-first docs rewrite + executed snippets (2026-08-12). The
  eight component pages now read as functionality overviews: per-parameter
  listings deleted from the pages and their content MOVED into the source
  docstrings (psf, psf_factory, psf_map, utils, templates, template_schemes,
  catalog, scene, scene_fitter, saturate, astrometry, verification,
  mock_mosaic, pipeline), so the autodoc API reference carries the detail;
  conf.py enables `ignore-module-all` and `napoleon_use_ivar` so the
  enriched docstrings all surface. Reference-style content kept on pages:
  RunConfig/FitConfig tables, output glossaries, flag tables, the
  extend_mode scheme table, the external-catalog contract. One wrong source
  docstring fixed en route (`get_bg_and_ivar` claimed four return values;
  it returns two). Then 21 small runnable snippets added across the pages,
  each executed verbatim before embedding with observed outputs as inline
  comments (deterministic seeds); an independent sweep re-extracted all 54
  python fences, classified 26 as self-contained, and re-ran them: 26/26
  pass, printed values match the inline comments (including the end-to-end
  mock + wiener-map + pipeline scenario snippet on simulation.md). Full
  pytest suite 216 passed; build has zero page-level warnings; examples
  3/3; sweep clean. Docs style preference recorded in agent memory:
  concept-first pages, details in docstrings/API.
- [x] Rigid shift fit for saturated-star scenes (2026-08-12). A
  saturated-star scene holds fragments of one star, which would fail
  every astrometry anchor cut (isolation against each other, star
  exclusion, sometimes minimum members) and so was solved flux-only —
  leaving a centroid-offset dipole in the residual. `Scene.solve` now
  detects an all-saturated scene, includes every member as an astrometry
  anchor, and forces the shift basis to order 0: one rigid (dx, dy) for
  the whole star. Verified on the UDS F770W satdemo patch: star 15871's
  five fragments share dx=+1.47, dy=+0.86 px and the residual dipole
  shrinks; synthetic regression test
  (`test_saturated_scene_fits_rigid_shift`) recovers an injected rigid
  shift under exclude-stars + isolation cuts. Note: per-fragment fluxes
  within a star group are degenerate (overlapping templates of one
  source) — the star's flux is the group sum.
- [x] Repair setting in the MINERVA run database + real-data validation
  (2026-08-12). `make_minerva_configs.py` now emits
  `"repair_saturated": true` + `repair_kwargs` (min_buffer_snr 200) and
  an explicit `wht_hi` (the bkgsub sci_hi breaks the `_sci`->`_wht`
  guess; found by the validation run) in every generated config — all 17
  field/band configs regenerated. Validated end-to-end on real UDS
  F770W data (`scratch/satdemo/`, 1' patch on saturated star 15871):
  in-memory repair at load (128 stars repaired/flagged on the full
  mosaic, diagnostics in `out_dir/repaired/`), star's 5 fragments =
  exactly one scene (scene 13, sole members, fitted flux 2.75e4 +- 80),
  per-star before/after comparison written, and the star's segments
  nulled in the neighbouring scenes' diagnostics while shown in its own.
- [x] In-memory saturation repair in the pipeline + per-star scenes
  (2026-08-12). `RunConfig.repair_saturated` (+ `repair_kwargs`) runs the
  saturation repair on the loaded `sci_hi`/`wht_hi`/catalog/segmap at
  `load_data` time via the new `repair.repair_in_memory` — mosaics on
  disk stay untouched, diagnostics go to `out_dir/repaired/`, repaired
  templates land in `*_stamps.fits` as usual, and the repaired wht
  (cores restored to the median donut weight) feeds the detection ivar.
  In-memory flagging keeps the wing segment labels
  (`zero_segments=False`) so every flagged row still gets a template;
  `Template.sat_group` (in `_META_ATTRS`) carries the star's group id
  from `FLAG_SATURATED_*`, and `generate_scenes` now groups saturated
  templates by it: one scene per saturated star (fragments fit jointly),
  own scene per template for legacy 0/1 flags, still created after
  `merge_small_scenes` so exempt from `scene_minimum_anchors`.
  `Scene.plot` gained `null_segments`: the pipeline nulls the saturated
  stars' segments in every other scene's diagnostic (they'd dominate the
  stretch) while their own scene shows them. Tests: `test_repair.py`
  `repair_in_memory` (wht filled, labels kept, grouping),
  `test_scene_saturated.py` group-scene/legacy/plot-null cases; docs
  `repair.md` "Running inside the photometry pipeline" section.
- [x] Docs feedback round (2026-08-12). (1) Parameter blocks now render as
  real definition lists — MyST's `deflist` extension was never enabled, so
  every `param:` block had rendered as plain paragraphs — styled by
  `docs/_static/custom.css` as quiet astropy-style panels (theme-aware,
  scoped to hand-written pages). (2) Component pages trimmed: driver and
  user-facing entry points keep full parameter documentation, internal
  helpers collapsed to one-line summaries with API links (psf, psf_maps,
  templates, catalog, fitting, preprocessing, simulation, diagnostics).
  (3) New/fixed figures: EE growth curve in the drizzled-stamps section;
  region montage regenerated with F444W roll diversity (106-133 deg, new
  generator scratch/wren/make_psf_report_figs_rolls.py); composite-anatomy
  figure rebuilt with all segments nulled in the halo panel and a
  full-coverage object (scratch/wren/tmplfig/mk_idl_figs_docfix.py);
  generated scene-partition illustration (scratch/docfigs/
  scene_partition_fig.py); saturation-repair diagnostic embedded in
  preprocessing.md. (4) catalog.md documents the external-catalog column
  contract (required id/x/y 0-indexed; optional deblend/saturation/
  flag_star/ra/dec/aperture columns), verified against _fit_catalog.
  (5) fitting.md states SparseFitter is not in the production path (scene
  solver does all fitting); the dead import in Pipeline.run removed.
  (6) Real bug found by the figure generation and fixed:
  `Templates.convolve_templates(inplace=True)` discarded the convolved
  result (kept unconvolved originals) — the path `from_image` uses;
  production (`inplace=False`) unaffected. Fixed + regression test;
  stale docs caveat updated. Examples 3/3, build and sweep clean.
- [x] Verification v4 (2026-08-12): `psf_wings` from main with the
  aperture estimator corrected by the recorded box EE
  (`ap_flux_total_<i> = ap_flux_corr_<i>/ee_psf_lo`; est1 panel reads it).
  Against IDL the offset flips from +0.04..+0.06 to -0.03..-0.07 mag
  (SNR>25, four bands). Settled from the recorded formalism: IDL `totcor
  = 1/ap_lo` on the unit-normalized model — NOT ee-corrected — so
  `ap_flux_corr` is the like-for-like column vs `flux_F` (its +0.05
  offset is a real model-EE difference in the composites' wings) and
  `ap_flux_total` is the truth-convention total, brighter than IDL by
  construction. Default scheme stays `psf_wings` (user decision).
  Mock leg reproduces v2 to four decimals (determinism check). Scene
  reporting now post-merge only (one INFO line; pre-merge component count
  demoted to DEBUG) after the 4867-scenes misreading. README + figures +
  json under `examples/minerva/verification/v4/`. Follow-up in TODO:
  carry `ap_flux_total` into the mock recovery table for a direct
  truth check of the aperture estimator.
- [x] Band-independent flag column + core segment identity (2026-08-12).
  The saturation flag column is now `FLAG_SATURATED_TMPL` by default
  (TMPL = template band; `filter_name` still overrides) so downstream
  code does not change when the detection band does — the CLI passes
  TMPL unconditionally and no longer requires `--filter` for the catalog
  step. The core fill in `flag_saturated_segments` was extended: besides
  the enclosed seg=0 region (now bounded by `r_out` against unrelated
  sky pockets), all repaired pixels within the fit radius `r_in` get the
  group id — previously core pixels belonging to a zeroed wing segment
  were left at 0, leaving a gap in the star's segment. The group-id row
  therefore keeps a segment covering the PSF-repaired core and a
  mophongo run on the repaired image models the star as a normal source
  (verified on UDS: 128/128 star centres carry their group id). The
  5-panel diagnostic's zeroed panel now masks the filled core too (the
  core belongs to the flagged star). Autobuilt STDPSFs now use
  `date_mode="cluster"` — one grid per observation epoch over the full
  data distribution, nearest-MJD per exposure — matching the pipeline's
  PSF policy instead of a single modal epoch.
- [x] Saturation-flag refinements from the UDS F444W inspection
  (2026-08-12). Four changes to `catalog.flag_saturated_segments` after
  reviewing real bright stars: (1) flux comparison restricted to the
  model support (ePSFs are zero beyond their native FOV — spike segments
  outside a small stamp could never flag); (2) model-flux noise floor
  `min_snr x sky_noise x sqrt(n_pix)` (default 5 sigma) — the large-FOV
  model otherwise flags every noise-level segment on a near-zero obs
  denominator (7628 -> 549 on UDS); (3) flag value is now the star's
  group id = lowest flagged segment id (membership encoding; `>0` is the
  boolean cut; contested segments join the star with larger model flux);
  (4) the undetected saturated core (seg=0 enclosed by the star) is set
  to the group id in the output segmap so the group-id row keeps a
  segment. Diagnostic now 5 panels: segmap before | sci before +
  to-flag | sci repaired + flagged | segmap after with star in one
  color | sci with flagged zeroed. UDS flagging rerun with the 30"
  GRID5 ePSF (npix=751): 549 segments / 128 stars, bright-star spikes
  now fully captured. Tests grown to 14 (group ids, core fill,
  truncated-stamp support, noise floor); notebook + docs/repair.md +
  `UDS/repair/v1/README.md` updated.
- [x] Verification v2 (psf_wings) + v3 (wren) on the merged code
  (2026-08-11/12), consolidated under `examples/minerva/verification/v2,v3`
  (per-version README, `runs/`, `uds_monu/` IDL leg with figures+json+log,
  `uds_sims/` mock leg with figures+json+log; driver
  `examples/minerva/run_verification_v2.py --version --scheme`, heavy
  products gitignored). Headline numbers (F770W mock, recovered/true):
  conv-fill 0.9998, psf_wings 0.9755 (points 1.008 / extended 0.971), wren
  0.9635 (points 0.968 / extended 0.972). Attribution settled in three
  steps: (1) the psf_wings extended-source deficit is wing *shape* (PSF
  wings cannot carry extended profiles; IDL classic shares the mechanism);
  (2) the apparent ~5% est1 offset against IDL is NOT hi-PSF truncation
  (8" F444W grids measure stamp sums 0.950 — the far field is stpsf
  throughput, not box truncation) but the aperture estimator never applying
  the recorded `ee_psf_lo` — IDL's totcor includes it (IDL 1.450 vs
  ap_corr 1.358 vs ap_corr/ee 1.480, within 2%); fixed by the new
  `ap_flux_total_<i>` column; (3) wren adds a ~3% point-source deficit
  (clipped noisy-data wings) and does not cure the extended one. Decision
  (TODO): revert default extend_mode to `psf_convolution`, optionally add
  an aperture-floor trusted-data variant. `ee_psf_lo` chain live end to
  end: 0 filter-mean fallbacks in every v2/v3 band.
- [x] UDS F444W repair run + example notebook (2026-08-11). Full-mosaic
  run of `mophongo-repair` on the MINERVA UDS F444W DR0.1 inputs
  (n3.0 mosaic, n3.0_v1.2 segmap, n3.0_m3.1_v1.2.1 SUPER catalog wMIRI):
  22455 interior holes, 129 saturated stars repaired (200-sig buffer
  filter), 292 catalog segments flagged FLAG_SATURATED_F444W. Outputs +
  README in `MINERVA/data/UDS/repair/v1/`. New concise walkthrough
  `examples/repair_uds_f444w.{py,ipynb}` (jupytext percent + executed
  notebook): one call per step, inspects per-star PNGs, flag log, and the
  default flag diagnostic — supersedes the long-form
  `examples/repair_saturate.*` for the repair/flag workflow. Fixes from
  the run: `repair_image` also writes the fit table as FITS
  (`_saturate_<mode>.fits`, types preserved; CSV bools read back as
  strings — `catalog._bool_column` now coerces `ok`/`flagged` wherever
  consumed), saturate status strings are ASCII (FITS-safe), flag
  diagnostic subsamples the display stretch on large mosaics.
- [x] Post-hoc saturated-segment flagging (2026-08-11). New
  `catalog.flag_saturated_segments`: non-destructive counterpart to
  `repair_saturated_catalog` for catalogs built before the repair (e.g.
  the MINERVA SUPER catalogs) — a segment is flagged
  `FLAG_SATURATED_<FILTER>=1` when the repair's best-fit star model
  contributes more than `flux_frac` (default 0.3) of its observed flux;
  rows are kept (order/matching preserved) and flagged labels are zeroed
  in the output segmap. `mophongo-repair --catalog/--segmap` now defaults
  to this flag mode (`--flux-frac`; `--merge` restores the destructive
  parent-merge), writes `<catalog>_flagged.fits` /
  `<segmap>_flagged.fits` / `<catalog>_flaglog.csv`, and by default a
  per-star diagnostic (`verification.plot_saturated_flag_diagnostic`,
  4 panels: segmap before | sci before | sci repaired + flagged overlay |
  segmap after with flagged=0). `drizzled_psf_stamp` is now public.
  Tests: `tests/test_repair.py` grown to 12 (flag mode, neighbour kept,
  merge mode, both CLI paths); docs `repair.md` updated.
- [x] Concept figures in the docs (2026-08-11). Sixteen PNGs under
  `docs/images/`, drawn from the wren report material (tmplfig, fitfig
  PDFs converted, ee_report, psf_check_figs, mock verification) and
  embedded across eight pages: scene fit example (overview, diagnostics),
  template composite anatomy + classic build steps (templates), shift
  linearization anatomy + damped-iteration convergence (fitting), kernel
  regularization diagnostic + star-vs-model blur + MIRI growth curves
  (psf), region tiling + per-region PSF/kernel montage (psf_maps), flux
  scale flow + stamp encircled energy (pipeline), mock flux recovery +
  mock kernel scan (simulation), subphot six-panel (diagnostics). Every
  candidate was viewed before selection; figures with internal tags
  ('ivo'/'wren'/'ifl' labels, run filenames) were rejected, filenames with
  survey ids renamed on copy. An independent verifier re-viewed all 16
  embedded figures: captions accurate, all under 400 KB, no internal text;
  build clean, sweep clean.
- [x] Scene diagnostic PNGs moved to `out_dir/scenes/` (2026-08-11).
  `write_outputs` wrote one `<name>_scene_<id>.png` per scene straight into
  `out_dir`, which for a full field is a few hundred files drowning the real
  products. They now go to a `scenes/` subdirectory, created only when
  `scene_plots` is set and there are scenes to plot; the scene catalog CSV
  stays in `out_dir`. Docs updated (`docs/outputs.md`, `alma.md`,
  `examples/canfar/MANUAL.md`); regression test
  `test_pipeline.py::test_write_outputs_puts_scene_plots_in_scenes_subdir`
  checks both the new location and that no `scenes/` dir appears when the
  plots are off. `tests/test_pipeline.py` + `test_pipeline_config.py`:
  29 passed.
- [x] Naming/metadata/guard hardening (2026-08-11), follow-ups to the EE fix:
  * One name for the scheme selector: `extend_mode` everywhere.
    `Pipeline(extend_mode=...)` and `run(extend_mode=...)` are canonical;
    `extend_templates=` still accepted as a deprecated alias (logs a
    warning, error if both given); the constructor override is stored as
    `extend_mode_override`, the resolved scheme as `extend_mode`.
    `verification.py` switched to the new kwarg.
  * `Template._META_ATTRS` + `copy_meta_to()`: the full metadata contract
    (ids, flags incl. star/deblend/saturated bits, EE corrections,
    fit state, extension_* provenance, shift vectors). `downsample`,
    `project_to_block_replicated_grid` and `convolve_cutout` now copy all
    of it instead of hand-picked subsets — the mechanism behind the
    ee_psf_lo audit bug. Regression test loops over `_META_ATTRS`.
  * NaN guards with clear WARNINGs at the EE choke points: invalid filter
    throughput (applies no correction), per-source `ee_psf_lo` fallback
    count > 0, and invalid PSF stamp sums in `_filter_psf_throughput`.
  Full suite 205 passed.
- [x] `ee_psf_lo` now survives resampling (2026-08-11): the audit's headline
  gap — `project_to_block_replicated_grid` and `Template.downsample` dropped
  `ee_psf_lo`/`ee_tmpl`, so every `k>1` run (all MINERVA bands) silently fell
  back to the filter-mean EE — is closed. Both methods now propagate
  `ee_psf_lo`, `ee_tmpl`, `template_norm`, `id_parent`, `id_scene`, `name`,
  so `flux_<i>_total` divides by the per-source encircled energy on the
  default upsample path. Regression test
  `test_template_convolution.py::test_resampling_preserves_ee_metadata`;
  full suite 205 passed. Real-data validation of the divisor stays open in
  TODO.md. Also confirmed the post-merge default build scheme is
  `psf_wings` (FitConfig.extend_mode, both API and config paths).
- [x] Docs updated to the template merge (2026-08-11, merge `6e6cec6`).
  Three verification agents brought every affected page to merged HEAD:
  pipeline.md's flow now documents the six-scheme template build
  (`FitConfig.extend_mode`, default `"psf_wings"`, legacy `extend_templates`
  as override) with the full new FitConfig block (scheme knobs +
  `astrom_damping` 0.8); templates.md gained the full scheme reference
  (composite formulas, params dataclasses, normalization order, new flag
  bits) and api.rst lists `mophongo.template_schemes`; fitting.md documents
  the damped shift iteration; outputs.md documents `scene_<i>`,
  `<name>_templates.fits`, and the `<name>.json` config snapshot;
  overview/quickstart no longer claim the array path defaults to truncated
  templates (both entry points resolve to `"psf_wings"`; `psfs[0]` is
  effectively required); quickstart's minimal config gained the now-required
  `wht_hi`; repair.md verified line-by-line against `mophongo.repair` and
  committed; psf_maps.md's stale extension paragraph rewritten. Migration
  note added: old configs carrying `extend_templates` fail to load. Stale
  source docstrings fixed (`_ensure_maps`, `repair_saturated_holes`
  pre-filter/repair-footprint, `template_schemes` blank line). Executable
  examples 3/3 pass at merged HEAD; build clean; sweep clean. Follow-ups
  recorded in TODO (headline: `RunConfig.driz_hi` looks accidentally dropped
  by the merge).
- [x] `extend_mode` rename (2026-08-11): the convolution-fill scheme is now
  `'psf_convolution'` (was `'psf'` — easily confused with the scaled-PSF
  `'psf_wings'` scheme; the two produced separately-verified IDL agreement
  and are different algorithms). `'psf'` stays as an alias in
  `EXTEND_MODE_ALIASES` and the legacy constructor map, so existing scripts
  and configs keep working; templates now stamp
  `extension_mode='psf_convolution'`. `_resolve_extend_mode` also applies
  the alias map to config values (previously `extend_mode='default'` via
  config would have raised). Full suite 204 passed.
- [x] Standalone saturation-repair entry point (2026-08-11). New
  `mophongo/repair.py` wraps `saturate.repair_saturated_holes` +
  `catalog.repair_saturated_catalog` into one runnable step for users who
  only want repaired FITS images and a flagged catalog: console script
  `mophongo-repair` / `python -m mophongo.repair sci.fits wht.fits
  [--catalog cat --segmap seg]`. Builds the `DrizzlePSF` from the mosaic
  (`_wcs.csv` auto-reconstructed; STDPSFs loaded from `--psf-dir` or built
  on demand via `PSFFactory.from_csv`), measures the PSF FWHM from a
  drizzled stamp when not given, writes `<sci>_repaired.fits` /
  `<wht>_repaired.fits` (+ `SATREPAI/SATMODE/SATNFIX/SATFILT` provenance
  keywords), the per-hole fit CSV, and optionally the repaired
  catalog/segmap with `FLAG_SATURATED_<FILTER>`. Offline tests in
  `tests/test_repair.py` (synthetic scene + fake ePSF, 9 tests).
  Post-review hardening: `mode` validated in the Python API, fit CSV /
  plot dir are mode-suffixed (`_saturate_repair.csv` vs
  `_saturate_subtract.csv`) so two-pass runs don't overwrite, the CLI
  refuses `--catalog` when no filter can be determined (no fabricated
  flag name), backend probes catch only `ValueError`, and subtract-mode
  output naming/weight semantics documented. New
  readthedocs page `docs/repair.md` (User guide), wired into
  `index.md`/`conf.py`/`api.rst`, cross-linked from `preprocessing.md`.
- [x] Cross-reference follow-ups (2026-08-11). `Pipeline.from_config` now
  accepts a directory holding exactly one `*.json` (ambiguity raises
  FileNotFoundError naming the candidates) and config-driven `run()`
  stamps the executed config to `<out_dir>/<name>.json`, so a finished run
  reopens with `from_config(out_dir).load_fit()` — the wren run-directory
  convention, restored with provenance (test:
  `test_from_config_accepts_directory`). Docs: pipeline.md documents the
  reopen path, the CWD caveat, and that catalog x/y are 0-indexed (1-based
  FITS-convention externals shift by one); psf.md states production keeps
  the default MIRI blur while star-comparison tests disable it;
  diagnostics.md documents plot_subphot's KeyError and the reopen recipe.
  scratch/wren/make_psf_report_figs.py: KERNEL/PIXFRAC reads guarded with
  the pipeline's square/0.75 fallbacks, pixel_ratio derived from the cube
  stamp-size ratio instead of hard-coded 2, deliberate no-blur choice
  documented. astrom_damping NOT documented: `git log main..template`
  shows the template branch is still unmerged and the field is absent at
  main HEAD.
- [x] Wren cross-reference (2026-08-11). Twelve agents (analysis +
  adversarial verification per script) cross-checked `scratch/wren/*.py`
  against main `3d9e7d4` and the docs; full report in
  `scratch/wren/CROSSREF_2026-08-11.md`. Two scripts were broken at main
  and are repaired in place (`make_compare_subphot.py`: `diagnose_subphot`
  renamed to `plot_subphot`, `from_config` now gets the JSON not the run
  directory) and `fit.py`'s docstring/label no longer call the
  template-branch `astrom_damping=0.8` "implemented"/"production default".
  Docs additions driven by the findings: x/y are 0-indexed (outputs.md),
  BUNIT pointer for flux units (outputs.md), scene ids are run-dependent
  partition labels (outputs.md), `DrizzlePSF.footprint`/`driz_footprint`
  attributes (psf.md), `target_label` on both verification entry points and
  the scenario `summary` keys (simulation.md);
  `_add_aperture_photometry`'s docstring now lists the columns actually
  written. Open items (reopen-from-out_dir, CWD-relative config paths,
  `plot_subphot` KeyError, astrom_damping-on-merge) recorded in TODO.md.
- [x] Both verification legs folded into one entry point,
  `scratch/wren/verification.py`: `mock` (injected-truth per-band mocks,
  implementation in `examples/minerva/run_verification.py`) and `idl`
  (subphot comparison, implementation in
  `scratch/wren/make_compare_idl_python.py`), default both, optional band
  list, combined summary in `scratch/wren/verification_summary.json`. Both
  legs re-verified through it; the seeded F770W mock reproduced its 0.9998
  median exactly. `verification.pdf` reproduction section updated.
- [x] Seven bug fixes from the docs-verification findings (2026-08-11).
  (1) `(weights <= 0) | np.isnan(weights)` precedence fix in
  `SparseFitter.model_image` and `Scene.residual` — NaN-weight pixels are
  zeroed again. (2) `Catalog._detect` no longer subtracts the estimated
  background a second time (`run()` already rebinds `self.sci`); a
  user-supplied background level still comes off. (3) `PSF.gaussian` takes
  `(size, fwhm[, fwhm_y, theta])`: `fwhm` required (was UnboundLocalError),
  second positional is `fwhm_y` — existing `gaussian(n, fx, fy)` call sites
  had been silently passing `fy` into `theta`. (4) `PSFRegionMap` pickles
  and deepcopies (`__getstate__` drops `_prepared`/`_geoms`).
  (5) `AstroCorrect.fit` pops from a copy of `astrom_kwargs`; the dead
  `AstroCorrect(config)` construction in `Pipeline.run` is gone.
  (6) `PSFSZ<i>`/`RCIRC<i>` metadata now use the native lo-band pixel scale
  recorded before the upsample path rebinds `wcs[ifilt]` (was k× too small;
  no code reads the keys back). (7) `generate_scenes` assigns
  `Template.id_scene`, so the stamps column and `plot_subphot`/`plot_result`
  see real scene membership (was constant 1). Regression tests:
  `tests/test_verification_fixes.py` (8 tests). fitting.md NaN wording
  restored.
- [x] Full independent verification of the docs set + theme (2026-08-11).
  Twelve agents re-checked every docs page claim-by-claim against source
  after the gap-fix round; 55 further corrections applied (wrong shift sign
  in fitting.md, unpopulated `Scene.flux/err`, per-source `ee_psf_lo`
  divisor, `find_stars` keyword-only marking, donut fit geometry, mock
  position-shift units, stamps "PSF cubes" claim, and similar). The three
  runnable examples (README, overview, quickstart array example) were
  executed verbatim against synthetic FITS inputs: all pass end-to-end with
  finite fluxes (README's old positional-argument example raised TypeError
  and was rewritten). Quickstart restructured config-first to match real
  usage, with a realistic annotated JSON config and the stepwise
  build_psfs/build_kernels/run/write_outputs session. Theme switched to
  furo (left section navigation, right per-page TOC, dark mode) with
  sphinx-copybutton; docs extra now sphinx/myst-parser/furo/
  sphinx-copybutton. Build has zero page-level warnings; built HTML swept
  for internal names (none). Verification also surfaced real code bugs —
  logged under the docs-verification entry in `TODO.md`, headline items: a
  verified `weights <= 0 | np.isnan(weights)` precedence bug in
  `fit.py`/`scene.py`, a verified double background subtraction in
  `catalog.py` with `estimate_background=True`, and the `write_wcs_csv`
  dead `continue`.
- [x] Injected-truth mock verification of the UDS MIRI setup, one 800-source
  realistic mock per band (F770W/F1280W/F1500W/F1800W), run through the
  package verification framework (`build_realistic_two_detector_mock` ->
  `build_wiener_psf_maps` -> `run_pipeline_extension_scenario("psf_wings")`)
  with the production settings of the real runs: the band's STPSF grid
  (`UDS_MIRI_<BAND>_OS4_GRID1`), the band's blur (0.08/0.12/0.18/0.24"),
  aperture (0.70/1.20/1.20/1.50") and scene limits 800/1000. Total-flux
  recovery is unbiased to <= 0.2% in every band (medians 0.9998/0.9983/
  0.9982/0.9983; 0.9986-1.0000 at SNR > 25), residual RMS 0.80x the noise
  floor, MAD pull sigma 0.72 at SNR >= 20. Two systematics recorded: a faint-
  end pull skew growing with wavelength (-0.05 -> -0.34), and the hi-res
  self-fit sitting 0.7% high in all four bands. All 696 fitted sources fell
  back to the filter-mean `ee_psf_lo` (the known propagation TODO), which on
  this mock costs nothing measurable.
  Driver: `examples/minerva/run_verification.py` (tracked); outputs under
  `examples/minerva/verification/uds_<band>/` + `summary_all.json`. The
  framework keys its lo-res slot `f770w` internally, so inside a band
  directory that label means "the lo-res band"; the driver docstring records
  the aliasing. `run_pipeline_extension_scenario` gained `fit_overrides`
  (defaults unchanged) so production apertures/scene limits reach the mock
  fit.
- [x] Full Read the Docs documentation set (2026-08-11). Twelve pages under
  `docs/`: overview, quickstart, pipeline (full `run()`/`Pipeline`/`RunConfig`/
  `FitConfig` parameter reference incl. per-frame WCS CSV generation),
  outputs (complete column glossary), diagnostics (`source_products`,
  `show_sources`, `diagnose_sources`, `plot_inputs`, `plot_subphot`,
  `log_run`, PSF diagnostic scans), and component pages for psf, psf_maps,
  templates, fitting (+shifts), catalog, preprocessing (saturate +
  astrometry), simulation (MockMosaic + verification). Every page was
  fact-checked against source by an independent verification pass (~30
  corrections applied: wrong defaults, wrong column semantics, examples that
  would not run) and a completeness critique closed 4 major/8 minor gaps
  (undocumented diagnostics methods, `point_like` vs `flag_star` bridge,
  WCS-CSV creation path, `astrom_model` "poly" not "polynomial").
  `conf.py`: `include_patterns` whitelist keeps internal dev notes
  unpublished; `myst_heading_anchors=3`; `dollarmath`+`amsmath` enabled
  (unparsed `$$` math previously produced setext-heading artifacts).
  Local build clean of page-level warnings; built HTML swept for internal
  names (none). Code-side findings routed to TODO: dead `continue` in
  `utils.write_wcs_csv`, `minerva_link` column in `write_outputs`, stale
  docstrings (`matching_kernel.recenter`, `Pipeline.run` return arity,
  `FitConfig.astrom_model` comment).
- [x] All scratch/wren reports scanned and verified against the code at
  `2ba747b` (canfar-toolkit), stale claims fixed, five PDFs rebuilt
  (2026-08-11). Versioned `_vN` files left untouched as history;
  `docs/ENCIRCLED_ENERGY.pdf` mirror refreshed.
  * `encircled_energy.pdf`: kernel-cache sentence was stale (maps ARE stamped
    with method/reg/psf_size since `bfa76d6`); flow-table `ee_tmpl <= 1` row
    corrected to `= 1` on the default path (normalise-then-blank is the
    template-branch scheme, stated); added the open `k>1` defect - projection
    drops `ee_psf_lo`, so the per-source divisor is inactive on MINERVA-like
    runs and every source falls back to the filter mean.
  * `fit.pdf`: documented solver is current except `astrom_damping=0.8`, which
    exists only on the `template` branch - stated at both mentions; product
    table corrected `<run>_templates.fits` -> `<run>_stamps.fits` (write_stamps/
    load_fit landed here in `b1cac9b`); author line rebranded off `flux-bug`.
  * `noise_background.pdf`: the "calibration factor is now logged" claim is
    template-branch only - `get_bg_and_ivar` on this line still reports
    nothing; `_load_detection_ivar` likewise (here `load_data` passes
    `weights=[None, ivar]`).
  * `psf.pdf`: practical-notes default list gains `f2100w = 0.30"`
    (extrapolated along F1280W-F1800W; re-measure on COSMOS/EGS).
  * `template_comparison.pdf`: status box added - the "ivo" column describes
    the `template` branch (`template_schemes.composite_psf_wings`,
    `template_norm`, native-sum kernels); on canfar-toolkit
    `extend_templates='psf_wings'` is now the RunConfig default but dispatches
    into the self-convolution `extend_with_psf_wings`, and `build_kernels`
    uses unit-sum shapes + unit-sum kernels, so the native-sum kernel
    discussion does not apply on that line.
  * `flux_estimator_comparison.pdf`: "Status at 2ba747b" paragraph - four
    audit findings resolved by the parallel commits (`extend_templates` in
    RunConfig defaulting to psf_wings and threaded with
    `psfs=[prm_hi, prm_lo]`; `_ensure_maps` reloads `prm_hi`;
    `generate_scenes` callable with its own defaults, `minimum_bright=10`;
    `scene_max_size=800`/`scene_max_merge_radius=1000`), the rest confirmed
    still live (ee_psf_lo projection loss, aperture/total scale split,
    absolute ridge, two aperture frames, weight-mask precedence bug, dead
    R_cat resolver, catalogue-column cut).
  * `verification.pdf`: verified current, no changes needed - its config-path
    and scene-limit claims match HEAD.
- [x] Read the Docs scaffold. Sphinx 8 + pydata-sphinx-theme + myst-parser as a
  `docs` extra in `pyproject.toml` (`poetry install --extras docs`). New
  `docs/conf.py` (autodoc/autosummary/napoleon, `include_patterns` limits the
  published set to `index.md`/`api.rst` so the internal dev notes in `docs/`
  stay unpublished), `docs/index.md` landing page, `docs/api.rst` autosummary
  over the public modules, and `.readthedocs.yaml` (pip install `.[docs]`,
  Python 3.12). Local `sphinx-build` succeeds; remaining warnings are
  pre-existing docstring formatting nits. `docs/_build/` and generated
  `docs/api/` stubs are gitignored. Remaining manual step: import the repo at
  readthedocs.org.
- [x] Verification report written: `scratch/wren/verification.pdf`
  (`verification.tex`, 7 pages, the four comparison figures included). Records
  scope, inputs/settings, the four measured comparison conventions, the two
  fixes (apertures, psf_wings wiring), the two rejected hypotheses (scene
  limits, magnitude-cut metric), the final r < 3' table, caveats and the
  reproduction commands.
- [x] Confirmed at r < 3' (~17.5k matched sources per band, 10x the 1'
  sample, ~1 h for the four bands): Estimator 1 vs IDL at SNR > 25 is
  +0.02/+0.01/+0.01/+0.03 mag with sigma 0.05-0.09. F1500W's mag < 24 median
  fell from +0.06 to +0.03, confirming the residual was small-sample noise;
  F1800W's magnitude-cut column stays inflated (shallow band, noise-dominated
  selection) while its SNR-cut number is clean. Generated configs now default
  to `r_trial` = 3.0.
- [x] IDL agreement closed in all four UDS MIRI bands. With the `psf_wings`
  extension wired in (below) and the comparison read at the median over
  SNR > 25 sources, Estimator 1 agrees with classic IDL subphot to
  +0.01/+0.01/+0.02/+0.03 mag (sigma 0.04/0.02/0.03/0.08) for
  F770W/F1280W/F1500W/F1800W on the r < 1' patch. The earlier "+0.3/+0.5 mag
  red-band systematic" was a metric artefact: an IDL mag < 24 cut in a shallow
  band is noise dominated and its mean inflates. Scene partitioning was ruled
  out twice: `scene_max_size` 500 -> 800 with `scene_max_merge_radius` inf ->
  1000 px changed nothing to the second decimal, and F1800W's bisected
  coupling threshold reaching 0.381 left the bright end untouched. The
  generated configs keep 800/1000; `make_compare_idl_python.py` quotes medians
  plus an SNR > 25 line and prints a per-band summary table.
- [x] Config-driven runs now extend templates with PSF wings, and it was the
  whole of the disagreement with classic IDL. The extension was unreachable
  from a config file: `RunConfig` had no `extend_templates` field,
  `Pipeline.from_config` builds through `cls.__new__` and never set the
  attribute, and `load_data` finished construction with no `extend_templates`
  argument, so it defaulted to `None` and the branch in `run` was skipped.
  Independently, that same call passed `psfs=[None, self.prm_lo]`, so even a
  set flag would have fallen back to the *low-res* MIRI PSF for extending
  40 mas templates. The `FitConfig` knobs that look like they control this
  (`extend_wings_background_only`, `skip_template_extension_for_deblended`,
  printed in every run log) were arguments to a call that never happened.
  Changes, all in `pipeline.py`:
  * `RunConfig.extend_templates`, default `"psf_wings"`.
  * `_ensure_maps` also loads the cached hi-res map into `prm_hi`.
  * `load_data` passes `psfs=[self.prm_hi, self.prm_lo]` and
    `extend_templates=cfg.extend_templates`.
  * `run` refreshes `psfs[0]` on the `load_data(kernels=False)` path.
  * an INFO line reports the count, the PSF map used and the two flags.
  Effect on UDS F770W (r < 1', 4583 sources), against the same run without the
  extension: `flux_1` x1.169 median and x1.103 in the bright decile,
  `ap_corr_1` x1.111, `ap_flux_corr_1` x1.095. The aperture flux itself is
  unchanged at the bright end, as it must be. Against IDL the aperture-to-total
  factor goes from 0.925 +- 0.02 to **0.999 +- 0.02** and Estimator 1 from
  +0.10 +- 0.03 to **+0.01 +- 0.05** mag. `tests/test_pipeline_config.py`
  passes (7).
- [x] Fitted-product convenience access (2026-08-11).
  `Pipeline.source_products(id)` collects everything the fit produced for
  one source from the in-memory state (post-`run()` or post-`load_fit()`,
  nothing recomputed): window-aligned `tmpl_hi/tmpl_lo` stamps and
  `img_hi/segmap/img_lo/model/residual` cutouts, the band PSFs at the
  source position, fitted scalars (`flux, err, err_pred, ee_psf_lo, flag,
  shift`), the fit-table row, and the window slices for further slicing.
  `Pipeline.show_sources(ids)` renders them one row per source (8 columns;
  image/model/residual share one display scale so the subtraction is
  judged by eye). Shared PSF resolution factored into `_band_psfs` (also
  used by `write_stamps`). Tested in
  `tests/test_pipeline.py::test_source_products_and_show_sources` plus a
  post-`load_fit` smoke check; full suite 137 passed.
- [x] Per-source stamp output + post-run restore (2026-08-11).
  `Pipeline.write_stamps()` writes one FITS per run,
  `<out_dir>/<name>_stamps.fits`, with stamps at native per-source sizes:
  the `SOURCES` bintable stores each template flattened in a
  variable-length array column (`tmpl_hi`, `tmpl_lo`, heap storage — no
  padding) next to its full geometry (`ny/nx`, origin `x0/y0`, source
  position `xs/ys`) and fit metadata (`flux, err, err_pred, flag, flag_hi,
  id_parent, id_scene, ee_psf_lo, ee_tmpl, shift_x/y`). Nothing that has
  its own save file is duplicated: PSFs stay in the `<name>_psf_*.geojson`
  maps (rows carry the region `key_psf_hi/lo`), configs stay in the run
  JSON, and the primary header holds only pointers (`RUNNAME`, `IFILT`)
  and grid shapes for staleness checks. Written from `write_outputs()`,
  gated by `RunConfig.save_stamps` (default on); `read_stamps()` returns
  per-source dicts with 2D arrays.
  `Pipeline.load_fit()` is the post-run counterpart of `load_data()`: it
  reads the fit table + residual, rebuilds all templates from the stamps
  file via the new `Template.from_stamp()` (bit-exact geometry round-trip),
  and recreates model images/bin factors so the instance matches a
  completed `run()`; a missing stamps file is regenerated through the same
  template code path `run()` uses (extraction/extension/convolution were
  factored into `_prepare_hi_templates`/`_convolved_templates`) with
  fluxes taken from the fit table, then rewritten. Scenes are not
  persisted (`all_scenes` stays empty) and regeneration is exact only for
  runs without applied astrometric shifts. Verified equal to the live
  post-run state — table, residual, model, per-template data/geometry/
  metadata, and regenerated-vs-original stamp files — by
  `tests/test_pipeline.py::test_load_fit_restores_post_run_state` and
  `test_write_stamps_variable_size_single_file`.
  Found while verifying, and fixed: `Templates.predicted_errors()` used to
  overwrite `tmpl.err` with the predicted error after `run()` snapshotted
  the catalog errors, so post-run `t.err` disagreed with `err_1` by 0.1-1%.
  It now stores the prediction on `tmpl.err_pred` only (`tmpl.err` keeps
  the solver error; pre-fit seeding uses the returned array, which no
  caller relied on as a side effect), and its zero-weight branch returns
  `inf` instead of an uninitialized `np.empty` slot. Regression-tested in
  `test_load_fit_restores_post_run_state` (`t.err == err_1`,
  `t.err_pred == err_pred_1`).
- [x] mophongo runs on the CANFAR Science Platform. `uds_770_dr0.1` completed
  headless on 2026-08-10: 14 min wall on 8 cores, 34 GB peak, 2242 fit-table
  rows, 7 scenes, 3.6 GB of outputs under `arc:home/ilabbe/run/out/`. Method,
  job scripts and the traps are in `scratch/canfar/RUNNING_ON_CANFAR.md` and
  `scratch/canfar/jobs/`. The draw is that every input is already on `/arc`, so
  no download is needed at all. (That first run fetched the plain F444W
  `_drc_sci` from the grizli S3 bucket unnecessarily: it is on arc under
  `mosaics/nircam/n3.0/grizli/`, and a truncated listing was misread.)
  Validated against a local run of the same commit: `flux_1` agrees to a median
  2.9e-09 (p99 9.5e-06), which is the CG tolerance, and `err_1` to 1.9e-11. The
  33 percent flux offset against the older `examples/uds_770_dr0.1/` outputs is
  `490e13c`, not the platform — a local rerun reproduces it identically. For one
  trial patch CANFAR is not faster (14 min on 8 cores vs 9.1 min locally, ~3 min
  of it importing from the NFS venv); the gain is that the inputs need no
  download. Compute is the `skaha` REST API (`/skaha/v1`, pass `version='v1'`),
  not ssh — the transfer endpoint is SFTP-only. `arc:home` mounts locally via
  FUSE-T (not macFUSE, whose kext cannot load on Apple Silicon with SIP on):
  `./canfar-mount.sh /home/ilabbe ~/canfar_home`.
- [x] Installing into a clean container exposed two packaging/CLI defects, both
  fixed 2026-08-10. Five directly-imported packages (`psutil`, `photutils`,
  `matplotlib`, `pysiaf`, `pillow`) were undeclared in `pyproject.toml`; four
  were masked by transitive installs until pip resolved `photutils` to 3.0.0,
  which dropped `IntegratedGaussianPRF` and broke `drizzlepac` 3.9.1. Added via
  `poetry add`, `photutils` bounded `>=2.2.0,<3.0.0`. Separately,
  `pipeline.py`'s `steps` argument combined `nargs="*"` with `choices`, and
  argparse checks the collected list against `choices` as one value, so a bare
  `python -m mophongo.pipeline <cfg>` died with `invalid choice: []`; `choices`
  is gone and the steps are validated explicitly. Verified by rebuilding a
  CANFAR venv from `pyproject.toml` alone with no manual pins, and by the full
  suite (130 passed).
- [x] The latest MINERVA reductions are staged for all three fields (UDS,
  COSMOS, EGS) and there is a run config for every field and MIRI band, 17 in
  all, in `examples/minerva/`. Versions: UDS n3.0/m3.1/n3.0_v1.2/
  n3.0_m3.1_v1.2.1, COSMOS n3.0/m3.0/n3.0_v1.0/n3.0_m3.0_v1.0.1, EGS n2.0/
  m2.1/n2.0_v1.3/n2.0_m2.1_v1.3.1. Pointers: `MINERVA/data/00WHERE` (what
  exists), `00CANFAR` (how to get it), `data/stage/README.md` (this staging).
  The sources are split, so staging is two scripts:
  * `data/stage/stage_nircam_s3.sh` -- NIRCam/HST from the public grizli S3
    bucket, `f277w`/`f356w`/`f444w` × (`_drc_sci_bkgsub`, `_drc_wht`,
    `_wcs.csv`). 21 files, 66 GB. `_drc_sci_bkgsub` replaces the plain
    `_drc_sci` of the DR0 configs; that is the flavour recommended for
    photometry. Transfers run `xargs -P` (default 6).
  * `data/stage/stage_canfar.py` (`walk`/`plan`/`get`/`link`) -- MIRI mosaics,
    both detection flavours' segmaps and SUPER catalogs, the wMIRI catalogs,
    MIRI PSFs and their encircled-energy tables, empirical PSFs, rms maps, from
    `arc:projects/minerva`. 351 files, 11 GB gzipped, no failures. `get` runs
    N `vcp` processes (default 6).
  Verified: every staged FITS opens, its last data row reads back, and its byte
  size equals `NAXIS1*NAXIS2*|BITPIX|/8`. Segmap grids match their NIRCam
  mosaics. Catalogs hold 345792 (UDS), 294126 (COSMOS), 520875 (EGS) sources.
  * `examples/make_minerva_configs.py` writes the configs, reading frame counts
    off the WCS tables and the trial-patch centre off a scan of the MIRI weight
    map, so no number is hand-copied. All 17 load as `RunConfig` and every
    input path resolves. The patch scan reports the weight percentile it
    reached: 83rd-99th for UDS and COSMOS, but 21st (F1280W) and 15th (F1800W)
    for EGS, whose MIRI footprints are too fragmented for a deep 1.2 arcmin box
    to fit. It relaxes its coverage requirement in steps and returns None (so
    the config becomes a full-field run) rather than emitting a centre in a
    zero-weight region, which an earlier version did silently.
  * `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC` gained `f2100w` = 0.30", extrapolated
    along the F1280W-F1800W trend. COSMOS and EGS have F2100W and the lookup
    returns `None` (no blur at all) for a missing key.
  Trial runs (`r_trial` = 0.6') in all four UDS MIRI bands, compared against
  classic IDL subphot by `scratch/wren/make_compare_idl_python.py`, which
  recreates the four-panel `compare_idl_vs_python_*` figure per band. Two
  conventions were established from the data rather than assumed: IDL
  `xdet`/`ydet` are 1-indexed FITS pixels on the 40 mas F444W grid (through
  that WCS they reproduce the SUPER catalog `ra`/`dec` to 0.0 mas, so the sky
  match is exact and every python source matches), and IDL fluxes are nJy
  against the mosaic's own 10 nJy unit, pinned by a bright-end ratio of
  0.092-0.099 with 0.03 dex scatter.
  The first pass showed +0.37 mag in panel (a), raw aperture flux, where wren's
  figure had +0.01. Cause: the generated configs inherited a flat
  `aperture_diam` = 0.5" from `uds_770_dr0.json`, while IDL uses 0.70/1.20/
  1.20/1.50" for F770W/F1280W/F1500W/F1800W -- values the SUPER catalogue
  confirms independently in its `Scale_factor_APER07_F770W`,
  `...APER12_F1280W`, `...APER12_F1500W`, `...APER15_F1800W` column names.
  Raw aperture fluxes are only comparable at the same aperture, so
  `make_minerva_configs.py` now carries `APERTURE_DIAM_ARCSEC` and the runs
  were repeated. After the fix, F770W panel (a) is mu = +0.01, sigma = 0.01,
  and the whole remaining disagreement sits in the aperture-to-total factor:
  panel (b) `ap_corr_1`/`totcor` = 0.93, which propagates to +0.10 mag in
  Estimator 1. That is the quantity the estimator port is about.
  F1500W and F1800W are much worse (+0.18 +- 0.57 and +0.36 +- 0.75 mag in
  panel (a)) and the cause looks like scene partitioning, not photometry:
  `scene_max_size` = 500 forces the adaptive coupling threshold to 0.193 at
  F1800W, ten times past the value wren sets by hand, and the band ends with
  more scenes than F770W despite a far broader PSF. See `TODO.md`.
  The comparison is now driven by a declarative `PANELS` table taken from
  `flux_estimator_comparison.tex` Sec. 9, six panels wide, and picks the
  `ivo:main` or `fork` column per panel by inspecting the fit table. Two
  panels have no `ivo:main` counterpart and are drawn blank and labelled: the
  shape ratio (IDL `psfcor<f>`, fork `apcor1_<i>`) and the internal F444W
  total (fork `f444w_ktot_<i>`).
  `flux_estimator_comparison.tex` gained Sec. 9.4 recording the recipe: the
  match convention, the unit factor, the aperture requirement, the panel map,
  and two corrections to earlier work. First, the document's warning that
  comparisons against `flux_F` must undo `invtotcorcat` does not hold for
  these catalogues -- applying the SUPER `tot_cor` degrades the bright-decile
  scatter from 0.033 to 0.158 dex, so `flux_F` is compared as released; a
  cross-check of `flux_Ff444w` against the catalogue's own F444W columns was
  inconclusive at better than 0.24 dex and is left open. Second, the existing
  `compare_idl_vs_python_chimean_f1500w_after.png` plots the fork's `apcor_1`
  against IDL `psfcor<f>` in panel (b2), which compares the full released
  correction against one of its factors; the like-for-like column is
  `apcor1_1`. PDF rebuilt, 30 pages, no undefined references.
  End-to-end check on `examples/minerva/uds_f770w.json` (`psfs kernels`): 32
  NIRCam and 9 MIRI ePSF grids loaded from the patterns, 1694 hi and 2911 lo
  PSFs drizzled, kernel map built at `wiener` `reg` = 5.62e-4 from the 21-point
  scan, kernel DC 0.999444 (range 0.999442-0.999448) before renormalisation.
  No errors. EGS has no ePSF grids yet, so its first run per band builds them
  through `psf_autobuild` and will be much slower.
- [x] Solved: why IDL/monu `flux_Ff444w` sits ~1.6x below the catalog
  `f_f444w`. It is a convention mismatch, not a flux error. `flux_Ff444w` is the
  raw `faper` sum on the detection image at fixed radius `subphot_raper`,
  zeropoint-scaled only - no PSF correction, no aperture-to-total, no neighbour
  subtraction (`dophot.pro:733,737-738,773`; detection column q=0 path, unlike
  the measured bands which get `f1*totcor1*invtotcorcat*zpscl` at `:824`).
  Measured the v3.0 F444W mosaic at the monu positions: ratio
  `flux_Ff444w/aper(r)` = 1.006+-0.022 at r=0.35" and flat in SNR and
  `use_aper` class, at no other radius - so the monu run used
  `subphot_raper=0.35"` (the archived `phot.param` says 0.9), and the column is
  in nJy. Closing test: `f_f444w/flux_Ff444w =
  tot_cor * aper(use_aper/2)/aper(0.35")` per source - observed/predicted =
  0.992+-0.017 (SNR>10), 0.999+-0.004 (SNR>100), corr 0.998. The ratio rises
  1.34 (faint, point-like: 1/EE(0.35")) to ~2 (bright, use_aper=1.0-1.4"), and
  neither `tot_cor` nor `fauto/faper` alone matches (both fall with SNR while
  the observed ratio rises). Also: the SUPER catalog has no `aper_corr` column,
  so `invtotcorcat=1` for the monu run and `flux_Ff770w` is the *uncorrected*
  IDL total (Estimator 1 on the IDL system). Figure:
  `scratch/wren/fig_idl_f444w_ratio.png` (all diagnostic figures now use
  matplotlib `layout='tight'`). Recorded in the report's column-name section
  (dagger footnote), `flux_estimator_comparison.pdf` rebuilt. Follow-up
  (2026-08-11): catalog-444-vs-IDL-444 comparisons removed everywhere (pointless
  by construction); new report section documents every subphot output column
  with formula and source line (`_org/_phot/_res/_model/_model_nn` identities,
  `_phot.cat` f/e/fcor/ecor/apcor/totcor/fnn/enn/forg, `_model.cat`
  shx/shy/fmodel/emodel/chi_red/chi_red_half/bg_ann/chi_ann, released
  flux_F/eflux_F/flux_contam/chi*/rbg_ann/contam/snr_nn/psfcor/totcor/wht/use
  and the full `use`-flag boolean). Released `psfcor` = internal `apcor1` =
  ap_hi/ap_lo measured on the source composite (psf_apercor defaults off, so the
  PSF branch is the fallback); released `totcor` = 1/ap_lo; `eflux_F` correctly
  carries the totcor1 scaling (Estimator-1 error convention).
- [x] Checked `examples/run_uds_770_wren.py` inputs: the wren run extracted
  templates from the aperpy-homogenized image
  (`...f444w-clear_drc_sci_f444w-matched.fits`) while (a) the weight is the
  *native* `_drc_wht.fits`, (b) `prm_444.psfs` are drizzled from the *native*
  STPSF grids, and (c) the MIRI matching kernels are built native-F444W ->
  MIRI. So her templates carry the aperpy homogenization kernel that neither
  the PSF map, the containment factors, nor the matching kernels know about,
  and the detection SNRs use a weight map whose noise correlations do not
  match the smoothed image. Consistent with the catalog side (SUPER is
  measured on matched images) but inconsistent with every native-PSF-derived
  correction in her chain. Any rerun should use the native F444W sci+wht or
  build the PSF/kernels from the matched PSF.
- [ ] Exhaustive audit of the implementation on main, written up as
  `scratch/wren/flux_estimator_comparison.pdf` (v4, 23 pp; source
  `flux_estimator_comparison_v4.tex`). 50 agents mapped `src/mophongo` across five
  areas and adversarially re-checked every claim at its cited `file:line`: 751 claims,
  682 confirmed, 62 imprecise, 4 wrong, plus 41 completeness gaps. Section 7 replaces the
  stale "current python implementation" section; sections 8 and 9 state the four
  estimators in main's conventions and measure the catalogue side.
  Headline findings:
  * **`ee_psf_lo` is destroyed in the default multi-resolution path** —
    `project_to_block_replicated_grid` does not copy it, so every source falls back to
    the filter-level mean and the encircled-energy chain is inactive on any k>1 run.
    Confirmed by execution. See TODO.
  * main implements Estimator 1 *in full*: `ap_corr_<i> = 1/ap_lo` = `totcor1`. The
    earlier reading that it "stopped one factor short" is obsolete.
  * the aperture family and the total family are on absolute scales differing by
    `1/S_lo`; `ap_flux_corr_<i>` is never divided by `ee_psf_lo`.
  * `_resolve_catalog_ap_radius_pix` (the only `R_cat` implementation) has zero callers,
    so the two-radius generalisation is unreachable.
  * `extend_templates` is not reachable from a run config, so config-driven runs fit bare
    segment cutouts.
  * the fit table drops every catalogue column except id/x/y and a short allowlist, so
    `fauto_KRON`/`faper_KRON`/`tot_cor`/`use_aper`/`f_f444w` cannot reach the estimators.
  * operator-precedence bug in the weight mask (`fit.py:269`, `scene.py:1192`):
    `w <= 0 | isnan(w)` parses as `w <= (0 | isnan(w))`, so NaN-weight pixels are never
    masked.
  * `generate_scenes` raises `TypeError` when called with its own defaults
    (`minimum_bright=None`).
- [ ] Merge path for the dev-wren aperture-correction / total-flux system
  decided and written up: `docs/WREN_MERGE_PATH.md`. Design only, no code
  changed yet. It restates `docs/FORK_AUDIT_WREN.md` against the settled
  encircled-energy chain and the `template` branch's rewritten builder, and
  sequences the work as PR-0 (land `template`) through PR-6.
  What the audit's tiers reduce to:
  * `PSFRegionMap.containment` is dropped, and replaced by a per-region curve
    of growth. wren needs `containment` because its stored stamps are
    unit-normalised; ours are absolutely calibrated, so `EE_true(r)` is the
    aperture sum on the stamp directly. What is genuinely missing is EE at an
    *arbitrary* radius — `refresh_ee` caches two scalars — which the Kron path
    needs. Add `refresh_cog`/`get_ee_at`, keyed on `resolve_key` rather than
    on `id(psf)` (wren fixed a real collision there: `get_psf` returns a fresh
    ndarray per call and CPython reuses freed ids).
  * The four estimators are **kept in full**, renamed `est3int -> est3` and
    `est3cat -> est4`:
    `ap_flux_est1 = (ap_model + res_sum)*totcor1` (IDL-exact),
    `ap_flux_est2 = ap_model*totcor1 + res_sum` (residual unscaled),
    `ap_flux_est3 = ap_model*apcor1*tcor_int + res_sum` (Kron-convention),
    `ap_flux_est4 = ap_model*apcor1*tcor_int*s_cat + res_sum` (catalogue-tied).
    `_model_kron` is not a fallback for truncated templates: it runs the
    catalogue's own Kron recipe on the extended model stamp, giving a noise-free
    `fauto/faper` for sources too faint to measure it on the data.
  * **wren's formulae cannot be transcribed verbatim.** wren carries the
    finite-stamp truncation through `containment` (`c_det`, `c_b`) because its
    stamps are unit-normalised; ours are absolute and carry it in `S_hi`/`S_lo`.
    Copying the expressions double-corrects by `S_hi/S_lo` = 4.6% on UDS. The
    substitution is `c_det -> S_hi = prm_hi.get_ee_box`, `c_b -> S_lo =
    ee_psf_lo`, giving `apF_corr = (apF_book + apF_blank)*S_hi` and
    `apB_corr = (apB_book + apB_blank)*S_lo`. Bridge identity to pin:
    `ap_flux_est2 == flux_<i>_total + res_sum` exactly, for an isolated point
    source under `psf_wings`.
  * `sample_psf_on_stamp` divides by `psf.sum()` before interpolation, so the
    `psf_wings` halo is unit-sum, not absolute: `model_total = template_norm /
    S_hi`, replacing wren's `trunc_denom`. Needs the hi-res PSF map wired into
    the config path (`psfs=[None, prm_lo]` today).
  * The SUPER catalog's `tot_cor` factorises, and the second factor is
    recoverable: `ee_kron_cat = (fauto_KRON/faper_KRON)/tot_cor`. Verified on
    288,153 UDS rows: bounded in (0,1] (0.13% exceed) and **a single global
    function of `kron_radius_circ`** - regressing on 200 quantile bins leaves
    NMAD 4.3e-4 (0.05%), and residual medians over a 4x4 sky grid span -6e-5 to
    +1e-5. So the catalog used one F444W growth curve for the whole field. The
    recovered curve is a NIRCam F444W EE: 0.583 at 0.125", 0.838 at 0.375"
    (the first Airy minimum), shoulder, 0.939 at 1.375" - and it does not reach
    1, so it carries its own normalisation convention. Three consequences:
    (a) a 288k-row acceptance gate for the new `PSFRegionMap.get_ee_at`, far
    harder than anything in wren's suite and free; (b) `s_cat` factorises
    exactly into `fauto_KRON/kron_flux_model` (Kron flux, model vs data) times
    `ee_kron_int/ee_kron_cat` (PSF EE, ours vs theirs), the second splitting
    again into a global-curve ratio and the +-1% per-region structure the
    catalog does not have - write all three; (c) `est4` gains a route that
    never divides by a noisy `f_f444w`. Derive the curve at run time, do not
    ship a table. Unexplained: 3.4% of sources sit >=2% below it, and
    `flag_kron` is 0 for every row in this release.
    wren's measured `s_cat` of 0.83 [0.65, 0.88] sits on top of median
    `ee_kron_cat` = 0.835 (5-95% 0.675-0.905), so its "morphology-dependent
    per-source difference" may be largely a missing `1/EE_kron`. Test that.
  * `tcor_int = ktot/(template_norm*apF_book)` carries **no** `S_hi`: the
    denominator is already the absolute model aperture flux. `S_hi` belongs in
    `apcor1` via `apF_corr`. Easiest mistake in the translation.
  * wren's Sec 5.3 bookkeeping invariant (`ap_model` from the fitted template's
    own aperture fraction, never a curve-of-growth substitute) is already
    satisfied here: `ap_model = fl * _aperture_sum_on_template(tmpl, r_img)`.
    Only the frame mismatch needs fixing. Comparison-axis rule adopted: `est1`
    /`est2`/`est3` on the IDL axis, `est4` only against the SUPER catalog.
  * `tot_cor` is noise-dominated where it matters: NMAD 1.67 (48%) at
    `f_f444w/e_f444w` < 3 against 0.18 (14%) above 100, and 57,622 of 345,792
    rows have no `kron_radius` at all. That is what `est3` supplies.
  * The `0x40`/`0x80` flag collision and the wren `convolve_cutout` hazard are
    both already resolved on `template` (extension flags moved to `0x100`/
    `0x200`; even-alignment convolution untouched).
  * The `utils` curve-of-growth helpers are retired in favour of what already
    exists in `psf.py` and `template_schemes.psf_ee_radius_pix`.
  * Both 345k-source performance cuts are scheduled: `_sources_with_coverage`
    (~57% of UDS sources never built) and `np.searchsorted` for the segment
    bbox lookup (3.894 -> 0.053 ms/source); ROI-restricted ownership is
    already on `template`.
  * The flux-block ridge is **not** solved (see TODO): the `flux-bug` fix
    removed a different, larger term. The remaining
    `lam_A = 1e-6 * median(diag(A))` added before whitening is the exact
    configuration wren measured, and it must be made relative before the
    estimator work lands.
- [x] `DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC` extended from F770W-only to all MIRI
  imaging bands used so far: 0.08" (F560W, F770W), 0.10" (F1000W,
  interpolated), 0.12" (F1280W), 0.18" (F1500W), 0.24" (F1800W). Values are
  the MINERVA-UDS star-test measurements (0.088/0.116/0.189/0.246") rounded;
  report at `scratch/wren/psf.pdf` (versioned `psf_vN.*`, figures rebuilt by
  `scratch/wren/make_psf_report_figs.py`). Tests touching the defaults pass
  (`test_pipeline_config.py`, `test_mock_mosaic.py`).
- [x] Cached run products carry provenance, and missing PSF grids build
  themselves. Three artefacts are cached across runs and each is now reused
  only when it matches the run asking for it.
  * STPSF grids. `_load_epsf` loads the grids matching `pattern_hi`/`pattern_lo`
    under `psf_dir`; when none match it derives the generator settings from the
    pattern itself (`_psf_factory_kwargs`: prefix, `num_psfs`, oversampling,
    detector sampling, MJD) and runs `PSFFactory.from_csv` on that band's
    exposure list, then reloads. Deriving from the pattern is what guarantees
    the generated filenames are found again. An empty result is now an error
    instead of a silent no-op that let a run continue with no PSF. Off with
    `psf_autobuild=False`; field of view via `psf_fov_arcsec`.
  * Drizzled PSF maps. Stamped with `pattern`, `psf_size` and `blur_fwhm`, and
    rebuilt when any of them differs.
  * Kernel maps. Stamped with `kernel_method`, `kernel_reg` and `psf_size`, and
    rebuilt when the method differs. The matching method is worth a few percent
    in the flux scale, so reusing a map built another way applied a correction
    the run had not asked for.
  Maps written before this carry no provenance columns and count as stale, so
  the first run against an existing output directory rebuilds them.
- [x] `Pipeline.run_all` writes a full log to `<out_dir>/<name>.log`. The
  package reports through both `logging` and bare `print`/`tqdm`, so the log
  tees stdout and stderr as well as attaching a handler, and keeps only the
  final state of each progress line. Appends across runs, with a header
  (timestamp, python, platform, out_dir) and a footer giving elapsed time or
  the exception. Available on its own as `Pipeline.log_run()`.
- [x] Encircled-energy bookkeeping settled end to end. One scalar converts a
  fitted amplitude to a total flux: `ee_psf_lo`, the absolute encircled energy
  of the low-resolution PSF stamp at the source position. For a point source
  `A = f * S_lo` exactly, for any weight map and independently of how the
  high-resolution stamp is truncated, because the kernel's target is the
  low-resolution shape and the `1/S_hi` from normalising the input cancels.
  Verified through the production code path (even stamps, real `Template`
  cutouts, `convolve_cutout`): `A/S_lo` = 1.00140 to 1.00160 for the JWST pair
  while `S_lo` moves from 0.85 to 1.00, and exact to five decimals for a
  compact Moffat pair. Report: `docs/ENCIRCLED_ENERGY.pdf` (LaTeX source, figures and the scripts that
  produce every number: `scratch/wren/encircled_energy.tex` and
  `scratch/wren/ee_report/`, local only).
  Changes:
  * `build_kernels` matches unit-sum PSF *shapes* and renormalises the kernel
    to unit sum, so it carries no flux scale of its own. Stored PSF maps keep
    their native sums.
  * `build_kernels` defaults to `method="wiener"` with the regularisation
    scanned once on the median of the hi/lo PSF stacks. The old
    `SplitCosineBellWindow` default biases the flux scale by 2.3-3.3% on the
    JWST pair, of which a stamp-independent 2.2% has the closed form
    `sum(W|P|^2)/sum(W^2|P|^2)`; wiener leaves 0.16%, flat in stamp size.
  * `optimize_matching_kernel_regularization` accepts `pixel_ratio`, so the
    scan scores the kernel that will actually be built.
  * `PSFRegionMap.refresh_ee` measures `ee_box` and `ee_rlim` per region once
    when `psfs` is set; `get_ee_box`/`get_ee_rlim` cost 11.8 us against 11.3 us
    for `get_psf`. Derived from the stamps in use, so the edge taper, negative
    clipping, drizzle kernel and blur are all inside the sum.
  * `Template.ee_psf_lo` and `Template.ee_tmpl`, both `nan` by default and
    carried through `convolve_cutout`. `flux_<i>_total` now divides by the
    per-source `ee_psf_lo`; the filter-level mean from
    `_filter_psf_throughput` is the fallback only.
  * `ee_tmpl` is a diagnostic and is deliberately never read as a correction.
    The amplitude scales with the leverage of the blanked pixels, not their
    flux, and neighbour segments lie in the faint wings: measured, dividing by
    it turns a 0.2% error into 0.9% isolated, and 3.3% into 6.7% for a pair at
    8 px.
  **Consequence:** `flux_<i>` and `flux_<i>_total` change on the next run, and
  cached kernel maps must be rebuilt with `build_kernels(overwrite=True)`.
  Full suite green (118 passed).
- [x] `build_kernels` now matches unit-sum PSF *shapes* (`pipeline.py`,
  `_unit_sum`), as `docs/PSF_SHAPE_THROUGHPUT_CONVENTION.md` has always
  prescribed. Previously it passed native-sum stamps, so `sum(kernel)` carried
  `sum(psf_lo)/sum(psf_hi)` and the kernel's own fidelity error was hidden
  inside what looked like a throughput factor. Measured on the cached UDS
  DR0.1 maps (1694 hi / 294 lo / 2911 kernel regions): `sum(psf_hi)` 0.96106,
  `sum(psf_lo)` 0.91699, old kernel DC 0.95512 over the range 0.9357-0.9671,
  matching `S_lo/S_hi` to 5.8e-6 per position. New kernel DC is 1.000005 with
  a maximum deviation of 6.1e-6, so the old spread was throughput, not
  regularization: there is nothing in the DC left to renormalize, and any
  future deviation from 1 is a window/regularization diagnostic.
  Amplitude effect, measured on region 0 with a unit-sum template and the
  resampled F770W stamp as data: `A` 0.930420 -> 0.889643, exactly
  `S_lo/S_hi` = 0.956173, and `A/S_lo` 1.013247 -> 0.968840. The remaining
  0.9688 is the SplitCosineBell kernel's projection efficiency, now isolated
  instead of cancelling against a +4.6% DC excess. `prm_hi`/`prm_lo` keep
  their native sums, so `_filter_psf_throughput` is unchanged.
  **Consequence:** `flux_<i>` and `flux_<i>_total` drop ~4.4% on UDS-like
  fields on the next kernel rebuild (`build_kernels(overwrite=True)`; cached
  kernel maps are reused otherwise). The divisor in `flux_<i>_total` is a
  separate open decision - see TODO. Supporting analysis:
  `scratch/ee_bookkeeping_facts.md`. Full suite green (118 passed).
- [x] `examples/check_psf.ipynb` rewritten as a drizzled-PSF versus empirical-star
  check on the MINERVA-UDS mosaics in F444W (n3.0 40 mas), F770W (m3.0 80 mas)
  and F1500W (DR0 v2.4 80 mas), branch `drizzlepsf_star_mismatch`.
  Stars, neighbour masks and isolation all come from the
  `n3.0_m3.1_v1.2.1` SUPER catalog; nothing is detected on the images. Only
  data-quality tests (finite pixels, non-zero weight, unsaturated core) touch
  the pixels. Outputs land in `examples/star_psf_check/`.
  MJD coverage of the updated STPSFs confirmed: F444W uses 29 of 32 loaded
  grids over 297 frames, F770W 9 of 9 over 229 frames, F1500W 4 of 4 over 162
  frames; worst frame-to-grid offset 2.0 d, median 0.6-1.1 d.
  The Gaussian-basis kernel fit runs on a core box (about three PSF FWHM), not
  the full stamp: over the full stamp the fit is dominated by a wing deficit in
  the drizzled model, and NNLS pays for it with a broad component that depresses
  the core and leaves the corrected model worse than the raw one. The wing ratio
  is now reported separately: azimuthal model/star is 0.94 at F444W, 0.75-0.80
  at F770W and 0.54-0.69 at F1500W (the MIRI numbers got worse once neighbour
  segments were masked, because the stars' own wings are no longer inflated by
  neighbour light).
  Star selection has two finders, `STAR_FINDER`. The catalog one (`flag_star`)
  is badly incomplete for this purpose: only 60 of its stars reach SNR 100 in
  F770W and 9 in F1500W, since the flag rides on the F444W detection. The
  default is now catalog-free: a plain peak search per band (median and MAD from
  a strided pixel sample, no background model, no ivar; exact zeros dropped, or
  the MAD collapses on the background-subtracted MIRI mosaics), the 2000
  brightest peaks, then a point-source cut on the ratio of two raw apertures,
  r = FWHM and r = 2 FWHM, with the FWHM from that band's drizzled PSF. The
  model gives ratio 0.817 / 0.811 / 0.820 and the stellar locus is +-0.04 around
  it. Survivors also carry `psf_corr`, the normalised correlation with the model
  PSF. This delivers 98 / 89 / 37 stars against 88 / 46 / 3 from the catalog —
  F1500W is the band it rescues.
  Sample: SNR > 150, unsaturated, covered, centroidable, and isolated by
  a PSF-based test — a neighbour's contribution inside the star's R80 aperture
  is computed from the band's own drizzled PSF shifted to the neighbour's
  separation, and anything above 20 % of the star's flux in that aperture is
  rejected. R80 is 0.254" (F444W), 0.495" (F770W), 0.777" (F1500W); a neighbour
  sitting at one R80 puts 0.35-0.42 of its flux inside the aperture. Of the 835
  flagged stars, 514 reach SNR 100 in F444W, 60 in F770W, 9 in F1500W; the
  analysis runs on 100, 49 and 3. Stars are about 1.8 mag fainter at 15 um than
  at 4.4 um (median catalog flux ratio 0.18), which is what makes F1500W thin.
  Neighbours are masked with their catalog segments (the n3.0_v1.2 segmap,
  whose labels are the catalog ids), resampled onto each stamp WCS and grown by
  0.05-0.30" so the 40 mas NIRCam segments still cover the wider MIRI PSF. Only
  the star's own segment plus a 2 px core is protected; guarding the whole
  normalisation aperture, as a first version did, left neighbours inside it
  unmasked, which is precisely the light that has to go.
  A last cut removes extended sources: the data/model growth-curve ratio at the
  data's own R80 must be at least 0.95. Galaxies with a compact core pass every
  point-source test and fail here — 12 of 100 in F444W, 3 of 49 in F770W.
  The kernel basis carries an explicit identity element, so "no broadening" is a
  solution the fit can select rather than one it must approximate with its
  narrowest Gaussian, and the run reports a no-blur baseline (the model at its
  best-fit amplitude) alongside the fitted kernel.
  All five bands now run: F444W plus the four MIRI bands. F1280W, F1500W and
  F1800W all come from the DR0 v2.4 release (no download was needed, they were
  already local); F1280W and F1500W sit in the 60881-60906 window, F1800W on the
  earlier 59789-60514 baseline, and each has matching MJD-resolved grids.
  Final samples: 98 stars (F444W), 89 (F770W), 37 (F1280W), 37 (F1500W),
  23 (F1800W). Every star gets its own stamp figure (284 in total); stale
  figures from an earlier run are cleared per band so the directory always
  matches the current sample.
  The MIRI diffusion kernel grows with wavelength — 0.084" (F770W), 0.111"
  (F1280W), 0.172" (F1500W), 0.200" (F1800W) — roughly a quarter to a third of
  the PSF FWHM in each band.
  The kernel fit now solves for a constant background at the same time, in real
  space with `scipy.optimize.lsq_linear` (kernel coefficients bounded
  non-negative, the offset free either way). The annulus median that
  `clean_with_catalog` subtracts is measured over a ring that still holds PSF
  wings, so it overshoots, and the growth curves used to peak and then turn back
  down. Two details matter. The design matrix is built with the pipeline's own
  operator: column zero is the untouched model (the identity, exact by
  construction) and the rest are `mock_mosaic.gaussian_blur_fourier` applied at
  each basis width — the same flux-conserving analytic transfer function
  `Pipeline._drizzle_lo_blurred` uses. An earlier version assembled the basis by
  hand instead, from a delta placed at `(n-1)//2`, which disagrees with the
  `n//2` kernel centre `utils.fftconvolve` documents; on the even stamps the
  pipeline uses that shifted the identity column by a pixel, and the only
  symptom was a fit that silently refused all pass-through weight while
  `resid_fit` rose to 0.29. Reusing the production operator removes the question
  entirely, and a check cell asserts per band that the identity column equals
  the model exactly (0.0) and that convolving with the returned kernel image
  reproduces the fitted model (round-trip 5-8e-8) on 100x100 and 125x125 stamps
  alike. Separately verified that the pipeline itself was never affected: its
  kernels come from `matching_kernel` on a PSF pair in the same frame, which
  cancels any centring convention (0.0000 px shift, even and odd), and its one
  analytic path, `gaussian_blur_psf`, goes through the same Fourier operator
  (0.0000 px, flux 1.000000).
  The offset must be constrained on the core box *plus* an outer ring
  (`BG_ANNULUS_FRAC = 0.8` of the stamp half-width): on the core box alone it is
  degenerate with the model's wing deficit and soaks it up as a positive
  constant, which made the F770W turnover worse (outer growth curve at 0.79 of
  its peak) instead of better. With the ring the outer growth curve sits at
  1.000 (F444W, F770W) and 0.996 (F1500W), against 0.998/0.998/0.950 before.
  Each star's fitted offset and registration shift are recorded, and the shift
  is printed in the ratio panel of its figure.
  `compare_psf_to_star` takes `save_curves=True` and writes the growth curves it
  plotted to `<figure>_cog.csv` (radius, encircled energy for data/psf/psf x
  diff, and both ratios). A stacking cell reads all of them and overplots the
  whole sample per band with a 16-84 percentile band, which is where the core
  mismatch shows up cleanly: the median data/psf ratio dips to 0.97 (F444W) and
  0.88-0.91 (F1280W-F1800W) at small radius.
  A pipeline would carry one number per band rather than a kernel per star, so
  the median `fwhm_eff` per band is applied as a single Gaussian to every model
  through `gaussian_blur_psf` — the same operator the pipeline broadens with —
  and its growth-curve ratio is stored alongside. The widths are 0.031" (F444W),
  0.088" (F770W), 0.116" (F1280W), 0.189" (F1500W) and 0.246" (F1800W), i.e.
  0.8-3.1 drizzle pixels. One blur per band tracks the median of the per-star
  fits closely: both sit within about 1-2 % of unity from the core out to the
  normalisation radius, so the per-star freedom buys little over a single
  per-band Gaussian.
  Results (median over stars): F444W puts 88 % of the diffusion-kernel weight on
  the identity — the core needs no extra smoothing, and the remaining 12 % is a
  0.090" FWHM halo term; the basis-free width check gives 0.036". F770W needs a
  real core term, 0.082" (kernel half-max) against 0.084" from the width check,
  the two agreeing. F1500W needs 0.176", with 0.173" from the width check.
  Core peak residual, no-blur baseline -> fitted kernel: 6.3 % -> 5.4 % (F444W),
  8.0 % -> 5.9 % (F770W), 6.8 % -> 5.2 % (F1500W). The no-blur baseline already
  wins for 19 % of F444W, 15 % of F770W and 11 % of F1500W stars, and the model
  is measurably broader than the star for 18 % and 13 % of F444W and F770W —
  cases no non-negative kernel can repair, since NNLS cannot sharpen.
  The multi-Gaussian kernel is well approximated by a single convolution with
  its largest-coefficient element: median |full basis - single element| is
  1.9 % of peak in F444W, 2.9 % in F770W, 4.4 % in F1500W, and 0.7-1.6 % on the
  growth curve.
  PSF-region-map PSFs (drizzled at the region centroid, as the pipeline uses
  them) need more smoothing than PSFs drizzled at the star: 0.053" versus 0.090"
  halo term in F444W, 0.107" versus 0.082" in F770W, 0.176" versus 0.176" in
  F1500W.
  Distributions plot `fwhm_half_all` and `fwhm_dom_all`, which report zero when
  the identity dominates, so stars needing no smoothing sit at zero instead of
  being counted at the width of whatever sliver of Gaussian weight they carry.
  A cross-band figure asks whether the extra width is intrinsic to the sources:
  the reference band is re-measured at every other band's star positions (the
  per-band samples are each band's brightest and barely overlap), and its kernel
  width is plotted against the other band's. The extended-source cut is switched
  off for that pass, since resolved objects are exactly what would produce a
  correlation. The result is no correlation worth the name — r = +0.16 to +0.19
  for the kernel width across the MIRI bands, and -0.14 to +0.06 for the
  point-like subsets — so the extra width is a per-band PSF-model property, not
  source size.
  A note on what that pass exposed: measured with the annulus-median background
  alone, most MIRI-selected stars failed the extended-source cut in F444W. With
  the offset fitted, nearly all pass. The over-subtraction was making point
  sources look resolved.
- [x] The outer growth-curve scatter is not a background-model failure, checked
  rather than assumed. Plane, 2D-quadratic and sigma-clipped background models
  were each measured against the constant on 30 stars per band: all agree within
  the noise (F770W ratio scatter 0.0222 / 0.0223 / 0.0232, F1500W 0.0167 /
  0.0168 / 0.0169). Enlarging the ring onto a 1.6x or 2.2x stamp changes nothing
  either. What the scatter does track is star brightness — correlation -0.4 to
  -0.6 between log flux and outer scatter, with the bright half scattering 2-3x
  less than the faint half (F770W 0.0048 versus 0.0140) — while its correlation
  with the size of the fitted offset is -0.07 to -0.33. It is sky noise divided
  by stellar flux. A far-field background (clipped median at 7-19", segments
  masked) does reduce the scatter by 20-50 %, but it measures essentially zero
  and so leaves the local bowl uncorrected, and the growth curves then turn over
  again (outer COG/peak 0.92-0.97 against 1.000). The local fitted offset is
  kept, since flattening the growth curve is the point.
- [x] Blur width against source size: for every object measured in both bands,
  the MIRI diffusion width is plotted against its F444W half-light radius
  (`r50_data`, from its own growth curve normalised at Rnorm). No consistent
  trend — r = +0.18 (F770W), -0.41 (F1280W), -0.30 (F1500W), +0.11 (F1800W) —
  which agrees with the band-versus-band comparison: the extra width belongs to
  the band, not to the sources.
- [x] Comparison/diagnostic figure changes in `utils.py`:
  `CircularApertureProfile._plot_cog` marks R50 instead of R20 alongside R80 and
  annotates both with rotated labels on the lines rather than legend entries;
  `_plot_ratio` marks the normalisation radius; `compare_psf_to_star` labels the
  fitted kernel a diffusion kernel and reports its effective (second-moment)
  FWHM, from the new `utils.kernel_effective_fwhm`, as green text in the ratio
  panel. The kernel stamp panel is gone (a smooth blob shows nothing), and the
  growth-curve legend is now a three-row R50/R80 table (data, psf, psf x diff)
  with colour-matched row labels. `compare_psf_to_star` also takes `clean=False`,
  so a caller that already masked neighbours keeps its own stamp instead of
  being re-cleaned by `clean_stamp` detection, and an optional `composite` RGB
  array that becomes a fifth image panel — the notebook fills it with an
  F444W + F770W Lupton composite. Channels are built as in the scene
  diagnostics (each over its own standard deviation, high-resolution band
  block-reduced onto the low-resolution grid), but with `stretch = 1`, `Q = 30`
  rather than the scene default `Q = 8`: `Q`, not `stretch`, is what keeps a
  bright core from collapsing into a flat blob, and raising `stretch` alone only
  darkens the frame. The greyscale panels also take `vmin`/`vmax` arguments and
  default to a wider `-5.3 .. -1.0` dex ramp, so sky noise occupies less of it.
- [x] `utils.clean_stamp` no longer crashes when `detect_sources` finds nothing
  (it passed `None` into `safe_dilate_segmentation`). It now logs and returns
  the background-subtracted stamp with an empty object mask. Hit by faint
  F1500W stars through `compare_psf_to_star`.
- [x] `scratch/wren/template_comparison.tex` Figs. 2-10 rebuilt on the shipped
  code. New `scratch/wren/tmplfig/shipped.py` holds the common inputs (cutout,
  resampled PSFs, `utils.matching_kernel`, the Fig. 1 segmentation rendering)
  and an `agrees()` guard; every panel intermediate is now rebuilt from the
  numbers the schemes return and checked against the shipped composite before
  it is drawn, so a divergence raises instead of being plotted. Fig. 5 was the
  substantive change: it still showed the old segment-masked `'none'` build and
  now dissects `composite_psf_wings`. Figs. 3, 4, 7, 8, 10 gained the
  segmentation panel; Figs. 4 and 7 gained the double-counting and truncation
  panels. The detection rms is measured once per source on a fixed 12" box, so
  the same galaxy reports the same SNR (21.5 mid, 5.2 faint) in every figure.
  Captions and the affected in-text numbers ($R_{95}$ 16.5 -> 20.6 px, panel
  letters) were resynced; `template_comparison.pdf` rebuilt, 12 pages.
- [x] UDS DR0.1 F770W rerun on the current code (`psf_wings` default scheme +
  `astrom_damping = 0.8`), 2242 sources in the 0.6' trial circle, ~3 min,
  converged in 5 astrometry passes (`scratch/wren/dr0.1_run_aug9b.log`).
  Against the previous (Aug 9 00:55) run of the same config: bright fluxes
  +4% median (`flux_1` new/old 1.040, 16/84 = 1.017/1.078 at SNR>5), bright
  errors unchanged (+2-4%), faint errors +49% median (the PSF-wing extension
  enlarges small templates), SNR>5 count 106 -> 105. The scene partition
  changed (6 scenes, membership 320/920/69/667/137/128 vs
  350/974/253/330/93/241), so scene PNGs are not one-to-one between runs.
  `positivity=True` puts 1015/2242 sources exactly at flux 0, including both
  monu `nondet_good` sources, so the faint end has no negative fluxes to
  average.
  Comparison products in `scratch/wren/compare/`, rebuilt with
  `make_compare.py` plus the new `make_compare_subphot.py`: the latter renders
  `Pipeline.diagnose_subphot` (via `from_config(run_dir)` + `load_fit`, no
  re-solve) at `size=195`, `nsig=3` for the 12 monu QA sources in the trial
  circle, into `compare/dr0.1/subphot_<MONU_ID>.png`, matching the monu
  stamps' 1170x780 layout pixel for pixel. Result: the four good/mid-SNR
  sources agree with monu to a constant factor 9.06-9.38 (a catalogue flux
  unit/zeropoint offset, not a fit difference), the faint ones scatter as
  expected at their SNR, and the close pair MIN_UDS48823/48824 sits low
  (6.9-7.0), i.e. we assign ~30% more flux to each component. The bright
  edge-on disk MIN_UDS48823 leaves a much stronger residual than monu's; 2x2
  block sums cut its residual std to 0.60 of the per-pixel value (0.46 for
  MIN_UDS38103, 0.84 for a faint source), so the excess is sub-block
  checkerboard power from fitting the 80mas F770W block-replicated onto the
  40mas grid, against monu's native 40mas drizzle.
- [x] Astrometric step damping: new `FitConfig.astrom_damping = 0.8`;
  `Scene.solve` scales each pass's per-template shift increment by it before
  `apply_template_shifts` (log line reports the applied factor). Rationale:
  the central-difference shift basis underestimates gradients of sharp
  structure, so the linearized Gauss-Newton step overshoots by k/sin(k) per
  mode — harmless for convolved templates (power-weighted factor 1.1–1.3)
  but able to approach the stability limit 2 for scenes dominated by
  marginally sampled cores. Damping guarantees contraction; the fixed point
  is unchanged (synthetic check: same offset to 1e-4 px, contraction ~7x
  per pass instead of ~20x, tol reached on pass 3 vs 2 — dashed curves in
  `scratch/wren/fitfig/fig_iteration_convergence.pdf`). `fit.pdf` gained a
  section deriving the k/sin(k) overshoot, the two convergence regimes, the
  stationary-point bias, and the ranked basis/scheme fixes (incl. the
  Aitken step-scale formula); the path-independence figure was dropped
  (point retained in text). Set 1.0 to restore
  the old behavior. Improvement roadmap for the basis/scheme (tangent-
  consistent B-spline derivative, kernel-side spectral gradient, direct
  chi2 scan, Aitken step-scale estimation) recorded in the
  `scratch/wren/fit.py` module docstring and `fit.pdf` Sec. 4. Tests:
  `test_pipeline`, `test_scene_saturated`, `test_scene_max_size`,
  `test_subphot_diag` — 24 passed with damping active.
- [x] `scratch/wren/fit.pdf` (+ `fit.tex`, figures from `scratch/wren/fit.py`
  into `fitfig/`): 4-page writeup of the scene solver for talks/paper — the
  linear model and scene partition, the joint flux+astrometry solve (shift
  basis, Chebyshev field, per-template dx/dy prediction), why the linearized
  step is a smoothed operator (Fourier transfer functions: Taylor truncation
  + central-difference low-pass vs measured cubic-spline response; 2D
  anatomy figure, linear error 5% of peak vs spline 2e-3 at 0.5 px), the
  iterate-linearize-reshift loop (TikZ flow diagram; synthetic convergence:
  increment decays ~20x/pass, injected (1.5,-0.8) px recovered to 1e-4 px),
  path independence of the applied-shift operator (single vs incremental
  spline MTF erosion), and why (dx, dy, flux) per template is the complete
  fit state (persisted-products table, offline reconstruction).
  Documents-only; no package code changed.
- [x] Repeatable run configs + fresh-session resume. `Pipeline.save_config()`
  writes the fully-explicit run config to `out_dir/<name>.json`: every
  `RunConfig` field and every *used* `FitConfig` setting with its resolved
  value, so the run stays repeatable even when code defaults change (e.g. the
  recent `template_dilate_segmap` 2 -> 0). Settings of template build schemes
  the run did not select are omitted (`wren_*`/`classic_*` pruned by
  `extend_mode`). `run()` snapshots automatically before fitting. `Pipeline.from_config` now also accepts a run *directory*
  (`<dir>/<dirname>.json` preferred, else the single `*.json` it contains,
  ambiguity raises), so `Pipeline.from_config("uds_770_dr0.1/")` resumes from
  the outputs. New `Pipeline.load_outputs()` loads a finished run's products
  (fit table + residual, memmapped) into a fresh session for catalog-level
  diagnostics; `repr` shows `[fitted]`, `info()` lists output-product
  presence (`out config/table/residual`). `write_outputs`/`load_outputs`/
  `info` share new `f_config`/`f_fit_table`/`f_residual` path properties.
  Generated `examples/uds_770_dr0.1/uds_770.json` from the current
  `examples/uds_770_dr0.1.json` (includes the new `wht_hi`, `extend_mode`,
  `wren_*`, `classic_*` fields) and verified resume on the real run: 2242
  rows, 106 at SNR>5, 25344x34560 residual. Tests: +4 in
  `tests/test_pipeline_config.py` (full-snapshot fields, directory
  resolution, resume, ambiguity) and +1 in `tests/test_pipeline_inspect.py`
  (run() snapshots before fitting); 26 passed across the three suites.
  Follow-up: full fit state is now persisted and image diagnostics work
  offline. `write_outputs` also writes `<name>_templates.fits`
  (`_template_fit_table`: per final template id, id_parent, x, y, fitted
  shift dx/dy, flux, err, scene id) — the only solve products a
  deterministic template rebuild cannot re-derive; everything else
  (raw/extended/convolved template shapes) reconstructs from sci_hi + segmap
  + config + the cached PSF/kernel maps, and the total model is
  `image - residual`. New `load_fit()` restores image-based state in a fresh
  session: loads data, re-upsamples the fitted band onto the reference grid
  exactly as `run()` does, derives `model_images` from the saved residual,
  and loads the template table. `diagnose_subphot` and `diagnose_sources` then
  work without a solve — each source's convolved template is rebuilt on
  demand (`_rebuild_source_stage_templates`) and the saved flux/shift
  applied; `diagnose_subphot`'s seg colouring falls back to template-table (or
  catalog) positions. Verified: resumed-session `diagnose_subphot` reproduces
  the in-session render bit-identically on the img/tmpl/seg/model/res
  panels (clean differs only through the rebuilt-template spline;
  `test_template_fit_table_and_resumed_render`), plus an end-to-end
  files-only resume test (`test_load_fit_offline_diagnostics`); 31 passed.
  Note: runs finished before this change have no `_templates.fits` —
  `load_outputs` warns, and resumed renders then use catalog fluxes with
  zero shift.
- [x] Template build schemes selectable with one knob, `FitConfig.extend_mode`
  (`'default'|'psf'|'psf_model'|'wren'|'classic'`), so the three codes of
  `scratch/wren/template_comparison.pdf` can be compared 1-1 on identical
  inputs. `Templates.extend_with_psf_wings` renamed `extend_with_psf` (it is a
  convolution with the PSF, not a scaled-PSF paste); `extension_mode` on those
  templates is now `"psf"`.
  * `'wren'` and `'classic'` are **build-time** schemes: their composite
    replaces the segment-masked data inside `Templates.extract_templates`,
    before the unit-sum normalisation, so `template_norm` covers the extended
    shape. `'psf'`/`'psf_model'` stay post-extraction passes (they reshape the
    cutout), leaving `templates_extracted` / `templates_extended` meaningful.
  * Ports live in the new `src/mophongo/template_schemes.py`, self-contained
    and importing nothing from the fit/catalog/pipeline layers: pure numpy in,
    `(composite, info)` out. Dispatch is one block in `extract_templates` plus
    `Pipeline._extend_scheme_kwargs`, so either scheme can be adapted or
    deleted as a unit.
  * `classic` = IDL `subphot.pro::build_cube` (:294-330) from the canonical
    source `~/Documents/Astro/PROG/idl/ifl/pro/fitphot/subphot.pro`:
    `m = S.D + f_psf (1-S).P` with `f_psf = sum_S P D / sum_S P^2` floored at
    0, the `f_psf<=0` bare-PSF branch, and the hard replacement by a point
    source below `tmpl_snrlo=15` measured against `robust_sigma` (astrolib
    biweight, ported). No dilation, no positivity clip, wings pasted over
    neighbouring segments too. Cutout floored at the detection PSF stamp — the
    resampled PSF is identically zero beyond it, so that footprint *is* the
    support IDL's whole-tile paste produces. Records IDL's log columns
    (`fpsf`, `flux_in_seg`, `added_flux`) in `Template.extend_info`. IDL's
    step 7 (normalise the *convolved* plane, then `apermask` at
    `ceil(ksz/2)`, `subphot.pro:324`) belongs to the convolution stage and is
    deliberately not reproduced.
  * `wren` = `dev-wren:templates.py::_extended_composite`: global
    area-weighted `build_ownership` (disk contest, ROI-restricted), support
    `own | (owned background within R95)`, one core weight
    `w(S_seg; 1.5*fit_snrlo_psf)` and one weight per 0.15" halo annulus forced
    monotone non-increasing outward and seeded at the core, blend
    `H = W D + (1-W) A_src P`, positivity clip *before* `template_norm`, plus
    `snr_seg`/`A_src`/`f_cut`/`flux_beyond_stamp`/`flux_beyond_aper`
    bookkeeping and `FLAG_PSF_EXTENDED`/`FLAG_EXTEND_FAILED`. Sizing chain
    (`r_fill = max(R95, r_aper + kernel_hw)`, `min_size = 2*ceil(r_fill)+2`)
    reproduced in `Pipeline._extend_scheme_kwargs`.
  * `Template.template_norm` is now recorded on **every** path (the one
    behaviour change to `'default'`), so the detection-band flux a template
    implies is reconstructable — `template_comparison.tex` Sec. 7 listed its
    absence as an ivo defect.
  * `Pipeline.load_data` now passes `psfs=[prm_hi, prm_lo]` instead of
    `[None, prm_lo]`, and `_ensure_maps` loads the cached `prm_hi` (it only
    ever loaded `prm_lo` + built the kernel map, so `psfs[0]` stayed `None`
    even after the `load_data` change). `_psf_for_template_extension` is now
    **strictly `psfs[0]`** and raises otherwise: the old fallback to `psfs[1]`
    is gone. In the two-image config layout `psfs[1]` is the **low-res** PSF,
    so `wren`/`classic`/`psf` would have built wings from the wrong band and
    derived `r_fill`/`R95` in lo-res pixels while applying them as hi-res
    radii, with no error. There is no correct fallback for a detection-band
    PSF, so the path is disallowed rather than substituted; on a config-driven
    run the cached detection map is built/loaded on demand instead.
    `verification.py` updated to pass the detection map at index 0. No
    `RunConfig` field reached the extension branches at all before; with
    `extend_mode` in the `fit` dict the whole family is now reachable from a
    JSON run. Only fitted bands (`ifilt >= 1`) feed throughput/PSF-EE
    bookkeeping, so index 0 is inert there.
  * `RunConfig.wht_hi` supplies the detection weight map, so the config-driven
    path gets `weights[0]` and neither scheme needs a global noise scalar.
    Every run must have one: `Pipeline.resolve_wht_hi()` takes `wht_hi` when
    set, else derives it from `sci_hi` by the grizli `_sci.fits` ->
    `_wht.fits` naming, and **raises** when neither resolves rather than
    degrading to one sky-sigma scalar for the whole mosaic. The pixels are
    read only when `extend_mode` is `'wren'` or `'classic'` —
    `'default'`/`'psf'` never touch `weights[0]`, and a full-field hi-res
    weight map costs as much memory as the mosaic.
    `Pipeline._load_detection_ivar` rescales it with the same
    `get_bg_and_ivar` the lo-res side uses. All four example configs now name
    `wht_hi` explicitly; verified end to end on COSMOS DR0.1 (32768x18944,
    99.8% covered, median ivar 431).
  * `RunConfig.driz_hi` removed. It was the DrizzlePSF footprint/grid source,
    defaulting to `sci_hi` and null in every config ever written. `DrizzlePSF`
    reads the header *and the pixels* of `driz_image` (`get_psf_radec`
    centroids on them), so pointing it at a different file than the one being
    fitted was the risky option, not the safe one — for a repaired mosaic the
    repaired pixels are what you want. One pointer, `sci_hi`.
  * Noise: with a detection inverse-variance map (`weights[0]`) **neither**
    scheme uses a global scalar — both take the formal `sqrt(sum 1/ivar)` over
    the mask. That is wren's primary path already; classic now shares it,
    since IDL's `sum_S D / (sqrt(n_S)*tmpl_rms)` is exactly that expression
    for uniform noise (IDL carries no ivar map at any stage and fits with
    `wts = 1/rms`, `subphot.pro:541,869`, so one scalar per tile was the best
    it could do). Test:
    `test_classic_ivar_noise_reduces_to_idl_scalar_for_uniform_weights`.
    The scalar fallbacks are for runs without detection weights, and are now
    measured over `covered_mask` (finite and non-zero): over a mosaic
    `load_data`'s `nan_to_num` turns the out-of-footprint margin into exact
    zeros, and that spike drags the MAD down — 0.05 -> 0.019 at 61% blank,
    exactly 0 at ~50%. Zero would have silently disabled IDL's low-SNR branch
    and driven every wren blend weight to 0, i.e. every template a bare point
    source with `extend_failed` False. Both schemes now raise instead of
    degrading silently (`WrenParams.bg_rms` / `FitConfig.wren_bg_rms` and
    `ClassicParams.rms` override; `classic_tmpl_snrlo=0` skips the branch as
    IDL's `keyword_set` guard does).
  * `Templates.extract_templates(dilate_segmap=...)` and
    `verification.run_pipeline_extension_scenario(template_dilate_segmap=...)`
    now default to **0**, matching `FitConfig.template_dilate_segmap`. The
    verification default was 4. (`mock_dilate_segmap` is unchanged: it grows
    the truth segmap, not a template.)
  * Verified on a synthetic faint source (fraction of a point source the
    support can hold): `default` 0.40, `wren` 0.955 (R95 cap), `classic` 1.00,
    `psf` 1.00; classic `added_flux` 2.49 and `f_psf` within 1% of the true
    amplitude — matching `template_comparison.tex` Fig. 1.
  * Tests: `tests/test_template_schemes.py` (36) + 2 in
    `tests/test_pipeline_inspect.py`. Suite: 157 passed +
    pre-existing `test_moffat_flux_recovery[psf_wing-3-psf]` failure. The three
    detection-PSF / noise-scalar defects above came out of an adversarial
    review of both ports against their reference sources (43 agents, 3
    confirmed of 20 raised).
- [x] Test suite fully green for the first time (118 passed, 0 failed).
  Two long-standing failures fixed, neither caused by the cleanup:
  `test_moffat_recovery[psf_wing-3-psf]` passed `kernels=` but never `psfs=`
  to `pipeline.run`, so the `extend_templates='psf'` branch hit
  "requires a high-resolution PSF in psfs[0]" and raised; the other four
  scenarios use `extend=None` and never enter that branch. The test now
  passes `fit_psfs = [psfs[0], psfs[0], psfs[1]]`, matching its
  `[hires, hires, lowres]` image list. `test_catalog_from_fits_smoke` was
  removed: it reads `data/uds-test-f444w_{sci,wht}.fits`, and CI checks out
  without `lfs: true`, so those arrive as LFS pointer stubs and astropy
  raises "No SIMPLE card found". It passed locally and failed on every CI
  run. Adding `lfs: true` to `actions/checkout` in `.github/workflows/ci.yml`
  would be the alternative fix if the coverage is wanted back.
- [x] Dead-code cleanup (branch `cleanup`, off `main` @ d30be0f, tagged
  `pre-cleanup`). `src/mophongo` went from 24,143 to 20,434 lines. Scan
  method and full inventory: `scratch/CLEANUP_SCAN.md`.
  Removed modules: `deblender.py`, `sim_data.py`, `astro_fit.py` (all three
  reachable only through commented-out imports) and `photutils_deblend.py`
  (unreferenced, and the source of three private-photutils imports flagged in
  GUIDE.md). `__init__.py` now re-exports `photutils.segmentation.deblend_sources`
  with no commented alternatives.
  `fit.py` lost the 328-line block under its own `# OBSOLETE BELOW` banner
  (`assemble_scene_system_old` referenced an undefined `tmpl_j`, so it could
  never have run), the orphaned module-level `build_normal_matrix(self)` that
  `build_normal()` dispatched to for `normal != 'tree'` (never bound to the
  class: that config raised AttributeError), the unused
  `sparse_cholesky`/`make_sparse_chol_prec` pair, and its duplicate copies of
  `build_scene_tree_from_normal`, `merge_small_scenes` and `summarize_scenes`
  (`solve_scene` now imports the `scene.py` versions lazily; `summarize_scenes`
  was byte-identical, `merge_small_scenes` differed only in a default).
  `build_normal()` and `solve()` now raise `ValueError` for the config values
  whose implementations never existed (`normal != 'tree'`,
  `solve_method != 'scene'`).
  The `run_scene_solver=False` "legacy solver" branch in `Pipeline.run()` is
  gone; the scene solver is the only fitting path and the flag now raises if
  set False. `Pipeline._add_templates_for_bad_fits` (its only call site was
  commented out), `_per_source_chi2` and `_stamp_slices_for_templates` went
  with it.
  The truncated-FFT kernel feature was inert end to end and is removed:
  `Templates.prepare_kernel_info` was the only writer of a non-zero
  `ee_rlim`, nothing called it, so `_crop_kernel`, `_prepare_fft_fast`, the
  `ee_rlim`/`ee_fraction` template attributes and `FitConfig.fft_fast` are
  all dropped. `FitConfig.block_size` was likewise never read.
  `utils.py` lost the unused PSF-basis zoo (`difference_of_gaussians`,
  `gaussian_laguerre`, `zernike`, `radial_bspline`,
  `positive_monotone_radial_bspline`, `starlet`, `eigen_psf`, `powerlaw`),
  `regularized_pixel_kernel_central`, `regularized_lstsq_kernel_central` and
  `CircularApertureProfile.moffat_fit`/`.moffat_fwhm`.
  `psf.py` lost `to_header` and `get_slice_wcs`, byte-identical copies of the
  `utils` functions it already imported (the local defs shadowed the imports);
  `saturate.py` now takes `get_slice_wcs` from `utils`. Its dead profile-fit
  cluster (`_fit_profile`, `fit_moffat`, `fit_gaussian`, `GaussianFit`,
  `MoffatFit`) is gone too.
  Also removed: `verification.plot_lw_coadd_diagnostic`,
  `astrometry.measure_template_shifts_old` with its now-orphaned
  `make_gradients`/`basis_matrix`, `PSFRegionMap.lookup_key_slow`,
  `Scene.augment_templates`, `Template.downsample_wcs_old`,
  `Template.block_aligned`, five dead `catalog.py` helpers plus its
  copy-pasted second definitions of `_mean_downsample` and `_expand_remap`
  (the later definitions shadowed the earlier ones) and the repeated mid-file
  import blocks in `catalog.py`, `scene.py` and `templates.py`, ~25 unused
  imports, ~110 lines of commented-out code, six tracked `.DS_Store` files
  and two zero-length `legacy/autopilot/.pdf` / `.pro` artifacts.
  Up/down-sampling was preserved as requested: `Template.downsample`,
  `project_to_block_replicated_grid`, `AlignedCutout.downsample`/`upsample`,
  `as_block_reduced`/`as_block_replicated`, `utils.rebin_wcs`,
  `downsample_psf`, `bin_factor_from_wcs` and both
  `multi_resolution_method` modes all stay. Helpers used only from `scratch/`
  and notebooks were kept and logged in TODO.md as untested public API.
  Second pass, on request: the legacy `SparseFitter` solver is gone too.
  Removed `SparseFitter.solve`, `.solve_scene`, `._solve_scenes_with_shifts`,
  `.fit`, `.flux_errors`/`._flux_errors`, the module-level
  `make_basis_per_scene`, `assemble_scene_system_self_AB`, `solve_scene_cg`
  and `_diag_inv_hutch` that only served them, `scene.summarize_scenes` (its
  last caller was `solve_scene`), and the dead-compute block in
  `SparseFitter.__init__` that built an S/N array and discarded it.
  `fit.py` is 1,686 -> 378 lines. `SparseFitter` is now a normal-matrix
  builder with model/residual images and covariance-free flux/error
  estimators; all flux solving goes through `SceneFitter`.
  `FitConfig` lost seven knobs nothing reads any more: `solve_method`,
  `fit_covariances`, `multi_tmpl_chi2_thresh`, `multi_tmpl_psf_core`,
  `multi_tmpl_colour`, `scene_merge_small`, `negative_snr_thresh`. The only
  files that passed them were four `scratch/claude_audit/v1/repros/` scripts
  (B020, B024, B085, B088), which will now raise `TypeError` if re-run.
  Tests: `test_fit.py::test_solve_scene_matches_global` deleted (it compared
  `solve()` against `solve_scene()`, i.e. the same code path);
  `test_scene_fitter.py::test_scene_solve_matches_legacy_solver` became
  `test_scene_solve_matches_dense_normal_solution`, checking `Scene.solve` on
  real templates against a dense solve of the same regularized system;
  the two `test_astrometry.py` tests that called `fitter.solve()` now solve
  with `SceneFitter.solve(fitter.ata, fitter.atb)` and write flux/err back
  onto the templates via a local `_assign_solution` helper (the removed
  `solve_scene` used to do that write-back, and `measure_template_shifts`
  gates on `tmpl.flux / tmpl.err`); `test_pipeline_multitemplate.py` dropped
  the now-nonexistent `multi_tmpl_chi2_thresh` kwarg.
  Suite: 118 passed + the pre-existing
  `test_moffat_recovery[psf_wing-3-psf]` failure (119 before, one test
  deleted).
  Third pass: the untracked `scratch/` tree was audited against the cleaned
  API. Static check of all 176 scripts (import resolution, module-attribute
  access, dataclass kwargs) plus an execution run of all 105
  `claude_audit/v1/repros/*/repro.py`. Result: 18 repros failed, 16 of them
  before reaching their subject. Repaired in place: `run_saturate.py`
  (docstring pointed at the retired `jwst_psf.psf_grid_from_csv`; now names
  `PSFFactory.from_csv` and warns that `PSF_FILTER`/`PSF_FILTER_LARGE` keep
  the legacy `..._OS4_GRID{N}` naming), `moffat_recovery/ab_with_wings.py` and
  `claude_audit/v1/repros/B069/repro.py` (both imported a class attribute as a
  module-level name — `Templates.extend_with_psf_wings` and
  `Template.FLAG_CONVOLVED` — and had been broken since they were written).
  Moved to `scratch/_obsolete_2026-08-09/` with a README explaining each:
  fifteen audit repros whose subject no longer exists, `build_large_psf.py`
  (superseded by `build_psfs_mjd_grid.py`), and
  `wren/uds_wren/run_uds_770_wren.py` (a byte-identical copy of the tracked
  `examples/run_uds_770_wren.py`, which targets the `wren/dev-wren` fork API
  and has never been runnable in this tree). Re-verified: the 90 remaining
  repros run, except B014 and B105, which fail inside live code — that is the
  bug each demonstrates, not staleness.
- [x] Astrometric anchor selection unified (post-merge follow-up):
  `astrom_isolation_thresh` default 0.5 -> 0.7 (wren value); new
  `astrom_exclude_stars: bool = False` makes star exclusion opt-in at both
  the solve-time bright mask and the merge-time mask (flips dev-wren's
  unconditional `& not_star`; unsaturated stars are the best anchors and
  saturated ones are already isolated into their own scenes). Isolation now
  also applies at merge time: `generate_scenes(isolation_thresh=...)` folds
  `_astrom_isolation_mask` into the bright mask counted by
  `merge_small_scenes`, so `scene_minimum_anchors` counts bright & isolated
  (& star-policy) sources and merged scenes are guaranteed usable anchors —
  the solve-time "astrometry skipped" branch becomes unreachable in
  practice. The full-field normal matrix makes merge-time dominance
  stricter than the per-scene solve-time one (out-of-scene neighbours still
  count). Test: `test_isolation_thresh_counts_only_isolated_toward_floor`.
  Suite: 119 passed + pre-existing moffat failure.
- [x] IDL subphot diagnostic port: `Pipeline.diagnose_subphot(source_id)` renders
  the legacy `subphot.pro::mkdiag`/`fptv` 6-panel PNG (img/tmpl/seg/model/
  res/clean, 2x3 at 2x nearest-neighbour zoom) pixel-for-pixel for 1-1
  comparison against IDL outputs (e.g. `scratch/wren/compare/monu/*.png`).
  Reproduced exactly from the canonical source
  (`~/Documents/Astro/PROG/idl/ifl/pro/fitphot/subphot.pro`): IDL `bytscl`
  (`floor(255.9999*frac)`), img/model/clean at `+-nsig*prms`, res at
  `(img-model)/err*(1-mask)` with `+-nsig` and
  `err = sqrt(prms^2+(sys_err*model)^2)` (sys_err 0.02), tmpl at
  `median +- 8*robust_sigma` (Tukey biweight c=6), seg = distance-sorted
  5-level gray cycle [0.2,0.8,0.4,0.6,1.0] minus `0.1*mask` at bytscl
  [-0.2,1] (unfitted segments keep raw ids -> clip white), circular `rlim`
  fit mask, and the `prms` estimator (aperture-scale `na = floor(raper*
  sqrt(pi)/sqrt(2))` block sums of masked residual background, 2-sigma
  clipped, `prms = rms/na`, block median re-subtracted from the display
  stamps). Defaults follow the survey-era run config (`phot.param`):
  `nsig=3` (`subphot_nsigma=3`; IDL code default was 5), `maskhi` off,
  `photbin=1`. Optional SNR-preserving display binning kept as `photbin`.
  White DejaVu-Bold labels at the IDL xyouts positions (+5, 20 below panel
  top). Requires fit grid == reference grid (upsample path). Tests:
  `tests/test_subphot_diag.py` (4: bytscl vs IDL values, fptv binning/zoom,
  full-render layout + masked-corner grays + labels, defaults/errors).
- [x] `template_comparison.pdf` v5 (12 pp): naming finalized — current python
  is **ivo** (memory `mophongo-version-names`), IDL = the `subphot.pro` run
  path only (rewrite behaviours purged from IDL descriptions: quadrature floor
  and competitive dilation are labelled "the rewrite (never ran)" everywhere,
  including the priors figure). Sections merged: single "How the template is
  constructed, end to end" with step-by-step + flow diagram + dissection
  figure per method (`fig_method_flows`, `fig_idl_dissect`, `fig_ivo_dissect`,
  matching the wren anatomy style, all on the same real mid source). Fig. 1
  support insert now lists segment | IDL | ivo | wren.
- [x] `FitConfig.template_dilate_segmap` default changed 2 -> 0 (`fit.py:105`).
  The IDL reference run path (`subphot.pro::build_cube`) uses the exact
  segment; dilation only adds a ring of sky noise, and
  `safe_dilate_segmentation`'s contested-background tie-break is catalog-id
  ordered. Wing recovery belongs to template extension, not dilation.
  41 tests pass (`test_pipeline*`, `test_fit`, `test_template_convolution`).
  `template_comparison.pdf` updated: "current python" renamed to **ifl**
  throughout, ifl column/figures regenerated with dilation off (faint-source
  support EE 0.69 -> 0.44).
- [x] Pre-run data loading + pipeline inspection. `Pipeline.load_data` gained
  `kernels=False` to load/preprocess images+catalog without touching the
  PSF/kernel maps (`run()` finishes them later via the new `_ensure_maps`
  helper, which also replaces the duplicated map-loading block in
  `load_data`). New `Pipeline.info()` prints a stage-aware summary: before
  loading it reports each input file's existence, size, shape/row/frame counts
  from headers only (no pixel data read); after `load_data` the image shapes,
  pixel scales, weights, segmap, catalog columns, and region maps; after
  `run()` the fit table and scenes. New `Pipeline.plot_inputs()` quicklook
  (hi-res sci with catalog overlay, lo-res sci, weight, segmap) and a concise
  `__repr__` (`<Pipeline 'uds_770' [configured|loaded|fitted] images=N
  sources=N>`). CLI steps `load` and `info` added
  (`python -m mophongo.pipeline cfg.json info`). Tests:
  `tests/test_pipeline_inspect.py` (9); `test_pipeline.py` +
  `test_pipeline_config.py` unchanged (19 passed).
- [x] `scratch/wren/template_comparison.pdf` (+ `.tex`, figures from
  `scratch/wren/tmplfig/mk_*.py`): 10-page explainer comparing template
  construction in IDL classic (`subphot.pro::build_cube` — verified as the code
  that produced the `uds_monu` QA run; `mophongo__define.pro` is an unfinished
  rewrite that never ran), current python, and the wren fork. Verified: the run
  path has NO segmap dilation (plain `tseg eq id`; competitive dilation
  `kseg > knn` exists only in the rewrite), the low-SNR PSF replacement WAS
  active (`phot.param:91 tmpl_snrlo = 15.0`, confirmed from LOG arithmetic —
  same threshold as wren's core-weight saturation), and IDL fits on the
  detection grid with one global rms. New "How wren determines a profile" deep
  dive: sizing chain diagram, 8-panel anatomy of `_extended_composite` on a
  real neighbour-truncated source, faint-limit blend takeover, and a knob
  reference (two knobs confirmed dead). Numerical rulings: our k² ivar
  convention is correct (wren errors x k/kappa: 2.0 isolated, 0.65 heavily
  blended — no scalar repair possible); missing block projection biases wren
  fluxes −4.9%. Documents-only; no code changed.
- [x] Canonical dr0.1 run at the new defaults (psf_size 4.0, scene_max_size
  500, floor 1e-3, r_trial 0.6', 2242 sources; stale 8" PSF/kernel caches
  cleared first): local split threshold 0.0379 confined to the one giant,
  674 tree scenes (max 494) -> 6 final scenes 350(5)/974(18)/253(8)/330(8)/
  93(8)/241(13). Scene 2 (974) exceeds the cap because the original merge is
  capless — under-bright fragments re-fuse past 500; acceptable soft-cap
  behaviour, noted for review. Fitted fluxes agree with the ps3 run (psf 3.0,
  cap 300) to med ratio 0.9998 at SNR > 10 (nmad 0.17 sigma, n=71).
  `scratch/wren/compare/` rebuilt from this run (6 scene pngs).
- [x] Defaults adopted after review of the ps3 result: `scene_max_size = 500`
  (FitConfig) and `psf_size: 4.0` in `uds_770_dr0.1.json` (RunConfig default
  was already 4.0; the json had overridden it to null/8" stamps). Smaller
  stamps both improve the fits and thin the coupling graph. EE inside the
  max inscribed circle (r = width/2, from the latest MJD UDS ePSF grids,
  normalized to full-stamp sum): F444W 0.983/0.996/— (4.1" grid stamp),
  F770W 0.909/0.935/0.994, F1280W 0.918/0.953/0.995, F1500W
  0.917/0.946/0.994, F1800W 0.918/0.932/0.994 for 3"/4"/8" widths; F1000W
  (no UDS grid, STPSF model) 0.942/0.959/0.996.
- [x] Scene size cap landed as one knob: `scene_max_size` (FitConfig,
  default None = original behaviour; now 500, see above). A component over the cap is split by
  bisecting over its own edge scores — threshold raised only inside that
  component, local leakage logged, rest of field untouched at the 1e-3
  floor. Original merge untouched. The wren discrepancy resolved first:
  coupling range = segment (+) kernel stamp = `psf_size`; our `null` = 8"
  stamps percolate (one 2241-template scene at 0.6'), wren's 3" reproduces
  wren exactly — including a 1738-template giant in this deepest-region
  patch, so wren's ~200-source scenes were typical-field, not algorithmic.
  On the 3" graph the giant breaks at local threshold 0.0289 (vs 0.16 at
  8"), below the 0.044 level where partition changes left fluxes unchanged
  (nmad 0.13 sigma). `uds_770_dr0.1_ps3.json` (psf_size 3.0, cap 300,
  r_trial 0.6') -> 9 compact scenes, 75-628 templates, 5-8 bright each.
  Adopting 3" stamps for science awaits the `flux_<i>_total` stamp-sum fix
  (+4.9% bias at 3.0", `TODO.md`). Tests: `tests/test_scene_max_size.py`
  (ceiling-not-target, other components untouched). `scratch/wren/compare/`
  now built from the ps3 run (9 scene pngs vs 5 wren + 12 monu).
- [x] Scene-partition experiments, reverted to the original two-step
  algorithm (coupling-threshold components + merge-small-scenes). A sequence
  of size-driven designs (`scene_max_size` with global/local threshold
  bisection, then bright-balanced, then coupling-guided merging) each fixed
  the previous one's failure but produced non-compact, oddly shaped scenes on
  the UDS F770W dr0.1 patch; per "make it simpler, not more complex" the
  original algorithm was restored (experimental code in `git stash`:
  "session scene-partition experiments (reverted)"). Durable measurements
  kept in `docs/SCENE_PARTITION.md`: the per-band threshold ladder collapses
  to 0.030 +/- 0.005 after dividing by SNR and PSF area (percolation + 1/SNR);
  the 8" `psf_size: null` stamps make the coupling graph percolate at any
  usable threshold while wren's 3" stamps ran fixed 1e-3 (template support is
  the scene-size lever, not the partition algorithm); partition changes at
  cut levels <= 0.044 leave photometry unchanged (nmad 0.13 sigma), while
  breaking the F770W giant (thresh ~0.1-0.16) moves bright fluxes 1-3%;
  elongated scenes are internally misaligned because offsets vary on ~arcmin
  scales. `uds_770_dr0.1.json`: `r_trial` 0.5' -> 0.6' (2242 sources),
  fit overrides back to `{fit_astrometry_joint, scene_minimum_anchors: 5,
  aperture_diam}`. `scratch/wren/compare/` regenerated by
  `scratch/wren/make_compare.py` at the 0.6' circle.
- [x] flux-estimator note v3 (`scratch/wren/flux_estimator_comparison_v3.tex`,
  compiled over the unversioned pdf; v1/v2 preserved): corrects v2's claim that
  IDL's overlapping wings are benignly apportioned by the solve. Data-derived
  cores carry neighbour wing light, so both wing placements fit a wrong profile
  — structured bridge residuals + biased amplitudes, worst for
  faint-beside-bright; region-integrated algebra shows IDL's contaminated wing
  normalisation cancels in the unit-sum (coarse allocation exact, bias
  intra-segment only) while background-only fill misallocates the cross-segment
  light at first order. Fix documented: subtract neighbour wing models at
  extraction, renormalise, then fill wings everywhere. Also added: low-SNR
  blend engagement numbers (quadratic w has 20% PSF at 2*S0 vs IDL quadrature's
  ~12%; `fit_snrlo_psf <= 0` disables) and the operating decision deferring the
  residual-region question for totals. COSMOS n3.0_v1.3 check: catalogue
  `tot_cor` = (fauto_KRON/faper_KRON) x 1/EE_f444w(kron_radius_circ) with ~1%
  scatter at fixed radius — decomposable and consistent, safe to adopt
  wholesale. `docs/FLUX_ESTIMATORS.md` (new section 2.3) and `TODO.md` updated
  to match. Documents-only; no code changed. Follow-up edits after review:
  "wings everywhere" renamed **full-complement fill** (wings never replace the
  source's own positive segment data — the fill region is the complement of
  `kseg ∧ data>0` within the stamp; "everywhere" wrongly implied inside own
  segment); and a wing-construction comparison added (IDL tile vs ifl
  `psf_wings` self-convolution vs wren radial blend) — the ifl construction
  convolves segment data with the PSF, i.e. wings follow PSF^⊛2, over-broad
  worst for compact sources, converging to the tile only far-field; wren
  (FORK_DIFF_WREN §4) blends data with a point-source tile per annulus.
  Terminology adopted: ifl = this repo, wren = dev-wren fork.
- [x] `docs/FORK_DIFF_WREN.md`: active-path comparison of
  `wrensuess/mophongo@dev-wren` against `flux-bug`, scoped to
  `examples/run_uds_770_wren.py` (fork) vs `run_uds_770_dr0.1.py` +
  `uds_770_dr0.1.json` (ours). Covers orchestration, `run()` step order,
  solver, template extension, PSF encircled-energy bookkeeping, and the output
  column sets. Two defects found on our side and recorded in `TODO.md`:
  `flux_<i>_total` divides by the low-res stamp sum when the unnormalized
  kernel's DC means the fitted amplitude carries the *detection* stamp sum
  (−0.7% at `psf_size: null`, +3.1% at the `RunConfig` default 4.0"), and
  `PSFSZ<i>`/`RCIRC<i>` are half their true value because `wcs[ifilt] =
  wcs[0]` aliases `self.wcs` before `_record_psf_ee` reads it. Fork-side
  findings include the merge-base kernel-region lookup bug that ours already
  fixed via `wcs_original`, `wcslin_pscale=1.0` masked by `renormalize=True`,
  an `eval_ePSF` 0.375-native-px centring error on even grids, and
  `containment` measured against the parent grid's own sum rather than the
  true total (+2.9% F444W / +1.5% F770W on every EE it feeds).
  Documents-only; no code changed.
- [x] `scratch/wren/flux_estimator_comparison_v2.pdf` (+ `_v2.tex` source, v1
  preserved as `_v1.pdf`): rewrite of Monu Sharma's estimator note. Adds the
  collapsed identities (`f2 = A + sum_Omega(res)` exactly, so Estimator 2 needs
  no aperture photometry at all), a table of the several PSF-EE corrections and
  the `apcor1`/`totcor1` naming trap, what the legacy IDL did (competitive
  dilation, wings over neighbours, low-SNR PSF prior at template SNR 15), the
  python column mapping including its `R_cat != R_img` generalisation, the
  residual-region choice, and an assessment against catalogue-matching and
  low-SNR robustness. Key conclusions: the total correction cancels in colour
  only if the same factor is used for all bands, so totals should use the
  catalogue's `tot_cor`; and bounding `ap_hi`/`ap_lo` by the PSF EE removes the
  40x correction tail without touching templates.
- [x] `docs/FLUX_ESTIMATORS.md`: analysis of the three total-flux estimators in
  `scratch/wren/flux_estimator_comparison.pdf` against the legacy IDL in
  `legacy/autopilot/mophongo__define.pro`. Establishes that (a) current python
  computes the PDF's `fcor1`, i.e. Estimator 1 stopped one factor short of
  total; (b) IDL templates were never truncated at the segmap — PSF wings were
  pasted outside every segment (`:218-219`) with a low-SNR quadrature PSF floor
  (`:215-216`), and IDL `apcor` was a pure detection-PSF EE at the Kron ellipse
  (`:1300-1303`); (c) python's inflated corrections for small segmaps are a
  template-truncation artifact, and `extend_templates` is currently unreachable
  from `RunConfig`, so both MINERVA runs ran truncated. Concludes the estimator
  choice is second-order to fixing templates. Documents-only; no code changed.
- [x] `docs/PHOTOMETRY_APERTURES.md`: technical explainer for the two photometry
  schemes (template-fit `flux_<i>`/`flux_<i>_total` with PSF throughput, and
  aperture photometry on model+residual with the `ap_*` columns), the two
  aperture radii and the grids they live on, and the exact set of aperpy catalog
  columns/meta the corrections require. Documents-only; no code changed. Notes
  four latent issues for later: `KERNEL` vs `sci_hi` consistency is unasserted,
  the `None` `r_cat` default mixes grids in the `downsample` path (the
  `upsample` path is correct via the `wcs[ifilt] = wcs[0]` rebinding at
  `pipeline.py:1207`), the documented 1.5xFWHM `r_cat` fallback is dead code,
  and `ap_flux_*` has no error column. Proposed refactor sketched in section 7.
- [x] UDS F770W run on the wren-era inputs, as a config-driven run:
  `examples/uds_770_dr0.1.json` + `examples/run_uds_770_dr0.1.py` reproduce
  `examples/run_uds_770_wren.py` with the modern `RunConfig` path and the same input
  data — grizli v8.0 `minerva-v3.0` 40 mas F444W, `uds-sbkgsub-v3.0` 80 mas MIRI F770W,
  `n3.0_v1.2` ACS+WEBB chi-mean segmap, `n3.0_m3.1_v1.2.1` SUPER catalog (wMIRI).
  Settings mirror `cosmos_770_dr0.1.json` so the two same-generation runs are
  comparable (`psf_size` null, blur "default", `fit_astrometry_joint`,
  `scene_minimum_anchors` 10, `aperture_diam` 0.5, `r_trial` 0.5'); frame counts
  297 (F444W) / 229 (F770W). Trial patch `[34.34914, -5.27462]`, the deepest fully
  F770W-covered 0.5' patch of the v3.0 mosaic (median wht 1.30e8, 2.0x the footprint
  median), near the DR0 patch `[34.4, -5.26]`.
  Two deliberate departures from the wren script: the template is the raw
  `_drc_sci` mosaic, not the aperpy `_sci_f444w-matched` image (not published on S3 or
  Drive, and the modern path drizzles its own ePSFs); and PSFs come from the MJD-tagged
  `UDS_..._MJD*_GRID25/9_OS4` grids rather than the static `OS4_GRID25` ones.
  Data provenance: the wren inputs lived on `/Volumes/DarkPhoenix` and were re-fetched
  from the master records — NIRCam from `s3://grizli-v2/MINERVA/mosaics/uds/40mas-v3.0/`
  (the Drive `UDS/Images/NIRCam` folder is empty), MIRI/segmap/catalog from the MINERVA
  Google Drive. Local data tree reorganised by field: `MINERVA/data/UDS/`
  (`DR0/` = the old `data/DR0`, plus `n3.0/`, `m3.0/`, `n3.0_v1.2_SEC/`,
  `n3.0_m3.1_v1.2.1/`, `v2/`, `n2.2_m2.0_v1.0/`) and `MINERVA/data/COSMOS/`
  (the old `data/DR0.1`, whose name wrongly implied a UDS release). `MINERVA/data/00WHERE`
  now lists the S3 and Drive master-record URLs; `UDS/README.md` and the updated
  `COSMOS/README.md` document the layout. `examples/uds_770_dr0.json` and
  `examples/cosmos_770_dr0.1.json` were repointed at the new paths.
  Trial run done (`examples/uds_770_dr0.1/`): 138641 sources inside the F770W footprint,
  1549 in the 0.5' patch (vs COSMOS's 507 — the UDS run uses the ACS+WEBB chi-mean
  detection, COSMOS the LW noise-equalised one, so the UDS patch reaches much fainter).
  Astrometry converged in 4 passes to a bulk MIRI-vs-NIRCam shift of (0.63, 1.91) px at
  40 mas; error calibration `err_1/err_pred_1` = 1.0003 (16/84: 1.0000/1.0136); no
  negative fluxes; 70 sources above SNR 5, 111 above SNR 3; realized `EEBOX1` 0.9675
  (`EECIRC1` 0.9649 at r 2.02", stamp 4.04"), so `flux_1_total/flux_1` = 1.034.
  `alma.md` in the repo root is the standalone setup/run instruction for a collaborator:
  branch checkout, S3 + Drive data fetch, `PSFFactory.from_csv` grid generation (the
  MJD-tagged grids are too large to track), config editing, run, outputs.
  Note: `.gitignore` had `examples/*`, so no run script or run config had ever been
  committed; `!examples/run_*.py` and `!examples/*.json` now track them.
- [x] PSF encircled energy is now **measured, not requested**. `ee_fraction` in
  `DrizzlePSF.get_psf_radec` was relative to the finite ePSF stamp sum
  (`cum /= cum[-1]`), so it was filter-dependent and a hi/lo pair sized by the same
  `ee_fraction` enclosed different absolute EE; it is now absolute against the
  calibrated grid (`cum /= oversample**2`), the 0.95 default is gone (`None`), and a
  request above the stamp's own `EE_stamp` raises with the achievable ceiling.
  `_ee_fraction_to_arcsec` is documented as what it is — a *predictor*, forced to work
  on the native oversampled grid because a size is needed before anything is drizzled,
  and it answers with a circular diameter that then becomes a square side.
  New `psf.stamp_encircled_energy(psf, pscale, ee_fraction=None)` measures the
  delivered stamp instead: `ee_box` (full square sum), `ee_circ` (inscribed circle),
  `r_circ`, `r_ee`; cubes reduce to per-stamp means, one stamp at a time so a
  few-hundred-region map does not allocate the whole sorted cumulative. Every
  `get_psf_radec` call now caches `psf_size` / `ee_box` / `ee_circ` / `r_circ` / `r_ee`
  / `ee_fraction_request` from the cube it produced (`_record_realized_ee`), logs
  requested-vs-realized, and warns when the drizzled stamp misses the request.
  `Pipeline.run` writes the same numbers per fitted filter into `cat.meta` as
  `EEBOX<i>` / `EECIRC<i>` / `RCIRC<i>` / `PSFSZ<i>` (<=8 chars, so they land in a FITS
  header without HIERARCH, with the description as the card comment); `EEBOX<i>` is by
  construction the `throughput_<i>` already used for `flux_<i>_total`.
  Measured on UDS DR0 F770W (`MJD60154`, 80 mas): request 0.95 -> 7.200" (90 pix) ->
  realized `ee_box` 0.9634, `ee_circ(3.600")` 0.9579. The gap to the request is
  quantize-up (+0.3-0.9%), circle-to-square corner flux (+0.6-0.8%), and a **+0.93%
  drizzle flux gain** (see TODO) — the last one is why the realized numbers, not the
  native growth curve, are what the catalog carries.
  Also added the missing `logger.warning` in `Template.downsample` for a cutout origin
  that is not `k`-aligned: the trailing low-res row/column is zero-filled and up to ~4%
  of template flux is silently dropped. Only reachable via
  `multi_resolution_method="downsample"`; the default `"upsample"` path goes through
  `project_to_block_replicated_grid` and is exact. `AlignedCutout.downsample` refuses
  the same condition outright.
- [x] MINERVA DR0.1 F770W run set up and validated against the DR0 run.
  **The DR0.1 delivery is COSMOS, not UDS** — all 23 Google-Drive zips in
  `MINERVA/data/DR0.1` are `MINERVA-COSMOS` (`n3.0` NIRCam/HST reduction, `m3.0` MIRI
  mosaics, `v1.3` aperpy catalogs); a grep for `uds` across them returns nothing. DR0.1
  is therefore a new field, not an updated UDS release, so the requested
  `uds_770_dr0.1.json` cannot exist and no source-level DR0/DR0.1 comparison is
  possible. New config is `examples/cosmos_770_dr0.1.json` + `run_770_dr0.1.py`,
  mirroring the DR0 run in every setting that transfers (LW noise-equalised detection to
  match DR0's `faper_f277w+f356w+f444w` catalog, `psf_size` null, blur "default",
  `fit_astrometry_joint`, `scene_minimum_anchors` 10, `aperture_diam` 0.5, `r_trial` 0.5').
  Unpacking notes, the S3 provenance of the NIRCam mosaic, and the two local fixes are
  documented in `MINERVA/data/DR0.1/README.md`.
  Data prep: the delivery's `n3.0/` (NIRCam) directory is **empty**, so the F444W
  template mosaic was pulled from `s3://grizli-v2/MINERVA/mosaics/cosmos/40mas-v3.0/`
  and verified to sit on the segmap grid exactly (18944x32768, CRVAL 150.13/2.325,
  CRPIX 9216.5/15872.5). The shipped MIRI csvs are named `cosmos-v3.0_f770_wcs.csv`
  (no trailing `w`), which `jwst_psf._FILTER_TOKEN` cannot parse — added `f770w`-style
  symlinks for all six MIRI filters rather than touching the parser.
  PSFs: 60 new COSMOS ePSF grids in `data/PSF` via
  `PSFFactory(prefix='COSMOS', date_mode='cluster', delta_day=2.0)` — 22 NIRCam F444W
  cluster dates x NRCA5/NRCB5 (GRID25 OS4, fov 4") + 16 MIRI F770W dates (GRID9 OS4,
  fov 8"), no failures. Frame counts 586 (F444W) / 518 (F770W) vs DR0's 297/228.
  Also fixed `examples/uds_770_dr0.json`, which carried `"psf_size": None` and so failed
  `json.loads` outright — now `null`.
  Trial patch `[150.17963, 2.43257]`, chosen as the deepest fully F770W-covered 0.5'
  patch (median wht 1.34e8, 2.0x the DR0 patch's 6.76e7); 507 LW sources vs DR0's 578,
  so the two runs are comparable in size. DR0 was re-run with current code into
  `uds_770_dr0_v3` as the baseline — `flux_1` bit-identical to the existing v2 outputs.
  Comparison (`examples/compare_dr0_dr0.1.py` -> `examples/dr0_vs_dr0.1/`): error
  calibration `err_1/err_pred_1` = 1.000 (16/84: 0.9997/1.001) in both; no negative
  fluxes in either; patch residual/science nmad 0.9784 vs 0.9782 (the 80 mas patch is
  noise-dominated, so this is a sanity check, not a discriminator); median aperture
  residual at source positions -0.012 vs +0.002 of the patch noise, and +0.22 vs +0.16
  for SNR>5 — the same mild under-subtraction of bright sources in both, slightly
  smaller in DR0.1. Astrometry converged in both (5 passes for DR0.1, 3 for DR0) to a
  bulk MIRI-vs-NIRCam offset of comparable size but different direction: DR0
  (-0.87, +0.23) px vs DR0.1 (-0.24, -0.96) px, i.e. ~0.036" vs ~0.040" at 40 mas.
  DR0.1 has higher SNR throughout (median SNR 1.95 vs 0.33, 23% vs 10% above SNR 5),
  which the 2x weight only partly explains — the DR0 patch centre was a fixed choice
  while the DR0.1 one was depth-optimised, so the two patches differ in source
  brightness as well as depth. Not yet done for COSMOS: the saturated-star repair pass
  DR0 used, so DR0.1 fits the raw F444W mosaic and the unrepaired LW segmap/catalog.
- [x] Config-driven runs built INTO `Pipeline` (no new class): `RunConfig` dataclass
  (JSON with `#` comments, unknown keys raise) + `Pipeline.from_config("run.json")` with
  step methods `build_psfs()` / `build_kernels()` / `run()` (auto-loads data) /
  `write_outputs()` / `run_all()`, geojson-cached in `out_dir`, all intermediates
  (`dpsf_*`, `prm_hi/lo/kern`, `images`, `catalog`, `table`, `residuals`, `scenes`)
  inspectable on the instance; CLI `python -m mophongo.pipeline config.json [steps]`.
  Folds in all run_*.py preprocessing (frame asserts, footprint filter, trial patch,
  bg/ivar, NaN guard, shared blur "default"). Fixes the legacy PSF-map misalignment
  (audit B011 pattern): band maps now carry PSFs at their OWN region centroids
  (lookup-safe); kernels are built from pairs drizzled at the hi/lo overlay centroids.
  `examples/run_770.py` reduced to a ~15-line cell script over
  `examples/uds_770_dr0.json`. Validated end-to-end on the DR0 trial patch: per-source
  `flux_1` identical to the script-era run (median ratio 1.00000, 16/84 both 1.00000);
  throughput 0.9170 vs 0.9181 reflects the corrected own-centroid lookup. Tests:
  new `tests/test_pipeline_config.py` (7); full suite 83 passed + known B018 failure.
  run_1280/1500/1800 migrate by copying the json config (change filter fields +
  psf_size; F1500W/F1800W still gated on kernel-window vetting per the audit plan).
- [x] Made the mock "extra" PSF blur a first-class shared setting: new
  `mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC = {"f770w": 0.08}` is the single source
  of truth; `MockMosaic.psf_gaussian_fwhm_arcsec` defaults from it;
  `verification.build_realistic_two_detector_mock` gained a settable
  `psf_gaussian_fwhm_arcsec` parameter (None -> MockMosaic default, 0/{} disables); and
  `examples/run_770.py` now broadens the F770W model PSFs with the same constant via
  `gaussian_blur_fourier` before kernel construction — previously the driver's kernel
  chain omitted the broadening that the realistic mocks (and real MIRI mosaics) carry,
  a model-too-narrow mismatch of the donut-residual type. Unified the operator end to
  end: new module-level `mock_mosaic.gaussian_blur_psf` (FWHM/sigma = 2.355 via
  `PSF_BLUR_FWHM_PER_SIGMA`) is now the single blur implementation — `blur_filter_psf`
  delegates to it, run_770.py calls it directly, and a new regression test pins the mock
  hook and the driver-style call bit-identical
  (`test_gaussian_blur_psf_is_the_single_shared_operator`). `verify_pipeline.ipynb`
  exposes `PSF_BLUR_FWHM_ARCSEC = None` (None -> shared default) and passes it through
  `build_realistic_two_detector_mock(psf_gaussian_fwhm_arcsec=...)`.
  DR0 trial A/B (psf_size=4, blur on vs off, 578 sources): blur is exactly
  flux-conserving (throughput unchanged to 1e-9); fitted fluxes rise by median +0.9%
  (16/84: -0.4%/+2.1%) for flux>10 sources — the broader model kernel recovers flux
  previously left in residuals. POLICY (standing): the scheme is verified ONLY on
  simulated data with injected truth (`verify_pipeline.ipynb` mock harness, <1-2% offset
  at any SNR); comparison to real photometry happens elsewhere, outside this repo's
  driver/verification outputs. Full suite after the changes: 76 passed + the known
  pre-existing test_moffat_recovery psf_wing failure (audit B018, unrelated).
- [x] Diagnosed the "donut" residuals in the 2026-06-10
  `verify_pipeline_realistic_out/template_extension_psf_wings` run vs the cleaner-looking
  May `mock_run_770`. Not a regression: stacked bright-source residuals show the old run's
  fractional systematics were 5–15x larger (hidden by ~14x fainter sources and fit-grid
  jailbar noise), while the new run's donuts are a ~0.3–1% of-peak systematic made visible
  by peak-pixel SNR ~1300 mock sources. The donut (negative core, positive ring at
  r≈0.2–0.4") is identical for `none` vs `psf_wings` template extension and for the
  no-shift control run — so template wings, astrometric shifting, and repeat-convolution
  are all excluded. Shape-normalized model/data stacks show the fitted model is ~2% too
  peaked in the core and ~3–5% deficient exactly at the F770W Airy ring (0.3"), net ~1.4%
  missing flux (`med_lo` 0.986): a painted-mock-PSF vs PSF-map/kernel chain mismatch in
  the verification harness (kernel reproduces the map target PSF to 0.02%), not a fitter
  bug. The residual.fits vs source-stage-diagnostic "contradiction" is display-only: both
  read `pipe.residuals`; the fit-grid panels are dominated by the period-2 jailbar
  sampling artifact, which cancels in the 2x2 native block-average that residual.fits gets.
- [x] Proved the donut root cause (multi-agent verified, adversarially reviewed): the mock's
  "realistic" F770W 0.08" Gaussian blur was applied through two different operators. Painting
  (`MockMosaic.blur_filter_psf`) upsampled the native 80mas stamps by `np.repeat` x2, blurred
  at 40mas, and 2x2 sum-binned back — mathematically an extra `tri(1,2,1)/4 x tri(1,2,1)/4`
  kernel at 40mas spacing (machine-exact impulse-response measurement; +800 mas^2/axis
  variance, sigma~28mas) on top of the intended Gaussian. The map/kernel path
  (`verification.apply_mock_filter_blur_on_grid`) blurred the 40mas map PSF directly — no
  sandwich. Replaying the exact painting operator reproduces the saved `mock_f770w_truth`
  stamps to rms 5e-5 of peak; base pixelization (drizzle-to-80 vs bin2(drizzle-to-40)) is
  symmetric to 6e-6; Wiener regularization, segmap truncation, drizzle-phase variation all
  excluded. The blur feature only entered at 6ed937b (2026-06-10) — the May `mock_run_770`
  predates it (no blur, data/model PSFs matched by construction), and `examples/run_770.py`
  applies no blur at all (real-data driver; the 0.08" blur is a deliberate mock-only
  data-vs-model mismatch).
- [x] Fixed the blur asymmetry: new `mock_mosaic.gaussian_blur_fourier` applies the exact
  analytic Gaussian transfer function in Fourier space on the PSF's own grid (grid-
  independent, flux-conserving, exact for sub-pixel sigma); `blur_filter_psf` uses it
  directly on the native grid (upsample/sum-bin sandwich deleted) and
  `apply_mock_filter_blur_on_grid` now delegates to `blur_filter_psf` so both paths are one
  operator. Replaced the two sandwich-characterization tests with a flux-conservation test
  and a 40mas-vs-80mas blur/bin commutation test (the property whose violation caused the
  donut). Mock regeneration required for any rerun of the verification harness.
- [x] End-to-end fix validation (reduced 250-source, 50% point-source scenario, auto-size
  mosaic): flux bias eliminated — `med_lo` 0.9990 (pre-fix 0.9865); the coherent donut
  (pre-fix +2.4e-4/flux positive ring at 16-22σ) is gone; shape-normalized model/data
  profile ratio flat within ±2% (pre-fix monotonic +2.6% core → −4.4% ring). A smaller,
  sign-flipped residual remains: −1.4e-4/flux ring at r=2.5 native px (−5σ), model
  marginally too broad. Tested and excluded blur-grid aliasing as its cause (blur-at-40mas
  -then-bin vs bin-then-blur-at-80mas differ by ≤0.12% on the real F770W PSF). Open
  candidates: F444W painting stamps are 4" while the map/kernel source PSF uses 8"
  (`build_realistic_two_detector_mock` `psf_size_arcsec`), and nsrc=250 measurement
  statistics — a same-nsrc pre/post A/B rerun would settle whether the remaining ±2%
  structure is real. Validation outputs + `donut_comparison.png` in the session scratchpad
  `fixcheck_out/`.
- [x] Scene diagnostics PNGs now render at 2x pixel sampling
  (`_diagnostic_pixel_sampling_dpi` gained an `oversample` parameter; scene plots use
  `oversample=2.0`, min/max dpi 400/2400).
- [x] Full code audit (2026-07): state analysis, verified bug hunt with repros, MIRI DR0
  operational plan, staged cleanup/refactor plan — reports in `scratch/claude_audit/v1/`
  (gitignored). Note: the historical "flux bug" was an earlier shift-block normalization
  error (since fixed), not `SparseFitter._flux_errors`; `SceneFitter` is the only live
  solver path, so fixes in obsolete paths are deprioritized.
- [x] Fixed the scene-solver flux-only fallback (audit B004/B013/B014): `SceneFitter.solve`
  now treats empty shift blocks (scene with <2 bright members, e.g. saturated-isolated
  scenes) as flux-only, and `Scene.solve` leaves templates unshifted instead of crashing
  with a size-0 matmul in the shift predictor. Verified with the audit repros; full suite
  unchanged (75 passed).
- [x] Changed the matching-kernel default to `recenter=False` in both
  `utils.matching_kernel` and `PSF.matching_kernel` (audit B007/B010): the
  quadratic-centroid recentring displaced even-parity kernels by 0.5 pix relative to the
  `N//2` convolution convention, and its hard-coded `fit_boxsize=7` was the documented
  F1500W failure mode — both moot with recentring off by default (opt-in remains).
- [x] Made `examples/run_770.py` runnable: restored the `psf_size = 2.0` definition
  (was commented out but still referenced), dropped `recenter=True` from kernel
  construction, and sanitized non-finite pixels (zero image + zero ivar) before the
  Pipeline call since the pipeline's finite check does not cover input images.
- [x] Ported `examples/run_770.py` to MINERVA-UDS DR0: repaired catalog/segmap/F444W sci
  from `examples/repair_saturate_out/`, DR0 v2.4 MIRI extrabkg mosaics with explicit
  shipped `uds-v2.4_f770w_wcs.csv` (auto_reconstruct never triggered; frame counts
  asserted 297/228), MJD-tagged ePSF patterns (`UDS_NRC.._F444W_MJD\d+_GRID25_OS4`,
  `UDS_MIRI_F770W_MJD\d+_GRID9_OS4`), geojson caches written to the output dir (never
  into read-only DR0), new preprocess step keeping only wht>0 footprint sources (~86k).
  `flag_artifact` rows (DR0 hand-drawn star/spike regions, not mophongo's) are KEPT in
  the fit — they carry real flux that must be modeled; filter downstream. Validated
  end-to-end on the r<0.5' trial patch (578 sources, 2 scenes, astrometry converged in
  3/5 passes, 17 GB peak).
- [x] Added a nearest-frame fallback to `DrizzlePSF.get_psf`: positions outside every
  exposure footprint (region-map sliver centroids sitting exactly on footprint
  boundaries — the source of the 9 empty kernels in the first DR0 trial) now use the
  closest frame instead of returning an empty stamp, with a logged warning including the
  distance. Verified on a previously-empty F770W position (stamp sum 0.864); focused
  tests (test_psf, test_psf_map, test_mock_mosaic, test_pipeline) 40 passed.
- [x] run_770.py: `psf_size = 4.0` for F770W; `psf_size = None` now means the full native
  ePSF stamp size as generated (passes `size=None, ee_fraction=None` to `get_psf_radec`).
  flag_artifact rows are kept in the fit (see above); PSF/kernel caches regenerated.
- [x] Audit bookkeeping convention: fixed findings are checkmarked in
  `scratch/claude_audit/v1/02_bug_report.md` section "0. Fix log" (entries are never
  removed, only marked). Currently marked fixed: B004/B007/B010/B013/B014/B016 + the
  unnumbered empty-PSF caveat; B002 has a driver-level mitigation only.
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
  - faint templates biasing the shift blocks: `astrom_minimum_snr=15` gives
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
- [x] Reran mock validation with explicit `reg_flux=0.0` and `astrom_reg=0.0`
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
