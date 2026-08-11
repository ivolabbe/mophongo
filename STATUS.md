# Status

This file records completed implementations, validation runs, and the current work state.

## Current Work
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
  ap_F/ap_B measured on the source composite (psf_apercor defaults off, so the
  PSF branch is the fallback); released `totcor` = 1/ap_B; `eflux_F` correctly
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
  * main implements Estimator 1 *in full*: `ap_corr_<i> = 1/ap_B` = `totcor1`. The
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
  `merge_small_scenes`, so `scene_minimum_bright` counts bright & isolated
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
  fit overrides back to `{fit_astrometry_joint, scene_minimum_bright: 5,
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
  catalogue's `tot_cor`; and bounding `ap_F`/`ap_B` by the PSF EE removes the
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
  `scene_minimum_bright` 10, `aperture_diam` 0.5, `r_trial` 0.5'); frame counts
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
  `fit_astrometry_joint`, `scene_minimum_bright` 10, `aperture_diam` 0.5, `r_trial` 0.5').
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
