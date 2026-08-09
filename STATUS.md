# Status

This file records completed implementations, validation runs, and the current work state.

## Current Work
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
  * Tests: `tests/test_template_schemes.py` (36). Suite: 155 passed +
    pre-existing `test_moffat_flux_recovery[psf_wing-3-psf]` failure. The three
    detection-PSF / noise-scalar defects above came out of an adversarial
    review of both ports against their reference sources (43 agents, 3
    confirmed of 20 raised).
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
- [x] IDL subphot diagnostic port: `Pipeline.plot_subphot(source_id)` renders
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
