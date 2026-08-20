# Changelog

## 0.1.0 (2026-08-20)

183 commits since `v0.0.3` (2025-09-02); 225 files, +124k/-22k lines.

The headline changes are the template build schemes, a rewritten astrometric
anchor solve, saturated-star repair, spatially varying PSF and kernel maps, and
the two cluster toolkits that run a MINERVA release end to end on CANFAR and
OzStar. Several flux-scale bugs were fixed along the way, so photometry from
this release is not directly comparable with `v0.0.3`.

### Flux scale and photometric corrections

- `e181ffd` Fix flux bias from absolute regularization in `SceneFitter`.
- `524d3ed` Fix aperture correction: use the post-convolution template total
  rather than the pre-convolution aperture sum.
- `ffb7cf6` Correct the encircled-energy chain to one per-source factor from
  amplitude to total flux; `dd2050f` drops the window method from the report.
- `2d2d014` Solve the flux block as NNLS by default instead of clipping after
  the fact (#111).
- `bf0f88b` Match convolution widths and widen the normal-matrix accumulators.
- `1d30f28` Apply `ee_psf_lo` to the aperture estimator as `ap_flux_total_<i>`.
- `a316486` Add the `tot_stamp` column; `totcor` is always EE-inclusive; warn on
  support parity.
- `d0d62c4` Classic-mophongo correction formalism: `totcor`, `psfcor`,
  `totcor_cat`. `4408bfc` corrects the IDL `totcor` reading (it is not
  EE-corrected) and settles the column roles.
- `c0c291d` Rename the aperture-EE symbols: `ap_F` to `ap_hi`, `ap_B` to
  `ap_lo`.
- `50bcba1` `reg_flux` gains three-state semantics: `None` auto, `0.0` none,
  positive explicit.
- `fb1e017` Quote the encircled-energy aperture in arcsec rather than in
  PSF-stamp pixels, which had made it 12.5x too wide.
- `84b3fea` Aperture sizing by encircled energy, Estimator 3, and field-free
  PSF grid names (#109).
- `bc4a61d` Document that aperture columns are not on the total-flux scale.

### Template schemes

- `0237390` Add the `extend_mode` selector with the wren and IDL-classic build
  schemes ported into `template_schemes.py`; `97318e9` fixes three defects in
  those ports.
- `708615f` New default scheme: least-squares PSF wings without double counting.
  `3a62147` renames it `psf_wings` and makes it the default; `15e0860` drops the
  invented `segment` alias.
- `bee997b` Rename `extend_mode` `psf` to `psf_convolution`, keeping `psf` as an
  alias; `2d629ea` unifies on `extend_mode` and copies full template metadata.
- `cca19f4` Disallow PSF fallback for template extension; use weights for
  classic SNR.
- `51bf69f` Add `RunConfig.wht_hi` for the SNR-weighted schemes; `d16c2aa`
  requires a detection weight map and drops the redundant `driz_hi` pointer;
  `cfa4f2a` logs the weight-map calibration factor.
- `dfe9a7f` Propagate `ee_psf_lo` / `ee_tmpl` through template resampling.
- `ff3b8d4` Do not build templates where the detection band has no coverage;
  `db36fe2` flags partly covered templates; `fa22f2a` sets `FLAG_OUTSIDE_WEIGHT`
  on templates the prune drops.

### Scenes and astrometry

- `663a19e` Add `scene_max_merge_radius` to `FitConfig`; `bfb09d1` and `4600789`
  bound the scene shape with it.
- `97eeca2` Add `scene_max_size`, a local threshold split of oversized scenes;
  `b394083` defaults it to 800, `75c17d2` raises it to 1000.
- `6beb05f` Improve astrometric source selection: exclude stars and blended
  sources. `d85a9d7` unifies anchor selection across solve and merge.
- `879e5b0` Record astrometry damping and the scene-solver writeup.
- `9e2c47d` Cap astrometric anchor leverage.
- `467fd3b` Exact astrometric blocks, background and ivar fixes, trial patches,
  aperture centring.
- `99e6095` Scene-by-scene astrometric loop (#105).
- `3990462` One astrometric anchor is enough to solve a scene's shift.
- `6e0f27a` Weight astrometric anchors by how much they agree with each other,
  in the new leaf module `astrom_robust.py` (#108). `bba78e3` stops the robust
  pass paying for work it discards.
- `f82b012` Assemble anchor local systems from slice intersections (#112).
- `eb0cdca` Switch `SceneFitter` solvers from CG to `spsolve`; `77030d1` drops
  the dead `cg_kwargs` (#110).
- `e77933e` Make the scene size cap bind, and stop factorising a dense matrix as
  sparse.
- `5f7d19d` Persist scene membership as `scene_<i>` in the fit table.
- `ae43bfc` Report only the post-merge scene structure at INFO.

### PSF handling

- `ec8252c` Auto-build missing PSF grids, and make every cached product declare
  its provenance.
- `c29b08e` Default PSF grids to one per epoch (#106); `c904b00` asks which
  epochs are missing rather than whether the grids agree.
- `e629e44` Build ePSF grids at a stated field of view rather than an inherited
  one; `4341cb5` always names the FOV in grid filenames; `a9afb17` resolves a
  frame's ePSF key across the FOV token.
- `68c99e9` Job to rename and stamp existing grids in place; `2695028` sweeps
  every family in one pass; `ab06f04` adds a standalone name converter.
- `c22a699` Per-band MIRI Gaussian blur defaults.
- `061b599` Spatially varying Wiener deconvolution toward theoretical PSFs.
- `ea9dde2` Match kernels from the cached band PSF maps instead of re-drizzling.
- `e2c2e35` Give written region maps a projection; `95e6113` corrects the record
  on the CRS warning.
- `f46676a` Check drizzled PSFs against MINERVA-UDS stars in five bands.
- `0de0a96` Pin one thread per PSF worker, and measure what the pool buys.
- `fffdbef` Add MIRI STPSFs.

### Saturated-star repair

- `d5c2d09` Add `saturate.py` with PSF repair and diagnostic SNR/polar panels.
- `7c48239` Add the standalone `mophongo-repair` CLI.
- `65d73f6` Share the repair cache across a multi-band campaign; `87c80ca`
  defaults `repair_cache_path` to `..` so a field shares one cache out of the
  box; `45b6f33` documents the `out_dir`-relative resolution and accepts
  directories; `805217f` reuses the cache and names the band in the
  weight-calibration log.
- `f631af8` Keep saturated stars visible in the scene template panel; `6dc7c47`
  shows the no-shift repair residual on the same scale as the shifted one.

### Performance and memory

- `471b511` Remove 4.5k lines of dead code across src, tests and docs (#98).
- `4e0a725` Cut full-field memory: drop redundant copies, band the whole-array
  passes (#102).
- `d72ac95` Store PSF, kernel and coadd arrays at float32; `1fac911` stops the
  float64 round trips on stamps and encircled energy; `0b4a904` solves PSF
  matching in float64 and stores the kernel float32 (#104).
- `0d7a7f3` Write stamps as HDF5; `ac050a0` writes them uncompressed.

### Cluster toolkits

- `1a8bded` Add the CANFAR toolkit and an A-Z manual for the Science Platform;
  `645b538` adds `campaign.py` to launch a whole campaign in one command.
- `c29b08e` Add the OzStar campaign toolkit; `e23a617` its run root and job
  scripts; `92b6c56` its release tooling.
- `a569304` Give CANFAR the OzStar shape: build the grids, repair, then fan out.
- `9966cd0` Shard the ePSF build over jobs rather than fields; `3ca7233` shards
  the OzStar build too and gives a campaign a memory; `61b665c` shards in
  contiguous blocks; `169a7be` fetches the OPDs once, serially, before the
  shards fan out.
- `0964ff4` Restructure CANFAR run trees around a release; `2c4fea0` keeps them
  off `/arc/home`; `d33cf88` points the toolkit at the run's own config
  directory.
- `ab06f04` and `5b985b3` give each run its own source checkout and venv, so a
  run's outputs can be tied to the commit that produced them; `636cf04` records
  the mophongo commit in the run log.
- `788e3e6` Add `sync`, to ship a code fix without rebuilding the venv.
- `352a3e3`, `2b10a6b` CANFAR resource defaults; `af12da3` treats vanished skaha
  sessions as terminal; `91b4a3c` sanitises session names; `8bad9df` fixes fetch
  naming and stops a missing file aborting the download.
- `663fbff`, `1db4eca` give the thin EGS MIRI bands a trial patch.
- `0baeca8` Correct the OzStar README and MANUAL against the code.

### Diagnostics, logging and run state

- `28a7ccf` IDL-style inspection panels, estimator and aperture docs.
- `d8e2a20` Draw the scene partition as blobs over the whole field; `149c302`
  redraws a finished run's scene figures without refitting; `a174873` refits one
  scene after the fact; `d24752b` draws it with the run's own diagnostic;
  `b8f17e0` annotates scene figures in-panel.
- `393e638` Single-print logging: modules stop owning handlers. `34b8eac` puts
  every library's records in the run log; `8d31afc` logs every CLI run to its
  output directory; `2df4600` silences the HIERARCH-card warnings.
- `ba693ab` Report wall time per section at the end of a run; `3643e7b` uses
  human-readable byte sizes.
- `498eac3` Restore fit state from stamps; `93bf0b5` reopens a finished run from
  `out_dir` alone.
- `4f7d57c` Repair `plot_result`, which could not have run, and cover it with a
  test.
- `1c9af60` Fix the two long-standing test failures; the suite is green.
- `c184b19` Drop em dashes from pipeline messages.

### Documentation

- `0c7abfd` Read the Docs scaffold; `e0be3c5` the full documentation set;
  `80516c2` an independent verification pass, config-first quickstart and the
  furo theme; `3d49b95` fixes seven bugs that pass surfaced.
- `eb2dd11` Concept-first pages with executed snippets, details moved to
  docstrings; `88db617` concept figures on every major page; `412dea4`
  definition-list rendering and trimmed pages.
- `d2c73ab` Entry-point comparison table (arrays vs JSON config); `cbbd7a1`
  `extend_mode` naming convention; `5905126` brings the site to the merged
  template-scheme pipeline; `8f5beba` aligns stale docstrings with behaviour.
- `55bf174` `LWBUG_ANALYSIS.md`: architecture map and F1500W residual diagnosis.
- `09bcbb9` Verified audit of the wren fork and her aperture-correction design.
- `3417c88` Close the gaps surfaced by the wren scratch cross-reference.

### Verification and mocks

- `733cb69` Add the mock mosaic generator; `0d915fd` fixes mock injection phase
  bugs and the verification harness.
- `388f91f` Versioned verification driver (v2/v3) with a scheme parameter;
  `5f42859` v3 against wren; `32ec286` v4 with `ee_psf_lo`-corrected aperture
  totals; `d550f6e` v6, support-matched IDL comparison at percent level;
  `69d330c` v8.
- `b16be80` `--psf-dir` / `--psf-size` overrides and band selection; `dea9d8c`
  `--r-trial` override and provenance-guarded cache seeding.

### Packaging and configuration

- `fbbf631` Declare the five dependencies mophongo imports but never listed.
- `83d5e6e` Accept segmaps stored as float, which COSMOS ships and photutils
  rejects.
- `77030d1` Read the lo-res filter from the mosaic header.
- `afdf1b2` Regenerate the 17 MINERVA configs from the fixed generator.
- `77b4c5b`, `92b6c56` Untrack generated run configs and outputs.
- `9f77504` Add a config-driven UDS F770W run on minerva-v3.0 inputs;
  `82fdfd0` tracks the example run scripts and configs.
- `243980b` Bump the package version to 0.1.0, which had been left at 0.0.1
  while the tags moved on.

The remaining commits in this range are merges, `STATUS.md` / `TODO.md`
bookkeeping, and repo guard-hook allowlists.
