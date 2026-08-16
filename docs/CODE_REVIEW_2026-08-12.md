# Deep code, numerical, and documentation review — 2026-08-12

## Executive conclusion

Mophongo has a substantial amount of thoughtful implementation and unusually
good scientific context, but it is **not yet safe to describe as
release-quality photometry software**. The main risk is not ordinary polish.
Several live paths can change fitted fluxes, select the wrong spatial PSF,
mis-calibrate inverse variance, corrupt reproducibility metadata, or silently
break catalog/segmentation invariants.

The most important findings are:

1. The joint astrometric system is not the normal equation of the documented
   linearized model for overlapping sources.
2. The final applied astrometric shift is rendered with fluxes solved on the
   preceding template basis; a direct probe found a maximum 5.8% amplitude
   change after the missing final flux-only solve.
3. The fitted-band background/inverse-variance estimator both dilates the
   wrong mask and scales its convolved detection threshold incorrectly. Its
   source-free verification cannot reveal the resulting contamination.
4. Template convolution/block projection can replace a parent WCS with a
   cutout WCS, moving the apparent source sky coordinate by tens of arcseconds
   and therefore selecting the wrong `PSFRegionMap` region.
5. Wren/classic config snapshots are corrupted by iterating the characters of
   a string as prefixes, defeating the promised reproducibility contract.
6. Multi-resolution coordinate expansion is both a hard `NameError` and
   mathematically offset; the duplicate saturated-star mapping has the same
   formula error.
7. Several repair/catalog paths can produce orphan catalog rows, merge
   unrelated stars, pair FITS data with the wrong header, or overwrite one
   output product with a different one when filenames collide.
8. Multiple PSF and mock-verification paths silently assume OS4, attach PSFs
   to the wrong geometries, reuse colliding cache names, correlate noise
   between filters, or mark sources as valid even when no source was painted.

There are **no P0 findings** (known unconditional loss of an existing input or
a demonstrated security issue). There are 23 P1 findings that should block a
science/release claim, plus 40 P2 defects and a longer set of documentation,
testing, architectural, and conceptual improvements.

This review is intentionally evidence-led. “Confirmed” means the behavior is
visible directly in source and was reproduced by a focused probe or follows
deterministically from the shown code. Conceptual risks are labelled
separately and are not presented as demonstrated bugs.

## Scope and baseline

Review baseline:

- repository HEAD: `a4ff14ba7f6791cc635f4e261e933f6af3133d76`
  (`a4ff14b`, 2026-08-12);
- reviewed package source: 26,569 lines under `src/mophongo/`;
- reviewed tests: 7,701 lines under `tests/`;
- reviewed top-level documentation: 12,043 Markdown/RST lines under `docs/`;
- read `AGENTS.md`, `GUIDE.md`, `STATUS.md`, and `TODO.md` in full;
- reviewed code comments/docstrings and searched explicit TODO/FIXME/bug/dead
  markers;
- compared all 14 public narrative pages with the deployed Read the Docs
  `latest` sources;
- inventoried all 19 top-level `scratch/wren/*.pdf` files (255 pages total)
  and all 18 top-level `scratch/wren/*.tex` files (14,607 lines);
- extracted every PDF, read the current unversioned TeX/PDF set, and visually
  inspected contact-sheet renders of all 88 pages in the seven current PDFs;
- treated suffixes such as `_v1`…`_v7` as historical snapshots, not as
  normative current design;
- ran the full test suite, focused module suites, a warnings-as-errors Sphinx
  build, package/lock checks, compile checks, and targeted numerical probes.

The working tree was already dirty and changed concurrently during the audit,
especially in MINERVA verification files, `mock_mosaic.py`, `verification.py`,
`STATUS.md`, and `TODO.md`. No pre-existing or concurrent changes were
reverted. File/line references point to the working-tree snapshot observed on
2026-08-12 and may move as that work continues.

## Severity rubric

- **P0 — critical:** unconditional destructive loss, security compromise, or
  universally invalid results. None confirmed.
- **P1 — high/release-blocking:** a live/common scientific path can return a
  wrong result, use the wrong geometry/calibration, lose reproducibility, or
  violate a core product invariant; also includes destructive output
  collisions and hard failures in advertised core workflows.
- **P2 — medium:** conditional scientific/API correctness, latent public API
  failures, misleading diagnostics, or provenance weaknesses that require a
  particular input or configuration.
- **P3 — low/debt:** maintainability, performance, documentation governance,
  packaging, and test-quality issues that do not by themselves demonstrate a
  wrong science product.

## P1 findings — release blockers

### P1-01 — The joint astrometric blocks omit cross-template terms

**Evidence.** `src/mophongo/scene.py:496-528` loops over each template and
builds `AB`, `BB`, and `bB` from that template's own gradient products. It does
not construct the global derivative columns of the scene. Consequently it
omits:

- `<T_j, -alpha_i grad(T_i) phi_i>` for `j != i`;
- derivative cross-products between distinct bright anchors;
- the x-y derivative block;
- faint-template flux rows coupled to the bright derivative columns (the
  early `continue` at lines 503-504).

A dense two-overlapping-Gaussian probe with one shared x-shift coefficient
gave:

| block | implementation | exact dense design |
|---|---:|---:|
| `AB[0]` | approximately 0 | -0.058879 |
| `AB[1]` | approximately 0 | +0.084113 |
| `BB` | 0.947782 | 0.685401 |

**Impact.** Flux and shift estimates can be biased precisely in the crowded
scenes for which a joint solver is needed. `docs/fitting.md:331-369` and
`scratch/wren/fit.tex` describe a genuine block normal equation, so the
implementation does not match the mathematical contract.

**Required fix/test.** Form scene-wide derivative columns
`B_k = sum_i -alpha_i grad(T_i) phi_ik`, then compute `T.T W B`, `B.T W B`,
and `B.T W image`. Add a dense-design equivalence test with overlapping
bright/faint templates, both axes, and polynomial order greater than zero.

### P1-02 — The final shifted model uses pre-shift fluxes

**Evidence.** `Scene.solve` solves on the current templates, applies the
predicted shifts, invalidates its matrix/vector, and retains the flux solution
from the pre-shift linearized system (`src/mophongo/scene.py:898-973`). The
pipeline can then declare convergence and immediately render those shifted
templates (`src/mophongo/pipeline.py:3333-3407`) without one final flux-only
solve on the final basis.

The current `scratch/wren/fit.tex:331-335` explicitly admits the inconsistency,
but calls it at most 0.05 pixels; the runtime default is now 0.1 pixels.

A synthetic pipeline probe with an injected `(-0.35, +0.55)` pixel shift
converged in two passes with a final 0.0903-pixel increment. Re-solving fluxes
on the final shifted templates changed:

- maximum nonzero-source relative flux: **5.824%**;
- median relative flux: **0.0739%**;
- chi-square: 10457.5707 -> 10454.2962 (`Delta chi2 = 3.2745`).

**Impact.** The stored fluxes, errors, residual, and rendered model are not a
stationary solution for the basis that is actually written. Non-converged
scenes are even less bounded.

**Required fix/test.** After the astrometry loop, run one flux-only solve for
every scene on the final templates, regardless of convergence verdict. Test
normal-equation stationarity and exact consistency among stored fluxes, model,
residual, and final stamps.

### P1-03 — `get_bg_and_ivar` has coupled mask-polarity and threshold errors

**Evidence.** In `src/mophongo/catalog.py:193-215`:

1. `bgmask = seg_all == 0` means `True` is background, but
   `binary_dilation(bgmask)` expands background and erodes the excluded source
   regions. It should dilate a source mask and invert it.
2. The bright detector convolves with an unnormalized 29-pixel disk, while
   the threshold uses the RMS factor for a normalized kernel. The coded
   factor is 0.1857 sigma; the actual white-noise RMS of that convolution is
   5.385 sigma, a factor of 29 difference.
3. The faint pass is at one sigma with `npixels=1`, so merely correcting the
   mask polarity would mask almost the entire noise field.

Quantified probes:

- source disks of radius 1/2/3/5/8 pixels had 100%/100%/97%/84%/59% of their
  source pixels wrongly re-admitted after the current dilation;
- in a 256x256 Gaussian-noise field, candidates covered about 52.7%; the
  current inverted dilation excluded only 2.9%, while correct source dilation
  would exclude 99.8%.

**Impact.** Both background subtraction and fitted-band inverse-variance
calibration can be contaminated by source flux. The current Wren
`noise_background` experiment is source-free and therefore cannot validate
this part of the algorithm.

**Required fix/test.** Normalize the detection kernel (or use the correct
unnormalized RMS), dilate `seg_all > 0`, retune the faint detector, and test
injected sources spanning brightness/size/coverage on top of correlated noise.
Acceptance must include background bias, recovered variance, mask occupancy,
and depth dependence, not only a source-free scalar-noise case.

### P1-04 — Non-finite science can crash or poison preprocessing before it is sanitized

**Evidence.** `get_bg_and_ivar` downsamples science and computes median/MAD
without a finite-science mask (`src/mophongo/catalog.py:171-188`). The
pipeline replaces non-finite science only after calling the estimator
(`src/mophongo/pipeline.py:1139-1144,1275-1287`). A single NaN can make the
statistics and detections NaN; when both detections are absent, `0 + 0`
produces a scalar `seg_all`, and the subsequent morphology call fails on
dimensionality. `bg_gaussian_normalized` also permits `NaN * 0` to spread
through filtering.

**Impact.** A single bad input pixel can abort config preprocessing or spread
NaNs into background and weight products despite later pipeline guards.

**Required fix/test.** Define a common finite-and-positive-weight mask before
binning, use masked block statistics, replace invalid samples explicitly in
normalized convolutions, and create a full-shape empty mask when there are no
detections. Test isolated NaN/inf values, blank images, zero-weight borders,
and all-invalid blocks.

### P1-05 — Template convolution and block projection corrupt WCS provenance

**Evidence.** `Template.convolve` passes cutout-local `self.wcs` as the WCS of
a new parent image (`src/mophongo/templates.py:711-718`); block projection does
the same at lines 859-866. The resulting `Template` stores that as
`wcs_original`. Spatial kernel and EE lookup later trusts this parent WCS
(`templates.py:1893-1901`). The source code even comments “note wcs origin is
wrong” at line 715.

A one-arcsecond TAN-WCS probe moved an unchanged source by about
`(-55, +50)` arcseconds after either operation.

**Impact.** Position-dependent convolution/throughput can query a completely
wrong `PSFRegionMap` region. This is scientific, not cosmetic metadata.

**Required fix/test.** Pass `self.wcs_original` as the parent WCS, preserve the
cutout WCS separately, and assert world-coordinate invariance before/after
convolution, padding, block projection, and resize on translated/rotated/SIP
WCSs.

### P1-06 — Wren/classic config snapshots delete unrelated settings

**Evidence.** `Pipeline.save_config` maps Wren/classic mode to a string and
then iterates it as though it were a tuple of prefixes
(`src/mophongo/pipeline.py:1337-1341`). Each character becomes a prefix to
remove.

Probe results:

- Wren snapshot retained only 22 of 45 `FitConfig` fields;
- classic retained 33 of 45;
- lost fields included aperture, astrometry, scene, normalization, and solver
  settings; classic could even lose `extend_mode`, `reg_flux`, and
  `astrom_reg`.

**Impact.** `<out_dir>/<name>.json` is not a reproducible description of the
executed run, contrary to `docs/pipeline.md:262-270`,
`docs/outputs.md:461-466`, and `STATUS.md:1437ff`.

**Required fix/test.** Use singleton tuples `("classic_",)` and
`("wren_",)`. Round-trip every `extend_mode` and assert equality for all active
fields, including non-default values.

### P1-07 — Pipeline input validation is inverted and NaN templates remain “valid”

**Evidence.** `src/mophongo/pipeline.py:3260-3265` checks image finiteness only
when an image is `None`, so real arrays are never checked and a `None` image is
passed to `np.isfinite`. `weights=None`, accepted by the constructor, is then
indexed. Independently, template normalization at
`src/mophongo/templates.py:1824-1842` treats `NaN != 0` as true, divides by NaN,
sets neither `FLAG_HAS_NAN` nor `FLAG_SUM_ZERO`, and leaves `FLAG_VALID` set.

A one-pixel NaN template probe produced NaN data, `template_norm`, and
`ee_tmpl` with only `FLAG_VALID` set.

**Impact.** Non-finite templates can reach normal equations without the flags
promised in `docs/outputs.md`. Assertions are also inappropriate for runtime
validation because optimized Python can remove them.

**Required fix/test.** Normalize optional sequences up front, validate lengths,
shapes, finite science, and weight semantics with explicit exceptions, and
enforce a finite positive template normalization contract. Test every flag
bit against malformed inputs.

### P1-08 — Documented `aperture_catalog` behavior is not implemented

**Evidence.** `src/mophongo/pipeline.py:3030-3031` treats every non-`None`
`aperture_catalog` as a table column key. A numeric value therefore raises
(`FitConfig(aperture_catalog=1.2)` reproduced an Astropy table `ValueError`).
The numeric/string resolver at lines 2745-2797 has no callers; aperture
measurement uses only `_resolve_image_ap_radius_pix` at 2951-2954. A string
column is copied but does not change the measurement.

**Impact.** The public contract in `docs/pipeline.md:548-552` and the aperture
technical notes is false. Users can either crash or believe a source-dependent
aperture was applied when it was not.

**Required fix/test.** Wire the catalog radius into numerator and denominator
on one coordinate/support convention, or remove/deprecate the option. Test
fixed numeric, per-row column, pixel/arcsec units, missing/invalid values, and
edge clipping.

### P1-09 — Multi-resolution fitting mutates native inputs and is not reentrant

**Evidence.** `src/mophongo/pipeline.py:3166-3185` replaces entries in
`self.images` and `self.wcs` during a run, but does not persist the matching
upsampled weight. A second run sees a high-resolution image/WCS paired with a
native weight. The downsample path mutates `PSFRegionMap.psfs`; tuple-backed
sequences fail assignment; diagnostics contain ad hoc repairs for the
resulting mismatch (`pipeline.py:4188-4198`).

**Impact.** Re-running a `Pipeline` instance, using immutable input sequences,
or inspecting state after a run can yield shape errors or changed results.

**Required fix/test.** Preserve immutable native inputs/maps and create
per-band working arrays for each run. Add two-run idempotence, tuple-input,
multi-band, and diagnostic-after-run tests.

### P1-10 — Coordinate expansion is a hard failure and not the inverse of binning

**Evidence.** `AlignedCutout.upsample` calls `expand_remap` without importing
it (`src/mophongo/templates.py:15,381-402`), producing `NameError` for every
factor greater than one. The implementation in `src/mophongo/utils.py:54-61`
uses `(x + (k-1)/2) * k`; the inverse of `(x - (k-1)/2) / k` is
`k*x + (k-1)/2`. `catalog._expand_remap` duplicates the wrong formula at
`src/mophongo/catalog.py:457-461`.

Round-trip offsets are 0.5 pixels for `k=2`, 2.0 for `k=3`, and 4.5 for `k=4`.

**Impact.** The public upsample API always fails, and saturated-star centroids
mapped from the binned grid are systematically displaced.

**Required fix/test.** Implement and import the true inverse once, remove the
duplicate, and test exact center round-trips for odd/even shapes, multiple
factors, and WCS world coordinates.

### P1-11 — Footprint filtering bypasses WCS and clips outside sources onto the edge

**Evidence.** `src/mophongo/pipeline.py:1250-1256` maps high-resolution catalog
positions to the low-resolution weight by division by a rounded scalar pixel
ratio. It ignores CRPIX/CRVAL offsets, rotation, distortion, and anisotropic
scale. It then clips out-of-bounds coordinates before testing weight, so an
outside source is retained whenever the nearest edge pixel is positive.

**Impact.** The default config path can retain outside sources or reject valid
ones for any grids that are not perfectly nested at a common origin.

**Required fix/test.** Transform high-pixel -> world -> low-pixel with the two
WCS objects, use an explicit bounds mask before indexing, and test offset,
rotated, SIP, non-square, and partial-overlap grids.

### P1-12 — Verification attaches overlay-centroid PSFs to band-region maps

**Evidence.** `src/mophongo/verification.py:551-553` evaluates source and target
PSFs at kernel-overlay centroids, then attaches those arrays to the original
source and target maps at lines 672-674. The number/order of overlay regions
need not match either band map. When one source region is split by two target
regions, both halves can receive the first overlay-centroid PSF under the
source key.

**Impact.** The realistic mock verifier can compare against PSFs that are not
the PSFs of the geometries it claims to validate. This weakens the evidence in
the current verification reports.

**Required fix/test.** Generate each band cube at that band map's own
representative points; generate only kernel pairs on the overlay. Assert
`max(psf_key)+1 == len(psfs)` and verify each representative point resolves to
the key whose PSF it generated.

### P1-13 — LW coadd has wrong variance propagation and bypasses even-kernel alignment

**Evidence.** `src/mophongo/utils.py:2504` calls SciPy's raw `fftconvolve`, even
though the package wrapper at lines 376-414 exists because SciPy's `same`
mode shifts even kernels by one pixel. Tests cover only odd 21/31-pixel coadd
PSFs. The propagated output weight uses `wht / sum(kernel**2)`
(`utils.py:2390-2393,2505`), which is exact only for constant variance and full
support. In general the marginal variance is convolution of input variance
with `kernel**2`, with masks and boundary coverage handled explicitly.

**Impact.** Coadd astrometry can shift for even kernels, while depth gradients,
holes, and edges receive incorrect weights. The result does not represent
off-diagonal correlated-noise covariance despite comments suggesting it does.

**Required fix/test.** Use the shared convolution wrapper and propagate
variance spatially. Add impulse tests with even kernels and Monte Carlo tests
across variable depth, masks, and edges.

### P1-14 — Oversampling is supported in metadata but hard-coded in physical sizing, EE, and filenames

**Evidence.** The ePSF loader records an oversampling value
(`src/mophongo/psf.py:1745-1751`), but `get_driz_cutout` divides shape by four
at 2351-2359 and `_ee_fraction_to_arcsec` hard-codes four/divides cumulative
flux by 16 at 2821-2842. Saturation sizing repeats OS4 at
`src/mophongo/saturate.py:543-564`. `PSFFactory` publicly accepts arbitrary
oversampling but filename generation hard-codes `OS4`, and resolved FOV/OS are
not part of the path/cache identity (`src/mophongo/psf_factory.py:229-303,
385-420`).

A factory with `oversample=2` produced an `..._OS4.fits` filename. OS1 EE would
be divided by 16.

**Impact.** Non-OS4 grids have wrong angular support and absolute EE, while
different builds can overwrite or reuse stale cache products. The API and
cache contract disagree.

**Required fix/test.** Resolve oversampling per selected grid, make it part of
all physical calculations and provenance, include FOV/OS/detector-sampling/
backend options in cache identity, and validate an existing file's header
before reuse. Test OS1/2/4/8 and detector-sampled grids.

### P1-15 — Ordinary Catalog states crash instead of producing a controlled empty result

**Evidence.** `Catalog._detect` dereferences `detect_sources(...).data` without
handling `None` (`src/mophongo/catalog.py:801-823`); a blank 64x64 image
reproduced `AttributeError`. `estimate_background=True,
estimate_ivar=False` leaves `self.ivar=None` and later evaluates its square
root (`catalog.py:842-861`), reproduced as `TypeError`.

**Impact.** Blank fields, aggressive thresholds, and a plausible public flag
combination fail unpredictably.

**Required fix/test.** Define the empty-catalog/empty-segmentation result,
always establish an input or estimated IVAR before detection, and validate
configuration combinations in `__post_init__`/`run`.

### P1-16 — Saturated catalog repair can violate catalog/segmentation identity

**Evidence.** `repair_saturated_catalog` allocates a new parent for each fit
while its child remap is last-wins (`src/mophongo/catalog.py:1385-1515`). One
child claimed by two nearby fits produced catalog ids `[2,3]` but segmentation
labels `[0,3]`: id 2 was an orphan. The same path ignores fitted
`shift_x/shift_y` when associating children, placing the PSF, and writing the
parent coordinate (`catalog.py:1385-1434,1492-1494`), unlike the flagging path.

**Impact.** The central invariant “catalog ids equal segmentation labels” can
be broken, and associations may be displaced by the allowed cumulative shift
(up to several pixels).

**Required fix/test.** Resolve fit/child conflicts globally before allocating
parents, apply one documented coordinate convention including fitted shifts,
and post-assert exact nonzero-label/catalog-id agreement. Test overlapping
stars, duplicate claims, shifts, and empty parents.

### P1-17 — Missing fit IDs merge distinct saturated stars into one group

**Evidence.** `flag_saturated_segments` keys every id-less fit row as `-1`
(`src/mophongo/catalog.py:1671-1696`). Windows and centers overwrite one
another. A two-distant-star probe reported two segments but one star/group.

**Impact.** An optional field silently changes scene grouping and catalog
flags, potentially coupling unrelated objects.

**Required fix/test.** Use a unique internal row identity; treat external `id`
only as metadata and reject duplicates when it is supplied.

### P1-18 — Repair output paths can collide and silently overwrite another product

**Evidence.** `src/mophongo/repair.py:330-362,831-839` derives output names
only from input stems and writes with `overwrite=True`. Inputs from different
directories with the same stem can make science and weight outputs identical;
the same applies to catalog and segmap outputs. The second write then replaces
the first with a different FITS product type.

**Impact.** A successful command can silently destroy its own earlier output.

**Required fix/test.** Resolve all output paths before writing, assert pairwise
distinctness, and require an explicit overwrite option for existing files.
Use product-role suffixes independent of ambiguous input stems.

### P1-19 — `Catalog.from_fits` can pair MEF data with the wrong header/WCS

**Evidence.** `src/mophongo/catalog.py:760-770` calls `fits.getdata(path)` and
`fits.getheader(path)` independently. On a normal MEF with no primary data and
SCI data in extension 1, the probe returned extension data but the WCS-less
primary header.

**Impact.** Catalog sky coordinates and downstream spatial matching can use a
header from a different HDU than the pixels.

**Required fix/test.** Select one data HDU in a context manager and copy both
data and header from it. Validate science/weight/segmentation shapes and grid
WCS agreement.

### P1-20 — Saturation acceptance and published metrics are stale after pedestal refitting

**Evidence.** `src/mophongo/saturate.py:981-1029` computes residual fraction,
ring SNR, and data/model statistics before an optional pedestal refit changes
the amplitude/final model. Acceptance at line 1075 and published metrics at
1147-1185 continue using the old values. A final re-drizzle failure is swallowed
at 937-945, leaving an old-position PSF paired with a new center.

**Impact.** A star may be accepted/rejected on a model different from the one
written, and diagnostics can describe neither the initial nor final fit
consistently.

**Required fix/test.** Recompute every quality metric after any parameter/model
change. On redrizzle failure, either revert position/model coherently or fail
with context; never silently mix states.

### P1-21 — Catalog flagging uses a field-center PSF and permits an invalid subtract workflow

**Evidence.** Core repair evaluates a PSF per star, but catalog flagging uses
one PSF drizzled at the mosaic center
(`src/mophongo/repair.py:199-231,679-695,949`). The CLI also permits
`--mode subtract --catalog` (`repair.py:906,931-950`), then performs catalog
decisions on PSF-subtracted/core-blanked science although the flagger requires
repaired/observed science (`catalog.py:1586-1588`).

**Impact.** On spatially varying mosaics, spikes/wings may be over- or
under-flagged. In subtract mode, flux-ratio logic is semantically invalid.

**Required fix/test.** Supply a per-row PSF callback/region cache evaluated at
the fitted coordinate. Reject subtract+catalog, or retain a correct reference
image specifically for catalog decisions.

### P1-22 — Mock validation contains filter, randomness, and truth-validity defects

**Evidence.** In `src/mophongo/mock_mosaic.py`:

- filter-family parsing strips only trailing `W` (`:411-420`), so valid
  `F410M`, `F470N`, and `F322W2` were classified as MIRI;
- each filter initializes its RNG from the same default seed (`:900,932`), so
  same-shaped bands receive the same standardized noise realization;
- a source is marked valid before paste bounds are checked (`:1310,
  1326-1329`), and a skipped edge paste does not clear validity/truth
  (`:1368-1387`).

**Impact.** Validation can use the wrong instrument family, understate
multiband uncertainty through correlated noise, and compare recovered flux to
truth for a source that was never painted.

**Required fix/test.** Parse the leading wavelength with an explicit JWST
filter table, derive independent deterministic child RNG streams, and define
validity only after a complete paste. Add multi-filter covariance and
edge-source truth-image consistency tests.

### P1-23 — `make_extended_grid` fails on the repository's shipped STDPSF grid

**Evidence.** `src/mophongo/jwst_psf.py:324` assumes
`emp_grid.meta["detector"]`. The shipped
`data/PSF/UDS_NRCA5_F444W_OS4_GRID25.fits` exposed only grid shape/positions
and oversampling through Photutils; `make_extended_grid(..., test=True)`
raised `KeyError`. The function also mutates input metadata and constructs
NIRCam unconditionally (`jwst_psf.py:324-334`).

**Impact.** An advertised PSF utility fails on the project's own data and
cannot correctly dispatch other instruments.

**Required fix/test.** Read detector/filter/instrument from the FITS header or
require explicit values, avoid mutating caller metadata, dispatch by
instrument, and round-trip every shipped grid family.

## P2 findings — conditional correctness and public API defects

### P2-01 — The alternate IVAR estimator is exported but uncallable

`calibrate_ivar_with_bg_median` reads local `bgmask` before assignment
(`src/mophongo/catalog.py:321-323`), reproduced as `UnboundLocalError`; it then
repeats the inverted background dilation at 358-361. Its docstring says block
sum while code uses mean. Delete it from the public surface or repair and test
it independently.

### P2-02 — “Safe” dilation is label-order dependent

`safe_dilate_segmentation` protects original segments but lets later labels
overwrite earlier dilated halos (`src/mophongo/catalog.py:392-414`). A
contested pixel changed to the later/higher label in a two-source probe.
`docs/catalog.md:203-205` falsely says neighbors never overwrite each other;
the Wren notes correctly describe the id bias. Use simultaneous nearest/
competitive assignment with a geometric tie-break.

### P2-03 — Non-joint astrometry options are accepted and ignored

`FitConfig.astrom_model` and `astrom_centroid` are documented and stored
(`src/mophongo/fit.py:75-80`), but the pipeline scene path does not dispatch to
them. `fit_astrometry_joint=False` is flux-only (`scene.py:881-884`), while the
joint path always reads polynomial order. Even an invalid model string is
accepted. Either implement the documented separate path or remove the fields
and correct `docs/fitting.md:468-475` / `docs/pipeline.md:518-537`.

### P2-04 — `generate_scenes` fails on its own defaults and empty input

The public function defaults `minimum_bright=None` and forwards it into a
numeric comparison; two disconnected templates reproduced `TypeError`
(`src/mophongo/scene.py:533-657`). Empty input reaches `labels.max()` around
line 691. This contradicts status claims that the default was fixed. Define
`None` semantics or an integer default and return a typed empty result.

### P2-05 — Parent/component bookkeeping tests a nonexistent attribute

`src/mophongo/pipeline.py:1393,2439-2441,2957,4144` checks `parent_id` before
reading the actual `id_parent`. It normally falls back to `id`, so prebuilt
components with distinct child/parent ids are booked or diagnosed under the
wrong source. This is already suspected in `TODO.md`; replace every branch
with one tested `id_parent` resolver.

### P2-06 — Multi-component output aggregation is not covariance-correct

The pipeline sums component errors in quadrature despite a joint correlated
solve (`src/mophongo/pipeline.py:2494-2503`), collapses duplicate ids in stamp
maps, and overwrites aperture rows component by component. Persist the relevant
covariance/linear-combination variance or explicitly declare multi-component
science output unsupported.

### P2-07 — Edge apertures use different numerator and denominator support

The raw aperture uses the in-image patch, while the correction denominator can
integrate the full padded template (`src/mophongo/pipeline.py:2799-2805,
2963-2988`). Near an edge, denominator flux exists where numerator data do
not. The current flux-estimator report identifies this correctly. Clip both to
one support or flag/exclude incomplete apertures.

### P2-08 — Stamp provenance checks are weaker than the documentation claims

Stamps store `RUNNAME` but the loader checks only filter number and shapes
(`src/mophongo/pipeline.py:1847-1860,2095-2149`). A same-shaped file from
another run can load silently. Validate run/config/input/PSF/kernel identity
and the full metadata contract; narrow `docs/outputs.md:449-471` until then.

### P2-09 — `SparseFitter.add_flux_priors` misindexes selected-length arrays

`src/mophongo/fit.py:333-380` broadcasts mean/sigma to the selected length and
then indexes them with original parameter indices. `idx=[2]` with one selected
mean reproduced `IndexError`. Normalize selection to integer indices and
separately accept scalar, full-length, or selected-length inputs.

### P2-10 — WCS shape axes are swapped and SIP rescaling is incomplete

`src/mophongo/templates.py:188-198,315-316` assigns `(ny,nx)` to both
`array_shape` and `pixel_shape`, although `pixel_shape` is `(nx,ny)`. A
rectangular `(7,11)` probe reported transposed metadata. Pixel rescaling moves
SIP CRPIX but not order-dependent polynomial coefficients. Test rectangular
and distorted WCS world-coordinate invariance; use Astropy-supported slicing/
resampling semantics where possible.

### P2-11 — Shift diagnostics can report zero and scalar calls return arrays

`src/mophongo/scene.py:977-1014` needs a spatial tree that scene construction
can discard; a real first-pass shift can therefore display as zero. Scalar
detection occurs after `np.atleast_1d`, so scalar inputs return length-one
arrays. Retain/evaluate the fitted polynomial independent of the tree and
capture scalar-ness before coercion.

### P2-12 — Saturated singleton scenes cannot fit the shift comments promise

Scene construction makes ungrouped saturated templates singletons and forces
them bright/order zero (`src/mophongo/scene.py:665-695,869-896`), but the block
builder refuses fewer than two bright templates (`:459-463`). The result is
always flux-only. Permit a stable singleton shift or remove the misleading
special-case claims.

### P2-13 — Residual FITS headers use the wrong physical metadata

`src/mophongo/pipeline.py:1762-1767` always begins from the high-resolution
science header. On an upsampled residual the WCS may be right, but BUNIT and
photometric calibration belong to the fitted band; on other projection modes
even WCS can be wrong. Build a fitted-band header and replace only grid/WCS
cards needed for the actual residual array.

### P2-14 — `psf_model` resizing drops scientific metadata

`src/mophongo/templates.py:1192-1225` manually copies a stale subset and loses
`is_star`, `sat_group`, EE/norm fields, extension mode, and provenance. This
can change scene grouping after resize. Use one exhaustive metadata contract
(`copy_meta_to`) and operation-specific overrides.

### P2-15 — JSON per-band aperture arrays are silently ignored

`src/mophongo/pipeline.py:2699-2715` recognizes only `np.ndarray`, while JSON
necessarily supplies a list. Lists fall through to a default; actual arrays
cannot be serialized by `save_config`. Normalize sequences during config
validation and serialize canonical lists.

### P2-16 — Partial IVAR coverage inflates scheme SNR

Template-scheme SNR functions sum flux over all segment pixels but variance
only over positive-weight pixels (`src/mophongo/template_schemes.py:511-523,
620-631,711-724`). Signal in invalid pixels therefore carries no noise. Use one
common valid mask or fail/fallback conservatively when coverage is incomplete.

### P2-17 — Scene-catalog mode exits the interpreter and ignores `out_dir`

`src/mophongo/pipeline.py:3300-3327` writes `scene_catalog_<i>.ecsv` to the
process CWD and calls `sys.exit()` from library code. This can terminate a
notebook, service, or test process. Return a result/early state and keep process
termination in an explicit CLI only.

### P2-18 — Region-map convolution corrupts integer/non-finite input

`PSFRegionMap.convolve_image` allocates output with the input dtype and casts
the floating convolution back (`src/mophongo/psf_map.py:715-757`). An int16
impulse convolved with a 3x3 mean kernel became all zeros. `nan_to_num` maps
infinity to a huge finite value; a probe produced values around `1.7e291`.
Return a floating dtype and explicitly mask all non-finite input.

### P2-19 — Region-map EE cache ignores in-place PSF changes

Cache validity uses only `id(self.psfs)` (`src/mophongo/psf_map.py:619-624`).
Multiplying a cube in place changed its sum from 9 to 18 while cached EE stayed
9. Encapsulate mutation with versioned invalidation, replace rather than mutate,
or expose read-only arrays.

### P2-20 — Sidecar naming can overwrite a region file

`src/mophongo/psf_map.py:769-785` uses string replacement of lowercase
`.geojson` to derive a FITS sidecar. For `.gpkg`, uppercase suffixes, or another
advertised driver, replacement is a no-op and FITS can target the geometry
path. Use `Path.with_suffix(".fits")` and explicitly limit supported formats.

### P2-21 — Matching-basis recentering swaps x/y

Photutils centroids are `(x,y)`, but `src/mophongo/psf.py:1553-1559,
1573-1579` assigns `ycen, xcen`. Asymmetric basis kernels shift along the
wrong axes. The target center also differs from the package's even-kernel
convention and `mode="nearest"` can change sum. Test asymmetric odd/even
kernels using the shared centering convention.

### P2-22 — `DrizzlePSF.register` can return/use unbound variables

On centroid failure, `dx`, `dy`, and `dr` are unset; with zero iterations,
`psf` is unset (`src/mophongo/psf.py:2918-2939`). Its docstring advertises an
absent argument. Validate iterations, handle a failed centroid explicitly,
and correct the API documentation.

### P2-23 — PSF realized metadata can be misaligned or stale

`stamp_encircled_energy` appends `r_ee` only for stamps that reach the
requested fraction (`src/mophongo/psf.py:2029-2046`), so per-stamp arrays lose
cube alignment; a two-stamp probe returned two EEs and one radius. Invalid
DrizzlePSF calls also leave prior realized size/EE fields attached to the new
request (`psf.py:2748-2755`). Append NaN placeholders and clear realized state
at method entry.

### P2-24 — Reciprocal pixel ratios and mixed-parity padding use wrong geometry

For `pixel_ratio < 1`, matching code shrinks the source by that ratio instead
of enlarging by its reciprocal (`src/mophongo/utils.py:740-745` and
`psf.py:1297-1301`). Separately, odd-to-even padding leaves a centered delta at
the lower central pixel, conflicting with the package's `shape//2` convention
(`utils.py:369-373`). Share one center-preserving pad/resample helper and test
reciprocal ratios plus all parity combinations.

### P2-25 — PSF factory date/cache behavior is not canonical

`_modal_mjd` can return a singleton observation MJD plus 2.5 days; ISO dates
work in the backend but `save=True` later casts the date to float; cache hits
are skipped without returning grids/paths and existing headers are not checked
(`src/mophongo/psf_factory.py:79-87,308-311,397-421`). Canonicalize through
`Time`, return a typed product record for hit/miss, and validate provenance.

### P2-26 — `write_stdpsf` corrupts tuple metadata

`src/mophongo/jwst_psf.py:388-471` interprets every tuple as FITS
`(value,comment)`. Photutils metadata such as `grid_shape=(5,5)` is therefore
written as value 5/comment 5. Eight-character key truncation has no collision
check and missing grids reach `len(None)`. Use an explicit FITS provenance
schema and validate required grid metadata.

### P2-27 — Empirical/theoretical PSF blending scales before pedestal correction

`src/mophongo/jwst_psf.py:263-272` solves the halo scale from the raw core and
then subtracts a pedestal, so the corrected core and halo no longer match at
the documented normalization radius. Estimate/subtract background first, then
fit the scale with finite/nonzero checks and a guaranteed background sample.

### P2-28 — Mock aperture truth ignores actual subpixel phase

`src/mophongo/mock_mosaic.py:1553-1564` centers apertures on the geometric stamp
center, while painting uses a phase-aware cutout origin. Even stamps can differ
by half a pixel. Integrate at each source coordinate relative to the actual
paste origin or measure directly from the painted truth image.

### P2-29 — Astrometry threshold and polynomial rank handling are broken

`AstroMap.snr_threshold` is not passed to `_measure`
(`src/mophongo/astrometry.py:411-447`). Missing SNR mutates the caller table to
exactly the threshold, then strict `>` rejects every row. Polynomial fitting
forms normal equations and calls `solve` without rank/sample checks
(`astrometry.py:181-189,325-332`); one order-2 source reproduced a singular
matrix. Pass configuration, avoid input mutation, and use weighted least
squares with rank-aware downgrade/failure.

### P2-30 — Several Catalog display/star APIs do not satisfy their signatures

- `find_stars` ignores `return_seg` and `chi2_max`, always returns a different
  tuple than annotated, and even-sized PSFs extract the wrong shape
  (`src/mophongo/catalog.py:903-970`).
- `show_stamp(ax=None)` creates but does not populate an axis (`:1026-1065`).
- `plot_bg` computes a fallback bin size but still divides by the original
  `None` (`:1082,1123`).
- `fit_psf_stamp` lacks finite/positive-sigma masking (`:430-446`).

All were reproduced. These should be fixed rather than documented as expected
workarounds in the public guide.

### P2-31 — Catalog implementation has shared mutable state, avoidable private API, and scaling debt

The private Photutils `DEFAULT_COLUMNS` import is immediately shadowed by a
local list; the dataclass default returns that same mutable global list; mid-file
imports repeat NumPy/Astropy/Scipy/Photutils blocks; package paths still print
instead of logging; `_deblend_label_info` scans the whole image once per label.
See `src/mophongo/catalog.py:22,65-74,417-454,549-595,708` and print sites around
798-949. Remove the private import, use immutable/copy defaults, consolidate
imports/logging, and vectorize the label contingency calculation.

### P2-32 — Saturation API exposes no-op/unchecked controls

`repair_saturated_holes` accepts any mode string; a typo follows the repair
branch. Exposed `sat_significance`, `max_resid_frac`, and `min_ring_snr` are
unused while a cutoff is hard-coded. Global sky includes non-finite science,
and OS4 sizing is assumed (`src/mophongo/saturate.py:543-725,1075`). Validate
mode and shapes, implement/remove knobs, filter finite sky, and read actual
oversampling.

### P2-33 — `repair_star` and repair FITS output mishandle MEF structure

`src/mophongo/repair.py:445-477` duplicates FITS traversal and can leave
`sci_arr` unbound when there is no data HDU; odd cutout sizes return one pixel
fewer. The general repair writer flattens an MEF into a new single-primary FITS,
discarding other extensions. Reuse `_read_image`, define exact size semantics,
and either preserve HDU structure or document/refuse unsupported MEFs.

### P2-34 — Public utility functions contain unconditional failures/no-ops

- `utils.rebin_wcs` references undefined `n`;
- `utils.retile_blocked` references undefined `time`;
- `CircularApertureProfile(pixel_scale=..., norm_radius=None)` divides by
  `None`;
- `write_wcs_csv` contains an unconditional `continue` and writes only a
  header (`src/mophongo/utils.py:145-165,651-661,1125-1127,2022-2037`).

These should be fixed, made private, or removed. Public docs currently expose
`write_wcs_csv` while admitting it is broken.

### P2-35 — Analytic PSF functions overstate discrete normalization

The Gaussian/Moffat utilities use continuous infinite-plane amplitudes, not
finite discrete-stamp normalization (`src/mophongo/utils.py:273-365`). Probes
gave finite 5x5 sums 1.117, 0.914, and 0.678 for representative profiles while
docstrings say “specified total flux”/normalized. Given the shape-throughput
policy, document this as a continuous-profile integral or explicitly normalize
the discrete array when that is the intended API.

### P2-36 — Positivity and error handling can return a model inconsistent with the stated solve

`SceneFitter` clips negative coefficients after an unconstrained solve without
refitting remaining active sources (`src/mophongo/scene_fitter.py:242-244,
299-300`). Its inverse-diagonal error path clamps negative values to `1e-12`,
which can hide indefinite systems (`:305-326`). Use NNLS/active set if
positivity is a science constraint, validate SPD/factorization, and surface
invalid covariance rather than manufacturing tiny errors.

### P2-37 — Sparse labels and template metadata operations have pathological behavior

`build_ownership` uses `find_objects`, whose allocation scales with maximum
label rather than source count (`src/mophongo/template_schemes.py:455-475`).
Large sparse catalog ids can consume excessive memory. `Template.pad` loses
metadata and has confusing `inplace=True` return behavior
(`templates.py:617-653`). Use compact label maps and the shared metadata-copy
contract.

### P2-38 — PSF lookup extrapolation and serialization are under-specified

`PSFRegionMap.lookup_key` / `DrizzlePSF.get_psf` fall back to the nearest
region/frame with no maximum distance, potentially extrapolating arbitrarily
far. Region serialization loses pscale, tolerances, footprints, and much
provenance (`src/mophongo/psf_map.py:266-299,552-592,769-785`; `psf.py:
2494-2508`). Add strict/maximum-distance modes, validate cube key/count, and
round-trip physical/provenance metadata.

### P2-39 — A verification diagnostic fabricates covariance

`src/mophongo/verification.py:1377-1378` constructs `0.1 * identity` and labels
it “Covariance Matrix” at 1440-1444. A reader can mistake a placeholder for
fitted uncertainty evidence. Accept the real covariance or render an explicit
“unavailable” panel.

### P2-40 — One test ignores `tmp_path` and writes outside the test sandbox

`tests/test_astrometry.py::test_polynomial_astrometry_reduces_residual`
overwrites its fixture with `Path("../tmp")`. The full suite therefore failed
only while saving `../tmp/diagnostic_poly_shift.png`; the numerical assertions
had passed. Use the supplied fixture and do not leave diagnostics outside the
test temp directory.

## Conceptual and architectural concerns

These are not all demonstrated bugs, but they should be resolved before the
scientific contract is considered stable.

### C-01 — Define the estimator being recommended, not merely every estimator that can be emitted

The aperture path multiplies model plus residual by a potentially large total
correction (`src/mophongo/pipeline.py:2981-3008`). This amplifies residual noise
and any residual pedestal. The current Wren flux-estimator report correctly
frames model-only, aperture, and residual-unscaled estimators as a
bias/variance trade, but the public docs do not choose a validated primary
estimator with a propagated uncertainty. Keep compatibility columns, but name
one science product only after injected-truth validation across size, blend,
SNR, position, and depth.

### C-02 — “Absolute” encircled energy is not yet established end to end

`scratch/wren/encircled_energy.tex` says the native STPSF stamp sum is an
absolute fraction of infinite-aperture flux, but its own scope later says
totals are relative to the model grid. Example parent grids sum to about 0.93
and 0.99, and the report normalizes growth curves to those grids. It is not yet
shown whether sub-unity parent sums are missing finite support, physical
throughput, optical loss, or an array normalization convention. Dividing an
already calibrated science amplitude by the wrong interpretation can
double-correct. Establish the calibration contract with a point-source truth
whose infinite-aperture normalization is independently known.

### C-03 — Scalar global noise calibration cannot represent mosaic depth variation

`get_bg_and_ivar` produces one scalar normalization per image. Even after its
mask/detection bugs are fixed, local correlated-noise and weight conventions
can vary with coverage, kernel, and exposure geometry. The Wren noise report
acknowledges this. Validate and, if needed, implement a robust spatial
calibration map at a scale tied to the measurement aperture.

### C-04 — Block replication requires stronger WCS admissibility checks

`get_wcs_pscale` and `bin_factor_from_wcs` use one axis. The pipeline assumes
integer, conformal, nested grids, but does not prove rotation/distortion/
pixel-area compatibility. Before block replication, check both axes and pixel
area (or the determinant), require close-to-integer isotropic ratio, and verify
the pixel-center transform. Otherwise reproject explicitly.

### C-05 — The validation comparison is not fully independent

The IDL and Python comparisons share detection data, mosaics, PSF assumptions,
selection cuts, and often templates or catalog conventions. Agreement is
valuable but does not establish absolute calibration independently. The
injected-truth leg is more independent, but currently has the PSF-map and mock
defects above. State validation layers separately: algebraic equivalence,
synthetic truth, code-to-code comparison, and external calibrated-star truth.

### C-06 — Pull width and residual/noise below one are diagnostic failures, not automatic success

`scratch/wren/verification.tex` reports residual/noise about 0.80 and pull
width about 0.72 while calling flux recovery validated. Those values suggest
overestimated noise/error, covariance with the fitted model, selection effects,
or a mismatch in the denominator. They require an explained noise model, not
only a median flux ratio.

### C-07 — Selection makes validation conditional

The reported comparisons exclude nonpositive, unmatched, deblended, or
unusable sources, and real-data conclusions are restricted to selected
high-SNR populations. Report completeness and failure rates by truth class;
otherwise an unbiased retained median can coexist with serious selection bias.

### C-08 — Module size and statefulness make invariants difficult to defend

`pipeline.py` is about 4,560 lines, `psf.py` about 3,108, and several classes
mix I/O, caching, mutation, diagnostics, and algorithms. The immediate fixes
should be scoped, but longer term the native-data/working-data/product state
needs explicit dataclasses and pure transformation boundaries. The current
module-boundary rules are good; the next step is enforcing state transitions
inside the large modules.

### C-09 — Cache provenance should be content/contract based

Map, stamp, and factory caches currently mix partial metadata, object identity,
filename identity, and mutable arrays. Define a versioned provenance record
containing input identity/hash, physical sampling, WCS/grid, algorithm,
parameters, dependency/backend versions, and output schema. Validate it in one
place for every cache product.

### C-10 — Public API policy is not explicit

Sphinx uses `ignore-module-all=True`, so many implementation helpers and broken
alternates become public even when module `__all__` tries to hide them. Decide
which functions/classes carry compatibility guarantees; make everything else
private or omit it from API stubs.

## Documentation reconciliation

### Deployed Read the Docs

Checks against the public site on 2026-08-12:

- `https://mophongo.readthedocs.io/en/latest/` returned HTTP 200.
- The deployed `_sources/*.md.txt` for all 14 narrative pages were
  byte-for-byte identical to local `docs/` sources.
- The sitemap advertises both `latest` and `stable`, but
  `https://mophongo.readthedocs.io/en/stable/` returned the Read the Docs 404
  page. A public release should not advertise an unusable stable version.
- A local `sphinx-build -W --keep-going -b html docs ...` completed output but
  exited nonzero with 17 warnings/errors. A stricter fresh/nitpicky build
  emitted hundreds of messages (many are offline intersphinx resolution), but
  the source-controlled parser/reference failures remain real.

Confirmed public documentation problems include:

- malformed docstrings in `catalog.calibrate_ivar_with_bg_median`,
  `fit.add_flux_priors`, `PSF.auto_matching_kernel_window`,
  `scene.build_scene_tree_from_normal`, `Scene.plot`, `templates.EXTEND_MODES`,
  `Templates.add_component`, and `Templates.apply_template_shifts`;
- `docs/pipeline.md:474-480` links nonexistent `Pipeline.plot_subphot`; the
  current helper is `diagnose_subphot`;
- `docs/catalog.md:22-30` still warns of double background subtraction even
  though source, STATUS, and TODO say it was fixed;
- `docs/catalog.md:177-185` documents broken `show_stamp`/`plot_bg` defaults as
  user workarounds instead of fixing them;
- `docs/catalog.md:189-205` describes source masking/dilation and safe dilation
  contrary to the live polarity/order behavior;
- `docs/fitting.md` and `docs/pipeline.md` describe a non-joint astrometry path
  that is absent;
- `docs/outputs.md` overstates cache validation and does not explain the wrong
  fitted-band header semantics;
- `docs/pipeline.md` exposes `write_wcs_csv` while acknowledging that it emits
  no rows;
- public API generation exposes broken/dead helpers because it ignores
  `__all__`.

`STATUS.md` claims zero page-level documentation warnings and contains an old
description of the theme/publication set; current `docs/conf.py` uses Furo and
publishes the full user guide. Add a docs CI job only after distinguishing
network-dependent intersphinx failures from source warnings, then make source
warnings fatal.

### `GUIDE.md`

The implementation guide is not a reliable current contract:

- it still says code belongs under `dotfit`;
- it says input weights are proportional to variance, while the package and
  all fitting math require inverse variance;
- it mandates one filter-average throughput and says not to apply
  region-dependent corrections, while code, STATUS, Wren EE notes, and public
  docs now use per-source `ee_psf_lo` with a filter fallback;
- it describes the active environment as Photutils 1.12 and a runtime ceiling
  near 1.13, while the reviewed Poetry environment imports Photutils 2.3.0,
  NumPy 2.2.6, Astropy 7.1.0, and DrizzlePac 3.9.1 successfully;
- its upgrade plan and private-API discussion mix historical investigation
  with current instructions.

Replace the document with a short set of current invariants and move historical
dependency investigations to a dated note.

### `STATUS.md` and `TODO.md`

Both files are valuable raw history but are too long and contradictory to act
as authoritative current state.

Examples:

- STATUS says the standalone scene default was fixed; code still defaults to
  failing `None`.
- STATUS says repeated mid-file catalog imports were removed; they remain.
- STATUS says fully reproducible snapshots; Wren/classic snapshots are
  corrupted.
- STATUS/TODO contain both completed and still-open variants of `wht_hi`,
  `PSFSZ/RCIRC`, `ee_psf_lo`, and documentation work.
- STATUS describes `ap_flux_total_<i>` as implemented in places, while TODO
  still requests it and current source/public docs do not expose that exact
  column.
- old line numbers and defaults (`astrom_shift_tol` 0.02/0.05) remain after
  the runtime default moved to 0.1.

Recommended structure:

1. a short current release matrix: supported entry points, invariants,
   validation status, and known blockers;
2. one prioritized TODO list with ids that link to this review/issues;
3. an archived chronological changelog for completed/historical notes;
4. no claim marked complete without a regression test or artifact link.

### Wren TeX/PDF set

All current PDFs rendered cleanly and are readable. The central problem is
scientific/version drift, not layout.

#### `encircled_energy.tex/pdf`

Strongest parts: clearly separates fitting shapes from stamp sums, documents
per-source EE, and quantifies spatial variation. Required corrections:

- distinguish infinite-aperture calibration from “relative to the finite
  parent model grid” throughout;
- reconcile the absolute-flux claim with parent-grid sums below one;
- cite the exact normalization convention of STPSF/STDPSF products and test it
  independently;
- update the flow/caching discussion for lost `pscale`/provenance and mutable
  EE cache behavior.

#### `fit.tex/pdf`

It is pinned to `a2fecb1`, uses a 0.05-pixel tolerance instead of 0.1, calls
stored shifts shifts “applied to the data” when templates are shifted, and
describes reconstruction differently from the current stamps-first
`load_fit`. Most importantly, lines 331-335 explicitly record P1-02 but frame
it as an accepted caveat. It should instead be an implementation blocker and
include the dense joint-block defect P1-01.

#### `flux_estimator_comparison.tex/pdf`

This is a valuable forensic notebook but not a coherent current specification.
It is pinned to older commits, interleaves a historical implementation with
later status paragraphs, and contains statements that are simultaneously
marked fixed and still described as live. It correctly identifies the
edge-aperture mismatch and order-dependent dilation. It incorrectly says
`generate_scenes` now defaults to 10, retains stale ridge/normalization
descriptions, and includes planned columns alongside actual outputs. Split it
into:

- a dated historical audit;
- a short current estimator contract generated/tested against output schema;
- a separate research note for proposed shrinkage/shape-aware estimators.

#### `noise_background.tex/pdf`

Its controlled tests validate scalar rescaling on source-free noise, not live
source masking. The description “mask sources, dilate” is opposite to current
mask polarity and omits the 29x threshold mismatch. It says
`_mean_downsample` is defined twice (now once) and calls the alternate estimator
merely unused rather than uncallable. Page 5 also exposes an internal home path,
branch, MINERVA release names, and workflow details; scrub or explicitly
approve those before public release.

#### `psf.tex/pdf`

This is the most internally coherent current note, but it overstates cache/
serialization completeness and arbitrary-oversampling support. It should cover
hard-coded OS4 paths, filename collisions, band-map versus overlay centroids,
unbounded nearest-region fallback, `pscale` persistence, cache invalidation,
and the absolute-parent-grid caveat.

#### `template_comparison.tex/pdf`

The note is transitional. Its status box says current main uses unit-sum
kernels, while its normalization ledger and later “ivo” analysis still derive
defects from an unnormalized kernel. The title metadata still names a
`flux-bug` branch. It correctly exposes id-ordered dilation and the unresolved
shape problem for extended sources. Rebuild from current source or label the
entire document historical; a small disclaimer is not enough when the body
contains mutually incompatible runtime descriptions.

#### `verification.tex/pdf`

It is stale against current STATUS v6/v7 and repeatedly says `ee_psf_lo`
propagation is inactive, which was fixed. Reproduction points to the old
driver and old support assumptions. “Independent IDL” should be qualified by
shared inputs/assumptions. Residual/noise 0.80 and pull width 0.72 need diagnosis,
not presentation as ideal validation. Add completeness/selection reporting,
separate truth/code-comparison claims, and regenerate only after P1-12 and
P1-22 are fixed.

#### Historical version handling

The `_vN` files are useful provenance but should live under an explicit
`archive/` with a generated index containing date, commit, superseded-by link,
and known invalidated claims. Top-level current-looking filenames should never
depend on readers recognizing suffix conventions.

### Code comments and docstrings

Comments are often excellent, but several actively conceal risk by normalizing
known defects:

- “note wcs origin is wrong” remains next to live WCS construction;
- `Template.id_parent` is called redundant while downstream code has four
  broken parent branches;
- the pipeline module header advertises nonexistent/future entry points;
- saturation group-id comments/docstrings alternate among “1,” lowest label,
  and nearest/core-reaching label;
- multiple config fields are described as live while ignored;
- package modules retain `@@@`, dead imports, local imports, commented-out
  alternatives, and print diagnostics in production paths.

A code comment should either state an invariant enforced by a test or link to a
tracked issue. Known live correctness defects should not remain only as
comments.

## Packaging, CI, and release engineering

Confirmed state:

- `pyproject.toml` version is `0.0.1` and description is empty;
- `mophongo.__version__` is absent;
- `Template` and `Templates` are imported at package top level but omitted from
  `__all__`;
- the package eagerly imports heavy catalog/PSF-map dependencies;
- CI runs only macOS/Python 3.12 and `pytest`; Python 3.11 is declared supported
  but not tested;
- CI has no docs, warning, lint/static, packaging/install, or coverage gate;
- no test currently guards the highest-risk contracts found here: dense
  astrometry equivalence, final basis stationarity, background source masks,
  config snapshot equality, WCS invariance, repair catalog/seg consistency,
  output path uniqueness, arbitrary oversampling, or mock noise independence.

Before a public release:

1. settle versioning and expose `__version__` from installed metadata;
2. add a real package description and supported-platform matrix;
3. test Linux and macOS on Python 3.11/3.12 (or narrow the declared range);
4. add package-build/install and strict source-doc builds;
5. treat unexpected warnings as failures after cleaning current warnings;
6. explicitly curate public API and internal/public documentation;
7. decide whether MINERVA/CANFAR/internal paths and artifacts belong in the
   public repository; no credential audit was performed as part of this code
   review.

## Validation performed

### Automated checks

- Full suite: **237 passed, 1 failed, 182 warnings in 76.11 s**.
  The one failure is P2-40: an otherwise-passing astrometry test tries to save
  outside the pytest temp directory.
- Focused solver/template/pipeline audit: **113 passed**.
- Focused PSF/WCS/mock/astrometry audit: **63 passed** before the same external
  diagnostic-path failure.
- Focused catalog/repair suites: **26 passed**.
- `poetry check`: passed.
- `python -m compileall -q src`: passed.
- `python -m pip check`: no broken requirements.
- Sphinx warnings-as-errors build: failed with 17 source/network warnings;
  output HTML was still generated.

Observed environment:

| component | version |
|---|---|
| Python | 3.12.9 |
| NumPy | 2.2.6 |
| SciPy | 1.16.0 |
| Astropy | 7.1.0 |
| Photutils | 2.3.0 |
| DrizzlePac | 3.9.1 |

### Focused reproductions

The audit ran small deterministic probes for, among others:

- exact versus implemented astrometric blocks;
- final shifted-basis flux stationarity;
- source-mask polarity/occupancy and convolved-noise scaling;
- NaN preprocessing and template flags;
- Wren/classic config round trips;
- WCS sky-coordinate invariance after template operations;
- coordinate bin/expand round trips;
- numeric catalog apertures;
- blank catalog detection;
- saturated child conflicts, id-less fit grouping, and catalog/seg labels;
- MEF data/header pairing;
- integer/infinite region-map convolution;
- arbitrary oversampling/factory filenames;
- shipped-grid extension;
- filter parsing, multiband RNG, and edge-paint validity;
- per-stamp EE alignment and cache invalidation;
- public utility failure paths.

### Positive checks and false positives ruled out

The audit also confirmed important correct behavior:

- `saturate.py` respects the documented module boundary and does not import
  catalog/segmentation/photometry state; `repair.py` is the permitted
  orchestrator.
- Native finite-stamp PSF sums are preserved, while pipeline matching uses
  unit-sum shapes.
- The shared `utils.fftconvolve` handles even-kernel alignment correctly.
- Inverse variance multiplied by `factor**2` on flux-conserving block
  replication preserves native chi-square/error scaling.
- Joint-solver Cholesky whitening/unwhitening is algebraically consistent.
- Template shift sign and x/y order passed to SciPy are correct.
- Template bounding-box maxima are inclusive; the scene `+1` shape is correct.
- Default `psf_wings` sub-unity sum after neighbor blanking is intentional.
- Adaptive `reg_flux` is relative to matrix scale; an older Wren “absolute
  ridge” claim is stale.
- `matching_kernel(recenter=False)` matches current implementation/docs.
- the earlier `PSF.gaussian` signature failure is fixed;
- PSF-region pickle/deepcopy rebuilding of prepared geometry works;
- invalid matching-PSF pixels and positive pixel-scale checks are handled;
- Read the Docs publication patterns correctly exclude internal uppercase
  developer notes from the rendered site;
- Poetry dependency metadata is internally consistent in the active env.

Passing existing tests does not negate the findings: many probes exercise
states absent from the suite, and `tests/test_catalog.py` in particular is only
one small deblend-info test.

## Recommended remediation sequence

### Gate 1 — Make fitted results mathematically self-consistent

1. Replace astrometric block construction with a dense-design-equivalent
   implementation (P1-01).
2. Add the final flux-only solve and stationarity test (P1-02).
3. Fix template WCS provenance, remapping, and footprint WCS use
   (P1-05, P1-10, P1-11).
4. Enforce finite inputs/templates and immutable per-run multiresolution state
   (P1-07, P1-09).

Do not interpret new verification medians until this gate passes.

### Gate 2 — Rebuild preprocessing and repair invariants

1. Repair/retune `get_bg_and_ivar` as one algorithm, not as a polarity-only
   patch (P1-03, P1-04).
2. Define empty Catalog behavior and valid flag combinations (P1-15).
3. Enforce catalog/segmentation bijection, unique star identity, applied shifts,
   and per-star PSFs (P1-16, P1-17, P1-21).
4. Make repair writes collision-safe and MEF-aware (P1-18, P1-19).
5. Recompute final saturation metrics (P1-20).

### Gate 3 — Make PSF and mock validation trustworthy

1. Remove OS4 assumptions and version all cache provenance (P1-14).
2. Generate PSF cubes on the geometry they belong to (P1-12).
3. Fix mock filter parsing, RNG independence, paint validity, and aperture
   phase (P1-22, P2-28).
4. Fix shipped-grid utilities and coadd alignment/variance (P1-13, P1-23).
5. Re-run synthetic truth with completeness, pull calibration, size/blend/SNR/
   depth/position strata, and failure counts.

### Gate 4 — Reconcile products, docs, and release claims

1. Fix snapshot and aperture contracts (P1-06, P1-08).
2. Resolve P2 output/provenance/API defects or remove them from the public API.
3. Replace GUIDE with current invariants; split STATUS/TODO history from current
   state.
4. Regenerate all current Wren reports from a named commit and archive versions.
5. Make Read the Docs strict-clean, publish a working stable version, and add
   docs/package matrices to CI.

## Minimum acceptance suite for the next review

A release candidate should not pass review without all of the following:

1. Dense-design equality for `A`, `AB`, `BB`, and RHS on overlapping scenes.
2. Final fitted-basis stationarity and model/residual/catalog consistency.
3. Background/IVAR recovery with injected sources, correlated noise, NaNs,
   depth gradients, and zero-weight borders.
4. World-coordinate invariance through every template/PSF resampling operation.
5. Config snapshot round-trip equality for every template mode.
6. Two-run pipeline idempotence with immutable native inputs.
7. Exact catalog-id/nonzero-seg-label equality after every repair mode.
8. Pairwise-distinct output-path validation and MEF data/header tests.
9. OS1/2/4/8 PSF sizing, EE, filename, and cache-provenance tests.
10. Source/target/overlay map-key-to-PSF geometry consistency.
11. Independent per-filter mock noise and truth-image/validity equality.
12. Even/odd kernel impulse alignment and spatial variance Monte Carlo.
13. Empty/blank/malformed public API tests for Catalog, Scene, PSF, and repair.
14. Strict local docs build with zero source warnings and a live stable RTD URL.
15. A verification report that publishes completeness, failures, median bias,
    scatter, pull center/width, residual/noise, and results by size/blend/SNR/
    depth/position — with every quantity tied to a committed artifact.

Until those gates are met, Mophongo should be described as an active research
implementation with promising validation, not as a fully validated public
photometry package.
