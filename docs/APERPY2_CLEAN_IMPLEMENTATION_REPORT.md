# Aperpy 2 to Mophongo: clean-slate aperture-photometry assessment

**Audit date:** 2026-08-14  
**Upstream revision:** [`astrowhit/aperpy@dfe8a43`](https://github.com/astrowhit/aperpy/tree/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038) (`aperpy-2`)  
**Question:** how difficult is a clean Mophongo implementation whose default
PSF model is `DrizzlePSF` plus `PSFRegionMap`?

## Executive decision

This is a **go**, with one architectural condition: implement a Mophongo-native
`AperturePipeline` beside the existing template-fitting `Pipeline`. Do not wrap
the upstream scripts and do not expand the current two-image fitting
orchestrator into a second, unrelated pipeline.

The short answer on difficulty is:

| Deliverable | Assessment | One experienced scientific-Python developer |
|---|---|---:|
| Replace Aperpy's empirical PSF with one representative `DrizzlePSF` per band | Easy, but only a demonstration | 2--4 days |
| Spatially matched fixed-aperture prototype on an existing catalog/segmap | Straightforward to moderate | 1--2 weeks |
| Native core MVP: detection, fixed/masked/isophotal/Kron photometry, adaptive aperture, local totals | Moderate | 4--6 weeks |
| Release-quality core: masks, uncertainties, performance, persistence, diagnostics, real-field validation | Moderate to hard | 8--12 weeks |
| Full Aperpy feature parity, including HST, survey flags, Gaia/dust, and EAZY integration | Large scope | 10--16 weeks |

These are engineering estimates, not measured schedules. They assume access to
representative mosaics, exposure-WCS tables, and an agreed reference catalog.

The important distinction is that the **PSF foundation is mostly present**.
Mophongo already constructs exposure-aware PSFs, partitions the mosaic by
coverage, builds local matching kernels, and convolves a whole image region by
region. The hard work is instead:

1. defining and validating a local curve-of-growth contract;
2. propagating masks and noise without inventing a false diagonal weight map;
3. reproducing SEP/Kron/catalog semantics intentionally;
4. hardening PSF artifacts, cache provenance, and no-coverage behavior; and
5. establishing scientific validation that upstream does not provide.

A useful summary score is **7/10 for architectural feasibility, 8/10 for PSF
machinery already available, and 4/10 for validation readiness**. Overall this
is a moderate clean implementation, not a difficult PSF research project.

Here, "clean" means a clean-slate, Mophongo-native design informed by Aperpy's
published behavior and catalog equations. It does **not** mean a formal
evidentiary clean-room process. A formal clean room would require an independent
specification team and an implementation team that had not inspected the
upstream source; this audit cannot serve that purpose.

## Audit basis and confidence

The audited branch is 17 commits ahead of, and zero behind, Aperpy's `main`
branch. It is an unreleased development branch: no release tag contains its
tip. The [`main...aperpy-2` comparison](https://github.com/astrowhit/aperpy/compare/abe3b57acaeb6e4d66362aa7dd857dd7b1eddfc6...dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038)
contains 2,030 additions and 579 deletions across 18 files.

The analysis used a commit-pinned source clone, inspected every pipeline
module, and compared the behavior with the current Mophongo source, tests, and
design notes. Aperpy was not run end to end because the repository contains no
test data, fixture, portable configuration, or reference products. Therefore:

- confidence in the architecture and algorithm map is high;
- confidence in exact column-by-column numerical parity is necessarily lower;
- effort estimates include building a reference fixture that upstream lacks.

The upstream `src/` tree is only 5,178 lines across 11 Python files. It has no
package metadata, package namespace, test directory, CI workflow, or library
API. Its pinned [`requirements.txt`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/requirements.txt)
is an environment snapshot rather than a minimal dependency declaration.
Most of the scientific dependencies already exist in Mophongo, including
NumPy, SciPy, Astropy, SEP, Photutils, scikit-image, OpenCV, and STPSF.

This small, script-oriented upstream surface is why a rewrite is less risky
than an adapter or direct port. The challenge is scientific fidelity, not code
volume.

## What Aperpy 2 does

The [`README`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/README.md)
describes a front-to-back aperture catalog pipeline for HST and JWST galaxy
fields. The actual driver is a 64-line script, [`bin/example.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/bin/example.py#L1-L64),
in which the user edits Boolean stage switches. It launches the remaining
scripts with `subprocess.run(..., shell=True)`. State moves through FITS files,
Astropy tables, NumPy pickles, global configuration dictionaries, filename
globs, and string substitutions.

The branch implements the following pipeline.

| Stage | Aperpy behavior | Clean Mophongo disposition |
|---|---|---|
| Resampling | Block-reduces selected short-wave images, borrows WCS header values from another mosaic, and moves originals into `raw/` | Do not port. Require an explicit aligned-grid manifest or use a real WCS reprojection/drizzle stage |
| Background | SEP background; optional field-specific median filtering for cluster light | Reuse Mophongo background primitives; keep cluster subtraction as an optional survey adapter |
| PSFs | Detects stars, recenters and stacks them into one empirical PSF per filter | Replace the default with a `DrizzlePSF`-derived map; retain empirical stars for QA/fallback |
| Matching | Computes one filter-level kernel to a global `MATCH_BAND` and convolves the entire mosaic | Use source/target map overlays and a local kernel map |
| Detection | Noise-equalized coadd or spatially rescaled chi-mean image | Port as pure, tested coadd strategies; Mophongo already has a simpler matched inverse-variance coadd |
| Extraction | SEP detection plus native and matched isophotal, circular, masked-circular, Kron/AUTO, and flux-radius measurements | Implement a pure measurement engine; preserve SEP semantics where catalog compatibility matters |
| Errors | Random empty apertures and a fitted correlated-noise curve | Reimplement with spatial/depth stratification and explicit kernel metadata |
| Catalog totals | Reference-band Kron/aperture correction, PSF growth-curve floor, then the same correction in every matched band | Preserve the estimator, but query the local target PSF curve of growth |
| Adaptive catalog | Chooses one of six fixed apertures from isophotal area, with a blend shrink factor | Port as a small pure selection function |
| Release extras | Star/artifact/coverage flags, external crossmatches, Galactic extinction, DS9 regions | Optional postprocessors, not aperture-core responsibilities |
| Photo-z | Runs EAZY and optional zeropoint iterations | Downstream adapter or optional extra |

Relevant source locations are:

- empirical PSF and kernel construction in
  [`make_psfs.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/make_psfs.py#L42-L177);
- whole-image convolution in
  [`convolve_images.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/convolve_images.py#L21-L110);
- detection builders in
  [`build_detection.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/build_detection.py#L21-L305);
- forced measurements and empty apertures in
  [`source_extract.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/source_extract.py#L439-L620);
- total-correction equations in
  [`combine_catalogs_kronlike.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/combine_catalogs_kronlike.py#L265-L359);
- adaptive aperture selection in
  [`make_supercatalog.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/make_supercatalog.py#L21-L61); and
- the empty-aperture implementation and noise-curve fit in
  [`webb_tools.py`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/webb_tools.py#L55-L333).

### The present PSF coupling is narrow

Despite `psf_tools.py` being one of the largest modules, Aperpy's runtime PSF
contract is essentially:

```text
filter -> one empirical PSF array
filter -> one matching-kernel array
reference filter -> one global curve of growth
```

There is no coordinate in that contract. `make_psfs.py` constructs one
empirical PSF and kernel per band; `convolve_images.py` applies that kernel to
the full image; and the catalog combiner loads one reference PSF for every
source's aperture correction. This concentration is favorable: only four
operations need a new abstraction--PSF construction, kernel construction,
image application, and growth-curve lookup.

### The total-flux estimator worth preserving

Let source `i` have reference-band AUTO flux `F_AUTO,i`, reference matched
aperture flux `F_ref,aper,i`, aperture radius `R_aper,i`, and circularized AUTO
radius `R_AUTO,i`. Let `EE_ref(R)` be the reference PSF curve of growth. The
essential Aperpy correction is:

```text
F_ref,total,i = F_AUTO,i / EE_ref(R_AUTO,i)
C_morph,i     = F_ref,total,i / F_ref,aper,i
C_floor,i     = 1 / EE_ref(R_aper,i)
C_total,i     = max(C_morph,i, C_floor,i)
F_band,total,i = F_band,aper,i(matched) * C_total,i
```

The same source-specific correction is applied to every PSF-matched band,
which is what preserves matched colors. A spatial implementation does not
need to change the estimator. It changes the curve to
`EE_ref(ra_i, dec_i, R)` and records the PSF-region key used.

## What Mophongo already supplies

Mophongo's current PSF model is substantially more capable than Aperpy's
global-array contract.

### `DrizzlePSF`

[`DrizzlePSF`](../src/mophongo/psf.py#L2258) evaluates a detector-, epoch-,
roll-, exposure-WCS-, and drizzle-aware effective PSF on the mosaic grid. Its
stamps retain their finite native integral. That is a crucial advantage over
Aperpy's empirical finite stamp plus externally grafted encircled-energy
table: the stamp can carry its realized throughput explicitly.

### `PSFRegionMap`

[`PSFRegionMap`](../src/mophongo/psf_map.py#L33) already provides:

- footprint-derived spatial partitions;
- source/target map overlays;
- sky-coordinate to PSF-key lookup;
- one PSF or kernel stamp per region;
- local matching-kernel construction with unit-DC enforcement and response
  diagnostics;
- persistence as GeoJSON plus a FITS cube; and
- region-wise whole-image convolution.

The last point materially reduces the migration effort. Contrary to what a
source-only comparison might suggest,
[`PSFRegionMap.convolve_image`](../src/mophongo/psf_map.py#L1137) already pads
each region, convolves its cutout with the region's kernel, and writes only
pixels inside that region. It explicitly handles integer input and non-finite
pixels. Existing tests cover region selection, flux conservation, uncovered
pixels, input promotion, non-finite values, and FITS wrappers in
[`test_psf_map_convolve.py`](../tests/test_psf_map_convolve.py).

A focused audit run of the existing convolution and realized-EE tests passed:

```text
poetry run pytest -q tests/test_psf_map_convolve.py tests/test_psf.py \
  -k 'convolve or encircled or realized_ee'
20 passed, 14 deselected
```

The operator is ready to reuse for science arrays. It is not yet a complete
photometry homogenizer because it does not propagate masks, variance, or
covariance, and its hard region selection needs seam validation.

### Existing orchestration and catalog pieces

[`Pipeline.build_psfs`](../src/mophongo/pipeline.py#L1211) and
[`Pipeline.build_kernels`](../src/mophongo/pipeline.py#L1253) demonstrate the
correct two-band construction pattern. The pipeline normalizes copies of the
PSFs for matching, keeps the native stamp sums for throughput, overlays the
source and target geometry, and builds unit-DC kernels.

[`Catalog`](../src/mophongo/catalog.py#L859) already owns detection,
segmentation, deblending, WCS positions, Kron quantities, and source-table
assembly. [`lw_detection_coadd`](../src/mophongo/utils.py#L2448) supplies a
tested static-PSF inverse-variance detection coadd. The existing pipeline also
contains relevant aperture/total-correction algebra, documented in
[Photometry and Aperture Schemes](./PHOTOMETRY_APERTURES.md) and
[Flux Estimators](./FLUX_ESTIMATORS.md).

These components are reusable seams, not a reason to put aperture-catalog
orchestration inside the existing `Pipeline`.

## Capability and gap matrix

| Required capability | Current Mophongo coverage | Incremental difficulty | Decision |
|---|---|---|---|
| Exposure-aware JWST PSFs | High: `DrizzlePSF` | Low after preflight defects are fixed | Reuse behind a provider contract |
| Spatial PSF tiling and lookup | High: `PSFRegionMap` | Low algorithmically; medium hardening | Reuse with strict coverage semantics |
| Local source-to-target kernels | High | Low | Reuse unit-sum shape/unit-DC convention |
| Full-image spatial convolution | Implemented and tested for science arrays | Medium integration | Extend for mask and variance products |
| Local arbitrary-radius PSF COG | Partial: only box and one inscribed-circle EE are public | Medium | Add first-class cached COG API |
| Absolute throughput correction | Strong convention; unresolved drizzle closure at about the percent level | Medium/high validation | Preserve, then close the calibration gate |
| Detection coadds | Static matched inverse-variance coadd exists; no region-map or Aperpy chi-mean implementation | Medium | Generalize as an aperture-pipeline stage |
| Detection/segmentation | Photutils-based `Catalog` exists | Medium for SEP numerical parity | Reuse ownership; make backend semantics explicit |
| Forced circular/isophotal/Kron measurements in every band | Only pieces exist; current aperture path measures template model plus residual | Medium | New pure aperture engine |
| Neighbor-masked measurements | Segmentation infrastructure exists | Medium | New measurement masks with explicit flags |
| Empty-aperture noise model | No reusable core equivalent | High scientific importance | New `ApertureNoiseModel` |
| Adaptive `SUPER` aperture | No standalone equivalent; algorithm is small | Low | Port as a pure selector |
| Aperpy-compatible release columns | Partial concepts, different native schema | Medium | Compatibility writer, not internal schema |
| HST model-PSF default | `PSFBackend` protocol exists, but only JWST is registered | High | Add HST provider or require explicit empirical fallback |
| Gaia, dust, EAZY, survey flags | Mostly outside Mophongo core | Medium/large but separable | Optional adapters |

## Scientific contracts that must be explicit

### 1. Shape and throughput are different quantities

The clean implementation must retain Mophongo's existing invariant:

- a **native PSF stamp** keeps its finite integral, which is throughput/EE
  metadata;
- a **matching shape** is a copy normalized to unit sum;
- a **matching kernel** has unit DC, so convolution preserves the science
  image's flux scale.

Native stamps must never be silently normalized in the provider or cache.
Conversely, their finite sums must never leak into a kernel. This is the
highest-value Mophongo convention to preserve; see
[PSF Shape and Throughput Convention](./PSF_SHAPE_THROUGHPUT_CONVENTION.md).

### 2. The target is spatial, but colors remain matched

For band `b`, build its native map `P_b(x)` and the target-band map `P_t(x)`.
Their footprint overlay carries a local kernel `K_b(x)` satisfying, within the
validated approximation,

```text
P_b(x) convolved with K_b(x) -> P_t(x).
```

The target PSF may still change across the field. That is acceptable: every
band at one object is matched to the same local target PSF. A global uniform
PSF is not required for matched colors, provided every total correction uses
the target COG at that object's coordinate.

The target policy must fail closed. `F444W` cannot simply be assumed broader
than every input PSF at every coordinate. The planner should measure local
source/target response feasibility, disallow deconvolution by default, and
either choose a broader analytic envelope, exclude an infeasible band/region,
or require an explicit override.

### 3. Local curve of growth is the main missing PSF API

`PSFRegionMap.refresh_ee` currently caches only the full square-stamp sum and
the sum inside its one inscribed circle. Aperpy needs arbitrary physical
radii, including a per-source Kron radius. Add an API with semantics such as:

```python
ee_at(coord, radii, *, normalization="absolute")
radius_at_ee(coord, fractions, *, normalization="absolute")
```

It should cache a monotone per-region curve, accept physical units, state the
center and subpixel-integration convention, retain region alignment when a
fraction is unreachable, and return finite-support/coverage quality flags.

`normalization="absolute"` is not cosmetic. Normalizing the stamp by its
finite sum before evaluating the COG erases the beyond-stamp throughput that
Mophongo deliberately preserves. The current private/diagnostic COG helpers
and [`Pipeline._totcor_cat`](../src/mophongo/pipeline.py#L3707) cannot be copied
blindly for this purpose because their normalization goals differ.

Requested radii can exceed the stamp or ePSF-grid support. The API must not
silently clamp or extrapolate. The run should build sufficient PSF support,
apply an explicit validated wing model, or flag the total as unsupported.

### 4. Noise needs two products, not one optimistic weight map

Aperpy's current convolution path convolves a standard-deviation image by the
matching kernel and then inverts its square
([source](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/convolve_images.py#L81-L89)).
For initially independent pixels, the diagonal variance instead follows

```text
Var_out(x) = sum_j K_x(j)^2 Var_in(x-j).
```

That is still only a diagonal approximation because drizzled mosaics already
contain covariance. The clean implementation should therefore produce:

1. a clearly named diagonal variance approximation using `K^2`, with the same
   spatial-region and valid-mask logic as science convolution; and
2. empirical empty-aperture scatter measured on the matched image, stratified
   by depth and kernel/coverage class, which calibrates the catalog errors.

For masks and zero-weight holes, zero-filled convolution is not sufficient.
Use a validity halo or normalized convolution policy, propagate a coverage
fraction, and flag apertures whose kernel support intersects invalid pixels.

### 5. Region maps approximate a continuous PSF

The present map evaluates one PSF at each exposure-combination region's
centroid. It captures changes in contributing exposures, detector, roll, and
epoch, but not continuous detector-position variation within a large region.
That is already an improvement over one global empirical PSF, but it needs an
acceptance test at region interiors and extrema. If residuals exceed budget,
subdivide adaptively or interpolate within regions.

Hard kernel boundaries also require tests with point sources and extended
sources centered on and straddling a boundary. Soft blending should be added
only if the measured seam error warrants it; it complicates the definition of
the local response and covariance.

### 6. Model PSFs need empirical validation

`DrizzlePSF` should be the default provider for JWST, not the only provider.
Observed stars remain necessary to diagnose focus, jitter, registration, and
model residuals. An empirical/static provider should be an explicit fallback
for missing provenance or unsupported telescopes and a standard QA comparison
for JWST, rather than silently replacing the default.

## Local preflight blockers before making this the default

The current PSF code has a strong scientific design, but several defects and
artifact-contract gaps must be fixed before an aperture pipeline can rely on
it as a production default.

### Blocking correctness issues

1. [`PSFFactory.build`](../src/mophongo/psf_factory.py#L386) references the
   undefined names `csv_path` and `mode` while stamping provenance. The public
   method currently raises `NameError` before backend generation. This audit
   reproduced the failure directly.
2. [`PSFFactory.from_csv`](../src/mophongo/psf_factory.py#L440) does not assign
   the CSV fingerprint/date-mode provenance before writing grids, although
   later cache checks expect it.
3. [`PSFFactory.filename`](../src/mophongo/psf_factory.py#L347) writes `OS4`
   for every oversampled grid instead of the selected oversampling factor.
4. The open drizzle flux-closure discrepancy documented in `TODO.md` is of
   order one percent and changes sign between examined bands. Absolute COG
   corrections cannot claim sub-percent accuracy until this is settled with
   injected sources.

### Artifact and lookup hardening

1. [`PSFRegionMap.to_file`](../src/mophongo/psf_map.py#L1247) does not
   round-trip pixel scale, geometry tolerances, input footprints, normalization
   convention, schema version, or full build provenance. The geometry/cube
   pair is not atomic or content-hashed.
2. Cached kernel validation in
   [`Pipeline.build_kernels`](../src/mophongo/pipeline.py#L1253) checks the
   method but not a requested regularization value or the complete
   source/target-map identity.
3. [`lookup_key` and `get_psf`](../src/mophongo/psf_map.py#L570) use an
   unbounded nearest-region or plane-zero fallback. Photometric corrections
   need strict no-coverage results and explicit quality flags. A nearest mode
   should be opt-in and distance-bounded.
4. Map cubes and EE caches are mutable. In-place cube changes can leave cached
   EE stale, and first-run versus reloaded cubes can have different dtypes.

These are bounded fixes, but they belong in phase zero. A new aperture engine
should consume a stable PSF provider contract rather than depend directly on
mutable `DrizzlePSF`/GeoPandas implementation details.

## Recommended architecture

### Keep the aperture engine separate

The existing `Pipeline` is organized around high-resolution template
extraction, scene partitioning, simultaneous fitting, and residuals. Aperpy is
organized around matched images and direct measurements. Combining them would
create ambiguous configuration and output semantics.

Use these components instead:

| Component | Responsibility |
|---|---|
| `BandInput` / `ApertureRunConfig` | Immutable, validated science/variance/mask/WCS/filter manifest; no inferred glob substitutions |
| `PSFProvider` | `stamp_at`, `region_key_at`, `ee_at`, support and provenance; implementations for cached region maps, live drizzle builds, and explicit static fallback |
| `PSFMatchPlan` | Target policy, source/target overlays, local kernels, feasibility metrics, cache identity |
| `SpatialHomogenizer` | Matched science, diagonal variance approximation, validity/coverage mask, kernel-response metadata |
| `Catalog` | Detection, segmentation, deblend provenance, label-to-row identity |
| `ApertureMeasurements` | Pure fixed, masked, isophotal, Kron/AUTO, and flux-radius operations |
| `ApertureNoiseModel` | Reproducible empty-aperture placement, depth/region grouping, robust scatter curves |
| `AperturePipeline` | Stage orchestration, caching, catalog joins, diagnostics, and products |
| Compatibility writer | Aperpy-style names and metadata without making them the internal schema |

The provider should return immutable records containing at least the stamp,
pixel scale, native throughput, support, region key, provenance, and quality
flags. Aperture code should not import `drizzlepac`, STPSF, GeoPandas, or the
template-fitting pipeline.

### Runtime flow

```text
explicit band manifest
        |
        +--> load/build native DrizzlePSF region maps
        |           |
        |           +--> validate target feasibility
        |           +--> overlay source and target maps
        |           +--> build/cache unit-DC kernel maps
        |
        +--> spatially match science + diagonal variance + valid mask
        |
        +--> build detection image, segmentation, and source table
        |
        +--> forced native/matched aperture and Kron measurements
        |
        +--> empty-aperture noise calibration by depth/kernel class
        |
        +--> local target COG + reference Kron/adaptive total correction
        |
        +--> native catalog + optional Aperpy compatibility products
```

For many bands, construct the target map once, then one source-target overlay
per band. The target band is an identity operation. Record at least the target
PSF key, per-band kernel key, local EE values used, matched-coverage fraction,
and every fallback flag in the catalog or referenced artifact manifest.

### Core versus adapters

Keep the core survey-neutral. Galactic extinction, Gaia crossmatches, named
artifact masks, release-specific flags, and EAZY should consume the finished
catalog. They should not be dependencies of aperture measurement. This follows
Mophongo's existing module-boundary rule: one domain should pass flat tables or
arrays to another rather than importing its internal state.

## What should not be ported

Do not carry these upstream engineering patterns into Mophongo:

- executable Python configuration with import-time filesystem reads;
- top-level script execution and shell-based stage orchestration;
- unchecked `glob(...)[0]` input discovery and filename string replacement;
- destructive movement of input mosaics during resampling;
- borrowed WCS headers in place of geometric reprojection;
- pickled neighbor dictionaries or object arrays as pipeline state;
- broad exception handling and silent fallbacks;
- convolved standard-deviation weight maps; or
- EAZY and survey-specific release logic in the photometry core.

Do preserve, with attribution and tests:

- the detection/coadd choices where scientifically desired;
- SEP aperture, Kron, masking, and flux-radius definitions;
- the reference-band total-correction estimator;
- adaptive-aperture selection semantics;
- familiar release columns through a compatibility layer; and
- the methodology citations requested by Aperpy's README.

The reason not to treat Aperpy itself as a golden oracle is practical, not
stylistic. It has no tests or fixtures and contains observable development
defects. Examples at the audited commit include a five-value crossmatch unpack
against a four-value return
([caller](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/combine_catalogs_kronlike.py#L953-L979),
[callee](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/webb_tools.py#L913-L916))
and an automatic-star flag derived from only the final filter's Boolean array
([source](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/src/combine_catalogs_kronlike.py#L483-L497)).
A frozen differential fixture remains useful for recovering intended column
semantics, but deviations must be adjudicated rather than copied blindly.

## Implementation sequence

### Phase 0: make the PSF contract dependable

1. Fix `PSFFactory` build/provenance/oversampling defects.
2. Define immutable PSF stamp/provider results and strict coverage behavior.
3. Version and fully fingerprint region-map artifacts and kernel caches.
4. Add absolute arbitrary-radius COG and inverse-COG APIs.
5. Pin shape/throughput and finite-support tests.

**Exit gate:** a cached map is scientifically identical to its live provider,
round-trips all units/provenance, and fails explicitly outside coverage.

### Phase 1: recover Aperpy semantics in a one-region mode

1. Freeze a small input/output fixture from the audited branch.
2. Implement pure SEP fixed/masked/isophotal/Kron/flux-radius functions.
3. Implement the reference correction and adaptive aperture selector.
4. Run with one static region per band and compare columns statistically.

This phase deliberately removes spatial variation. It separates catalog
definition differences from PSF-map differences.

**Exit gate:** catalog IDs, masks, aperture geometry, Kron quantities, and
correction equations agree within documented numerical tolerances.

### Phase 2: make region maps the default

1. Extract shared PSF-map planning from the current two-band pipeline.
2. Build one source-to-target kernel map per band.
3. Match science with the existing region operator.
4. Add coverage-aware mask and `K^2` diagonal-variance paths.
5. Replace the global target COG with local absolute COG lookups.

**Exit gate:** injected point sources and extended sources recover flux and
color without position- or boundary-dependent bias above the agreed budget.

### Phase 3: scientific errors and production behavior

1. Implement empty-aperture sampling by depth/kernel class.
2. Calibrate catalog errors against matched-image blank sky.
3. Add deterministic seeds, diagnostics, resumable artifacts, and memory-aware
   region batching.
4. Validate model PSFs against real stars and resolve absolute drizzle flux
   closure.

**Exit gate:** empirical/predicted noise and stellar COG residuals satisfy the
release gates below on at least two fields with different coverage geometry.

### Phase 4: optional parity

Add only the required release flags, HST provider/fallback, extinction,
crossmatches, and EAZY adapter. These do not gate the core aperture engine.

## Validation plan and proposed release gates

Because upstream supplies no tests, validation is a first-class workstream.

### Unit and property tests

- every native stamp is finite, two-dimensional, carries a positive pixel
  scale, and declares absolute versus unit-sum semantics;
- every matching kernel has unit DC and its realized source response matches
  the target COG/core/centroid within tolerance;
- a constant image remains constant away from invalid edges;
- a delta source conserves flux in every region and when placed on both sides
  of a boundary;
- `K^2` variance agrees with white-noise Monte Carlo before drizzle covariance;
- masks, no-coverage pixels, and insufficient PSF support yield explicit flags;
- COG radius interpolation is monotone and region-key aligned;
- segment label, source ID, and table row cannot drift;
- circular, masked, isophotal, and Kron measurements pass analytic fixtures.

### Differential fixture

Freeze one small real or realistic field under the audited Aperpy commit,
including intermediate matched images and per-band tables. Compare the
one-region implementation column by column. Treat the comparison as a
semantic recovery tool, not an unquestionable truth source; document every
intentional correction, especially weights and crossmatches.

### Injected-truth validation

Use point sources, Sérsic profiles, close blends, exposure edges, invalid-pixel
holes, and sources straddling PSF-region boundaries. Vary source brightness,
size, color, subpixel phase, and local coverage. Test both reference-based
totals and raw matched-aperture colors.

### Real-field validation

- compare predicted and observed stellar PSF/COG residuals versus coordinate,
  detector, epoch, and region key;
- compare blank-aperture RMS versus aperture size, depth, and kernel class;
- compare empirical and model PSFs without silently calibrating one into the
  other;
- record runtime, peak RSS, region count, FFT count, and cache reuse.

Suggested initial release budgets are:

| Metric | Proposed gate |
|---|---:|
| Matched-response centroid residual | `< 0.02` pixel |
| Point-source aperture/EE closure | `< 0.5--1%` at catalog apertures |
| Position-dependent injected photometric bias | `< 1%` |
| Predicted versus empirical blank-aperture RMS | within `5--10%` per calibrated bin |
| Kernel DC error | numerical tolerance, with no catalog flux scale in the kernel |

The 0.5--1% photometric gate cannot be advertised until the existing
DrizzlePSF absolute-flux discrepancy is resolved. These thresholds are
starting acceptance criteria, not universal instrument requirements.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Target is narrower than a native PSF in some region | Ringing, noise amplification, biased colors | Per-region feasibility test; forward-only default; broader target or exclusion |
| Drizzle covariance is represented as diagonal IVAR | Underestimated errors | `K^2` only as a named approximation; empirical empty apertures are authoritative |
| Absolute stamp sum is normalized away | Percent-level total-flux bias | Typed normalization contract; unit tests; separate shape and throughput fields |
| Local COG exceeds finite support | Silent overcorrection | Support-aware API; larger grids/stamps or explicit flagged wing model |
| Hard region boundary or coarse centroid sampling | Spatial discontinuity | Boundary and intra-region injections; adaptive subdivision before blending |
| Missing/incorrect exposure provenance | Wrong model PSF with plausible output | Content-hashed, versioned artifacts; strict cache validation |
| SEP and Photutils Kron semantics diverge | Catalog incompatibility | SEP compatibility mode and frozen analytic/differential fixtures |
| Model PSF differs from stars | Spatial photometric residual | Mandatory empirical-star QA and explicit fallback policy |
| Region count times band count is expensive | Runtime and memory blow-up | Cache maps, merge scientifically equivalent regions, batch FFTs, measure RSS |
| HST data lack a registered model backend | Default cannot cover advertised scope | Add HST backend or require an explicit empirical provider |

## Recommendation

Proceed with a clean-slate `AperturePipeline` and make
`psf_mode="drizzle_region"` the strict default for supported JWST inputs. The
default should require exposure provenance, complete map coverage, a feasible
forward target, and validated PSF support. Provide an explicit
`psf_mode="empirical_static"` fallback for HST, nonstandard mosaics, and model
validation; do not fall back silently.

The first deliverable should be a vertical slice on a frozen small field:
explicit inputs, cached local PSFs/kernels, matched images, fixed apertures,
`K^2` diagonal variance, empirical blank apertures, local target EE, and a FITS
catalog. This exercises every risky boundary without first absorbing the
survey-specific release surface.

Do not use an Aperpy adapter as the final architecture. It can be useful as a
temporary comparison harness, but it preserves the global configuration,
weight propagation, and provenance problems the clean implementation is meant
to remove. The best long-term path is to retain Aperpy's measurement semantics
and catalog estimator while making Mophongo's spatial PSF formalism the native
center of the design.

## Licensing and attribution

Aperpy carries a
[`BSD License`](https://github.com/astrowhit/aperpy/blob/dfe8a43cd2c76a3f886e0b506cf33d2b3a2f0038/LICENSE)
with the three standard redistribution/non-endorsement conditions (the
non-endorsement clause names `aperphotpy`). Mophongo is MIT licensed. A direct
copy or derived port is permitted, but the upstream copyright notice,
conditions, and disclaimer must accompany derived code; it should not be
silently relabeled as MIT-only. A behavior-level rewrite should still cite the
methodology and record the pinned source revision used for the specification.
This paragraph is an engineering provenance recommendation, not legal advice.

## Related Mophongo design notes

- [Spatially varying PSFs](./psf_maps.md)
- [PSF behavior and `DrizzlePSF`](./psf.md)
- [PSF Shape and Throughput Convention](./PSF_SHAPE_THROUGHPUT_CONVENTION.md)
- [Photometry and Aperture Schemes](./PHOTOMETRY_APERTURES.md)
- [Flux Estimators](./FLUX_ESTIMATORS.md)
- [Wren Merge Path](./WREN_MERGE_PATH.md)
