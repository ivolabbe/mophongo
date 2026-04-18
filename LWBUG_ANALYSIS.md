# Mophongo — Codebase Reference and `lwbug` Diagnostic

Working notes on the `lwbug` branch. This file combines:

1. A full architectural map of the `mophongo` package, and
2. A root-cause diagnosis of the large residuals seen in the F1500W fit
   (`examples/run_1500.py`) compared to the clean F770W fit
   (`examples/run_770.py`).

All claims cite `file:line` against the state of `main` at the tip of the
`lwbug` branch. No behaviour has been changed — this is analysis only.

---

## Part I — Architecture

### 1. Scientific purpose

Mophongo performs template-fitting photometry across bands at different
angular resolutions. The canonical use case: a deep NIRCam F444W mosaic
(40 mas/px) provides morphology and the detection catalogue, while
lower-resolution bands (MIRI F770W/F1500W at 80 mas/px) are fitted for
source fluxes using per-source templates convolved to the target band's
PSF.

For each source `i` the pipeline solves

    sum_i f_i * T_i(x, y) ≈ I(x, y)

with `T_i` the unit-normalised, PSF-matched template and `I` the target
image. Because templates are unit-normalised before convolution
(`templates.py:1013-1014`), the recovered `f_i` is the total flux of
source `i` in the target band. This yields *matched-aperture* photometry
across all bands, which is a prerequisite for SEDs and photo-z.

The MINERVA-UDS survey (JWST NIRCam + MIRI) is the primary driver;
example scripts `examples/run_770.py` and `examples/run_1500.py` both
fit a single-pointing MIRI mosaic against a co-added NIRCam LW
detection image.

### 2. Module map (src/mophongo/)

| Module | Role | Key public objects |
| --- | --- | --- |
| `pipeline.py` (1030 LOC) | Top-level orchestrator. Sequences template extraction, kernel dispatch, resolution handling, scene generation, solving, aperture photometry, catalogue assembly. | `Pipeline`, `Pipeline.run` |
| `templates.py` (1186 LOC) | Per-source cutouts, convolution, down/upsampling, weight pruning, shift application. | `Template` (extends `Cutout2D`), `Templates`, `extend_with_psf_wings` |
| `fit.py` (1655 LOC) | Sparse normal-matrix assembly, scene-based and LSQR solvers, flux-error estimation. | `FitConfig`, `SparseFitter`, `build_normal_tree`, `solve_scene`, `_solve_scenes_with_shifts` |
| `scene.py` (1042 LOC) | Connected-component grouping of overlapping templates; per-scene joint flux+astrometry solve. | `Scene`, `generate_scenes`, `merge_small_scenes` |
| `scene_fitter.py` | Stateless CG solver consuming pre-assembled A/b (+ shift blocks). | `SceneFitter`, `build_normal` |
| `psf.py` (1421 LOC) | `PSF` dataclass with analytic constructors; JWST/HST `DrizzlePSF` that projects model PSFs to mosaic WCS. | `PSF`, `DrizzlePSF` |
| `psf_map.py` | Spatial lookup of PSFs/kernels by sky position via STRtree over detector footprints. | `PSFRegionMap` |
| `jwst_psf.py` | STPSF grid extension helpers. | `make_extended_grid`, `blend_psf` |
| `astrometry.py` | Chebyshev basis, polynomial predictor, stored-shift application. | `cheb_basis`, `AstroCorrect`, `build_poly_predictor` |
| `astro_fit.py` | Legacy global joint flux+astrometry (now superseded by per-scene path). | `GlobalAstroFitter` |
| `catalog.py` | photutils-based source detection + background/ivar helpers. Not invoked when an external catalogue is provided. | `Catalog`, `get_bg_and_ivar` |
| `deblender.py`, `photutils_deblend.py` | Experimental/standalone deblenders. Not wired into `Pipeline.run`. | — |
| `sim_data.py` | Synthetic mosaic generator for tests. | `make_mosaic_dataset` |
| `utils.py` (2042 LOC) | `matching_kernel`, `bin_factor_from_wcs`, `downsample_psf`, analytic profiles, WCS helpers, `AlignedCutout`. | — |

### 3. Core abstractions

**`Template` / `Templates`** (`templates.py:368`, `templates.py:683`).
Each `Template` extends `astropy.nddata.Cutout2D` in `"partial"` mode.
Two slice pairs are carried on every template:

- `slices_original` — location of the template footprint inside the full
  image array.
- `slices_cutout` — corresponding region inside `Template.data`.

Every normal-matrix operation uses the idiom
`t.data[t.slices_cutout]` indexed against `image[t.slices_original]`
(e.g. `fit.py:769`). This is the mechanism that makes edge templates
correct and sparse assembly unambiguous. After
`convolve_cutout` (`templates.py:481`) the template grows by
`(kernel_size - 1)` in each dimension and the slices are recomputed on
the new `Template` so the mapping to the parent image remains valid.
Shifts from astrometry iterations are stored in `t.to_shift` and
applied in-place by `Templates.apply_template_shifts` using cubic
spline interpolation (`templates.py:800-820`).

**`FitConfig` / `SparseFitter`** (`fit.py:49`, `fit.py:746`). Important
`FitConfig` fields:

| Field | Default | Role |
| --- | --- | --- |
| `solve_method` | `"scene"` | `"scene"` / `"all"` / `"lo"` (LSQR) |
| `positivity` | `True` | Clamp negative fluxes |
| `reg` | `0.0` | Ridge (auto if 0) |
| `fit_astrometry_niter` | `2` | Outer astrometry iterations |
| `fit_astrometry_joint` | `True` | Joint flux+shift solve per scene |
| `astrom_model` | `"gp"` | `"polynomial"` or `"gp"` |
| `astrom_kwargs` | `{"poly": {"order": 0}}` | Order of the shift basis |
| `scene_coupling_thresh` | `1e-3` | Off-diagonal threshold for scene split |
| `scene_minimum_bright` | `5` | Bright sources needed per scene for astrometry |
| `snr_thresh_astrom` | `15.0` | SNR cut defining "bright" |
| `fft_fast` | `False` | If non-zero, crop kernel per source by encircled energy |
| `fit_covariances` | `False` | Full off-diagonal errors via Hutchinson |
| `aperture_diam` | `None` | Diameter for the aperture-photometry pass |

`SparseFitter.build_normal_tree` uses a Shapely `STRtree` over template
bounding boxes to assemble the symmetric normal matrix
`A^T W A` and RHS `A^T W b`. Diagonals are `sum(t_i^2 * w)`, off-diagonals
`sum(t_i * t_j * w)` over the footprint intersection. The matrix is
sparse because most template pairs do not overlap.

The default solve path is **`solve_scene`** (`fit.py:1074`): whiten the
normal matrix as `A_w = D^{-1} A D^{-1}` with `D = sqrt(diag(A + reg I))`,
split into scenes by coupling-based connected components
(`build_scene_tree_from_normal`), merge small scenes
(`merge_small_scenes`), then solve each scene independently with CG
(MINRES fallback). If `fit_astrometry_joint=True`, each scene assembles
the extended block system `[A_w AB_w; AB_w^T BB_reg]` and solves for
both fluxes and Chebyshev shift coefficients.

**`GlobalAstroFitter`** (`astro_fit.py:21`) is the legacy single-scene
variant that augments the global normal system with astrometric
columns. It is superseded by the per-scene path but still present.

**`Scene` / `SceneFitter`** (`scene.py:567`, `scene_fitter.py:114`).
A `Scene` is a dataclass with a list of connected `Template`s, cached
normal blocks (`A`, `b`), a bounding box, and a `SceneFitter`. Scene
formation in `generate_scenes` (`scene.py:474`) builds the full normal
matrix once, splits into connected components by coupling strength,
then merges components with fewer than `scene_minimum_bright` bright
sources into their nearest neighbours via an STRtree-accelerated
union-find. Inside `Scene.solve`, `make_scene_basis` constructs the
Chebyshev basis evaluated at each bright template position (scaled to
`[-1, 1]` over the scene extent); `assemble_scene_system_AB` forms the
coupling block `AB` and Hessian block `BB`. The whitened block system
is solved jointly; `AstroCorrect.build_poly_predictor` evaluates the
shifts at each template position and stores them on `t.to_shift`. If
`apply_shifts=True`, templates are shifted in-place and the cached `A`
and `b` are cleared so the next iteration rebuilds them on the shifted
pixels.

**`PSF` / `PSFRegionMap` / `DrizzlePSF`** (`psf.py`, `psf_map.py`). The
PSF is spatially variable because each pixel in the mosaic is drizzled
from a different subset of detector exposures. `PSFRegionMap.regions`
is a GeoDataFrame with a `psf_key` column; `psfs` is an
`ndarray(N_regions, H, W)` holding one stamp per region. `get_psf(ra,
dec)` issues an STRtree query and returns the stamp for the containing
(or nearest) region. This is the mechanism called from
`Templates.convolve_templates` (`templates.py:1060-1068`) to fetch a
per-source matching kernel. `DrizzlePSF` sits upstream — it reads the
mosaic and a CSV of rate-file WCS headers, builds footprint polygons,
and drizzles model PSF stamps onto the mosaic grid to populate
`PSFRegionMap.psfs`.

**`utils.matching_kernel`** (`utils.py:383`). Computes `K` such that
`PSF_hi * K ≈ PSF_lo`:

1. If `pixel_ratio != 1.0`, `scipy.ndimage.zoom` the low-resolution PSF
   onto the high-resolution grid.
2. Zero-pad both PSFs to a common shape.
3. Call `photutils.psf.matching.create_matching_kernel` with a
   `SplitCosineBellWindow(alpha=0.4, beta=0.1)` (hard-coded). The window
   suppresses high-frequency deconvolution noise where the high-res
   PSF's FFT has little power.
4. If `recenter=True`: estimate the centroid via `centroid_com`
   (initial guess), refine with `centroid_quadratic(kernel, xpeak,
   ypeak, fit_boxsize=7)`, and shift the kernel to that centroid.

The kernel is *not* normalised to unit sum after windowing/recentering.

### 4. End-to-end data flow — MINERVA-UDS example

Both `run_770.py` and `run_1500.py` follow the same structure:

1. **PSF and kernel map construction** (cached on disk). `DrizzlePSF`
   reads the mosaic header and rate-file WCS CSV. `PSFRegionMap.from_footprints(...)
   .overlay_with(driz_footprint)` builds region polygons clipped to the
   mosaic. `prm_kern = prm_444.overlay_with(prm_miri)` gives regions
   where both PSFs are defined. `dpsf.get_psf_radec(pos, size=psf_size)`
   drizzles one stamp per region centroid. `utils.matching_kernel(psf_444,
   psf_miri, recenter=True, pixel_ratio=2)` produces one kernel per
   region; these are stored on `prm_kern.psfs` and written to geojson.
2. **Inputs**: F444W detection image (as template source), MIRI sci+wht,
   LW segmap, MINERVA super-catalogue (optionally trimmed to a test
   circle via `r_trial`). `get_bg_and_ivar(sci_miri, wht_miri, 64.0)`
   returns the background-subtracted image and an ivar map.
3. **`Pipeline.__init__`** (`pipeline.py:132`) stores all arrays;
   `psfs=[None, prm_miri]`, `kernels=[None, prm_kern]`.
4. **`Pipeline.run`** (`pipeline.py:517`):
   - `Templates.extract_templates` (4a): unit-normalised, segmap-masked
     cutouts at each catalogue position.
   - **Multi-resolution decision** (`pipeline.py:598`):
     `k = bin_factor_from_wcs(wcs[0], wcs[i])`. For 40 mas vs 80 mas,
     `k=2`. With `config.multi_resolution_method="upsample"` (default),
     `block_replicate(images[i], k, conserve_sum=True)` expands the MIRI
     image to 40 mas and `wcs[i] = wcs[0]`. The `"downsample"` branch
     downsamples templates and kernels instead.
   - `tmpls_lo.prune_outside_weight(weights_i)`: drop templates with no
     weight overlap.
   - `tmpls_lo.convolve_templates(kernel, inplace=False)`
     (`pipeline.py:629`). Per template, if `kernel` is a
     `PSFRegionMap`, the template WCS converts the template centre to
     `(ra, dec)` and `kernel.get_psf(ra, dec)` returns the matching
     kernel stamp; then `Template.convolve_cutout(kern)` produces the
     enlarged convolved template.
   - **Scene loop** (`pipeline.py:637-697`): `generate_scenes` builds
     and splits the normal matrix. The astrometry iteration loop runs
     `for j in range(niter_scene)`: each scene calls `scn.set_band(...)`
     then `scn.solve(config=config, apply_shifts=True)`. Templates are
     shifted in-place between iterations; `A/b` are cleared so the next
     pass rebuilds on shifted pixels.
   - Residual assembly sums each scene's model image and subtracts
     from the input image.
   - `_update_catalog_with_fluxes` and `_add_aperture_photometry`
     produce the final output table.
5. **Outputs**: `residual.fits`, `fit_table.fits`, per-scene PNG
   diagnostics (`Scene.plot`).

### 5. Conventions

- No `astropy.units` on array data; pixel scales are extracted as
  arcseconds via `proj_plane_pixel_scales(wcs) * 3600`.
- WCS origin 0 everywhere (`wcs_pix2world(..., 0)`).
- Segmentation-map convention: `0 = background`, positive integers =
  labels. Template extraction reads the label at
  `segm.data[int(y), int(x)]`; label `0` or out-of-bounds skips the
  source silently.
- Weight convention: `w ∝ 1/variance`, used directly in the normal
  matrix.
- Flux errors: default is `1 / sqrt(diag(A_w))` (ignores off-diagonal
  covariance). `FitConfig.fit_covariances=True` enables Hutchinson
  stochastic covariance estimation (`fit.py:1233`).

### 6. Inputs, assumptions, and implicit contracts

Hard preconditions:

- Integer pixel-scale ratio between detection and target images.
  `bin_factor_from_wcs` raises if `|ratio - round(ratio)| > 0.001`.
- `Template.downsample` requires that the cutout origin and size are
  multiples of `k` (`templates.py:632`); the `AlignedCutout` helper
  exists to enforce this but `Template` itself still uses plain
  `Cutout2D`.

Silent failure modes:

- **PSF stamp size vs kernel size vs template size**. The template
  convolution in `Template.convolve_cutout` uses `fftconvolve(...,
  mode="full")`, so the *template* does not truncate the convolution.
  However, the *kernel itself* is truncated at the PSF stamp boundary
  upstream, in the user script
  (`dpsf.get_psf_radec(pos, size=psf_size)`). A too-small `psf_size`
  clips the PSF wings *before* the kernel is built, which then
  propagates as a normalisation / wing deficit at every source.
- **Centroid stability under broad PSFs**. `centroid_quadratic` in
  `matching_kernel` uses a fixed `fit_boxsize=7` (`utils.py:468`). A
  broad PSF kernel (F1500W has ~12.5 high-res px FWHM) is poorly fit
  by a 7×7 quadratic; the recovered centroid can drift by 0.5-1 px,
  translating to a uniform template offset at every source.
- **Catalogue position vs segmap label**. If the catalogue `(x, y)`
  falls on `segm.data == 0`, the source is dropped silently
  (`templates.py:989`). Only a summary count is logged at
  `pipeline.py:581`.
- **Boundary weight coverage**. `prune_outside_weight` keeps templates
  with partial coverage; no flag is set, so low-SNR edge sources look
  like normal ones.

### 7. Checklist gaps directly relevant to residuals

From `CHECKLIST.md:134-140` (reproduced for reference; verbatim status):

- `[ ]` Automated determination of optimal convolution kernel.
- `[ ]` Deduplicate templates by weighted-overlap cosine similarity.
- `[ ]` Strong residuals:
  - `[ ]` Handle saturated stars in F444W (catalog pre-pass detection).
  - `[ ]` Fit both F444W and MIRI as PSF for centroid, mask the core.
- `[ ]` Wavelength-dependent morphology (bluer band, residual-driven
  PSF core).

These are the pre-existing open items most relevant to the current
bug.

---

## Part II — `lwbug` diagnostic: why F1500W residuals are worse than F770W

### Observable

- `examples/uds_770/uds_770_v0.1_scene_1.png` — residual looks clean;
  `Max shift: 0.20 pix`.
- `examples/uds_1500/uds_1500_scene_1.png` — source-centred dipoles and
  wing residuals at every bright source; `Max shift: 1.32 pix`.

Both runs use the same config:
`FitConfig(fit_astrometry_niter=2, fit_astrometry_joint=True,
scene_minimum_bright=10, aperture_diam=0.5)`, the same detection image,
and the same MINERVA catalogue. The only real difference in the user
script is the PSF stamp size: `psf_size = 2.0"` (770) vs `size = 8.0"`
(1500) — a concession to the broader F1500W PSF.

At 80 mas / 40 mas, `pixel_ratio = 2` in both cases. Both images go
through the `"upsample"` multi-resolution path identically.

### Ruled out

| Hypothesis | Reason ruled out |
| --- | --- |
| Template stamp truncating the convolution | `Template.convolve_cutout` uses `fftconvolve(..., mode="full")` (`templates.py:481`); output is `n_tmpl + n_kern - 1`. No truncation. |
| `ee_rlim` / `ee_fraction` kernel cropping | `Templates.prepare_kernel_info` (`templates.py:856`) is never called from `Pipeline.run`. All templates have `ee_rlim = 0`, so the cropping branch at `templates.py:1070` is dead code for these runs. |
| `extend_templates` parameter | Accepted on `Pipeline.__init__` (`pipeline.py:171`) but never referenced inside `Pipeline.run`. Dead. |
| Pixel-ratio / resolution mismatch | Both bands `k=2` via `bin_factor_from_wcs`; both take the same upsample branch. |

### Primary root cause — matching-kernel recentring failure

`utils.matching_kernel` at `utils.py:465-482`:

```python
xcom, ycom = centroid_com(kernel)
xcen, ycen = centroid_quadratic(kernel, xpeak=xcom, ypeak=ycom, fit_boxsize=7)
```

`fit_boxsize=7` is hard-coded. The kernel built for F770W uses a 2"
PSF stamp → 50×50 px at 40 mas; the F770W PSF FWHM is ~6-7 high-res
pixels, well-sampled by a 7×7 quadratic fit. The F1500W kernel uses an
8" PSF stamp → 200×200 px; the F1500W PSF FWHM is ~12-13 high-res
pixels, with broad, extended wings. A 7×7 quadratic fit around the
COM-estimated peak of such a broad, potentially ringing kernel is
ill-conditioned: the recovered centroid can drift ~0.5-1 pixel from
the true centre.

Once the kernel is recentred to the wrong location, every convolved
template is uniformly displaced by that same sub-pixel offset relative
to where the flux actually sits in the MIRI image. The joint
astrometry solver in `Scene.solve` then absorbs this as a ~1.3 pixel
apparent shift. The displayed "Max shift: 1.32 pix" in the Model
panel is exactly this kernel-recentring residual, not a real WCS
offset.

Dipole residuals at every bright source are the signature: the model
is shifted by ~1 px relative to the truth, and an order-0 polynomial
can remove only the mean offset across the scene, leaving a
position-dependent sub-pixel residual everywhere.

### Strong secondary contributor — scene collapse

`examples/uds_1500/` contains a single scene plot
(`uds_1500_scene_1.png`); `examples/uds_770/` contains ~164 scene
plots. With `scene_minimum_bright=10` and `snr_thresh_astrom=15`, most
F1500W sources fail the SNR cut, and `merge_small_scenes`
(`scene.py`) merges scenes until each has ≥ 10 bright sources — the
field is small enough (trial radius 1′) that this collapses to a
single mega-scene covering all sources.

A single scene with `astrom_kwargs["poly"]["order"]=0` fits one global
`(dx, dy)` for the whole field. That can only absorb the mean
template-kernel displacement. Any spatial variation in the
kernel-centring error (e.g. because different PSF regions have slightly
different kernel centroids) becomes a per-source dipole in the
residual.

The 770 case does not hit this because many more sources are bright
enough to form independent, well-constrained scenes.

### Contributing factor — kernel normalisation

`matching_kernel` does not re-normalise to unit sum after
`create_matching_kernel` or after the sub-pixel shift
(`utils.py:453-482`). `SplitCosineBellWindow(alpha=0.4, beta=0.1)` is
hard-coded. For a broad PSF with significant power at the window
cut-off, the output kernel sum can deviate from unity at the few-percent
level. This alone produces a uniform flux bias, not dipoles, but it
compounds with the centring error and creates visible wing residuals
after the joint astrometry tries to compensate.

### Evidence summary

| File:line | Fact |
| --- | --- |
| `utils.py:465-482` | `fit_boxsize=7` hard-coded in `matching_kernel` recentring. |
| `utils.py:453-455` | `SplitCosineBellWindow(alpha=0.4, beta=0.1)` hard-coded; no adaptation to PSF width. |
| `templates.py:481` | `fftconvolve(..., mode="full")` — template convolution does not truncate. |
| `templates.py:1070-1071` | `ee_rlim` kernel-cropping branch is unreachable in these runs. |
| `pipeline.py:171` | `extend_templates` parameter accepted but never used in `run`. |
| `fit.py:70-71` | `astrom_kwargs={"poly": {"order": 0}}` — constant per-scene shift only. |
| `run_770.py:17` / `run_1500.py:48` | `psf_size=2.0"` vs `size=8.0"`. |

### Diagnostic experiments to run before fixing

1. Immediately after the `matching_kernel(...)` call in
   `run_1500.py:74-76`, log `kernel.sum()` and
   `photutils.centroids.centroid_com(kernel)` for every region. Compare
   against the array centre `(N/2, N/2)`. If the centroid deviates by
   more than ~0.3 px, this is confirmation.
2. Regenerate the 1500 kernel geojson with `recenter=False` and rerun
   the pipeline. If residuals become clean (dipoles vanish, "Max shift"
   drops), the recentring is the bug.
3. Add `kernel /= kernel.sum()` immediately after `matching_kernel`
   returns (in the user script, not the library) and rerun. This
   tests whether normalisation alone recovers the flux scale.
4. Rerun with `FitConfig(fit_astrometry_niter=0)`. If residuals *improve*,
   the astrometry solver is absorbing the kernel error as a bogus shift
   and making things worse.
5. Rerun with `scene_minimum_bright=3` (or lower). If scenes split and
   per-scene shifts differ, the scene-collapse story is confirmed.
6. Rerun with `astrom_kwargs={"poly": {"order": 1}}`. A linear shift
   field within the single mega-scene should absorb position-dependent
   centring errors and improve residuals.

### Suggested fix (not implemented)

In priority order:

1. `utils.py:468` — make `fit_boxsize` adaptive. A reasonable rule:
   `fit_boxsize = max(7, 2 * int(round(fwhm_px)) | 1)` where `fwhm_px`
   is estimated from a simple moments analysis of the kernel peak.
   Alternatively, skip `centroid_quadratic` entirely when the kernel
   half-light radius is > `fit_boxsize/2` and use `centroid_com` alone.
2. `utils.py:482` (after the recentring shift) — add
   `kernel /= kernel.sum()` so the convolved template carries unit
   integrated flux regardless of window roll-off.
3. `templates.py:1068` — log a warning when `|kern.sum() - 1| > 1e-3`
   so future mis-normalised kernels are caught.
4. `pipeline.py:171` — either wire `extend_templates` through to the
   template-extraction step (so the template is padded to accommodate
   the post-convolution footprint) or remove it as dead API.

These are all small, local changes. The architectural shape of the
pipeline is correct; the bug is a hard-coded constant (`fit_boxsize=7`)
plus an absent normalisation, both in `utils.matching_kernel`.

---

## Files essential to the analysis

- `src/mophongo/utils.py` — `matching_kernel`, `bin_factor_from_wcs`,
  `downsample_psf`.
- `src/mophongo/templates.py` — `Template`, `Templates`, convolution,
  downsampling, slice bookkeeping.
- `src/mophongo/pipeline.py` — orchestration, resolution handling,
  astrometry loop.
- `src/mophongo/fit.py` — `FitConfig`, sparse normal matrix, scene-based
  solver.
- `src/mophongo/scene.py` + `scene_fitter.py` — scene formation,
  merging, joint flux+shift solve.
- `src/mophongo/psf_map.py`, `psf.py` — per-position kernel lookup and
  drizzled PSFs.
- `src/mophongo/astrometry.py`, `astro_fit.py` — Chebyshev astrometric
  basis and legacy global fitter.
- `examples/run_770.py`, `examples/run_1500.py` — production usage.
- `CHECKLIST.md` — design intent, open gaps (esp. "strong residuals"
  subsection).
