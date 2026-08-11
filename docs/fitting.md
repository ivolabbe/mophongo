# Fitting and astrometric shifts

Once per-source templates have been extracted and PSF-matched to a
low-resolution band (see {doc}`templates` and {doc}`psf`), photometry reduces
to a weighted linear least-squares problem: the band image is modeled as a sum
of templates, each scaled by one free amplitude. This page documents the three
modules that own that solve: `mophongo.fit` (normal-matrix assembly,
{class}`~mophongo.fit.FitConfig`), `mophongo.scene` (partitioning the field
into independent scenes, per-scene astrometric shifts), and
`mophongo.scene_fitter` (the stateless linear solver). In normal use
{meth}`mophongo.pipeline.Pipeline.run` drives all of this; the classes below
matter when you tune {class}`~mophongo.fit.FitConfig` or inspect fits
scene by scene.

## The linear model and normal equations

For templates $T_i$ and image $I$ with per-pixel inverse-variance weights $w$
(the weight convention throughout mophongo), the fitted amplitudes
$\alpha$ minimize $\sum_\mathrm{pix} w\,(I - \sum_i \alpha_i T_i)^2$. The
normal equations are $A\alpha = b$ with

$$
A_{ij} = \sum_\mathrm{pix} T_i\, w\, T_j, \qquad
b_i = \sum_\mathrm{pix} T_i\, w\, I .
$$

$A$ is sparse: only templates with overlapping footprints couple. Overlaps are
found with an STRtree spatial index over template bounding boxes, and each
$A_{ij}$ is accumulated over the pixel intersection of the two cutouts.
$A_{ij}$ is the integral of $T_i T_j$, so off-diagonal terms measure how much
flux from one source leaks into the footprint of another — the same quantity
later reused to partition the field into scenes.

### SparseFitter

{class}`mophongo.fit.SparseFitter` assembles the normal equations and provides
model/residual images and quick covariance-free estimators. Flux solving
itself lives in {class}`mophongo.scene_fitter.SceneFitter`.

Constructor parameters:

`templates` (`list[Template]`)
: Templates on the fit grid, one per source. Each template gets
  `is_flux = True` and a column index `col_idx` assigned.

`image` (`np.ndarray`)
: The low-resolution science image being fit.

`weights` (`np.ndarray | None`, default `None`)
: Inverse-variance weight map, same shape as `image`. `None` uses unit
  weights.

`config` ({class}`~mophongo.fit.FitConfig` `| None`, default `None`)
: Fit configuration; `None` uses a default `FitConfig()`.

Public methods:

`build_normal()`
: Dispatches on `config.normal`; only `"tree"` (the STRtree builder) is
  implemented, and any other value raises `ValueError`.

`build_normal_tree()`
: Builds the sparse normal matrix and right-hand side, cached on the `ata`
  and `atb` properties (which build lazily on first access). Templates whose
  weighted norm falls below `1e-6` times the median norm are counted and
  logged but kept.

`add_flux_priors(idx, mu, sigma, *, floor=1e-12)`
: Adds Gaussian flux priors $(x_i-\mu_i)^2/\sigma_i^2$ for the templates in
  integer index array `idx`: each selected diagonal gains a precision
  $\lambda_i = 1/\sigma_i^2$ and the right-hand side gains
  $\lambda_i \mu_i$. `mu` and `sigma` may be scalars or arrays; `floor`
  bounds `sigma` away from zero.

`model_image()` / `residual()`
: Full-frame model $\sum_i \alpha_i T_i$ from the stored `solution`, and
  `image - model`. Model pixels where the weight is non-positive or NaN are
  zeroed.

`quick_flux(templates=None)`
: Per-template estimate $\sum I\,T_i / \sum T_i^2$ over each template's own
  footprint, ignoring neighbors (delegates to
  {func}`mophongo.templates.Templates.quick_flux`).

`predicted_errors(templates=None)`
: Per-template $1/\sqrt{\sum w\,T_i^2}$, ignoring covariance (see
  [Error estimates](#error-estimates-err-vs-err_pred)).

`flux_and_rms(templates=None)`
: Returns `(flux, rms)`; uses fluxes already stored on the templates when
  present, otherwise `quick_flux`, paired with `predicted_errors`.

## Error estimates: `err` vs `err_pred`

Two uncertainty estimates are produced for every source, and both appear in
the output catalog (`err_<i>` and `err_pred_<i>`; see {doc}`outputs`):

- **`err`** — the solver error, $\sqrt{[\hat A^{-1}]_{ii}}$ computed from
  the (whitened) normal matrix of the source's scene. It includes the
  covariance with overlapping neighbors: blended sources get larger `err`
  than isolated ones of the same brightness. When astrometric shift
  parameters are fit jointly, the shift block is marginalized out first via
  the Schur complement $S = A - A_B B^{-1} A_B^\top$, so `err` also accounts
  for the flux–shift covariance. For scenes up to 500 templates the inverse
  diagonal is obtained by dense inversion; larger scenes use one sparse LU
  factorization and per-column back-solves.
- **`err_pred`** — the isolated-source prediction
  $1/\sqrt{\sum_\mathrm{pix} w\,T_i^2}$, which ignores template covariance
  entirely. It is a depth map in disguise: it depends only on the weight map
  and template shape, not on the fit, and is the floor `err` approaches for
  isolated sources. {func}`mophongo.templates.Templates.predicted_errors`
  stores it on `tmpl.err_pred` and never overwrites `tmpl.err`.

Comparing the two flags blending: `err / err_pred` substantially above 1
means neighbors inflate the uncertainty.

## Flags

The direct sparse solver always reports a zero exit flag (there is no
iterative convergence to fail), returned as the dict `{"cg_info": 0}`;
{meth}`Scene.solve <mophongo.scene.Scene.solve>` passes it through as `info`. Per-template quality flags live on
`Template.flag` as a bitmask (defined in {class}`mophongo.templates.Template`):
`FLAG_VALID = 0x01`, `FLAG_CONVOLVED = 0x02`, `FLAG_SUM_ZERO = 0x04`,
`FLAG_HAS_NAN = 0x08`, `FLAG_OUTSIDE_WEIGHT = 0x10`, `FLAG_SHIFTED = 0x20`,
`FLAG_DEBLENDED = 0x40`, `FLAG_SATURATED = 0x80`. During fitting,
`predicted_errors` sets `FLAG_SUM_ZERO` (and returns an infinite `err_pred`)
for templates with zero weighted support. The bitmask is written to the
`flag` column of the fit table.

## Scenes: partitioning the fit

Solving one global system for every source in a mosaic is unnecessary: the
normal matrix decomposes into connected components of mutually overlapping
templates. Mophongo cuts the field into **scenes** — groups of templates
whose cross-coupling is strong enough to matter — and solves each scene
independently with a direct method. Besides speed, scenes are the natural
unit for local astrometric corrections: each scene fits its own smooth shift
field.

### Coupling threshold

{func}`mophongo.scene.build_scene_tree_from_normal` builds the partition from
the normal matrix alone. Using diagonal-only amplitudes
$\alpha_i = b_i/A_{ii}$, the predicted cross-leakage between sources $i$ and
$j$ is scored as

$$
s_{ij} = \max\!\left(
  \frac{|A_{ij}\alpha_j|}{A_{ii}|\alpha_i|},\;
  \frac{|A_{ij}\alpha_i|}{A_{jj}|\alpha_j|}
\right),
$$

i.e. the neighbor's flux inside source $i$'s footprint relative to $i$'s own.
An edge connects $i$ and $j$ when $s_{ij}$ exceeds the coupling threshold;
scenes are the connected components. A lower threshold merges more sources
into fewer, larger scenes.

Parameters of `build_scene_tree_from_normal(ATA, ATb, *, coupling_thresh=0.01,
max_size=None, return_0_based=False)`:

`ATA` (sparse `(n, n)`), `ATb` (`(n,)` array)
: Unwhitened normal matrix and right-hand side.

`coupling_thresh` (`float`, default `0.01`)
: Leakage score above which two templates are joined into one scene.

`max_size` (`int | None`, default `None`)
: Soft cap on templates per scene. An oversized component is split by
  bisecting over that component's own edge scores for the smallest *local*
  threshold whose pieces all fit, so strong couplings elsewhere in the field
  are never cut on behalf of one crowded region. The accepted local leakage
  is logged. `None` disables the cap.

`return_0_based` (`bool`, default `False`)
: Return labels `0..K-1` instead of the default `1..K`.

Returns `(labels, nscene)`.

### Merging small scenes

The astrometric shift fit needs several bright anchors per scene, so scenes
with too few bright members are merged into their nearest neighbor by
{func}`mophongo.scene.merge_small_scenes`:

`labels` (`np.ndarray`)
: Scene labels per template.

`templates` (`list[Template]`)
: Templates, for centroid positions.

`bright_mask` (`np.ndarray` of bool)
: Which templates count as bright anchors.

`order` (`int`, default `1`)
: Present in the signature but unused by the current implementation.

`minimum_bright` (`int`, default `10`)
: Scenes with fewer bright members than this are merged.

`max_merge_radius` (`float`, default `np.inf`)
: Maximum centroid distance in pixels over which an underfilled scene may
  merge, keeping merges local. Underfilled scenes with no neighbor within
  the radius stay as they are.

`max_iter` (`int`, default `64`)
: Maximum merge rounds.

Returns 1-based labels and the scene count.

### generate_scenes

{func}`mophongo.scene.generate_scenes` is the one-call entry point used by the
pipeline: it builds the normal system, partitions it, merges small scenes, and
returns {class}`~mophongo.scene.Scene` objects carrying their sub-blocks of
`ATA`/`ATb`.

```python
from mophongo.fit import FitConfig
from mophongo.scene import generate_scenes

cfg = FitConfig(fit_astrometry_niter=3)
scenes, labels = generate_scenes(
    templates, image, weight,
    coupling_thresh=cfg.scene_coupling_thresh,
    max_size=cfg.scene_max_size,
    minimum_bright=cfg.scene_minimum_bright,
)
for scn in scenes:
    scn.set_band(image, weight, config=cfg)
    flux, err, shifts, info = scn.solve(config=cfg, apply_shifts=True)
```

Parameters:

`templates` (`Sequence[Template]`), `image` (`np.ndarray`), `weight`
(`np.ndarray | None`, default `None`)
: Fit inputs; `None` weight means unit weights.

`coupling_thresh` (`float`, default `0.01`)
: Passed to `build_scene_tree_from_normal`. The pipeline passes
  `FitConfig.scene_coupling_thresh` (default `1e-3`).

`max_size` (`int | None`, default `None`)
: Soft per-scene template cap (pipeline passes `FitConfig.scene_max_size`,
  default 800).

`snr_thresh_astrom` (`float`, default `7.0`)
: Bright-anchor cut on the SNR proxy $b_i/\sqrt{A_{ii}}$.

`isolation_thresh` (`float`, default `0.0`)
: If positive, a template only counts as a bright anchor when its own flux
  dominance within its footprint (self flux over self plus neighbor flux,
  from the full-field normal matrix) meets this fraction.

`minimum_bright` (`int | None`, default `None`)
: Minimum bright anchors per scene, forwarded to `merge_small_scenes`. Pass
  an integer (the pipeline passes `FitConfig.scene_minimum_bright`): the
  `None` default is forwarded unchanged and fails inside
  `merge_small_scenes`.

`max_merge_radius` (`float`, default `np.inf`)
: Merge radius in pixels, forwarded to `merge_small_scenes`.

`exclude_stars` (`bool`, default `False`)
: Remove templates with `is_star` set from the bright-anchor mask.

`isolate_saturated` (`bool`, default `True`)
: Move saturated/repaired templates into singleton scenes. Their PSF wings
  extend far beyond their segment and would corrupt the flux solution of
  every neighbor caught in the same coupling graph.

### The Scene dataclass

{class}`mophongo.scene.Scene` is a container for one scene's templates and
fit state. Fields:

`id` (`int`)
: 1-based scene label.

`templates` (`list[Template]`)
: Scene members, in scene-local order.

`fitter` ({class}`~mophongo.scene_fitter.SceneFitter`)
: Stateless solver instance.

`bbox` (`tuple[int, int, int, int] | None`, default `None`)
: Union bounding box `(y0, y1, x0, x1)` of the member templates.

`image`, `weights` (`np.ndarray | None`, default `None`)
: Full-frame band image and inverse-variance weights (sliced per template).

`config` ({class}`~mophongo.fit.FitConfig` `| None`, default `None`)
: Per-scene fit configuration.

`shift_basis` (`list | None`, default `None`)
: `[basis, (x0, y0), (Sx, Sy)]` stored by `solve()` for shift evaluation.

`flux`, `err`, `shifts` (`np.ndarray | None`, default `None`)
: Solution arrays (fluxes and errors are also written onto the templates).

`is_bright` (`np.ndarray | None`, default `None`)
: Per-template bright-anchor mask.

`solution` (`SimpleNamespace | None`, default `None`)
: Full solver result (`flux`, `err`, `shifts`, `info`).

`A` (`csr_matrix | None`), `b` (`np.ndarray | None`), `tree`
(`STRtree | None`), all default `None`
: Scene-local normal block, right-hand side, and spatial index; rebuilt from
  the current band by `solve()` when absent.

Methods:

`set_band(image, weight=None, psf=None, config=None)`
: Cache band data on the scene. `psf` is accepted but currently unused.

`solve(*, config=None, apply_shifts=True, **kwargs)`
: Solve the scene and return `(flux, err, shifts, info)`. Rebuilds `A`/`b`
  from the current band if needed, recomputes the bright mask (SNR proxy
  above `config.snr_thresh_astrom`, isolation above
  `config.astrom_isolation_thresh`, optional star exclusion), then either
  solves flux-only (when `config.fit_astrometry_joint` is false or
  `fit_astrometry_niter <= 0`) or the joint flux+shift system. Results are
  stored on the scene and on each template (`tmpl.flux`, `tmpl.err`,
  `tmpl.is_bright`). With `apply_shifts=True` the fitted shift field is
  evaluated at each template position and applied (see below), and `A`/`b`
  are cleared so the next pass rebuilds them against the shifted templates.

`shift_at(x, y)`
: Evaluate the applied shift at positions `(x, y)` by nearest-template
  lookup; returns `(dx, dy)` arrays (zeros when no shift fit exists).

`model_image()` / `residual()`
: Scene model and image-minus-model over the scene bounding box; residual
  pixels with non-positive or NaN weight are zeroed.

`plot(tmpl_image, seg_image, display_sig=3.0, display_sig_by_title=None,
residual_image=None, ax=None, **imshow_kwargs)`
: Six-panel diagnostic (template, image, model, segmap, residual, color
  composite) with the fitted shift field drawn as arrows on the model panel.
  `tmpl_image` and `seg_image` are the full-frame high-resolution image and
  segmentation map; `display_sig` scales the grayscale stretch (per-panel
  overrides via `display_sig_by_title`); `residual_image`, if given, is a
  full-frame residual with all scenes subtracted (otherwise the panel shows
  `self.residual()`, which still contains other scenes' sources). See
  {doc}`diagnostics`.

`create_scene_graph(templates)` / `overlay_scene_graph(templates, shape)`
: Static helpers that label connected components by bounding-box overlap
  alone (no coupling threshold), and paint those labels into an image of
  `shape`. Diagnostic aids; the pipeline partition uses
  `build_scene_tree_from_normal`.

## The SceneFitter solver

{class}`mophongo.scene_fitter.SceneFitter` is stateless: all inputs arrive as
arguments and results are returned, so the same instance serves every scene.

`SceneFitter.solve(A, b, *, AB=None, BB=None, bB=None, config=None,
cg_kwargs=None)`
: Solve the scene system and return a namespace with `flux`, `err`,
  `shifts`, `info`. The flux block receives a small adaptive ridge:
  `config.reg_flux` if positive, else `1e-6` times the median positive
  diagonal of `A`. When shift blocks `AB`, `BB`, `bB` are supplied and
  non-empty, the shift block is regularized by `config.reg_astrom` times the
  median diagonal of `BB` and solved jointly; empty shift blocks (a scene
  with fewer than two bright members) fall back to flux-only. Despite the
  name, `cg_kwargs` is unused as of this writing: the solve is a direct
  sparse factorization (`scipy.sparse.linalg.spsolve`), not conjugate
  gradients, and `info` is always the dict `{"cg_info": 0}`.

`SceneFitter.solve_flux(A, b, config=None)`
: Flux-only path. The matrix is whitened by its diagonal,
  $\hat A = D^{-1} A D^{-1}$ with $D = \mathrm{diag}(\sqrt{A_{ii}})$, solved
  directly, and unwhitened. Errors are $\sqrt{\mathrm{diag}(\hat
  A^{-1})}/d$. If `config.positivity` is true, negative fluxes are clipped
  to zero after the solve (a post-hoc clamp, not a constrained NNLS solve).

`build_normal(templates, image, weights)`
: Module-level, stateless clone of `SparseFitter.build_normal_tree`; returns
  `(ATA, ATb, rtree)`.

## Astrometric shift blocks

Residual astrometry between the detection image and a band — distortion
residuals, guide-star catalog offsets — biases blended photometry. Mophongo
fits a smooth per-scene shift field jointly with the fluxes. To first order a
template offset by $(\delta x, \delta y)$ is

$$
T_i(\mathbf{r} - \boldsymbol{\delta}) \approx
T_i - \delta x\,\partial_x T_i - \delta y\,\partial_y T_i ,
$$

with gradients from `np.gradient` of the template stamp. The shift field is a
Chebyshev polynomial over scene coordinates scaled to roughly $[-1, 1]$:
$\delta x(x,y) = \sum_k \beta^x_k\,\phi_k(u,v)$ and likewise for $\delta y$,
where $\phi_k$ are the 2-D Chebyshev cross terms
({func}`mophongo.astrometry.cheb_basis`) of order
`FitConfig.astrom_kwargs["poly"]["order"]`. Order 0 fits one constant offset
per scene. The joint system is the block matrix

$$
\begin{pmatrix} A & A_B \\ A_B^\top & B \end{pmatrix}
\begin{pmatrix} \alpha \\ \beta \end{pmatrix}
=
\begin{pmatrix} b \\ b_B \end{pmatrix},
$$

whitened (flux block by its diagonal, shift block by Cholesky) and solved
directly. Only bright anchors — templates passing the SNR, isolation, and
optional star cuts — contribute to the coupling and shift blocks, using their
diagonal-only flux estimates as the amplitude scale.

### Building the blocks

{func}`mophongo.scene.make_scene_basis`, called as
`make_scene_basis(templates, bright, order=1)`, returns
per-template basis vectors (or `None` for faint members), the scene center
`(x0, y0)`, and half-range scales `(Sx, Sy)` computed from the bright members.

{func}`mophongo.scene.assemble_scene_system_AB` builds `AB`, `BB`, `bB` for
one scene:

`templates` (`list[Template]`), `image`, `weights` (`np.ndarray`)
: Scene members and full-frame arrays (sliced per template).

`basis_vals` (`list[np.ndarray | None]`)
: Output of `make_scene_basis`, aligned to `templates`.

`alpha0` (`np.ndarray | float | None`)
: Flux seeds scaling the gradient terms — an array of shape `(n_scene,)`, a
  scalar broadcast to all, or `None` for zeros. `Scene.solve` passes the
  diagonal-only estimates $b_i/A_{ii}$.

`order` (`int`, default `1`)
: Chebyshev order (sizes the shift block).

`include_y` (`bool`, default `True`)
: Fit $\delta y$ as well as $\delta x$.

`ab_from_bright_only` (`bool`, default `True`)
: Restrict flux–shift coupling rows to bright members.

A scene needs at least two bright members; otherwise empty blocks are
returned and the solver falls back to flux-only, leaving that scene's
templates unshifted (logged as a warning).

### Applying fitted shifts and iterating

After a joint solve, `Scene.solve` evaluates the polynomial at every template
position (via {func}`mophongo.astrometry.AstroCorrect.build_poly_predictor`)
and stores `(dx, dy)` on `tmpl.to_shift`. With `apply_shifts=True`,
{func}`mophongo.templates.Templates.apply_template_shifts` resamples each
template by $(-dx, -dy)$ — the fitted shift moves the image onto the
template, so the template moves the opposite way — always interpolating from
the original unshifted stamp so repeated passes do not accumulate
interpolation smoothing. Shifts below 0.01 pixel are skipped.

The linearization only captures part of a large offset per pass, so the
pipeline iterates: up to `FitConfig.fit_astrometry_niter` solve/apply passes
per band, stopping early once the largest per-template shift increment falls
below `FitConfig.astrom_shift_tol` (in fit-grid pixels).

## Fitting-related FitConfig fields

{class}`mophongo.fit.FitConfig` collects solver, astrometry, scene, and
photometry settings. The fields below control fitting; the full reference,
including aperture and template-extraction fields, is on the {doc}`pipeline`
page.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `positivity` | `bool` | `True` | Clip negative fitted fluxes to zero after the solve. |
| `reg_flux` | `float` | `0.0` | Ridge added to the flux block diagonal; `0` uses an adaptive `1e-6` times the matrix scale. |
| `bad_value` | `float` | `np.nan` | Fill value for missing catalog entries. |
| `cg_kwargs` | `dict` | `{"M": None, "maxiter": 500, "atol": 1e-6}` | Iterative-solver options; unused by the current direct solver. |
| `fit_astrometry_niter` | `int` | `5` | Maximum astrometry solve/apply passes per band; `0` disables shift fitting. |
| `astrom_shift_tol` | `float` | `0.05` | Stop iterating once the largest per-template shift increment (fit-grid pixels) drops below this. |
| `fit_astrometry_joint` | `bool` | `True` | Fit shifts jointly with fluxes inside each scene; if `False`, shifts come from the separate {class}`mophongo.astrometry.AstroCorrect` step. |
| `reg_astrom` | `float` | `1e-4` | Ridge on the shift block, relative to its diagonal scale. |
| `snr_thresh_astrom` | `float` | `15.0` | Minimum SNR proxy $b_i/\sqrt{A_{ii}}$ for a bright astrometric anchor; `0` keeps all. |
| `astrom_isolation_thresh` | `float` | `0.7` | Minimum flux dominance (0–1) within its own footprint for a template to anchor astrometry; `0.0` disables the cut. |
| `astrom_exclude_stars` | `bool` | `False` | Exclude `is_star` templates from the shift fit. Off by default: unsaturated stars are the best anchors, and saturated ones already sit in singleton scenes. |
| `astrom_model` | `str` | `"gp"` | Model for the separate (non-joint) astrometry step: `"poly"` or `"gp"`; any other value raises `ValueError`. |
| `astrom_centroid` | `str` | `"centroid"` | Shift measurement for the separate step: `"centroid"` or `"correlation"`. |
| `astrom_kwargs` | `dict` | `{"poly": {"order": 0}, "gp": {"length_scale": 400}}` | Per-model options; the joint scene fit reads `astrom_kwargs["poly"]["order"]`. |
| `multi_resolution_method` | `str` | `"upsample"` | Multi-resolution handling (see {doc}`pipeline`). |
| `normal` | `str` | `"tree"` | Normal-matrix builder; only `"tree"` is implemented. |
| `scene_minimum_bright` | `int` | `5` | Minimum bright anchors per scene; smaller scenes are merged. `None` derives it from the polynomial order in `__post_init__`. |
| `run_scene_solver` | `bool` | `True` | Must remain `True`; the scene solver is the only fitting path and `False` raises. |
| `scene_coupling_thresh` | `float` | `1e-3` | Leakage score above which templates share a scene. |
| `scene_max_size` | `int \| None` | `800` | Soft cap on templates per scene, enforced by local threshold-raising. `None` disables. |
| `scene_max_merge_radius` | `float` | `1000.0` | Maximum distance (pixels) over which underfilled scenes merge. |
| `generate_scene_catalog` | `bool` | `False` | Write `scene_catalog_<i>.ecsv` and exit without fitting. |

Aperture fields (`aperture_diam`, `aperture_catalog`, `aperture_units`) and
template-extraction fields (`template_dilate_segmap`,
`skip_template_extension_for_deblended`, `extend_wings_background_only`) are
documented on the {doc}`pipeline` and {doc}`templates` pages.

## Relation to catalog outputs

For each band `i`, {meth}`mophongo.pipeline.Pipeline.run` writes the raw
fitted template amplitudes as `flux_<i>` with solver errors `err_<i>` and
predicted errors `err_pred_<i>`. Because fitting uses unit-sum PSF shapes,
these are modeled-stamp fluxes; the throughput-corrected totals
(`flux_<i>_total`, divided by the filter-level finite-stamp PSF throughput)
and the fitted shifts (`shift_x`, `shift_y`) are described in {doc}`outputs`.
