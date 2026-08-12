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
$A_{ij}$ is the weighted integral of $T_i T_j$, so off-diagonal terms measure
how much flux from one source leaks into the footprint of another — the same
quantity later reused to partition the field into scenes.

### SparseFitter

{class}`mophongo.fit.SparseFitter` is no longer part of the production
fitting path: {meth}`mophongo.pipeline.Pipeline.run` never instantiates it,
and all fitting is done by the scene solver — {class}`~mophongo.scene.Scene`,
{class}`~mophongo.scene_fitter.SceneFitter`, and
{func}`mophongo.scene_fitter.build_normal`, a stateless clone of
`SparseFitter.build_normal_tree`. `SparseFitter` remains a public standalone
class, exercised by the test suite, that assembles the normal equations
described above for a flat template list. In brief:

- {meth}`~mophongo.fit.SparseFitter.build_normal` (and the STRtree builder
  {meth}`~mophongo.fit.SparseFitter.build_normal_tree` behind it) builds the
  sparse normal matrix and right-hand side, cached on the lazy `ata`/`atb`
  properties.
- {meth}`~mophongo.fit.SparseFitter.add_flux_priors` adds per-template
  Gaussian flux priors to the system; an interactive hook, unused by the
  package.
- {meth}`~mophongo.fit.SparseFitter.model_image` /
  {meth}`~mophongo.fit.SparseFitter.residual` render the full-frame model
  $\sum_i \alpha_i T_i$ from a stored solution, and `image - model`.
- {meth}`~mophongo.fit.SparseFitter.quick_flux`,
  {meth}`~mophongo.fit.SparseFitter.predicted_errors`, and
  {meth}`~mophongo.fit.SparseFitter.flux_and_rms` give quick per-template
  flux and error estimates that ignore neighbor covariance.

## Error estimates: `err` vs `err_pred`

Two uncertainty estimates are produced for every source, and both appear in
the output catalog (`err_<i>` and `err_pred_<i>`; see {doc}`outputs`):

- **`err`** — the solver error, $\sqrt{[\hat A^{-1}]_{ii}}\,/\,d_i$ with
  $d_i = \sqrt{A_{ii}}$, computed from the whitened normal matrix
  $\hat A = D^{-1} A D^{-1}$ of the source's scene. It includes the
  covariance with overlapping neighbors: blended sources get larger `err`
  than isolated ones of the same brightness. When astrometric shift
  parameters are fit jointly, the shift block is marginalized out first:
  $\hat A$ is replaced by the Schur complement
  $S = A - A_B B^{-1} A_B^\top$, formed in the whitened basis where $B$
  becomes the identity, so `err` also accounts for the flux–shift covariance.
  For scenes up to 500 templates the inverse diagonal is obtained by dense
  inversion; larger scenes use one sparse LU factorization and per-column
  back-solves.
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

```{figure} images/scene_partition.png
:width: 100%
:alt: Three panels showing a synthetic field with segmentation outlines, the same segments colored by scene membership, and a histogram of scene sizes.

Scene partition of a synthetic field of 120 sources. Left: the fit image with
segmentation outlines. Middle: each segment colored by the scene it belongs
to — templates whose cross-coupling exceeds the threshold share a scene and
are solved together, while isolated sources form single-template scenes.
Right: the resulting scene-size distribution, from singletons to one
34-source group.
```

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
An edge connects $i$ and $j$ when $s_{ij}$ reaches the coupling threshold;
scenes are the connected components. A lower threshold merges more sources
into fewer, larger scenes. Given the unwhitened `ATA`/`ATb`,
`build_scene_tree_from_normal` returns `(labels, nscene)`; an optional soft
`max_size` cap splits an oversized component by raising the threshold
locally within that component, so strong couplings elsewhere in the field
are never cut on behalf of one crowded region. The pipeline calls it
through `generate_scenes`.

### Merging small scenes

The astrometric shift fit needs several bright anchors per scene, so
{func}`mophongo.scene.merge_small_scenes` iteratively merges scenes with
fewer than `minimum_bright` bright members into their nearest neighboring
scene, out to at most `max_merge_radius` pixels so merges stay local
(underfilled scenes with no neighbor within the radius stay as they are).
The pipeline calls it through `generate_scenes`.

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
: `shifts` holds the fitted Chebyshev coefficients after a joint solve.
  `flux` and `err` are declared but never filled by `solve()`: per-source
  results are written onto `solution` and onto each template.

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
  stored on the scene (`solution`, `shifts`) and on each template
  (`tmpl.flux`, `tmpl.err`, `tmpl.is_bright`). After a joint solve the fitted
  shift field is always evaluated at each template position, scaled by
  `config.astrom_damping`, and stored on
  `tmpl.to_shift`; `apply_shifts=True` additionally resamples the templates
  (see below) and clears `A`/`b` so the next pass rebuilds them against the
  shifted templates.

`shift_at(x, y)`
: Evaluate the already-applied shift at positions `(x, y)` by nearest-template
  lookup; returns `(dx, dy)` arrays. It returns zeros unless the scene has both
  a shift fit and a spatial index, and `tree` is only populated when `solve()`
  rebuilds `A`/`b` itself — so a scene straight out of `generate_scenes`
  returns zeros on its first pass.

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
  `self.residual()` with the segment pixels of other scenes blanked, so their
  unsubtracted wings still show). See {doc}`diagnostics`.

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
  median positive diagonal of `BB` and solved jointly; empty shift blocks (a
  scene with fewer than two bright members) fall back to flux-only. Despite the
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

```{figure} images/shift_linearization.png
:width: 100%
:alt: Six panels showing a template, its x-gradient basis, the linear shift model, the spline-shifted template, and the residuals of both against the exactly shifted template.

Anatomy of one linearized shift step, for a 0.5-pixel offset in $x$. The
solver models the shifted template as the template (top left) plus the
gradient basis $-\partial T/\partial x$ (top center) scaled by $\delta x$
(top right); the approximation error against the exactly shifted template is
a few per cent of the peak (bottom center, stretched $\times 100$). The
cubic-spline resampling that then applies the accumulated shift (bottom
left) is accurate to about $2\times10^{-3}$ of the peak (bottom right), so
the linear step only steers the iteration while the applied shift is
effectively exact.
```

### Building the blocks

Two helpers, both called by `Scene.solve`, assemble the shift blocks.
{func}`mophongo.scene.make_scene_basis` returns per-template Chebyshev basis
vectors (or `None` for faint members) plus the scene center and half-range
scales computed from the bright members.
{func}`mophongo.scene.assemble_scene_system_AB` then builds the flux–shift
coupling block `AB`, the shift block `BB`, and the right-hand side `bB` for
one scene, seeding the gradient terms with the diagonal-only flux estimates
$b_i/A_{ii}$.

A scene needs at least two bright members; otherwise empty blocks are
returned and the solver falls back to flux-only, leaving that scene's
templates unshifted (logged as a warning).

### Applying fitted shifts and iterating

After a joint solve, `Scene.solve` evaluates the polynomial at every template
position (via {func}`mophongo.astrometry.AstroCorrect.build_poly_predictor`),
scales the result by `FitConfig.astrom_damping`, and stores the damped
`(dx, dy)` on `tmpl.to_shift`. The fitted shift is the offset of
the source in the band image relative to its template — the linearized model
is $T_i(\mathbf{r} - \boldsymbol{\delta})$, the template displaced by
$+\boldsymbol{\delta}$ — so with `apply_shifts=True`
{func}`mophongo.templates.Templates.apply_template_shifts` resamples each
template by $(+dx, +dy)$ onto the image, always interpolating from the
original unshifted stamp with the accumulated total shift so repeated passes
do not accumulate interpolation smoothing. The accumulated offset is tracked
on `tmpl.shifted` and `FLAG_SHIFTED` is set. Passes whose increment is below
0.01 pixel in both axes are skipped.

The linearization only captures part of a large offset per pass, so the
pipeline iterates: up to `FitConfig.fit_astrometry_niter` solve/apply passes
per band, stopping early once the largest change in accumulated per-template
shift falls below `FitConfig.astrom_shift_tol` (in fit-grid pixels; the test
measures the applied, i.e. damped, increment). The loop always runs at least
once, so `fit_astrometry_niter = 0` still gives each band a single flux-only
pass.

Each pass is also damped, because the same linearization can err in the other
direction. The gradients $\partial_x T_i$, $\partial_y T_i$ are central
differences of the template stamp, which underestimate the gradient of
structure near the sampling limit: a mode of wavenumber $k$ (radians per pixel)
has its gradient underestimated by $\sin k / k$, so the shift solved from it is
too large by $k / \sin k$. A scene dominated by marginally sampled cores can
therefore step past the true offset and oscillate instead of converging.
`FitConfig.astrom_damping` (default `0.8`) scales each pass's increment before
it is applied, which keeps the iteration contracting at the cost of roughly one
extra pass; `1.0` recovers the undamped step.

```{figure} images/shift_iteration_damping.png
:width: 100%
:alt: Two panels showing the per-pass shift increment and the flux error of the joint solve, for damped and undamped iterations recovering a 1.5-pixel offset.

Solve/apply iteration recovering a true offset of $(1.5, -0.8)$ pixels. The
per-pass shift increment shrinks geometrically until it crosses
`astrom_shift_tol` (left, dashed line), after which the loop stops; the flux
error of the joint solve (right) reaches its floor within two or three
passes. On this well-sampled source damping (0.8, the production default)
costs about one extra pass relative to the undamped step.
```

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
| `astrom_damping` | `float` | `0.8` | Factor applied to each pass's fitted shift increment before it is applied to the templates; `1.0` is undamped. |
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

Aperture fields (`aperture_diam`, `aperture_catalog`, `aperture_units`),
template-extraction fields (`template_dilate_segmap`,
`skip_template_extension_for_deblended`, `extend_wings_background_only`) and
the template build-scheme fields (`extend_mode` and the per-scheme knobs it
selects between) are documented on the {doc}`pipeline` and {doc}`templates`
pages.

## Relation to catalog outputs

For each band `i`, {meth}`mophongo.pipeline.Pipeline.run` writes the raw
fitted template amplitudes as `flux_<i>` with solver errors `err_<i>` and
predicted errors `err_pred_<i>`. Because fitting uses unit-sum PSF shapes,
these are modeled-stamp fluxes; the throughput-corrected totals
(`flux_<i>_total`, divided by the per-source encircled energy `ee_psf_lo` of
the low-resolution PSF stamp, or by the filter-level throughput where that is
missing) and the fitted shifts (`shift_x`, `shift_y`) are described in
{doc}`outputs`.
