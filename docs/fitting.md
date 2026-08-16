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

```python
import numpy as np
from mophongo.psf import PSF
from mophongo.scene_fitter import build_normal
from mophongo.templates import Templates

rng = np.random.default_rng(11)
hires = rng.normal(0, 1e-3, (101, 101))         # detection image, FWHM 3
band = rng.normal(0, 1e-3, (101, 101))          # fit image, FWHM 5
positions, fluxes = [(45.0, 50.0), (53.0, 50.0)], (50.0, 20.0)
for (x, y), flux in zip(positions, fluxes):
    sly, slx = slice(int(y) - 15, int(y) + 16), slice(int(x) - 15, int(x) + 16)
    hires[sly, slx] += flux * PSF.gaussian(31, 3.0).array
    band[sly, slx] += flux * PSF.gaussian(31, 5.0).array
yy, xx = np.indices(hires.shape)
near = np.argmin([np.hypot(xx - x, yy - y) for x, y in positions], axis=0)
segmap = np.where(hires > 5e-3, near + 1, 0)    # nearest-source segments
tmpls = Templates.from_image(hires, segmap, positions, kernel=PSF.gaussian(31, 4.0).array)
ATA, ATb, _ = build_normal(list(tmpls), band, np.ones_like(band))
print(np.round(ATA.toarray() / ATA.diagonal()[:, None], 3))  # [[1. 0.028], [0.028 1.]]
```

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

```python
import numpy as np
from mophongo.psf import PSF
from mophongo.scene import generate_scenes
from mophongo.templates import Templates

rng = np.random.default_rng(11)
hires = rng.normal(0, 1e-3, (161, 161))         # detection image, FWHM 3
band = rng.normal(0, 1e-3, (161, 161))          # fit image, FWHM 5
xy = rng.uniform(20, 140, size=(40, 2))
for (x, y), flux in zip(xy, rng.uniform(5, 50, size=40)):
    sly, slx = slice(int(y) - 15, int(y) + 16), slice(int(x) - 15, int(x) + 16)
    hires[sly, slx] += flux * PSF.gaussian(31, 3.0).array
    band[sly, slx] += flux * PSF.gaussian(31, 5.0).array
yy, xx = np.indices(hires.shape)
near = np.argmin([np.hypot(xx - x, yy - y) for x, y in xy], axis=0)
segmap = np.where(hires > 5e-3, near + 1, 0)    # nearest-source segments
tmpls = Templates.from_image(hires, segmap, xy, kernel=PSF.gaussian(31, 4.0).array)
for thresh in (1e-1, 1e-3):
    scenes, labels = generate_scenes(list(tmpls), band, coupling_thresh=thresh, minimum_bright=0)
    print(thresh, len(scenes), np.bincount(labels)[1:].max())  # 0.1 33 3 / 0.001 22 6
```

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
`ATA`/`ATb`. The partition is set by the coupling threshold (the pipeline
passes `FitConfig.scene_coupling_thresh`, default `1e-3`) and the soft
per-scene cap `max_size`; the merge step is set by the bright-anchor
definition (SNR and isolation cuts) and the merge radius.

Saturated or repaired templates are held out of that partitioning
altogether (`isolate_saturated`, on by default). Their PSF wings extend far
beyond their segment, so leaving them in the coupling graph glues every
neighbour under the wings into one scene — and the star is then pulled out
again, leaving that scene shaped by a member it no longer has. Everything
else is partitioned on its own, and the saturated templates get scenes
afterwards: fragments sharing a `sat_group` id (the star's core segment id,
from `FLAG_SATURATED_*` — see {doc}`repair`) are the oversplit pieces of one
star and go into one scene together, fitted jointly with a single rigid
shift. These scenes never pass through the merge step, so they are exempt
from `minimum_bright` by construction. All parameters and defaults are
documented in the {doc}`api` reference.

```python
from mophongo.fit import FitConfig
from mophongo.scene import generate_scenes

cfg = FitConfig(fit_astrometry_niter=3)
scenes, labels = generate_scenes(
    templates, image, weight,
    coupling_thresh=cfg.scene_coupling_thresh,
    max_size=cfg.scene_max_size,
    minimum_bright=cfg.scene_minimum_anchors,
)
for scn in scenes:
    scn.set_band(image, weight, config=cfg)
    flux, err, shifts, info = scn.solve(config=cfg, apply_shifts=True)
```

### Working with a Scene

{class}`mophongo.scene.Scene` holds one scene's templates together with its
fit state: the scene-local normal block, the band image and weights, the
bright-anchor mask, and — after a solve — the full result on `solution`.
The workflow is the loop in the example above:
{meth}`~mophongo.scene.Scene.set_band` caches a band's image and weights,
and {meth}`~mophongo.scene.Scene.solve` solves the scene, flux-only or
jointly with the per-scene shift field when `FitConfig.fit_astrometry_joint`
is on, writing per-source results onto each template (`tmpl.flux`,
`tmpl.err`). With `apply_shifts=True` a joint solve also resamples the
templates by the fitted, damped shifts and clears the cached normal block so
the next pass rebuilds it against the shifted templates (see
[Applying fitted shifts and iterating](#applying-fitted-shifts-and-iterating)).

```python
import numpy as np
from mophongo.fit import FitConfig
from mophongo.psf import PSF
from mophongo.scene import generate_scenes
from mophongo.templates import Templates

rng = np.random.default_rng(11)
hires = rng.normal(0, 1e-3, (101, 101))         # detection image, FWHM 3
band = rng.normal(0, 1e-3, (101, 101))          # fit image, FWHM 5
positions, fluxes = [(45.0, 50.0), (53.0, 50.0)], (50.0, 20.0)
for (x, y), flux in zip(positions, fluxes):
    sly, slx = slice(int(y) - 15, int(y) + 16), slice(int(x) - 15, int(x) + 16)
    hires[sly, slx] += flux * PSF.gaussian(31, 3.0).array
    band[sly, slx] += flux * PSF.gaussian(31, 5.0).array
yy, xx = np.indices(hires.shape)
near = np.argmin([np.hypot(xx - x, yy - y) for x, y in positions], axis=0)
segmap = np.where(hires > 5e-3, near + 1, 0)
tmpls = Templates.from_image(hires, segmap, positions, kernel=PSF.gaussian(31, 4.0).array)
wht = np.full(band.shape, 1e6)                  # inverse variance of the 1e-3 noise
scenes, _ = generate_scenes(list(tmpls), band, wht, coupling_thresh=1e-3, minimum_bright=0)
flux, err, shifts, info = scenes[0].solve(config=FitConfig(fit_astrometry_niter=0))
print(np.round(flux, 2), np.round(err, 4))      # [50.01 19.95] [0.0075 0.0075]
```

For inspection, {meth}`~mophongo.scene.Scene.model_image` and
{meth}`~mophongo.scene.Scene.residual` render the scene model and
image-minus-model over the scene bounding box,
{meth}`~mophongo.scene.Scene.shift_at` evaluates the applied shift at
arbitrary positions by nearest-template lookup, and
{meth}`~mophongo.scene.Scene.plot` draws the six-panel scene diagnostic
(template, image, model, segmap, residual, color composite) with the fitted
shift field as arrows on the model panel — see {doc}`diagnostics`. The
static helpers {meth}`~mophongo.scene.Scene.create_scene_graph` and
{meth}`~mophongo.scene.Scene.overlay_scene_graph` label connected
components by bounding-box overlap alone, as a diagnostic aid; the pipeline
partition uses {func}`~mophongo.scene.build_scene_tree_from_normal`.

## The SceneFitter solver

{class}`mophongo.scene_fitter.SceneFitter` is stateless: all inputs arrive as
arguments and results are returned, so the same instance serves every scene.
`Scene.solve` hands it the normal block and optional shift blocks;
{meth}`~mophongo.scene_fitter.SceneFitter.solve` returns a namespace with
`flux`, `err`, `shifts`, `info`. The flux block receives a small adaptive
ridge (`FitConfig.reg_flux`; `None` applies an adaptive `1e-6` times the matrix scale, `0.0` none at all) and
the shift block a relative ridge `FitConfig.astrom_reg`; a scene with fewer
than two bright members has empty shift blocks and falls back to the
flux-only path, {meth}`~mophongo.scene_fitter.SceneFitter.solve_flux`, which
whitens the matrix by its diagonal as described under
[Error estimates](#error-estimates-err-vs-err_pred) above. Despite some
argument names, the solve is a direct sparse factorization
(`scipy.sparse.linalg.spsolve`), not conjugate gradients, and `info` is
always `{"cg_info": 0}`. If `FitConfig.positivity` is true, negative fluxes
are clipped to zero after the solve (a post-hoc clamp, not a constrained
NNLS solve). {func}`mophongo.scene_fitter.build_normal` is the module-level,
stateless clone of `SparseFitter.build_normal_tree` that assembles
`(ATA, ATb, rtree)` for a template list.

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

The shift columns of the design are scene-wide sums over the bright anchors,

$$
B_k = -\sum_i \alpha_i\,\phi_k(u_i, v_i)\,\nabla T_i ,
$$

so `AB` $= A^\top W B$, `BB` $= B^\top W B$ and `bB` $= B^\top W d$ keep the
cross-template terms: a flux row couples to its neighbours' gradients,
distinct anchors couple to each other, and the $x$–$y$ block is populated.
Accumulating only each template's own gradient products instead is exact for
an isolated anchor, but in a blend it reads the residual dipole of an
overlapping neighbour as a shift — on a perfectly aligned synthetic blend
that produced a spurious order-1 shift field of 0.05 px rms, and it shrank a
recovered 0.30 px offset to 0.18 px at 6 px separation.

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

Convergence is judged per scene, not per band. A scene's solve reads only its
own templates, image and weights, so once its increment drops below the
tolerance it cannot start moving again and it leaves the pass list; the
remaining passes iterate only the scenes still moving. Each scene records what
happened in `Scene.astrom_step` (its last increment), `Scene.astrom_niter`
(passes used) and `Scene.astrom_converged`. Scenes still above tolerance when
the budget runs out hold their last iterate rather than a converged solution,
and the run logs a warning naming the worst of them — usually crowded scenes,
or ones whose bright anchors are marginally sampled.

A verdict is only recorded where a shift was actually fitted.
`astrom_converged` stays `None` — and `flag_astrom` `-1` — for a flux-only
run and for scenes with too few bright anchors to carry a shift block. Those
templates trivially do not move, and reporting that as convergence would
claim an astrometric solution that was never solved for.

Once the passes finish, every scene gets one more flux-only solve on its
final templates, converged or not. A pass solves fluxes and shifts together
and *then* resamples the templates, so the fluxes it produced belong to the
basis as it stood before that pass's shift — the last applied shift is never
accounted for. Without the closing pass the stored fluxes, errors, model and
residual would describe a template basis that no longer exists. The extra
solve leaves the fitted shifts alone and costs one pass.

The tolerance is a stopping rule, not an accuracy claim, and `0.1` fit-grid
pixels is chosen against the noise rather than against the arithmetic. The
weakest scene the anchor cuts admit — `scene_minimum_anchors` = 3 members at
`astrom_minimum_snr` = 15 — has a centroid good to roughly
$\sigma_{\rm PSF}/({\rm SNR}\sqrt{N}) \approx 0.10$ fit pixels, so iterating
below that measures nothing; and the systematic floor from PSF and kernel
mismatch is a bias, which no number of passes removes. Tightening the
tolerance buys passes, not precision. Because the fit grid is the
high-resolution grid on the upsample path, and not the grid of the band being
fitted, the run logs the tolerance in mas alongside pixels.

### Anchor leverage

Written out, the shift block is a weighted least-squares fit of per-anchor
shifts onto the basis:

$$
I_i = \alpha_i^2 \langle G_x, w, G_x\rangle, \qquad
\mathrm{d}x_i = -\frac{\langle G_x, w, r\rangle}{\alpha_i \langle G_x, w, G_x\rangle},
\qquad
BB = \sum_i I_i S_i S_i^\top, \quad
bB = \sum_i I_i\, \mathrm{d}x_i\, S_i .
$$

Leverage $I_i$ grows as flux squared, so one bright source can carry a scene.
That matters when the source is extended with an asymmetric colour gradient:
its residual is a dipole aligned with its own template gradient, formally
indistinguishable from a shift, and it drags the fitted field. No per-source
test on the residual separates the two cases — the residual really does look
like a shift.

`FitConfig.astrom_leverage_cap` bounds the damage without pretending to
identify it. Anchors above the quantile $I_\mathrm{cap}$ have their
contribution to $AB$, $BB$ and $bB$ scaled by $I_\mathrm{cap}/I_i$, which is
the system you get by scaling that source's pixel weights inside the shift
equations only: the anchor still measures the same $\mathrm{d}x_i$, it just
counts less. The flux block is untouched, so photometry does not change.
$I_i$ depends on the template, its flux seed and the weight map but not on
the residual, so the cap costs nothing — it is applied while the blocks are
assembled, with no extra pass.

Its limitation is the flip side of its simplicity: it clips the brightest
anchors, which are often the best ones, and does nothing in a scene where the
offending source is the only bright member. Separating a real offset from a
colour gradient needs the coherence of *neighbouring* anchors, since real
offsets are smooth in position while morphology-driven pseudo-shifts are
random per source; that robust variant is listed in `TODO.md`.

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
:alt: Two panels showing the per-pass applied shift increment and the flux error of the joint solve, for damped and undamped iterations recovering a 1.5-pixel offset.

Solve/apply iteration on a synthetic scene whose band image carries a true
offset of $(1.5, -0.8)$ pixels. The applied increment shrinks by roughly an
order of magnitude per pass until it crosses `astrom_shift_tol` (left, dashed
line), after which the scene leaves the loop; below 0.01 pixel in both axes
(dotted) nothing is applied at all and the templates stop moving, which is
why the series ends. The flux error of the joint solve (right) is already at
its floor by pass two, so the passes a looser tolerance skips cost no
accuracy. Damping (0.8, the production default) costs about one extra pass
relative to the undamped step.
```

## Fitting-related FitConfig fields

{class}`mophongo.fit.FitConfig` collects solver, astrometry, scene, and
photometry settings. The fields below control fitting; the full reference,
including aperture and template-extraction fields, is on the {doc}`pipeline`
page.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `positivity` | `bool` | `True` | Clip negative fitted fluxes to zero after the solve. |
| `reg_flux` | `float \| None` | `None` | Ridge added to the flux block diagonal. `None` = adaptive (`1e-6` times the matrix scale), `0.0` = genuinely unregularized, positive = that value. JSON configs write `null` for the default. |
| `bad_value` | `float` | `np.nan` | Fill value for missing catalog entries. |
| `cg_kwargs` | `dict` | `{"M": None, "maxiter": 500, "atol": 1e-6}` | Iterative-solver options; unused by the current direct solver. |
| `fit_astrometry_niter` | `int` | `5` | Maximum astrometry solve/apply passes per band; `0` disables shift fitting. |
| `astrom_shift_tol` | `float` | `0.1` | Stop iterating a scene once its largest per-template shift increment (fit-grid pixels) drops below this. |
| `astrom_damping` | `float` | `0.8` | Factor applied to each pass's fitted shift increment before it is applied to the templates; `1.0` is undamped. |
| `fit_astrometry_joint` | `bool` | `True` | Fit shifts jointly with fluxes inside each scene; if `False`, shifts come from the separate {class}`mophongo.astrometry.AstroCorrect` step. |
| `astrom_reg` | `float` | `1e-4` | Ridge on the shift block, relative to its diagonal scale. |
| `astrom_minimum_snr` | `float` | `15.0` | Minimum SNR proxy $b_i/\sqrt{A_{ii}}$ for a bright astrometric anchor; `0` keeps all. |
| `astrom_isolation_thresh` | `float` | `0.7` | Minimum flux dominance (0–1) within its own footprint for a template to anchor astrometry; `0.0` disables the cut. Roughly a separation cut: 0.6 admits blends down to ~1.2 PSF sigma, 0.7 to ~2. Not superseded by `astrom_robust` — see below. |
| `astrom_leverage_cap` | `float \| None` | `0.9` | Cap each anchor's leverage at this quantile of the scene's anchor information. Bounds how much one bright source can move the shift field; `None` leaves the weights alone. |
| `astrom_robust` | `bool` | `True` | Weight each anchor by how well it agrees with the shift field its neighbours define, and by how well its own stamp fits once it is allowed to move ({mod}`mophongo.astrom_robust`). Set `False` to recover the unweighted fit. |
| `astrom_exclude_stars` | `bool` | `False` | Exclude `is_star` templates from the shift fit. Off by default: unsaturated stars are the best anchors, and saturated ones already sit in singleton scenes. |
| `astrom_model` | `str` | `"poly"` | Model for the separate (non-joint) astrometry step: `"poly"` or `"gp"`; any other value raises `ValueError`. The joint path ignores it and always uses the polynomial basis. |
| `astrom_centroid` | `str` | `"centroid"` | Shift measurement for the separate step: `"centroid"` or `"correlation"`. |
| `astrom_kwargs` | `dict` | `{"poly": {"order": 0}, "gp": {"length_scale": 400}}` | Per-model options; the joint scene fit reads `astrom_kwargs["poly"]["order"]`. |
| `multi_resolution_method` | `str` | `"upsample"` | Multi-resolution handling (see {doc}`pipeline`). |
| `normal` | `str` | `"tree"` | Normal-matrix builder; only `"tree"` is implemented. |
| `scene_minimum_anchors` | `int \| None` | `None` | Minimum bright anchors per scene; smaller scenes are merged. `None` derives it from the polynomial order as `(order+1)(order+2)+1` — 3 at order 0, 7 at order 1. Also the gate for `astrom_robust`. |
| `run_scene_solver` | `bool` | `True` | Must remain `True`; the scene solver is the only fitting path and `False` raises. |
| `scene_coupling_thresh` | `float` | `1e-3` | Leakage score above which templates share a scene. |
| `scene_max_size` | `int \| None` | `800` | Soft cap on templates per scene, enforced by local threshold-raising. `None` disables. |
| `scene_max_merge_radius` | `float` | `1500.0` | The scene length scale (pixels): scenes wider than this are split, underfilled scenes look no further for a merge partner, and a merge that would exceed it is refused. `np.inf` disables all three. |
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
