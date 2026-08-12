# Template construction

Templates are the morphological basis of the fit: one cutout per source, taken
from the high-resolution detection image inside that source's segmentation
footprint, normalized to unit sum, optionally extended beyond the footprint,
and convolved with a PSF-matching kernel to the resolution of each measurement
band. The fitter ({doc}`fitting`) then solves for one amplitude per template.
This page documents {class}`mophongo.templates.Template`,
{class}`mophongo.templates.Templates`, the supporting cutout machinery in
`mophongo.templates`, and the alternative build schemes in
{mod}`mophongo.template_schemes`. For where template construction sits in the
full run, see {doc}`pipeline`; for kernel construction, see {doc}`psf`.

## Coordinate conventions

Every template carries two pixel coordinate systems:

- **Original coordinates**: pixel indices on the full parent image the cutout
  was taken from. Attributes with an `_original` suffix
  (`slices_original`, `bbox_original`, `input_position_original`,
  `position_original`) live in this frame.
- **Cutout coordinates**: pixel indices inside the template's own `data`
  array. Attributes with a `_cutout` suffix (`slices_cutout`,
  `input_position_cutout`, `position_cutout`) live in this frame.

Cutouts are made in "partial" mode: a template near the image edge keeps its
requested size, and pixels falling outside the parent image are zero-filled.
`slices_original` then selects the parent-image region that actually overlaps
the cutout, and `slices_cutout` selects the corresponding region inside
`data`; the two select arrays of identical shape, so
`image[tmpl.slices_original]` and `tmpl.data[tmpl.slices_cutout]` are always
pixel-aligned. Code that combines a template with a full image (residual
accumulation, weight lookups, quick flux estimates) uses this slice pair.

Positions follow the 0-based, pixel-center convention: integer coordinate
`(x, y)` is the center of pixel `[y, x]`. `input_position_*` keeps the exact
fractional source position; `position_*` is its half-up-rounded integer
counterpart.

## Template

{class}`mophongo.templates.Template` subclasses
`astropy.nddata.Cutout2D` and adds slice bookkeeping, provenance flags, and
fit results.

### Constructor

The constructor cuts a `(ny, nx)` stamp out of a full parent image at an
`(x, y)` position, always in `mode="partial"` with `fill_value=0.0`, storing
the source id as `Template.id` and keeping any parent WCS as `wcs_original`.
Templates are normally built for you by
{meth}`~mophongo.templates.Templates.extract_templates`; see the
{class}`~mophongo.templates.Template` API page for the full signature.

### Attributes set at construction

Beyond the Cutout2D geometry, a template records:

- `id`, `id_parent`, `id_scene`: source label, parent label, and scene
  membership (scene grouping is described in {doc}`fitting`).
- `deblend_parent_label` (`int | None`), `deblend_nchildren` (`int`):
  deblending provenance from the catalog ({doc}`catalog`).
- `flag` (`int`): bitwise diagnostic flags (below).
- `is_star` (`bool`): set by the pipeline from the catalog star flag.
- `flux`, `err`, `err_pred`, `wnorm` (`float`): fitted amplitude, solver
  error, predicted error, and weighted norm, filled in during fitting.
- `ee_psf_lo` (`float`): encircled energy of the low-resolution PSF stamp at
  this position, set by {meth}`~mophongo.templates.Templates.convolve_templates`;
  `NaN` until then.
- `ee_tmpl` (`float`): the template's own sum, recorded at the end of
  extraction and of the `"psf_convolution"` extension pass. Both steps renormalize the
  stamp, so it is 1.0 for any nonzero template and 0.0 for a zero-sum one —
  with one deliberate exception, the `"psf_wings"` build scheme, whose stamp
  is normalized before neighbor-owned pixels are dropped and therefore sums to
  slightly less than one (see [Template build schemes](#template-build-schemes)).
  It stays `NaN` when neither step ran (`extend_with_psf_model` never sets it).
  Wing flux withheld from a neighboring segment is reported by
  `extension_blocked_sum` on the `"psf_convolution"` path and by `wing_frac_lost` in
  `extend_info` on the `"psf_wings"` path. Diagnostic only: the fitted
  amplitude does not scale with it.
- `template_norm` (`float`): the stamp sum divided out by the unit-sum
  normalization, so `template_norm * data` reconstructs the composite and the
  implied detection-band flux stays known. Initialized to 0.0. On the `"psf_convolution"`
  extension path it means something different: the input template is already
  unit-sum there, so the recorded value is the multiplicative wing boost rather
  than a detection-band flux.
- `extension_mode` (`str`, default `"none"`): which build scheme or extension
  pass produced the current pixels.
- `extend_info` (`dict[str, float]`): per-source bookkeeping returned by an
  alternative build scheme; empty for `extend_mode="none"` and for the
  post-extraction passes, which record their diagnostics as individual
  `extension_*` attributes instead.
- `to_shift`, `shifted` (`np.ndarray`, length 2): pending and accumulated
  astrometric shifts in `(dx, dy)`.

### Flags

`Template.flag` is a bitmask built from class constants:
`FLAG_VALID` (0x01), `FLAG_CONVOLVED` (0x02), `FLAG_SUM_ZERO` (0x04),
`FLAG_HAS_NAN` (0x08), `FLAG_OUTSIDE_WEIGHT` (0x10), `FLAG_SHIFTED` (0x20),
`FLAG_DEBLENDED` (0x40), `FLAG_SATURATED` (0x80), `FLAG_PSF_EXTENDED` (0x100),
and `FLAG_EXTEND_FAILED` (0x200). Two properties wrap the provenance bits:
`is_deblended` marks templates that come from deblended catalog children, and
`is_saturated` marks saturated/repaired sources, which the scene builder
isolates into their own scene so their PSF wings do not contaminate
neighboring flux solutions. The last two bits are set by the build-time
schemes: `FLAG_PSF_EXTENDED` when a scheme blended any PSF model into the
template, `FLAG_EXTEND_FAILED` when extension was attempted but the PSF was
unusable.

### Geometry helpers

These per-template methods are internal plumbing used by the pipeline; each
link leads to the full API documentation.

- {meth}`~mophongo.templates.Template.from_stamp` rebuilds a template from
  serialized stamp pixels plus geometry, the inverse of the stamp output
  described in {doc}`outputs`.
- {meth}`~mophongo.templates.Template.pad` returns a new template enlarged by
  a given (even) padding, keeping original-image coordinates consistent.
- {meth}`~mophongo.templates.Template.convolve_cutout` convolves the template
  with a centered kernel, returning a new template enlarged to hold the full
  convolution result with provenance and `ee_*` metadata propagated and
  `FLAG_CONVOLVED` set.
- {meth}`~mophongo.templates.Template.downsample` performs flux-conserving
  `k`-fold binning aligned to the global high-resolution grid (the
  "downsample" multi-resolution path); binning is exact only for `k`-aligned
  cutouts, and misaligned origins lose the trailing row/column with a warning.
- {meth}`~mophongo.templates.Template.project_to_block_replicated_grid`
  projects the template onto the globally aligned block-replicated grid of
  the upsampled multi-resolution path ({doc}`pipeline`): the template is
  integrated over native-pixel blocks and the block means replicated back,
  so it lives in the same pixel basis as the block-replicated image it is
  fitted against.

## Templates

{class}`mophongo.templates.Templates` is the container for a band's template
list. It is iterable and indexable; `len(tmpls)` gives the template count, and
the `templates` property returns the underlying list. The class attribute
`min_size = 8` sets the minimum cutout size in pixels. After extraction the
container also stores `segmap` (the dilated segmentation map) and
`original_shape`.

### Extraction entry points

#### Templates.from_image

```python
Templates.from_image(hires_image, segmap, positions, kernel=None,
                     extension=None, wcs=None)
```

Convenience constructor: extracts templates and, when a kernel is given,
calls {meth}`~mophongo.templates.Templates.convolve_templates` with
`inplace=True`, so the stored templates are the convolved ones.

- `hires_image` (`np.ndarray`): high-resolution detection image.
- `segmap` (`np.ndarray`): segmentation map on the same grid; each pixel
  belongs to at most one source label.
- `positions` (iterable of `(x, y)`): source positions in pixel coordinates.
- `kernel` (`np.ndarray | None`, default `None`): PSF-matching kernel; when
  given, {meth}`~mophongo.templates.Templates.convolve_templates` is applied.
- `extension` (`np.ndarray | str | None`, default `None`): accepted but
  currently unused. `from_image` extracts with `extend_mode="none"`; to build
  extended templates, call
  {meth}`~mophongo.templates.Templates.extract_templates` directly with an
  `extend_mode`, or apply
  {meth}`~mophongo.templates.Templates.extend_with_psf` or
  {meth}`~mophongo.templates.Templates.extend_with_psf_model` afterwards.
- `wcs` (`WCS | None`, default `None`): WCS of the high-resolution image.

#### Templates.extract_templates

```python
tmpls = Templates()
templates = tmpls.extract_templates(hires_image, segmap, positions,
                                    wcs=None, dilate_segmap=0, *,
                                    extend_mode="none", detection_psf=None,
                                    detection_weight=None, wren=None,
                                    classic=None, psf_wings=None)
```

- `hires_image` (`np.ndarray`): high-resolution detection image.
- `segmap` (`np.ndarray`): segmentation map.
- `positions` (iterable of `(x, y)`): source positions; non-finite or
  out-of-bounds positions, and positions falling on segmentation background,
  are skipped silently.
- `wcs` (`WCS | None`, default `None`): image WCS. The `"wren"` scheme uses it
  to convert its halo annulus width from arcsec to pixels.
- `dilate_segmap` (`int`, default `0`): disk radius in pixels used to dilate
  each segment into background only before cutting; dilation never lets
  segments overlap neighbors. The default matches
  `FitConfig.template_dilate_segmap`, so pipeline runs and direct calls agree.
  Dilation is off because a dilated ring is mostly sky noise and its
  contested-background tie-break is catalog-id ordered rather than geometric;
  recovering the PSF wings is the job of the build scheme, and neither
  reference scheme dilates.

The keyword-only parameters select and configure the build scheme:

- `extend_mode` (`str`, default `"none"`): one of `EXTEND_MODES`, or the alias
  `"default"`. Any other value raises `ValueError`. See
  [Template build schemes](#template-build-schemes) below. Note that the
  default here is `"none"`, whereas `FitConfig.extend_mode` — what a pipeline
  run uses — defaults to `"psf_wings"`.
- `detection_psf` (`np.ndarray | PSFRegionMap | None`, default `None`):
  high-resolution PSF on the detection grid. Required by every build-time
  scheme; omitting it with one of those modes raises `ValueError`. A region
  map is looked up per source at the template's sky position, so each source
  gets its local PSF; a single derived template size is set from the *widest*
  member of the map ({func}`~mophongo.template_schemes.representative_psf`).
- `detection_weight` (`np.ndarray | None`, default `None`): detection-band
  inverse variance on the detection grid. All three build-time schemes measure
  a source SNR; with a weight map they use the formal per-pixel noise
  `sqrt(sum 1/ivar)`, without one they fall back to a single scalar noise
  measured once per extraction (below).
- `wren`, `classic`, `psf_wings`
  ({class}`~mophongo.template_schemes.WrenParams`,
  {class}`~mophongo.template_schemes.ClassicParams`,
  {class}`~mophongo.template_schemes.PsfWingsParams`, all default `None`):
  per-scheme knobs, documented under
  [Scheme parameters](#scheme-parameters). Defaults are used when omitted;
  parameters of the schemes that are not selected are ignored.

For each source the cutout size is the segment bounding box made symmetric
about the source position, with a floor of `min_size`. The build-time schemes
raise that floor so the stamp holds the support they build over: `"classic"`
and `"psf_wings"` floor it at the largest detection-PSF stamp dimension,
`"wren"` at `2*ceil(r_fill) + 1`, each rounded up to an even value.

With `extend_mode="none"` (and with the two post-extraction modes, which only
differ after extraction) pixels outside the source's own segment are zeroed.
A build-time scheme instead replaces the whole stamp with its composite before
normalization. Either way the stamp is then normalized to unit sum and the
pre-normalization total is kept as `template_norm`; a template whose sum is
zero gets `FLAG_SUM_ZERO` instead. The single exception is `"psf_wings"`,
listed in `PRENORMALISED_MODES`, which normalizes inside the scheme (below).
Because templates are unit-normalized, the fitted amplitude is directly the
source flux in the modeled stamp (see the shape-versus-throughput convention in
{doc}`psf` and the `flux_<i>` vs `flux_<i>_total` columns in {doc}`outputs`).

#### Templates.from_cutout_models

{meth}`~mophongo.templates.Templates.from_cutout_models` builds templates
from precomputed source-model cutouts, bypassing segmentation, for use with
externally generated models. By default the cutouts are taken as per-unit-flux
models as given; pass `normalize=True` to renormalize each to unit sum.

### Template build schemes

Extraction on its own truncates each template at its segmentation footprint,
so the template lacks the source's PSF wings and the fitted flux is biased low,
worst for faint sources. `extend_mode` selects how the missing support is
supplied. It takes one of the six values in
`mophongo.templates.EXTEND_MODES` plus the alias `"default"`, which resolves
to `"psf_wings"` (`EXTEND_MODE_ALIASES`).

The modes split into two families. The three listed in `BUILD_TIME_MODES` are
composites implemented in {mod}`mophongo.template_schemes` and built inside
{meth}`~mophongo.templates.Templates.extract_templates`: they replace the
segment-masked data before the unit-sum normalization, so `template_norm`
covers the extended shape. The other two reshape the cutout and therefore run
as a separate pass after extraction, driven by their own methods.

| `extend_mode` | stage | what supplies the missing support |
| --- | --- | --- |
| `"none"` | — | nothing; the template stops at its segment |
| `"psf_wings"` (alias `"default"`) | build | least-squares-scaled PSF outside the segment, smooth faint limit ({func}`~mophongo.template_schemes.composite_psf_wings`) |
| `"wren"` | build | area-weighted ownership plus an SNR-graded blend towards a core-anchored PSF ({func}`~mophongo.template_schemes.composite_wren`) |
| `"classic"` | build | exact segment data plus a least-squares-scaled PSF over the whole stamp ({func}`~mophongo.template_schemes.composite_classic`) |
| `"psf"` | post-extraction | the template convolved with the detection PSF fills its zero pixels ({meth}`~mophongo.templates.Templates.extend_with_psf`) |
| `"psf_model"` | post-extraction | the best-fitting PSF-convolved Gaussian ({meth}`~mophongo.templates.Templates.extend_with_psf_model`) |

```{figure} images/template_psf_halo_anatomy.png
:width: 100%
:alt: A classic-scheme template split into segmentation map, full template, on-segment data, halo outside all segments, and the PSF used to build it.

A build-time composite template from a real F770W run, decomposed into its
parts: inside the source's segmentation footprint the template is the
detection data, and outside it carries a flux-matched PSF halo, supplying the
wing flux that a segment-truncated template would miss. The halo panel nulls
every segment, its own and its neighbours', so only background pixels and the
PSF halo remain.
```

The build-time schemes are self-contained ports kept out of
`mophongo.templates` so that the alternatives can be compared one to one and
either can be adapted or dropped as a unit. Nothing in
`mophongo.template_schemes` imports the fitting, catalog, or pipeline layers:
every entry point takes plain numpy arrays and returns a composite stamp plus
a flat dict of per-source bookkeeping, which
{meth}`~mophongo.templates.Templates.extract_templates` stores on the template
as `extend_info`.

#### psf_wings

The default. The halo amplitude is an ordinary least-squares fit of the PSF to
the data inside the exact segment, $f_{\rm psf} = \sum_S P D / \sum_S P^2$
floored at zero, the same amplitude the `"classic"` scheme uses. It differs
from `"classic"` in two ways.

The faint limit is smooth rather than a switch. One weight
$W = w(S_{\rm seg};\ \texttt{snrlo\_psf})$ is computed for the whole stamp from
the in-segment SNR and the composite is blended towards the bare scaled PSF:
$m = W\,[S D + (1-S) f_{\rm psf} P] + (1-W)\,f_{\rm psf} P$. Outside the
segment the two terms coincide, so the blend is a no-op there and the wings
keep full strength at every SNR; as the SNR goes to zero the whole stamp,
segment included, becomes $f_{\rm psf} P$, with no discontinuity on the way. A
degenerate fit ($f_{\rm psf} \le 0$) falls back to a bare point source.

The composite is also normalized before neighbor-owned pixels are dropped, as
described under *Normalization order* below. `extend_info` records `fpsf`,
`snr_seg`, `w_core`, `template_norm` (the whole-stamp sum before
normalization), `wing_frac_lost` (the fraction zeroed as neighbor-owned), and
`psf_extended`.

#### classic

A port of the IDL `subphot.pro::build_cube` template builder. The composite is
$m = S D + f_{\rm psf} (1-S) P$ with the same least-squares halo amplitude:
fitting the PSF to the data down-weights pixels where the PSF is faint, whereas
the wren scheme uses a flux ratio. When the in-segment SNR falls below
`tmpl_snrlo` the template is *replaced* by the scaled point source rather than
blended towards one, reproducing IDL's hard switch; `force_psf` applies that
replacement unconditionally, matching IDL's `/psf` keyword.

Two properties of the original are preserved deliberately. The support is the
whole stamp including pixels belonging to *other* segments, since $(1-S)$ does
not exclude neighbors — which is why the cutout is floored at the
detection-PSF stamp, beyond which the resampled PSF, and the composite with
it, is identically zero. And there is no positivity clip: negative sky inside
the segment is kept. `extend_info` records `fpsf`, `snr_seg`, `flux_in_seg`,
`added_flux`, and `psf_replaced`.

One deviation from the original is intentional. IDL carried no inverse-variance
map at any stage, so its low-SNR test used one scalar noise per tile; when a
calibrated `detection_weight` is supplied the formal per-pixel noise
$\sqrt{\sum_S 1/\mathrm{ivar}}$ is used instead. The two agree exactly where
the noise is uniform.

```{figure} images/classic_scheme_steps.png
:width: 100%
:alt: Step-by-step construction of a classic composite template, from segment core and scaled PSF wings to the convolved, normalized, aperture-masked stamp.

Construction of a classic composite on a mid-SNR source. The segment core
keeps the raw data with its negatives, the least-squares-scaled PSF supplies
the wings over the whole tile including neighboring segments, and the
convolved stamp is normalized before the circular mask is applied; the growth
curves show the wing flux the composite adds over the bare segment.
```

#### wren

A port of the wren fork's `_extended_composite`. It first partitions the
background globally: each background pixel goes to the segment with the largest
area inside a disk of radius `max_radius_pix` around it, and segment pixels
always keep their own label, so the partition is disjoint by construction
({func}`~mophongo.template_schemes.build_ownership`). This is area-weighted
rather than a distance Voronoi, so a large segment wins more inter-source
territory than a small one; ties go to the lowest label. The map is built once
per extraction, restricted to the labels that can affect a pixel some retained
cutout actually reads.

Each template is then a single radial, SNR-weighted linear blend between the
data and a core-anchored PSF model over the source's own territory,

$$
H = W D + (1 - W) M, \qquad M = A_{\rm src} P,
\qquad A_{\rm src} = \frac{\sum_S \max(D, 0)}{\sum_S P},
$$

with one weight $w_{\rm core} = w(S_{\rm seg};\ 1.5\,\texttt{fit\_snrlo\_psf})$
over the segment and one weight $w(S_k;\ \texttt{wings\_snr\_psf})$ per radial
halo annulus over owned background. The halo ladder is forced monotone
non-increasing outward and seeded at $w_{\rm core}$, so trust in the data never
increases with radius and a faint core caps its own halo. The support is the
segment plus owned background inside the encircled-energy radius; halo weights
are measured out to `max_radius_pix`, which is at least that radius by
construction, so the extra margin buys ownership reach only. Non-finite data
pixels take the model whatever their annulus.

The blend weight itself,
{func}`~mophongo.template_schemes.blend_weight`, is 1 at and above its onset
threshold and a power law of index `blend_p` below it, so the smooth rolloff
reduces to the IDL hard switch as `blend_p` goes to infinity. Both `"wren"` and
`"psf_wings"` use it.

Unlike the other two composites, this one is returned unnormalized and without
a positivity clip: {meth}`~mophongo.templates.Templates.extract_templates`
clips it and then normalizes, in that order, which is what wren does (and which
biases the stored norm high). Without a usable PSF the scheme falls back to raw
data over the larger `max_radius_pix` support and sets `FLAG_EXTEND_FAILED`.
`extend_info` records `snr_seg`, `w_core`, `A_src`, `f_cut`,
`flux_beyond_stamp`, `flux_beyond_aper`, `apcor_from_psf`, `psf_extended`, and
`extend_failed`.

#### Noise for the scheme SNR

Every build-time scheme grades data against a PSF model by SNR, so each needs a
noise estimate. With `detection_weight` all three use the formal per-pixel
noise. Without it:

- `"classic"` and `"psf_wings"` measure a single scalar detection rms once per
  extraction with {func}`~mophongo.template_schemes.detection_rms`, a biweight
  scale estimate ported from IDL astrolib `robust_sigma` — sources included,
  since the biweight downweights them, which is the original intent. The
  scheme's own `rms` field overrides the measurement. It is only needed when
  the faint-source onset (`tmpl_snrlo` or `snrlo_psf`) is above zero, and a
  non-positive result raises `ValueError` rather than silently disabling the
  faint branch.
- `"wren"` measures a sigma-clipped sky sigma over unsegmented pixels
  ({func}`~mophongo.template_schemes.sky_sigma`), overridable with
  `WrenParams.bg_rms`. Every wren weight comes from an SNR, so a missing noise
  estimate would drive every weight to zero and turn every template into a bare
  point source; a non-positive value raises `ValueError`.

Both estimators ignore exactly-zero pixels
({func}`~mophongo.template_schemes.covered_mask`). In a drizzled mosaic those
are the uncovered margin rather than sky, and once that spike at zero holds the
median it collapses the median absolute deviation to zero. Both are also
measured on a regular subsample, since these are global scalars that the
reference codes measured on a single tile.

#### Normalization order

The `"psf_wings"` composite is normalized over the whole stamp *before*
neighbor-owned pixels are zeroed, and that ordering is the point of the scheme.
Wing flux landing on a neighboring segment is counted in the normalizer and
only then dropped, so the template sums to slightly less than one, short by
exactly that flux, and the neighbor's own template fits the light instead of it
being counted twice. Renormalizing afterwards would undo that protection, which
is why `"psf_wings"` is listed in `PRENORMALISED_MODES` and
{meth}`~mophongo.templates.Templates.extract_templates` leaves its stamp alone,
taking `template_norm` from the scheme's own bookkeeping. The ordering
reproduces the effect of IDL's normalize-then-mask step while masking by
segment ownership rather than by a circular aperture. Whether neighbor pixels
are zeroed at all is controlled by `PsfWingsParams.background_only`.

The other two composites never zero anything and are normalized to unit sum by
{meth}`~mophongo.templates.Templates.extract_templates` in the usual way, but
for opposite reasons. `"classic"` keeps its scaled PSF on neighboring segments,
so that flux stays in the template and in its normalizer, and two blended
neighbors can both model it. `"wren"` never claims those pixels: its support is
restricted to the territory the ownership map assigns to the source, so the
partition prevents the double counting up front.

#### Post-extraction passes

The two remaining modes run after extraction as method calls on the container.
The pipeline applies them for `extend_mode="psf_convolution"` (legacy alias
`"psf"`) and `"psf_model"`
respectively, the latter with `mode="model"`. Both accept either a single PSF
array or a spatially varying {class}`mophongo.psf_map.PSFRegionMap`
({doc}`psf_maps`), normalize the PSF to a unit-sum shape for the morphology
operation, and record the native finite-stamp sum only as throughput metadata
(`extension_psf_throughput`).

Both treat deblended children specially through `skip_deblended`. By default
deblended child templates are extended like any other template; setting
`FitConfig(skip_template_extension_for_deblended=True)` makes the pipeline pass
`skip_deblended=True`, leaving children unextended (they are copied through
with `extension_mode = "none"` and
`extension_skip_reason = "is_deblended"`). The build-time schemes have no such
opt-out: they apply to every source.

#### Templates.extend_with_psf

{meth}`~mophongo.templates.Templates.extend_with_psf` fills zero-valued
template pixels with the local high-resolution PSF response.
Nonzero pixels are trusted measured source pixels; the sparse template is
convolved with the local PSF, only the zero pixels receive the convolved
values, and the completed stamp is renormalized to unit sum. The wings
inserted here are therefore the template *convolved* with the detection PSF,
not a scaled PSF model — contrast
{func}`~mophongo.template_schemes.composite_classic`, which pastes a
least-squares-scaled PSF, and
{func}`~mophongo.template_schemes.composite_wren`, which blends towards a
core-anchored one. By default (`background_only=True`) the fill is restricted
to segmentation-background pixels: pixels owned by a different segment keep
their zero value, so blended neighbors model their own light there instead of
receiving this source's extrapolated wings. Per-template `extension_*`
diagnostics record the inserted, blocked, and throughput sums; on this path
`template_norm` is the multiplicative wing boost (IDL's `added_flux`) rather
than a detection-band flux, since the input template is already unit-sum.

#### Templates.extend_with_psf_model

{meth}`~mophongo.templates.Templates.extend_with_psf_model` fits a grid of
PSF-convolved circular Gaussian models to the segment pixels (scale chosen by
least squares on the segment only) and uses the best model to complete the
template: `mode="wings"` keeps the extracted segment pixels and fills only
the outside, `mode="model"` replaces the whole template by the best-fitting
model (what the pipeline's `extend_mode="psf_model"` pass uses). The
completed stamp is renormalized to unit sum, with the best-fit sigma and fit
score recorded as `extension_*` diagnostics.

### Scheme parameters

Each build-time scheme carries its knobs in a dataclass, passed to
{meth}`~mophongo.templates.Templates.extract_templates` through the `wren`,
`classic`, and `psf_wings` keywords. Defaults are used when the keyword is
omitted.

#### PsfWingsParams

{class}`mophongo.template_schemes.PsfWingsParams`.

- `snrlo_psf` (`float`, default `5.0`): in-segment SNR at which the template is
  pure data. Below it the core rolls off smoothly towards the scaled PSF,
  reaching a pure point source as the SNR goes to zero. This replaces IDL's
  hard switch at `tmpl_snrlo`.
- `blend_p` (`float`, default `2.0`): rolloff exponent of that weight.
- `background_only` (`bool`, default `True`): null pixels owned by a
  *different* segment after normalization, so a neighbor's wings are not
  counted twice.
- `rms` (`float | None`, default `None`): fallback scalar detection noise for
  the SNR when no inverse-variance map is supplied. `None` measures it with
  {func}`~mophongo.template_schemes.detection_rms`.

#### ClassicParams

{class}`mophongo.template_schemes.ClassicParams`.

- `tmpl_snrlo` (`float`, default `15.0`): in-segment SNR below which the
  template is replaced by a pure point source; the IDL parameter file sets the
  same value. Zero or negative disables the branch, matching IDL's
  `keyword_set(tmpl_snrlo)` guard.
- `rms` (`float | None`, default `None`): fallback detection-image noise, used
  only when no inverse-variance map is supplied. `None` measures it once per
  extraction with {func}`~mophongo.template_schemes.robust_sigma`, IDL's
  `robust_sigma(ttmpl)`. A calibrated inverse-variance map supersedes it.
- `force_psf` (`bool`, default `False`): IDL's `/psf` keyword — build every
  template as a pure point source.

#### WrenParams

{class}`mophongo.template_schemes.WrenParams`.

- `max_radius_pix` (`float`, default `0.0`): ownership-contest disk radius and
  outer reach of the halo annuli, in detection pixels (wren's `r_fill`). Zero
  or negative derives it from the detection PSF via
  {func}`~mophongo.template_schemes.wren_fill_radius`, which returns
  `max(R_ee, r_aper + kernel_half_width)`: the template must cover the
  measurement aperture plus a convolution margin and never be smaller than the
  encircled-energy cap.
- `psf_ee_radius_pix` (`float | None`, default `None`): hard cap on the
  composite support (wren's `R95`). `None` derives it from the detection PSF at
  `ee_fraction`.
- `aperture_radius_pix` (`float | None`, default `None`): measurement-aperture
  radius on the detection grid, used only for the `flux_beyond_aper` crowding
  bookkeeping. `None` disables that bookkeeping.
- `ee_fraction` (`float`, default `0.95`): encircled-energy fraction defining
  the support cap. The radius is computed against the *stamp* sum, not the true
  total PSF ({func}`~mophongo.template_schemes.psf_ee_radius_pix`).
- `fit_snrlo_psf` (`float`, default `10.0`): the core-weight onset is
  `1.5 * fit_snrlo_psf`, the same SNR at which the IDL code switches to a pure
  point source.
- `wings_snr_psf` (`float`, default `3.0`): per-annulus weight onset.
- `blend_p` (`float`, default `2.0`): rolloff exponent of the blend weight.
- `blend_annulus` (`float`, default `0.15`): halo annulus width in arcsec,
  converted with the cutout WCS; without a WCS it falls back to 4 detection
  pixels.
- `containment` (`float`, default `1.0`): detection-PSF stamp containment used
  by the `flux_beyond_stamp` bookkeeping; 1.0 disables it.
- `bg_rms` (`float | None`, default `None`): explicit detection-image sky rms,
  used when no inverse-variance map is supplied. `None` measures it with
  {func}`~mophongo.template_schemes.sky_sigma`.

#### Selecting a scheme in a run

A pipeline run selects the scheme with `FitConfig.extend_mode`, which defaults
to `"psf_wings"`. The `Pipeline` constructor takes an `extend_mode`
argument that overrides it when given (`extend_templates` is its deprecated
alias); left at `None`, the config field decides ({doc}`pipeline`). The pipeline builds the parameter dataclasses from
`FitConfig` fields: `psf_wings_snrlo`, `psf_wings_blend_p`, `psf_wings_rms`,
and `extend_wings_background_only` for `PsfWingsParams`; `classic_tmpl_snrlo`
and `classic_rms` for `ClassicParams`; `wren_ee_fraction`,
`wren_fit_snrlo_psf`, `wren_wings_snr_psf`, `wren_blend_p`,
`wren_blend_annulus`, and `wren_bg_rms` for `WrenParams`, whose radii it
derives from the configured measurement aperture and the widest matching
kernel. `ClassicParams.force_psf` and `WrenParams.containment` have no config
field and stay at their defaults. Settings belonging to a scheme the run did
not select are dropped from the saved config snapshot.

The detection PSF passed to the build-time schemes is strictly `psfs[0]`,
the band the templates live on; no other index is substituted, because a
lower-resolution PSF would silently produce wrong wings and wrong extension
radii. The detection weight map is resolved on every run but its pixels are
read only when the selected mode uses them.

### Convolution to a measurement band

#### Templates.convolve_templates

```python
Templates.convolve_templates(kernel, inplace=False, psf_lo=None)
```

- `kernel` (`np.ndarray | PSFRegionMap | None`): PSF-matching kernel at the
  template resolution, or a spatial kernel map looked up at each source's sky
  position. Identity (delta-function) kernels skip the convolution and leave
  the template pixels unchanged. `None` behaves like an identity kernel for
  every template.
- `inplace` (`bool`, default `False`): with `True`, the internal template
  list is updated with the convolved (enlarged) templates and returned;
  with `False` the internal list is untouched and a new list of convolved
  templates is returned.
- `psf_lo` (`PSFRegionMap | None`, default `None`): PSF map of the target
  band. When given, each output template records `ee_psf_lo`, the encircled
  energy of that band's PSF stamp at the source position, used downstream to
  convert fitted amplitudes to total fluxes.

Kernels are built from unit-sum PSF shapes (see {doc}`psf`); convolution
enlarges each cutout by the kernel size with even padding and sets
`FLAG_CONVOLVED`.

### Other methods

These container methods are called by the pipeline; each link leads to the
full API documentation.

- {meth}`~mophongo.templates.Templates.add_component` clones a parent
  template and appends a new component template (used for extra fit
  components such as astrometric gradients), returning `None` when the
  component is nearly parallel to the parent.
- {meth}`~mophongo.templates.Templates.apply_template_shifts` applies each
  template's pending `to_shift = (dx, dy)` astrometric offset in place with
  cubic-spline interpolation, always restarting from a cached unshifted copy
  so repeated passes do not compound the interpolation smoothing.
- {meth}`~mophongo.templates.Templates.quick_flux` returns per-source scalar
  least-squares amplitudes over each template footprint, ignoring source
  blending; used as an initial estimate before the full sparse fit.
- {meth}`~mophongo.templates.Templates.predicted_errors` returns per-source
  flux uncertainties that ignore template covariance, stored on
  `tmpl.err_pred` only (the solver error `tmpl.err` is never overwritten).
- {meth}`~mophongo.templates.Templates.prune_outside_weight` removes
  templates lying entirely on non-positive weight and returns the survivors.

## AlignedCutout and WCS scaling

{class}`mophongo.templates.AlignedCutout` is a lighter cutout used
internally where grid alignment matters more than Cutout2D compatibility: the
lower bound and shape of the cutout are forced to multiples of `align`, which
keeps multi-resolution binning exact. It exposes the same
`slices_original`/`slices_cutout` bookkeeping as `Template`, plus block
reduce/replicate array helpers and a flux-conserving `downsample`. As of this
writing `upsample` depends on a position-remapping helper that
`mophongo.templates` does not import, so calls with `factor > 1` fail; use
`as_block_replicated` for the array-only operation.

{func}`mophongo.templates.scale_wcs_pixel` supports these resamplings: it
rescales a WCS to a coarser or finer pixel grid while preserving sky
coordinates, remapping CRPIX and SIP reference pixels as needed.
