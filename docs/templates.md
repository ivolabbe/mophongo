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

```python
Template(data, position, size, label=None, copy=True, wcs=None, **kwargs)
```

- `data` (`np.ndarray`): full parent image to cut from.
- `position` (`tuple[float, float]`): source position `(x, y)` in original
  pixel coordinates.
- `size` (`tuple[int, int]`): cutout size `(ny, nx)` in pixels.
- `label` (`int | None`, default `None`): source id, stored as `Template.id`.
- `copy` (`bool`, default `True`): copy the pixel data rather than keeping a
  view into the parent image.
- `wcs` (`astropy.wcs.WCS | None`, default `None`): WCS of the parent image;
  Cutout2D adjusts it to the cutout frame, and the parent WCS is kept as
  `wcs_original`.

The constructor always uses `mode="partial"` with `fill_value=0.0`.

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
  extraction and of the `"psf"` extension pass. Both steps renormalize the
  stamp, so it is 1.0 for any nonzero template and 0.0 for a zero-sum one —
  with one deliberate exception, the `"psf_wings"` build scheme, whose stamp
  is normalized before neighbor-owned pixels are dropped and therefore sums to
  slightly less than one (see [Template build schemes](#template-build-schemes)).
  It stays `NaN` when neither step ran (`extend_with_psf_model` never sets it).
  Wing flux withheld from a neighboring segment is reported by
  `extension_blocked_sum` on the `"psf"` path and by `wing_frac_lost` in
  `extend_info` on the `"psf_wings"` path. Diagnostic only: the fitted
  amplitude does not scale with it.
- `template_norm` (`float`): the stamp sum divided out by the unit-sum
  normalization, so `template_norm * data` reconstructs the composite and the
  implied detection-band flux stays known. Initialized to 0.0. On the `"psf"`
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

### Template.from_stamp

```python
Template.from_stamp(data, origin, input_position_original, shape_original,
                    *, wcs=None, label=None, parent_image=None)
```

Class method that rebuilds a template from serialized stamp pixels plus
geometry (the inverse of the stamp output described in {doc}`outputs`).

- `data` (`np.ndarray`): stamp pixels, shape `(ny, nx)`.
- `origin` (`tuple[int, int]`): original-grid pixel `(x, y)` of `data[0, 0]`;
  may be negative for edge-padded cutouts.
- `input_position_original` (`tuple[float, float]`): source position `(x, y)`
  on the original grid.
- `shape_original` (`tuple[int, int]`): shape of the full parent image, which
  sets the clipped `slices_original`/`slices_cutout` pair.
- `wcs` (`WCS | None`, default `None`): WCS of the full parent image.
- `label` (`int | None`, default `None`): source id.
- `parent_image` (`np.ndarray | None`, default `None`): optional zero array of
  `shape_original` reused across calls to avoid per-source allocations.

### Template.pad

```python
Template.pad(padding, original_shape, *, image=None, inplace=False)
```

Returns a new template enlarged by `padding`, keeping original-image
coordinates consistent.

- `padding` (`tuple[int, int]`): total extra `(ny, nx)` pixels; halved per
  side, so odd values are rounded down to keep the padding even.
- `original_shape` (`tuple[int, int]`): accepted for the call signature; the
  parent shape actually used is the stored `shape_input`.
- `image` (`np.ndarray | None`, default `None`): parent array to cut from; a
  zero array of the original shape is created when omitted.
- `inplace` (`bool`, default `False`): when `True`, the current instance is
  updated to the padded geometry as well.

### Template.convolve_cutout

```python
Template.convolve_cutout(kernel, *, parent_image=None, preserve_dtype=True)
```

Convolves the template with a centered kernel and returns a new
`Template` whose geometry is already enlarged to hold the full convolution
result (`mode="full"`), with even padding on both sides. The source position,
provenance attributes, and `ee_*` metadata are propagated, and
`FLAG_CONVOLVED` is set.

- `kernel` (`np.ndarray`): 2-D centered convolution kernel.
- `parent_image` (`np.ndarray | None`, default `None`): reference to the full
  parent image; a zero dummy of the original shape is created when omitted.
  It is never copied.
- `preserve_dtype` (`bool`, default `True`): cast the result back to the input
  `data` dtype instead of keeping the float64 that FFT convolution returns.

### Template.downsample

```python
Template.downsample(k, image=None, wcs_lo=None)
```

Flux-conserving `k`-fold binning aligned to the global high-resolution grid,
used on the "downsample" multi-resolution path.

- `k` (`int`): integer bin factor; `k=1` returns a deep copy.
- `image` (`np.ndarray | None`, default `None`): low-resolution parent array;
  a zero array of the binned original shape is created when omitted.
- `wcs_lo` (`WCS | None`, default `None`): WCS of the low-resolution grid.

Binning is exact only when the cutout origin and shape are divisible by `k`.
For a misaligned origin the routine bins the largest `k`-aligned block, logs a
warning, and leaves the trailing low-resolution row/column zero-filled, losing
that flux; the warning recommends the `upsample` multi-resolution method for
exact alignment.

### Template.project_to_block_replicated_grid

```python
Template.project_to_block_replicated_grid(factor, *, parent_image=None,
                                          preserve_dtype=True)
```

Projects the template onto the globally aligned block-replicated grid used by
the upsampled multi-resolution fitting path (see {doc}`pipeline`). On that
path each native low-resolution pixel is represented as a constant
`factor x factor` block; a template fitted against such an image must live in
the same pixel basis, or the residual depends on where the source sits inside
the native pixel. The method integrates the template over global native-pixel
blocks and replicates the block means back onto the high-resolution grid,
enlarging the cutout to whole native-pixel boundaries.

- `factor` (`int`): block-replication factor; `factor=1` returns a deep copy.
- `parent_image` (`np.ndarray | None`, default `None`): full parent image
  reference; a zero dummy is created when omitted.
- `preserve_dtype` (`bool`, default `True`): cast the result back to the input
  dtype.

The projected template keeps `input_position_original`, `position_original`,
the recomputed cutout-frame positions, `flag`, `deblend_parent_label`, and
`deblend_nchildren`. Other metadata (fit results, `ee_*` values, shift state,
`id_scene`) are freshly initialized on the returned template, as of this
writing.

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
`inplace=True` (see the in-place caveat under that method).

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

```python
Templates.from_cutout_models(cutouts, positions, ids, *,
                             original_shape, wcs=None, normalize=False)
```

Builds templates from precomputed source-model cutouts, bypassing
segmentation, for use with externally generated models.

- `cutouts` (iterable of `np.ndarray`): per-source model stamps; each stamp's
  shape must match the template cutout built at its position.
- `positions` (iterable of `(x, y)`): source positions.
- `ids` (iterable of `int`): source labels.
- `original_shape` (`tuple[int, int]`, keyword-only): shape of the parent
  grid the cutouts refer to.
- `wcs` (`WCS | None`, default `None`): parent-grid WCS.
- `normalize` (`bool`, default `False`): normalize each cutout to unit sum.
  When `False`, cutouts are interpreted as per-unit-flux models as given.

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

```python
Templates.extend_with_psf(psf, *, skip_deblended=False,
                          background_only=True, inplace=False)
```

Fills zero-valued template pixels with the local high-resolution PSF response.
Nonzero pixels are trusted measured source pixels; the sparse template is
convolved with the local PSF, only the zero pixels receive the convolved
values, and the completed stamp is renormalized to unit sum. The wings
inserted here are therefore the template *convolved* with the detection PSF,
not a scaled PSF model — contrast
{func}`~mophongo.template_schemes.composite_classic`, which pastes a
least-squares-scaled PSF, and
{func}`~mophongo.template_schemes.composite_wren`, which blends towards a
core-anchored one.

- `psf` (`np.ndarray | PSFRegionMap`): high-resolution PSF or spatial PSF map.
- `skip_deblended` (`bool`, default `False`): copy deblended-child templates
  through unextended.
- `background_only` (`bool`, default `True`): restrict the fill to background
  pixels of the segmentation map stored by `extract_templates`. Pixels owned
  by a different segment keep their zero value, so blended neighbors model
  their own light there instead of receiving this source's extrapolated
  wings; pixels outside the image footprint count as background. When no
  segmap was recorded (prebuilt templates), all zero pixels are filled.
- `inplace` (`bool`, default `False`): replace the internal template list.

Returns the list of completed templates. Each records diagnostics:
`extension_mode` (`"psf"`), `extension_psf_sum`,
`extension_psf_throughput`, `extension_core_sum`, `extension_pre_norm_sum`,
`extension_filled_sum`, `extension_filled_fraction` (wing flux inserted as a
fraction of the pre-normalization total), and `extension_blocked_sum` (wing
flux withheld because it fell on a neighbor's segment).

`template_norm` means something different on this path than at extraction:
the input template is already unit-sum, so the recorded pre-normalization sum
is the multiplicative wing boost (IDL's `added_flux`) rather than the
detection-band flux inside the segment.

#### Templates.extend_with_psf_model

```python
Templates.extend_with_psf_model(psf, *,
    gaussian_sigmas=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    target_shape=None, mode="wings", skip_deblended=False, inplace=False)
```

Fits a grid of PSF-convolved circular Gaussian models to the segment pixels
(scale chosen by least squares on the segment only) and uses the best model to
complete the template.

- `psf` (`np.ndarray | PSFRegionMap`): high-resolution PSF or spatial PSF map.
- `gaussian_sigmas` (sequence of `float`, default
  `(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)`): Gaussian widths in pixels to try;
  `0.0` is the bare PSF. Must be non-empty and non-negative.
- `target_shape` (`tuple[int, int] | None`, default `None`): minimum output
  stamp shape; the stamp is always at least as large as the template and the
  PSF, and is padded to even dimensions.
- `mode` (`str`, default `"wings"`): `"wings"` keeps the extracted segment
  pixels and fills only the pixels outside the segment with the scaled model;
  `"model"` replaces the whole template by the best-fitting model. The
  pipeline's `extend_mode="psf_model"` pass calls this method with
  `mode="model"`.
- `skip_deblended` (`bool`, default `False`): as above.
- `inplace` (`bool`, default `False`): replace the internal template list.

The completed stamp is renormalized to unit sum. Diagnostics recorded per
template: `extension_mode` (the `mode` value, `"wings"` or `"model"` — not
`"psf_model"`), `extension_sigma_pix` (best-fit Gaussian sigma),
`extension_score` (residual sum of squares on segment pixels),
`extension_segment_fraction` (model flux inside the segment), and
`extension_psf_throughput`.

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
to `"psf_wings"`. The `Pipeline` constructor keeps an `extend_templates`
argument that overrides it when given; left at `None`, the config field
decides ({doc}`pipeline`). The pipeline builds the parameter dataclasses from
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
- `inplace` (`bool`, default `False`): with `True`, non-identity kernels
  still produce new enlarged templates but the internal list keeps the
  original objects; as of this writing the in-place path returns the internal
  list without substituting the convolved results, so prefer
  `inplace=False` and use the returned list.
- `psf_lo` (`PSFRegionMap | None`, default `None`): PSF map of the target
  band. When given, each output template records `ee_psf_lo`, the encircled
  energy of that band's PSF stamp at the source position, used downstream to
  convert fitted amplitudes to total fluxes.

Kernels are built from unit-sum PSF shapes (see {doc}`psf`); convolution
enlarges each cutout by the kernel size with even padding and sets
`FLAG_CONVOLVED`.

### Other methods

#### Templates.add_component

```python
Templates.add_component(parent, data, component, **kwargs)
```

Clones `parent` and appends a new component template (used for extra fit
components such as astrometric gradients).

- `parent` (`Template`): template providing the spatial metadata.
- `data` (`np.ndarray`): pixel data for the component; must match
  `parent.data` in shape.
- `component` (`str`): informational tag stored on the clone.
- `**kwargs`: additional attributes set on the clone.

Returns the new template, or `None` when the component is nearly parallel to
the parent (normalized inner product above 0.999) and would make the normal
equations degenerate.

#### Templates.apply_template_shifts

```python
Templates.apply_template_shifts(templates)
```

Static method; applies each template's pending `to_shift = (dx, dy)` offset in
place with cubic-spline interpolation. The sign convention: `(dx, dy)` is the
image-to-template correction predicted by astrometry, so the template is
shifted by `(-dx, -dy)` internally. Shifts below 0.01 pixel are skipped.
Interpolation always starts from a cached unshifted copy of the data with the
accumulated total shift, so repeated astrometric passes do not compound the
interpolation smoothing. Applied templates accumulate into `shifted`, reset
`to_shift` to zero, and gain `FLAG_SHIFTED`.

- `templates` (sequence of `Template`): templates to shift.

#### Templates.quick_flux

```python
Templates.quick_flux(templates, image)
```

Static method; returns per-source scalar least-squares amplitudes
`sum(image * t) / sum(t**2)` over each template footprint, ignoring source
blending. Used as an initial estimate before the full sparse fit. Stores the
value on `tmpl.flux` and returns the array.

- `templates` (`list[Template]`): templates to measure.
- `image` (`np.ndarray`): image in original coordinates.

#### Templates.predicted_errors

```python
Templates.predicted_errors(templates, weights)
```

Static method; returns per-source flux uncertainties
`1 / sqrt(sum(w * t**2))` that ignore template covariance. Weights are
inverse variance. The prediction is stored on `tmpl.err_pred` only; the
solver error `tmpl.err` is never overwritten. Templates with zero total
weight get `FLAG_SUM_ZERO` and an infinite predicted error.

- `templates` (`list[Template]`): templates to evaluate.
- `weights` (`np.ndarray`): inverse-variance map in original coordinates.

#### Templates.prune_outside_weight

```python
Templates.prune_outside_weight(weight, rtol=1e-8)
```

Removes templates whose weighted norm `sum(d * w * d)` over their footprint
falls below `rtol` times the median norm — sources lying entirely on
non-positive weight. Stores the norm on `tmpl.wnorm`, replaces the internal
list, and returns the survivors.

- `weight` (`np.ndarray`): inverse-variance map aligned with the original
  image shape.
- `rtol` (`float`, default `1e-8`): relative tolerance on the median weighted
  norm.

## AlignedCutout and WCS scaling

{class}`mophongo.templates.AlignedCutout` is a lighter cutout used where
grid alignment matters more than Cutout2D compatibility: the lower bound and
shape of the cutout are forced to multiples of `align`, which keeps
multi-resolution binning exact.

```python
AlignedCutout(data, position, size, *, align=1, copy=False,
              fill_value=0.0, wcs=None)
```

- `data` (`np.ndarray`): 2-D parent array.
- `position` (`tuple[float, float]`): `(x, y)` pixel-center position.
- `size` (`tuple[int, int] | int`): minimum `(ny, nx)` size; the actual cutout
  may be enlarged to satisfy alignment.
- `align` (`int`, default `1`): per-axis alignment; the lower bound and the
  shape become multiples of this value.
- `copy` (`bool`, default `False`): copy pixels instead of viewing the parent
  (a copy is forced when the cutout extends outside the parent).
- `fill_value` (`float`, default `0.0`): value for out-of-image pixels.
- `wcs` (`WCS | None`, default `None`): parent WCS; adjusted to the cutout,
  including SIP terms when present.

It exposes the same `slices_original`/`slices_cutout` bookkeeping as
`Template`, plus array helpers `as_block_reduced(factor, func=np.sum)` and
`as_block_replicated(factor, conserve_sum=True)`, and geometry-aware
`downsample(factor)` (flux-conserving; requires origin and shape divisible by
`factor`) and `upsample(factor, conserve_sum=True)`. As of this writing
`upsample` depends on a position-remapping helper that `mophongo.templates`
does not import, so calls with `factor > 1` fail; use `as_block_replicated`
for the array-only operation.

{func}`mophongo.templates.scale_wcs_pixel` supports these resamplings:

```python
scale_wcs_pixel(wcs, pixel_scale_factor, new_shape=None)
```

- `wcs` (`WCS | None`): input WCS; `None` passes through.
- `pixel_scale_factor` (`float`): factor greater than one enlarges pixels
  (downsampling), smaller than one shrinks them (upsampling). Sky coordinates
  are preserved by scaling the CD/CDELT matrix and remapping CRPIX (and SIP
  reference pixels when present).
- `new_shape` (`tuple[int, int] | None`, default `None`): pixel shape recorded
  on the returned WCS.
