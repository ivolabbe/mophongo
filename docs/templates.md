# Template construction

Templates are the morphological basis of the fit: one cutout per source, taken
from the high-resolution detection image inside that source's segmentation
footprint, normalized to unit sum, optionally extended beyond the footprint,
and convolved with a PSF-matching kernel to the resolution of each measurement
band. The fitter ({doc}`fitting`) then solves for one amplitude per template.
This page documents {class}`mophongo.templates.Template`,
{class}`mophongo.templates.Templates`, and the supporting cutout machinery in
`mophongo.templates`. For where template construction sits in the full run, see
{doc}`pipeline`; for kernel construction, see {doc}`psf`.

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
- `ee_tmpl` (`float`): fraction of the normalized source model this template
  retains after construction (below one when wing flux is withheld from
  neighboring segments).
- `to_shift`, `shifted` (`np.ndarray`, length 2): pending and accumulated
  astrometric shifts in `(dx, dy)`.

### Flags

`Template.flag` is a bitmask built from class constants:
`FLAG_VALID` (0x01), `FLAG_CONVOLVED` (0x02), `FLAG_SUM_ZERO` (0x04),
`FLAG_HAS_NAN` (0x08), `FLAG_OUTSIDE_WEIGHT` (0x10), `FLAG_SHIFTED` (0x20),
`FLAG_DEBLENDED` (0x40), and `FLAG_SATURATED` (0x80). Two properties wrap the
provenance bits: `is_deblended` marks templates that come from deblended
catalog children, and `is_saturated` marks saturated/repaired sources, which
the scene builder isolates into their own scene so their PSF wings do not
contaminate neighboring flux solutions.

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
`deblend_nchildren`. Other metadata (fit results, `ee_*` values, shift state)
are freshly initialized on the returned template, as of this writing.

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
  currently unused; call
  {meth}`~mophongo.templates.Templates.extend_with_psf_wings` or
  {meth}`~mophongo.templates.Templates.extend_with_psf_model` explicitly.
- `wcs` (`WCS | None`, default `None`): WCS of the high-resolution image.

#### Templates.extract_templates

```python
tmpls = Templates()
templates = tmpls.extract_templates(hires_image, segmap, positions,
                                    wcs=None, dilate_segmap=2)
```

- `hires_image` (`np.ndarray`): high-resolution detection image.
- `segmap` (`np.ndarray`): segmentation map.
- `positions` (iterable of `(x, y)`): source positions; non-finite or
  out-of-bounds positions, and positions falling on segmentation background,
  are skipped silently.
- `wcs` (`WCS | None`, default `None`): image WCS.
- `dilate_segmap` (`int`, default `2`): disk radius in pixels used to dilate
  each segment into background only before cutting. Detection segmaps often
  capture only the bright core of a point source; without dilation the
  template misses the PSF wings and the fit biases low. Dilation never lets
  segments overlap neighbors. Pass `0` to disable. This default applies only
  to direct calls: pipeline runs pass `FitConfig.template_dilate_segmap`
  instead, which defaults to `0` because wing recovery there is handled by
  template extension (see {doc}`fitting`).

For each source the cutout size is the segment bounding box made symmetric
about the source position, with a floor of `min_size`. Pixels outside the
source's own (dilated) segment are zeroed, and the cutout is normalized to
unit sum; a template whose sum is zero gets `FLAG_SUM_ZERO` instead. Because
templates are unit-normalized, the fitted amplitude is directly the source
flux in the modeled stamp (see the shape-versus-throughput convention in
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

### Template extension

Extraction truncates each template at its segmentation footprint, so the
template lacks the source's PSF wings. Two extension methods complete the
missing support; the pipeline selects between them with the
`extend_templates` option (`"psf_wings"` or `"psf_model"`). The run-level
`RunConfig` defaults to `"psf_wings"`; the `Pipeline` constructor argument
itself defaults to `None`, which leaves templates truncated
({doc}`pipeline`). Both accept either a single
PSF array or a spatially varying {class}`mophongo.psf_map.PSFRegionMap`
({doc}`psf_maps`), normalize the PSF to a unit-sum shape for the morphology
operation, and record the native finite-stamp sum only as throughput metadata
(`extension_psf_throughput`).

Both methods treat deblended children specially through `skip_deblended`.
By default deblended child templates are extended like any other template;
setting `FitConfig(skip_template_extension_for_deblended=True)` makes the
pipeline pass `skip_deblended=True`, leaving children unextended (they are
copied through with `extension_mode = "none"` and
`extension_skip_reason = "is_deblended"`).

#### Templates.extend_with_psf_wings

```python
Templates.extend_with_psf_wings(psf, *, skip_deblended=False,
                                background_only=True, inplace=False)
```

Fills zero-valued template pixels with the local high-resolution PSF response.
Nonzero pixels are trusted measured source pixels; the sparse template is
convolved with the local PSF, only the zero pixels receive the convolved
values, and the completed stamp is renormalized to unit sum.

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
`extension_mode` (`"psf_wings"`), `extension_psf_sum`,
`extension_psf_throughput`, `extension_core_sum`, `extension_pre_norm_sum`,
`extension_filled_sum`, `extension_filled_fraction` (wing flux inserted as a
fraction of the pre-normalization total), and `extension_blocked_sum` (wing
flux withheld because it fell on a neighbor's segment).

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
  `"model"` replaces the whole template by the best-fitting model.
- `skip_deblended` (`bool`, default `False`): as above.
- `inplace` (`bool`, default `False`): replace the internal template list.

The completed stamp is renormalized to unit sum. Diagnostics recorded per
template: `extension_mode`, `extension_sigma_pix` (best-fit Gaussian sigma),
`extension_score` (residual sum of squares on segment pixels),
`extension_segment_fraction` (model flux inside the segment), and
`extension_psf_throughput`.

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
