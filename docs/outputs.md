# Outputs

A config-driven run ({doc}`pipeline`) writes its products to `out_dir`, each
file prefixed with the run `name` from the JSON config:

| File | Written by | Contents |
| --- | --- | --- |
| `<name>_fit_table.fits` | {meth}`~mophongo.pipeline.Pipeline.write_outputs` | the photometry catalog (see below) |
| `<name>_residual.fits` | `write_outputs` | data minus model on the reference grid |
| `<name>_templates.fits` | `write_outputs` | per-template fit state: amplitudes, applied shifts, scene membership |
| `<name>_stamps.fits` | {meth}`~mophongo.pipeline.Pipeline.write_stamps` (when `save_stamps` is set) | per-source template stamps and fit metadata |
| `<name>_scene_catalog.csv` | `write_outputs` | one row per fitted scene |
| `<name>_scene_<id>.png` | `write_outputs` (when `scene_plots` is set) | per-scene diagnostic figures |
| `<name>_psf_hi.geojson`, `<name>_psf_lo.geojson`, `<name>_kernel.geojson` | `build_psfs` / `build_kernels` | cached PSF and kernel region maps ({doc}`psf_maps`) |
| `<name>.json` | {meth}`~mophongo.pipeline.Pipeline.save_config`, called by `run` | fully explicit snapshot of the executed config |
| `<name>.log` | {meth}`~mophongo.pipeline.Pipeline.run_all` | full log of the run |

In-memory runs return the same products directly:
{meth}`mophongo.pipeline.Pipeline.run` returns `(table, residuals)` and leaves
`table`, `residuals`, and `model_images` on the instance.

Throughout, `<i>` is the fitted image index, 1-based: image 0 is the
high-resolution reference, images 1 and up are the fitted bands. A
config-driven run fits one band, so its columns end in `_1`.

## The fit table

`<name>_fit_table.fits` (in memory: `Pipeline.table`) holds one row per input
catalog source that survived preprocessing (footprint and trial-region cuts).
Fluxes are in the pixel units of the fitted image — check the mosaic's
`BUNIT` header keyword for the physical conversion (e.g. a mosaic in units
of 10 nJy per pixel yields fluxes in the same 10 nJy unit). Sources for
which no template was fitted keep `FitConfig.bad_value` (default `NaN`) in
every measurement column.

### Identity and provenance columns

`id`, `x`, `y`
: Source id and position in pixels of the high-resolution reference image,
  copied from the input catalog. Positions are 0-indexed (numpy/`origin=0`
  convention); catalogs written by 1-indexed FITS tools are offset by one
  pixel unless converted.

`is_deblended`, `deblend_parent_label`, `deblend_nchildren`
: Deblending provenance, copied through when present in the input catalog.

`FLAG_SATURATED_*`
: Saturation flags from preprocessing ({doc}`preprocessing`), copied through
  when present. Flagged sources are isolated into their own scenes.

When `FitConfig.aperture_catalog` names a catalog column, that column is also
copied through.

### Flux and error columns

`flux_<i>`
: Raw fitted template amplitude. Templates are normalized to unit sum and
  convolved with unit-sum matching kernels, so this is the flux contained in
  the modeled PSF support, without any correction for flux outside the finite
  PSF stamp. Under the default `"psf_wings"` build scheme a template sums to
  slightly less than one, because the wing flux that fell on a neighbouring
  segment is dropped after the normalization and fitted by that neighbour's
  own template; the amplitude is not rescaled for it (the blanked pixels
  carry flux but almost no fitting weight — see `ee_tmpl` below). Each
  amplitude is written to the catalog row carrying the template's `id`;
  catalog deblend children are rows of their own, so a child keeps its own
  flux instead of having it folded into its parent.

`err_<i>`
: 1-sigma uncertainty on `flux_<i>` from the solver:
  `sqrt(diag(A^-1))` of the scene's normal matrix (solved in whitened
  variables for conditioning, then unwhitened), so it includes the
  covariance penalty from overlapping neighbors. When a scene also solves for
  astrometric shifts, the shift block is marginalized out through its Schur
  complement first.

`err_pred_<i>`
: Predicted uncertainty ignoring template covariance,
  `1 / sqrt(sum(w * T^2))` over the template footprint with `w` the inverse
  variance ({meth}`mophongo.templates.Templates.predicted_errors`). Comparing
  `err_<i>` to `err_pred_<i>` measures how much blending inflates the error.

`throughput_<i>`
: Filter-level PSF throughput: the mean finite-support stamp sum of the
  band's PSFs, i.e. the realized encircled energy of the square PSF stamp.
  One scalar per band; used as the fallback encircled energy for templates
  without a per-source value.

`flux_<i>_total`, `err_<i>_total`, `err_pred_<i>_total`
: Total-flux versions of the three columns above: each template's amplitude
  (and errors) divided by `ee_psf_lo`, the encircled energy of the
  low-resolution PSF stamp at that source's position. Templates whose
  `ee_psf_lo` is unset fall back to `throughput_<i>`. This is the number to
  use as a total flux; `flux_<i>` deliberately keeps the uncorrected amplitude
  (see the shape-vs-throughput convention in {doc}`psf`).

`scene_<i>`
: Id of the scene the source was fitted in, taken from the scene objects
  themselves rather than from the partitioning labels, so it names the scene
  that actually held the fitted template. Sources with no template keep `-1`,
  and when several templates book their flux under one catalog id, the first
  of them names the scene. The scene objects are not persisted, so this
  column (with `id_scene` in the stamps and per-template files) is how scene
  membership survives a run: without it,
  {meth}`~mophongo.pipeline.Pipeline.load_fit` could only recover it by
  refitting. Scene ids label one run's partition and are not stable across
  runs with different scene settings.

### Aperture columns

Aperture photometry on the model-plus-residual image is always attempted; the
aperture radius comes from `FitConfig.aperture_diam` (a fixed diameter in
`aperture_units`, one value per band as an array, or `None` for a default
aperture radius of 1.5 times the band PSF FWHM).

`aper_<i>`
: The aperture diameter used, in arcsec. Only written when
  `FitConfig.aperture_diam` is set.

`ap_flux_<i>`
: Raw aperture sum on model + residual at the source position.

`ap_model_<i>`
: Aperture sum on the source's own model only.

`ap_corr_<i>`
: Aperture-to-total correction: total flux of the convolved template divided
  by its flux inside the aperture, i.e. `1 / EE_template(r)`.

`ap_flux_corr_<i>`
: `ap_flux_<i> * ap_corr_<i>`, the aperture flux corrected to total.

### Header metadata

The table's FITS header records the realized PSF encircled energies used for
the total-flux correction, per band `<i>`:

`EEBOX<i>`
: Realized encircled energy of the full square PSF stamp — the value written
  as `throughput_<i>`.

`PSFSZ<i>`
: Delivered PSF stamp side in arcsec.

`EECIRC<i>`, `RCIRC<i>`
: Encircled energy within the stamp's inscribed circle, and that circle's
  radius in arcsec.

All values are measured on the PSF stamps the fit actually used, after
drizzling and any configured broadening.

## Residual images

`Pipeline.run()` produces one residual image per fitted band
(`Pipeline.residuals`, index `<i> - 1`): the fitted image minus the best-fit
model of all templates. On the default multi-resolution path
(`multi_resolution_method="upsample"`) the fitted image was block-replicated
onto the reference grid, so the residual shares the reference grid and WCS;
`write_outputs` accordingly writes `<name>_residual.fits` with the
high-resolution science header. The matching model images are in
`Pipeline.model_images` (`images[i] - residual`) but are not written to disk.

## Scene catalog

`<name>_scene_catalog.csv` has one row per fitted scene ({doc}`fitting`):
`id` (scene id), `n_templates`, `is_bright` (number of bright anchor sources),
and `ra`, `dec` of the scene center, plus a URL column linking each position
to an external sky viewer. With `scene_plots` enabled, each scene also gets a
`<name>_scene_<id>.png` diagnostic figure.

Scene ids are labels of one run's fit partition, not stable source
identifiers: membership depends on the scene-construction settings
(`scene_coupling_thresh`, `scene_max_size`, `scene_max_merge_radius`), so
scene-id-keyed products from runs with different settings are not comparable
by id.

## The per-template fit table

`<name>_templates.fits` is one row per fitted template of the first fitted
band, holding what a deterministic rebuild of the templates cannot re-derive:
`id` and `id_parent`, the reference-grid position `x`, `y`, the applied
astrometric shift `dx`, `dy`, the fitted `flux` and `err`, and `id_scene`.
The fit table aggregates per catalog source, so a source split into several
fitted components appears there once; this file keeps the components
separate. {meth}`~mophongo.pipeline.Pipeline.load_outputs` reads it into
`Pipeline.template_table`, and `load_fit` uses it to reapply the exact
amplitudes and shifts when it has to regenerate the stamps.

## The stamps file

`<name>_stamps.fits` stores every fitted template at its native, per-source
size, plus the metadata needed to restore the post-run state with
{meth}`~mophongo.pipeline.Pipeline.load_fit`. Data that already have their own
files are not duplicated: PSFs stay in the cached geojson region maps and each
source carries only a lookup key into them; the run configuration stays in its
JSON file.

### Layout

- **Primary HDU** — header only:
  - `NSRC`: number of `SOURCES` rows
  - `IFILT`: fitted image index the `*_lo` columns belong to
  - `RUNNAME`: run name, pointing at the JSON/geojson files of the run
  - `NX_HI`, `NY_HI`: reference-grid width and height in pixels
  - `NX_LO`, `NY_LO`: fitting-grid width and height in pixels

  `load_fit` compares the grid shapes against the loaded images and rejects a
  stale stamps file.

- **`SOURCES`** — one binary-table row per fitted template:

  `id`, `x`, `y`
  : Source id and reference-grid position.

  `flux`, `err`
  : Fitted amplitude and solver uncertainty of this template.

  `tmpl_hi`, `ny_hi`, `nx_hi`, `x0_hi`, `y0_hi`, `xs_hi`, `ys_hi`
  : High-resolution template pixels as a flattened variable-length array
    (reshape to `(ny_hi, nx_hi)`), the original-grid pixel coordinates of
    `data[0, 0]` (`x0`, `y0`; may be negative for cutouts padded past the
    image edge), and the source position on that grid (`xs`, `ys`). Empty
    (`ny_hi = nx_hi = 0`, `x0 = y0 = -1`) when the source has no
    high-resolution template.

  `tmpl_lo`, `ny_lo`, `nx_lo`, `x0_lo`, `y0_lo`, `xs_lo`, `ys_lo`
  : The same for the convolved template on the fitting grid.

  `key_psf_hi`, `key_psf_lo`
  : `psf_key` into the band's cached PSF region map
    (`<name>_psf_hi.geojson` / `<name>_psf_lo.geojson`); 0 for a static PSF
    array, -1 when the band has no PSF.

  `flag_hi`, `flag`
  : Bitwise diagnostic flags of the high-resolution and convolved template
    (bit values below).

  `id_parent`, `id_scene`
  : Catalog id the template's flux is booked under, and the scene it belongs
    to. `id_parent` is a placeholder as the pipeline stands: it equals `id`
    for every template it builds. `id_scene` is the real scene id, stamped
    onto each template by {func}`mophongo.scene.generate_scenes`, and matches
    the `scene_<i>` column of the fit table.

  `ee_psf_lo`
  : Encircled energy of the low-resolution PSF stamp at this position — the
    per-source divisor behind `flux_<i>_total`.

  `ee_tmpl`
  : Fraction of the normalized source model this template retains
    (`sum(data)` after construction). Below one when wing flux was handed to
    a neighbor. Diagnostic only; it is not applied to the fluxes, because the
    blanked pixels carry flux but almost no fitting weight.

  `err_pred`
  : Predicted covariance-free uncertainty (as `err_pred_<i>`, per template).

  `shift_x`, `shift_y`
  : Astrometric shift applied to the template during the fit, in fitting-grid
    pixels; 0 when astrometric fitting was disabled.

{meth}`~mophongo.pipeline.Pipeline.read_stamps` reads the file back as a list
of per-source dicts with the `tmpl_hi` / `tmpl_lo` arrays restored to 2D.

### Template flag bits

`flag` and `flag_hi` are bitwise ORs of {class}`mophongo.templates.Template`
constants:

| Bit | Value | Name | Meaning |
| --- | --- | --- | --- |
| 0 | 1 | `FLAG_VALID` | template was constructed (set on every template) |
| 1 | 2 | `FLAG_CONVOLVED` | template has been convolved with a matching kernel |
| 2 | 4 | `FLAG_SUM_ZERO` | template sum (or its fitting weight) is zero; its flux carries no information |
| 3 | 8 | `FLAG_HAS_NAN` | template contained NaN values |
| 4 | 16 | `FLAG_OUTSIDE_WEIGHT` | template footprint falls outside the positive weight map |
| 5 | 32 | `FLAG_SHIFTED` | an astrometric shift was applied |
| 6 | 64 | `FLAG_DEBLENDED` | source is a catalog deblend child (provenance) |
| 7 | 128 | `FLAG_SATURATED` | source carried a `FLAG_SATURATED_*` catalog flag (provenance) |
| 8 | 256 | `FLAG_PSF_EXTENDED` | the build scheme blended PSF wings into the template |
| 9 | 512 | `FLAG_EXTEND_FAILED` | extension was attempted but the PSF was unusable |

`FLAG_PSF_EXTENDED` and `FLAG_EXTEND_FAILED` are set by the build-time
schemes only (`"psf_wings"`, `"wren"`, `"classic"`). A template is flagged as
PSF-extended when its blend weight fell below one, that is, when the scheme
mixed a scaled PSF into the data rather than keeping the data alone; the
post-extraction modes `"psf"` and `"psf_model"` set neither flag.

`FLAG_HAS_NAN` and `FLAG_OUTSIDE_WEIGHT` are declared but nothing in the
current code sets them: templates with no useful overlap with the weight map
are dropped by `Templates.prune_outside_weight` before the fit rather than
flagged, and never reach the stamps file at all.

As of this writing these flags live in the stamps file (and on the in-memory
`Template` objects), not as a column of the fit table.

## Method reference

### `Pipeline.write_outputs()`

No parameters. Requires a completed {meth}`~mophongo.pipeline.Pipeline.run`
on a config-driven pipeline. Writes the residual FITS, fit table,
per-template fit table, scene catalog and plots, and (when the config's
`save_stamps` is true) the stamps file. Returns `self`.

### `Pipeline.write_stamps(path=None, *, ifilt=1)`

`path` : `str | os.PathLike | None`, default `None`
: Output file. Defaults to `<out_dir>/<name>_stamps.fits` for config-driven
  runs; required otherwise.

`ifilt` : `int`, default `1`
: Fitted image index (1-based). Must be between 1 and `len(images) - 1`.

Returns the `Path` of the written file.

### `Pipeline.read_stamps(path)` (static)

`path` : `str | os.PathLike`
: A stamps file written by `write_stamps`.

Returns a list of dicts, one per source, with the scalar `SOURCES` columns
plus `tmpl_hi` / `tmpl_lo` as 2D arrays. PSF stamps are looked up separately
through `key_psf_hi` / `key_psf_lo` in the cached PSF region maps.

### `Pipeline.load_fit(ifilt=1)`

`ifilt` : `int`, default `1`
: Fitted image index (1-based).

Counterpart of `load_data` for the post-run state: loads the data when
needed, reads the written products through
{meth}`~mophongo.pipeline.Pipeline.load_outputs`, rebuilds the fitted
templates from `<name>_stamps.fits`, and recreates the derived state (grid
upsampling, model image) so the instance matches a completed `run()` without
refitting — the diagnostic methods in {doc}`diagnostics` then work as usual.
Requires a config-driven pipeline; returns `self`.

Not restored: the scene objects (`all_scenes` stays empty; membership
survives as `id_scene` and the `scene_<i>` column), and the pre-extension
template pixels (`templates_extracted` then equals `templates_extended`).
When the stamps file is missing, the templates are regenerated through the
same code path `run()` uses and the stamps file is written back. The fitted
amplitudes, uncertainties and shifts are then reapplied per component from
`<name>_templates.fits`; for runs written before that file existed they come
from the fit table instead, and the rebuild reproduces the fitted templates
exactly only when the run applied no astrometric shifts.

## Provenance

Products record what produced them, so stale caches fail loudly instead of
silently mixing configurations:

- **PSF and kernel maps** (`<name>_*.geojson`) carry provenance columns that
  round-trip through the geojson: the PSF maps record the ePSF filename
  pattern, stamp size, and Gaussian broadening FWHM; the kernel map records
  the matching method, regularization value, and stamp size. `build_psfs`
  reuses cached PSF maps only when all their recorded fields match the
  current config; `build_kernels` reuses a cached kernel map only when its
  recorded method matches the requested one. Otherwise the map is rebuilt.
- **The config snapshot** (`<name>.json`) is written by `run()` before it
  fits, with every `RunConfig` field and every used `FitConfig` setting at
  its resolved value, so the run stays reproducible when code defaults change
  later. Settings belonging to template build schemes the run did not select
  are left out. It is a valid input config: `Pipeline.from_config(out_dir)`
  reopens the run from it.
- **The fit table header** carries the `EEBOX` / `PSFSZ` / `EECIRC` / `RCIRC`
  encircled-energy cards described above, keeping the aperture-correction
  reference with the catalog that used it.
- **The stamps header** names the run (`RUNNAME`) and records the grid shapes
  that `load_fit` validates against the loaded images.
- **The run log** (`<name>.log`, written by `run_all` or the
  `Pipeline.log_run` context manager) captures everything the run emits —
  logging records, warnings, and progress output — with a timestamped banner
  recording the Python version and platform. Successive runs append.
- **The residual FITS** inherits the full header, including the WCS, of the
  high-resolution science image.
