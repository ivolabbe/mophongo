# Outputs

A config-driven run ({doc}`pipeline`) writes its products to `out_dir`, each
file prefixed with the run `name` from the JSON config:

| File | Written by | Contents |
| --- | --- | --- |
| `<name>_fit_table.fits` | {meth}`~mophongo.pipeline.Pipeline.write_outputs` | the photometry catalog (see below) |
| `<name>_residual.fits` | `write_outputs` | data minus model on the reference grid |
| `<name>_templates.fits` | `write_outputs` | per-template fit state: amplitudes, applied shifts, scene membership |
| `<name>_stamps.fits` | {meth}`~mophongo.pipeline.Pipeline.write_stamps` (when `save_stamps` is set) | per-source template stamps and fit metadata |
| `<name>_scene_catalog.csv` | `write_outputs` | one row per fitted scene: position and total astrometric shift |
| `scenes/<name>_scene_<id>.png` | `write_outputs` (when `scene_plots` is set) | per-scene diagnostic figures |
| `<name>_shift_field.png` | `write_outputs` (when astrometry was solved) | map of the fitted astrometric shift field |
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

When `FitConfig.phot.aperture_catalog` names a catalog column, that column is also
copied through.

### Flux and error columns

`flux_<i>`
: Raw fitted template amplitude: the flux inside the modeled PSF support,
  with no correction for flux outside the finite PSF stamp. Templates and
  matching kernels are unit-sum, which puts the amplitude on that scale; under
  the default `"psf_wings"` scheme a template sums to slightly less than one
  and the amplitude is not rescaled for it — the *Normalization order* section
  of {doc}`templates` gives the reason. Each amplitude is written to the
  catalog row carrying the template's `id`; catalog deblend children are rows
  of their own, so a child keeps its own flux instead of having it folded into
  its parent.

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
  (see the shape-vs-throughput convention in {doc}`psf`). The PSF encircled
  energy is the right divisor for a point source; a source resolved at the
  band's resolution loses a little more light past the same support, by the
  factor `psfcor_<i>` below (*Encircled energy of a template* in
  {doc}`templates`).

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

`flag_astrom_<i>`
: The scene-level astrometry verdict, inherited by every source fitted in
  that scene: `0` when the scene's shifts converged, `1` when it was still
  moving after `FitConfig.fit_astrometry_niter` passes, `-1` for sources with
  no template. Convergence is a property of the scene — all its members came
  out of the same solve/apply passes — so a `1` marks the whole group whose
  positions, and therefore fluxes, are the last iterate rather than a
  converged solution. The run logs a warning naming the worst offenders, and
  `<name>_scene_catalog.csv` carries the same verdict per scene along with
  `astrom_niter` (passes used) and `astrom_step` (last increment, pixels).

`astrom_weight_<i>`
: The weight this source carried as an astrometric anchor, from
  `FitConfig.astrom_robust`: `1.0` when the robust pass is off, declined the
  scene, or found nothing to downweight, and `0.0` for a rejected anchor.
  Non-anchors read `1.0` — they never had a vote to lose. Where a source has
  several templates the weakest is taken, so the column answers "was any part
  of this source thrown out of the shift fit". The column is written whether
  or not the pass is enabled, so an A/B pair of runs differs in the values
  rather than in the schema.

### Aperture columns

Aperture photometry on the model-plus-residual image is always attempted; the
aperture radius comes from `FitConfig.phot.aperture_diam` (a fixed diameter in
`phot.units`, or one value per band as an array). Left at `None` — the default
— the aperture is instead sized by encircled energy through
`FitConfig.phot.aperture_ee`, default `0.70`: the diameter enclosing that
fraction of the band's *model* PSF, meaning the drizzled stamp after the
Gaussian diffusion blur, which is the PSF the matching kernel was built
against. Sizing by EE is what keeps a colour free of aperture corrections to
first order — the same EE in every band is the same correction factor in every
band, and it cancels in the ratio, which a fixed angular diameter cannot do
because the PSF width runs with wavelength. It also sits near the SNR optimum:
for a background-limited point source `SNR ~ EE(r)/r`, which for a Gaussian
peaks at `r = 1.585 sigma`, i.e. `1.35 x FWHM` in diameter, enclosing ~71%.
An explicit `aperture_diam` always wins, so a run tied to an external catalog's
aperture is never silently resized. Where the PSF stamp never reaches the
requested fraction inside its inscribed circle, the run warns and falls back to
1.5 times the band PSF FWHM rather than extrapolating past the stamp.

The radius actually measured in is, per source, **the larger of the catalog
aperture and the encircled-energy aperture**. The two ends of the catalog fail
in opposite directions: a bright extended source has a wide catalog aperture,
and measuring the band in that same aperture leaves the catalog correction as a
pure PSF correction over a shared radius; a faint source sits on the catalog's
aperture floor (72.4% of the MINERVA SUPER catalog is at 0.2"), which at MIRI
resolution is well inside the PSF core, and the encircled-energy aperture is
the floor that catches it near maximum SNR. Both raw sums are written whichever
way the rule falls, so the choice is auditable per source.

`aper_<i>`
: The aperture diameter actually used, in arcsec, per source — the larger of
  the two. Always written. Under `aperture_ee` the diameter is derived from the
  band's model PSF rather than configured, so this column is the only record of
  the realized size, and it is what you need to compare two bands or compare
  this run against a fixed-diameter one.

`aper_ee_<i>`
: The encircled-energy aperture diameter, in arcsec. A band constant, so it is
  the same for every source; the value `aper_<i>` falls back to.

`ap_flux_ee_<i>`
: Raw aperture sum on (model + residual) in the encircled-energy aperture.

`ap_flux_catap_<i>`
: Raw aperture sum on (model + residual) in the catalog aperture. Equals
  `ap_flux_ee_<i>` when no catalog aperture is configured. Not to be confused
  with `ap_flux_cat_<i>`, which is a corrected *total* on the catalog's Kron
  convention, not a raw sum.

`ap_res_<i>`
: `sum_Omega(res)`: the residual map summed over `disk(aper_<i>/2)` with other
  sources' segment pixels zeroed. The unscaled term of Estimator 3.

`ap_flux_est3_<i>`
: Estimator 3, `ap_model_<i> * psfcor_<i> * totcor_cat + ap_res_<i>`. Written
  only where `totcor_cat` and `psfcor_<i>` are both finite, so it needs the
  `phot.kron_flux_col` / `aper_flux_col` / `kron_radius_col` knobs set.

With `src_tmpl` the unit-normalized high-resolution composite `H` and
`src_img` the unit-normalized band-convolved composite `H*K`,
`ap_hi = aper(src_tmpl, R)` and `ap_lo = aper(src_img, R)` are their
encircled energies at the aperture radius.

```{note}
**Naming rule.** A correction name carries `tot` only if it includes the
encircled-energy term. A correction that stops at the edge of the model's
own finite support is not a total, whatever other codes call it. So
`stampcor` (`1/ap_lo`) has no EE and is deliberately not `totcor`, while
`totcor` (`1/(ap_lo*ee_psf_lo)`) does and keeps the name. Classic IDL
releases the first of these *as* `totcor`, and comparing it against a
quantity that does include the EE is the usual way to manufacture a
few-percent offset between the two codes. When writing an estimator, prefer
the factored `psfcor * totcor_cat` over a bare `totcor`: the factors state
which convention is meant and the bare name does not.
```

`ap_flux_<i>`
: Raw aperture sum on model + residual at the source position.

`ap_model_<i>`
: Aperture sum on the source's own model only.

`ee_psf_lo_<i>`
: The per-source finite-support box encircled energy applied in
  `totcor_<i>` (filter-mean fallback where a template has no recorded
  value) — the same factor `flux_<i>_total` divides by.

`stampcor_<i>`
: `1 / ap_lo` alone: the aperture carried to the total of the model on its
  own finite stamp support, with **no** EE factor. Named `stampcor` rather
  than `tot*` on purpose — see the naming rule below. This is the quantity
  classic IDL releases as `totcor`, which is misnamed by the same rule;
  compare the two only when both runs use the same PSF support. (Written as
  `tot_stamp_<i>` before 2026-08-12.)

`totcor_<i>`
: Aperture-to-total correction, `1 / (ap_lo * ee_psf_lo)`. This one earns
  the name: it always includes the beyond-support encircled energy, like a
  catalog aperture-to-total. The support-only piece is `stampcor_<i>`.

`psfcor_<i>`
: `ap_hi / ap_lo`, the source's own high-res to low-res band EE ratio at the
  aperture radius (classic mophongo's PSF/shape correction).

`ap_flux_corr_<i>`
: `ap_flux_<i> * totcor_<i>`, the aperture flux corrected to total. On the
  same absolute scale as `flux_<i>_total`; for an isolated point source the
  two converge.

`totcor_cat`
: Band-independent catalog-side aperture-to-total,
  `(f_kron / f_aper) / EE_H(k * R_kron)`: the detection catalog's
  Kron-to-aperture flux ratio times the inverse encircled energy of the
  high-resolution PSF at the scaled circularized Kron radius. Written only
  when `FitConfig.phot.kron_flux_col`, `phot.aper_flux_col` and
  `phot.kron_radius_col` name existing catalog columns (`phot.kron_k` scales
  the radius; SExtractor AUTO convention is 2.5). This is the quantity the
  flux-estimator report called `tcorH`.

`ap_flux_cat_<i>`
: `ap_flux_<i> * psfcor_<i> * totcor_cat`: the aperture flux carried onto
  the detection catalog's Kron total convention, for catalog-type
  comparisons.

### How the aperture enters the flux estimator

Estimator 3 of the flux-estimator comparison report (its Eq. 12) is written to
`ap_flux_est3_<i>`:

```
f3     = aper(model - model_nn, R) * psfcor * totcor_cat + sum_Omega(res)
psfcor = ap_hi / ap_lo
Omega  = disk(R), other sources' segment pixels zeroed
```

`R` is `aper_<i>`, the band's own measurement aperture after the max rule
below. The aperture-to-total is built as `psfcor * totcor_cat` — the
high-res→low-res band correction times the high-res aperture-to-total — rather
than from a bare `1/ap_lo`, which avoids the name `totcor` entirely; that name
has meant both the with-EE and the without-EE convention in different
codebases. `f3` is a total in the full sense, because `totcor_cat` carries the
encircled-energy term.

read left to right as three separate jobs:

- `aper(model - model_nn, R) * psfcor` — the best-fit model, with the
  neighbours' models subtracted, summed in the measurement aperture and then
  carried onto the high-resolution template's PSF. This is one joint
  aperture-and-PSF correction rather than two: `psfcor` is a ratio of two
  encircled energies at the *same* radius, so the aperture size cancels out of
  it and only the hi/lo PSF difference remains. Written as `ap_model_<i>`.
- `totcor_cat` — the correction to a genuine total, taken from the detection
  catalog: the F444W Kron-to-aperture ratio times the inverse encircled energy
  outside the scaled Kron radius. Catalog-tied, so results on this estimator
  compare only against the release catalog, never against the internal
  `totcor_<i>` convention.
- `sum_Omega(res)` — the residual summed over the same disk, with other
  sources' segment pixels zeroed, written as `ap_res_<i>`. **Added unscaled.**
  The neighbours are already subtracted in the model, but their residuals are
  not this source's to claim, and inside a shared aperture they would be. This
  is the data-driven term: where the template is wrong, it is what keeps the
  estimator honest.

The unscaled residual is what separates `ap_flux_est3_<i>` from
`ap_flux_cat_<i>`. The latter is `ap_flux_<i> * psfcor * totcor_cat`, and
`ap_flux_<i>` already contains the residual, so it applies the
aperture-to-total correction to the residual as well as to the model. The two
differ by `ap_res * (psfcor * totcor_cat - 1)` and agree only where the
residual vanishes. They are different estimators, not two spellings of one.

The aperture rule is what makes the first two terms compose cleanly. When the
catalog aperture wins, the band is measured at the catalog's own `R_phi`, and
`totcor_cat` — which was itself defined at that radius on the detection band —
needs no aperture-size adjustment at all: `psfcor` is left doing pure PSF work.
When the source sits on the catalog's aperture floor, that radius is far inside
the MIRI core and the encircled-energy aperture takes over, trading an exactly
matched radius for the SNR that would otherwise be thrown away. The larger of
the two is the right pick in both regimes, and the two raw sums
(`ap_flux_catap_<i>`, `ap_flux_ee_<i>`) are both written so the trade is
visible per source.

The residual term is why the floor should not simply be made large: it grows
with aperture area, so a wide aperture admits more noise and more of any
neighbour the fit failed to subtract. The model term is already corrected, so
buying wing-insensitivity with a wide floor mostly buys noise in `res` — which
is why `aperture_ee` defaults near the SNR optimum rather than to a near-total
aperture.

`aper_<i>` records the realized diameter per source, which is the number to
quote when reporting any of this.

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
`id` (scene id), `n_templates`, `n_anchor` (sources that passed the
astrometric anchor cuts — SNR, isolation, and optional star exclusion — so it
counts more than brightness despite the older `is_bright` spelling), `ra`,
`dec` of the scene center, the total astrometric shift `dx`, `dy` at that
center in reference-grid pixels (NaN where the scene solved no astrometry),
`sigma_shift`, `shift_rms`, `chi2_dof`, and a `minerva_link` column linking
each position to an external sky viewer. That column is written only when
`RunConfig.fitsmap_url` gives a FITSMap root — a full URL, or a bare
`<field>/<release>` which gets the default host in front. Unset or empty drops
the column, since a guessed link is worse than none.

`dx`, `dy` is the mean shift actually applied to the scene's templates, at
the centroid of those templates. It is not a field evaluation: accumulated
shifts are a sum of damped increments, each fitted at whatever the previous
pass left behind, so at order >= 1 the total is not in general representable
by the functional form of any single pass. `Scene.shifts` holds only the last
increment's coefficients and is never written here.

`sigma_shift` is the formal 1-sigma on that shift, propagated from the shift
block's covariance with the fluxes marginalized out, evaluated at the scene
centre and averaged over the two axes. It is what gives `dx`, `dy` a scale: a
0.2 px shift means nothing until you know whether it was measured to 0.02 px or
to 0.5 px. It is the *last pass's* number -- passes re-measure the same pixels,
so at convergence the accumulated shift is determined about as well as any one
pass determines it. Read it as the scale on the total, not as an error that
shrinks with the number of passes.

`chi2_dof` is the reduced chi-square over the scene's bounding box, against the
residual with *every* scene's model subtracted, with one free parameter per
template plus the shift coefficients. Sorting on it is the direct way to find
the scenes worth looking at.

`shift_rms` is the spread of those applied shifts about their mean. At the
default order 0 it is exactly zero for any number of passes: the field is a
constant, so every template moves by the same amount every pass and a scene
that walks rather than settles walks rigidly. At higher order it is the
amplitude of the gradient the field actually carried, which is the direct way
to see whether the extra terms did anything.

Three columns record how the shift was reached: `astrom_niter` (solve/apply
passes this scene used before it dropped out of the loop), `astrom_step`
(its last shift increment in pixels) and `flag_astrom` (`0` converged, `1`
still moving when the budget ran out, `-1` no verdict). Every source of the
scene inherits `flag_astrom` as the fit table's `flag_astrom_<i>`.

`astrom_floor` is what `FitConfig.astrom_robust` measured, NaN where the pass
never ran or declined. It is the **systematic error floor on an anchor's
position**: the extra per-axis scatter, in fit-grid pixels, that must be added
in quadrature to the anchors' formal errors before their disagreement with the
fitted shift field is statistically consistent. Formally it is the `s >= 0`
solving

    median over anchors and axes of  r_ia^2 / (v_i + s^2)  =  median(chi^2_1)

with `r_ia` the anchor's implied shift minus the fitted field and `v_i = 1 /
I_i` its formal variance. Median-based, so one wild anchor cannot set it, and
zero when the anchors already agree within their errors.

Read it against `astrom_shift_tol`: a floor several times the tolerance means
the anchors do not agree at the level the loop is trying to converge to, and
the shift is limited by template and PSF fidelity rather than by noise. It is
also the ceiling on how much any one anchor can be trusted, which is what
`astrom_leverage_cap` approximates with a quantile.

Expect the NaN to be the common case rather than the exception. The pass needs
`max(FitConfig.scene_minimum_anchors, 2p)` usable anchors for a `p`-term field
per axis, and declines otherwise; on a COSMOS F770W run with
`scene_minimum_anchors=7` that was 792 of 1434 scenes, every one of them a
scene with too few anchors rather than a scene where the measurement failed.
Note that `n_anchor` counts the sources that passed the SNR and isolation
cuts, which is one step ahead of the gate: an anchor whose residual carried no
finite information does not count toward it.

The `SCENES` extension of the fit table carries the rest of the robust
verdict: `astrom_robust` (`1` if the pass judged this scene), `astrom_nreject`
(anchors rejected outright) and `astrom_neff` (anchors surviving rejection --
`n_anchor` counts those that passed the cuts, before any rejection).
Comparing these against a run with the flag off is the intended way to judge
whether the weighting earned its place on a given field.

With `scene_plots` enabled, scenes also get a `<name>_scene_<id>.png`
diagnostic figure, written to a `scenes/` subdirectory of `out_dir` (created
only when the plots are requested). `scene_plots_max` decides how many:
by default the 100 worst by `chi2_dof` and the 100 with the largest
`astrom_floor`, since a full field solves ~1600 scenes and rendering all of
them costs the better part of an hour. Each figure is sampled to its own
scene rather than drawn at a fixed size, so the PNG is as large as the scene
needs and no larger. The partition as a whole gets
`<name>_scene_map.png` in `out_dir`: the full field with every segment
colored by the scene that fitted it, and each scene's bounding box drawn in
the same color. Fields wider than 4000 pixels are decimated for the display,
reducing each block by its largest label so single-pixel segments survive
rather than falling between samples; the title states the ratio. Bounding
boxes are drawn for up to 250 scenes, past which they overlap into noise.

Alongside it, `<name>_scene_blobs.png` draws the same partition from the other
direction: each scene as the convex hull of its template positions, filled in
its own colour and labelled with its scene id in grey. It never touches the
mosaic or the segmentation map, so it is a vector figure whose cost is the
number of scenes rather than the size of the field, and it needs no
decimation. Scenes are drawn largest first, so a compact scene inside a
sprawling one stays visible. Scenes narrower than 60 pixels are drawn as
circles rather than hulls -- a hull over three sources is a sliver that reads
as noise at field scale -- and are left unlabelled, since on a real partition
most scenes are small and numbering every one of them buries the figure.

In a scene diagnostic, segments listed as saturated elsewhere in the field
stay visible in the image panel but are excluded from its display scale,
which their brightness would otherwise flatten; the residual panel nulls
them, since the fit residual under a saturated core is meaningless. A scene's
own saturated members are never treated this way.

Scene ids are labels of one run's fit partition, not stable source
identifiers: membership depends on the scene-construction settings
(`scene_coupling_thresh`, `scene_max_size`, `scene_max_merge_radius`), so
scene-id-keyed products from runs with different settings are not comparable
by id.

## Shift field

`<name>_shift_field.png` maps the astrometric solution over the field, and is
written whenever at least one scene solved for shifts. Every such scene
contributes `2**order` arrows, `order` being the Chebyshev order of its shift
basis (`fit["astrom_kwargs"]["poly"]["order"]`, forced to 0 for a saturated-star
scene): order 0 puts a single arrow at the scene center, order 1 two spread
along the scene's longer axis, order 2 a 2x2 grid. Each arrow runs from the
template position toward where the source is measured in the fitted band, so
it points the same way as the `dx`, `dy` of the per-template fit table. The
scene id labels each scene in light gray next to its first arrow.

Positions and arrows are in RA/Dec degrees, drawn with aspect `1/cos(dec)` so
angles are undistorted, RA increasing to the left. Shifts are sub-pixel, so
arrows carry a common magnification set from the 90th percentile of their
length — a single runaway scene would otherwise shrink every other arrow to
nothing — and the legend arrow gives that percentile in pixels and arcsec.

`Scene.shifts` holds the coefficients of the *last* astrometric iteration
only, so the plotted field is refit at the same order to the accumulated
`Template.shifted` values, which are the total applied offsets.
{meth}`~mophongo.pipeline.Pipeline.plot_shift_field` returns `(fig, ax)`, or
None when no scene solved for astrometry.

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
| 4 | 16 | `FLAG_OUTSIDE_WEIGHT` | no support on the fitted band's weight map; the template is dropped from that band |
| 5 | 32 | `FLAG_SHIFTED` | an astrometric shift was applied |
| 6 | 64 | `FLAG_DEBLENDED` | source is a catalog deblend child (provenance) |
| 7 | 128 | `FLAG_SATURATED` | source carried a `FLAG_SATURATED_*` catalog flag (provenance) |
| 8 | 256 | `FLAG_PSF_EXTENDED` | the build scheme blended PSF wings into the template |
| 9 | 512 | `FLAG_EXTEND_FAILED` | extension was attempted but the PSF was unusable |
| 10 | 1024 | `FLAG_NO_COVERAGE` | part of the segment falls where the detection band has no exposure |

`FLAG_NO_COVERAGE` marks a source the detection image only partly sees, which
a combined catalog and segmap can produce: another band defined the segment and
this one has no exposure under part of it. The template is still built and
fitted -- the blank pixels drop out of the unit-sum normalisation -- but the
flux covers the exposed pixels only and is a lower limit on the segment's
light. A source whose *position* is uncovered gets no template at all and does
not reach the fit table, so the flag marks the partial case by construction.
It requires a weight map: `extract_templates` sets it only when
`detection_weight` is given, and treats an all-zero map as no information.

`FLAG_PSF_EXTENDED` and `FLAG_EXTEND_FAILED` are set by the build-time
schemes only (`"psf_wings"`, `"wren"`, `"classic"`). A template is flagged as
PSF-extended when its blend weight fell below one, that is, when the scheme
mixed a scaled PSF into the data rather than keeping the data alone; the
post-extraction modes `"psf"` and `"psf_model"` set neither flag.

`FLAG_OUTSIDE_WEIGHT` is set by `Templates.prune_outside_weight`, which then
drops the template: the band's weight map gives it no support, so its flux
would carry no information. It is a *per-band* judgement, unlike
`FLAG_NO_COVERAGE`, which describes the detection image and is therefore the
same for all bands -- a source can be pruned from F1800W and fitted in F770W.
Because the template is dropped, the flag never reaches a row of that band's
outputs; it stays on the hi-res template the pruned copy came from, and the
count (with the ids at debug level) goes to the log. `FLAG_HAS_NAN` is
declared and nothing sets it.

As of this writing these flags live in the stamps file (and on the in-memory
`Template` objects), not as a column of the fit table.

## Method reference

### `Pipeline.write_outputs()`

No parameters. Requires a completed {meth}`~mophongo.pipeline.Pipeline.run`
on a config-driven pipeline. Writes the residual FITS, fit table,
per-template fit table, scene catalog and plots, the shift field, and (when
the config's `save_stamps` is true) the stamps file. Returns `self`.

### `Pipeline.plot_shift_field(*, save=None, arrow_frac=0.05)`

`save` : `str | os.PathLike | None`, default `None`
: Path to save the figure to.

`arrow_frac` : `float`, default `0.05`
: Length of the 90th-percentile arrow as a fraction of the field span, which
  sets the common magnification.

Returns `(fig, ax)`, or None when no scene solved for astrometry (or the
pipeline has no reference WCS). See [Shift field](#shift-field).

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
