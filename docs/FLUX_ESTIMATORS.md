# Total-flux estimators: legacy IDL vs current python

Analysis of the three estimators in `scratch/wren/flux_estimator_comparison.pdf`
(Monu Sharma), the legacy IDL implementation in `legacy/autopilot/`, and what
the current python pipeline actually computes.

The write-up for circulation is `scratch/wren/flux_estimator_comparison.pdf`
(source: `flux_estimator_comparison.tex`; `_v1`.. `_v4` preserved alongside). It is
canonical for the algebra and for the assessment against catalogue-matching and
low-SNR goals. This file is the repo-facing summary.

Terminology: **ifl** = this repo (`ivolabbe/mophongo`, `flux-bug`); **wren** =
`wrensuess/mophongo@dev-wren` (see `docs/FORK_DIFF_WREN.md`); **IDL** =
`legacy/autopilot/`.

Two questions drive this: **what the profile-shape-dependent correction may
legitimately be applied to**, and **over what region the residual should be
integrated**. The short answer to both is that they matter far less than the
template-truncation problem sitting underneath them.

Companion to [PHOTOMETRY_APERTURES.md](./PHOTOMETRY_APERTURES.md) (column
definitions, aperture plumbing) and
[PSF_SHAPE_THROUGHPUT_CONVENTION.md](./PSF_SHAPE_THROUGHPUT_CONVENTION.md)
(shape vs throughput).

## 1. Notation, and how the python columns map onto it

The PDF works with per-source saved images:

```text
_phot = _org - _model_nn          neighbour-subtracted
_res  = _org - _model             full residual
_phot = (_model - _model_nn) + _res = central model + residual
```

Current python builds exactly `_phot`, though it never names it
(`pipeline.py:986-988`):

```python
patch = residual[tmpl.slices_original] + fl * tmpl.data[tmpl.slices_cutout]
```

`residual` has *all* models subtracted, so adding back this source's model
leaves `_org - _model_nn`. Identical construction.

With unit-sum templates, the two aperture sums of the PDF are precisely the
python `num` and `den`:

| PDF | python | meaning |
|---|---|---|
| `ap_hi = aper(src_tmpl, R_φ)` | `num` (`pipeline.py:1000`) | high-res composite EE at `R_φ` |
| `ap_lo = aper(src_img, R_φ)` | `den` (`pipeline.py:1007`) | band-convolved composite EE at `R_φ` |
| `psfcor = ap_hi/ap_lo` (IDL `apcor1`) | `psfcor_<i>` | high-res → low-res band EE ratio |
| `stampcor = 1/ap_lo` (IDL `totcor1`) | `stampcor_<i>` | aperture → support total, **no EE** |
| `aper(_phot, R_φ)` | `ap_flux_<i>` | raw neighbour-subtracted aperture flux |
| `aper(_phot, R_φ)·apcor1` = `fcor1` | `ap_flux_corr_<i>` | band-PSF-corrected aperture flux |
| `A` (NNLS amplitude, `fmodel`) | `flux_<i>` | fitted template amplitude |

> **Superseded 2026-08-12.** The table above is updated to current names and
> columns; the assessment paragraphs below it are historical. Main now writes
> `psfcor_<i>`, `stampcor_<i>` and `totcor_<i>`, so Estimator 1 is computable
> from the table. Naming follows the rule in the tex report's "Naming" section
> and in {doc}`outputs`: a name carries `tot` only when it includes the
> encircled-energy term, which is why `1/ap_lo` is `stampcor` and not `totcor`.

**Historical (pre-2026-08-12): current python implements Estimator 1, stopped one factor short of total.**
`ap_flux_corr_<i>` is the PDF's `fcor1`, not `f_t^(1)`; reaching a total needs
one further multiplication by a `tcor_H`-like factor. Verified numerically on
the DR0.1 COSMOS F770W run (`flux_1 > 5`, n=119): `den` = 0.468, `num` = 0.662,
`ap_corr_1` = 1.393 = `num/den`.

## 2. What the legacy IDL actually did

This is the part that was unknown, and it answers Wren's closing question in
`scratch/wren/00NOTES` ("in particular, knowing what original mophongo did to
build templates… were they also truncated at the segmap?").

### 2.1 Templates were segment-limited, but to a *competitively grown* segment

`legacy/autopilot/mophongo__define.pro:274-276, 299-303`:

```idl
gc = growthcurve(self.getpsf(1,1), rhalf=rhalf_det)
kgrow = apermask(oddsize(round(rhalf_det*2.0)))
...
; grow mask of object and neigbors, keep grown pixels dominated by object
knn  = convol(seg ne s.id and seg ne 0, kgrow)
kseg = convol(seg eq s.id, kgrow)
kseg = kseg gt knn and kseg ne 0
```

The growth kernel is a disk of diameter ≈ 2·`rhalf` of the detection PSF (≈ 2 px
radius for NIRCam F444W at 40 mas — coincidentally the python
`template_dilate_segmap=2` default). The two convolutions count own-segment
versus neighbour-segment pixels within that disk, and the mask keeps only pixels
where the source's own segment area *dominates*. A competitive, Voronoi-like
dilation: order-independent and symmetric between neighbours, unlike python's
`safe_dilate_segmentation` (`catalog.py:353-375`), which dilates only into
background and lets the last-written label win in contested territory.

### 2.2 PSF wings filled everything outside that mask, neighbours included

`legacy/autopilot/mophongo__define.pro:315-332`:

```idl
; SNR in tmpl image segment, reject segment pixels < 0 -> replaced by PSF wings
iseg = where(kseg and det_img gt 0, complement=inoseg,/null)
f_tmpl = total(det_img[iseg])
e_tmpl = sqrt(iseg.length/det_ivar)
psf_tile = extrac(psf,psz[0]/2-xpeak,psz[1]/2-ypeak,tsz[0],tsz[1])

; at low snr add in point source prior to tmpl in quadrature
if f_tmpl/e_tmpl lt 1.5*fit_snrlo_psf then $
   det_img[iseg] = sqrt(det_img[iseg]^2 + (e_tmpl*fit_snrlo_psf*psf_tile[iseg])^2)

; add PSF wings outside the segmap, normalized to flux in segment
det_img[inoseg] = psf_tile[inoseg]*total(det_img[iseg])/total(psf_tile[iseg])
det_img /= total(det_img)
```

Three things, in order:

1. **Negative segment pixels are excluded** from the segment and treated as
   outside-segment — they get PSF wings rather than being carried as noise.
2. **Low-SNR templates are blended toward a point source in quadrature**, not
   switched to one. The threshold is `1.5 × fit_snrlo_psf`, and
   `autopilot.pro:1353` sets `fit_snrlo_psf = 10.0`, so the prior engages below
   **template SNR 15** and the added component is a unit-sum PSF scaled to
   **SNR exactly 10**. As `f_tmpl/e_tmpl → 0` the template asymptotes to the PSF
   (the code comment says so explicitly). `fit_snrhi_psf = 150.0` is unrelated:
   it adds an *extra free* PSF component to the basis for very bright sources
   with anomalous chi² (`:1412, :1485`).
3. **PSF wings fill everything outside the grown mask.** `inoseg` is the
   complement of (`kseg` ∧ `det_img > 0`) over the *whole stamp*, so it contains
   blank sky, **neighbour-owned segment pixels**, and non-positive pixels inside
   the source's own segment. All three get wings, normalized so the amplitude
   matches the segment flux ratio. Then unit-normalized. The fill never
   replaces the source's own positive segment data — the region is the
   complement of (`kseg` ∧ data>0) *within the stamp* ("full-complement
   fill").

So IDL templates **overlap their neighbours**: each source's model puts PSF-wing
light under the neighbours' segments. Python's `psf_wings` mode does the
opposite — with `extend_wings_background_only=True` (default), neighbour-owned
pixels keep their zero value, so templates stay disjoint. **Neither placement is
unbiased today**: the data-derived cores already carry the neighbours' wing
light, so both bases fit a wrong profile — see §2.3.

The IDL composite `H` is therefore *always* (segment data + PSF wings), for every
source. For a source whose segment is small, `H` is very nearly a pure PSF —
which is exactly why IDL's `apcor` could be a PSF-only quantity without
inconsistency.

### 2.3 Cross-contamination is a bias, not a degeneracy (v3 correction)

Every data-derived core is raw detection data inside the (grown) segment, so it
contains the neighbours' wing light: `T_A|seg_A ∝ (f_A·S_A + f_B·S_B)|seg_A`.
This is common to every scheme that cuts templates from the data. Wing placement
only chooses where the *additional* error lives: full-complement fill (IDL —
wings on every stamp pixel outside the source's own *positive* segment,
neighbour segments included, own data never) represents A's light under `seg_B`
twice (analytic in `T_A`, data in `T_B`'s core); background-only (python
default) withholds it from `T_A`, so it is carried only by `T_B` —
misattributed.

Neither basis spans the truth: for a pair no amplitudes reproduce
`f_A·S_A + f_B·S_B` (matching the `P_A`/`P_B` content per region forces `a=1`
then `b=0`, which fails in `seg_B`). The fit leaves structured residuals along
the bridge between the pair and returns biased amplitudes. An earlier framing
of the overlap as benign non-orthogonality ("the solve apportions the shared
flux", cost only in covariance) was wrong — that holds for overlapping
*correct* profiles, not contaminated ones.

Region-integrated refinement (point-source pair, own/cross/sky PSF fractions
`e, c, s`): IDL's wing scaling to the *contaminated* segment flux cancels in
the unit normalisation, so the coarse IDL template shape is exactly the clean
`[e, c, s]` and region-level flux allocation is exact — the surviving IDL bias
is intra-segment only (the neighbour-shaped gradient inside the core), small
for comparable pairs, unbounded for faint-beside-bright where
`f_nn·c/(f·e) > 1`. Background-only is worse: coarse shape `[e, 0, s]`, a
first-order misallocation of the cross-segment light. Ordering:
full-complement < background-only in bias; both fail faint-beside-bright. The
competitive mask cannot help — `kseg > knn` counts segment pixels, and the
contamination is smooth *unsegmented* wing light.

**Fix at extraction**: pass 0 as now; pass 1 re-extracts each core from
`D − Σ_{j≠i} W_j` (the other sources' analytic wing models, which extension
already builds) and renormalises wings to the decontaminated segment flux. One
iteration converges (a wing model ~10% accurate leaves percent-level
contamination). With clean cores **full-complement fill becomes the correct
placement** — each photon represented exactly once — and background-only
becomes the wrong choice. Validate on mock pairs: (separation × flux ratio) ×
{truncated, background-only, full-complement, full-complement+subtracted}.

### 2.4 The corrections were PSF-based, not composite-based

`mophongo__define.pro:1300-1303`:

```idl
mask_shrink = sqrt(od.kron_area/(noverlap+od.kron_area))  ; kron ellipse, masked pix
faper, detpsf, !null, !null, od.kron_major*mask_shrink, inv_det_apcor, $
       elon=od.kron_major/od.kron_minor, theta=od.theta
op.apcor  = 1.0 / inv_det_apcor
op.totcor = (od.flux_auto / op.flux_auto) * op.apcor
```

- `apcor` is the inverse encircled energy of the **detection PSF** inside the
  Kron ellipse — a pure PSF quantity, with no dependence on the source's own
  composite.
- `totcor` multiplies that by the Kron flux ratio between the detection image
  (`od`) and the PSF-matched photometry image (`op`), the two measured with
  different Kron factors (`par.detect_kron` vs `par.phot_kron`,
  `:1293-1298`).
- `mask_shrink` shrinks the Kron ellipse to account for neighbour-masked pixels
  before the PSF EE is evaluated.

Structurally this is the PDF's `tcor_H = (FLUX_AUTO/FLUX_APER) × 1/EE_H(k·R_CIRC)`
(Eq. 8): a measured aperture→Kron ratio times a PSF-only Kron→total term. The
profile-shape dependence enters through *measured Kron fluxes*, never through
the template.

This is exactly why Wren saw "the idl is giving corrections close to the psf EE
correction alone". It is not an approximation in the IDL — it is the design.

## 3. Why python's correction blows up and the IDL's does not

Current python truncates hard at the segment (`templates.py:1544`):

```python
cut.data[cut.slices_cutout] *= (segm.data[cut.slices_original] == label)
```

then unit-normalizes (`:1549-1551`). PSF-wing extension exists
(`extend_templates in {"psf", "psf_wings"}`) but **is unreachable from a run
config**: `RunConfig` has no `extend_templates` field, `from_config`
(`pipeline.py:387-406`) never sets it, and `load_data`'s `Pipeline.__init__`
call (`pipeline.py:602-612`) omits it, so it falls back to the `__init__`
default of `None`. Both MINERVA runs therefore used truncated templates with
only `template_dilate_segmap=2` applied.

The consequence is not primarily in `apcor1`. For a compact source with
`seg ⊆ disk(R_φ)`, `ap_hi → 1` and `apcor1 → 1/ap_lo`, which is bounded and
well behaved. The damage is in any factor that references an **external total**:

- the fitted amplitude `A` is the total of a template that has no wings, so `A`
  is biased low — Ivo's estimate is up to ~10% at the bright end and worse at
  the faint end;
- `est3cat`-style corrections form `catalog_total_f444w / internal_f444w`, and
  with a truncated `internal` that ratio absorbs the entire missing-wing flux;
- because the missing wings carry no shape information in the truncated model,
  the absorbing factor is shape-blind and scatters — Wren's "giant vertical
  band", with corrections up to 40×, concentrated in sources with segmaps
  smaller than the measurement aperture (<200 px in the F1800W split).

**The bias is at the template level, and correction-level fixes only relocate
it.** IDL's `A` was already close to unbiased because its `H` had wings, which
is why its `totcor` could be a mild, PSF-dominated correction. Reproducing IDL's
*corrections* without reproducing IDL's *templates* cannot work.

Ivo's Slack note (`00NOTES:195-196`) says the same thing from the other
direction: template extension "is effectively an aperture correction".

### Wing construction: three variants, none correct on its own

| | wing shape | amplitude | region | SNR gating |
|---|---|---|---|---|
| IDL | PSF tile at the peak | segment-flux ratio | full complement | quadrature core prior, S<15 |
| ifl `psf_wings` | data ⊛ PSF (self-convolution) | unit-norm after fill | background-only | none |
| wren (default-on) | `W·data + (1−W)·tile` per 0.15″ annulus | `A_src` segment ratio | own segment + owned background | annulus onset S=3, core onset S=15 |

The ifl self-convolution over-broadens: the observed data are already
intrinsic⊛PSF, so the predicted wings follow PSF^⊛2 — worst exactly where the
IDL tile is exact (compact sources, ~√2 wider for a Gaussian core), converging
to the tile only in the far field where `(D·1_seg) ⊛ P → f_seg·P(r)`. The
over-broadening sits in the transition annulus just outside the segment, where
the truncation correction lives. The tile is wrong-shaped for resolved and
multi-peaked sources. Wren's radial blend has the right structure — data where
they constrain, tile where they don't — but `M` is always a point source
(under-estimates resolved halos), the weight `min(1,(S/S0)²)` has a kink at
`S=S0` (the smooth `S²/(S²+S0²)` does not), extension never enters neighbour
segments, and the raw-data halo re-admits neighbour wing light at `W≈1`
(`docs/FORK_DIFF_WREN.md` §4). The mock pair grid should compare all three plus
the decontaminated full-complement variant of §2.3.

## 4. Question 1 — what may the shape correction be applied to?

### 4.1 The estimators

```text
f1 = aper(_phot, R_φ)·totcor1                          = A + aper(_res,R_φ)·totcor1
f2 = aper(_phot - _res, R_φ)·totcor1     + Σ_Ω(_res)   = A + Σ_Ω(_res)      exactly
f3 = aper(_phot - _res, R_φ)·apcor1·tcor_H + Σ_Ω(_res) ≈ A + Σ_Ω(_res)
```

The collapse to `A` is exact for `f2` because `A·ap_lo·(1/ap_lo) = A` (PDF
Eq. 14); `f3` differs only by `apcor1·tcor_H·ap_lo`, which is unity to the ~10%
by which the two constructions of the aperture-to-total correction disagree.

`f1 - f2` (PDF Eq. 17) splits into two terms:

```text
aper(_res,R_φ)·(totcor1 - 1)      over-extrapolation bias, ≥ 0 for positive residual
aper(_res,R_φ) - Σ_Ω(_res)        extra seg residual,      ≤ 0 since disk(R_φ) ⊆ Ω
```

### 4.2 The case against scaling the residual

**Conceptual (PDF §4.1, and Ivo's `00NOTES:199`).** `totcor` is a shape-based
scaling, valid for flux that follows the model shape. The residual is by
definition the part of the data that does *not* follow that shape. Multiplying
it by `totcor` applies a shape correction to flux of unknown shape.

**Magnitude.** `totcor1` is 3.3–4.0 in the PDF's MIRI sample, so `(totcor1 - 1)`
is 2.3–3.0. Applied to an in-aperture residual worth a few percent of
`aper(_phot,R_φ)`, that yields the measured 8–13% per-source inflation of `f1`
over `f2`, band-independent across f560w/f770w/f1000w.

**Sign.** The over-extrapolation term is ≥ 0 whenever the in-aperture residual
is positive, so the inflation is a bias, not scatter. Our own MINERVA runs
independently confirm the residual is positive at source positions: the median
aperture residual at SNR > 5 sources is +0.22σ (DR0/UDS) and +0.16σ
(DR0.1/COSMOS). Mild bright-source under-subtraction is the norm, so `f1` is
systematically high.

**Noise — the argument the PDF does not make, and the one that matters most for
Wren's use cases.** For faint sources the in-aperture residual *is* noise. `f1`
multiplies it by `totcor1 ≈ 3-4`, so the residual's contribution to the variance
of `f1` is ~10-16× that in `f2`. For the faint and upper-limit regime Wren
flagged, this is a larger effect than the 8-13% median bias, and it is invisible
unless the error is propagated through `totcor1` as well. Any `f1`-style column
whose error is not scaled by `totcor1` is overconfident by that factor.

**Fair point for `f1`.** If the template shape is right, `f1` and `f2` agree; the
disagreement *is* a measure of template-shape error. So `f1 - f2` is worth
keeping as a diagnostic even once `f2`/`f3` are the reported values.

### 4.3 Neighbour subtraction is settled, and worth stating quantitatively

All three estimators use `_phot`, never the raw aperture. Ivo's argument
(`00NOTES:198`): if template fitting is 10% accurate and a blended neighbour
contributes 30% of the aperture flux, subtracting the neighbour's model reduces
its contribution to a 3% error. The template only has to be roughly right for
subtraction to win by an order of magnitude. There is no regime where the
un-subtracted aperture flux is preferable.

### 4.4 `f2` is nearly free in the current code

Since `f2 = A + Σ_Ω(_res)` exactly, and `A` is already the column `flux_<i>`,
Estimator 2 needs **no aperture photometry at all** for its first term — just one
masked sum over the residual. In current column terms:

```text
f2_total = flux_<i> / throughput_<i>  +  Σ_Ω(residual)
```

The throughput division applies to the model term only. That is the same
principle as §4.2 applied to the finite-PSF-support correction: it is a
model-shape quantity, so it scales the model and not the residual. Pleasingly
self-consistent.

The caveat from §3 stands: `A` is "total" only if `H` is correct out to
infinity. `f2` is the right estimator, but its accuracy is bounded entirely by
template extension.

## 5. Question 2 — over what region should the residual be integrated?

The PDF uses `Ω = seg_H ∪ disk(R_φ)` (Eq. 13), which is never smaller than
either component: it reduces to the disk for compact sources
(`seg_H ⊆ disk(R_φ)`) and to the segment for resolved ones.

| region | captures | misses | noise |
|---|---|---|---|
| `disk(R_φ)` | compact residual | resolved flux outside the aperture | lowest |
| `seg_H` | resolved residual | wing mismatch between seg edge and `R_φ` — where PSF error lives | middling |
| `seg_H ∪ disk(R_φ)` | both | — | highest of the three |
| Kron ellipse | matches the catalog's own aperture geometry | — | highest |

The trade is bias against variance, and it changes sign with source type:

- **Bright resolved** — use `Ω` or the Kron ellipse. Bias dominates; the extra
  area pays for itself.
- **Faint compact** — `Ω` collapses to `disk(R_φ)` automatically. Nothing is
  lost, no decision needed.
- **Faint resolved** — the bad case. `Ω` is large, the residual sum is
  dominated by noise, and `Σ_Ω(_res)` becomes a random walk added to a
  well-measured `A`. An SNR-dependent shrink toward `disk(R_φ)` is defensible
  here, and it is the direct analogue of what the IDL did in *template* space
  with its low-SNR quadrature PSF floor: let the model take over where the data
  cannot constrain it.

A note on `Ω` and the segmap it is built from: `seg_H` should be the *dilated*
segment actually used for template extraction (`template_dilate_segmap`, default
2 px), not the raw catalog segment, or the region and the model disagree about
where the source ends.

**Noise on `Σ_Ω(_res)` in the upsample path.** The block-replication convention
(`conserve_sum=True` for science, ivar copied then × `k²`,
`pipeline.py:115-132`) is self-consistent for this sum: each subpixel carries
`σ_native/k`, and `k²` of them summed in quadrature return `σ_native` per native
pixel. So naive per-pixel propagation on the upsampled grid gives the right
answer. What it does *not* capture is the correlation between *native* pixels
introduced by drizzling, which inflates the true variance of any large-area sum.
Any error bar on `Σ_Ω(_res)` should be flagged as a lower bound.

## 6. Recommendations

Ordered by impact.

1. **Make template extension reachable, and turn it on.** Add
   `extend_templates` to `RunConfig` and pass it through `load_data`. Without
   this the estimator choice is second-order — the templates are biased low and
   every correction downstream inherits it. This is the single change that
   closes most of the python/IDL gap.
2. **Decontaminate template cores at extraction** (§2.3): subtract the other
   sources' wing models before re-extracting cores, renormalise wings to the
   decontaminated segment flux, then apply the full-complement fill (IDL
   placement) rather than background-only.
3. **Add the low-SNR PSF blend, linear rather than IDL's quadrature**:
   `H = w·H_data + (1−w)·PSF` with `w = S²/(S²+S₀²)`; `fit_snrlo_psf <= 0`
   disables cleanly. The quadratic weight carries 20% PSF at `S = 2·S₀` (IDL's
   quadrature adds ~12% there) — exponent/`S₀` set on injected-truth mocks.
   Also adopt the IDL's exclusion of non-positive segment pixels (`:204`).
4. **Write `f2` as a column**: `flux_<i>/throughput_<i> + Σ_Ω(residual)`. One
   masked sum; no new aperture machinery.
5. **Record the residual sum over each region separately** — `disk(R_φ)`,
   `seg_H`, and `Ω`. *(Deferred: the matched quantity uses the disk residual
   only — not testing colour gradients now, and the catalogue's adaptive
   `use_aper` grows with source size. Kept for the totals path.)*
6. **Propagate errors on the residual term**, and if any `f1`-style column is
   retained, scale its error by `totcor1` too. Flag drizzle correlation as an
   uncaptured term.
7. **Keep `f1 - f2` as a diagnostic.** It measures template-shape error, which
   is the quantity actually under investigation.

## 7. Open questions

- **`totcor1` vs `apcor1 × tcor_H` disagree by ~10%** (median ratio 1.094-1.109
  across three MIRI bands, per-source MAD 8-9%). The PDF attributes this to
  mophongo's internal `H`-based aperture-to-total versus the catalogue-side
  Kron geometry, and expects them to diverge most where high-res → low-res
  colour gradients exist. That is testable on injected-truth mocks, and per
  project policy it should be tested there rather than by comparison to real
  photometry.
- **Does extending templates close the `totcor1` vs `tcor_H` gap?** Both sides
  of the identity `totcor1 = apcor1 × tcor_H` (PDF Eq. 9) depend on `H`. If
  truncation is the dominant error, extension should shrink the ~10%
  discrepancy, and the identity becomes a useful internal consistency check
  rather than a known-broken one.
- **Wren's residual F1800W offset in both raw and total flux** (`00NOTES:138`)
  survives the Estimator 3 fix, so it is not an aperture-correction problem.
  Separate investigation; the long-wavelength scene-size and astrometry-mask
  issues in the same note are the more likely culprits.
