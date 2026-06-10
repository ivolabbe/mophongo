# Flux Recovery Debug Synthesis

Date: 2026-05-03

This report consolidates the debugging results for the recent Mophongo flux
recovery failures: scene-solver flux regularization, PSF flux conservation,
kernel normalization, convolution placement, centroiding, realistic PSF
templates, and the remaining template-support failure.

## Executive Conclusion

The current large realistic-PSF residuals are **not** caused by the flux
solver, a one-pixel convolution shift, `Cutout2D` parity, DrizzlePSF
normalization, the windowed PSF matching kernel, or pixel-scale centroid
scatter in the fitted model stamps.

The remaining production failure is that the no-extension F444W segmentation
templates are incomplete source models for F770W. They are finite-support,
unit-normalized cutouts. They fit their own finite-support F444W basis cleanly,
but after F444W->F770W convolution they do not describe the full F770W source
light distribution. The next implementation target should be template
support/ownership outside the segmentation footprint.

Current status:

```text
scene-solver flux regularization bug              fixed
DrizzlePSF finite-flux conservation semantics     fixed/documented
unit-template/kernel normalization convention     fixed/documented
dummy identity convolution path                   removed
Template.convolve_cutout placement convention     validated
Cutout2D origin-parity hypothesis                 falsified
no-kernel F444W source-stamp placement            pass
exact F770W true-template reference               pass
fitted model-stamp centroid scatter               not pixel-scale for bright sources
F770W true PSF 0.08 arcsec Gaussian convention    fixed/tested
no-extension F444W->F770W extracted templates     fail residual hard requirement
```

## 0. F770W True PSF Convention

The true F770W mock PSF is the F770W drizzle/STPSF response convolved with a
0.08 arcsec FWHM Gaussian. That is 2 pixels on the 40 mas reference grid and
1 native F770W pixel on the 80 mas grid. The broadening must therefore be
applied on the 40 mas sampling before returning to an 80 mas F770W-grid PSF.

This is distinct from the intrinsic source-size parameter:

```text
source_sigma_pix=2              # intrinsic source morphology on 40 mas grid
psf_gaussian_fwhm_arcsec=0.08   # extra F770W PSF broadening
```

The corrected code records the convention explicitly in the mock truth table:

```text
psf_gaussian_fwhm_arcsec_f770w = 0.08
psf_gaussian_fwhm_pix_f770w    = 1.0
psf_gaussian_fwhm_arcsec_f444w = 0.0
```

Regression tests now prove that the native F770W-grid blur is equivalent to
manual 40 mas upsampling, Gaussian convolution, and block-summing back to the
80 mas grid, and is not the old direct native-grid shortcut.

## Relevant Reports And Artifacts

Primary reports:

```text
FLUXBUG.md
REALISTIC_PSF_FLUX_RECOVERY_REPORT.md
FLUX_RECOVERY_DEBUG_SYNTHESIS_2026-05-03.md
```

Key 300-source run:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-extension none \
  --kernel-grid-nside 4 \
  --out scratch/codex_debug_realistic_300_centroid_diagnostic
```

Key outputs:

```text
scratch/codex_debug_realistic_300_centroid_diagnostic/realistic_gaussian_summary.txt
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2_true_templates.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/model_centroid_shifts_sigma_2.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/model_centroid_shifts_sigma_2.csv
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/source_table_sigma_2.csv
```

## 1. Scene-Solver Flux Bug

### Symptom

The Moffat mock validation showed a roughly 10% flux deficit even in the
high-resolution self-fit:

```text
no_extension  hi=0.9023  lo=0.8796
dilated3      hi=0.9324  lo=0.8990
```

Because the high-resolution branch did not require a PSF-matching kernel, the
kernel could not be the primary cause.

### Root Cause

`SceneFitter.solve()` was using `config.reg_astrom` as a photometric ridge on
the flux normal matrix. `reg_astrom` belongs only to the astrometric shift
block. Applying it to fluxes suppresses broad, low-norm templates.

Decisive scan:

```text
scene flux-only, reg_astrom=1e-04:
  hi=0.9324  lo=0.8991

scene flux-only, reg_astrom=1e-08:
  hi=0.9746  lo=0.9864

scene flux-only, reg_astrom=0:
  hi=0.9746  lo=0.9864

scene flux-only, reg_astrom=1e-02:
  hi=0.1853  lo=0.0977
```

### Fix

Separate photometric and astrometric regularization:

```text
flux block:       reg_flux   * scale(A)
astrometry block: reg_astrom * scale(BB)
```

Also fixed:

```text
SceneFitter.solve_flux static signature
fit_astrometry_niter=0 now reaches flux-only behavior
```

Recommendation: keep `reg_astrom` completely out of any flux-only matrix.
Any future solver path must have a regression test proving this separation.

## 2. PSF Flux Conservation And Kernel Normalization

### Correct Convention

Extracted image templates are unit normalized. Matching kernels are not unit
normalized by default. Their sums are physically meaningful because they carry
the finite-aperture PSF integral ratio between source and target PSFs.

The corrected convention is:

```text
unit F444W source template * non-unit F444W->F770W kernel = F770W template basis
```

For DrizzlePSF, the desired behavior is flux conservation, not an after-the-fact
"renormalization":

```text
drizzled PSF sum == finite evaluated ePSF/STPSF aperture integral
```

This holds for full stamps and must also hold for partial stamps. The public
`renormalize` keyword was removed to avoid conflating flux conservation with
unit normalization.

### Recommendation

Do not reintroduce any hidden unit normalization in:

```text
DrizzlePSF.get_psf*
MockMosaic PSF hooks
PSF.from_array
matching-kernel construction
```

If a caller needs a unit template, normalize the extracted source template, not
the physical PSF or matching kernel.

## 3. Kernel Matching

The windowed F444W->F770W kernel is not the current limiting issue. The kernel
diagnostics show fractional-percent agreement in the relevant PSF radial and
growth-curve metrics, while the problematic bright-source residuals are at the
5-10% level.

Important corrections already made:

```text
F770W target PSF is drizzled directly onto the F444W WCS grid
F770W blur is applied in angular units on the actual PSF array grid
kernel diagnostic no longer includes unrelated pipeline metadata
psf_matching_diagnostic.png was removed as redundant
```

Do not use an unwindowed Fourier-ratio kernel as a "fix". It can reduce some
PSF-stamp residuals by fitting high-frequency sampling/noise structure, but it
creates an aliased kernel and is not a valid production path.

Recommendation: keep the windowed kernel, keep kernel diagnostics separated
from template/fitting diagnostics, and treat any remaining >3% source residual
as a template or fitting-basis problem until proven otherwise.

## 4. Convolution And Alignment

### Identity/No-Convolution Path

The old dummy one-pixel identity convolution was removed. If no convolution is
needed, `kernel=None` now reaches the no-convolution branch.

### Production Convolution Path

The active production path is:

```text
Pipeline.run
  -> Templates.extract_templates(...)
       -> Template(...), subclass of astropy.nddata.Cutout2D
  -> Templates.convolve_templates(...)
       -> Template.convolve_cutout(...)
       -> mophongo.utils.fftconvolve(..., mode="full")
```

`AlignedCutout` exists, but it is not used by `Pipeline.run` or
`Templates.extract_templates`.

### Tests

The production extraction/convolution path was tested for all lower-left
origin parities:

```text
origin parities: (0,0), (0,1), (1,0), (1,1)
kernel: asymmetric even-sized kernel
comparison: local Template.convolve_cutout model vs full-image fftconvolve(..., mode="same")
agreement: <= 1e-12
```

Forced aligned-cutout A/B on the 300-source realistic run:

```text
ordinary extraction origin parities = [(0,0), (0,1), (1,0), (1,1)]
forced aligned origin parities      = [(0,0)]

ordinary median F770W ratio         = 0.941882
forced-aligned median F770W ratio   = 0.941882

ordinary bright peak residual median       = 0.05023
forced-aligned bright peak residual median = 0.05023

ordinary bright >3% residuals       = 80 / 92
forced-aligned bright >3% residuals = 80 / 92
```

Conclusion: the current F770W residual failure is not a `Cutout2D` parity or
lower-left alignment bug.

## 5. Centroiding

### Known Photutils Centroid Bug

`photutils.centroids.centroid_quadratic` can fail on large mosaics because it
constructs a poorly conditioned quadratic design matrix using absolute pixel
indices. The workaround is to centroid on a local cutout and add the cutout
origin back afterward. This is documented in `GUIDE.md`.

All model/image centroid tests in the realistic diagnostic use local cutouts.

### Fitted Model-Stamp Centroid Test

The final fitted model template stamps were centroided and compared to exact
injected F770W templates and local F770W image cutouts.

All sources:

```text
model - exact true-template radial offset:
  median = 0.092 F770W native pix
  p84    = 0.215 F770W native pix
  p95    = 0.337 F770W native pix
  >0.5 pix = 8 / 263
  >1.0 pix = 2 / 263

model - local noiseless image radial offset:
  median = 0.108 F770W native pix
  p84    = 0.347 F770W native pix
  p95    = 1.468 F770W native pix
  >0.5 pix = 36 / 263
  >1.0 pix = 20 / 263
```

Bright sources (`SNR >= 50`):

```text
model - exact true-template radial offset:
  median = 0.066 F770W native pix
  p84    = 0.120 F770W native pix
  p95    = 0.156 F770W native pix
  max    = 0.279 F770W native pix
  >0.5 pix = 0 / 92
  >1.0 pix = 0 / 92

model - local noiseless image radial offset:
  median = 0.076 F770W native pix
  p84    = 0.142 F770W native pix
  p95    = 1.025 F770W native pix
  >0.5 pix = 7 / 92
  >1.0 pix = 5 / 92
```

The exact-template comparison is the clean placement test. It compares one
fitted model stamp to one injected source template and shows no pixel-scale
scatter for the bright sources. Larger image-cutout outliers are expected in
combined images because blends and local contamination shift image centroids.

Conclusion: the bright-source residual morphology is not explained by
pixel-scale scatter in the final fitted model-template centroids.

## 6. Realistic Template Results

### No-Extension Baseline

Command:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-extension none \
  --kernel-grid-nside 4 \
  --out scratch/codex_debug_realistic_300_centroid_diagnostic
```

Result:

```text
requested sources                              = 300
fitted sources                                 = 263
template extension                             = none
fit astrometry iterations                      = 0
median F444W template support fraction          = 0.89698
median F444W self-fit flux ratio                = 0.90523
median F770W extracted-template flux ratio      = 0.94188
SNR>=50 F770W extracted-template median ratio   = 0.95517
extracted model/truth integer shift             = (0, 0)
F770W source-pixel pull std, r<=3 pix           = 1.70025
F770W source-aperture |pull| 95%, r<=3 pix      = 1.10486
bright peak residual median                     = 5.023% of source peak
bright peak residual 95%                         = 9.302% of source peak
bright sources above 3% peak residual            = 80 / 92
```

This fails. It is not close to the hard requirement that source-stamp residuals
be noise-like.

### No-Kernel F444W Control

Same 263 sources, same segmentation, same template dilation, but F444W fitted
to F444W with `kernel=None`:

```text
median F444W flux ratio                 = 0.90523
source-pixel pull std, r<=3 pix         = 0.000151
source-aperture |pull| 95%, r<=3 pix    = 0.00245
source dipole median                    = 0.0000018
bright >3% peak residuals               = 0 / 92
```

This proves the extracted templates are placed consistently when no
PSF-matching transfer is required. The scalar flux is low because the
finite-support template is unit-normalized and captures only about 90% of the
source flux; the source-stamp placement itself is clean.

### Exact F770W True-Template Reference

Exact injected F770W templates fitted directly to F770W:

```text
noisy exact-F770W-template median flux ratio        = 0.99913
noiseless exact-F770W-template median flux ratio    = 1.00000
noisy exact-F770W-template source-pixel pull std    = 0.97690
noisy exact-F770W-template aperture |pull| 95%      = 0.90888
noiseless exact-F770W residual std/noise            = 0.00003
true-template model/truth integer shift             = (0, 0)
```

This passes. The true-template diagnostic image is clean:

```text
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2_true_templates.png
```

This proves the F770W image, F770W true templates, source placement, residual
construction, and solver are capable of noise-like residuals.

## 7. Why The Residuals Can Be Asymmetric

Asymmetric residual pulls do not automatically imply a pixel-shift bug. In the
current no-extension path, the template support is defined by thresholded and
deblended segmentation footprints. Those footprints are not symmetric source
models:

```text
segmentation truncates wings at the detection threshold
deblending assigns overlap pixels discretely
neighbor contamination is masked or assigned asymmetrically
unit normalization preserves the asymmetric finite support
convolution spreads that asymmetric support into the F770W basis
```

The centroid tests show that the final fitted model stamps are not displaced by
about a pixel relative to the exact source templates for bright objects. The
residual shape is therefore a template-shape/support/ownership failure, not a
confirmed astrometric shift failure.

## 8. Diagnostics That Should Remain Standard

Keep these diagnostics in every 300-source realistic validation:

```text
flux-ratio plot with SNR<20 and SNR>=20 residual histograms
point-source markers in flux-ratio plots
bright-source residual-pull stamp mosaic
true-template F770W image diagnostic
model-centroid shift table and plot
source-level residual metrics:
  source-pixel pull std
  aperture pull p95
  core-minus-ring pull
  dipole metric
  peak residual fraction for SNR>=50
```

The diagnostic metadata should stay readable and not be placed as clipped title
text. Kernel diagnostics should remain separation-of-concerns plots and should
not include detection/template metadata.

## Recommendations

### Immediate Recommendation: Implement Template Ownership Extension

Implement model-weighted template ownership outside the segmentation footprint:

```text
ownership_i(x, y) = model_i(x, y) / sum_j model_j(x, y)
```

Use this ownership to extend each source template beyond the hard segmentation
mask using a controlled observed/model hybrid. This directly addresses the
actual failure mode: hard segmentation masks truncate wings and assign blended
pixels discretely.

Recommended first implementation:

```text
1. Build provisional per-source models from PSF/Gaussian or simple profile fits.
2. Define an extension zone around each segment, not a full-image operation.
3. Compute fractional ownership in the extension zone.
4. Fill missing template support using owned high-resolution image pixels where reliable.
5. Fall back to model wings where high-resolution pixels are too noisy or blended.
6. Normalize final extracted/completed templates to unit sum.
7. Re-run the 300-source acceptance test.
```

### Secondary Recommendation: Keep PSF/Gaussian Completion As A Benchmark

PSF/Gaussian wing completion is useful as a quick diagnostic and baseline, but
it should not be accepted as the final production solution unless it passes the
same source-stamp residual tests. It is likely too simple for blended or
extended sources.

The simplest zero-fill PSF-wing completion has now been implemented and tested
with 300 sources:

```text
template_dilate_segmap=0
template_extension=psf_wings
median F770W ratio = 1.00239
SNR>=50 median F770W ratio = 1.00423
source-pixel pull std = 1.75195
bright SNR>=50 peak residual median = 5.147%
bright SNR>=50 peak residual p95 = 9.163%
bright SNR>=50 peak residual >3% = 81/92
```

This confirms the expectation: simple PSF-wing completion largely fixes the
median flux scale, but it does not make the fitted source-stamp residuals
noise-like. It should remain a benchmark, not the final production answer.

### Longer-Term Recommendation: Two-Component Templates

For real galaxies and color gradients, introduce linked components:

```text
component A: observed high-resolution core/template
component B: analytic or PSF-derived wings/profile
```

This is more flexible but increases degeneracy, so it should follow after the
ownership model is stable.

### Do Not Move This Into DrizzlePSF

Do not solve template-support or segmentation ownership by changing DrizzlePSF.
DrizzlePSF owns PSF projection and flux conservation. It should not know about
segmentation, source templates, photometry, or residual fitting.

### Do Not Use Unwindowed Kernel Fits

Do not replace the windowed kernel with an unwindowed Fourier-ratio kernel.
That path can fit high-frequency sampling/noise artifacts and is not a valid
solution for a 5-10% source-template residual problem.

## Acceptance Criteria Going Forward

A future fix should not be accepted unless all of the following pass:

```text
requested sources >= 300
template_extension mode recorded explicitly
exact/noiseless F770W true-template median ratio = 1.00000
noisy exact F770W true-template source residuals are noise-like
no-kernel F444W control has no source-stamp offset residuals
production F770W median flux ratio is within 1-2%
SNR>=50 production F770W median flux ratio is within 1-2%
production F770W source-pixel pull std is near 1
production F770W source-aperture |pull| p95 is near the true-template reference
bright-source peak residuals are not systematically >3%
fitted model-template centroid offsets for SNR>=50 are not pixel-scale
diagnostics include readable detection/template/kernel metadata
```

Current no-extension baseline fails the production F770W flux and residual
criteria. Exact F770W true templates pass. Therefore the next fix must improve
template construction, not the solver, PSF projection, or kernel matching.

## 2026-05-03 Native-Pixel Phase Sampling A/B

The single-frame F444W/F770W mock was compared against a regenerated mock with
eight deterministic native-pixel phase dithers per filter:

```text
command difference: --n-dither 8
requested sources: 300
position mode: isolated
template_dilate_segmap: 0
template_extension: psf_wings
F770W target PSF: drizzle/STPSF response convolved with 0.08 arcsec FWHM
```

Result:

```text
metric                                    single-frame   8-dither
n_fit                                             259        258
segmentation_blended                               0          0
median F770W ratio                           0.98657    0.98713
SNR>=50 median F770W ratio                   0.98792    0.98703
median integer model/truth shift              (0,0)      (0,0)
source-pixel pull std                        2.21744    1.45480
source-aperture |pull| p95                   1.41089    1.61411
bright SNR>=50 peak residual median          6.187%     4.379%
bright SNR>=50 peak residual p95            12.083%     9.163%
bright SNR>=50 peak residual >3%              90/93      75/93
F770W true-template source-pixel pull std    0.96310    0.97777
true-F444W-through-kernel pull std           2.22625    1.46164
```

Conclusion: better native phase sampling reduces the F444W-through-kernel
source residual excess, but it does not make the production source stamps
noise-like. The exact F770W true-template path stays noise-like in both runs.
The residual failure therefore cannot be explained by the solver, by
segmentation blending in this isolated run, or by a random one-pixel placement
offset. The remaining failure is tied to the F444W-through-kernel template
basis.

The apparent smaller area in `diagnostic_sigma_2*.png` is intentional display
cropping in the validation script: `_save_image_diagnostic` center-crops the
panels to at most 1200 F444W pixels across, with the F770W panels scaled by the
40/80 mas pixel ratio. `mock_mosaic.png` shows the full generated footprint.
