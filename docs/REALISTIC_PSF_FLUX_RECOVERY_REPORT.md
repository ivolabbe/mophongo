# Realistic PSF Flux Recovery Report

Date: 2026-05-03

## Executive Summary

The current production path is **not** using any template extension. It uses
F444W templates extracted from the segmentation map, normalizes each extracted
cutout to unit sum, then optionally convolves those templates for the lower
resolution band.

The suspected odd/even `Cutout2D` lower-left alignment bug was tested directly
and is **not** the origin of the current >3% F770W source-stamp residuals.
Forcing every extracted template onto an even/even lower-left origin changes
the template basis only at numerical roundoff levels after the production
convolution/projection path, and it leaves the 300-source F770W residual
metrics unchanged.

The decisive result is:

- `AlignedCutout` exists, but `Pipeline.run` does **not** use it.
- The production path uses `Templates.extract_templates(...)`, which creates
  `Template(...)`, a subclass of `astropy.nddata.Cutout2D`.
- `Template.extract_templates + Template.convolve_cutout` matches full-image
  convolution for all lower-left origin parities to `1e-12` in unit tests.
- The no-kernel F444W self-fit has no source-stamp offset residuals.
- The F770W exact true-template fit is clean.
- The current F444W-segmentation-template to F770W fit still fails because the
  extracted segmentation templates are incomplete source models for F770W.

The bug to fix next is template support/ownership outside the segmentation
footprint, not `Cutout2D` parity and not the windowed PSF matching kernel.

Update on the F770W PSF convention: the true mock F770W PSF is now defined as
the F770W drizzle/STPSF response convolved with a 0.08 arcsec FWHM Gaussian.
That is 2 pixels on the 40 mas reference grid and 1 native F770W pixel. The
blur is applied on the 40 mas sampling before returning to the requested
F770W-grid PSF. It is not "two F770W pixels."

## Code Path Audit

Current pipeline extraction path:

```text
Pipeline.run
  -> Templates.extract_templates(...)
       -> Template(hires_image, pos, (height, width), ...)
            -> astropy.nddata.Cutout2D(...)
  -> Templates.convolve_templates(kernel, ...)
       -> Template.convolve_cutout(...)
       -> mophongo.utils.fftconvolve(..., mode="full")
```

`AlignedCutout` is defined in `src/mophongo/templates.py`, but it is not used
by `Pipeline.run` or `Templates.extract_templates`. It is therefore not the
active production path.

The active no-convolution path is also clean now:

```text
kernels=[None, None] reaches Templates.convolve_templates(kernel=None)
identity/dummy delta convolution is no longer used
```

## Tests Run

### 1. Isolated Cutout2D + Convolution Regression

Added/ran:

```bash
poetry run pytest tests/test_template_convolution.py -q
```

Result:

```text
5 passed
```

The new regression test exercises the production path:

```text
Templates.extract_templates(...)
Templates.convolve_templates(...)
Template.convolve_cutout(...)
```

It uses:

```text
all four lower-left origin parities: (0,0), (0,1), (1,0), (1,1)
an asymmetric even-sized kernel
asymmetric source footprints
comparison against full-image fftconvolve(..., mode="same")
```

Measured agreement:

```text
max absolute difference <= 1e-12
```

This directly falsifies the hypothesis that the current
`Template(Cutout2D)` extraction plus production convolution operation
introduces a parity-dependent pixel shift.

### 2. 300-Source No-Extension Baseline

Command:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-extension none \
  --kernel-grid-nside 4 \
  --out scratch/codex_debug_realistic_300_no_extension_baseline
```

Important products:

```text
scratch/codex_debug_realistic_300_no_extension_baseline/realistic_gaussian_summary.txt
scratch/codex_debug_realistic_300_no_extension_baseline/sigma_2/source_table_sigma_2.csv
scratch/codex_debug_realistic_300_no_extension_baseline/sigma_2/diagnostic_sigma_2.png
scratch/codex_debug_realistic_300_no_extension_baseline/sigma_2/psf_kernel_window_diagnostic.png
```

The current diagnostic run with centroid plots is:

```text
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2_true_templates.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/model_centroid_shifts_sigma_2.png
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/model_centroid_shifts_sigma_2.csv
```

Baseline result:

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

This fails the hard residual requirement. The scalar fluxes are also biased
low in the no-extension baseline, consistent with the extracted templates
capturing only about 90% of the F444W source flux before unit normalization.

### 3. No-Convolution F444W Control

Same 263 detected sources, same segmentation, same template dilation, but:

```text
image fit: F444W -> F444W
kernel: None
template extension: none
```

Result:

```text
fitted sources                                      = 263
median F444W flux ratio                             = 0.90523
source-pixel pull std, r<=3 pix                     = 0.000151
source-aperture |pull| 95%, r<=3 pix                = 0.00245
source dipole median                                = 0.0000018
source dipole 95%                                   = 0.000036
bright peak residual median                         = 0.00010% of source peak
bright peak residual 95%                            = 0.00014% of source peak
bright sources above 3% peak residual                = 0 / 92
```

This proves the ordinary extracted templates are placed consistently when no
PSF-matching convolution is required. The F444W scalar ratio is low because
the extracted segmentation template is a unit-normalized finite-support
template, not a total-flux template. The source-stamp residuals, however, do
not show the offset failure.

### 4. Exact F770W True-Template Reference

The exact F770W source templates are the same finite source templates that
were injected into the F770W mock image. This is the reference for the solver,
source placement, and F770W residual-noise requirement.

Result in the same 300-source run:

```text
noisy exact-F770W-template median flux ratio        = 0.99913
noiseless exact-F770W-template median flux ratio    = 1.00000
noisy exact-F770W-template source-pixel pull std    = 0.97690
noisy exact-F770W-template aperture |pull| 95%      = 0.90888
noiseless exact-F770W residual std/noise            = 0.00003
true-template model/truth integer shift             = (0, 0)
```

This passes. Therefore the F770W image, F770W template placement, linear
solver, and residual construction are not the cause of the large residuals.

The true-template image diagnostic was produced explicitly:

```text
scratch/codex_debug_realistic_300_centroid_diagnostic/sigma_2/diagnostic_sigma_2_true_templates.png
```

Its residual image and bright-source residual-pull stamps are noise-like. This
is the expected behavior: if the exact injected F770W templates are fitted
directly to the F770W image, there is no coherent source residual.

## Fitted Model-Stamp Centroid Test

To test the remaining offset concern directly, the final fitted model template
stamp for each source was centroided and compared to:

```text
1. the exact injected F770W source-template centroid
2. the local noiseless F770W mock-image cutout centroid
3. the local noisy F770W data-image cutout centroid
```

Centroids were measured only on local cutouts, not on full mosaics, to avoid
the known large-index centroid failure mode. The model centroid uses the
positive core of the final fitted F444W->F770W template stamp that actually
enters the residual image.

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

The exact-template comparison is the clean placement test because it compares
one fitted model stamp to one injected source template. It shows no pixel-scale
scatter for the bright sources. The larger image-cutout outliers come from
local blends and centroid contamination in the combined image, not from a
global fitted-stamp placement shift.

Conclusion: the asymmetric bright-source residual pulls are not explained by
pixel-scale centroid scatter in the fitted model stamps. They remain a
template-shape/support problem.

### 5. Forced Aligned-Cutout A/B

A scratch A/B was run on the same 263 fitted sources and same 300-source mock:

```text
A: ordinary Templates.extract_templates output
B: manually rebuilt templates with every lower-left origin forced to even/even
```

The aligned templates were then sent through the same production
`convolve_templates(...)` and upsampled-grid projection path.

Origin parities:

```text
ordinary extraction origin parities = [(0,0), (0,1), (1,0), (1,1)]
forced aligned origin parities      = [(0,0)]
```

Full-grid basis comparison:

```text
extracted-template full-grid max abs diff            = 7.704e-04
extracted-template full-grid L1 diff                 = 6.086e-03
convolved/projected full-grid max abs diff           = 1.068e-05
convolved/projected full-grid L1 diff                = 3.901e-03
```

Fit/residual comparison:

```text
ordinary median F770W ratio                          = 0.941882
forced-aligned median F770W ratio                    = 0.941882
ordinary bright peak residual median                 = 0.05023
forced-aligned bright peak residual median           = 0.05023
ordinary bright peak residual 95%                    = 0.09302
forced-aligned bright peak residual 95%              = 0.09302
ordinary bright sources above 3%                     = 80 / 92
forced-aligned bright sources above 3%               = 80 / 92
```

This is the decisive alignment test. If the lower-left origin parity were the
cause of the F770W residuals, forcing all origins to even/even would change the
model basis and the residual metrics. It does not.

## Kernel Diagnostic Status

The kernel diagnostic is now kept separate from detection/template metadata.
It reports only the PSF-matching problem. The current diagnostic title is:

```text
F444W -> F770W PSF matching kernel, region 0
```

The windowed kernel is not the current failure mode. The kernel diagnostic
shows the matched F444W PSF and F770W PSF agree at the fractional-percent level
in the relevant radial/growth metrics. The current residual problem is much
larger than that and survives kernel-separation controls in the way described
above.

Do not use an unwindowed Fourier-ratio kernel as a fix. It reduces some PSF
stamp residuals by fitting high-frequency sampling/noise structure and is not
a valid production solution.

## Diagnosis

The current >3% F770W residuals are not caused by:

```text
Cutout2D odd/even lower-left origin parity
the removed dummy identity convolution
the current Template.convolve_cutout placement convention
the F770W true-template fit path
the linear solver
integer astrometric shifts
pixel-scale scatter in the final fitted model-template centroids
kernel diagnostic metadata or plotting
```

The remaining failure is the F444W-derived template model supplied to the
F770W fit. In the no-extension baseline, the segmentation templates contain
only finite, detected F444W support. They are then unit-normalized. That makes
the template core too important relative to missing wings and neighboring
unassigned flux. After F444W->F770W convolution this produces coherent
bright-source residuals even though the integer shift metric is zero.

The residuals can look asymmetric because the segmentation support is itself
asymmetric: thresholded/deblended masks truncate source wings differently on
different sides and assign blended pixels discretely. Unit-normalizing that
finite asymmetric support preserves placement but changes the profile and wing
ownership. The centroid plot verifies that this is not a one-pixel placement
failure for the bright sources.

This also explains why the no-kernel F444W control can have clean stamp
placement while the F770W convolved fit fails: the no-kernel case is fitting
the exact same finite-support F444W basis back to the F444W image, while the
F770W case requires that finite-support F444W basis to predict the full F770W
source morphology.

## Template Extension Options

These are design options only. None is implemented as the production solution
in the baseline above.

### Option 1: PSF/Gaussian Wing Completion

Fit a compact PSF-convolved analytic model to pixels inside the trusted
segmentation footprint, then use it to fill only the missing support outside
the segment.

Implementation ease: highest.

Accuracy: useful for stars and compact simple galaxies. It is a good
diagnostic baseline, but it will not handle blends, complex galaxies, or color
morphology robustly by itself.

### Option 2: Model-Weighted Template Ownership

Build provisional source models and assign pixels outside the segmentation map
by fractional ownership:

```text
ownership_i(x, y) = model_i(x, y) / sum_j model_j(x, y)
```

Then extend each template using either the observed high-resolution image, a
model, or an observed/model hybrid weighted by that ownership.

Implementation ease: moderate.

Accuracy: best next production target. It directly addresses the actual
failure: hard segmentation masks truncate wings and make background/neighbor
pixels unowned or exclusively owned by one source.

### Option 3: Two-Component Source Templates

Represent each source with linked components:

```text
component A: observed high-resolution segmentation/core template
component B: PSF/analytic wing or profile template
```

The fit can solve the components with constraints or priors.

Implementation ease: lowest.

Accuracy: likely best long-term option for real galaxies and wavelength
morphology changes, but it increases degeneracy with neighboring sources and
background unless constrained carefully.

Recommended path: implement Option 2 first. It is the most direct response to
the observed failure and should improve both isolated-source wings and crowded
source ownership. Option 1 is useful as a fast benchmark. Option 3 should come
after ownership is stable.

## Acceptance Criteria

Do not call the realistic test fixed unless all of these pass:

```text
requested sources >= 300
template_extension=none baseline documented separately from any extension A/B
all model/truth integer shift medians are zero
no-kernel F444W control has no source-stamp offset residuals
exact/noiseless F770W true-template median ratio = 1.00000
noisy F770W true-template source residual pulls are noise-like
production F770W template path median fluxes agree within 1-2%
production F770W bright-source stamp pulls are noise-like
diagnostics include detection sigma, npixels, filter, catalog dilation,
truth-label remap dilation, pipeline template ndilate, template-extension mode,
kernel grid nside, and astrometry iterations
```

Current status:

```text
Cutout2D/convolution parity regression            PASS
no-kernel F444W source-stamp placement             PASS
F770W exact true-template reference                PASS
F770W no-extension extracted-template flux         FAIL
F770W no-extension bright-source residual pulls    FAIL
F770W psf_wings, ndilate=0 median flux             PASS
F770W psf_wings, ndilate=0 bright residual pulls   FAIL
```

The next implementation should target template support and ownership outside
the segmentation map. It should not change `DrizzlePSF`, should not move this
responsibility into PSF generation, and should not replace the windowed kernel
with an unwindowed Fourier-ratio kernel.

## PSF-Wing Completion A/B

After this report was first written, the simplest PSF-wing completion option
was implemented and tested directly:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-dilate-segmap 0 \
  --template-extension psf_wings \
  --kernel-grid-nside 4 \
  --out scratch/codex_debug_realistic_300_psf_wings_nd0
```

Implementation convention:

```text
extract templates with template_dilate_segmap=0
trust existing template zero pixels as outside-segment support
convolve the already-extracted template with the raw local high-resolution PSF
replace zero-valued template pixels with the convolved values
normalize the completed template to unit sum
```

The local high-resolution PSF is intentionally not unit-normalized for this
operation.

Result:

```text
n_fit = 263
median F770W ratio = 1.00239
SNR>=50 median F770W ratio = 1.00423
median integer model/truth shift = (0, 0)
source-pixel pull std = 1.75195
source-aperture |pull| p95 = 1.33067
bright SNR>=50 peak residual median = 5.147%
bright SNR>=50 peak residual p95 = 9.163%
bright SNR>=50 peak residual >3% = 81/92
```

Conclusion: this PSF-wing completion is useful because it fixes the median
flux scale, but it does not satisfy the residual hard requirement. The
remaining failure is not an overall missing-flux scalar; it is still a
template-shape/ownership problem in the F770W fitting basis.

## F770W 0.08 Arcsec PSF Convention Check

The F770W PSF broadening convention was corrected after the first `psf_wings`
run. The previous implementation could interpret the requested value as two
native F770W pixels, i.e. 0.16 arcsec, which is not the intended target PSF.

The corrected implementation stores the convention in angular units:

```text
psf_gaussian_fwhm_arcsec["f770w"] = 0.08
psf_gaussian_fwhm_pix["f770w"]    = 1.0  # native 80 mas F770W pixels
psf_gaussian_fwhm_pix on 40 mas grid = 2.0
```

Regression coverage:

```bash
poetry run pytest \
  tests/test_mock_mosaic.py::test_mock_mosaic_psf_blur_preserves_native_edge_loss \
  tests/test_mock_mosaic.py::test_mock_mosaic_f770w_blur_is_sampled_on_40mas_grid \
  tests/test_pipeline.py::test_pipeline_accepts_prebuilt_templates \
  tests/test_template_convolution.py -q
```

Result:

```text
10 passed
```

The second test distinguishes the corrected path from a direct native-grid
one-pixel Gaussian convolution by comparing against a manual 40 mas upsample,
Gaussian blur, and block-sum back to the 80 mas grid.

The 300-source isolated `psf_wings` validation was rerun with the corrected
F770W PSF:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-dilate-segmap 0 \
  --template-extension psf_wings \
  --kernel-grid-nside 4 \
  --position-mode isolated \
  --position-min-sep-pix 40 \
  --out scratch/codex_debug_realistic_300_psf_wings_isolated_f770blur008
```

Result:

```text
n_fit = 259
segmentation_blended = 0
segmentation_unblended = 259
truth catalog F770W PSF Gaussian FWHM = 0.08 arcsec = 1.0 native F770W pix
median F770W ratio = 0.98657
SNR>=50 median F770W ratio = 0.98792
median integer model/truth shift = (0, 0)
source-pixel pull std = 2.21744
source-aperture |pull| p95 = 1.41089
bright SNR>=50 peak residual median = 6.187%
bright SNR>=50 peak residual p95 = 12.083%
bright SNR>=50 peak residual >3% = 90/93
```

Conclusion: the F770W PSF definition is now correct and tested, but this
correction does not solve the bright-source residual hard failure. The
remaining problem is still in the F444W-derived template basis used for the
F770W fit.

## Native-Pixel Phase Sampling A/B

The coarse-sampling hypothesis was tested by regenerating the 300-source
isolated validation with eight deterministic native-pixel phase dithers per
filter:

```bash
poetry run python scratch/run_realistic_mosaic_gaussian_sweep.py \
  --sigmas 2 \
  --nsrc 300 \
  --fit-astrometry-niter 0 \
  --template-dilate-segmap 0 \
  --template-extension psf_wings \
  --kernel-grid-nside 4 \
  --position-mode isolated \
  --position-min-sep-pix 40 \
  --n-dither 8 \
  --out scratch/codex_debug_realistic_300_psf_wings_isolated_dither8
```

Single-frame versus 8-dither result:

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

Conclusion: native-pixel phase sampling is a real contributor to the
source-stamp residual excess, but it is not the full origin. The F770W
true-template control remains noise-like, while both the production
F444W-derived templates and the true-F444W-through-kernel control retain an
excess. The remaining failure therefore belongs to the F444W-through-kernel
template basis, not to the linear solver, segmentation blending, or a random
one-pixel model-placement offset.
