# PSF Shape and Throughput Convention

## Problem

Real pixelated PSF stamps always have finite support. A useful JWST PSF stamp
may contain roughly 95% of the total light, with the remaining flux living in
very broad wings outside the modeled pixel array. Treating that missing flux as
part of the convolution kernel has made the pipeline hard to reason about:
unit-normalized source templates, native-sum PSFs, and finite-support target
PSFs can silently mix incompatible amplitude conventions.

## Adopted Convention

Mophongo separates PSF shape from PSF throughput.

- `psf_shape`: the finite PSF stamp divided by its own finite-stamp sum. This
  is unit-sum and is used for morphology, template extension, PSF matching
  kernels, and convolution.
- `psf_throughput`: the original finite-stamp sum. This records the enclosed
  fraction of the total PSF flux represented by the stamp.

The low-level PSF arrays are still allowed to preserve native finite-support
sums. The normalization happens at the point where a PSF is used as a shape
basis for fitting.

## Fitting Rule

The fitted template amplitude is a modeled-image flux in the finite-support
basis. If the science image was generated or calibrated against a native PSF
whose finite stamp has throughput `f`, then the total-flux estimate is:

```text
flux_total = flux_model / f
```

For flux-recovery plots against a truth catalog whose `flux_true` is the total
source flux before finite-stamp loss, compare `flux_total` to `flux_true`.

## Kernel Rule

PSF matching kernels used by the template-fitting pipeline should be built from
unit-sum shapes:

```text
shape_hi = psf_hi_native / sum(psf_hi_native)
shape_lo = psf_lo_native / sum(psf_lo_native)
kernel = matching_kernel(shape_hi, shape_lo)
```

The kernel should have a DC sum near one. Any deviation should reflect numerical
or regularization effects, not the missing 5% of out-of-stamp PSF flux.

The low-level `mophongo.utils.matching_kernel` routine intentionally preserves
whatever sums the caller passes in. That behavior is useful for diagnostics and
controlled tests, but pipeline callers should pass unit-sum PSF shapes unless
they explicitly want a throughput-carrying kernel.

## Template Extension Rule

Template extraction normalizes each source template to unit sum. Therefore PSFs
used for `psf_wings` or `psf_model` template extension must also be treated as
unit-sum shapes. The native PSF sum is diagnostic metadata only; it must not
alter the relative amount of filled wing light.

## Mock Data Rule

Mock generation may preserve native PSF finite-support sums when painting
sources. In that case, the image contains `flux_true * psf_throughput` inside
the modeled stamp. Pipeline recovery should report both:

- `flux_model`: fitted modeled-stamp amplitude.
- `flux_total`: `flux_model / psf_throughput`.

This keeps simulations realistic while keeping the fitting algebra simple.

## Practical Check

For a stable run:

- native F444W and F770W PSF sums may be around `0.95-0.98`;
- unit PSF shapes used for kernels should sum to `1`;
- the matching kernel sum should be near `1`;
- raw modeled flux ratios may sit near the target throughput;
- throughput-corrected flux ratios should be near unity when the model is
  otherwise unbiased.
