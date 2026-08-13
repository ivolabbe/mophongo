# Precision and memory

Mophongo runs on mosaics large enough that the width of an intermediate array
decides whether a field fits in memory at all. A MINERVA UDS detection grid is
34560 x 25344 = 876 Mpx, so one full-field array is 3.5 GB at float32 and
7.0 GB at float64, and a full-field F770W fit holds 138,610 templates whose
pixels come to 5.95 GB per set. This page records which arrays are float64 on
purpose, which are float32 on purpose, and how to tell the difference when
adding code.

## The rule

**Anything held at scale is single precision. Use float64 only where the extra
precision is needed.**

"At scale" means either one large array -- an image, a weight map, a scene
plane -- or a large number of small ones, which in practice means stamps: a
full field carries 138,610 templates, so a stamp dtype is a field-scale
decision even though each stamp is 40 kB.

The precision *is* needed wherever the array **is** the arithmetic of a solve:
a normal-equation matrix, its whitening or factorisation, a covariance or error
propagation, or WCS and sky arithmetic. Everywhere else -- anything that only
*stores* values derived from float32 pixels and is consumed by a reduction or
written into a float32 array -- narrow it.

Two mechanical consequences that any narrowing has to respect.

### Narrowing storage requires widening the accumulator

`np.sum` on a float32 array accumulates in float32. Narrowing a buffer without
annotating the reduction that consumes it trades a real memory saving for a
silent precision loss:

```python
wi = float(np.sum(cut_i * w_i * cut_i, dtype=np.float64))
```

The `dtype=np.float64` is not decoration. Without it these entries carry about
seven significant digits into a float64 normal matrix, and the whitening,
Cholesky and `spsolve` chain downstream inherits that. `np.dot` and the BLAS
paths are worse: they cannot be annotated at all, so a float32 operand there
fixes the accumulator width too.

### NEP 50: Python scalars are weak, numpy scalars are strong

```python
py_float   * arr32   ->  float32     # weak scalar adopts the array dtype
np.float64 * arr32   ->  float64     # strong scalar promotes the array
```

This decides more than it looks. `Template.flux` is an element of the solve
result, so it is an `np.float64`; `t.flux * t.data` therefore forms a float64
product and rounds once when accumulated into a float32 buffer. Wrapping it as
`float(t.flux)` would compute the product in float32 and round twice, for no
saving -- the temporary is one stamp either way. Reach for `float()` to control
promotion only when the array being promoted is large.

### scipy promotes to the wider operand

`fftconvolve(float32_image, float64_kernel)` runs the whole transform at
float64. Narrowing one side alone is a no-op; both have to match, or the width
has to be forced at the point of use. Two places in the codebase were paying
this: the detection convolution, where `Gaussian2DKernel(...).array` is always
float64 (25.4 GB peak against 14.5 GB matched, on the 876 Mpx grid), and the
per-template convolution against a float64 matching-kernel cube.

## What is float64 on purpose

Do not narrow these. They are the arithmetic of the solve, and a float32 solve
on a normal matrix was measured at 0.2-29% relative error across condition
numbers 5.7e4 to 8.0e6.

| Where | What |
|---|---|
| `scene_fitter.build_normal` | `ata`, `atb` -- the normal matrix and its RHS |
| `SceneFitter.solve_flux`, `_solve_flux_and_shifts` | `A_w`, `b_w`, the diagonal whitening, the joint KKT matrix handed to `spsolve` |
| `_solve_flux_and_shifts` | `cholesky(BB)`, `inv(L)`, the un-whitening solve |
| `_solve_flux_and_shifts` | `S_w = A_w - AB_w @ AB_w.T`, the Schur complement inverted for the reported 1-sigma errors; the subtraction is cancellation-prone by construction |
| `SceneFitter._flux_errors` | the inversion itself; its `np.maximum(diag, 1e-12)` floor would silently clamp a float32-corrupted diagonal into a fake tight error bar |
| `assemble_scene_system_AB` | `BB`, `AB`, `bB`, and the `Bq`/`Bl` shift columns -- the `+=` across overlapping anchors rounds *before* any reduction, so a wider accumulator on the final sum cannot recover it |
| `astrometry`, WCS | sky coordinates and shift-field fits |
| `_create_matching_kernel_no_normalize`, `_matching_kernel_tikhonov`, `_matching_kernel_wiener` | the PSF-matching inversion, upcast at entry |

PSF matching deserves its own note, because it is the one place where the
*inputs* are float32 by policy and the *computation* still has to be float64.
Deconvolution divides by an OTF that falls to zero at high frequency, so input
rounding is amplified by exactly the factor the regularisation exists to
bound -- and `numpy.fft` preserves single precision, so float32 stamps would
otherwise carry the whole inversion through in `complex64`. The three matching
functions upcast at entry. The *resulting* kernel is stored float32 like any
other stamp: it is the solve that needs the digits, not the answer.

The flux-block ridge is `1e-6 x median(diag)`, which is below float32's 1.2e-7
epsilon -- at float32 the regularisation would be indistinguishable from noise.

## What is float32 on purpose

Pixel data is float32 from the mosaics inward: science images, weights,
template stamps, residuals, and the per-scene model plane in
`Scene.model_image`. The convolution in `Templates.convolve_templates` matches
the kernel to the stamp width rather than upcasting the stamp.

Two arrays are deliberately *not* narrowed even though they look like
candidates:

- **PSF stamps in a `PSFRegionMap`.** Their finite sums are throughput
  metadata that converts a fitted amplitude to a total flux. At float32 a
  unit-normalised stamp sums to 1 +/- 5e-9, which is enough to push a reported
  encircled energy above 1. The width is matched at the point of use instead.
- **Segmentation maps.** `as_label_array` returns an integer input untouched
  whatever its width or byte order. FITS is big-endian, so a `BITPIX=32` map
  arrives as `>i4` -- which *is* int32, differing only in byte order -- and the
  full-field read hands over a memmap view whose pages are file-backed. Rewriting
  it as native int32 would convert 3.5 GB of reclaimable page cache into
  anonymous memory. Only float segmaps (COSMOS ships float64 labels) are cast,
  in bands, to int32.

## Where the memory actually goes

Worth stating plainly, because it is the thing dtype audits get wrong: on the
fitting path, float64 is not the problem. Everything narrowable there totals
about 0.36 GB. A full-field fit peaks holding **three full template sets at
once** -- the hi-res set, the convolved set, and the pre-shift copies the
astrometric solve keeps -- which is 18.1 GB of a 32.65 GB numpy-allocated peak,
all of it already float32.

The lever that matters is therefore not width but redundancy: how many copies
of the template set are live, and for how long. See `STATUS.md` for the
measured breakdown.

The one large float64 win is on the **detection** path, not the fitting path:
matching the detection kernel to the image width in `Catalog._detect` takes
that convolution from 25.4 GB to 14.5 GB.

## Checklist for new code

- Allocating a buffer whose size scales with the image or a scene bounding box?
  Give it an explicit dtype. `np.zeros(shape)` is float64.
- Narrowing a buffer? Find every reduction that consumes it and add
  `dtype=np.float64`.
- Convolving? Check both operands have the same width, including kernels from
  `astropy.convolution`, which are always float64.
- Multiplying an array by a scalar that came out of a solve? That scalar is an
  `np.float64` and will promote the array.
- Touching a normal matrix, a factorisation or an error estimate? Leave it
  float64.
