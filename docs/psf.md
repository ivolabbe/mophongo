# PSF models and matching kernels

Mophongo fits low-resolution images with templates derived from a
high-resolution detection image, so every band needs a PSF model and a
convolution kernel that transforms the high-resolution PSF into the
low-resolution one. Three classes provide this machinery:

- {class}`mophongo.psf.PSF` wraps a discrete PSF stamp and builds matching
  kernels (windowed Fourier ratio, Tikhonov, Wiener, ForWaRD), including
  optimizers for the window and regularization parameters.
- {class}`mophongo.psf.DrizzlePSF` drizzles STDPSF-format effective PSF
  (ePSF) grids through the per-exposure WCS onto the mosaic grid, so the
  model PSF at any position reflects the actual exposure stack, roll angles,
  and drizzle parameters.
- {class}`mophongo.psf_factory.PSFFactory` generates STDPSF grids with
  `stpsf` for the observation epoch, via the backend utilities in
  `mophongo.jwst_psf`.

Spatially varying PSFs across a mosaic are handled by region maps; see
{doc}`psf_maps`. The pipeline wires these pieces together automatically; see
{doc}`pipeline`. Kernel and PSF diagnostics are described in
{doc}`diagnostics`.

## Shape versus throughput convention

Fitting uses unit-sum PSF *shapes*. A realistic pixelated PSF stamp sums to
less than one because flux exists outside the finite stamp; that stamp sum is
kept separately as filter-level throughput metadata. The consequences:

- Pipeline-facing code normalizes PSF stamps to unit sum before building
  matching kernels or extending templates, and records the native stamp sum.
- {func}`mophongo.utils.matching_kernel` does not normalize its inputs or
  output: if `sum(psf_lo) / sum(psf_hi)` is not one, that ratio propagates
  into `sum(kernel)`. Pass unit-sum shapes unless you deliberately want a
  throughput-carrying diagnostic kernel.
- {meth}`mophongo.pipeline.Pipeline.run` writes raw fitted template
  amplitudes as `flux_<i>` and throughput-corrected totals as
  `flux_<i>_total` (see {doc}`outputs`).

## The `PSF` class

{class}`mophongo.psf.PSF` is a small dataclass wrapping a pixel-grid PSF
stamp (`array`, with optional `wcs` and `pos` metadata). Analytic stamps come
from {meth}`~mophongo.psf.PSF.moffat`, {meth}`~mophongo.psf.PSF.gaussian`,
and {meth}`~mophongo.psf.PSF.delta` (a single unit pixel, useful as an
identity PSF); {meth}`~mophongo.psf.PSF.from_array` wraps an existing pixel
array unchanged, and {meth}`~mophongo.psf.PSF.from_data` cuts a stamp out of
an image around a star — given in pixel or sky coordinates — optionally
recentering on the quadratic centroid and carrying the cutout WCS.

```python
import numpy as np
from mophongo.psf import PSF

psf_hi = PSF.moffat(101, fwhm_x=2.5, fwhm_y=2.5, beta=2.8)  # detection band
psf_lo = PSF.moffat(101, fwhm_x=6.0, fwhm_y=6.0, beta=2.8)  # broader band
print(psf_hi.array.shape, round(psf_hi.array.sum(), 4), round(psf_lo.array.sum(), 4))  # (101, 101) 1.0 0.9997
```

### Building a matching kernel

{meth}`~mophongo.psf.PSF.matching_kernel` returns the convolution kernel `k`
such that `psf_hi * k ≈ psf_lo` (as `float32`), zeroing non-finite pixels
and zero-padding PSFs of unequal shape to a common grid first. The `method`
argument selects the regularization — `"window"` (the default), `"tikhonov"`,
`"wiener"`, or `"forward"`, described below — with a single dimensionless
scalar, `reg` (default `1e-3`), controlling the non-window methods.

```python
kernel = psf_hi.matching_kernel(psf_lo, method="wiener", reg=3e-4)
```

The underlying function is {func}`mophongo.utils.matching_kernel`, which adds
a `pixel_ratio` argument for PSFs sampled on different pixel scales: the pair
is brought onto a common grid with flux-conserving cubic interpolation before
Fourier inversion, keeping integer scale ratios registered with the
pipeline's nested block grids. Kernel convolution throughout mophongo uses
{func}`mophongo.utils.fftconvolve`, which crops the full convolution from
`kernel.shape // 2` so odd- and even-sized centered kernels follow the same
convention (SciPy's `mode="same"` is offset by one pixel for even kernels).

```python
import numpy as np
from mophongo.psf import PSF
from mophongo.utils import fftconvolve

psf_hi = PSF.moffat(101, fwhm_x=2.5, fwhm_y=2.5, beta=2.8)
psf_lo = PSF.moffat(101, fwhm_x=6.0, fwhm_y=6.0, beta=2.8)
kernel = psf_hi.matching_kernel(psf_lo, method="tikhonov", reg=1e-5)
matched = fftconvolve(psf_hi.array, kernel)
print(f"{np.abs(matched - psf_lo.array).max():.1e}")  # 7.1e-07
```

### Regularization methods

**`"window"`**
Direct Fourier ratio `H_lo / H_hi` apodized by a window function.
  Accepted aliases: `"scb"`, `"split_cosine_bell"`, `"tukey"`. The window
  object is any callable of the stamp shape, typically photutils'
  `SplitCosineBellWindow(alpha, beta)` (`beta` is the flat inner fraction,
  `alpha` the tapered fraction) or `TukeyWindow`.

**`"tikhonov"`** (alias `"ridge"`)
`K = conj(H_hi) H_lo / (|H_hi|^2 + lambda)` with
  `lambda = reg * max(|H_hi|^2)`.

**`"wiener"`**
`K = conj(H_hi) P_xx H_lo / (|H_hi|^2 P_xx + lambda)`. With the default
  flat `signal_psd` this is mathematically identical to Tikhonov; pass an
  explicit prior (e.g. `|H_lo|^2`) to differ.

**`"forward"`** (aliases `"forwardrd"`, `"fourier_wavelet"`)
ForWaRD Fourier + wavelet regularized deconvolution (Neelamani et al.
  2004): a Tikhonov Fourier inverse followed by stationary-wavelet hard
  thresholding and an optional wavelet-domain Wiener step.

### Optimizing the kernel parameters

All optimizers share one figure of merit: mean squared encircled-energy
(growth-curve) mismatch, plus mean squared log radial-profile mismatch inside
a core radius, plus a kernel-stability penalty
`R(K) = w_hf * HF(K) + w_c * C(K)^2`, where `HF` is the fraction of kernel
Fourier power near the Nyquist scale and `C = sum(|K|)/|sum(K)| - 1`
measures positive/negative flux cancellation.

{meth}`~mophongo.psf.PSF.auto_matching_kernel_window` is the production entry
point for the window method: it grid-searches the split-cosine-bell
parameters and returns the `SplitCosineBellWindow` to pass to
{meth}`~mophongo.psf.PSF.matching_kernel`. A named figure-of-merit preset
(`fom`, default `"c2"`: growth MSE + core MSE + `1e-3 * C(K)^2`) sets the
stability weights, and `diagnostic_path` writes the standard PNG diagnostic.
It wraps {meth}`~mophongo.psf.PSF.optimize_matching_kernel_window`, the
underlying grid search with explicit stability weights, which returns a
{class}`mophongo.psf.MatchingKernelWindowFit` carrying the best window
parameters, kernel, matched PSF, and the full score and error-component
grids.

For the regularized methods,
{meth}`~mophongo.psf.PSF.optimize_matching_kernel_regularization` scans the
scalar regularization parameter over a logarithmic grid using the same
figure of merit and returns a {class}`mophongo.psf.MatchingKernelRegFit`
with the best `reg`, kernel, matched PSF, and scan vectors. Its
`diagnostic_path` option writes the standard diagnostic figure; use it
rather than inventing ad hoc kernel diagnostics.

```python
import numpy as np
from mophongo.psf import PSF

rng = np.random.default_rng(11)
psf_hi = PSF.moffat(101, fwhm_x=2.5, fwhm_y=2.5, beta=2.8)
target = PSF.moffat(101, fwhm_x=6.0, fwhm_y=6.0, beta=2.8).array
target += rng.normal(0, 2e-5, target.shape)   # a realistically noisy target
fit = psf_hi.optimize_matching_kernel_regularization(
    PSF.from_array(target), method="wiener")
print(f"best reg = {fit.reg:.2e}, kernel {fit.kernel.shape}")  # best reg = 5.62e-03, kernel (101, 101)
```

```{figure} images/kernel_diagnostic.png
:width: 100%
:alt: Regularization-scan diagnostic with FOM scan, profiles, growth-curve ratio, PSF stamps, kernel, matched PSF, and residual

The standard diagnostic written by `optimize_matching_kernel_regularization`
(here a Wiener scan matching an F444W PSF to a blurred F770W target). The top
row shows the figure-of-merit scan over lambda with the selected minimum, the
matched versus target radial profiles, and their growth-curve ratio against
the ±2% band; below are the source and target PSFs, the kernel, the matched
PSF, and the residual, with the scan configuration and error terms listed.
```

{meth}`~mophongo.psf.PSF.matching_kernel_basis` builds an alternative kernel
from a linear fit to a stack of kernel basis images
({func}`mophongo.utils.fit_kernel_fourier`).

```python
import numpy as np
from mophongo.psf import PSF

psf_hi = PSF.from_array(stamp_hi / stamp_hi.sum())  # unit-sum shapes
psf_lo = PSF.from_array(stamp_lo / stamp_lo.sum())

window = psf_hi.auto_matching_kernel_window(psf_lo, diagnostic_path="diag/")
kernel = psf_hi.matching_kernel(psf_lo, window=window)
```

## `DrizzlePSF`: model PSFs on the mosaic grid

{class}`mophongo.psf.DrizzlePSF` reproduces the mosaic PSF at any sky
position by evaluating the loaded ePSF grid in each contributing exposure
frame and drizzling the stamps onto the output WCS with the same kernel and
`pixfrac` as the mosaic (read from the mosaic header keywords `KERNEL` and
`PIXFRAC` when present).

A `DrizzlePSF` is constructed from the drizzled mosaic FITS (`driz_image`,
whose header defines the output WCS and pixel scale) and a per-exposure WCS
table, the `*_wcs.csv` written alongside mophongo mosaics. The CSV path is
derived from the mosaic filename when not given, and a missing CSV is
reconstructed from public MAST cal-file headers; the static helper doing the
parsing is {meth}`~mophongo.psf.DrizzlePSF.read_wcs_csv`. After construction
two public attributes expose the exposure geometry: `DrizzlePSF.footprint`,
a dict mapping each frame key to its sky-footprint `shapely` Polygon, and
`DrizzlePSF.driz_footprint`, the mosaic outline Polygon. Region maps are
built from these footprints ({doc}`psf_maps`).

```python
from mophongo.psf import DrizzlePSF

dpsf = DrizzlePSF(driz_image="uds_f444w_drz_sci.fits")
dpsf.load_jwst_stdpsf(local_dir="data/PSF", filter_pattern="F444W")
```

### Loading ePSF grids

{meth}`~mophongo.psf.DrizzlePSF.load_jwst_stdpsf` forwards to
`EffectivePSF.load_jwst_stdpsf`, which fills the `epsf` dictionary keyed by
the STDPSF filename stem (basename without `.fits`). In the local-directory
mode the pipeline uses, it recursively loads the `*.fits` files under
`local_dir` whose basenames match the `filter_pattern` regex, applying a
cosine edge taper (4 native pixels by default) to remove finite-grid edge
discontinuities without renormalizing the loaded planes. Without both
arguments it downloads library STDPSF files from the STScI JWST1PASS
archive instead.

Each grid stores its spatial knot positions (`IPSFX*`/`JPSFY*` header
keywords) and oversampling factor (`OVERSAMP`, default 4); local-directory
loads also record the epoch (`MJD-AVG`) that drives the lookup described
below. `EffectivePSF.get_at_position(x, y, filter, rot90=0)`
bilinearly interpolates the tile grid to a detector position, and
`eval_ePSF` evaluates the oversampled stamp at sub-pixel offsets with cubic
interpolation.

### MJD-aware key resolution

When `get_psf` looks up the ePSF for an exposure, the user-supplied filter
pattern is resolved per frame: a literal `NRC..` token is replaced by the
NIRCam detector decoded from the exposure filename (so one pattern covers
all SCAs), the pattern is regex-matched against loaded keys, and when
several epochs match, the grid whose stored MJD is nearest the frame's
`MJD-AVG` is chosen. Loading several `..._MJD{int}_...` grids per filter
therefore gives each exposure the wavefront model closest to its epoch.

### Producing PSF stamps

{meth}`~mophongo.psf.DrizzlePSF.get_psf` drizzles the PSF model at one sky
position onto an output WCS (default: the full mosaic grid). The returned
stamp is the finite-integral flux landing on the output footprint; it is
deliberately not renormalized (see the shape/throughput convention above).

{meth}`~mophongo.psf.DrizzlePSF.get_psf_radec` is the workhorse for building
per-region PSF cubes ({doc}`psf_maps`): it returns a `(N, size, size)` cube
of drizzled PSFs at a list of `(ra, dec)` positions. The stamp size can be
given in arcsec or output pixels, or derived from a requested absolute
encircled-energy fraction, `ee_fraction`; derived sizes are rounded up to a
multiple of 0.160 arcsec so that mosaics on the nested 20/40/80 mas pixel
ladder share integer pixel ratios for clean block-binning. After each call
the delivered (not requested) stamp properties — `psf_size`, `ee_box`,
`ee_circ` and friends — are measured on the returned cube and stored on the
instance; re-measure if you modify the cube afterwards.

```python
cube = dpsf.get_psf_radec(positions, ee_fraction=0.90)
```

The measurement itself is {func}`mophongo.psf.stamp_encircled_energy`, which
works on one stamp or a whole cube: its `ee_box` (the full square-stamp sum)
is the quantity that converts a fitted template amplitude into a total flux,
while `ee_circ` (the inscribed-circle sum) is what to compare against
tabulated encircled-energy curves.

```python
import numpy as np
from mophongo.psf import PSF, stamp_encircled_energy

wide = PSF.gaussian(101, fwhm=10.0).array   # unit flux on the large grid
stamp = wide[38:63, 38:63]                  # 25x25 crop truncates the wings
ee = stamp_encircled_energy(stamp, pscale=0.04)
print(f"ee_box={ee['ee_box']:.3f}  ee_circ={ee['ee_circ']:.3f}")  # ee_box=0.994  ee_circ=0.987
```

{meth}`~mophongo.psf.DrizzlePSF.get_driz_cutout` returns an
`astropy.nddata.Cutout2D` of the mosaic (with WCS) around a position — the
cutout WCS is what `get_psf` drizzles onto — and
{meth}`~mophongo.psf.DrizzlePSF.register` iteratively shifts the model
position until the drizzled PSF centroid matches the data centroid,
returning `((ra, dec), data_cutout, psf_model)` — useful for verifying
astrometric registration against isolated stars.

## `PSFFactory`: generating PSF grids

{class}`mophongo.psf_factory.PSFFactory` is a configure-once dataclass that
dispatches PSF construction to a telescope backend registered in
`mophongo.psf_factory.BACKENDS` (currently JWST via
{class}`mophongo.jwst_psf.JWSTBackend`; new telescopes register a backend
implementing the same small protocol). The factory holds the grid layout and
sampling defaults, the output directory, and the epoch-selection defaults;
every field can be overridden per call. Saved grids follow the canonical
filename order `{prefix}_{DET}_{FILT}[_MJD{int}]_GRID{N}_{OS4|DET}.fits`,
and stripping the MJD token yields the key used by `DrizzlePSF` for
nearest-MJD lookup.

{meth}`~mophongo.psf_factory.PSFFactory.build` builds one grid explicitly
for a given telescope/instrument/filter/detector, using the wavefront model
(OPD) nearest the requested date.
{meth}`~mophongo.psf_factory.PSFFactory.from_csv` builds every grid a mosaic
needs from its per-exposure `*_wcs.csv` listing: telescope, instrument,
filter, and detector list are decoded from the CSV, and one file is produced
per `(detector, date)` pair, skipping existing files unless `overwrite` is
set. The epoch selection behind it is
{func}`~mophongo.psf_factory.dates_from_csv`, which turns the CSV's
`mjd-avg` column into one or more grid dates via the `date_mode` setting —
`"modal"` (densest window, the default), `"median"`/`"mean"`, `"cluster"`
(one date per epoch cluster), `"all"`, or a literal date used as-is.

```python
from mophongo.psf_factory import PSFFactory

fac = PSFFactory(prefix="MYPROJ", outdir="data/PSF", num_psfs=9)
fac.from_csv("image_wcs.csv", date_mode="cluster", save=True)

grid = fac.build(telescope="JWST", instrument="NIRCAM",
                 detector="NRCA5", filter="F444W", date=60000.0)
```

### Automatic grid generation and per-band blur defaults

The pipeline calls `PSFFactory.from_csv` itself when its ePSF loader finds
no files matching the configured filename pattern and `psf_autobuild` is on
(the default); see {doc}`pipeline`. The pipeline can also broaden the
low-resolution model PSF by an extra Gaussian before kernel construction
(`psf_blur_fwhm`): the `"default"` setting looks the FWHM up per filter in
`mophongo.mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC`, which carries MIRI
defaults (0.08 arcsec at F560W/F770W rising to 0.30 arcsec at F2100W, no
broadening for unlisted filters), accounting for the broadening of real
MIRI mosaics relative to the optical `stpsf` model. Production runs should
keep this default blur on for MIRI bands; disable it (`psf_blur_fwhm=None`)
only when deliberately testing the unblurred optical model, for example
when comparing drizzled model PSFs against real stars to measure the
broadening itself.

```{figure} images/star_vs_model_f770w.png
:width: 100%
:alt: Real F770W star compared with the drizzled model PSF, with residuals, radial profiles, and growth curves

A drizzled F770W model PSF compared with a real star: star cutout, model,
and residuals before and after convolving the model with a small Gaussian
diffusion kernel, with radial profiles and growth curves below. The
unmodified model growth curve runs a few percent low in the core; the
blurred model tracks the star to within a percent at all radii.
```

```{figure} images/miri_blur_growth_curves.png
:width: 100%
:alt: Stacked star-to-model growth-curve ratios per band, before and after a single per-band Gaussian blur

Stacked star-to-model growth-curve ratios per band. Against the raw model
(top) the ratio falls below one in the core, increasingly so toward longer
MIRI wavelengths; a single per-band Gaussian blur (bottom, best-fit FWHM
rising from 0.03 arcsec at F444W to 0.25 arcsec at F1800W) flattens the
median ratio, which is what the per-filter blur defaults encode.
```

## `jwst_psf` utilities

{func}`~mophongo.jwst_psf.build_jwst_psf` is the low-level `stpsf` wrapper
returning a `photutils.psf.GriddedPSFModel` for one instrument/filter
(/detector for NIRCam), using the measured wavefront OPD nearest `date` and
recording the epoch as `MJD-AVG`.

{func}`~mophongo.jwst_psf.blend_psf` blends an empirical PSF core into a
theoretical halo with a linear taper and matched enclosed flux.

{func}`~mophongo.jwst_psf.make_extended_grid` applies
{func}`~mophongo.jwst_psf.blend_psf` to every position of an empirical
`STDPSFGrid`, attaching `stpsf` theoretical wings out to a given radius.

{func}`~mophongo.jwst_psf.write_stdpsf` writes a grid (a
`GriddedPSFModel`/`STDPSFGrid` or a raw cube with knot arrays) in the
STDPSF FITS format that `EffectivePSF.load_jwst_stdpsf` reads.

The full signatures of everything above are in the {doc}`api` reference.
