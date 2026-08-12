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

{class}`mophongo.psf.PSF` is a dataclass holding a pixel-grid PSF:

- `array` (*np.ndarray*) — the PSF stamp; converted to `float` on
  construction. Also available as the `data` property.
- `wcs` (*astropy.wcs.WCS or None*, default `None`) — optional WCS of the
  stamp.
- `pos` (*tuple[float, float] or None*, default `None`) — position of the
  PSF center in cutout coordinates, set by {meth}`~mophongo.psf.PSF.from_data`.

### Constructors

**`PSF.moffat(size, fwhm_x, fwhm_y, beta, theta=0.0)`**
Normalized elliptical Moffat profile.
  - `size` (*int or (int, int)*) — stamp shape; a scalar gives a square.
  - `fwhm_x`, `fwhm_y` (*float*) — FWHM along each axis in pixels.
  - `beta` (*float*) — Moffat power-law index.
  - `theta` (*float*, default `0.0`) — rotation angle in radians.

**`PSF.gaussian(size, fwhm=None, theta=0.0)`**
Normalized elliptical Gaussian profile.
  - `size` (*int or (int, int)*) — stamp shape.
  - `fwhm` (*float or (float, float)*, default `None`) — FWHM in pixels; a
    2-tuple is interpreted as `(fwhm_x, fwhm_y)`.
  - `theta` (*float*, default `0.0`) — rotation angle in radians.

**`PSF.delta(size=3)`**
Single unit pixel at the stamp center; useful as an identity PSF.
  - `size` (*int*, default `3`) — side length; should be odd so the delta
    pixel is centered.

**`PSF.from_array(array)`**
Wrap an arbitrary pixel array without modification.

**`PSF.from_data(data, position=None, *, search_boxsize=None, fit_boxsize=5, size=51, wcs=None, verbose=False)`**
Extract a PSF stamp from an image around a star.
  - `data` (*np.ndarray*) — image containing the star.
  - `position` (*(x, y) in pixels, or (ra, dec) astropy Quantities with
    `wcs`*, default `None`) — approximate star position; `None` uses the
    image center.
  - `search_boxsize` (*int or (int, int) or None*, default `None`) — search
    box for `photutils.centroids.centroid_quadratic`; `None` skips
    recentering entirely.
  - `fit_boxsize` (*int or (int, int)*, default `5`) — centroid fit box.
  - `size` (*int*, default `51`) — square cutout side in pixels.
  - `wcs` (*WCS or None*, default `None`) — image WCS, required for sky
    coordinates and propagated to the cutout.
  - `verbose` (*bool*, default `False`) — print centroid information.

### Building a matching kernel

**`PSF.matching_kernel(other, window=None, *, recenter=False, method="window", reg=1e-3, wavelet="db4", levels=3, threshold_factor=3.0, noise_sigma=None, forward_wavelet_wiener=True, signal_psd=None)`**
Return the convolution kernel `k` such that `psf_hi * k ≈ psf_lo`
  (returned as `float32`). Non-finite pixels are zeroed and PSFs of unequal
  shape are zero-padded to a common grid first.
  - `other` (*PSF or np.ndarray*) — target (low-resolution) PSF.
  - `window` (*callable or None*, default `None`) — Fourier-domain window
    for `method="window"`; defaults to
    `photutils.psf.matching.SplitCosineBellWindow(alpha=0.4, beta=0.1)`.
  - `recenter` (*bool*, default `False`) — shift the kernel to its centroid
    (center-of-mass first guess refined by a quadratic centroid, cubic
    interpolation with zero-padding so the shift conserves flux).
  - `method` (*str*, default `"window"`) — one of `"window"`,
    `"tikhonov"`, `"wiener"`, `"forward"` (details below).
  - `reg` (*float*, default `1e-3`) — regularization parameter for the
    Tikhonov, Wiener, and ForWaRD methods, scaled internally by the peak of
    the inversion denominator (`max(|H_hi|^2)`; `max(|H_hi|^2 P_xx)` for
    Wiener) so it is dimensionless.
  - `wavelet` (*str*, default `"db4"`), `levels` (*int*, default `3`),
    `threshold_factor` (*float*, default `3.0`), `noise_sigma` (*float or
    None*, default `None`), `forward_wavelet_wiener` (*bool*, default
    `True`) — ForWaRD-only options; `threshold_factor` sets the hard
    threshold on wavelet detail coefficients in units of per-subband noise,
    and `forward_wavelet_wiener=False` skips the final wavelet-domain
    Wiener step.
  - `signal_psd` (*np.ndarray or None*, default `None`) — signal power
    spectral density prior for `method="wiener"`.

The underlying function is {func}`mophongo.utils.matching_kernel`, which
accepts the same arguments plus `pixel_ratio` (*float*, default `1.0`): when
the two PSFs are sampled on different pixel scales, one of them is rescaled by
that factor with flux-conserving cubic interpolation
({func}`mophongo.utils.resize_flux_conserving_inter_cubic`) before Fourier
inversion. A ratio above one upsamples `psf_lo` onto the finer grid, which is
how the pipeline passes it (the low-to-high pixel-scale ratio); a ratio below
one downsamples `psf_hi` instead. The resize uses the same pixel-extent
convention as the pipeline's nested block grids, so integer scale ratios stay
registered. Kernel convolution throughout mophongo uses
{func}`mophongo.utils.fftconvolve`, which crops the full convolution from
`kernel.shape // 2` so odd- and even-sized centered kernels follow the same
convention (SciPy's `mode="same"` is offset by one pixel for even kernels).

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

**`PSF.auto_matching_kernel_window(other, *, fom="c2", alpha_grid=None, beta_grid=None, grid_oversample=2, core_radius=None, growth_weight=1.0, core_weight=1.0, l2_weight=0.0, reg_lambda=1e-3, kernel_high_frequency_radius=0.7, recenter=False, diagnostic_path=None, source_label="source PSF", target_label="target PSF", diagnostic_title=None, aperture_radius=None, return_result=False)`**
Production entry point: grid-search the split-cosine-bell window and
  return the `SplitCosineBellWindow` to pass to
  {meth}`~mophongo.psf.PSF.matching_kernel`.
  - `fom` (*str*, default `"c2"`) — figure-of-merit preset; `"c2"` is an
    alias for `"growth_core_cancel"` (growth MSE + core MSE +
    `1e-3 * C(K)^2`). Other presets: `"growth_core_only"`,
    `"growth_core_hf"`, `"growth_core_hf_cancel"`.
  - `alpha_grid`, `beta_grid` (*np.ndarray or None*) — explicit search
    grids; defaults span alpha 0.02–0.90 and beta 0.05–0.95, refined by
    `grid_oversample` (*int*, default `2`).
  - `core_radius` (*float or None*) — core-profile radius in pixels;
    default is a quarter of the stamp side.
  - `growth_weight`, `core_weight`, `l2_weight` (*float*, defaults `1.0`,
    `1.0`, `0.0`) — weights of the growth-curve, core-profile, and
    image-space MSE terms.
  - `reg_lambda` (*float*, default `1e-3`) — overall weight of the kernel
    stability term.
  - `kernel_high_frequency_radius` (*float*, default `0.7`) — Fourier
    radius in Nyquist units above which kernel power counts as ringing.
  - `recenter` (*bool*, default `False`) — passed through to the kernel
    builder.
  - `diagnostic_path` (*str or Path or None*) — write the standard PNG
    diagnostic (score grid, radial profiles, growth-curve ratio, kernel,
    matched PSF, residual). A directory path writes
    `diagnostic_window.png` inside it.
  - `return_result` (*bool*, default `False`) — also return the full
    `MatchingKernelWindowFit`.

**`PSF.optimize_matching_kernel_window(...)`**
The underlying grid search; same search and weighting parameters, but with
  explicit `kernel_regularization_weight`, `kernel_high_frequency_weight`,
  and `kernel_cancellation_weight` instead of a named preset, and without
  the `fom`, diagnostic, and `return_result` options. Returns a
  {class}`mophongo.psf.MatchingKernelWindowFit` dataclass carrying the best
  `alpha`, `beta`, `score`, `kernel`, and `matched_psf`, the 2-D score and
  error-component grids evaluated over (`alpha_grid`, `beta_grid`), and the
  target/matched radial-profile and growth-curve vectors.

**`PSF.optimize_matching_kernel_regularization(other, *, method="tikhonov", reg_grid=None, pixel_ratio=1.0, core_radius=None, growth_weight=1.0, core_weight=1.0, l2_weight=0.0, kernel_regularization_weight=1e-3, kernel_high_frequency_radius=0.7, kernel_high_frequency_weight=0.0, kernel_cancellation_weight=1.0, recenter=False, wavelet="db4", levels=3, threshold_factor=3.0, noise_sigma=None, forward_wavelet_wiener=True, signal_psd=None, diagnostic_path=None, source_label="source PSF", target_label="target PSF", diagnostic_title=None, aperture_radius=None, diagnostic_note=None)`**
1-D scan of the scalar regularization parameter for
  `method="tikhonov" | "wiener" | "forward"`, using the same figure of
  merit. `reg_grid` defaults to `np.logspace(-6, -1, 21)`. `pixel_ratio`
  brings the pair onto a common grid before scanning, exactly as
  `matching_kernel` would. Passing `diagnostic_path` writes the standard
  diagnostic figure, named `diagnostic_<method>.png` when the path is a
  directory; use this rather than ad hoc diagnostic figures. Returns a
  {class}`mophongo.psf.MatchingKernelRegFit` with `method`, the best `reg`,
  `score`, `kernel`, `matched_psf`, the 1-D scan grids, the profile vectors,
  and an `extra` dict recording the scan configuration.

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

**`DrizzlePSF(flt_files=None, info=None, driz_image=None, driz_hdu=None, full_flt_weight=True, csv_file=None, epsf_obj=None)`**
- `driz_image` (*str*) — path to the drizzled mosaic FITS; its header
    defines the output WCS and pixel scale.
  - `csv_file` (*str or None*, default `None`) — per-exposure WCS table
    (`*_wcs.csv`). When `None`, the path is derived from `driz_image` by
    replacing the `_drz_sci/_drc_sci/_sci` suffix with `_wcs.csv`; a
    missing CSV is reconstructed from public MAST cal-file headers.
  - `info` (*tuple or None*, default `None`) — pre-parsed
    `(flt_keys, wcs, footprints, headers)` as returned by
    {meth}`~mophongo.psf.DrizzlePSF.read_wcs_csv`; bypasses the CSV read.
  - `driz_hdu` (*HDU or None*, default `None`) — use this HDU's header
    instead of reading `driz_image` from disk.
  - `epsf_obj` (*EffectivePSF or None*, default `None`) — ePSF container;
    a fresh empty {class}`mophongo.psf.EffectivePSF` is created when
    `None`.
  - `flt_files`, `full_flt_weight` — accepted for backward compatibility;
    as of this writing neither is used (exposure files come from the CSV,
    and per-frame weights come from the `EXPTIME` column).

After construction two public attributes expose the exposure geometry:
`DrizzlePSF.footprint`, a dict mapping each frame key to its sky-footprint
`shapely` Polygon, and `DrizzlePSF.driz_footprint`, the mosaic outline
Polygon. Region maps are built from these footprints ({doc}`psf_maps`).

{meth}`~mophongo.psf.DrizzlePSF.read_wcs_csv` is the static helper that
parses the WCS CSV into `(flt_keys, wcs_dict, footprints, headers)`,
regenerating a missing CSV from public archive headers by default.

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

**`DrizzlePSF.get_psf(ra, dec, filter=None, pixfrac=0.75, kernel="square", verbose=False, wcs_slice=None, get_extended=True, get_weight=False, ds9=None, npix=None, xphase=0, yphase=0, taper_alpha=None, return_hdul=False)`**
Drizzle the PSF model at one position onto `wcs_slice` (default: the full
  mosaic WCS). The returned stamp is the finite-integral flux landing on
  the output footprint; it is deliberately not renormalized (see the
  shape/throughput convention above).
  - `filter` (*str or None*) — ePSF key or regex; `None` reuses the
    pattern given at load time.
  - `pixfrac` (*float*, default `0.75`), `kernel` (*str*, default
    `"square"`) — drizzle parameters, used exactly as passed; the mosaic
    header override (`KERNEL`/`PIXFRAC`) is applied by the callers
    `get_psf_radec` and `register`, not by `get_psf` itself.
  - `wcs_slice` (*WCS or None*) — output WCS defining the stamp footprint.
  - `get_extended` (*bool*, default `True`) — add the extended halo model
    if one is loaded for the filter.
  - `npix` (*int or None*) — half-size of the evaluated input grid in
    detector pixels; `None` derives it from the output footprint plus a
    half-`pixfrac` margin.
  - `xphase`, `yphase` (*float*, default `0`) — extra sub-pixel phase
    offsets applied to the ePSF evaluation.
  - `taper_alpha` (*float or None*, default `None`) — apply a Tukey taper
    of this alpha over the drizzled footprint to suppress edge
    discontinuities.
  - `return_hdul` (*bool*, default `False`) — return a FITS HDUList with
    the stamp WCS instead of a bare array.
  - `get_weight`, `ds9` — accepted but as of this writing unused.

**`DrizzlePSF.get_psf_radec(positions, *, filter=None, size=None, ee_fraction=None, size_quantum_arcsec=0.160, parity="even", verbose=False, kernel="square", pixfrac=0.75)`**
Return a `(N, size, size)` cube of drizzled PSFs at a list of `(ra, dec)`
  positions — the workhorse for building per-region PSF cubes (see
  {doc}`psf_maps`).
  - `size` (*float, int, or None*) — stamp size; a float is arcsec, an int
    is output pixels, `None` derives the size from `ee_fraction` or falls
    back to the native ePSF stamp size.
  - `ee_fraction` (*float or None*) — request the diameter enclosing this
    absolute encircled-energy fraction of the PSF (the ePSFs are
    absolutely calibrated, so a request above the finite-stamp total
    raises). The size is rounded up to a multiple of
    `size_quantum_arcsec` (*float*, default `0.160`, i.e. 2 × 80 mas) so
    that mosaics on a nested 20/40/80 mas pixel ladder share integer pixel
    ratios for clean block-binning.
  - `parity` (*{"even", "odd", "any"}*, default `"even"`) — parity of the
    pixel size; `"even"` keeps the stamp parity stable under integer
    block replication, `"odd"` centers the stamp on the requested
    position.
  - After each call the delivered (not requested) stamp properties are
    stored on the instance, measured on the returned cube by
    {func}`mophongo.psf.stamp_encircled_energy`: `psf_size` (arcsec side),
    `ee_box`, `ee_circ`, `r_circ`, `r_ee`, and `ee_fraction_request`.
    Re-measure if you modify the cube afterwards.

**`DrizzlePSF.get_driz_cutout(ra, dec, size=None, size_native=None, recenter=False, search_boxsize=11, fit_boxsize=5, cutout_data=None, verbose=False)`**
Return an `astropy.nddata.Cutout2D` of the mosaic (with WCS) around a
  position, optionally recentered on the local quadratic centroid; the
  cutout WCS is what `get_psf` drizzles onto. `size` is in output pixels;
  `size_native` in detector pixels; `cutout_data` substitutes an in-memory
  array (or list of arrays) for the mosaic file.

{meth}`~mophongo.psf.DrizzlePSF.register` iteratively shifts the model
position until the drizzled PSF centroid matches the data centroid,
returning `((ra, dec), data_cutout, psf_model)` — useful for verifying
astrometric registration against isolated stars.

**`stamp_encircled_energy(psf, pscale, *, ee_fraction=None, per_stamp=False)`**
Measure the realized encircled energy of one stamp or a cube. Returns a
  dict with `ee_box` (full square-stamp sum — the quantity that converts a
  fitted amplitude into a total flux), `ee_circ` (sum inside the inscribed
  circle), `r_circ`, and `r_ee` (radius enclosing `ee_fraction`, `nan` when
  not requested or not reached). With `per_stamp=True` the values are
  returned per cube plane instead of averaged.

## `PSFFactory`: generating PSF grids

{class}`mophongo.psf_factory.PSFFactory` is a configure-once dataclass that
dispatches PSF construction to a telescope backend registered in
`mophongo.psf_factory.BACKENDS` (currently JWST via
{class}`mophongo.jwst_psf.JWSTBackend`; new telescopes register a backend
implementing the same small protocol). Factory fields (all defaults
overridable per call):

- `prefix` (*str*, default `"STDPSF"`) — filename prefix (project tag).
- `outdir` (*str or None*, default `None`) — output directory for saved
  FITS grids; created on demand.
- `num_psfs` (*int*, default `1`) — PSFs per grid; must be a perfect square
  laid out across the detector.
- `oversample` (*int*, default `4`) — pixel-space oversampling.
- `fov_arcsec` (*float or None*, default `None`) — field of view per PSF;
  `None` omits the keyword so `stpsf` applies its own pixel-based default
  (4.09 arcsec for NIRCam, 8.10 for MIRI).
- `use_detsampled_psf` (*bool*, default `False`) — write detector-sampled
  rather than oversampled PSFs.
- `date_mode` (*str, float, or astropy Time*, default `"modal"`) — default
  epoch-selection mode (below).
- `span` (*float*, default `5.0`) — window width in days for `"modal"`.
- `delta_day` (*float*, default `2.0`) — cluster radius in days for
  `"cluster"`.
- `include_mjd` (*bool*, default `True`) — embed `_MJD{int}` in saved
  filenames; the canonical order is
  `{prefix}_{DET}_{FILT}[_MJD{int}]_GRID{N}_{OS4|DET}.fits`, and stripping
  the MJD token yields the key used by `DrizzlePSF` for nearest-MJD lookup.
- `overwrite` (*bool*, default `False`), `verbose` (*bool*, default
  `False`).

**`PSFFactory.build(*, telescope, instrument, filter, detector=None, date=None, num_psfs=None, oversample=None, fov_arcsec=None, use_detsampled_psf=None, save=False, **backend_kw)`**
Build one PSF grid explicitly. `date` may be an MJD float, ISO string, or
  `astropy.time.Time`; when given, the wavefront model (OPD) nearest that
  epoch is used. `save=True` writes the grid to `outdir` under the
  canonical filename.

**`PSFFactory.from_csv(csv_path, *, detector=None, date_mode=None, span=None, delta_day=None, num_psfs=None, oversample=None, fov_arcsec=None, use_detsampled_psf=None, save=True)`**
Build every grid needed for a mosaic from its per-exposure `*_wcs.csv`
  listing: the telescope and instrument are decoded from the exposure
  filenames in the `file` column, the filter from the CSV filename, and the
  detector list from the backend (every NIRCam SCA that sees the filter,
  otherwise the detector decoded from the filenames). One file is produced
  per `(detector, date)` pair. Existing files are skipped unless `overwrite`
  is set. `date_mode` may also be an iterable of modes to combine.

{func}`~mophongo.psf_factory.dates_from_csv` performs the epoch selection
behind `from_csv`, turning the CSV's `mjd-avg` column into one or more grid
dates via the `"modal"`, `"median"`/`"mean"`, `"cluster"`, or `"all"` modes
(or a literal date, used as-is).

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
