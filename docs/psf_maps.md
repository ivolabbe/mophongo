# Spatially varying PSFs: region maps

A mosaic built from many exposures has no single PSF. Each exposure
contributes a PSF whose shape depends on detector, position angle, and epoch,
so the effective PSF at a sky position depends on which exposures cover that
position. Mophongo represents this with {class}`mophongo.psf_map.PSFRegionMap`:
a partition of the sky into polygonal regions, each covered by a fixed set of
exposures, with one PSF (or matching kernel) stamp per region. Lookups at a
sky position then reduce to a point-in-polygon query followed by an array
index.

Region maps are used in two roles:

- **PSF maps** (one per band): each region carries the drizzled PSF evaluated
  at the region centroid. Built by
  {meth}`mophongo.pipeline.Pipeline.build_psfs`.
- **Kernel maps** (one per fitted band pair): each region of the
  high-resolution/low-resolution overlay carries the PSF-matching kernel for
  that region. Built by {meth}`mophongo.pipeline.Pipeline.build_kernels` and
  consumed during template convolution ({doc}`templates`).

For how single PSFs are drizzled and matched, see {doc}`psf`. For where these
maps sit in the full run, see {doc}`pipeline`.

## How regions are formed

{meth}`~mophongo.psf_map.PSFRegionMap.from_footprints` takes a mapping of
frame identifier to footprint polygon (in sky coordinates, degrees) and
intersects the footprints sequentially. Every distinct set of overlapping
frames becomes one region, stored in a `geopandas.GeoDataFrame` with a
`frame_list` column and an integer `psf_key` label. Three tolerances control
the geometry cleanup:

- polygons are buffered out and back by `buffer_tol` to seal gaps narrower
  than twice that value, then snapped to a `snap_tol` grid;
- regions smaller than an area threshold derived from `area_factor` and
  `buffer_tol` are dissolved into the neighbour with the longest shared
  boundary, so slivers from nearly-coincident exposure edges do not each get
  their own PSF.

`psf_key` values are renumbered to run consecutively from 0, and the stamp
cube in `psfs` is indexed by `psf_key`, so the cube must stay aligned with
the regions table.

```python
from mophongo.psf_map import PSFRegionMap

prm = PSFRegionMap.from_footprints(footprints, name="F444W")
key = prm.lookup_key(ra, dec)      # integer psf_key, or None
prm.psfs = stamps                  # (n_regions, ny, nx) cube, indexed by psf_key
psf = prm.get_psf(ra, dec)         # 2-D stamp for that region
fig, ax = prm.plot()               # colored region overview
```

```python
from shapely.geometry import box
from mophongo.psf_map import PSFRegionMap

footprints = {
    "expA": box(10.000, 0.000, 10.010, 0.010),
    "expB": box(10.005, 0.000, 10.015, 0.010),
    "expC": box(10.000, 0.005, 10.010, 0.015),
}
prm = PSFRegionMap.from_footprints(footprints, name="demo")
print(len(prm.regions), prm.lookup_key(10.0075, 0.0025), prm.lookup_key(10.002, 0.012))  # 6 1 5
```

```{figure} images/region_map_tiling.png
:width: 100%
:alt: Region maps for a hi-res and a lo-res band, their kernel overlay, and a zoom.

Region maps built from exposure footprints. Top: a hi-res band map (F444W,
1694 regions) and a lo-res band map (F770W, 294 regions), each polygon one
distinct set of overlapping exposures. Bottom: the hi-by-lo overlay used for
the kernel map (2911 regions), with a 2-arcminute zoom showing how the
pairwise intersections tile the sky.
```

## PSFRegionMap

`PSFRegionMap` is a dataclass. Its fields:

`regions` : `geopandas.GeoDataFrame`, required
: One row per region, with at least a `geometry` column and an integer
  `psf_key` column. Factory-built maps also carry `frame_list` (contributing
  frame identifiers) and `pa_list`. Extra columns round-trip through GeoJSON,
  which the pipeline uses for provenance stamping (see below); tuple-valued
  columns such as `frame_list` come back as their string repr.

`snap_tol` : `float`, default `0.2 / 3600` (degrees)
: Snap grid passed to `shapely.set_precision` during footprint preprocessing.

`buffer_tol` : `float`, default `1.0 / 3600` (degrees)
: Half-width of the ±buffer used to seal gaps narrower than
  `2 * buffer_tol`.

`area_factor` : `float`, default `200.0`
: Scale factor for the minimum region area used by sliver merging. As of this
  writing the two construction paths derive the threshold differently: the
  plain constructor uses `area_factor * buffer_tol`, while
  `from_footprints` uses `area_factor * buffer_tol**2`.

`name` : `str | None`, default `None`
: Human-readable label, used in plot titles and derived-map names.

`footprints` : `Mapping[Hashable, Polygon]`, default `{}`
: The original input footprints, kept for reference.

`psfs` : `np.ndarray | None`, default `None`
: Optional stamp cube of shape `(n_regions, ny, nx)`, indexed by `psf_key`.
  Holds PSF stamps in PSF maps and matching kernels in kernel maps.

`pscale` : `float`, default `1.0`
: Pixel scale of `psfs` in arcsec per pixel. Only sets the units of
  {attr}`~mophongo.psf_map.PSFRegionMap.r_lim`; encircled-energy fractions
  are unaffected.

The spatial index (`tree`, a Shapely `STRtree`) is derived state, rebuilt by
`__post_init__` and again by every factory. `__getstate__` drops `tree` and
`__setstate__` rebuilds it, but the prepared-geometry cache built alongside it
is not dropped, so as of this writing pickling or deep-copying a region map
raises `PicklingError: Prepared geometries cannot be pickled`.

### Building and loading maps

{meth}`~mophongo.psf_map.PSFRegionMap.from_footprints` builds a map from a
`(frame_id -> footprint polygon)` mapping as described above; note its
factory defaults for the geometry tolerances (`snap_tol=0.5/3600`,
`area_factor=100.0`) differ from the dataclass defaults. Passing per-frame
`wcs` together with a position-angle tolerance, `pa_tol`, keys regions by
their set of PA classes instead of the exact frame set, which coarsens the
map; `pa_tol=0` (the default) disables orientation coarsening.

{meth}`~mophongo.psf_map.PSFRegionMap.to_file` writes the regions table
(geometry plus all columns, including provenance) to GeoJSON and, when
`psfs` is set, the stamp cube to a FITS file with the same base name.
{meth}`~mophongo.psf_map.PSFRegionMap.from_geojson` reverses both, but only
`regions` and `psfs` are stored: the tolerances, `pscale`, and `footprints`
come back as constructor defaults unless passed again as keyword arguments,
and `name` is reset to the file's base name.

{meth}`~mophongo.psf_map.PSFRegionMap.plot` draws the regions colored by a
column (default `psf_key`), forwarding extra keywords to
`GeoDataFrame.plot`, and returns `(fig, ax)` with the x-axis inverted (RA
increasing left).

### Looking up the PSF at a position

{meth}`~mophongo.psf_map.PSFRegionMap.lookup_key` returns the integer
`psf_key` at a sky position in degrees (the CRS of `regions`), an O(log N)
STRtree point-in-polygon query. When the point falls inside no region (for
example just outside the mosaic edge) it returns the key of the nearest
region, or `None` with `nearest=False`.
{meth}`~mophongo.psf_map.PSFRegionMap.get_psf` returns the corresponding
2-D stamp `psfs[key]`; a `None`/NaN position or failed lookup logs a warning
and returns the stamp at index 0.

```python
import numpy as np
from shapely.geometry import box
from mophongo.psf import PSF
from mophongo.psf_map import PSFRegionMap

footprints = {"expA": box(10.000, 0.0, 10.010, 0.010),
              "expB": box(10.005, 0.0, 10.015, 0.010)}
prm = PSFRegionMap.from_footprints(footprints)
prm.psfs = np.stack([PSF.gaussian(25, fwhm=2.0 + k).array
                     for k in range(len(prm.regions))])
key = prm.lookup_key(10.0075, 0.005)        # inside the A-B overlap
print(key, prm.get_psf(10.0075, 0.005).shape)  # 1 (25, 25)
```

For a map cached by a run, `mophongo psf <map.geojson> <ra> <dec>` writes
that stamp to FITS with a WCS centered on the position (see
{doc}`diagnostics`).

### Convolving a whole image with a map

A map holds a different stamp in every region, so a mosaic cannot be
convolved in one pass. {meth}`~mophongo.psf_map.PSFRegionMap.convolve_image`
does it region by region: each region is cut out with a border wide enough
that its own pixels see nothing of the cut (`buffer`, by default half the
largest stamp), convolved with that region's stamp, and only the pixels
*inside* the region polygon are written into the output. Cutouts overlap, the
kept pixels never do, so nothing is double-counted and the seams carry no
discontinuity beyond the difference between the two kernels themselves.

This is the operation to reach for when a PSF-matched *image* is wanted rather
than PSF-matched templates — smoothing a detection image to another band's
resolution, for instance, with the same kernel map the fit itself used.

Two entry points, one per level. The method works on arrays and needs the
image's WCS, since the region polygons are in degrees:

```python
from astropy.io import fits
from astropy.wcs import WCS
from mophongo.psf_map import PSFRegionMap

prm = PSFRegionMap.from_geojson("out/uds_f770w/uds_f770w_kernel.geojson")
sci = fits.getdata("f444w_sci.fits")
matched = prm.convolve_image(sci, WCS(fits.getheader("f444w_sci.fits")))
```

{func}`mophongo.psf_map.convolve_fits` is the file-level wrapper: it reads the
science image and its WCS, convolves, and writes the result under the input
header plus `CONVMAP` and `CONVNREG` provenance keywords. The map may be an
object or the GeoJSON a run left behind (its `.fits` stamp sidecar is picked
up alongside), so a matched image is one call with nothing loaded by hand:

```python
from mophongo.psf_map import convolve_fits

convolve_fits("f444w_sci.fits", "out/uds_f770w/uds_f770w_kernel.geojson",
              "f444w_matched_f770w.fits")
```

Pixels that fall in no region — outside the footprint the map was built from
— are set to `fill_value` (0 by default), and their count is logged. Regions
are the drizzle-footprint intersections, so a kernel map covers only the
overlap of the two bands: convolving a full hi-res mosaic with a kernel map
leaves everything outside the lo-res band's coverage empty, which is
generally what you want.

`convolve_image` always returns floating-point data. Integer inputs are
promoted rather than truncating the convolution, and non-finite input pixels
are explicitly replaced by zero instead of allowing `+/-inf` to become huge
finite values. For a signed deconvolution kernel, bad pixels should normally
be inpainted or masked with a halo before this call: zero-filled holes ring.

### Matching to a theoretical analytic target

The same map machinery can sharpen an image toward a theoretical target PSF.
{meth}`~mophongo.psf_map.PSFRegionMap.gaussian_psf_map` constructs one
noise-free, unit-sum Gaussian per region. The target's core is phase-matched
to the measured subpixel centroid of the corresponding source PSF by default;
otherwise a fixed array-center target would turn the region-dependent drizzle
phase into an astrometric shift. A larger `shape` zero-pads the finite source
support during inversion and suppresses circular FFT wraparound. It is kernel
support, not a claim that the physical PSF itself is hundreds of pixels wide.
{meth}`~mophongo.psf_map.PSFRegionMap.moffat_psf_map` provides the same path
for a circular Moffat target and records its `beta`. Winged targets are
rendered directly on the requested support before discrete unit
normalization; rendering a small Moffat and then padding it would truncate
and renormalize the intended wings.

{meth}`~mophongo.psf_map.PSFRegionMap.matching_kernel_map` then delegates each
source/target pair to the existing {func}`mophongo.utils.matching_kernel`,
normalizes both PSF shapes and the final kernel to unit sum, and returns a new
kernel map ready for `convolve_image`:

```python
from astropy.io import fits
from astropy.wcs import WCS
from mophongo.psf_map import PSFRegionMap

source = PSFRegionMap.from_geojson("f444w_psf.geojson", pscale=0.04)
target = source.gaussian_psf_map(
    0.10 / source.pscale,  # 0.1 arcsec = 2.5 pixels
    shape=512,             # padded support for an aggressive inverse kernel
    phase_match=True,
)
kernels = source.matching_kernel_map(
    target, method="wiener", reg=1e-3,
)
image = fits.getdata("f444w_sci.fits")
sharpened = kernels.convolve_image(image, WCS(fits.getheader("f444w_sci.fits")))
```

A Moffat sensitivity target uses the identical kernel path:

```python
target = source.moffat_psf_map(
    0.10 / source.pscale,
    beta=2.5,
    shape=160,
    phase_match=True,
)
kernels = source.matching_kernel_map(
    target, method="wiener", reg=2e-4,
)
```

A Moffat is an empirical winged profile, not a physical JWST diffraction
model, so compare its realized encircled energy as well as its core FWHM.
Changing the analytic target cannot restore modes beyond the F444W optical
cutoff.

The strictly positive regularization is a required argument on purpose. The standard automatic
PSF-matching score was tuned for stable *smoothing* kernels and can choose a
broader response when the target is narrower than the source. With no
`signal_psd`, Mophongo's Wiener method uses a flat signal spectrum and is
mathematically the Tikhonov solution. Scan `reg` against the science image;
do not infer the output resolution from the requested Gaussian alone.

Every returned region records diagnostics including `kernel_noise_gain`
(`sqrt(sum(kernel**2))`, the white-noise RMS factor), `kernel_l1`, negative
kernel flux, absolute and fractional edge L1, realized
`response_fwhm_[xy]_pix`, recovered
target peak, negative response flux, normalized L2 PSF residual, and residual centroid
shift. Real drizzle noise is correlated, so its blank-sky RMS and power
spectrum still need to be measured on the convolved image. A sharpened result
should be described as **regularized toward** the target unless the realized
response and noise diagnostics establish otherwise.

For an inverse kernel, a small fractional edge L1 can hide a material tail
when the total kernel L1 is large. Check both the fractional and absolute
outer-edge L1 and repeat selected kernels at doubled support. The UDS driver
marks a scan point support-limited when the fraction exceeds `1e-3` or the
absolute edge L1 exceeds `1e-2`; limited points remain in its CSV but are not
connected into the resolution/noise curve.

The site-local real-data driver `examples/run_uds_f444w_deconvolution.py`
applies this path to a 1024-pixel MINERVA UDS F444W patch, scans the
resolution/noise tradeoff, and writes the selected FITS images, target/kernel
maps, CSV metrics, and diagnostic figures. The mosaics and production PSF map
are too large for the repository: provide a local MINERVA JSON config with
aligned `sci_hi`/`wht_hi`/`segmap` paths and pass its matching map with
`--psf-map`. The
driver reports both release-segmentation-masked field scatter and fixed
empty-aperture RMS. These include correlated background and residual ringing;
neither is a propagated WHT. The input WHT product is labelled native-only.
It accepts `--target-model gaussian|moffat` and `--target-beta`. On the UDS
patch a 160-pixel, 0.10-arcsec Moffat (`beta=2.5`, `reg=2e-4`) realizes a
0.139-arcsec core. The same-width 0.10-arcsec Gaussian compromise has about
four times as much integrated negative response; the Moffat result still has
visible rings and must be treated as a profile-sensitivity test. A
0.14-arcsec Gaussian is the safer low-amplification option but realizes only
about 0.153 arcsec at `reg=1e-4`. A 0.08-arcsec Gaussian is more ill
conditioned, not less.

Forward matching from F356W to the broader F444W PSF is a different problem:
it mostly attenuates high frequencies and should reduce white-noise RMS. The
reverse F444W-to-F356W direction is deconvolution and has the same missing-mode
limit as sharpening toward an analytic target. Signed lobes in an
F356W-to-F444W kernel can correct diffraction/drizzle structure without
making the operation a deconvolution.

### Derived maps

{meth}`~mophongo.psf_map.PSFRegionMap.group_by_pa` returns a new map in
which regions whose contributing frames share a single position-angle class
(derived from the supplied FITS headers) and the same relative per-detector
exposure-time profile are dissolved into one region, reducing the region
count on mosaics with many same-orientation exposures.

{meth}`~mophongo.psf_map.PSFRegionMap.overlay_with` returns a new map whose
regions are the pairwise intersections with a second map (or a single
clipping polygon), recording the parent keys as `psf_key_1`/`psf_key_2`.
The pipeline uses the map-map form to build the kernel map on the joint
hi/lo geometry, and the polygon form to clip a footprint map to the
drizzled mosaic outline; the result carries no `psfs`, the caller fills
them in.

### Encircled energy

When `psfs` holds absolutely calibrated stamps, their finite sums are
realized encircled energies (the shape-versus-throughput convention of
{doc}`psf`). The map measures and caches them:

{meth}`~mophongo.psf_map.PSFRegionMap.refresh_ee` measures the encircled
energy of every stamp via {func}`mophongo.psf.stamp_encircled_energy`; it
runs automatically on construction and needs a manual call only if `psfs`
is mutated in place. The results are exposed as the per-`psf_key` arrays
`ee_box` (full square stamp) and `ee_rlim` (inscribed circle), the
inscribed-circle radius `r_lim` (in units of `pscale`), and the position
lookups `get_ee_box(ra, dec)` / `get_ee_rlim(ra, dec)`, which share the
NaN/miss fallback of `get_psf`. Accessing any of these on a map without
`psfs` raises `ValueError`.

```python
import numpy as np
from shapely.geometry import box
from mophongo.psf import PSF
from mophongo.psf_map import PSFRegionMap

footprints = {"expA": box(10.000, 0.0, 10.010, 0.010),
              "expB": box(10.005, 0.0, 10.015, 0.010)}
prm = PSFRegionMap.from_footprints(footprints, name="demo")
wide = PSF.gaussian(101, fwhm=10.0).array
prm.psfs = np.stack([wide[38:63, 38:63]] * len(prm.regions))  # truncated stamps
print(np.round(prm.ee_box, 3), round(prm.get_ee_box(10.0075, 0.005), 3))  # [0.994 0.994 0.994] 0.994
```

## How the pipeline builds and consumes region maps

{meth}`mophongo.pipeline.Pipeline.build_psfs` builds one map per band. For
each band it forms the region map from the exposure footprints of the
band's {class}`mophongo.psf.DrizzlePSF`, clips it to the drizzled mosaic
outline with `overlay_with`, and drizzles a PSF at every region centroid.
The low-resolution cube optionally receives a Gaussian broadening (the
`psf_blur_fwhm` run setting). Stamps keep their native sums, so a stamp sum
is a realized encircled energy: {meth}`mophongo.pipeline.Pipeline.run` divides
each fitted amplitude (`flux_<i>`) by the encircled energy of the lo-res stamp
at that source's position to get the total (`flux_<i>_total`), and falls back
to the filter-level mean stamp sum for templates that never saw a PSF map.
The two maps are written to
`<name>_psf_hi.geojson` and `<name>_psf_lo.geojson` (plus the companion
`.fits` cubes) in the output directory.

{meth}`mophongo.pipeline.Pipeline.build_kernels` overlays the hi and lo
geometry maps and computes one matching kernel per overlay region with
{func}`mophongo.utils.matching_kernel`, passing its own `method` argument,
which defaults to `"wiener"` rather than the `"window"` default of
`matching_kernel` itself. Kernels are matched between unit-sum PSF *shapes*;
for regularized methods (anything but `"window"`), when the regularization
parameter is not given it is optimized once on the median PSF shape with
{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization` and reused
for every region. The finished kernels are renormalized to unit sum, so the
kernel carries no flux scale of its own, and the map is written to
`<name>_kernel.geojson`.

```{figure} images/region_psf_kernels.png
:width: 100%
:alt: Per-region PSFs for three bands and the corresponding matching kernels.

Per-region stamps from the built maps; each column is one region, ordered by
the mean position angle of the contributing F444W frames (106–133 deg). The
PSF drizzled at the region centroid changes with the number and roll angles
of the contributing frames (top three rows: F444W, F770W, F1800W), and each
region of the overlay carries its own matching kernel (bottom row).
```

During template preparation, template convolution
({meth}`mophongo.templates.Templates.convolve_templates`) accepts either a
single kernel array or a kernel `PSFRegionMap`; in the map case each
template's kernel is looked up at the template's sky position with
`get_psf`. The hi-res PSF map supplies the detection-band PSF that the
template build scheme (`FitConfig.extend_mode`, default `"psf_wings"`)
looks up at each source position — the composite schemes scale it into the
template halo, and the `"psf_convolution"` post-pass fills zero-valued pixels with the
template convolved by it (see {doc}`templates`). The lo-res map provides the
encircled-energy metadata recorded in the output catalog, both the per-source
`ee_psf_lo` and the filter-level values in `cat.meta`.

### Provenance and staleness

Both builders stamp provenance onto the map before writing: extra columns on
`regions` (which round-trip through GeoJSON) record what produced the map.
PSF maps record the ePSF filename pattern, the stamp size, and the blur
FWHM; kernel maps record `kernel_method`, `kernel_reg`, and `psf_size`. On a
later run the cached file is reloaded and its provenance compared with the
current configuration:

- a cached PSF map is reused only when pattern, size, and blur all match;
  any disagreement (or a map predating provenance stamping) triggers a
  rebuild, with the offending field named in the log;
- a cached kernel map is reused only when the matching method matches; a
  method change forces a rebuild, since the method affects the flux scale at
  the percent level.

Passing `overwrite=True` to either builder forces a rebuild regardless. Maps
are cached per run name, so distinct configurations should use distinct run
names or output directories.
