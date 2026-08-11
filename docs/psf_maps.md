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

## PSFRegionMap

`PSFRegionMap` is a dataclass. Its fields:

`regions` : `geopandas.GeoDataFrame`, required
: One row per region, with at least a `geometry` column and an integer
  `psf_key` column. Factory-built maps also carry `frame_list` (contributing
  frame identifiers) and `pa_list`. Extra columns round-trip through GeoJSON,
  which the pipeline uses for provenance stamping (see below).

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

The spatial index (`tree`, a Shapely `STRtree`) is derived state: it is
rebuilt in `__post_init__`, dropped on pickling, and restored on unpickling,
so region maps deep-copy and pickle cleanly.

### Constructors

#### `from_footprints`

```python
PSFRegionMap.from_footprints(
    footprints,
    *,
    crs="EPSG:4326",
    snap_tol=0.5 / 3600,
    buffer_tol=1.0 / 3600,
    area_factor=100.0,
    wcs=None,
    pa_tol=0.0,
    name=None,
)
```

`footprints` : `Mapping[Hashable, Polygon]`
: Mapping of frame identifier to footprint polygon, in degrees.

`crs` : `str | None`, default `"EPSG:4326"`
: Coordinate reference system assigned to the regions GeoDataFrame.

`snap_tol`, `buffer_tol`, `area_factor`
: Geometry tolerances as above. The factory defaults
  (`snap_tol=0.5/3600`, `area_factor=100.0`) differ from the dataclass
  defaults; both are in degrees.

`wcs` : `Mapping[Hashable, WCS] | None`, default `None`
: Optional per-frame WCS used only for orientation bucketing.

`pa_tol` : `float`, default `0.0`
: Tolerance in degrees for grouping frames by position angle. With
  `pa_tol > 0` and `wcs` given, frames are tagged with a PA class and regions
  are keyed by their set of PA classes rather than the exact frame set, which
  coarsens the map. `0` disables orientation coarsening.

`name` : `str | None`, default `None`
: Label for the resulting map.

#### `from_geojson`

```python
PSFRegionMap.from_geojson(geojson_path, **kwargs)
```

`geojson_path` : `str`
: Path to a GeoJSON file previously written by
  {meth}`~mophongo.psf_map.PSFRegionMap.to_file`. If a FITS file with the
  same base name (`.geojson` replaced by `.fits`) exists, its data become
  `psfs`; otherwise a warning is logged and `psfs` stays `None`.

`**kwargs`
: Passed to the `PSFRegionMap` constructor (for example `pscale`).

The map's `name` is set to the file's base name.

### Lookup

#### `lookup_key`

```python
prm.lookup_key(ra, dec, nearest=True)
```

`ra`, `dec` : `float`
: Sky position in degrees (the CRS of `regions`).

`nearest` : `bool`, default `True`
: When the point falls inside no region (for example just outside the mosaic
  edge), return the key of the nearest region instead of `None`.

Returns the integer `psf_key`, or `None` on a miss with `nearest=False`. The
query uses the STRtree plus prepared geometries and is O(log N).

#### `get_psf`

```python
prm.get_psf(ra, dec)
```

`ra`, `dec` : `float | None`
: Sky position in degrees. If either is `None` or NaN, or the lookup fails,
  a warning is logged and the stamp at index 0 is returned.

Returns the 2-D stamp `psfs[key]` for the region containing (or nearest to)
the position. Requires `psfs` to be set.

### Derived maps

#### `group_by_pa`

```python
prm.group_by_pa(pa_tol, hdrs, crs="EPSG:4326")
```

`pa_tol` : `float`
: PA class width in degrees.

`hdrs` : `Mapping[Hashable, fits.Header]`
: FITS headers per frame identifier; used to derive each frame's WCS (for the
  PA class), detector name (parsed from the identifier), and `EXPTIME`.

`crs` : `str | None`, default `"EPSG:4326"`
: Accepted for interface symmetry; as of this writing it is not used in the
  merge.

Returns a new `PSFRegionMap` in which regions whose contributing frames share
a single PA class and the same relative per-detector exposure-time profile
are dissolved into one region. Regions mixing PA classes are kept separate.
This reduces the region count on mosaics with many same-orientation
exposures.

#### `overlay_with`

```python
prm.overlay_with(other)
```

`other` : `PSFRegionMap | shapely.geometry.Polygon`
: A second region map, or a single clipping polygon.

Returns a new `PSFRegionMap` whose regions are the pairwise intersections.
Each output region records the parent keys as `psf_key_1` (and `psf_key_2`
for a map-map overlay) and gets a fresh consecutive `psf_key`. The pipeline
uses the map-map form to build the kernel map on the joint hi/lo geometry,
and the polygon form to clip a footprint map to the drizzled mosaic outline.
The result carries no `psfs`; the caller fills them in.

### Encircled energy

When `psfs` holds absolutely calibrated stamps, their finite sums are
realized encircled energies (the shape-versus-throughput convention of
{doc}`psf`). The map measures and caches them:

#### `refresh_ee`

```python
prm.refresh_ee()
```

No parameters. Measures the encircled energy of every stamp via
{func}`mophongo.psf.stamp_encircled_energy`. Called automatically on
construction; call it again only if `psfs` is mutated in place, since
replacement of the array is detected automatically.

`ee_box` : property, `np.ndarray`
: Encircled energy in the full square stamp, one value per `psf_key`.

`ee_rlim` : property, `np.ndarray`
: Encircled energy within the inscribed circle, one value per `psf_key`.

`r_lim` : property, `float`
: Inscribed-circle radius of the stamps, in units of `pscale`.

`get_ee_box(ra, dec)` / `get_ee_rlim(ra, dec)` : `float`
: Position lookups of the two arrays; `ra`, `dec` in degrees, with the same
  NaN/miss fallback to index 0 as `get_psf`.

Accessing any of these on a map without `psfs` raises `ValueError`.

### Serialization

#### `to_file`

```python
prm.to_file(filename, driver="GeoJSON")
```

`filename` : path-like
: Output path for the regions table (conventionally `.geojson`).

`driver` : `str`, default `"GeoJSON"`
: Any `geopandas.GeoDataFrame.to_file` driver.

Writes the regions (geometry plus all columns, including provenance) to
`filename`; if `psfs` is set, the cube is written to a FITS file with the
same base name. `from_geojson` reverses both, but only `regions` and `psfs`
are stored: `snap_tol`, `buffer_tol`, `area_factor`, `pscale`, and
`footprints` are not written to the file and come back as constructor
defaults unless passed again via `from_geojson` kwargs, and `name` is reset
to the file's base name.

#### `plot`

```python
prm.plot(column="psf_key", ax=None, edgecolor="k", cmap="tab20", **kwargs)
```

`column` : `str`, default `"psf_key"`
: Regions column to color by.

`ax` : matplotlib axis, optional
: Axis to draw into; a new figure is created when `None`.

`edgecolor` : default `"k"`; `cmap` : default `"tab20"`; `**kwargs`
: Forwarded to `GeoDataFrame.plot`.

Returns `(fig, ax)` with the x-axis inverted (RA increasing left).

## How the pipeline builds and consumes region maps

{meth}`mophongo.pipeline.Pipeline.build_psfs` builds one map per band. For
each band it forms the region map from the exposure footprints of the
band's {class}`mophongo.psf.DrizzlePSF`, clips it to the drizzled mosaic
outline with `overlay_with`, and drizzles a PSF at every region centroid.
The low-resolution cube optionally receives a Gaussian broadening (the
`psf_blur_fwhm` run setting). Stamps keep their native sums: the finite
stamp sum is filter-level throughput metadata, later applied by
{meth}`mophongo.pipeline.Pipeline.run` to convert fitted amplitudes
(`flux_<i>`) into totals (`flux_<i>_total`). The two maps are written to
`<name>_psf_hi.geojson` and `<name>_psf_lo.geojson` (plus the companion
`.fits` cubes) in the output directory.

{meth}`mophongo.pipeline.Pipeline.build_kernels` overlays the hi and lo
geometry maps and computes one matching kernel per overlay region with
{func}`mophongo.utils.matching_kernel` (default `method="wiener"`). Kernels
are matched between unit-sum PSF *shapes*; for regularized methods, when the
regularization parameter is not given it is optimized once on the median PSF
shape with
{meth}`mophongo.psf.PSF.optimize_matching_kernel_regularization` and reused
for every region. The finished kernels are renormalized to unit sum, so the
kernel carries no flux scale of its own, and the map is written to
`<name>_kernel.geojson`.

During template preparation, template convolution
({meth}`mophongo.templates.Templates.convolve_templates`) accepts either a
single kernel array or a kernel `PSFRegionMap`; in the map case each
template's kernel is looked up at the template's sky position with
`get_psf`. The hi-res PSF map serves template extension (analytic wings are
matched to the local hi-res PSF), and the lo-res map provides the
encircled-energy metadata recorded in the output catalog.

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
