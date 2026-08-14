"""Utilities for PSF region mapping from exposure footprints."""

from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS.*distance",
    category=UserWarning,
    module=r"mophongo\.psf_map"
)

from dataclasses import dataclass, field
from typing import Mapping, Hashable
import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import prepared 
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from astropy.wcs import WCS
import logging

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
#  Main public dataclass
# ────────────────────────────────────────────────────────────────────
@dataclass
class PSFRegionMap:
    """Lookup table that maps a sky position → *psf_key*.

    Parameters (all degree units; factory defaults are 0.2″)
    ----------------------------------------------------------------
    snap_tol     snap grid for Shapely ``set_precision``.
    buffer_tol   ±buffer used to seal <2·buffer_tol gaps.
    area_factor  area_min = area_factor × buffer_tol.
    """

    regions: gpd.GeoDataFrame
    snap_tol: float = 0.2 / 3600
    buffer_tol: float = 1.0 / 3600
    area_factor: float = 200.0
    name: str | None = None
    tree: STRtree = field(init=False, repr=False)
    footprints: Mapping[Hashable, Polygon] = field(default_factory=dict, repr=False)
    
    # optional ndarray to store PSF kernels as a lookup table
    psfs: np.ndarray | None = None
    # pixel scale of ``psfs``; only sets the units of ``r_lim``.  Left at 1.0
    # the radii are in pixels, while the encircled-energy fractions are
    # unaffected either way.
    pscale: float = 1.0

    # encircled-energy cache, filled by refresh_ee().  Declared without
    # annotations so the dataclass does not treat them as fields, and set here
    # so the constructors that bypass __post_init__ still find them.
    _ee_src = None
    _ee_box = None
    _ee_rlim = None
    _r_lim = float("nan")

    # ------------------------------------------------------------------
    # orientation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _pa_class(wcs: WCS, tol: float) -> int:
        """Return orientation bucket index for ``wcs`` with width ``tol`` degrees."""
        pa = (np.rad2deg(np.arctan2(wcs.wcs.cd[0, 1], wcs.wcs.cd[0, 0])) + 360.0) % 360.0
        if tol <= 0:
            return int(round(pa))  # effectively unique per degree
        return int(np.round(pa / tol))

    @staticmethod
    def _parse_detector_from_key(key: str) -> str:
        """
        Parse detector name from a FITS filename or key.
        Supports NIRCam (_nrcalong_rate.fits), MIRI (_mirimage_rate.fits), and JWST convention.
        """
        key = key.lower()
        match = re.search(r'_nrc([ab]\w+)_rate\.fits', key)
        if match:
            return f'NRC{match.group(1).upper()}'
        match = re.search(r'_mirimage_rate\.fits', key)
        if match:
            return 'MIRIMAGE'
        match = re.search(r'_([a-z0-9]+)_rate\.fits', key)
        if match:
            return match.group(1).upper()
        return 'UNKNOWN'

    # ───────────── private derived constants ──────────────
    def __post_init__(self) -> None:
        self._area_min = self.area_factor * self.buffer_tol
        # One width for every map, whichever constructor was used. PSF and
        # kernel stamps multiply float32 image and template pixels, and scipy
        # promotes to the wider operand, so a float64 cube doubles the
        # workspace of every convolution that touches it: 233 MB for a
        # 100x100x2911 matching-kernel cube, and twice the FFT width across
        # 138,610 template convolutions.
        #
        # float32 is ample. These stamps weight pixels; they are not summed
        # over long chains. The one derived quantity with a hard bound is the
        # encircled energy, and on the real UDS maps narrowing moves ee_box by
        # 2.6e-10 (0.916986720542 -> 0.916986720803) against a physical value
        # of 0.92-0.96 -- and the delivered psf_hi map has always been stored
        # BITPIX -32 anyway, so the hi-res band has run at this width all along.
        # Normalising here rather than in from_geojson keeps a map built in
        # memory bit-comparable with the same map round-tripped through disk.
        if self.psfs is not None and np.asarray(self.psfs).dtype != np.float32:
            self.psfs = np.asarray(self.psfs, dtype=np.float32)
        self._rebuild_spatial_index()
        self._ee_src: int | None = None
        self.refresh_ee()

    def _rebuild_spatial_index(self) -> None:
        """Rebuild geometry lookup caches after replacing ``regions``."""
        self._geoms     = list(self.regions.geometry)          # keep a plain list for STRtree
        self.tree       = STRtree(self._geoms)

        self._prepared  = [prepared.prep(g) for g in self._geoms]

        self._keys      = self.regions["psf_key"].to_numpy()

    # ----------------------------------------------------------------
    #  Make deepcopy / pickle work
    # ----------------------------------------------------------------
    def __getstate__(self):
        """
        Return a picklable representation of the object.

        We drop the STRtree (and anything else that can’t be pickled)
        and rebuild it in __setstate__.
        """
        state = self.__dict__.copy()
        # STRtree and prepared geometries are not picklable; __setstate__
        # rebuilds all three from ``regions``.
        state["tree"] = None
        state["_prepared"] = None
        state["_geoms"] = None
        return state

    def __setstate__(self, state):
        """
        Restore the object and rebuild the spatial index.
        """
        self.__dict__.update(state)
        if self.tree is None and hasattr(self, "regions"):
            self._rebuild_spatial_index()


    # =================================================================
    # public factory
    # =================================================================
    @classmethod
    def from_footprints(
        cls,
        footprints: Mapping[Hashable, Polygon],
        *,
        crs: str | None = "EPSG:4326",
        snap_tol: float = 0.5 / 3600,
        buffer_tol: float = 1.0 / 3600,
        area_factor: float = 100.0,
        wcs: Mapping[Hashable, WCS] | None = None,
        pa_tol: float = 0.0,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """
        Build a PSFRegionMap from ``(frame_id → footprint polygon)``.

        Footprints are intersected sequentially; every distinct set of
        overlapping frames becomes one region, stored with a ``frame_list``
        column and an integer ``psf_key`` label. ``psf_key`` values are
        renumbered to run consecutively from 0.

        Parameters
        ----------
        footprints : Mapping[Hashable, Polygon]
            Mapping of frame identifier to footprint polygon, in degrees.
        crs : str, optional
            Coordinate reference system assigned to the regions
            GeoDataFrame. Default ``"EPSG:4326"``.
        snap_tol, buffer_tol, area_factor : float, optional
            Geometry-cleanup tolerances, in degrees; see the class docstring.
            Note the factory defaults (``snap_tol=0.5/3600``,
            ``area_factor=100.0``) differ from the dataclass defaults, and
            the sliver-area threshold here is
            ``area_factor * buffer_tol**2``.
        wcs : Mapping[Hashable, WCS], optional
            Optional mapping of frame identifier to its ``WCS``, used only
            for orientation bucketing.
        pa_tol : float, optional
            Tolerance in degrees for grouping frames by position angle. With
            ``pa_tol > 0`` and ``wcs`` given, frames are tagged with a PA
            class and regions are keyed by their set of PA classes rather
            than the exact frame set, which coarsens the map. ``0`` (default)
            disables orientation coarsening.
        name : str, optional
            Label for the resulting map.
        """
        self = cls.__new__(cls)
        self.snap_tol = snap_tol
        self.buffer_tol = buffer_tol
        self.area_factor = area_factor
        self._area_min = area_factor * buffer_tol**2
        self.footprints = dict(footprints)  # Save original footprints
        self.name = name 

        pa_class = None
        if wcs is not None and pa_tol > 0:
            pa_class = {fid: cls._pa_class(wcs[fid], pa_tol) for fid in footprints}

        # --- Previous functionality: full overlap logic ---
        regions: list[tuple[Polygon, set[Hashable]]] = []
        for fid, poly in footprints.items():
            poly = self._preprocess(poly)
            new_regions: list[tuple[Polygon, set[Hashable]]] = []
            for geom, frames in regions:
                if geom.intersects(poly):
                    inter = geom.intersection(poly)
                    if not inter.is_empty and inter.area > 0:
                        token = (
                            (pa_class[fid], fid) if pa_class is not None else fid
                        )
                        new_regions.append((inter, frames | {token}))
                    diff = geom.difference(poly)
                    if not diff.is_empty and diff.area > 0:
                        new_regions.append((diff, frames))
                    poly = poly.difference(geom)
                else:
                    new_regions.append((geom, frames))
            if not poly.is_empty and poly.area > 0:
                token = (
                    (pa_class[fid], fid) if pa_class is not None else fid
                )
                new_regions.append((poly, {token}))
            regions = new_regions

        records = []
        for geom, frames in regions:
            if pa_class is not None:
                pa_list = tuple(sorted({p for p, _ in frames}))
                fid_list = tuple(sorted(f for _, f in frames))
            else:
                pa_list = ()
                fid_list = tuple(sorted(frames))

            if geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    records.append({
                        "geometry": part,
                        "frame_list": fid_list,
                        "pa_list": pa_list,
                    })
            else:
                records.append({
                    "geometry": geom,
                    "frame_list": fid_list,
                    "pa_list": pa_list,
                })

        gdf = gpd.GeoDataFrame(records, crs=crs)
        group_col = "pa_list" if pa_class is not None else "frame_list"
        gdf["psf_key"] = gdf.groupby(group_col).ngroup()

        self.regions = self._merge_slivers(gdf).reset_index(drop=True)

        # Remove non-polygon geometries
        self.regions = self.regions[self.regions.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)

        # Renumber psf_key to be consecutive starting from 0
        unique_keys = sorted(self.regions['psf_key'].unique())
        key_mapping = {old_key: new_key for new_key, old_key in enumerate(unique_keys)}
        self.regions['psf_key'] = self.regions['psf_key'].map(key_mapping)

        self._rebuild_spatial_index()
        return self

    @classmethod
    def from_geojson(cls, geojson_path, **kwargs):
        """
        Create a PSFRegionMap from a GeoJSON file written by :meth:`to_file`.

        If a FITS file with the same base name (``.geojson`` replaced by
        ``.fits``) exists, its data become ``psfs``; otherwise a warning is
        logged and ``psfs`` stays ``None``. The map's ``name`` is set to the
        file's base name. Only ``regions`` and ``psfs`` are stored on disk:
        tolerances, ``pscale``, and ``footprints`` come back as constructor
        defaults unless passed again via ``kwargs``.

        Parameters
        ----------
        geojson_path : str or Path
            Path to the GeoJSON file.
        kwargs : dict
            Additional arguments for the PSFRegionMap constructor
            (for example ``pscale``).
        """
        from astropy.io import fits
        # str(): a Path would take .replace below as the rename method
        geojson_path = str(geojson_path)
        regions_gdf = gpd.read_file(geojson_path)

        # load PSFs if available
        psfs = None
        psfs_file = geojson_path.replace('.geojson', '.fits')
        if os.path.exists(psfs_file):
            # __post_init__ narrows this to float32; see the note there
            psfs = fits.getdata(psfs_file)
        else:
            logging.warning(f"No PSFs found for {geojson_path}, using None.")

        base_name = os.path.splitext(os.path.basename(geojson_path))[0]
        return cls(regions=regions_gdf, psfs=psfs, name=base_name, **kwargs)

    # =================================================================
    # public grouping methods
    # =================================================================
    def group_by_pa(
        self,
        pa_tol: float,
        hdrs: Mapping[Hashable, fits.Header],
        crs: str | None = "EPSG:4326",
    ) -> "PSFRegionMap":
        """
        Merge regions where all contributing frames share the same PA class AND 
        detector exposure time profile (relative contributions).
        """
        from collections import defaultdict
        
        # Create WCS and extract info from headers
        pa_class = {}
        detector_class = {}
        exposure_times = {}
        
        for fid, hdr in hdrs.items():
            wcs = WCS(hdr, relax=True)
            pa_class[fid] = self._pa_class(wcs, pa_tol)
            detector_class[fid] = self._parse_detector_from_key(str(fid))
            exposure_times[fid] = hdr.get('EXPTIME', 0.0)
        
        def get_detector_exposure_profile(frame_list):
            """
            Calculate RELATIVE exposure time contribution per detector for a region.
            Returns a frozenset of (detector, relative_exposure_fraction) tuples.
            """
            detector_exposures = defaultdict(float)
            
            # Sum absolute exposure times per detector
            for fid in frame_list:
                detector = detector_class[fid]
                exp_time = exposure_times[fid]
                detector_exposures[detector] += exp_time
            
            # Calculate total exposure time across all detectors
            total_exp_time = sum(detector_exposures.values())
            
            # Convert to relative fractions (rounded for comparison stability)
            if total_exp_time > 0:
                relative_exposures = {
                    detector: round(exp_time / total_exp_time, 6)  # Round to 6 decimal places
                    for detector, exp_time in detector_exposures.items()
                }
            else:
                relative_exposures = {detector: 0.0 for detector in detector_exposures}
            
            # Return as frozenset for hashability and set comparison
            return frozenset(relative_exposures.items())
        
        # For each region, get the (PA, detector_exposure_profile) combination
        regions = self.regions.copy()
        regions["pa_detector_profile"] = regions["frame_list"].apply(
            lambda fl: (
                tuple(sorted({pa_class[fid] for fid in fl})),  # PA classes
                get_detector_exposure_profile(fl)  # Relative detector exposure profile
            )
        )
        
        # Only dissolve regions where PA is homogeneous
        def can_merge(profile):
            pa_classes, det_exp_profile = profile
            return len(pa_classes) == 1  # Homogeneous PA
        
        homogeneous = regions[regions["pa_detector_profile"].apply(can_merge)].copy()
        inhomogeneous = regions[~regions["pa_detector_profile"].apply(can_merge)].copy()

        # Dissolve homogeneous regions by (PA, relative_detector_exposure_profile)
        if not homogeneous.empty:
            homogeneous["merge_key"] = homogeneous["pa_detector_profile"]
            homogeneous["psf_key"] = homogeneous.groupby("merge_key").ngroup()
            dissolved = homogeneous.dissolve(by="merge_key", as_index=False, aggfunc="first")
        else:
            dissolved = gpd.GeoDataFrame()

        # Keep inhomogeneous regions separate with unique psf_keys
        if not inhomogeneous.empty:
            start_key = dissolved["psf_key"].max() + 1 if not dissolved.empty else 0
            inhomogeneous["psf_key"] = range(start_key, start_key + len(inhomogeneous))
            inhomogeneous["merge_key"] = inhomogeneous["pa_detector_profile"]

        # Combine dissolved and inhomogeneous regions
        if not dissolved.empty and not inhomogeneous.empty:
            final = pd.concat([dissolved, inhomogeneous], ignore_index=True)
        elif not dissolved.empty:
            final = dissolved
        elif not inhomogeneous.empty:
            final = inhomogeneous
        else:
            final = gpd.GeoDataFrame()
        
        # Renumber psf_key to be consecutive starting from 0
        if not final.empty:
            unique_keys = sorted(final['psf_key'].unique())
            key_mapping = {old_key: new_key for new_key, old_key in enumerate(unique_keys)}
            final['psf_key'] = final['psf_key'].map(key_mapping)
            final = final[final.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)

        # Build new PSFRegionMap
        new_map = PSFRegionMap(
            regions=final.reset_index(drop=True),
            snap_tol=self.snap_tol,
            buffer_tol=self.buffer_tol,
            area_factor=self.area_factor,
            footprints=self.footprints,
            name = (self.name or '') + ' by PA'
        )
        new_map._rebuild_spatial_index()
        return new_map

    # =================================================================
    # public overlay methods
    # =================================================================
    def overlay_with(self, other) -> "PSFRegionMap":
        """
        Compute the overlay (intersection) of this PSFRegionMap with another PSFRegionMap
        or a single Polygon. Returns a new PSFRegionMap whose regions are the spatial
        intersections of the input maps, recording the parent keys as
        ``psf_key_1``/``psf_key_2`` (``psf_key_1`` only for a Polygon). The
        result carries no ``psfs``; the caller fills them in.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        overlays = []
        if isinstance(other, PSFRegionMap):
            # Overlay with another PSFRegionMap
            for i, reg1 in self.regions.iterrows():
                for j, reg2 in other.regions.iterrows():
                    intersection = reg1.geometry.intersection(reg2.geometry)
                    if not intersection.is_empty:
                        overlays.append({
                            "geometry": intersection,
                            "psf_key_1": reg1.psf_key,
                            "psf_key_2": reg2.psf_key
                        })
        elif isinstance(other, Polygon):
            # Overlay with a single Polygon
            for i, reg1 in self.regions.iterrows():
                intersection = reg1.geometry.intersection(other)
                if not intersection.is_empty:
                    overlays.append({
                        "geometry": intersection,
                        "psf_key_1": reg1.psf_key
                    })
        else:
            raise TypeError("overlay_with: 'other' must be a PSFRegionMap or a shapely Polygon.")

        # Build GeoDataFrame
        overlay_gdf = gpd.GeoDataFrame(overlays)
        overlay_gdf = overlay_gdf[overlay_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        overlay_gdf["psf_key"] = overlay_gdf.index  # Assign new unique keys

        # Return new PSFRegionMap
        new_map = PSFRegionMap(
            regions=overlay_gdf,
            snap_tol=self.snap_tol,
            buffer_tol=self.buffer_tol,
            area_factor=self.area_factor,
            footprints=None,
            name = f"{self.name}" + (f" overlay with {other.name}" if isinstance(other, PSFRegionMap) else "")
        )
        new_map._rebuild_spatial_index()
        return new_map

    # =================================================================
    # private helpers (operate on *self.* tolerances)
    # =================================================================
    def _preprocess(self, poly: Polygon) -> Polygon:
        """±buffer then snap to grid."""
        if self.buffer_tol:
            poly = poly.buffer(+self.buffer_tol, join_style="mitre")
            poly = poly.buffer(-self.buffer_tol, join_style="mitre")
        return shapely.set_precision(poly, self.snap_tol)

    def _merge_slivers(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Dissolve regions whose area < self._area_min."""
        if gdf.empty:
            return gdf

        gdf = gdf.copy()
        gdf["area"] = gdf.geometry.area
        rtree = STRtree(list(gdf.geometry))

        small_idx = gdf.query("area < @self._area_min").index
        for idx in small_idx:
            poly = gdf.at[idx, "geometry"]
            if poly is None or poly.is_empty:
                continue
            nbrs = [
                j
                for j in rtree.query(poly)
                if (
                    j != idx
                    and gdf.at[j, "geometry"] is not None
                    and not gdf.at[j, "geometry"].is_empty
                    and poly.touches(gdf.at[j, "geometry"])
                )
            ]
            if not nbrs:
                continue
            
            # Check if poly.boundary is valid before using it
            if poly.boundary is None:
                continue
                
            # Filter again for safety in lambda
            nbr = max(
                nbrs,
                key=lambda j: (
                    poly.boundary.intersection(gdf.at[j, "geometry"]).length
                    if (gdf.at[j, "geometry"] is not None 
                        and not gdf.at[j, "geometry"].is_empty 
                        and poly.boundary is not None)
                    else -1
                ),
            )
            gdf.at[idx, "psf_key"] = gdf.at[nbr, "psf_key"]

        result = gdf.dissolve(by="psf_key", as_index=False, aggfunc="first").drop(
            columns="area"
        )
        # Remove non-polygon geometries
        result = result[result.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        return result

    def plot(self, column: str = "psf_key", ax=None, edgecolor="k", cmap="tab20", **kwargs):
        """
        Plot the PSF regions, inverting the x-axis.
        If ax is None, creates a new figure and axis.
        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        self.regions.plot(column=column, ax=ax, edgecolor=edgecolor, cmap=cmap, **kwargs)
        ax.set_title(self.name or "PSF Regions")
        ax.invert_xaxis()
        return fig, ax

    # -------------------------------------------------------------------
    # fast lookup
    # -------------------------------------------------------------------
    def lookup_key(self, ra: float, dec: float, nearest: bool = True) -> int | None:
        """
        Return psf_key containing (ra, dec) in deg, or nearest one if `nearest` is True.
        Runs in O(log N) for both hit & miss.
        """
        pt = Point(ra, dec)

        # ---------- exact hit (typ. 1–3 candidates) --------------------
        for idx in self.tree.query(pt):                     # fast candidate search
            if self._prepared[idx].contains(pt):            # prepared geometry ⇒ µs
                return int(self._keys[idx])

        # ---------- fallback: nearest ----------------------------------
        if nearest and len(self._geoms):
            try:
                idx = self.tree.nearest(pt)                 # shapely ≥2.0
            except AttributeError:                          # older shapely / pygeos
                # GeoPandas >=0.10 has the same via spatial index
                idx = self.regions.sindex.nearest(pt)[1][0]
            return int(self._keys[idx])

        return None

    def get_psf(self, ra: float | None, dec: float | None) -> np.ndarray | None:
        """Return the 2-D stamp ``psfs[key]`` for the region at (ra, dec).

        The position is in degrees (the CRS of ``regions``); the containing
        region is used, or the nearest one outside the map. If either
        coordinate is ``None`` or NaN, or the lookup fails, a warning is
        logged and the stamp at index 0 is returned. Requires ``psfs``.
        """
        if ra is None or dec is None or np.isnan(ra) or np.isnan(dec):
            key = 0
            logging.warning("RA/Dec is None or NaN, returning default kernel at index 0.")
        else:
            key = self.lookup_key(ra, dec)
            if key is None or np.isnan(key):
                logging.warning("key are requested ra,dec is None or NaN, returning default kernel at index 0.")
                key = 0

        return self.psfs[key]

    def _dense_psf_keys(self) -> np.ndarray:
        """Validate and return the dense integer key sequence of ``psfs``."""
        if self.psfs is None:
            raise ValueError(f"PSFRegionMap {self.name!r} has no psfs")
        cube = np.asarray(self.psfs)
        if cube.ndim != 3:
            raise ValueError(
                f"PSFRegionMap psfs must have shape (n, ny, nx), got {cube.shape}"
            )
        raw_keys = np.asarray(self.regions["psf_key"])
        numeric_keys = np.asarray(raw_keys, dtype=float)
        if not np.all(np.isfinite(numeric_keys)) or not np.all(
            numeric_keys == np.round(numeric_keys)
        ):
            raise ValueError(f"PSFRegionMap psf_key values must be finite integers: {raw_keys}")
        keys = np.unique(numeric_keys.astype(int))
        expected = np.arange(len(keys), dtype=int)
        if not np.array_equal(keys, expected) or cube.shape[0] != len(expected):
            raise ValueError(
                "PSFRegionMap requires dense psf_key values 0..N-1 aligned "
                f"with one PSF plane per key; keys={keys.tolist()}, "
                f"psfs.shape[0]={cube.shape[0]}"
            )
        return expected

    def _analytic_psf_map(
        self,
        model: str,
        fwhm_pix: float,
        *,
        beta: float | None = None,
        shape: int | tuple[int, int] | None = None,
        phase_match: bool = True,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """Build a phase-matched analytic target map for public wrappers."""
        from .psf import PSF, psf_core_centroid

        if model not in {"gaussian", "moffat"}:
            raise ValueError(f"unsupported analytic target model {model!r}")
        keys = self._dense_psf_keys()
        source = np.asarray(self.psfs)
        if not np.isfinite(fwhm_pix) or float(fwhm_pix) <= 0.0:
            raise ValueError(f"fwhm_pix must be positive, got {fwhm_pix!r}")
        if model == "moffat" and (
            beta is None or not np.isfinite(beta) or float(beta) <= 1.0
        ):
            raise ValueError(
                f"Moffat beta must be finite and greater than one, got {beta!r}"
            )
        if shape is None:
            target_shape = tuple(map(int, source.shape[-2:]))
        elif isinstance(shape, (int, np.integer)):
            target_shape = (int(shape), int(shape))
        else:
            target_shape = tuple(map(int, shape))
            if len(target_shape) != 2:
                raise ValueError(f"shape must have two axes, got {shape!r}")
        if any(n <= 0 for n in target_shape):
            raise ValueError(f"shape axes must be positive, got {target_shape}")
        if any(t < s for t, s in zip(target_shape, source.shape[-2:])):
            raise ValueError(
                f"target shape {target_shape} is smaller than source PSFs "
                f"{source.shape[-2:]}"
            )

        pad_y = (target_shape[0] - source.shape[-2]) // 2
        pad_x = (target_shape[1] - source.shape[-1]) // 2
        targets: list[np.ndarray] = []
        raw_sums: list[float] = []
        for key in keys:
            if phase_match:
                xc, yc = psf_core_centroid(source[int(key)])
                xc += pad_x
                yc += pad_y
            else:
                xc = (target_shape[1] - 1) / 2.0
                yc = (target_shape[0] - 1) / 2.0
            if model == "gaussian":
                target = PSF.gaussian(
                    target_shape, fwhm=float(fwhm_pix), x0=xc, y0=yc
                ).array
            else:
                target = PSF.moffat(
                    target_shape,
                    float(fwhm_pix),
                    float(fwhm_pix),
                    float(beta),
                    x0=xc,
                    y0=yc,
                ).array
            target_sum = float(np.sum(target))
            if not np.isfinite(target_sum) or target_sum <= 0.0:
                raise ValueError(
                    f"invalid {model} target sum at psf_key={key}"
                )
            raw_sums.append(target_sum)
            targets.append(target / target_sum)

        regions = self.regions.copy()
        key_rows = np.asarray(regions["psf_key"], dtype=int)
        regions["target_model"] = model
        regions["target_fwhm_pix"] = float(fwhm_pix)
        regions["target_fwhm"] = float(fwhm_pix) * float(self.pscale)
        regions["target_discrete_sum"] = np.asarray(raw_sums)[key_rows]
        regions["target_phase_match"] = bool(phase_match)
        regions["target_ny"] = int(target_shape[0])
        regions["target_nx"] = int(target_shape[1])
        if model == "moffat":
            regions["target_beta"] = float(beta)
        model_suffix = model
        if model == "moffat":
            model_suffix += f"_beta{float(beta):g}"
        target_map = PSFRegionMap(
            regions=regions,
            snap_tol=self.snap_tol,
            buffer_tol=self.buffer_tol,
            area_factor=self.area_factor,
            footprints=self.footprints,
            name=(
                name
                or f"{self.name or 'psf'}_{model_suffix}_fwhm{fwhm_pix:g}px"
            ),
            pscale=self.pscale,
        )
        # Assign after construction: target/kernel maps do not need their
        # signed growth curves eagerly, and a large padded cube can make that
        # irrelevant calculation expensive. The EE cache remains lazy.
        target_map.psfs = np.asarray(targets, dtype=np.float32)
        return target_map

    def gaussian_psf_map(
        self,
        fwhm_pix: float,
        *,
        shape: int | tuple[int, int] | None = None,
        phase_match: bool = True,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """Return a theoretical Gaussian PSF map on this map's geometry.

        One noise-free, discrete-unit-sum Gaussian is generated per source
        region. With ``phase_match=True`` its core is placed at that region's
        measured subpixel centroid, preserving astrometry in a subsequent
        matching kernel.

        Args:
            fwhm_pix: Gaussian full width at half maximum in map pixels.
            shape: Output stamp shape. A scalar gives a square stamp. Larger
                shapes provide padded support for inverse kernels.
            phase_match: Match the source PSF's fitted core centroid.
            name: Optional map name.

        Returns:
            A theoretical Gaussian :class:`PSFRegionMap`.

        Raises:
            ValueError: If a parameter or the source map is invalid.
        """
        return self._analytic_psf_map(
            "gaussian",
            fwhm_pix,
            shape=shape,
            phase_match=phase_match,
            name=name,
        )

    def moffat_psf_map(
        self,
        fwhm_pix: float,
        beta: float,
        *,
        shape: int | tuple[int, int] | None = None,
        phase_match: bool = True,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """Return a theoretical circular Moffat PSF map.

        The finite sampled targets are explicitly normalized to unit sum and
        generated directly on the requested support. This matters for the
        Moffat wings; making a small target and zero-padding it would silently
        truncate and renormalize the intended profile.

        Args:
            fwhm_pix: Moffat full width at half maximum in map pixels.
            beta: Power-law index greater than one.
            shape: Output stamp shape. A scalar gives a square stamp. Larger
                shapes provide padded support for inverse kernels.
            phase_match: Match the source PSF's fitted core centroid.
            name: Optional map name.

        Returns:
            A theoretical Moffat :class:`PSFRegionMap`.

        Raises:
            ValueError: If a parameter or the source map is invalid.
        """
        return self._analytic_psf_map(
            "moffat",
            fwhm_pix,
            beta=beta,
            shape=shape,
            phase_match=phase_match,
            name=name,
        )

    def matching_kernel_map(
        self,
        target: "PSFRegionMap | np.ndarray",
        *,
        method: str = "wiener",
        reg: float,
        signal_psd: np.ndarray | None = None,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """Build per-region kernels matching this PSF map to ``target``.

        The source and target stamps are converted to unit-sum PSF shapes,
        passed to Mophongo's existing :func:`mophongo.utils.matching_kernel`,
        and the resulting kernels are normalized to unit DC.  The latter is
        required because regularization attenuates the zero-frequency term;
        without it, convolving a science image would change its flux scale.

        This method deliberately requires an explicit regularization value.
        The standard automatic PSF-matching figure of merit was designed for
        stable smoothing kernels and can choose a broadened response when the
        requested target is narrower than the source.  For deconvolution,
        callers should scan ``reg`` and inspect the diagnostic columns written
        to the returned region table.

        Args:
            target: A 2-D target used for every region, a cube with one target
                per source key, or another aligned :class:`PSFRegionMap` such
                as :meth:`gaussian_psf_map`. Array targets are assumed to be
                sampled on this source map's pixel grid; target maps must
                carry the same ``pscale``.
            method: Matching-kernel method understood by
                :func:`mophongo.utils.matching_kernel`.
            reg: Required, strictly positive dimensionless regularization.
                ``1e-3`` is a useful conservative starting point for
                sharpening; it is not a claim that the requested target
                resolution is achieved.
            signal_psd: Optional Wiener signal power spectrum.  With ``None``
                the current Wiener implementation uses a flat prior and is
                mathematically the Tikhonov solution.
            name: Optional output map name.

        Returns:
            A kernel :class:`PSFRegionMap`.  Its region table includes the
            white-noise RMS gain, kernel cancellation, edge support, realized
            core width, target peak recovery, negative response flux, and
            residual astrometric shift for every key.

        Raises:
            ValueError: If either map is malformed, target keys are not
                aligned, a stamp cannot be normalized, or a kernel has zero
                DC response.
        """
        from .psf import psf_core_centroid, psf_core_fwhm
        from .utils import fftconvolve, matching_kernel, pad_to_shape

        keys = self._dense_psf_keys()
        source_cube = np.asarray(self.psfs, dtype=float)
        if not np.isfinite(reg) or float(reg) <= 0.0:
            raise ValueError(f"reg must be a finite positive scalar, got {reg!r}")

        target_name = "array"
        target_regions = None
        if isinstance(target, PSFRegionMap):
            target_keys = target._dense_psf_keys()
            if not np.array_equal(keys, target_keys):
                raise ValueError("source and target PSFRegionMap keys are not aligned")
            if not np.isclose(
                float(self.pscale), float(target.pscale), rtol=1e-10, atol=0.0
            ):
                raise ValueError(
                    "source and target PSFRegionMap pixel scales differ: "
                    f"{self.pscale!r} vs {target.pscale!r}"
                )
            target_cube = np.asarray(target.psfs, dtype=float)
            target_name = target.name or "PSFRegionMap"
            target_regions = target.regions
        else:
            target_cube = np.asarray(target, dtype=float)
            if target_cube.ndim == 2:
                target_cube = np.broadcast_to(
                    target_cube, (len(keys),) + target_cube.shape
                )
            elif target_cube.ndim != 3 or target_cube.shape[0] != len(keys):
                raise ValueError(
                    "target must be 2-D or have one 2-D plane per psf_key; "
                    f"got shape {target_cube.shape}"
                )

        metric_names = (
            "kernel_sum_raw",
            "kernel_noise_gain",
            "kernel_l1",
            "kernel_negative_flux",
            "kernel_edge_l1",
            "kernel_edge_l1_fraction",
            "response_fwhm_x_pix",
            "response_fwhm_y_pix",
            "response_target_peak",
            "response_negative_flux",
            "response_l2_fraction",
            "response_shift_x_pix",
            "response_shift_y_pix",
        )
        metrics = {field: np.full(len(keys), np.nan, dtype=float) for field in metric_names}
        kernels: list[np.ndarray] = []

        for key in keys:
            source = np.where(np.isfinite(source_cube[int(key)]), source_cube[int(key)], 0.0)
            target_psf = np.where(
                np.isfinite(target_cube[int(key)]), target_cube[int(key)], 0.0
            )
            source_sum = float(np.sum(source))
            target_sum = float(np.sum(target_psf))
            if source_sum <= 0.0 or not np.isfinite(source_sum):
                raise ValueError(f"source PSF at psf_key={key} has invalid sum {source_sum}")
            if target_sum <= 0.0 or not np.isfinite(target_sum):
                raise ValueError(f"target PSF at psf_key={key} has invalid sum {target_sum}")
            source /= source_sum
            target_psf /= target_sum

            kernel = matching_kernel(
                source,
                target_psf,
                method=method,
                reg=float(reg),
                recenter=False,
                signal_psd=signal_psd,
            )
            raw_sum = float(np.sum(kernel))
            if not np.isfinite(raw_sum) or abs(raw_sum) <= np.finfo(float).eps:
                raise ValueError(
                    f"kernel at psf_key={key} has invalid DC sum {raw_sum}"
                )
            kernel = np.asarray(kernel, dtype=float) / raw_sum

            # PSFRegionMap stores stamp cubes as float32.  Signed inverse
            # kernels can contain enough cancellation that this cast moves
            # their DC sum measurably away from one, so normalize and validate
            # the representation that will actually be convolved.  A tiny
            # residual is placed at the kernel origin; if float32 cannot
            # represent that correction, the requested inversion is too
            # unstable for this map format and must use stronger regularization.
            kernel_stored = np.asarray(kernel, dtype=np.float32)
            stored_sum = float(np.sum(kernel_stored, dtype=np.float64))
            if not np.isfinite(stored_sum) or abs(stored_sum) <= np.finfo(float).eps:
                raise ValueError(
                    f"float32 kernel at psf_key={key} has invalid DC sum "
                    f"{stored_sum}; increase reg"
                )
            kernel_stored /= np.float32(stored_sum)
            origin = tuple(int(n // 2) for n in kernel_stored.shape)
            stored_sum = float(np.sum(kernel_stored, dtype=np.float64))
            kernel_stored[origin] += np.float32(1.0 - stored_sum)
            stored_sum = float(np.sum(kernel_stored, dtype=np.float64))
            if not np.isclose(stored_sum, 1.0, rtol=0.0, atol=2e-6):
                raise ValueError(
                    f"float32 kernel at psf_key={key} cannot preserve unit DC "
                    f"(sum={stored_sum}); increase reg"
                )
            kernels.append(kernel_stored)
            kernel = np.asarray(kernel_stored, dtype=float)

            shape = tuple(map(int, kernel.shape))
            source_padded = pad_to_shape(source, shape)
            target_padded = pad_to_shape(target_psf, shape)
            response = fftconvolve(source_padded, kernel, mode="same")
            response_fwhm = psf_core_fwhm(response)
            target_centroid = psf_core_centroid(target_padded)
            response_centroid = psf_core_centroid(response)
            border = max(1, min(shape) // 32)
            edge = np.zeros(shape, dtype=bool)
            edge[:border, :] = True
            edge[-border:, :] = True
            edge[:, :border] = True
            edge[:, -border:] = True
            l1 = float(np.sum(np.abs(kernel)))
            edge_l1 = float(np.sum(np.abs(kernel[edge])))
            target_peak = float(np.max(target_padded))

            metrics["kernel_sum_raw"][int(key)] = raw_sum
            metrics["kernel_noise_gain"][int(key)] = float(np.sqrt(np.sum(kernel**2)))
            metrics["kernel_l1"][int(key)] = l1
            metrics["kernel_negative_flux"][int(key)] = float(
                -np.sum(np.minimum(kernel, 0.0))
            )
            metrics["kernel_edge_l1"][int(key)] = edge_l1
            metrics["kernel_edge_l1_fraction"][int(key)] = (
                edge_l1 / l1 if l1 > 0.0 else np.nan
            )
            metrics["response_fwhm_x_pix"][int(key)] = response_fwhm[0]
            metrics["response_fwhm_y_pix"][int(key)] = response_fwhm[1]
            metrics["response_target_peak"][int(key)] = (
                float(np.max(response)) / target_peak if target_peak > 0.0 else np.nan
            )
            metrics["response_negative_flux"][int(key)] = float(
                -np.sum(np.minimum(response, 0.0))
            )
            target_l2 = float(np.linalg.norm(target_padded))
            metrics["response_l2_fraction"][int(key)] = (
                float(np.linalg.norm(response - target_padded)) / target_l2
                if target_l2 > 0.0 else np.nan
            )
            metrics["response_shift_x_pix"][int(key)] = (
                response_centroid[0] - target_centroid[0]
            )
            metrics["response_shift_y_pix"][int(key)] = (
                response_centroid[1] - target_centroid[1]
            )

        regions = self.regions.copy()
        key_rows = np.asarray(regions["psf_key"], dtype=int)
        regions["kernel_method"] = str(method)
        regions["kernel_reg"] = float(reg)
        regions["kernel_source"] = self.name or "PSFRegionMap"
        regions["kernel_target"] = target_name
        regions["kernel_signal_psd"] = "flat" if signal_psd is None else "provided"
        for field, values in metrics.items():
            regions[field] = values[key_rows]

        if target_regions is not None:
            target_key_rows = np.asarray(target_regions["psf_key"], dtype=int)
            for field in target_regions.columns:
                if not str(field).startswith("target_"):
                    continue
                values_by_key = []
                for key in keys:
                    rows = np.flatnonzero(target_key_rows == int(key))
                    if rows.size == 0:
                        raise ValueError(f"target metadata missing psf_key={key}")
                    values_by_key.append(target_regions.iloc[int(rows[0])][field])
                regions[field] = np.asarray(values_by_key)[key_rows]

        kernel_map = PSFRegionMap(
            regions=regions,
            snap_tol=self.snap_tol,
            buffer_tol=self.buffer_tol,
            area_factor=self.area_factor,
            footprints=self.footprints,
            name=name or f"{self.name or 'psf'}_{method}_kernel_{target_name}",
            pscale=self.pscale,
        )
        kernel_map.psfs = np.asarray(kernels, dtype=np.float32)
        return kernel_map

    # -------------------------------------------------------------------
    # encircled energy of the stored stamps
    # -------------------------------------------------------------------
    def refresh_ee(self) -> None:
        """Measure the encircled energy of every stamp in ``psfs``.

        Called once when the map is constructed and again whenever ``psfs`` is
        replaced, so the lookups below are a single array index.  The values
        are derived from the stamps themselves rather than stored alongside
        them, which means the drizzle kernel, any edge taper and any
        broadening are already inside the sum.  For absolutely calibrated
        stamps (STPSF ``NORMALIZ = first``, drizzled with ``in_units="cps"``)
        they are absolute encircled energies.
        """
        if self.psfs is None:
            self._ee_box = self._ee_rlim = None
            self._r_lim = float("nan")
            self._ee_src = None
            return
        from .psf import stamp_encircled_energy

        ee = stamp_encircled_energy(np.asarray(self.psfs), self.pscale, per_stamp=True)
        self._ee_box = ee["ee_box"]
        self._ee_rlim = ee["ee_circ"]
        self._r_lim = float(ee["r_circ"])
        self._ee_src = id(self.psfs)

    def _ee_arrays(self) -> None:
        """Recompute only if ``psfs`` was replaced since the last measurement."""
        if self._ee_src != id(self.psfs):
            self.refresh_ee()
        if self._ee_box is None:
            raise ValueError(
                f"PSFRegionMap {self.name!r} has no psfs; cannot report encircled energy"
            )

    @property
    def ee_box(self) -> np.ndarray:
        """Encircled energy in the square stamp, one value per psf_key."""
        self._ee_arrays()
        return self._ee_box

    @property
    def ee_rlim(self) -> np.ndarray:
        """Encircled energy in the inscribed circle, one value per psf_key."""
        self._ee_arrays()
        return self._ee_rlim

    @property
    def r_lim(self) -> float:
        """Inscribed-circle radius of the stamps, in units of ``pscale``."""
        self._ee_arrays()
        return self._r_lim

    def _ee_at(self, arr: np.ndarray, ra: float | None, dec: float | None) -> float:
        if ra is None or dec is None or np.isnan(ra) or np.isnan(dec):
            logging.warning("RA/Dec is None or NaN, returning encircled energy at index 0.")
            return float(arr[0])
        key = self.lookup_key(ra, dec)
        if key is None or np.isnan(key):
            logging.warning("no region at requested ra,dec; returning index 0.")
            key = 0
        return float(arr[int(key)])

    def get_ee_box(self, ra: float | None, dec: float | None) -> float:
        """Encircled energy in the square stamp at a sky position."""
        return self._ee_at(self.ee_box, ra, dec)

    def get_ee_rlim(self, ra: float | None, dec: float | None) -> float:
        """Encircled energy in the inscribed circle at a sky position."""
        return self._ee_at(self.ee_rlim, ra, dec)

    # -------------------------------------------------------------------
    # region-wise convolution
    # -------------------------------------------------------------------
    def convolve_image(
        self,
        image: np.ndarray,
        wcs: WCS,
        *,
        buffer: int | None = None,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Convolve a full image with the per-region stamp of this map.

        The stamps of a kernel map vary from region to region, so a whole
        mosaic cannot be convolved in one pass. Each region is instead cut out
        with a border wide enough that its own pixels see no edge (half the
        kernel, ``buffer``), convolved with that region's stamp, and only the
        pixels *inside* the region polygon are written back. Overlaps between
        cutouts are therefore never double-counted and the seams carry no
        discontinuity beyond the difference between the two kernels.

        For FITS files on disk use :func:`convolve_fits`, which wraps this
        with the I/O.

        Args:
            image: 2-D science image.
            wcs: WCS of ``image``, used to place the region polygons (which
                are in degrees) on the pixel grid.
            buffer: Border in pixels added around each region before
                convolving. Defaults to half the largest stamp, which is the
                width at which a region's own pixels are unaffected by the
                cut.
            fill_value: Value for pixels covered by no region -- outside the
                exposure footprint the map was built from.

        Returns:
            Convolved image with the same shape as ``image``. Floating input
            retains its dtype; integer input is promoted to floating point.

        Raises:
            ValueError: If the map has no ``psfs`` to convolve with.
        """
        from shapely import contains_xy as _contains_xy
        from shapely.ops import transform as _shapely_transform

        from .utils import fftconvolve

        if self.psfs is None:
            raise ValueError("this PSFRegionMap has no psfs to convolve with")

        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"image must be 2-D, got shape {image.shape}")

        stamps = np.asarray(self.psfs)
        if buffer is None:
            buffer = int(max(stamps.shape[-2:]) // 2 + 1)
        buffer = int(buffer)

        # Convolution is intrinsically floating point. Keeping an integer
        # input dtype truncated every result, and np.nan_to_num's default
        # treatment of +/-inf injected enormous finite values that a signed
        # deconvolution kernel then spread across a region. Promote integers
        # and replace every non-finite input explicitly with zero.
        out_dtype = np.result_type(image.dtype, stamps.dtype, np.float32)
        if not np.issubdtype(out_dtype, np.floating):
            out_dtype = np.dtype(np.float32)
        out = np.full(image.shape, fill_value, dtype=out_dtype)
        covered = np.zeros(image.shape, dtype=bool)
        ny, nx = image.shape

        for geom, key in zip(self._geoms, self._keys):
            if geom.is_empty:
                continue
            # Work in pixel space: map the polygon once (vertices, holes and
            # all parts) instead of running the WCS over every pixel of the
            # cutout, which for a detector-sized region is millions of points.
            poly = _shapely_transform(
                lambda x, y, z=None: tuple(
                    wcs.all_world2pix(np.asarray(x), np.asarray(y), 0)
                ),
                geom,
            )
            bx0, by0, bx1, by1 = poly.bounds
            x0 = max(int(np.floor(bx0)) - buffer, 0)
            x1 = min(int(np.ceil(bx1)) + buffer + 1, nx)
            y0 = max(int(np.floor(by0)) - buffer, 0)
            y1 = min(int(np.ceil(by1)) + buffer + 1, ny)
            if x1 <= x0 or y1 <= y0:
                continue

            cut = np.asarray(image[y0:y1, x0:x1], dtype=out_dtype)
            clean = np.where(np.isfinite(cut), cut, 0.0)
            conv = fftconvolve(clean, stamps[int(key)], mode="same")

            # keep only the pixels of this region, tested against the polygon
            # itself rather than its bounding box
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inside = _contains_xy(poly, xx.astype(float), yy.astype(float))
            if not inside.any():
                continue
            out[y0:y1, x0:x1][inside] = conv[inside].astype(out.dtype, copy=False)
            covered[y0:y1, x0:x1] |= inside

        n_missing = int((~covered).sum())
        if n_missing:
            logger.info(
                "convolve_image: %d of %d pixels (%.2f%%) lie in no region and "
                "were set to %g", n_missing, image.size,
                100.0 * n_missing / image.size, fill_value,
            )
        return out

    def to_file(self, filename, driver="GeoJSON"):
        """
        Save regions to GeoJSON and PSFs to a .fits file with the same base name.

        The regions table is written with all its columns (including
        provenance), using any ``geopandas`` driver. :meth:`from_geojson`
        reverses both files, but only ``regions`` and ``psfs`` round-trip;
        tolerances, ``pscale``, and ``footprints`` are not stored, and
        tuple-valued columns such as ``frame_list`` come back as their
        string repr.
        """
        from astropy.io import fits
        # Save regions
        self.regions.to_file(filename, driver=driver)
        # Save PSFs if present
        if self.psfs is not None:
            fits.writeto(str(filename).replace('.geojson', '.fits'), self.psfs, overwrite=True)


# ────────────────────────────────────────────────────────────────────
#  file-level convenience
# ────────────────────────────────────────────────────────────────────
def convolve_fits(
    sci: str | os.PathLike,
    region_map: "PSFRegionMap | str | os.PathLike",
    out_path: str | os.PathLike,
    *,
    ext: int | str = 0,
    buffer: int | None = None,
    fill_value: float = 0.0,
    overwrite: bool = True,
) -> str:
    """Convolve a science image on disk with a region-mapped kernel map.

    File-level wrapper around :meth:`PSFRegionMap.convolve_image`: reads the
    image and its WCS, convolves region by region, and writes the result with
    the original header plus the provenance of the map used.

    Args:
        sci: Path to the science FITS file.
        region_map: A :class:`PSFRegionMap`, or the path of a GeoJSON written
            by :meth:`PSFRegionMap.to_file` (its ``.fits`` sidecar of stamps
            is picked up alongside).
        out_path: Where to write the convolved image.
        ext: Extension of ``sci`` holding the image.
        buffer: Border in pixels around each region, see
            :meth:`PSFRegionMap.convolve_image`.
        fill_value: Value for pixels covered by no region.
        overwrite: Overwrite ``out_path`` if it exists.

    Returns:
        The path written, as a string.
    """
    from astropy.io import fits

    prm = (region_map if isinstance(region_map, PSFRegionMap)
           else PSFRegionMap.from_geojson(region_map))

    with fits.open(sci) as hdul:
        header = hdul[ext].header.copy()
        data = np.asarray(hdul[ext].data)
    out = prm.convolve_image(data, WCS(header), buffer=buffer,
                             fill_value=fill_value)

    header["CONVMAP"] = (os.path.basename(str(prm.name or region_map))[:40],
                         "kernel region map")
    header["CONVNREG"] = (len(prm.regions), "regions convolved separately")
    fits.writeto(out_path, out, header, overwrite=overwrite)
    logger.info("wrote %s (%d regions, buffer %s)", out_path, len(prm.regions),
                buffer if buffer is not None else "auto")
    return str(out_path)
