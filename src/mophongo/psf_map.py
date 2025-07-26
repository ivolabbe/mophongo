"""Utilities for PSF region mapping from exposure footprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Hashable
import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from astropy.wcs import WCS


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
        self.tree = STRtree(self.regions.geometry.to_list())

    # =================================================================
    # public factory
    # =================================================================
    @classmethod
    def from_footprints(
        cls,
        footprints: Mapping[Hashable, Polygon],
        *,
        crs: str | None = "EPSG:4326",
        snap_tol: float = 0.2 / 3600,
        buffer_tol: float = 1.0 / 3600,
        area_factor: float = 200.0,
        wcs: Mapping[Hashable, WCS] | None = None,
        pa_tol: float = 0.0,
        name: str | None = None,
    ) -> "PSFRegionMap":
        """
        Build a PSFRegionMap from ``(frame_id → footprint polygon)``.

        If dissolve_by_pa is True and pa_tol > 0, regions are dissolved by PA class,
        resulting in one region per PA bucket. Otherwise, previous overlap logic is used.

        Parameters
        ----------
        footprints : Mapping[Hashable, Polygon]
            Mapping of frame identifier to footprint polygon.
        wcs : Mapping[Hashable, WCS], optional
            Optional mapping of frame identifier to its ``WCS`` for orientation bucketing.
        pa_tol : float, optional
            Tolerance in degrees for grouping frames by position angle.
            ``0`` disables orientation coarsening.
        dissolve_by_pa : bool, optional
            If True, dissolve regions by PA class (one region per PA).
        All other tolerances are given in degrees.
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

        self.tree = STRtree(self.regions.geometry.to_list())
        return self

    @classmethod
    def from_geojson(cls, geojson_path, **kwargs):
        """
        Create a PSFRegionMap from a GeoJSON file.
        
        Parameters
        ----------
        geojson_path : str or Path
            Path to the GeoJSON file.
        kwargs : dict
            Additional arguments for PSFRegionMap constructor.
        """
        from astropy.io import fits
        regions_gdf = gpd.read_file(geojson_path)

        # load PSFs if available
        psfs_file = geojson_path.replace('.geojson', '.fits')
        if os.path.exists(psfs_file):
            psfs = fits.getdata(psfs_file)
        else:
            logging.warning(f"No PSFs found for {geojson_path}, using None.")

        return cls(regions=regions_gdf, psfs=psfs, **kwargs)

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
        new_map.tree = STRtree(new_map.regions.geometry.to_list()) if not final.empty else STRtree([])
        return new_map

    # =================================================================
    # public overlay methods
    # =================================================================
    def overlay_with(self, other) -> "PSFRegionMap":
        """
        Compute the overlay (intersection) of this PSFRegionMap with another PSFRegionMap
        or a single Polygon. Returns a new PSFRegionMap whose regions are the spatial
        intersections of the input maps, with psf_key pairs (or single key if Polygon).
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
            footprints=None
        )
        new_map.tree = STRtree(new_map.regions.geometry.to_list())
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
        ax.invert_xaxis()
        return fig, ax

    def lookup_key(self, ra: float, dec: float, nearest: bool = True) -> int | None:
        """Return the integer *psf_key* at (ra, dec) in deg.
        If not inside any region, optionally return the nearest region's psf_key by boundary.
        """
        pt = Point(ra, dec)
        for idx in self.tree.query(pt):
            if self.regions.geometry.iloc[idx].contains(pt):
                return int(self.regions.psf_key.iloc[idx])
        if nearest and len(self.regions) > 0:
            # Find the region with the closest boundary to the point
            distances = self.regions.geometry.boundary.distance(pt)
            nearest_idx = distances.idxmin()
            return int(self.regions.psf_key.iloc[nearest_idx])
        return None

    def get_psf(self, ra: float | None, dec: float | None) -> np.ndarray | None:
        if ra is None or dec is None or np.isnan(ra) or np.isnan(dec):
            key = 0
            logging.warning("RA/Dec is None or NaN, returning default kernel at index 0.")
        else:
            key = self.lookup_key(ra, dec)
            if key is None or np.isnan(key):
                logging.warning("key are requested ra,dec is None or NaN, returning default kernel at index 0.")
                key = 0

        return self.psfs[key]

    def to_file(self, filename, driver="GeoJSON"):
        """
        Save regions to GeoJSON and PSFs to a .fits file with the same base name.
        """
        from astropy.io import fits
        # Save regions
        self.regions.to_file(filename, driver=driver)
        # Save PSFs if present
        if self.psfs is not None:
            fits.writeto(str(filename).replace('.geojson', '.fits'), self.psfs, overwrite=True)
