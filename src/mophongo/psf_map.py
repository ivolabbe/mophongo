"""Utilities for PSF region mapping from exposure footprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Hashable

from astropy.wcs import WCS

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree


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
    buffer_tol: float = 0.2 / 3600
    area_factor: float = 100.0

    tree: STRtree = field(init=False, repr=False)

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
        area_factor: float = 300.0,
        wcs: Mapping[Hashable, WCS] | None = None,
        pa_tol: float = 1.0,
    ) -> "PSFRegionMap":
        """Build a PSFRegionMap from ``(frame_id → footprint polygon)``.
        
        Parameters
        ----------
        footprints : Mapping[Hashable, Polygon]
            Mapping of frame identifiers to footprint polygons.
        wcs
            Optional mapping dictionary of frame identifier to its ``WCS`` for
            orientation bucketing.
        pa_tol
            Tolerance in degrees for grouping frames by position angle.
            ``0`` disables orientation coarsening.
        All other tolerances are given in degrees.
        crs : str or None, optional
            Coordinate reference system of the input footprints. Defaults to
            "EPSG:4326".
        snap_tol : float, optional
            Tolerance for ``shapely.set_precision`` in degrees.
        buffer_tol : float, optional
            Buffer size used to close small gaps in degrees.
        area_factor : float, optional
            Scale factor applied to the buffer_tol**2 area when defining the minimum
            region area to retain

        Returns
        -------
        PSFRegionMap
            New instance constructed from the provided footprints.
        """
        self = cls.__new__(cls)
        self.snap_tol = snap_tol
        self.buffer_tol = buffer_tol
        self.area_factor = area_factor
        self._area_min = area_factor * buffer_tol
        pa_class = None
        if wcs is not None and pa_tol > 0:
            pa_class = {fid: cls._pa_class(wcs[fid], pa_tol) for fid in footprints}

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

        self.tree = STRtree(self.regions.geometry.to_list())
        return self

    # =================================================================
    # public query helpers
    # =================================================================
    def lookup_key(self, ra: float, dec: float) -> int | None:
        """Return the integer *psf_key* at (ra, dec) in deg; None if outside."""
        pt = Point(ra, dec)
        for idx in self.tree.query(pt):
            if self.regions.geometry.iloc[idx].contains(pt):
                return int(self.regions.psf_key.iloc[idx])
        return None

    def build_psf(self, key: int):
        """Placeholder stub – replace with real PSF synthesis."""
        return f"psf-{key}"

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
                    if (
                        gdf.at[j, "geometry"] is not None
                        and not gdf.at[j, "geometry"].is_empty
                        and poly.boundary is not None
                    )
                    else -1
                ),
            )
            gdf.at[idx, "psf_key"] = gdf.at[nbr, "psf_key"]

        return gdf.dissolve(by="psf_key", as_index=False, aggfunc="first").drop(
            columns="area"
        )
