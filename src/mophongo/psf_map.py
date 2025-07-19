"""Utilities for PSF region mapping from exposure footprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Hashable

import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree


@dataclass
class PSFRegionMap:
    """Lookup table for unique composite PSFs.

    Parameters
    ----------
    regions : GeoDataFrame
        Table with ``geometry`` polygons and ``psf_key`` identifiers.
    """

    regions: gpd.GeoDataFrame
    tree: STRtree = field(init=False)

    def __post_init__(self) -> None:
        self.tree = STRtree(self.regions.geometry.to_list())

    @classmethod
    def from_footprints(
        cls, footprints: Mapping[Hashable, Polygon], crs: str | None = "EPSG:4326"
    ) -> "PSFRegionMap":
        """Create a :class:`PSFRegionMap` from input footprints.

        Parameters
        ----------
        footprints
            Mapping from frame identifier to polygon footprint.
        crs
            Coordinate reference system of the polygons. Defaults to ``EPSG:4326``.
        """
        regions: list[tuple[Polygon, set[Hashable]]] = []
        for fid, poly in footprints.items():
            new_regions: list[tuple[Polygon, set[Hashable]]] = []
            for geom, frames in regions:
                if geom.intersects(poly):
                    inter = geom.intersection(poly)
                    if not inter.is_empty and inter.area > 0:
                        new_regions.append((inter, frames | {fid}))
                    diff = geom.difference(poly)
                    if not diff.is_empty and diff.area > 0:
                        new_regions.append((diff, frames))
                    poly = poly.difference(geom)
                else:
                    new_regions.append((geom, frames))
            if not poly.is_empty and poly.area > 0:
                new_regions.append((poly, {fid}))
            regions = new_regions

        records = []
        for geom, frames in regions:
            if geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    records.append({"geometry": part, "frame_list": tuple(sorted(frames))})
            else:
                records.append({"geometry": geom, "frame_list": tuple(sorted(frames))})

        gdf = gpd.GeoDataFrame(records, crs=crs)
        gdf["psf_key"] = gdf.groupby("frame_list").ngroup()
        gdf = gdf.reset_index(drop=True)
        return cls(regions=gdf[["geometry", "frame_list", "psf_key"]])

    def lookup_key(self, ra: float, dec: float) -> int | None:
        """Return the PSF key for a sky position."""
        pt = Point(ra, dec)
        for idx in self.tree.query(pt):
            geom = self.regions.geometry.iloc[idx]
            if geom.contains(pt):
                return int(self.regions.psf_key.iloc[idx])
        return None

    def build_psf(self, key: int):
        """Placeholder for PSF construction."""
        return f"psf-{key}"

