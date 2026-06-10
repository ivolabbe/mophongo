import numpy as np
import shapely.geometry as sgeom
from astropy.wcs import WCS
from shapely.affinity import translate

from mophongo.psf_map import PSFRegionMap


base = sgeom.box(0, 0, 1, 1)


def test_lookup():
    regmap = PSFRegionMap.from_footprints(
        {"A": base, "B": translate(base, 0.5, 0.5)}, crs=None
    )
    key = regmap.lookup_key(0.1, 0.1)
    assert key is not None
    frames = regmap.regions.query("psf_key == @key").frame_list.iloc[0]
    assert frames == ("A",)


def _make_wcs(pa):
    w = WCS(naxis=2)
    scale = 1.0 / 3600
    theta = np.deg2rad(pa)
    w.wcs.cd = np.array(
        [
            [scale * np.cos(theta), -scale * np.sin(theta)],
            [scale * np.sin(theta), scale * np.cos(theta)],
        ]
    )
    return w


def test_pa_coarsening():
    fp = {
        "A": base,
        "B": translate(base, 1.1, 0),
        "C": translate(base, 0, 1.1),
    }
    wcs = {"A": _make_wcs(0.1), "B": _make_wcs(0.2), "C": _make_wcs(90.0)}
    regmap = PSFRegionMap.from_footprints(fp, wcs=wcs, pa_tol=1.0, crs=None)
    key_a = regmap.lookup_key(0.5, 0.5)
    key_b = regmap.lookup_key(1.6, 0.5)
    key_c = regmap.lookup_key(0.5, 1.6)

    assert key_a == key_b
    assert key_c != key_a
