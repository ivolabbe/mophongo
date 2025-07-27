import pytest

import numpy as np
import shapely.geometry as sgeom
import matplotlib.pyplot as plt
from shapely.affinity import translate

from mophongo.psf_map import PSFRegionMap
from astropy.wcs import WCS

# synthetic 2 × 2 dither with sub-pixel offsets
base = sgeom.box(0, 0, 1, 1)
footprints = {
    "A": base,
    "B": translate(base, 0.5, 0.5),  # ~0.108", 0.072"
}

def test_region_count():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    # With snapping & merge there should be ≤ 3 regions (A, B, A∩B)
    assert len(regmap.regions) <= 3

def test_no_tiny_regions():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    area_min = regmap.area_factor * regmap.buffer_tol**2
    assert (regmap.regions.geometry.area >= area_min).all()

def test_lookup():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    # pick a point firmly inside footprint 'A' only
    key = regmap.lookup_key(0.1, 0.1)
    assert key is not None
    frames = regmap.regions.query("psf_key == @key").frame_list.iloc[0]
    assert frames == ("A",)

def test_plot(tmp_path):
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    fig, ax = plt.subplots()
    regmap.regions.plot(column="psf_key", ax=ax, edgecolor="k", cmap="tab20")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    out = tmp_path / "psf_region_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    assert out.exists()


def _make_wcs(pa):
    w = WCS(naxis=2)
    scale = 1.0 / 3600
    theta = np.deg2rad(pa)
    w.wcs.cd = np.array(
        [[scale * np.cos(theta), -scale * np.sin(theta)],
         [scale * np.sin(theta), scale * np.cos(theta)]]
    )
    return w


def test_pa_coarsening():
    fp = {
        "A": base,
        "B": translate(base, 1.1, 0),
        "C": translate(base, 0, 1.1),
    }
    wcs = {
        "A": _make_wcs(0.1),
        "B": _make_wcs(0.2),
        "C": _make_wcs(90.0),
    }
    regmap = PSFRegionMap.from_footprints(fp, wcs=wcs, pa_tol=1.0, crs=None)
    key_a = regmap.lookup_key(0.5, 0.5)
    key_b = regmap.lookup_key(1.6, 0.5)
    key_c = regmap.lookup_key(0.5, 1.6)
    assert key_a == key_b
    assert key_c != key_a


@pytest.mark.skipif(1, reason="uses external data")
def test_psf_region_map_from_file(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import shapely.geometry as sgeom
    from shapely.affinity import translate

    from mophongo.psf_map import PSFRegionMap

    import importlib
    from pathlib import Path
    import mophongo.psf
    importlib.reload(mophongo.psf)
    from mophongo.psf import DrizzlePSF, PSF

    filt = 'F444W'
    data_dir = Path(__file__).resolve().parent.parent / "data"
    drz_file = str(data_dir / f"uds-test-{filt.lower()}_sci.fits")
    csv_file = str(data_dir / f"uds-test-{filt.lower()}_wcs.csv")

    # The target mosaic and its output WCS doesn't necessarily have to be the same drz_file
    dpsf = DrizzlePSF(driz_image=drz_file,csv_file=csv_file)

    # extract the first 10 footprints from dpsf.footprint
    footprint = {k: v for i, (k, v) in enumerate(dpsf.footprint.items()) if i < 15}

#    prm = PSFRegionMap.from_footprints(dpsf.footprint, buffer_tol=1.0/3600, area_factor=300)
#    prm = PSFRegionMap.from_footprints(footprint)
    prm = PSFRegionMap.from_footprints(footprint, pa_tol=1.0)
    prm.plot()

