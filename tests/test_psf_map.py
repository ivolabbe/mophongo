import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
from shapely.affinity import translate

from mophongo.psf_map import PSFRegionMap

# synthetic 2 × 2 dither with sub-pixel offsets
base = sgeom.box(0, 0, 1, 1)
footprints = {
    "A": base,
    "B": translate(base, 0.00003, 0.00002),  # ~0.108", 0.072"
}

def test_region_count():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    # With snapping & merge there should be ≤ 3 regions (A, B, A∩B)
    assert len(regmap.regions) <= 3

def test_no_tiny_regions():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    area_min = regmap.area_factor * np.pi * (regmap.fwhm / 2) ** 2
    assert (regmap.regions.geometry.area >= area_min).all()

def test_lookup():
    regmap = PSFRegionMap.from_footprints(footprints, crs=None)
    # pick a point firmly inside footprint 'A' only
    key = regmap.lookup_key(2.5e-05, 0.5)
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


def test_psf_region_map_from_file(tmp_path):
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

    prm = PSFRegionMap.from_footprints(dpsf.footprint)
    prm.regions.plot(column="psf_key", ax=ax, edgecolor="k", cmap="tab20")

 
