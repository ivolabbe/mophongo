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
    footprint = {k: v for i, (k, v) in enumerate(dpsf.footprint.items()) if i < 10}

#    prm = PSFRegionMap.from_footprints(dpsf.footprint, buffer_tol=1.0/3600, area_factor=300)
    prm = PSFRegionMap.from_footprints(footprint)

    fig, ax = plt.subplots()
    prm.regions.plot(column="psf_key", ax=ax, edgecolor="k", cmap="tab20")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_ylim(-5.27,-5.24)
    ax.set_xlim(34.48,34.52)
    ax.set_ylim(-5.12,-5.07)
    ax.set_xlim(34.25,34.32)
    plt.show()

    plt.hist(prm.regions.area, bins=50,range=(0.1,1000),log=True)
#    ax.set_ylim(-5.24,-5.20)
#    ax.set_xlim(34.46,34.50)
 
