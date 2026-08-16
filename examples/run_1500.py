# %%
from pathlib import Path
import numpy as np
from astropy.io import fits
import logging

# logging.basicConfig(level=logging.ERROR)

miri_filt = "1500"

img_dir = Path("/Users/ivo/Astro/PROJECTS/MINERVA/data/v2/")
field = "uds-grizli-v8.0-minerva-v2.2-40mas"
field_miri = "uds-sbkgsub-v2.3-80mas"
sci_444 = img_dir / f"{field}-f444w-clear_drc_sci.fits"
sci_miri = img_dir / f"{field_miri}-f{miri_filt}w_drz_sci.fits"

from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap
import glob

csv444 = glob.glob(str(img_dir) + "/uds*444*_wcs.csv")[0]
csvmiri = glob.glob(str(img_dir) + f"/uds*{miri_filt}*_wcs.csv")[0]

# initialize drizzler; also reads the associated _wcs.csv files (rate files wcs/header information)
dpsf_444 = DrizzlePSF(driz_image=str(sci_444), csv_file=csv444)
dpsf_miri = DrizzlePSF(driz_image=str(sci_miri), csv_file=csvmiri)

# map unique detector overlaps; keep only footprints overlapping the target mosaic
prm_444 = PSFRegionMap.from_footprints(dpsf_444.footprint, name="F444W").overlay_with(
    dpsf_444.driz_footprint
)
prm_miri = PSFRegionMap.from_footprints(dpsf_miri.footprint, name=f"F{miri_filt}W").overlay_with(
    dpsf_miri.driz_footprint
)

# %%

# compute overlay regions unique to both PSFs
prm_kern = prm_444.overlay_with(prm_miri)
prm_444.plot()
prm_miri.plot()
prm_kern.plot()
prm_kern.regions

import mophongo.utils as utils

psf_dir = Path("../data/PSF")
# stpsf_444 = "UDS_NRC.._F444W_OS4_GRID25"
# stpsf_miri = f"UDS_MIRI_F{miri_filt}W_OS4_GRID9"
stpsf_444 = "UDS_NRC.._F444W_OS4_GRID1"
stpsf_miri = f"UDS_MIRI_F{miri_filt}W_OS4_GRID1"
size = 8.0

if not (img_dir / f"{field_miri}-f{miri_filt}w_psf.geojson").exists():
    # centroid positions of the regions: DONT drop poins here bc PSF list will be off
    pos = [np.squeeze(p.xy) for p in prm_kern.regions.geometry.centroid]

    # load webb psfs
    dpsf_444.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_444)
    dpsf_miri.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_miri)

    # drizzle at centroid positions, size of stamp in arcsec
    prm_444.psfs = dpsf_444.get_psf_radec(pos, size=size)
    prm_miri.psfs = dpsf_miri.get_psf_radec(pos, size=size)

    # store the PSFs + region maps
    prm_444.to_file(img_dir / f"{field}-f444w_psf.geojson")
    prm_miri.to_file(img_dir / f"{field_miri}-f{miri_filt}w_psf.geojson")

    # match kernels
    # @@@ need a better way to determine best fft window shape
    # @@@ No PSF found, position possibly outside footprint for 34.38666666666666, -5.243333333333333 in filter UDS_MIRI_F770W_OS4_GRID9. Returning empty output.
    # in this case, return nearest PSF, not empty
    pixel_ratio = round(dpsf_miri.driz_pscale / dpsf_444.driz_pscale)

    # @@@ optional add gaussian_filter(X, 0.1)
    kernels = [
        utils.matching_kernel(psf_444, psf_miri, recenter=True, pixel_ratio=pixel_ratio)
        for psf_444, psf_miri in zip(prm_444.psfs, prm_miri.psfs)
    ]

    prm_kern.psfs = np.asarray(kernels)
    prm_kern.to_file(img_dir / f"{field_miri}-f444w_kernel_f{miri_filt}w.geojson")


# run photometry

from pathlib import Path
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from shapely.geometry import Polygon
from shapely import points
from astropy.table import Table
from astropy.io import fits
from astropy.table import Table
from mophongo.psf_map import PSFRegionMap
from mophongo.fit import FitConfig
from mophongo.catalog import Catalog, get_bg_and_ivar
from mophongo.pipeline import Pipeline

# for testing, first run on small patch r < 0.5 arcmin
r_trial = 0.5

out_root = Path("uds_" + miri_filt)
out_root.mkdir(exist_ok=True)

img_dir = Path("/Users/ivo/Astro/PROJECTS/MINERVA/data/v2/")
cat_dir = Path("/Users/ivo/Astro/PROJECTS/MINERVA/data/n2.2_m2.0_v1.0/")

cat_file = cat_dir / "MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits"
fseg_LW = cat_dir / "LW_f277w-f356w-f444w_SEGMAP.fits"
cat = Table.read(cat_file)

field = "uds-grizli-v8.0-minerva-v2.2-40mas"
field_miri = "uds-sbkgsub-v2.3-80mas"
miri_ext = "_drz"
nircam_ext = "-clear_drc"

# PSF + kernel maps
prm_444 = PSFRegionMap.from_geojson(str(img_dir / f"{field}-f444w_psf.geojson"))
prm_miri = PSFRegionMap.from_geojson(str(img_dir / f"{field_miri}-f{miri_filt}w_psf.geojson"))
prm_kern = PSFRegionMap.from_geojson(str(img_dir / f"{field_miri}-f444w_kernel_f{miri_filt}w.geojson"))

# NIRCam images
# fsci_444 = img_dir / f'{field}-f444w{nircam_ext}_sci.fits'
fsci_444 = cat_dir / "LW_f277w-f356w-f444w_KRON_Kf444w_optavg.fits"
wcs_444 = WCS(fits.getheader(fsci_444))
# MIRI images
fsci_miri = img_dir / f"{field_miri}-f{miri_filt}w{miri_ext}_sci.fits"
fwht_miri = img_dir / f"{field_miri}-f{miri_filt}w{miri_ext}_wht.fits"
wcs_miri = WCS(fits.getheader(fsci_miri))

# load images + segmap
tmpl_444 = fits.getdata(fsci_444)
sci_miri = fits.getdata(fsci_miri)
wht_miri = fits.getdata(fwht_miri)
segmap = fits.getdata(fseg_LW)

# background and inverse variance calibration
(bg_miri, ivar_miri) = get_bg_and_ivar(sci_miri, wht_miri, bg_filter_sigma=64.0)

if locals().get("r_trial", 0) > 0:
    #    run only on small subsection of the catalog for testing
    coords = SkyCoord(ra=cat["ra"], dec=cat["dec"])
    ref = SkyCoord(ra=34.4 * u.deg, dec=-5.26 * u.deg)
    mask = coords.separation(ref) < r_trial * u.arcmin
    cat = cat[mask]

# first fit, no shifts: first image is template, 2nd and on the fitting images
config = FitConfig(
    fit_astrometry_niter=2, fit_astrometry_joint=True, scene_minimum_anchors=10, aperture_diam=0.5
)

pipe = Pipeline(
    [tmpl_444, sci_miri - bg_miri],
    segmap,
    weights=[None, ivar_miri],
    catalog=cat,
    psfs=[None, prm_miri],
    kernels=[None, prm_kern],
    wcs=[wcs_444, wcs_miri],
)

table, res = pipe.run(config=config)

fits.writeto(
    out_root / (str(out_root) + "_residual.fits"), res[0], fits.getheader(fsci_miri), overwrite=True
)
table.write(out_root / (str(out_root) + "_fit_table.fits"), overwrite=True)

from matplotlib import pyplot as plt

scenes = pipe.all_scenes[0]
for i in range(len(scenes)):
    print(
        f"scene {i}, id {scenes[i].id}, sources {len(scenes[i].templates)}, bright {scenes[i].is_bright.sum()}"
    )
    fig, ax = scenes[i].plot(tmpl_444, segmap, display_sig=5)
    fig.savefig(out_root / (str(out_root) + f"_scene_{scenes[i].id}.png"), dpi=300)
    plt.close(fig)

# %%
ras, decs = [], []
for s in scenes:
    xy_mean = np.mean([t.position_original for t in s.templates], axis=0)
    ra, dec = wcs_444.wcs_pix2world([xy_mean], 0)[0]
    ras.append(ra)
    decs.append(dec)

scene_table = Table(
    {
        "id": [s.id for s in scenes],
        "n_templates": [len(s.templates) for s in scenes],
        "is_bright": [s.is_bright.sum() for s in scenes],
        "ra": ras,
        "dec": decs,
    }
)
scene_table["minerva_link"] = [
    f"https://minerva.colorado.edu/?ra={ra}&dec={dec}&zoom=7"
    for ra, dec in zip(scene_table["ra"], scene_table["dec"])
]
scene_table.write(out_root / (str(out_root) + f"_scene_catalog.csv"), format="ascii.csv", overwrite=True)

# %%
