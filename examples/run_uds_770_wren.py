# %%
from pathlib import Path
import numpy as np
from astropy.io import fits
import logging

# see output on https://drive.google.com/drive/folders/1d1LU57xqk2sVRr6PtiTSG3-XMEGC06nC

logging.getLogger("drizzle").setLevel(logging.ERROR)
# for testing, first run on small patch r < 0.5 arcmin
# r_trial = 1

miri_filt = "1800"
version = 'test_aug3'  # internal Kron total (tcor_int) + catalog tie (s_cat)
psf_size = 3.0  # psf size in arcsec (was 2.0 in v0.1; 3.0 balances EE coverage vs. speed)

# set aperture diameter to match what yoshi used
aps = {'770': 0.35*2, '1280': 0.6*2, '1500': 0.6*2, '1800': 0.75*2}
aperture_diam = aps[miri_filt]

# set coupling threshold based on filter
thresh = {'770': 1e-3, '1280': 5e-3, '1500': 0.01, '1800': 0.02} #5e-3 f1280w,#1e-3 f770w, 0.02 F1800w
coupling_thresh = thresh[miri_filt]


img_dir = Path("/Volumes/DarkPhoenix/Surveys/MINERVA/UDS/images/")
data_dir = Path("data")
field = "uds-grizli-v8.0-minerva-v3.0-40mas"
field_miri = "uds-sbkgsub-v3.0-80mas"
sci_444 = img_dir / f"NIRCam/v3.0/{field}-f444w-clear_drc_sci_f444w-matched.fits"
sci_miri = img_dir / f"MIRI/v3.0/{field_miri}-f{miri_filt}w_drz_sci_extrabkg.fits"

from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap

# WCS CSVs: F444W uses local v2.3 file (same observations); MIRI uses v3.0
csv444 = str(img_dir / "NIRCam/v3.0/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv")
csvmiri = str(img_dir / f"MIRI/v3.0/{field_miri}-f{miri_filt}w_wcs.csv")

# PSF/kernel geojson names — tagged with psf_size to avoid overwriting v0.1 files
psf_tag = f"_psf{int(psf_size)}as"
psf_444_geojson  = img_dir / f"{field}-f444w_psf{psf_tag}.geojson"
psf_miri_geojson = img_dir / f"{field_miri}-f{miri_filt}w_psf{psf_tag}.geojson"
kern_geojson     = img_dir / f"{field_miri}-f444w_kernel_f{miri_filt}w{psf_tag}.geojson"

# initialize drizzler; also reads the associated _wcs.csv files (rate files wcs/header information)
dpsf_444  = DrizzlePSF(driz_image=str(sci_444),  csv_file=csv444)
dpsf_miri = DrizzlePSF(driz_image=str(sci_miri), csv_file=csvmiri)

# map unique detector overlaps; keep only footprints overlapping the target mosaic
prm_444  = PSFRegionMap.from_footprints(dpsf_444.footprint,  name="F444W").overlay_with(dpsf_444.driz_footprint)
prm_miri = PSFRegionMap.from_footprints(dpsf_miri.footprint, name=f"F{miri_filt}W").overlay_with(dpsf_miri.driz_footprint)

# compute overlay regions unique to both PSFs
prm_kern = prm_444.overlay_with(prm_miri)

import mophongo.utils as utils

psf_dir  = data_dir / "PSF"
stpsf_444  = "UDS_NRC.._F444W_OS4_GRID25"
stpsf_miri = f"UDS_MIRI_F{miri_filt}W_OS4_GRID9"

if not psf_miri_geojson.exists():
    print(f"Computing PSFs and kernels (psf_size={psf_size} arcsec) ...")
    # centroid positions of the regions: DONT drop points here bc PSF list will be off
    pos = [np.squeeze(p.xy) for p in prm_kern.regions.geometry.centroid]

    # load webb psfs
    dpsf_444.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_444)
    dpsf_miri.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_miri)

    # drizzle at centroid positions
    prm_444.psfs  = dpsf_444.get_psf_radec(pos,  size=psf_size)
    prm_miri.psfs = dpsf_miri.get_psf_radec(pos, size=psf_size)

    # PSF stamp containment: fraction of each PSF's TRUE total flux (measured on
    # the 8" STPSF grid parents) captured by the finite region-map stamp -- the
    # stamps only hold ~92-96% of the flux, which left every PSF-path correction
    # low by that fraction (docs/aperture_corrections.md Sec 4.1/5.2). Use the
    # ACTUAL stamp width (region-map PSF shape x mosaic pixel scale), not
    # psf_size, since the drizzled stamp can differ from the requested size.
    grid_444 = np.concatenate(
        [fits.getdata(f) for f in sorted(psf_dir.glob("UDS_NRC*_F444W_OS4_GRID25.fits"))]
    )
    grid_miri = fits.getdata(psf_dir / f"UDS_MIRI_F{miri_filt}W_OS4_GRID9.fits")
    width_444  = prm_444.psfs.shape[-1]  * dpsf_444.driz_pscale
    width_miri = prm_miri.psfs.shape[-1] * dpsf_miri.driz_pscale
    prm_444.containment  = utils.psf_stamp_containment(grid_444,  0.063 / 4, width_444)
    prm_miri.containment = utils.psf_stamp_containment(grid_miri, 0.11 / 4,  width_miri)
    print(f"  PSF stamp containment: F444W={prm_444.containment:.4f} (width {width_444:.2f}\"), "
          f"F{miri_filt}W={prm_miri.containment:.4f} (width {width_miri:.2f}\")")

    # store the PSFs + region maps
    prm_444.to_file(str(psf_444_geojson))
    prm_miri.to_file(str(psf_miri_geojson))

    # match kernels
    pixel_ratio = round(dpsf_miri.driz_pscale / dpsf_444.driz_pscale)
    kernels = [
        utils.matching_kernel(psf_444, psf_miri, recenter=True, pixel_ratio=pixel_ratio)
        for psf_444, psf_miri in zip(prm_444.psfs, prm_miri.psfs)
    ]
    prm_kern.psfs = np.asarray(kernels)
    prm_kern.to_file(str(kern_geojson))
    print(f"Saved PSFs and kernels.")
else:
    print(f"Loading cached PSFs and kernels from {psf_tag} geojsons ...")

# run photometry

from pathlib import Path
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from mophongo.psf_map import PSFRegionMap
from mophongo.fit import FitConfig
from mophongo.catalog import Catalog, get_bg_and_ivar
from mophongo.pipeline import Pipeline

out_root = Path("uds_" + miri_filt)
out_root.mkdir(exist_ok=True)

cat_dir  = Path("/Volumes/DarkPhoenix/Surveys/MINERVA/UDS/catalogs/n3.0_m3.1_v1.2.1/")
cat_file = cat_dir / "MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits" #"LW/MINERVA-UDS_n3.0_v1.2_LW_Kf444w_SUPER_CATALOG.fits" #
fseg_LW  = cat_dir / "MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits" #"LW/MINERVA-UDS_n3.0_v1.2_LW_SEGMAP.fits" 
cat = Table.read(cat_file)

miri_ext   = "_drz"
nircam_ext = "-clear_drc"

# PSF + kernel maps
prm_444  = PSFRegionMap.from_geojson(str(psf_444_geojson))
prm_miri = PSFRegionMap.from_geojson(str(psf_miri_geojson))
prm_kern = PSFRegionMap.from_geojson(str(kern_geojson))

# Images
fsci_444  = img_dir / f"NIRCam/v3.0/{field}-f444w-clear_drc_sci_f444w-matched.fits"
fwht_444 = img_dir / f"NIRCam/v3.0/{field}-f444w-clear_drc_wht.fits"  
wcs_444   = WCS(fits.getheader(fsci_444))
fsci_miri = img_dir / f"MIRI/v3.0/{field_miri}-f{miri_filt}w_drz_sci_extrabkg.fits"
fwht_miri = img_dir / f"MIRI/v3.0/{field_miri}-f{miri_filt}w_drz_wht.fits"
wcs_miri  = WCS(fits.getheader(fsci_miri))

# load images + segmap
tmpl_444 = fits.getdata(fsci_444)
sci_miri  = fits.getdata(fsci_miri)
wht_miri  = fits.getdata(fwht_miri)
wht_444  = fits.getdata(fwht_444)
segmap    = fits.getdata(fseg_LW)

# background and inverse variance calibration
bg_miri, ivar_miri = get_bg_and_ivar(sci_miri, wht_miri, bg_filter_sigma=64.0)

if locals().get("r_trial", 0) > 0:
    coords = SkyCoord(ra=cat["ra"], dec=cat["dec"])
    ref    = SkyCoord(ra=34.4 * u.deg, dec=-5.14 * u.deg)
    mask   = coords.separation(ref) < r_trial * u.arcmin
    cat    = cat[mask]

config = FitConfig(
    fit_astrometry_niter=2, 
    scene_minimum_anchors=5,
    scene_coupling_thresh=coupling_thresh,
    astrom_isolation_thresh=0.7,
    aperture_diam=aperture_diam,
    f444w_col='f_f444w',
    template_extend_mode='data',
    # Stage-3b two-step catalog tie (tcor_int + s_cat): f444w_aper_col sets the
    # Kron circular-radius floor (r_floor_pix); f444w_totcor_col is currently
    # unused (kept for diagnostics/alternative ties).
    f444w_totcor_col='tot_cor',   # NIRCam catalog aperture(color)->total factor
    f444w_aper_col='use_aper',    # NIRCam catalog color-aperture diameter (arcsec)
)

pipe = Pipeline(
    [tmpl_444, sci_miri - bg_miri],
    segmap,
    weights=[wht_444, ivar_miri],
    catalog=cat,
    psfs=[prm_444, prm_miri],
    kernels=[None, prm_kern],
    wcs=[wcs_444, wcs_miri],
)

table, res = pipe.run(config=config)

fits.writeto(
    out_root / (str(out_root) + f"_{version}_residual.fits"),
    res[0],
    fits.getheader(fsci_444),
    overwrite=True,
)
table.write(out_root / (str(out_root) + f"_{version}_fit_table.fits"), overwrite=True)

# # scene plots — skip for speed; uncomment to regenerate
# from matplotlib import pyplot as plt
# scenes = pipe.all_scenes[0]
# for i in range(len(scenes)):
#     fig, ax = scenes[i].plot(tmpl_444, segmap, display_sig=5)
#     fig.savefig(out_root / (str(out_root) + f"_{version}_scene_{scenes[i].id}.png"), dpi=300)
#     plt.close(fig)

# %%
from astropy.table import Table
from astropy.wcs import WCS

scenes = pipe.all_scenes[0]
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
    f"https://minerva.colorado.edu/uds/?ra={ra}&dec={dec}&zoom=7"
    for ra, dec in zip(scene_table["ra"], scene_table["dec"])
]
scene_table.write(
    out_root / (str(out_root) + f"_{version}_scene_catalog.csv"),
    format="ascii.csv",
    overwrite=True,
)

print(f"Done. Outputs in {out_root}/ with version tag {version}.")

# %%
