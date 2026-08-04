# %%
"""Run the full mophongo pipeline on the mock_demo output.

Modeled on run_770.py, but:
- no r_trial subsection
- MIRI picks psf_size automatically from the 95% EE growth curve
  (DrizzlePSF.get_psf_radec with ee_fraction=0.95); F444W reuses
  dpsf_miri.psf_size so paired cutouts block-bin cleanly
- segmap + catalog are auto-detected on the mock F444W via Catalog
"""
from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap
from mophongo.catalog import Catalog, get_bg_and_ivar
from mophongo.fit import FitConfig
from mophongo.pipeline import Pipeline
import mophongo.utils as utils

# %%
# inputs from mock_demo
mock_dir = Path(__file__).parent / "mock_out_demo"
psf_dir = Path(__file__).parent.parent / "data" / "PSF"

miri_filt = "770"
version = "v0.1"

sci_444 = mock_dir / "mock_f444w_sci.fits"
wht_444 = mock_dir / "mock_f444w_wht.fits"
csv_444 = mock_dir / "mock_f444w_wcs.csv"
sci_miri_f = mock_dir / f"mock_f{miri_filt}w_sci.fits"
wht_miri_f = mock_dir / f"mock_f{miri_filt}w_wht.fits"
csv_miri = mock_dir / f"mock_f{miri_filt}w_wcs.csv"

out_root = Path(__file__).parent / f"mock_run_{miri_filt}"
out_root.mkdir(exist_ok=True)

stpsf_444 = "UDS_NRC.._F444W_OS4_GRID1"
stpsf_miri = f"UDS_MIRI_F{miri_filt}W_OS4_GRID1"

# %%
# drizzle PSF models
dpsf_444 = DrizzlePSF(driz_image=str(sci_444), csv_file=str(csv_444))
dpsf_miri = DrizzlePSF(driz_image=str(sci_miri_f), csv_file=str(csv_miri))

# unique detector-overlap regions, clipped to the mosaic footprint
prm_444 = PSFRegionMap.from_footprints(
    dpsf_444.footprint, name="F444W"
).overlay_with(dpsf_444.driz_footprint)
prm_miri = PSFRegionMap.from_footprints(
    dpsf_miri.footprint, name=f"F{miri_filt}W"
).overlay_with(dpsf_miri.driz_footprint)
prm_kern = prm_444.overlay_with(prm_miri)

# %%
# build or load PSFs + kernels
psf_444_file = out_root / "prm_f444w_psf.geojson"
psf_miri_file = out_root / f"prm_f{miri_filt}w_psf.geojson"
kern_file = out_root / f"prm_f444w_kernel_f{miri_filt}w.geojson"

if not kern_file.exists():
    pos = [np.squeeze(p.xy) for p in prm_kern.regions.geometry.centroid]

    dpsf_444.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_444)
    dpsf_miri.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_miri)

    # MIRI picks the size via 95% EE, F444W inherits it so pixel ratio stays clean
    prm_miri.psfs = dpsf_miri.get_psf_radec(pos, ee_fraction=0.95, verbose=True)
    prm_444.psfs = dpsf_444.get_psf_radec(pos, size=dpsf_miri.psf_size, verbose=True)
    logging.info(
        "psf_size MIRI=%.3f\" -> F444W cutout %.3f\"",
        dpsf_miri.psf_size, dpsf_444.psf_size,
    )

    prm_444.to_file(psf_444_file)
    prm_miri.to_file(psf_miri_file)

    pixel_ratio = round(dpsf_miri.driz_pscale / dpsf_444.driz_pscale)
    kernels = [
        utils.matching_kernel(p444, pmiri, recenter=True, pixel_ratio=pixel_ratio)
        for p444, pmiri in zip(prm_444.psfs, prm_miri.psfs)
    ]
    prm_kern.psfs = np.asarray(kernels)
    prm_kern.to_file(kern_file)
else:
    prm_444 = PSFRegionMap.from_geojson(str(psf_444_file))
    prm_miri = PSFRegionMap.from_geojson(str(psf_miri_file))
    prm_kern = PSFRegionMap.from_geojson(str(kern_file))

# %%
# detection + segmap on the mock F444W
wcs_444 = WCS(fits.getheader(sci_444))
wcs_miri = WCS(fits.getheader(sci_miri_f))

cat_obj = Catalog.from_fits(
    sci_444, wht_444,
    estimate_background=True,
    estimate_ivar=True,
)
cat = cat_obj.table
segmap = cat_obj.segmap.data

tmpl_444 = fits.getdata(sci_444).astype(np.float32)
sci_miri = fits.getdata(sci_miri_f).astype(np.float32)
wht_miri = fits.getdata(wht_miri_f).astype(np.float32)

bg_miri, ivar_miri = get_bg_and_ivar(sci_miri, wht_miri, bg_filter_sigma=64.0)

# %%
config = FitConfig(
    reg_flux=0.0,
    reg_astrom=0.0,
    fit_astrometry_niter=2,
    fit_astrometry_joint=True,
    scene_minimum_bright=10,
    aperture_diam=0.5,
    template_dilate_segmap=12,
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

# %%
fits.writeto(
    out_root / f"mock_{miri_filt}_{version}_residual.fits",
    res[0], fits.getheader(sci_444), overwrite=True,
)
table.write(
    out_root / f"mock_{miri_filt}_{version}_fit_table.fits", overwrite=True,
)

# match truth back onto the fit table so accuracy is easy to inspect
truth = Table.read(mock_dir / "mock_truth.ecsv")
truth.write(
    out_root / f"mock_{miri_filt}_{version}_truth.ecsv", overwrite=True,
)

from matplotlib import pyplot as plt

scenes = pipe.all_scenes[0]
for s in scenes:
    print(f"scene {s.id} sources {len(s.templates)} bright {s.is_bright.sum()}")
    fig, _ = s.plot(tmpl_444, segmap, display_sig=5)
    fig.savefig(out_root / f"mock_{miri_filt}_{version}_scene_{s.id}.png", dpi=200)
    plt.close(fig)

# %%
