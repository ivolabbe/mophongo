# %%
"""High-SNR mock: identical to mock_demo but SNR ~ log-uniform(100, 10000).

Regenerates mock_out_hisnr/ (mosaics + truth) and runs the F770W pipeline
into mock_run_770_hisnr/. Everything else (pointings, STPSF, noise seed)
mirrors examples/mock_demo.ipynb + examples/run_mock.py.
"""
import os, logging
os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
logging.getLogger("pysiaf").setLevel(logging.ERROR)
logging.getLogger("numexpr").setLevel(logging.WARNING)

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from mophongo.mock_mosaic import MockMosaic, Pointing
from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap
from mophongo.catalog import Catalog, get_bg_and_ivar
from mophongo.fit import FitConfig
from mophongo.pipeline import Pipeline
import mophongo.utils as utils

here = Path(__file__).parent
mock_dir = here / "mock_out_hisnr"
mock_dir.mkdir(exist_ok=True)
psf_dir = here.parent / "data" / "PSF"
out_root = here / "mock_run_770_hisnr"
out_root.mkdir(exist_ok=True)

miri_filt = "770"
version = "v0.1"
stpsf_444 = "UDS_NRC.._F444W_OS4_GRID1"
stpsf_miri = f"UDS_MIRI_F{miri_filt}W_OS4_GRID1"

# %% 1. build mock (only if truth is missing)
truth_file = mock_dir / "mock_truth.ecsv"
if not truth_file.exists():
    center = (34.50, -5.20)
    MIRI_DET_WIDTH_DEG = 1032 * 0.110 / 3600.0
    ra2 = center[0] - 0.4 * MIRI_DET_WIDTH_DEG / np.cos(np.deg2rad(center[1]))
    dec2 = center[1] + 0.4 * MIRI_DET_WIDTH_DEG
    mock = MockMosaic(
        out_dir=mock_dir, center_radec=center,
        nircam_lw_frames={"f444w": [Pointing(*center, pa=0.0),
                                    Pointing(*center, pa=30.0)]},
        miri_frames={"f770w": [Pointing(*center, pa=0.0),
                               Pointing(ra2, dec2, pa=-30.0)]},
        mosaic_pscale="nircam_lw",
        exptime={"f444w": 418.734, "f770w": 444.006},
        psf_size_arcsec={"f444w": 8.0, "f770w": 8.0},
        pixfrac={"nircam_lw": 0.75, "miri": 1.00},
        stpsf_patterns={"f444w": stpsf_444, "f770w": stpsf_miri},
        stpsf_dir=psf_dir,
        snr_range=(100.0, 10000.0),
        apertures_arcsec=(0.32, 0.7),
        noise_seed=42,
    )
    import json
    mock.to_json(mock_dir / "mock_config.json")
    paths, noise_info, dpsfs, truth = mock.build(n_sources=1000)
    mock.report()
else:
    logging.info("re-using existing mock in %s", mock_dir)

# %% 2. pipeline inputs
sci_444 = mock_dir / "mock_f444w_sci.fits"
wht_444 = mock_dir / "mock_f444w_wht.fits"
csv_444 = mock_dir / "mock_f444w_wcs.csv"
sci_miri_f = mock_dir / f"mock_f{miri_filt}w_sci.fits"
wht_miri_f = mock_dir / f"mock_f{miri_filt}w_wht.fits"
csv_miri = mock_dir / f"mock_f{miri_filt}w_wcs.csv"

dpsf_444 = DrizzlePSF(driz_image=str(sci_444), csv_file=str(csv_444))
dpsf_miri = DrizzlePSF(driz_image=str(sci_miri_f), csv_file=str(csv_miri))
prm_444 = PSFRegionMap.from_footprints(dpsf_444.footprint, name="F444W").overlay_with(dpsf_444.driz_footprint)
prm_miri = PSFRegionMap.from_footprints(dpsf_miri.footprint, name=f"F{miri_filt}W").overlay_with(dpsf_miri.driz_footprint)
prm_kern = prm_444.overlay_with(prm_miri)

psf_444_file = out_root / "prm_f444w_psf.geojson"
psf_miri_file = out_root / f"prm_f{miri_filt}w_psf.geojson"
kern_file = out_root / f"prm_f444w_kernel_f{miri_filt}w.geojson"

if not kern_file.exists():
    pos = [np.squeeze(p.xy) for p in prm_kern.regions.geometry.centroid]
    dpsf_444.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_444)
    dpsf_miri.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=stpsf_miri)
    # Debug: use the full 8" stamp for fit PSF + kernel (same extent as injection)
    PSF_SIZE_ARCSEC = 8.0
    prm_miri.psfs = dpsf_miri.get_psf_radec(pos, size=PSF_SIZE_ARCSEC, verbose=True)
    prm_444.psfs = dpsf_444.get_psf_radec(pos, size=PSF_SIZE_ARCSEC, verbose=True)
    prm_444.to_file(psf_444_file)
    prm_miri.to_file(psf_miri_file)
    pixel_ratio = round(dpsf_miri.driz_pscale / dpsf_444.driz_pscale)
    kernels = [utils.matching_kernel(p444, pmiri, recenter=True, pixel_ratio=pixel_ratio)
               for p444, pmiri in zip(prm_444.psfs, prm_miri.psfs)]
    prm_kern.psfs = np.asarray(kernels)
    prm_kern.to_file(kern_file)
else:
    prm_444 = PSFRegionMap.from_geojson(str(psf_444_file))
    prm_miri = PSFRegionMap.from_geojson(str(psf_miri_file))
    prm_kern = PSFRegionMap.from_geojson(str(kern_file))

# %% 3. detection + fit
wcs_444 = WCS(fits.getheader(sci_444))
wcs_miri = WCS(fits.getheader(sci_miri_f))
cat_obj = Catalog.from_fits(sci_444, wht_444, estimate_background=True, estimate_ivar=True)
cat = cat_obj.table
segmap = cat_obj.segmap.data

tmpl_444 = fits.getdata(sci_444).astype(np.float32)
sci_miri = fits.getdata(sci_miri_f).astype(np.float32)
wht_miri = fits.getdata(wht_miri_f).astype(np.float32)
bg_miri, ivar_miri = get_bg_and_ivar(sci_miri, wht_miri, bg_filter_sigma=64.0)

config = FitConfig(
    fit_astrometry_niter=2,
    fit_astrometry_joint=True,
    scene_minimum_anchors=10,
    aperture_diam=0.5,
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

fits.writeto(out_root / f"mock_{miri_filt}_{version}_residual.fits",
             res[0], fits.getheader(sci_444), overwrite=True)
table.write(out_root / f"mock_{miri_filt}_{version}_fit_table.fits", overwrite=True)
Table.read(mock_dir / "mock_truth.ecsv").write(
    out_root / f"mock_{miri_filt}_{version}_truth.ecsv", overwrite=True)

from matplotlib import pyplot as plt
for s in pipe.all_scenes[0]:
    print(f"scene {s.id} sources {len(s.templates)} bright {s.is_bright.sum()}")
    fig, _ = s.plot(
        tmpl_444, segmap, display_sig=1.0,
        display_sig_by_title={"Template": 0.25, "Image": 2.0, "Model": 2.0, "Residual": 3.0},
    )
    fig.savefig(out_root / f"mock_{miri_filt}_{version}_scene_{s.id}.png", dpi=200)
    plt.close(fig)
