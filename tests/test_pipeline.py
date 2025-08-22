import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.nddata import Cutout2D
from photutils.psf.matching import SplitCosineBellWindow, TukeyWindow
import mophongo.utils as mutils

import mophongo.pipeline as pipeline
from utils import (
    make_simple_data,
    save_diagnostic_image,
    save_flux_vs_truth_plot,
)

from astroquery.mast import Observations, Mast
from astropy.io import fits
from astropy.table import Table


def test_download_rate():

    from astroquery.mast import Observations, Mast
    from astropy.io import fits
    from astropy.table import Table

    # ---- 1) Identify the dataset (obs_id) ----
    dataset = "jw01837001001_06101_00002"  # your valid example (no detector/suffix)

    # (Optional) If you might hit EAP data, login first:
    # Mast.login(token="YOUR_MAST_TOKEN")  # see https://mast.stsci.edu for token

    # Try the standard observations search
    obs = Observations.query_criteria(obs_collection="JWST", obs_id=dataset)
    if len(obs) == 0:
        # Fallback: Advanced CAOM “Filtered” service on obs_id
        params = {
            "columns": "*",
            "filters": [
                {"paramName": "obs_collection", "values": [{"value": "JWST"}]},
                {"paramName": "obs_id", "values": [{"value": dataset, "operator": "="}]},
            ],
        }
        tbl = Mast.service_request("Mast.Caom.Filtered", params)  # TableRows JSON
        obs = Table(tbl) if len(tbl) > 0 else obs  # keep Table-like for get_product_list
    # (If still empty: dataset is wrong or still proprietary to another account.)

    # ---- 2) Get products and keep rate/cal FITS for any detector ----
    prods = Observations.get_product_list(obs)

    want = Observations.filter_products(
        prods,
        productFilename=[f"{dataset}_*rate.fits", f"{dataset}_*cal.fits"],
        extension="fits",
        mrp_only=False,
    )

    # Convenience: turn MAST dataURI into a direct download URL
    def mast_url_from_uri(data_uri: str) -> str:
        return f"https://mast.stsci.edu/api/v0.1/Download/file/?uri={data_uri}"

    urls = [mast_url_from_uri(u) for u in want["dataURI"]]

    # ---- 3) Read ONLY headers (no full download) ----
    # Astropy will use HTTP Range requests with fsspec; it streams just what it needs.
    u = urls[0]
    with fits.open(u, use_fsspec=True) as hdul:
        phdr = hdul[0].header
        # example: read SCI header too (if present)
        # ehdr = hdul["SCI"].header

    print(phdr.tostring(sep="\n")[:1000])


def test_pipeline_flux_recovery(tmp_path):
    #    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(seed=5, nsrc=300, size=501, ndilate=1, peak_snr=1)
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms)

    images, segmap, catalog, psfs, truth_img, wht = make_simple_data(
        seed=5, nsrc=151, size=301, ndilate=2, peak_snr=1.5
    )
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms, extend_templates='psf')

    # add the hires images as the first fitting image, so that we can compare fluxes
    images.insert(0, images[0])
    wht.insert(0, wht[0])
    psfs.insert(0, psfs[0])
    # images are: hires, hires, lowres
    # psfs are:   hires, hires, lowres
    # so this would add psf hires wings to templates, and result in a delta function for kernel
    dirac = lambda n: ((np.arange(n)[:, None] == n // 2) & (np.arange(n) == n // 2)).astype(float)

    kernel = [mutils.matching_kernel(psfs[0], psf) for psf in psfs]
    kernel[0] = kernel[1] = dirac(3)  # no kernel for the first image, it is the hires image
    table, resid, templates = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=kernel
    )

    # @@@ sometimes flux_true is NEGATIVE?
    table["flux_true"] = catalog["flux_true"]  # add flux_true to the table

    # Plot for high-res (flux_0) vs truth
    flux_hi_plot = tmp_path / "flux_hi_vs_true.png"
    save_flux_vs_truth_plot(
        flux_hi_plot,
        table["flux_true"],
        table["flux_1"],
        error=table["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_hi_plot.exists()

    # Plot for low-res (flux_1) vs truth
    flux_lo_plot = tmp_path / "flux_lo_vs_true.png"
    save_flux_vs_truth_plot(
        flux_lo_plot,
        table["flux_true"],
        table["flux_2"],
        error=table["err_2"],  # Add error column
        label="Flux (lowres)",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_plot.exists()

    # Plot for flux_lo vs flux_hi with error propagation
    flux_lo_hi_plot = tmp_path / "flux_lo_vs_hi.png"
    # Calculate combined error for hires vs lowres comparison
    combined_error = np.sqrt(table["err_1"] ** 2 + table["err_2"] ** 2)
    save_flux_vs_truth_plot(
        flux_lo_hi_plot,
        table["flux_1"],
        table["flux_2"],
        error=combined_error,
        label="Flux (lowres) vs (hires)",
        xlabel="Recovered Flux (hires)",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_hi_plot.exists()

    # ----------------------------------- separate run for high-res, using the truth image as templates
    # images, segmap, catalog=catalog, psfs=psfs,  wht_images=wht)
    table_true, resid_hi, templates_true = pipeline.run(
        [truth_img, images[1]],
        segmap,
        catalog=catalog,
        kernels=[dirac(3), psfs[1]],
        weights=[np.zeros(wht[0].shape), wht[1]],
    )
    # Plot for high-res (flux_0) vs truth
    flux_true_plot = tmp_path / "flux_hi_vs_true_truemodel.png"
    save_flux_vs_truth_plot(
        flux_true_plot,
        table_true["flux_true"],
        table_true["flux_1"],
        error=table_true["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_true_plot.exists()

    model = images[2] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(
        fname, truth_img, images[1], images[2], model, resid[1], segmap=segmap, catalog=catalog
    )
    fname = tmp_path / "diagnostic_hires.png"
    model = images[1] - resid[0]
    save_diagnostic_image(
        fname, truth_img, images[0], images[1], model, resid[0], segmap=segmap, catalog=catalog
    )

    fname = tmp_path / "diagnostic_hires_truemodel.png"
    model = images[1] - resid_hi[0]
    save_diagnostic_image(
        fname, truth_img, truth_img, images[1], model, resid_hi[0], segmap=segmap, catalog=catalog
    )
    assert fname.exists()

    # Report statistics for flux recovery
    for idx in range(1, len(psfs)):
        col = f"flux_{idx}"
        ratio = np.array(table[col]) / np.array(table["flux_true"])
        print(
            f"flux_{idx}/flux_true percentiles: 5th={np.percentile(ratio,5):.2f}, "
            f"16th={np.percentile(ratio,16):.2f}, 50th={np.percentile(ratio,50):.2f}, "
            f"84th={np.percentile(ratio,84):.2f}, 95th={np.percentile(ratio,95):.2f}"
        )

    # sanity check on propagated errors for low-res image
    from mophongo.psf import PSF
    from mophongo.templates import Templates

    psf_hi = PSF.from_array(psfs[1])
    psf_lo = PSF.from_array(psfs[2])
    kernel = psf_hi.matching_kernel(psf_lo)
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    noise_std = wht[1][0, 0]
    err_pred = np.array([noise_std / np.sqrt((t.data**2).sum()) for t in tmpls.templates])
    ratio_err = table["err_1"] / err_pred
    assert np.allclose(np.mean(ratio_err), 1.0, atol=3)

    # Write catalog with all columns formatted to 3 digits after the decimal
    for col in table.colnames:
        if table[col].dtype.kind in "fc":  # float or complex
            table[col].info.format = ".3f"

    cat_file = tmp_path / "photometry.cat"
    table.write(cat_file, format="ascii.commented_header")
    assert cat_file.exists()

    loaded = Table.read(cat_file, format="ascii.commented_header")
    assert len(loaded) == len(table)


def test_pipeline_astrometry(tmp_path):
    return
    from scipy.ndimage import shift as nd_shift, map_coordinates
    from mophongo.fit import FitConfig, SparseFitter
    from mophongo.astro_fit import GlobalAstroFitter

    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=20, size=151, peak_snr=1, seed=11, border_size=15
    )

    h, w = images[0].shape
    y, x = np.mgrid[0:h, 0:w]
    shift_x = -1.5 * x / w + 0.5 * (x / w) ** 2  # quadratic in x
    shift_y = -2.0 * y / h + 0.3 * (y / h) ** 2  # quadratic in y
    shift_field = np.sqrt(shift_x**2 + shift_y**2)
    images[1] = map_coordinates(images[0], [y - shift_y, x - shift_x], order=3, mode="constant")
    print(f"Shift field: {shift_field.min()} to {shift_field.max()} pixels")

    kern1 = mutils.matching_kernel(psfs[0], psfs[1])

    config = FitConfig(
        fit_astrometry_niter=2,
        astrom_basis_order=1,
        reg_astrom=1e-4,
        snr_thresh_astrom=10.0,
    )
    table, res0, fit0 = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=[None, kern1], config=config
    )
