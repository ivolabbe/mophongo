import numpy as np
from mophongo.pipeline import run_photometry
from mophongo.sim_data import make_mosaic_dataset
from utils import save_flux_vs_truth_plot, save_diagnostic_image


def test_realistic_pipeline(tmp_path):
    ds = make_mosaic_dataset(seed=3)

    # average PSF for detection
    psf_det = (ds.psf_f444w[0] + ds.psf_f444w[1]) / 2
    psfs = [psf_det, ds.psf_f770w[0]]

    images = [ds.f444w, ds.f770w]
    wht = [np.ones_like(ds.f444w), np.ones_like(ds.f770w) / 25.0]

    table, resid, _ = run_photometry(images, ds.segmap, ds.catalog, psfs, wht)

    table['flux_true'] = ds.catalog['flux_true'] * ds.catalog['ratio_770']

    flux_plot = tmp_path / "flux_vs_true.png"
    save_flux_vs_truth_plot(flux_plot, table['flux_true'], table['flux_1'], table['err_1'])
    assert flux_plot.exists()

    model = images[1] - resid[0]
    diag = tmp_path / "diagnostic.png"
    save_diagnostic_image(diag, ds.truth_f770w, ds.f444w, images[1], model, resid[0], segmap=ds.segmap, catalog=ds.catalog)
    assert diag.exists()
 
