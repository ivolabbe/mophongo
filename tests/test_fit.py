import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.fit import FitConfig, SparseFitter
from mophongo.psf import PSF
from mophongo.templates import Templates
from utils import make_simple_data, save_fit_diagnostic


def test_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth, truth_img = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel
    )
    fitter = SparseFitter(list(tmpls), images[1], np.ones_like(images[1]), FitConfig())
    fitter.build_normal_matrix()
    x, info = fitter.solve()

    assert info == 0
    assert np.allclose(x, truth, rtol=2e-2)
    model = fitter.model_image()
    fname = tmp_path / "fit.png"
    save_fit_diagnostic(fname, images[1], model, fitter.residual())
    assert fname.exists()


def test_ata_symmetry():
    images, segmap, catalog, psfs, _, _ = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel
    )
    fitter = SparseFitter(list(tmpls), images[1], np.ones_like(images[1]), FitConfig())
    fitter.build_normal_matrix()
    ata = fitter.ata.toarray()
    assert np.allclose(ata, ata.T)
