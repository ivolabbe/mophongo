import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.fit import FitConfig, SparseFitter
from mophongo.psf import PSF
from mophongo.templates import Templates, Template 
from utils import make_simple_data, save_fit_diagnostic


def test_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )

    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    fitter.build_normal_matrix()
    x, err, info = fitter.solve()

    assert info == 0
#    assert np.allclose(x, np.array(catalog['flux_true']), rtol=1e-1)
    model = fitter.model_image()
    fname = tmp_path / "fit.png"
    save_fit_diagnostic(fname, images[1], model, fitter.residual())

    assert fname.exists()


def test_ata_symmetry():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    fitter.build_normal_matrix()
    ata = fitter.ata.toarray()
    assert np.allclose(ata, ata.T)


def test_zero_weight_template_dropped():
    img = np.zeros((4, 4))
    weights = np.ones_like(img)
    weights[2:4, 2:4] = 0

    t1 = Template(img, (1, 1), (2, 2))
    t1.data[:] = 1.0
    t2 = Template(img, (3, 3), (2, 2))
    t2.data[:] = 1.0

    fitter = SparseFitter([t1, t2], img, weights, FitConfig())
    fitter.build_normal_matrix()

    assert len(fitter.templates) == 1


def test_flux_errors_regularized():
    img = np.zeros((3, 3))
    weights = np.ones_like(img)
    tmpl_data = np.zeros((3, 3))
    tmpl_data[1, 1] = 1.0

    t1 = Template(img, (1, 1), (3, 3))
    t1.data[:] = tmpl_data
    t2 = Template(img, (1, 1), (3, 3))
    tmpl_data2 = tmpl_data.copy()
    tmpl_data2[0, 0] = 0.1  # break perfect degeneracy
    t2.data[:] = tmpl_data2

    fitter = SparseFitter([t1, t2], img, weights, FitConfig())
    fitter.build_normal_matrix()
    _, _, _ = fitter.solve()
    err = fitter.flux_errors()

    assert err.size == 2
    assert np.all(np.isfinite(err))


def test_build_normal_matrix_new_equivalence():
    return
    images, segmap, catalog, psfs, _, rms = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    old = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    new = extract_templates(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )

    fitter_old = SparseFitter(old.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    fitter_new = SparseFitter(new, images[1], 1.0 / rms[1] ** 2, FitConfig())
    fitter_old.build_normal_matrix()
    fitter_new.build_normal_matrix_new()

    np.testing.assert_allclose(fitter_old.ata.toarray(), fitter_new._ata.toarray())
    np.testing.assert_allclose(fitter_old.atb, fitter_new._atb)

    fitter_old.solve()
    fitter_new.solve()
    model_old = fitter_old.model_image()
    model_new = fitter_new.model_image_new()
    np.testing.assert_allclose(model_old, model_new)
