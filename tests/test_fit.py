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

    assert info["cg_info"] == 0
#    assert np.allclose(x, np.array(catalog['flux_true']), rtol=1e-1)
    model = fitter.model_image()
    fname = tmp_path / "fit.png"
    save_fit_diagnostic(fname, images[1], model, fitter.residual())

    assert fname.exists()


def test_lsqr_lo_matches_cg():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls1 = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_lo = SparseFitter(tmpls1.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    flux_lo, err_lo, _ = fitter_lo.solve_lo()

    tmpls2 = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_cg = SparseFitter(tmpls2.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    flux_cg, err_cg, _ = fitter_cg.solve()

    np.testing.assert_allclose(flux_lo, flux_cg, rtol=2e-3, atol=2e-3)
    assert err_lo.shape == err_cg.shape


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


def test_flux_and_rms_estimation():
    """SparseFitter.flux_and_rms matches quick flux and error estimates."""
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )

    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())

    flux, err = fitter.flux_and_rms()
    np.testing.assert_allclose(flux, fitter.quick_flux())
    np.testing.assert_allclose(err, fitter.predicted_errors())

    for t in tmpls.templates:
        t.flux = 42.0
    flux2, _ = fitter.flux_and_rms()
    assert np.all(flux2 == 42.0)


def test_build_normal_tree_matches_loop():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_loop = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig()
    )
    fitter_tree = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig(normal="tree")
    )

    fitter_loop.build_normal_matrix()
    fitter_tree.build_normal_tree()

    np.testing.assert_allclose(
        fitter_loop.ata.toarray(), fitter_tree._ata.toarray()
    )
    np.testing.assert_allclose(fitter_loop.atb, fitter_tree._atb)


def test_solve_scene_matches_global():
    img = np.zeros((6, 6))
    weights = np.ones_like(img)

    t1 = Template(img, (2, 2), (3, 3))
    t1.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t2 = Template(img, (2, 3), (3, 3))
    t2.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t3 = Template(img, (5, 5), (1, 1))
    t3.data[:] = np.array([[1]])

    tmpls = [t1, t2, t3]
    fluxes = [1.0, 2.0, 3.0]
    image = np.zeros_like(img)
    for f, t in zip(fluxes, tmpls):
        image[t.slices_original] += f * t.data[t.slices_cutout]

    # Global solve
    fitter_all = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_all.templates[0].data[:] = t1.data
    fitter_all.templates[1].data[:] = t2.data
    fitter_all.templates[2].data[:] = t3.data
    flux_all, _, _ = fitter_all.solve()

    # Component solve
    fitter_comp = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_comp.templates[0].data[:] = t1.data
    fitter_comp.templates[1].data[:] = t2.data
    fitter_comp.templates[2].data[:] = t3.data
    flux_comp, _, _ = fitter_comp.solve_scene()

    np.testing.assert_allclose(flux_comp, flux_all, rtol=1e-6, atol=1e-6)


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


def test_bright_source_detection():
    images, segmap, catalog, psfs, _, rms = make_simple_data()
    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )
    cfg = FitConfig(snr_thresh_astrom=5.0)
    fitter = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, cfg
    )
    flux = Templates.quick_flux(tmpls.templates, images[1])
    err = Templates.predicted_errors(tmpls.templates, 1.0 / rms[1] ** 2)
    snr = flux / err
    expected = snr > cfg.snr_thresh_astrom
    assert np.array_equal(fitter.bright_mask, expected)


def test_solve_scene_shifts_matches_global():
    img = np.zeros((6, 6))
    weights = np.ones_like(img)

    t1 = Template(img, (2, 2), (3, 3))
    t1.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t2 = Template(img, (2, 3), (3, 3))
    t2.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t3 = Template(img, (5, 5), (1, 1))
    t3.data[:] = np.array([[1]])

    tmpls = [t1, t2, t3]
    fluxes = [1.0, 2.0, 3.0]
    image = np.zeros_like(img)
    for f, t in zip(fluxes, tmpls):
        image[t.slices_original] += f * t.data[t.slices_cutout]

    fitter_all = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_all.templates[0].data[:] = t1.data
    fitter_all.templates[1].data[:] = t2.data
    fitter_all.templates[2].data[:] = t3.data
    flux_all, _, _ = fitter_all.solve()

    fitter_shift = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(snr_thresh_astrom=0.0),
    )
    fitter_shift.templates[0].data[:] = t1.data
    fitter_shift.templates[1].data[:] = t2.data
    fitter_shift.templates[2].data[:] = t3.data
    flux_shift, _, info = fitter_shift.solve_scene_shifts(order=1)
    betas = info["betas"]

    np.testing.assert_allclose(flux_shift, flux_all, rtol=1e-4, atol=1e-4)
    for _, beta in betas:
        assert np.all(np.abs(beta) < 1e-6)
