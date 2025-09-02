import numpy as np
import pytest
import scipy.sparse as sp

from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.scene import Scene
from mophongo.templates import Templates, Template
from mophongo.fit import SparseFitter, FitConfig
from mophongo.psf import PSF
from utils import make_simple_data


def test_scene_fitter_flux_only():
    A = sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    alpha, err, beta, info = SceneFitter.solve(A, b)
    expected = np.linalg.solve(A.toarray(), b)
    cov = np.linalg.inv(A.toarray())
    expected_err = np.sqrt(np.diag(cov))
    assert info == 0
    assert beta is None
    np.testing.assert_allclose(alpha, expected)
    np.testing.assert_allclose(err, expected_err)


def test_scene_fitter_with_shift_block():
    A = sp.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    AB = sp.csr_matrix([[1.0], [2.0]])
    BB = sp.csr_matrix([[3.0]])
    bB = np.array([0.5])
    alpha, err, beta, info = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB)
    M = np.block([[A.toarray(), AB.toarray()], [AB.T.toarray(), BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    np.testing.assert_allclose(alpha, dense[:2])
    np.testing.assert_allclose(beta, dense[2:])


@pytest.mark.parametrize("order", [0, 1, 2])
def test_solve_flux_and_shifts_matches_dense(order):
    nA = 4
    p = (order + 1) * (order + 2) // 2
    nB = p * 2
    A = sp.eye(nA, format="csr") * 2.0
    AB = np.full((nA, nB), 0.1)
    BB = sp.eye(nB, format="csr") * 3.0
    b = np.arange(1, nA + 1, dtype=float)
    bB = np.arange(1, nB + 1, dtype=float)
    cfg = FitConfig(cg_kwargs={"rtol": 1e-10, "maxiter": 1000})
    x, err, beta, info = SceneFitter._solve_flux_and_shifts(
        A, b, sp.csr_matrix(AB), BB, bB, config=cfg
    )
    M = np.block([[A.toarray(), AB], [AB.T, BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    cov = np.linalg.inv(M)
    np.testing.assert_allclose(x, dense[:nA], rtol=1e-3)
    np.testing.assert_allclose(beta, dense[nA:], rtol=1e-3)
    np.testing.assert_allclose(err, np.sqrt(np.diag(cov)[:nA]), rtol=1e-3)
    assert info["cg_info"] == 0


def test_scene_graph_and_residuals():
    img = np.zeros((10, 10))
    size = (3, 3)
    t1 = Template(img, (2, 2), size, label=1)
    t2 = Template(img, (2, 3), size, label=2)
    t3 = Template(img, (7, 7), size, label=3)
    scene_labels = Scene.create_scene_graph([t1, t2, t3])
    assert scene_labels[t1.id - 1] == scene_labels[t2.id - 1]
    assert scene_labels[t1.id - 1] != scene_labels[t3.id - 1]
    seg, labels = Scene.overlay_scene_graph([t1, t2, t3], img.shape)
    assert seg[t1.bbox[0], t1.bbox[2]] == seg[t2.bbox[0], t2.bbox[2]]
    assert seg[t3.bbox[0], t3.bbox[2]] != seg[t1.bbox[0], t1.bbox[2]]
    # Residuals
    for tmpl in (t1, t2, t3):
        tmpl.data[...] = 1.0
    sc = Scene([t1, t2], SceneFitter())
    coeffs = np.array([2.0, 3.0])
    res = sc.add_residuals(np.zeros_like(img), coeffs)
    expected = np.zeros_like(img)
    expected[t1.slices_original] -= 2.0
    expected[t2.slices_original] -= 3.0
    np.testing.assert_array_almost_equal(res, expected)


@pytest.mark.parametrize("order", [1, 2])
def test_scene_solve_matches_legacy_solver(order):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=51, peak_snr=5, seed=order
    )
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)
    image = images[1]
    weight = wht[1]
    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=0.0,
        astrom_kwargs={"poly": {"order": order}},
    )
    fitter = SparseFitter(tmpls.templates, image, weight, cfg)
    A, b, _ = build_normal(tmpls.templates, image, weight)
    d = np.sqrt(A.diagonal())
    Dinv = sp.diags(1.0 / d)
    A_w = Dinv @ A @ Dinv
    b_w = b / d
    scene_ids = np.ones(len(tmpls.templates), dtype=int)
    bright = np.ones(len(tmpls.templates), dtype=bool)
    alpha_legacy, err_legacy, betas, infos = fitter._solve_scenes_with_shifts(
        A_w,
        b_w,
        d,
        scene_ids,
        tmpls.templates,
        bright,
        order=order,
        include_y=True,
        ab_from_bright_only=True,
    )
    beta_legacy = betas[0][1]

    scene = Scene(id=1, templates=list(tmpls.templates), fitter=SceneFitter())
    scene.A = A
    scene.b = b
    scene.image = image
    scene.weights = weight
    flux, err, beta_scene, info = scene.solve(
        config=FitConfig(
            fit_astrometry_joint=True,
            snr_thresh_astrom=0.0,
            astrom_kwargs={"poly": {"order": order}},
        ),
        apply_shifts=False,
    )
    np.testing.assert_allclose(flux, alpha_legacy, rtol=1e-3)
