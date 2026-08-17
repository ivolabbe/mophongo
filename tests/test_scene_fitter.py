"""Solving a scene's normal equations: the flux block, the shift block, and
the quick flux/error estimates that stand in for a solve."""

import numpy as np
import scipy.sparse as sp

from mophongo.fit import FitConfig, SparseFitter
from mophongo.psf import PSF
from mophongo.scene import Scene
from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.templates import Templates
from utils import make_simple_data


def test_scene_fitter_flux_only():
    A = sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    cfg = FitConfig(reg_flux=1e-12)
    sol = SceneFitter.solve(A, b, config=cfg)
    expected = np.linalg.solve(A.toarray() + np.eye(A.shape[0]) * cfg.reg_flux, b)

    assert sol.info["solver"] == "fnnls"  # nnls is the default
    assert sol.shifts is None
    np.testing.assert_allclose(sol.flux, expected)
    assert np.all(np.isfinite(sol.err))


def test_scene_fitter_astrom_reg_does_not_regularize_flux():
    A = sp.csr_matrix([[1e-3]])
    b = np.array([1e-3])
    sol = SceneFitter.solve(A, b, config=FitConfig(reg_flux=0.0, astrom_reg=1e-2))
    np.testing.assert_allclose(sol.flux, [1.0], rtol=1e-5)


def test_scene_fitter_with_shift_block():
    A = sp.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    AB = sp.csr_matrix([[1.0], [2.0]])
    BB = sp.csr_matrix([[3.0]])
    bB = np.array([0.5])
    cfg = FitConfig(reg_flux=1e-12, astrom_reg=0.0, fit_method="lls")
    sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg)
    M = np.block([[A.toarray(), AB.toarray()], [AB.T.toarray(), BB.toarray()]])
    M[: A.shape[0], : A.shape[0]] += np.eye(A.shape[0]) * cfg.reg_flux
    dense = np.linalg.solve(M, np.concatenate([b, bB]))
    np.testing.assert_allclose(sol.flux, dense[:2])
    np.testing.assert_allclose(sol.shifts, dense[2:])


def test_solve_can_skip_the_flux_errors():
    """``errors=False`` leaves the fluxes alone and says so with NaN.

    ``sqrt(diag(A^-1))`` costs a factorization plus a back-solve per template,
    which a caller that only wants the fluxes -- the seed of an intermediate
    residual -- should not have to pay. NaN rather than zero so a skipped
    error cannot be mistaken for a measured one.
    """
    A = sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    AB = sp.csr_matrix([[1.0], [2.0]])
    BB = sp.csr_matrix([[3.0]])
    bB = np.array([0.5])
    cfg = FitConfig(reg_flux=1e-12, astrom_reg=0.0, fit_method="lls")

    for blocks in ({}, dict(AB=AB, BB=BB, bB=bB)):
        full = SceneFitter.solve(A, b, config=cfg, **blocks)
        cheap = SceneFitter.solve(A, b, config=cfg, errors=False, **blocks)
        np.testing.assert_allclose(cheap.flux, full.flux)
        if blocks:
            np.testing.assert_allclose(cheap.shifts, full.shifts)
        assert np.all(np.isfinite(full.err))
        assert np.all(np.isnan(cheap.err))


def test_scene_solve_matches_dense_normal_solution():
    """Scene.solve on real templates reproduces a dense solve of the same system."""
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=51, peak_snr=5, seed=1
    )
    kernel = PSF.from_array(psfs[0]).matching_kernel(PSF.from_array(psfs[1]))
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    image = images[1]
    weight = wht[1]
    A, b, _ = build_normal(tmpls.templates, image, weight)

    cfg = FitConfig(reg_flux=1e-12, fit_method="lls", fit_astrometry_niter=0)
    scene = Scene(id=1, templates=list(tmpls.templates), fitter=SceneFitter())
    scene.A = A
    scene.b = b
    scene.image = image
    scene.weights = weight
    flux, *_ = scene.solve(config=cfg, apply_shifts=False)

    expected = np.linalg.solve(A.toarray() + np.eye(A.shape[0]) * cfg.reg_flux, b)
    np.testing.assert_allclose(flux, expected, rtol=1e-3, atol=1e-6)


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

    for tmpl in tmpls.templates:
        tmpl.flux = 42.0
    flux2, _ = fitter.flux_and_rms()
    assert np.all(flux2 == 42.0)
