"""The non-joint astrometry path: ``AstroCorrect``, ``AstroMap`` and the
Chebyshev basis they share.

The joint scene solver is covered exactly, and to much tighter tolerances,
in ``test_scene_astrometry_blocks.py`` (the normal-equation blocks against a
dense design) and ``test_scene_astrometry_robust.py`` (per-anchor measurement
and robust weighting). What is left here is the separate-step estimator that
those two never touch, plus the one scaling property of the shift block that
only shows when ``alpha0`` is varied by hand.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.ndimage import shift as nd_shift

from utils import make_simple_data
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from mophongo.fit import SparseFitter, FitConfig
from mophongo.astrometry import AstroCorrect, AstroMap, cheb_basis
from mophongo.scene import make_scene_basis, assemble_scene_system_AB
from mophongo.scene_fitter import SceneFitter, build_normal


def _assign_solution(fitter, sol) -> None:
    """Write a SceneFitter solution back onto the fitter and its templates."""
    fitter.solution = sol.flux
    for tmpl, flux, err in zip(fitter.templates, sol.flux, sol.err):
        tmpl.flux = float(flux)
        tmpl.err = float(err)


# ---------------------------------------------------------------------------
# AstroCorrect: the separate-step estimator
# ---------------------------------------------------------------------------


def test_polynomial_astrometry_reduces_residual():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=42
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    shx, shy = 0.6, -0.5
    images[1] = nd_shift(images[1], (shy, shx))

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal()
    sol = SceneFitter.solve(fitter.ata, fitter.atb, config=FitConfig())
    _assign_solution(fitter, sol)
    res0 = fitter.residual()

    ac = AstroCorrect(FitConfig())
    ac.fit(tmpls.templates, res0, fitter.solution)

    rhx, rhy = ac(np.array([[50.0, 50.0]]))

    assert abs(rhx[0] - shx) < 0.3
    assert abs(rhy[0] - shy) < 0.3


def test_gp_astrometry_returns_models():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=101, peak_snr=5, seed=1
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    images[1] = nd_shift(images[1], (-0.5, 0.6))

    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal()
    sol = SceneFitter.solve(fitter.ata, fitter.atb, config=FitConfig())
    _assign_solution(fitter, sol)
    res = fitter.residual()

    cfg = FitConfig(astrom_model="gp", astrom_kwargs={"gp": {"length_scale": 30.0}})
    ac = AstroCorrect(cfg)
    ac.fit(tmpls.templates, res, fitter.solution)

    dx, dy = ac(np.array([[50.0, 50.0]]))
    assert isinstance(float(dx[0]), float)
    assert isinstance(float(dy[0]), float)


def test_astromap_recovers_shift():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=7
    )
    shx, shy = 0.4, -0.3
    shifted = nd_shift(images[0], (shy, shx))
    amap = AstroMap(order=1, snr_threshold=3.0)
    catalog["snr"] = 100.0  # make_simple_data has no snr column; pass the cut
    amap.fit(images[0], shifted, catalog)
    dx, dy = amap(np.array([[75.0, 75.0]]))
    assert abs(dx[0] - shx) < 0.3
    assert abs(dy[0] - shy) < 0.3


# ---------------------------------------------------------------------------
# applying a solved field, and the basis it is expressed in
# ---------------------------------------------------------------------------


def test_apply_template_shifts_uses_shift_field():
    data = np.zeros((7, 7))
    data[3, 3] = 1.0
    tmpl = Template(data, (3.0, 3.0), (7, 7), label=1)
    tmpl.to_shift = np.array([0.5, -0.25])

    Templates.apply_template_shifts([tmpl])

    expected = nd_shift(
        data,
        (-0.25, 0.5),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    assert np.allclose(tmpl.data, expected)
    # ours accumulates the applied shift on the template instead of moving
    # the catalog position
    assert np.allclose(tmpl.shifted, (0.5, -0.25))
    assert np.allclose(tmpl.to_shift, (0.0, 0.0))


def test_build_poly_predictor_returns_expected_shift():
    order = 1
    betax = np.array([1.0, 0.2, -0.3])
    betay = np.array([-0.5, 0.1, 0.05])
    coeffs = np.concatenate([betax, betay])
    x0, y0 = 10.0, 20.0
    predict = AstroCorrect.build_poly_predictor(coeffs, x0, y0, order)

    x, y = 11.0, 19.0
    dx, dy = predict(x, y)
    phi = cheb_basis(x - x0, y - y0, order)
    assert np.allclose(dx, phi @ betax)
    assert np.allclose(dy, phi @ betay)


def test_cheb_basis_handles_domain_edges():
    order = 2
    phi = cheb_basis(1.0, -1.0, order)
    tx = [chebval(1.0, [0] * i + [1]) for i in range(order + 1)]
    ty = [chebval(-1.0, [0] * j + [1]) for j in range(order + 1)]
    expected = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            expected.append(tx[i] * ty[j])
    assert np.allclose(phi, expected)


# ---------------------------------------------------------------------------
# the one shift-block property that only a hand-scaled alpha0 exposes
# ---------------------------------------------------------------------------


def test_scene_shift_depends_on_alpha0_scale():
    """Shift estimate is proportional to 1/alpha0: biased fluxes bias shifts.

    The shift signal is bB proportional to alpha0 x image_gradient and the
    gradient information matrix BB to alpha0 squared. Their ratio
    shift ~ bB/BB goes as 1/alpha0. Scaling alpha0 by k must scale the
    recovered shift by 1/k -- which is how the old astrom_reg flux bias
    (alpha0 ~ 0.57x) was inflating shifts by ~1.75x.
    """
    from mophongo.scene import Scene

    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=101, peak_snr=30, seed=55
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    true_dx, true_dy = 0.5, 0.4
    science = nd_shift(images[1], (true_dy, true_dx))
    weight = wht[1]

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    cfg = FitConfig(
        fit_astrometry_joint=True,
        astrom_minimum_snr=0.0,
        scene_minimum_anchors=1,
        scene_coupling_thresh=0.005,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    # One scene from all templates, so the has_shift >= 2 requirement is met
    # and the test stays on the alpha0 -> shift dependency rather than on
    # scene partitioning.
    all_templates = list(tmpls.templates)
    A, b, _ = build_normal(all_templates, science, weight)
    scene = Scene(
        id=1,
        templates=all_templates,
        fitter=SceneFitter(),
        image=science,
        weights=weight,
        config=cfg,
    )
    scene.A = A
    scene.b = b
    d = A.diagonal()
    alpha0_true = np.divide(b, d, out=np.zeros_like(b), where=d > 0)
    bright_mask = np.ones(len(scene.templates), dtype=bool)
    basis, (x0, y0), (Sx, Sy) = make_scene_basis(scene.templates, bright_mask, order=0)

    recovered = {}
    for scale in (0.5, 1.0, 2.0):
        AB, BB, bB = assemble_scene_system_AB(
            scene.templates, science, weight, basis,
            alpha0=alpha0_true * scale, order=0, include_y=True,
        )
        sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg)
        if sol.shifts is not None and len(sol.shifts) >= 2:
            # order=0: shifts = [beta_x, beta_y], each length p=1
            recovered[scale] = (float(sol.shifts[0]), float(sol.shifts[1]))

    assert set(recovered) >= {0.5, 1.0, 2.0}, "not all alpha0 scales produced a shift"

    dx_half, _ = recovered[0.5]
    dx_one, _ = recovered[1.0]
    dx_two, _ = recovered[2.0]

    ratio_half = dx_half / dx_one
    ratio_two = dx_two / dx_one
    assert abs(ratio_half - 2.0) < 0.5, f"expected ratio ~2, got {ratio_half:.3f}"
    assert abs(ratio_two - 0.5) < 0.25, f"expected ratio ~0.5, got {ratio_two:.3f}"

    # the unbiased (scale=1.0) estimate matches the injected shift
    assert abs(dx_one - true_dx) < 0.3
    assert abs(recovered[1.0][1] - true_dy) < 0.3
