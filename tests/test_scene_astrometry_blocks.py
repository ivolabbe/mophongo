"""Exactness of the joint flux/shift normal-equation blocks.

``assemble_scene_system_AB`` must build the blocks of the linearized joint
design documented in ``docs/fitting.md``:

    model    m = sum_i alpha_i [T_i - dx_i grad_x(T_i) - dy_i grad_y(T_i)]
    columns  A_j   = T_j
             B_k^x = -sum_i alpha_i S_ik grad_x(T_i)      (likewise B^y)
    blocks   AB = A^T W B ,  BB = B^T W B ,  bB = B^T W d

The reference implementation below forms those columns densely over the whole
image and contracts them directly, sharing no code with the module under
test, so it is an independent oracle rather than a restatement.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.ndimage import shift as nd_shift

from mophongo.astrometry import AstroCorrect, cheb_basis
from mophongo.fit import FitConfig
from mophongo.scene import assemble_scene_system_AB, make_scene_basis
from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.templates import Template

NY = NX = 200
HALF = 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gauss_template(xc: float, yc: float, sigma: float, label: int = 1) -> Template:
    """Unit-sum Gaussian stamp registered at ``(xc, yc)`` on the (NY, NX) grid."""
    n = 2 * HALF + 1
    x0, y0 = int(round(xc)) - HALF, int(round(yc)) - HALF
    yy, xx = np.mgrid[y0 : y0 + n, x0 : x0 + n]
    g = np.exp(-0.5 * (((xx - xc) / sigma) ** 2 + ((yy - yc) / sigma) ** 2))
    g /= g.sum()
    return Template.from_stamp(g, (x0, y0), (xc, yc), (NY, NX), label=label)


def _embed(tmpl: Template) -> np.ndarray:
    """Template pixels placed on the full image grid."""
    out = np.zeros((NY, NX), dtype=float)
    out[tmpl.slices_original] = tmpl.data[tmpl.slices_cutout]
    return out


def _embed_gradients(tmpl: Template) -> tuple[np.ndarray, np.ndarray]:
    """``(grad_x, grad_y)`` of the stamp, placed on the full image grid."""
    gy, gx = np.gradient(tmpl.data.astype(float))
    gx_full = np.zeros((NY, NX), dtype=float)
    gy_full = np.zeros((NY, NX), dtype=float)
    gx_full[tmpl.slices_original] = gx[tmpl.slices_cutout]
    gy_full[tmpl.slices_original] = gy[tmpl.slices_cutout]
    return gx_full, gy_full


def _dense_blocks(templates, image, weights, basis_vals, alpha, order, include_y=True):
    """Exact ``(AB, BB, bB)`` from the dense joint design matrix."""
    p = len(cheb_basis(0.0, 0.0, order))
    nB = p * (2 if include_y else 1)

    acols = np.stack([_embed(t).ravel() for t in templates], axis=1)
    bcols = np.zeros((NY * NX, nB), dtype=float)
    for i, tmpl in enumerate(templates):
        basis = basis_vals[i]
        if basis is None:  # faint members carry no shift coefficient
            continue
        gx, gy = _embed_gradients(tmpl)
        for k in range(p):
            bcols[:, k] += -alpha[i] * basis[k] * gx.ravel()
            if include_y:
                bcols[:, p + k] += -alpha[i] * basis[k] * gy.ravel()

    w = weights.ravel()
    d = image.ravel()
    return (
        acols.T @ (w[:, None] * bcols),
        bcols.T @ (w[:, None] * bcols),
        bcols.T @ (w * d),
    )


def _blend_scene():
    """Twelve sources in six close pairs plus two isolated members."""
    positions = [
        (60, 60), (64, 62), (95, 70), (99, 74), (130, 60), (134, 63),
        (70, 120), (74, 124), (120, 130), (125, 133), (150, 100), (95, 150),
    ]
    sigmas = [2.6, 2.2, 2.8, 2.3, 2.5, 2.1, 2.7, 2.2, 2.4, 2.6, 2.5, 2.3]
    alpha = np.array([1.0, 0.7, 1.2, 0.6, 0.9, 0.8, 1.1, 0.5, 1.0, 0.7, 0.9, 1.0])
    templates = [
        _gauss_template(x, y, s, label=i + 1)
        for i, ((x, y), s) in enumerate(zip(positions, sigmas))
    ]
    return templates, alpha


def _model_image(templates, alpha, dx=0.0, dy=0.0):
    """Noise-free image of the templates, optionally displaced by (dx, dy)."""
    image = np.zeros((NY, NX), dtype=float)
    for a, tmpl in zip(alpha, templates):
        full = a * _embed(tmpl)
        if dx or dy:
            full = nd_shift(full, (dy, dx), order=3)
        image += full
    return image


def _ramp_weights():
    """Smooth non-uniform weights, so no block survives by symmetry alone."""
    y, x = np.mgrid[0:NY, 0:NX]
    return 1.0 + 0.5 * (x / NX) + 0.3 * np.sin(y / 23.0)


# ---------------------------------------------------------------------------
# block exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("include_y", [True, False])
def test_blocks_match_dense_design(order, include_y):
    """AB, BB and bB equal the dense joint design's normal-equation blocks."""
    templates, alpha = _blend_scene()
    weights = _ramp_weights()
    image = _model_image(templates, alpha, dx=0.3, dy=-0.2)

    # a mixed scene: the last three members are faint (no shift coefficient)
    is_bright = np.ones(len(templates), dtype=bool)
    is_bright[-3:] = False
    basis, _ctr, _scl = make_scene_basis(templates, is_bright, order=order)
    basis = list(basis)

    AB, BB, bB = assemble_scene_system_AB(
        templates, image, weights, basis,
        alpha0=alpha, order=order, include_y=include_y,
    )
    AB_ref, BB_ref, bB_ref = _dense_blocks(
        templates, image, weights, basis, alpha, order, include_y=include_y
    )

    scale = np.abs(BB_ref).max()
    assert np.allclose(AB.toarray(), AB_ref, atol=1e-12 * scale, rtol=1e-9)
    assert np.allclose(BB.toarray(), BB_ref, atol=1e-12 * scale, rtol=1e-9)
    assert np.allclose(bB, bB_ref, atol=1e-12 * scale, rtol=1e-9)


def test_bb_carries_cross_anchor_and_xy_blocks():
    """The dropped terms are non-negligible, so the test above has teeth."""
    templates, alpha = _blend_scene()
    weights = np.ones((NY, NX))
    image = _model_image(templates, alpha)
    is_bright = np.ones(len(templates), dtype=bool)
    basis, _c, _s = make_scene_basis(templates, is_bright, order=1)
    basis = list(basis)
    p = len(basis[0])

    _AB, BB, _bB = assemble_scene_system_AB(
        templates, image, weights, basis, alpha0=alpha, order=1
    )
    BB = BB.toarray()

    # x-y coupling block is populated (it is identically zero if only
    # per-anchor x-x and y-y products are accumulated)
    xy = BB[0:p, p : 2 * p]
    assert np.abs(xy).max() > 1e-3 * np.abs(np.diag(BB)).max()

    # cross-anchor terms lower the shift information relative to the
    # per-anchor-only sum, which double counts overlapping gradients
    per_anchor = 0.0
    for i, tmpl in enumerate(templates):
        gx, _gy = _embed_gradients(tmpl)
        per_anchor += alpha[i] ** 2 * basis[i][0] ** 2 * float(np.sum(gx * weights * gx))
    assert BB[0, 0] < 0.95 * per_anchor


def test_faint_templates_couple_to_the_shift_columns():
    """A faint template overlapping a bright anchor has a non-zero AB row."""
    bright_a = _gauss_template(60.0, 60.0, 2.6, label=1)
    bright_b = _gauss_template(66.0, 60.0, 2.4, label=2)
    faint = _gauss_template(63.0, 60.0, 2.0, label=3)  # sits between them
    templates = [bright_a, bright_b, faint]
    alpha = np.array([1.0, 0.8, 0.05])

    weights = _ramp_weights()
    image = _model_image(templates, alpha, dx=0.2)
    basis = [cheb_basis(0.0, 0.0, 0), cheb_basis(0.0, 0.0, 0), None]

    AB, _BB, _bB = assemble_scene_system_AB(
        templates, image, weights, basis, alpha0=alpha, order=0
    )
    AB_ref, _, _ = _dense_blocks(templates, image, weights, basis, alpha, 0)

    faint_row = AB.toarray()[2]
    assert np.abs(faint_row).max() > 0.0
    assert np.allclose(faint_row, AB_ref[2], rtol=1e-9, atol=1e-18)


# ---------------------------------------------------------------------------
# solver behaviour
# ---------------------------------------------------------------------------


def _fit_shifts(templates, alpha, image, weights, order, cfg):
    """Solve the joint system and return the per-source (dx, dy)."""
    is_bright = np.ones(len(templates), dtype=bool)
    basis, ctr, scl = make_scene_basis(templates, is_bright, order=order)
    AB, BB, bB = assemble_scene_system_AB(
        templates, image, weights, list(basis), alpha0=alpha, order=order
    )
    A, b, _ = build_normal(templates, image, weights)
    sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg)
    predict = AstroCorrect.build_poly_predictor(sol.shifts, ctr[0], ctr[1], order, scl[0], scl[1])
    pts = np.array([t.input_position_original for t in templates], dtype=float)
    dx, dy = predict(pts[:, 0], pts[:, 1])
    return dx, dy, sol


@pytest.mark.parametrize("order", [0, 1])
def test_no_spurious_shift_on_aligned_blend(order):
    """Perfectly aligned, perfectly modelled blend must fit zero shift.

    Dropping the cross-template terms leaves a residual dipole signature that
    is indistinguishable from a shift, which at order 1 produced a spurious
    shift field of ~0.05 px rms (0.16 px peak) on exactly this scene.

    Unregularized, the exact blocks put the solution at machine zero. The
    remaining floor under default settings is the adaptive ``reg_flux``
    ridge perturbing the fluxes, not the blocks.
    """
    templates, alpha = _blend_scene()
    weights = np.ones((NY, NX))
    image = _model_image(templates, alpha)  # data == model, truth is zero

    dx, dy, _sol = _fit_shifts(
        templates, alpha, image, weights, order, FitConfig(reg_flux=0.0)
    )
    assert np.abs(dx).max() < 1e-12, f"spurious dx up to {np.abs(dx).max():.4g} px"
    assert np.abs(dy).max() < 1e-12, f"spurious dy up to {np.abs(dy).max():.4g} px"

    dx, dy, _sol = _fit_shifts(templates, alpha, image, weights, order, FitConfig())
    assert np.abs(dx).max() < 1e-4, f"spurious dx up to {np.abs(dx).max():.4g} px"
    assert np.abs(dy).max() < 1e-4, f"spurious dy up to {np.abs(dy).max():.4g} px"


def test_blended_shift_recovery_matches_isolated():
    """Shift recovery must not degrade with blending.

    The per-anchor blocks overstate the shift information (they miss the
    negative cross-anchor gradient products), which shrinks the step: a
    0.30 px offset came back as 0.18 px at 6 px separation.
    """
    cfg = FitConfig()
    truth = 0.30
    recovered = {}
    for separation in (6.0, 8.0, 20.0):
        a = _gauss_template(60.0, 60.0, 2.5, label=1)
        b = _gauss_template(60.0 + separation, 60.0, 2.2, label=2)
        templates = [a, b]
        alpha = np.array([1.0, 0.7])
        weights = np.ones((NY, NX))
        image = _model_image(templates, alpha, dx=truth)
        dx, _dy, _sol = _fit_shifts(templates, alpha, image, weights, 0, cfg)
        recovered[separation] = float(dx[0])

    # 20 px is effectively isolated for 25 px stamps: it sets the achievable
    # accuracy, which is limited only by the first-order linearization
    isolated = recovered[20.0]
    assert abs(isolated - truth) < 0.05 * truth
    for separation, value in recovered.items():
        assert abs(value - truth) < 0.10 * truth, (
            f"separation {separation} px recovered {value:.4f} vs truth {truth}"
        )


def test_flux_errors_inflated_by_shift_covariance():
    """Blended fluxes must inherit uncertainty from the shift block.

    ``SceneFitter`` marginalizes via ``S_w = A_w - AB_w AB_w^T``; with an
    all-but-zero AB the shift covariance never reaches the flux errors.
    """
    templates, alpha = _blend_scene()
    weights = np.full((NY, NX), 1e4)
    image = _model_image(templates, alpha)
    is_bright = np.ones(len(templates), dtype=bool)
    basis, _c, _s = make_scene_basis(templates, is_bright, order=1)

    AB, BB, bB = assemble_scene_system_AB(
        templates, image, weights, list(basis), alpha0=alpha, order=1
    )
    A, b, _ = build_normal(templates, image, weights)
    joint = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=FitConfig())
    flux_only = SceneFitter.solve(A, b, config=FitConfig())

    ratio = joint.err / flux_only.err
    # blended members (the six close pairs) gain, isolated ones do not
    assert ratio[:10].max() > 1.05
    assert np.all(ratio >= 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# leverage cap
# ---------------------------------------------------------------------------


def test_leverage_cap_reduces_to_per_anchor_form_when_isolated():
    """With no overlaps the capped blocks equal the per-anchor contract.

    ``leverage_cap`` is documented as a weight on the shift equations: it
    scales an anchor's information ``I_i`` and its RHS by the same factor so
    the implied shift ``dx_i`` is untouched. That must still hold exactly.
    """
    positions = [(40, 40), (100, 40), (160, 40), (40, 110), (100, 110), (160, 110)]
    templates = [_gauss_template(x, y, 2.5, label=i + 1) for i, (x, y) in enumerate(positions)]
    alpha = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # first anchor dominates
    weights = _ramp_weights()
    image = _model_image(templates, alpha, dx=0.25, dy=-0.1)
    basis = [cheb_basis(0.0, 0.0, 0)] * len(templates)
    cap = 0.5

    AB, BB, bB = assemble_scene_system_AB(
        templates, image, weights, basis, alpha0=alpha, order=0, leverage_cap=cap,
    )

    # explicit per-anchor reference, straight from the documented formulas
    info = np.zeros(len(templates))
    prods = []
    for i, tmpl in enumerate(templates):
        gx, gy = _embed_gradients(tmpl)
        gxx = float(np.sum(gx * weights * gx))
        gyy = float(np.sum(gy * weights * gy))
        gx_ip = float(np.sum(_embed(tmpl) * weights * gx))
        gy_ip = float(np.sum(_embed(tmpl) * weights * gy))
        gx_d = float(np.sum(gx * weights * image))
        gy_d = float(np.sum(gy * weights * image))
        prods.append((gxx, gyy, gx_ip, gy_ip, gx_d, gy_d))
        info[i] = alpha[i] ** 2 * 0.5 * (gxx + gyy)

    lev = np.ones(len(templates))
    thresh = float(np.quantile(info[info > 0], cap))
    over = info > thresh
    lev[over] = thresh / info[over]
    assert over.any(), "cap did not engage; test would be vacuous"

    AB_ref = np.zeros((len(templates), 2))
    BB_ref = np.zeros((2, 2))
    bB_ref = np.zeros(2)
    for i, (gxx, gyy, gx_ip, gy_ip, gx_d, gy_d) in enumerate(prods):
        a_i, wl = alpha[i], lev[i]
        AB_ref[i, 0] = -a_i * wl * gx_ip
        AB_ref[i, 1] = -a_i * wl * gy_ip
        BB_ref[0, 0] += wl * a_i**2 * gxx
        BB_ref[1, 1] += wl * a_i**2 * gyy
        bB_ref[0] += -a_i * wl * gx_d
        bB_ref[1] += -a_i * wl * gy_d

    scale = np.abs(BB_ref).max()
    assert np.allclose(AB.toarray(), AB_ref, atol=1e-11 * scale, rtol=1e-8)
    assert np.allclose(BB.toarray(), BB_ref, atol=1e-11 * scale, rtol=1e-8)
    assert np.allclose(bB, bB_ref, atol=1e-11 * scale, rtol=1e-8)


def test_leverage_cap_preserves_implied_shift():
    """Capping an anchor changes its weight, not the shift it implies."""
    positions = [(40, 40), (100, 40), (160, 40), (40, 110)]
    templates = [_gauss_template(x, y, 2.5, label=i + 1) for i, (x, y) in enumerate(positions)]
    alpha = np.array([6.0, 1.0, 1.0, 1.0])
    weights = np.ones((NY, NX))
    image = _model_image(templates, alpha, dx=0.25)
    basis = [cheb_basis(0.0, 0.0, 0)] * len(templates)

    def implied_dx(cap):
        _AB, BB, bB = assemble_scene_system_AB(
            templates, image, weights, basis, alpha0=alpha, order=0, leverage_cap=cap,
        )
        BB = BB.toarray()
        return float(np.linalg.solve(BB, bB)[0])

    # isolated anchors all see the same true shift, so down-weighting one of
    # them cannot move the solution
    assert implied_dx(0.5) == pytest.approx(implied_dx(None), rel=1e-9)


def test_a_single_anchor_still_solves_the_order_zero_shift():
    """One anchor is enough: it constrains dx and dy, which is the whole basis.

    The old rule needed two and left single-anchor scenes unshifted, which is
    a scene whose one good source carries a measurable offset that nothing
    ever applied.
    """
    truth = (0.35, -0.2)
    tmpl = _gauss_template(100.0, 100.0, 2.5)
    faint = _gauss_template(140.0, 100.0, 2.5, label=2)
    templates = [tmpl, faint]

    # image = the template shifted by the truth, so the offset is recoverable
    image = np.zeros((NY, NX), dtype=float)
    moved = nd_shift(_embed(tmpl), (truth[1], truth[0]), order=3)
    image += moved + 0.15 * _embed(faint)
    weights = np.ones((NY, NX), dtype=float)

    # only the first template passes the anchor cuts
    basis_vals, _centre, _scale = make_scene_basis(
        templates, np.array([True, False]), order=0
    )
    assert sum(b is not None for b in basis_vals) == 1

    A, b, _ = build_normal(templates, image, weights)
    AB, BB, bB = assemble_scene_system_AB(
        templates, image, weights, basis_vals,
        alpha0=np.array([1.0, 0.15]), order=0,
    )
    assert AB.shape[1] == 2, "one anchor must still open the two shift columns"

    flux, _err, shifts, _info = SceneFitter._solve_flux_and_shifts(
        A.tocsr(), b, AB, BB, bB, FitConfig(),
    )
    assert shifts is not None and len(shifts) == 2
    # sign convention aside, the magnitude must come back to within a fraction
    # of a pixel rather than being zeroed
    assert np.hypot(*shifts) == pytest.approx(np.hypot(*truth), rel=0.35)
