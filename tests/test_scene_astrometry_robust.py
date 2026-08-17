"""Per-anchor shift measurement and robust weighting, at scene level.

Two things are checked here that the pure-numpy tests in
``test_astrom_robust.py`` cannot see:

* :func:`~mophongo.scene.measure_anchor_shifts` measures what it claims to --
  the implied displacement, its information, and the part of the residual that
  a displacement cannot explain;
* a scene whose brightest anchor lies gets the right shift with
  ``FitConfig.astrom_robust`` on, and the wrong one with it off.

Scenes are noiseless and built from the exact linearized model, so any error is
the estimator's own.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from mophongo.fit import FitConfig
from mophongo.scene import (
    Scene,
    _bbox_union,
    _scene_residual,
    assemble_scene_system_AB,
    make_scene_basis,
    measure_anchor_shifts,
)
from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.templates import Template

NY = NX = 320
HALF = 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stamp(xc: float, yc: float, sigma: float, skew: float = 0.0) -> np.ndarray:
    n = 2 * HALF + 1
    x0, y0 = int(round(xc)) - HALF, int(round(yc)) - HALF
    yy, xx = np.mgrid[y0 : y0 + n, x0 : x0 + n]
    r2 = ((xx - xc) / sigma) ** 2 + ((yy - yc) / sigma) ** 2
    g = np.clip(np.exp(-0.5 * r2) * (1.0 + skew * (xx - xc) / sigma), 0.0, None)
    return g / g.sum()


def _tmpl(xc: float, yc: float, sigma: float, label: int, skew: float = 0.0) -> Template:
    x0, y0 = int(round(xc)) - HALF, int(round(yc)) - HALF
    return Template.from_stamp(
        _stamp(xc, yc, sigma, skew), (x0, y0), (xc, yc), (NY, NX), label=label
    )


def _shifted(stamp: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """First-order displacement of ``stamp`` -- the exact model of the solve."""
    gy, gx = np.gradient(stamp.astype(float))
    return stamp - shift[0] * gx - shift[1] * gy


def _grid(n: int = 12, sigma: float = 2.5) -> list[Template]:
    """``n`` well-separated Gaussians on a coarse grid."""
    out = []
    k = 0
    for row in range(3):
        for col in range(4):
            if k >= n:
                break
            out.append(
                _tmpl(45.0 + 75.0 * col, 45.0 + 100.0 * row, sigma, label=k + 1)
            )
            k += 1
    return out


def _paint(
    templates: list[Template],
    fluxes: np.ndarray,
    shifts: list[tuple[float, float]],
    stamps: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Image from per-source flux, shift and (optionally) a different profile.

    ``stamps`` lets a source be *painted* with a profile the template does not
    have, which is how a colour gradient shows up: the residual then holds
    structure no displacement can absorb.
    """
    img = np.zeros((NY, NX), dtype=float)
    for i, (t, f, sh) in enumerate(zip(templates, fluxes, shifts)):
        src = t.data if stamps is None else stamps[i]
        img[t.slices_original] += f * _shifted(src, sh)[t.slices_cutout]
    return img


def _cfg(**kw) -> FitConfig:
    base = dict(
        reg_flux=0.0,
        astrom_reg=0.0,
        fit_method="lls",
        fit_astrometry_joint=True,
        fit_astrometry_niter=1,
        astrom_minimum_snr=0.0,
        astrom_isolation_thresh=0.0,
        astrom_leverage_cap=None,
        astrom_damping=1.0,
    )
    base.update(kw)
    return FitConfig(**base)


def _solve_scene(templates, image, weights, cfg):
    scn = Scene(
        id=1,
        templates=templates,
        fitter=SceneFitter(),
        bbox=_bbox_union(templates),
    )
    scn.set_band(image, weights, config=cfg)
    scn.solve(config=cfg, apply_shifts=False)
    return scn


# ---------------------------------------------------------------------------
# measure_anchor_shifts
# ---------------------------------------------------------------------------


def _measure_anchor_shifts_padded(templates, resid, weights, origin, bright_idx, alpha):
    """Reference: every column laid out on the neighbourhood's union footprint.

    The obvious way to write the local system, and the way it was written
    before the columns were reduced to slice intersections. Kept here so the
    shipped version has something to be equal to -- the slice arithmetic is
    where a rewrite like that goes wrong, and it goes wrong quietly.
    """
    n = len(templates)
    y0, x0 = origin
    eps = np.full((n, 2), np.nan)
    info = np.zeros(n)
    chi2_red = np.full(n, np.nan)
    slices = [t.slices_original for t in templates]
    anchors = {int(k) for k in bright_idx}

    def _grad(j):
        arr = np.asarray(templates[j].data, dtype=float)
        if min(arr.shape) < 2:
            return None
        gy, gx = np.gradient(arr)
        sc = templates[j].slices_cutout
        return -gx[sc], -gy[sc]

    for i in map(int, bright_idx):
        a = float(alpha[i])
        if not np.isfinite(a) or a == 0.0 or _grad(i) is None:
            continue
        si = slices[i]
        nb = [j for j, sj in enumerate(slices)
              if sj[0].start < si[0].stop and si[0].start < sj[0].stop
              and sj[1].start < si[1].stop and si[1].start < sj[1].stop]
        fy0 = min(slices[j][0].start for j in nb)
        fy1 = max(slices[j][0].stop for j in nb)
        fx0 = min(slices[j][1].start for j in nb)
        fx1 = max(slices[j][1].stop for j in nb)
        w = np.asarray(weights[fy0:fy1, fx0:fx1], dtype=float)
        r = resid[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]

        def _place(j, values):
            out = np.zeros((fy1 - fy0, fx1 - fx0))
            sj = slices[j]
            out[sj[0].start - fy0 : sj[0].stop - fy0,
                sj[1].start - fx0 : sj[1].stop - fx0] = values
            return out

        cols = [_place(j, templates[j].data[templates[j].slices_cutout]) for j in nb]
        for k in nb:
            if k in anchors and k != i and _grad(k) is not None:
                g = _grad(k)
                cols += [_place(k, g[0]), _place(k, g[1])]
        g_i = _grad(i)
        cols += [_place(i, g_i[0]), _place(i, g_i[1])]

        ncol = len(cols)
        nrest = ncol - 2
        M = np.empty((ncol, ncol))
        s = np.empty(ncol)
        for p_ in range(ncol):
            cw = cols[p_] * w
            s[p_] = float(np.sum(cw * r))
            for q_ in range(p_, ncol):
                M[p_, q_] = M[q_, p_] = float(np.sum(cw * cols[q_]))

        try:
            theta = np.linalg.solve(M, s)
            rest_inv = np.linalg.solve(M[:nrest, :nrest], M[:nrest, nrest:])
        except np.linalg.LinAlgError:
            continue
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(rest_inv))):
            continue
        schur = M[nrest:, nrest:] - M[nrest:, :nrest] @ rest_inv
        iso = 0.5 * (schur[0, 0] + schur[1, 1])
        if not (iso > 0):
            continue
        eps[i] = theta[nrest:] / a
        info[i] = a**2 * iso

        model = sum(th * c for c, th in zip(cols, theta))
        own = (slice(si[0].start - fy0, si[0].stop - fy0),
               slice(si[1].start - fx0, si[1].stop - fx0))
        w_own = w[own]
        dof = int(np.count_nonzero(w_own > 0)) - ncol
        if dof > 0:
            left = r[own] - model[own]
            chi2_red[i] = float(np.sum(left * w_own * left)) / dof
    return eps, info, chi2_red


def _ragged_scene(seed: int = 3):
    """Overlapping templates of different sizes, some clipped by the frame.

    Equal-sized stamps on a regular grid would let a wrong offset cancel. The
    sizes differ, the centres are off-pixel, two sit close enough to overlap
    heavily, and one runs off the edge so its ``slices_cutout`` is a strict
    sub-slice of its own data.
    """
    rng = np.random.default_rng(seed)
    spec = [
        (40.3, 60.7, 9, 3.0), (46.9, 63.1, 14, 4.0), (58.2, 57.4, 7, 2.0),
        (95.6, 61.2, 18, 5.0), (101.1, 66.8, 11, 3.5), (8.4, 62.5, 20, 4.5),
        (150.7, 60.1, 12, 3.0), (156.3, 55.9, 16, 4.0),
    ]
    templates = []
    for label, (xc, yc, half, sigma) in enumerate(spec, start=1):
        nn = 2 * half + 1
        ix, iy = int(round(xc)) - half, int(round(yc)) - half
        yy, xx = np.mgrid[iy : iy + nn, ix : ix + nn]
        g = np.exp(-0.5 * (((xx - xc) / sigma) ** 2 + ((yy - yc) / sigma) ** 2))
        templates.append(
            Template.from_stamp(g / g.sum(), (ix, iy), (xc, yc), (NY, NX), label=label)
        )
    flux = np.array([120.0, 60.0, 25.0, 200.0, 45.0, 90.0, 150.0, 70.0])
    img = _paint(templates, flux, [(0.18, -0.09)] * len(templates))
    img += rng.normal(0.0, 1e-4, img.shape)
    weights = rng.uniform(0.4, 1.6, img.shape)
    weights[:, :6] = 0.0  # a strip with no exposure, as a real weight map has
    return templates, img, weights, flux


def test_local_systems_match_the_padded_reference():
    """The intersection assembly is bookkeeping, not a new estimator.

    Columns are zero off their own template's slice, so restricting every
    inner product to a slice intersection can only drop zeros. Checked against
    the padded reference on overlapping templates of different sizes, one of
    them clipped by the frame, over a weight map with a dead strip.
    """
    templates, img, weights, flux = _ragged_scene()
    A, b, _ = build_normal(templates, img, weights)
    resid, y0, x0 = _scene_residual(templates, img, flux)
    alpha = np.asarray(b / np.maximum(A.diagonal(), 1e-12))
    bright = [0, 1, 3, 5, 6]

    want = _measure_anchor_shifts_padded(
        templates, resid, weights, (y0, x0), bright, alpha
    )
    got = measure_anchor_shifts(templates, resid, weights, (y0, x0), bright, alpha)

    for name, w_, g_ in zip(("eps", "info", "chi2_red"), want, got):
        assert np.array_equal(np.isnan(w_), np.isnan(g_)), name
        np.testing.assert_allclose(g_, w_, rtol=1e-10, atol=1e-14, err_msg=name)
    assert np.isfinite(want[0][bright]).all(), "reference measured nothing"


def test_anchor_shift_is_recovered_exactly_for_isolated_anchors():
    """Noiseless, isolated, exact model: eps is the injected shift."""
    shift = (0.17, -0.09)
    tm = _grid(6)
    f = np.array([100.0, 60.0, 40.0, 30.0, 20.0, 15.0])
    img = _paint(tm, f, [shift] * len(tm))
    W = np.ones((NY, NX))

    resid, y0, x0 = _scene_residual(tm, img, np.zeros(len(tm)))
    assert np.allclose(resid, img[y0 : y0 + resid.shape[0], x0 : x0 + resid.shape[1]])

    A, b, _ = build_normal(tm, img, W)
    flux0 = SceneFitter.solve(A, b, config=_cfg()).flux
    resid, y0, x0 = _scene_residual(tm, img, flux0)
    eps, info, chi2 = measure_anchor_shifts(
        tm, resid, W, (y0, x0), range(len(tm)), flux0
    )

    assert np.allclose(eps, np.array(shift), atol=1e-6)
    assert np.all(info > 0)
    # a pure displacement lives entirely in the gradient span, so nothing is
    # left once it is projected out
    assert np.all(chi2 < 1e-12)


def test_information_scales_as_flux_squared():
    """The leverage problem in one assertion."""
    tm = _grid(2)
    f = np.array([100.0, 10.0])
    img = _paint(tm, f, [(0.1, 0.1)] * 2)
    W = np.ones((NY, NX))

    A, b, _ = build_normal(tm, img, W)
    flux0 = SceneFitter.solve(A, b, config=_cfg()).flux
    resid, y0, x0 = _scene_residual(tm, img, flux0)
    _, info, _ = measure_anchor_shifts(tm, resid, W, (y0, x0), range(2), flux0)

    assert info[0] / info[1] == pytest.approx(100.0, rel=0.05)


def test_chi2_separates_a_wrong_profile_from_a_real_displacement():
    """The discriminator: what is left after the anchor's own shift is removed.

    Source 0 has genuinely moved; source 1 has not moved at all but is painted
    with a profile its template does not have. Both leave a large residual;
    only the second leaves one a displacement cannot explain.
    """
    tm = _grid(2)
    f = np.array([100.0, 100.0])
    stamps = [tm[0].data, _stamp(120.0, 45.0, 2.5, skew=0.8)]
    img = _paint(tm, f, [(0.25, -0.15), (0.0, 0.0)], stamps=stamps)
    W = np.ones((NY, NX))

    A, b, _ = build_normal(tm, img, W)
    flux0 = SceneFitter.solve(A, b, config=_cfg()).flux
    resid, y0, x0 = _scene_residual(tm, img, flux0)
    eps, _, chi2 = measure_anchor_shifts(tm, resid, W, (y0, x0), range(2), flux0)

    assert np.allclose(eps[0], (0.25, -0.15), atol=1e-6)
    assert chi2[0] < 1e-12
    assert chi2[1] > 1e3 * max(chi2[0], 1e-15)


def _blend(neighbour_shift):
    """Bright anchor with a faint blended neighbour, plus an isolated twin.

    Index 0 is the blended anchor, 1 its neighbour (which fails the isolation
    cut), 2 an identical but isolated copy of 0 -- the control.
    """
    shift = (0.20, -0.12)
    tm = [
        _tmpl(60.0, 100.0, 2.5, 1),
        _tmpl(64.0, 100.0, 2.0, 2),
        _tmpl(200.0, 100.0, 2.5, 3),
        _tmpl(200.0, 220.0, 2.5, 4),
    ]
    f = np.array([100.0, 12.0, 100.0, 60.0])
    img = _paint(tm, f, [shift, neighbour_shift, shift, shift])
    W = np.ones((NY, NX))

    A, b, _ = build_normal(tm, img, W)
    d = np.asarray(A.diagonal(), dtype=float)
    alpha0 = np.divide(b, d, out=np.zeros_like(b), where=d > 0)
    flux0 = SceneFitter.solve(A, b, config=_cfg()).flux
    resid, y0, x0 = _scene_residual(tm, img, flux0)
    eps, info, chi2 = measure_anchor_shifts(
        tm, resid, W, (y0, x0), [0, 2, 3], alpha0
    )
    return shift, eps, info, chi2


def test_a_blended_anchor_is_measured_conditionally_on_its_neighbour():
    """A neighbour's free flux must not be allowed to eat the anchor's dipole.

    The anchor's shift is measured against a residual whose fluxes are already
    fitted, and a neighbour sitting to one side can absorb part of a dipole by
    adjusting its own brightness. Measured marginally -- projecting out only
    the anchor's own flux -- this anchor read 0.095 px against a true 0.20,
    while its isolated twin read 0.20 exactly. Every flux overlapping the
    anchor has to be in the local system, or blended anchors report
    systematically small shifts and the robust pass rejects them for
    disagreeing.
    """
    truth, eps, info, chi2 = _blend(neighbour_shift=(0.0, 0.0))

    assert np.abs(eps[2] - np.array(truth)).max() < 1e-9  # isolated twin
    assert np.abs(eps[0] - np.array(truth)).max() < 0.02  # blended anchor
    # the blend does cost information, and that is reported rather than hidden
    assert 0.5 < info[0] / info[2] < 1.0
    # ...but it is not a misfit: nothing is unexplained
    assert chi2[0] < 1e-20


def test_an_unmodelled_neighbour_shift_shows_up_as_misfit():
    """A non-anchor that has also moved leaves structure behind.

    Its displacement is not in the model -- only anchors get derivative
    columns -- so its dipole cannot be absorbed by any flux and lands in the
    anchor's misfit instead. That is the honest place for it: the anchor's
    stamp genuinely no longer explains its own footprint.
    """
    truth, eps, info, chi2 = _blend(neighbour_shift=(0.20, -0.12))

    assert chi2[0] > 1e6 * chi2[2]
    # the shift is still recovered far better than the marginal estimate gave
    assert np.abs(eps[0] - np.array(truth)).max() < 0.05


def _blended_pair(sep, f_neighbour, anchors):
    """A blended pair at (60, 100) plus six isolated anchors.

    ``anchors`` selects which templates get a free displacement, so the same
    scene can be measured with the neighbour treated as an anchor or not.
    """
    shift = (0.20, -0.12)
    tm = [_tmpl(60.0, 100.0, 2.5, 1), _tmpl(60.0 + sep, 100.0, 2.5, 2)]
    tm += [_tmpl(150.0 + 70 * (k % 3), 100.0 + 90 * (k // 3), 2.5, 10 + k)
           for k in range(6)]
    f = np.array([100.0, f_neighbour] + [60.0] * 6)
    img = _paint(tm, f, [shift] * 8)
    W = np.ones((NY, NX))

    A, b, _ = build_normal(tm, img, W)
    d = np.asarray(A.diagonal(), dtype=float)
    alpha0 = np.divide(b, d, out=np.zeros_like(b), where=d > 0)
    flux0 = SceneFitter.solve(A, b, config=_cfg()).flux
    resid, y0, x0 = _scene_residual(tm, img, flux0)
    eps, info, chi2 = measure_anchor_shifts(
        tm, resid, W, (y0, x0), anchors, alpha0
    )
    return shift, eps, info, chi2


@pytest.mark.parametrize("sep,f_n", [(6.0, 80.0), (6.0, 12.0), (3.0, 80.0)])
def test_two_blended_anchors_are_measured_against_each_other(sep, f_n):
    """When the neighbour is an anchor too, its displacement must be free.

    Two overlapping sources that have both moved leave overlapping dipoles.
    Fitting only one anchor's gradient columns splits that badly: with the
    neighbour's flux free but its *shift* held at zero, a pair at 6 px read
    0.089 and 0.039 px against a true 0.20 -- both low, both agreeing with
    each other, and both carrying more information than the honest anchors
    they disagreed with. Giving every overlapping anchor its own free
    displacement is what makes each estimate conditional rather than a
    one-at-a-time guess.
    """
    truth, eps, info, chi2 = _blended_pair(sep, f_n, anchors=list(range(8)))

    assert np.abs(eps[2] - np.array(truth)).max() < 1e-9  # isolated control
    for i in (0, 1):
        bias = np.abs(eps[i] - np.array(truth)).max()
        sigma = 1.0 / np.sqrt(info[i])
        # The invariant that matters: whatever bias survives must sit inside
        # the anchor's own error bar, or the robust pass would read a blend as
        # a liar and reject a perfectly good anchor.
        assert bias < sigma, f"anchor {i}: bias {bias:.3f} >= sigma {sigma:.3f}"
        assert bias < 0.15


def test_a_degenerate_blend_reports_its_own_uncertainty():
    """Heavier blending buys a larger error bar, not a more confident wrong answer."""
    _, _, info_wide, _ = _blended_pair(8.0, 80.0, anchors=list(range(8)))
    _, _, info_tight, _ = _blended_pair(2.0, 80.0, anchors=list(range(8)))

    assert info_tight[0] < info_wide[0]
    assert info_wide[0] > info_wide[2]  # still brighter than the isolated ones


def test_non_anchors_come_back_blank():
    tm = _grid(3)
    img = _paint(tm, np.array([50.0, 50.0, 50.0]), [(0.1, 0.0)] * 3)
    W = np.ones((NY, NX))
    resid, y0, x0 = _scene_residual(tm, img, np.zeros(3))

    eps, info, chi2 = measure_anchor_shifts(tm, resid, W, (y0, x0), [0, 2], np.ones(3))

    assert np.all(np.isnan(eps[1]))
    assert info[1] == 0.0
    assert np.isnan(chi2[1])
    assert np.all(np.isfinite(eps[[0, 2]]))


# ---------------------------------------------------------------------------
# anchor_weights plumbing
# ---------------------------------------------------------------------------


def test_zero_weight_anchor_is_identical_to_never_being_bright():
    """Hard rejection is exact -- that is why the module rejects rather than
    tapers. Compare zeroing an anchor's weight against dropping it from the
    bright mask, which is the same thing the blocks should see."""
    tm = _grid(4)
    f = np.array([100.0, 50.0, 40.0, 30.0])
    img = _paint(tm, f, [(0.2, -0.1)] * 4)
    W = np.ones((NY, NX))

    bright_all = np.array([True, True, True, True])
    bright_drop = np.array([False, True, True, True])
    basis_all, _, _ = make_scene_basis(tm, bright_all, order=0)
    basis_drop, _, _ = make_scene_basis(tm, bright_drop, order=0)

    kw = dict(alpha0=f, order=0, include_y=True, leverage_cap=None)
    AB_w, BB_w, bB_w = assemble_scene_system_AB(
        tm, img, W, basis_all, anchor_weights=np.array([0.0, 1.0, 1.0, 1.0]), **kw
    )
    AB_d, BB_d, bB_d = assemble_scene_system_AB(tm, img, W, basis_drop, **kw)

    assert np.allclose(AB_w.toarray(), AB_d.toarray())
    assert np.allclose(BB_w.toarray(), BB_d.toarray())
    assert np.allclose(bB_w, bB_d)


def test_unit_anchor_weights_change_nothing():
    tm = _grid(4)
    f = np.array([100.0, 50.0, 40.0, 30.0])
    img = _paint(tm, f, [(0.2, -0.1)] * 4)
    W = np.ones((NY, NX))
    basis, _, _ = make_scene_basis(tm, np.ones(4, bool), order=0)

    kw = dict(alpha0=f, order=0, include_y=True, leverage_cap=None)
    a0 = assemble_scene_system_AB(tm, img, W, basis, **kw)
    a1 = assemble_scene_system_AB(tm, img, W, basis, anchor_weights=np.ones(4), **kw)

    assert np.allclose(a0[0].toarray(), a1[0].toarray())
    assert np.allclose(a0[1].toarray(), a1[1].toarray())
    assert np.allclose(a0[2], a1[2])


def test_anchor_weights_shape_is_checked():
    tm = _grid(3)
    img = _paint(tm, np.ones(3) * 50, [(0.1, 0.0)] * 3)
    W = np.ones((NY, NX))
    basis, _, _ = make_scene_basis(tm, np.ones(3, bool), order=0)
    with pytest.raises(ValueError, match="anchor_weights"):
        assemble_scene_system_AB(
            tm, img, W, basis, alpha0=np.ones(3), order=0,
            anchor_weights=np.ones(2),
        )


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_a_dominant_liar_pulls_the_scene_and_robust_weighting_stops_it():
    """One bright anchor displaced differently from the other eleven.

    This is the colour-gradient case in its purest form: the offending source's
    residual *is* a clean dipole, so nothing about its own stamp gives it away
    -- only the eleven anchors that disagree with it do.
    """
    truth = (0.20, -0.12)
    tm = _grid(12)
    f = np.full(12, 20.0)
    f[0] = 400.0  # 20x brighter -> 400x the leverage
    shifts = [truth] * 12
    shifts[0] = (0.90, 0.70)  # its dipole is morphology, not motion
    img = _paint(tm, f, shifts)
    W = np.ones((NY, NX))

    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True))

    err_naive = np.abs(naive.shifts - np.array(truth)).max()
    err_robust = np.abs(robust.shifts - np.array(truth)).max()

    assert err_naive > 0.3, "the liar should dominate without robust weighting"
    assert err_robust < 0.02
    assert robust.anchor_report.applied
    assert robust.anchor_report.n_rejected >= 1
    assert tm[0].astrom_weight == 0.0
    assert min(t.astrom_weight for t in tm[1:]) > 0.3


def test_a_wrong_profile_anchor_is_caught_too():
    """Same setup, but the offender's residual is not a clean dipole.

    Here the per-anchor chi-square sees it directly, without needing the
    neighbours -- the two layers cover different failures.
    """
    truth = (0.18, -0.10)
    tm = _grid(12)
    f = np.full(12, 20.0)
    f[0] = 400.0
    stamps = [t.data for t in tm]
    stamps[0] = _stamp(45.0, 45.0, 2.5, skew=1.5)
    img = _paint(tm, f, [truth] * 12, stamps=stamps)
    W = np.ones((NY, NX))

    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True))

    err_naive = np.abs(naive.shifts - np.array(truth)).max()
    err_robust = np.abs(robust.shifts - np.array(truth)).max()

    assert err_naive > 5.0 * err_robust
    assert err_robust < 0.02
    assert tm[0].astrom_weight < 0.5


def test_robust_weighting_leaves_an_honest_scene_where_it_was():
    """No liar: turning it on must not move the answer appreciably."""
    truth = (0.22, -0.14)
    tm = _grid(12)
    f = np.linspace(20.0, 200.0, 12)
    img = _paint(tm, f, [truth] * 12)
    W = np.ones((NY, NX))

    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True))

    assert np.abs(naive.shifts - np.array(truth)).max() < 0.01
    assert np.abs(robust.shifts - np.array(truth)).max() < 0.01
    assert robust.anchor_report.n_rejected == 0


@pytest.mark.parametrize("liar", [True, False])
def test_behaviour_survives_realistic_noise(liar):
    """With noise the honest anchors no longer agree exactly.

    Both directions matter: the liar must still be caught when the majority
    only agrees to within its errors, and the majority must not start
    rejecting each other when there is no liar to find.
    """
    rng = np.random.default_rng(101)
    truth = (0.20, -0.12)
    tm = _grid(12)
    f = np.full(12, 300.0)
    if liar:
        f[0] = 3000.0
    shifts = [truth] * 12
    if liar:
        shifts[0] = (0.85, 0.65)
    sigma = 0.02
    img = _paint(tm, f, shifts) + rng.normal(0.0, sigma, (NY, NX))
    W = np.full((NY, NX), 1.0 / sigma**2)

    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True))
    err_naive = np.abs(naive.shifts - np.array(truth)).max()
    err_robust = np.abs(robust.shifts - np.array(truth)).max()

    assert robust.anchor_report.applied
    if liar:
        assert err_naive > 0.2
        assert err_robust < 0.03
        assert tm[0].astrom_weight == 0.0
        # the honest eleven survive
        assert sum(t.astrom_weight > 0 for t in tm[1:]) >= 10
    else:
        assert err_naive < 0.03
        assert err_robust < 0.03
        assert robust.anchor_report.n_rejected == 0


def test_small_scene_is_left_alone_and_reports_why():
    """Below the anchor gate the pass declines rather than guessing."""
    truth = (0.15, -0.05)
    tm = _grid(2)
    f = np.array([100.0, 50.0])
    img = _paint(tm, f, [truth] * 2)
    W = np.ones((NY, NX))

    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True))

    assert np.allclose(naive.shifts, robust.shifts)
    assert not robust.anchor_report.applied
    assert "required" in robust.anchor_report.reason
    assert all(t.astrom_weight == 1.0 for t in tm)


def test_the_gate_is_scene_minimum_anchors():
    """The robust gate and the scene-merging floor are one number.

    A scene merged up to ``scene_minimum_anchors`` anchors is by construction
    large enough for the robust pass to judge, so the two must not drift
    apart.
    """
    truth = (0.15, -0.05)
    tm = _grid(6)
    f = np.linspace(100.0, 25.0, 6)
    img = _paint(tm, f, [truth] * 6)
    W = np.ones((NY, NX))

    under = _solve_scene(tm, img, W, _cfg(astrom_robust=True, scene_minimum_anchors=9))
    over = _solve_scene(tm, img, W, _cfg(astrom_robust=True, scene_minimum_anchors=4))

    assert not under.anchor_report.applied
    assert "9 required" in under.anchor_report.reason
    assert over.anchor_report.applied


def test_scene_minimum_anchors_is_a_flat_floor_the_basis_can_raise():
    """One number, whatever the astrometric model.

    The floor used to be derived from the polynomial order. It is a
    statistical floor rather than an algebraic one, so it does not follow the
    order; where a wider basis does need more anchors, ``anchor_gate`` is what
    raises it.
    """
    from mophongo.astrom_robust import anchor_gate

    assert FitConfig().scene_minimum_anchors == 10
    assert (
        FitConfig(
            astrom_kwargs={"poly": {"order": 1}, "gp": {"length_scale": 400}}
        ).scene_minimum_anchors
        == 10
    )
    assert FitConfig(scene_minimum_anchors=11).scene_minimum_anchors == 11

    assert anchor_gate(10, 1) == 10  # order 0: the floor binds
    assert anchor_gate(10, 6) == 12  # order 2: the basis binds


def test_a_scene_below_the_gate_is_never_measured(monkeypatch):
    """The gate is checked before the measurement, not after.

    ``measure_anchor_shifts`` fits a local least-squares system per anchor over
    its whole neighbourhood, which is the expensive part of the pass. A scene
    short of the anchor gate can only be declined, so the work must not happen
    at all.
    """
    import mophongo.scene as scene_mod

    calls = []
    real = scene_mod.measure_anchor_shifts

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scene_mod, "measure_anchor_shifts", counting)

    tm = _grid(6)
    f = np.linspace(100.0, 25.0, 6)
    img = _paint(tm, f, [(0.15, -0.05)] * 6)
    W = np.ones((NY, NX))

    under = _solve_scene(tm, img, W, _cfg(astrom_robust=True, scene_minimum_anchors=9))
    assert not under.anchor_report.applied
    assert calls == []

    over = _solve_scene(tm, img, W, _cfg(astrom_robust=True, scene_minimum_anchors=4))
    assert over.anchor_report.applied
    assert len(calls) == 1


def test_robust_weighting_supersedes_the_leverage_cap():
    """Where the measured weight applies, the blind quantile stands down.

    Both bound one anchor's pull. Leaving the cap on as well would keep
    clipping the brightest anchors -- usually the best ones -- on top of a
    ceiling already set from the data.
    """
    truth = (0.20, -0.12)
    tm = _grid(12)
    f = np.linspace(20.0, 400.0, 12)
    img = _paint(tm, f, [truth] * 12)
    W = np.ones((NY, NX))

    with_cap = _solve_scene(tm, img, W, _cfg(astrom_robust=True, astrom_leverage_cap=0.5))
    no_cap = _solve_scene(tm, img, W, _cfg(astrom_robust=True, astrom_leverage_cap=None))

    assert with_cap.anchor_report.applied
    assert np.allclose(with_cap.shifts, no_cap.shifts)

    # ...but a scene the robust pass declines still gets the cap. The two
    # anchors must disagree for any weighting to show, so give them
    # different shifts.
    small = _grid(2)
    fs = np.array([400.0, 20.0])
    img_s = _paint(small, fs, [truth, (0.6, 0.4)])
    capped = _solve_scene(small, img_s, W, _cfg(astrom_robust=True, astrom_leverage_cap=0.5))
    uncapped = _solve_scene(small, img_s, W, _cfg(astrom_robust=True, astrom_leverage_cap=None))
    assert not capped.anchor_report.applied
    assert not np.allclose(capped.shifts, uncapped.shifts)


def _mixed_field(n_iso, n_pair, seed, sep=4.0, liar=False, sigma=0.02):
    """``n_pair`` blended pairs at random orientation plus ``n_iso`` singletons."""
    rng = np.random.default_rng(seed)
    slots = [(50.0 + 80.0 * (i % 3), 50.0 + 85.0 * (i // 3)) for i in range(9)]
    rng.shuffle(slots)
    tm, f, sh, k = [], [], [], 0
    truth = (0.20, -0.12)
    for i in range(n_pair):
        x, y = slots[i]
        th = rng.uniform(0, 2 * np.pi)
        tm += [
            _tmpl(x, y, 2.5, k + 1),
            _tmpl(x + sep * np.cos(th), y + sep * np.sin(th), 2.5, k + 2),
        ]
        k += 2
        f += [300.0, 240.0]
        sh += [truth, truth]
    for i in range(n_iso):
        x, y = slots[n_pair + i]
        tm.append(_tmpl(x, y, 2.5, k + 1))
        k += 1
        f.append(300.0)
        sh.append(truth)
    f = np.array(f, dtype=float)
    if liar:
        f[0] = 3000.0
        sh[0] = (0.85, 0.65)
    img = _paint(tm, f, sh) + rng.normal(0.0, sigma, (NY, NX))
    return truth, tm, img, np.full((NY, NX), 1.0 / sigma**2)


def _beta_err(tm, img, W, iso, robust):
    truth = (0.20, -0.12)
    scn = _solve_scene(
        tm, img, W,
        # a nine-template field cannot clear the default anchor floor, and
        # these two tests are about the weighting rather than the gate
        _cfg(astrom_isolation_thresh=iso, astrom_minimum_snr=0.0,
             astrom_robust=robust, scene_minimum_anchors=3),
    )
    return np.abs(scn.shifts - np.array(truth)).max(), scn


def test_robust_weighting_does_not_replace_the_isolation_cut():
    """The two guard against different failures and neither subsumes the other.

    A blended anchor's residual shift is *shrunk* toward zero by what remains
    approximate in the local system -- and shrinkage is coherent, so every
    blended anchor in the field is biased the same way whatever its pair's
    orientation. They therefore agree with each other. Robust weighting is
    majority rule: when the blended anchors outnumber the clean ones it
    follows them, and the systematic floor it fits to their scatter shifts
    weight further toward them. Only the isolation cut can remove that
    population, because there is no disagreement for the robust pass to see.
    """
    truth, tm, img, W = _mixed_field(n_iso=3, n_pair=3, seed=11)

    admitted, scn_a = _beta_err(tm, img, W, iso=0.0, robust=True)
    admitted_off, _ = _beta_err(tm, img, W, iso=0.0, robust=False)
    excluded, scn_e = _beta_err(tm, img, W, iso=0.7, robust=True)

    # the cut is what buys the accuracy, by a wide margin
    assert excluded < 0.02
    assert admitted > 10 * excluded
    # and with the blends admitted the robust pass does not rescue it -- it
    # makes it worse, because the biased anchors are the majority
    assert admitted > admitted_off
    assert scn_a.anchor_report.applied
    assert scn_a.anchor_report.n_rejected == 0


def test_robust_weighting_is_what_catches_a_minority_outlier():
    """The complementary half: one liar, which no isolation cut can see."""
    truth, tm, img, W = _mixed_field(n_iso=3, n_pair=3, seed=11, liar=True)

    off, _ = _beta_err(tm, img, W, iso=0.5, robust=False)
    on, scn = _beta_err(tm, img, W, iso=0.5, robust=True)

    assert off > 0.4
    assert on < 0.2
    assert on < 0.25 * off
    assert scn.anchor_report.n_rejected >= 1


def _ab_pipeline(robust: bool):
    """Run the same mock field twice, with the flag on and off."""
    from astropy.wcs import WCS
    from scipy.ndimage import shift as nd_shift

    from mophongo import pipeline
    from mophongo import utils as mutils
    from utils import make_simple_data

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=40, size=201, ndilate=2, peak_snr=50.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    images = [images[0], nd_shift(images[1], (0.95, -0.75), order=3)]

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100.0, 100.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-1e-5, 1e-5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    cfg = FitConfig(
        fit_astrometry_niter=5,
        astrom_shift_tol=0.004,
        astrom_minimum_snr=5.0,
        astrom_isolation_thresh=0.0,
        scene_minimum_anchors=2,
        astrom_robust=robust,
    )
    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
        wcs=[wcs, wcs], config=cfg,
    )
    table, _ = pipe.run()
    return pipe, table


def test_the_flag_reaches_the_solve_and_off_is_the_old_behaviour():
    """A/B handle: one flag, and `False` must reproduce the previous run.

    The point of keeping this switchable is that two runs of the same field
    differ only in this flag, so the comparison is attributable. That only
    holds if `False` touches nothing.
    """
    off, tab_off = _ab_pipeline(robust=False)
    on, tab_on = _ab_pipeline(robust=True)

    scenes_off = sorted(off.all_scenes[0], key=lambda s: s.id)
    scenes_on = sorted(on.all_scenes[0], key=lambda s: s.id)

    # off: the pass never runs, and every source keeps unit weight
    assert all(s.anchor_report is None for s in scenes_off)
    assert all(t.astrom_weight == 1.0 for s in scenes_off for t in s.templates)
    assert np.all(tab_off["astrom_weight_1"] == 1.0)

    # on: it ran, and the verdict is recorded per source
    assert any(s.anchor_report is not None for s in scenes_on)
    assert np.all(np.isfinite(tab_on["astrom_weight_1"]))
    assert tab_on["astrom_weight_1"].min() >= 0.0
    assert tab_on["astrom_weight_1"].max() <= 1.0

    # the schema is the same either way, so an A/B pair differs in values
    # rather than in columns
    assert set(tab_off.colnames) == set(tab_on.colnames)


@pytest.mark.parametrize("order", [0, 1])
def test_robust_weighting_works_at_higher_order(order):
    """Order enters only through the basis width, so both orders behave."""
    tm = _grid(12)
    f = np.full(12, 20.0)
    f[0] = 400.0
    # a genuine linear gradient across the field, plus one liar
    pos = np.array([t.position_original for t in tm], dtype=float)
    true_shift = [
        (0.10 + 4e-4 * (x - NX / 2), -0.05 - 3e-4 * (y - NY / 2)) for x, y in pos
    ]
    shifts = list(true_shift)
    shifts[0] = (1.0, 0.8)
    img = _paint(tm, f, shifts)
    W = np.ones((NY, NX))

    kw = dict(astrom_kwargs={"poly": {"order": order}, "gp": {"length_scale": 400}})
    naive = _solve_scene(tm, img, W, _cfg(astrom_robust=False, **kw))
    robust = _solve_scene(tm, img, W, _cfg(astrom_robust=True, **kw))

    # compare the field at the honest anchors, not the coefficients
    def _at(scn):
        from mophongo.astrometry import AstroCorrect

        _, (x0, y0), (sx, sy) = scn.shift_basis
        predict = AstroCorrect.build_poly_predictor(scn.shifts, x0, y0, order, sx, sy)
        dx, dy = predict(pos[1:, 0], pos[1:, 1])
        return np.stack([dx, dy], axis=1)

    want = np.array(true_shift[1:])
    err_naive = np.abs(_at(naive) - want).max()
    err_robust = np.abs(_at(robust) - want).max()

    assert robust.anchor_report.applied
    assert err_robust < 0.05
    assert err_robust < 0.3 * err_naive


def test_scene_shift_is_the_mean_of_applied_template_shifts():
    """The reported scene shift is what was applied, not a refit of it.

    Accumulated shifts are a sum of damped increments, each fitted at whatever
    the previous pass left behind, so at order >= 1 the total is not in general
    representable by the functional form of any one pass. Fitting the form back
    to it approximates something already known exactly.
    """
    tm = _grid(6)
    for i, t in enumerate(tm):
        t.shifted = np.array([0.1 * i, -0.05 * i], dtype=float)
    scn = Scene(id=1, templates=tm, fitter=SceneFitter(), bbox=_bbox_union(tm))

    want = np.mean([[0.1 * i, -0.05 * i] for i in range(6)], axis=0)
    assert np.allclose(scn.mean_shift(), want)


def test_scene_shift_ignores_templates_without_a_finite_shift():
    tm = _grid(4)
    for t in tm:
        t.shifted = np.array([0.2, -0.1], dtype=float)
    tm[0].shifted = np.array([np.nan, np.nan], dtype=float)
    scn = Scene(id=1, templates=tm, fitter=SceneFitter(), bbox=_bbox_union(tm))

    assert np.allclose(scn.mean_shift(), [0.2, -0.1])


def test_scene_shift_is_zero_without_templates():
    scn = Scene(id=1, templates=[], fitter=SceneFitter())
    assert np.allclose(scn.mean_shift(), [0.0, 0.0])


def test_shift_error_scales_with_the_noise():
    """The formal 1-sigma on a scene's shift, so `dx`/`dy` have a scale.

    Without it a 0.2 px shift cannot be told from zero, and `astrom_floor` --
    an *excess* over exactly this -- cannot be read at all.
    """
    truth = (0.20, -0.12)
    tm = _grid(12)
    f = np.full(12, 300.0)

    errs = []
    for sigma in (0.02, 0.08):
        rng = np.random.default_rng(5)
        img = _paint(tm, f, [truth] * 12) + rng.normal(0.0, sigma, (NY, NX))
        W = np.full((NY, NX), 1.0 / sigma**2)
        scn = _solve_scene(tm, img, W, _cfg())
        e = scn.shift_error()
        assert np.isfinite(e) and e > 0
        errs.append(e)

    # 4x the noise, 4x the positional uncertainty
    assert errs[1] / errs[0] == pytest.approx(4.0, rel=0.25)


def test_shift_error_is_nan_without_a_shift_fit():
    tm = _grid(6)
    img = _paint(tm, np.full(6, 100.0), [(0.1, -0.05)] * 6)
    scn = _solve_scene(tm, img, np.ones((NY, NX)), _cfg(fit_astrometry_niter=0))
    assert np.isnan(scn.shift_error())


def test_shift_error_brackets_the_actual_error():
    """A formal sigma that does not contain the truth is not worth writing."""
    truth = (0.20, -0.12)
    tm = _grid(12)
    f = np.full(12, 300.0)
    sigma = 0.03
    inside = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        img = _paint(tm, f, [truth] * 12) + rng.normal(0.0, sigma, (NY, NX))
        W = np.full((NY, NX), 1.0 / sigma**2)
        scn = _solve_scene(tm, img, W, _cfg())
        miss = np.abs(np.asarray(scn.shifts) - np.array(truth)).max()
        inside += miss < 3.0 * scn.shift_error()
    assert inside >= 7


def test_chi2_dof_is_near_one_for_a_good_fit_and_large_for_a_bad_one():
    """Ranks scenes by how badly they are fitted, which is the point."""
    truth = (0.15, -0.08)
    tm = _grid(9)
    f = np.full(9, 200.0)
    sigma = 0.05
    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, sigma, (NY, NX))
    W = np.full((NY, NX), 1.0 / sigma**2)

    good = _solve_scene(tm, _paint(tm, f, [truth] * 9) + noise, W, _cfg())
    assert good.chi2_dof() == pytest.approx(1.0, rel=0.15)

    # same scene, but one source painted with a profile its template lacks
    stamps = [t.data for t in tm]
    stamps[0] = _stamp(45.0, 45.0, 2.5, skew=1.5)
    bad = _solve_scene(
        tm, _paint(tm, f, [truth] * 9, stamps=stamps) + noise, W, _cfg()
    )
    assert bad.chi2_dof() > 2.0 * good.chi2_dof()


def test_chi2_dof_prefers_the_global_residual():
    """Passed a residual, it uses it rather than the scene's own."""
    tm = _grid(6)
    img = _paint(tm, np.full(6, 100.0), [(0.1, -0.05)] * 6)
    W = np.ones((NY, NX))
    scn = _solve_scene(tm, img, W, _cfg())

    perfect = np.zeros((NY, NX))
    assert scn.chi2_dof(perfect) == pytest.approx(0.0, abs=1e-12)
    assert scn.chi2_dof() > 0.0
