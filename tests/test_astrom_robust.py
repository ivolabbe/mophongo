"""Robust anchor weighting, exercised on synthetic anchor tables.

The tables here are what :func:`mophongo.scene.measure_anchor_shifts` produces
-- implied shift, information and reduced chi-square per anchor -- so the
weighting can be judged on its own, without a scene, an image or a solve in the
way.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from mophongo.astrom_robust import AnchorWeights, robust_anchor_weights
from mophongo.astrometry import cheb_basis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _basis(pos: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev rows for anchors at normalized positions ``pos`` in [-1, 1]."""
    return np.array([cheb_basis(x, y, order) for x, y in pos])


def _table(
    rng: np.random.Generator,
    m: int,
    order: int,
    *,
    coeff: np.ndarray | None = None,
    sys_floor: float = 0.0,
    info_lo: float = 1e2,
    info_hi: float = 1e4,
):
    """Anchor table drawn from a known field, formal errors plus a floor.

    Returns ``(eps, info, basis, coeff)`` with ``coeff`` of shape ``(2, p)``.
    """
    pos = rng.uniform(-1.0, 1.0, size=(m, 2))
    B = _basis(pos, order)
    p = B.shape[1]
    if coeff is None:
        coeff = np.zeros((2, p))
        coeff[:, 0] = [0.20, -0.11]
        if p > 1:
            coeff[:, 1:] = rng.uniform(-0.03, 0.03, size=(2, p - 1))
    info = 10 ** rng.uniform(np.log10(info_lo), np.log10(info_hi), size=m)
    sigma = np.sqrt(1.0 / info + sys_floor**2)
    eps = B @ coeff.T + rng.normal(0.0, sigma[:, None], size=(m, 2))
    return eps, info, B, coeff


def _plain_wls(eps: np.ndarray, info: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Information-weighted least squares -- what the solve does unweighted."""
    root = np.sqrt(info)[:, None]
    coeff, *_ = np.linalg.lstsq(B * root, eps * root, rcond=None)
    return coeff.T


# ---------------------------------------------------------------------------
# clean data must be left alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_clean_anchors_keep_their_weight_and_recover_the_field(order):
    """No outlier, no systematic: nothing is rejected and the field is right."""
    rng = np.random.default_rng(11)
    eps, info, B, coeff = _table(rng, 60, order)

    res = robust_anchor_weights(eps, info, B)

    assert res.applied
    assert res.n_rejected == 0
    # Tukey costs efficiency on clean data by design -- with 2m standardized
    # residuals the largest is a couple of sigma and gets visibly downweighted.
    # The mean is what must stay near one.
    assert res.weight.min() > 0.15
    assert res.weight.mean() > 0.80
    assert res.sys_floor < 0.01
    assert np.allclose(res.coeff, coeff, atol=0.01)


def test_no_systematic_means_a_floor_well_below_the_formal_errors():
    """The floor is an excess, so with none present it must be negligible.

    Not identically zero: the estimator is one-sided, so the sampling scatter
    of the median standardized residual leaks into a small positive floor. That
    errs toward flatter weights, which is the safe direction.
    """
    rng = np.random.default_rng(3)
    eps, info, B, _ = _table(rng, 80, 0, sys_floor=0.0)
    res = robust_anchor_weights(eps, info, B)
    assert res.sys_floor < 0.3 * np.median(1.0 / np.sqrt(info))


# ---------------------------------------------------------------------------
# the systematic floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s_true,rtol",
    [
        # A floor below most of the formal errors (which span 0.010-0.098 px
        # here) is only weakly constrained -- it is buried in the statistical
        # scatter it has to be separated from -- so it reads low.
        (0.01, 0.5),
        (0.03, 0.2),
        (0.10, 0.2),
    ],
)
def test_systematic_floor_is_recovered(s_true, rtol):
    """Excess scatter beyond the formal errors comes back as ``sys_floor``."""
    rng = np.random.default_rng(19)
    eps, info, B, _ = _table(rng, 200, 0, sys_floor=s_true)

    res = robust_anchor_weights(eps, info, B)

    assert res.applied
    assert res.sys_floor == pytest.approx(s_true, rel=rtol)


def test_floor_caps_the_leverage_of_a_very_bright_anchor():
    """With a floor, weight saturates at 1/s^2 however bright the anchor is.

    This is the property ``astrom_leverage_cap`` approximates with a quantile:
    an anchor 10^4 times better determined than the floor allows must not
    count 10^4 times more.
    """
    rng = np.random.default_rng(23)
    s_true = 0.05
    eps, info, B, _ = _table(rng, 120, 0, sys_floor=s_true)
    # one anchor with vastly more formal information than the rest
    info[0] = 1e8
    eps[0] = eps[1:].mean(axis=0) + np.array([s_true, -s_true])

    res = robust_anchor_weights(eps, info, B)
    omega = res.weight * info

    assert res.applied
    # its post-weight influence is within a factor of a few of the others',
    # not the 10^4 its raw information would buy
    assert omega[0] < 5.0 * np.median(omega[1:])
    assert res.weight[0] < 1e-4


# ---------------------------------------------------------------------------
# outlier rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_one_dominant_outlier_is_rejected_at_any_order(order):
    """The failure this module exists for: brightest anchor, wrong shift."""
    rng = np.random.default_rng(5)
    m = 40
    eps, info, B, coeff = _table(rng, m, order, info_lo=1e3, info_hi=1e4)
    # a bright, extended source with a colour gradient: 30x the information of
    # its neighbours and a half-pixel pseudo-shift that is pure morphology
    info[0] = 30.0 * info.max()
    eps[0] = B[0] @ coeff.T + np.array([0.5, -0.4])

    naive = _plain_wls(eps, info, B)
    res = robust_anchor_weights(eps, info, B)

    assert res.applied
    assert res.weight[0] == 0.0, "the offending anchor must be rejected outright"
    # naive fit is dragged toward the outlier; robust fit is not
    err_naive = np.abs(naive[:, 0] - coeff[:, 0]).max()
    err_robust = np.abs(res.coeff[:, 0] - coeff[:, 0]).max()
    assert err_naive > 0.05
    assert err_robust < 0.01
    assert err_robust < 0.1 * err_naive


def test_a_genuinely_offset_scene_is_not_rejected():
    """A real offset is coherent, so no anchor disagrees and none is cut.

    The discriminator is disagreement with the neighbours, not shift size --
    a scene that really has moved a long way must keep all of its anchors.
    """
    rng = np.random.default_rng(29)
    eps, info, B, coeff = _table(rng, 50, 0)
    eps += np.array([1.7, -2.3])  # whole scene displaced

    res = robust_anchor_weights(eps, info, B)

    assert res.applied
    assert res.n_rejected == 0
    assert np.allclose(res.coeff[:, 0], coeff[:, 0] + np.array([1.7, -2.3]), atol=0.01)


def test_a_minority_of_outliers_does_not_capture_the_fit():
    """Three bright liars against thirty honest anchors lose."""
    rng = np.random.default_rng(31)
    eps, info, B, coeff = _table(rng, 33, 0, info_lo=1e3, info_hi=3e3)
    for i in (0, 1, 2):
        info[i] = 50.0 * info.max()
        eps[i] = coeff[:, 0] + np.array([0.6, 0.6])

    res = robust_anchor_weights(eps, info, B)

    assert res.n_rejected >= 3
    assert np.allclose(res.coeff[:, 0], coeff[:, 0], atol=0.02)


# ---------------------------------------------------------------------------
# chi2 inflation
# ---------------------------------------------------------------------------


def test_chi2_inflation_downweights_a_stamp_that_does_not_fit():
    """An anchor whose own stamp misfits is trusted less, outlier or not."""
    rng = np.random.default_rng(37)
    eps, info, B, _ = _table(rng, 40, 0)
    chi2 = np.ones(40)
    chi2[0] = 25.0  # unmodelled structure after its private shift is removed

    plain = robust_anchor_weights(eps, info, B)
    inflated = robust_anchor_weights(eps, info, B, chi2_red=chi2)

    assert inflated.weight[0] < 0.1 * plain.weight[0]
    # everyone else is untouched
    assert np.allclose(inflated.weight[1:], plain.weight[1:], rtol=0.2)


def test_chi2_inflation_stands_down_when_everything_fits_absurdly_well():
    """Ratios between numbers far inside the noise are roundoff, not evidence."""
    rng = np.random.default_rng(83)
    eps, info, B, _ = _table(rng, 40, 0)
    chi2 = rng.uniform(1e-30, 1e-28, size=40)  # noiseless-model roundoff

    plain = robust_anchor_weights(eps, info, B)
    tiny = robust_anchor_weights(eps, info, B, chi2_red=chi2)

    assert np.allclose(plain.weight, tiny.weight)


def test_chi2_inflation_is_relative_so_a_global_offset_does_nothing():
    """Correlated pixels bias every reduced chi-square the same way."""
    rng = np.random.default_rng(41)
    eps, info, B, _ = _table(rng, 40, 0)
    chi2 = rng.uniform(0.8, 1.2, size=40)

    a = robust_anchor_weights(eps, info, B, chi2_red=chi2)
    b = robust_anchor_weights(eps, info, B, chi2_red=7.5 * chi2)

    assert np.allclose(a.weight, b.weight)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order,m", [(0, 4), (1, 5), (2, 10)])
def test_gate_leaves_small_scenes_alone(order, m):
    """Below 2p anchors (or ``min_anchors``) there is nothing to judge.

    The gate is ``max(min_anchors, 2p)``: 5, 6 and 12 for orders 0, 1 and 2.
    """
    rng = np.random.default_rng(43)
    eps, info, B, _ = _table(rng, m, order)

    res = robust_anchor_weights(eps, info, B, min_anchors=5)

    assert not res.applied
    assert np.all(res.weight == 1.0)
    assert "required" in res.reason


def test_single_anchor_is_a_no_op():
    """One anchor: the weight is a global scale and cannot move the field."""
    res = robust_anchor_weights(
        np.array([[0.2, -0.1]]), np.array([1e4]), np.ones((1, 1))
    )
    assert not res.applied
    assert res.weight == pytest.approx(1.0)


def test_non_finite_anchors_are_dropped_not_propagated():
    rng = np.random.default_rng(47)
    eps, info, B, coeff = _table(rng, 30, 0)
    eps[0] = np.nan
    info[1] = 0.0
    info[2] = -1.0

    res = robust_anchor_weights(eps, info, B)

    assert res.applied
    assert np.all(res.weight[:3] == 0.0)
    assert np.all(np.isfinite(res.weight))
    assert np.allclose(res.coeff[:, 0], coeff[:, 0], atol=0.02)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="anchor count"):
        robust_anchor_weights(np.zeros((5, 2)), np.ones(4), np.ones((5, 1)))


def test_weights_never_exceed_unity():
    """``lev_w`` is a downweight; it must never amplify an anchor."""
    rng = np.random.default_rng(53)
    for seed_order in (0, 1, 2):
        eps, info, B, _ = _table(rng, 60, seed_order, sys_floor=0.02)
        res = robust_anchor_weights(eps, info, B)
        assert res.weight.max() <= 1.0
        assert res.weight.min() >= 0.0
