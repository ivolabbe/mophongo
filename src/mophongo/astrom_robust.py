"""Robust weighting of a scene's astrometric anchors.

The scene shift field is fitted by least squares, so each anchor's pull scales
as its Fisher information ``I_i = alpha_i^2 <grad T_i, w, grad T_i>`` -- as the
*square* of its flux. One bright extended source with an asymmetric colour
gradient therefore carries a scene on its own: its residual is a dipole aligned
with the template gradient, formally indistinguishable from a displacement.
``FitConfig.astrom_leverage_cap`` bounds that pull by clipping the top quantile
of ``I``, which is blind (it clips the brightest, which are usually the *best*
anchors) and arbitrary (a quantile, not a physical scale).

This module replaces the quantile with a measurement. Given a table of
per-anchor implied shifts and their information -- what
:func:`mophongo.scene.measure_anchor_shifts` produces -- it fits the shift
field robustly and returns a per-anchor weight built from two terms:

**Systematic floor.** Anchors scatter about the fitted field by more than their
formal errors whenever template morphology, PSF matching or colour gradients
limit the centroid. Estimating that excess as a common floor ``s`` and using
``v_i + s^2`` in place of ``v_i`` makes an anchor's weight saturate at
``1/s^2`` however bright it is. That is what ``astrom_leverage_cap`` was
approximating, with the cap now set by the data rather than by a quantile.

**Outlier rejection.** A source whose residual dipole comes from morphology
rather than displacement disagrees with its neighbours: a real offset is smooth
in position (the premise of the field model), while morphology-driven
pseudo-shifts are random per source. Tukey's biweight on the standardized
residual about the robustly fitted field rejects those, and rejects them
*hard* -- see :func:`robust_anchor_weights` for why a hard cutoff is preferable
to a Huber tail here.

Both terms need several anchors to mean anything, so the whole pass is gated on
anchor count and reports itself as inactive below it (see ``min_anchors``). In
particular nothing here can help a scene whose only bright member is the
offender: with one anchor the weight is a global scale, and a global scale
cannot change where the field lands.

The module is a leaf: plain arrays in, plain arrays out, no imports from
``scene``, ``fit``, ``templates`` or ``pipeline``. That keeps it usable by the
non-joint path in :mod:`mophongo.astrometry`, whose ``fit_polynomial_field``
weights anchors by SNR^2 and rejects nothing, i.e. fails the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AnchorWeights",
    "anchor_gate",
    "inactive_anchor_weights",
    "robust_anchor_weights",
]

#: Tukey biweight tuning constant, in units of the robust scale. 4.685 gives
#: 95% efficiency against a Gaussian and rejects outright beyond 4.685 sigma.
TUKEY_C = 4.685

#: Smallest scene-median reduced chi-square at which the per-anchor misfit
#: inflation is believed. A scene fitting a hundred times inside its own noise
#: has nothing to say about which of its anchors is misfit, and the ratios
#: between such numbers are floating-point noise.
CHI2_FLOOR = 1e-2

#: Median of a chi^2 with one degree of freedom. The systematic floor is set by
#: matching the median standardized square residual to this value, which is the
#: robust analogue of "reduced chi^2 equals one".
CHI2_1_MEDIAN = 0.4549364231195728


@dataclass
class AnchorWeights:
    """Outcome of a robust pass over one scene's anchor table.

    Attributes
    ----------
    weight
        ``(m,)`` multiplier for each anchor's leverage, in ``[0, 1]``. This is
        ``omega_i / I_i``: the ratio of the weight the robust fit gave the
        anchor to the information it carries on its own, which is exactly the
        factor :func:`mophongo.scene.assemble_scene_system_AB` needs to scale
        that anchor's contribution to the shift blocks. All ones when the pass
        is gated off.
    coeff
        ``(2, p)`` Chebyshev coefficients of the robustly fitted field, x then
        y. Diagnostic only -- the joint solve refits the field with these
        weights, it does not adopt these coefficients.
    field
        ``(m, 2)`` fitted field evaluated at the anchors.
    resid
        ``(m, 2)`` ``eps - field``, the disagreement each anchor was judged on.
    sys_floor
        Estimated systematic floor ``s``, in pixels. Zero when the anchors
        scatter no more than their formal errors allow.
    n_rejected
        Anchors given zero weight.
    n_eff
        Effective number of anchors surviving rejection,
        ``(sum w)^2 / sum w^2`` over the robustness weights. Deliberately not
        taken over the information-scaled weights, which would confuse a wide
        flux range with a heavy rejection.
    applied
        False when the pass was gated off or backed out, leaving ``weight``
        all ones.
    reason
        Why, when ``applied`` is False. Empty otherwise.
    """

    weight: np.ndarray
    coeff: np.ndarray
    field: np.ndarray
    resid: np.ndarray
    sys_floor: float
    n_rejected: int
    n_eff: float
    applied: bool
    reason: str = ""


def anchor_gate(min_anchors: int, n_terms: int) -> int:
    """Anchors a scene needs before the robust pass will judge it.

    A robust scatter about an ``n_terms``-parameter fit per axis needs
    comfortably more points than parameters, so the floor is the larger of the
    caller's ``min_anchors`` and twice the width of the basis.

    Exposed because the caller wants it *before* measuring: the measurement
    that fills the anchor table is the expensive part of the pass, and a scene
    that cannot clear this gate would only have it thrown away.

    Parameters
    ----------
    min_anchors
        Caller's floor, normally ``FitConfig.scene_minimum_anchors``.
    n_terms
        Width ``p`` of the shift-field basis.

    Returns
    -------
    int
        Minimum usable anchors.
    """
    return max(int(min_anchors), 2 * int(n_terms))


def inactive_anchor_weights(
    n_anchors: int, n_terms: int, reason: str, *, n_axes: int = 2
) -> AnchorWeights:
    """Unit-weight verdict for a scene the robust pass will not judge.

    Parameters
    ----------
    n_anchors, n_terms
        Anchor count ``m`` and basis width ``p`` the verdict is shaped for.
    reason
        Why the pass declined; carried on the report.
    n_axes
        Number of shift axes, i.e. the width of ``eps``.

    Returns
    -------
    AnchorWeights
        ``applied`` False, weights all ones.
    """
    m, p, k = int(n_anchors), int(n_terms), int(n_axes)
    return AnchorWeights(
        weight=np.ones(m, dtype=float),
        coeff=np.zeros((k, p)),
        field=np.zeros((m, k)),
        resid=np.zeros((m, k)),
        sys_floor=0.0,
        n_rejected=0,
        n_eff=float(m),
        applied=False,
        reason=reason,
    )



def _wls(basis: np.ndarray, values: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Weighted least squares of ``values`` on ``basis``, one column per axis.

    Parameters
    ----------
    basis
        ``(m, p)`` design matrix.
    values
        ``(m, k)`` observations, fitted independently per column.
    omega
        ``(m,)`` non-negative weights.

    Returns
    -------
    ndarray
        ``(k, p)`` coefficients. Rank-deficient systems fall back to the
        least-norm solution rather than raising, which is what
        ``np.linalg.lstsq`` gives.
    """
    root = np.sqrt(np.maximum(omega, 0.0))[:, None]
    coeff, *_ = np.linalg.lstsq(basis * root, values * root, rcond=None)
    return coeff.T


def _robust_start(
    basis: np.ndarray,
    values: np.ndarray,
    usable: np.ndarray,
    *,
    n_iter: int = 5,
    tukey_c: float = TUKEY_C,
) -> np.ndarray:
    """Leverage-blind robust fit: one anchor, one vote.

    The starting point decides everything about what follows. Beginning from
    the information-weighted least-squares fit does not work here, because that
    fit is the very thing under suspicion: an anchor with a hundred times the
    leverage of its neighbours *is* the least-squares answer, so every honest
    anchor looks like the outlier and the reweighting locks onto them. Huber
    does not rescue it either -- the leverage sits in the weights, not in the
    design, so the anchor that ought to be doubted has the smallest
    standardized residual of all.

    Ignoring the information entirely breaks that. With equal votes a lone
    liar is outnumbered whatever its flux, the fit lands near the majority, and
    the liar acquires the large residual it deserves. The price is statistical
    efficiency, which is why this is only the *start*: the information-weighted
    pass in :func:`robust_anchor_weights` recomputes the weights from scratch
    about this fit, so a bright anchor that was right is readmitted.

    Parameters
    ----------
    basis
        ``(m, p)`` design matrix.
    values
        ``(m, k)`` observations.
    usable
        ``(m,)`` mask of anchors that may vote.
    n_iter
        Reweighting steps.
    tukey_c
        Tukey tuning constant, in units of the residual MAD.

    Returns
    -------
    ndarray
        ``(k, p)`` starting coefficients.
    """
    w = usable.astype(float)
    coeff = _wls(basis, values, w)
    for _ in range(n_iter):
        r = values - basis @ coeff.T
        peak = np.max(np.abs(r), axis=1)
        scale = 1.4826 * float(np.median(np.abs(r[usable])))
        if scale > 0:
            u = peak / (tukey_c * scale)
        else:
            # An exact fit to the majority: anything off it is an outlier, and
            # anything on it is perfect. Guards against 0/0 as well.
            u = np.where(peak > 0, np.inf, 0.0)
        w = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0) * usable
        if not np.any(w > 0):
            break
        coeff = _wls(basis, values, w)
    return coeff


def _systematic_floor(resid: np.ndarray, var: np.ndarray) -> float:
    """Excess scatter of ``resid`` beyond the formal variances ``var``.

    Solves ``median_i,a[ resid_ia^2 / (var_i + s^2) ] = median(chi^2_1)`` for
    ``s^2 >= 0`` by bisection. The left side decreases monotonically in
    ``s^2``, so the root is unique when it exists; if the anchors already
    scatter *less* than their formal errors the answer is zero.

    Using the median rather than the mean is what keeps a single wild anchor
    from inflating the floor for everybody -- which would defeat the purpose,
    since a floor large enough to accommodate the outlier also flattens the
    weights of the anchors that were right.

    Parameters
    ----------
    resid
        ``(m, k)`` residuals about the fitted field.
    var
        ``(m,)`` formal variances, broadcast across the ``k`` axes.

    Returns
    -------
    float
        ``s`` in the units of ``resid`` (pixels).
    """
    r2 = np.asarray(resid, float) ** 2
    v = np.asarray(var, float)[:, None]

    def excess(s2: float) -> float:
        return float(np.median(r2 / (v + s2))) - CHI2_1_MEDIAN

    if excess(0.0) <= 0.0:
        return 0.0
    # The median of r^2/(v + s^2) falls below the target once s^2 exceeds the
    # median of r^2, so that is a guaranteed bracket.
    hi = float(np.median(r2)) / CHI2_1_MEDIAN + np.median(v)
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(0.5 * (lo + hi)))


def robust_anchor_weights(
    eps: np.ndarray,
    info: np.ndarray,
    basis: np.ndarray,
    *,
    chi2_red: np.ndarray | None = None,
    min_anchors: int = 5,
    tukey_c: float = TUKEY_C,
    n_iter: int = 8,
) -> AnchorWeights:
    """Weight a scene's anchors by agreement with the fitted shift field.

    Fits the field to the per-anchor implied shifts by iteratively reweighted
    least squares, estimating a common systematic floor as it goes, and returns
    the ratio of each anchor's final weight to its own information. Works at
    any polynomial order: the order enters only through the width of ``basis``.

    The estimator is of MM type: a high-breakdown starting fit that ignores
    the anchors' information entirely (:func:`_robust_start`), followed by
    redescending Tukey steps that put the information back. Both halves are
    needed. Without the leverage-blind start the reweighting *masks*: the
    dominant anchor is the least-squares answer, so the honest anchors carry
    the large residuals and get cut instead. A Huber warm-up does not
    substitute for it either -- its weights decay only as ``1/u``, so against
    an anchor with 400x the leverage of its neighbours a tenfold downweight
    still leaves it holding most of the vote, and the fit walks back onto it.

    Rejection is hard (Tukey), not a decaying tail (Huber), for a reason
    specific to how the weight is consumed. The weight enters
    :func:`mophongo.scene.assemble_scene_system_AB` through ``lev_w``, whose
    two-power split is exact for the shift-only block but leaves the
    flux-marginalization term ``AB^T A^-1 AB`` scaled as ``c_i c_j`` where the
    information it corrects scales as ``sqrt(c_i c_j)``. That mismatch is worst
    at intermediate weights and vanishes identically at ``c_i = 0``, where the
    anchor becomes exactly equivalent to one that was never bright.

    Parameters
    ----------
    eps
        ``(m, 2)`` implied shift of each anchor, in pixels, measured against
        the flux-only residual.
    info
        ``(m,)`` Fisher information of each implied shift, ``1 / var(eps)``,
        averaged over the two axes. Non-positive entries are treated as
        uninformative and get zero weight.
    basis
        ``(m, p)`` shift-field basis evaluated at the anchors, i.e. the rows
        :func:`mophongo.scene.make_scene_basis` produced for these anchors.
    chi2_red
        ``(m,)`` reduced chi-square of each anchor's own residual after its
        private shift and flux are projected out. Used *relative to the scene
        median* to inflate that anchor's variance: a source whose stamp does
        not fit even after being allowed to move is not a source whose position
        should be trusted. Relative because the absolute value is not
        trustworthy -- drizzled pixels are correlated, so the nominal degrees
        of freedom are too generous. ``None`` skips the inflation, as do
        scenes with fewer than three usable values, where the median it would
        be measured against means nothing.
    min_anchors
        Floor on the anchor count. The effective gate is
        ``max(min_anchors, 2 * p)``: a robust scatter about a ``p``-term fit
        per axis needs comfortably more points than parameters.
    tukey_c
        Tukey tuning constant in units of the robust scale.
    n_iter
        Reweighting steps after the high-breakdown start.

    Returns
    -------
    AnchorWeights
        With ``applied`` False and unit weights whenever the scene is too
        small to judge, the field cannot be fitted, or rejection would leave
        too few anchors standing.
    """
    eps = np.atleast_2d(np.asarray(eps, dtype=float))
    info = np.asarray(info, dtype=float).ravel()
    basis = np.atleast_2d(np.asarray(basis, dtype=float))
    m, p = basis.shape

    def _inactive(reason: str) -> AnchorWeights:
        return inactive_anchor_weights(
            m, p, reason, n_axes=eps.shape[1] if eps.size else 2
        )

    if eps.shape[0] != m or info.size != m:
        raise ValueError(
            f"eps {eps.shape}, info {info.shape} and basis {basis.shape} "
            "must agree on the anchor count"
        )

    usable = np.isfinite(info) & (info > 0) & np.all(np.isfinite(eps), axis=1)
    # Unusable rows carry zero weight throughout, but a NaN would still poison
    # the weighted design matrix (0 * nan = nan), so blank them here.
    eps = np.where(usable[:, None], eps, 0.0)
    gate = anchor_gate(min_anchors, p)
    if int(usable.sum()) < gate:
        return _inactive(
            f"{int(usable.sum())} usable anchor(s) < {gate} required for "
            f"order with {p} term(s)"
        )

    # Per-anchor variance of the implied shift, inflated where the anchor's own
    # stamp does not fit even after its private shift is removed.
    var = np.where(usable, 1.0 / np.where(usable, info, 1.0), np.inf)
    if chi2_red is not None:
        c = np.asarray(chi2_red, dtype=float).ravel()
        good = usable & np.isfinite(c) & (c > 0)
        # The inflation is a ratio, so it needs a scene whose misfits mean
        # something. Below CHI2_FLOOR every anchor already fits orders of
        # magnitude inside its own noise, and the ratios between such numbers
        # are roundoff rather than evidence.
        if good.sum() >= 3 and np.median(c[good]) >= CHI2_FLOOR:
            var = var * np.where(good, np.maximum(1.0, c / np.median(c[good])), 1.0)

    # Start from a fit no single anchor can buy, then let the information back
    # in. Weights are recomputed from scratch about the current fit on every
    # pass, so nothing the start rejected is permanently barred.
    coeff = _robust_start(basis, eps, usable, tukey_c=tukey_c)
    if not np.all(np.isfinite(coeff)):
        return _inactive("shift-field fit did not produce finite coefficients")

    w = usable.astype(float)
    s = 0.0
    resid = eps - basis @ coeff.T
    for _ in range(n_iter):
        resid = eps - basis @ coeff.T
        s = _systematic_floor(resid[usable], var[usable])

        sigma = np.sqrt(var + s**2)
        # One weight per anchor, not one per axis: an anchor whose residual
        # dipole comes from its morphology is untrustworthy in both.
        u = np.max(np.abs(resid), axis=1) / np.maximum(sigma, 1e-30)
        t = u / tukey_c
        w = np.where((t < 1.0) & usable, (1.0 - t**2) ** 2, 0.0)

        omega = np.where(usable, w / (var + s**2), 0.0)
        if not np.any(omega > 0):
            return _inactive("every anchor lost its weight during the fit")
        coeff = _wls(basis, eps, omega)
        if not np.all(np.isfinite(coeff)):
            return _inactive("shift-field fit did not produce finite coefficients")

    resid = eps - basis @ coeff.T
    omega = np.where(usable, w / (var + s**2), 0.0)

    # Count survivors, not influence. The effective count is taken on the
    # robustness weights alone: computing it on `omega` would fold in the
    # anchors' natural flux range, so a scene of six honest anchors spanning
    # 4x in flux would report ~3.9 "effective" anchors and be refused for a
    # concentration that is not rejection at all.
    n_eff = float(w.sum() ** 2 / np.sum(w**2)) if np.any(w > 0) else 0.0
    if n_eff < gate:
        return _inactive(
            f"rejection left {n_eff:.1f} effective anchor(s) < {gate} required"
        )

    # lev_w scales the information the blocks carry for this anchor from I_i to
    # the weight the robust fit actually gave it.
    weight = np.where(usable, omega / np.where(usable, info, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    n_rej = int(np.sum(usable & (w <= 0.0)))
    logger.debug(
        "[astrom] robust anchors: m=%d p=%d floor=%.4f px rejected=%d "
        "n_eff=%.1f weight %.3g-%.3g",
        m, p, s, n_rej, n_eff, float(weight.min()), float(weight.max()),
    )
    return AnchorWeights(
        weight=weight,
        coeff=coeff,
        field=basis @ coeff.T,
        resid=resid,
        sys_floor=float(s),
        n_rejected=n_rej,
        n_eff=n_eff,
        applied=True,
    )
