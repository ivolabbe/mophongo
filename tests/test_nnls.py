"""Non-negative flux solving.

`FitConfig.fit_method` picks between "lls" (keep negatives), "clip" (clamp the
unconstrained solution) and "nnls" (solve under the constraint). Clipping is not
the constrained optimum -- pinning one template at zero changes what its
neighbours should be -- and these tests pin that difference, and check the FNNLS
path against `scipy.optimize.nnls` on the equivalent dense problem.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import nnls

from mophongo.fit import FitConfig
from mophongo.scene_fitter import SceneFitter


def _normal(M: np.ndarray, d: np.ndarray):
    """The (Gram, projected data) pair a scene actually holds."""
    return sp.csr_matrix(M.T @ M), M.T @ d


def test_fnnls_matches_scipy_nnls_on_random_problems():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, n = 40, 6
        M = rng.standard_normal((m, n))
        d = rng.standard_normal(m)
        A, b = _normal(M, d)

        x, _passive = SceneFitter.fnnls(A, b)
        want, _ = nnls(M, d)

        np.testing.assert_allclose(x, want, atol=1e-8)
        assert np.all(x >= 0.0)


def test_fnnls_is_exact_when_the_solution_is_interior():
    """With no active constraint it must reproduce the plain solve."""
    rng = np.random.default_rng(3)
    M = rng.standard_normal((60, 4))
    truth = np.array([5.0, 3.0, 8.0, 1.0])
    d = M @ truth
    A, b = _normal(M, d)

    x, passive = SceneFitter.fnnls(A, b)

    np.testing.assert_allclose(x, truth, rtol=1e-8)
    assert passive.all()


def test_clip_and_nnls_differ_on_a_blend():
    """The case the constraint exists for.

    Two heavily overlapping templates: the fit drives one negative, and the
    other absorbs flux to compensate. Clipping zeroes the negative one and
    leaves the inflated neighbour untouched; NNLS re-solves it.
    """
    x = np.linspace(-4, 4, 200)
    t1 = np.exp(-0.5 * ((x - 0.4) / 1.0) ** 2)
    t2 = np.exp(-0.5 * ((x + 0.4) / 1.0) ** 2)  # ~85% correlated with t1
    M = np.column_stack([t1, t2])
    # data consistent with only the second template present
    d = 10.0 * t2
    # push the first negative
    d = d - 1.5 * t1
    A, b = _normal(M, d)

    uncon = np.linalg.solve(A.toarray(), b)
    assert uncon[0] < 0, "fixture must drive a template negative"

    clipped = np.maximum(0.0, uncon)
    x_nnls, passive = SceneFitter.fnnls(A, b)

    assert x_nnls[0] == pytest.approx(0.0, abs=1e-9)
    assert not passive[0] and passive[1]
    # the surviving template is re-solved, so it does NOT keep the clipped value
    assert abs(x_nnls[1] - clipped[1]) > 1e-6

    # and NNLS is the better fit of the two feasible answers
    resid = lambda v: float(np.sum((M @ v - d) ** 2))
    assert resid(x_nnls) < resid(clipped)


def test_solve_flux_nnls_path_is_selected_by_config():
    rng = np.random.default_rng(11)
    M = rng.standard_normal((80, 5))
    d = M @ np.array([4.0, -2.0, 6.0, 0.5, -1.0])
    A, b = _normal(M, d)

    clip = SceneFitter.solve_flux(A, b, FitConfig(fit_method="clip"))
    nnls_ = SceneFitter.solve_flux(A, b, FitConfig(fit_method="nnls"))

    assert clip[2]["solver"] == "spsolve"
    assert nnls_[2]["solver"] == "fnnls"
    assert np.all(clip[0] >= 0) and np.all(nnls_[0] >= 0)
    # same fixture drives components negative, so the two must disagree
    assert not np.allclose(clip[0], nnls_[0])


def test_unconstrained_fluxes_are_always_returned():
    """A clipped zero is otherwise unrecoverable, and the faint end needs it."""
    rng = np.random.default_rng(5)
    M = rng.standard_normal((60, 4))
    d = M @ np.array([3.0, -1.0, 2.0, -0.5])
    A, b = _normal(M, d)

    for method in ("clip", "nnls"):
        cfg = FitConfig(fit_method=method)
        x, _err, info = SceneFitter.solve_flux(A, b, cfg)
        uncon = info["flux_uncon"]

        assert np.any(uncon < 0), "fixture must produce negative fluxes"
        np.testing.assert_allclose(
            uncon, np.linalg.solve(A.toarray(), b), rtol=1e-7
        )
        assert np.all(x >= 0.0)
        assert info["at_bound"].any()


def test_lls_leaves_negatives_alone():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((60, 3))
    d = M @ np.array([2.0, -3.0, 1.0])
    A, b = _normal(M, d)

    x, _err, info = SceneFitter.solve_flux(A, b, FitConfig(fit_method="lls"))

    assert np.any(x < 0)
    np.testing.assert_allclose(x, info["flux_uncon"], rtol=1e-12)
    assert "at_bound" not in info


def test_errors_are_the_unconstrained_ones_and_bound_flags_say_so():
    """The error is not re-derived under the constraint.

    `sqrt(diag(A^-1))` describes the unconstrained estimator. At an active
    bound the sampling distribution is truncated and asymmetric, so the number
    reported there is not a 1-sigma interval for that parameter -- which is
    what `at_bound` exists to say.
    """
    rng = np.random.default_rng(13)
    M = rng.standard_normal((70, 4))
    d = M @ np.array([3.0, -2.0, 1.0, 4.0])
    A, b = _normal(M, d)

    free = SceneFitter.solve_flux(A, b, FitConfig(fit_method="lls"))
    bound = SceneFitter.solve_flux(A, b, FitConfig(fit_method="nnls"))

    np.testing.assert_allclose(free[1], bound[1], rtol=1e-12)
    assert bound[2]["at_bound"].any()


def test_nnls_is_the_default():
    assert FitConfig().fit_method == "nnls"


def test_unknown_fit_method_raises():
    with pytest.raises(ValueError, match="fit_method"):
        FitConfig(fit_method="magic")


def test_clip_reproduces_the_previous_default_exactly():
    """The old behaviour must stay recoverable for A/B work.

    `clip` is what every run before this change did: solve unconstrained, then
    clamp. Pin it against that definition rather than against a recorded array,
    so the guarantee survives a refactor of the solve.
    """
    rng = np.random.default_rng(21)
    M = rng.standard_normal((90, 6))
    d = M @ np.array([5.0, -2.0, 3.0, -0.5, 1.0, -4.0])
    A, b = _normal(M, d)

    lls, _e1, _i1 = SceneFitter.solve_flux(A, b, FitConfig(fit_method="lls"))
    clip, _e2, _i2 = SceneFitter.solve_flux(A, b, FitConfig(fit_method="clip"))

    np.testing.assert_allclose(clip, np.maximum(0.0, lls), rtol=0, atol=0)
    assert np.any(lls < 0), "fixture must exercise the clamp"


def test_all_three_methods_recover_injected_truth_when_it_is_positive():
    """With no negative truth, the constraint must not cost anything.

    An estimator that only helps by biasing would show up here as `nnls`
    disagreeing with `lls` on a clean positive problem.
    """
    rng = np.random.default_rng(31)
    M = np.abs(rng.standard_normal((300, 8)))
    truth = np.array([10.0, 4.0, 7.0, 1.0, 12.0, 3.0, 5.0, 8.0])
    d = M @ truth + 0.01 * rng.standard_normal(300)
    A, b = _normal(M, d)

    got = {m: SceneFitter.solve_flux(A, b, FitConfig(fit_method=m))[0]
           for m in ("lls", "clip", "nnls")}

    for method, x in got.items():
        np.testing.assert_allclose(x, truth, rtol=2e-2, err_msg=method)
    np.testing.assert_allclose(got["nnls"], got["lls"], rtol=1e-6)


def test_nnls_beats_clip_at_recovering_truth_in_a_blend():
    """The A/B that motivates the default.

    A faint source sits on a bright neighbour. Noise drives the faint one
    negative; clipping zeroes it and leaves the bright one carrying the excess,
    while NNLS re-solves the bright one back toward its true flux.
    """
    x = np.linspace(-6, 6, 400)
    bright = np.exp(-0.5 * ((x - 0.5) / 1.2) ** 2)
    faint = np.exp(-0.5 * ((x + 0.5) / 1.2) ** 2)
    M = np.column_stack([faint, bright])
    truth = np.array([0.0, 20.0])          # the faint source is truly absent
    rng = np.random.default_rng(41)

    clip_err, nnls_err = [], []
    for _ in range(40):
        d = M @ truth + 0.5 * rng.standard_normal(x.size)
        A, b = _normal(M, d)
        clip = SceneFitter.solve_flux(A, b, FitConfig(fit_method="clip"))[0]
        nn = SceneFitter.solve_flux(A, b, FitConfig(fit_method="nnls"))[0]
        clip_err.append(abs(clip[1] - truth[1]))
        nnls_err.append(abs(nn[1] - truth[1]))

    # the bright neighbour is recovered better once the survivors are re-solved
    assert np.median(nnls_err) < np.median(clip_err)


def test_warm_start_agrees_with_scipy_when_many_components_are_pinned():
    """Exercises the KKT shortcut, its refinement, and the exact fallback.

    The shortcut returns early only when the KKT test passes, so a disagreement
    here would mean the test itself is wrong -- which is the thing worth
    guarding, since the fast path is what production runs take.
    """
    rng = np.random.default_rng(101)
    for _ in range(5):
        m, n = 200, 40
        M = rng.standard_normal((m, n))
        truth = rng.standard_normal(n) * 5
        d = M @ truth + 0.5 * rng.standard_normal(m)
        A, b = _normal(M, d)

        x, passive = SceneFitter.fnnls(A, b)
        want, _ = nnls(M, d)

        np.testing.assert_allclose(x, want, atol=1e-6)
        assert np.all(x[~passive] == 0.0)
        assert (~passive).any(), "fixture must pin something"


def test_all_positive_scene_returns_the_plain_solve_at_scale():
    """The common case must not pay for the constraint.

    Lawson-Hanson from x = 0 admits one component per iteration, so without a
    warm start a fully positive scene of this size costs hundreds of growing
    solves to reproduce what one solve gives.
    """
    n = 300
    rng = np.random.default_rng(102)
    A = sp.diags(np.ones(n)).tolil()
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 0.3
    A = A.tocsr()
    truth = np.abs(rng.standard_normal(n)) * 10 + 1.0
    b = A @ truth

    x, passive = SceneFitter.fnnls(A, b)

    assert passive.all()
    np.testing.assert_allclose(x, truth, rtol=1e-8)


def test_shifts_are_never_constrained_in_the_joint_system():
    """`free` components may go negative and are never pinned."""
    rng = np.random.default_rng(103)
    na, nb, m = 6, 3, 200
    M = np.abs(rng.standard_normal((m, na)))
    S = rng.standard_normal((m, nb))
    D = np.column_stack([M, S])
    free = np.concatenate([np.zeros(na, bool), np.ones(nb, bool)])

    truth = np.concatenate([[5.0, -4.0, 8.0, 2.0, 6.0, 4.0], [-2.0, 1.0, -3.0]])
    A, b = _normal(D, D @ truth)

    x, passive = SceneFitter.fnnls(A, b, free=free)

    assert np.all(x[:na] >= -1e-12), "fluxes are constrained"
    assert passive[na:].all(), "shift components must never be pinned"
    assert np.any(x[na:] < 0), "a shift must be free to go negative"
    # with nothing pinned, the joint answer is the unconstrained one
    x2, passive2 = SceneFitter.fnnls(*_normal(D, D @ np.abs(truth)), free=free)
    np.testing.assert_allclose(
        x2, np.linalg.solve(D.T @ D, D.T @ (D @ np.abs(truth))), atol=1e-8
    )
    assert passive2.all()
