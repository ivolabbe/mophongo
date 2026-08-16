"""Tests for the exploratory MINERVA SED estimator comparison."""

from __future__ import annotations

import numpy as np
import pytest

# The module under test lives in `scratch/`, which is gitignored, so it is
# absent from a fresh checkout and from CI. Skip rather than fail collection:
# the experiment is exploratory and its test is only meaningful where the
# scratch tree exists.
sed_experiment = pytest.importorskip(
    "scratch.minerva_sed_estimator_experiment",
    reason="scratch/ is gitignored and absent from this checkout",
)

ESTIMATORS = sed_experiment.ESTIMATORS
estimate_cell = sed_experiment.estimate_cell
split_half_cell = sed_experiment.split_half_cell


def test_estimator_experiment_retains_signed_values_and_exposes_ivw_bias():
    values = np.array([-1.0, 0.0, 1.0, 10.0])
    errors = np.array([1.0, 1.0, 1.0, 0.001])
    estimates = estimate_cell(values, errors)

    assert estimates.values["equal"] == 2.5
    assert estimates.values["ivw_raw_failure"] > 9.99
    assert estimates.effective_count["equal"] == 4.0
    assert estimates.effective_count["ivw_raw_failure"] < 1.01
    assert estimates.values["winsor_0p5"] < estimates.values["equal"]


def test_split_half_experiment_is_seed_reproducible():
    rng = np.random.default_rng(31)
    values = rng.normal(size=101)
    errors = rng.uniform(0.2, 2.0, size=101)

    first_full, first_error = split_half_cell(
        values,
        errors,
        repeats=8,
        rng=np.random.default_rng(987),
    )
    second_full, second_error = split_half_cell(
        values,
        errors,
        repeats=8,
        rng=np.random.default_rng(987),
    )

    assert first_full == second_full
    assert first_error == second_error
    assert set(first_error) == set(ESTIMATORS)
    assert all(np.isfinite(first_error[name]) for name in ESTIMATORS)


def test_scatter_regularization_uses_equal_weights_for_zero_scatter():
    estimates = estimate_cell(
        np.ones(4),
        np.array([0.001, 0.1, 1.0, 100.0]),
    )
    assert estimates.values["scatter_regularized"] == 1.0
    assert estimates.effective_count["scatter_regularized"] == 4.0
