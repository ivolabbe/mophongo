"""Tests for broadband SED rasterization and redshift stacking."""

from __future__ import annotations

import numpy as np

from mophongo.sed_stack import (
    binned_measurement_statistics,
    combine_sed_stacks,
    filter_interval_wavelength_edges,
    fnu_to_flam_proxy,
    normalize_at_rest_wavelength,
    paint_filter_lines,
    redshift_bin_edges,
    stack_filter_seds,
    stack_interpolated_seds,
)


def test_binned_statistics_retain_signed_values_and_empirical_scatter():
    statistics = binned_measurement_statistics(
        values=np.array([[-1.0], [0.0], [1.0]]),
        errors=np.ones((3, 1)),
        valid=np.ones((3, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 1.0]),
        redshift_edges=np.array([0.0, 1.0]),
    )
    assert statistics.count[0, 0] == 3
    assert statistics.mean[0, 0] == 0.0
    assert statistics.median[0, 0] == 0.0
    assert statistics.winsorized_mean[0, 0] == 0.0
    np.testing.assert_allclose(statistics.standard_error[0, 0], 1.0 / np.sqrt(3.0))
    assert statistics.inverse_variance_mean[0, 0] == 0.0
    assert statistics.inverse_variance_effective_count[0, 0] == 3.0
    assert statistics.regularized_weighted_mean[0, 0] == 0.0
    assert statistics.regularized_effective_count[0, 0] == 3.0


def test_binned_inverse_variance_diagnostic_caps_dominant_weight():
    common = dict(
        values=np.array([[0.0], [0.0], [10.0]]),
        errors=np.array([[1.0], [1.0], [0.01]]),
        valid=np.ones((3, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 0.3]),
        redshift_edges=np.array([0.0, 1.0]),
    )
    capped = binned_measurement_statistics(
        **common, weight_cap_percentile=50.0
    )
    uncapped = binned_measurement_statistics(
        **common, weight_cap_percentile=100.0
    )
    np.testing.assert_allclose(capped.inverse_variance_mean[0, 0], 10.0 / 3.0)
    np.testing.assert_allclose(
        capped.inverse_variance_effective_count[0, 0], 3.0
    )
    np.testing.assert_allclose(capped.maximum_weight_fraction[0, 0], 1.0 / 3.0)
    assert uncapped.inverse_variance_mean[0, 0] > 9.9
    assert uncapped.inverse_variance_effective_count[0, 0] < 1.01
    assert capped.regularized_effective_count[0, 0] > 2.9


def test_binned_winsorized_mean_limits_outlier_without_dropping_it():
    statistics = binned_measurement_statistics(
        values=np.array([[0.0], [0.0], [0.0], [100.0]]),
        errors=np.ones((4, 1)),
        valid=np.ones((4, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 0.3, 0.4]),
        redshift_edges=np.array([0.0, 1.0]),
        winsor_tail_percent=25.0,
    )
    assert statistics.count[0, 0] == 4
    assert statistics.mean[0, 0] == 25.0
    np.testing.assert_allclose(statistics.winsorized_mean[0, 0], 6.25)


def test_regularized_weights_fall_back_to_equal_for_zero_population_scatter():
    statistics = binned_measurement_statistics(
        values=np.ones((3, 1)),
        errors=np.array([[0.001], [1.0], [100.0]]),
        valid=np.ones((3, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 0.3]),
        redshift_edges=np.array([0.0, 1.0]),
    )
    assert statistics.regularized_weighted_mean[0, 0] == 1.0
    assert statistics.regularized_effective_count[0, 0] == 3.0
    np.testing.assert_allclose(
        statistics.regularized_maximum_weight_fraction[0, 0], 1.0 / 3.0
    )


def test_redshift_edges_are_uniform_in_log_one_plus_z():
    edges = redshift_bin_edges(6.0, fractional_step=0.05)
    assert edges[-1] >= 6.0
    np.testing.assert_allclose(
        np.diff(edges),
        0.05 * (1.0 + edges[:-1]),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(np.diff(np.log1p(edges)), np.log(1.05))


def test_filter_interval_edges_use_physical_boundaries_and_domain_limits():
    edges = filter_interval_wavelength_edges(
        filter_blue=np.array([2.0, 3.0, 9.0]),
        filter_red=np.array([4.0, 6.0, 12.0]),
        minimum=1.0,
        maximum=10.0,
    )
    np.testing.assert_array_equal(edges, [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 10.0])


def test_filter_interval_cells_average_only_the_physical_overlap():
    wavelength_edges = filter_interval_wavelength_edges(
        filter_blue=np.array([2.0, 3.0]),
        filter_red=np.array([4.0, 6.0]),
        minimum=1.0,
        maximum=8.0,
    )
    stack = stack_filter_seds(
        values=np.array([[2.0, 4.0]]),
        valid=np.ones((1, 2), dtype=bool),
        redshift=np.array([0.2]),
        filter_blue=np.array([2.0, 3.0]),
        filter_red=np.array([4.0, 6.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    np.testing.assert_array_equal(wavelength_edges, [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(stack.mean[0, 1:4], [2.0, 3.0, 4.0])
    assert np.isnan(stack.mean[0, 0])
    assert np.isnan(stack.mean[0, 4])


def test_filter_interval_overlap_counts_galaxies_not_measurements():
    wavelength_edges = filter_interval_wavelength_edges(
        filter_blue=np.array([2.0, 3.0]),
        filter_red=np.array([4.0, 6.0]),
        minimum=1.0,
        maximum=8.0,
    )
    stack = stack_filter_seds(
        values=np.array([[2.0, 4.0], [8.0, 100.0]]),
        valid=np.array([[True, True], [True, False]]),
        redshift=np.array([0.2, 0.3]),
        filter_blue=np.array([2.0, 3.0]),
        filter_red=np.array([4.0, 6.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    # In the shared 3--4 interval, galaxy one contributes (2+4)/2 = 3 and
    # galaxy two contributes 8 once.  The galaxy-level mean is therefore 5.5.
    assert stack.mean[0, 2] == 5.5
    assert stack.count[0, 2] == 2


def test_flat_flam_normalizes_to_one_and_requires_bracketing():
    pivot = np.array([4000.0, 6000.0, 9000.0])
    # F_nu proportional to lambda^2 is flat in F_lambda.
    flux_fnu = np.square(pivot / 5000.0)[None, :].repeat(2, axis=0)
    error_fnu = 0.01 * flux_fnu
    valid = np.ones_like(flux_fnu, dtype=bool)
    redshift = np.array([0.0, 1.0])

    flux_flam = fnu_to_flam_proxy(flux_fnu, pivot)
    error_flam = fnu_to_flam_proxy(error_fnu, pivot)
    result = normalize_at_rest_wavelength(
        flux_flam,
        error_flam,
        valid,
        pivot,
        redshift,
        min_snr=5,
    )

    assert result.selected.tolist() == [True, False]
    np.testing.assert_allclose(result.values[0], 1.0)
    assert np.all(~result.valid[1])


def test_three_band_local_normalization_improves_intercept_snr():
    pivot = np.array([4000.0, 5000.0, 6250.0])
    flux = np.ones((1, 3))
    error = np.ones_like(flux)
    valid = np.ones_like(flux, dtype=bool)
    two = normalize_at_rest_wavelength(
        flux,
        error,
        valid,
        pivot,
        np.array([0.0]),
        min_snr=0,
        n_nearest=2,
    )
    three = normalize_at_rest_wavelength(
        flux,
        error,
        valid,
        pivot,
        np.array([0.0]),
        min_snr=0,
        n_nearest=3,
    )
    assert two.normalization_band_count[0] == 2
    assert three.normalization_band_count[0] == 3
    np.testing.assert_allclose(two.normalization, 1.0)
    np.testing.assert_allclose(three.normalization, 1.0)
    assert three.normalization_error[0] < two.normalization_error[0]


def test_negative_flux_is_retained_after_positive_normalization():
    pivot = np.array([4000.0, 6000.0, 8000.0])
    flux = np.array([[1.0, 1.0, -0.25]])
    error = np.full_like(flux, 0.01)
    result = normalize_at_rest_wavelength(
        flux,
        error,
        np.ones_like(flux, dtype=bool),
        pivot,
        np.array([0.0]),
        min_snr=5,
    )
    assert result.selected[0]
    assert result.valid[0, 2]
    assert result.values[0, 2] < 0


def test_paint_filter_lines_averages_overlap_once_per_galaxy():
    wavelength_edges = np.geomspace(1.0, 8.0, 25)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    values = np.array([[2.0, 4.0]], dtype=float)
    valid = np.ones_like(values, dtype=bool)
    lines = paint_filter_lines(
        values,
        valid,
        filter_blue=np.array([1.5, 2.5]),
        filter_red=np.array([4.0, 6.0]),
        wavelength_edges=wavelength_edges,
    )
    np.testing.assert_allclose(lines[0, (centers >= 1.5) & (centers < 2.5)], 2.0)
    np.testing.assert_allclose(lines[0, (centers >= 2.5) & (centers <= 4.0)], 3.0)
    np.testing.assert_allclose(lines[0, (centers > 4.0) & (centers <= 6.0)], 4.0)
    assert np.all(np.isnan(lines[0, (centers < 1.5) | (centers > 6.0)]))


def test_rest_frame_filter_support_scales_by_one_plus_z():
    wavelength_edges = np.geomspace(1.0, 8.0, 97)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    values = np.ones((2, 1))
    valid = np.ones_like(values, dtype=bool)
    lines = paint_filter_lines(
        values,
        valid,
        filter_blue=np.array([4.0]),
        filter_red=np.array([6.0]),
        wavelength_edges=wavelength_edges,
        redshift=np.array([0.0, 1.0]),
        rest_frame=True,
    )
    support_z0 = np.flatnonzero(np.isfinite(lines[0]))
    support_z1 = np.flatnonzero(np.isfinite(lines[1]))
    assert np.all((centers[support_z0] >= 4.0) & (centers[support_z0] <= 6.0))
    assert np.all((centers[support_z1] >= 2.0) & (centers[support_z1] <= 3.0))


def test_stack_is_nanmean_and_chunk_invariant():
    wavelength_edges = np.geomspace(1.0, 8.0, 65)
    z_edges = np.array([0.0, 1.0, 3.0])
    z = np.array([0.2, 0.4, 1.2])
    values = np.array([[1.0], [3.0], [7.0]])
    valid = np.ones_like(values, dtype=bool)
    kwargs = dict(
        values=values,
        valid=valid,
        redshift=z,
        filter_blue=np.array([2.0]),
        filter_red=np.array([4.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=z_edges,
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    one = stack_filter_seds(**kwargs, chunk_size=1)
    many = stack_filter_seds(**kwargs, chunk_size=10)
    np.testing.assert_allclose(one.mean, many.mean, equal_nan=True)
    np.testing.assert_array_equal(one.count, many.count)
    assert one.galaxies_per_bin.tolist() == [2, 1]
    finite0 = np.isfinite(one.mean[0])
    finite1 = np.isfinite(one.mean[1])
    np.testing.assert_allclose(one.mean[0, finite0], 2.0)
    np.testing.assert_allclose(one.mean[1, finite1], 7.0)


def test_zero_and_negative_sed_pixels_enter_mean_and_valid_count():
    wavelength_edges = np.geomspace(1.0, 8.0, 33)
    stack = stack_filter_seds(
        values=np.array([[-1.0], [0.0], [1.0]]),
        valid=np.ones((3, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 0.3]),
        filter_blue=np.array([2.0]),
        filter_red=np.array([4.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    covered = stack.count[0] > 0
    assert np.all(stack.count[0, covered] == 3)
    np.testing.assert_allclose(stack.mean[0, covered], 0.0)


def test_interval_sweep_matches_explicit_galaxy_rasters():
    rng = np.random.default_rng(913)
    n_source = 37
    values = rng.normal(size=(n_source, 5))
    values[0, 0] = 0.0
    values[1, 1] = -2.0
    values[2, 2] = np.nan
    valid = rng.random(values.shape) > 0.25
    redshift = rng.uniform(0.0, 1.99, n_source)
    redshift_edges = np.array([0.0, 0.5, 1.0, 2.0])
    filter_blue = np.array([0.7, 1.0, 1.6, 2.0, 4.0])
    filter_red = np.array([1.8, 2.4, 3.0, 5.0, 8.0])
    wavelength_edges = np.geomspace(0.3, 9.0, 258)

    for rest_frame in (False, True):
        stack = stack_filter_seds(
            values,
            valid,
            redshift,
            filter_blue,
            filter_red,
            wavelength_edges,
            redshift_edges,
            rest_frame=rest_frame,
            chunk_size=7,
            minimum_count=0,
            minimum_fraction=0.0,
        )
        expected_sum = np.zeros_like(stack.mean, dtype=float)
        expected_count = np.zeros_like(stack.count)
        expected_galaxies = np.zeros(3, dtype=np.int32)
        z_bin = np.searchsorted(redshift_edges, redshift, side="right") - 1
        for bin_index in range(3):
            rows = np.flatnonzero(z_bin == bin_index)
            expected_galaxies[bin_index] = rows.size
            lines = paint_filter_lines(
                values[rows],
                valid[rows],
                filter_blue,
                filter_red,
                wavelength_edges,
                redshift=redshift[rows] if rest_frame else None,
                rest_frame=rest_frame,
            )
            finite = np.isfinite(lines)
            expected_sum[bin_index] = np.nansum(lines, axis=0)
            expected_count[bin_index] = np.sum(finite, axis=0)
        expected_mean = np.full(expected_sum.shape, np.nan)
        np.divide(
            expected_sum,
            expected_count,
            out=expected_mean,
            where=expected_count > 0,
        )
        np.testing.assert_array_equal(stack.count, expected_count)
        np.testing.assert_array_equal(stack.galaxies_per_bin, expected_galaxies)
        np.testing.assert_allclose(
            stack.mean,
            expected_mean,
            rtol=5e-6,
            atol=1e-7,
            equal_nan=True,
        )


def test_hundred_thousand_wavelength_bins_without_source_cube():
    stack = stack_filter_seds(
        values=np.array([[-1.0], [1.0]]),
        valid=np.ones((2, 1), dtype=bool),
        redshift=np.array([0.1, 0.2]),
        filter_blue=np.array([2.0]),
        filter_red=np.array([4.0]),
        wavelength_edges=np.geomspace(1.0, 8.0, 100_001),
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    assert stack.mean.shape == (1, 100_000)
    covered = stack.count[0] > 0
    assert np.all(stack.count[0, covered] == 2)
    np.testing.assert_allclose(stack.mean[0, covered], 0.0)


def test_contributor_threshold_masks_sparse_fringe():
    wavelength_edges = np.geomspace(1.0, 8.0, 33)
    stack = stack_filter_seds(
        values=np.array([[1.0], [2.0], [3.0]]),
        valid=np.ones((3, 1), dtype=bool),
        redshift=np.array([0.1, 0.2, 0.3]),
        filter_blue=np.array([2.0]),
        filter_red=np.array([4.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=4,
        minimum_fraction=0.0,
    )
    assert np.nanmax(stack.count) == 3
    assert np.all(np.isnan(stack.mean))


def test_concatenated_fields_are_galaxy_weighted_and_counts_add():
    wavelength_edges = np.array([1.0, 2.0, 4.0, 8.0])
    common = dict(
        filter_blue=np.array([2.0]),
        filter_red=np.array([4.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    field_a_values = np.array([[1.0], [1.0], [1.0]])
    field_b_values = np.array([[7.0]])
    field_a = stack_filter_seds(
        field_a_values,
        np.ones_like(field_a_values, dtype=bool),
        np.array([0.1, 0.2, 0.3]),
        **common,
    )
    field_b = stack_filter_seds(
        field_b_values,
        np.ones_like(field_b_values, dtype=bool),
        np.array([0.4]),
        **common,
    )
    combined_values = np.concatenate((field_a_values, field_b_values))
    combined = stack_filter_seds(
        combined_values,
        np.ones_like(combined_values, dtype=bool),
        np.array([0.1, 0.2, 0.3, 0.4]),
        **common,
    )
    covered = combined.count[0] > 0
    np.testing.assert_allclose(combined.mean[0, covered], 2.5)
    np.testing.assert_array_equal(
        combined.count, field_a.count + field_b.count
    )


def test_interpolated_stack_is_linear_in_flux_versus_log_wavelength():
    wavelength_edges = np.geomspace(1.0, 4.0, 17)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    pivots = np.array([1.0, 2.0, 4.0])
    values = np.array([[-1.0, 1.0, 3.0]])
    stack = stack_interpolated_seds(
        values=values,
        valid=np.ones_like(values, dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=pivots,
        filter_blue=np.array([0.8, 1.5, 3.0]),
        filter_red=np.array([1.6, 3.0, 4.5]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    covered = (centers >= pivots[0]) & (centers < pivots[-1])
    expected = np.interp(np.log(centers[covered]), np.log(pivots), values[0])
    np.testing.assert_allclose(stack.mean[0, covered], expected, atol=2e-7)
    np.testing.assert_array_equal(stack.count[0, covered], 1)
    assert np.any(stack.mean[0, covered] < 0)
    assert np.any(stack.mean[0, covered] > 0)


def test_interpolation_breaks_at_invalid_band_and_true_filter_gap():
    wavelength_edges = np.geomspace(1.0, 8.0, 49)
    stack = stack_interpolated_seds(
        values=np.array([[1.0, 2.0, 4.0, 8.0]]),
        valid=np.array([[True, True, False, True]]),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([1.0, 2.0, 4.0, 8.0]),
        filter_blue=np.array([0.8, 1.5, 3.5, 7.5]),
        filter_red=np.array([1.6, 2.5, 4.5, 8.5]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    assert np.all(stack.count[0, (centers >= 1.0) & (centers < 2.0)] == 1)
    assert np.all(stack.count[0, (centers > 2.5) & (centers < 7.5)] == 0)
    assert np.all(stack.count[0, (centers >= 7.5) & (centers <= 8.0)] == 1)

    gap = stack_interpolated_seds(
        values=np.array([[1.0, 4.0]]),
        valid=np.ones((1, 2), dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([1.0, 3.0]),
        filter_blue=np.array([0.8, 2.8]),
        filter_red=np.array([1.2, 3.2]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    assert np.all(gap.count[0, (centers > 1.2) & (centers < 2.8)] == 0)
    assert np.all(np.isnan(gap.mean[0, (centers > 1.2) & (centers < 2.8)]))


def test_interpolated_stack_shifts_pivots_to_rest_frame():
    wavelength_edges = np.geomspace(0.5, 4.0, 49)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    stack = stack_interpolated_seds(
        values=np.array([[2.0, 6.0]]),
        valid=np.ones((1, 2), dtype=bool),
        redshift=np.array([1.0]),
        pivot_wavelength=np.array([2.0, 4.0]),
        filter_blue=np.array([1.6, 3.0]),
        filter_red=np.array([3.0, 5.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 2.0]),
        rest_frame=True,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    covered = (centers >= 1.0) & (centers < 2.0)
    expected = np.interp(
        np.log(centers[covered]), np.log([1.0, 2.0]), [2.0, 6.0]
    )
    np.testing.assert_allclose(stack.mean[0, covered], expected, atol=2e-7)
    outside_support = (centers < 0.8) | (centers > 2.5)
    assert np.all(stack.count[0, outside_support] == 0)


def test_combined_interpolated_fields_equal_concatenated_galaxy_stack():
    wavelength_edges = np.geomspace(1.0, 4.0, 17)
    common = dict(
        pivot_wavelength=np.array([1.0, 2.0, 4.0]),
        filter_blue=np.array([0.8, 1.5, 3.0]),
        filter_red=np.array([1.6, 3.0, 4.5]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=0,
        minimum_fraction=0.0,
    )
    field_a_values = np.array([[1.0, 2.0, 4.0]] * 3)
    field_b_values = np.array([[7.0, 8.0, 10.0]])
    field_a = stack_interpolated_seds(
        field_a_values,
        np.ones_like(field_a_values, dtype=bool),
        np.array([0.1, 0.2, 0.3]),
        **common,
    )
    field_b = stack_interpolated_seds(
        field_b_values,
        np.ones_like(field_b_values, dtype=bool),
        np.array([0.4]),
        **common,
    )
    combined = combine_sed_stacks(
        [field_a, field_b], minimum_count=1, minimum_fraction=0.0
    )
    concatenated_values = np.concatenate((field_a_values, field_b_values))
    direct = stack_interpolated_seds(
        concatenated_values,
        np.ones_like(concatenated_values, dtype=bool),
        np.array([0.1, 0.2, 0.3, 0.4]),
        **{**common, "minimum_count": 1},
    )
    np.testing.assert_allclose(combined.mean, direct.mean, equal_nan=True, atol=2e-7)
    np.testing.assert_array_equal(combined.count, direct.count)
    np.testing.assert_array_equal(
        combined.galaxies_per_bin, direct.galaxies_per_bin
    )


def test_interpolated_single_band_component_uses_exact_filter_support():
    wavelength_edges = np.geomspace(1.0, 8.0, 97)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    stack = stack_interpolated_seds(
        values=np.array([[0.0]]),
        valid=np.ones((1, 1), dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([3.0]),
        filter_blue=np.array([2.0]),
        filter_red=np.array([5.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    covered = (centers >= 2.0) & (centers <= 5.0)
    np.testing.assert_array_equal(stack.count[0, covered], 1)
    np.testing.assert_allclose(stack.mean[0, covered], 0.0)
    np.testing.assert_array_equal(stack.count[0, ~covered], 0)


def test_nearly_coincident_pivots_are_averaged_once_per_galaxy():
    wavelength_edges = np.geomspace(1.0, 4.0, 33)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    stack = stack_interpolated_seds(
        values=np.array([[2.0, 4.0, 9.0]]),
        valid=np.ones((1, 3), dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([1.99, 2.01, 3.0]),
        filter_blue=np.array([1.5, 1.6, 2.4]),
        filter_red=np.array([2.5, 2.6, 3.5]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        coincident_fraction=0.05,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    near_merged_pivot = np.argmin(np.abs(centers - 2.0))
    assert stack.count[0, near_merged_pivot] == 1
    assert abs(stack.mean[0, near_merged_pivot] - 3.0) < 0.2


def test_nested_wide_filter_keeps_coverage_component_connected():
    wavelength_edges = np.geomspace(1.0, 6.0, 97)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    stack = stack_interpolated_seds(
        values=np.array([[1.0, 2.0, 3.0]]),
        valid=np.ones((1, 3), dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([2.0, 3.0, 4.0]),
        filter_blue=np.array([1.0, 2.5, 3.8]),
        filter_red=np.array([4.2, 3.5, 5.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    bridge = (centers >= 3.5) & (centers < 4.0)
    np.testing.assert_array_equal(stack.count[0, bridge], 1)
    assert np.all(np.isfinite(stack.mean[0, bridge]))


def test_nested_filter_sets_full_component_outer_support():
    wavelength_edges = np.geomspace(0.8, 5.0, 129)
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    common = dict(
        values=np.array([[2.0, 4.0]]),
        valid=np.ones((1, 2), dtype=bool),
        redshift=np.array([0.2]),
        pivot_wavelength=np.array([2.0, 3.0]),
        wavelength_edges=wavelength_edges,
        redshift_edges=np.array([0.0, 1.0]),
        rest_frame=False,
        minimum_count=1,
        minimum_fraction=0.0,
    )
    wide_right = stack_interpolated_seds(
        **common,
        filter_blue=np.array([1.8, 1.0]),
        filter_red=np.array([2.2, 4.0]),
    )
    np.testing.assert_array_equal(
        wide_right.count[0, (centers >= 1.0) & (centers <= 4.0)], 1
    )
    wide_left = stack_interpolated_seds(
        **common,
        filter_blue=np.array([1.0, 2.8]),
        filter_red=np.array([4.0, 3.2]),
    )
    np.testing.assert_array_equal(
        wide_left.count[0, (centers >= 1.0) & (centers <= 4.0)], 1
    )
