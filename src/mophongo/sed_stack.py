"""Broadband SED rasterization and redshift stacking.

The helpers in this module turn one broadband SED per galaxy into a
single-pixel-high wavelength raster.  A filter measurement is represented by
a top hat over that filter's half-maximum wavelength interval.  Pixels not
covered by a valid measurement stay NaN, and overlapping filters are averaged
within a galaxy before galaxies are averaged in redshift bins.

The routines operate on plain NumPy arrays.  Survey-specific catalog loading,
quality flags, filter metadata, plotting, and file output belong in callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SEDNormalization:
    """SEDs normalized at a common rest wavelength.

    Attributes:
        values: Normalized flux-density array with shape ``(n_source, n_band)``.
        valid: Valid-measurement mask with the same shape as ``values``.
        normalization: Interpolated normalization flux density per source.
        normalization_error: Propagated interpolation error per source.
        normalization_band_count: Number of local bands used for each source.
        selected: Sources with a bracketed, positive, sufficiently significant
            normalization.
    """

    values: np.ndarray
    valid: np.ndarray
    normalization: np.ndarray
    normalization_error: np.ndarray
    normalization_band_count: np.ndarray
    selected: np.ndarray


@dataclass(frozen=True)
class SEDStack:
    """A wavelength-by-redshift stack and its sampling metadata.

    Attributes:
        mean: Mean normalized flux density, shape ``(n_redshift, n_wavelength)``.
        count: Number of contributing galaxies at every stack pixel.
        wavelength_edges: Wavelength-bin edges in the caller's units.
        redshift_edges: Redshift-bin edges.
        galaxies_per_bin: Number of selected galaxies assigned to each
            redshift bin before wavelength-coverage masking.
    """

    mean: np.ndarray
    count: np.ndarray
    wavelength_edges: np.ndarray
    redshift_edges: np.ndarray
    galaxies_per_bin: np.ndarray


def fnu_to_flam_proxy(
    flux_fnu: np.ndarray,
    pivot_wavelength: np.ndarray,
    reference_wavelength: float = 5000.0,
) -> np.ndarray:
    """Convert ``F_nu`` to a quantity proportional to ``F_lambda``.

    The physical factor of the speed of light cancels when each galaxy is
    normalized.  Scaling by ``reference_wavelength**2`` keeps the numerical
    values close to the input flux-density scale.

    Args:
        flux_fnu: Flux density with bands along the last axis.
        pivot_wavelength: Positive pivot wavelength for every band.
        reference_wavelength: Arbitrary positive wavelength used only for
            numerical scaling, in the same units as ``pivot_wavelength``.

    Returns:
        An array proportional to ``F_lambda`` with the same shape as
        ``flux_fnu``.

    Raises:
        ValueError: Shapes are inconsistent or a wavelength is non-positive.
    """

    flux = np.asarray(flux_fnu)
    pivot = np.asarray(pivot_wavelength, dtype=float)
    if flux.shape[-1] != pivot.size:
        raise ValueError(
            "the last flux axis must match pivot_wavelength: "
            f"{flux.shape[-1]} != {pivot.size}"
        )
    if reference_wavelength <= 0 or np.any(~np.isfinite(pivot)) or np.any(pivot <= 0):
        raise ValueError("wavelengths must be finite and positive")
    scale = np.square(reference_wavelength / pivot)
    return np.asarray(flux * scale, dtype=np.result_type(flux.dtype, np.float32))


def normalize_at_rest_wavelength(
    flux_flam: np.ndarray,
    error_flam: np.ndarray,
    valid: np.ndarray,
    pivot_wavelength: np.ndarray,
    redshift: np.ndarray,
    rest_wavelength: float = 5000.0,
    min_snr: float = 5.0,
    n_nearest: int = 3,
) -> SEDNormalization:
    """Normalize SEDs by a local fit at a common rest wavelength.

    For each source, the nearest valid filters around
    ``rest_wavelength * (1 + redshift)`` are fit with an inverse-variance
    weighted straight line in ``F_lambda`` versus log-wavelength.  The
    intercept at the requested wavelength is the normalization.  The fit
    always contains the nearest bracketing filters and uses up to
    ``n_nearest`` bands, so two bands reduce exactly to linear interpolation
    while a third band improves the local signal-to-noise.  The function
    never extrapolates.  Negative flux measurements remain valid SED samples,
    but the fitted normalization itself must be positive.

    Args:
        flux_flam: Quantity proportional to ``F_lambda``, shape
            ``(n_source, n_band)``.
        error_flam: One-sigma errors in the same scale and shape.
        valid: Boolean measurement mask.  Validity is further restricted to
            finite fluxes and finite, positive errors.
        pivot_wavelength: Strictly increasing band pivots.
        redshift: Source redshifts, shape ``(n_source,)``.
        rest_wavelength: Rest wavelength of the normalization.
        min_snr: Minimum interpolated normalization signal-to-noise.  Set to
            zero to require only a positive normalization.
        n_nearest: Maximum number of nearest local bands in the fit.  Must be
            at least two.

    Returns:
        Normalized values, masks, and normalization diagnostics.

    Raises:
        ValueError: Input shapes, wavelengths, redshifts, or ordering are
            invalid.
    """

    flux = np.asarray(flux_flam)
    error = np.asarray(error_flam)
    mask = np.asarray(valid, dtype=bool).copy()
    pivot = np.asarray(pivot_wavelength, dtype=float)
    z = np.asarray(redshift, dtype=float)

    if flux.ndim != 2:
        raise ValueError("flux_flam must be a two-dimensional array")
    if error.shape != flux.shape or mask.shape != flux.shape:
        raise ValueError("flux, error, and valid arrays must have identical shapes")
    if pivot.shape != (flux.shape[1],):
        raise ValueError("pivot_wavelength must have one value per band")
    if z.shape != (flux.shape[0],):
        raise ValueError("redshift must have one value per source")
    if rest_wavelength <= 0 or np.any(~np.isfinite(pivot)) or np.any(pivot <= 0):
        raise ValueError("wavelengths must be finite and positive")
    if np.any(np.diff(pivot) <= 0):
        raise ValueError("pivot_wavelength must be strictly increasing")
    if np.any(~np.isfinite(z)) or np.any(z <= -1):
        raise ValueError("redshifts must be finite and greater than -1")
    if min_snr < 0:
        raise ValueError("min_snr must be non-negative")
    if not isinstance(n_nearest, (int, np.integer)) or n_nearest < 2:
        raise ValueError("n_nearest must be an integer of at least two")

    mask &= np.isfinite(flux) & np.isfinite(error) & (error > 0)
    target = rest_wavelength * (1.0 + z)
    band_index = np.arange(pivot.size)

    below = mask & (pivot[None, :] <= target[:, None])
    above = mask & (pivot[None, :] >= target[:, None])
    lower = np.max(np.where(below, band_index[None, :], -1), axis=1)
    upper = np.min(
        np.where(above, band_index[None, :], pivot.size), axis=1
    )
    bracketed = (lower >= 0) & (upper < pivot.size)

    chosen = np.zeros(mask.shape, dtype=bool)
    rows = np.flatnonzero(bracketed)
    if rows.size:
        chosen[rows, lower[rows]] = True
        chosen[rows, upper[rows]] = True

    available_count = np.sum(mask, axis=1, dtype=np.int16)
    target_count = np.minimum(available_count, n_nearest)
    distance = np.abs(np.log(pivot)[None, :] - np.log(target)[:, None])
    distance = np.where(mask & ~chosen, distance, np.inf)
    for _ in range(n_nearest):
        chosen_count = np.sum(chosen, axis=1, dtype=np.int16)
        needs_band = bracketed & (chosen_count < target_count)
        if not np.any(needs_band):
            break
        extra = np.argmin(distance, axis=1)
        extra_distance = distance[np.arange(distance.shape[0]), extra]
        add = needs_band & np.isfinite(extra_distance)
        add_rows = np.flatnonzero(add)
        chosen[add_rows, extra[add_rows]] = True
        distance[add_rows, extra[add_rows]] = np.inf

    sum_w = np.zeros(flux.shape[0], dtype=float)
    sum_wx = np.zeros_like(sum_w)
    sum_wxx = np.zeros_like(sum_w)
    sum_wy = np.zeros_like(sum_w)
    sum_wxy = np.zeros_like(sum_w)
    for band in range(pivot.size):
        band_rows = np.flatnonzero(chosen[:, band])
        if not band_rows.size:
            continue
        x = np.log(pivot[band] / target[band_rows])
        weight = np.square(1.0 / error[band_rows, band])
        y = flux[band_rows, band]
        sum_w[band_rows] += weight
        sum_wx[band_rows] += weight * x
        sum_wxx[band_rows] += weight * x * x
        sum_wy[band_rows] += weight * y
        sum_wxy[band_rows] += weight * x * y

    determinant = sum_w * sum_wxx - np.square(sum_wx)
    fit_ok = bracketed & np.isfinite(determinant) & (determinant > 0)
    normalization = np.full(flux.shape[0], np.nan, dtype=float)
    normalization_error = np.full(flux.shape[0], np.nan, dtype=float)
    normalization[fit_ok] = (
        sum_wxx[fit_ok] * sum_wy[fit_ok]
        - sum_wx[fit_ok] * sum_wxy[fit_ok]
    ) / determinant[fit_ok]
    normalization_error[fit_ok] = np.sqrt(
        sum_wxx[fit_ok] / determinant[fit_ok]
    )
    normalization_band_count = np.sum(chosen, axis=1, dtype=np.int16)

    snr = np.divide(
        normalization,
        normalization_error,
        out=np.full_like(normalization, np.nan),
        where=normalization_error > 0,
    )
    selected = (
        bracketed
        & np.isfinite(normalization)
        & np.isfinite(normalization_error)
        & (normalization > 0)
        & (normalization_error > 0)
        & (snr >= min_snr)
    )

    values = np.full(flux.shape, np.nan, dtype=np.float32)
    if np.any(selected):
        values[selected] = np.asarray(
            flux[selected] / normalization[selected, None], dtype=np.float32
        )
    normalized_valid = mask & selected[:, None] & np.isfinite(values)
    return SEDNormalization(
        values=values,
        valid=normalized_valid,
        normalization=normalization,
        normalization_error=normalization_error,
        normalization_band_count=normalization_band_count,
        selected=selected,
    )


def redshift_bin_edges(
    z_max: float,
    fractional_step: float = 0.05,
    z_min: float = 0.0,
) -> np.ndarray:
    """Return bins uniform in log(1 + z).

    Adjacent edges obey
    ``z[k + 1] - z[k] = fractional_step * (1 + z[k])``.  The final edge is
    the first one greater than or equal to ``z_max``.

    Args:
        z_max: Requested upper redshift limit.
        fractional_step: Fractional bin width in ``1 + z``.
        z_min: Lower redshift edge.

    Returns:
        Monotonically increasing redshift edges.

    Raises:
        ValueError: Limits or step are invalid.
    """

    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min < 0 or z_max <= z_min:
        raise ValueError("require finite redshifts with 0 <= z_min < z_max")
    if not np.isfinite(fractional_step) or fractional_step <= 0:
        raise ValueError("fractional_step must be finite and positive")
    delta = np.log1p(fractional_step)
    span = np.log1p(z_max) - np.log1p(z_min)
    n_bin = int(np.ceil(span / delta))
    log_edges = np.log1p(z_min) + delta * np.arange(n_bin + 1)
    return np.expm1(log_edges)


def filter_interval_wavelength_edges(
    filter_blue: np.ndarray,
    filter_red: np.ndarray,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Partition wavelength at the exact boundaries of filter intervals.

    The returned bins are the maximal wavelength intervals over which the set
    of active filters is constant.  Thus an isolated filter is represented by
    one cell spanning its full half-maximum width.  Overlapping filters are
    split only at their physical boundaries, which allows their fluxes to be
    averaged exactly in the shared portion without imposing an arbitrary
    wavelength sampling.

    Args:
        filter_blue: Blue half-maximum edge for every filter.
        filter_red: Red half-maximum edge for every filter.
        minimum: Optional lower wavelength limit.  Defaults to the bluest edge.
        maximum: Optional upper wavelength limit.  Defaults to the reddest edge.

    Returns:
        Strictly increasing wavelength-bin edges.

    Raises:
        ValueError: Filter intervals or optional limits are invalid.
    """

    blue = np.asarray(filter_blue, dtype=float)
    red = np.asarray(filter_red, dtype=float)
    if blue.ndim != 1 or blue.size == 0 or red.shape != blue.shape:
        raise ValueError(
            "filter edges must be matching non-empty one-dimensional arrays"
        )
    if (
        np.any(~np.isfinite(blue))
        or np.any(~np.isfinite(red))
        or np.any(blue <= 0)
        or np.any(red <= blue)
    ):
        raise ValueError("filter intervals must be finite, positive, and ordered")

    lower = float(np.min(blue)) if minimum is None else float(minimum)
    upper = float(np.max(red)) if maximum is None else float(maximum)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0
        or upper <= lower
    ):
        raise ValueError("wavelength limits must be finite, positive, and ordered")

    intersects = (red > lower) & (blue < upper)
    boundaries = [np.array([lower, upper], dtype=float)]
    if np.any(intersects):
        boundaries.extend(
            (
                np.clip(blue[intersects], lower, upper),
                np.clip(red[intersects], lower, upper),
            )
        )
    return np.unique(np.concatenate(boundaries))


def paint_filter_lines(
    values: np.ndarray,
    valid: np.ndarray,
    filter_blue: np.ndarray,
    filter_red: np.ndarray,
    wavelength_edges: np.ndarray,
    redshift: np.ndarray | None = None,
    rest_frame: bool = False,
) -> np.ndarray:
    """Paint broadband measurements into one horizontal raster per source.

    A measurement is constant between its blue and red filter edges.  Pixels
    outside all valid filters are NaN.  If valid filters overlap, their values
    are averaged within the source so that the source still contributes only
    once when lines are stacked.

    Args:
        values: Flux values, shape ``(n_source, n_band)``.
        valid: Boolean mask with the same shape.
        filter_blue: Blue half-maximum edge for every filter.
        filter_red: Red half-maximum edge for every filter.
        wavelength_edges: Positive raster-bin edges.
        redshift: Redshift per source, required for a rest-frame raster.
        rest_frame: Divide every filter edge by ``1 + redshift``.

    Returns:
        A float32 array of shape ``(n_source, n_wavelength)``.

    Raises:
        ValueError: Shapes, intervals, or wavelength edges are invalid.
    """

    data = np.asarray(values)
    mask = np.asarray(valid, dtype=bool).copy()
    blue = np.asarray(filter_blue, dtype=float)
    red = np.asarray(filter_red, dtype=float)
    edges = np.asarray(wavelength_edges, dtype=float)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("values and valid must be matching two-dimensional arrays")
    if blue.shape != (data.shape[1],) or red.shape != blue.shape:
        raise ValueError("filter edges must have one value per band")
    if np.any(~np.isfinite(blue)) or np.any(~np.isfinite(red)) or np.any(blue <= 0):
        raise ValueError("filter edges must be finite and positive")
    if np.any(red <= blue):
        raise ValueError("every red filter edge must exceed its blue edge")
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)):
        raise ValueError("wavelength_edges must be a finite one-dimensional array")
    if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("wavelength_edges must be positive and strictly increasing")

    n_source, n_band = data.shape
    if rest_frame:
        if redshift is None:
            raise ValueError("redshift is required when rest_frame=True")
        z = np.asarray(redshift, dtype=float)
        if z.shape != (n_source,) or np.any(~np.isfinite(z)) or np.any(z <= -1):
            raise ValueError("redshift must be finite, greater than -1, and match sources")
        divisor = 1.0 + z[:, None]
        blue_2d = blue[None, :] / divisor
        red_2d = red[None, :] / divisor
    else:
        blue_2d = np.broadcast_to(blue, (n_source, n_band))
        red_2d = np.broadcast_to(red, (n_source, n_band))

    centers = np.sqrt(edges[:-1] * edges[1:])
    start = np.searchsorted(centers, blue_2d, side="left")
    stop = np.searchsorted(centers, red_2d, side="right")
    mask &= np.isfinite(data) & (stop > start)

    signal_diff = np.zeros((n_source, centers.size + 1), dtype=np.float32)
    overlap_diff = np.zeros((n_source, centers.size + 1), dtype=np.int16)
    for band in range(n_band):
        rows = np.flatnonzero(mask[:, band])
        if not rows.size:
            continue
        i0 = start[rows, band]
        i1 = stop[rows, band]
        band_values = np.asarray(data[rows, band], dtype=np.float32)
        np.add.at(signal_diff, (rows, i0), band_values)
        np.add.at(signal_diff, (rows, i1), -band_values)
        np.add.at(overlap_diff, (rows, i0), 1)
        np.add.at(overlap_diff, (rows, i1), -1)

    signal = np.cumsum(signal_diff[:, :-1], axis=1, dtype=np.float32)
    overlap = np.cumsum(overlap_diff[:, :-1], axis=1, dtype=np.int16)
    lines = np.full(signal.shape, np.nan, dtype=np.float32)
    np.divide(signal, overlap, out=lines, where=overlap > 0)
    return lines


def stack_filter_seds(
    values: np.ndarray,
    valid: np.ndarray,
    redshift: np.ndarray,
    filter_blue: np.ndarray,
    filter_red: np.ndarray,
    wavelength_edges: np.ndarray,
    redshift_edges: np.ndarray,
    *,
    rest_frame: bool,
    chunk_size: int = 1024,
    minimum_count: int = 5,
    minimum_fraction: float = 0.01,
) -> SEDStack:
    """Nanmean normalized SED rasters in redshift bins.

    The aggregation sweeps the filter start/stop events of each galaxy and
    accumulates exact range differences directly into the stack.  The full
    ``source x wavelength`` cube is never allocated, so very finely sampled
    wavelength grids remain practical.  Means are unweighted per galaxy;
    measurement errors are used for validity and normalization selection, not
    as inverse-variance weights.  Every finite valid value contributes,
    including zero and negative flux densities.

    Args:
        values: Normalized ``F_lambda`` values, shape
            ``(n_source, n_band)``.
        valid: Valid-measurement mask with the same shape.
        redshift: Source redshifts.
        filter_blue: Blue half-maximum filter edges.
        filter_red: Red half-maximum filter edges.
        wavelength_edges: Raster wavelength-bin edges.
        redshift_edges: Stack redshift-bin edges.
        rest_frame: Whether to shift filter intervals to rest wavelength.
        chunk_size: Maximum galaxies whose filter events are swept at once.
        minimum_count: Absolute minimum contributors for a retained pixel.
        minimum_fraction: Also require this fraction of the galaxies assigned
            to that redshift bin.

    Returns:
        Mean stack, contribution counts, and bin metadata.

    Raises:
        ValueError: Inputs or thresholds are invalid.
    """

    data = np.asarray(values)
    mask = np.asarray(valid, dtype=bool).copy()
    z = np.asarray(redshift, dtype=float)
    z_edges = np.asarray(redshift_edges, dtype=float)
    wave_edges = np.asarray(wavelength_edges, dtype=float)
    blue = np.asarray(filter_blue, dtype=float)
    red = np.asarray(filter_red, dtype=float)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("values and valid must be matching two-dimensional arrays")
    if z.shape != (data.shape[0],):
        raise ValueError("redshift must have one value per source")
    if np.any(~np.isfinite(z)) or np.any(z <= -1):
        raise ValueError("redshift must be finite and greater than -1")
    if blue.shape != (data.shape[1],) or red.shape != blue.shape:
        raise ValueError("filter edges must have one value per band")
    if (
        np.any(~np.isfinite(blue))
        or np.any(~np.isfinite(red))
        or np.any(blue <= 0)
        or np.any(red <= blue)
    ):
        raise ValueError("filter intervals must be finite, positive, and ordered")
    if (
        wave_edges.ndim != 1
        or wave_edges.size < 2
        or np.any(~np.isfinite(wave_edges))
        or np.any(wave_edges <= 0)
        or np.any(np.diff(wave_edges) <= 0)
    ):
        raise ValueError("wavelength_edges must be positive and strictly increasing")
    if (
        z_edges.ndim != 1
        or z_edges.size < 2
        or np.any(~np.isfinite(z_edges))
        or np.any(np.diff(z_edges) <= 0)
    ):
        raise ValueError("redshift_edges must be strictly increasing")
    if chunk_size <= 0 or minimum_count < 0 or not (0 <= minimum_fraction <= 1):
        raise ValueError("invalid chunk or contributor threshold")

    n_redshift = z_edges.size - 1
    n_wavelength = wave_edges.size - 1
    sum_diff = np.zeros((n_redshift, n_wavelength + 1), dtype=np.float64)
    count_diff = np.zeros((n_redshift, n_wavelength + 1), dtype=np.int32)
    z_bin = np.searchsorted(z_edges, z, side="right") - 1
    # Include a source exactly on the final edge in the final bin.
    z_bin[z == z_edges[-1]] = n_redshift - 1
    in_redshift_range = (z_bin >= 0) & (z_bin < n_redshift)
    galaxies_per_bin = np.bincount(
        z_bin[in_redshift_range], minlength=n_redshift
    ).astype(np.int32)
    members = np.flatnonzero(in_redshift_range)
    centers = np.sqrt(wave_edges[:-1] * wave_edges[1:])
    for offset in range(0, members.size, chunk_size):
        rows = members[offset : offset + chunk_size]
        if rest_frame:
            divisor = 1.0 + z[rows, None]
            blue_2d = blue[None, :] / divisor
            red_2d = red[None, :] / divisor
        else:
            blue_2d = np.broadcast_to(blue, (rows.size, blue.size))
            red_2d = np.broadcast_to(red, (rows.size, red.size))

        start = np.searchsorted(centers, blue_2d, side="left")
        stop = np.searchsorted(centers, red_2d, side="right")
        interval_valid = (
            mask[rows]
            & np.isfinite(data[rows])
            & (stop > start)
        )
        start = np.where(interval_valid, start, 0)
        stop = np.where(interval_valid, stop, 0)
        signal = np.where(interval_valid, data[rows], 0.0)

        event_position = np.concatenate((start, stop), axis=1)
        event_signal = np.concatenate((signal, -signal), axis=1)
        event_overlap = np.concatenate(
            (interval_valid.astype(np.int16), -interval_valid.astype(np.int16)),
            axis=1,
        )
        order = np.argsort(event_position, axis=1, kind="stable")
        event_position = np.take_along_axis(event_position, order, axis=1)
        event_signal = np.take_along_axis(event_signal, order, axis=1)
        event_overlap = np.take_along_axis(event_overlap, order, axis=1)

        active_signal = np.cumsum(event_signal, axis=1, dtype=np.float64)
        active_overlap = np.cumsum(event_overlap, axis=1, dtype=np.int16)
        segment_start = event_position[:, :-1]
        segment_stop = event_position[:, 1:]
        segment_overlap = active_overlap[:, :-1]
        segment_valid = (segment_stop > segment_start) & (segment_overlap > 0)
        segment_value = np.zeros(segment_start.shape, dtype=np.float64)
        np.divide(
            active_signal[:, :-1],
            segment_overlap,
            out=segment_value,
            where=segment_valid,
        )

        stack_row = np.broadcast_to(z_bin[rows, None], segment_start.shape)
        rr = stack_row[segment_valid]
        i0 = segment_start[segment_valid]
        i1 = segment_stop[segment_valid]
        vv = segment_value[segment_valid]
        np.add.at(sum_diff, (rr, i0), vv)
        np.add.at(sum_diff, (rr, i1), -vv)
        np.add.at(count_diff, (rr, i0), 1)
        np.add.at(count_diff, (rr, i1), -1)

    sums = np.cumsum(sum_diff[:, :-1], axis=1, dtype=np.float64)
    counts = np.cumsum(count_diff[:, :-1], axis=1, dtype=np.int32)

    mean = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    for bin_index, n_galaxy in enumerate(galaxies_per_bin):
        required = max(minimum_count, int(np.ceil(minimum_fraction * n_galaxy)))
        mean[bin_index, counts[bin_index] < required] = np.nan

    return SEDStack(
        mean=mean,
        count=counts,
        wavelength_edges=wave_edges,
        redshift_edges=z_edges,
        galaxies_per_bin=galaxies_per_bin,
    )


def stack_interpolated_seds(
    values: np.ndarray,
    valid: np.ndarray,
    redshift: np.ndarray,
    pivot_wavelength: np.ndarray,
    filter_blue: np.ndarray,
    filter_red: np.ndarray,
    wavelength_edges: np.ndarray,
    redshift_edges: np.ndarray,
    *,
    rest_frame: bool,
    coincident_fraction: float = 0.05,
    minimum_count: int = 5,
    minimum_fraction: float = 0.01,
) -> SEDStack:
    """Stack piecewise-linear broadband SEDs in redshift bins.

    Every galaxy is interpolated in linear flux versus log wavelength only
    between adjacent valid knots whose half-maximum filter intervals overlap.
    Real gaps in that galaxy's valid filter coverage therefore remain NaN.
    Each connected component is held constant from its first/last pivot to
    its physical half-maximum boundary; no sloped extrapolation is performed.
    The interpolation is evaluated analytically with range-difference arrays,
    so the full ``source x wavelength`` raster is never allocated.

    The resulting estimand is the same equal-galaxy mean used by
    :func:`stack_filter_seds`: every galaxy contributes at most once to a
    wavelength cell, regardless of how many bands exist elsewhere in its SED.
    Finite zero and negative fluxes remain valid samples.

    Args:
        values: Normalized ``F_lambda`` values, shape
            ``(n_source, n_band)``. Bands must be in increasing pivot order.
        valid: Valid-measurement mask with the same shape.
        redshift: Source redshifts.
        pivot_wavelength: Strictly increasing band pivots.
        filter_blue: Blue half-maximum edge for every band.
        filter_red: Red half-maximum edge for every band.
        wavelength_edges: Output wavelength-bin edges.
        redshift_edges: Stack redshift-bin edges.
        rest_frame: Whether to divide pivots by ``1 + redshift`` before
            interpolation.
        coincident_fraction: Merge consecutive pivots separated by less than
            this fraction of the narrower filter's half-maximum width.  Their
            valid values are arithmetic-averaged within each galaxy, avoiding
            arbitrary sharp transitions between effectively duplicate knots.
        minimum_count: Absolute minimum contributors for a retained pixel.
        minimum_fraction: Also require this fraction of the galaxies assigned
            to that redshift bin.

    Returns:
        Mean interpolated stack, contribution counts, and bin metadata.

    Raises:
        ValueError: Inputs, ordering, or contributor thresholds are invalid.
    """

    data = np.asarray(values)
    mask = np.asarray(valid, dtype=bool).copy()
    z = np.asarray(redshift, dtype=float)
    pivot = np.asarray(pivot_wavelength, dtype=float)
    blue = np.asarray(filter_blue, dtype=float)
    red = np.asarray(filter_red, dtype=float)
    wave_edges = np.asarray(wavelength_edges, dtype=float)
    z_edges = np.asarray(redshift_edges, dtype=float)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("values and valid must be matching two-dimensional arrays")
    if z.shape != (data.shape[0],):
        raise ValueError("redshift must have one value per source")
    if np.any(~np.isfinite(z)) or np.any(z <= -1):
        raise ValueError("redshift must be finite and greater than -1")
    if pivot.shape != (data.shape[1],):
        raise ValueError("pivot_wavelength must have one value per band")
    if (
        np.any(~np.isfinite(pivot))
        or np.any(pivot <= 0)
        or np.any(np.diff(pivot) <= 0)
    ):
        raise ValueError("pivot_wavelength must be finite, positive, and increasing")
    if blue.shape != pivot.shape or red.shape != pivot.shape:
        raise ValueError("filter edges must have one value per band")
    if (
        np.any(~np.isfinite(blue))
        or np.any(~np.isfinite(red))
        or np.any(blue <= 0)
        or np.any(red <= blue)
    ):
        raise ValueError("filter intervals must be finite, positive, and ordered")
    if (
        wave_edges.ndim != 1
        or wave_edges.size < 2
        or np.any(~np.isfinite(wave_edges))
        or np.any(wave_edges <= 0)
        or np.any(np.diff(wave_edges) <= 0)
    ):
        raise ValueError("wavelength_edges must be positive and strictly increasing")
    if (
        z_edges.ndim != 1
        or z_edges.size < 2
        or np.any(~np.isfinite(z_edges))
        or np.any(np.diff(z_edges) <= 0)
    ):
        raise ValueError("redshift_edges must be strictly increasing")
    if not np.isfinite(coincident_fraction) or coincident_fraction < 0:
        raise ValueError("coincident_fraction must be finite and non-negative")
    if minimum_count < 0 or not (0 <= minimum_fraction <= 1):
        raise ValueError("invalid contributor threshold")

    mask &= np.isfinite(data)
    filter_width = red - blue
    group_starts = [0]
    for band in range(1, pivot.size):
        separation = pivot[band] - pivot[band - 1]
        tolerance = coincident_fraction * min(
            filter_width[band - 1], filter_width[band]
        )
        if separation > tolerance:
            group_starts.append(band)
    group_stops = group_starts[1:] + [pivot.size]
    n_group = len(group_starts)
    grouped_values = np.full((data.shape[0], n_group), np.nan, dtype=np.float32)
    grouped_valid = np.zeros((data.shape[0], n_group), dtype=bool)
    grouped_pivot = np.full((data.shape[0], n_group), np.nan, dtype=np.float64)
    grouped_blue = np.full_like(grouped_pivot, np.nan)
    grouped_red = np.full_like(grouped_pivot, np.nan)
    grouped_index = np.full((data.shape[0], n_group), -1, dtype=np.int16)
    for group, (start_band, stop_band) in enumerate(zip(group_starts, group_stops)):
        local_valid = mask[:, start_band:stop_band]
        local_count = np.sum(local_valid, axis=1, dtype=np.int16)
        rows = np.flatnonzero(local_count > 0)
        if not rows.size:
            continue
        local_values = np.where(
            local_valid[rows], data[rows, start_band:stop_band], 0.0
        )
        grouped_values[rows, group] = np.asarray(
            np.sum(local_values, axis=1, dtype=np.float64) / local_count[rows],
            dtype=np.float32,
        )
        grouped_pivot[rows, group] = np.sum(
            local_valid[rows] * pivot[None, start_band:stop_band], axis=1
        ) / local_count[rows]
        grouped_blue[rows, group] = np.min(
            np.where(
                local_valid[rows], blue[None, start_band:stop_band], np.inf
            ),
            axis=1,
        )
        grouped_red[rows, group] = np.max(
            np.where(
                local_valid[rows], red[None, start_band:stop_band], -np.inf
            ),
            axis=1,
        )
        local_band_indices = np.arange(start_band, stop_band, dtype=np.int16)
        grouped_index[rows, group] = np.min(
            np.where(local_valid[rows], local_band_indices[None, :], pivot.size),
            axis=1,
        )
        grouped_valid[rows, group] = True

    n_redshift = z_edges.size - 1
    n_wavelength = wave_edges.size - 1
    intercept_diff = np.zeros((n_redshift, n_wavelength + 1), dtype=np.float64)
    slope_diff = np.zeros_like(intercept_diff)
    count_diff = np.zeros((n_redshift, n_wavelength + 1), dtype=np.int32)
    z_bin = np.searchsorted(z_edges, z, side="right") - 1
    z_bin[z == z_edges[-1]] = n_redshift - 1
    in_redshift_range = (z_bin >= 0) & (z_bin < n_redshift)
    galaxies_per_bin = np.bincount(
        z_bin[in_redshift_range], minlength=n_redshift
    ).astype(np.int32)
    log_centers = np.log(np.sqrt(wave_edges[:-1] * wave_edges[1:]))

    def add_segments(
        rows: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        intercept: np.ndarray,
        slope: np.ndarray,
        *,
        include_right: bool,
    ) -> None:
        """Add affine-in-log-wavelength segments to the stack events."""

        if not rows.size:
            return
        if rest_frame:
            divisor = 1.0 + z[rows]
            left = left / divisor
            right = right / divisor
            intercept = intercept + slope * np.log(divisor)
        start = np.searchsorted(log_centers, np.log(left), side="left")
        stop = np.searchsorted(
            log_centers,
            np.log(right),
            side="right" if include_right else "left",
        )
        contributes = stop > start
        if not np.any(contributes):
            return
        rows = rows[contributes]
        start = start[contributes]
        stop = stop[contributes]
        intercept = intercept[contributes]
        slope = slope[contributes]
        stack_row = z_bin[rows]
        np.add.at(intercept_diff, (stack_row, start), intercept)
        np.add.at(intercept_diff, (stack_row, stop), -intercept)
        np.add.at(slope_diff, (stack_row, start), slope)
        np.add.at(slope_diff, (stack_row, stop), -slope)
        np.add.at(count_diff, (stack_row, start), 1)
        np.add.at(count_diff, (stack_row, stop), -1)

    connected_previous = np.zeros(grouped_valid.shape, dtype=bool)
    connected_next = np.zeros(grouped_valid.shape, dtype=bool)
    connected_blue = grouped_blue.copy()
    connected_red = grouped_red.copy()
    next_valid_group = np.full(grouped_valid.shape, -1, dtype=np.int16)
    next_index = np.full(data.shape[0], -1, dtype=np.int16)
    for group in range(n_group - 1, -1, -1):
        next_valid_group[:, group] = next_index
        next_index = np.where(grouped_valid[:, group], group, next_index)
    for left_group in range(max(0, n_group - 1)):
        right_group = next_valid_group[:, left_group]
        pair_valid = (
            in_redshift_range
            & grouped_valid[:, left_group]
            & (right_group >= 0)
        )
        rows = np.flatnonzero(pair_valid)
        if not rows.size:
            continue
        right_for_row = right_group[rows]
        left_band = grouped_index[rows, left_group]
        right_band = grouped_index[rows, right_for_row]
        connected = np.zeros(rows.size, dtype=bool)
        for local_row, (source, first_band, last_band) in enumerate(
            zip(rows, left_band, right_band)
        ):
            interval_bands = np.flatnonzero(
                mask[source]
                & (red >= pivot[first_band])
                & (blue <= pivot[last_band])
            )
            order = interval_bands[np.argsort(blue[interval_bands])]
            component_end = -np.inf
            for band in order:
                if blue[band] > component_end:
                    if component_end >= pivot[first_band]:
                        break
                    component_end = red[band]
                else:
                    component_end = max(component_end, red[band])
                if component_end >= pivot[last_band]:
                    connected[local_row] = True
                    break
        rows = rows[connected]
        right_for_row = right_for_row[connected]
        if not rows.size:
            continue
        connected_next[rows, left_group] = True
        connected_previous[rows, right_for_row] = True
        component_blue = np.minimum(
            connected_blue[rows, left_group], grouped_blue[rows, right_for_row]
        )
        component_red = np.maximum(
            connected_red[rows, left_group], grouped_red[rows, right_for_row]
        )
        connected_blue[rows, left_group] = component_blue
        connected_blue[rows, right_for_row] = component_blue
        connected_red[rows, left_group] = component_red
        connected_red[rows, right_for_row] = component_red
        left_pivot = grouped_pivot[rows, left_group]
        right_pivot = grouped_pivot[rows, right_for_row]
        y0 = np.asarray(grouped_values[rows, left_group], dtype=np.float64)
        y1 = np.asarray(grouped_values[rows, right_for_row], dtype=np.float64)
        slope = (y1 - y0) / np.log(right_pivot / left_pivot)
        intercept = y0 - slope * np.log(left_pivot)
        add_segments(
            rows,
            left_pivot,
            right_pivot,
            intercept,
            slope,
            include_right=False,
        )

    for group in range(n_group):
        left_rows = np.flatnonzero(
            in_redshift_range
            & grouped_valid[:, group]
            & ~connected_previous[:, group]
        )
        add_segments(
            left_rows,
            connected_blue[left_rows, group],
            grouped_pivot[left_rows, group],
            np.asarray(grouped_values[left_rows, group], dtype=np.float64),
            np.zeros(left_rows.size, dtype=np.float64),
            include_right=False,
        )
        right_rows = np.flatnonzero(
            in_redshift_range
            & grouped_valid[:, group]
            & ~connected_next[:, group]
        )
        add_segments(
            right_rows,
            grouped_pivot[right_rows, group],
            connected_red[right_rows, group],
            np.asarray(grouped_values[right_rows, group], dtype=np.float64),
            np.zeros(right_rows.size, dtype=np.float64),
            include_right=True,
        )

    intercept_sum = np.cumsum(intercept_diff[:, :-1], axis=1, dtype=np.float64)
    slope_sum = np.cumsum(slope_diff[:, :-1], axis=1, dtype=np.float64)
    counts = np.cumsum(count_diff[:, :-1], axis=1, dtype=np.int32)
    sums = intercept_sum + slope_sum * log_centers[None, :]
    mean = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    for bin_index, n_galaxy in enumerate(galaxies_per_bin):
        required = max(minimum_count, int(np.ceil(minimum_fraction * n_galaxy)))
        mean[bin_index, counts[bin_index] < required] = np.nan

    return SEDStack(
        mean=mean,
        count=counts,
        wavelength_edges=wave_edges,
        redshift_edges=z_edges,
        galaxies_per_bin=galaxies_per_bin,
    )


def combine_sed_stacks(
    stacks: list[SEDStack] | tuple[SEDStack, ...],
    *,
    minimum_count: int = 5,
    minimum_fraction: float = 0.01,
) -> SEDStack:
    """Combine unmasked field stacks with exact galaxy weighting.

    Input stacks must use identical axes and retain a finite mean wherever
    their count is positive.  In practice, field stacks intended for this
    helper should be made with ``minimum_count=0`` and
    ``minimum_fraction=0``.  Their reconstructed sums and contributor counts
    are then added before the requested all-field threshold is applied.

    Args:
        stacks: Non-empty sequence of compatible field stacks.
        minimum_count: Absolute minimum contributors for a retained pixel.
        minimum_fraction: Also require this fraction of the galaxies assigned
            to the combined redshift bin.

    Returns:
        Exact galaxy-weighted combined stack.

    Raises:
        ValueError: Stacks are absent, incompatible, threshold-masked, or
            thresholds are invalid.
    """

    if not stacks:
        raise ValueError("at least one SED stack is required")
    if minimum_count < 0 or not (0 <= minimum_fraction <= 1):
        raise ValueError("invalid contributor threshold")
    reference = stacks[0]
    total_count = np.zeros(reference.count.shape, dtype=np.int64)
    total_sum = np.zeros(reference.mean.shape, dtype=np.float64)
    galaxies_per_bin = np.zeros(reference.galaxies_per_bin.shape, dtype=np.int64)
    for stack in stacks:
        if (
            stack.mean.shape != reference.mean.shape
            or stack.count.shape != reference.count.shape
            or not np.array_equal(stack.wavelength_edges, reference.wavelength_edges)
            or not np.array_equal(stack.redshift_edges, reference.redshift_edges)
            or stack.galaxies_per_bin.shape != reference.galaxies_per_bin.shape
        ):
            raise ValueError("SED stacks must have identical axes and shapes")
        if np.any(stack.count < 0):
            raise ValueError("SED stack counts must be non-negative")
        covered = stack.count > 0
        if np.any(covered & ~np.isfinite(stack.mean)):
            raise ValueError(
                "cannot combine threshold-masked stacks; make field stacks with "
                "minimum_count=0 and minimum_fraction=0"
            )
        total_sum[covered] += (
            stack.mean[covered].astype(np.float64) * stack.count[covered]
        )
        total_count += stack.count.astype(np.int64)
        galaxies_per_bin += stack.galaxies_per_bin.astype(np.int64)

    if np.max(total_count, initial=0) > np.iinfo(np.int32).max:
        raise ValueError("combined contributor count exceeds int32 range")
    if np.max(galaxies_per_bin, initial=0) > np.iinfo(np.int32).max:
        raise ValueError("combined galaxy count exceeds int32 range")
    counts = total_count.astype(np.int32)
    galaxies = galaxies_per_bin.astype(np.int32)
    mean = np.full(reference.mean.shape, np.nan, dtype=np.float32)
    np.divide(total_sum, counts, out=mean, where=counts > 0)
    for bin_index, n_galaxy in enumerate(galaxies):
        required = max(minimum_count, int(np.ceil(minimum_fraction * n_galaxy)))
        mean[bin_index, counts[bin_index] < required] = np.nan

    return SEDStack(
        mean=mean,
        count=counts,
        wavelength_edges=reference.wavelength_edges,
        redshift_edges=reference.redshift_edges,
        galaxies_per_bin=galaxies,
    )
