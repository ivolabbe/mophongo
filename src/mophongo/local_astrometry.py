"""Local astrometric correction utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
from astropy.nddata import Cutout2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .templates import Template
from . import astrometry

def shifts_at_positions(
    positions: np.ndarray,  # shape (N, 2) with (x, y) coordinates
    coeff_x: np.ndarray, 
    coeff_y: np.ndarray, 
    order: int,
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct shifts at multiple positions."""
    
    phi = np.array([
        astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order)
        for x, y in positions
    ])
    
    dx = phi @ coeff_x  # Matrix multiplication
    dy = phi @ coeff_y
    
    return dx, dy

# def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """Return the normalized cross-correlation of ``a`` and ``b``."""
#     fa = np.fft.fft2(a)
#     fb = np.fft.fft2(b)
#     cc = np.fft.ifft2(fa * np.conj(fb))
#     return np.abs(np.fft.fftshift(cc))


def _compute_snr(cc: np.ndarray) -> float:
    """Estimate a signal-to-noise ratio from a cross-correlation map."""
    return (cc.max() - np.median(cc)) / (cc.std() + 1e-12)


from photutils.centroids import centroid_quadratic, centroid_com


def measure_template_shifts(
    templates: Sequence[Template],
    coeffs: np.ndarray,
    residual: np.ndarray,
    box_size: int = 5,
    snr_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate template shifts from residual centroids.

    Parameters
    ----------
    templates
        List of PSF-matched templates.
    residual
        Residual image ``I_770 - (I_444 \* K)`` from a first-pass fit.
    box_size
        Size of the square stamp used for the cross-correlation.
    snr_threshold
        Minimum S/N of the cross-correlation peak to keep the measurement.

    Returns
    -------
    positions : ndarray of shape (N, 2)
        Template positions ``(x, y)`` for which a reliable shift was found.
    dx, dy : ndarray
        Measured shifts in pixels along x and y.
    weights : ndarray
        Weights proportional to ``S/N^2`` for each measurement.
    """
    half = box_size // 2
    ny, nx = residual.shape
    positions = []
    dx = []
    dy = []
    weights = []

    for j, (tmpl, coeff) in enumerate(zip(templates, coeffs)):

        x_pix, y_pix = tmpl.input_position_original
        x_stamp, y_stamp = tmpl.input_position_cutout

        if tmpl.flux <= 0 or tmpl.err <= 0:
            continue

        snr = (tmpl.flux/tmpl.err)
        if snr < snr_threshold:
            continue

        cutout_res = Cutout2D(residual, position=(x_pix, y_pix), size=3*box_size+1, mode='partial')            
        cutout_tmpl = Cutout2D(tmpl.data, position=(x_stamp, y_stamp), size=3*box_size+1, mode='partial')            

        # measure shift in residual + best-fit, relative to the template 
        model = coeff * cutout_tmpl.data + cutout_res.data
        xc, yc = cutout_tmpl.input_position_cutout

        cx_model, cy_model = centroid_quadratic(model, xpeak=xc, ypeak=yc, fit_boxsize=box_size)
        cx_tmp, cy_tmp = centroid_quadratic(cutout_tmpl.data, xpeak=xc, ypeak=yc, fit_boxsize=box_size)
        shift_est = np.array([cx_model - cx_tmp, cy_model - cy_tmp])
#        print(j,snr,shift_est)

        if np.isnan(shift_est).any():
#            print(f"NaN shift estimate for template {j} at position ({x_pix}, {y_pix}) remeasure:")
            cx_model, cy_model = centroid_com(model)
            cx_tmp, cy_tmp = centroid_com(cutout_tmpl.data)
            shift_est = np.array([cx_model - cx_tmp, cy_model - cy_tmp])
  #          print(j,snr,shift_est)
            if np.isnan(shift_est).any():
 #               print(f"NaN shift: SKIP")
                continue

        positions.append((x_pix, y_pix))
        dx.append(shift_est[0])
        dy.append(shift_est[1])
        weights.append(snr**2)

    return (
        np.array(positions, dtype=float),
        np.array(dx, dtype=float),
        np.array(dy, dtype=float),
        np.array(weights, dtype=float),
    )


def fit_polynomial_field(
    positions: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    weights: np.ndarray,
    order: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 2-D Chebyshev polynomials to the measured shifts."""
    phi = np.array(
        [
            astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order)
            for x, y in positions
        ]
    )
    w = np.diag(weights)
    ata = phi.T @ w @ phi
    at_dx = phi.T @ (weights * dx)
    at_dy = phi.T @ (weights * dy)
    coeff_x = np.linalg.solve(ata, at_dx)
    coeff_y = np.linalg.solve(ata, at_dy)
    return coeff_x, coeff_y


def apply_polynomial_correction(
    templates: Sequence[Template],
    coeff_x: np.ndarray,
    coeff_y: np.ndarray,
    order: int,
    shape: tuple[int, int],
) -> None:
    """Apply polynomial shifts to templates in place."""
    for tmpl in templates:
        x_pix, y_pix = tmpl.position_original
        phi = astrometry.cheb_basis(x_pix / (shape[1] - 1), y_pix / (shape[0] - 1), order)
        dx = float(np.dot(coeff_x, phi))
        dy = float(np.dot(coeff_y, phi))
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            continue
        tmpl.data = nd_shift(
            tmpl.data,
            (dy, dx),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        tmpl.shifted_position_original = (x_pix - dx, y_pix - dy)
        tmpl.shift += [dx, dy]


def correct_astrometry_polynomial(
    templates: Sequence[Template],
    residual: np.ndarray,
    coeffs: np.ndarray,
    *,
    order: int = 3,
    box_size: int = 9,
    snr_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure and correct local astrometric offsets with polynomials."""

    pos, dx, dy, weights = measure_template_shifts(
        templates, coeffs, residual, box_size, snr_threshold
    )
    if len(pos) == 0:
        n = astrometry.n_terms(order)
        return np.zeros(n), np.zeros(n)
    
    coeff_x, coeff_y = fit_polynomial_field(pos, dx, dy, weights, order, residual.shape)
    apply_polynomial_correction(templates, coeff_x, coeff_y, order, residual.shape)

    return coeff_x, coeff_y


def fit_gp_field(
    positions: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    weights: np.ndarray,
    *,
    length_scale: float = 500.0,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Fit Gaussian Process models to the measured shifts.

    Parameters
    ----------
    positions
        Array of ``(x, y)`` coordinates.
    dx, dy
        Measured shifts along x and y in pixels.
    weights
        Weights proportional to ``S/N^2`` for each measurement.
    length_scale
        Characteristic length scale of the RBF kernel in pixels.

    Returns
    -------
    tuple
        Gaussian Process models ``(gp_x, gp_y)`` for the x and y shifts.
    """

    X = positions
    err = 1.0 / np.sqrt(weights)
    base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=length_scale, length_scale_bounds=(10.0, 5000.0)
    )

    kernel_x = base_kernel + WhiteKernel(
        noise_level=err.mean() ** 2, noise_level_bounds=(1e-6, 1e2)
    )
    gp_x = GaussianProcessRegressor(
        kernel=kernel_x,
        alpha=err**2,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp_x.fit(X, dx)

    kernel_y = base_kernel + WhiteKernel(
        noise_level=err.mean() ** 2, noise_level_bounds=(1e-6, 1e2)
    )
    gp_y = GaussianProcessRegressor(
        kernel=kernel_y,
        alpha=err**2,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp_y.fit(X, dy)

    return gp_x, gp_y


def apply_gp_correction(
    templates: Sequence[Template],
    gp_x: GaussianProcessRegressor,
    gp_y: GaussianProcessRegressor,
) -> None:
    """Apply GP-derived shifts to templates in place."""

    positions = np.array([t.position_original for t in templates], dtype=float)
    dx_pred = gp_x.predict(positions)
    dy_pred = gp_y.predict(positions)

    for tmpl, dx, dy in zip(templates, dx_pred, dy_pred):
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            continue
        x_pix, y_pix = tmpl.position_original
        tmpl.data = nd_shift(
            tmpl.data,
            (dy, dx),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        tmpl.shifted_position_original = (x_pix - dx, y_pix - dy)
        tmpl.shift += [dx, dy]


def correct_astrometry_gp(
    templates: Sequence[Template],
    residual: np.ndarray,
    coeffs: np.ndarray,
    *,
    box_size: int = 9,
    snr_threshold: float = 5.0,
    length_scale: float = 500.0,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Measure and correct local astrometric offsets with Gaussian processes."""

    pos, dx, dy, weights = measure_template_shifts(
        templates, coeffs, residual, box_size, snr_threshold
    )
    if len(pos) == 0:
        dummy_kernel = ConstantKernel(1.0) * RBF(length_scale)
        return (
            GaussianProcessRegressor(kernel=dummy_kernel),
            GaussianProcessRegressor(kernel=dummy_kernel),
        )

    gp_x, gp_y = fit_gp_field(pos, dx, dy, weights, length_scale=length_scale)
    apply_gp_correction(templates, gp_x, gp_y)

    return gp_x, gp_y
