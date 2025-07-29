"""Local astrometric correction utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

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


from photutils.centroids import centroid_quadratic


def measure_template_shifts(
    templates: Sequence[Template],
    coeffs: np.ndarray,
    residual: np.ndarray,
    box_size: int = 11,
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
        x_pix, y_pix = [int(round(v)) for v in tmpl.position_original]
        y0 = max(y_pix - half, 0)
        y1 = min(y_pix + half + 1, ny)
        x0 = max(x_pix - half, 0)
        x1 = min(x_pix + half + 1, nx)
        if y0 >= y1 or x0 >= x1:
            continue

        sl_res = (slice(y0, y1), slice(x0, x1))
        sl_tmp = (
            slice(
                tmpl.slices_cutout[0].start + y0 - tmpl.slices_original[0].start,
                tmpl.slices_cutout[0].start + y1 - tmpl.slices_original[0].start,
            ),
            slice(
                tmpl.slices_cutout[1].start + x0 - tmpl.slices_original[1].start,
                tmpl.slices_cutout[1].start + x1 - tmpl.slices_original[1].start,
            ),
        )

        stamp_res = residual[sl_res]
        stamp_tmp = tmpl.data[sl_tmp]
        mny = min(stamp_res.shape[0], stamp_tmp.shape[0])
        mnx = min(stamp_res.shape[1], stamp_tmp.shape[1])
        if mny <= 1 or mnx <= 1:
            continue
        stamp_res = stamp_res[:mny, :mnx]
        stamp_tmp = stamp_tmp[:mny, :mnx]

        model = coeff * stamp_tmp + stamp_res
        try:
            cy_model, cx_model = centroid_quadratic(model)
            cy_tmp, cx_tmp = centroid_quadratic(stamp_tmp)
        except Exception:
            continue

        shift_est = np.array([cx_model - cx_tmp, cy_model - cy_tmp])

        # estimate S/N from coefficient and local residual dispersion
        noise = np.std(stamp_res)
        snr = 0.0 if noise == 0 else abs(coeff) / noise

        # DO SIMPLE SHIFT ESTIMATE 
        # 1. calculate SNR from weight  + image 
        # 1. only keep sources with SNR > fitconfig.snr_thresh_astrom
        # 2. construct the model + residual for each stamp res + coeff * tmpl.data
        # 3. measure centroid on the model + residual
        # 4. measure centroid on the template
        # 5. calculate the shift as the difference of the centroids
        # 6. apply that shift to the template data in place, record shift in template.shift
        # 7 double check the SIGN of the shift. 

    # m = t.data[t.slices_cutout] * coeff
    # r = resid0[t.slices_original]
    # mr = m +r 
    # rm_xy = centroid_quadratic(r+m)
    # m_xy = centroid_quadratic(m)
    # print(f"Shifted residual centroid: {rm_xy}, model centroid: {m_xy} shift: {m_xy- rm_xy}")


        if snr < snr_threshold:
            continue

        tmpl.data = nd_shift(
            tmpl.data,
            (-shift_est[1], -shift_est[0]),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        tmpl.position_original = (x_pix - shift_est[0], y_pix - shift_est[1])
        tmpl.shift += [-shift_est[0], -shift_est[1]]

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
        tmpl.position_original = (x_pix - dx, y_pix - dy)


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
