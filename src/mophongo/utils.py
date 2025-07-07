"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import numpy as np


def elliptical_moffat(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Moffat profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
    return amplitude * (1 + r2) ** (-beta)


def elliptical_gaussian(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Gaussian profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    r2 = (xr / sigma_x) ** 2 + (yr / sigma_y) ** 2
    return amplitude * np.exp(-0.5 * r2)


def measure_shape(data: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``x_c``, ``y_c``, ``sigma_x``, ``sigma_y``, and ``theta`` of ``data``.

    Parameters
    ----------
    data : ndarray
        Pixel data.
    mask : ndarray
        Boolean mask selecting the object pixels.
    """
    y_idx, x_idx = np.indices(data.shape)
    flux = float(data[mask].sum())
    y_c = float((y_idx[mask] * data[mask]).sum() / flux)
    x_c = float((x_idx[mask] * data[mask]).sum() / flux)
    y_rel = y_idx - y_c
    x_rel = x_idx - x_c
    cov_xx = float((data[mask] * x_rel[mask] ** 2).sum() / flux)
    cov_yy = float((data[mask] * y_rel[mask] ** 2).sum() / flux)
    cov_xy = float((data[mask] * x_rel[mask] * y_rel[mask]).sum() / flux)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    sigma_x = float(np.sqrt(vals[0]))
    sigma_y = float(np.sqrt(vals[1]))
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return x_c, y_c, sigma_x, sigma_y, theta

