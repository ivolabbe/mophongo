"""Local astrometric correction utilities.

This module provides a single :class:`AstroCorrect` front-end which measures
per-template shifts and fits a smooth field using either a polynomial or a
Gaussian-process model.  Additional algorithms can be plugged in by adding
private ``_fit_*`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com, centroid_quadratic
from scipy.ndimage import shift as nd_shift
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from . import astrometry
from .templates import Template


def shifts_at_positions(
    positions: np.ndarray,
    coeff_x: np.ndarray,
    coeff_y: np.ndarray,
    order: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct shifts at multiple positions.

    Parameters
    ----------
    positions : ndarray
        ``(N, 2)`` array with ``(x, y)`` coordinates.
    coeff_x, coeff_y : ndarray
        Polynomial coefficients for the x and y directions.
    order : int
        Polynomial order.
    shape : tuple of int
        Shape of the image ``(ny, nx)`` used for normalisation.

    Returns
    -------
    tuple of ndarray
        Predicted ``(dx, dy)`` shifts for the input positions.
    """

    phi = np.array(
        [astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order) for x, y in positions]
    )
    dx = phi @ coeff_x
    dy = phi @ coeff_y
    return dx, dy


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
    templates : Sequence[Template]
        List of PSF-matched templates.
    coeffs : ndarray
        Fitted amplitudes for each template.
    residual : ndarray
        Residual image from a first-pass fit.
    box_size : int, optional
        Size of the square stamp used for centroiding.
    snr_threshold : float, optional
        Minimum ``S/N`` of a template to keep the measurement.

    Returns
    -------
    positions : ndarray of shape ``(N, 2)``
        Template positions ``(x, y)`` for which a reliable shift was found.
    dx, dy : ndarray
        Measured shifts in pixels along x and y.
    weights : ndarray
        Weights proportional to ``S/N^2`` for each measurement.
    """

    positions: list[tuple[float, float]] = []
    dx: list[float] = []
    dy: list[float] = []
    weights: list[float] = []

    for tmpl, coeff in zip(templates, coeffs):
        x_pix, y_pix = tmpl.input_position_original
        x_stamp, y_stamp = tmpl.input_position_cutout

        if tmpl.flux <= 0 or tmpl.err <= 0:
            continue

        snr = tmpl.flux / tmpl.err
        if snr < snr_threshold:
            continue

        cutout_res = Cutout2D(residual, position=(x_pix, y_pix), size=3 * box_size + 1, mode="partial")
        cutout_tmpl = Cutout2D(tmpl.data, position=(x_stamp, y_stamp), size=3 * box_size + 1, mode="partial")

        model = coeff * cutout_tmpl.data + cutout_res.data
        xc, yc = cutout_tmpl.input_position_cutout

        cx_model, cy_model = centroid_quadratic(model, xpeak=xc, ypeak=yc, fit_boxsize=box_size)
        cx_tmp, cy_tmp = centroid_quadratic(cutout_tmpl.data, xpeak=xc, ypeak=yc, fit_boxsize=box_size)
        shift_est = np.array([cx_model - cx_tmp, cy_model - cy_tmp])

        if np.isnan(shift_est).any():
            cx_model, cy_model = centroid_com(model)
            cx_tmp, cy_tmp = centroid_com(cutout_tmpl.data)
            shift_est = np.array([cx_model - cx_tmp, cy_model - cy_tmp])
            if np.isnan(shift_est).any():
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


@dataclass
class AstroCorrect:
    """Homogeneous front-end for local astrometry corrections.

    Parameters
    ----------
    cfg : FitConfig
        Configuration carrying the astrometry settings.
    """

    cfg: "FitConfig"

    _predict: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, default=lambda p: (np.zeros(len(p)), np.zeros(len(p)))
    )

    def __call__(self, x: float | np.ndarray, y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the predicted shift at a position or array of positions."""

        if y is None:
            pos = np.asarray(x, float)
            if pos.ndim == 1:
                x_shape = () if np.isscalar(x) else (pos.shape[0],)
                pos = pos.reshape(-1, 2)
            else:
                x_shape = pos.shape[:-1]
        else:
            x_arr = np.asarray(x, float)
            y_arr = np.asarray(y, float)
            x_shape = np.broadcast(x_arr, y_arr).shape
            pos = np.c_[x_arr.ravel(), y_arr.ravel()]
        dx, dy = self._predict(pos)
        return dx.reshape(x_shape), dy.reshape(x_shape)

    def fit(
        self,
        templates: Sequence[Template],
        residual: np.ndarray,
        coeffs: np.ndarray,
    ) -> None:
        """Measure per-source shifts, fit a smooth field and update templates."""

        model = self.cfg.astrom_model.lower()
        kwargs = dict(
            snr_threshold=self.cfg.snr_thresh_astrom,
            **self.cfg.astrom_kwargs,  # Use astrom_kwargs directly
        )

        pos, dx, dy, w = measure_template_shifts(
            templates,
            coeffs,
            residual,
            box_size=kwargs.pop("box_size", 7),
            snr_threshold=kwargs.pop("snr_threshold"),
        )

        if len(pos) == 0:
            self._predict = lambda p: (np.zeros(len(p)), np.zeros(len(p)))
            return

        if model == "poly":
            self._predict = self._fit_polynomial(pos, dx, dy, w, shape=residual.shape, **kwargs)
        elif model == "gp":
            self._predict = self._fit_gp(pos, dx, dy, w, **kwargs)
        else:
            raise ValueError(f"Unknown astrom_model '{model}'")

        dxx, dyy = self(np.array([t.input_position_original for t in templates]))
        for tmpl, dxi, dyi in zip(templates, dxx, dyy):
            if abs(dxi) < 1e-2 and abs(dyi) < 1e-2:
                continue
            x0, y0 = tmpl.input_position_original
            tmpl.data = nd_shift(
                tmpl.data,
                (dyi, dxi),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            tmpl.input_position_original = (x0 - dxi, y0 - dyi)
            tmpl.shift += [dxi, dyi]

    def _fit_polynomial(
        self,
        pos: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        w: np.ndarray,
        *,
        order: int = 2,
        shape: tuple[int, int],
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        print('POLY ORDER:', order)
        phi = np.array(
            [astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order) for x, y in pos]
        )
        W = np.diag(w)
        coeff_x = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dx))
        coeff_y = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dy))

        def _predict(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            phi_p = np.array(
                [astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order) for x, y in p]
            )
            return phi_p @ coeff_x, phi_p @ coeff_y

        return _predict

    def _fit_gp(
        self,
        pos: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        w: np.ndarray,
        *,
        length_scale: float = 300.0,
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:

        print('GP LENGTH SCALE:', length_scale)
        err = 1 / np.sqrt(w)
        base = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale, (10.0, 5000.0))
        gpx = GaussianProcessRegressor(
            base + WhiteKernel(err.mean() ** 2, (1e-6, 1e2)),
            alpha=err**2,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=0,
        )
        gpy = clone(gpx)
        gpx.fit(pos, dx)
        gpy.fit(pos, dy)

        def _predict(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return gpx.predict(p), gpy.predict(p)

        return _predict

