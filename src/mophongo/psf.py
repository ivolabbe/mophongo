"""Point spread function utilities.

This module provides a :class:`PSF` class which wraps a pixel grid
representation of a point spread function. Instances can be created from
analytic profiles (Moffat, Gaussian) or directly from a user supplied
array. A method is included to compute a matching kernel between two PSFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.optimize import least_squares

import numpy as np
from photutils.psf import matching
from photutils.psf.matching import TukeyWindow

from .utils import elliptical_gaussian, elliptical_moffat, measure_shape


def _moffat_psf(
    size: int | tuple[int, int], fwhm_x: float, fwhm_y: float, beta: float, theta: float = 0.0
) -> np.ndarray:
    """Return a 2-D elliptical Moffat PSF normalized to unit sum."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    psf = elliptical_moffat(
        y,
        x,
        1.0,
        fwhm_x,
        fwhm_y,
        beta,
        theta,
        cx,
        cy,
    )
    psf /= psf.sum()
    return psf


def _gaussian_psf(
    size: int | tuple[int, int], fwhm_x: float, fwhm_y: float, theta: float = 0.0
) -> np.ndarray:
    """Return a 2-D elliptical Gaussian PSF normalized to unit sum."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    psf = elliptical_gaussian(
        y,
        x,
        1.0,
        fwhm_x,
        fwhm_y,
        theta,
        cx,
        cy,
    )
    psf /= psf.sum()
    return psf


@dataclass
class MoffatFit:
    """Parameters describing a fitted Moffat profile."""

    fwhm_x: float
    fwhm_y: float
    beta: float
    theta: float


@dataclass
class GaussianFit:
    """Parameters describing a fitted Gaussian profile."""

    fwhm_x: float
    fwhm_y: float
    theta: float


@dataclass
class PSF:
    """Discrete point spread function."""

    array: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.array, dtype=float)
        s = arr.sum()
        if s != 0:
            arr = arr / s
        self.array = arr

    @classmethod
    def moffat(
        cls,
        size: int | tuple[int, int],
        fwhm_x: float,
        fwhm_y: float,
        beta: float,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Moffat PSF."""
        return cls(_moffat_psf(size, fwhm_x, fwhm_y, beta, theta))

    @classmethod
    def gaussian(
        cls,
        size: int | tuple[int, int],
        fwhm_x: float,
        fwhm_y: float,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Gaussian PSF."""
        return cls(_gaussian_psf(size, fwhm_x, fwhm_y, theta))

    @classmethod
    def from_array(cls, array: np.ndarray) -> "PSF":
        """Create a PSF from an arbitrary pixel array."""
        return cls(array)

    def matching_kernel(self, other: "PSF", window: object | None = None) -> np.ndarray:
        """Return the convolution kernel that matches ``self`` to ``other``."""
        return psf_matching_kernel(self.array, other.array, window=window)

    def fit_moffat(self) -> MoffatFit:
        """Fit a 2-D Moffat profile to the PSF data."""

        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2
        cx = (self.array.shape[1] - 1) / 2

        _, _, sigma_x, sigma_y, theta0 = measure_shape(self.array, np.ones_like(self.array, dtype=bool))
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2
        p0 = np.array([2.355 * sigma_x, 2.355 * sigma_y, 2.5, theta0])
        bounds = ([1e-3, 1e-3, 0.5, -np.pi / 2], [np.inf, np.inf, 20.0, np.pi / 2])

        def residual(p: np.ndarray) -> np.ndarray:
            fwhm_x, fwhm_y, beta, theta = p
            model = elliptical_moffat(
                y,
                x,
                1.0,
                fwhm_x,
                fwhm_y,
                beta,
                theta,
                cx,
                cy,
            )
            model /= model.sum()
            return (model - self.array).ravel()

        result = least_squares(residual, p0, bounds=bounds)
        fwhm_x, fwhm_y, beta, theta = result.x
        return MoffatFit(float(fwhm_x), float(fwhm_y), float(beta), float(theta))

    def fit_gaussian(self) -> GaussianFit:
        """Fit a 2-D Gaussian profile to the PSF data."""

        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2
        cx = (self.array.shape[1] - 1) / 2

        _, _, sigma_x, sigma_y, theta0 = measure_shape(self.array, np.ones_like(self.array, dtype=bool))
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2
        p0 = np.array([2.355 * sigma_x, 2.355 * sigma_y, theta0])
        bounds = ([1e-3, 1e-3, -np.pi / 2], [np.inf, np.inf, np.pi / 2])

        def residual(p: np.ndarray) -> np.ndarray:
            fwhm_x, fwhm_y, theta = p
            model = elliptical_gaussian(
                y,
                x,
                1.0,
                fwhm_x,
                fwhm_y,
                theta,
                cx,
                cy,
            )
            model /= model.sum()
            return (model - self.array).ravel()

        result = least_squares(residual, p0, bounds=bounds)
        fwhm_x, fwhm_y, theta = result.x
        return GaussianFit(float(fwhm_x), float(fwhm_y), float(theta))


def moffat_psf(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float = 0.0,
) -> np.ndarray:
    """Return a normalized Moffat PSF array.

    This is a convenience wrapper around ``PSF.moffat`` returning the pixel
    array directly.
    """
    return PSF.moffat(size, fwhm_x, fwhm_y, beta, theta).array


def pad_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad array with zeros to center it in the target shape."""
    py = (shape[0] - arr.shape[0]) // 2
    px = (shape[1] - arr.shape[1]) // 2
    return np.pad(arr, ((py, shape[0] - arr.shape[0] - py), (px, shape[1] - arr.shape[1] - px)))


def psf_matching_kernel(
    psf_hi: np.ndarray, psf_lo: np.ndarray, *, window: object | None = None
) -> np.ndarray:
    """Compute a convolution kernel matching ``psf_hi`` to ``psf_lo``.

    The kernel ``k`` is defined such that ``psf_hi * k \approx psf_lo`` when
    convolved. ``photutils.psf.matching.create_matching_kernel`` is used under
    the hood. If the two PSFs have different shapes they are zero padded to a
    common grid before computing the kernel.

    Parameters
    ----------
    psf_hi, psf_lo:
        High- and low-resolution PSF arrays normalized to unit sum. They may
        have different shapes.
    window : optional
        Window function passed to ``create_matching_kernel``. Defaults to TukeyWindow(alpha=0.5).

    Returns
    -------
    kernel: ``np.ndarray``
        Convolution kernel with shape equal to the larger of the two input PSFs.
    """
    if psf_hi.shape != psf_lo.shape:
        ny = max(psf_hi.shape[0], psf_lo.shape[0])
        nx = max(psf_hi.shape[1], psf_lo.shape[1])
        shape = (ny, nx)
        psf_hi = pad_to_shape(psf_hi, shape)
        psf_lo = pad_to_shape(psf_lo, shape)

    if window is None:
        window = TukeyWindow(alpha=0.4)

    kernel = matching.create_matching_kernel(psf_hi, psf_lo, window=window)
    return np.asarray(kernel)
