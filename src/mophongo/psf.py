"""Point spread function utilities.

This module provides a :class:`PSF` class which wraps a pixel grid
representation of a point spread function. Instances can be created from
analytic profiles (Moffat, Gaussian) or directly from a user supplied
array. A method is included to compute a matching kernel between two PSFs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    x = x - cx
    y = y - cy

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t

    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))

    r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
    psf = (1 + r2) ** (-beta)
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
    x = x - cx
    y = y - cy

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t

    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))

    r2 = (xr / sigma_x) ** 2 + (yr / sigma_y) ** 2
    psf = np.exp(-0.5 * r2)
    psf /= psf.sum()
    return psf


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

    def matching_kernel(self, other: "PSF", reg: float = 1e-3) -> np.ndarray:
        """Return the convolution kernel that matches ``self`` to ``other``."""
        return psf_matching_kernel(self.array, other.array, reg)


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


def psf_matching_kernel(psf_hi: np.ndarray, psf_lo: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """Compute a convolution kernel matching ``psf_hi`` to ``psf_lo``.

    The kernel ``k`` is defined such that ``psf_hi * k \approx psf_lo`` when
    convolved. The computation is done in the Fourier domain with a small
    regularization term to avoid division by zero.

    Parameters
    ----------
    psf_hi, psf_lo:
        High- and low-resolution PSF arrays of identical shape. They must be
        normalized to unit sum.
    reg:
        Regularization parameter added to the denominator in Fourier space.

    Returns
    -------
    kernel: ``np.ndarray``
        Convolution kernel with the same shape as the input PSFs.
    """
    if psf_hi.shape != psf_lo.shape:
        raise ValueError("Input PSFs must have the same shape")

    f_hi = np.fft.fft2(psf_hi)
    f_lo = np.fft.fft2(psf_lo)
    kernel_freq = f_lo / (f_hi + reg)
    kernel = np.fft.ifft2(kernel_freq).real
    return kernel
