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

from .utils import elliptical_gaussian, elliptical_moffat, measure_shape, moffat, gaussian


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
        from .utils import moffat as moffat_psf

        return cls(moffat_psf(size, fwhm_x, fwhm_y, beta, theta))

    @classmethod
    def gaussian(
        cls,
        size: int | tuple[int, int],
        fwhm_x: float,
        fwhm_y: float,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Gaussian PSF."""
        from .utils import gaussian as gaussian_psf

        return cls(gaussian_psf(size, fwhm_x, fwhm_y, theta))

    @classmethod
    def delta(cls, size: int = 3) -> "PSF":
        """Create a symmetric delta function PSF.

        Parameters
        ----------
        size : int, optional
            Length of each side of the square PSF array. ``size`` should be odd
            to center the delta pixel. Defaults to ``3``.

        Returns
        -------
        PSF
            PSF instance containing a single central pixel with unit flux.
        """

        array = np.zeros((size, size), dtype=float)
        cy = size // 2
        cx = size // 2
        array[cy, cx] = 1.0
        return cls(array)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "PSF":
        """Create a PSF from an arbitrary pixel array."""
        return cls(array)

    def matching_kernel(self, other: "PSF" | np.ndarray, window: object | None = None) -> np.ndarray:
        """Return the convolution kernel that matches ``self`` to ``other``.
        
        Parameters
        ----------
        other : PSF or np.ndarray
            The target PSF. If np.ndarray, it's assumed to be a normalized PSF.
        window : optional
            Window function passed to ``create_matching_kernel``. Defaults to TukeyWindow(alpha=0.4).
        
        Returns
        -------
        kernel : np.ndarray
            Convolution kernel that matches self to other.
        """
        psf_hi = self.array
        
        # Handle both PSF objects and numpy arrays
        if isinstance(other, PSF):
            psf_lo = other.array
        else:
            psf_lo = np.asarray(other, dtype=float)
            # Normalize if not already normalized
            if psf_lo.sum() != 0:
                psf_lo = psf_lo / psf_lo.sum()
        
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

    def fit_moffat(self, free_params: str = "fwhm_x,fwhm_y,beta,theta") -> MoffatFit:
        """Fit a 2-D Moffat profile to the PSF data.
        
        Parameters
        ----------
        free_params : str
            Comma-separated list of parameters to fit. Options: 'fwhm_x', 'fwhm_y', 'beta', 'theta'
            Default: "fwhm_x,fwhm_y,beta,theta"
        """
        from .utils import moffat
        
        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2
        cx = (self.array.shape[1] - 1) / 2

        # Get initial values from measure_shape
        _, _, sigma_x, sigma_y, theta0 = measure_shape(self.array, np.ones_like(self.array, dtype=bool))
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2
        
        # Initial parameter values
        params = {
            'fwhm_x': 2.355 * sigma_x,
            'fwhm_y': 2.355 * sigma_y,
            'beta': 2.5,
            'theta': theta0
        }
        
        # Parse free parameters
        free_list = [p.strip() for p in free_params.split(',')]
        all_params = ['fwhm_x', 'fwhm_y', 'beta', 'theta']
        
        # Create parameter vector and bounds for free parameters only
        p0 = []
        bounds_lower = []
        bounds_upper = []
        param_map = {}
        
        for i, param in enumerate(all_params):
            if param in free_list:
                p0.append(params[param])
                param_map[len(p0) - 1] = param
                if param in ['fwhm_x', 'fwhm_y']:
                    bounds_lower.append(1e-3)
                    bounds_upper.append(np.inf)
                elif param == 'beta':
                    bounds_lower.append(0.5)
                    bounds_upper.append(20.0)
                elif param == 'theta':
                    bounds_lower.append(-np.pi / 2)
                    bounds_upper.append(np.pi / 2)
        
        bounds = (bounds_lower, bounds_upper)
        
        def residual(p: np.ndarray) -> np.ndarray:
            # Update parameters with fitted values
            current_params = params.copy()
            for i, param_name in param_map.items():
                current_params[param_name] = p[i]
            
            model = moffat(
                self.array.shape,
                current_params['fwhm_x'],
                current_params['fwhm_y'],
                current_params['beta'],
                current_params['theta'],
                x0=cx,
                y0=cy
            )
            return (model - self.array).ravel()

        result = least_squares(residual, p0, bounds=bounds)
        
        # Update fitted parameters
        for i, param_name in param_map.items():
            params[param_name] = float(result.x[i])
        
        return MoffatFit(params['fwhm_x'], params['fwhm_y'], params['beta'], params['theta'])

    def fit_gaussian(self, free_params: str = "fwhm_x,fwhm_y,theta") -> GaussianFit:
        """Fit a 2-D Gaussian profile to the PSF data.
        
        Parameters
        ----------
        free_params : str
            Comma-separated list of parameters to fit. Options: 'fwhm_x', 'fwhm_y', 'theta'
            Default: "fwhm_x,fwhm_y,theta"
        """
        from .utils import gaussian
        
        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2
        cx = (self.array.shape[1] - 1) / 2

        # Get initial values from measure_shape
        _, _, sigma_x, sigma_y, theta0 = measure_shape(self.array, np.ones_like(self.array, dtype=bool))
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2
        
        # Initial parameter values
        params = {
            'fwhm_x': 2.355 * sigma_x,
            'fwhm_y': 2.355 * sigma_y,
            'theta': theta0
        }
        
        # Parse free parameters
        free_list = [p.strip() for p in free_params.split(',')]
        all_params = ['fwhm_x', 'fwhm_y', 'theta']
        
        # Create parameter vector and bounds for free parameters only
        p0 = []
        bounds_lower = []
        bounds_upper = []
        param_map = {}
        
        for i, param in enumerate(all_params):
            if param in free_list:
                p0.append(params[param])
                param_map[len(p0) - 1] = param
                if param in ['fwhm_x', 'fwhm_y']:
                    bounds_lower.append(1e-3)
                    bounds_upper.append(np.inf)
                elif param == 'theta':
                    bounds_lower.append(-np.pi / 2)
                    bounds_upper.append(np.pi / 2)
        
        bounds = (bounds_lower, bounds_upper)
        
        def residual(p: np.ndarray) -> np.ndarray:
            # Update parameters with fitted values
            current_params = params.copy()
            for i, param_name in param_map.items():
                current_params[param_name] = p[i]
            
            model = gaussian(
                self.array.shape,
                current_params['fwhm_x'],
                current_params['fwhm_y'],
                current_params['theta'],
                x0=cx,
                y0=cy
            )
            return (model - self.array).ravel()

        result = least_squares(residual, p0, bounds=bounds)
        
        # Update fitted parameters
        for i, param_name in param_map.items():
            params[param_name] = float(result.x[i])
        
        return GaussianFit(params['fwhm_x'], params['fwhm_y'], params['theta'])


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
