"""Point spread function utilities.

This module provides a :class:`PSF` class which wraps a pixel grid
representation of a point spread function. Instances can be created from
analytic profiles (Moffat, Gaussian) or directly from a user supplied
array. A method is included to compute a matching kernel between two PSFs.
"""

from __future__ import annotations

from collections import OrderedDict

import logging
import os
import numpy as np
from scipy.ndimage import shift as shift
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from drizzlepac import adrizzle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.data import download_file
from photutils.psf.matching import TukeyWindow
from photutils.centroids import centroid_quadratic

from .utils import (
    measure_shape,
    get_wcs_pscale,
    get_slice_wcs,
    to_header,
    fit_kernel_fourier,
    pad_to_shape,
    matching_kernel
)
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
 

logger = logging.getLogger(__name__)

@dataclass
class GaussianFit:
    """Parameters describing a fitted Gaussian profile."""

    fwhm_x: float
    fwhm_y: float
    theta: float
    xc: float
    yc: float
    flux: float
    shape: tuple = None  # Store the original array shape
    
    def model(self) -> np.ndarray:
        """Generate the best fit Gaussian model."""
        from .utils import gaussian
        return gaussian(self.shape, self.fwhm_x, self.fwhm_y, self.theta, 
                       x0=self.xc, y0=self.yc, flux=self.flux)

@dataclass
class MoffatFit:
    """Parameters describing a fitted Moffat profile."""

    fwhm_x: float
    fwhm_y: float
    beta: float
    theta: float
    xc: float
    yc: float
    flux: float
    shape: tuple = None  # Store the original array shape
    
    def model(self) -> np.ndarray:
        """Generate the best fit Moffat model."""
        from .utils import moffat
        return moffat(self.shape, self.fwhm_x, self.fwhm_y, self.beta, 
                     self.theta, x0=self.xc, y0=self.yc, flux=self.flux)

@dataclass
class PSF:
    """Discrete point spread function."""

    array: np.ndarray
    wcs: WCS | None = None
    pos: tuple[float, float] | None = None

    @property
    def data(self) -> np.ndarray:
        """Alias for the PSF array."""
        return self.array

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
        fwhm: float | tuple[float, float] | None = None,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Gaussian PSF.

        Parameters
        ----------
        size : int or tuple of int
            Size of the PSF array.
        fwhm_x, fwhm_y : float, optional
            FWHM along x and y axes.
        fwhm : float or tuple, optional
            If given, overrides fwhm_x and fwhm_y. If tuple, interpreted as (fwhm_x, fwhm_y).
        theta : float, optional
            Rotation angle in radians.

        Returns
        -------
        PSF
            PSF instance with a Gaussian profile.
        """
        from .utils import gaussian

        # Handle fwhm as tuple or float
        if fwhm is not None:
            if isinstance(fwhm, (tuple, list)) and len(fwhm) == 2:
                fwhm_x, fwhm_y = fwhm
            else:
                fwhm_x = fwhm_y = fwhm

        return cls(gaussian(size, fwhm_x, fwhm_y, theta=theta))

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

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        position: tuple[float, float] | tuple["Quantity", "Quantity"]
        | None = None,
        *,
        search_boxsize: int | tuple[int, int] | None = None,
        fit_boxsize: int | tuple[int, int] = 5,
        size: int = 51,
        wcs: WCS | None = None,
        verbose: bool = False,
    ) -> "PSF":
        """Extract a PSF from ``data`` around an approximate star position.

        Parameters
        ----------
        data : ndarray
            Image containing the star.
        position : tuple of float or tuple of astropy Quantity
            Approximate ``(x, y)`` pixel coordinates of the star, or (ra, dec) as astropy Quantities (e.g. with unit deg).
        search_boxsize, fit_boxsize : int or tuple of int, optional
            Passed to :func:`photutils.centroids.centroid_quadratic`.
        size : int, optional
            Cutout size. The PSF will be a square array of this shape.
        wcs : astropy.wcs.WCS, optional
            WCS object for the image.

        Returns
        -------
        PSF
            PSF instance extracted from the image.
        """
        from astropy.nddata import Cutout2D
        from photutils.centroids import centroid_quadratic
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if position is None:
            # get center pixel of ndarray
            position_pix = ((data.shape[1] - 1) // 2, (data.shape[0] - 1) / 2)
        else:
            # If position is given as (Quantity, Quantity) and wcs is supplied, convert to pixel
            if (hasattr(position[0], "unit") and hasattr(position[1], "unit")
                    and wcs is not None):
                sky = SkyCoord(position[0], position[1])
                x, y = wcs.world_to_pixel(sky)
                position_pix = (x, y)
            else:
                position_pix = position

    # If either search_boxsize or fit_boxsize is None, skip recentering
        if search_boxsize is None:
            x_cen, y_cen = position_pix
        else:
            x_cen, y_cen = centroid_quadratic(
                data,
                xpeak=position_pix[0],
                ypeak=position_pix[1],
                fit_boxsize=fit_boxsize,
                search_boxsize=search_boxsize,
            )
        if verbose:
            if search_boxsize is not None:
                print(f"original position: ({position_pix})")
            print(f"Centroid position: ({x_cen}, {y_cen})")

        cut = Cutout2D(
            data,
            (x_cen, y_cen),  # Use unrounded center
            (size, size),
            mode="partial",
            wcs=wcs,
            fill_value=0.0,
            copy=True,
        )
        return cls(array=np.asarray(cut.data),
                   wcs=cut.wcs,
                   pos=cut.input_position_cutout)

    def matching_kernel(
        self,
        other: "PSF" | np.ndarray,
        window: object | None = None,
        *,
        recenter: bool = True,
    ) -> np.ndarray:
        """Return the convolution kernel that matches ``self`` to ``other``.
        
        Parameters
        ----------
        other : PSF or np.ndarray
            The target PSF. If np.ndarray, it's assumed to be a normalized PSF.
        window : optional
            Window function passed to ``create_matching_kernel``. Defaults to TukeyWindow(alpha=0.4).
        
        Parameters
        ----------
        recenter : bool, optional
            If ``True`` the resulting kernel is shifted to its centroid using
            bicubic interpolation. Defaults to ``True``.

        Returns
        -------
        kernel : np.ndarray
            Convolution kernel that matches self to other.
        """
        psf_hi = self.array

        # Handle both PSF objects and numpy arrays
        if isinstance(other, PSF):
            psf_lo = other.array
        elif hasattr(other, "array"):
            psf_lo = np.asarray(other.array, dtype=float)
        else:
            psf_lo = np.asarray(other, dtype=float)
        # Normalize if not already normalized
        if psf_lo.sum() != 0:
            psf_lo = psf_lo / psf_lo.sum()

        kernel = matching_kernel(psf_hi,
                                     psf_lo,
                                     window=window,
                                     recenter=recenter)
        return kernel.astype(np.float32)

    def matching_kernel_basis(
        self,
        other: "PSF" | np.ndarray,
        basis: np.ndarray,
        *,
        method: str = "lstsq",
        recenter: bool = True,
    ) -> np.ndarray:
        """Return convolution kernel using a Fourier basis fit."""

        psf_hi = self.array
        psf_lo = other.array if isinstance(other, PSF) else np.asarray(other, dtype=float)
        if psf_lo.sum() != 0:
            psf_lo = psf_lo / psf_lo.sum()

        if psf_hi.shape != psf_lo.shape:
            ny = max(psf_hi.shape[0], psf_lo.shape[0])
            nx = max(psf_hi.shape[1], psf_lo.shape[1])
            shape = (ny, nx)
            psf_hi = pad_to_shape(psf_hi, shape)
            psf_lo = pad_to_shape(psf_lo, shape)

        if basis.shape[:2] != psf_hi.shape:
            basis = np.stack(
                [pad_to_shape(basis[:, :, i], psf_hi.shape) for i in range(basis.shape[2])],
                axis=2,
            )

        kernel, _ = fit_kernel_fourier(psf_hi, psf_lo, basis, method=method)
        if recenter:
            ycen, xcen = centroid_quadratic(kernel, fit_boxsize=5)
            if not np.isnan(ycen) and not np.isnan(xcen):
                cy = (kernel.shape[0] - 1) / 2
                cx = (kernel.shape[1] - 1) / 2
                kernel = shift(kernel, (cy - ycen, cx - xcen), order=3, mode="nearest")
        return kernel

    def _fit_profile(self, model_func, default_params, free_params, xc=None, yc=None, result_class=None):
        """Shared fitting logic for both Gaussian and Moffat profiles."""
        from scipy.optimize import least_squares
        
        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2 if yc is None else yc
        cx = (self.array.shape[1] - 1) / 2 if xc is None else xc

        _, _, sigma_x, sigma_y, theta0 = measure_shape(
            self.array, np.ones_like(self.array, dtype=bool))
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2

        params = default_params.copy()
        params.update({
            'fwhm_x': 2.355 * sigma_x,
            'fwhm_y': 2.355 * sigma_y,
            'theta': theta0,
            'xc': cx,
            'yc': cy,
            'flux': self.array.sum()  # Initial flux estimate
        })

        # Build optimization parameter list and mapping
        free_list = [p.strip() for p in free_params.split(',')]
        opt_params = []
        param_map = {}
        bounds_lower, bounds_upper = [], []
        
        for param in free_list:
            if param == 'fwhm':  # Special case for symmetric fwhm
                # Use fwhm_x as the initial value for symmetric fitting
                opt_params.append(params['fwhm_x'])
                param_map[param] = len(opt_params) - 1
                bounds_lower.append(1e-3)
                bounds_upper.append(np.inf)
            elif param.startswith('fwhm') and param in params:
                opt_params.append(params[param])
                param_map[param] = len(opt_params) - 1
                bounds_lower.append(1e-3)
                bounds_upper.append(np.inf)
            elif param in params:
                opt_params.append(params[param])
                param_map[param] = len(opt_params) - 1
                
                # Set bounds based on parameter type
                if param == 'beta':
                    bounds_lower.append(0.5)
                    bounds_upper.append(20.0)
                elif param == 'theta':
                    bounds_lower.append(-np.pi / 2)
                    bounds_upper.append(np.pi / 2)
                elif param in ['xc', 'yc']:
                    max_val = self.array.shape[1 if param == 'xc' else 0] - 1
                    bounds_lower.append(0)
                    bounds_upper.append(max_val)
                elif param == 'flux':
                    bounds_lower.append(1e-10)
                    bounds_upper.append(np.inf)

        def residual(p):
            # Map optimization parameters back to model parameters
            current_params = params.copy()
            
            for param_name, idx in param_map.items():
                if param_name == 'fwhm':  # Symmetric case
                    current_params['fwhm_x'] = current_params['fwhm_y'] = p[idx]
                else:
                    current_params[param_name] = p[idx]
            
            model = model_func(self.array.shape, **current_params)
            return (model - self.array).ravel()

        result = least_squares(residual, opt_params, bounds=(bounds_lower, bounds_upper))
        
        # Update parameters with fitted values
        for param_name, idx in param_map.items():
            if param_name == 'fwhm':  # Symmetric case
                fwhm_val = float(result.x[idx])
                params['fwhm_x'] = params['fwhm_y'] = fwhm_val
            else:
                params[param_name] = float(result.x[idx])
        
        # Return result with appropriate parameter names
        result_params = {}
        for field_name in result_class.__annotations__:
            if field_name != 'shape':  # Skip the shape field
                result_params[field_name] = params[field_name]
        
        # Add the shape information
        result_params['shape'] = self.array.shape
        
        return result_class(**result_params)

    def fit_moffat(self, free_params: str = "fwhm_x,fwhm_y,beta,theta,flux", 
                   xc: float = None, yc: float = None) -> MoffatFit:
        from .utils import moffat
        
        def model_func(shape, fwhm_x, fwhm_y, beta, theta, xc, yc, flux, **kwargs):
            return moffat(shape, fwhm_x, fwhm_y, beta, theta, x0=xc, y0=yc, flux=flux)
        
        return self._fit_profile(
            model_func, {'beta': 2.5}, free_params, xc, yc, MoffatFit
        )

    def fit_gaussian(self, free_params: str = "fwhm_x,fwhm_y,theta,flux",
                     xc: float = None, yc: float = None) -> GaussianFit:
        from .utils import gaussian
        
        def model_func(shape, fwhm_x, fwhm_y, theta, xc, yc, flux, **kwargs):
            return gaussian(shape, fwhm_x, fwhm_y, theta, x0=xc, y0=yc, flux=flux)
        
        return self._fit_profile(
            model_func, {}, free_params, xc, yc, GaussianFit
        )


def psf_matching_kernel_basis(
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    basis: np.ndarray,
    *,
    recenter: bool = False,
) -> np.ndarray:
    """Match ``psf_hi`` to ``psf_lo`` using basis function fitting."""

    kernel, _ = fit_kernel_fourier(psf_hi, psf_lo, basis)
    if recenter:
        cy = (kernel.shape[0] - 1) / 2
        cx = (kernel.shape[1] - 1) / 2
        ycen, xcen = centroid_quadratic(kernel, xpeak=cx, ypeak=cy, fit_boxsize=5)
        if not np.isnan(ycen) and not np.isnan(xcen):
            kernel = shift(kernel, (cy - ycen, cx - xcen), order=3, mode="nearest")
        else:
            logger.warning("Centroiding failed, kernel not recentered.")
    return kernel


from pathlib import Path
import re

# ---------------------------------------------------------------------
# Minimal EffectivePSF implementation (JWST STDPSF)
# ---------------------------------------------------------------------
# @@@ change this to an overloaded astropy PSF gridded model 
class EffectivePSF:

    def __init__(self, **kwargs):
        self.epsf = OrderedDict()
        self.extended_epsf = {}
        self.extended_N = None
#        if kwargs.get("jwst_stdpsf", True):
#            self.load_jwst_stdpsf()

    def load_jwst_stdpsf(
        self,
        miri_filters=None,
        nircam_sw_filters=None,
        nircam_sw_detectors=None,
        nircam_lw_filters=None,
        nircam_lw_detectors=None,
        miri_extended=True,
        clip_negative=False,
        local_dir=None,
        filter_pattern=None,
        use_astropy_cache=True,
        verbose=False,
    ):
        """Download JWST STDPSF models."""

        # If local_dir is specified, use it to find files
        if local_dir is not None and filter_pattern is not None:
            self.filter_pattern = filter_pattern
            p = Path(local_dir)
            files_dir = list(p.rglob('*.fits'))
            #rx = re.compile(filter_pattern)
            rx = re.compile(f"{filter_pattern}(?!_EXTENDED)")
            files = [f for f in files_dir if rx.search(os.path.basename(f))]
            for f in files:
                with fits.open(f) as im:
                    h = im[0].header
                    if verbose:
                        hstr = (
                            f"{h.get('NAXIS1', '?')}x{h.get('NAXIS2', '?')}x{h.get('NAXIS3', '?')} "
                            f"{h.get('INSTRUME', '?')} "
                            f"{h.get('DETECTOR', '?')} "
                            f"{h.get('FILTER',   '?')} "
                            f"{float(h.get('MJD-AVG', 0.0)):6.1f}"
                        )
                        print(f"Loading {f} {hstr}")       
                    data = np.array([d.T for d in im[0].data]).T
                    if clip_negative:
                        data[data < 0] = 0
                    key = os.path.basename(f).split(".fits")[0]
                    self.epsf[key] = data
            return

        if miri_filters is None:
            miri_filters = [
                #                "F560W",
                "F770W",
                # "F1000W",
                # "F1130W",
                # "F1280W",
                # "F1500W",
                # "F1800W",
                # "F2100W",
                # "F2550W",
            ]
        if nircam_sw_filters is None:
            nircam_sw_filters = ["F200W"]
        if nircam_sw_detectors is None:
            nircam_sw_detectors = [
                "A1",
                "A2",
                "A3",
                "A4",
                "B1",
                "B2",
                "B3",
                "B4",
            ]
        if nircam_lw_filters is None:
            nircam_lw_filters = ["F444W"]
        if nircam_lw_detectors is None:
            nircam_lw_detectors = ["AL", "BL"]

        base = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/"
        miri_path = ("MIRI/EXTENDED/STDPSF_MIRI_{filter}_EXTENDED.fits"
                     if miri_extended else "MIRI/STDPSF_MIRI_{filter}.fits")

        for filt in miri_filters:
            url = base + miri_path.format(filter=filt)
            try:
                file_obj = download_file(url, cache=use_astropy_cache)
                with fits.open(file_obj) as im:
                    data = np.array([d.T for d in im[0].data]).T
                    if clip_negative:
                        data[data < 0] = 0
                    key = os.path.basename(url.split(".fits")[0])
                    self.epsf[key] = data
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        sw_path = "NIRCam/SWC/{filter}/STDPSF_NRC{detector}_{filter}.fits"
        for filt in nircam_sw_filters:
            for det in nircam_sw_detectors:
                url = base + sw_path.format(filter=filt, detector=det)
                try:
                    file_obj = download_file(url, cache=use_astropy_cache)
                    with fits.open(file_obj) as im:
                        data = np.array([d.T for d in im[0].data]).T
                        if clip_negative:
                            data[data < 0] = 0
                        key = os.path.basename(url.split(".fits")[0])
                        self.epsf[key] = data
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

        lw_path = "NIRCam/LWC/STDPSF_NRC{detector}_{filter}.fits"
        for filt in nircam_lw_filters:
            for det in nircam_lw_detectors:
                url = base + lw_path.format(filter=filt, detector=det)
                try:
                    file_obj = download_file(url, cache=use_astropy_cache)
                    with fits.open(file_obj) as im:
                        data = np.array([d.T for d in im[0].data]).T
                        if clip_negative:
                            data[data < 0] = 0
                        key = os.path.basename(url.split(".fits")[0])
                        key = key.replace(f"{det}_", f"{det}ONG_")
                        self.epsf[key] = data
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

    # do this with PSFgriddedmodel.eval
    # and change hardcoded depenendence on grid size and detector oversampling 
    # --- PSF evaluation -------------------------------------------------
    def get_at_position(self, x, y, filter, rot90=0):
        """Interpolate the ePSF grid to a detector position."""
        epsf = self.epsf[filter]

        self.eval_psf_type = "HST/Optical"

        if  "MIRI" in filter:
            self.eval_psf_type = "MIRI"
            ndet = int(np.sqrt(epsf.shape[2]))
            rx = np.interp(x, [1, 358, 1032], [1, 2, 3]) - 1
            ry = np.interp(y, [1, 512, 1024], [1, 2, 3]) - 1
            nx = np.clip(int(rx), 0, 2)
            ny = np.clip(int(ry), 0, 2)
            fx = rx - nx
            fy = ry - ny
            if ndet == 1:
                psf_xy = epsf[:, :, 0]
            else:
                psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * ndet]
                psf_xy += fx * (1 - fy) * epsf[:, :, nx + 1 + ny * ndet]
                psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * ndet]
                psf_xy += fx * fy * epsf[:, :, nx + 1 + (ny + 1) * ndet]
            psf_xy = psf_xy.T

        elif "NRC" in filter:
            self.eval_psf_type = "NRC"
            # ndet = int(np.sqrt(epsf.shape[2]))
            # rx = np.interp(x, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5]) - 1
            # ry = np.interp(y, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5]) - 1
            # nx = np.clip(int(rx), 0, 4)
            # ny = np.clip(int(ry), 0, 4)
            # fx = rx - nx
            # fy = ry - ny
            # if ndet == 1:
            #     psf_xy = epsf[:, :, 0]
            # else:
            #     print(filter, ndet, nx, ny)
            #     print(filter rx, ry, fx, fy, nx + (ny + 1) * ndet)
            #     psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * ndet]
            #     psf_xy += fx * (1 - fy) * epsf[:, :, nx + 1 + ny * ndet]
            #     psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * ndet]
            #     psf_xy += fx * fy * epsf[:, :, nx + 1 + (ny + 1) * ndet]
            # psf_xy = psf_xy.T
            # # Use grid-agnostic robust interpolation
            xk = [0, 512, 1024, 1536, 2048]
            yk = [0, 512, 1024, 1536, 2048]
            nxps, nyps = len(xk), len(yk)
            ndet = nxps  # for square grid

            # Fractional grid indices
            rx = np.interp(x, xk, np.arange(nxps))
            ry = np.interp(y, yk, np.arange(nyps))
            ix = int(np.floor(rx))
            iy = int(np.floor(ry))
            fx = rx - ix
            fy = ry - iy

            # Robustly clip indices so ix+1, iy+1 are in bounds
            ix0 = np.clip(ix, 0, nxps - 2)
            iy0 = np.clip(iy, 0, nyps - 2)
            ix1 = ix0 + 1
            iy1 = iy0 + 1

            # Index into the cube
            psf_xy = (
                (1 - fx) * (1 - fy) * epsf[:, :, ix0 + iy0 * ndet]
                + fx * (1 - fy) * epsf[:, :, ix1 + iy0 * ndet]
                + (1 - fx) * fy * epsf[:, :, ix0 + iy1 * ndet]
                + fx * fy * epsf[:, :, ix1 + iy1 * ndet]
            )
            psf_xy = psf_xy.T
        else:
            psf_xy = epsf[:, :, 0]

        if rot90 != 0:
            psf_xy = np.rot90(psf_xy, rot90)

        return psf_xy

    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        """Evaluate the PSF at sub‑pixel offsets."""
        from scipy.ndimage import map_coordinates

        if self.eval_psf_type in ["WFC3/IR", "HST/Optical"]:
            ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
            coords = np.array([50 + 4 * dx[ok], 50 + 4 * dy[ok]])
        else:
            sh = psf_xy.shape
            size = (sh[0] - 1) // 4
            x0 = size * 2
            cen = (x0 - 1) // 2
            ok = (np.abs(dx) <= cen) & (np.abs(dy) <= cen)
            coords = np.array([x0 + 4 * dx[ok], x0 + 4 * dy[ok]])

        interp_map = map_coordinates(psf_xy, coords, order=3)
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map

        if extended_data is not None:
            ok = np.abs(dx) < self.extended_N
            ok &= np.abs(dy) < self.extended_N
            x0 = self.extended_N
            coords = np.array([x0 + dy[ok], x0 + dx[ok]])
            out[ok] += map_coordinates(extended_data, coords, order=0)

        return out



# ---------------------------------------------------------------------
# Basic WCS utilities
# ---------------------------------------------------------------------
def get_wcs_pscale(wcs, set_attribute=True):
    """Pixel scale in arcsec from a ``WCS`` object."""
    from numpy.linalg import det

    if isinstance(wcs, fits.Header):
        wcs = WCS(wcs, relax=True)

    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
        detv = det(wcs.wcs.cd)
    else:
        detv = det(wcs.wcs.pc)

    pscale = np.sqrt(np.abs(detv)) * 3600.0
    if set_attribute:
        wcs.pscale = pscale
    return pscale


def to_header(wcs, add_naxis=True, relax=True, key=None):
    """Convert WCS to a FITS header with a few extra keywords."""
    hdr = wcs.to_header(relax=relax, key=key)
    if add_naxis:
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs.pixel_shape[0]
            hdr["NAXIS2"] = wcs.pixel_shape[1]
        elif hasattr(wcs, "_naxis1"):
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs._naxis1
            hdr["NAXIS2"] = wcs._naxis2

    if hasattr(wcs.wcs, "cd"):
        for i in [0, 1]:
            for j in [0, 1]:
                hdr[f"CD{i + 1}_{j + 1}"] = wcs.wcs.cd[i][j]

    if hasattr(wcs, "sip") and wcs.sip is not None:
        hdr["SIPCRPX1"], hdr["SIPCRPX2"] = wcs.sip.crpix
    return hdr


def get_slice_wcs(wcs, slx, sly):
    """Slice a WCS while propagating SIP and distortion keywords."""
    nx = slx.stop - slx.start
    ny = sly.stop - sly.start
    swcs = wcs.slice((sly, slx))

    if hasattr(swcs, "_naxis1"):
        swcs.naxis1 = swcs._naxis1 = nx
        swcs.naxis2 = swcs._naxis2 = ny
    else:
        swcs._naxis = [nx, ny]
        swcs._naxis1 = nx
        swcs._naxis2 = ny

    if hasattr(swcs, "sip") and swcs.sip is not None:
        for c in [0, 1]:
            swcs.sip.crpix[c] = swcs.wcs.crpix[c]

    acs = [4096 / 2, 2048 / 2]
    dx = swcs.wcs.crpix[0] - acs[0]
    dy = swcs.wcs.crpix[1] - acs[1]
    for ext in ["cpdis1", "cpdis2", "det2im1", "det2im2"]:
        if hasattr(swcs, ext):
            extw = getattr(swcs, ext)
            if extw is not None:
                extw.crval[0] += dx
                extw.crval[1] += dy
                setattr(swcs, ext, extw)
    return swcs




# ---------------------------------------------------------------------
# Drizzle PSF class
# ---------------------------------------------------------------------

class DrizzlePSF:

    def __init__(
        self,
        flt_files=None,
        info=None,
        driz_image=None,
        driz_hdu=None,
        full_flt_weight=True,
        csv_file=None,
        epsf_obj=None,
    ):

        import warnings
        from astropy.wcs import FITSFixedWarning
        warnings.simplefilter('ignore', FITSFixedWarning)

        if info is None:
            info = self.read_wcs_csv(driz_image, csv_file=csv_file)

        self.flt_keys, self.wcs, self.footprint, self.hdrs = info
        self.flt_files = list({k[0] for k in self.flt_keys})

        if epsf_obj is None:
#            epsf_obj = NEffectivePSF()
            epsf_obj = EffectivePSF()
        self.epsf_obj = epsf_obj

        if driz_hdu is None:
            self.driz_image = driz_image
            self.driz_header = fits.getheader(driz_image)
        else:
            self.driz_image = driz_image
            self.driz_header = driz_hdu.header
            
        self.driz_wcs = WCS(self.driz_header)
        self.driz_pscale = get_wcs_pscale(self.driz_wcs)
        self.driz_wcs.pscale = self.driz_pscale
        self.driz_footprint = Polygon(self.driz_wcs.calc_footprint())

        self._next_odd_int = lambda x: int(round(x)) | 1


    # ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    # WCS information from CSV
    # ---------------------------------------------------------------------
    @staticmethod
    def read_wcs_csv(drz_file:str, csv_file=None):
        """Read exposure WCS info from a CSV table."""
        if csv_file is None:
            csv_file = (
                drz_file.split("_drz_sci")[0].split("_drc_sci")[0].split("_sci")[0] + "_wcs.csv"
            )
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file {csv_file} not found")

        tab = Table.read(csv_file, format="csv")
        flt_keys = []
        wcs_dict = {}
        footprints = {}
        hdrs = {}

        for row in tab:
            key = (row["file"], row["ext"])
            hdr = fits.Header()
            for col in tab.colnames:
                hdr[col] = row[col]

            wcs = WCS(hdr, relax=True)
            get_wcs_pscale(wcs)
            wcs.expweight = hdr.get("EXPTIME", 1)

            flt_keys.append(key)
            hdrs[key] = hdr
            wcs_dict[key] = wcs
            footprints[key] = Polygon(wcs.calc_footprint())

        return flt_keys, wcs_dict, footprints, hdrs

    @staticmethod
    def _get_empty_driz(wcs):
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            sh = wcs.pixel_shape[::-1]
        else:
            if (not hasattr(wcs, "_naxis1")) and hasattr(wcs, "_naxis"):
                wcs._naxis1, wcs._naxis2 = wcs._naxis
            sh = (wcs._naxis2, wcs._naxis1)

        outsci = np.zeros(sh, dtype=np.float32)
        outwht = np.zeros(sh, dtype=np.float32)
        outctx = np.zeros(sh, dtype=np.int32)
        return outsci, outwht, outctx

    # always return odd size
    def get_driz_cutout(self,
                        ra,
                        dec,
                        size=None,
                        size_native=None,
                        recenter=False,
                        search_boxsize=11,
                        fit_boxsize=5,
                        cutout_data = None,                
                        verbose=False):
        """Return a drizzle Cutout2D, including WCS."""

        # default size to size of the ePSF model
        if size is None:
            if size_native is None:    # get from the first filter
                first_key, first_value = next(iter(self.epsf_obj.epsf.items()))
                size_native = first_value.shape[0] / 4  # 4x oversampling
                if verbose:
                    print(f"Using native size {size_native} from {first_key} assuming 4x oversampling.")

            size = size_native * self.wcs[self.flt_keys[0]].pscale / self.driz_pscale
            
        size_odd = self._next_odd_int(size)

        xc, yc = self.driz_wcs.world_to_pixel_values(ra, dec)

        if cutout_data is None:
            data = fits.getdata(self.driz_image)         
        else:
            data = cutout_data
        
        # if data is 3D cube or list of 2D images, take loop over and append to output list
        # check if data is a 2D array or a list of 2D arrays
        if not isinstance(data, list):
            if data.ndim == 2: 
                data = [data]

        # get accurate centroid from first image
        if recenter:
            xc, yc = centroid_quadratic(
                data[0], xpeak = xc, ypeak = yc,
                fit_boxsize=fit_boxsize,
                search_boxsize=search_boxsize
            )

        # get cutouts for all images
        cutout_list = []
        for data_i in data:
            cutout = Cutout2D(
                data_i,
                (xc, yc),
                (size_odd, size_odd),
                wcs=self.driz_wcs,
                mode="partial",
                fill_value=0.0,
                copy=True,
            )
            cutout_list.append(cutout)

        return cutout_list if len(cutout_list) > 1 else cutout_list[0]
                

    # ---------------------------------------------------------------
    def get_psf(
        self,
        ra,
        dec,
        filter=None,
        pixfrac=0.1,
        kernel="point",
        verbose=False,
        wcs_slice=None,
        get_extended=True,
        get_weight=False,
        ds9=None,
        npix=None,
        renormalize=True,
        xphase=0,
        yphase=0,
        taper_alpha=0.05, # radial percent of tapering
        return_hdul=False,
    ):
        """Drizzle a PSF model at ``ra``, ``dec`` onto ``wcs_slice``."""
        if wcs_slice is None:
            wcs_slice = self.driz_wcs.copy()
        
        # default: adopt the filter pattern used to load the ePSF models 
        if filter is None:
            filter = self.epsf_obj.filter_pattern

        outsci, outwht, outctx = self._get_empty_driz(wcs_slice)

        tukey_taper = TukeyWindow(alpha=0.05)(outsci.shape)

        if npix is None:    
            # Calculate npix based on the WCS pixel scale            
            N = outsci.shape[0] // 2
            npix = int(np.ceil((N * self.driz_pscale / self.wcs[self.flt_keys[0]].pscale)))            

        pix = np.arange(-npix, npix + 1)
        for key in self.flt_keys:
            if self.footprint[key].contains(Point(ra, dec)):
                file, ext = key

                xy = self.wcs[key].all_world2pix([[ra, dec]], 0)[0]

                xyp = np.asarray(xy, dtype=int)
                dx = xy[0] - int(xy[0]) + xphase
                dy = xy[1] - int(xy[1]) + yphase
                chip_offset = 2051 if ext == 2 else 0

                # here get the riġht inst, dector, and MJD
                # for NIRCam select detector from flt file name if the filter is a regexp
                # do this with a more robust lookup. Parse the file name into a instrument / detector
                # @@ then pull the PSF from the ePSF object 
                if 'NRC..' in filter:
                    det = Path(file).stem.split('_')[-2][0:5].upper()
                    det = det.replace('L','5')
                    flt_filter = filter.replace('NRC..', det)
                else:
                    flt_filter = filter

                if verbose:
                    print(   f"Position: {xy}, Filter: {flt_filter}, in frame: {file}[SCI,{ext}]" )

                psf_xy = self.epsf_obj.get_at_position(xy[0],
                                                       xy[1] + chip_offset,
                                                       filter=flt_filter)
                yp, xp = np.meshgrid(pix - dy, pix - dx, indexing="ij")
                extended_data = (self.epsf_obj.extended_epsf.get(flt_filter)
                                 if get_extended else None)
                psf = self.epsf_obj.eval_ePSF(psf_xy,  xp,  yp,
                                              extended_data=extended_data)

                flt_weight = self.wcs[key].expweight
                N = npix
                slx = slice(xyp[0] - N, xyp[0] + N + 1)
                sly = slice(xyp[1] - N, xyp[1] + N + 1)
                if hasattr(flt_weight, "ndim") and flt_weight.ndim == 2:
                    wslx = slice(xyp[0] - N + 32, xyp[0] + N + 1 + 32)
                    wsly = slice(xyp[1] - N + 32, xyp[1] + N + 1 + 32)
                    flt_weight = self.wcs[key].expweight[wsly, wslx]

                psf_wcs = get_slice_wcs(self.wcs[key], slx, sly)
                psf_wcs.pscale = get_wcs_pscale(self.wcs[key])

                adrizzle.do_driz(
                    psf,
                    psf_wcs,
                    (psf * 0 + flt_weight).astype(outwht.dtype),
                    wcs_slice,
                    outsci,
                    outwht,
                    outctx,
                    1.0,
                    "cps",
                    1,
                    wcslin_pscale=1.0,
                    uniqid=1,
                    pixfrac=pixfrac,
                    kernel=kernel,
                    fillval=0,
                    stepsize=10,
                    wcsmap=None,
                )

        # taper PSF to avoid discontinuities at the edges and ringing
        if taper_alpha is not None and taper_alpha > 0:
            # rtaper is maximum radial extent of drizzled footprint
            shape = int(np.sqrt((outwht > 0).sum()))
            tukey_taper = pad_to_shape(
                TukeyWindow(alpha=taper_alpha)((shape,shape)),
                outsci.shape
            )                        
            outsci *= tukey_taper 

        if 'psf' not in locals():
            logger.warning(
                f"No PSF found, position possibly outside footprint for {ra}, {dec} in filter {filter}. Returning empty output."
            )
            scale = 1.0
        else:
            scale = psf.sum() / outsci.sum() if renormalize else 1.0
        
        if return_hdul is True:
            return fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(data=outsci * scale, header=to_header(wcs_slice))
            ])
        else:
            return outsci * scale

    def get_psf_radec(
        self,
        positions: list[tuple[float, float]],
        *,
        filter: str | None = None,
        size: int,
        verbose: bool = False,
    ) -> np.ndarray:
        """Return a cube of drizzled PSFs evaluated at given coordinates.

        Parameters
        ----------
        positions : list of tuple(float, float)
            World coordinate pairs ``(ra, dec)`` in degrees.
        filter : str
            Filter key or regular expression selecting the PSF model.
        size : int
            Cutout size in drizzle pixels for each PSF model.
        verbose : bool, optional
            Emit progress information if ``True``.

        Returns
        -------
        np.ndarray
            Array of shape ``(Npos, size, size)`` containing the drizzled PSFs.
        """

        psf_cube: list[np.ndarray] = []
        for ra, dec in positions:
            cutout = self.get_driz_cutout(
                ra,
                dec,
                size=size,
                verbose=verbose,
                recenter=False,
                search_boxsize=11,                
            )

            psf = self.get_psf(
                ra=ra,
                dec=dec,
                filter=filter,
                wcs_slice=cutout.wcs,
                kernel=self.driz_header["KERNEL"],
                pixfrac=self.driz_header["PIXFRAC"],
                verbose=verbose,
                npix=size // 2,
            )
            psf_cube.append(psf)

        return np.asarray(psf_cube)
    
    def register(
        self,
        ra: float,
        dec: float,
        filter: str,
        size: int = 15,
        max_iterations: int = 3,
        convergence_threshold: float = 0.05,
        verbose: bool = False,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """Register a PSF model to match the data centroid.

        Parameters
        ----------
        cutout : `~astropy.nddata.Cutout2D`
            Data cutout used for registration.
        filter : str
            Filter key or regexp identifying the PSF model.
        max_iterations : int, optional
            Maximum number of centering iterations.
        convergence_threshold : float, optional
            Convergence threshold in pixels.
        verbose : bool, optional
            If ``True`` emit progress information through the logger.

        Returns
        -------
        position : tuple of float
            Final world coordinate position ``(ra, dec)``.
        data : ndarray
            Data cutout used for registration.
        psf_model : ndarray
            Registered PSF model array.
        """

        cutout = self.get_driz_cutout(ra, dec, size=size, recenter=True)

        xi, yi = cutout.input_position_cutout
        for i in range(max_iterations):
            ri, di = cutout.wcs.pixel_to_world_values(xi, yi)

            psf = self.get_psf(
                ra=ri,
                dec=di,
                filter=filter,
                wcs_slice=cutout.wcs,
                kernel=self.driz_header["KERNEL"],
                pixfrac=self.driz_header["PIXFRAC"],
                verbose=verbose,
            )

            xc, yc = centroid_quadratic(
                 psf,
                 xpeak=cutout.input_position_cutout[0],
                 ypeak=cutout.input_position_cutout[1],
                 fit_boxsize=5,
            )

            if not np.isnan(yc) and not np.isnan(xc):
                dx = cutout.input_position_cutout[0] - xc
                dy = cutout.input_position_cutout[1] - yc
                dr = np.hypot(dx, dy)
            else:
                logger.warning("Centroiding failed, psf not recentered.")

            if verbose:
                print(f"Iteration {i + 1}: Centroid box5 shift: {dx:.3f}, {dy:.3f}, dr= {dr:.3f}")
            xi += dx
            yi += dy

            if dr < convergence_threshold:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
        else:
            if verbose:
                print(f"Maximum iterations {max_iterations} reached")

  
        ri, di = cutout.wcs.pixel_to_world_values(xi, yi)
        return (ri, di), cutout.data, psf

# ------------------------------------------------------------------
# EffectivePSF  —  now grid-agnostic (MIRI & NIRCam)
# ------------------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
import os, re
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file

class NEffectivePSF:
    """
    Minimal JWST STDPSF loader/evaluator that *learns* the grid break-points
    (IPSFX## / JPSFY##) from every cube it opens, so it works with any SIAF
    release.
    """
    # ──────────────────────────────────────────────────────────────
    def __init__(self):
        self.epsf           = OrderedDict()   # key → (Ny, Nx, Ncube)
        self.grid_breaks    = {}              # key → {'x':[...], 'y':[...]}
        self.extended_epsf  = {}              # unchanged
        self.extended_N     = None
        self.eval_psf_type  = None            # set in get_at_position

    # ──────────────────────────────────────────────────────────────
    # 1. LOAD CUBES ─ exactly as before, but store the break-points
    # ──────────────────────────────────────────────────────────────
    def _store_cube(self, key, hdu, clip_negative=False):
        """Helper: transpose to (Ny,Nx,N), clip <0, save cube & breaks."""
        dat = np.array([d.T for d in hdu.data]).T
        if clip_negative:
            dat[dat < 0] = 0
        self.epsf[key] = dat

        hdr  = hdu.header
        nxps = hdr.get('NXPSFS', 1)
        nyps = hdr.get('NYPSFS', 1)
        xk   = [hdr[f'IPSFX{i:02d}'] for i in range(1, nxps+1)]
        yk   = [hdr[f'JPSFY{i:02d}'] for i in range(1, nyps+1)]
        self.grid_breaks[key] = {'x': xk, 'y': yk}

    def load_jwst_stdpsf(
        self,
        *,
        clip_negative=False,
        local_dir=None,
        filter_pattern=None,
        verbose=False,
    ):
        """Load cubes from STScI site *or* a local directory (unchanged API)."""
        # ─── Local directory mode ───────────────────────────────────────
        if local_dir and filter_pattern:
            self.filter_pattern = filter_pattern
            regex = re.compile(filter_pattern, re.IGNORECASE)
            for fp in Path(local_dir).rglob("*.fits"):
                if regex.search(fp.name):
                    with fits.open(fp) as hdul:
                        if verbose: print(f"Loading {fp}")
                        key = fp.stem
                        self._store_cube(key, hdul[0], clip_negative)
            return

        # # ─── Remote STScI buckets ───────────────────────────────────────
        # base = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/"
        # get = lambda url: download_file(url, cache=use_astropy_cache)
        # # ---- MIRI ----
        # miri_fmt = ("MIRI/EXTENDED/STDPSF_MIRI_{filt}_EXTENDED.fits"
        #             if miri_extended else "MIRI/STDPSF_MIRI_{filt}.fits")
   
    # ──────────────────────────────────────────────────────────────
    # 2. GET AT POSITION ─ use stored break-points, not literals
    # ──────────────────────────────────────────────────────────────
    def get_at_position(self, x, y, filter, rot90=0):
        """Return the oversampled PSF at (x,y) detector coords."""
        epsf = self.epsf[filter]            # cube (Ny,Nx,N)
        br   = self.grid_breaks[filter]     # {'x': [...], 'y': [...]}

        # Determine flavour
        self.eval_psf_type = "HST/Optical"
        if  'MIRI' in  filter: self.eval_psf_type = "MIRI"
        if  'NRC' in  filter: self.eval_psf_type = "NRC"

        # ---- generic 2×2 (MIRI) or 3×3 / 5×5 (NIRCam) bilinear blend
        xk, yk = br['x'], br['y']
        nxps, nyps = len(xk), len(yk)
        ndet = int(np.sqrt(epsf.shape[2]))  # 3×3 → 3 etc.

        # 0-based fractional indices within the grid
        rx = np.interp(x, xk, np.arange(nxps)) - 0
        ry = np.interp(y, yk, np.arange(nyps)) - 0
        ix, iy  = np.clip(rx.astype(int), 0, nxps-2), np.clip(ry.astype(int), 0, nyps-2)
        fx, fy  = rx - ix, ry - iy

        # Bilinear combination
        psf_xy  = (1-fx)*(1-fy)*epsf[:,:, ix   + iy   *ndet]
        psf_xy +=   fx *(1-fy)*epsf[:,:, ix+1 + iy   *ndet]
        psf_xy += (1-fx)*  fy *epsf[:,:, ix   +(iy+1)*ndet]
        psf_xy +=   fx *  fy *epsf[:,:, ix+1 +(iy+1)*ndet]
        psf_xy  = psf_xy.T          # your historical transpose

        if rot90:
            psf_xy = np.rot90(psf_xy, rot90)

        return psf_xy

    # ──────────────────────────────────────────────────────────────
    # 3. eval_ePSF unchanged
    # ──────────────────────────────────────────────────────────────
    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        from scipy.ndimage import map_coordinates

        if self.eval_psf_type in ("WFC3/IR", "HST/Optical"):
            ok = (np.abs(dx)<=12.5) & (np.abs(dy)<=12.5)
            coords = np.array([50 + 4*dx[ok], 50 + 4*dy[ok]])
        else:
            sz   = (psf_xy.shape[0]-1)//4
            x0   = sz*2
            cen  = (x0-1)//2
            ok   = (np.abs(dx)<=cen) & (np.abs(dy)<=cen)
            coords = np.array([x0 + 4*dx[ok], x0 + 4*dy[ok]])

        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = map_coordinates(psf_xy, coords, order=3)

        # optional extended halo
        if extended_data is not None:
            ok2 = (np.abs(dx)<self.extended_N) & (np.abs(dy)<self.extended_N)
            coords = np.array([self.extended_N+dy[ok2],
                               self.extended_N+dx[ok2]])
            out[ok2] += map_coordinates(extended_data, coords, order=0)
        return out
