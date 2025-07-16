"""Point spread function utilities.

This module provides a :class:`PSF` class which wraps a pixel grid
representation of a point spread function. Instances can be created from
analytic profiles (Moffat, Gaussian) or directly from a user supplied
array. A method is included to compute a matching kernel between two PSFs.
"""

from __future__ import annotations

from collections import OrderedDict

import os
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import shift as nd_shift
from dataclasses import dataclass
from shapely.geometry import Point
from drizzlepac import adrizzle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.data import download_file
from photutils.psf import matching
from photutils.psf.matching import TukeyWindow
from photutils.centroids import centroid_quadratic

from .utils import measure_shape, get_wcs_pscale, get_slice_wcs, to_header
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

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

        kernel = psf_matching_kernel(psf_hi,
                                     psf_lo,
                                     window=window,
                                     recenter=recenter)
        return kernel

    def fit_moffat(self,
                   free_params: str = "fwhm_x,fwhm_y,beta,theta") -> MoffatFit:
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
        _, _, sigma_x, sigma_y, theta0 = measure_shape(
            self.array, np.ones_like(self.array, dtype=bool))
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

            model = moffat(self.array.shape,
                           current_params['fwhm_x'],
                           current_params['fwhm_y'],
                           current_params['beta'],
                           current_params['theta'],
                           x0=cx,
                           y0=cy)
            return (model - self.array).ravel()

        result = least_squares(residual, p0, bounds=bounds)

        # Update fitted parameters
        for i, param_name in param_map.items():
            params[param_name] = float(result.x[i])

        return MoffatFit(params['fwhm_x'], params['fwhm_y'], params['beta'],
                         params['theta'])

    def fit_gaussian(self,
                     free_params: str = "fwhm_x,fwhm_y,theta") -> GaussianFit:
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
        _, _, sigma_x, sigma_y, theta0 = measure_shape(
            self.array, np.ones_like(self.array, dtype=bool))
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

            model = gaussian(self.array.shape,
                             current_params['fwhm_x'],
                             current_params['fwhm_y'],
                             current_params['theta'],
                             x0=cx,
                             y0=cy)
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
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    *,
    window: object | None = None,
    recenter: bool = True,
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
    recenter : bool, optional
        If ``True`` the resulting kernel is shifted to its centroid using
        bicubic interpolation. Defaults to ``True``.

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
    kernel = np.asarray(kernel)
    if recenter:
        ycen, xcen = centroid_quadratic(kernel, fit_boxsize=5)
        cy = (kernel.shape[0] - 1) / 2
        cx = (kernel.shape[1] - 1) / 2
        kernel = nd_shift(kernel, (cy - ycen, cx - xcen), order=3, mode="nearest")
    return kernel


from pathlib import Path
import re

# ---------------------------------------------------------------------
# Minimal EffectivePSF implementation (JWST STDPSF)
# ---------------------------------------------------------------------
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
    ):
        """Download JWST STDPSF models."""

        # If local_dir is specified, use it to find files
        if local_dir is not None and filter_pattern is not None:
            p = Path(local_dir)
            files_dir = list(p.rglob('*.fits'))
            rx = re.compile(filter_pattern)
            files = [f for f in files_dir if rx.search(os.path.basename(f))]
            for f in files:
                with fits.open(f) as im:
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

    # --- PSF evaluation -------------------------------------------------
    def get_at_position(self, x, y, filter, rot90=0):
        """Interpolate the ePSF grid to a detector position."""
        epsf = self.epsf[filter]
        psf_type = "HST/Optical"
        if filter.startswith("STDPSF_MIRI"):
            psf_type = "STDPSF_MIRI"
        elif filter.startswith("STDPSF_NRC"):
            psf_type = "STDPSF_NRC"

        self.eval_psf_type = psf_type

        if psf_type == "STDPSF_MIRI":
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

        elif psf_type == "STDPSF_NRC":
            ndet = int(np.sqrt(epsf.shape[2]))
            rx = np.interp(x, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5]) - 1
            ry = np.interp(y, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5]) - 1
            nx = np.clip(int(rx), 0, 4)
            ny = np.clip(int(ry), 0, 4)
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
        if info is None:
            info = read_wcs_csv(driz_image, csv_file=csv_file)

        self.flt_keys, self.wcs, self.footprint = info
        self.flt_files = list({k[0] for k in self.flt_keys})

        if epsf_obj is None:
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

    # ---------------------------------------------------------------
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

    def get_driz_cutout(self,
                        ra,
                        dec,
                        size=None,
                        N=None,
                        size_native=31,
                        odd=True):
        """Return a drizzle cutout or its WCS."""
        xy = self.driz_wcs.all_world2pix([[ra, dec]], 0)[0]
        xyp = np.asarray(np.round(xy), dtype=int)

        if size is None:
            N = int((size_native * self.wcs[self.flt_keys[0]].pscale /
                     self.driz_pscale) // 2)
        else:
            N = int(size // 2)

        size_psf = 2 * N + odd

        with fits.open(self.driz_image) as im:
            cutout = Cutout2D(
                im[0].data,
                SkyCoord(ra, dec, unit="deg"),
                (size_psf, size_psf),
                wcs=self.driz_wcs,
                mode="partial",
                fill_value=0.0,
                copy=True,
            )
        return cutout

    # ---------------------------------------------------------------
    def get_psf(
        self,
        ra,
        dec,
        filter,
        pixfrac=0.1,
        kernel="point",
        verbose=True,
        wcs_slice=None,
        get_extended=True,
        get_weight=False,
        ds9=None,
        npix=13,
        renormalize=True,
        xphase=0,
        yphase=0,
    ):
        """Drizzle a PSF model at ``ra``, ``dec`` onto ``wcs_slice``."""
        pix = np.arange(-npix, npix + 1)
        if wcs_slice is None:
            wcs_slice = self.driz_wcs.copy()

        outsci, outwht, outctx = self._get_empty_driz(wcs_slice)

        for key in self.flt_keys:
            if self.footprint[key].contains(Point(ra, dec)):
                file, ext = key

                xy = self.wcs[key].all_world2pix([[ra, dec]], 0)[0]

                xyp = np.asarray(xy, dtype=int)
                dx = xy[0] - int(xy[0]) + xphase
                dy = xy[1] - int(xy[1]) + yphase
                chip_offset = 2051 if ext == 2 else 0

                # for NIRCam select detector from flt file name
                if 'NRC*' in filter:
                    det = Path(file).stem.split('_')[-2][0:5].upper()
                    flt_filter = filter.replace('NRC*', det)
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
                psf = self.epsf_obj.eval_ePSF(psf_xy,
                                              xp,
                                              yp,
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

        scale = 1.0 / outsci.sum() * psf.sum() if renormalize else 1.0
        hdu = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=outsci * scale, header=to_header(wcs_slice))
        ])
        return hdu
