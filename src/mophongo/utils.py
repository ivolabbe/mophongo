"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import os
import copy
import numpy as np
import scipy
from scipy.ndimage import shift
from astropy.nddata import block_reduce

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from shapely.geometry import Polygon
from scipy.special import hermite
from scipy.interpolate import PchipInterpolator
from astropy.utils import lazyproperty
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Moffat1D
from photutils.profiles import RadialProfile, CurveOfGrowth
from photutils.centroids import centroid_quadratic, centroid_com

from scipy.special import eval_hermite      # physicists' Hermite
from photutils.psf import matching

import logging
logger = logging.getLogger(__name__)


def bin2d_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Block-average a 2-D array by an integer factor using ``block_reduce``.

    Trailing rows/columns that do not fit an exact ``k``×``k`` block are
    discarded, matching ``astropy.nddata.block_reduce`` semantics.
    """
    if k == 1:
        return arr

    return block_reduce(arr, k, func=np.mean)


def downsample_psf(psf: np.ndarray, k: int) -> np.ndarray:
    """Downsample a PSF by an integer factor, preserving the centroid.

    Parameters
    ----------
    psf : np.ndarray
        Input PSF array centred at ``(shape-1)/2``.
    k : int
        Integer binning factor.

    Returns
    -------
    np.ndarray
        Downsampled and re-centered PSF array.
    """
    if k == 1:
        return psf

    # only shift center if k a multiple of 2 and psf.shape is odd 
    if (k % 2 == 0) and (psf.shape[0] % 2 == 1):
        shift_hi = (k - 1) / 2.0
        psf = shift(
            psf,
            shift=(-shift_hi, -shift_hi),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return block_reduce(psf, k, func=np.mean)


def bin_factor_from_wcs(w_det: WCS, w_img: WCS, tol: float = 0.02) -> int:
    """Return the integer pixel-scale factor between two WCS objects.

    Parameters
    ----------
    w_det : WCS
        WCS of the detection image (higher resolution).
    w_img : WCS
        WCS of the target image.
    tol : float, optional
        Tolerance on the ratio between the scales to still be considered an
        integer. Defaults to 0.02.

    Returns
    -------
    int
        Integer binning factor ``k``. Always at least one.

    Raises
    ------
    ValueError
        If the pixel-scale ratio deviates from an integer by more than ``tol``.
    """
    s_det = proj_plane_pixel_scales(w_det)[0] * 3600.0
    s_img = proj_plane_pixel_scales(w_img)[0] * 3600.0
    ratio = s_img / s_det
    k = int(round(ratio))
    if abs(ratio - k) > tol:
        raise ValueError(
            f"Pixel-scale ratio {ratio:.3f} not within {tol*100:.1f}% of an integer – "
            "cannot block-average safely."
        )
    return max(k, 1)

def rebin_wcs(wcs: WCS, n: int) -> WCS:
    """
    Up‐ or down‐sample a WCS by a power of two, *exactly* preserving the
    tangent point (CRVALs), and updating CRPIX and NAXIS accordingly.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Original WCS.
    n : int
        Power of two to scale by:
          - n > 0 : down‐bin by 2**n  (pixels get larger)
          - n < 0 : up‐sample by 2**|n| (pixels get smaller)

    Returns
    -------
    new_wcs : astropy.wcs.WCS
        The rebinned WCS.
    """
    factor = 2 ** n
    new_wcs = copy.deepcopy(wcs)

    # — scale the CD or CDELT matrix
    if getattr(new_wcs.wcs, 'cd', None) is not None:
        new_wcs.wcs.cd = new_wcs.wcs.cd / factor
    else:
        new_wcs.wcs.cdelt = new_wcs.wcs.cdelt / factor

    # — shift CRPIX so that CRVAL stays fixed on the sky:
    #    new_crpix = (old_crpix - 0.5)/factor + 0.5
    old_crpix = new_wcs.wcs.crpix.copy()
    new_wcs.wcs.crpix = (old_crpix - 0.5) / factor + 0.5

    # — update the “NAXIS” so to_header() will emit the right shape
    #    (Astropy uses .pixel_shape if present, else _naxis1/_naxis2)
    if hasattr(new_wcs, 'pixel_shape') and new_wcs.pixel_shape is not None:
        ny, nx = new_wcs.pixel_shape
        new_wcs.pixel_shape = (int(ny // factor), int(nx // factor))
    else:
        # fallback into the private attributes
        if hasattr(new_wcs.wcs, '_naxis1'):
            new_wcs.wcs._naxis1 = int(new_wcs.wcs._naxis1 // factor)
            new_wcs.wcs._naxis2 = int(new_wcs.wcs._naxis2 // factor)
        if hasattr(new_wcs.wcs, '_naxis'):
            na = new_wcs.wcs._naxis
            new_wcs.wcs._naxis = [int(na[0] // factor), int(na[1] // factor)]

    # re‐initialize internally computed stuff
    new_wcs.wcs.set()

    return new_wcs


# model based stuff
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


def moffat(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Moffat PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy

    # Convert flux to amplitude analytically
    # For a Moffat profile: flux = amplitude * pi * alpha_x * alpha_y / (beta - 1)
    # where alpha = fwhm / (2 * sqrt(2^(1/beta) - 1))
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    amplitude = flux * (beta - 1) / (np.pi * alpha_x * alpha_y)

    psf = elliptical_moffat(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        beta,
        theta,
        x0,
        y0,
    )
    return psf


def gaussian(
    size: int | tuple[int, int],
    fwhm_x: float | None = None,
    fwhm_y: float | None = None,
    fwhm: float | None = None,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Gaussian PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    if fwhm is not None:
        if isinstance(fwhm, (list, tuple, np.ndarray)) and len(fwhm) == 2:
            fwhm_x, fwhm_y = fwhm
        else:
            fwhm_x = fwhm_y = fwhm

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy

    # Convert flux to amplitude analytically
    # For a Gaussian profile: flux = amplitude * 2 * pi * sigma_x * sigma_y
    # where sigma = fwhm / (2 * sqrt(2 * ln(2)))
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    amplitude = flux / (2 * np.pi * sigma_x * sigma_y)

    psf = elliptical_gaussian(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        theta,
        x0,
        y0,
    )
    return psf


def pad_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad array with zeros to center it in the target shape."""
    py = (shape[0] - arr.shape[0]) // 2
    px = (shape[1] - arr.shape[1]) // 2
    return np.pad(arr, ((py, shape[0] - arr.shape[0] - py), (px, shape[1] - arr.shape[1] - px)))


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    pad_before = (pad_y, pad_x)
    pad_after = (ky - 1 - pad_y, kx - 1 - pad_x)
    padded = np.pad(image, (pad_before, pad_after), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)



# wrapper around astropy matching kernel that handles padding and recentering
def matching_kernel(
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
        Window function passed to ``create_matching_kernel``. Defaults to SplitCosineBellWindow.
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

    if not np.isfinite(psf_hi).all():
        logger.warning(
            "psf 1 contains non-finite values, setting elements to zero ")
        psf_hi[~np.isfinite(psf_hi)] = 0.0

    if not np.isfinite(psf_lo).all():
        logger.warning(
            "psf 2 contains non-finite values, setting elements to zero ")
        psf_lo[~np.isfinite(psf_lo)] = 0.0

    if window is None:
         window = matching.SplitCosineBellWindow(alpha=0.4, beta=0.1)
#        window = matching.TukeyWindow(alpha=0.4)

    kernel = matching.create_matching_kernel(psf_hi, psf_lo, window=window)
    kernel = np.asarray(kernel)

    if not np.isfinite(kernel).all():
        logger.warning(
            "Kernel contains non-finite values, returning zero kernel.")
        return np.zeros_like(kernel)

    if recenter:
        cy = (kernel.shape[0] - 1) / 2
        cx = (kernel.shape[1] - 1) / 2
        ycen, xcen = centroid_quadratic(kernel,
                                        xpeak=cx,
                                        ypeak=cy,
                                        fit_boxsize=5)
        if np.isnan(ycen) or np.isnan(xcen):
            # fallback to centroid_com if quadratic fails
            ycen, xcen = centroid_com(kernel)

        if not np.isnan(ycen) and not np.isnan(xcen):
            kernel = shift(kernel, (cy - ycen, cx - xcen),
                           order=3,
                           mode="nearest")
        else:
            logger.warning("Centroiding failed, kernel not recentered.")

    return kernel


# ------------------------------------------------------------------
# 2. Fourier-space kernel fit
# ------------------------------------------------------------------
def fit_kernel_fourier(img_hi, img_lo, basis, method="lstsq"):
    """
    Solve  FFT(img_hi) * FFT(basis_k) * c_k  =  FFT(img_lo)
    and return a centred real-space kernel.
    
    Parameters
    ----------
    img_hi : array_like
        High-resolution input image
    img_lo : array_like  
        Low-resolution target image
    basis : array_like
        Basis functions (size, size, n_basis)
    method : {"lstsq", "nnls"}, optional
        Fitting method. "lstsq" uses standard least squares, 
        "nnls" uses non-negative least squares. Default is "lstsq".
    """
    from scipy.optimize import nnls

    n_pix = img_hi.size                    # = size²

    # 1.  Shift arrays so that the PSF/basis centre is at pixel (0,0) **before** FFT
    f_hi    = np.fft.fft2(np.fft.ifftshift(img_hi))
    f_lo    = np.fft.fft2(np.fft.ifftshift(img_lo))
    f_basis = np.fft.fft2(np.fft.ifftshift(basis, axes=(0, 1)), axes=(0, 1))

    # 2.  Build the least-squares matrix in Fourier space
    nb = basis.shape[-1]
    A  = (f_basis * f_hi[..., None] / n_pix).reshape(-1, nb)   # IDL normalisation
    b  = (f_lo / n_pix).ravel()

    if method == "lstsq":
        A_ri = np.vstack([A.real, A.imag])
        b_ri = np.concatenate([b.real, b.imag])
        coeffs, *_ = np.linalg.lstsq(A_ri, b_ri, rcond=None)
    elif method == "nnls":
        A_ri = np.vstack([A.real, A.imag])
        b_ri = np.concatenate([b.real, b.imag])
        coeffs, _ = nnls(A_ri, b_ri)
    else:
        raise ValueError(f"method must be 'lstsq' or 'nnls', got '{method}'")

    # 3.  Back to real space – already centred, so **no fftshift here**
    kernel = np.tensordot(basis, coeffs, axes=([-1], [0]))
    kernel /= kernel.sum()                 # final normalisation

    return kernel, coeffs

def regularized_pixel_kernel_central(psf_hi, psf_lo, kernel_size=20, alpha=0.01, method="ridge"):
    """
    Fit only the central pixels of the kernel with regularization.
    
    Parameters
    ----------
    psf_hi : array
        High-resolution PSF (input)
    psf_lo : array  
        Low-resolution PSF (target)
    kernel_size : int
        Size of central kernel region to fit (e.g., 20 for 20x20)
    alpha : float
        Regularization strength
    method : {"ridge", "smooth", "total_variation"}
        Type of regularization
    """
    from scipy.optimize import minimize
    from scipy.ndimage import laplace, sobel

    full_size = psf_hi.shape[0]

    # Create full kernel with central region to optimize
    def make_full_kernel(central_params):
        kernel = np.zeros((full_size, full_size))
        center = full_size // 2
        half_k = kernel_size // 2

        central_kernel = central_params.reshape((kernel_size, kernel_size))
        kernel[center-half_k:center+half_k, center-half_k:center+half_k] = central_kernel
        return kernel

    # Initial guess: delta function in center
    x0 = np.zeros(kernel_size * kernel_size)
    x0[len(x0)//2] = 1.0

    def objective(x):
        kernel = make_full_kernel(x)

        # Data fidelity term
        conv = convolve2d(psf_hi, kernel)
        data_term = np.sum((conv - psf_lo)**2)

        # Regularization on central region only
        central = x.reshape((kernel_size, kernel_size))
        if method == "ridge":
            reg_term = alpha * np.sum(central**2)
        elif method == "smooth":
            smooth = laplace(central)
            reg_term = alpha * np.sum(smooth**2)
        elif method == "total_variation":
            grad_x = sobel(central, axis=0)
            grad_y = sobel(central, axis=1)
            reg_term = alpha * np.sum(np.sqrt(grad_x**2 + grad_y**2))
        else:
            reg_term = 0

        return data_term + reg_term

    # Optimize with positivity constraint
    bounds = [(0, None) for _ in range(len(x0))]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    kernel = make_full_kernel(result.x)
    kernel /= kernel.sum()
    return kernel

def regularized_lstsq_kernel_central(psf_hi, psf_lo, kernel_size=20, alpha=0.01, smooth_type="laplacian"):
    """
    Direct least squares with regularization, fitting only central kernel pixels.
    """
    full_size = psf_hi.shape[0]
    center = full_size // 2
    half_k = kernel_size // 2

    # Only fit central region
    n_central = kernel_size * kernel_size

    # Build convolution matrix for central pixels only
    A = np.zeros((full_size * full_size, n_central))

    idx = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Create test kernel with single pixel
            kernel_test = np.zeros((full_size, full_size))
            ki = center - half_k + i
            kj = center - half_k + j
            kernel_test[ki, kj] = 1.0

            # Convolve and store
            conv = convolve2d(psf_hi, kernel_test)
            A[:, idx] = conv.flatten()
            idx += 1

    # Build regularization matrix for central region
    if smooth_type == "laplacian":
        L = np.zeros((n_central, n_central))
        for i in range(1, kernel_size-1):
            for j in range(1, kernel_size-1):
                idx = i * kernel_size + j
                L[idx, idx] = -4
                if i > 0: L[idx, (i-1)*kernel_size + j] = 1
                if i < kernel_size-1: L[idx, (i+1)*kernel_size + j] = 1
                if j > 0: L[idx, i*kernel_size + (j-1)] = 1
                if j < kernel_size-1: L[idx, i*kernel_size + (j+1)] = 1
    elif smooth_type == "identity":
        L = np.eye(n_central)
    else:
        L = np.zeros((n_central, n_central))

    # Solve regularized system
    AtA = A.T @ A
    LtL = L.T @ L
    b = psf_lo.flatten()
    Atb = A.T @ b

    central_flat = np.linalg.solve(AtA + alpha * LtL, Atb)

    # Construct full kernel
    kernel = np.zeros((full_size, full_size))
    central_kernel = central_flat.reshape((kernel_size, kernel_size))
    kernel[center-half_k:center+half_k, center-half_k:center+half_k] = central_kernel

    # Ensure positivity and normalize
    kernel = np.maximum(kernel, 0)
    kernel /= kernel.sum()

    return kernel

# ------------------------------------------------------------------
# 1. 2-D Gauss–Hermite basis (physicists' convention)
# ------------------------------------------------------------------
def gauss_hermite_basis(order: int, scales, size: int):
    """
    Return an (size, size, Nbasis) cube of 2-D Gauss–Hermite functions.
    H_{i}(x) H_{j}(y) e^{-(x²+y²)/2s²},   0 ≤ i+j ≤ order  for each scale s.
    Zeroth component is unit-sum, all others are zero-sum.
    """
    y, x = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2                        # geometric centre
    basis = []

    for s in scales:
        xn = (x - cx) / s
        yn = (y - cy) / s
        g  = np.exp(-0.5 * (xn**2 + yn**2))        # isotropic Gaussian

        for i in range(order + 1):
            for j in range(order + 1 - i):
                b = g * eval_hermite(i, xn) * eval_hermite(j, yn)

                if i == 0 and j == 0:
                    b /= b.sum()                   # unit DC
                else:
                    b -= b.mean()                  # kill residual DC

                basis.append(b)

    # Stack as (Ny, Nx, Nbasis)
    return np.stack(basis, axis=-1)


def multi_gaussian_basis(scales: list[float], size: int) -> np.ndarray:
    """Return a set of Gaussian basis functions with varying width."""

    gauss_list = [gaussian(size, s, s) for s in scales]
    basis = np.stack(gauss_list, axis=2)
    basis /= basis.sum(axis=(0, 1))
    return basis


def difference_of_gaussians_basis(scales: list[float], size: int) -> np.ndarray:
    """Return Difference-of-Gaussians (DoG) basis functions.

    Parameters
    ----------
    scales : list of float
        FWHM values for the Gaussian stack.  Must be in increasing order.
    size : int
        Output array size (``size`` \times ``size``).

    Notes
    -----
    If ``n`` scales are provided, ``n-1`` DoG modes are returned.  Each
    mode is the difference between successive Gaussians and has zero sum.
    """

    gauss_list = [gaussian(size, s, s) for s in scales]
    dog_list = [gauss_list[i + 1] - gauss_list[i] for i in range(len(scales) - 1)]
    basis = np.stack(dog_list, axis=2)
    basis -= basis.mean(axis=(0, 1))
    return basis


def gaussian_laguerre_basis(
    nmax: int, m_list: list[int], scales: list[float], size: int
) -> np.ndarray:
    """Return Gaussian-Laguerre (polar shapelet) basis functions.

    Parameters
    ----------
    nmax : int
        Maximum radial order ``n`` of the Laguerre polynomial.
    m_list : list of int
        Azimuthal orders ``m`` (typically ``[0, 2, 4]``).
    scales : list of float
        Gaussian ``FWHM`` values controlling the radial scale.
    size : int
        Output array size.
    """

    y, x = np.mgrid[:size, :size]
    cy = cx = (size - 1) / 2
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)

    basis = []
    for s in scales:
        sigma = s / (2 * np.sqrt(2 * np.log(2)))
        rsq = (r / sigma) ** 2
        g = np.exp(-0.5 * rsq)

        for n in range(nmax + 1):
            for m in m_list:
                if m > n or (n - m) % 2:
                    continue
                radial = scipy.special.genlaguerre((n - m) // 2, m)(rsq)
                b = g * radial * np.cos(m * theta)
                if n == 0 and m == 0:
                    b /= b.sum()
                else:
                    b -= b.mean()
                basis.append(b)

    return np.stack(basis, axis=-1)


def zernike_basis(
    nmax: int, m_list: list[int], sigma: float, size: int
) -> np.ndarray:
    """Return Zernike polynomial basis functions with a Gaussian envelope."""

    y, x = np.mgrid[:size, :size]
    cy = cx = (size - 1) / 2
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    rho = r / (size / 2)
    env = np.exp(-0.5 * (r / sigma) ** 2)

    def _zernike(n, m, rho):
        out = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            c = (
                (-1) ** k
                * scipy.special.factorial(n - k)
                / (
                    scipy.special.factorial(k)
                    * scipy.special.factorial((n + m) // 2 - k)
                    * scipy.special.factorial((n - m) // 2 - k)
                )
            )
            out += c * rho ** (n - 2 * k)
        return out

    basis = []
    for n in range(nmax + 1):
        for m in m_list:
            if m > n or (n - m) % 2:
                continue
            z = _zernike(n, m, rho) * np.cos(m * theta) * env
            if n == 0 and m == 0:
                z /= z.sum()
            else:
                z -= z.mean()
            basis.append(z)

    return np.stack(basis, axis=-1)


from scipy.interpolate import BSpline
def radial_bspline_basis(
    size: int, n_knots: int = 6, degree: int = 3, rmin: float = 0.5, rmax: float | None = None,
    spacing: str = "log", verbose: bool = False
) -> np.ndarray:
    """Radial B-spline basis on a radius grid.
    
    Parameters
    ----------
    size : int
        Output array size (size x size).
    n_knots : int, optional
        Number of knots for the B-spline basis. Default is 6.
    degree : int, optional
        Degree of the B-spline basis. Default is 3.
    rmin : float, optional
        Minimum radius for knot placement. Default is 0.5.
    rmax : float or None, optional
        Maximum radius for knot placement. If None, uses maximum image radius.
    spacing : {"log", "linear"}, optional
        Knot spacing type. Default is "log".
    verbose : bool, optional
        If True, print knot radii. Default is False.
    """

    from scipy.interpolate import BSpline

    y, x = np.mgrid[:size, :size]
    cy = cx = (size - 1) / 2
    r = np.hypot(x - cx, y - cy)

    # Use provided rmax or default to maximum radius
    if rmax is None:
        rmax = r.max()

    r = np.where(r == 0, rmin / 10, r)

    if spacing == "log":
        log_r = np.log10(r)
        knots = np.linspace(np.log10(rmin), np.log10(rmax), n_knots)
        if verbose:
            print("Knot radii:", 10**knots)
        t = np.pad(knots, (degree, degree), mode="edge")
        design = BSpline.design_matrix(log_r.ravel(), t, degree, extrapolate=True).toarray()
    elif spacing == "linear":
        knots = np.linspace(rmin, rmax, n_knots)
        if verbose:
            print("Knot radii:", knots)
        t = np.pad(knots, (degree, degree), mode="edge")
        design = BSpline.design_matrix(r.ravel(), t, degree, extrapolate=True).toarray()
    else:
        raise ValueError(f"spacing must be 'log' or 'linear', got '{spacing}'")

    basis = design.reshape(size, size, -1)
    for i in range(basis.shape[-1]):
        if i == 0:
            basis[:, :, i] /= basis[:, :, i].sum()
        else:
            basis[:, :, i] -= basis[:, :, i].mean()
    return basis

def positive_monotone_radial_bspline(
    size: int | tuple[int, int],
    *,
    n_knots: int = 8,
    degree: int  = 3,
    r_min: float = 0.5,
    r_max: float | None = None,
    log_spacing: bool = True,
) -> np.ndarray:
    """
    Returns a basis cube (ny, nx, n_basis) where every slice is
    - strictly ≥ 0
    - monotonically non-increasing with radius
    - the first slice integrates to 1, others to <1 (so flux can only be
      *moved outward*, never created or destroyed).
    """
    # ------------------------------------------------------------
    # radius map
    # ------------------------------------------------------------
    ny, nx = (size, size) if isinstance(size, int) else size
    y, x   = np.mgrid[:ny, :nx]
    r      = np.hypot(x-(nx-1)/2, y-(ny-1)/2)
    if r_max is None or r_max < r.max():
        r_max = r.max() + 1e-3
    r = np.where(r == 0, r_min/10, r)

    # ------------------------------------------------------------
    # knot vector in log or linear space *identical* to query domain
    # ------------------------------------------------------------
    if log_spacing:
        knots = np.log10(np.geomspace(r_min, r_max, n_knots))
        r_q   = np.log10(r.ravel())
    else:
        knots = np.linspace(r_min, r_max, n_knots)
        r_q   = r.ravel()

    t = np.pad(knots, (degree, degree), mode="edge")

    # ordinary (non-integrated) B-spline bumps  (Npix × Nb)
    dm = BSpline.design_matrix(r_q, t, degree, extrapolate=True).toarray()
    dm = dm.reshape(ny, nx, -1)

    # ------------------------------------------------------------
    # cumulative, positive & monotone
    # ------------------------------------------------------------
    cum = np.cumsum(dm, axis=-1)             # C_k(r) = Σ_{i≤k} B_i(r)

    # normalise: first mode unit integral, others zero-mean w.r.t. previous
    cum[..., 0] /= cum[..., 0].sum()         # DC, area = 1
    for k in range(1, cum.shape[-1]):
        delta = cum[..., k] - cum[..., k-1]
        delta -= delta.mean()                # remove tiny DC drift
        cum[..., k] = cum[..., k-1] + delta  # restore monotonicity

    return cum


def starlet_basis(size: int, n_scales: int = 5) -> np.ndarray:
    """Return à-trous starlet wavelet basis."""
    from scipy.ndimage import convolve

    delta = np.zeros((size, size))
    cy = cx = size // 2
    delta[cy, cx] = 1.0
    h_1d = np.array([1, 4, 6, 4, 1], dtype=float) / 16
    h = np.outer(h_1d, h_1d)

    c = delta
    basis = []
    for j in range(n_scales):
        if j == 0:
            kernel = h
        else:
            # Use scipy.ndimage.zoom with order=0 for dilation
            from scipy.ndimage import zoom
            step = 2 ** j
            # Create dilated kernel by upsampling with zeros
            kernel = np.kron(h, np.ones((step, step))) / (step * step)

        sm = convolve(c, kernel, mode="nearest")
        w = c - sm
        w -= w.mean()
        basis.append(w)
        c = sm
    return np.stack(basis, axis=-1)


def eigen_psf_basis(stack: np.ndarray, n_modes: int) -> np.ndarray:
    """Return data-driven eigen-PSF basis using SVD."""

    npsf, ny, nx = stack.shape
    U, S, Vt = np.linalg.svd(stack.reshape(npsf, ny * nx), full_matrices=False)
    basis = Vt[:n_modes].reshape(n_modes, ny, nx).transpose(1, 2, 0)
    for i in range(basis.shape[-1]):
        if i == 0:
            basis[:, :, i] /= basis[:, :, i].sum()
        else:
            basis[:, :, i] -= basis[:, :, i].mean()
    return basis


def powerlaw_basis(
    slopes: list[float], size: int, sigma: float | None = None, r0: float = 1.0
) -> np.ndarray:
    """Return power-law radial basis functions with Gaussian taper."""

    y, x = np.mgrid[:size, :size]
    cy = cx = (size - 1) / 2
    r = np.hypot(x - cx, y - cy)
    if sigma is None:
        sigma = size / 3
    taper = np.exp(-0.5 * (r / sigma) ** 2)

    basis = []
    for s in slopes:
        b = ((r + r0) / r0) ** s * taper
        b -= b.mean()
        basis.append(b)

    return np.stack(basis, axis=-1)



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
# WCS information from CSV
# ---------------------------------------------------------------------
def read_wcs_csv(drz_file, csv_file=None):
    """Read exposure WCS info from a CSV table."""
    if csv_file is None:
        csv_file = (
            drz_file.split("_drz_sci")[0].split("_drc_sci")[0] + "_wcs.csv"
        )
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")

    tab = Table.read(csv_file, format="csv")
    flt_keys = []
    wcs_dict = {}
    footprints = {}

    for row in tab:
        key = (row["file"], row["ext"])
        hdr = fits.Header()
        for col in tab.colnames:
            hdr[col] = row[col]

        wcs = WCS(hdr, relax=True)
        get_wcs_pscale(wcs)
        wcs.expweight = hdr.get("EXPTIME", 1)

        flt_keys.append(key)
        wcs_dict[key] = wcs
        footprints[key] = Polygon(wcs.calc_footprint())

    return flt_keys, wcs_dict, footprints


class CircularApertureProfile(RadialProfile):
    """Combined radial profile and curve of growth for a source.

    This class extends :class:`photutils.profiles.RadialProfile` by
    computing a matching :class:`photutils.profiles.CurveOfGrowth` and
    providing convenience methods for normalization, 1D model fitting
    and plotting.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Background subtracted image data.
    xycen : tuple of 2 floats, optional
        ``(x, y)`` pixel coordinate of the source centre. If None, use image center.
    radii : 1D array_like, optional
        Radii defining the edges for the radial profile annuli. If None, use [0, 0.5, 1, 2, 4, ...] pix.
    cog_radii : 1D array_like, optional
        Radii for the curve of growth apertures.  If `None`, the values
        of ``radii[1:]`` are used.
    recenter : bool, optional
        If True, recenter using centroid_quadratic. Default False.
    centroid_kwargs : dict, optional
        Passed to centroid_quadratic. Defaults to {'search_boxsize': 11, 'fit_boxsize': 5}.
    name : str, optional
        Name of the profile used for plot legends.
    norm_radius : float, optional
        Radius at which to normalise both profiles.
    error, mask, method, subpixels : optional
        Passed to :class:`photutils.profiles.RadialProfile` and
        :class:`photutils.profiles.CurveOfGrowth`.
    """

    def __init__(self,
                 data: np.ndarray,
                 xycen: tuple[float, float] | None = None,
                 radii: np.ndarray | None = None,
                 *,
                 cog_radii: np.ndarray | None = None,
                 recenter: bool = False,
                 centroid_kwargs: dict = None,
                 error: np.ndarray | None = None,
                 mask: np.ndarray | None = None,
                 method: str = "exact",
                 subpixels: int = 5,
                 name: str | None = None,
                 norm_radius: float | None = None,
                 pixel_scale: float | None = None) -> None:
        from photutils.centroids import centroid_quadratic

        # Set centroid kwargs defaults
        if centroid_kwargs is None:
            centroid_kwargs = {"search_boxsize": 11, "fit_boxsize": 5}

        # Default xycen: image center
        if xycen is None:
            ny, nx = data.shape
            xycen = ((nx - 1) / 2, (ny - 1) / 2)

        # Optionally recenter using centroid_quadratic
        if recenter:
            xp, yp = xycen
            xycen = centroid_quadratic(data,
                                       xpeak=xp,
                                       ypeak=yp,
                                       **centroid_kwargs)

        # Default radii: logarithmic bins from 0, 0.5, 1, 2, 4, ... up to edge of image
        if radii is None:
            ny, nx = data.shape
            maxrad = min(nx, ny) / 2
            radii = np.unique(
                np.concatenate([
                    np.array([0, 0.5, 1]),
                    np.logspace(np.log10(2), np.log10(maxrad), num=101)
                ]))
            radii = radii[radii <= maxrad]

        # Always set cog_radii = radii[1:] if not provided
        if cog_radii is None:
            cog_radii = radii[1:]

        super().__init__(
            data,
            xycen,
            radii,
            error=error,
            mask=mask,
            method=method,
            subpixels=subpixels,
        )

        self.cog = CurveOfGrowth(
            data,
            xycen,
            cog_radii,
            error=error,
            mask=mask,
            method=method,
            subpixels=subpixels,
        )

        self.name = name
        self.pixel_scale = pixel_scale

        # save indices into data
        yidx, xidx = np.indices(self.data.shape)
        radii = np.hypot(xidx - self.xycen[0], yidx - self.xycen[1])
        mask = radii <= np.max(self.radii)
        self._data_indices = np.where(mask)

        if self.pixel_scale is not None:
            norm_radius = norm_radius / pixel_scale

        self.norm_radius = norm_radius  # Always set this attribute
        if norm_radius is not None:
            self.normalize(norm_radius)

    def normalize(self, norm_radius: float | None = None) -> None:
        """Normalize the radial profile and curve of growth."""

        if norm_radius is not None:
            self.norm_radius = norm_radius

        if self.norm_radius is None:
            raise ValueError("norm_radius must be provided")

        # Use standard linear interpolation instead of PchipInterpolator


#        rp_val = np.interp(self.norm_radius, self.radius, self.profile)
        cog_val = np.interp(self.norm_radius, self.cog.radius,
                            self.cog.profile)
        if np.isfinite(cog_val) and cog_val != 0:
            self.cog.normalization_value *= cog_val
            self.cog.__dict__["profile"] = self.cog.profile / cog_val
            self.cog.__dict__[
                "profile_error"] = self.cog.profile_error / cog_val

            self.normalization_value *= cog_val
            self.__dict__["profile"] = self.profile / cog_val
            self.__dict__["profile_error"] = self.profile_error / cog_val

    @lazyproperty
    def moffat_fit(self):
        """Return a 1D Moffat model fitted to the radial profile."""

        profile = self.profile[self._profile_nanmask]
        radius = self.radius[self._profile_nanmask]
        amplitude = float(np.max(profile))
        gamma = float(radius[np.argmax(profile < amplitude / 2)] or 1.0)
        m_init = Moffat1D(amplitude=amplitude, x_0=0.0, gamma=gamma, alpha=2.5)
        m_init.x_0.fixed = True
        fitter = TRFLSQFitter()
        return fitter(m_init, radius, profile)

    @lazyproperty
    def moffat_fwhm(self) -> float:
        """Full width at half maximum (FWHM) of the fitted Moffat."""

        fit = self.moffat_fit
        return 2 * fit.gamma.value * np.sqrt(2**(1 / fit.alpha.value) - 1)

    def cog_ratio(self, other: "CircularApertureProfile") -> np.ndarray:
        """Return the ratio of this curve of growth to another."""

        interp = PchipInterpolator(other.cog.radius,
                                   other.cog.profile,
                                   extrapolate=False)(self.cog.radius)
        return self.cog.profile / interp

    def _radius_unit(self):
        return "arcsec" if self.pixel_scale is not None else "pix"

    def _convert_radius(self, r):
        return r * self.pixel_scale if self.pixel_scale is not None else r

    def _plot_radial_profile(self, ax, color="C0", **kwargs) -> None:
        label = self.name or "profile"
        radius = self._convert_radius(self.radius)
        ax.plot(radius, self.profile, label=label, color=color, **kwargs)
        # Overplot uncertainty as filled area
        if (hasattr(self, "profile_error") and self.profile_error is not None
                and self.profile_error.shape == self.profile.shape
                and self.profile_error.size > 0):
            ax.fill_between(
                radius,
                self.profile - self.profile_error,
                self.profile + self.profile_error,
                color=color,
                alpha=0.3,
                linewidth=0,
            )
        ax.set_yscale("log")
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("Normalized Profile")

        gfwhm = self.gaussian_fwhm
        ax.axvline(
            self._convert_radius(gfwhm / 2),
            color=color,
            ls="--",
            label=
            f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}"
        )
        ax.set_ylim(np.max(self.profile) / 3e5, np.max(self.profile) * 2.0)
        ax.legend()

    def _plot_cog(self, ax, color="C0", **kwargs) -> None:
        label = self.name or "profile"
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, self.cog.profile, label=label, color=color, **kwargs)
        # Overplot uncertainty as filled area
        if (hasattr(self.cog, "profile_error")
                and self.cog.profile_error is not None
                and self.cog.profile_error.shape == self.cog.profile.shape
                and self.cog.profile_error.size > 0):
            ax.fill_between(
                radius,
                self.cog.profile - self.cog.profile_error,
                self.cog.profile + self.cog.profile_error,
                color=color,
                alpha=0.3,
                linewidth=0,
            )
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("Encircled Energy")
        if self.norm_radius is not None:
            r20 = self.cog.calc_radius_at_ee(0.2)
            r80 = self.cog.calc_radius_at_ee(0.8)
            ax.axvline(
                self._convert_radius(r20),
                color=color,
                ls=":",
                label=
                f"R20 {self._convert_radius(r20):.2f} {self._radius_unit()}",
                **kwargs)
            ax.axvline(
                self._convert_radius(r80),
                color=color,
                ls="--",
                label=
                f"R80 {self._convert_radius(r80):.2f} {self._radius_unit()}",
                **kwargs)
        ax.set_ylim(0, 1.05)
        ax.legend()

    def _plot_ratio(self,
                    other: "CircularApertureProfile",
                    ax,
                    ylabel='',
                    color='k',
                    **kwargs) -> None:
        ratio = self.cog_ratio(other)
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, ratio, color=color, label=ylabel, **kwargs)
        # Overplot uncertainty as filled area if at least one profile has error
        err1 = getattr(self.cog, "profile_error", None)
        err2 = getattr(other.cog, "profile_error", None)
        val1 = self.cog.profile
        interp_val2 = PchipInterpolator(other.cog.radius,
                                        other.cog.profile,
                                        extrapolate=False)(self.cog.radius)
        err_ratio = None
        if (err1 is not None and err1.shape == val1.shape and err1.size
                > 0) and (err2 is not None and err2.shape == interp_val2.shape
                          and err2.size > 0):
            # Both have errors: propagate
            interp_err2 = PchipInterpolator(other.cog.radius,
                                            err2,
                                            extrapolate=False)(self.cog.radius)
            err_ratio = ratio * np.sqrt((err1 / val1)**2 +
                                        (interp_err2 / interp_val2)**2)
        elif err1 is not None and err1.shape == val1.shape and err1.size > 0:
            # Only self has error
            err_ratio = ratio * (err1 / val1)
        elif err2 is not None and err2.shape == interp_val2.shape and err2.size > 0:
            # Only other has error
            interp_err2 = PchipInterpolator(other.cog.radius,
                                            err2,
                                            extrapolate=False)(self.cog.radius)
            err_ratio = ratio * (interp_err2 / interp_val2)
        # Plot error band if available
        if err_ratio is not None:
            ax.fill_between(
                radius,
                ratio - err_ratio,
                ratio + err_ratio,
                color=color,
                alpha=0.3,
                linewidth=0,
            )
        ax.axhline(1.0, ls="-", color='gray')
        gfwhm = self.gaussian_fwhm
        ax.axvline(
            self._convert_radius(gfwhm / 2),
            color=color,
            ls="--",
            label=
            f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}",
            **kwargs)
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("COG Ratio " + ylabel)
        ax.set_ylim(0.8, 1.2)

    def plot(self,
             *,
             ax: list | None = None,
             cog_ratio: bool = True,
             **kwargs: dict) -> tuple["matplotlib.figure.Figure", list]:
        """Plot radial profile and curve of growth."""

        import matplotlib.pyplot as plt

        #        ncols = 3 if cog_ratio else 2
        if ax is None:
            fig, ax = plt.subplots(1,
                                   2 + cog_ratio,
                                   figsize=(4 * (2 + cog_ratio), 4))
            ax = ax.flatten()
        else:
            fig = ax[0].figure

        # Main profile: blue
        self._plot_radial_profile(ax[0], color='C0')
        self._plot_cog(ax[1], color='C0')

        fig.tight_layout()
        return fig, ax

    def plot_other(self,
                   other_profile,
                   ax=None,
                   color='C4',
                   cog_ratio=True,
                   **kwargs):
        """
        Plot only the other CircularApertureProfile on the provided axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1,
                                   2 + cog_ratio,
                                   figsize=(5 * (1 + cog_ratio), 3))
            ax = ax.flatten()
        else:
            fig = ax[0].figure

        other_profile._plot_radial_profile(ax[0], color=color, **kwargs)
        other_profile._plot_cog(ax[1], color=color, **kwargs)
        self._plot_ratio(other_profile,
                         ax[2],
                         ylabel=self.name + " / " + other_profile.name,
                         color=color,
                         **kwargs)
        fig.tight_layout()

        return ax

def clean_stamp(
    data,
    weight=None,
    scl=None,
    offset=2e-5,
    kws=None,
    w=3,
    threshold=3.0,
    verbose=False,
    imshow=False,
):
    """
    Produce a cleaned stamp for growth curve comparisons.
    
    Parameters
    ----------
    data : np.ndarray
        Input image stamp.
    scl : float, optional
        Normalization factor for display.
    offset : float, optional
        Offset for log10 display.
    kws : dict, optional
        Keyword arguments for imshow.
    w : int, optional
        Smoothing kernel width.
    verbose : bool, optional
        Print statistics if True.
    imshow : bool, optional
        Show diagnostic figure if True.
    
    Returns
    -------
    img : np.ndarray
        Cleaned image stamp.
    obj_mask : np.ndarray
        Object mask.
    bg_mask : np.ndarray
        Background mask.
    bg_level : float
        Estimated background level.
    """
    import matplotlib.pyplot as plt
    from mophongo.catalog import safe_dilate_segmentation
    from photutils.segmentation import detect_sources, SegmentationImage
    from astropy.stats import sigma_clipped_stats

    detimg = data.copy()
    if weight is not None:
        detimg *= np.sqrt(weight)
    detimg = convolve2d(detimg, np.ones((w, w)) / w**2)
    det_mean, det_median, det_std = sigma_clipped_stats(detimg, sigma=3)

    detimg -= det_median

    if verbose:
        print(f"Mean: {det_mean}, Median: {det_median}, Std: {det_std}")

    seg = detect_sources(detimg, threshold=threshold * det_std, npixels=3 * w**2)
    seg = SegmentationImage(safe_dilate_segmentation(seg, selem=np.ones((2*w, 2*w))))

    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    label = seg.data[cy, cx]

    obj_mask = seg.data == label
    nn_mask = (seg.data > 0) & ~obj_mask
    bg_mask = seg.data == 0
    bg_level = np.nanmedian(data[bg_mask])
    img = data.copy() - bg_level
    img[nn_mask] = np.minimum(img[::-1, ::-1], img)[nn_mask]

    if imshow:
        if kws is None:
            kws = dict(vmin=-5.3, vmax=-1.5, cmap='bone_r', origin='lower')

        scl = np.sum(data[obj_mask])
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(np.log10(data / scl + offset), **kws)
        seg.imshow(ax[0], alpha=0.3)
        ax[1].imshow(np.log10(img / scl + offset), **kws)
        plt.show()

    if verbose:
        print(f"subtracted bg_level: {bg_level}")

    return img, obj_mask, bg_level


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def compare_psf_to_star(
    cutout_data_in,
    psf_data_in,
    weight=None,
    kernel=None,
    Rnorm=None,
    pixel_scale=1.0,
    to_file=None,
    offset=2e-5,
    title_prefix='',
    fit_kernel=False,
    register_psf=False,
    **kwargs
):
    """
    Compare a PSF to a real star cutout, optionally convolving with a kernel.
    Produces a 2-row figure: 5 images on top, 3 profiles below.

    Parameters
    ----------
    cutout_data : np.ndarray
        Star cutout image.
    psf_data : np.ndarray
        PSF cutout image.
    kernel : np.ndarray, optional
        Convolution kernel. If None, will fit a kernel using multi_gaussian_basis.
    Rnorm : float, optional
        Normalization radius in arcsec (for profiles).
    pscale : float, optional
        Pixel scale in arcsec/pixel.
    to_file : str or Path, optional
        If given, save the figure to this file.
    offset : float, optional
        Offset for log10 display.
    title_prefix : str, optional
        Prefix for plot titles.
    kwargs : dict
        Passed to CircularApertureProfile.
    """
    from .utils import CircularApertureProfile, multi_gaussian_basis, fit_kernel_fourier, convolve2d

    # remove neighbors and subtract background
    cutout_data, obj_mask, bg_level = clean_stamp(cutout_data_in.copy(), imshow=False)

    if weight is not None:
        error = np.sqrt(1.0 / weight)
        error[~np.isfinite(error)] = 0.0
    else:
        error = None

    # shift align PSF to cutout centroid
    psf_data = psf_data_in.copy()
    if register_psf:
        from scipy.ndimage import shift
        from photutils.centroids import centroid_quadratic
        psf_xycen = centroid_quadratic(psf_data, xpeak=psf_data.shape[1]//2, ypeak=psf_data.shape[0]//2)
        cutout_xycen = centroid_quadratic(cutout_data, xpeak=cutout_data.shape[1]//2, ypeak=cutout_data.shape[0]//2)
        print('shifting by', cutout_xycen - psf_xycen)
        psf_data = shift(psf_data, psf_xycen - cutout_xycen,order=3)

    # --- Scale PSF to data ---
    if Rnorm is None:
        Rnorm = 2.0 * pixel_scale * (cutout_data.shape[0] // 2)

    mask = np.hypot(*np.indices(cutout_data.shape) - cutout_data.shape[0]//2) < (Rnorm / pixel_scale)
    scl = (cutout_data * psf_data)[mask].sum() / (psf_data[mask]**2).sum()

    # --- Profiles ---
    rp_data = CircularApertureProfile(cutout_data, error=error, name='data', norm_radius=Rnorm, pixel_scale=pixel_scale, recenter=True, **kwargs)
    rp_psf = CircularApertureProfile(psf_data, name='psf', norm_radius=Rnorm, pixel_scale=pixel_scale, recenter=True, **kwargs)

    # --- Kernel and convolution ---
    if fit_kernel:
        scales = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        basis = multi_gaussian_basis(scales, cutout_data.shape[0])
        kernel, coeffs = fit_kernel_fourier(psf_data, cutout_data, basis, method="nnls")
        print(f"Fitted kernel coefficients: {coeffs} for gaussian fwhm {scales} ")

    if kernel is not None:
        conv = convolve2d(psf_data, kernel)
        rp_conv = CircularApertureProfile(conv, name='psf x kernel', norm_radius=Rnorm, pixel_scale=pixel_scale, recenter=True, **kwargs)

    # --- Plotting ---
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1.])

    # First row: 5 images
    ax1 = [fig.add_subplot(gs[0, i]) for i in range(5)]
    titles = ['data', 'psf', 'data - psf', '', '']
    kws = dict(vmin=-5.3, vmax=-1.5, cmap='bone_r', origin='lower')
    for a, title in zip(ax1, titles):
        a.set_title(f"{title_prefix}{title}")
        a.axis('off')
    ax1[0].imshow(np.log10(cutout_data/scl + offset), **kws)
    ax1[1].imshow(np.log10(psf_data + offset), **kws)
    ax1[2].imshow(np.log10(cutout_data/scl - psf_data + offset), **kws)
    if kernel is not None:
        ax1[3].imshow(np.log10(cutout_data/scl - conv + offset), **kws)
        ax1[4].imshow(np.log10(pad_to_shape(kernel,conv.shape) + offset), **kws)
        ax1[3].set_title('data - psf x kernel')
        ax1[4].set_title('kernel')

    # Add filename as a textbox in the left top corner
    if to_file is not None:
        import os
        plot_base = os.path.splitext(os.path.basename(to_file))[0].replace('_', ' ')
        fig.text(0.01, 0.98, plot_base, fontsize=14, fontweight='medium',
                 va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Second row: 3 profiles spanning all columns
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.2)
    ax2 = [fig.add_subplot(subgs[0, i]) for i in range(3)]
    _ = rp_data.plot(ax=ax2)
    _ = rp_data.plot_other(rp_psf, ax=ax2, color='C3', alpha=0.5)
    for a in ax2:
        a.set_xlim(0,Rnorm * 1.3)
    if kernel is not None:
        _ = rp_data.plot_other(rp_conv, ax=ax2, color='C2', alpha=0.5)
    for a, title in zip(ax2, ['profile', 'growthcurve', 'ratio of growthcurves']):
        a.set_title(title)

    plt.tight_layout()
    if to_file is not None:
        fig.savefig(to_file, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig
