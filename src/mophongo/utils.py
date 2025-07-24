"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import os
import numpy as np
import scipy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from shapely.geometry import Polygon
from scipy.special import hermite
from scipy.interpolate import PchipInterpolator
from astropy.utils import lazyproperty
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Moffat1D
from photutils.profiles import RadialProfile, CurveOfGrowth

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
    fwhm_x: float,
    fwhm_y: float,
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


import numpy as np
from scipy.special import eval_hermite      # physicists' Hermite


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
        conv = _convolve2d(psf_hi, kernel)
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
            conv = _convolve2d(psf_hi, kernel_test)
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

    def __init__(
        self,
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
        pixel_scale: float | None = None,  # <-- add pixel_scale argument
    ) -> None:
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
            xycen = centroid_quadratic(data, xpeak=xp, ypeak=yp, **centroid_kwargs)

        # Default radii: logarithmic bins from 0, 0.5, 1, 2, 4, ... up to edge of image
        if radii is None:
            ny, nx = data.shape
            maxrad = min(nx, ny) / 2
            radii = np.unique(
                np.concatenate([
                    np.array([0, 0.5, 1]),
                    np.logspace(np.log10(2), np.log10(maxrad), num=101)
                ])
            )
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
        self.norm_radius = norm_radius
        self.pixel_scale = pixel_scale

        # save indices into data 
        yidx, xidx = np.indices(self.data.shape)
        radii = np.hypot(xidx - self.xycen[0], yidx - self.xycen[1])
        mask = radii <= np.max(self.radii)
        self._data_indices = np.where(mask)

        if norm_radius is not None:
            self.normalize(norm_radius)

    def normalize(self, norm_radius: float | None = None) -> None:
        """Normalize the radial profile and curve of growth."""

        if norm_radius is not None:
            self.norm_radius = norm_radius

        if self.norm_radius is None:
            raise ValueError("norm_radius must be provided")

        rp_val = PchipInterpolator(self.radius, self.profile, extrapolate=False)(
            self.norm_radius
        )
        if np.isfinite(rp_val) and rp_val != 0:
            self.normalization_value *= rp_val
            self.__dict__["profile"] = self.profile / rp_val
            self.__dict__["profile_error"] = self.profile_error / rp_val

        cog_val = PchipInterpolator(
            self.cog.radius, self.cog.profile, extrapolate=False
        )(self.norm_radius)
        if np.isfinite(cog_val) and cog_val != 0:
            self.cog.normalization_value *= cog_val
            self.cog.__dict__["profile"] = self.cog.profile / cog_val
            self.cog.__dict__["profile_error"] = self.cog.profile_error / cog_val

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
        return 2 * fit.gamma.value * np.sqrt(2 ** (1 / fit.alpha.value) - 1)

    def cog_ratio(self, other: "CircularApertureProfile") -> np.ndarray:
        """Return the ratio of this curve of growth to another."""

        interp = PchipInterpolator(
            other.cog.radius, other.cog.profile, extrapolate=False
        )(self.cog.radius)
        return self.cog.profile / interp

    def _radius_unit(self):
        return "arcsec" if self.pixel_scale is not None else "pix"

    def _convert_radius(self, r):
        return r * self.pixel_scale if self.pixel_scale is not None else r

    def _plot_radial_profile(self, ax, color="C0", **kwargs) -> None:
        label = self.name or "profile"
        radius = self._convert_radius(self.radius)
        ax.plot(radius, self.profile, label=label, color=color, **kwargs)
        ax.set_yscale("log")
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("Normalized Profile")
        gfwhm = self.gaussian_fwhm
        ax.axvline(self._convert_radius(gfwhm / 2), color=color, ls="--", label=f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}")
        ax.set_ylim(np.max(self.profile) / 1e5, np.max(self.profile) * 1.3)
        ax.legend()

    def _plot_cog(self, ax, color="C0", **kwargs) -> None:
        label = self.name or "profile"
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, self.cog.profile, label=label, color=color, **kwargs)
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("Encircled Energy")
        if self.norm_radius is not None:
            r20 = self.cog.calc_radius_at_ee(0.2)
            r80 = self.cog.calc_radius_at_ee(0.8)
            ax.axvline(self._convert_radius(r20), color=color, ls=":", label=f"R20 {self._convert_radius(r20):.2f} {self._radius_unit()}", **kwargs)
            ax.axvline(self._convert_radius(r80), color=color, ls="--", label=f"R80 {self._convert_radius(r80):.2f} {self._radius_unit()}", **kwargs)
        ax.set_ylim(0, 1.05)
        ax.legend()

    def _plot_ratio(self, other: "CircularApertureProfile", ax, ylabel='', color='k', **kwargs) -> None:
        ratio = self.cog_ratio(other)
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, ratio, color=color, label=ylabel, **kwargs)
        ax.axhline(1.0, ls="-", color='gray')
        gfwhm = self.gaussian_fwhm
        ax.axvline(self._convert_radius(gfwhm / 2), color=color, ls="--", label=f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}", **kwargs)
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("COG Ratio " + ylabel)
        ax.set_ylim(0.8, 1.2)

    def plot(
        self,
        *,
        axes: list | None = None,
        cog_ratio: bool = True,
        **kwargs: dict
    ) -> tuple["matplotlib.figure.Figure", list]:
        """Plot radial profile and curve of growth."""

        import matplotlib.pyplot as plt

#        ncols = 3 if cog_ratio else 2
        if axes is None:
            fig, axes = plt.subplots(1 + cog_ratio, 2, figsize=(4 * 2, 3.5 * (1+ cog_ratio)))
            axes = axes.flatten()
        else:
            fig = axes[0].figure
        
        # Main profile: blue
        self._plot_radial_profile(axes[0], color='C0')
        self._plot_cog(axes[1], color='C0')

        fig.tight_layout()
        return fig, axes

    def plot_other(self, other_profile, axes=None, color='C4', cog_ratio=True, **kwargs):
        """
        Plot only the other CircularApertureProfile on the provided axes.
        """
        if axes is None:
            fig, axes = plt.subplots(1 + cog_ratio, 2, figsize=(4 * 2, 3.5 * (1+ cog_ratio)))
            axes = axes.flatten()
        else:
            fig = axes[0].figure
        
        other_profile._plot_radial_profile(axes[0], color=color, **kwargs)
        other_profile._plot_cog(axes[1], color=color, **kwargs)
        self._plot_ratio(other_profile, axes[2], ylabel=self.name + " / " + other_profile.name, color=color, **kwargs)
        fig.tight_layout()

        return axes

def clean_stamp(
    data,
    weight=None,
    scl=None,
    offset=2e-5,
    kws=None,
    w=3,
    threshold=2.5,
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
    from mophongo.templates import _convolve2d
    from mophongo.catalog import safe_dilate_segmentation
    from photutils.segmentation import detect_sources, SegmentationImage
    from astropy.stats import sigma_clipped_stats 

    detimg = data.copy()
    if weight is not None:
        detimg *= np.sqrt(weight)
    detimg = _convolve2d(detimg, np.ones((w, w)) / w**2)
    det_mean, det_median, det_std = sigma_clipped_stats(detimg, sigma=3)
    
    detimg -= det_median

    if verbose:
        print(f"Mean: {det_mean}, Median: {det_median}, Std: {det_std}")

    seg = detect_sources(detimg, threshold=threshold * det_std, npixels=3 * w**2)
    seg = SegmentationImage(safe_dilate_segmentation(seg.data, selem=np.ones((w, w))))

    obj_mask = seg.data.astype(bool)
    bg_mask = ~obj_mask
    bg_level = np.nanmedian(data[bg_mask])
    img = data.copy() - bg_level
    img[obj_mask] = np.minimum(img[::-1, ::-1], img)[obj_mask]

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
