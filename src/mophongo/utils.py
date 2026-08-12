"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import os
import copy
import numpy as np
import cv2
import scipy
from scipy.ndimage import shift
from scipy.signal import fftconvolve as _scipy_fftconvolve
from astropy.nddata import block_reduce

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table
from shapely.geometry import Polygon
from scipy.interpolate import PchipInterpolator
from photutils.profiles import RadialProfile, CurveOfGrowth
from photutils.centroids import centroid_quadratic, centroid_com

from scipy.special import eval_hermite  # physicists' Hermite
from photutils.psf import matching

import logging

logger = logging.getLogger(__name__)


# remaps (center-of-pixel convention; origin at pixel centers, 0-based)

def as_label_array(segmap: np.ndarray) -> np.ndarray:
    """Return ``segmap`` as an integer label array.

    Segmaps are label images, but releases differ on how they store them: the
    MINERVA UDS and EGS maps are ``int32`` while COSMOS ships the same labels as
    ``float64``. ``SegmentationImage`` rejects anything non-integer, so cast
    when the values are whole numbers and refuse when they are not, rather than
    truncating real fractions silently.
    """
    arr = np.asarray(segmap)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size and not np.all(finite == np.rint(finite)):
        raise ValueError(
            "segmap holds non-integer values and cannot be used as labels"
        )
    logger.info("segmap stored as %s; casting labels to int32", arr.dtype)
    return np.nan_to_num(arr).astype(np.int32)


def bin_remap(x: float | tuple[float, float], k: int) -> np.ndarray:
    shift = (k - 1) / 2.0
    return (np.array(x, dtype=np.float64) - shift) / k


def expand_remap(x: float | tuple[float, float], k: int) -> np.ndarray:
    shift = (k - 1) / 2.0
    return (np.array(x, dtype=np.float64) + shift) * k


def intersection(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int] | None:
    y0 = max(a[0], b[0])
    y1 = min(a[1], b[1])
    x0 = max(a[2], b[2])
    x1 = min(a[3], b[3])
    if y0 >= y1 or x0 >= x1:
        return None
    return y0, y1, x0, x1


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

    # only downsample if k a multiple of 2 and
    # correct for drift of center from odd to even size
    if (k % 2 == 0) and (psf.shape[0] % 2 == 1):
        shift_hi = (k - 1) / 2.0
        psf = shift(
            psf,
            #            shift=(-shift_hi, -shift_hi), # @@@ check direction
            shift=(shift_hi, shift_hi),  # @@@ check direction
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return block_reduce(psf, k, func=np.sum)


def bin_factor_from_wcs(w_det: WCS, w_img: WCS, tol: float = 0.001) -> int:
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


def rebin_wcs(wcs: WCS, factor: int) -> WCS:
    """
    Up‐ or down‐sample a WCS by factor, *exactly* preserving the
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
    factor = 2**n
    new_wcs = copy.deepcopy(wcs)

    # — scale the CD or CDELT matrix
    if getattr(new_wcs.wcs, "cd", None) is not None:
        new_wcs.wcs.cd = new_wcs.wcs.cd / factor
    else:
        new_wcs.wcs.cdelt = new_wcs.wcs.cdelt / factor

    # — shift CRPIX so that CRVAL stays fixed on the sky:
    #    new_crpix = (old_crpix - 0.5)/factor + 0.5
    old_crpix = new_wcs.wcs.crpix.copy()
    new_wcs.wcs.crpix = (old_crpix - 0.5) / factor + 0.5

    # — update the “NAXIS” so to_header() will emit the right shape
    #    (Astropy uses .pixel_shape if present, else _naxis1/_naxis2)
    if hasattr(new_wcs, "pixel_shape") and new_wcs.pixel_shape is not None:
        ny, nx = new_wcs.pixel_shape
        new_wcs.pixel_shape = (int(ny // factor), int(nx // factor))
    else:
        # fallback into the private attributes
        if hasattr(new_wcs.wcs, "_naxis1"):
            new_wcs.wcs._naxis1 = int(new_wcs.wcs._naxis1 // factor)
            new_wcs.wcs._naxis2 = int(new_wcs.wcs._naxis2 // factor)
        if hasattr(new_wcs.wcs, "_naxis"):
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


def fftconvolve(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    mode: str = "same",
) -> np.ndarray:
    """Convolve using Mophongo's centered-kernel convention.

    For ``mode="same"``, this avoids ``scipy.signal.fftconvolve(...,
    mode="same")`` because SciPy's generic central crop is offset by one pixel
    for even-sized kernels relative to the centered-kernel convention used by
    Mophongo's matching kernels.  Cropping the full convolution from
    ``kernel.shape // 2`` keeps odd and even kernels on the same convention.

    Parameters
    ----------
    image : np.ndarray
        Image or PSF to convolve.
    kernel : np.ndarray
        Centered convolution kernel.
    mode : {"same", "full"}
        Output shape. ``"same"`` returns an image-sized centered crop.
        ``"full"`` returns the complete true convolution.

    Returns
    -------
    np.ndarray
        Convolved image.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    full = _scipy_fftconvolve(image, kernel, mode="full")
    if mode == "full":
        return full
    if mode != "same":
        raise ValueError(f"mode must be 'same' or 'full', got {mode!r}")
    y0 = kernel.shape[0] // 2
    x0 = kernel.shape[1] // 2
    return full[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]]


def resize_flux_conserving_inter_cubic(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize an image with OpenCV cubic interpolation and conserve total flux.

    This uses the same pixel-extent convention as Mophongo's nested block
    grids.  Integer 80 -> 40 or 160 -> 40 upsampling therefore stays registered
    with the block-sum convention while avoiding the centroid offsets from
    SciPy's default zoom coordinates.
    """
    if factor <= 0:
        raise ValueError(f"resize factor must be positive, got {factor!r}")
    image = np.asarray(image, dtype=np.float32)
    ny, nx = image.shape
    out_ny = max(1, int(round(ny * factor)))
    out_nx = max(1, int(round(nx * factor)))
    resized = cv2.resize(
        image,
        dsize=(out_nx, out_ny),
        fx=0.0,
        fy=0.0,
        interpolation=cv2.INTER_CUBIC,
    )
    return resized.astype(np.float64, copy=False) / float(factor * factor)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve and return an image-sized result using the shared convention."""
    return fftconvolve(image, kernel, mode="same")


def _create_matching_kernel_no_normalize(
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    *,
    window: object | None = None,
) -> np.ndarray:
    """Create a Fourier-ratio PSF matching kernel without flux normalization."""
    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))
    good = np.abs(source_otf) > (np.finfo(float).eps * np.nanmax(np.abs(source_otf)))
    ratio = np.zeros_like(target_otf, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[good] = target_otf[good] / source_otf[good]
    if window is not None:
        ratio *= window(target_psf.shape)
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ratio))))


def _matching_kernel_tikhonov(
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    *,
    reg: float = 1e-3,
) -> np.ndarray:
    """Tikhonov-regularized Fourier inversion ``K = conj(H_hi)*H_lo / (|H_hi|^2 + lambda)``.

    ``reg`` is scaled by ``max(|H_hi|^2)`` so the parameter is dimensionless
    and easily comparable across PSF pairs.
    """
    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))
    h2 = np.abs(source_otf) ** 2
    lam = float(reg) * float(np.max(h2))
    filt = np.conj(source_otf) * target_otf / (h2 + lam)
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filt))))


def _matching_kernel_wiener(
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    *,
    reg: float = 1e-3,
    signal_psd: np.ndarray | None = None,
) -> np.ndarray:
    """Wiener-regularized Fourier inversion.

    ``K = conj(H_hi) P_xx H_lo / (|H_hi|^2 P_xx + lambda * max(|H_hi|^2 P_xx))``.

    With the default flat ``signal_psd`` this is mathematically identical to
    Tikhonov; the path is kept so callers can pass an explicit Wiener prior
    (e.g. ``|H_lo|^2`` or a 1/f spectrum).
    """
    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))
    if signal_psd is None:
        signal_psd = np.ones(source_otf.shape, dtype=float)
    else:
        signal_psd = np.asarray(signal_psd, dtype=float)
    h2 = np.abs(source_otf) ** 2
    num = np.conj(source_otf) * signal_psd * target_otf
    den_base = h2 * signal_psd
    lam = float(reg) * float(np.max(den_base))
    filt = num / (den_base + lam)
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filt))))


def _pad_to_multiple(arr: np.ndarray, factor: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Zero-pad ``arr`` so each axis is a multiple of ``factor``.

    Pad widths are chosen so the input's array center ``N // 2`` is preserved
    at the padded array's center ``N_pad // 2``.  Without this constraint
    asymmetric padding (e.g. 121 -> 128 with top=3, bot=4) introduces a
    half-pixel shift relative to ``np.fft`` conventions that survives the
    wavelet round-trip and appears as a dipole in residuals.
    """
    ny, nx = arr.shape
    ny_pad = ((ny + factor - 1) // factor) * factor
    nx_pad = ((nx + factor - 1) // factor) * factor
    top = (ny_pad // 2) - (ny // 2)
    bot = ny_pad - ny - top
    left = (nx_pad // 2) - (nx // 2)
    right = nx_pad - nx - left
    padded = np.pad(arr, ((top, bot), (left, right)), mode="constant")
    return padded, (top, bot, left, right)


def _unpad(arr: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    top, bot, left, right = pads
    ny, nx = arr.shape
    return arr[top : ny - bot if bot > 0 else ny, left : nx - right if right > 0 else nx]


def _matching_kernel_forward(
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    *,
    reg: float = 1e-3,
    wavelet: str = "db4",
    levels: int = 3,
    threshold_factor: float = 3.0,
    noise_sigma: float | None = None,
    apply_wavelet_wiener: bool = True,
) -> np.ndarray:
    """ForWaRD (Fourier+wavelet regularized deconvolution) matching kernel.

    Reference
    ---------
    Neelamani, Choi, Baraniuk, IEEE TSP 52, 418 (2004),
    "ForWaRD: Fourier-Wavelet Regularized Deconvolution for Ill-Conditioned
    Systems."  Adapted to PSF matching by recovering the kernel ``K`` from the
    observation ``psf_lo = psf_hi * K`` (no measurement noise; the regularizer
    represents model mismatch).

    Steps
    -----
    1. Tikhonov-regularized Fourier inverse of ``psf_hi`` applied to ``psf_lo``
       gives an initial kernel estimate ``K1``.
    2. Redundant (stationary) wavelet decomposition of ``K1``.
    3. Per-subband noise variance estimated from the wavelet decomposition of
       the Tikhonov inverse impulse response (paper eq. for noise propagation).
    4. Hard thresholding of detail coefficients with threshold
       ``threshold_factor * sigma_subband`` gives a "reference" estimate
       ``K_ref``.
    5. Optional wavelet-domain Wiener step that filters ``K1`` using
       ``K_ref^2 / (K_ref^2 + sigma_subband^2)`` as the empirical Wiener gain.
    """
    import pywt

    factor = 1 << int(levels)
    src_padded, pads = _pad_to_multiple(source_psf, factor)
    tgt_padded, _ = _pad_to_multiple(target_psf, factor)

    src_otf = np.fft.fftshift(np.fft.fft2(src_padded))
    tgt_otf = np.fft.fftshift(np.fft.fft2(tgt_padded))
    h2 = np.abs(src_otf) ** 2
    lam = float(reg) * float(np.max(h2))
    inv_filter = np.conj(src_otf) / (h2 + lam)

    K1_otf = inv_filter * tgt_otf
    K1 = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(K1_otf))))
    inv_imp = np.real(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(inv_filter)))
    )

    if noise_sigma is None:
        coeffs0 = pywt.wavedec2(K1, wavelet, level=1)
        _, (_, _, cD) = coeffs0[0], coeffs0[1]
        noise_sigma = float(np.median(np.abs(cD)) / 0.6745)
        if noise_sigma <= 0:
            noise_sigma = float(np.std(K1)) * 1e-3

    coeffs_K = pywt.swt2(K1, wavelet, level=levels, trim_approx=False, norm=False)
    coeffs_inv = pywt.swt2(inv_imp, wavelet, level=levels, trim_approx=False, norm=False)

    sigma_sub: list[tuple[float, float, float]] = []
    thresh_levels: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for (aK, (h_K, v_K, d_K)), (_, (h_i, v_i, d_i)) in zip(coeffs_K, coeffs_inv):
        s_h = noise_sigma * float(np.sqrt(np.mean(h_i ** 2)))
        s_v = noise_sigma * float(np.sqrt(np.mean(v_i ** 2)))
        s_d = noise_sigma * float(np.sqrt(np.mean(d_i ** 2)))
        sigma_sub.append((s_h, s_v, s_d))
        thr_h = threshold_factor * s_h
        thr_v = threshold_factor * s_v
        thr_d = threshold_factor * s_d
        thresh_levels.append(
            (
                aK,
                (
                    pywt.threshold(h_K, thr_h, mode="hard"),
                    pywt.threshold(v_K, thr_v, mode="hard"),
                    pywt.threshold(d_K, thr_d, mode="hard"),
                ),
            )
        )
    K_ref = pywt.iswt2(thresh_levels, wavelet, norm=False)

    if not apply_wavelet_wiener:
        return _unpad(K_ref, pads)

    coeffs_ref = pywt.swt2(K_ref, wavelet, level=levels, trim_approx=False, norm=False)
    wiener_levels: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for (aK, (h_K, v_K, d_K)), (_, (h_R, v_R, d_R)), (s_h, s_v, s_d) in zip(
        coeffs_K, coeffs_ref, sigma_sub
    ):
        def _gain(detail_K: np.ndarray, detail_ref: np.ndarray, sigma: float) -> np.ndarray:
            sig2 = detail_ref ** 2
            n_var = sigma ** 2
            denom = sig2 + n_var
            gain = np.where(denom > 0, sig2 / denom, 0.0)
            return detail_K * gain

        wiener_levels.append(
            (
                aK,
                (
                    _gain(h_K, h_R, s_h),
                    _gain(v_K, v_R, s_v),
                    _gain(d_K, d_R, s_d),
                ),
            )
        )
    K_final = pywt.iswt2(wiener_levels, wavelet, norm=False)
    return _unpad(K_final, pads)


def retile_blocked(arr, B=64):
    """Return a blocked copy with shape (nby, nbx, B, B), plus timings."""
    H, W = arr.shape
    if H % B != 0 or W % B != 0:
        raise ValueError(f"Array shape {H}x{W} must be divisible by block size {B}.")
    HH, WW = (H // B) * B, (W // B) * B
    t0 = time.perf_counter()
    blocked = arr[:HH, :WW].reshape(HH // B, B, WW // B, B).swapaxes(1, 2).copy()
    t = time.perf_counter() - t0
    bytes_moved = blocked.nbytes * 2  # read + write
    return blocked.reshape(H, W), HH, WW, t, bytes_moved


# wrapper around astropy matching kernel that handles padding and recentering
def matching_kernel(
    psf_hi_in: np.ndarray,
    psf_lo_in: np.ndarray,
    *,
    window: object | None = None,
    recenter: bool = False,
    pixel_ratio: float = 1.0,
    method: str = "window",
    reg: float = 1e-3,
    wavelet: str = "db4",
    levels: int = 3,
    threshold_factor: float = 3.0,
    noise_sigma: float | None = None,
    forward_wavelet_wiener: bool = True,
    signal_psd: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a convolution kernel matching ``psf_hi`` to ``psf_lo``.

    The kernel ``k`` is defined such that ``psf_hi * k \approx psf_lo`` when
    convolved.  The Fourier-ratio implementation intentionally does not
    normalize the input PSFs or output kernel; if ``sum(psf_lo) / sum(psf_hi)``
    is not one, that normalization propagates into ``sum(k)``.  If the two
    PSFs have different shapes they are zero padded to a common grid before
    computing the kernel.

    Note
    ----
    This is a low-level routine. Pipeline-facing fitting code should normally
    pass unit-sum PSF shapes and keep finite-stamp PSF sums as throughput
    metadata for final flux corrections. Passing native-sum PSFs here is still
    supported for explicit diagnostics or tests that need a throughput-carrying
    kernel.

    Parameters
    ----------
    psf_hi, psf_lo:
        High- and low-resolution PSF arrays. They may have different shapes.
    window : optional
        Fourier-domain window function used when ``method="window"``.
        Defaults to ``SplitCosineBellWindow(alpha=0.4, beta=0.1)``.
    recenter : bool, optional
        If ``True`` the resulting kernel is shifted to its centroid: a
        center-of-mass first guess refined by a quadratic centroid, applied
        with cubic interpolation and zero padding so the shift conserves
        flux. Defaults to ``False``.
    pixel_ratio : float, optional
        Pixel-scale ratio between the two PSFs. A ratio above one upsamples
        ``psf_lo`` onto the finer grid (how the pipeline passes it, as the
        low-to-high pixel-scale ratio); a ratio below one downsamples
        ``psf_hi`` instead. The resize is flux-conserving cubic interpolation
        (:func:`resize_flux_conserving_inter_cubic`) using the same
        pixel-extent convention as the pipeline's nested block grids, so
        integer scale ratios stay registered. Defaults to ``1.0`` (no
        resampling).
    method : str, optional
        ``"window"`` (default), ``"tikhonov"``, ``"wiener"``, or ``"forward"``
        (ForWaRD Fourier+wavelet regularized deconvolution).
    reg : float, optional
        Regularization parameter for the non-window methods, scaled
        internally by the peak of the inversion denominator
        (``max(|H_hi|^2)``; ``max(|H_hi|^2 P_xx)`` for Wiener) so it is
        dimensionless.

    Returns
    -------
    kernel: ``np.ndarray``
        Convolution kernel with shape equal to the larger of the two input PSFs.
    """
    # pixel_ratio > 0.  For non-unity ratios use OpenCV cubic resize, scaled by
    # factor**2 to conserve integrated flux and preserve the nested block-grid
    # centroid convention.
    if pixel_ratio == 1.0:
        psf_hi = psf_hi_in.copy()
        psf_lo = psf_lo_in.copy()
    else:
        if pixel_ratio > 1.0:
            psf_hi = psf_hi_in.copy()
            psf_lo = resize_flux_conserving_inter_cubic(psf_lo_in, pixel_ratio)
        else:
            psf_lo = psf_lo_in.copy()
            psf_hi = resize_flux_conserving_inter_cubic(psf_hi_in, pixel_ratio)

    if psf_hi.shape != psf_lo.shape:
        ny = max(psf_hi.shape[0], psf_lo.shape[0])
        nx = max(psf_hi.shape[1], psf_lo.shape[1])
        shape = (ny, nx)
        psf_hi = pad_to_shape(psf_hi, shape)
        psf_lo = pad_to_shape(psf_lo, shape)

    if not np.isfinite(psf_hi).all():
        logger.warning("psf 1 contains non-finite values, setting elements to zero ")
        psf_hi[~np.isfinite(psf_hi)] = 0.0

    if not np.isfinite(psf_lo).all():
        logger.warning("psf 2 contains non-finite values, setting elements to zero ")
        psf_lo[~np.isfinite(psf_lo)] = 0.0

    method_key = method.strip().lower()
    if method_key in ("window", "scb", "split_cosine_bell", "tukey"):
        if window is None:
            window = matching.SplitCosineBellWindow(alpha=0.4, beta=0.1)
        kernel = _create_matching_kernel_no_normalize(psf_hi, psf_lo, window=window)
    elif method_key in ("tikhonov", "ridge"):
        kernel = _matching_kernel_tikhonov(psf_hi, psf_lo, reg=reg)
    elif method_key == "wiener":
        kernel = _matching_kernel_wiener(psf_hi, psf_lo, reg=reg, signal_psd=signal_psd)
    elif method_key in ("forward", "forwardrd", "fourier_wavelet"):
        kernel = _matching_kernel_forward(
            psf_hi,
            psf_lo,
            reg=reg,
            wavelet=wavelet,
            levels=levels,
            threshold_factor=threshold_factor,
            noise_sigma=noise_sigma,
            apply_wavelet_wiener=forward_wavelet_wiener,
        )
    else:
        raise ValueError(
            f"Unknown matching kernel method {method!r}. "
            "Expected one of: window, tikhonov, wiener, forward."
        )
    kernel = np.asarray(kernel)

    if not np.isfinite(kernel).all():
        logger.warning("Kernel contains non-finite values, returning zero kernel.")
        return np.zeros_like(kernel)

    if recenter:
        # first guess is center of mass, then fit quadratic centroid
        xcom, ycom = centroid_com(kernel)
        xcen, ycen = centroid_quadratic(kernel, xpeak=xcom, ypeak=ycom, fit_boxsize=7)
        if np.isnan(ycen) or np.isnan(xcen):
            xcen, ycen = xcom, ycom

        if not np.isnan(ycen) and not np.isnan(xcen):
            cx = (kernel.shape[1] - 1) / 2
            cy = (kernel.shape[0] - 1) / 2
            # constant-pad so the shift is flux-conserving (mode="nearest"
            # would replicate edge values and inject bias into kernel.sum).
            kernel = shift(kernel, (cy - ycen, cx - xcen),
                           order=3, mode="constant", cval=0.0)
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

    n_pix = img_hi.size  # = size²

    # 1.  Shift arrays so that the PSF/basis centre is at pixel (0,0) **before** FFT
    f_hi = np.fft.fft2(np.fft.ifftshift(img_hi))
    f_lo = np.fft.fft2(np.fft.ifftshift(img_lo))
    f_basis = np.fft.fft2(np.fft.ifftshift(basis, axes=(0, 1)), axes=(0, 1))

    # 2.  Build the least-squares matrix in Fourier space
    nb = basis.shape[-1]
    A = (f_basis * f_hi[..., None] / n_pix).reshape(-1, nb)  # IDL normalisation
    b = (f_lo / n_pix).ravel()

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

    # 3.  Back to real space – already centred, so **no fftshift here**.
    # Do not normalize the solved kernel: its DC term carries the native
    # target/source flux ratio.
    kernel = np.tensordot(basis, coeffs, axes=([-1], [0]))

    return kernel, coeffs




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
    cx = cy = (size - 1) / 2  # geometric centre
    basis = []

    for s in scales:
        xn = (x - cx) / s
        yn = (y - cy) / s
        g = np.exp(-0.5 * (xn**2 + yn**2))  # isotropic Gaussian

        for i in range(order + 1):
            for j in range(order + 1 - i):
                b = g * eval_hermite(i, xn) * eval_hermite(j, yn)

                if i == 0 and j == 0:
                    b /= b.sum()  # unit DC
                else:
                    b -= b.mean()  # kill residual DC

                basis.append(b)

    # Stack as (Ny, Nx, Nbasis)
    return np.stack(basis, axis=-1)


def multi_gaussian_basis(scales: list[float], size: int) -> np.ndarray:
    """Return a set of Gaussian basis functions with varying width."""

    gauss_list = [gaussian(size, s, s) for s in scales]
    basis = np.stack(gauss_list, axis=2)
    basis /= basis.sum(axis=(0, 1))
    return basis




# ---------------------------------------------------------------------
# Basic WCS utilities
# ---------------------------------------------------------------------
def get_wcs_pscale(wcs, set_attribute=True):
    """Pixel scale in arcsec from a ``WCS`` object."""
    from numpy.linalg import det

    if isinstance(wcs, fits.Header):
        wcs = WCS(wcs, relax=True)

    # assumes wcs in degrees
    pscale = wcs.proj_plane_pixel_scales()[0].value * 3600

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
        csv_file = drz_file.split("_drz_sci")[0].split("_drc_sci")[0] + "_wcs.csv"
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
        pixel_scale: float | None = None,
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
                np.concatenate(
                    [np.array([0, 0.5, 1]), np.logspace(np.log10(2), np.log10(maxrad), num=101)]
                )
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
        cog_val = np.interp(self.norm_radius, self.cog.radius, self.cog.profile)
        if np.isfinite(cog_val) and cog_val != 0:
            self.cog.normalization_value *= cog_val
            self.cog.__dict__["profile"] = self.cog.profile / cog_val
            self.cog.__dict__["profile_error"] = self.cog.profile_error / cog_val

            self.normalization_value *= cog_val
            self.__dict__["profile"] = self.profile / cog_val
            self.__dict__["profile_error"] = self.profile_error / cog_val


    def cog_ratio(self, other: "CircularApertureProfile") -> np.ndarray:
        """Return the ratio of this curve of growth to another."""

        interp = PchipInterpolator(other.cog.radius, other.cog.profile, extrapolate=False)(
            self.cog.radius
        )
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
        if (
            hasattr(self, "profile_error")
            and self.profile_error is not None
            and self.profile_error.shape == self.profile.shape
            and self.profile_error.size > 0
        ):
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
            label=f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}",
        )
        ax.set_ylim(np.max(self.profile) / 3e5, np.max(self.profile) * 2.0)
        ax.legend()

    def _plot_cog(self, ax, color="C0", **kwargs) -> None:
        label = self.name or "profile"
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, self.cog.profile, label=label, color=color, **kwargs)
        # Overplot uncertainty as filled area
        if (
            hasattr(self.cog, "profile_error")
            and self.cog.profile_error is not None
            and self.cog.profile_error.shape == self.cog.profile.shape
            and self.cog.profile_error.size > 0
        ):
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
            # values kept for callers that tabulate them instead of using the legend
            self.r50 = self.cog.calc_radius_at_ee(0.5)
            self.r80 = self.cog.calc_radius_at_ee(0.8)
            for r, ls, tag in ((self.r50, ":", "R50"), (self.r80, "--", "R80")):
                rx = self._convert_radius(r)
                ax.axvline(rx, color=color, ls=ls, **kwargs)
                ax.text(
                    rx,
                    0.02,
                    tag,
                    rotation=90,
                    color=color,
                    fontsize=8,
                    ha="right",
                    va="bottom",
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_ylim(0, 1.05)
        ax.legend()

    def _plot_ratio(
        self, other: "CircularApertureProfile", ax, ylabel="", color="k", **kwargs
    ) -> None:
        ratio = self.cog_ratio(other)
        radius = self._convert_radius(self.cog.radius)
        ax.plot(radius, ratio, color=color, label=ylabel, **kwargs)
        # Overplot uncertainty as filled area if at least one profile has error
        err1 = getattr(self.cog, "profile_error", None)
        err2 = getattr(other.cog, "profile_error", None)
        val1 = self.cog.profile
        interp_val2 = PchipInterpolator(other.cog.radius, other.cog.profile, extrapolate=False)(
            self.cog.radius
        )
        err_ratio = None
        if (err1 is not None and err1.shape == val1.shape and err1.size > 0) and (
            err2 is not None and err2.shape == interp_val2.shape and err2.size > 0
        ):
            # Both have errors: propagate
            interp_err2 = PchipInterpolator(other.cog.radius, err2, extrapolate=False)(
                self.cog.radius
            )
            err_ratio = ratio * np.sqrt((err1 / val1) ** 2 + (interp_err2 / interp_val2) ** 2)
        elif err1 is not None and err1.shape == val1.shape and err1.size > 0:
            # Only self has error
            err_ratio = ratio * (err1 / val1)
        elif err2 is not None and err2.shape == interp_val2.shape and err2.size > 0:
            # Only other has error
            interp_err2 = PchipInterpolator(other.cog.radius, err2, extrapolate=False)(
                self.cog.radius
            )
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
        ax.axhline(1.0, ls="-", color="gray")
        gfwhm = self.gaussian_fwhm
        ax.axvline(
            self._convert_radius(gfwhm / 2),
            color=color,
            ls="--",
            label=f"Gauss FWHM {self._convert_radius(gfwhm):.2f} {self._radius_unit()}",
            **kwargs,
        )
        if self.norm_radius is not None:
            # the curves are normalized here, so the ratio is 1 by construction
            rn = self._convert_radius(self.norm_radius)
            ax.axvline(rn, color="gray", ls="-.")
            ax.text(
                rn,
                0.02,
                f"Rnorm {rn:.2f} {self._radius_unit()}",
                rotation=90,
                color="gray",
                fontsize=8,
                ha="right",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlabel(f"Radius ({self._radius_unit()})")
        ax.set_ylabel("COG Ratio " + ylabel)
        ax.set_ylim(0.8, 1.2)

    def plot(
        self, *, ax: list | None = None, cog_ratio: bool = True, **kwargs: dict
    ) -> tuple["matplotlib.figure.Figure", list]:
        """Plot radial profile and curve of growth."""

        import matplotlib.pyplot as plt

        #        ncols = 3 if cog_ratio else 2
        if ax is None:
            fig, ax = plt.subplots(1, 2 + cog_ratio, figsize=(4 * (2 + cog_ratio), 4))
            ax = ax.flatten()
        else:
            fig = ax[0].figure

        # Main profile: blue
        self._plot_radial_profile(ax[0], color="C0")
        self._plot_cog(ax[1], color="C0")

        fig.tight_layout()
        return fig, ax

    def plot_other(self, other_profile, ax=None, color="C4", cog_ratio=True, **kwargs):
        """
        Plot only the other CircularApertureProfile on the provided axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2 + cog_ratio, figsize=(5 * (1 + cog_ratio), 3))
            ax = ax.flatten()
        else:
            fig = ax[0].figure

        other_profile._plot_radial_profile(ax[0], color=color, **kwargs)
        other_profile._plot_cog(ax[1], color=color, **kwargs)
        self._plot_ratio(
            other_profile,
            ax[2],
            ylabel=self.name + " / " + other_profile.name,
            color=color,
            **kwargs,
        )
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
    if seg is None:
        # nothing above threshold: no neighbours to mask, only remove the background
        logger.warning("clean_stamp: no source detected, returning background-subtracted stamp")
        bg_level = np.nanmedian(data)
        return data.copy() - bg_level, np.zeros(data.shape, dtype=bool), bg_level
    seg = SegmentationImage(safe_dilate_segmentation(seg, selem=np.ones((2 * w, 2 * w))))

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
            kws = dict(vmin=-5.3, vmax=-1.5, cmap="bone_r", origin="lower")

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


def kernel_effective_fwhm(kernel: np.ndarray, pixel_scale: float = 1.0) -> float:
    """Second-moment FWHM of a non-negative convolution kernel.

    Zero for a pure delta (no diffusion), and the FWHM itself for a single
    Gaussian, so a delta-plus-broad mixture reports the width its diffuse part
    actually carries.
    """
    k = np.clip(np.asarray(kernel, dtype=float), 0.0, None)
    total = k.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    yy, xx = np.indices(k.shape)
    r2 = (yy - (k.shape[0] - 1) / 2) ** 2 + (xx - (k.shape[1] - 1) / 2) ** 2
    sigma = np.sqrt((k * r2).sum() / total / 2.0)
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma * pixel_scale)


def compare_psf_to_star(
    cutout_data_in,
    psf_data_in,
    weight=None,
    kernel=None,
    Rnorm=None,
    pixel_scale=1.0,
    to_file=None,
    offset=2e-5,
    title_prefix="",
    fit_kernel=False,
    register_psf=False,
    clean=True,
    composite=None,
    composite_title="",
    shift=None,
    kernel_fwhm=None,
    save_curves=False,
    vmin=-5.3,
    vmax=-1.0,
    **kwargs,
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
    from .utils import (
        CircularApertureProfile,
        multi_gaussian_basis,
        fit_kernel_fourier,
        convolve2d,
    )

    # remove neighbors and subtract background, unless the caller already did
    if clean:
        cutout_data, obj_mask, bg_level = clean_stamp(cutout_data_in.copy(), imshow=False)
    else:
        cutout_data = cutout_data_in.copy()

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

        psf_xycen = centroid_quadratic(
            psf_data, xpeak=psf_data.shape[1] // 2, ypeak=psf_data.shape[0] // 2
        )
        cutout_xycen = centroid_quadratic(
            cutout_data, xpeak=cutout_data.shape[1] // 2, ypeak=cutout_data.shape[0] // 2
        )
        print("shifting by", cutout_xycen - psf_xycen)
        psf_data = shift(psf_data, psf_xycen - cutout_xycen, order=3)

    # --- Scale PSF to data ---
    if Rnorm is None:
        Rnorm = 2.0 * pixel_scale * (cutout_data.shape[0] // 2)

    mask = np.hypot(*np.indices(cutout_data.shape) - cutout_data.shape[0] // 2) < (
        Rnorm / pixel_scale
    )
    scl = (cutout_data * psf_data)[mask].sum() / (psf_data[mask] ** 2).sum()

    # --- Profiles ---
    rp_data = CircularApertureProfile(
        cutout_data,
        error=error,
        name="data",
        norm_radius=Rnorm,
        pixel_scale=pixel_scale,
        recenter=True,
        **kwargs,
    )
    rp_psf = CircularApertureProfile(
        psf_data, name="psf", norm_radius=Rnorm, pixel_scale=pixel_scale, recenter=True, **kwargs
    )

    # --- Kernel and convolution ---
    if fit_kernel:
        scales = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        basis = multi_gaussian_basis(scales, cutout_data.shape[0])
        kernel, coeffs = fit_kernel_fourier(psf_data, cutout_data, basis, method="nnls")
        print(f"Fitted kernel coefficients: {coeffs} for gaussian fwhm {scales} ")

    if kernel is not None:
        conv = convolve2d(psf_data, kernel)
        rp_conv = CircularApertureProfile(
            conv,
            name="psf x diff",
            norm_radius=Rnorm,
            pixel_scale=pixel_scale,
            recenter=True,
            **kwargs,
        )

    # --- Plotting ---
    ncol = 4 + (composite is not None)
    fig = plt.figure(figsize=(3.75 * ncol, 8))
    gs = gridspec.GridSpec(2, ncol, height_ratios=[1, 1.0])

    # First row: 4 images, plus an optional colour composite. The diffusion
    # kernel is a smooth blob and carries no visual information, so its width is
    # reported in the ratio panel instead of getting a panel of its own.
    ax1 = [fig.add_subplot(gs[0, i]) for i in range(ncol)]
    titles = ["data", "psf", "data - psf", ""] + ([""] if composite is not None else [])
    # log10 display range in dex. The top used to sit at -1.5; -1.0 widens the
    # ramp, so the sky noise takes up less of it and the panels read calmer
    kws = dict(vmin=vmin, vmax=vmax, cmap="bone_r", origin="lower")
    for a, title in zip(ax1, titles):
        a.set_title(f"{title_prefix}{title}")
        a.axis("off")
    ax1[0].imshow(np.log10(cutout_data / scl + offset), **kws)
    ax1[1].imshow(np.log10(psf_data + offset), **kws)
    ax1[2].imshow(np.log10(cutout_data / scl - psf_data + offset), **kws)
    if kernel is not None:
        # `conv = psf*kernel` is fit to `cutout_data` in absolute units, while
        # the other panels work in PSF-flux units (cutout_data/scl). Divide
        # `conv` by `scl` to keep the residual in the same units.
        ax1[3].imshow(np.log10((cutout_data - conv) / scl + offset), **kws)
        ax1[3].set_title("data - psf x diff")
        # a caller that fitted the kernel knows its width analytically; measuring
        # the second moment off the image instead picks up far-field FFT residue
        # and overstates it several-fold on a large stamp
        fwhm_diff = (kernel_effective_fwhm(kernel, pixel_scale) if kernel_fwhm is None
                     else float(kernel_fwhm))
        unit = "arcsec" if pixel_scale != 1.0 else "pix"

    if composite is not None:
        ax1[-1].imshow(composite, origin="lower")
        ax1[-1].set_title(composite_title or "colour composite")

    # Add filename as a textbox in the left top corner
    if to_file is not None:
        import os

        plot_base = os.path.splitext(os.path.basename(to_file))[0].replace("_", " ")
        fig.text(
            0.01,
            0.98,
            plot_base,
            fontsize=14,
            fontweight="medium",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    # Second row: 3 profiles spanning all columns
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.2)
    ax2 = [fig.add_subplot(subgs[0, i]) for i in range(3)]
    _ = rp_data.plot(ax=ax2)
    _ = rp_data.plot_other(rp_psf, ax=ax2, color="C3", alpha=0.5)
    for a in ax2:
        a.set_xlim(0, Rnorm * 1.3)
    if kernel is not None:
        _ = rp_data.plot_other(rp_conv, ax=ax2, color="C2", alpha=0.5)
    for a, title in zip(ax2, ["profile", "growthcurve", "ratio of growthcurves"]):
        a.set_title(title)

    # growth-curve panel: the radii are marked on the curves, so trade the
    # legend for a compact R50/R80 table
    entries = [("data", rp_data, "C0"), ("psf", rp_psf, "C3")]
    if kernel is not None:
        entries.append(("psf x diff", rp_conv, "C2"))
    legend = ax2[1].get_legend()
    if legend is not None:
        legend.remove()
    table = ax2[1].table(
        cellText=[
            [f"{p._convert_radius(p.cog.calc_radius_at_ee(0.5)):.2f}",
             f"{p._convert_radius(p.cog.calc_radius_at_ee(0.8)):.2f}"]
            for _, p, _ in entries
        ],
        rowLabels=[name for name, _, _ in entries],
        colLabels=['R50 (")', 'R80 (")'],
        colWidths=[0.22, 0.22],
        cellLoc="center",
        loc="lower right",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for row, (_, _, color) in enumerate(entries, start=1):
        table[row, -1].get_text().set_color(color)

    # the diffusion kernel itself: one number, next to the curve it produced
    if kernel is not None:
        ax2[2].text(
            0.04,
            0.06,
            f"diffusion kernel eff FWHM {fwhm_diff:.3f} {unit}",
            color="C2",
            fontsize=9,
            transform=ax2[2].transAxes,
        )
    if shift is not None:
        ax2[2].text(
            0.04,
            0.13,
            f"registration shift dx {shift[0]:+.2f}, dy {shift[1]:+.2f} pix",
            color="0.35",
            fontsize=9,
            transform=ax2[2].transAxes,
        )

    # the growth curves themselves, so many stars can be overplotted later
    if save_curves and to_file is not None:
        import os

        from astropy.table import Table

        curves = Table()
        curves["radius"] = rp_data._convert_radius(rp_data.cog.radius)
        curves["ee_data"] = rp_data.cog.profile
        curves["ee_psf"] = np.interp(rp_data.cog.radius, rp_psf.cog.radius, rp_psf.cog.profile)
        curves["ratio_psf"] = rp_data.cog_ratio(rp_psf)
        if kernel is not None:
            curves["ee_conv"] = np.interp(
                rp_data.cog.radius, rp_conv.cog.radius, rp_conv.cog.profile
            )
            curves["ratio_conv"] = rp_data.cog_ratio(rp_conv)
        curves.meta["norm_radius"] = rp_data._convert_radius(rp_data.norm_radius)
        curves.write(os.path.splitext(str(to_file))[0] + "_cog.csv", overwrite=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])   # leave room for the corner label
    if to_file is not None:
        fig.savefig(to_file, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


import os, re, csv, logging, glob
from pathlib import Path
from astropy.io import fits

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("jwst_wcs")

# --- CSV schema (note: no 'xposure' column; we map header XPOSURE/EXPTIME/EFFEXPTM -> exptime) ---
FIELDS = [
    "file",
    "ext",
    "exptime",
    "wcsaxes",
    "crpix1",
    "crpix2",
    "cd1_1",
    "cd1_2",
    "cd2_1",
    "cd2_2",
    "cdelt1",
    "cdelt2",
    "cunit1",
    "cunit2",
    "ctype1",
    "ctype2",
    "crval1",
    "crval2",
    "lonpole",
    "latpole",
    "wcsname",
    "mjdref",
    "date-beg",
    "mjd-beg",
    "date-avg",
    "mjd-avg",
    "date-end",
    "mjd-end",
    "telapse",
    "obsgeo-x",
    "obsgeo-y",
    "obsgeo-z",
    "radesys",
    "velosys",
    "a_order",
    "a_0_2",
    "a_0_3",
    "a_0_4",
    "a_0_5",
    "a_1_1",
    "a_1_2",
    "a_1_3",
    "a_1_4",
    "a_2_0",
    "a_2_1",
    "a_2_2",
    "a_2_3",
    "a_3_0",
    "a_3_1",
    "a_3_2",
    "a_4_0",
    "a_4_1",
    "a_5_0",
    "b_order",
    "b_0_2",
    "b_0_3",
    "b_0_4",
    "b_0_5",
    "b_1_1",
    "b_1_2",
    "b_1_3",
    "b_1_4",
    "b_2_0",
    "b_2_1",
    "b_2_2",
    "b_2_3",
    "b_3_0",
    "b_3_1",
    "b_3_2",
    "b_4_0",
    "b_4_1",
    "b_5_0",
    "naxis",
    "naxis1",
    "naxis2",
    "sipcrpx1",
    "sipcrpx2",
]

SIP_A_KEYS = ["A_ORDER"] + [
    f"A_{i}_{j}"
    for (i, j) in [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (5, 0),
    ]
]
SIP_B_KEYS = ["B_ORDER"] + [
    f"B_{i}_{j}"
    for (i, j) in [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (5, 0),
    ]
]

MAST_TOKEN = os.environ.get("MAST_TOKEN")  # optional for proprietary
FSSPEC_HEADERS = {"Authorization": f"token {MAST_TOKEN}"} if MAST_TOKEN else {}


def mast_url_for_filename(filename: str) -> str:
    return f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{filename}"


# dataset prefix + detector
JW_DATASET_RE = re.compile(r"(jw\d{11}_[0-9]{5}_[0-9]{5})_([a-z0-9]+)", re.IGNORECASE | re.DOTALL)


def extract_dataset_from_comments(hdr: fits.Header):
    """Robustly gather COMMENT text (cards can be split/continued)."""
    comments = []
    for card in hdr.cards:
        if card.keyword == "COMMENT" and card.value is not None:
            comments.append(str(card.value))
    blob = " ".join(comments)  # join; breaks across cards are common
    pairs = JW_DATASET_RE.findall(blob)
    out, seen = [], set()
    for ds, det in pairs:
        key = (ds.lower(), det.lower())
        if key not in seen:
            seen.add(key)
            out.append(ds + "_" + det + "_rate.fits")
    return out


def cd_from_header(h):
    if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        return h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"]
    if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2")):
        return (
            h["CDELT1"] * h["PC1_1"],
            h["CDELT1"] * h["PC1_2"],
            h["CDELT2"] * h["PC2_1"],
            h["CDELT2"] * h["PC2_2"],
        )
    return (None, None, None, None)


def mjdref_from_header(h):
    if "MJDREF" in h:
        return h["MJDREF"]
    a, b = h.get("MJDREFI", 0.0), h.get("MJDREFF", 0.0)
    try:
        return float(a) + float(b)
    except Exception:
        return None


def pick_exptime(h):
    # Map header exposure keywords into single CSV 'exptime'
    for k in ("XPOSURE", "EXPTIME", "EFFEXPTM"):
        if k in h:
            return h[k]
    return None


def open_remote_sci_header(url: str):
    """Open remote FITS; return (header, ext_index). Try 'SCI' else [1]."""
    with fits.open(
        url, use_fsspec=True, fsspec_kwargs={"headers": FSSPEC_HEADERS} if FSSPEC_HEADERS else {}
    ) as hdul:
        try:
            hdr = hdul["SCI"].header
            return hdr, hdul.index_of("SCI")
        except Exception:
            hdr = hdul[1].header
            return hdr, 1


def row_from_header(filename: str, ext: int, h: fits.Header):
    cd11, cd12, cd21, cd22 = cd_from_header(h)
    g = lambda k: h.get(k)
    row = {
        "file": filename,
        "ext": ext,
        "exptime": pick_exptime(h),
        "wcsaxes": g("WCSAXES"),
        "crpix1": g("CRPIX1"),
        "crpix2": g("CRPIX2"),
        "cd1_1": cd11,
        "cd1_2": cd12,
        "cd2_1": cd21,
        "cd2_2": cd22,
        "cdelt1": g("CDELT1"),
        "cdelt2": g("CDELT2"),
        "cunit1": g("CUNIT1"),
        "cunit2": g("CUNIT2"),
        "ctype1": g("CTYPE1"),
        "ctype2": g("CTYPE2"),
        "crval1": g("CRVAL1"),
        "crval2": g("CRVAL2"),
        "lonpole": g("LONPOLE"),
        "latpole": g("LATPOLE"),
        "wcsname": g("WCSNAME"),
        "mjdref": mjdref_from_header(h),
        "date-beg": g("DATE-BEG"),
        "mjd-beg": g("MJD-BEG"),
        "date-avg": g("DATE-AVG"),
        "mjd-avg": g("MJD-AVG"),
        "date-end": g("DATE-END"),
        "mjd-end": g("MJD-END"),
        "telapse": g("TELAPSE"),
        "obsgeo-x": g("OBSGEO-X"),
        "obsgeo-y": g("OBSGEO-Y"),
        "obsgeo-z": g("OBSGEO-Z"),
        "radesys": g("RADESYS"),
        "velosys": g("VELOSYS"),
        "naxis": g("NAXIS"),
        "naxis1": g("NAXIS1"),
        "naxis2": g("NAXIS2"),
        "sipcrpx1": g("SIPCRPX1"),
        "sipcrpx2": g("SIPCRPX2"),
    }
    for k in SIP_A_KEYS:
        row[k.lower()] = g(k)
    for k in SIP_B_KEYS:
        row[k.lower()] = g(k)
    for f in FIELDS:
        row.setdefault(f, None)
    return row


def output_csv_path(mosaic_path: Path) -> Path:
    base = mosaic_path.stem
    for suf in ("_drz_wht", "_drz_sci", "_i2d", "_wht", "_sci"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return mosaic_path.with_name(f"{base}_wcs.csv")


# write_wcs_csv_from_mosaic('uds-sbkgsub-v2.0-80mas-f770w_drz_wht.fits')
# write_wcs_csv_from_mosaic('data/*/F770W/stage2/*cal.fits', out_csv='uds-v2.0_f770_wcs.csv')
def write_wcs_csv(mosaic_or_glob: Path | str, out_csv: str | None = None):
    if any(ch in str(mosaic_or_glob) for ch in "*?["):
        files = sorted(glob.glob(str(mosaic_or_glob)))
        if not files:
            raise RuntimeError(f"Glob matched zero files: {mosaic_or_glob}")  # minimal guard
        files = [(Path(f).name, f) for f in files]
        base_for_csv = Path(files[0][1])  # <-- needed for output name
    else:
        hdr0 = fits.getheader(mosaic_or_glob, ext=0)  # PRIMARY only
        files = extract_dataset_from_comments(hdr0)
        if not files:
            raise RuntimeError("No JWST dataset references found in PRIMARY COMMENT cards.")
        files = [(f, mast_url_for_filename(f)) for f in files]
        base_for_csv = Path(mosaic_or_glob)

    rows = []
    for file_name, path_or_url in files:
        print(f"Processing {path_or_url}...")
        continue
        try:
            hdr, ext = open_remote_sci_header(path_or_url)  # SCI else [1]
        except Exception as e:
            log.warning(f"could not open rate header for {path_or_url}: {e}")
            continue
        rows.append(row_from_header(file_name, ext, hdr))

    out_csv_path = Path(out_csv) if out_csv else output_csv_path(base_for_csv)
    with out_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    log.info(f"Wrote {len(rows)} rows -> {out_csv_path}")
    return out_csv_path, len(rows)


# Online: parse mosaic → fetch headers from MAST
# write_wcs_csv_from_mosaic("uds-sbkgsub-v2.0-80mas-f770w_drz_wht.fits")

# Offline: glob local CAL/RATE files; override output name
# write_wcs_csv_from_mosaic("data/*/F770W/stage2/*cal.fits", out_csv="uds-v2.0_f770_wcs.csv")


# ===========================================================================
# Grizli _wcs.csv reconstruction from public MAST _cal.fits headers
# ===========================================================================
# Used by :class:`mophongo.psf.DrizzlePSF` (via ``read_wcs_csv``) when the
# companion ``<stem>_wcs.csv`` is missing. Reads only the FITS primary
# header byte range from S3/MAST per frame; no full-file downloads.

import io as _io
import time as _time
import urllib.request as _urlreq
from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
from typing import Any as _Any

import pandas as _pd

_RECON_STEM_RE = re.compile(r"(jw\d{11}_\d{5}_\d{5}_[a-z]+\d*)")
_RECON_MOSAIC_SUFFIX_RE = re.compile(r"_dr[cz]_sci(_extrabkg)?\.fits$")

RECON_COLS: tuple[str, ...] = (
    "file,ext,exptime,wcsaxes,crpix1,crpix2,cd1_1,cd1_2,cd2_1,cd2_2,"
    "cdelt1,cdelt2,cunit1,cunit2,ctype1,ctype2,crval1,crval2,lonpole,latpole,"
    "wcsname,mjdref,date-beg,mjd-beg,date-avg,mjd-avg,date-end,mjd-end,"
    "xposure,telapse,obsgeo-x,obsgeo-y,obsgeo-z,radesys,velosys,"
    "a_order,a_0_2,a_0_3,a_0_4,a_0_5,a_1_1,a_1_2,a_1_3,a_1_4,"
    "a_2_0,a_2_1,a_2_2,a_2_3,a_3_0,a_3_1,a_3_2,a_4_0,a_4_1,a_5_0,"
    "b_order,b_0_2,b_0_3,b_0_4,b_0_5,b_1_1,b_1_2,b_1_3,b_1_4,"
    "b_2_0,b_2_1,b_2_2,b_2_3,b_3_0,b_3_1,b_3_2,b_4_0,b_4_1,b_5_0,"
    "naxis,naxis1,naxis2,sipcrpx1,sipcrpx2"
).split(",")

_RECON_PRIMARY_ALIASES = {
    "exptime": "EFFEXPTM",
    "mjd-beg": "EXPSTART",
    "mjd-avg": "EXPMID",
    "mjd-end": "EXPEND",
}

# NIRCam grizli convention: subtract one frame from start/mid (reset frame).
# MIRI files use a different readout convention; do NOT shift.
_NIRCAM_DETECTORS = {
    f"nrc{ab}{n}" for ab in "ab" for n in ["1", "2", "3", "4", "long"]
}


def canonical_stem(file_name: str) -> str:
    """Return canonical MAST stem (without ``_cal.fits``/``_rate.fits``).

    Grizli stores intermediate names such as
    ``jw{...}_mirimage_masked_sbkgsub_tweak_cal.fits``; this maps to
    ``jw{...}_mirimage`` so it can be turned into a MAST URL.
    """
    m = _RECON_STEM_RE.search(file_name)
    if not m:
        raise ValueError(f"cannot parse canonical stem from {file_name!r}")
    return m.group(1)


def recon_detector(file_name: str) -> str:
    """Return detector token (e.g. ``nrca5``, ``mirimage``)."""
    return canonical_stem(file_name).rsplit("_", 1)[-1]


def s3_url(file_name: str) -> str:
    """Public MAST S3 cal-file URL for any grizli intermediate name."""
    stem = canonical_stem(file_name)
    full = stem.split("_", 1)[0]
    prog = full[:7]
    return (
        f"https://stpubdata.s3.amazonaws.com/jwst/public/{prog}/{full}/"
        f"{stem}_cal.fits"
    )


def mast_url(file_name: str) -> str:
    """Public MAST download URL for any grizli intermediate name."""
    stem = canonical_stem(file_name)
    return (
        "https://mast.stsci.edu/api/v0.1/Download/file"
        f"?uri=mast:JWST/product/{stem}_cal.fits"
    )


def _recon_list_flt(mosaic: Path) -> list[str]:
    h = fits.getheader(str(mosaic))
    return [h[k] for k in h if k.startswith("FLT") and k[3:].isdigit()]


def _recon_list_from_csv(csv_path: Path) -> list[str]:
    return _pd.read_csv(csv_path, usecols=["file"])["file"].tolist()


def _recon_list_from_comments(mosaic: Path) -> list[str]:
    """Parse 'Files used to create mosaic:' COMMENT block (MIRI mosaics).

    Filenames wrap across two COMMENT cards with no separator; cards are
    concatenated and split on ``.fits`` to recover each path.
    """
    h = fits.getheader(str(mosaic))
    if "COMMENT" not in h:
        return []
    lines = [str(c) for c in h["COMMENT"]]
    start = None
    for i, ln in enumerate(lines):
        if "Files used to create mosaic" in ln:
            start = i + 1
            break
    if start is None:
        return []
    joined = "".join(lines[start:])
    parts = joined.split(".fits")
    out: list[str] = []
    for chunk in parts[:-1]:
        path = chunk.strip().split()[-1] + ".fits"
        out.append(Path(path).name)
    return out


def _recon_list_inputs(
    mosaic: Path,
    companion_csv: Path | None,
    filelist: Path | None,
) -> list[str]:
    if filelist is not None:
        return [
            ln.strip() for ln in filelist.read_text().splitlines() if ln.strip()
        ]
    flt = _recon_list_flt(mosaic)
    if flt:
        logger.info(
            "using FLT* keys from mosaic header (%d entries)", len(flt)
        )
        return flt
    com = _recon_list_from_comments(mosaic)
    if com:
        logger.info(
            "using 'Files used' COMMENT block (%d entries)", len(com)
        )
        return com
    if companion_csv and companion_csv.exists():
        names = _recon_list_from_csv(companion_csv)
        logger.info(
            "mosaic has no FLT* keys; using `file` column of %s (%d entries)",
            companion_csv.name, len(names),
        )
        return names
    raise RuntimeError(
        f"no FLT* keys or 'Files used' COMMENT block in {mosaic.name} "
        f"and no companion csv {companion_csv}; pass filelist= explicitly"
    )


def _recon_header_to_row(
    orig_name: str, pri: fits.Header, sci: fits.Header
) -> dict[str, _Any]:
    row: dict[str, _Any] = dict.fromkeys(RECON_COLS)
    row["file"] = orig_name
    row["ext"] = 1
    wcs = WCS(sci)
    for key in RECON_COLS:
        if key in ("file", "ext", "sipcrpx1", "sipcrpx2"):
            continue
        k = key.upper()
        v = sci.get(k)
        if v is None:
            v = pri.get(k)
        if v is not None:
            row[key] = v
    for csv_key, pri_key in _RECON_PRIMARY_ALIASES.items():
        v = pri.get(pri_key)
        if v is not None:
            row[csv_key] = v
    if all(row.get(k) is None for k in ("cd1_1", "cd1_2", "cd2_1", "cd2_2")):
        cd = wcs.pixel_scale_matrix
        row["cd1_1"] = cd[0, 0]
        row["cd1_2"] = cd[0, 1]
        row["cd2_1"] = cd[1, 0]
        row["cd2_2"] = cd[1, 1]
    if recon_detector(orig_name) in _NIRCAM_DETECTORS:
        tframe = pri.get("TFRAME")
        if tframe is not None:
            dt = tframe / 86400.0
            if row.get("mjd-beg") is not None:
                row["mjd-beg"] = row["mjd-beg"] - dt
            if row.get("mjd-avg") is not None:
                row["mjd-avg"] = row["mjd-avg"] - dt / 2.0
    row["sipcrpx1"] = sci.get("CRPIX1")
    row["sipcrpx2"] = sci.get("CRPIX2")
    return row


def _recon_range(
    orig_name: str, url: str, header_bytes: int = 300_000,
) -> dict[str, _Any]:
    req = _urlreq.Request(
        url, headers={"Range": f"bytes=0-{header_bytes - 1}"}
    )
    with _urlreq.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    with fits.open(
        _io.BytesIO(payload), lazy_load_hdus=True, ignore_missing_end=True
    ) as hdul:
        return _recon_header_to_row(orig_name, hdul[0].header, hdul["SCI"].header)


def reconstruct_row(orig_name: str, source: str = "s3") -> dict[str, _Any]:
    """Reconstruct one CSV row from public ``_cal.fits`` header range.

    Parameters
    ----------
    orig_name
        Grizli intermediate filename (canonicalized internally).
    source
        ``"s3"`` (anonymous, fast) or ``"mast"`` (CDN URL).
    """
    url = s3_url(orig_name) if source == "s3" else mast_url(orig_name)
    return _recon_range(orig_name, url)


def _recon_many(
    names: list[str], workers: int, source: str = "s3",
) -> tuple[list[dict[str, _Any]], list[tuple[str, str]]]:
    rows: dict[str, dict[str, _Any]] = {}
    fails: list[tuple[str, str]] = []
    with _TPE(max_workers=workers) as ex:
        fut_map = {ex.submit(reconstruct_row, n, source): n for n in names}
        done = 0
        total = len(names)
        for fut in _as_completed(fut_map):
            n = fut_map[fut]
            done += 1
            try:
                rows[n] = fut.result()
            except Exception as e:
                fails.append((n, str(e)))
                logger.error("fail %s: %s", n, e)
            if done % 25 == 0 or done == total:
                logger.info("fetched %d/%d", done, total)
    return [rows[n] for n in names if n in rows], fails


def reconstruct_wcs_default_paths(mosaic: Path) -> tuple[Path, Path]:
    """Return ``(<stem>_wcs.recon.csv, <stem>_wcs.csv)`` next to *mosaic*."""
    stem = _RECON_MOSAIC_SUFFIX_RE.sub("", mosaic.name)
    base = mosaic.parent
    return base / f"{stem}_wcs.recon.csv", base / f"{stem}_wcs.csv"


def reconstruct_wcs(
    mosaic: str | Path,
    *,
    out_csv: str | Path | None = None,
    source: str = "s3",
    workers: int = 32,
    filelist: str | Path | None = None,
    limit: int = 0,
    companion_csv: str | Path | None = None,
) -> _pd.DataFrame:
    """Reconstruct grizli ``_wcs.csv`` for *mosaic* and write it to disk.

    Parameters
    ----------
    mosaic
        Drizzled mosaic FITS path.
    out_csv
        Output CSV path. Default ``<stem>_wcs.recon.csv`` next to *mosaic*.
    source
        Header range source: ``"s3"`` (default) or ``"mast"``.
    workers
        Parallel header fetches.
    filelist
        Optional text file with one input filename per line.
    limit
        If >0, reconstruct only the first *limit* inputs (debugging).
    companion_csv
        Fallback file-list source if mosaic has no FLT*/COMMENT block.

    Returns
    -------
    pandas.DataFrame
        Reconstructed CSV contents (also written to *out_csv*).
    """
    mosaic = Path(mosaic)
    if not mosaic.exists():
        raise FileNotFoundError(f"mosaic not found: {mosaic}")

    default_out, default_actual = reconstruct_wcs_default_paths(mosaic)
    out_csv = Path(out_csv) if out_csv else default_out
    if companion_csv is not None:
        fallback: Path | None = Path(companion_csv)
    elif default_actual.exists():
        fallback = default_actual
    else:
        fallback = None

    names = _recon_list_inputs(
        mosaic, fallback, Path(filelist) if filelist else None
    )
    if limit and limit > 0:
        names = names[:limit]
        logger.info("limited to first %d", len(names))

    t0 = _time.time()
    rows, fails = _recon_many(names, workers=workers, source=source)
    logger.info(
        "done in %.1f s  (rows=%d  fails=%d)",
        _time.time() - t0, len(rows), len(fails),
    )

    df = _pd.DataFrame(rows, columns=list(RECON_COLS))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(
        "wrote %s  (%d rows, %d cols)", out_csv, len(df), len(df.columns)
    )
    return df


# ===========================================================================
# PSF-matched inverse-variance coadd (LW detection image)
# ===========================================================================
def lw_detection_coadd(
    bands: list[dict[str, _Any]],
    target_psf: np.ndarray,
    *,
    target_index: int | None = None,
    method: str = "wiener",
    reg_grid: np.ndarray | None = None,
    output_sci: str | Path | None = None,
    output_wht: str | Path | None = None,
    header: fits.Header | None = None,
    diagnostic_dir: str | Path | None = None,
    return_kernels: bool = False,
) -> dict[str, _Any]:
    """Inverse-variance coadd PSF-matched to *target_psf*.

    For each input band (except *target_index*):

    1. Find the optimal regularization parameter using
       :meth:`mophongo.psf.PSF.optimize_matching_kernel_regularization`
       with the requested *method* (default ``"wiener"``).
    2. Convolve the science image with the resulting kernel.
    3. Scale the per-pixel weight by ``1 / Σ(kernel ** 2)`` so the
       coadded inverse variance accounts for the correlated noise that
       convolution introduces.

    The PSF-matched ``(sci, wht)`` arrays are then combined via standard
    inverse-variance weighting:

    .. math::

        I_{\\rm coadd} = \\frac{\\sum_i w_i \\cdot I_i}{\\sum_i w_i},
        \\quad W_{\\rm coadd} = \\sum_i w_i

    Parameters
    ----------
    bands
        List of dicts, one per band. Required keys:

        - ``sci`` : ``np.ndarray`` or FITS path
        - ``wht`` : ``np.ndarray`` or FITS path
        - ``psf`` : ``np.ndarray`` — the band PSF (sum-normalized)
        - ``name`` : ``str`` (optional, used in diagnostics)

    target_psf
        Reference PSF for matching (e.g. F444W).
    target_index
        Index of *bands* whose PSF already matches *target_psf*. Its
        ``sci`` is left unconvolved and ``wht`` is used as-is. If
        ``None`` (default), every band is matched.
    method
        Matching-kernel regularization method passed to
        :func:`mophongo.utils.matching_kernel` /
        :meth:`PSF.optimize_matching_kernel_regularization`.
    reg_grid
        Regularization-parameter grid for the optimizer. Default uses
        the optimizer's own default.
    output_sci, output_wht
        Optional FITS paths to write the coadded science and weight
        images. Uses *header* if supplied.
    header
        FITS header to attach to the written outputs (typically the
        target band's science header).
    diagnostic_dir
        Optional directory for per-band PSF matching diagnostics
        (one PNG per matched band).
    return_kernels
        If True, the per-band kernels are included in the returned dict.

    Returns
    -------
    dict
        ``{"sci": coadd_sci, "wht": coadd_wht,
        "info": [{"name", "reg", "score", "sum_k2"}, ...]}``.
        Adds ``"kernels": [...]`` when *return_kernels* is True.
    """
    from .psf import PSF  # local import to avoid circular dependency

    if not bands:
        raise ValueError("bands list is empty")

    def _load_array(x: _Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        return fits.getdata(str(x)).astype(np.float64)

    target_psf = np.asarray(target_psf, dtype=float)
    target_psf = target_psf / target_psf.sum()
    target_psf_obj = PSF(target_psf)

    diag_dir = Path(diagnostic_dir) if diagnostic_dir else None
    if diag_dir is not None:
        diag_dir.mkdir(parents=True, exist_ok=True)

    sum_wI: np.ndarray | None = None
    sum_w: np.ndarray | None = None
    info: list[dict[str, _Any]] = []
    kernels: list[np.ndarray] = []

    for i, band in enumerate(bands):
        name = str(band.get("name", f"band{i}"))
        sci = _load_array(band["sci"])
        wht = _load_array(band["wht"]).astype(np.float64)
        psf = np.asarray(band["psf"], dtype=float)
        psf = psf / psf.sum()

        is_target = (target_index is not None and i == target_index)
        if is_target:
            kernel = None
            sum_k2 = 1.0
            reg = float("nan")
            score = float("nan")
            sci_match = sci.astype(np.float64, copy=False)
            wht_match = wht
        else:
            psf_obj = PSF(psf)
            diag_path = (
                diag_dir / f"matched_{name}.png" if diag_dir is not None else None
            )
            result = psf_obj.optimize_matching_kernel_regularization(
                target_psf_obj,
                method=method,
                reg_grid=reg_grid,
                diagnostic_path=str(diag_path) if diag_path else None,
                source_label=f"{name} PSF",
                target_label="target PSF",
            )
            kernel = np.asarray(result.kernel, dtype=float)
            sum_k2 = float(np.sum(kernel * kernel))
            if not np.isfinite(sum_k2) or sum_k2 <= 0:
                raise RuntimeError(
                    f"non-positive sum(kernel**2) for band {name!r}"
                )
            reg = float(result.reg)
            score = float(result.score)
            sci_match = _scipy_fftconvolve(sci.astype(np.float64), kernel, mode="same")
            wht_match = wht / sum_k2

        if sum_wI is None:
            sum_wI = wht_match * sci_match
            sum_w = wht_match.copy()
        else:
            if sci_match.shape != sum_wI.shape:
                raise ValueError(
                    f"band {name!r} shape {sci_match.shape} != target "
                    f"shape {sum_wI.shape}"
                )
            sum_wI += wht_match * sci_match
            sum_w += wht_match

        info.append({
            "name": name, "reg": reg, "score": score, "sum_k2": sum_k2,
        })
        if return_kernels and kernel is not None:
            kernels.append(kernel)
        elif return_kernels:
            kernels.append(np.array([[1.0]], dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        coadd_sci = np.where(sum_w > 0, sum_wI / sum_w, 0.0)
    coadd_wht = sum_w

    if output_sci is not None:
        fits.writeto(
            str(output_sci),
            coadd_sci.astype(np.float32),
            header=header,
            overwrite=True,
        )
        logger.info("wrote coadd sci -> %s", output_sci)
    if output_wht is not None:
        fits.writeto(
            str(output_wht),
            coadd_wht.astype(np.float32),
            header=header,
            overwrite=True,
        )
        logger.info("wrote coadd wht -> %s", output_wht)

    out = {"sci": coadd_sci, "wht": coadd_wht, "info": info}
    if return_kernels:
        out["kernels"] = kernels
    return out
