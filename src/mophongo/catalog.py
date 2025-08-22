"""Basic source catalog creation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from astropy.convolution import Gaussian2DKernel, Box2DKernel
from astropy.io import fits
from astropy.table import Table
from photutils.background import MADStdBackgroundRMS
from astropy.stats import SigmaClip, mad_std
from astropy.wcs import WCS
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    SegmentationImage,
)
from photutils.segmentation.catalog import DEFAULT_COLUMNS
from photutils.segmentation import deblend_sources
from skimage.morphology import binary_dilation, dilation, disk  # , square
from skimage.segmentation import watershed
from skimage.measure import label

from itertools import product

from astropy.nddata import block_reduce, block_replicate

import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

__all__ = [
    "Catalog",
]


def enlarge_slice(slc, shape, pad):
    """
    Enlarge a 2D slice by pad pixels on each side, clipped to array boundaries.

    Parameters
    ----------
    slc : tuple of slices
        (slice_y, slice_x)
    shape : tuple
        (ny, nx) shape of the array
    pad : int
        Number of pixels to pad on each side

    Returns
    -------
    tuple of slices
        Enlarged (slice_y, slice_x)
    """
    y0 = max(slc[0].start - pad, 0)
    y1 = min(slc[0].stop + pad, shape[0])
    x0 = max(slc[1].start - pad, 0)
    x1 = min(slc[1].stop + pad, shape[1])
    return (slice(y0, y1), slice(x0, x1))


from scipy.ndimage import binary_dilation


import numpy as np

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import binary_dilation, gaussian_filter, zoom
from astropy.stats import mad_std
from photutils.segmentation import detect_sources
from skimage.morphology import disk

# --- helpers ---------------------------------------------------------------


def _mean_downsample(arr: np.ndarray, fact: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    H, W = a.shape
    H2, W2 = (H // fact) * fact, (W // fact) * fact
    if H2 != H or W2 != W:
        a = a[:H2, :W2]
    a = a.reshape(H2 // fact, fact, W2 // fact, fact)
    return a.mean(axis=(1, 3), dtype=np.float32)


def bg_gaussian_normalized(img, bgmask, sigma=20.0, truncate=3.0):
    """Mask-aware smoothing: (G*(I*M)) / (G*M)."""
    M = bgmask.astype(np.float32)
    num = gaussian_filter(
        img.astype(np.float32) * M, sigma=sigma, truncate=truncate, mode="nearest"
    )
    den = gaussian_filter(M, sigma=sigma, truncate=truncate, mode="nearest")
    out = np.zeros_like(num, dtype=np.float32)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    if np.any(~ok):
        # one broad fill pass (inpaint)
        out2 = gaussian_filter(out, sigma=4 * sigma, truncate=3.0, mode="nearest")
        out[~ok] = out2[~ok]
    return out


# --- main ------------------------------------------------------------------


def expand_to_full(img_binned: np.ndarray, step: int, full_shape: tuple[int, int]) -> np.ndarray:
    """
    Linearly upsample a coarse image/mask to `full_shape` with bilinear interpolation.

    Parameters
    ----------
    img_binned : (Hc, Wc) coarse array (float or bool)
    step       : nominal binning factor (unused in math, kept for API symmetry)
    full_shape : (H, W) target shape

    Returns
    -------
    (H, W) float32 interpolated array in [0, 1] if input was a mask.
    """
    Hc, Wc = img_binned.shape
    H, W = full_shape
    zy = H / float(Hc)
    zx = W / float(Wc)
    out = zoom(img_binned.astype(np.float32), (zy, zx), order=1, mode="nearest", prefilter=False)
    # Ensure exact shape
    out = out[:H, :W]
    if out.shape[0] < H or out.shape[1] < W:
        out = np.pad(out, ((0, H - out.shape[0]), (0, W - out.shape[1])), mode="edge")
    return out.astype(np.float32)


def get_bg_and_ivar(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    bg_filter_sigma: float = 64.0,
    detect_thresh: float = 1.0,
    dilate: int = 3,
):
    """
    Fit a smooth background on the coarse grid (mask-aware), subtract it,
    measure robust σ on bg pixels, and rescale the full-res ivar.

    Returns
    -------
    ivar_new     : float32 ndarray (H, W)
    bg_coarse    : float32 ndarray (Hc, Wc)
    det_coarse   : float32 ndarray (Hc, Wc) after bg subtraction
    bgmask_full  : float32 ndarray (H, W) linearly interpolated mask in [0,1]
    """
    step = np.floor(np.sqrt(bg_filter_sigma)).astype(int)
    min_npixels_bright = step**2
    min_npixels_faint = 1

    s = np.asarray(sci, dtype=np.float32)
    w = np.asarray(wht, dtype=np.float32)
    valid_w = np.isfinite(w) & (w > 0)
    w = np.where(valid_w, w, 0.0).astype(np.float32)

    # 1) coarse block means
    s_bin = _mean_downsample(s, step)
    w_bin = _mean_downsample(w, step)
    pos = w_bin > 0

    # 2) coarse detection image (S/N)
    det = np.zeros_like(s_bin, dtype=np.float32)
    det[pos] = s_bin[pos] * np.sqrt(w_bin[pos], dtype=np.float32)

    # 3) robust baseline
    med0 = np.median(det).astype(np.float32)
    nmad0 = np.median(np.abs(det - med0) * np.float32(1.4826)).astype(np.float32)
    sigma0 = nmad0 if nmad0 > 0 else np.std(det).astype(np.float32)

    # 4) quick source mask & local bg pre-estimate (on det)
    mask_src0 = det > (med0 + 3.0 * sigma0)

    # smooth DET for bright detection; use convolution noise factor
    kern = disk(max(dilate, 1)).astype(np.float32)
    detc = fftconvolve(det - med0, kern, mode="same")
    sigma_correct = np.sqrt((kern**2).sum()) / kern.sum()

    seg_bright = detect_sources(
        detc,
        threshold=detect_thresh * sigma_correct * sigma0,
        npixels=min_npixels_bright,
        connectivity=8,
    )
    seg_faint = detect_sources(
        det, threshold=detect_thresh * sigma0, npixels=min_npixels_faint, connectivity=8
    )

    seg_all = (seg_bright.data if seg_bright is not None else 0) + (
        seg_faint.data if seg_faint is not None else 0
    )

    bgmask = seg_all == 0
    if dilate > 0:
        bgmask = binary_dilation(bgmask, structure=kern)
    bgmask &= pos  # exclude zero-weight tiles

    # 5) fit smooth background on the COARSE SCI (not det)
    #    convert sigma to coarse pixels
    bg_sigma_bin = max(float(step), 8.0)
    bg_img_bin = bg_gaussian_normalized(s_bin, bgmask, sigma=bg_sigma_bin, truncate=3.0)

    # 6) subtract bg and re-measure σ on bg pixels
    s_bin_bsub = s_bin - bg_img_bin
    det_bsub = np.zeros_like(det, dtype=np.float32)
    det_bsub[pos] = s_bin_bsub[pos] * np.sqrt(w_bin[pos], dtype=np.float32)

    bg_ok = bgmask & np.isfinite(det_bsub)
    if not np.any(bg_ok):
        # fallback: use all valid pixels
        bg_ok = pos
    sigma_bin = mad_std(det_bsub[bg_ok].astype(np.float32))
    sigma_true = np.float32(step) * np.float32(sigma_bin)

    # 7) rescale full-res weights
    scale = np.float32(1.0) / (sigma_true * sigma_true + np.float32(1e-30))
    ivar_new = np.where(valid_w, (w * scale).astype(np.float32), 0.0).astype(np.float32)

    # Linearly upsample bgmask to full resolution
    bg_img = expand_to_full(bg_img_bin.astype(np.float32), step, s.shape)
    bg_img[~valid_w] = 0.0  # zero out invalid pixels

    return bg_img, ivar_new


def calibrate_ivar_with_bg_median(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    bg_scale: int = 64,  # area in native px; bin factor = sqrt(bg_scale)
    detect_sigma: float = 2.0,  # n-threshold in coarse S/N units
    ndilate: int = 2,  # dilation radius on coarse grid
    bg_smooth_sigma_bin: float = 2.0,  # Gaussian sigma (coarse px) for bg smoothing
) -> tuple[np.ndarray, np.ndarray]:
    """
    Background/noise calibration via block-sum + median detrending and two-pass detection.

    Steps
    -----
    - Bin by N = round(sqrt(bg_scale)) using SUM for science and MEAN for weights.
    - Build det image: det_bin = sci_bin * sqrt(w_bin).
    - Median-filter det_bin (size=N) and subtract for initial trend removal.
    - Estimate σ via MAD on detrended det_bin.
    - Two-pass detection on det_bin:
        1) convolve with disk(2), detect at detect_sigma*σ*sigma_conv_correct, npixels=N*N
        2) detect on raw detrended det_bin at detect_sigma*σ
        Combine masks and dilate by ndilate.
    - Measure background on sci_bin with bg_gaussian_normalized + bgmask.
    - Recompute σ on bg pixels after bg subtraction, then correct for bin: σ_full = σ_bin / N.
    - Rescale full-res weights by 1/σ_full^2 and upsample per-pixel background.

    Returns
    -------
    ivar_new : float32 (H, W)
    bg_full  : float32 (H, W)
    """
    s = np.asarray(sci, dtype=np.float32)
    w = np.asarray(wht, dtype=np.float32)

    valid_w = np.isfinite(w) & (w > 0)
    w = np.where(valid_w, w, 0.0).astype(np.float32)

    N = max(1, int(round(np.sqrt(float(bg_scale)))))

    # Block-sum science; weights downsampled by mean
    s_bin = block_reduce(s, N, func=np.mean).astype(np.float32)
    w_bin = _mean_downsample(w, N)
    pos = w_bin > 0

    # Noise-equalised coarse DET
    det_bin = np.zeros_like(s_bin, dtype=np.float32)
    det_bin[pos] = s_bin[pos] * np.sqrt(w_bin[pos])

    # Median filter detrending on DET
    k_med = max(5, N)
    bg_bin = bg_gaussian_normalized(s_bin, bgmask, sigma=float(bg_smooth_sigma_bin), truncate=3.0)

    det_trend = median_filter(det_bin, size=k_med, mode="nearest")
    det0 = det_bin - det_trend

    # Robust σ on detrended DET
    ok0 = np.isfinite(det0) & pos
    if not np.any(ok0):
        ok0 = pos
    sigma0 = mad_std(det0[ok0].astype(np.float32))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = np.std(det0[ok0].astype(np.float32))

    # Two-pass detection
    k = disk(2).astype(np.float32)
    detc = fftconvolve(det0, k, mode="same")
    sigma_conv = np.sqrt((k**2).sum()) / k.sum()

    seg1 = detect_sources(
        detc,
        threshold=float(detect_sigma) * float(sigma0) * float(sigma_conv),
        npixels=N * N,
        connectivity=8,
    )
    seg2 = detect_sources(
        det0,
        threshold=float(detect_sigma) * float(sigma0),
        npixels=3,  # conservative second pass
        connectivity=8,
    )

    m1 = (seg1.data > 0) if (seg1 is not None) else 0
    m2 = (seg2.data > 0) if (seg2 is not None) else 0
    seg_mask = (m1 | m2).astype(bool)

    # Background mask = not detected and valid weight
    bgmask = (~seg_mask) & pos
    if ndilate > 0:
        bgmask = binary_dilation(bgmask, structure=disk(int(ndilate)))

    # Background on SUM-binned science, mask-aware smoothing
    bg_bin = bg_gaussian_normalized(s_bin, bgmask, sigma=float(bg_smooth_sigma_bin), truncate=3.0)

    # Recompute σ on bg pixels after bg subtraction (in DET space)
    s_bin_bsub = s_bin - bg_bin
    det_bsub = np.zeros_like(det_bin, dtype=np.float32)
    det_bsub[pos] = s_bin_bsub[pos] * np.sqrt(w_bin[pos])

    ok_bg = bgmask & np.isfinite(det_bsub)
    if not np.any(ok_bg):
        ok_bg = pos
    sigma_bin = mad_std(det_bsub[ok_bg].astype(np.float32))
    if not np.isfinite(sigma_bin) or sigma_bin <= 0:
        sigma_bin = np.std(det_bsub[ok_bg].astype(np.float32))

    # Bin-correct to native pixel units
    sigma_full = float(sigma_bin) * float(N)

    # Rescale full-res inverse variance
    scale = np.float32(1.0) / (np.float32(sigma_full) ** 2 + np.float32(1e-30))
    ivar_new = np.where(valid_w, (w * scale).astype(np.float32), 0.0).astype(np.float32)

    # Convert SUM background back to per-pixel MEAN before upsampling
    bg_full = expand_to_full(bg_bin.astype(np.float32), N, s.shape)
    bg_full[~valid_w] = 0.0

    return ivar_new, bg_full


def safe_dilate_segmentation(segmap: SegmentationImage, selem=disk(1.5)):
    """
    Efficiently dilate segments in a SegmentationImage, only into background.
    Works on small enlarged slices for each segment for speed.
    """

    result = np.zeros_like(segmap.data)
    pad = max(selem.shape) // 2
    arr_shape = segmap.data.shape
    for segment in segmap.segments:
        seg_id = segment.label
        if seg_id == 0:
            continue  # skip background
        slc = enlarge_slice(segment.slices, arr_shape, pad)
        mask = segmap.data[slc] == seg_id
        dilated = binary_dilation(mask, selem)
        bg_mask = segmap.data[slc] == 0
        dilated_bg = np.logical_and(dilated, bg_mask)
        sub_result = result[slc]
        sub_result[dilated_bg] = seg_id
        sub_result[mask] = seg_id  # retain original
        result[slc] = sub_result
    return result


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import binary_dilation as _dilate
from astropy.stats import mad_std


def _mean_downsample(arr, fact):
    """Fast block-reduce by *fact*×*fact* using strides, no Python loops."""
    ny, nx = arr.shape
    ny2, nx2 = ny // fact, nx // fact
    trimmed = arr[: ny2 * fact, : nx2 * fact]  # drop edge pixels
    view = trimmed.reshape(ny2, fact, nx2, fact)
    return view.mean(axis=(1, 3), dtype=arr.dtype)


def _sigma_clip(arr, sigma=3.0):
    """Vectorised σ-clip that returns a boolean mask."""
    med = np.median(arr)
    dev = sigma * mad_std(arr, ignore_nan=True)
    return np.abs(arr - med) > dev


def noise_equalised_image(data: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Return image divided by the per-pixel noise."""
    if weight is None:
        return data
    return data * np.sqrt(weight)


def detect_peaks(
    img_eq: np.ndarray,
    sigma: float = 3.0,
    npix_min: int = 5,
    kernel_w: int = 3,
) -> tuple[SegmentationImage, SourceCatalog]:
    """Detect peaks in a noise-equalised image."""

    kernel = Gaussian2DKernel(kernel_w / 2.355, x_size=npix_min, y_size=npix_min)

    sm = fftconvolve(img_eq, kernel.array, mode="same")
    std = mad_std(sm)
    seg = detect_sources(sm, sigma * std, npixels=npix_min)
    props = SourceCatalog(img_eq, seg)
    return seg, props


def fit_psf_stamp(
    data: np.ndarray,
    sigma: np.ndarray,
    psf_model: np.ndarray,
) -> tuple[float, float]:
    """Fit a PSF to a small stamp and return flux and reduced chi^2."""

    y, x = np.indices(data.shape)
    flat = np.ones_like(psf_model)

    A = np.vstack([(psf_model / sigma), (flat / sigma)]).reshape(2, -1).T
    b = (data / sigma).ravel()
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    model = coeff[0] * psf_model + coeff[1]
    chi2 = np.sum(((data - model) ** 2) / sigma**2)
    dof = data.size - 2
    return coeff[0], chi2 / dof


def vet_by_chi2(star_list: Table, chi2_max: float = 3.0) -> Table:
    """Filter table rows by reduced chi^2."""

    mask = star_list["chi2_red"] < chi2_max
    return star_list[mask]


import numpy as np
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.stats import mad_std
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import minimum_filter


def _expand_remap(pos_xy, k):
    # center-of-pixel convention (pixel centers at integers)
    shift = (k - 1) / 2.0
    x, y = pos_xy
    return (x + shift) * k, (y + shift) * k


import numpy as np
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.stats import mad_std
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import minimum_filter


def _expand_remap(pos_xy, k):
    # center-of-pixel convention (pixel centers at integers)
    shift = (k - 1) / 2.0
    x, y = pos_xy
    return (x + shift) * k, (y + shift) * k


def find_saturated_stars(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    nbin: int = 8,
    ncen: int = 3,  # odd; min over ncen×ncen binned pixels at centroid
    sigma: float = 5.0,
    npixels: int = 50,
    return_seg: bool = False,
):
    """
    Fast saturated-star finder with neighborhood min in the binned weight.

    Steps:
      1) sci_b = mean nbin×nbin; wht_b_min = min nbin×nbin
      2) det = sci_b * sqrt(max(wht_b_min, 0))  (noise–equalized)
      3) detect_sources(det, threshold=sigma*mad_std(det), npixels=npixels)
      4) for each source, compute min over an ncen×ncen window of wht_b_min
         centered on the (binned) centroid; flag saturated if that min ≤ 0
      5) return centroids on binned grid and mapped back to full-res
    """
    # sanitize & bin
    sci = np.asarray(sci, dtype=np.float32)
    wht = np.asarray(wht, dtype=np.float32)
    wht = np.where(np.isfinite(wht) & (wht > 0), wht, 0.0)

    sci_b = block_reduce(sci, (nbin, nbin), func=np.mean).astype(np.float32)
    wht_b_min = block_reduce(wht, (nbin, nbin), func=np.min).astype(np.float32)

    # detector image & threshold
    det = sci_b * np.sqrt(np.maximum(wht_b_min, 0.0, dtype=np.float32))
    thr = float(sigma * mad_std(det, ignore_nan=True))

    # empty outputs on degenerate case
    def _empty(seg=None):
        out = Table(
            names=["id", "x_b", "y_b", "x", "y", "npix_b", "sat_flag"],
            dtype=[int, float, float, float, float, int, bool],
        )
        return (out, seg) if return_seg else out

    if not np.isfinite(thr) or thr <= 0:
        return _empty()

    seg = detect_sources(det, threshold=thr, npixels=npixels)
    if seg is None or seg.nlabels == 0:
        return _empty(seg)

    # catalog on the binned grid
    cat = SourceCatalog(sci_b, seg)
    xb = np.asarray(cat.xcentroid.value, dtype=np.float32)
    yb = np.asarray(cat.ycentroid.value, dtype=np.float32)

    # min over ncen×ncen around each centroid (on binned weight map)
    if ncen is None or ncen < 1:
        ncen = 1
    if ncen % 2 == 0:
        ncen += 1  # force odd
    wht_b_cenmin = minimum_filter(wht_b_min, size=(ncen, ncen), mode="nearest")

    yb_i = np.clip(np.rint(yb).astype(int), 0, wht_b_cenmin.shape[0] - 1)
    xb_i = np.clip(np.rint(xb).astype(int), 0, wht_b_cenmin.shape[1] - 1)
    sat_flag = wht_b_cenmin[yb_i, xb_i] <= 0.0

    # map centroids back to full-res
    x_full, y_full = _expand_remap((xb, yb), nbin)

    out = Table()
    out["id"] = np.asarray(cat.labels, dtype=int)
    out["x_b"] = xb
    out["y_b"] = yb
    out["x"] = x_full.astype(np.float32)
    out["y"] = y_full.astype(np.float32)
    out["npix_b"] = np.asarray(cat.area.value, dtype=int)  # binned-pixel area
    out["sat_flag"] = sat_flag
    # optionally expose the actual min value for debugging:
    # out["wht_b_cenmin"] = wht_b_cenmin[yb_i, xb_i].astype(np.float32)

    return (out, seg) if return_seg else out


# Add 'eccentricity' to the default columns for SourceCatalog
# if 'eccentricity' not in DEFAULT_COLUMNS:
#    DEFAULT_COLUMNS.append(['ra','dec','eccentricity'])

DEFAULT_COLUMNS = [
    "label",
    "xcentroid",
    "ycentroid",
    "sky_centroid",
    "area",
    "semimajor_sigma",
    "semiminor_sigma",
    "kron_radius",
    "eccentricity",
    "orientation",
    "min_value",
    "max_value",
    "local_background",
    "segment_flux",
    "segment_fluxerr",
    "kron_flux",
    "kron_fluxerr",
]


@dataclass
class CatConfig:
    """Configuration options for :class:`SparseFitter`."""

    # aperture in
    aperture: float | str = "use_aper"
    aperture_units: str = "arcsec"


@dataclass
class Catalog:
    """Create a catalog from a science image and weight map."""

    sci: np.ndarray
    wht: np.ndarray
    nbin: int = 4
    estimate_background: bool = False
    estimate_ivar: bool = False

    background: float = 0.0
    ivar: np.ndarray | None = None
    segmap: SegmentationImage | None = None
    catalog: SourceCatalog | None = None
    table: Table | None = None
    det_img: np.ndarray | None = None
    params: dict[str, float | int] = field(default_factory=dict)
    header: fits.Header | None = None
    wcs: WCS | None = None
    default_columns: list[str] = field(default_factory=lambda: DEFAULT_COLUMNS)

    def __post_init__(self) -> None:
        defaults = {
            "kernel_size": 3.5,
            "detect_npixels": 5,
            "detect_threshold": 2.0,
            "dilate_segmap": 2,
            "deblend_mode": "exponential",
            "deblend_nlevels": 32,
            "deblend_contrast": 1e-4,
            "deblend_compactness": 0.0,
            "background_filter_sigma": 64.0,
        }
        defaults.update(self.params)
        self.params = defaults

    @classmethod
    def from_fits(
        cls,
        sci: str | Path | np.ndarray,
        wht: str | Path | np.ndarray,
        *,
        segmap: str | Path | np.ndarray | SegmentationImage | None = None,
        header: fits.Header | None = None,
        **kwargs,
    ) -> "Catalog":
        # Load sci and wht if they are file paths, force float32
        if isinstance(sci, (str, Path)):
            sci_data = fits.getdata(sci).astype(np.float32)
            header = fits.getheader(sci)
        else:
            sci_data = np.asarray(sci).astype(np.float32)

        if isinstance(wht, (str, Path)):
            wht_data = fits.getdata(wht).astype(np.float32)
        else:
            wht_data = np.asarray(wht).astype(np.float32)

        # Handle segmap
        segmap_obj = None
        if segmap is not None:
            if isinstance(segmap, (str, Path)):
                segmap_obj = SegmentationImage(fits.getdata(segmap))
            elif isinstance(segmap, np.ndarray):
                segmap_obj = SegmentationImage(segmap)
            elif isinstance(segmap, SegmentationImage):
                segmap_obj = segmap
            else:
                raise ValueError("segmap must be a filename, ndarray, or SegmentationImage")

        obj = cls(sci_data, wht_data, segmap=segmap_obj, header=header, **kwargs)
        obj.run()
        return obj

    def _detect(self) -> None:
        self.det_img = (self.sci - self.background) * np.sqrt(self.ivar)
        kernel_pix = int(2 * self.params["kernel_size"]) | 1  # ensure odd size
        kernel = Gaussian2DKernel(
            self.params["kernel_size"] / 2.355, x_size=kernel_pix, y_size=kernel_pix
        )
        print(f"Convolving with kernel size {self.params['kernel_size']} pixels")
        smooth = fftconvolve(self.det_img, kernel.array, mode="same")
        print("Detecting sources...")
        segmap = detect_sources(
            smooth,
            threshold=float(self.params["detect_threshold"]),
            npixels=self.params["detect_npixels"],
        )
        # Dilate the segmentation map to include more pixels
        if self.params["dilate_segmap"] > 0:
            print(f"Dilating segmentation map with size {self.params['dilate_segmap']}")
            segmap.data = safe_dilate_segmentation(segmap, disk(self.params["dilate_segmap"]))
        if self.params["deblend_mode"] is not None:
            segmap = deblend_sources(
                self.det_img,
                segmap,
                npixels=self.params["detect_npixels"],
                mode=self.params["deblend_mode"],
                nlevels=int(self.params["deblend_nlevels"]),
                contrast=float(self.params["deblend_contrast"]),
                connectivity=8,
                progress_bar=False,
                #            compactness=float(self.params.get("deblend_compactness", 0.0)),
            )
        self.segmap = segmap

    def run(self) -> None:
        if self.estimate_background or self.estimate_ivar:
            print("Estimating background and inverse variance...")
            background, ivar = get_bg_and_ivar(
                self.sci,
                self.wht,
                bg_filter_sigma=self.params.get("background_filter_sigma", 64.0),
            )
            # ivar, background = calibrate_ivar_with_bg_median(
            #     self.sci, self.wht, bg_scale=self.nbin
            # )

            if self.estimate_ivar:
                self.ivar = ivar
            if self.estimate_background:
                print("Subtracting background...")
                self.background = background
                self.sci = self.sci - self.background

        else:  # assume wht is inverse variance
            self.ivar = self.wht

        if self.segmap is None:
            self._detect()

        if self.wcs is None and self.header is not None:
            self.wcs = WCS(self.header)

        self.catalog = SourceCatalog(
            self.sci, self.segmap, error=np.sqrt(1.0 / self.ivar), wcs=self.wcs
        )
        # Compute r50 and sharpness for each source and add to table
        self.table = self.catalog.to_table(self.default_columns)
        self.table["r50"] = self.catalog.fluxfrac_radius(0.5).value
        self.table["eccentricity"] = self.catalog.eccentricity.value
        self.table["sharpness"] = (
            self.catalog.max_value * np.pi * self.table["r50"] ** 2 / self.catalog.segment_flux
        ).value
        self.table["snr"] = self.table["segment_flux"] / self.table["segment_fluxerr"]
        self.table.rename_columns(["label", "xcentroid", "ycentroid"], ["id", "x", "y"])
        if "sky_centroid" in self.table.colnames:
            self.table["ra"] = [
                sc.ra.deg if sc is not None else np.nan for sc in self.table["sky_centroid"]
            ]
            self.table["dec"] = [
                sc.dec.deg if sc is not None else np.nan for sc in self.table["sky_centroid"]
            ]
            self.table.remove_column("sky_centroid")

    def find_stars(
        self,
        *,
        psf: np.ndarray | None = None,
        snr_min: float = 100,
        r50_max: float = 5,
        eccen_max: float = 0.2,
        sharp_lohi: tuple[float, float] = (0.2, 1.2),
        chi2_max: float = 3.0,
        return_seg: bool = False,
    ) -> Table | tuple[Table, SegmentationImage]:
        """Find point sources in the catalog image."""

        point_like = (
            (self.table["r50"] < r50_max)
            & (self.table["eccentricity"] < eccen_max)
            & (self.table["sharpness"] > sharp_lohi[0])
            & (self.table["sharpness"] < sharp_lohi[1])
        )
        self.table["point_like"] = point_like

        table = self.table.copy()
        print("found", len(table), "sources")

        idx_stars = np.where(point_like & (table["snr"] > snr_min))[0]
        table = table[idx_stars]
        print("kept", len(table), "point-like sources")

        if psf is not None and len(table) > 0:
            print("fitting PSF to stamps")
            chi2 = []
            flux_psf = []
            half = psf.shape[0] // 2
            for row in table:
                y0 = int(row["y"])
                x0 = int(row["x"])
                y_slice = slice(max(0, y0 - half), min(self.sci.shape[0], y0 + half + 1))
                x_slice = slice(max(0, x0 - half), min(self.sci.shape[1], x0 + half + 1))
                stamp = self.sci[y_slice, x_slice]
                sigma_im = np.sqrt(1.0 / self.ivar[y_slice, x_slice])
                if stamp.shape != psf.shape:
                    chi2.append(np.inf)
                    flux_psf.append(np.nan)
                    continue
                flux, c2 = fit_psf_stamp(stamp, sigma_im, psf)
                chi2.append(c2)
                flux_psf.append(flux)
            table["flux_psf"] = flux_psf
            table["chi2_red"] = chi2
        #            table = vet_by_chi2(table, chi2_max)

        return table, idx_stars

    def show_stamp(
        self,
        idnum: int,
        offset: float = 3e-5,
        buffer: int = 20,
        ax=None,
        cmap="gray",
        alpha=0.2,
        keys: list[str] | None = None,
    ):
        """
        Show a cutout stamp of the source with segmentation mask overlay.

        Parameters
        ----------
        idnum : int
            Catalog row index (not segment label).
        buffer : int, optional
            Number of pixels to pad around the segmentation footprint.
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates a new figure.
        cmap : str, optional
            Colormap for the image.
        alpha : float, optional
            Alpha for the segmentation mask overlay.
        label : list of str, optional
            List of column names to display as text in the top left.
        """
        if self.segmap is None or self.catalog is None:
            raise RuntimeError("Catalog must be detected (run .run()) before calling showstamp.")

        idx = self.table["id"] == idnum
        row = self.table[idx][0]

        x = int(round(row["x"]))
        y = int(round(row["y"]))
        segm = self.segmap  # Already a SegmentationImage
        label_val = segm.data[y, x]
        if label_val == 0:
            raise ValueError("Selected position is not inside any segment.")

        idx = segm.get_index(label_val)
        bbox = segm.bbox[idx]
        # Expand bbox by buffer
        iymin = max(bbox.iymin - buffer, 0)
        iymax = min(bbox.iymax + buffer, self.sci.shape[0] - 1)
        ixmin = max(bbox.ixmin - buffer, 0)
        ixmax = min(bbox.ixmax + buffer, self.sci.shape[1] - 1)

        stamp = self.sci[iymin : iymax + 1, ixmin : ixmax + 1]
        segmask = segm.data[iymin : iymax + 1, ixmin : ixmax + 1]
        #        segmask[segmask != 0] = segmask[segmask != 0] - label_val + 1  # Keep only the selected segment
        scl = stamp[segmask == label_val].sum()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure  # <-- add this line
            titles = ["data", "psf", "data - psf", "data - psf x kernel", "kernel"]
            kws = dict(
                vmin=-5.3, vmax=-1.5, cmap="bone_r", origin="lower", interpolation="nearest"
            )

            from matplotlib.colors import ListedColormap

            # Create a new colormap with the first color transparent
            cmap = segm.cmap
            cmap_mod = ListedColormap(np.vstack(([0, 0, 0, 0], cmap(np.arange(1, cmap.N)))))
            ax.imshow(np.log10(stamp / scl + offset), **kws)
            ax.imshow(segmask, origin="lower", cmap=cmap_mod, alpha=alpha)
            ax.axis("off")
            ax.set_title(f"ID {idnum}")
            text_lines = []
            if keys:
                for col in keys:
                    val = row[col]
                    try:
                        sval = f"{val:.2f}"
                    except Exception:
                        sval = str(val)
                    text_lines.append(f"{col}: {sval}")
            if text_lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(text_lines),
                    color="w",
                    fontsize=10,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.4, lw=0),
                )

        return fig, ax

    def plot_bg(
        self,
        nbin: int | None = None,
        *,
        fac: float = 1.0,
        figsize: tuple[int, int] = (20, 10),
    ):
        """
        Show 2x2 panels on a downsampled grid:
          [0] image, [1] image with sources plotted (segmap overlay if present),
          [2] background, [3] bg-subtracted and noise-equalised.

        Uses cached self.background, self.ivar/self.wht, self.segmap.data only.
        """
        nb = int(nbin if nbin is not None else self.nbin)

        # reconstruct full image (in case self.sci is already bg-subtracted)
        bg_full = self.background
        if np.isscalar(bg_full):
            bg_img = float(bg_full)
        else:
            bg_img = np.asarray(bg_full, dtype=np.float32)

        sci_full = self.sci + (0.0 if np.isscalar(bg_img) else bg_img)

        # weights
        w_full = self.ivar if self.ivar is not None else self.wht
        w_full = np.asarray(w_full, dtype=np.float32)
        w_full = np.where(np.isfinite(w_full) & (w_full > 0), w_full, 0.0)

        # downsample
        s_bin = _mean_downsample(np.asarray(sci_full, dtype=np.float32), nb)
        if np.isscalar(bg_img):
            bg_bin = np.full_like(s_bin, float(bg_img), dtype=np.float32)
        else:
            bg_bin = _mean_downsample(bg_img.astype(np.float32), nb)

        w_bin = _mean_downsample(w_full, nb)
        pos = w_bin > 0

        # noise-equalised, bg-subtracted on coarse grid
        det_bin = np.zeros_like(s_bin, dtype=np.float32)
        det_bin[pos] = (s_bin[pos] - bg_bin[pos]) * np.sqrt(w_bin[pos])
        mscale = np.median(w_bin[pos]) if np.any(pos) else 1.0

        # segmap overlay (if available)
        seg_overlay = None
        if (
            getattr(self, "segmap", None) is not None
            and getattr(self.segmap, "data", None) is not None
        ):
            seg = (self.segmap.data > 0).astype(np.float32)
            seg_overlay = _mean_downsample(seg, nb) > 0.0

        # plot
        v = float(np.std(s_bin)) * fac / nbin
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax = ax.flatten()

        ax[0].imshow(s_bin, origin="lower", cmap="gray", vmin=-v, vmax=v)
        ax[0].set_title("image (binned)")

        ax[1].imshow(s_bin, origin="lower", cmap="gray", vmin=-v, vmax=v)
        ax[1].set_title("sources plotted")
        if seg_overlay is not None:
            from matplotlib.colors import ListedColormap

            cmap = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 0.5]])
            ax[1].imshow(
                seg_overlay.astype(int), origin="lower", cmap=cmap, interpolation="nearest"
            )

        ax[2].imshow(bg_bin, origin="lower", cmap="gray", vmin=-v, vmax=v)
        ax[2].set_title("background (binned)")

        ax[3].imshow(
            det_bin,
            origin="lower",
            cmap="gray",
            vmin=-v * np.sqrt(mscale),
            vmax=v * np.sqrt(mscale),
            interpolation="nearest",
        )
        ax[3].set_title("bg-subtracted × sqrt(w)")

        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        return fig, ax
