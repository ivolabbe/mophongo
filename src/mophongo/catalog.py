"""Basic source catalog creation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std
from astropy.wcs import WCS
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    SegmentationImage,
)
from photutils.segmentation.catalog import DEFAULT_COLUMNS
from photutils.segmentation import deblend_sources
from skimage.morphology import binary_dilation, dilation, disk  # , square
from skimage.measure import label


from astropy.nddata import block_reduce

import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

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
from scipy.ndimage import binary_dilation, gaussian_filter, zoom
from astropy.stats import mad_std
from photutils.segmentation import detect_sources
from skimage.morphology import disk

from .utils import as_label_array, fftconvolve

# --- helpers ---------------------------------------------------------------


def bg_gaussian_normalized(img, bgmask, sigma=20.0, truncate=3.0):
    """Mask-aware smoothing: (G*(I*M)) / (G*M).

    Non-finite samples are dropped from the mask and replaced explicitly.
    ``NaN * 0`` is ``NaN``, so zeroing them through the mask alone lets one
    bad pixel spread across the whole smoothing footprint even when the mask
    already excludes it.
    """
    M = (np.asarray(bgmask, dtype=bool) & np.isfinite(img)).astype(np.float32)
    num = gaussian_filter(
        np.where(M > 0, img, 0.0).astype(np.float32),
        sigma=sigma, truncate=truncate, mode="nearest",
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


#: Minimum fraction of valid coarse blocks that must survive the source mask
#: for the weight calibration to be believable. Below this the background fit
#: has nothing to smooth over and the measured scatter collapses; the
#: estimator warns and leaves the weight map unscaled instead.
MIN_BG_FRACTION = 0.02


def coarse_source_mask(
    det: np.ndarray,
    sigma0: float,
    *,
    detect_thresh: float = 2.5,
    faint_thresh: float = 4.0,
    dilate: int = 3,
    min_npixels_bright: int = 64,
    min_npixels_faint: int = 3,
) -> np.ndarray:
    """Flag source pixels on a coarse noise-equalised detection image.

    Two passes: a smoothed one for extended flux and an unsmoothed per-pixel
    one for compact sources, unioned and then dilated.

    Args:
        det: Coarse detection image (science times sqrt(weight)), already
            median-subtracted.
        sigma0: Robust per-pixel sigma of ``det``.
        detect_thresh: Bright-pass threshold in units of the sigma of the
            *smoothed* image.
        faint_thresh: Faint-pass threshold in units of ``sigma0``. Higher
            than ``detect_thresh`` because nothing suppresses the noise
            behind it.
        dilate: Disk radius for the smoothing kernel and the final dilation.
        min_npixels_bright: Minimum connected area for the bright pass.
        min_npixels_faint: Minimum connected area for the faint pass. A lone
            pixel over threshold is a noise spike, not a source.

    Returns:
        Boolean array, ``True`` where a source is flagged.
    """
    # The smoothing kernel is normalised so that ``detect_thresh`` is in units
    # of the smoothed noise. An unnormalised disk multiplies white noise by
    # sqrt(N) while the 1/sqrt(N) factor belongs to the normalised kernel, so
    # mixing them puts the threshold a factor N (29 for disk(3)) too low and
    # flags roughly half of a pure noise field.
    kern = disk(max(dilate, 1)).astype(np.float32)
    kern_norm = kern / kern.sum()
    detc = fftconvolve(det, kern_norm, mode="same")
    sigma_smooth = np.sqrt((kern_norm**2).sum()) * sigma0

    seg_bright = detect_sources(
        detc,
        threshold=detect_thresh * sigma_smooth,
        npixels=min_npixels_bright,
        connectivity=8,
    )
    seg_faint = detect_sources(
        det,
        threshold=faint_thresh * sigma0,
        npixels=min_npixels_faint,
        connectivity=8,
    )

    src_mask = np.zeros(det.shape, dtype=bool)
    if seg_bright is not None:
        src_mask |= np.asarray(seg_bright.data) > 0
    if seg_faint is not None:
        src_mask |= np.asarray(seg_faint.data) > 0

    # Dilate the SOURCE mask. Dilating the background mask instead grows
    # background *into* the sources and erodes the exclusion, which re-admits
    # most of a compact source's own pixels.
    if dilate > 0:
        src_mask = binary_dilation(src_mask, structure=kern > 0)
    return src_mask


def get_bg_and_ivar(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    bg_filter_sigma: float = 64.0,
    detect_thresh: float = 2.5,
    faint_thresh: float = 4.0,
    dilate: int = 3,
    label: str = "",
    need_bg: bool = True,
):
    """
    Fit a smooth background on the coarse grid (mask-aware), subtract it,
    measure robust σ on bg pixels, and rescale the full-res ivar.

    Bins the image, masks sources with a two-pass detection (bright,
    smoothed + faint, per-pixel), fits a mask-aware smoothed background on
    the coarse grid, and measures the robust scatter of background pixels
    after subtraction.

    Parameters
    ----------
    sci, wht
        Science image and weight map.
    bg_filter_sigma
        Sets the coarse-grid bin factor (``floor(sqrt(bg_filter_sigma))``)
        and the scale of the mask-aware Gaussian background smoothing.
    detect_thresh
        Bright-pass threshold for masking sources out of the background fit,
        in units of the sigma of the *smoothed* coarse detection image.
    faint_thresh
        Faint-pass threshold, in units of the per-pixel sigma of the
        unsmoothed coarse detection image. Higher than ``detect_thresh``
        because this pass has no smoothing to suppress noise behind it.
    dilate
        Disk radius for smoothing the coarse detection image and for
        dilating the source mask before it is excluded.
    need_bg
        Build the full-resolution background image. Pass ``False`` when only
        the calibrated weights are wanted: on a mosaic-sized input the
        background alone is another 3.5 GB, and its median is logged here
        anyway. ``bg_img`` is then ``None``.

    Returns
    -------
    bg_img   : float32 ndarray (H, W), background interpolated back to
               full resolution, or None when ``need_bg`` is False
    ivar_new : float32 ndarray (H, W), weight map rescaled to a calibrated
               inverse variance
    """
    step = np.floor(np.sqrt(bg_filter_sigma)).astype(int)
    min_npixels_bright = step**2
    # a single pixel over threshold is a noise spike, not a source: at the
    # faint pass's per-pixel threshold, requiring a connected triple is what
    # keeps the mask off the noise field
    min_npixels_faint = 3

    s = np.asarray(sci, dtype=np.float32)
    w = np.asarray(wht, dtype=np.float32)
    valid_w = np.isfinite(w) & (w > 0)
    # One common mask: a pixel counts only where BOTH science and weight are
    # finite. Non-finite science is replaced here rather than downstream --
    # a single NaN otherwise spreads over its whole block in the mean, into
    # the median and MAD, and makes every statistic and both outputs NaN.
    valid = valid_w & np.isfinite(s)

    # 1) coarse block means, taken over the valid pixels of each block, so a
    #    bad pixel costs its block sample size instead of poisoning it.
    #    Masked full-resolution copies of sci and wht are never materialised:
    #    see _valid_block_means.
    vfrac, s_sum_bin, w_sum_bin = _valid_block_means(s, w, valid, step)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_bin = np.where(vfrac > 0, s_sum_bin / vfrac, 0.0).astype(np.float32)
        w_bin = np.where(vfrac > 0, w_sum_bin / vfrac, 0.0).astype(np.float32)
    pos = (vfrac > 0) & (w_bin > 0)

    # 2) coarse detection image (S/N)
    det = np.zeros_like(s_bin, dtype=np.float32)
    det[pos] = s_bin[pos] * np.sqrt(w_bin[pos], dtype=np.float32)

    # 3) robust baseline
    # Robust baseline over the *valid* blocks only. Pixels outside the
    # footprint -- a mosaic edge, a chip gap, a trial patch clipped at the
    # boundary -- are zero-filled, and folding those zeros into the median and
    # the MAD drags sigma0 down. That is not a harmless bias: the detection
    # threshold below scales with sigma0, so an underestimate flags the whole
    # field as source, leaves no background pixels, and the calibration then
    # measures the scatter of a background fit that has interpolated the data.
    if np.any(pos):
        med0 = np.median(det[pos]).astype(np.float32)
        nmad0 = np.median(np.abs(det[pos] - med0) * np.float32(1.4826)).astype(np.float32)
        sigma0 = nmad0 if nmad0 > 0 else np.std(det[pos]).astype(np.float32)
    else:
        med0 = np.float32(0.0)
        sigma0 = np.float32(0.0)

    # 4) source mask (smoothed bright pass + unsmoothed faint pass, dilated)
    bgmask = ~coarse_source_mask(
        det - med0,
        sigma0,
        detect_thresh=detect_thresh,
        faint_thresh=faint_thresh,
        dilate=dilate,
        min_npixels_bright=min_npixels_bright,
        min_npixels_faint=min_npixels_faint,
    )
    bgmask &= pos  # exclude zero-weight tiles

    # A background fit needs background. When the source mask has eaten the
    # field there is nothing to smooth over, the fit tracks the data pixel for
    # pixel, and the residual scatter -- hence sigma_true -- collapses towards
    # zero, which would scale the inverse variance up without bound. Refuse
    # rather than return a calibration measured on nothing.
    n_pos = int(pos.sum())
    bg_frac = float(bgmask.sum()) / max(n_pos, 1)
    degenerate = n_pos == 0 or bg_frac < MIN_BG_FRACTION

    # 5) fit smooth background on the COARSE SCI (not det)
    #    convert sigma to coarse pixels
    bg_sigma_bin = max(float(step), 8.0)
    bg_img_bin = bg_gaussian_normalized(s_bin, bgmask, sigma=bg_sigma_bin, truncate=3.0)

    # 6) subtract bg and re-measure σ on bg pixels
    s_bin_bsub = s_bin - bg_img_bin
    det_bsub = np.zeros_like(det, dtype=np.float32)
    det_bsub[pos] = s_bin_bsub[pos] * np.sqrt(w_bin[pos], dtype=np.float32)

    # the step**2 block-averaging factor below assumes a full block, so
    # measure sigma on blocks that are (almost) entirely valid
    bg_ok = bgmask & np.isfinite(det_bsub) & (vfrac > 0.9)
    if not np.any(bg_ok):
        bg_ok = bgmask & np.isfinite(det_bsub)
    if not np.any(bg_ok):
        # fallback: use all valid pixels
        bg_ok = pos
    sigma_bin = mad_std(det_bsub[bg_ok].astype(np.float32)) if np.any(bg_ok) else np.nan
    # ``det_bsub`` is the coarse residual in units of the weight map's own claimed
    # sigma. Block-averaging step^2 independent pixels divides its scatter by
    # ``step``, so ``sigma_true`` is 1 exactly when ``wht`` IS a calibrated
    # inverse variance and the noise is uncorrelated. Otherwise it is the factor
    # by which the real noise exceeds what the weight map claims, and it absorbs
    # BOTH an arbitrary weight normalisation (drizzle weights, exposure time, ...)
    # and the drizzle pixel-to-pixel correlation, because it is measured after
    # resampling on a step x step block scale.
    sigma_true = np.float32(step) * np.float32(sigma_bin)
    if degenerate or not np.isfinite(sigma_true) or sigma_true <= 0:
        # no usable background sample (blank input, or the source mask covered
        # the field): leave the weights as they are rather than scaling them
        # by a number measured on a background fit that had nothing to fit
        logger.warning(
            "weight calibration%s: unusable background sample "
            "(%.1f%% of %d valid blocks, sigma_true=%r); leaving the weight "
            "map unscaled",
            f" [{label}]" if label else "", 100.0 * bg_frac, n_pos,
            float(sigma_true),
        )
        sigma_true = np.float32(1.0)

    # 7) rescale full-res weights.  ``valid``, not ``valid_w``: the weights used
    # above were masked by both, and a pixel whose science value is non-finite
    # carries no information regardless of what its weight claims.
    scale = np.float32(1.0) / (sigma_true * sigma_true + np.float32(1e-30))
    ivar_new = np.multiply(w, scale, dtype=np.float32)
    np.copyto(ivar_new, np.float32(0.0), where=~valid)
    # sigma_true = 1 exactly when the weight map is a calibrated inverse
    # variance; treat a 20% band as "consistent" (drizzle correlation and
    # normalisation conventions both land inside it when the map is honest)
    if 0.8 <= float(sigma_true) <= 1.2:
        verdict = "consistent with the weight image"
    else:
        verdict = (
            "INCONSISTENT with the weight image (its claimed noise is off "
            f"by x{float(sigma_true):.4g})"
        )
    # Median weight for the log line only. Boolean-indexing the full mask
    # copies the mosaic, and np.median partitions a second copy of that, so
    # sample every 8th pixel in each direction instead: 1/64 of the memory for
    # a median quoted to four figures in a message. The median ivar follows
    # from it exactly, since ivar is w * scale wherever the pixel is valid.
    sub = (slice(None, None, 8), slice(None, None, 8))
    vsub = valid[sub]
    med_w = float(np.median(w[sub][vsub])) if np.any(vsub) else np.nan
    logger.info(
        "weight calibration%s: correction factor to wht = %.4g (ivar x %.4g), "
        "sigma_true=%.4g -> %s; measured on %dx%d blocks; "
        "median wht %.4g -> ivar %.4g; median background %.4g",
        f" [{label}]" if label else "",
        float(scale), float(scale), float(sigma_true), verdict, step, step,
        med_w, med_w * float(scale),
        float(np.median(bg_img_bin[bgmask])) if np.any(bgmask) else np.nan,
    )

    if not need_bg:
        # the caller only wants the calibrated weights; the full-resolution
        # background is another array the size of the mosaic
        return None, ivar_new

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
from astropy.stats import mad_std


def _mean_downsample(arr, fact):
    """Fast block-reduce by *fact*×*fact* using strides, no Python loops."""
    ny, nx = arr.shape
    ny2, nx2 = ny // fact, nx // fact
    trimmed = arr[: ny2 * fact, : nx2 * fact]  # drop edge pixels
    view = trimmed.reshape(ny2, fact, nx2, fact)
    return view.mean(axis=(1, 3), dtype=arr.dtype)


def _valid_block_means(
    s: np.ndarray,
    w: np.ndarray,
    valid: np.ndarray,
    step: int,
    band_blocks: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block means of ``valid`` and of ``s`` and ``w`` zeroed outside it.

    Identical to :func:`_mean_downsample` applied to the three masked
    full-resolution arrays, but computed one band of ``band_blocks`` coarse rows
    at a time. The whole-array form needs three full-resolution temporaries --
    masked copies of the science and weight images plus a float32 cast of the
    mask -- and ``np.where(mask, float32_array, 0.0)`` promotes to float64 on
    the way, so on a 876 Mpx mosaic it costs about 21 GB to produce three coarse
    arrays of 200 kB each.

    Args:
        s: Science image, float32.
        w: Weight image, float32.
        valid: Boolean mask, same shape.
        step: Block size; trailing rows/columns outside a whole block are
            dropped exactly as :func:`_mean_downsample` drops them.
        band_blocks: Coarse rows reduced per pass; sets the temporary size.

    Returns:
        ``(vfrac, s_masked_mean, w_masked_mean)`` on the coarse grid, float32.
    """
    ny2, nx2 = s.shape[0] // step, s.shape[1] // step
    vfrac = np.empty((ny2, nx2), dtype=np.float32)
    s_bin = np.empty((ny2, nx2), dtype=np.float32)
    w_bin = np.empty((ny2, nx2), dtype=np.float32)
    zero = np.float32(0.0)  # a float32 scalar, so np.where stays in float32
    cols = slice(0, nx2 * step)
    for b0 in range(0, ny2, band_blocks):
        b1 = min(b0 + band_blocks, ny2)
        rows = slice(b0 * step, b1 * step)
        chunk = valid[rows, cols]
        vfrac[b0:b1] = _mean_downsample(chunk.astype(np.float32), step)
        # np.where, not multiplication by the mask: 0 * NaN is NaN, which is
        # precisely the poisoning the mask exists to prevent
        s_bin[b0:b1] = _mean_downsample(np.where(chunk, s[rows, cols], zero), step)
        w_bin[b0:b1] = _mean_downsample(np.where(chunk, w[rows, cols], zero), step)
    return vfrac, s_bin, w_bin


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


def _deblend_label_info(
    final_segmap: np.ndarray | SegmentationImage,
    parent_segmap: np.ndarray | SegmentationImage | None,
) -> dict[int, tuple[int, int, bool]]:
    """Map final segmentation labels to their pre-deblend parent labels.

    A final label is marked deblended when its pre-deblend parent label split
    into more than one final child label. This records the catalog/deblender
    provenance, not whether multiple truth sources overlap a label.
    """
    final_data = np.asarray(final_segmap.data if isinstance(final_segmap, SegmentationImage) else final_segmap)
    labels = np.unique(final_data)
    labels = labels[labels > 0]
    if parent_segmap is None:
        return {int(label): (int(label), 1, False) for label in labels}

    parent_data = np.asarray(parent_segmap.data if isinstance(parent_segmap, SegmentationImage) else parent_segmap)
    parent_by_label: dict[int, int] = {}
    for label in labels:
        parents = parent_data[final_data == label]
        parents = parents[parents > 0]
        if parents.size == 0:
            parent_by_label[int(label)] = int(label)
            continue
        vals, counts = np.unique(parents, return_counts=True)
        parent_by_label[int(label)] = int(vals[np.argmax(counts)])

    child_counts: dict[int, int] = {}
    for parent in parent_by_label.values():
        child_counts[parent] = child_counts.get(parent, 0) + 1

    return {
        label: (parent, child_counts[parent], child_counts[parent] > 1)
        for label, parent in parent_by_label.items()
    }




@dataclass
class Catalog:
    """Create a catalog from a science image and weight map.

    :meth:`run` (called automatically by :meth:`from_fits`) turns ``sci``
    and ``wht`` into a segmentation map plus a measurement table whose
    integer segment labels equal ``table['id']``.

    Attributes
    ----------
    sci
        Science image (2D). With ``estimate_background=True``, :meth:`run`
        rebinds this attribute to the background-subtracted image; the
        caller's array is not modified.
    wht
        Weight map, interpreted as inverse variance unless
        ``estimate_ivar=True``.
    nbin
        Binning factor used by the :meth:`plot_bg` diagnostic display.
    estimate_background
        Fit and subtract a smooth background via :func:`get_bg_and_ivar`
        before detection.
    estimate_ivar
        Recalibrate the inverse variance from the measured background-pixel
        scatter (also via :func:`get_bg_and_ivar`) instead of trusting
        ``wht``.
    background
        Background level or map; filled by :meth:`run` when
        ``estimate_background=True``.
    ivar
        Inverse-variance map. :meth:`run` sets it to ``wht`` when no
        estimation is requested, or to the recalibrated map when
        ``estimate_ivar=True``. ``estimate_background=True`` on its own
        leaves ``ivar`` unset and :meth:`run` fails; enable
        ``estimate_ivar`` as well, or pass ``ivar`` yourself.
    segmap
        Segmentation map. If provided, detection is skipped; otherwise
        filled by :meth:`run`.
    parent_segmap
        Copy of the segmentation map before deblending, used to record
        deblend provenance.
    catalog
        Underlying :class:`photutils.segmentation.SourceCatalog`, filled by
        :meth:`run`.
    table
        Measurement table, filled by :meth:`run`.
    det_img
        Noise-equalised detection image, filled during detection.
    params
        Detection and deblending parameters; user-supplied entries are
        merged over these defaults:

        - ``kernel_size`` (3.5): FWHM in pixels of the Gaussian smoothing
          kernel applied to the detection image (kernel sigma is
          ``kernel_size / 2.355``, stamp ``int(2 * kernel_size) | 1`` pixels
          on a side).
        - ``detect_npixels`` (5): minimum connected pixels for a detection;
          also passed to the deblender.
        - ``detect_threshold`` (2.0): threshold applied to the smoothed,
          noise-equalised detection image.
        - ``dilate_segmap`` (2): disk radius for
          :func:`safe_dilate_segmentation`; 0 disables dilation.
        - ``deblend_mode`` (``"exponential"``): mode passed to
          ``photutils.segmentation.deblend_sources``; ``None`` skips
          deblending entirely.
        - ``deblend_nlevels`` (32): multi-thresholding levels for the
          deblender.
        - ``deblend_contrast`` (1e-4): minimum flux fraction of a deblended
          child.
        - ``deblend_compactness`` (0.0): reserved; currently not forwarded
          to the deblender.
        - ``background_filter_sigma`` (64.0): forwarded to
          :func:`get_bg_and_ivar` as ``bg_filter_sigma``.
    header
        FITS header used to construct a WCS when one is not given; filled
        by :meth:`from_fits` when ``sci`` is a filename.
    wcs
        World coordinate system for sky positions; built from ``header``
        if absent.
    default_columns
        photutils ``SourceCatalog`` columns exported to ``table``.
    """

    sci: np.ndarray
    wht: np.ndarray
    nbin: int = 4
    estimate_background: bool = False
    estimate_ivar: bool = False

    background: float = 0.0
    ivar: np.ndarray | None = None
    segmap: SegmentationImage | None = None
    parent_segmap: SegmentationImage | None = None
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
        """Build a catalog from FITS files or arrays and run it.

        Parameters
        ----------
        sci
            Science image or FITS filename. When a filename, the header is
            read and stored for WCS construction.
        wht
            Weight map or FITS filename.
        segmap
            External segmentation map (filename, array, or
            ``SegmentationImage``). When given, detection and deblending are
            skipped and sources are measured within the provided segments.
        header
            Header to use when ``sci`` is an array.
        kwargs
            Forwarded to the :class:`Catalog` constructor (e.g.
            ``estimate_background``, ``params``).

        Returns
        -------
        Catalog
            Instance with ``table`` and ``segmap`` populated
            (:meth:`run` has been called).
        """
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
                segmap_obj = SegmentationImage(as_label_array(fits.getdata(segmap)))
            elif isinstance(segmap, np.ndarray):
                segmap_obj = SegmentationImage(as_label_array(segmap))
            elif isinstance(segmap, SegmentationImage):
                segmap_obj = segmap
            else:
                raise ValueError("segmap must be a filename, ndarray, or SegmentationImage")

        obj = cls(sci_data, wht_data, segmap=segmap_obj, header=header, **kwargs)
        obj.run()
        return obj

    def _detect(self) -> None:
        # run() already subtracted the estimated background from self.sci;
        # subtracting it here again would remove it twice. Only a
        # user-supplied background level still needs to come off.
        bg = 0.0 if self.estimate_background else self.background
        self.det_img = (self.sci - bg) * np.sqrt(self.ivar)
        kernel_pix = int(2 * self.params["kernel_size"]) | 1  # ensure odd size
        kernel = Gaussian2DKernel(
            self.params["kernel_size"] / 2.355, x_size=kernel_pix, y_size=kernel_pix
        )
        print(f"Convolving with kernel size {self.params['kernel_size']} pixels")
        # Gaussian2DKernel.array is always float64, and scipy promotes to the
        # wider operand, so an unconverted kernel drags a float32 detection
        # mosaic through the whole convolution at double width: 25.4 GB peak
        # on the 876 Mpx MINERVA grid against 14.5 GB matched. The FFT sums
        # inside pocketfft, so there is no accumulator to annotate; the error
        # is ~log2(N) eps32 = 2e-6 relative, on a map thresholded at 2 sigma.
        smooth = fftconvolve(
            self.det_img, kernel.array.astype(self.det_img.dtype, copy=False), mode="same"
        )
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
        self.parent_segmap = SegmentationImage(np.array(segmap.data, copy=True))
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
        """Detect, deblend, and measure sources; fills ``table`` and ``segmap``.

        Steps, in order: optional background / inverse-variance estimation;
        detection, dilation, and deblending (only if ``segmap`` is not
        already set); WCS construction from ``header``; measurement with
        ``SourceCatalog(sci, segmap, error=np.sqrt(1.0 / ivar), wcs=wcs)``;
        and assembly of ``table``. Beyond ``default_columns`` the table
        gains ``id``/``x``/``y`` (renamed from
        ``label``/``xcentroid``/``ycentroid``), ``ra``/``dec`` in degrees
        when a WCS is available (the ``sky_centroid`` column is removed),
        ``r50`` (``fluxfrac_radius(0.5)``), ``sharpness``
        (``max_value * pi * r50**2 / segment_flux``, near unity for point
        sources), ``snr`` (``segment_flux / segment_fluxerr``), and the
        deblend provenance columns ``deblend_parent_label``,
        ``deblend_nchildren``, and ``is_deblended``.
        """
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
        deblend_info = _deblend_label_info(self.segmap, self.parent_segmap)
        self.table["deblend_parent_label"] = np.array(
            [deblend_info.get(int(label), (int(label), 1, False))[0] for label in self.table["id"]],
            dtype=np.int32,
        )
        self.table["deblend_nchildren"] = np.array(
            [deblend_info.get(int(label), (int(label), 1, False))[1] for label in self.table["id"]],
            dtype=np.int32,
        )
        self.table["is_deblended"] = np.array(
            [deblend_info.get(int(label), (int(label), 1, False))[2] for label in self.table["id"]],
            dtype=bool,
        )
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
        """Select point-like sources and optionally fit a PSF stamp to each.

        Candidates must pass ``r50 < r50_max``, ``eccentricity < eccen_max``,
        and ``sharp_lohi[0] < sharpness < sharp_lohi[1]``; a boolean
        ``point_like`` column recording that cut is added to ``self.table``
        as a side effect. The returned table keeps only candidates with
        ``snr > snr_min``. When ``psf`` is given, each kept source is fit
        with :func:`fit_psf_stamp` and the returned table gains ``flux_psf``
        and ``chi2_red`` columns; ``chi2_max`` is not applied inside the
        method, so filter on ``chi2_red`` yourself. ``return_seg`` currently
        has no effect.

        Returns
        -------
        table, idx_stars
            Table of the selected sources and their row indices in
            ``self.table``.
        """

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


# --------------------------------------------------------------------------
# Saturation ↔ segmentation interop
# --------------------------------------------------------------------------
# These helpers consume the per-source table produced by
# :func:`mophongo.saturate.repair_saturated_holes` (columns ``xc, yc,
# r_equiv``, optionally ``r_out``) and act on a segmentation map. They
# live here, NOT in ``saturate.py``: the saturation algorithm does not
# need to know about catalogs, and segmap manipulation is the catalog
# module's responsibility.


def _bool_column(col) -> np.ndarray:
    """Boolean array from a table column, handling CSV round-trips.

    A bool column written to CSV reads back as the strings
    ``"True"/"False"``; a plain ``astype(bool)`` would turn every
    non-empty string into True.
    """
    arr = np.asarray(col)
    if arr.dtype.kind in "US":
        return np.char.lower(arr.astype(str)) == "true"
    return arr.astype(bool)


def merge_segments_at_holes(
    seg: np.ndarray,
    holes: "Table",
    *,
    dilate_factor: float = 8.0,
    dilate_min: float = 5.0,
) -> dict[int, list[int]]:
    """For each repaired hole, list segmentation labels within a radius.

    Parameters
    ----------
    seg
        2D integer segmentation map.
    holes
        Astropy ``Table`` with columns ``id, yc, xc, r_equiv``. Typically
        the ``fits`` table returned by
        :func:`mophongo.saturate.repair_saturated_holes` (or the holes
        catalog from :func:`mophongo.saturate.find_wht_holes`).
    dilate_factor, dilate_min
        Search radius around each hole is
        ``max(r_equiv * dilate_factor, dilate_min)`` pixels. The factor
        captures spike halos: brighter (larger-hole) stars have wider
        diffraction spikes that fragment the segmap.

    Returns
    -------
    dict
        ``{hole_id: sorted list of segmentation label ids inside r_merge}``.

    Notes
    -----
    Pure read-only inspection — does not modify ``seg``. Callers can use
    the returned mapping to relabel a segmap (e.g. by collapsing all
    matched labels of one hole into a single ID), but that policy choice
    is left to the caller.
    """
    H, W = seg.shape
    out: dict[int, list[int]] = {}
    for row in holes:
        yc = float(row["yc"])
        xc = float(row["xc"])
        r_eq = float(row["r_equiv"])
        r_merge = max(r_eq * dilate_factor, float(dilate_min))
        y0 = max(0, int(np.floor(yc - r_merge)))
        y1 = min(H, int(np.ceil(yc + r_merge)) + 1)
        x0 = max(0, int(np.floor(xc - r_merge)))
        x1 = min(W, int(np.ceil(xc + r_merge)) + 1)
        sub = seg[y0:y1, x0:x1]
        if sub.size == 0 or not np.any(sub > 0):
            out[int(row["id"])] = []
            continue
        ys_local, xs_local = np.where(sub > 0)
        ys = ys_local + y0
        xs = xs_local + x0
        d2 = (ys - yc) ** 2 + (xs - xc) ** 2
        within = d2 <= r_merge * r_merge
        if not within.any():
            out[int(row["id"])] = []
            continue
        ids = np.unique(sub[ys_local[within], xs_local[within]])
        out[int(row["id"])] = sorted(int(v) for v in ids)
    return out


def repair_saturated_catalog(
    catalog: Table,
    segmap: np.ndarray,
    fit_table: Table,
    *,
    fwhm_pix: float,
    filter_name: str,
    n_fwhm: float = 5.0,
    id_col: str = "id",
    x_col: str = "x",
    y_col: str = "y",
    pad: int | None = None,
    sci: np.ndarray | None = None,
    psf_stamp: np.ndarray | None = None,
    amplitude_col: str = "amplitude",
    flux_frac_thresh: float = 0.5,
) -> Tuple[Table, np.ndarray, Table]:
    """Merge oversplit saturated-star segments + catalog rows.

    For each successfully fit star in ``fit_table`` (rows with
    ``ok=True``):

    1. Collect all segmentation labels intersecting a circle of radius
       ``n_fwhm * fwhm_pix`` pixels around ``(xc, yc)``.
    2. Allocate a new parent label
       (``max(segmap) + 1``, incrementing).
    3. Close interior holes in the union of those child labels via
       ``scipy.ndimage.binary_fill_holes`` (the saturated core, which is
       background in the input segmap, becomes part of the parent).
       Newly-filled pixels are restricted to the merge circle.
    4. Drop the child rows from ``catalog`` and add a single parent row
       inherited from the brightest child (largest segmap area) with
       updated ``id``, ``x``, ``y`` from the PSF fit, and
       ``FLAG_SATURATED_<FILTER>`` set to 1.
    5. Bulk-remap all child labels to the parent label via a LUT pass.

    The weight map (segmap input here) is **not** grown to a PSF
    isophote — the saturated hole in ``wht`` is left untouched. Only
    the segmap is closed so the merged segment has no interior gap.

    Parameters
    ----------
    catalog
        Source catalog. Must contain ``id_col`` matching segmap labels.
    segmap
        2D integer segmentation map. Modified copy returned, not in
        place.
    fit_table
        Table from
        :func:`mophongo.saturate.repair_saturated_holes`. Required
        columns: ``xc, yc``. Optional: ``ok`` (only rows with
        ``ok=True`` are processed).
    fwhm_pix
        PSF FWHM in segmap pixels.
    filter_name
        Filter (e.g. ``"F444W"``). Used to name the new flag column
        ``FLAG_SATURATED_<filter_name.upper()>``.
    n_fwhm
        Merge radius in units of FWHM. Default 5.
    id_col, x_col, y_col
        Catalog column names.
    pad
        Extra padding (pixels) around each merge bbox when computing the
        local member mask. Default ``max(8, ceil(r_merge * 0.5))``.

    sci, psf_stamp, amplitude_col, flux_frac_thresh
        Optional PSF-flux filter on candidate child segments. When
        ``sci`` and ``psf_stamp`` are both provided, each segment label
        inside the merge circle is kept only if the PSF model flux
        within the segment is ``>= flux_frac_thresh`` of the science
        flux in those same pixels. Independent neighbour sources whose
        own flux dominates over the saturated star's PSF wings are
        therefore preserved instead of being merged. ``psf_stamp``
        must be sum-normalised and at the same pixel scale as
        ``segmap``; ``fit_table[amplitude_col]`` provides the per-star
        scaling.

    Notes
    -----
    Parent IDs are allocated as the next free integer
    ``max(segmap.max(), catalog[id_col].max()) + 1``, incrementing per
    star. The ``FLAG_SATURATED_<FILTER>`` column is the canonical signal
    that a row is a repaired parent.

    Returns
    -------
    new_catalog : Table
        Catalog with child rows removed, parent rows appended, and the
        ``FLAG_SATURATED_<FILTER>`` column added/updated.
    new_segmap : np.ndarray
        Repaired segmentation map.
    merge_log : Table
        One row per merged star with columns
        ``parent_id, fit_id, xc, yc, n_children, children`` (the
        latter a comma-separated string of child label IDs).
    """
    if fwhm_pix <= 0:
        raise ValueError("fwhm_pix must be positive")
    if id_col not in catalog.colnames:
        raise ValueError(f"catalog missing id column {id_col!r}")
    for col in (x_col, y_col):
        if col not in catalog.colnames:
            raise ValueError(f"catalog missing column {col!r}")
    for col in ("xc", "yc"):
        if col not in fit_table.colnames:
            raise ValueError(f"fit_table missing column {col!r}")

    catalog = catalog.copy()
    segmap = np.array(segmap, copy=True)
    H, W = segmap.shape

    flag_col = f"FLAG_SATURATED_{filter_name.upper()}"
    if flag_col not in catalog.colnames:
        catalog[flag_col] = np.zeros(len(catalog), dtype=np.int8)
    else:
        catalog[flag_col] = np.asarray(catalog[flag_col]).astype(np.int8)

    if "ok" in fit_table.colnames:
        ok_mask = _bool_column(fit_table["ok"])
    else:
        ok_mask = np.ones(len(fit_table), dtype=bool)

    r_merge = float(n_fwhm) * float(fwhm_pix)
    r2 = r_merge * r_merge
    if pad is None:
        pad = max(8, int(np.ceil(r_merge * 0.5)))

    id_to_index: dict[int, int] = {
        int(rid): i for i, rid in enumerate(catalog[id_col])
    }

    remap: dict[int, int] = {}
    drop_ids: set[int] = set()
    parents_tbl = catalog[:0].copy()
    log_rows: list[tuple] = []
    cat_id_max = int(np.asarray(catalog[id_col]).max()) if len(catalog) else 0
    next_id = max(int(segmap.max()), cat_id_max) + 1

    for row in fit_table[ok_mask]:
        xc = float(row["xc"])
        yc = float(row["yc"])

        y0 = max(0, int(np.floor(yc - r_merge)))
        y1 = min(H, int(np.ceil(yc + r_merge)) + 1)
        x0 = max(0, int(np.floor(xc - r_merge)))
        x1 = min(W, int(np.ceil(xc + r_merge)) + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        circle = (yy - yc) ** 2 + (xx - xc) ** 2 <= r2
        sub = segmap[y0:y1, x0:x1]
        labels_in = np.unique(sub[circle])
        labels_in = labels_in[labels_in > 0]
        if labels_in.size == 0:
            continue

        # Local bbox enlarged with pad for closure.
        yy0 = max(0, y0 - pad)
        yy1 = min(H, y1 + pad)
        xx0 = max(0, x0 - pad)
        xx1 = min(W, x1 + pad)
        local_seg = segmap[yy0:yy1, xx0:xx1]

        # PSF-flux filter: keep only segments whose PSF-model flux is
        # ``>= flux_frac_thresh * sci flux``. Independent neighbour
        # sources whose own flux dominates the saturated star's wings
        # are preserved.
        if sci is not None and psf_stamp is not None:
            amp = float(row[amplitude_col]) if amplitude_col in row.colnames else 1.0
            ph, pw = psf_stamp.shape
            psf_cy = (ph - 1) // 2
            psf_cx = (pw - 1) // 2
            yi = int(round(yc))
            xi = int(round(xc))
            # Destination window in global coords
            dy0 = yi - psf_cy; dy1 = dy0 + ph
            dx0 = xi - psf_cx; dx1 = dx0 + pw
            # Clip vs local bbox
            ay0 = max(yy0, dy0); ay1 = min(yy1, dy1)
            ax0 = max(xx0, dx0); ax1 = min(xx1, dx1)
            psf_model = np.zeros_like(local_seg, dtype=float)
            if ay1 > ay0 and ax1 > ax0:
                psf_model[
                    ay0 - yy0: ay1 - yy0, ax0 - xx0: ax1 - xx0
                ] = amp * psf_stamp[
                    ay0 - dy0: ay1 - dy0, ax0 - dx0: ax1 - dx0
                ]
            sci_local = sci[yy0:yy1, xx0:xx1]
            kept: list[int] = []
            for lbl in labels_in.tolist():
                m = (local_seg == int(lbl))
                if not m.any():
                    continue
                data_flux = float(sci_local[m].sum())
                psf_flux = float(psf_model[m].sum())
                if data_flux <= 0:
                    continue
                if psf_flux / data_flux >= flux_frac_thresh:
                    kept.append(int(lbl))
            if not kept:
                continue
            labels_in = np.array(kept, dtype=labels_in.dtype)

        local_member = np.isin(local_seg, labels_in)

        # Pick template row by largest child area (within local bbox).
        children_present = [
            int(lbl) for lbl in labels_in.tolist() if int(lbl) in id_to_index
        ]
        template_lbl: int | None = None
        if children_present:
            areas = {
                lbl: int(np.sum(local_seg == lbl)) for lbl in children_present
            }
            template_lbl = max(areas, key=areas.get)

        parent_id = next_id
        next_id += 1

        # Close ONLY the saturated core (originally background, seg==0).
        # Non-merging child segments that happen to sit inside the
        # binary-fill-holes hull must be preserved — they are unrelated
        # sources whose segments survive the repair untouched.
        local_filled = ndi.binary_fill_holes(local_member)
        new_pix = local_filled & ~local_member & (local_seg == 0)
        if new_pix.any():
            ys_n, xs_n = np.where(new_pix)
            yg = ys_n + yy0
            xg = xs_n + xx0
            inside = (yg - yc) ** 2 + (xg - xc) ** 2 <= r2
            yg = yg[inside]
            xg = xg[inside]
            if yg.size:
                segmap[yg, xg] = parent_id

        for lbl in labels_in.tolist():
            remap[int(lbl)] = parent_id

        # Add parent row to catalog. Inherit from brightest child (largest
        # segmap area). If no child has a catalog row, skip the catalog
        # update for this star; segmap is still merged.
        if template_lbl is not None:
            template_row = catalog[id_to_index[template_lbl]]
            parents_tbl.add_row(list(template_row))
            parents_tbl[id_col][-1] = parent_id
            parents_tbl[x_col][-1] = xc
            parents_tbl[y_col][-1] = yc
            parents_tbl[flag_col][-1] = 1
            drop_ids.update(
                int(lbl) for lbl in labels_in.tolist()
                if int(lbl) in id_to_index
            )
        log_rows.append((
            parent_id,
            int(row["id"]) if "id" in row.colnames else -1,
            xc, yc,
            int(labels_in.size),
            ",".join(str(int(lbl)) for lbl in labels_in.tolist()),
        ))

    # Bulk LUT remap of old child labels → parent ids.
    if remap:
        max_label = int(segmap.max())
        lut = np.arange(max_label + 1, dtype=segmap.dtype)
        for old, new in remap.items():
            if 0 <= old <= max_label:
                lut[old] = new
        segmap = lut[segmap]

    # Drop child rows, append parents.
    if drop_ids:
        ids_arr = np.asarray(catalog[id_col])
        keep = ~np.isin(ids_arr, np.array(sorted(drop_ids)))
        catalog = catalog[keep]
    if len(parents_tbl):
        from astropy.table import vstack as _vstack

        catalog = _vstack([catalog, parents_tbl], join_type="exact")

    merge_log = Table(
        rows=log_rows or None,
        names=["parent_id", "fit_id", "xc", "yc", "n_children", "children"],
        dtype=[int, int, float, float, int, str],
    )
    return catalog, segmap, merge_log


def flag_saturated_segments(
    catalog: Table,
    segmap: np.ndarray,
    fit_table: Table,
    *,
    sci: np.ndarray,
    psf_stamp: np.ndarray,
    filter_name: str = "TMPL",
    flux_frac: float = 0.3,
    min_snr: float = 5.0,
    halo_nsigma: float = 5.0,
    sky_noise: float | None = None,
    amplitude_col: str = "amplitude",
    id_col: str = "id",
    zero_segments: bool = True,
) -> Tuple[Table, np.ndarray, Table]:
    """Flag catalog segments dominated by a repaired star's PSF model.

    Post-hoc, non-destructive counterpart to
    :func:`repair_saturated_catalog` for catalogs built *before* the
    saturation repair: no rows are dropped or added and no segmentation
    labels are merged. For each successful fit in *fit_table* the star
    model ``A * psf_stamp`` is placed at the fitted centre, and every
    segmentation label whose model flux exceeds
    ``flux_frac x observed flux`` — both summed over the segment pixels
    inside the stamp's support (where the model is non-zero) — is deemed
    part of the saturated star. Its catalog row gets
    ``FLAG_SATURATED_<FILTER>`` set to the star's **group id** — the
    lowest flagged segment id of that star — so rows sharing a value
    belong to the same star (0 = not saturated; ``flag > 0`` is the
    boolean cut). Flagged labels are set to 0 in the returned
    segmentation map (``zero_segments``), and the undetected saturated
    core — ``seg == 0`` pixels enclosed by the star's flagged segments —
    is set to the group id, so the group-id row keeps a segment covering
    the repaired core. A segment claimed by two stars joins the one
    whose model flux in it is larger.

    Parameters
    ----------
    catalog
        Source catalog whose ``id_col`` matches segmap labels. A copy
        is returned with the flag column added/updated.
    segmap
        2D integer segmentation map. A (possibly masked) copy is
        returned.
    fit_table
        Table from :func:`mophongo.saturate.repair_saturated_holes`.
        Required columns: ``xc, yc, amplitude``; optional ``ok``
        (only ``ok=True`` rows are used), ``shift_x, shift_y``
        (added to ``xc, yc`` for the model centre), and ``r_in, r_out``
        (repair geometry, used to bound the core fill).
    sci
        Science image on the segmap pixel grid. Use the *repaired*
        image so the observed segment fluxes include the filled cores.
    psf_stamp
        Sum-normalised PSF stamp at the segmap pixel scale. Its size
        limits how far from the star segments can be flagged; segments
        entirely outside the stamp window are never flagged.
    filter_name
        Suffix for the flag column name. Default ``"TMPL"`` →
        ``FLAG_SATURATED_TMPL``: the repair runs on the template
        (high-resolution detection) band, which varies between surveys,
        so the column name stays band-independent.
    flux_frac
        Flag a segment when ``model_flux > flux_frac * observed_flux``.
    min_snr
        Noise floor: in addition to the flux-ratio test, the model flux
        in the segment must exceed ``min_snr x sky_noise x sqrt(n_pix)``.
        Without it, every faint segment inside the stamp support would
        be flagged whenever its noise-dominated observed flux sums to
        about zero.
    halo_nsigma
        Second, independent criterion: flag a segment when the model's
        own mean surface brightness over it exceeds
        ``halo_nsigma x sky_noise`` per pixel, whatever the flux ratio.
        The ratio test asks whether the star dominates the segment, and
        misses the bright diffraction spikes of a saturated star, whose
        real wings run far above the ePSF's (measured: spike segments at
        3-29 per cent of their observed flux, where flagged segments sit
        at 85-130 per cent). Those same segments still carry a model
        halo tens of sigma above the background, which is what this
        catches. Set to ``0`` to disable.
    sky_noise
        Per-pixel background rms for the noise floor. Default: robust
        ``mad_std`` of a subsample of *sci*.
    zero_segments
        Also zero the flagged labels in the returned segmap.

    Returns
    -------
    new_catalog : Table
        Catalog with the flag column set on flagged rows.
    new_segmap : np.ndarray
        Segmentation map with flagged labels zeroed (or an unchanged
        copy when ``zero_segments=False``).
    flag_log : Table
        One row per (star, segment) pair evaluated: ``fit_id, xc, yc,
        seg_id, npix, obs_flux, model_flux, frac, halo_sig, flagged``,
        where ``halo_sig`` is the model's mean surface brightness over
        the segment in units of ``sky_noise``.
    """
    if flux_frac <= 0:
        raise ValueError("flux_frac must be positive")
    if sky_noise is None:
        step = max(1, int(np.sqrt(sci.size / 4_000_000)))
        sample = np.asarray(sci[::step, ::step], dtype=np.float64)
        sample = sample[np.isfinite(sample) & (sample != 0)]
        sky_noise = float(mad_std(sample)) if sample.size else 0.0
        logger.info("[catalog] flag noise floor: sky_noise=%.4g", sky_noise)
    for col in ("xc", "yc", amplitude_col):
        if col not in fit_table.colnames:
            raise ValueError(f"fit_table missing column {col!r}")
    if id_col not in catalog.colnames:
        raise ValueError(f"catalog missing id column {id_col!r}")
    if sci.shape != segmap.shape:
        raise ValueError("sci and segmap shapes differ")

    catalog = catalog.copy()
    segmap = np.array(segmap, copy=True)
    H, W = segmap.shape
    sh, sw = psf_stamp.shape

    flag_col = f"FLAG_SATURATED_{filter_name.upper()}"
    if flag_col not in catalog.colnames:
        catalog[flag_col] = np.zeros(len(catalog), dtype=np.int64)
    else:
        catalog[flag_col] = np.asarray(catalog[flag_col]).astype(np.int64)

    if "ok" in fit_table.colnames:
        ok_mask = _bool_column(fit_table["ok"])
    else:
        ok_mask = np.ones(len(fit_table), dtype=bool)

    has_shift = ("shift_x" in fit_table.colnames
                 and "shift_y" in fit_table.colnames)

    # label -> (model_flux, fit_id) of the star claiming it most strongly
    claims: dict[int, tuple[float, int]] = {}
    star_windows: dict[int, tuple[int, int, int, int]] = {}
    star_centres: dict[int, tuple[float, float]] = {}
    log_rows: list[tuple] = []
    for row in fit_table[ok_mask]:
        amp = float(row[amplitude_col])
        if not np.isfinite(amp) or amp <= 0:
            continue
        xc = float(row["xc"])
        yc = float(row["yc"])
        if has_shift:
            xc += float(row["shift_x"])
            yc += float(row["shift_y"])
        fit_id = int(row["id"]) if "id" in fit_table.colnames else -1

        # Stamp window on the image grid, clipped to the bounds.
        y0 = int(round(yc)) - sh // 2
        x0 = int(round(xc)) - sw // 2
        iy0, iy1 = max(0, y0), min(H, y0 + sh)
        ix0, ix1 = max(0, x0), min(W, x0 + sw)
        if iy1 <= iy0 or ix1 <= ix0:
            continue
        star_windows[fit_id] = (iy0, iy1, ix0, ix1)
        star_centres[fit_id] = (yc, xc)
        model = amp * psf_stamp[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        seg_cut = segmap[iy0:iy1, ix0:ix1]
        sci_cut = np.asarray(sci[iy0:iy1, ix0:ix1], dtype=np.float64)

        # Compare fluxes only where the model is defined (stamp support).
        # Empirical ePSFs are zero beyond their native field of view, so
        # without this cut a large segment dilutes its ratio with pixels
        # the model cannot predict.
        support = model > 0
        for lbl in np.unique(seg_cut[(seg_cut > 0) & support]).tolist():
            m = (seg_cut == lbl) & support
            npix = int(m.sum())
            obs = float(np.nansum(sci_cut[m]))
            pred = float(np.sum(model[m]))
            frac = pred / obs if obs > 0 else np.inf
            floor = float(min_snr) * sky_noise * float(np.sqrt(npix))
            # the star dominates the segment ...
            dominates = pred > floor and (obs <= 0 or pred > flux_frac * obs)
            # ... or its halo alone is bright here, which is how the
            # diffraction spikes get caught (see halo_nsigma above)
            halo_sig = (pred / npix) / sky_noise if npix and sky_noise > 0 else 0.0
            flagged = dominates or (halo_nsigma > 0 and halo_sig > halo_nsigma)
            if flagged:
                prev = claims.get(int(lbl))
                if prev is None or pred > prev[0]:
                    claims[int(lbl)] = (pred, fit_id)
            log_rows.append(
                (fit_id, xc, yc, int(lbl), npix, obs, pred, frac, halo_sig, flagged)
            )

    # Group id per star: the flagged label that owns the star's core, i.e.
    # the one reaching closest to the fitted centre. That label's catalog row
    # sits on the star itself, so it survives the pipeline's footprint and
    # trial cuts and the filled core it labels is modelled by a template in
    # the right place. Naming the group by the lowest id instead can pick a
    # spike fragment tens of arcsec out, whose row may be cut for having no
    # coverage in the fitted band -- and then nothing models the core.
    star_labels: dict[int, list[int]] = {}
    for lbl, (_, star) in claims.items():
        star_labels.setdefault(star, []).append(lbl)

    def _core_label(star: int, labels: list[int]) -> int:
        win, centre = star_windows.get(star), star_centres.get(star)
        if win is None or centre is None:
            return min(labels)
        iy0, iy1, ix0, ix1 = win
        sub = segmap[iy0:iy1, ix0:ix1]
        yc, xc = centre
        yy, xx = np.indices(sub.shape)
        rr = np.hypot(yy + iy0 - yc, xx + ix0 - xc)
        reach = {}
        for lbl in labels:
            m = sub == lbl
            if m.any():
                reach[lbl] = float(rr[m].min())
        if not reach:
            return min(labels)
        best_r = min(reach.values())
        # Fragments that reach the centre equally closely (oversplit cores,
        # symmetric wedges) are all "the core"; take the lowest id among
        # them so the group id stays stable rather than turning on
        # sub-pixel differences.
        tol = 1.0 + 0.05 * best_r
        return min(lbl for lbl, r in reach.items() if r <= best_r + tol)

    label_group: dict[int, int] = {}
    for star, labels in star_labels.items():
        gid = _core_label(star, labels)
        for lbl in labels:
            label_group[lbl] = gid

    if label_group:
        flag_vals = np.asarray(catalog[flag_col])
        for i, idv in enumerate(np.asarray(catalog[id_col])):
            gid = label_group.get(int(idv))
            if gid is not None:
                flag_vals[i] = gid
        catalog[flag_col] = flag_vals
        seg_orig = segmap
        if zero_segments:
            max_label = int(segmap.max())
            lut = np.arange(max_label + 1, dtype=segmap.dtype)
            for lbl in label_group:
                if 0 <= lbl <= max_label:
                    lut[lbl] = 0
            segmap = lut[segmap]
        # Fill the saturated core so the group-id catalog row keeps a
        # segment covering the repaired pixels and mophongo models the
        # star as a normal source:
        #   * seg=0 pixels enclosed by the star's flagged segments
        #     (bounded by r_out to skip unrelated enclosed sky pockets);
        #   * everything within r_in of the fitted centre — the repair
        #     replaced those pixels with the PSF model, including ones
        #     that belonged to (now zeroed) flagged wing segments.
        star_geom: dict[int, tuple[float, float, float, float]] = {}
        if all(c in fit_table.colnames for c in ("r_in", "r_out")):
            for row in fit_table[ok_mask]:
                fid = int(row["id"]) if "id" in fit_table.colnames else -1
                xg = float(row["xc"]) + (float(row["shift_x"]) if has_shift else 0.0)
                yg = float(row["yc"]) + (float(row["shift_y"]) if has_shift else 0.0)
                star_geom[fid] = (yg, xg, float(row["r_in"]), float(row["r_out"]))
        for star, labels in star_labels.items():
            gid = label_group[labels[0]]
            win = star_windows.get(star)
            if win is None:
                continue
            iy0, iy1, ix0, ix1 = win
            sub_orig = seg_orig[iy0:iy1, ix0:ix1]
            sub_out = segmap[iy0:iy1, ix0:ix1]
            union = np.isin(sub_orig, np.array(labels))
            if not union.any():
                continue
            fill = ndi.binary_fill_holes(union) & ~union & (sub_orig == 0)
            geom = star_geom.get(star)
            if geom is not None:
                yg, xg, r_in, r_out = geom
                yy, xx = np.indices(sub_orig.shape)
                rr = np.hypot(yy + iy0 - yg, xx + ix0 - xg)
                fill = (fill & (rr <= r_out)) | (
                    (rr <= r_in) & ((sub_orig == 0) | union)
                )
            fill &= sub_out == 0
            if fill.any():
                sub_out[fill] = gid
    logger.info(
        "[catalog] flagged %d segments (%d catalog rows, %d stars) in %s",
        len(label_group),
        int(np.sum(np.asarray(catalog[flag_col]) > 0)),
        len(star_labels),
        flag_col,
    )

    flag_log = Table(
        rows=log_rows or None,
        names=["fit_id", "xc", "yc", "seg_id", "npix",
               "obs_flux", "model_flux", "frac", "halo_sig", "flagged"],
        dtype=[int, float, float, int, int, float, float, float, float, bool],
    )
    flag_log["group_id"] = [
        label_group.get(int(s), 0) if f else 0
        for s, f in zip(flag_log["seg_id"], flag_log["flagged"])
    ] if len(flag_log) else np.zeros(0, dtype=int)
    return catalog, segmap, flag_log
