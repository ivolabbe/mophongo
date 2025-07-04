from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    pad_before = (pad_y, pad_x)
    pad_after = (ky - 1 - pad_y, kx - 1 - pad_x)
    padded = np.pad(image, (pad_before, pad_after), mode="constant")
    # Use sliding windows for convolution
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)


@dataclass
class Template:
    array: np.ndarray
    bbox: Tuple[int, int, int, int]


def extract_templates(
    hires_image: np.ndarray,
    segmap: np.ndarray,
    positions: Iterable[Tuple[float, float]],
    kernel: np.ndarray,
) -> List[Template]:
    """Extract PSF-matched templates for a list of source positions.

    Parameters
    ----------
    hires_image : np.ndarray
        High-resolution image array.
    segmap : np.ndarray
        Segmentation map with integer labels identifying sources.
    positions : iterable of tuple
        List of (y, x) positions corresponding to objects.
    kernel : np.ndarray
        Convolution kernel to match the high-resolution PSF to the low-resolution PSF.

    Returns
    -------
    templates : list of Template
        Each template contains the PSF-matched cutout and its bounding box as
        (y0, y1, x0, x1) in the high-resolution image coordinates.
    """
    if hires_image.shape != segmap.shape:
        raise ValueError("hires_image and segmap must have the same shape")

    templates: List[Template] = []
    kernel = kernel / kernel.sum()

    segm = SegmentationImage(segmap)

    for pos in positions:
        y, x = int(round(pos[0])), int(round(pos[1]))
        if y < 0 or y >= segm.data.shape[0] or x < 0 or x >= segm.data.shape[1]:
            continue
        label = segm.data[y, x]
        if label == 0:
            continue

        idx = segm.get_index(label)
        bbox = segm.bbox[idx]

        ky, kx = kernel.shape
        pad_y, pad_x = ky // 2, kx // 2

        y0_ext = bbox.iymin - pad_y
        y1_ext = bbox.iymax + pad_y
        x0_ext = bbox.ixmin - pad_x
        x1_ext = bbox.ixmax + pad_x

        height = y1_ext - y0_ext
        width = x1_ext - x0_ext
        center = ((x0_ext + x1_ext) / 2.0, (y0_ext + y1_ext) / 2.0)

        cut = Cutout2D(hires_image, center, (height, width), mode="partial", fill_value=0.0)
        mask_cut = Cutout2D((segm.data == label).astype(hires_image.dtype), center, (height, width), mode="partial", fill_value=0.0)
        cutout = cut.data * mask_cut.data

        flux = cutout.sum()
        if flux == 0:
            continue

        cutout_norm = cutout / flux
        conv = _convolve2d(cutout_norm, kernel)
        templates.append(Template(conv, (y0_ext, y1_ext, x0_ext, x1_ext)))

    return templates
