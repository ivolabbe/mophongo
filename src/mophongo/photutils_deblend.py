"""Photutils deblend wrapper with compactness option."""

from __future__ import annotations

import warnings
from multiprocessing import cpu_count, get_context
from typing import Iterable

import numpy as np
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from photutils.segmentation import SegmentationImage
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._progress_bars import add_progress_bar

__all__ = ["deblend_sources"]


def deblend_sources(
    data: np.ndarray,
    segment_img: SegmentationImage,
    npixels: int,
    *,
    labels: int | Iterable[int] | None = None,
    nlevels: int = 32,
    contrast: float = 0.001,
    mode: str = "exponential",
    connectivity: int = 8,
    relabel: bool = True,
    nproc: int = 1,
    progress_bar: bool = True,
    compactness: float = 0.0,
) -> SegmentationImage:
    """Deblend sources using photutils with watershed compactness.
    
    This is a wrapper around photutils.segmentation.deblend_sources that
    adds support for the compactness parameter in watershed segmentation.
    
    Parameters
    ----------
    data : np.ndarray
        The 2D array of the image.
    segment_img : SegmentationImage
        The segmentation image to deblend.
    npixels : int
        The minimum number of connected pixels for an object.
    labels : int or array_like of int, optional
        The label numbers to deblend. If None, all labels are deblended.
    nlevels : int, optional
        Number of multi-thresholding levels for deblending.
    contrast : float, optional
        Fraction of total flux that a local peak must have to be deblended.
    mode : {'exponential', 'linear', 'sinh'}, optional
        Mode for spacing between multi-thresholding levels.
    connectivity : {8, 4}, optional
        Pixel connectivity for grouping pixels.
    relabel : bool, optional
        Whether to relabel consecutively starting from 1.
    nproc : int, optional
        Number of processes for multiprocessing.
    progress_bar : bool, optional
        Whether to display a progress bar.
    compactness : float, optional
        Compactness parameter for watershed (0 = no compactness).
        
    Returns
    -------
    SegmentationImage
        Deblended segmentation image.
    """
    
    # For compatibility, if compactness is 0, use standard photutils deblending
    if compactness == 0.0:
        from photutils.segmentation import deblend_sources as photutils_deblend
        return photutils_deblend(
            data, segment_img, npixels,
            labels=labels, nlevels=nlevels, contrast=contrast,
            mode=mode, connectivity=connectivity, relabel=relabel,
            nproc=nproc, progress_bar=progress_bar
        )
    
    # Custom implementation with compactness support
    if isinstance(data, Quantity):
        data = data.value

    if not isinstance(segment_img, SegmentationImage):
        raise ValueError("segment_img must be a SegmentationImage")

    if segment_img.shape != data.shape:
        raise ValueError("The data and segmentation image must have the same shape")

    if nlevels < 1:
        raise ValueError("nlevels must be >= 1")
    if contrast < 0 or contrast > 1:
        raise ValueError("contrast must be >= 0 and <= 1")

    if contrast == 1:
        return segment_img.copy()

    if mode not in ("exponential", "linear", "sinh"):
        raise ValueError('mode must be "exponential", "linear", or "sinh"')

    if labels is None:
        labels = segment_img.labels
    else:
        labels = np.atleast_1d(labels)
        segment_img.check_labels(labels)

    mask = segment_img.areas[segment_img.get_indices(labels)] >= (npixels * 2)
    labels = labels[mask]

    footprint = _make_binary_structure(data.ndim, connectivity)

    if nproc is None:
        nproc = cpu_count()

    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    last_label = segment_img.max_label
    indices = segment_img.get_indices(labels)

    all_source_data = []
    all_source_segments = []
    all_source_slices = []
    for label, idx in zip(labels, indices):
        source_slice = segment_img.slices[idx]
        source_data = data[source_slice]
        source_segment = object.__new__(SegmentationImage)
        source_segment._data = segment_img.data[source_slice]
        source_segment.keep_labels(label)
        all_source_data.append(source_data)
        all_source_segments.append(source_segment)
        all_source_slices.append(source_slice)

    if nproc == 1:
        if progress_bar:
            desc = "Deblending"
            all_source_data = add_progress_bar(all_source_data, desc=desc)

        all_source_deblends = []
        for source_data, source_segment in zip(all_source_data, all_source_segments):
            source_deblended = _deblend_source_compact(
                source_data,
                source_segment,
                npixels,
                footprint,
                nlevels,
                contrast,
                mode,
                compactness,
            )
            all_source_deblends.append(source_deblended)
    else:
        # Multiprocessing implementation would go here
        # For now, fall back to serial processing
        warnings.warn(
            "Multiprocessing with compactness not yet implemented, using serial processing",
            AstropyUserWarning
        )
        all_source_deblends = []
        for source_data, source_segment in zip(all_source_data, all_source_segments):
            source_deblended = _deblend_source_compact(
                source_data,
                source_segment,
                npixels,
                footprint,
                nlevels,
                contrast,
                mode,
                compactness,
            )
            all_source_deblends.append(source_deblended)

    for label, source_deblended, source_slice in zip(labels, all_source_deblends, all_source_slices):
        if source_deblended is not None:
            segment_mask = source_deblended.data > 0
            segm_deblended._data[source_slice][segment_mask] = (
                source_deblended.data[segment_mask] + last_label
            )
            last_label += source_deblended.nlabels

    return segm_deblended


def _deblend_source_compact(
    source_data: np.ndarray,
    source_segment: SegmentationImage,
    npixels: int,
    footprint: np.ndarray,
    nlevels: int,
    contrast: float,
    mode: str,
    compactness: float,
) -> SegmentationImage | None:
    """Deblend a single source using watershed with compactness."""
    
    # If compactness is 0, use standard photutils
    if compactness == 0.0:
        from photutils.segmentation import deblend_sources as photutils_deblend
        return photutils_deblend(
            source_data, source_segment, npixels,
            nlevels=nlevels, contrast=contrast, mode=mode,
            connectivity=8 if footprint.sum() == 8 else 4,
            relabel=True, nproc=1, progress_bar=False
        )
    
    # Custom watershed implementation with compactness
    try:
        from skimage.segmentation import watershed
        from scipy.ndimage import sum_labels
        from photutils.segmentation.detect import _detect_sources
        
        # Get source mask and properties
        label = source_segment.labels[0]
        segment_mask = source_segment.data == label
        
        if not np.any(segment_mask):
            return None
            
        data_values = source_data[segment_mask]
        source_min = np.nanmin(data_values)
        source_max = np.nanmax(data_values)
        source_sum = np.nansum(data_values)
        
        if source_min == source_max:
            return None
        
        # Compute thresholds
        if mode == 'exponential' and source_min <= 0:
            mode = 'linear'
            
        if mode == 'linear':
            thresholds = np.linspace(source_min, source_max, nlevels + 2)[1:-1]
        elif mode == 'sinh':
            a = 0.25
            norm_thresh = np.linspace(0, 1, nlevels + 2)[1:-1]
            thresholds = np.sinh(norm_thresh / a) / np.sinh(1.0 / a)
            thresholds = thresholds * (source_max - source_min) + source_min
        elif mode == 'exponential':
            norm_thresh = np.linspace(0, 1, nlevels + 2)[1:-1]
            thresholds = source_min * (source_max / source_min) ** norm_thresh
        
        # Find markers using multi-thresholding
        markers = None
        for threshold in thresholds:
            segm = _detect_sources(source_data, threshold, npixels,
                                 footprint, segment_mask,
                                 relabel=False, return_segmimg=False)
            if segm is not None and len(np.unique(segm[segm > 0])) > 1:
                markers = segm
                break
        
        if markers is None:
            return None
        
        # Apply watershed with compactness
        remove_marker = True
        while remove_marker:
            markers = watershed(
                -source_data,
                markers,
                mask=segment_mask,
                connectivity=footprint,
                compactness=compactness,
            )
            
            labels = np.unique(markers[markers != 0])
            if labels.size == 1:
                remove_marker = False
            else:
                flux_frac = sum_labels(source_data, markers, index=labels) / source_sum
                remove_marker = any(flux_frac < contrast)
                if remove_marker:
                    markers[markers == labels[np.argmin(flux_frac)]] = 0
        
        if len(np.unique(markers[markers > 0])) <= 1:
            return None
        
        # Create output segmentation
        result = object.__new__(SegmentationImage)
        result._data = markers.astype(np.int32)
        
        return result
        
    except ImportError:
        warnings.warn(
            "scikit-image not available, falling back to standard deblending",
            AstropyUserWarning
        )
        from photutils.segmentation import deblend_sources as photutils_deblend
        return photutils_deblend(
            source_data, source_segment, npixels,
            nlevels=nlevels, contrast=contrast, mode=mode,
            connectivity=8 if footprint.sum() == 8 else 4,
            relabel=True, nproc=1, progress_bar=False
        )

