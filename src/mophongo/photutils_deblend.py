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
from photutils.segmentation.deblend import _Deblender

__all__ = ["deblend_sources"]


class _CompactnessDeblender(_Deblender):
    """Deblender using watershed with compactness."""

    def __init__(
        self,
        source_data: np.ndarray,
        source_segment: SegmentationImage,
        npixels: int,
        footprint: np.ndarray,
        nlevels: int,
        contrast: float,
        mode: str,
        *,
        compactness: float = 0.0,
    ) -> None:
        super().__init__(source_data, source_segment, npixels, footprint, nlevels, contrast, mode)
        self.compactness = compactness

    def apply_watershed(self, markers: Iterable[SegmentationImage]) -> np.ndarray:
        from scipy.ndimage import sum_labels
        from skimage.segmentation import watershed

        markers = markers[-1].data
        remove_marker = True
        while remove_marker:
            markers = watershed(
                -self.source_data,
                markers,
                mask=self.segment_mask,
                connectivity=self.footprint,
                compactness=self.compactness,
            )
            labels = np.unique(markers[markers != 0])
            if labels.size == 1:
                remove_marker = False
            else:
                flux_frac = sum_labels(self.source_data, markers, index=labels) / self.source_sum
                remove_marker = any(flux_frac < self.contrast)
                if remove_marker:
                    markers[markers == labels[np.argmin(flux_frac)]] = 0.0
        return markers


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
    deblender = _CompactnessDeblender(
        source_data,
        source_segment,
        npixels,
        footprint,
        nlevels,
        contrast,
        mode,
        compactness=compactness,
    )
    return deblender.deblend_source()


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
    """Deblend sources using photutils with watershed compactness."""

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
            deblender = _CompactnessDeblender(
                source_data,
                source_segment,
                npixels,
                footprint,
                nlevels,
                contrast,
                mode,
                compactness=compactness,
            )
            source_deblended = deblender.deblend_source()
            all_source_deblends.append(source_deblended)
    else:
        nlabels = len(labels)
        args_all = zip(
            all_source_data,
            all_source_segments,
            (npixels,) * nlabels,
            (footprint,) * nlabels,
            (nlevels,) * nlabels,
            (contrast,) * nlabels,
            (mode,) * nlabels,
            (compactness,) * nlabels,
        )

        if progress_bar:
            desc = "Deblending"
            args_all = add_progress_bar(args_all, total=nlabels, desc=desc)

        with get_context("spawn").Pool(processes=nproc) as executor:
            all_source_deblends = executor.starmap(_deblend_source_compact, args_all)

    nonposmin_labels = []
    nmarkers_labels = []
    for label, source_deblended, source_slice in zip(labels, all_source_deblends, all_source_slices):
        if source_deblended is not None:
            segment_mask = source_deblended.data > 0
            segm_deblended._data[source_slice][segment_mask] = (
                source_deblended.data[segment_mask] + last_label
            )
            last_label += source_deblended.nlabels

            if hasattr(source_deblended, "warnings"):
                if source_deblended.warnings.get("nonposmin", None) is not None:
                    nonposmin_labels.append(label)
                if source_deblended.warnings.get("nmarkers", None) is not None:
                    nmarkers_labels.append(label)

    if nonposmin_labels or nmarkers_labels:
        segm_deblended.info = {"warnings": {}}
        warnings.warn(
            "The deblending mode of one or more source labels from "
            "the input segmentation image was changed from "
            f"{mode!r} to 'linear'. See the 'info' attribute "
            "for the list of affected input labels.",
            AstropyUserWarning,
        )

        if nonposmin_labels:
            warn = {
                "message": f"Deblending mode changed from {mode} to "
                "linear due to non-positive minimum data values.",
                "input_labels": np.array(nonposmin_labels),
            }
            segm_deblended.info["warnings"]["nonposmin"] = warn

        if nmarkers_labels:
            warn = {
                "message": f"Deblending mode changed from {mode} to "
                "linear due to too many potential deblended sources.",
                "input_labels": np.array(nmarkers_labels),
            }
            segm_deblended.info["warnings"]["nmarkers"] = warn

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended

