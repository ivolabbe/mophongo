"""Symmetry-based deblender utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls
from photutils.segmentation import SegmentationImage
from skimage.feature import peak_local_max
from skimage.measure import label

__all__ = ["deblend_sources_symmetry"]


def find_top_peaks(
    image: np.ndarray,
    segm: SegmentationImage,
    nmax: int,
    *,
    min_distance: int = 3,
    threshold_abs: float | None = None,
    threshold_rel: float = 0.0,
) -> Dict[int, List[Tuple[int, int]]]:
    """Find up to ``nmax`` brightest peaks per segmentation island."""
    peaks: Dict[int, List[Tuple[int, int]]] = {}
    for label_id in segm.labels:
        if label_id == 0:
            continue
        idx = segm.get_index(label_id)
        slc = segm.slices[idx]
        mask = segm.data[slc] == label_id
        sub = image[slc]
        masked = np.where(mask, sub, sub.min() - 1.0)
        loc = peak_local_max(
            masked,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            exclude_border=False,
        )
        if loc.size == 0:
            # fall back to brightest pixel
            idx = np.unravel_index(np.argmax(masked), masked.shape)
            loc = np.array([idx])
        values = sub[tuple(loc.T)]
        order = np.argsort(values)[::-1][:nmax]
        loc = loc[order]
        loc[:, 0] += slc[0].start
        loc[:, 1] += slc[1].start
        peaks[label_id] = [tuple(p) for p in loc]
    return peaks


def build_templates(
    image: np.ndarray,
    segm: SegmentationImage,
    peaks: Dict[int, List[Tuple[int, int]]],
    *,
    eps: float,
    radius: int | None = None,
) -> tuple[Dict[int, Tuple[np.ndarray, Tuple[int, int]]], Dict[int, int]]:
    """Create symmetry templates around each peak."""
    templates: Dict[int, Tuple[np.ndarray, Tuple[int, int]]] = {}
    mapping: Dict[int, int] = {}
    cid = 1
    for parent, pts in peaks.items():
        for py, px in pts:
            if radius is None:
                idx = segm.get_index(parent)
                slc = segm.slices[idx]
                r = max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start) // 2
            else:
                r = radius
            y0 = max(0, py - r)
            x0 = max(0, px - r)
            y1 = min(image.shape[0], py + r + 1)
            x1 = min(image.shape[1], px + r + 1)
            stamp = image[y0:y1, x0:x1]
            rot = stamp[::-1, ::-1]
            tmpl = np.minimum(stamp, rot)
            tmpl[tmpl < eps] = 0.0
            total = tmpl.sum()
            if total <= 0:
                continue
            tmpl /= total
            templates[cid] = (tmpl, (y0, x0))
            mapping[cid] = parent
            cid += 1
    return templates, mapping


def fit_templates(
    image: np.ndarray,
    segm: SegmentationImage,
    templates: Dict[int, Tuple[np.ndarray, Tuple[int, int]]],
    mapping: Dict[int, int],
) -> Dict[int, float]:
    """Fit template fluxes with non-negative least squares."""
    fluxes: Dict[int, float] = {}
    groups: Dict[int, List[int]] = defaultdict(list)
    for cid, parent in mapping.items():
        groups[parent].append(cid)
    for parent, cids in groups.items():
        mask = segm.data == parent
        if mask.sum() == 0:
            continue
        coords = np.nonzero(mask)
        b = image[mask].ravel()
        A = np.zeros((b.size, len(cids)))
        for j, cid in enumerate(cids):
            tmpl, (y0, x0) = templates[cid]
            h, w = tmpl.shape
            arr = np.zeros_like(segm.data, dtype=float)
            y1 = min(y0 + h, arr.shape[0])
            x1 = min(x0 + w, arr.shape[1])
            arr[y0:y1, x0:x1] = tmpl[: y1 - y0, : x1 - x0]
            A[:, j] = arr[mask]
        if A.size == 0:
            continue
        sol, _ = nnls(A, b)
        for j, cid in enumerate(cids):
            fluxes[cid] = sol[j]
    return fluxes


def assign_pixels(
    segm: SegmentationImage,
    templates: Dict[int, Tuple[np.ndarray, Tuple[int, int]]],
    fluxes: Dict[int, float],
    mapping: Dict[int, int],
) -> tuple[np.ndarray, Dict[int, int]]:
    """Assign pixels to children based on template models."""
    new_seg = segm.data.copy()
    next_label = new_seg.max() + 1
    child_labels: Dict[int, int] = {}
    groups: Dict[int, List[int]] = defaultdict(list)
    for cid, parent in mapping.items():
        groups[parent].append(cid)
    for parent, cids in groups.items():
        if len(cids) <= 1:
            continue
        mask = segm.data == parent
        stack = np.zeros((len(cids),) + segm.data.shape, dtype=float)
        for i, cid in enumerate(cids):
            tmpl, (y0, x0) = templates[cid]
            h, w = tmpl.shape
            y1 = min(y0 + h, stack.shape[1])
            x1 = min(x0 + w, stack.shape[2])
            stack[i, y0:y1, x0:x1] = tmpl[: y1 - y0, : x1 - x0] * fluxes.get(cid, 0.0)
        best = np.argmax(stack[:, mask], axis=0)
        for i, cid in enumerate(cids):
            lbl = next_label + i
            assign_mask = np.zeros_like(new_seg, dtype=bool)
            assign_mask[mask] = best == i
            new_seg[assign_mask] = lbl
            child_labels[lbl] = parent
        next_label += len(cids)
    return new_seg, child_labels


def clean_and_relabel(
    seg: np.ndarray, mapping: Dict[int, int], *, min_pix: int
) -> tuple[np.ndarray, Dict[int, int]]:
    """Remove tiny children and relabel sequentially."""
    for child, parent in list(mapping.items()):
        if np.sum(seg == child) < min_pix:
            seg[seg == child] = parent
            del mapping[child]
    seg = label(seg, background=0)
    return seg, mapping


def converged(old: np.ndarray, new: np.ndarray) -> bool:
    """Check convergence via exact label match."""
    return np.array_equal(old, new)


def deblend_sources_symmetry(
    image: np.ndarray,
    segmap: SegmentationImage | np.ndarray,
    *,
    nmax: int = 5,
    eps: float | None = None,
    min_pix: int = 5,
    max_iter: int = 10,
    peak_min_distance: int = 3,
    peak_threshold_abs: float | None = None,
    peak_threshold_rel: float = 0.0,
    radius: int | None = None,
) -> SegmentationImage:
    """Deblend a segmentation map using symmetry templates."""
    if not isinstance(segmap, SegmentationImage):
        segm = SegmentationImage(np.asarray(segmap))
    else:
        segm = segmap
    seg_arr = segm.data.copy()
    eps = eps or 1e-3 * np.percentile(image, 99.5)

    for _ in range(max_iter):
        segm = SegmentationImage(seg_arr)
        peaks = find_top_peaks(
            image,
            segm,
            nmax,
            min_distance=peak_min_distance,
            threshold_abs=peak_threshold_abs,
            threshold_rel=peak_threshold_rel,
        )
        if all(len(p) <= 1 for p in peaks.values()):
            break
        templates, mapping = build_templates(
            image, segm, peaks, eps=eps, radius=radius
        )
        fluxes = fit_templates(image, segm, templates, mapping)
        seg_new, mapping = assign_pixels(segm, templates, fluxes, mapping)
        seg_new, mapping = clean_and_relabel(seg_new, mapping, min_pix=min_pix)
        if converged(seg_arr, seg_new):
            seg_arr = seg_new
            break
        seg_arr = seg_new
    return SegmentationImage(seg_arr)
