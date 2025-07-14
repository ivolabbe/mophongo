"""Symmetry-based deblender utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls
from photutils.segmentation import SegmentationImage
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import rotate, shift

from .utils import measure_shape, elliptical_gaussian, elliptical_moffat

__all__ = ["deblend_sources_symmetry", "deblend_sources_hybrid"]


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


def _hybrid_models(templates: dict[int, tuple[np.ndarray, tuple[int, int]]]) -> dict[int, np.ndarray]:
    """Return analytic probability models for templates."""
    models: dict[int, np.ndarray] = {}
    for cid, (tmpl, _) in templates.items():
        mask = tmpl > 0
        if not np.any(mask):
            continue
        x_c, y_c, sigma_x, sigma_y, theta = measure_shape(tmpl, mask)
        ratio = sigma_x / max(sigma_y, 1e-6)
        y, x = np.indices(tmpl.shape)
        if ratio > 1.5 or ratio < (1 / 1.5):
            model = elliptical_gaussian(
                y,
                x,
                1.0,
                2.355 * sigma_x,
                2.355 * sigma_y,
                theta,
                x_c,
                y_c,
            )
        else:
            r = np.hypot(y - y_c, x - x_c)
            mask_r = r > 0
            if np.sum(mask_r) > 0:
                slope, _ = np.polyfit(np.log(r[mask_r]), np.log(tmpl[mask_r] + 1e-6), 1)
                beta = np.clip(-slope / 2.0, 1.0, 10.0)
            else:
                beta = 2.5
            model = elliptical_moffat(
                y,
                x,
                1.0,
                2.355 * sigma_x,
                2.355 * sigma_y,
                beta,
                theta,
                x_c,
                y_c,
            )
        if model.sum() > 0:
            model = model / model.sum()
        models[cid] = model
    return models


def deblend_sources_hybrid(
    image: np.ndarray,
    segmap: SegmentationImage | np.ndarray,
    *,
    nmax: int = 5,
    eps: float | None = None,
    min_pix: int = 5,
    peak_min_distance: int = 3,
    peak_threshold_abs: float | None = None,
    peak_threshold_rel: float = 0.0,
    radius: int | None = None,
) -> SegmentationImage:
    """Deblend a segmentation map using analytic symmetry templates."""
    if not isinstance(segmap, SegmentationImage):
        segm = SegmentationImage(np.asarray(segmap))
    else:
        segm = segmap
    seg_arr = segm.data.copy()
    eps = eps or max(1e-4, 1e-3 * np.percentile(image, 99.5))

    peaks = find_top_peaks(
        image,
        segm,
        nmax,
        min_distance=peak_min_distance,
        threshold_abs=peak_threshold_abs,
        threshold_rel=peak_threshold_rel,
    )
    templates, mapping = build_templates(image, segm, peaks, eps=eps, radius=radius)

    # Recompute templates with floor distribution
    for cid, (tmpl, (y0, x0)) in templates.items():
        cy = tmpl.shape[0] // 2
        cx = tmpl.shape[1] // 2
        peak = tmpl[cy, cx]
        floor = max(eps, peak * 1e-4)
        y, x = np.indices(tmpl.shape)
        dist = np.hypot(y - cy, x - cx)
        mask = tmpl < floor
        tmpl[mask] = floor / (dist[mask] + 1e-3)
        if tmpl.sum() > 0:
            tmpl /= tmpl.sum()
        templates[cid] = (tmpl, (y0, x0))

    models = _hybrid_models(templates)
    new_seg = seg_arr.copy()
    next_label = new_seg.max() + 1
    groups: dict[int, list[int]] = defaultdict(list)
    for cid, parent in mapping.items():
        groups[parent].append(cid)
    for parent, cids in groups.items():
        if len(cids) <= 1:
            continue
        mask = seg_arr == parent
        stack = np.zeros((len(cids),) + seg_arr.shape, dtype=float)
        for i, cid in enumerate(cids):
            model = models.get(cid)
            if model is None:
                continue
            tmpl, (y0, x0) = templates[cid]
            h, w = model.shape
            y1 = min(y0 + h, stack.shape[1])
            x1 = min(x0 + w, stack.shape[2])
            stack[i, y0:y1, x0:x1] = model[: y1 - y0, : x1 - x0]
        best = np.argmax(stack[:, mask], axis=0)
        for i, cid in enumerate(cids):
            lbl = next_label + i
            assign_mask = np.zeros_like(new_seg, dtype=bool)
            assign_mask[mask] = best == i
            new_seg[assign_mask] = lbl
        next_label += len(cids)
    new_seg, _ = clean_and_relabel(new_seg, {}, min_pix=min_pix)
    return SegmentationImage(new_seg)
