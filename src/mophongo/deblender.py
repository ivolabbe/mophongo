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
            num_peaks=nmax,
        )

        if loc.size == 0:
            # fall back to brightest pixel
            idx = np.unravel_index(np.argmax(masked), masked.shape)
            loc = np.array([idx])


#        values = sub[tuple(loc.T)]
#        order = np.argsort(values)[::-1][:nmax]
#        loc = loc[order]
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
    """Create symmetry templates around each peak, only for segments with multiple peaks."""
    templates: Dict[int, Tuple[np.ndarray, Tuple[int, int]]] = {}
    mapping: Dict[int, int] = {}
    cid = 1

    for parent, pts in peaks.items():
        # Skip segments with only one peak - leave them unchanged
        if len(pts) <= 1:
            continue

        print(f"Processing segment {parent} with {len(pts)} peaks: {pts}")

        # Get the parent segment mask and bounds
        idx = segm.get_index(parent)
        slc = segm.slices[idx]
        parent_mask = segm.data[slc] == parent
        segment_data = image[slc]

        for py, px in pts:
            # Convert global peak coordinates to local coordinates within the slice
            local_py = py - slc[0].start
            local_px = px - slc[1].start

            print(
                f"  Peak at global ({py}, {px}), local ({local_py}, {local_px})"
            )

            # Create template stamp - same size as segment
            stamp = np.zeros_like(segment_data)

            # Only fill pixels that belong to the parent segment
            stamp[parent_mask] = segment_data[parent_mask]

            # Apply 180-degree rotation around this specific peak
            h, w = stamp.shape

            # Create rotated stamp using proper indexing
            rot_stamp = np.zeros_like(stamp)

            # For each pixel, calculate where it should come from after rotation
            for y in range(h):
                for x in range(w):
                    # Calculate rotated coordinates
                    y_rot = 2 * local_py - y
                    x_rot = 2 * local_px - x

                    # Check bounds and copy if valid
                    if (0 <= y_rot < h and 0 <= x_rot < w and
                            parent_mask[y, x]):  # Only process parent pixels
                        rot_stamp[y, x] = stamp[int(y_rot), int(x_rot)]

            # Create symmetric template by taking minimum
            tmpl = np.minimum(stamp, rot_stamp)

            # Apply threshold
            tmpl[tmpl < eps] = eps

            # where template is eps, reduce it by distance to the peak
            # Compute distance from each pixel to the peak
            yy, xx = np.indices(tmpl.shape)
            dist = np.sqrt((yy - local_py) ** 2 + (xx - local_px) ** 2)
            # Only modify where tmpl == eps (i.e., below threshold)
            mask_eps = tmpl == eps
            # Reduce value further away from the peak (e.g., inverse with distance + 1)
            tmpl[mask_eps] = eps / (dist[mask_eps] + 1)
            
            # Only keep pixels within the original parent segment
            tmpl = np.where(parent_mask, tmpl, 0.0)

            # Normalize
            total = tmpl.sum()
            if total <= eps:
                print(f"    Template sum too low: {total}")
                continue
            tmpl /= total

            print(f"    Created template {cid} with sum {total:.6f}")

            # Store template with global coordinates
            templates[cid] = (tmpl, (slc[0].start, slc[1].start))
            mapping[cid] = parent
            cid += 1

    print(
        f"Created {len(templates)} templates for segments with multiple peaks")
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
    next_label = new_seg.max() + 1  # This starts at ~264
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
            lbl = next_label + i  # BUG: This increments for EVERY parent!
            assign_mask = np.zeros_like(new_seg, dtype=bool)
            assign_mask[mask] = best == i
            new_seg[assign_mask] = lbl
            child_labels[lbl] = parent
        next_label += len(cids)  # BUG: This keeps incrementing!
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
    eps: float = 1e-8,
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
    eps = eps or 1e-8 * np.percentile(image, 99.5)

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
