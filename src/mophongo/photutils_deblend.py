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



def _steepest_descent_labels(
    image: np.ndarray,
    seed_mask: np.ndarray,
    mask: np.ndarray,
    *,
    connectivity: int = 8,
) -> np.ndarray:
    """Assign pixels to seeds by steepest descent."""

    ny, nx = image.shape
    labels = np.zeros_like(image, dtype=int)

    # connectivity offsets
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    # label seeds
    seed_positions = np.column_stack(np.nonzero(seed_mask))
    for i, (y, x) in enumerate(seed_positions, start=1):
        labels[y, x] = i

    # assign remaining pixels
    unlabeled = np.column_stack(np.nonzero(mask & (labels == 0)))
    for y0, x0 in unlabeled:
        path = []
        y, x = y0, x0
        while True:
            if labels[y, x] > 0:
                lbl = labels[y, x]
                break
            path.append((y, x))
            max_val = image[y, x]
            yn, xn = y, x
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx]:
                    val = image[yy, xx]
                    if val > max_val:
                        max_val = val
                        yn, xn = yy, xx
            if (yn, xn) == (y, x):
                lbl = 0
                break
            y, x = yn, xn
        for py, px in path:
            labels[py, px] = lbl

    return labels


def deblend_sources_lutz(
    det_image: np.ndarray,
    segmap: SegmentationImage,
    *,
    npixels: int = 5,
    contrast: float = 1e-3,
) -> SegmentationImage:
    """Deblend segmentation map using a SEP-like algorithm."""
    
    from skimage.measure import label
    from skimage.segmentation import watershed
    
    new_seg = np.zeros_like(segmap.data, dtype=int)
    next_label = 1

    for segment in segmap.segments:
        seg_id = segment.label
        
        # Extract subimages using the segment's own slices
        slices = segment.slices
        subimg = det_image[slices]
        submask = segment.data != 0
        
        print(f"Segment {seg_id}: slices={slices}, mask_pixels={submask.sum()}")
        
        if submask.sum() == 0:
            continue

        # Check for degenerate dimensions that cause max_tree to fail
        shape = subimg.shape
        if min(shape) <= 1 or submask.sum() < 2:
            print(f"  Skipping degenerate segment: shape={shape}, pixels={submask.sum()}")
            # Just assign a single label to the whole segment
            new_seg[slices][submask] = next_label
            next_label += 1
            continue

        # Calculate total segment flux for absolute threshold
        total_segment_flux = np.sum(subimg[submask])
        flux_threshold = contrast * total_segment_flux
        
        # Max-tree analysis to find local peaks
        minval = float(subimg.min()) - 1.0
        work = np.where(submask, subimg, minval)

        print('work',work.shape, work.min(), work.max())
        # Ensure work array has no invalid values
        work = np.nan_to_num(work, nan=minval, posinf=subimg.max(), neginf=minval)

        try:
            # Get max-tree - connectivity should be 2 for 2D images
            parent, tree_order = max_tree(work, connectivity=2)
            parent = parent.ravel()
        except ValueError as e:
            print(f"  Max-tree failed for segment {seg_id}: {e}")
            # Fall back to single label
            new_seg[slices][submask] = next_label
            next_label += 1
            continue
        
        # Calculate tree attributes (area and flux) bottom-up
        n_nodes = parent.size
        area = np.ones(n_nodes, dtype=int)
        flux = work.ravel().copy()
        
        # Build tree attributes bottom-up
        for node_idx in tree_order[::-1]:
            parent_idx = parent[node_idx]
            if parent_idx != node_idx:  # Not root
                area[parent_idx] += area[node_idx]
                flux[parent_idx] += flux[node_idx]
        
        # Find leaves (nodes with no children)
        has_children = np.zeros(n_nodes, dtype=bool)
        for node_idx in range(n_nodes):
            parent_idx = parent[node_idx]
            if parent_idx != node_idx:  # Not root
                has_children[parent_idx] = True
        
        mask_flat = submask.ravel()
        leaves = np.where(~has_children & mask_flat)[0]
        
        print(f"  Found {len(leaves)} total leaves")
        print(f"  Total segment flux: {total_segment_flux:.3f}, threshold: {flux_threshold:.6f}")
        
        # Apply absolute flux threshold (like photutils)
        valid_seeds = []
        for leaf_idx in leaves:
            leaf_flux = flux[leaf_idx]
            
            print(f"    Leaf {leaf_idx}: flux={leaf_flux:.6f}, "
                  f"fraction={leaf_flux/total_segment_flux:.6f}")
            
            # Use absolute flux threshold instead of relative contrast
            if leaf_flux >= flux_threshold and leaf_flux > 0:
                valid_seeds.append(leaf_idx)
        
        print(f"  Found {len(valid_seeds)} valid seeds after flux threshold filtering")
        
        if len(valid_seeds) <= 1:
            # No deblending needed
            new_seg[slices][submask] = next_label
            next_label += 1
            continue
        
        # Create markers for watershed using only surviving leaves
        markers = np.zeros_like(submask, dtype=int)
        for i, seed_idx in enumerate(valid_seeds):
            row, col = np.unravel_index(seed_idx, submask.shape)
            markers[row, col] = i + 1
        
        # Apply watershed
        labels_sub = watershed(-subimg, markers=markers, mask=submask)
        
        # Apply area filtering to the actual watershed regions
        final_labels = np.unique(labels_sub[labels_sub > 0])
        print(f"  Before area filtering: {len(final_labels)} regions")
        
        for lbl in final_labels:
            region_mask = labels_sub == lbl
            region_size = region_mask.sum()
            if region_size < npixels:
                print(f"    -> Removing region {lbl} ({region_size} < {npixels} pixels)")
                labels_sub[region_mask] = 0
        
        # Add to final segmentation with consecutive labeling
        final_labels = np.unique(labels_sub[labels_sub > 0])
        print(f"  Final: {len(final_labels)} regions after area filtering")
        
        if len(final_labels) > 0:
            for i, lbl in enumerate(final_labels):
                region_mask = labels_sub == lbl
                labels_sub[region_mask] = next_label + i
            
            valid_mask = labels_sub > 0
            new_seg[slices][valid_mask] = labels_sub[valid_mask]
            next_label += len(final_labels)

    return SegmentationImage(new_seg)


def deblend_sources_color(
    sci1: np.ndarray,
    ivar1: np.ndarray,
    sci2: np.ndarray,
    ivar2: np.ndarray,
    *,
    kernel_size: float = 3.0,
    detect_threshold: float = 1.0,
    npixels: int = 5,
    nlevels: int = 128,
    contrast: float = 1e-4,
    color_thresh: float = 0.3,
    nsigma: float = 3.0,
    compactness: float = 0.0,
) -> SegmentationImage:
    """Chi² detection and colour-aware deblending.

    The two input images are combined into a chi² detection image.  Standard
    :func:`photutils.segmentation.deblend_sources` is applied and the resulting
    children are kept only if their colour contrast relative to the parent
    region exceeds ``color_thresh`` with a significance larger than
    ``nsigma``.
    """

    from astropy.convolution import convolve

    chi2 = (sci1 * np.sqrt(ivar1)) ** 2 + (sci2 * np.sqrt(ivar2)) ** 2

    kpix = int(2 * kernel_size + 1)
    kernel = Gaussian2DKernel(kernel_size / 2.355, x_size=kpix, y_size=kpix)
    chi2_smooth = convolve(chi2, kernel, normalize_kernel=True)

    seg_det = detect_sources(chi2_smooth, detect_threshold, npixels=npixels)
    seg_deb = deblend_sources(
        chi2,
        seg_det,
        npixels=npixels,
        nlevels=nlevels,
        mode='exponential',
        contrast=contrast,
        progress_bar=False,
#        compactness=compactness,
    )

    print(f"Initial detection: {len(seg_det.labels)} sources")
    print(f"After deblending: {len(seg_deb.labels)} sources")

    seg_final = seg_deb.data.copy()

    for seg_id in seg_det.labels:
        parent_mask = seg_det.data == seg_id
        child_ids = np.unique(seg_deb.data[parent_mask])
        child_ids = child_ids[child_ids != 0]
        
        print(f"\nSource {seg_id}: {len(child_ids)} children")
        
        if len(child_ids) <= 1:
#            print(f"  No deblending for source {seg_id}")
            continue

        # Parent properties
        f1_p = sci1[parent_mask].sum()
        f2_p = sci2[parent_mask].sum()
        var1_p = (1.0 / ivar1[parent_mask]).sum()
        var2_p = (1.0 / ivar2[parent_mask]).sum()
        color_p = f1_p / max(f2_p, 1e-9)
        sn1_p = f1_p / np.sqrt(var1_p) if var1_p > 0 else 0.0
        sn2_p = f2_p / np.sqrt(var2_p) if var2_p > 0 else 0.0
        sig_p = (1 / np.log(10)) * np.sqrt(
            (1 / sn1_p**2 if sn1_p > 0 else np.inf)
            + (1 / sn2_p**2 if sn2_p > 0 else np.inf)
        )
        chi_p = chi2[parent_mask].sum()
        
        if len(child_ids) > 10:        
            print(f"  Parent: flux1={f1_p:.3f}, flux2={f2_p:.3f}, color={color_p:.3f}, "
              f"chi2={chi_p:.3f}, sig={sig_p:.3f}")

        children_kept = 0
        children_rejected = 0

        for cid in child_ids:
            mask = seg_deb.data == cid
            f1 = sci1[mask].sum()
            f2 = sci2[mask].sum()
            var1 = (1.0 / ivar1[mask]).sum()
            var2 = (1.0 / ivar2[mask]).sum()
            chi_c = chi2[mask].sum()
            
            # Calculate noise thresholds (1-sigma)
            noise1 = np.sqrt(var1) if var1 > 0 else np.inf
            noise2 = np.sqrt(var2) if var2 > 0 else np.inf
            
            # Apply noise floor to fluxes for color calculation
            f1_thresh = max(f1, noise1)  # Use 1-sigma if flux < noise
            f2_thresh = max(f2, noise2)
            f1_p_thresh = max(f1_p, np.sqrt(var1_p)) if var1_p > 0 else f1_p
            f2_p_thresh = max(f2_p, np.sqrt(var2_p)) if var2_p > 0 else f2_p
            
            # Calculate colors using thresholded fluxes
            color_c = f1_thresh / f2_thresh
            color_p = f1_p_thresh / f2_p_thresh
            
            # Signal-to-noise ratios with 1-sigma floor for non-detections
            sn1 = max(f1 / noise1 if noise1 > 0 else 0.0, 1.0)
            sn2 = max(f2 / noise2 if noise2 > 0 else 0.0, 1.0)
            sn1_p = max(f1_p / np.sqrt(var1_p) if var1_p > 0 else 0.0, 1.0)
            sn2_p = max(f2_p / np.sqrt(var2_p) if var2_p > 0 else 0.0, 1.0)

            # Color uncertainties in log10 space using floored S/N
            sig_c = (1 / np.log(10)) * np.sqrt(
                (1 / sn1**2) + (1 / sn2**2)
            )
            sig_p = (1 / np.log(10)) * np.sqrt(
                (1 / sn1_p**2) + (1 / sn2_p**2)
            )
            
            # Color difference in log10 space
            delta = np.abs(np.log10(color_c) - np.log10(color_p))
            sig_delta = np.sqrt(sig_c**2 + sig_p**2)
            
            # Check rejection criteria
            flux_ratio = chi_c / chi_p
            color_diff = delta
            color_significance = delta / sig_delta if sig_delta > 0 and np.isfinite(sig_delta) else 0
            
            reject_flux = flux_ratio < contrast
            reject_color_diff = color_diff < color_thresh
            reject_color_sig = color_significance < nsigma
            
            # Additional check: reject if both bands are pure noise
            both_noise = (sn1 < 1.0 and sn2 < 1.0)
            if len(child_ids) > 10:
                print(f"    Child {cid}: flux1={f1:.3f}±{noise1:.3f} (S/N={sn1:.1f}), "
                    f"flux2={f2:.3f}±{noise2:.3f} (S/N={sn2:.1f})")
                print(f"      Color: {color_c:.3f} (thresh flux), actual: {f1/max(f2,1e-9):.3f}")
                print(f"      Chi2 ratio: {flux_ratio:.6f} (thresh={contrast}, reject={reject_flux})")
                print(f"      Color diff: {color_diff:.3f} (thresh={color_thresh}, reject={reject_color_diff})")
                print(f"      Color sig:  {color_significance:.3f} (thresh={nsigma}, reject={reject_color_sig})")
                print(f"      Both noise: {both_noise}")

            if (reject_flux or reject_color_diff or reject_color_sig or both_noise):
                if len(child_ids) > 10:
                    print(f"      -> REJECTED")
                seg_final[mask] = seg_id
                children_rejected += 1
            else:
                if len(child_ids) > 10:
                    print(f"      -> KEPT as separate source")
                children_kept += 1
        
        if len(child_ids) > 10:
            print(f"  Final: {children_kept} children kept, {children_rejected} rejected")

    seg_final = label(seg_final, connectivity=2)
    print(f"Final segmentation: {len(np.unique(seg_final))-1} sources")
    return SegmentationImage(seg_final)
