"""Basic source catalog creation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from photutils.background import MADStdBackgroundRMS
from astropy.stats import SigmaClip
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    deblend_sources,
    SegmentationImage,
)
from skimage.morphology import dilation, disk, max_tree
from skimage.segmentation import watershed
from skimage.measure import label

from itertools import product

from astropy.nddata import block_reduce, block_replicate

__all__ = [
    "Catalog",
    "estimate_background",
    "calibrate_wht",
    "deblend_sources_lutz",
]


def safe_dilate_segmentation(segmap, selem=disk(1)):
    result = np.zeros_like(segmap)
    for seg_id in np.unique(segmap):
        if seg_id == 0:
            continue  # skip background
        mask = segmap == seg_id
        dilated = dilation(mask, selem)
        # Only allow dilation into background
        dilated = np.logical_and(dilated, segmap == 0)
        result[dilated] = seg_id
        result[mask] = seg_id  # retain original
    return result


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

        # Calculate total segment flux for absolute threshold
        total_segment_flux = np.sum(subimg[submask])
        flux_threshold = contrast * total_segment_flux
        
        # Max-tree analysis to find local peaks
        minval = float(subimg.min()) - 1.0
        work = np.where(submask, subimg, minval)
        
        # Get max-tree
        parent, tree_order = max_tree(work, connectivity=2)
        parent = parent.ravel()
        
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


def estimate_background(sci: np.ndarray, nbin: int = 4) -> float:
    """Estimate image background using sigma-clipped median."""
    binned = block_reduce(sci, (nbin, nbin), func=np.mean)
    clipped = SigmaClip(sigma=3.0)(binned)
    return float(np.median(clipped))


def calibrate_wht(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    background: float = 0.0,
    nbin: int = 4,
    ndilate: int = 3,
) -> np.ndarray:
    """Calibrate weight map using noise estimates from the image."""
    sci_sub = sci - background
    sci_bin = block_reduce(sci_sub, (nbin, nbin), func=np.mean)
    wht_bin = block_reduce(wht, (nbin, nbin), func=np.mean)
    det_bin = sci_bin * np.sqrt(wht_bin)
    clipped = SigmaClip(sigma=3.0)(det_bin)
    mask = clipped.mask
    if ndilate > 0:
        mask = binary_dilation(mask, disk(ndilate))
    std = MADStdBackgroundRMS()(det_bin[~mask])
    sqrt_wht = np.sqrt(wht_bin) / std
    wht_bin_cal = sqrt_wht**2
    expanded = block_replicate(wht_bin_cal, (nbin, nbin), conserve_sum=False)
    ny, nx = sci.shape
    if expanded.shape[0] < ny or expanded.shape[1] < nx:
        pad_y = ny - expanded.shape[0]
        pad_x = nx - expanded.shape[1]
        expanded = np.pad(expanded, ((0, pad_y), (0, pad_x)), mode="edge")
    return expanded[:ny, :nx] / (nbin**2)


@dataclass
class Catalog:
    """Create a catalog from a science image and weight map."""

    sci: np.ndarray
    wht: np.ndarray
    nbin: int = 4
    estimate_background: bool = False
    calibrate_wht: bool = False

    background: float = 0.0
    ivar: np.ndarray | None = None
    segmap: np.ndarray | None = None
    catalog: Table | None = None
    det_catalog: SourceCatalog | None = None
    det_img: np.ndarray | None = None
    params: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        defaults = {
            "kernel_size": 4.0,
            "detect_npixels": 5,
            "dilate_segmap": 3,
            "deblend_mode": "exponential",
            "deblend_nlevels": 64,
            "deblend_contrast": 1e-3,
        }
        defaults.update(self.params)
        self.params = defaults

    def _detect(self) -> None:
        self.det_img = (self.sci - self.background) * np.sqrt(self.ivar)
        kernel_pix = int(2 * self.params["kernel_size"] + 1)
        kernel = Gaussian2DKernel(
            self.params["kernel_size"] / 2.355, x_size=kernel_pix, y_size=kernel_pix
        )
        from astropy.convolution import convolve

        smooth = convolve(self.det_img, kernel, normalize_kernel=True)
        seg = detect_sources(
            smooth,
            threshold=float(self.params["detect_threshold"]),
            npixels=self.params["detect_npixels"],
        )
        # Dilate the segmentation map to include more pixels
        if self.params["dilate_segmap"] > 0:
            seg.data = safe_dilate_segmentation(
                seg.data, disk(self.params["dilate_segmap"])
            )
        seg = deblend_sources(
            self.det_img,
            seg,
            npixels=self.params["detect_npixels"],
            mode=self.params["deblend_mode"],
            nlevels=int(self.params["deblend_nlevels"]),
            contrast=float(self.params["deblend_contrast"]),
            connectivity=8,
            progress_bar=False,
        )
        self.segmap = seg
        self.det_catalog = SourceCatalog(
            self.sci,
            seg,
            error=np.sqrt(1.0 / self.ivar),
        )
        self.catalog = self.det_catalog.to_table()

    def run(
        self, ivar_outfile: str | Path | None = None, header: fits.Header | None = None
    ) -> None:
        if self.estimate_background:
            self.background = estimate_background(self.sci, self.nbin)
            self.sci = self.sci - self.background
        if self.calibrate_wht:
            self.ivar = calibrate_wht(
                self.sci,
                self.wht,
                background=self.background,
                nbin=self.nbin,
                ndilate=self.params["dilate_segmap"],
            )
        else:
            self.ivar = self.wht
        if ivar_outfile is not None:
            fits.writeto(
                ivar_outfile,
                self.ivar.astype(np.float32),
                header=header,
                overwrite=True,
            )
        self._detect()

    @classmethod
    def from_fits(
        cls,
        sci_file: str | Path,
        wht_file: str | Path,
        *,
        ivar_outfile: str | Path | None = None,
        **kwargs,
    ) -> "Catalog":
        sci = fits.getdata(sci_file)
        wht = fits.getdata(wht_file)
        header = fits.getheader(sci_file)
        obj = cls(np.asarray(sci, dtype=float), np.asarray(wht, dtype=float), **kwargs)
        obj.run(ivar_outfile=ivar_outfile, header=header)
        return obj
