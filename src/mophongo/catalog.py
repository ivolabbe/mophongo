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
from photutils.segmentation import SourceCatalog, detect_sources, deblend_sources, SegmentationImage
from skimage.morphology import dilation, disk, max_tree

from astropy.nddata import block_reduce, block_replicate

__all__ = ["Catalog", "estimate_background", "calibrate_wht"]

# ------------------------------------------------------------------
#  utilities (unchanged)
# ------------------------------------------------------------------
def _offsets(connectivity=2):
    return [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),         ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)] if connectivity == 2 else \
           [(-1,0), (0,-1), (0,1), (1,0)]

def _lutz_assign(img, markers, mask, connectivity=2, start_id=1):
    seg       = markers.astype(np.int32, copy=True)
    next_id   = start_id
    h, w      = img.shape
    flat_I    = img.ravel()
    flat_S    = seg.ravel()
    flat_mask = mask.ravel()

    order     = np.argsort(flat_I[flat_mask & (flat_S == 0)])[::-1]
    pix_idx   = np.flatnonzero(flat_mask & (flat_S == 0))[order]
    nbr_off   = _offsets(connectivity)

    for p in pix_idx:
        r, c    = divmod(p, w)
        best_int, best_lab = -np.inf, 0
        for dr, dc in nbr_off:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                q, lab = rr * w + cc, flat_S[rr * w + cc]
                if lab and flat_I[q] > best_int:
                    best_int, best_lab = flat_I[q], lab
        if best_lab:
            flat_S[p] = best_lab
        else:                     # new isolated maximum
            flat_S[p] = next_id
            next_id  += 1
    return seg, next_id

def _tree_attributes(parent, trav, img_flat):
    area  = np.ones(parent.size, int)
    flux  = img_flat.astype(float).copy()
    for p in trav[::-1]:               # bottom-up
        par = parent.flat[p]
        if par != p:
            area[par] += area[p]
            flux[par] += flux[p]
    return area, flux

def _surviving_leaves(parent, area, flux, contrast, npixels):
    is_child = np.zeros(parent.size, bool)
    is_child[parent.flat] = True
    leaves   = np.flatnonzero(~is_child)
    keep = []
    for leaf in leaves:
        par = parent.flat[leaf]
        if par == leaf or (area[leaf] >= npixels and
                           flux[leaf] >= contrast * flux[par]):
            keep.append(leaf)
    return np.array(keep, int)

# ------------------------------------------------------------------
#  deblend ONE island  ---------------------------------------------
# ------------------------------------------------------------------
def deblend_island(sub, sub_mask,
                   npixels=5, contrast=0.005,
                   connectivity=2, start_id=1):
    parent, trav = max_tree(sub, connectivity=connectivity)
    area, flux   = _tree_attributes(parent, trav, sub.ravel())
    leaves       = _surviving_leaves(parent, area, flux,
                                     contrast, npixels)
    if leaves.size == 0:
        return np.zeros_like(sub, np.int32), start_id

    # seed image
    markers = np.zeros_like(sub, np.int32)
    nid = start_id
    for leaf in leaves:
        r, c = np.unravel_index(leaf, sub.shape)
        markers[r, c] = nid
        nid += 1

    return _lutz_assign(sub, markers, sub_mask,
                        connectivity, nid)[0], nid

# ------------------------------------------------------------------
#  NEW: relabel consecutive  ---------------------------------------
# ------------------------------------------------------------------
def _relabel_consecutive(seg):
    labels = np.unique(seg)
    labels = labels[labels != 0]
    if (labels == np.arange(1, len(labels)+1)).all():
        return seg                       # already consecutive
    lut = np.zeros(seg.max()+1, seg.dtype)
    lut[labels] = np.arange(1, len(labels)+1, dtype=seg.dtype)
    return lut[seg]

# ------------------------------------------------------------------
#  MAIN deblend function taking Photutils SegmentationImage --------
# ------------------------------------------------------------------
def deblend_sources_lutz(image,
                    segm,                           # SegmentationImage | ndarray
                    npixels=5, contrast=0.005,
                    connectivity=2, relabel=True):
    """
    Parameters
    ----------
    image : 2-D ndarray  – science frame
    segm  : photutils.segmentation.SegmentationImage
            or labelled ndarray (islands)
    Returns
    -------
    seg_full : ndarray[int32] – final SExtractor-like segmentation
    """
    if isinstance(segm, SegmentationImage):
        seg_data = segm.data
        slices   = segm.slices
        n_isles  = segm.nlabels
    else:                                # plain ndarray fallback
        seg_data = np.asanyarray(segm)
        n_isles  = int(seg_data.max())
        slices   = [None] * n_isles
        # build bounding-box list on the fly
        for lab in range(1, n_isles+1):
            coords = np.nonzero(seg_data == lab)
            slices[lab-1] = (slice(coords[0].min(), coords[0].max()+1),
                             slice(coords[1].min(), coords[1].max()+1))

    seg_full = np.zeros_like(image, np.int32)
    next_id  = 1

    for lab in range(1, n_isles+1):
        slc       = slices[lab-1]
        sub       = image[slc]
        sub_mask  = seg_data[slc] == lab

        seg_sub, next_id = deblend_island(sub, sub_mask,
                                          npixels, contrast,
                                          connectivity, next_id)
        seg_full[slc][seg_sub > 0] = seg_sub[seg_sub > 0]

    if relabel:
        seg_full = _relabel_consecutive(seg_full)

    return SegmentationImage(seg_full)

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
            "deblend_mode": 'exponential',
            "deblend_nlevels": 64,
            "deblend_contrast": 1e-3,
        }
        defaults.update(self.params)
        self.params = defaults

    def _detect(self) -> None:
        self.det_img = (self.sci - self.background) * np.sqrt(self.ivar)
        kernel_pix = int(2 * self.params["kernel_size"] + 1)
        kernel = Gaussian2DKernel(self.params["kernel_size"] / 2.355,
                                  x_size=kernel_pix,
                                  y_size=kernel_pix)
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
                seg.data, disk(self.params["dilate_segmap"]))
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

    def run(self,
            ivar_outfile: str | Path | None = None,
            header: fits.Header | None = None) -> None:
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
        obj = cls(np.asarray(sci, dtype=float), np.asarray(wht, dtype=float),
                  **kwargs)
        obj.run(ivar_outfile=ivar_outfile, header=header)
        return obj
