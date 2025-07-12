"""Basic source catalog creation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from photutils.background import MADStdBackgroundRMS
from astropy.stats import SigmaClip
from photutils.segmentation import SourceCatalog, detect_sources, deblend_sources
from skimage.morphology import binary_dilation, disk
from astropy.nddata import block_reduce, block_replicate

__all__ = ["Catalog", "estimate_background", "calibrate_wht"]


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
    ndilate: int = 2,
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
    ndilate: int = 2
    kernel_size: int = 5
    estimate_background: bool = False
    calibrate_wht: bool = False

    background: float = 0.0
    ivar: np.ndarray | None = None
    segmap: np.ndarray | None = None
    catalog: Table | None = None
    det_img: np.ndarray | None = None

    def _detect(self) -> None:
        self.det_img = (self.sci - self.background) * np.sqrt(self.ivar)
        kernel = Gaussian2DKernel(2.0 / 2.355, x_size=self.kernel_size, y_size=self.kernel_size)
        from astropy.convolution import convolve
        smooth = convolve(self.det_img, kernel, normalize_kernel=True)
        seg = detect_sources(smooth, threshold=2.0, npixels=5)
        seg = deblend_sources(self.det_img, seg, npixels=5, nlevels=32, contrast=1e-4, progress_bar=False)
        self.segmap = seg.data
        self.catalog = SourceCatalog(self.sci, seg, error=np.sqrt(1.0 / self.ivar))
        self.table = catalog.to_table()

    def run(self, ivar_outfile: str | Path | None = None, header: fits.Header | None = None) -> None:
        if self.estimate_background:
            self.background = estimate_background(self.sci, self.nbin)
            self.sci = self.sci - self.background
        if self.calibrate_wht:
            self.ivar = calibrate_wht(
                self.sci,
                self.wht,
                background=self.background,
                nbin=self.nbin,
                ndilate=self.ndilate,
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
