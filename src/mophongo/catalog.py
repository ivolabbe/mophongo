"""Basic source catalog creation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve
from astropy.io import fits
from astropy.table import Table
from photutils.background import MADStdBackgroundRMS
from astropy.stats import SigmaClip, mad_std
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    SegmentationImage,
)
from photutils.segmentation import deblend_sources
from skimage.morphology import dilation, disk, max_tree, square
from skimage.segmentation import watershed
from skimage.measure import label

from itertools import product

from astropy.nddata import block_reduce, block_replicate

__all__ = [
    "Catalog",
    "estimate_background",
    "calibrate_wht",
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
        mask = safe_dilate_segmentation(mask, disk(ndilate))
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


def noise_equalised_image(data: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Return image divided by the per-pixel noise."""
    if weight is None:
        return data
    return data * np.sqrt(weight)


def detect_peaks(
    img_eq: np.ndarray,
    sigma: float = 3.0,
    npix_min: int = 5,
    kernel_w: int = 3,
) -> tuple[SegmentationImage, SourceCatalog]:
    """Detect peaks in a noise-equalised image."""

    sm = convolve(img_eq, Box2DKernel(kernel_w), normalize_kernel=True)
    std = mad_std(sm)
    seg = detect_sources(sm, sigma * std, npixels=npix_min)
    props = SourceCatalog(img_eq, seg)
    return seg, props


def point_like_mask(
    props: SourceCatalog,
    r50_max_pix: float = 1.5,
    elong_max: float = 1.3,
    sharp_lohi: tuple[float, float] = (0.2, 1.2),
) -> np.ndarray:
    """Return mask for point-like sources."""

    good = []
    for prop in props:
        r50 = prop.equivalent_radius.value
        elong = prop.elongation.value
        sharp = prop.max_value * np.pi * r50 ** 2 / prop.segment_flux
        good.append(
            (r50 < r50_max_pix)
            and (elong < elong_max)
            and (sharp > sharp_lohi[0])
            and (sharp < sharp_lohi[1])
        )
    return np.array(good, dtype=bool)


def fit_psf_stamp(
    data: np.ndarray,
    sigma: np.ndarray,
    psf_model: np.ndarray,
) -> tuple[float, float]:
    """Fit a PSF to a small stamp and return flux and reduced chi^2."""

    y, x = np.indices(data.shape)
    flat = np.ones_like(psf_model)

    A = np.vstack([(psf_model / sigma), (flat / sigma)]).reshape(2, -1).T
    b = (data / sigma).ravel()
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    model = coeff[0] * psf_model + coeff[1]
    chi2 = np.sum(((data - model) ** 2) / sigma ** 2)
    dof = data.size - 2
    return coeff[0], chi2 / dof


def vet_by_chi2(star_list: Table, chi2_max: float = 3.0) -> Table:
    """Filter table rows by reduced chi^2."""

    mask = star_list['chi2_red'] < chi2_max
    return star_list[mask]


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
            "kernel_size": 3.5,
            "detect_npixels": 5,
            "dilate_segmap": 3,
            "deblend_mode": "exponential",
            "deblend_nlevels": 32,
            "deblend_contrast": 1e-4,
            "deblend_compactness": 0.0,
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
                seg.data, square(self.params["dilate_segmap"])
            )
        if self.params["deblend_mode"] is not None:
            seg = deblend_sources(
                self.det_img,
                seg,
                npixels=self.params["detect_npixels"],
                mode=self.params["deblend_mode"],
                nlevels=int(self.params["deblend_nlevels"]),
                contrast=float(self.params["deblend_contrast"]),
                connectivity=8,
                progress_bar=False,
    #            compactness=float(self.params.get("deblend_compactness", 0.0)),
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

    def find_stars(
        self,
        *,
        psf: np.ndarray | None = None,
        sigma: float = 3.0,
        npix_min: int = 5,
        kernel_w: int = 3,
        r50_max_pix: float = 1.5,
        elong_max: float = 1.3,
        sharp_lohi: tuple[float, float] = (0.2, 1.2),
        chi2_max: float = 3.0,
        return_seg: bool = False,
    ) -> Table | tuple[Table, SegmentationImage]:
        """Find point sources in the catalog image."""

        if self.ivar is None:
            raise RuntimeError("Run the catalog first to compute ivar")

        img_eq = noise_equalised_image(self.sci - self.background, self.ivar)
        seg, props_eq = detect_peaks(img_eq, sigma, npix_min, kernel_w)

        props = SourceCatalog(
            self.sci,
            seg,
            error=np.sqrt(1.0 / self.ivar),
        )

        keep = point_like_mask(props_eq, r50_max_pix, elong_max, sharp_lohi)
        table = props.to_table()[keep]
        table['x'] = table['xcentroid']
        table['y'] = table['ycentroid']
        if 'segment_fluxerr' in table.colnames:
            snr = table['segment_flux'] / table['segment_fluxerr']
        else:
            snr = np.full(len(table), np.nan)
        table['flux'] = table['segment_flux']
        table['snr'] = snr

        if psf is not None and len(table) > 0:
            chi2 = []
            flux_psf = []
            half = psf.shape[0] // 2
            for row in table:
                y0 = int(row['y'])
                x0 = int(row['x'])
                y_slice = slice(max(0, y0 - half), min(self.sci.shape[0], y0 + half + 1))
                x_slice = slice(max(0, x0 - half), min(self.sci.shape[1], x0 + half + 1))
                stamp = self.sci[y_slice, x_slice]
                sigma_im = np.sqrt(1.0 / self.ivar[y_slice, x_slice])
                if stamp.shape != psf.shape:
                    chi2.append(np.inf)
                    flux_psf.append(np.nan)
                    continue
                flux, c2 = fit_psf_stamp(stamp, sigma_im, psf)
                chi2.append(c2)
                flux_psf.append(flux)
            table['flux_psf'] = flux_psf
            table['chi2_red'] = chi2
            table = vet_by_chi2(table, chi2_max)

        return (table, seg) if return_seg else table

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
