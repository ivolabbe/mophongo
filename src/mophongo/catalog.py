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
from astropy.wcs import WCS
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    SegmentationImage,
)
from photutils.segmentation.catalog import DEFAULT_COLUMNS
from photutils.segmentation import deblend_sources
from skimage.morphology import binary_dilation, dilation, disk #, square
from skimage.segmentation import watershed
from skimage.measure import label

from itertools import product

from astropy.nddata import block_reduce, block_replicate

import matplotlib.pyplot as plt

__all__ = [
    "Catalog",
    "estimate_background",
    "calibrate_wht",
]

def enlarge_slice(slc, shape, pad):
    """
    Enlarge a 2D slice by pad pixels on each side, clipped to array boundaries.

    Parameters
    ----------
    slc : tuple of slices
        (slice_y, slice_x)
    shape : tuple
        (ny, nx) shape of the array
    pad : int
        Number of pixels to pad on each side

    Returns
    -------
    tuple of slices
        Enlarged (slice_y, slice_x)
    """
    y0 = max(slc[0].start - pad, 0)
    y1 = min(slc[0].stop + pad, shape[0])
    x0 = max(slc[1].start - pad, 0)
    x1 = min(slc[1].stop + pad, shape[1])
    return (slice(y0, y1), slice(x0, x1))

def safe_dilate_segmentation(segmap: SegmentationImage | np.ndarray, selem=disk(1.5)):
    """Dilate segmentation labels without merging distinct segments."""
    if isinstance(segmap, np.ndarray):
        segmap = SegmentationImage(segmap)
    result = np.zeros_like(segmap.data)
    pad = max(selem.shape) // 2
    arr_shape = segmap.data.shape
    for segment in segmap.segments:
        seg_id = segment.label
        if seg_id == 0:
            continue  # skip background
        slc = enlarge_slice(segment.slices, arr_shape, pad)
        mask = segmap.data[slc] == seg_id
        dilated = binary_dilation(mask, selem)
        bg_mask = segmap.data[slc] == 0
        dilated_bg = np.logical_and(dilated, bg_mask)
        sub_result = result[slc]
        sub_result[dilated_bg] = seg_id
        sub_result[mask] = seg_id  # retain original
        result[slc] = sub_result
    return result

def measure_background(sci: np.ndarray, nbin: int = 4) -> float:
    """Estimate image background using sigma-clipped median."""
    print('Measuring background...')
    binned = block_reduce(sci, (nbin, nbin), func=np.mean)
    clipped = SigmaClip(sigma=3.0)(binned)
    return float(np.median(clipped))


def measure_ivar(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    background: float = 0.0,
    nbin: int = 3,
    ndilate: int = 3,
) -> np.ndarray:
    """Calibrate weight map using noise estimates from the image."""
    print("Measuring inverse variance map...") 
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
    return (expanded[:ny, :nx] / (nbin**2)).astype(np.float32)


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


# Add 'eccentricity' to the default columns for SourceCatalog
#if 'eccentricity' not in DEFAULT_COLUMNS:
#    DEFAULT_COLUMNS.append(['ra','dec','eccentricity'])

DEFAULT_COLUMNS = ['label',
 'xcentroid',
 'ycentroid',
 'sky_centroid',
 'area',
 'semimajor_sigma',
 'semiminor_sigma',
 'kron_radius',
 'eccentricity',
 'orientation',
 'min_value',
 'max_value',
 'local_background',
 'segment_flux',
 'segment_fluxerr',
 'kron_flux',
 'kron_fluxerr']

@dataclass
class Catalog:
    """Create a catalog from a science image and weight map."""

    sci: np.ndarray
    wht: np.ndarray
    nbin: int = 4
    estimate_background: bool = False
    estimate_ivar: bool = False

    background: float = 0.0
    ivar: np.ndarray | None = None
    segmap: np.ndarray | None = None
    catalog: Table | None = None
    det_catalog: SourceCatalog | None = None
    det_img: np.ndarray | None = None
    params: dict[str, float | int] = field(default_factory=dict)
    default_columns: list[str] = field(default_factory=lambda: DEFAULT_COLUMNS)
    wcs: WCS | None = None

    def __post_init__(self) -> None:
        defaults = {
            "kernel_size": 3.5,
            "detect_npixels": 5,
            "detect_threshold": 2.0,
            "dilate_segmap": 2,
            "deblend_mode": "exponential",
            "deblend_nlevels": 32,
            "deblend_contrast": 1e-4,
            "deblend_compactness": 0.0,
        }
        defaults.update(self.params)
        self.params = defaults

    def _detect(self) -> None:
        self.det_img = (self.sci - self.background) * np.sqrt(self.ivar)
        kernel_pix = int(2 * self.params["kernel_size"]) | 1 # ensure odd size
        kernel = Gaussian2DKernel(
            self.params["kernel_size"] / 2.355, x_size=kernel_pix, y_size=kernel_pix
        )
        from astropy.convolution import convolve

        print(f"Convolving with kernel size {self.params['kernel_size']} pixels")
        smooth = convolve(self.det_img, kernel, normalize_kernel=True)
        print("Detecting sources...")
        seg = detect_sources(
            smooth,
            threshold=float(self.params["detect_threshold"]),
            npixels=self.params["detect_npixels"],
        )
        # Dilate the segmentation map to include more pixels
        if self.params["dilate_segmap"] > 0:
            print(f"Dilating segmentation map with size {self.params['dilate_segmap']}")
            seg.data = safe_dilate_segmentation(
                seg, disk(self.params["dilate_segmap"])
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
        self.catalog = SourceCatalog(
            self.sci,
            seg,
            error=np.sqrt(1.0 / self.ivar),
            wcs=self.wcs
        )
        # Compute r50 and sharpness for each source and add to table
        self.table = self.catalog.to_table(self.default_columns)
        self.table['r50'] = self.catalog.fluxfrac_radius(0.5).value
#        self.table.rename_column('label', 'id')
        self.table['eccentricity'] = self.catalog.eccentricity.value
        self.table['sharpness'] =  (self.catalog.max_value * np.pi * self.table['r50']**2 / self.catalog.segment_flux).value        
        self.table['snr'] = self.table['segment_flux'] / self.table['segment_fluxerr']
        self.table.rename_columns(['label','xcentroid', 'ycentroid'],['id','x', 'y'])  
        if 'sky_centroid' in self.table.colnames:
            self.table['ra'] = [sc.ra.deg if sc is not None else np.nan for sc in self.table['sky_centroid']]
            self.table['dec'] = [sc.dec.deg if sc is not None else np.nan for sc in self.table['sky_centroid']]
            self.table.remove_column('sky_centroid')
     
    def run(
        self, ivar_outfile: str | Path | None = None, header: fits.Header | None = None
    ) -> None:
        if self.estimate_background:
            self.background = measure_background(self.sci, self.nbin)
            self.sci = self.sci - self.background
        if self.estimate_ivar:
            self.ivar = measure_ivar(
                self.sci,
                self.wht,
                background=self.background,
                nbin=self.nbin,
                ndilate=self.params["dilate_segmap"],
            )
        else:
            # assume wht is inverse variance
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
        snr_min: float = 100,
        r50_max: float = 5,
        eccen_max: float = 0.2,
        sharp_lohi: tuple[float, float] = (0.2, 1.2),
        chi2_max: float = 3.0,
        return_seg: bool = False,
    ) -> Table | tuple[Table, SegmentationImage]:
        """Find point sources in the catalog image."""

        point_like = (
            (self.table['r50'] < r50_max)
            & (self.table['eccentricity'] < eccen_max)
            & (self.table['sharpness'] > sharp_lohi[0])
            & (self.table['sharpness'] < sharp_lohi[1])
        )
        self.table['point_like'] = point_like

        table = self.table.copy()
        print('found',len(table), 'sources')

        idx_stars = np.where(point_like & (table['snr'] > snr_min))[0]
        table = table[idx_stars]
        print('kept',len(table), 'point-like sources')

        if psf is not None and len(table) > 0:
            print('fitting PSF to stamps')
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
#            table = vet_by_chi2(table, chi2_max)

        return table, idx_stars

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
        obj = cls(np.asarray(sci, dtype=float), np.asarray(wht, dtype=float), wcs=WCS(header), **kwargs)
        obj.run(ivar_outfile=ivar_outfile, header=header)
        return obj

    def show_stamp(
        self,
        idnum: int,
        offset: float = 3e-5,
        buffer: int = 20,
        ax=None,
        cmap='gray',
        alpha=0.2,
        keys: list[str] | None = None,
    ):
        """
        Show a cutout stamp of the source with segmentation mask overlay.

        Parameters
        ----------
        idnum : int
            Catalog row index (not segment label).
        buffer : int, optional
            Number of pixels to pad around the segmentation footprint.
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates a new figure.
        cmap : str, optional
            Colormap for the image.
        alpha : float, optional
            Alpha for the segmentation mask overlay.
        label : list of str, optional
            List of column names to display as text in the top left.
        """
        if self.segmap is None or self.catalog is None:
            raise RuntimeError("Catalog must be detected (run .run()) before calling showstamp.")

        idx = self.table['id'] == idnum
        row = self.table[idx][0]

        x = int(round(row['x']))
        y = int(round(row['y']))
        segm = self.segmap  # Already a SegmentationImage
        label_val = segm.data[y, x]
        if label_val == 0:
            raise ValueError("Selected position is not inside any segment.")

        idx = segm.get_index(label_val)
        bbox = segm.bbox[idx]
        # Expand bbox by buffer
        iymin = max(bbox.iymin - buffer, 0)
        iymax = min(bbox.iymax + buffer, self.sci.shape[0] - 1)
        ixmin = max(bbox.ixmin - buffer, 0)
        ixmax = min(bbox.ixmax + buffer, self.sci.shape[1] - 1)

        stamp = self.sci[iymin:iymax+1, ixmin:ixmax+1]
        segmask = segm.data[iymin:iymax+1, ixmin:ixmax+1]
#        segmask[segmask != 0] = segmask[segmask != 0] - label_val + 1  # Keep only the selected segment
        scl = stamp[segmask == label_val].sum()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure  # <-- add this line
            titles = ['data', 'psf', 'data - psf', 'data - psf x kernel', 'kernel']
            kws = dict(vmin=-5.3, vmax=-1.5, cmap='bone_r', origin='lower', interpolation='nearest')

            from matplotlib.colors import ListedColormap
            # Create a new colormap with the first color transparent
            cmap = segm.cmap
            cmap_mod = ListedColormap(
                np.vstack(([0, 0, 0, 0], cmap(np.arange(1, cmap.N))))
            )
            ax.imshow(np.log10(stamp / scl + offset), **kws)
            ax.imshow(segmask, origin='lower', cmap=cmap_mod, alpha=alpha)
            ax.axis('off')
            ax.set_title(f"ID {idnum}")
            text_lines = []
            if keys:
                for col in keys:
                    val = row[col]
                    try:
                        sval = f"{val:.2f}"
                    except Exception:
                        sval = str(val)
                    text_lines.append(f"{col}: {sval}")
            if text_lines:
                ax.text(
                    0.02, 0.98, "\n".join(text_lines),
                    color='w', fontsize=10, ha='left', va='top',
                    transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.4, lw=0)
                )

        return fig, ax
