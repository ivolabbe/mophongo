"""Synthetic mosaic dataset generator for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
from astropy.wcs import WCS
from photutils.datasets import make_model_image, make_model_params
from shapely.geometry import Polygon
from reproject import reproject_interp

from .psf import PSF
from .templates import _convolve2d
from .psf_map import PSFRegionMap


@dataclass
class Frame:
    """Simple image frame with WCS."""

    data: np.ndarray
    wcs: WCS


@dataclass
class MosaicDataset:
    """Container holding the simulated dataset."""

    truth_f444w: np.ndarray
    truth_f770w: np.ndarray
    f444w: np.ndarray
    f770w: np.ndarray
    segmap: np.ndarray
    catalog: Table
    psf_f444w: List[np.ndarray]
    psf_f770w: List[np.ndarray]
    prm_f444w: PSFRegionMap
    prm_f770w: PSFRegionMap


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _make_wcs(scale_arcsec: float, size: int, pa: float = 0.0,
              center: Tuple[float, float] = (34.5, -5.2)) -> WCS:
    """Create a simple TAN WCS."""
    scale = scale_arcsec / 3600.0
    theta = np.deg2rad(pa)
    w = WCS(naxis=2)
    cd = np.array([
        [-scale * np.cos(theta), scale * np.sin(theta)],
        [scale * np.sin(theta), scale * np.cos(theta)],
    ])
    w.wcs.cd = cd
    w.wcs.crval = list(center)
    w.wcs.crpix = [size / 2, size / 2]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (size, size)
    w._naxis1 = size
    w._naxis2 = size
    return w


def _generate_truth(size: int, nsrc: int, seed: int = 1) -> Tuple[np.ndarray, Table, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    params = make_model_params(
        (size, size),
        nsrc,
        x_name="x_mean",
        y_name="y_mean",
        min_separation=20,
        border_size=40,
        seed=rng,
        amplitude=(1.0, 100.0),
        x_stddev=(1.5, 3.0),
        y_stddev=(1.5, 3.0),
        theta=(0, np.pi),
    )
    truth = make_model_image((size, size), Gaussian2D(), params,
                             bbox_factor=4.0, x_name="x_mean", y_name="y_mean")
    # add a few bright delta stars
    for _ in range(3):
        x = rng.uniform(50, size - 50)
        y = rng.uniform(50, size - 50)
        truth[int(y), int(x)] += 200.0
    flux_true = (
        params["amplitude"] * 2 * np.pi * params["x_stddev"] * params["y_stddev"]
    )
    ratio = rng.uniform(1.0, 10.0, size=len(flux_true))
    catalog = Table({
        "id": params["id"],
        "y": params["y_mean"],
        "x": params["x_mean"],
        "flux_true": flux_true,
        "ratio_770": ratio,
    })
    return truth, catalog, ratio, params


def make_mosaic_dataset(seed: int = 1) -> MosaicDataset:
    """Generate a simplified two-filter mosaic dataset."""
    # global mosaic
    pscale = 0.04  # arcsec/pix
    mosaic_size = int(45 / pscale)
    mosaic_wcs = _make_wcs(pscale, mosaic_size, 0.0)

    truth, catalog, ratio, params = _generate_truth(mosaic_size, 50, seed)

    # F444W frames
    pscale = 0.063 # arcsec/pix
    frame_size = int(30 / pscale)
    wcs_a = _make_wcs(pscale, frame_size, -20.0)
    wcs_b = _make_wcs(pscale, frame_size, 20.0)

    psf_a = PSF.gaussian(31, 2.0, 2.0).array
    psf_b = PSF.gaussian(31, 2.2, 2.2).array

    frames_f444w = []
    for wcs, psf in zip([wcs_a, wcs_b], [psf_a, psf_b]):
        data_full = _convolve2d(truth, psf)
        data_frame, _ = reproject_interp((data_full, mosaic_wcs), wcs,
                                         shape_out=(frame_size, frame_size))
        frames_f444w.append(data_frame)

    # F770W single frame
    pscale = 0.11 # arcsec/pix
    frame_size_770 = int(30 / pscale)
    wcs_c = _make_wcs(pscale, frame_size_770, 0.0)
    psf_c = PSF.gaussian(31, 3.5, 3.5).array
    
    params_770 = params.copy()
    params_770['amplitude'] = params['amplitude'] * ratio
    truth_770 = make_model_image(
        (mosaic_size, mosaic_size),
        Gaussian2D(),
        params_770,
        bbox_factor=4.0,
        x_name="x_mean",
        y_name="y_mean",
    )
    data_c_full = _convolve2d(truth_770, psf_c)
    frame_f770w, _ = reproject_interp((data_c_full, mosaic_wcs), wcs_c,
                                      shape_out=(frame_size_770, frame_size_770))

    # combine F444W frames (simple average)
    mosaic_f444w = np.zeros((mosaic_size, mosaic_size))
    weights = np.zeros((mosaic_size, mosaic_size))
    for frame, wcs in zip(frames_f444w, [wcs_a, wcs_b]):
        img, fp = reproject_interp((frame, wcs), mosaic_wcs,
                                   shape_out=(mosaic_size, mosaic_size))
        weight = np.isfinite(img).astype(float)
        mosaic_f444w += np.nan_to_num(img) * weight
        weights += weight
    weights[weights == 0] = np.inf
    mosaic_f444w /= weights

    # F770W mosaic
    mosaic_f770w, _ = reproject_interp((frame_f770w, wcs_c), mosaic_wcs,
                                       shape_out=(mosaic_size, mosaic_size))

    # noise
    rng = np.random.default_rng(seed + 1)
    noise_444 = 1.0
    noise_770 = 5.0
    mosaic_f444w += rng.normal(scale=noise_444, size=mosaic_f444w.shape)
    mosaic_f770w += rng.normal(scale=noise_770, size=mosaic_f770w.shape)

    # segmentation from F444W
    from photutils.segmentation import detect_sources
    from photutils.segmentation import deblend_sources
    from skimage.morphology import disk
    from .catalog import safe_dilate_segmentation

    seg = detect_sources(mosaic_f444w, threshold=2 * noise_444, npixels=5)
    seg.data = safe_dilate_segmentation(seg.data, selem=disk(2))
    seg = deblend_sources(mosaic_f444w, seg, npixels=5, nlevels=32,
                          contrast=0.0001, progress_bar=False)
    segmap = seg.data

    # PSF region maps
    footprints_444 = {
        "A": Polygon(wcs_a.calc_footprint()),
        "B": Polygon(wcs_b.calc_footprint()),
    }
    prm_444 = PSFRegionMap.from_footprints(footprints_444)

    footprints_770 = {
        "C": Polygon(wcs_c.calc_footprint()),
    }
    prm_770 = PSFRegionMap.from_footprints(footprints_770)

    return MosaicDataset(
        truth_f444w=truth,
        truth_f770w=truth_770,
        f444w=mosaic_f444w,
        f770w=mosaic_f770w,
        segmap=segmap,
        catalog=catalog,
        psf_f444w=[psf_a, psf_b],
        psf_f770w=[psf_c],
        prm_f444w=prm_444,
        prm_f770w=prm_770,
    )

