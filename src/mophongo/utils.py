"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from shapely.geometry import Polygon

# model based stuff 
def elliptical_moffat(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Moffat profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
    return amplitude * (1 + r2) ** (-beta)


def elliptical_gaussian(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Gaussian profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    r2 = (xr / sigma_x) ** 2 + (yr / sigma_y) ** 2
    return amplitude * np.exp(-0.5 * r2)


def measure_shape(data: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``x_c``, ``y_c``, ``sigma_x``, ``sigma_y``, and ``theta`` of ``data``.

    Parameters
    ----------
    data : ndarray
        Pixel data.
    mask : ndarray
        Boolean mask selecting the object pixels.
    """
    y_idx, x_idx = np.indices(data.shape)
    flux = float(data[mask].sum())
    y_c = float((y_idx[mask] * data[mask]).sum() / flux)
    x_c = float((x_idx[mask] * data[mask]).sum() / flux)
    y_rel = y_idx - y_c
    x_rel = x_idx - x_c
    cov_xx = float((data[mask] * x_rel[mask] ** 2).sum() / flux)
    cov_yy = float((data[mask] * y_rel[mask] ** 2).sum() / flux)
    cov_xy = float((data[mask] * x_rel[mask] * y_rel[mask]).sum() / flux)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    sigma_x = float(np.sqrt(vals[0]))
    sigma_y = float(np.sqrt(vals[1]))
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return x_c, y_c, sigma_x, sigma_y, theta


def moffat(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Moffat PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy
    
    # Convert flux to amplitude analytically
    # For a Moffat profile: flux = amplitude * pi * alpha_x * alpha_y / (beta - 1)
    # where alpha = fwhm / (2 * sqrt(2^(1/beta) - 1))
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    amplitude = flux * (beta - 1) / (np.pi * alpha_x * alpha_y)
    
    psf = elliptical_moffat(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        beta,
        theta,
        x0,
        y0,
    )
    return psf


def gaussian(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Gaussian PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy
    
    # Convert flux to amplitude analytically
    # For a Gaussian profile: flux = amplitude * 2 * pi * sigma_x * sigma_y
    # where sigma = fwhm / (2 * sqrt(2 * ln(2)))
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    amplitude = flux / (2 * np.pi * sigma_x * sigma_y)
    
    psf = elliptical_gaussian(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        theta,
        x0,
        y0,
    )
    return psf


# ---------------------------------------------------------------------
# Basic WCS utilities
# ---------------------------------------------------------------------
def get_wcs_pscale(wcs, set_attribute=True):
    """Pixel scale in arcsec from a ``WCS`` object."""
    from numpy.linalg import det

    if isinstance(wcs, fits.Header):
        wcs = WCS(wcs, relax=True)

    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
        detv = det(wcs.wcs.cd)
    else:
        detv = det(wcs.wcs.pc)

    pscale = np.sqrt(np.abs(detv)) * 3600.0
    if set_attribute:
        wcs.pscale = pscale
    return pscale


def to_header(wcs, add_naxis=True, relax=True, key=None):
    """Convert WCS to a FITS header with a few extra keywords."""
    hdr = wcs.to_header(relax=relax, key=key)
    if add_naxis:
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs.pixel_shape[0]
            hdr["NAXIS2"] = wcs.pixel_shape[1]
        elif hasattr(wcs, "_naxis1"):
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs._naxis1
            hdr["NAXIS2"] = wcs._naxis2

    if hasattr(wcs.wcs, "cd"):
        for i in [0, 1]:
            for j in [0, 1]:
                hdr[f"CD{i + 1}_{j + 1}"] = wcs.wcs.cd[i][j]

    if hasattr(wcs, "sip") and wcs.sip is not None:
        hdr["SIPCRPX1"], hdr["SIPCRPX2"] = wcs.sip.crpix
    return hdr


def get_slice_wcs(wcs, slx, sly):
    """Slice a WCS while propagating SIP and distortion keywords."""
    nx = slx.stop - slx.start
    ny = sly.stop - sly.start
    swcs = wcs.slice((sly, slx))

    if hasattr(swcs, "_naxis1"):
        swcs.naxis1 = swcs._naxis1 = nx
        swcs.naxis2 = swcs._naxis2 = ny
    else:
        swcs._naxis = [nx, ny]
        swcs._naxis1 = nx
        swcs._naxis2 = ny

    if hasattr(swcs, "sip") and swcs.sip is not None:
        for c in [0, 1]:
            swcs.sip.crpix[c] = swcs.wcs.crpix[c]

    acs = [4096 / 2, 2048 / 2]
    dx = swcs.wcs.crpix[0] - acs[0]
    dy = swcs.wcs.crpix[1] - acs[1]
    for ext in ["cpdis1", "cpdis2", "det2im1", "det2im2"]:
        if hasattr(swcs, ext):
            extw = getattr(swcs, ext)
            if extw is not None:
                extw.crval[0] += dx
                extw.crval[1] += dy
                setattr(swcs, ext, extw)
    return swcs


# ---------------------------------------------------------------------
# WCS information from CSV
# ---------------------------------------------------------------------
def read_wcs_csv(drz_file, csv_file=None):
    """Read exposure WCS info from a CSV table."""
    if csv_file is None:
        csv_file = (
            drz_file.split("_drz_sci")[0].split("_drc_sci")[0] + "_wcs.csv"
        )
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")

    tab = Table.read(csv_file, format="csv")
    flt_keys = []
    wcs_dict = {}
    footprints = {}

    for row in tab:
        key = (row["file"], row["ext"])
        hdr = fits.Header()
        for col in tab.colnames:
            hdr[col] = row[col]

        wcs = WCS(hdr, relax=True)
        get_wcs_pscale(wcs)
        wcs.expweight = hdr.get("EXPTIME", 1)

        flt_keys.append(key)
        wcs_dict[key] = wcs
        footprints[key] = Polygon(wcs.calc_footprint())

    return flt_keys, wcs_dict, footprints
