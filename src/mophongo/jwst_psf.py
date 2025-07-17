"""Utilities for extending JWST STDPSF grids with theoretical halos."""

from __future__ import annotations

import os
import math

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import STDPSFGrid, GriddedPSFModel

import stpsf

__all__ = ["make_extended_grid", "blend_psf"]


def blend_psf(
    emp_psf: np.ndarray,
    th_psf: np.ndarray,
    Rcore_px: int,
    Rmax_px: int,
    Rtaper_px: float,
) -> np.ndarray:
    """Blend empirical and theoretical PSFs.

    Parameters
    ----------
    emp_psf : ndarray
        Empirical PSF array.
    th_psf : ndarray
        Theoretical PSF array (assumed larger than the final size).
    Rcore_px : int
        Radius of the empirical core in oversampled pixels.
    Rmax_px : int
        Desired output radius in oversampled pixels.
    Rtaper_px : float
        Width of the blend region in oversampled pixels.

    Returns
    -------
    ndarray
        Normalised blended PSF of shape ``(2*Rmax_px+1, 2*Rmax_px+1)``.
    """
    y0_th = th_psf.shape[0] // 2
    x0_th = th_psf.shape[1] // 2
    out = th_psf[
        y0_th - Rmax_px : y0_th + Rmax_px + 1,
        x0_th - Rmax_px : x0_th + Rmax_px + 1,
    ].copy()

    Ny_emp = emp_psf.shape[0]
    y0_emp = Ny_emp // 2
    x0_emp = Ny_emp // 2
    yoff = Rmax_px - y0_emp
    xoff = yoff
    out[yoff : yoff + Ny_emp, xoff : xoff + Ny_emp] = emp_psf

    yy, xx = np.indices(out.shape)
    r = np.hypot(yy - Rmax_px, xx - Rmax_px)
    w = np.zeros_like(out)
    w[r <= Rcore_px] = 1.0
    m = (r > Rcore_px) & (r < Rcore_px + Rtaper_px)
    w[m] = 0.5 * (1 + np.cos(math.pi * (r[m] - Rcore_px) / Rtaper_px))

    th_crop = th_psf[
        y0_th - Rmax_px : y0_th + Rmax_px + 1,
        x0_th - Rmax_px : x0_th + Rmax_px + 1,
    ]
    out = w * out + (1 - w) * th_crop
    out /= out.sum()
    return out


def _extract_theory_data(hdul: fits.HDUList, n: int) -> np.ndarray:
    if "DET_SAMP" in hdul:
        data = hdul["DET_SAMP"].data
    else:
        data = np.stack([hdul[i + 1].data for i in range(n)], axis=0)
    return np.asarray(data)


def make_extended_grid(
    emp: str | STDPSFGrid,
    Rmax: float,
    *,
    Rtaper: float = 0.1,
    pixscale: float = 0.063,
) -> GriddedPSFModel:
    """Create an extended JWST PSF grid.

    Parameters
    ----------
    emp : str or STDPSFGrid
        Path to an STDPSF FITS file or an ``STDPSFGrid`` instance.
    Rmax : float
        Outer radius of the final PSF in arcsec.
    Rtaper : float, optional
        Width of the blending region in arcsec. Default is 0.1.
    pixscale : float, optional
        Detector pixel scale in arcsec/px. Defaults to ``0.063`` for
        the NIRCam long wavelength channel.

    Returns
    -------
    GriddedPSFModel
        New grid containing blended empirical cores and theoretical halos.
    """
    if isinstance(emp, (str, bytes, os.PathLike)):
        emp_grid = STDPSFGrid(emp)  # type: ignore[arg-type]
    else:
        emp_grid = emp

    oversamp = emp_grid.oversampling
    grid_xy = emp_grid.grid_xypos
    det_name = emp_grid.meta.get("detector", "NRC")
    filt_name = emp_grid.meta.get("filter", "F200W")
    Nemp, Ny_emp, _ = emp_grid.data.shape
    Rcore_px = (Ny_emp - 1) // 2

    nrc = stpsf.NIRCam()
    nrc.filter = filt_name
    nrc.detector = det_name

    th_raw = nrc.psf_grid(
        num_psfs=len(grid_xy),
        all_detectors=False,
        oversample=oversamp,
        fov_arcsec=2 * Rmax,
    )
    th_dat = _extract_theory_data(th_raw, len(grid_xy))

    Rmax_px = math.ceil(Rmax / (pixscale / oversamp))
    Rtaper_px = Rtaper / (pixscale / oversamp)
    Nfinal = 2 * Rmax_px + 1

    out_arr = np.empty((Nemp, Nfinal, Nfinal), dtype=float)
    for i in range(Nemp):
        out_arr[i] = blend_psf(
            emp_grid.data[i], th_dat[i], Rcore_px, Rmax_px, Rtaper_px
        )

    meta = {
        "grid_xypos": grid_xy,
        "oversampling": oversamp,
        "telescope": "JWST",
        "instrument": "NIRCam",
        "detector": det_name,
        "filter": filt_name,
        "grid_shape": emp_grid.meta.get("grid_shape"),
        "Rcore_px": Rcore_px,
        "Rmax_px": Rmax_px,
        "Rtaper_px": Rtaper_px,
        "note": "empirical STDPSF core + stpsf halo",
    }
    nd = NDData(out_arr, meta=meta)
    return GriddedPSFModel(nd)
