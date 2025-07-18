"""Utilities for extending JWST STDPSF grids with theoretical halos."""

from __future__ import annotations

import os
import math

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from photutils.psf import STDPSFGrid, GriddedPSFModel
from datetime import datetime

import stpsf

__all__ = ["make_extended_grid", "blend_psf"]


def blend_psf(
    core_psf: np.ndarray,
    ext_psf: np.ndarray,
    Rcore_px: int = 0,
    Rtaper_px: float = 1,
    Rnorm_px: float = 30,
    buf_px: int = 4,
    subtract_bg: bool = True,
    *,
    test: bool = False,
) -> np.ndarray:
    """
    Blend empirical and theoretical PSFs with smooth transition and core normalization.
    Uses a linear taper inward from R_inner.

    Returns
    -------
    ndarray
        Normalised blended PSF.
    """
    from astropy.nddata import Cutout2D

    core_shape = core_psf.shape
    pos = np.asarray(ext_psf.shape) // 2
    ext_cutout = Cutout2D(ext_psf, position=pos, size=core_shape)
    ext_cutout_data = ext_cutout.data

    N = core_psf.shape[0]//2
    r = np.hypot(*np.indices(core_psf.shape) - N)  

    core_psf_out = core_psf
    if subtract_bg:
        bgmask = ~(core_psf > np.nanpercentile(core_psf[core_psf > 0.0],10) ) & (r < N - buf_px)
        if np.any(bgmask):
            core_psf_out = core_psf - np.nanmedian(core_psf[bgmask])
            print('subtracting background:', np.nanmedian(core_psf[bgmask]))
   
    buf_px = int(buf_px)
    R_inner = min(Rcore_px, core_psf_out.shape[0] // 2 - buf_px)
    Rtaper_px = max(int(Rtaper_px), 1)  # ensure at least 1 pixel

    # Linear taper inward from R_inner
    w = np.ones_like(ext_cutout_data)
    annulus = (r > R_inner - Rtaper_px) & (r <= R_inner)
    w[annulus] = 1 - (r[annulus] - (R_inner - Rtaper_px)) / Rtaper_px
    w[r > R_inner] = 0.0
    print(f"R_inner: {R_inner}, Rtaper_px: {Rtaper_px}, R_norm: {Rnorm_px}, #pix in annulus: {np.sum(annulus)}")

    # Scaling for normalization in the blend region
    mask_norm = r <= Rnorm_px
    scl_ext = core_psf_out[mask_norm].sum() / ext_cutout_data[mask_norm].sum()

    # Insert blended core into full halo (no extra scaling of full halo)
    blend_psf = ext_psf.copy() * scl_ext
    
    # Blend only the ext_cutout region
    blend_core = w * core_psf_out + (1 - w) * ext_cutout_data * scl_ext
    blend_psf[ext_cutout.slices_original] = np.maximum(blend_core, 0)

    # Normalize total sum to 1
#    blend_psf /= blend_psf.sum()
    if test:
        return blend_psf, w, blend_core, ext_cutout_data, ext_cutout.slices_original
    return blend_psf 

 
def make_extended_grid(
    emp: str | STDPSFGrid,
    Rmax: float,
    *,
    Rtaper: float = 0.2,
    Rnorm: float = 0.5,
    verbose: bool = False,
    test: bool = False,
) -> GriddedPSFModel:
    """Create an extended JWST PSF grid.

    Parameters
    ----------
    emp : str or STDPSFGrid
        Path to an STDPSF FITS file or an ``STDPSFGrid`` instance.
    Rmax : float
        Outer radius of the final PSF in arcsec.
    Rtaper : float, optional
        Width of the blending region in arcsec. Default is 0.2.
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

    # Ensure the detector name is compatible with stpsf
    if emp_grid.meta['detector'][-1] == 'L': 
        emp_grid.meta['detector'] = emp_grid.meta['detector'][:-1] + '5'
   
    oversamp = emp_grid.oversampling[0]
    grid_xy = emp_grid.grid_xypos
    det_name = emp_grid.meta.get("detector", "NRC")
    filt_name = emp_grid.meta.get("filter", "F200W")
    Nemp, Ny_emp, _ = emp_grid.data.shape
    Rcore_px = (Ny_emp - 1) // 2

    nrc = stpsf.NIRCam()
    nrc.filter = filt_name
    nrc.detector = det_name
    nrc.options['parity'] = 'odd'
    
    if test:
        grid_xy = np.array([[0, 0]])
        Nemp = 1

    theo_grid = nrc.psf_grid(
        num_psfs=Nemp,
        all_detectors=False,
        oversample=oversamp,
        fov_arcsec=2 * Rmax,
        verbose=verbose,
    )

#    if test:
#        return theo_grid

    Rnorm_px = Rnorm / (nrc.pixelscale / oversamp)
    Rtaper_px = Rtaper / (nrc.pixelscale / oversamp)

    n_outpix = theo_grid.data[0].shape[0]
    out_arr = np.empty((Nemp, n_outpix, n_outpix), dtype=float)
    print(out_arr.shape, emp_grid.data.shape, theo_grid.data.shape)
    for i in range(Nemp):
        out_arr[i] = blend_psf(
            emp_grid.data[i], theo_grid.data[i], Rcore_px, Rtaper_px = Rtaper_px, Rnorm_px = Rnorm_px
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
        "Rtaper_px": Rtaper_px,
        "Rmax_as": Rmax,
        "Rnorm_as": Rnorm,
        "note": "empirical STDPSF core + stpsf halo",
        "pixscale": nrc.pixelscale,
    }
    nd = NDData(out_arr, meta=meta)
    return GriddedPSFModel(nd)

def write_stdpsf(
        filename: str,
        psf_cube: np.ndarray = None,
        xgrid: np.ndarray = None,
        ygrid: np.ndarray = None,
        *,
        detector: str | None = None,
        filt: str | None = None,
        overwrite: bool = False,
        history: str | None = None,
        verbose: bool = False,
):
    """
    Write an STScI-standard ePSF FITS (“STDPSF”) file.

    Parameters
    ----------
    filename : str
        Output path.
    psf_cube : (Npsf, Ny, Nx) `~numpy.ndarray` or STDPSFGrid
        Stack of oversampled ePSFs ordered (row-major) the same way
        `STDPSFGrid` expects: first varying Y, then X. If a STDPSFGrid is passed,
        all info is extracted from it.
    xgrid, ygrid : 1-D `~numpy.ndarray`
        Detector X and Y pixel *centres* (0-indexed) of the grid nodes,
        length NXPSFs and NYPSFs respectively.
    detector, filt : str, optional
        Written to the header for convenience (not used by the reader).
    overwrite : bool, optional
        Pass ``True`` to clobber an existing file.
    history : str, optional
        Extra text to place in HISTORY cards.
    """
    # -------- handle STDPSFGrid input ---------------------------------------
    if hasattr(psf_cube, "data") and hasattr(psf_cube, "meta"):
        emp_grid = psf_cube
        psf_cube = emp_grid.data
        grid_xy = emp_grid.grid_xypos
        # Extract sorted unique x/y grid positions
        xgrid = np.unique(grid_xy[:, 0])
        ygrid = np.unique(grid_xy[:, 1])
        detector = emp_grid.meta.get("detector", detector)
        filt = emp_grid.meta.get("filter", filt)

    psf_cube = np.asarray(psf_cube, dtype="float32")
    if psf_cube.ndim != 3:
        raise ValueError("psf_cube must be a 3-D array (Npsf, Ny, Nx)")
    npsf, ny, nx = psf_cube.shape

    xgrid = np.asarray(xgrid, dtype=int)
    ygrid = np.asarray(ygrid, dtype=int)

    nxpsfs = len(xgrid)
    nypsfs = len(ygrid)
    if npsf != nxpsfs * nypsfs:
        raise ValueError(
            "psf_cube.shape[0] must equal len(xgrid) × len(ygrid) "
            f"({npsf} ≠ {nxpsfs*nypsfs})"
        )

    # -------- build the primary HDU -----------------------------------------
    hdu = fits.PrimaryHDU(psf_cube)

    # Mandatory keywords expected by _read_stdpsf
    hdr = hdu.header
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = npsf
    hdr["NXPSFs"] = nxpsfs
    hdr["NYPSFs"] = nypsfs

    # The original spec stores 1-indexed pixel numbers ➜ add 1 here
    for i, xv in enumerate(xgrid, 1):
        hdr[f"IPSFX{i:02d}"] = int(xv + 1)
    for i, yv in enumerate(ygrid, 1):
        hdr[f"JPSFY{i:02d}"] = int(yv + 1)

    # Optional convenience keywords
    if detector is not None:
        hdr["DETECTOR"] = detector
    if filt is not None:
        hdr["FILTER"] = filt

    # Date/time stamp (matches style of existing libraries)
    now = datetime.utcnow()
    hdr["DATE"] = now.strftime("%Y-%m-%d")
    hdr["TIME"] = now.strftime("%H:%M:%S")

    # Optional HISTORY lines
    hdr.add_history("File written by write_stdpsf()")
    if history:
        for line in history.splitlines():
            hdr.add_history(line.strip())

    # -------- write to disk --------------------------------------------------
    hdu.writeto(filename, overwrite=overwrite)

    if verbose:
        print(f"Wrote {npsf} PSFs to {filename}")
        print(f"Grid: {nxpsfs} × {nypsfs} positions")
        print(f"Detector: {detector}, Filter: {filt}")
        print(f"Pixel grid X: {xgrid}")
        print(f"Pixel grid Y: {ygrid}")
