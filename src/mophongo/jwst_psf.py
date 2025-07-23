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

#e WebbPSF in in-flight sim mode, I need a fits file with several header keywords (e.g., INSTRUME, FILTER, DATE-OBS, TIME-OBS, APERNAME…).
#  In UDS F770W, I have found an i2d image file stored in the google drive (jw1837_miri_F770W_mosaic_v0p1_uds1_masked_sbkgsub_i2d.fits) 
# has this information in its header so I have used this file to produce the in-flight WebbPSF of UDS F770W, 


# ──────────────────────────────────────────────────────────
# 1.  detector token → (instrument, detector)
# ──────────────────────────────────────────────────────────
_PAT = {
    r'nrca[1-4]' : ("NIRCAM", lambda m: m.group(0).upper()),
    r'nrcb[1-4]' : ("NIRCAM", lambda m: m.group(0).upper()),
    r'nrcalong'  : ("NIRCAM", "NRCA5"),     # long-wave detectors
    r'nrcblong'  : ("NIRCAM", "NRCB5"),
    r'nrca5'     : ("NIRCAM", "NRCA5"),
    r'nrcb5'     : ("NIRCAM", "NRCB5"),
    r'nrs[12]'   : ("NIRSPEC", lambda m: m.group(0).upper()),
    r'nis'       : ("NIRISS",  "NIS"),
    r'mirimage'  : ("MIRI",    "MIRI"),
}

def _decode(fname: str):
    low = fname.lower()
    for pat, (inst, det) in _PAT.items():
        m = re.search(pat, low)
        if m:
            return inst, det(m) if callable(det) else det
    raise ValueError(f"Cannot decode detector from '{fname}'")

# ──────────────────────────────────────────────────────────
# 2.  filter from CSV file name (- or _ delimiters both ok)
# ──────────────────────────────────────────────────────────
import re, pathlib
_FILTER_TOKEN = re.compile(r'[-_]f\d{3,4}[a-z]\d?_?', re.IGNORECASE)

def _filter_from_csv_path(csv_path: str | pathlib.Path) -> str:
    m = _FILTER_TOKEN.search(str(csv_path).lower())
    if not m:
        raise ValueError(f"Filter token not found in '{csv_path}'")
    return m.group(0).lstrip('-_').rstrip('_').upper()   # → 'F444W', 'F115W2' …

# ──────────────────────────────────────────────────────────
# 3.  densest-window finder  (returns centre & mask)
# ──────────────────────────────────────────────────────────
import numpy as np
def _modal_mjd(arr, span=3.0):
    arr  = np.asarray(arr, float)
    sort = np.sort(arr)
    best_cnt, best_i = 0, 0
    for i, v in enumerate(sort):
        cnt = np.searchsorted(sort, v + span) - i
        if cnt > best_cnt:
            best_cnt, best_i = cnt, i
    lo   = sort[best_i]
    mask = (arr >= lo) & (arr <= lo + span)
    centre = lo + span / 2
    return centre, mask


# ----------------------------------------------------------------------
# accepted short-/long-wave NIRCam filters  (minimal examples – extend!)
# ----------------------------------------------------------------------
_NIRCAM_SW = {
    'F070W','F090W','F115W','F150W','F140M','F182M','F200W','F250M',
}
_NIRCAM_LW = {
    'F277W','F356W','F444W','F410M','F430M','F460M','F480M',
}

# ---------- public helper -------------------------------------------------
def psf_grid_from_csv(
    csv_path,
    *,
    detector   = None,          # ← NEW
    num_psfs   = 1,
    oversample = 4,
    fov_arcsec = None,
    span       = 5.0,           # days around the modal MJD
    save       = False,
    outdir     = None,
    use_detsampled_psf = False,
    prefix     = 'STDPSF',
    postfix    = '',
    verbose    = False,
    overwrite = False,
):
    """
    Build one or several STPSF PSF grids for the dominant epoch in *csv_path*.

    Parameters
    ----------
    detector : str or None
        • ``"NRCA1"``, ``"NRCB5"`` … → build only that detector  
        • ``None`` **and instrument = NIRCam** → build all SW or LW SCAs
          appropriate for the filter.
    Other parameters are forwarded to ``Instrument.psf_grid``; the routine
    always sets ``save=``, ``outdir=`` and an auto‐generated ``outfile`` for
    each grid.
    """
    import pandas as pd, re, pathlib, numpy as np
    from astropy.time import Time
    import stpsf

    # --- CSV → dominant row ------------------------------------------------
    tab = pd.read_csv(csv_path)
    if 'mjd-avg' not in tab.columns:
        raise ValueError("CSV must contain 'mjd-avg'")
    centre, mask = _modal_mjd(tab['mjd-avg'], span)
    row = tab[mask].iloc[0]

    inst_name, det_from_file = _decode(row['file'])
    filt = _filter_from_csv_path(csv_path)

    # --- decide list of detectors -----------------------------------------
    if detector:                                   # user forces one SCA
        det_list = [detector.upper()]
    elif inst_name == 'NIRCAM':
        if filt in _NIRCAM_LW:
            det_list = ['NRCA5', 'NRCB5']
        else:                                      # treat everything else as SW
            det_list = ['NRCA1','NRCA2','NRCA3','NRCA4',
                         'NRCB1','NRCB2','NRCB3','NRCB4']
    else:                                          # non-NIRCam: single detector
        det_list = [det_from_file]

    grids = []                                     # collect returned HDULists
    date_obs = Time(centre, format='mjd')

    if verbose:
        print(f"{inst_name}  filter={filt}  date={date_obs.isot[:10]}  "
              f"detectors={','.join(det_list)}  "
              f"({mask.sum()} files in ±{span/2:.1f} d)")

    # --- loop over detectors ----------------------------------------------
    for det in det_list:
        outfile = pathlib.Path('_'.join(filter(None, (prefix, det, filt, postfix))) + '.fits')
        print('OUTFILE:', outfile, )
        if (outdir / outfile).exists() and not overwrite:
            print(f"Skipping {outfile} (exists, overwrite={overwrite})")
            continue

        match inst_name:
            case 'NIRCAM':
                inst = stpsf.NIRCam();  inst.detector = det; inst.filter = filt
            case 'MIRI':
                inst = stpsf.MIRI(); inst.filter = filt
            case 'NIRISS':
                inst = stpsf.NIRISS();     inst.filter = filt
            case 'NIRSPEC':
                inst = stpsf.NIRSpec();    inst.filter = filt
            case _:
                raise RuntimeError(f"Unsupported instrument {inst_name}")

        inst.load_wss_opd_by_date(date_obs, choice='closest')
        inst.options['parity'] = 'odd'

        if save:
            os.makedirs(outdir, exist_ok=True)
        
        grid = inst.psf_grid(
            num_psfs      = num_psfs,
            oversample    = oversample,
            fov_arcsec    = fov_arcsec,
            all_detectors = False,
            use_detsampled_psf = use_detsampled_psf,
            verbose       = verbose,            
#            single_psf_centered = True,   
#            save       = False,
#            outdir     = None,
#            outfile    = outfile,
        )

        grid.meta['DATE-OBS'] = date_obs.isot
        grid.meta['MJD-AVG'] = centre
        grids.append(grid)

        if save:
            write_stdpsf(outdir / outfile, grid, overwrite=True, verbose=True)
        
    return grids




def blend_psf(
    core_psf: np.ndarray,
    ext_psf: np.ndarray,
    Rcore_px: int = 0,
    Rtaper_px: float = 1,
    Rnorm_px: float = 30,
    buf_px: int = 4,
    subtract_bg: bool = True,
    bg_pct: float = 15.0,      # Percentile for background subtraction        
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
    core_psf_out = core_psf
    
    core_psf_out = core_psf
    
    pos = np.asarray(ext_psf.shape) // 2
    ext_cutout = Cutout2D(ext_psf, position=pos, size=core_shape)
    ext_cutout_data = ext_cutout.data

    N = core_psf.shape[0]//2
    r = np.hypot(*np.indices(core_psf.shape) - N)  

    
    # Scaling for normalization in the blend region
    mask_norm = r <= min(Rnorm_px, N)
    scl_ext = core_psf_out[mask_norm].sum() / ext_cutout_data[mask_norm].sum()

    
    # Scaling for normalization in the blend region
    mask_norm = r <= min(Rnorm_px, N)
    scl_ext = core_psf_out[mask_norm].sum() / ext_cutout_data[mask_norm].sum()

    if subtract_bg:
        bgmask = ~(core_psf > np.nanpercentile(core_psf[core_psf > 0.0],bg_pct) ) & (r < N - buf_px)
        bgmask = ~(core_psf > np.nanpercentile(core_psf[core_psf > 0.0],bg_pct) ) & (r < N - buf_px)
        if np.any(bgmask):
            core_psf_out = core_psf - np.nanmedian((core_psf - ext_cutout_data)[bgmask])  
            print('percentile background core:', np.nanmedian((core_psf)[bgmask]))  
            print('percentile background extended:', np.nanmedian((ext_cutout_data)[bgmask]))  
            print('subtracting background -(core - extended):', np.nanmedian((core_psf - ext_cutout_data)[bgmask]))  
            core_psf_out = core_psf - np.nanmedian((core_psf - ext_cutout_data)[bgmask])  
            print('percentile background core:', np.nanmedian((core_psf)[bgmask]))  
            print('percentile background extended:', np.nanmedian((ext_cutout_data)[bgmask]))  
            print('subtracting background -(core - extended):', np.nanmedian((core_psf - ext_cutout_data)[bgmask]))  
   
    buf_px = int(buf_px)
    R_inner = min(Rcore_px, core_psf_out.shape[0] // 2 - buf_px)
    Rtaper_px = max(int(Rtaper_px), 1)  # ensure at least 1 pixel

    # Linear taper inward from R_inner
    w = np.ones_like(ext_cutout_data)
    annulus = (r > R_inner - Rtaper_px) & (r <= R_inner)
    w[annulus] = 1 - (r[annulus] - (R_inner - Rtaper_px)) / Rtaper_px
    w[r > R_inner] = 0.0
    print(f"R_inner: {R_inner}, Rtaper_px: {Rtaper_px}, R_norm: {Rnorm_px}, #pix in annulus: {np.sum(annulus)}")


    # Insert blended core into full halo (no extra scaling of full halo)
    blended = ext_psf.copy() * scl_ext
    
    # Blend only the ext_cutout region
    blend_core = w * core_psf_out + (1 - w) * ext_cutout_data * scl_ext
    blended[ext_cutout.slices_original] = np.maximum(blend_core, 0)

    # Normalize total sum to 1
#    blend_psf /= blend_psf.sum()
    if test:
        return blended, w, blend_core, ext_cutout_data, ext_cutout.slices_original
    return blended 

 
def make_extended_grid(
    emp: str | STDPSFGrid,
    Rmax: float,
    *,
    Rtaper: float = 0.2,
    Rnorm: float = 0.5,
    verbose: bool = False,
    subtract_bg=True,       # subtract dc offset from ePSF
    bg_pct: float = 15.0,    # Percentile for dc offset 
    return_stpsf: bool = True,
    subtract_bg=True,       # subtract dc offset from ePSF
    bg_pct: float = 15.0,    # Percentile for dc offset 
    return_stpsf: bool = True,
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

    st_grid = nrc.psf_grid(
    st_grid = nrc.psf_grid(
        num_psfs=Nemp,
        all_detectors=False,
        oversample=oversamp,
        fov_arcsec=2 * Rmax,
        verbose=verbose,
    )

    Rnorm_px = Rnorm / (nrc.pixelscale / oversamp)
    Rtaper_px = Rtaper / (nrc.pixelscale / oversamp)

    n_outpix = st_grid.data[0].shape[0]
    n_outpix = st_grid.data[0].shape[0]
    out_arr = np.empty((Nemp, n_outpix, n_outpix), dtype=float)
    print(out_arr.shape, emp_grid.data.shape, st_grid.data.shape)
    print(out_arr.shape, emp_grid.data.shape, st_grid.data.shape)
    for i in range(Nemp):
        out_arr[i] = blend_psf(
            emp_grid.data[i], st_grid.data[i], 
            Rcore_px, Rtaper_px = Rtaper_px, Rnorm_px = Rnorm_px, 
            subtract_bg=subtract_bg, bg_pct=bg_pct
            emp_grid.data[i], st_grid.data[i], 
            Rcore_px, Rtaper_px = Rtaper_px, Rnorm_px = Rnorm_px, 
            subtract_bg=subtract_bg, bg_pct=bg_pct
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
 
    gpm = GriddedPSFModel( NDData(out_arr, meta=meta))
    if return_stpsf:
        return gpm, GriddedPSFModel( NDData(st_grid.data, meta=meta))
    else:
        return gpm
import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.io import fits

import re
def _fits_key(name: str) -> str:
    key = re.sub(r'[^A-Z0-9-]', '', name.upper())[:8]
    return key if key and key[0].isalpha() else 'METAKEY'

# ─────────────────────────────────────────────────────────────────────
# Main writer
# ─────────────────────────────────────────────────────────────────────
def write_stdpsf(
    filename: str | Path,
    psf_grid=None,                     # NEW name; raw cube still accepted
    xgrid: np.ndarray | None = None,
    ygrid: np.ndarray | None = None,
    *,
    detector: str | None = None,
    filt: str | None = None,
    overwrite: bool = False,
    history: str | None = None,
    verbose: bool = False,
    filename: str | Path,
    psf_grid=None,                     # NEW name; raw cube still accepted
    xgrid: np.ndarray | None = None,
    ygrid: np.ndarray | None = None,
    *,
    detector: str | None = None,
    filt: str | None = None,
    overwrite: bool = False,
    history: str | None = None,
    verbose: bool = False,
):
    """
    Write a JWST “STDPSF” file.
    Write a JWST “STDPSF” file.

    Parameters
    ----------
    filename  :  str or Path
        Destination path.
    psf_grid  :  STDPSFGrid object **or** (N, Y, X) float32 array.
    xgrid/ygrid
        1-D arrays of detector-pixel centres for the PSF grid.
        *Ignored* when a STDPSFGrid is passed (taken from its meta).
    filename  :  str or Path
        Destination path.
    psf_grid  :  STDPSFGrid object **or** (N, Y, X) float32 array.
    xgrid/ygrid
        1-D arrays of detector-pixel centres for the PSF grid.
        *Ignored* when a STDPSFGrid is passed (taken from its meta).
    """

    # ───────────────────── accept either STDPSFGrid or raw cube ──────────
    if hasattr(psf_grid, "data") and hasattr(psf_grid, "meta"):
        grid_obj = psf_grid
        cube     = np.asarray(grid_obj.data,  dtype='float32')
        xgrid    = np.unique(grid_obj.grid_xypos[:, 0]).astype(int)
        ygrid    = np.unique(grid_obj.grid_xypos[:, 1]).astype(int)
        detector = detector or grid_obj.meta.get("detector")
        filt     = filt     or grid_obj.meta.get("filter")
        meta     = dict(grid_obj.meta)          # copy – we'll pop below
    else:
        cube  = np.asarray(psf_grid, dtype='float32')
        meta  = {}                              # nothing extra to copy

    if cube.ndim != 3:
        raise ValueError("psf_grid/cube must be a 3-D array (N, Y, X)")

    npsf, ny, nx = cube.shape

    # ───────────────────── accept either STDPSFGrid or raw cube ──────────
    if hasattr(psf_grid, "data") and hasattr(psf_grid, "meta"):
        grid_obj = psf_grid
        cube     = np.asarray(grid_obj.data,  dtype='float32')
        xgrid    = np.unique(grid_obj.grid_xypos[:, 0]).astype(int)
        ygrid    = np.unique(grid_obj.grid_xypos[:, 1]).astype(int)
        detector = detector or grid_obj.meta.get("detector")
        filt     = filt     or grid_obj.meta.get("filter")
        meta     = dict(grid_obj.meta)          # copy – we'll pop below
    else:
        cube  = np.asarray(psf_grid, dtype='float32')
        meta  = {}                              # nothing extra to copy

    if cube.ndim != 3:
        raise ValueError("psf_grid/cube must be a 3-D array (N, Y, X)")

    npsf, ny, nx = cube.shape
    xgrid = np.asarray(xgrid, dtype=int)
    ygrid = np.asarray(ygrid, dtype=int)
    if npsf != len(xgrid) * len(ygrid):
    if npsf != len(xgrid) * len(ygrid):
        raise ValueError(
            f"psf_grid.shape[0] ({npsf}) ≠ len(xgrid)*len(ygrid) "
            f"({len(xgrid)*len(ygrid)})"
            f"psf_grid.shape[0] ({npsf}) ≠ len(xgrid)*len(ygrid) "
            f"({len(xgrid)*len(ygrid)})"
        )

    # ───────────────────── primary HDU and required keywords ─────────────
    hdu  = fits.PrimaryHDU(cube)
    hdr  = hdu.header
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['NAXIS3'] = npsf
    hdr['NXPSFs'] = len(xgrid)
    hdr['NYPSFs'] = len(ygrid)
    # ───────────────────── primary HDU and required keywords ─────────────
    hdu  = fits.PrimaryHDU(cube)
    hdr  = hdu.header
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['NAXIS3'] = npsf
    hdr['NXPSFs'] = len(xgrid)
    hdr['NYPSFs'] = len(ygrid)

    # detector grid positions – store pixel numbers 1-indexed
    # detector grid positions – store pixel numbers 1-indexed
    for i, xv in enumerate(xgrid, 1):
        hdr[f'IPSFX{i:02d}'] = int(xv + 1)
        hdr[f'IPSFX{i:02d}'] = int(xv + 1)
    for i, yv in enumerate(ygrid, 1):
        hdr[f'JPSFY{i:02d}'] = int(yv + 1)
        hdr[f'JPSFY{i:02d}'] = int(yv + 1)

    # convenience keywords
    if detector:
        hdr['DETECTOR'] = detector
    if filt:
        hdr['FILTER']   = filt
    # convenience keywords
    if detector:
        hdr['DETECTOR'] = detector
    if filt:
        hdr['FILTER']   = filt

    # ───────────────── copy every meta entry into the header ──────────────
    for key, raw in meta.items():

        # skip if explicitly handled above
        if key.lower() in {'detector', 'filter'}:
            continue

        # unpack "(value, comment)" or fall back to plain value
        if isinstance(raw, tuple) and len(raw) >= 1:
            val     = raw[0]
            comment = raw[1] if len(raw) > 1 else ''
        else:
            val     = raw
            comment = f'From meta: {key}'

        # FITS keyword (≤8 chars, alnum only)
        kw = _fits_key(key)

        # truncate very long strings so they fit (FITS allows 68 chars)
        if isinstance(val, str):
            val = val[:68]

        try:
            hdr[kw] = (val, comment)
        except Exception:
            # fall back to string representation if value type not supported
            hdr[kw] = (str(val)[:68], comment)

    # ───────────────────── date / time / history ────────────────────────
    # ───────────────── copy every meta entry into the header ──────────────
    for key, raw in meta.items():

        # skip if explicitly handled above
        if key.lower() in {'detector', 'filter'}:
            continue

        # unpack "(value, comment)" or fall back to plain value
        if isinstance(raw, tuple) and len(raw) >= 1:
            val     = raw[0]
            comment = raw[1] if len(raw) > 1 else ''
        else:
            val     = raw
            comment = f'From meta: {key}'

        # FITS keyword (≤8 chars, alnum only)
        kw = _fits_key(key)

        # truncate very long strings so they fit (FITS allows 68 chars)
        if isinstance(val, str):
            val = val[:68]

        try:
            hdr[kw] = (val, comment)
        except Exception:
            # fall back to string representation if value type not supported
            hdr[kw] = (str(val)[:68], comment)

    # ───────────────────── date / time / history ────────────────────────
    now = datetime.utcnow()
    hdr['DATE'] = now.strftime('%Y-%m-%d')
    hdr['TIME'] = now.strftime('%H:%M:%S')
    hdr.add_history('File written by write_stdpsf')
    hdr['DATE'] = now.strftime('%Y-%m-%d')
    hdr['TIME'] = now.strftime('%H:%M:%S')
    hdr.add_history('File written by write_stdpsf')
    if history:
        for line in history.splitlines():
            hdr.add_history(line.strip())

    # ───────────────────── write file ────────────────────────────────────
    # ───────────────────── write file ────────────────────────────────────
    hdu.writeto(filename, overwrite=overwrite)
    if verbose:
        print(f"Wrote {npsf} PSFs ➜ {filename}")
        print(f"Wrote {npsf} PSFs ➜ {filename}")
