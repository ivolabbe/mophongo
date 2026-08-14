"""JWST PSF backend for :mod:`mophongo.psf_factory`.

Thin, readable wrappers around ``stpsf``:

* :func:`build_jwst_psf` -- build one PSF / PSF grid for an explicit
  instrument / detector / filter / date.
* :data:`jwst_backend` -- :class:`JWSTBackend` instance plugged into the
  telescope-agnostic :class:`mophongo.psf_factory.PSFFactory`.

Plus two PSF-shaping helpers retained from the original implementation:

* :func:`blend_psf` -- blend an empirical PSF core with a theoretical halo.
* :func:`make_extended_grid` -- apply :func:`blend_psf` to every position
  of an empirical ``STDPSFGrid``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import stpsf
from astropy.io import fits
from astropy.nddata import NDData
from astropy.time import Time
from photutils.psf import GriddedPSFModel, STDPSFGrid

logger = logging.getLogger(__name__)

__all__ = [
    "build_jwst_psf",
    "JWSTBackend",
    "jwst_backend",
    "make_extended_grid",
    "blend_psf",
    "write_stdpsf",
]


# ──────────────────────────────────────────────────────────────────────────
# Filename token decoders
# ──────────────────────────────────────────────────────────────────────────
# Detector token -> (instrument, canonical SCA). 'long' aliases listed first.
_DETECTOR_PATTERNS: list[tuple[str, str, object]] = [
    (r"nrcalong", "NIRCAM", "NRCA5"),
    (r"nrcblong", "NIRCAM", "NRCB5"),
    (r"nrca5", "NIRCAM", "NRCA5"),
    (r"nrcb5", "NIRCAM", "NRCB5"),
    (r"nrca[1-4]", "NIRCAM", lambda m: m.group(0).upper()),
    (r"nrcb[1-4]", "NIRCAM", lambda m: m.group(0).upper()),
    (r"nrs[12]", "NIRSPEC", lambda m: m.group(0).upper()),
    (r"nis\b", "NIRISS", "NIS"),
    (r"mirimage", "MIRI", "MIRI"),
]

_FILTER_TOKEN = re.compile(r"[-_](f\d{3,4}[a-z]\d?)(?:[-_]|$)", re.IGNORECASE)

# NIRCam channel partitioning. Anything not in LW is treated as SW.
_NIRCAM_LW = {
    "F250M", "F277W", "F300M", "F322W2", "F323N", "F335M", "F356W",
    "F360M", "F405N", "F410M", "F430M", "F444W", "F460M", "F466N",
    "F470N", "F480M",
}


def _decode_jwst_filename(name: str) -> tuple[str, str]:
    """Decode ``(instrument, detector)`` from a JWST rate-file name."""
    low = name.lower()
    for pat, inst, det in _DETECTOR_PATTERNS:
        m = re.search(pat, low)
        if m:
            return inst, det(m) if callable(det) else det
    raise ValueError(f"Cannot decode JWST detector from '{name}'")


def _filter_from_path(path: str | os.PathLike) -> str:
    """Extract the ``Fxxx[xM|W|N]`` filter token from a filename/path."""
    m = _FILTER_TOKEN.search(str(path).lower())
    if not m:
        raise ValueError(f"Filter token not found in '{path}'")
    return m.group(1).upper()


def _stpsf_instrument(instrument: str):
    inst = instrument.upper()
    if inst == "NIRCAM":
        return stpsf.NIRCam()
    if inst == "MIRI":
        return stpsf.MIRI()
    if inst == "NIRISS":
        return stpsf.NIRISS()
    if inst == "NIRSPEC":
        return stpsf.NIRSpec()
    raise ValueError(f"Unsupported JWST instrument: {instrument!r}")


# ──────────────────────────────────────────────────────────────────────────
# Low-level PSF builder
# ──────────────────────────────────────────────────────────────────────────
def build_jwst_psf(
    *,
    instrument: str,
    filter: str,
    detector: str | None = None,
    date: float | str | Time | None = None,
    num_psfs: int = 1,
    oversample: int = 4,
    fov_arcsec: float = 5.0,
    use_detsampled_psf: bool = False,
    parity: str = "odd",
    opd_choice: str = "closest",
    verbose: bool = False,
) -> GriddedPSFModel:
    """Build a single STPSF PSF / PSF grid.

    Parameters
    ----------
    instrument
        ``'NIRCAM'``, ``'MIRI'``, ``'NIRISS'``, or ``'NIRSPEC'``.
    filter
        Filter name e.g. ``'F444W'``.
    detector
        NIRCam SCA name (e.g. ``'NRCA5'``). Required for NIRCam; ignored
        for single-detector instruments.
    date
        MJD (float), ISO date string, or :class:`astropy.time.Time`. When
        given, the wavefront OPD nearest in time is loaded.
    num_psfs
        Total number of PSFs in the grid. Must be a perfect square
        (1, 4, 9, 16, 25, ...); ``stpsf`` lays them out as
        ``sqrt(num_psfs) x sqrt(num_psfs)`` across the detector.
    oversample
        Pixel-space oversampling factor used by ``stpsf``.
    fov_arcsec
        Field of view of each PSF in arcsec.
    use_detsampled_psf
        If True, return detector-sampled PSFs (``oversample`` is then the
        internal calculation factor only).
    parity
        ``'odd'`` (default) or ``'even'`` -- forwarded to ``inst.options``.
    opd_choice
        ``load_wss_opd_by_date`` selection mode.
    verbose
        Passed through to ``inst.psf_grid``.
    """
    inst = _stpsf_instrument(instrument)
    inst.filter = filter
    if instrument.upper() == "NIRCAM":
        if not detector:
            raise ValueError("NIRCam requires an explicit detector (e.g. 'NRCA5')")
        inst.detector = detector
    inst.options["parity"] = parity

    if date is not None:
        if isinstance(date, Time):
            date_obj = date
        elif isinstance(date, (int, float)):
            date_obj = Time(float(date), format="mjd")
        else:
            date_obj = Time(date)
        inst.load_wss_opd_by_date(date_obj, choice=opd_choice)
    else:
        date_obj = None

    # Unset means DEFAULT_FOV_PIXELS, asked for in pixels rather than left to
    # stpsf's fallback. Same grid either way today, but stating it keeps the
    # size in this repo: stpsf's fallback is a default like any other, and a
    # config that sets no field of view should still produce the same grids
    # next year. Passing fov_arcsec=None instead would raise -- stpsf divides
    # it by the pixel scale whenever the key is present.
    grid_kwargs = dict(
        num_psfs=num_psfs,
        oversample=oversample,
        all_detectors=False,
        use_detsampled_psf=use_detsampled_psf,
        verbose=verbose,
    )
    if fov_arcsec is not None:
        grid_kwargs["fov_arcsec"] = fov_arcsec
    else:
        grid_kwargs["fov_pixels"] = DEFAULT_FOV_PIXELS
    grid = inst.psf_grid(**grid_kwargs)
    if date_obj is not None:
        grid.meta["DATE-OBS"] = date_obj.isot
        grid.meta["MJD-AVG"] = float(date_obj.mjd)
    return grid


# ──────────────────────────────────────────────────────────────────────────
# Backend adapter
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class JWSTBackend:
    """JWST telescope backend for :class:`PSFFactory`."""

    name: str = "JWST"

    @staticmethod
    def detect(filename: str) -> bool:
        low = filename.lower()
        return any(re.search(p, low) for p, *_ in _DETECTOR_PATTERNS) or low.startswith("jw")

    @staticmethod
    def decode_filename(name: str) -> tuple[str, str]:
        return _decode_jwst_filename(name)

    @staticmethod
    def filter_from_path(path: str | os.PathLike) -> str:
        return _filter_from_path(path)

    @staticmethod
    def detectors_for_filter(instrument: str, filt: str) -> list[str]:
        if instrument.upper() != "NIRCAM":
            return []  # caller falls back to detector decoded from the CSV
        if filt.upper() in _NIRCAM_LW:
            return ["NRCA5", "NRCB5"]
        return [f"NRC{ab}{i}" for ab in "AB" for i in range(1, 5)]

    @staticmethod
    def build(**kwargs) -> GriddedPSFModel:
        return build_jwst_psf(**kwargs)


jwst_backend = JWSTBackend()


# ──────────────────────────────────────────────────────────────────────────
# Empirical-core / theoretical-halo blending
# ──────────────────────────────────────────────────────────────────────────
def blend_psf(
    core_psf: np.ndarray,
    ext_psf: np.ndarray,
    Rcore_px: int = 0,
    Rtaper_px: float = 1,
    Rnorm_px: float = 30,
    buf_px: int = 4,
    subtract_bg: bool = True,
    bg_pct: float = 15.0,
    *,
    test: bool = False,
) -> np.ndarray:
    """Blend empirical core with theoretical halo.

    Inserts ``core_psf`` into the centre of ``ext_psf`` with a linear taper
    of width ``Rtaper_px`` inward from ``Rcore_px``. The halo is rescaled so
    its enclosed flux inside ``Rnorm_px`` matches that of the core. When
    ``subtract_bg``, a DC offset estimated in an outer annulus (pixels below
    the ``bg_pct`` percentile, excluding a ``buf_px`` edge) is removed from
    the core before blending.
    """
    from astropy.nddata import Cutout2D

    core_shape = core_psf.shape
    pos = np.asarray(ext_psf.shape) // 2
    ext_cutout = Cutout2D(ext_psf, position=pos, size=core_shape)
    ext_cutout_data = ext_cutout.data

    N = core_psf.shape[0] // 2
    r = np.hypot(*np.indices(core_psf.shape) - N)

    mask_norm = r <= min(Rnorm_px, N)
    scl_ext = core_psf[mask_norm].sum() / ext_cutout_data[mask_norm].sum()

    core_psf_out = core_psf
    if subtract_bg:
        bgmask = ~(core_psf > np.nanpercentile(core_psf[core_psf > 0.0], bg_pct)) & (r < N - buf_px)
        if np.any(bgmask):
            offset = np.nanmedian((core_psf - ext_cutout_data)[bgmask])
            core_psf_out = core_psf - offset
            logger.debug("blend_psf: bg offset (core-ext) = %.3g", offset)

    buf_px = int(buf_px)
    R_inner = min(Rcore_px, core_psf_out.shape[0] // 2 - buf_px)
    Rtaper_px = max(int(Rtaper_px), 1)

    w = np.ones_like(ext_cutout_data)
    annulus = (r > R_inner - Rtaper_px) & (r <= R_inner)
    w[annulus] = 1 - (r[annulus] - (R_inner - Rtaper_px)) / Rtaper_px
    w[r > R_inner] = 0.0

    blended = ext_psf.copy() * scl_ext
    blend_core = w * core_psf_out + (1 - w) * ext_cutout_data * scl_ext
    blended[ext_cutout.slices_original] = np.maximum(blend_core, 0)

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
    subtract_bg: bool = True,
    bg_pct: float = 15.0,
    return_stpsf: bool = True,
    test: bool = False,
) -> GriddedPSFModel:
    """Extend an empirical PSF grid with theoretical wings from ``stpsf``.

    Parameters
    ----------
    emp
        Path to an STDPSF FITS file or an :class:`STDPSFGrid` instance.
    Rmax
        Outer radius of the final PSF in arcsec.
    Rtaper, Rnorm
        Blend taper width and normalisation radius in arcsec.
    subtract_bg, bg_pct
        Background subtraction control (see :func:`blend_psf`).
    return_stpsf
        If True (default), also return the bare theoretical halo grid.
    """
    if isinstance(emp, (str, bytes, os.PathLike)):
        emp_grid = STDPSFGrid(emp)
    else:
        emp_grid = emp

    if emp_grid.meta["detector"][-1] == "L":  # 'NRCALONG' -> 'NRCA5'
        emp_grid.meta["detector"] = emp_grid.meta["detector"][:-1] + "5"

    oversamp = emp_grid.oversampling[0]
    grid_xy = emp_grid.grid_xypos
    det_name = emp_grid.meta.get("detector", "NRC")
    filt_name = emp_grid.meta.get("filter", "F200W")
    Nemp, Ny_emp, _ = emp_grid.data.shape
    Rcore_px = (Ny_emp - 1) // 2

    nrc = stpsf.NIRCam()
    nrc.filter = filt_name
    nrc.detector = det_name
    nrc.options["parity"] = "odd"

    if test:
        grid_xy = np.array([[0, 0]])
        Nemp = 1

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
    out_arr = np.empty((Nemp, n_outpix, n_outpix), dtype=float)
    for i in range(Nemp):
        out_arr[i] = blend_psf(
            emp_grid.data[i], st_grid.data[i],
            Rcore_px, Rtaper_px=Rtaper_px, Rnorm_px=Rnorm_px,
            subtract_bg=subtract_bg, bg_pct=bg_pct,
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

    gpm = GriddedPSFModel(NDData(out_arr, meta=meta))
    if return_stpsf:
        return gpm, GriddedPSFModel(NDData(st_grid.data, meta=meta))
    return gpm


# ──────────────────────────────────────────────────────────────────────────
# STDPSF-format FITS writer
# ──────────────────────────────────────────────────────────────────────────
#: Grid width in native detector pixels when ``fov_arcsec`` is not given.
#: This is ``stpsf``'s own default (``CreatePSFLibrary`` falls back to 101 when
#: neither ``fov_pixels`` nor ``fov_arcsec`` is passed), stated here rather
#: than left implicit so the filename can carry the field of view a grid was
#: actually built at, and so a future change to that default cannot silently
#: change what the grids contain. 101 is odd, which keeps the PSF centred on a
#: pixel; see ``parity`` in :func:`build_jwst_psf`.
DEFAULT_FOV_PIXELS = 101

#: Native pixel scales [arcsec] used when ``stpsf`` cannot be queried (no
#: reference data installed). Only a fallback: the live instrument model is
#: asked first, so these cannot drift into the filenames unnoticed.
_FALLBACK_PIXELSCALE = {"NIRCAM_LW": 0.06290713, "NIRCAM_SW": 0.03122585,
                        "MIRI": 0.110917025}


@lru_cache(maxsize=None)
def default_fov_arcsec(detector_or_instrument: str) -> float | None:
    """FOV of a grid built without an explicit ``fov_arcsec``, in arcsec.

    :data:`DEFAULT_FOV_PIXELS` native pixels wide, so the answer is per
    detector rather than per instrument: the NIRCam long-wave SCAs are 63
    mas/pixel and the short-wave ones 31, MIRI 111. Returns ``None`` for a
    name that names neither.
    """
    key = str(detector_or_instrument or "").upper()
    if key.startswith("NRC") or key == "NIRCAM":
        try:
            inst = stpsf.NIRCam()
            if key.startswith("NRC"):
                inst.detector = key
            scale = float(inst.pixelscale)
        except Exception:  # noqa: BLE001 - no stpsf reference data
            long_wave = key.endswith("5") or key.endswith("LONG")
            scale = _FALLBACK_PIXELSCALE["NIRCAM_LW" if long_wave else "NIRCAM_SW"]
    elif key.startswith("MIRI"):
        try:
            scale = float(stpsf.MIRI().pixelscale)
        except Exception:  # noqa: BLE001 - no stpsf reference data
            scale = _FALLBACK_PIXELSCALE["MIRI"]
    else:
        return None
    return DEFAULT_FOV_PIXELS * scale


def fov_agnostic_pattern(pattern: str) -> str:
    """Let a pattern without an ``_FOV`` token match filenames that have one.

    Grids are now named with the field of view always present
    (``..._MJD60308_FOV4_GRID25_OS4``), so FOV4 and FOV8 sets can share a
    directory instead of needing one each. Patterns written before that -- in
    every config generated so far -- have no ``_FOV`` and would match none of
    the new files. Inserting an optional token keeps them working.

    A pattern that already names an FOV is returned unchanged: it is being
    specific on purpose, and the 30" halo grids depend on that.

    The relaxation is deliberately loose, and can match a grid built at another
    field of view with the same GRID/OS layout. That is the ambiguity the token
    exists to remove, so new configs should name the FOV they want; this is for
    reading what is already on disk.
    """
    text = str(pattern)
    if "_FOV" in text or "_GRID" not in text:
        return text
    return text.replace("_GRID", r"(?:_FOV\d+)?_GRID", 1)


def _fits_key(name: str) -> str:
    key = re.sub(r"[^A-Z0-9-]", "", name.upper())[:8]
    return key if key and key[0].isalpha() else "METAKEY"


def read_stdpsf_provenance(path: str | Path) -> dict[str, str]:
    """Return the ``HIERARCH MPH *`` cards of a grid file as a plain dict.

    Empty when the file predates provenance stamping, which is not the same
    as matching: an unstamped grid cannot be shown to agree with anything.
    """
    from astropy.io import fits

    try:
        hdr = fits.getheader(path)
    except OSError:
        return {}
    out = {}
    for key in hdr:
        if key.startswith("MPH "):
            out[key[4:].strip().lower()] = str(hdr[key]).strip()
    return out


def write_stdpsf(
    filename: str | Path,
    psf_grid=None,
    xgrid: np.ndarray | None = None,
    ygrid: np.ndarray | None = None,
    *,
    detector: str | None = None,
    filt: str | None = None,
    overwrite: bool = False,
    history: str | None = None,
    provenance: dict | None = None,
    verbose: bool = False,
):
    """Write a JWST STDPSF-format FITS file.

    Parameters
    ----------
    filename
        Destination path.
    psf_grid
        :class:`STDPSFGrid` / :class:`GriddedPSFModel` instance **or** raw
        ``(N, Y, X)`` float array.
    xgrid, ygrid
        1-D arrays of detector-pixel centres for the PSF grid. Ignored when
        a grid object is passed (taken from its ``grid_xypos``).
    detector, filt
        Override values written to ``DETECTOR`` / ``FILTER`` keywords. If
        omitted they are taken from ``psf_grid.meta``.
    provenance
        What this grid was built from, written as ``HIERARCH MPH <KEY>``
        cards and read back by :func:`read_stdpsf_provenance`. The grid
        filename records the detector, filter, MJD, grid size and
        oversampling, but not the exposure list it was derived from nor the
        date mode that chose the MJDs — so a grid built one epoch per band
        is indistinguishable on disk from one built per epoch, and a cache
        of the first kind is silently reused forever. These cards are what
        makes that detectable.
    """
    if hasattr(psf_grid, "data") and hasattr(psf_grid, "meta"):
        cube = np.asarray(psf_grid.data, dtype="float32")
        xgrid = np.unique(psf_grid.grid_xypos[:, 0]).astype(int)
        ygrid = np.unique(psf_grid.grid_xypos[:, 1]).astype(int)
        detector = detector or psf_grid.meta.get("detector")
        filt = filt or psf_grid.meta.get("filter")
        meta = dict(psf_grid.meta)
    else:
        cube = np.asarray(psf_grid, dtype="float32")
        meta = {}

    if cube.ndim != 3:
        raise ValueError("psf_grid/cube must be a 3-D array (N, Y, X)")
    npsf, ny, nx = cube.shape
    if npsf != len(xgrid) * len(ygrid):
        raise ValueError(
            f"psf_grid.shape[0] ({npsf}) != len(xgrid)*len(ygrid) "
            f"({len(xgrid) * len(ygrid)})"
        )

    hdu = fits.PrimaryHDU(cube)
    hdr = hdu.header
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = npsf
    hdr["NXPSFs"] = len(xgrid)
    hdr["NYPSFs"] = len(ygrid)
    for i, xv in enumerate(xgrid, 1):
        hdr[f"IPSFX{i:02d}"] = int(xv + 1)
    for i, yv in enumerate(ygrid, 1):
        hdr[f"JPSFY{i:02d}"] = int(yv + 1)
    if detector:
        hdr["DETECTOR"] = detector
    if filt:
        hdr["FILTER"] = filt

    for key, raw in meta.items():
        if key.lower() in {"detector", "filter"}:
            continue
        if isinstance(raw, tuple) and len(raw) >= 1:
            val = raw[0]
            comment = raw[1] if len(raw) > 1 else ""
        else:
            val, comment = raw, f"From meta: {key}"
        kw = _fits_key(key)
        if isinstance(val, str):
            val = val[:68]
        try:
            hdr[kw] = (val, comment)
        except Exception:
            hdr[kw] = (str(val)[:68], comment)

    for key, val in (provenance or {}).items():
        card = f"HIERARCH MPH {str(key).upper()}"
        hdr[card] = val if isinstance(val, (int, float)) else str(val)[:60]

    now = datetime.now(timezone.utc)
    hdr["DATE"] = now.strftime("%Y-%m-%d")
    hdr["TIME"] = now.strftime("%H:%M:%S")
    hdr.add_history("File written by write_stdpsf")
    if history:
        for line in history.splitlines():
            hdr.add_history(line.strip())

    hdu.writeto(filename, overwrite=overwrite)
    if verbose:
        logger.info("Wrote %d PSFs -> %s", npsf, filename)
