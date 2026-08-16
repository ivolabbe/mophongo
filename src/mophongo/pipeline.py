"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

import json
import os
import re
import warnings
import psutil
import h5py
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence
from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict
from collections.abc import Sequence as _SequenceABC

from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, block_replicate, block_reduce
from astropy.stats import mad_std
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs.utils import proj_plane_pixel_scales

from .psf_map import PSFRegionMap
from .utils import as_label_array, bin_factor_from_wcs, downsample_psf, bin_remap
from .templates import EXTEND_MODE_ALIASES, EXTEND_MODES, Templates, Template, _slices_from_bbox
from . import template_schemes
from .fit import FitConfig as _FitConfig
from .scene import generate_scenes

import logging

logger = logging.getLogger(__name__)

memory = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9


def _fmt_hms(seconds: float) -> str:
    """``1h02m03s`` / ``2m03s`` / ``3.4s`` -- whichever fits the magnitude."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, sec = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def human_bytes(n: float, binary: bool = True) -> str:
    """Format a byte count with a unit that keeps it readable.

    Run logs quote sizes spanning kilobytes (one PSF stamp) to tens of
    gigabytes (a full-field stamps file), and a fixed unit makes one end or
    the other unreadable -- "12005.8 MB" is a number nobody parses at a
    glance. Binary units by default, since these are memory and array sizes.

    Args:
        n: Size in bytes.
        binary: Use KiB/MiB/GiB (1024) rather than kB/MB/GB (1000).

    Returns:
        The size and its unit, e.g. ``"11.7 GiB"``.
    """
    step = 1024.0 if binary else 1000.0
    units = ("B", "KiB", "MiB", "GiB", "TiB") if binary else ("B", "kB", "MB", "GB", "TB")
    size = float(n)
    for unit in units[:-1]:
        if abs(size) < step:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} {units[-1]}"


def _ignore_hierarch_warnings() -> None:
    """Install the filter that silences astropy's per-card HIERARCH warning.

    Keywords longer than eight characters -- ``PHOT_UNIT``, ``WEBBSTARFILT``,
    ``APER_DIAM``, ``SHRINK_FACTOR`` and the rest of what an input catalog
    carries in ``Table.meta`` -- round-trip as HIERARCH cards by design.
    Astropy warns once per card, and twice per card once its own warning
    logging has a handler, which at run scale is pure noise.

    The caller is responsible for restoring ``warnings.filters``; use
    :func:`_quiet_hierarch_warnings` unless the filter state is already being
    saved (:meth:`Pipeline.log_run`).
    """
    from astropy.io.fits.verify import VerifyWarning

    warnings.filterwarnings(
        "ignore",
        category=VerifyWarning,
        message=".*a HIERARCH card will be created.*",
    )


@contextmanager
def _quiet_hierarch_warnings():
    """Scoped form of :func:`_ignore_hierarch_warnings` for a write path.

    ``log_run`` filters these for everything it wraps, which since 2026-08-13
    is every CLI invocation. It is not the only way in: a notebook or script
    driving :class:`Pipeline` directly never enters that block, and the filter
    belongs with the writes that provoke the warning in any case.
    """
    with warnings.catch_warnings():
        _ignore_hierarch_warnings()
        yield


# shift-field arrows span this fraction of a scene's template extent, so that
# the outermost samples stay inside the scene rather than sitting on its edge
_SHIFT_SAMPLE_SPREAD = 0.7


@dataclass
class RunConfig:
    """Config-file inputs for one filter fit (one hi-res + one lo-res band).

    Loaded from JSON (``#``-comment lines allowed) via :meth:`from_json` and
    consumed by :meth:`Pipeline.from_config`. Unknown keys raise, so typos
    fail loudly.
    """

    # run label; prefixes every output file
    name: str
    # output directory: products + psf/kernel geojson caches (never inputs)
    out_dir: str
    # --- inputs -----------------------------------------------------------
    sci_hi: str  # high-resolution template image (FITS)
    segmap: str  # segmentation map on the hi-res grid (labels = catalog ids)
    catalog: str  # source catalog with id, x, y (hi-res pixels), ra, dec
    sci_lo: str  # low-resolution science mosaic to fit
    wht_lo: str  # low-resolution weight map
    csv_hi: str  # per-frame WCS csv of the hi-res mosaic
    csv_lo: str  # per-frame WCS csv of the lo-res mosaic
    # High-resolution weight map: the detection-band counterpart of wht_lo, and
    # what turns the detection image into a calibrated inverse variance. Every
    # run must have one, so the path is resolved (and its absence raised) on
    # every run; the pixels are read only when the selected extend_mode uses
    # them, since a full-field hi-res weight map costs as much memory as the
    # mosaic itself. None means "derive from sci_hi by the standard
    # '_sci' -> '_wht' naming" (see :meth:`Pipeline.resolve_wht_hi`).
    wht_hi: str | None = None
    # --- PSFs -------------------------------------------------------------
    psf_dir: str = "data/PSF"
    pattern_hi: str = ""  # STDPSF filename regex for the hi-res band
    pattern_lo: str = ""  # STDPSF filename regex for the lo-res band
    filter_lo: str = ""  # lo-res filter name, e.g. "f770w" (blur lookup)
    # PSF stamp size in arcsec; None = full native ePSF stamp as generated
    psf_size: float | None = 4.0
    psf_autobuild: bool = True  # generate missing PSF grids with PSFFactory
    # What to do when the exposure list implies epochs that no grid provides
    # and `psf_autobuild` is off (see Pipeline._missing_psf_dates). With
    # autobuild on -- the default -- the missing epochs are simply built and
    # this never applies.
    #   "warn"  - name the missing epochs and fit with the ones that exist
    #   "error" - refuse to run
    #   "off"   - do not look; load whatever matches the pattern
    # Grids themselves carry no provenance: a grid is its detector, filter,
    # epoch and field of view, all of them in its filename, and none of them
    # a function of the rest of the exposure list. Adding a frame to that list
    # leaves every existing grid correct and asks for at most one more.
    psf_provenance: str = "warn"
    # Processes used when building ePSF grids. One (detector, date) grid is an
    # independent job of tens to low hundreds of MB, so this scales with cores
    # rather than with memory. It parallelises within one pattern only: bands
    # of a field share pattern_hi, so two bands building at once still race on
    # the same F444W filenames -- which is why a campaign builds one band of a
    # field first and fans out afterwards (see docs/campaigns.md).
    psf_workers: int = 1
    psf_fov_arcsec: float | None = None  # PSFFactory field of view; None = backend default
    # Which exposure dates get their own grid when autobuilding. "all" (one
    # per unique integer MJD) is the default because the grids are MJD-tagged
    # and looked up by nearest date: collapsing an exposure list that spans
    # years onto one date ("modal") or a few ("cluster") silently discards
    # that. See psf_factory.dates_from_csv for the full set of modes.
    psf_date_mode: str | float = "all"
    # extra Gaussian broadening of the lo-res model PSF (FWHM arcsec);
    # "default" = mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC[filter_lo],
    # a number = that value, None = no broadening
    psf_blur_fwhm: float | str | None = "default"
    # optional [n_frames_hi, n_frames_lo] sanity assert on the WCS csvs
    expect_frames: list[int] | None = None
    # --- templates --------------------------------------------------------
    # The template build scheme is selected by ``fit["extend_mode"]``
    # (FitConfig.extend_mode): 'default' -> 'psf_wings'. Without an extension
    # the total flux is biased low, badly so for faint sources.
    # --- preprocessing ----------------------------------------------------
    # In-memory saturated-star repair at load time: fill wht=0 cores in
    # sci_hi/wht_hi with the fitted PSF model and flag the star-dominated
    # segments in the catalog (FLAG_SATURATED_TMPL group ids; one scene per
    # star). Nothing on disk changes; diagnostics go to out_dir/repaired/.
    repair_saturated: bool = False
    # extra kwargs for mophongo.repair.repair_in_memory (e.g. min_buffer_snr,
    # flux_frac, min_snr, stamp_npix)
    repair_kwargs: dict[str, Any] = field(default_factory=dict)
    # Large-FOV STDPSF pattern for the repair's halo model (full halo +
    # diffraction spikes, so segments far from the core can be flagged).
    # Empty (default) derives '{prefix}_{det}_{filt}_MJD\d+_FOV30_GRID1_OS4'
    # from pattern_hi and, with psf_autobuild, generates the grids once
    # (~minutes; cached in psf_dir). The core fit always keeps the
    # MJD-matched pattern_hi ePSFs; the halo model is grafted outside
    # their support.
    repair_psf_pattern: str = ""
    # Reuse a previous run's repair from the cache file when its recorded
    # inputs (sci_hi/wht_hi paths + mtimes, patterns, repair_kwargs) match;
    # the PSF fit is then skipped and the cached pixel patches and catalog
    # flags are applied instead.
    repair_reuse: bool = True
    # Cache location. Relative paths resolve against out_dir (never the
    # CWD); a directory-valued path gets "repair_cache.fits" appended. The
    # repair depends only on detection-side inputs, so the default ".."
    # (= <out_dir>/../repair_cache.fits) is shared by every band whose
    # out_dir sits in the same field directory: band 1 fits, bands 2..N
    # reload. Set None for a per-run cache in out_dir/repaired/.
    repair_cache_path: str | None = ".."
    bg_filter_sigma: float = 64.0  # get_bg_and_ivar background filter
    footprint_filter: bool = True  # keep only sources with wht_lo > 0
    # Trial patch for fast iteration: ``{"center": [ra, dec], "radius": 1.5}``
    # (degrees, arcmin), optionally ``"margin"`` in arcsec. ``None`` = full
    # field. When set, only the patch is read off disk and only sources inside
    # ``radius`` are fitted, but every array keeps its full-frame shape so
    # pixel coordinates, slices and WCS are unchanged.
    #
    # The margin covers what reaches outside the patch: PSF support (half of
    # ``psf_size`` — 4" for production and canfar runs, 8" for the verification
    # runs), template stamps and convolution wings. The 60" default clears
    # both by a wide margin and still reads a few per cent of a mosaic.
    #
    # A trial run is not a subset of a production run: the background and the
    # ivar calibration are measured on the patch, so ``sigma_true`` and hence
    # the flux errors differ from a full-field run. Use it to iterate, not to
    # produce release numbers.
    trial: dict[str, Any] | None = None
    # Viewer path for the scene-catalog `minerva_link` column, as
    # `<field>/<release>`. None derives the field from the leading token of
    # `name` and pairs it with `minerva_release`; "" drops the column.
    minerva_viewer: str | None = None
    minerva_release: str = "DR0"
    # --- fitting ----------------------------------------------------------
    fit: dict[str, Any] = field(default_factory=dict)  # FitConfig kwargs
    scene_plots: bool = True  # write per-scene diagnostic PNGs
    # per-source stamps FITS: tmpl_hi/tmpl_lo at native sizes + per-source PSF
    # region keys (PSF stamps stay in the cached <name>_psf_*.geojson maps)
    save_stamps: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        """Load a config from JSON; ``#``-comment lines are stripped."""
        text = Path(path).read_text()
        clean = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        data = json.loads(clean)
        known = {f.name for f in fields(cls)}
        legacy = {"r_trial", "trial_center"} & set(data)
        if legacy:
            raise ValueError(
                f"{sorted(legacy)} were replaced by a single `trial` field: "
                'trial={"center": [ra, dec], "radius": <arcmin>}. '
                "Regenerate the config (examples/make_minerva_configs.py) or "
                "edit it by hand."
            )
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        from dataclasses import asdict

        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def trial_geometry(self) -> tuple[tuple[float, float], float, float] | None:
        """``((ra, dec), radius_arcmin, margin_arcsec)``, or None for a full run.

        Raises:
            ValueError: ``trial`` is set but incomplete.
        """
        if not self.trial:
            return None
        center = self.trial.get("center")
        radius = float(self.trial.get("radius") or 0.0)
        margin = float(self.trial.get("margin", 60.0))
        if radius <= 0:
            return None
        if center is None or len(center) != 2:
            raise ValueError(
                'trial needs {"center": [ra, dec], "radius": <arcmin>}; '
                f"got center={center!r}"
            )
        unknown = set(self.trial) - {"center", "radius", "margin"}
        if unknown:
            raise ValueError(f"unknown trial keys: {sorted(unknown)}")
        return (float(center[0]), float(center[1])), radius, margin


def _trial_pixel_box(
    wcs: WCS, shape: tuple[int, int], center: tuple[float, float],
    radius_arcmin: float, margin_arcsec: float,
) -> tuple[int, int, int, int]:
    """Pixel ``(y0, y1, x0, x1)`` of the trial patch, clipped to ``shape``.

    The margin covers what reaches outside the patch: PSF support, template
    stamps and convolution wings. Sources are still selected on ``radius``;
    the margin only widens the pixels that are read.
    """
    scale = float(proj_plane_pixel_scales(wcs)[0]) * 3600.0  # arcsec / pixel
    x, y = wcs.all_world2pix(center[0], center[1], 0)
    half = (radius_arcmin * 60.0 + margin_arcsec) / scale
    y0 = max(0, int(np.floor(float(y) - half)))
    y1 = min(int(shape[0]), int(np.ceil(float(y) + half)) + 1)
    x0 = max(0, int(np.floor(float(x) - half)))
    x1 = min(int(shape[1]), int(np.ceil(float(x) + half)) + 1)
    if y0 >= y1 or x0 >= x1:
        raise ValueError(
            f"trial patch at {center} falls outside the image ({shape})"
        )
    return y0, y1, x0, x1


def _box_slice(box: tuple[int, int, int, int] | None) -> tuple[slice, slice]:
    """``box`` as a 2-D slice, or the whole array when there is no box.

    Basic slicing gives a *view*, so writing through it edits the parent
    array in place and touches only the box's pages.
    """
    if box is None:
        return (slice(None), slice(None))
    y0, y1, x0, x1 = box
    return (slice(y0, y1), slice(x0, x1))


def _upsample_boxed(
    image: np.ndarray,
    weight: np.ndarray | None,
    factor: int,
    box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``_upsample_flux_conserving_image_and_ivar`` over ``box`` only.

    The upsampled arrays are full-size on the reference grid, so pixel
    coordinates are unchanged, but only the box is replicated and written.
    On a trial patch the alternative is materialising the whole reference
    grid (876 Mpx = 3.5 GB per array for a MINERVA mosaic) to hold data that
    was never read.
    """
    k = int(factor)
    if box is None:
        return _upsample_flux_conserving_image_and_ivar(image, weight, k)

    y0, y1, x0, x1 = box
    sub_img, sub_wht = _upsample_flux_conserving_image_and_ivar(
        image[y0:y1, x0:x1], None if weight is None else weight[y0:y1, x0:x1], k
    )
    shape_hi = (image.shape[0] * k, image.shape[1] * k)
    hi_sl = (slice(y0 * k, y1 * k), slice(x0 * k, x1 * k))
    image_hi = np.zeros(shape_hi, dtype=sub_img.dtype)
    image_hi[hi_sl] = sub_img
    weight_hi = None
    if sub_wht is not None:
        weight_hi = np.zeros(shape_hi, dtype=sub_wht.dtype)
        weight_hi[hi_sl] = sub_wht
    return image_hi, weight_hi


def _read_image(path: str | Path, box: tuple[int, int, int, int] | None = None):
    """Read a 2-D image, optionally only the pixels inside ``box``.

    With a box, ``hdu.section`` reads just those rows off disk and the result
    is placed into a full-shape array. Untouched pages of that array are never
    faulted in, so it costs the box in resident memory while every pixel
    coordinate, slice and WCS in the pipeline keeps its full-frame meaning.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        idx = next(
            (i for i, h in enumerate(hdul) if int(h.header.get("NAXIS", 0)) == 2),
            None,
        )
        if idx is None:
            raise ValueError(f"no 2-D image HDU in {path}")
        hdu = hdul[idx]
        shape = (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))
        if box is None:
            return np.asarray(hdu.data)
        y0, y1, x0, x1 = box
        sub = np.asarray(hdu.section[y0:y1, x0:x1])
        full = np.zeros(shape, dtype=sub.dtype)
        full[y0:y1, x0:x1] = sub
        return full


def _apply_repair_patches(
    patches: Table, sci: np.ndarray, segmap: np.ndarray
) -> None:
    """Write a repair patch table's sci/segmap pixels into ``sci``/``segmap``.

    Shared by the cache-reuse path and the fresh-repair path so both end up
    with the same representation: the original mosaics, patched in place over
    the saturated cores only.
    """
    yy = np.asarray(patches["y"])
    xx = np.asarray(patches["x"])
    sci[yy, xx] = np.asarray(patches["sci"], sci.dtype)
    segmap[yy, xx] = np.asarray(patches["seg"], segmap.dtype)


def _upsample_flux_conserving_image_and_ivar(
    image: np.ndarray,
    weight: np.ndarray | None,
    factor: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Upsample a low-res image and inverse variance onto the reference grid.

    Flux-conserving block replication divides each native pixel value across
    ``factor**2`` subpixels. To preserve the native chi-square on that
    replicated basis, per-subpixel inverse variance must be multiplied by
    ``factor**2``. Do not use the flux-conserving default for weights.
    """
    k = int(factor)
    # conserve_sum=True divides by k**2 in float64, so on a full field it
    # builds a 7 GB intermediate to deliver a 3.5 GB float32 array. Replicate
    # without it (float32 in, float32 out) and do the same division in place;
    # dividing by an integer square is exact in binary floating point.
    image_hi = block_replicate(image, k, conserve_sum=False).astype(np.float32, copy=False)
    image_hi /= np.float32(k * k)
    weight_hi = None
    if weight is not None:
        weight_hi = block_replicate(weight, k, conserve_sum=False).astype(np.float32, copy=False)
        weight_hi *= np.float32(k * k)
    return image_hi, weight_hi


# should support PSFRegionMap as well, like in template.convolve_templates
#   ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
# else:
#     ra, dec = x, y
# kern = kernel.get_psf(ra, dec)


def _extract_psf_at(tmpl: Template, psf: np.ndarray | PSFRegionMap) -> np.ndarray:
    """Return a PSF stamp matching the template size.

    Parameters
    ----------
    tmpl : Template
        Template object providing position and size information
    psf : np.ndarray or PSFRegionMap
        Either a static PSF array or a PSFRegionMap for spatially varying PSFs

    Returns
    -------
    np.ndarray
        PSF stamp normalized to sum=1, matching template size
    """
    from scipy.ndimage import shift

    # Get the PSF array - either directly or via lookup
    if isinstance(psf, PSFRegionMap):
        # Look up PSF at template position
        x, y = tmpl.input_position_original
        w_lookup = tmpl.wcs_original if getattr(tmpl, "wcs_original", None) is not None else tmpl.wcs
        if w_lookup is not None:
            ra, dec = w_lookup.wcs_pix2world(x, y, 0)
        else:
            ra, dec = x, y
        psf_array = psf.get_psf(ra, dec)
        if psf_array is None:
            raise ValueError(f"No PSF found at position ({ra}, {dec})")
    else:
        # Use static PSF array
        psf_array = psf

    ny, nx = tmpl.data.shape
    cx_psf, cy_psf = psf_array.shape[1] // 2, psf_array.shape[0] // 2

    xc, yc = tmpl.input_position_cutout
    dx = xc - (nx // 2)
    dy = yc - (ny // 2)

    shifted = shift(psf_array, shift=(dy, dx), order=3, mode="constant", cval=0.0, prefilter=False)
    cut = Cutout2D(
        shifted,
        (cx_psf, cy_psf),
        tmpl.data.shape,
        mode="partial",
        fill_value=0.0,
    )
    stamp = cut.data.copy()
    s = stamp.sum()
    if s > 0:
        stamp /= s
    return stamp


def normalize(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` scaled to unit sum, leaving non-positive sums untouched.

    Separates PSF *shape* from PSF *throughput*: the caller keeps the native
    stamp sum as metadata and passes the shape on to kernel construction.
    """
    a = np.asarray(arr, dtype=float)
    s = float(np.nansum(a))
    return a / s if np.isfinite(s) and s > 0.0 else a


_PSF_PATTERN_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9+-]+)_(?P<det>.+?)_(?P<filt>[Ff]\d+[A-Za-z]*)"
    r"(?P<mjd>_MJD[^_]+)?(?:_FOV(?P<fov>\d+))?_GRID(?P<n>\d+)_(?P<samp>OS\d+|DET)$"
)


def _psf_factory_kwargs(pattern: str) -> dict[str, Any]:
    """Recover PSFFactory settings from an STDPSF filename pattern.

    The pattern is what the loader searches with, so deriving the generator
    settings from it is what guarantees the generated files are found again.
    ``PSFFactory.filename`` builds
    ``{prefix}_{DET}_{FILT}[_MJD{int}]_GRID{N}_{OS4|DET}.fits``; the detector is
    left to the factory, which decodes it per exposure from the CSV.
    """
    m = _PSF_PATTERN_RE.match(pattern.strip())
    if m is None:
        raise ValueError(
            f"cannot derive PSFFactory settings from pattern {pattern!r}; "
            "expected {prefix}_{DET}_{FILT}[_MJD..]_GRID{N}_{OS4|DET}"
        )
    samp = m.group("samp")
    return {
        "prefix": m.group("prefix"),
        "num_psfs": int(m.group("n")),
        "oversample": 4 if samp == "DET" else int(samp[2:]),
        "use_detsampled_psf": samp == "DET",
        "include_mjd": m.group("mjd") is not None,
        # FOV token (e.g. _FOV30 on the large halo grids): generate at
        # that field of view and keep the token in the saved filenames.
        **(
            {"fov_arcsec": float(m.group("fov")), "include_fov": True}
            if m.group("fov") is not None else {}
        ),
    }


def _stamp_provenance(prm: PSFRegionMap, **fields: Any) -> None:
    """Record on a region map what produced it, as geojson-round-tripping columns."""
    for key, value in fields.items():
        prm.regions[key] = value


def _provenance(prm: PSFRegionMap, key: str) -> Any:
    """Read one provenance field back, or None when the map predates it."""
    if key not in prm.regions.columns or not len(prm.regions):
        return None
    return prm.regions[key].iloc[0]


def _provenance_matches(prm: PSFRegionMap, want: dict[str, Any]) -> str | None:
    """Return the first field that disagrees with *want*, or None if all match."""
    for key, value in want.items():
        got = _provenance(prm, key)
        if got is None:
            return key
        if isinstance(value, float):
            if not np.isclose(float(got), value, rtol=0, atol=1e-9):
                return key
        elif str(got) != str(value):
            return key
    return None


class _ModelImages(_SequenceABC):
    """Lazy ``image - residual`` per fitted band, in place of a stored list.

    The model image is fully determined by two arrays the run already keeps,
    so storing it as well costs a third full-field array per band (3.5 GB on a
    MINERVA mosaic) for a value nothing in the fit reads -- only the
    diagnostics do. Those index this exactly like the list it replaces; the
    band asked for last is cached, so a per-source loop subtracts the mosaic
    once rather than once per source.
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline
        self._cache: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(getattr(self._pipeline, "residuals", []) or [])

    def __getitem__(self, idx: int) -> np.ndarray:
        n = len(self)
        i = idx + n if idx < 0 else idx
        if not 0 <= i < n:
            raise IndexError(f"no model image for band index {idx}")
        if self._cache is not None and self._cache[0] == i:
            return self._cache[1]
        # residual i belongs to image i + 1: index 0 is the detection band
        model = self._pipeline.images[i + 1] - self._pipeline.residuals[i]
        self._cache = (i, model)
        return model


class _Tee:
    """Write to a stream and a log file at once, one whole line at a time.

    Progress bars rewrite a line with carriage returns; only the last state of
    such a line reaches the file, so the log stays readable.
    """

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle.write(line.rsplit("\r", 1)[-1] + "\n")
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()


def _filter_psf_throughput(
    psf: np.ndarray | PSFRegionMap | None,
    explicit_throughput: float | None = None,
) -> float:
    """Return one finite-support PSF throughput correction for a fitted filter.

    For absolutely calibrated PSF stamps this mean stamp sum *is* the realized
    encircled energy of the square stamp (``ee_box`` in
    :func:`mophongo.psf.stamp_encircled_energy`), measured on the PSFs the
    pipeline actually uses -- after drizzling and after any broadening -- so it
    needs no correction for distortion or size quantization.

    The PSF core may vary spatially, but the missing far-wing flux correction is
    treated as a single filter-level scalar.  Pipeline-facing callers that pass
    unit-sum PSF shapes should provide ``explicit_throughput`` when the native
    finite PSF support sum is not one.
    """
    if explicit_throughput is not None and np.isfinite(explicit_throughput) and explicit_throughput > 0.0:
        return float(explicit_throughput)
    if psf is None:
        return 1.0
    arr = psf.psfs if isinstance(psf, PSFRegionMap) else psf
    if arr is None:
        return 1.0
    sums = np.nansum(np.where(np.isfinite(arr), arr, 0.0), axis=(-2, -1))
    sums = np.asarray(sums, dtype=float)
    valid = np.isfinite(sums) & (sums > 0.0)
    if np.any(valid):
        if not np.all(valid):
            logger.warning(
                "%d of %d PSF stamps have a non-finite or non-positive sum; "
                "the filter throughput averages the remaining %d",
                int(np.sum(~valid)), sums.size, int(np.sum(valid)),
            )
        return float(np.nanmean(sums[valid]))
    logger.warning(
        "no PSF stamp has a finite positive sum; filter throughput set to 1 "
        "(NO missing-flux correction will be applied)"
    )
    return 1.0


def _record_psf_ee(
    cat: Table,
    psf: np.ndarray | PSFRegionMap | None,
    pscale: float | None,
    idx: int,
    throughput: float,
) -> None:
    """Record realized PSF encircled energy for filter ``idx`` in ``cat.meta``.

    ``throughput`` is the square-stamp sum already applied to convert fitted
    amplitudes into ``flux_<idx>_total``, so writing it alongside the circular
    numbers keeps the aperture-correction reference with the catalog that used
    it.  All values come from the final PSF stamps, i.e. post-drizzle and
    post-broadening; nothing here is a request.

    Keys are kept to eight characters so they survive a FITS header without
    HIERARCH, and carry their description as the card comment.
    """
    from .psf import stamp_encircled_energy

    cat.meta[f"EEBOX{idx}"] = (
        float(throughput),
        "realized PSF encircled energy, full stamp",
    )
    arr = psf.psfs if isinstance(psf, PSFRegionMap) else psf
    if arr is None or not pscale or pscale <= 0.0:
        return
    arr = np.asarray(arr, dtype=float)
    if arr.ndim < 2:
        return

    ee = stamp_encircled_energy(arr, float(pscale))
    cat.meta[f"PSFSZ{idx}"] = (
        float(arr.shape[-1]) * float(pscale),
        "delivered PSF stamp side [arcsec]",
    )
    cat.meta[f"EECIRC{idx}"] = (
        ee["ee_circ"],
        f"realized PSF encircled energy within RCIRC{idx}",
    )
    cat.meta[f"RCIRC{idx}"] = (
        ee["r_circ"],
        "inscribed-circle radius of PSF stamp [arcsec]",
    )


def _bytscl(a: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """IDL ``bytscl``: linear byte scale, ``floor(255.9999*(x-mn)/(mx-mn))``."""
    f = (np.clip(np.asarray(a, dtype=float), mn, mx) - mn) * (255.9999 / (mx - mn))
    return np.minimum(f, 255.0).astype(np.uint8)


def _idl_robust_sigma(a: np.ndarray) -> float:
    """IDL ``robust_sigma``: Tukey biweight scale with tuning constant 6."""
    from astropy.stats import biweight_scale

    s = float(biweight_scale(np.asarray(a, dtype=float).ravel(), c=6.0))
    if not np.isfinite(s) or s <= 0.0:
        s = float(np.std(a))
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    return s


def _fptv_panel(
    img: np.ndarray,
    *,
    mm: tuple[float, float] | None = None,
    fac: float = 5.0,
    bin: int = 1,
    os: int = 2,
) -> np.ndarray:
    """One diagnostic panel, replicating IDL subphot's ``fptv``.

    Optional SNR-preserving display binning (block mean times ``sqrt(bin)``,
    then nearest-neighbour replication), byte scaling to ``mm`` or, when ``mm``
    is None, to ``median(img) +- fac*robust_sigma(img)``, and ``os``-times
    nearest-neighbour upsampling.
    """
    img = np.asarray(img, dtype=float).copy()
    sz = img.shape[0]
    b = int(bin)
    if b > 1:
        bsz = sz // b
        n = bsz * b
        blk = img[:n, :n].reshape(bsz, b, bsz, b).mean(axis=(1, 3)) * np.sqrt(b)
        img[:n, :n] = np.repeat(np.repeat(blk, b, axis=0), b, axis=1)
    if mm is None:
        rms = _idl_robust_sigma(img)
        img = img - np.median(img)
        mm = (-fac * rms, fac * rms)
    out = _bytscl(img, mm[0], mm[1])
    return np.repeat(np.repeat(out, os, axis=0), os, axis=1)


@dataclass
class SceneRefit:
    """One scene solved once or twice, and the comparison between the two.

    Produced by :meth:`Pipeline.refit_scene`. ``baseline`` is ``None`` when
    nothing was changed (or ``baseline=False``), in which case the deltas are
    against the scene's own solve and therefore zero.
    """

    id_scene: int
    ids: np.ndarray
    changed: dict[str, tuple]
    variant: Any
    baseline: Any
    pipeline: Any
    ifilt: int = 1

    def _chi2(self, scene) -> float:
        """Weighted chi2 over the scene's bounding box.

        The one scalar worth comparing: same pixels, same weights, both sides.
        """
        from .templates import _slices_from_bbox

        # Scene.residual() and .model_image() are already bbox-sized; only the
        # full-frame arrays need slicing.
        sl = _slices_from_bbox(scene.bbox)
        w = scene.weights[sl]
        r = scene.residual()
        good = np.isfinite(r) & np.isfinite(w) & (w > 0)
        return float(np.sum(w[good] * r[good] ** 2))

    @property
    def chi2(self) -> float:
        return self._chi2(self.variant)

    @property
    def dchi2(self) -> float:
        """Variant minus baseline; negative means the change helped."""
        if self.baseline is None:
            return 0.0
        return self._chi2(self.variant) - self._chi2(self.baseline)

    def table(self):
        """Per-source fluxes of both solves, with the fractional change."""
        from astropy.table import Table

        var = {int(t.id): t for t in self.variant.templates}
        rows = {"id": [], "flux": [], "err": []}
        if self.baseline is not None:
            rows |= {"flux_base": [], "err_base": [], "dflux_sigma": []}
            base = {int(t.id): t for t in self.baseline.templates}
        for sid in sorted(var):
            t = var[sid]
            rows["id"].append(sid)
            rows["flux"].append(float(t.flux))
            rows["err"].append(float(t.err))
            if self.baseline is not None:
                b = base.get(sid)
                rows["flux_base"].append(float(b.flux) if b else np.nan)
                rows["err_base"].append(float(b.err) if b else np.nan)
                # in units of the baseline error: "did this move by more than
                # the fit claims it knows the flux to"
                denom = float(b.err) if b and b.err else np.nan
                rows["dflux_sigma"].append((float(t.flux) - float(b.flux)) / denom
                                           if denom and np.isfinite(denom) else np.nan)
        return Table(rows)

    @property
    def leakage(self) -> float:
        """Largest coupling between this scene and a source outside it.

        Freezing membership is the control for an A/B, but it assumes the
        frozen set is still separable from its neighbours. This measures that
        assumption the way ``generate_scenes`` defines it -- the relative flux
        one source's template can absorb from another -- so a value above
        ``FitConfig.scene_coupling_thresh`` says the partition would have
        changed and the comparison has stopped being clean.

        Uses the restored fit's templates for the outsiders, which is what a
        rerun would have grouped against. ``nan`` when no restored template
        overlaps the scene.
        """
        from .scene_fitter import build_normal

        outside = [t for t in (getattr(self.pipeline, "templates", None) or [])
                   if int(t.id) not in set(int(i) for i in self.ids)
                   and _bbox_overlaps(getattr(t, "bbox_original", None), self.variant.bbox)]
        if not outside:
            return float("nan")
        members = list(self.variant.templates)
        ata, _, _ = build_normal(members + outside, self.variant.image,
                                 self.variant.weights)
        ata = ata.tocsr()
        flux = np.array([float(t.flux) for t in members + outside])
        diag = np.abs(ata.diagonal())
        n_in = len(members)
        worst = 0.0
        for i in range(n_in):
            for j in range(n_in, len(flux)):
                aij = abs(ata[i, j])
                if not aij:
                    continue
                fi, fj = abs(flux[i]), abs(flux[j])
                if diag[i] * fi > 0:
                    worst = max(worst, aij * fj / (diag[i] * fi))
                if diag[j] * fj > 0:
                    worst = max(worst, aij * fi / (diag[j] * fj))
        return float(worst)

    def plot_scene(self, which: str = "variant", *, display_sig: float = 5.0,
                   path: str | Path | None = None, **kwargs):
        """The run's own six-panel scene diagnostic, for a refitted scene.

        The same figure ``write_outputs`` writes to ``scenes/`` -- template,
        image, model, segmap, residual, colour composite, with the fitted shift
        field on the model panel -- drawn from this refit's solve rather than
        the run's. Takes the detection image and segmap from the pipeline, as
        the run does.

        The residual panel shows this scene's own residual, not the global one:
        the saved full-field residual has the *run's* model subtracted, so
        using it here would show the old fit under a new solve.

        Args:
            which: ``"variant"`` (default) or ``"baseline"``.
            display_sig: greyscale stretch, as in :meth:`Scene.plot`.
            path: save here when given.
            **kwargs: forwarded to :meth:`Scene.plot`.
        """
        scene = self.baseline if which == "baseline" else self.variant
        if scene is None:
            raise ValueError(f"no {which} solve on this refit")
        fig, _ = scene.plot(self.pipeline.images[0], self.pipeline.segmap,
                            display_sig=display_sig, **kwargs)
        if path is not None:
            fig.savefig(path, dpi=200, bbox_inches="tight")
        return fig

    def plot(self, path: str | Path | None = None):
        """Data, both models and both residuals, on one stretch."""
        import matplotlib.pyplot as plt

        scenes = [("variant", self.variant)]
        if self.baseline is not None:
            scenes.insert(0, ("baseline", self.baseline))
        from .templates import _slices_from_bbox

        sl = _slices_from_bbox(self.variant.bbox)
        data = np.asarray(self.variant.image)[sl]
        vmax = float(np.nanpercentile(data, 99.5)) or 1.0
        kw = dict(origin="lower", vmin=-0.1 * vmax, vmax=vmax, cmap="gray_r")

        ncol = 1 + 2 * len(scenes)
        fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.4))
        axes[0].imshow(data, **kw)
        axes[0].set_title(f"scene {self.id_scene}  ({len(self.ids)} src)")
        for k, (label, scene) in enumerate(scenes):
            model = np.asarray(scene.model_image())
            resid = np.asarray(scene.residual())
            axes[1 + 2 * k].imshow(model, **kw)
            axes[1 + 2 * k].set_title(f"model {label}")
            axes[2 + 2 * k].imshow(resid, **kw)
            axes[2 + 2 * k].set_title(f"resid {label}  chi2={self._chi2(scene):.4g}")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        if self.changed:
            fig.suptitle(", ".join(f"{k}: {a!r} -> {b!r}"
                                   for k, (a, b) in self.changed.items()), fontsize=9)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130, bbox_inches="tight")
        return fig


def _config_values_equal(a, b) -> bool:
    """Whether two config values are the same, counting NaN as itself.

    Several ``FitConfig`` defaults are NaN sentinels, and ``nan != nan`` would
    report every one of them as a change on every refit.
    """
    if a is b:
        return True
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:      # non-numeric (strings, dicts, None)
        return bool(a == b)


def _bbox_overlaps(a, b) -> bool:
    """Whether two ``(y0, y1, x0, x1)`` boxes intersect; ``None`` never does."""
    if a is None or b is None:
        return False
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


class Pipeline:
    """Photometry pipeline orchestrator.

    Parameters mirror :func:`run` for backwards compatibility. After
    calling :meth:`run` the resulting catalog, residual images and fitter
    instance are stored on the object and returned.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        segmap: np.ndarray,
        *,
        catalog: Table | None = None,
        psfs: Sequence[np.ndarray] | None = None,
        weights: Sequence[np.ndarray] | None = None,
        wht_images: Sequence[np.ndarray] | None = None,
        kernels: Sequence[np.ndarray | PSFRegionMap] | None = None,
        psf_throughputs: Sequence[float] | None = None,
        wcs: Sequence[WCS] | None = None,
        window: Window | None = None,
        extend_mode: str | None = None,
        extend_templates: str | None = None,
        templates: Templates | Sequence[Template] | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if psfs is not None and len(images) != len(psfs):
            raise ValueError("Number of images and PSFs must match")
        if extend_templates is not None:
            if extend_mode is not None:
                raise ValueError("pass extend_mode only; extend_templates is its deprecated alias")
            logger.warning("Pipeline(extend_templates=...) is deprecated; use extend_mode=...")
            extend_mode = extend_templates
        if weights is None and wht_images is not None:
            weights = wht_images
        if weights is not None and len(weights) != len(images):
            raise ValueError("Number of weight images must match number of images")
        if psf_throughputs is not None and len(psf_throughputs) != len(images):
            raise ValueError("Number of PSF throughputs must match number of images")

        if config is None:
            config = _FitConfig()

        self.images = images
        self.segmap = segmap
        self.catalog = catalog
        self.psfs = psfs
        self.weights = weights
        self.kernels = kernels
        self.psf_throughputs = psf_throughputs
        self.wcs = wcs
        self.window = window
        # Constructor override; None -> FitConfig.extend_mode decides (see
        # _resolve_extend_mode). The resolved scheme lives in self.extend_mode.
        self.extend_mode_override = extend_mode
        self.input_templates = templates
        self.config = config
        self.extend_mode = self._resolve_extend_mode(config)

        if kernels is None:
            kernels = [None] * len(images)
        if psfs is None:
            psfs = [None] * len(images)

        self.residuals: list[np.ndarray] = []
        # Pixel box of the trial patch on the hi/lo grids, or None for a
        # full-field run; set by load_data and used to scope the whole-array
        # passes (background/ivar, saturation repair).
        self.trial_box_hi: tuple[int, int, int, int] | None = None
        self.trial_box_lo: tuple[int, int, int, int] | None = None
        self.fit: list[np.ndarray] = []
        self.astro: list[np.ndarray] = []
        #        self.templates: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.tmpls: Templates | None = None
        self.templates_extracted: Templates | None = None
        self.templates_extended: Templates | None = None
        # derived from images/residuals on access, never stored (see _ModelImages)
        self.model_images = _ModelImages(self)

        if not hasattr(self, "run_config"):
            self.run_config = None

        logger.info("Pipeline (init) memory: %.1f GB", memory())

    # ------------------------------------------------------------------
    # Config-driven runs: Pipeline.from_config("run.json") + step methods.
    # Notebook-friendly: every step is one call, every intermediate stays
    # on the instance, expensive products are geojson-cached in out_dir.
    #
    #   pipe = Pipeline.from_config("uds_770_dr0.json")
    #   pipe.run_all()          # or: build_psfs / build_kernels / run /
    #                           #     write_outputs individually
    #
    # Command line:  python -m mophongo.pipeline uds_770_dr0.json [steps]
    #
    # PSF/kernel alignment: each band map carries PSFs drizzled at its OWN
    # region centroids (safe for position lookups), while matching kernels
    # are built from PSF pairs drizzled at the hi/lo overlay centroids.
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_config_path(path: str | Path) -> Path:
        """Resolve a config argument that may be a JSON file or a run directory.

        A directory resolves to ``<dir>/<dirname>.json`` (the snapshot written
        by :meth:`save_config`), or to the single ``*.json`` file it contains.
        """
        p = Path(path)
        if not p.is_dir():
            return p
        named = p / f"{p.name}.json"
        if named.exists():
            return named
        candidates = sorted(p.glob("*.json"))
        if len(candidates) == 1:
            return candidates[0]
        found = [f.name for f in candidates] or "none"
        raise FileNotFoundError(
            f"expected one run config JSON in {p}, found {found}"
        )

    @classmethod
    def from_config(cls, path: str | Path | RunConfig) -> "Pipeline":
        """Create a deferred Pipeline from a JSON run config.

        ``path`` may be the config JSON, a run directory containing one (see
        :meth:`_resolve_config_path`; e.g. a finished run's ``out_dir``), or
        a :class:`RunConfig`. Data are loaded lazily: :meth:`run` (or
        :meth:`load_data`) reads the images and finishes construction;
        :meth:`load_outputs` resumes a finished run. Relative paths inside
        the config still resolve against the process working directory.
        """
        cfg = path if isinstance(path, RunConfig) else RunConfig.from_json(cls._resolve_config_path(path))
        obj = cls.__new__(cls)
        obj.run_config = cfg
        obj.out_dir = Path(cfg.out_dir)
        obj.out_dir.mkdir(parents=True, exist_ok=True)
        obj.images = None
        obj.table = None
        obj.dpsf_hi = None
        obj.dpsf_lo = None
        obj.prm_hi = None
        obj.prm_lo = None
        obj.prm_kern = None
        obj._epsf_loaded = False
        return obj

    # -- cache paths -------------------------------------------------------
    @property
    def f_psf_hi(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_psf_hi.geojson"

    @property
    def f_psf_lo(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_psf_lo.geojson"

    @property
    def f_kernel(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_kernel.geojson"

    # -- output paths ------------------------------------------------------
    @property
    def f_config(self) -> Path:
        return self.out_dir / f"{self.run_config.name}.json"

    @property
    def f_fit_table(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_fit_table.fits"

    @property
    def f_residual(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_residual.fits"

    @property
    def f_templates(self) -> Path:
        return self.out_dir / f"{self.run_config.name}_templates.fits"

    @property
    def scenes(self):
        """Scenes of the (first) fitted band, once :meth:`run` has completed."""
        return self.all_scenes[0] if getattr(self, "all_scenes", None) else None

    def scene_ids(self, ifilt: int = 1) -> np.ndarray:
        """Scene labels present in a loaded fit, largest scene first."""
        table = self._scene_membership_table()
        labels, counts = np.unique(np.asarray(table["id_scene"], int), return_counts=True)
        return labels[np.argsort(-counts)]

    def _scene_membership_table(self):
        """``id``/``id_scene`` for this run, however it is currently held.

        Live scenes when the fit is still in memory, otherwise the per-template
        table :meth:`load_outputs` reads -- scene objects are not persisted, but
        their membership is, which is the whole reason a refit can freeze it.
        """
        scenes = self.scenes
        if scenes:
            return Table({
                "id": [int(t.id) for s in scenes for t in s.templates],
                "id_scene": [int(s.id) for s in scenes for t in s.templates],
            })
        for name in ("template_table", "table"):
            tab = getattr(self, name, None)
            if tab is not None and "id_scene" in tab.colnames:
                return tab
        raise RuntimeError(
            "no scene membership available: run() or load_fit() first, and for "
            "a restored fit make sure <name>_templates.fits is beside the fit "
            "table (it carries id_scene; the fit table does not)"
        )

    def refit_scene(
        self,
        id_scene: int,
        *,
        ifilt: int = 1,
        config: "_FitConfig | None" = None,
        baseline: bool = True,
        **overrides,
    ) -> "SceneRefit":
        """Re-extract and re-solve one scene, optionally against a baseline.

        The point is an A/B on exactly the same sources: membership is frozen
        to the ``id_scene`` the run recorded, so the source set, the segmap,
        the band pixels and the weights are identical on both sides and the
        only thing that varies is what you changed. Everything downstream of
        hi-res extraction is fair game -- ``extend_mode`` and the build-scheme
        parameters, the solver and astrometry settings -- because both sides
        are rebuilt through the same path the run used
        (:meth:`_prepare_hi_templates` then :meth:`_convolved_templates`).

        Extraction of a subset is exact rather than approximate:
        ``extract_templates`` is positions-driven, each cutout is that source's
        own segment bbox, and the build schemes are per-template leaf
        functions. Ten sources extracted alone are the same pixels as those ten
        extracted alongside a hundred thousand others.

        Args:
            id_scene: scene label from the restored fit.
            ifilt: fitted image index (1-based).
            config: complete replacement :class:`FitConfig`.
            baseline: also solve with the run's own config, for comparison.
                Always re-derived here rather than read from the fit table, so
                both sides have been resampled the same number of times.
            **overrides: individual ``FitConfig`` fields, applied to the run's
                config with :func:`dataclasses.replace`.

        Returns:
            :class:`SceneRefit` holding both scenes, the flux table, chi2 over
            the scene bbox, and the coupling to sources outside the frozen set.

        Two caveats this cannot remove. Freezing membership is the control for
        an A/B, but it is not what a full rerun would do: a change that widens
        the templates could merge or split scenes, and ``leakage`` is the
        measurement that says when that has started to matter. And a
        scene-level improvement is a screening result, not a field-level one.
        """
        if getattr(self, "run_config", None) is None:
            raise RuntimeError("refit_scene requires a config-driven pipeline")
        if self.images is None:
            raise RuntimeError("call load_fit() (or load_data()) first")

        base_cfg = self.config
        if config is not None and overrides:
            raise ValueError("pass either config= or individual overrides, not both")
        if config is not None:
            variant_cfg = config
        elif overrides:
            known = {f.name for f in fields(base_cfg)}
            unknown = set(overrides) - known
            if unknown:
                raise ValueError(
                    f"unknown FitConfig field(s): {sorted(unknown)}; "
                    f"known fields include {sorted(known)[:8]} ..."
                )
            variant_cfg = replace(base_cfg, **overrides)
        else:
            variant_cfg = base_cfg

        table = self._scene_membership_table()
        ids = np.asarray(table["id"], int)[np.asarray(table["id_scene"], int) == int(id_scene)]
        if not len(ids):
            raise ValueError(f"no sources with id_scene == {id_scene}")
        logger.info("scene %d: %d source(s) frozen from the recorded fit",
                    id_scene, len(ids))

        changed = {f.name: (getattr(base_cfg, f.name), getattr(variant_cfg, f.name))
                   for f in fields(base_cfg)
                   if not _config_values_equal(getattr(base_cfg, f.name),
                                               getattr(variant_cfg, f.name))}

        var_scene = self._solve_frozen_scene(id_scene, ids, variant_cfg, ifilt)
        base_scene = (self._solve_frozen_scene(id_scene, ids, base_cfg, ifilt)
                      if baseline and changed else None)
        return SceneRefit(id_scene=int(id_scene), ids=ids, changed=changed,
                          variant=var_scene, baseline=base_scene, pipeline=self,
                          ifilt=ifilt)

    def _solve_frozen_scene(self, id_scene: int, ids: np.ndarray,
                            config: "_FitConfig", ifilt: int):
        """Extract, convolve and solve one frozen source set. Restores state.

        ``_convolved_templates`` is not idempotent: on the upsample path it
        rebinds ``images[ifilt]`` and ``wcs[ifilt]`` to their reference-grid
        versions and appends to ``fit_bin_factors``. Calling it twice in a
        session would upsample an already-upsampled image, silently and
        wrongly, so everything it touches is snapshotted and put back.
        """
        from .scene import Scene, _bbox_union
        from .scene_fitter import SceneFitter

        snapshot = {
            "images": list(self.images),
            "wcs": list(self.wcs) if self.wcs is not None else None,
            "weights": list(self.weights) if self.weights is not None else None,
            "fit_bin_factors": list(getattr(self, "fit_bin_factors", [])),
            "tmpls": getattr(self, "tmpls", None),
            "templates_extracted": getattr(self, "templates_extracted", None),
            "templates_extended": getattr(self, "templates_extended", None),
            "extend_mode": getattr(self, "extend_mode", None),
        }
        try:
            cat = self.catalog[np.isin(np.asarray(self.catalog["id"], int), ids)]
            self._prepare_hi_templates(cat, config)
            templates, weights_i = self._convolved_templates(ifilt, config)
            scene = Scene(id=int(id_scene), templates=templates,
                          fitter=SceneFitter(), bbox=_bbox_union(templates))
            scene.set_band(self.images[ifilt], weights_i, config=config)
            scene.solve(config=config)
            # the band arrays the scene keeps are the ones it was solved
            # against, which on the upsample path are NOT the restored ones
            return scene
        finally:
            self.images = snapshot["images"]
            if snapshot["wcs"] is not None:
                self.wcs = snapshot["wcs"]
            if snapshot["weights"] is not None:
                self.weights = snapshot["weights"]
            self.fit_bin_factors = snapshot["fit_bin_factors"]
            for key in ("tmpls", "templates_extracted", "templates_extended",
                        "extend_mode"):
                if snapshot[key] is not None:
                    setattr(self, key, snapshot[key])

    # -- shared helpers ----------------------------------------------------
    def _blur_fwhm(self) -> float | None:
        blur = self.run_config.psf_blur_fwhm
        if blur == "default":
            from .mock_mosaic import DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC

            return DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC.get(self.run_config.filter_lo)
        return float(blur) if blur else None

    def _size_kw(self) -> dict:
        if self.run_config.psf_size is not None:
            return dict(size=self.run_config.psf_size)
        return dict(size=None, ee_fraction=None)

    def _ensure_dpsfs(self, load_epsf: bool = False) -> None:
        """Instantiate the DrizzlePSF pair (and optionally load the ePSFs)."""
        from .psf import DrizzlePSF

        cfg = self.run_config
        if self.dpsf_hi is None:
            self.dpsf_hi = DrizzlePSF(
                driz_image=str(cfg.sci_hi), csv_file=str(cfg.csv_hi)
            )
            self.dpsf_lo = DrizzlePSF(
                driz_image=str(cfg.sci_lo), csv_file=str(cfg.csv_lo)
            )
            if cfg.expect_frames:
                n_hi, n_lo = cfg.expect_frames
                got_hi = len(self.dpsf_hi.footprint)
                got_lo = len(self.dpsf_lo.footprint)
                if got_hi != n_hi or got_lo != n_lo:
                    raise ValueError(
                        f"frame-count mismatch: hi {got_hi} (expect {n_hi}), "
                        f"lo {got_lo} (expect {n_lo})"
                    )
        if load_epsf and not self._epsf_loaded:
            self._load_epsf(self.dpsf_hi, cfg.pattern_hi, cfg.csv_hi, "hi")
            self._load_epsf(self.dpsf_lo, cfg.pattern_lo, cfg.csv_lo, "lo")
            self._epsf_loaded = True

    def _missing_psf_dates(self, pattern: str, csv: str) -> list[float]:
        """Epochs this band needs for which no grid file exists.

        A grid is a function of its detector, filter, epoch and field of view,
        and every one of those is in its own filename. Whether the *set* is
        complete is a different question, and not one any single file can
        answer: it is the dates the exposure list implies under
        ``psf_date_mode``, minus the dates already on disk. Asking it this way
        is what makes adding a frame to the exposure list cost one grid instead
        of all of them -- the existing epochs are still exactly right.

        Matching is on the ``_MJD<int>`` token, which is how the files are
        named (:meth:`PSFFactory.filename` rounds), so it works for the
        non-integer dates ``cluster`` and ``modal`` produce as well as for the
        integer ones of ``all``. Field of view is deliberately not compared:
        the drizzled stamp may be smaller than the grid it comes from, and
        :meth:`_check_psf_size_fits_grids` is what says so when it is not.
        """
        from .jwst_psf import fov_agnostic_pattern
        from .psf_factory import dates_from_csv

        cfg = self.run_config
        if str(getattr(cfg, "psf_provenance", "warn")).lower() == "off":
            return []
        want = dates_from_csv(csv, cfg.psf_date_mode)
        psf_dir = Path(cfg.psf_dir)
        rx = re.compile(fov_agnostic_pattern(pattern))
        have: set[int] = set()
        if psf_dir.is_dir():
            for path in psf_dir.glob("*.fits"):
                if not rx.search(path.name):
                    continue
                token = re.search(r"_MJD(\d+)", path.name)
                if token:
                    have.add(int(token.group(1)))
        return [d for d in want if int(round(d)) not in have]

    def _existing_grid_fov(self, pattern: str) -> float | None:
        """Largest field of view already on disk for ``pattern``, if any.

        Neither the pattern nor ``psf_fov_arcsec`` is required to name a
        field of view. When one isn't given, a missing epoch should still
        match whatever this band's grids already use rather than falling
        back to the backend default -- otherwise one config can quietly
        produce a mixed FOV4/FOV6 set. First-ever build for a pattern has
        nothing to match, so this returns ``None`` and the backend default
        applies; :meth:`_check_psf_size_fits_grids` warns afterwards if the
        result is too small for ``psf_size``.
        """
        from .jwst_psf import fov_agnostic_pattern

        psf_dir = Path(self.run_config.psf_dir)
        if not psf_dir.is_dir():
            return None
        rx = re.compile(fov_agnostic_pattern(pattern))
        fovs = [
            float(m.group(1))
            for path in psf_dir.glob("*.fits")
            if rx.search(path.name)
            for m in [re.search(r"_FOV(\d+)", path.name)]
            if m
        ]
        return max(fovs) if fovs else None

    def _load_epsf(self, dpsf, pattern: str, csv: str, band: str) -> None:
        """Load the ePSF grids for one band, building the epochs that are absent.

        Completeness is a question about dates, not about files: the exposure
        list under ``psf_date_mode`` says which epochs this band needs, and
        :meth:`_missing_psf_dates` says which of them no file provides. Only
        those are built, one call per epoch, so a band already holding 99 of
        100 epochs costs one grid rather than a hundred.

        This is what catches the case a match count cannot: a set built under
        the old ``date_mode="modal"`` default holds one grid for a band whose
        exposures span years, matches the pattern, loads, and looks fine. Here
        it is one epoch present and the rest missing.
        """
        cfg = self.run_config
        kw = _psf_factory_kwargs(pattern)
        fov = kw.get("fov_arcsec", cfg.psf_fov_arcsec)
        if fov is None:
            fov = self._existing_grid_fov(pattern)
        missing = self._missing_psf_dates(pattern, csv)
        mode = str(getattr(cfg, "psf_provenance", "warn")).lower()

        if missing and not cfg.psf_autobuild:
            summary = (
                f"{band}-res band: {len(missing)} of "
                f"{len(missing) + len(set(dpsf.epsf_obj.epsf))} epoch(s) that "
                f"{Path(csv).name} implies under date_mode="
                f"{cfg.psf_date_mode!r} have no grid under {cfg.psf_dir} "
                f"(first MJD{int(round(missing[0]))}), and psf_autobuild is off"
            )
            if mode == "error":
                raise FileNotFoundError(summary)
            logger.warning("%s; fitting with the epochs that are there", summary)

        if missing and cfg.psf_autobuild:
            from .psf_factory import PSFFactory

            logger.warning(
                "%s-res band: building %d missing epoch(s) for %r from %s "
                "(minutes each, cached in %s)",
                band, len(missing), pattern, Path(csv).name, cfg.psf_dir,
            )
            Path(cfg.psf_dir).mkdir(parents=True, exist_ok=True)
            # An _FOV token in the pattern carries its own field of view;
            # otherwise psf_fov_arcsec applies, then the FOV already on disk
            # (`fov`, resolved above), then the backend default.
            kw.pop("fov_arcsec", None)
            # One call per epoch: a literal MJD as date_mode selects exactly
            # that date (psf_factory.dates_from_csv), so nothing already on
            # disk is touched and no other epoch is recomputed.
            for mjd in missing:
                PSFFactory(outdir=str(cfg.psf_dir), fov_arcsec=fov,
                           date_mode=float(mjd),
                           workers=int(getattr(cfg, "psf_workers", 1) or 1),
                           **kw).from_csv(str(csv), save=True)

        dpsf.epsf_obj.load_jwst_stdpsf(local_dir=str(cfg.psf_dir), filter_pattern=pattern)
        if not dpsf.epsf_obj.epsf:
            raise FileNotFoundError(
                f"no PSF grids under {cfg.psf_dir} match {pattern!r} for the "
                f"{band}-res band"
                + ("" if cfg.psf_autobuild else " and psf_autobuild is off")
            )
        logger.info(
            "%s-res band: loaded %d ePSF grid(s) matching %r",
            band, len(dpsf.epsf_obj.epsf), pattern,
        )
        self._check_psf_size_fits_grids(dpsf, band)

    def _check_psf_size_fits_grids(self, dpsf, band: str) -> None:
        """Warn when the requested stamp is wider than the grids that feed it.

        ``psf_size`` is the drizzled stamp side in arcsec; a grid's ``FOV``
        header is its own side in arcsec. Asking for more than the grid holds
        does not fail: ``eval_ePSF`` returns zero outside the grid's support,
        so the stamp is simply padded with zeros and the missing wings are
        quietly absent from the kernel and from every template. That is worth
        a line in the log rather than a silent truncation of the PSF.
        """
        size = self.run_config.psf_size
        if not size:
            return
        meta_by_key = getattr(dpsf.epsf_obj, "epsf_meta", None) or {}
        fovs = {key: meta.get("fov") for key, meta in meta_by_key.items()}
        short = {key: fov for key, fov in fovs.items()
                 if fov is not None and float(size) > float(fov)}
        if not short:
            return
        worst = min(short, key=lambda k: short[k])
        logger.warning(
            "%s-res band: psf_size=%.3g\" exceeds the field of view of %d of "
            "%d ePSF grid(s); the stamp is zero-padded beyond the grid and the "
            "PSF wings outside it are lost. Smallest: %s at FOV=%.3g\". Rebuild "
            "those grids at a larger fov_arcsec, or lower psf_size.",
            band, float(size), len(short), len(fovs), worst, short[worst],
        )

    def _region_maps(self):
        """Region maps for both bands (geometry only, deterministic)."""
        prm_hi = PSFRegionMap.from_footprints(
            self.dpsf_hi.footprint, name="hi"
        ).overlay_with(self.dpsf_hi.driz_footprint)
        prm_lo = PSFRegionMap.from_footprints(
            self.dpsf_lo.footprint, name="lo"
        ).overlay_with(self.dpsf_lo.driz_footprint)
        return prm_hi, prm_lo

    @staticmethod
    def _centroids(prm) -> list[np.ndarray]:
        # do NOT drop any region: the PSF cube must stay index-aligned
        return [np.squeeze(p.xy) for p in prm.regions.geometry.centroid]

    def _drizzle_lo_blurred(self, positions) -> np.ndarray:
        """Lo-res PSF cube at ``positions`` with the configured broadening."""
        cube = self.dpsf_lo.get_psf_radec(positions, **self._size_kw())
        blur = self._blur_fwhm()
        if blur:
            from .mock_mosaic import gaussian_blur_psf

            cube = gaussian_blur_psf(cube, blur, self.dpsf_lo.driz_pscale)
        return cube

    # -- step 1: per-band PSF region maps ---------------------------------
    def build_psfs(self, overwrite: bool = False) -> "Pipeline":
        """Build (or reload) per-band PSF maps with PSFs at their own centroids."""
        self._ensure_dpsfs()
        cfg = self.run_config
        want_hi = {"pattern": cfg.pattern_hi, "psf_size": float(cfg.psf_size or 0.0),
                   "blur_fwhm": 0.0}
        want_lo = {"pattern": cfg.pattern_lo, "psf_size": float(cfg.psf_size or 0.0),
                   "blur_fwhm": float(self._blur_fwhm() or 0.0)}
        if self.f_psf_hi.exists() and self.f_psf_lo.exists() and not overwrite:
            cached_hi = PSFRegionMap.from_geojson(str(self.f_psf_hi))
            cached_lo = PSFRegionMap.from_geojson(str(self.f_psf_lo))
            stale = (_provenance_matches(cached_hi, want_hi)
                     or _provenance_matches(cached_lo, want_lo))
            if stale is None:
                self.prm_hi, self.prm_lo = cached_hi, cached_lo
                logger.info(
                    "loaded cached PSF maps from %s (psf_size=%.3g, blur=%.3g)",
                    self.out_dir, want_lo["psf_size"], want_lo["blur_fwhm"],
                )
                return self
            logger.warning(
                "cached PSF maps in %s disagree on %r; rebuilding",
                self.out_dir, stale,
            )

        self._ensure_dpsfs(load_epsf=True)
        prm_hi, prm_lo = self._region_maps()
        prm_hi.psfs = self.dpsf_hi.get_psf_radec(
            self._centroids(prm_hi), **self._size_kw()
        )
        prm_lo.psfs = self._drizzle_lo_blurred(self._centroids(prm_lo))
        blur = self._blur_fwhm()
        if blur:
            logger.info('applied %.3f" FWHM Gaussian broadening to lo-res PSFs', blur)
        _stamp_provenance(prm_hi, **want_hi)
        _stamp_provenance(prm_lo, **want_lo)
        prm_hi.to_file(self.f_psf_hi)
        prm_lo.to_file(self.f_psf_lo)
        self.prm_hi, self.prm_lo = prm_hi, prm_lo
        return self

    # -- step 2: matching-kernel map --------------------------------------
    def build_kernels(
        self,
        overwrite: bool = False,
        *,
        method: str = "wiener",
        reg: float | None = None,
    ) -> "Pipeline":
        """Build (or reload) the matching-kernel map on the hi/lo overlay.

        Args:
            overwrite: Rebuild even if a cached kernel map exists.
            method: Matching method passed to :func:`mophongo.utils.matching_kernel`.
                Defaults to ``"wiener"``, matching the verification harness.
            reg: Regularization parameter. When ``None`` and the method is
                regularized, it is optimized once on a representative region.

        A cached map records the method and regularization that produced it and
        is reused only when the method matches; otherwise it is rebuilt. The
        matching method is worth a few percent in the flux scale
        (docs/ENCIRCLED_ENERGY.pdf), so silently reusing a map built another way
        would apply a correction the run did not ask for.
        """
        from . import utils

        if self.f_kernel.exists() and not overwrite:
            cached = PSFRegionMap.from_geojson(str(self.f_kernel))
            cached_method = _provenance(cached, "kernel_method")
            cached_reg = _provenance(cached, "kernel_reg")
            cached_reg = float("nan") if cached_reg is None else float(cached_reg)
            if cached_method is not None and str(cached_method) == method:
                self.prm_kern = cached
                logger.info(
                    "loaded cached kernel map %s (method=%s, reg=%.4g)",
                    self.f_kernel, cached_method, cached_reg,
                )
                return self
            logger.warning(
                "cached kernel map %s was built with method=%s; this run wants "
                "%s, so it is being rebuilt",
                self.f_kernel, cached_method or "unrecorded", method,
            )

        self._ensure_dpsfs(load_epsf=True)
        prm_hi_geom, prm_lo_geom = self._region_maps()
        prm_kern = prm_hi_geom.overlay_with(prm_lo_geom)
        pos = self._centroids(prm_kern)

        psf_hi = self.dpsf_hi.get_psf_radec(pos, **self._size_kw())
        psf_lo = self._drizzle_lo_blurred(pos)

        pixel_ratio = round(self.dpsf_lo.driz_pscale / self.dpsf_hi.driz_pscale)
        # Kernels are matched between unit-sum PSF *shapes*
        # (docs/PSF_SHAPE_THROUGHPUT_CONVENTION.md).  Feeding native-sum stamps
        # would make sum(kernel) carry sum(psf_lo)/sum(psf_hi), which then hides
        # the kernel's own fidelity error inside a throughput factor.  The maps
        # written by :meth:`build_psfs` keep their native sums; only the copies
        # used here are normalized.
        shapes_hi = [normalize(p) for p in psf_hi]
        shapes_lo = [normalize(p) for p in psf_lo]

        # The default SplitCosineBell window low-passes the model, which biases
        # every fitted amplitude high by sum(W|P|^2)/sum(W^2|P|^2) -- 2.2% on the
        # F444W/F770W pair, independent of stamp size.  A regularized method with
        # an optimized parameter avoids it (docs/ENCIRCLED_ENERGY.pdf).
        kw: dict[str, Any] = {"method": method}
        if method.strip().lower() != "window":
            if reg is None:
                # One scan on the median shape, reused for every region: a
                # per-region grid search would cost 21 kernels per region.
                from .psf import PSF

                med_hi = normalize(np.median(np.asarray(shapes_hi), axis=0))
                med_lo = normalize(np.median(np.asarray(shapes_lo), axis=0))
                fit = PSF.from_array(med_hi).optimize_matching_kernel_regularization(
                    PSF.from_array(med_lo),
                    method=method,
                    pixel_ratio=pixel_ratio,
                    recenter=False,
                    growth_weight=1.0,
                    core_weight=1.0,
                    l2_weight=1.0,
                    kernel_regularization_weight=1e-3,
                )
                reg = float(fit.reg)
                logger.info(
                    "%s regularization from a %d-point scan on the median PSF: "
                    "reg=%.4g (score %.5g)",
                    method, len(fit.reg_grid), reg, float(fit.score),
                )
            kw["reg"] = reg
        kernels = [
            utils.matching_kernel(s_hi, s_lo, pixel_ratio=pixel_ratio, **kw)
            for s_hi, s_lo in zip(shapes_hi, shapes_lo)
        ]
        # Renormalize to unit sum.  Unit-sum inputs already put sum(k) within a
        # part in 1e3 of one, so this only removes the residual regularization
        # DC, but it guarantees the kernel carries no flux scale of its own:
        # the total flux correction is then ee_psf_lo and nothing else.  It says
        # nothing about whether the kernel has the right *shape*, which is a
        # separate term (docs/ENCIRCLED_ENERGY.pdf).
        raw_sums = np.array([float(np.nansum(k)) for k in kernels])
        prm_kern.psfs = np.asarray([normalize(k) for k in kernels])
        # Stamp the provenance so a cached map is never reused under a
        # different method.  These round-trip through the geojson as columns.
        _stamp_provenance(
            prm_kern,
            kernel_method=method,
            kernel_reg=float("nan") if reg is None else float(reg),
            psf_size=float(self.run_config.psf_size or 0.0),
        )
        logger.info(
            "kernel map: method=%s reg=%s, DC before renormalization "
            "mean %.6f, range %.6f-%.6f; renormalized to 1",
            method, "n/a" if reg is None else f"{reg:.4g}",
            float(np.nanmean(raw_sums)),
            float(np.nanmin(raw_sums)),
            float(np.nanmax(raw_sums)),
        )
        prm_kern.to_file(self.f_kernel)
        self.prm_kern = prm_kern
        return self

    def _ensure_maps(self) -> None:
        """Load the cached PSF maps and build the kernel map if missing.

        The hi-res map is needed as well as the lo-res one: it is the
        detection-band PSF the template build scheme reads from ``psfs[0]``
        (every ``extend_mode`` except ``'none'``).
        """
        if self.prm_lo is None and self.f_psf_lo.exists():
            self.prm_lo = PSFRegionMap.from_geojson(str(self.f_psf_lo))
        if self.prm_hi is None and self.f_psf_hi.exists():
            self.prm_hi = PSFRegionMap.from_geojson(str(self.f_psf_hi))
        if self.prm_kern is None:
            self.build_kernels()

    def resolve_wht_hi(self) -> Path:
        """Resolve the detection-band weight map, raising if there is none.

        ``RunConfig.wht_hi`` when set, else ``sci_hi`` with the standard
        ``_sci.fits`` -> ``_wht.fits`` substitution. A run without a detection
        weight map has no calibrated detection noise, so this raises rather
        than degrading: the SNR-weighted build schemes would silently fall back
        to one sky-sigma scalar for the whole mosaic.
        """
        cfg = self.run_config
        if cfg.wht_hi is not None:
            path = Path(cfg.wht_hi)
            if not path.exists():
                raise FileNotFoundError(f"wht_hi does not exist: {path}")
            return path
        guess = Path(str(cfg.sci_hi).replace("_sci.fits", "_wht.fits"))
        if guess == Path(cfg.sci_hi) or not guess.exists():
            raise FileNotFoundError(
                "no detection-band weight map: wht_hi is unset and the standard "
                f"'_sci.fits' -> '_wht.fits' sibling of sci_hi does not exist ({guess}). "
                "Set wht_hi in the run config -- without it the detection noise is "
                "uncalibrated and the SNR-weighted build schemes fall back to a "
                "single sky-sigma scalar for the whole mosaic."
            )
        return guess

    @staticmethod
    def _bg_and_ivar_boxed(
        sci: np.ndarray,
        wht: np.ndarray,
        box: tuple[int, int, int, int] | None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``get_bg_and_ivar`` restricted to ``box``, returned full-shape.

        A trial run's arrays are full-shape but only the box was read, so the
        rest is zero. Estimating on the whole array would both fault in the
        entire mosaic and measure the noise of a field of zeros; running on
        the slice keeps the cost and the statistics on real pixels. Outside
        the box the returned ivar is zero, so those pixels carry no weight.
        """
        from .catalog import get_bg_and_ivar

        if box is None:
            return get_bg_and_ivar(sci, wht, **kwargs)
        y0, y1, x0, x1 = box
        bg_sub, ivar_sub = get_bg_and_ivar(sci[y0:y1, x0:x1], wht[y0:y1, x0:x1], **kwargs)
        ivar = np.zeros(sci.shape, dtype=ivar_sub.dtype)
        ivar[y0:y1, x0:x1] = ivar_sub
        del ivar_sub
        if bg_sub is None:  # need_bg=False
            return None, ivar
        bg = np.zeros(sci.shape, dtype=bg_sub.dtype)
        bg[y0:y1, x0:x1] = bg_sub
        return bg, ivar

    def _repair_provenance(self, pattern_halo: str) -> dict[str, str]:
        """What the repair depended on; a cache is valid only if all match."""
        cfg = self.run_config
        prov: dict[str, str] = {
            "sci_hi": str(cfg.sci_hi),
            "wht_hi": str(self.resolve_wht_hi()),
            "pattern": cfg.pattern_hi,
            "halo": pattern_halo or "",
            "kwargs": json.dumps(cfg.repair_kwargs or {}, sort_keys=True),
            # A trial run only repairs its own patch, so its cache must not
            # satisfy a full-field run (or a differently-placed patch).
            "trial_box": json.dumps(getattr(self, "trial_box_hi", None)),
        }
        for key in ("sci_hi", "wht_hi"):
            prov[key + "_mtime"] = str(int(Path(prov[key]).stat().st_mtime))
        return prov

    def _save_repair_cache(
        self,
        path: Path,
        prov: dict[str, str],
        sci0: np.ndarray, wht0: np.ndarray, seg0: np.ndarray,
        rep: dict[str, Any],
        cat0: Table,
    ) -> None:
        """Persist the repair as pixel patches + catalog flag columns.

        The repair only touches saturated cores and flag columns, so the
        cache stores diffs, not mosaics: a PATCHES bintable of changed
        pixels (sci/wht/segmap values) and a FLAGS bintable of the
        catalog's ``FLAG_SATURATED_*`` columns, holding only the ids that
        carry a nonzero flag.

        The patch table is also left on the instance as ``_repair_patches``,
        so :meth:`load_data` can replay it onto a fresh memory map instead of
        holding the repaired mosaics in anonymous memory.
        """
        from astropy.io import fits

        # Changed pixels, found one band of rows at a time. The whole-array
        # form builds three full-field boolean arrays plus the temporaries of
        # the two ORs -- 4.4 GB on a MINERVA detection grid -- at the one
        # moment the pre-repair and post-repair mosaics are both in memory.
        sci_new = np.asarray(rep["sci"])
        wht_new = np.asarray(rep["wht"])
        seg_new = np.asarray(rep["segmap"])
        ny, nx = sci0.shape
        band = max(1, (1 << 22) // max(nx, 1))  # ~4 Mpx of booleans per pass
        ys: list[np.ndarray] = []
        xs: list[np.ndarray] = []
        for y0 in range(0, ny, band):
            y1 = min(y0 + band, ny)
            changed = sci_new[y0:y1] != sci0[y0:y1]
            changed |= wht_new[y0:y1] != wht0[y0:y1]
            changed |= seg_new[y0:y1] != seg0[y0:y1]
            by, bx = np.nonzero(changed)
            if by.size:
                ys.append(by + y0)
                xs.append(bx)
        yy = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
        xx = np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)
        patches = Table({
            "y": yy.astype(np.int32), "x": xx.astype(np.int32),
            "sci": np.asarray(rep["sci"])[yy, xx].astype(np.float32),
            "wht": np.asarray(rep["wht"])[yy, xx].astype(np.float32),
            "seg": np.asarray(rep["segmap"])[yy, xx].astype(np.int64),
        })
        cat = rep["catalog"]
        self._repair_patches = patches
        flag_cols = [c for c in cat.colnames if c.startswith("FLAG_SATURATED_")]
        # Only the flagged ids. A field flags a few hundred segments out of a
        # few hundred thousand sources, and the loader rebuilds each column
        # from zero, so the unflagged rows carry no information.
        keep = np.zeros(len(cat), dtype=bool)
        for c in flag_cols:
            keep |= np.asarray(cat[c]) != 0
        flags = Table({"id": np.asarray(cat["id"], np.int64)[keep]})
        for c in flag_cols:
            flags[c] = np.asarray(cat[c])[keep]
        hdr = fits.Header()
        for key, value in prov.items():
            hdr[f"HIERARCH REPAIR {key.upper()}"] = value
        hdr["NPATCH"] = len(patches)
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.BinTableHDU(patches, name="PATCHES"),
            fits.BinTableHDU(flags, name="FLAGS"),
        ]).writeto(path, overwrite=True)
        logger.info("repair cache written: %d pixel patches, %d flag rows -> %s",
                    len(patches), len(flags), path)

    def _load_repair_cache(
        self,
        path: Path,
        prov: dict[str, str],
        sci: np.ndarray, wht0: np.ndarray, seg: np.ndarray,
        cat: Table,
    ) -> tuple[np.ndarray, np.ndarray, Table, np.ndarray] | None:
        """Apply a matching repair cache; None when absent, stale, or unreadable.

        Unreadable counts as absent. The cache is a plain optimisation -- the
        repair can always be re-run -- and by default every band of a field
        shares one path (``repair_cache_path`` resolves to ``out_dir/..``), so
        a campaign that submits its bands together can have one job reading the
        file while another is still writing it. Letting a truncated read kill
        the run would trade a recomputable cache for a lost fit.
        """
        from astropy.io import fits

        if not path.exists():
            return None
        try:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                for key, value in prov.items():
                    got = hdr.get(f"REPAIR {key.upper()}")
                    if got is None or str(got) != value:
                        logger.warning(
                            "repair cache %s is stale (%s changed); re-running repair",
                            path.name, key,
                        )
                        return None
                patches = Table(hdul["PATCHES"].data)
                flags = Table(hdul["FLAGS"].data)
        except (OSError, KeyError, ValueError) as exc:
            # truncated or half-written file, or one missing its tables
            logger.warning(
                "repair cache %s is unreadable (%s: %s); re-running repair",
                path.name, type(exc).__name__, exc,
            )
            return None
        wht = wht0.copy()
        _apply_repair_patches(patches, sci, seg)
        yy, xx = np.asarray(patches["y"]), np.asarray(patches["x"])
        wht[yy, xx] = np.asarray(patches["wht"], wht.dtype)
        by_id = {int(i): k for k, i in enumerate(cat["id"])}
        rows = np.array([by_id.get(int(i), -1) for i in flags["id"]], dtype=np.int64)
        hit = rows >= 0
        for col in flags.colnames:
            if col == "id":
                continue
            # rebuilt from zero, not merged: the cache lists only flagged ids,
            # so an unlisted id means "not flagged" rather than "unknown"
            cat[col] = np.zeros(len(cat), dtype=np.asarray(flags[col]).dtype)
            cat[col][rows[hit]] = np.asarray(flags[col])[hit]
        n_flagged = int(np.sum(np.any(np.column_stack(
            [np.asarray(flags[c]) != 0 for c in flags.colnames if c != "id"]
        ), axis=1))) if len(flags.colnames) > 1 else 0
        logger.info(
            "repair reloaded from cache: %d pixel patches, %d flagged rows (%s)",
            len(patches), n_flagged, path.name,
        )
        return sci, wht, cat, seg

    def _repair_halo_pattern(self) -> str:
        """Canonical large-FOV halo-grid pattern derived from ``pattern_hi``.

        Keeps the prefix/detector/filter of the photometry grids and swaps
        the sampling tail for the 30" single-position halo layout:
        ``{prefix}_{det}_{filt}_MJD\\d+_FOV30_GRID1_OS4``. Building these
        is a one-off of a few minutes per detector (stpsf at 30" FOV),
        cached in ``psf_dir``; :meth:`_load_epsf` handles the build when
        ``psf_autobuild`` is on.
        """
        m = _PSF_PATTERN_RE.match(self.run_config.pattern_hi.strip())
        if m is None:
            logger.warning(
                "cannot derive a halo-grid pattern from pattern_hi=%r; "
                "set repair_psf_pattern explicitly",
                self.run_config.pattern_hi,
            )
            return ""
        return (
            f"{m.group('prefix')}_{m.group('det')}_{m.group('filt')}"
            r"_MJD\d+_FOV30_GRID1_OS4"
        )

    def _load_detection_ivar(
        self, sci_hi: np.ndarray, wht_hi: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Detection-band inverse variance, or None when the run does not read it.

        The weight map is always resolved (:meth:`resolve_wht_hi`, which raises
        if there is none), but its pixels are read only by the build schemes
        that weight real data against a PSF model by SNR: ``'wren'`` for every
        weight, ``'classic'`` for its low-SNR point-source branch.
        ``'default'`` and ``'psf_convolution'`` never touch ``weights[0]``, and a
        full-field hi-res weight map costs as much memory as the mosaic itself.

        Args:
            wht_hi: In-memory weight override — the saturation-repaired
                weights when ``repair_saturated`` is on, so the filled cores
                keep their restored (non-zero) weights.
        """
        from astropy.io import fits
        from .catalog import get_bg_and_ivar

        cfg = self.run_config
        path = self.resolve_wht_hi()
        mode = str(cfg.fit.get("extend_mode", _FitConfig.extend_mode) or "none").lower()
        mode = EXTEND_MODE_ALIASES.get(mode, mode)
        if mode not in ("psf_wings", "wren", "classic"):
            logger.info("detection weight map %s (not read: extend_mode=%r)", path.name, mode)
            return None

        box_hi = getattr(self, "trial_box_hi", None)
        # The detection background is measured but NOT subtracted: that would
        # change 'default' templates too. It matters for the extended schemes,
        # which blend raw data over a large halo, so get_bg_and_ivar reports its
        # median -- which is the only use this path ever had for it, hence
        # need_bg=False rather than a second mosaic-sized array here.
        _, ivar_hi = self._bg_and_ivar_boxed(
            sci_hi,
            wht_hi if wht_hi is not None else _read_image(path, box_hi),
            box_hi,
            bg_filter_sigma=cfg.bg_filter_sigma,
            label=f"detection band, {path.name}",
            need_bg=False,
        )
        logger.info("detection ivar from %s (background not subtracted)", path.name)
        return ivar_hi

    # -- step 3: data ------------------------------------------------------
    def load_data(self, kernels: bool = True) -> "Pipeline":
        """Load images/segmap/catalog, preprocess, and finish construction.

        Args:
            kernels: When ``False``, skip loading/building the PSF and kernel
                maps so data can be loaded and inspected quickly before a run
                (:meth:`run` finishes the maps later).
        """
        from astropy.io import fits
        from .catalog import get_bg_and_ivar

        cfg = self.run_config
        wcs_hi = WCS(fits.getheader(cfg.sci_hi))
        wcs_lo = WCS(fits.getheader(cfg.sci_lo))

        # Trial patch: read only those pixels, but into full-shape arrays so
        # nothing downstream has to know. self.trial_box_hi/_lo also scope the
        # whole-array passes (background/ivar, saturation repair), which would
        # otherwise fault in the entire mosaic and undo the saving.
        geom = cfg.trial_geometry()
        box_hi = box_lo = None
        if geom is not None:
            center, radius, margin = geom
            hdr_hi = fits.getheader(cfg.sci_hi)
            hdr_lo = fits.getheader(cfg.sci_lo)
            shape_hi = (int(hdr_hi["NAXIS2"]), int(hdr_hi["NAXIS1"]))
            shape_lo = (int(hdr_lo["NAXIS2"]), int(hdr_lo["NAXIS1"]))
            box_hi = _trial_pixel_box(wcs_hi, shape_hi, center, radius, margin)
            box_lo = _trial_pixel_box(wcs_lo, shape_lo, center, radius, margin)
            logger.warning(
                "TRIAL RUN: r=%.2f' + %.0f\" margin at (%.5f, %.5f); reading "
                "hi %dx%d of %dx%d (%.1f%%), lo %dx%d of %dx%d. Background and "
                "ivar are calibrated on the patch, so fluxes and errors will "
                "NOT match a full-field run — iterate with this, do not "
                "release it.",
                radius, margin, center[0], center[1],
                box_hi[1] - box_hi[0], box_hi[3] - box_hi[2], *shape_hi,
                100.0 * (box_hi[1] - box_hi[0]) * (box_hi[3] - box_hi[2])
                / float(shape_hi[0] * shape_hi[1]),
                box_lo[1] - box_lo[0], box_lo[3] - box_lo[2], *shape_lo,
            )
        self.trial_box_hi, self.trial_box_lo = box_hi, box_lo

        tmpl_hi = _read_image(cfg.sci_hi, box_hi)
        sci_lo = _read_image(cfg.sci_lo, box_lo)
        wht_lo = _read_image(cfg.wht_lo, box_lo)
        # Normalise the label dtype once, here at the boundary: releases differ
        # (MINERVA COSMOS ships float64 where UDS and EGS ship int32) and every
        # downstream SegmentationImage would otherwise have to defend itself.
        segmap = as_label_array(_read_image(cfg.segmap, box_hi))
        cat = Table.read(cfg.catalog)

        wht_hi_repaired: np.ndarray | None = None
        if cfg.repair_saturated:
            from .repair import repair_in_memory

            self._ensure_dpsfs(load_epsf=True)
            # Large-FOV grids for the halo model (halo + spikes); the core
            # fit keeps the MJD-matched pattern_hi ePSFs and the halo is
            # grafted outside their support. Same two-PSF split as the
            # original standalone repair flow.
            pattern_halo = cfg.repair_psf_pattern or self._repair_halo_pattern()
            stamp_dpsf = None
            if pattern_halo and pattern_halo != cfg.pattern_hi:
                from .psf import DrizzlePSF

                stamp_dpsf = DrizzlePSF(
                    driz_image=str(cfg.sci_hi),
                    info=(self.dpsf_hi.flt_keys, self.dpsf_hi.wcs,
                          self.dpsf_hi.footprint, self.dpsf_hi.hdrs),
                )
                try:
                    self._load_epsf(stamp_dpsf, pattern_halo, cfg.csv_hi, "halo")
                except (FileNotFoundError, ValueError) as exc:
                    # Missing grids with autobuild off, or a pattern the
                    # autobuild grammar cannot parse (e.g. a legacy-order
                    # spelling): degrade to the pattern_hi reach instead of
                    # failing the run — the repair itself is unaffected.
                    logger.warning(
                        "halo PSF grids unavailable for %r (%s); flag reach "
                        "limited to the pattern_hi field of view",
                        pattern_halo, exc,
                    )
                    stamp_dpsf, pattern_halo = None, ""
            wht0 = _read_image(self.resolve_wht_hi(), box_hi)
            if cfg.repair_cache_path:
                cache_path = Path(cfg.repair_cache_path)
                if not cache_path.is_absolute():
                    # resolves against out_dir, NOT the process CWD: multi-band
                    # configs whose out_dirs share a field directory can all
                    # point at "../<field>_repair_cache.fits"
                    cache_path = self.out_dir / cache_path
                if cache_path.is_dir() or not cache_path.suffix:
                    cache_path = cache_path / "repair_cache.fits"
            else:
                cache_path = self.out_dir / "repaired" / "repair_cache.fits"
            prov = self._repair_provenance(pattern_halo)
            cached = None
            if cfg.repair_reuse:
                cached = self._load_repair_cache(
                    cache_path, prov, tmpl_hi, wht0, segmap, cat
                )
            if cached is not None:
                tmpl_hi, wht_hi_repaired, cat = cached[0], cached[1], cached[2]
                segmap = as_label_array(cached[3])
            else:
                sci0 = tmpl_hi.copy()
                seg0 = segmap.copy()
                rep = repair_in_memory(
                    tmpl_hi, wht0,
                    dpsf=self.dpsf_hi, wcs=wcs_hi, psf_pattern=cfg.pattern_hi,
                    catalog=cat, segmap=segmap,
                    stamp_dpsf=stamp_dpsf,
                    stamp_pattern=pattern_halo or None,
                    out_dir=self.out_dir / "repaired",
                    plots=cfg.scene_plots,
                    **(cfg.repair_kwargs or {}),
                )
                self._save_repair_cache(cache_path, prov, sci0, wht0, seg0, rep, cat)
                # the pre-repair snapshots exist only to be written to the
                # cache, and they are two mosaic-sized arrays
                del sci0, seg0
                wht_hi_repaired = rep["wht"]
                cat = rep["catalog"]
                # `repair_saturated_holes` returns fresh full-field copies of
                # sci and segmap (saturate.py:733), so holding them costs two
                # mosaics of anonymous memory for a result that differs from
                # the inputs only over the saturated cores. Replay the patch
                # table onto fresh maps of the originals instead, exactly as
                # the reuse path above does: astropy maps a read-only HDU
                # copy-on-write, so only the patched pages go private and the
                # rest stays evictable page cache.
                tmpl_hi = _read_image(cfg.sci_hi, box_hi)
                segmap = as_label_array(_read_image(cfg.segmap, box_hi))
                _apply_repair_patches(self._repair_patches, tmpl_hi, segmap)
                del rep
            # the raw hi-res weight map is superseded by the repaired one
            del wht0

        if cfg.footprint_filter:
            scale_hi = proj_plane_pixel_scales(wcs_hi)[0]
            scale_lo = proj_plane_pixel_scales(wcs_lo)[0]
            k = round(float(scale_lo / scale_hi))
            ix = np.clip((np.asarray(cat["x"]) / k).astype(int), 0, wht_lo.shape[1] - 1)
            iy = np.clip((np.asarray(cat["y"]) / k).astype(int), 0, wht_lo.shape[0] - 1)
            cat = cat[wht_lo[iy, ix] > 0]
            logger.info("%d sources inside the lo-res footprint", len(cat))

        if geom is not None:
            import astropy.units as u
            from astropy.coordinates import SkyCoord

            center, radius, _margin = geom
            coords = SkyCoord(
                ra=np.asarray(cat["ra"], float) * u.deg,
                dec=np.asarray(cat["dec"], float) * u.deg,
            )
            ref = SkyCoord(ra=center[0] * u.deg, dec=center[1] * u.deg)
            cat = cat[coords.separation(ref) < radius * u.arcmin]
            logger.info("trial radius %.2f': %d sources", radius, len(cat))

        bg, ivar = self._bg_and_ivar_boxed(
            sci_lo, wht_lo, box_lo,
            bg_filter_sigma=cfg.bg_filter_sigma,
            label=f"{cfg.filter_lo or 'lo band'}, {Path(cfg.wht_lo).name}",
        )
        # Background subtraction and the non-finite guard, over the trial box
        # only. Whole-array arithmetic here would touch every page and fault
        # the mosaic that was deliberately not read back into memory; outside
        # the box the arrays are zero, which is already what the guard wants.
        sl_lo = _box_slice(box_lo)
        # np.zeros, not np.zeros_like: zeros_like is empty_like + memset and
        # so writes every page, which on a trial run materialises the whole
        # grid that was deliberately never read
        sci_fit = np.zeros(sci_lo.shape, dtype=sci_lo.dtype)
        sub = sci_lo[sl_lo] - bg[sl_lo]
        # zero non-finite pixels in image AND weight so they carry no information
        bad = ~np.isfinite(sub)
        sub[bad] = 0.0
        sci_fit[sl_lo] = sub
        ivar_box = ivar[sl_lo]
        ivar_box[bad] = 0.0
        ivar_box[~np.isfinite(ivar_box)] = 0.0

        ivar_hi = self._load_detection_ivar(tmpl_hi, wht_hi=wht_hi_repaired)
        sl_hi = _box_slice(box_hi)
        tmpl_box = tmpl_hi[sl_hi]
        bad_hi = ~np.isfinite(tmpl_box)
        np.nan_to_num(tmpl_box, copy=False)
        if ivar_hi is not None:
            ivar_hi_box = ivar_hi[sl_hi]
            ivar_hi_box[bad_hi] = 0.0
            ivar_hi_box[~np.isfinite(ivar_hi_box)] = 0.0

        if kernels:
            self._ensure_maps()

        # finish construction: regular __init__ on the loaded products.
        # psfs[0] is the hi-res map, which template extension needs; it is None
        # only when the maps were skipped (``kernels=False``), and `run` builds
        # them before it reaches the extension.
        Pipeline.__init__(
            self,
            [tmpl_hi, sci_fit],
            segmap,
            weights=[ivar_hi, ivar],
            catalog=cat,
            # psfs[0] is the detection-band map: template extension / the
            # 'wren' and 'classic' build schemes look up their detection PSF
            # there. Only fitted bands (ifilt >= 1) feed the throughput and
            # PSF-EE bookkeeping, so index 0 is inert for those.
            psfs=[self.prm_hi, self.prm_lo],
            kernels=[None, self.prm_kern],
            wcs=[wcs_hi, wcs_lo],
            config=_FitConfig(**cfg.fit),
        )
        # __init__ resets the trial boxes; they describe the data just loaded,
        # so restore them for the repair provenance and the detection-band ivar
        self.trial_box_hi, self.trial_box_lo = box_hi, box_lo
        return self

    # -- config snapshot + resume ------------------------------------------
    def save_config(self, path: str | Path | None = None) -> Path:
        """Write the fully-explicit run config to ``out_dir/<name>.json``.

        Every :class:`RunConfig` field and every *used* :class:`FitConfig`
        setting is written with its resolved value, so the run stays
        repeatable even if code defaults change later. Settings of template
        build schemes the run did not select (``wren_*``/``classic_*`` for
        other ``extend_mode`` values) are omitted. :meth:`run` calls this
        automatically; :meth:`from_config` accepts the snapshot (or its
        directory) back.
        """
        from dataclasses import asdict, replace
        from datetime import date

        if getattr(self, "config", None) is not None:
            fit = asdict(self.config)
        else:
            fit = asdict(_FitConfig(**self.run_config.fit))
        # drop settings of template build schemes this run did not use
        mode = fit.get("extend_mode", "default")
        for prefix in {"wren": "classic_", "classic": "wren_"}.get(
            mode, ("wren_", "classic_")
        ):
            fit = {k: v for k, v in fit.items() if not k.startswith(prefix)}
        cfg = replace(self.run_config, fit=fit)
        out = Path(path) if path is not None else self.f_config
        header = (
            f"# full '{cfg.name}' run config snapshot, {date.today()}: all\n"
            "# RunConfig and FitConfig settings explicit (Pipeline.save_config)\n"
        )
        out.write_text(header + json.dumps(asdict(cfg), indent=2) + "\n")
        logger.info("wrote full run config to %s", out)
        return out

    def load_outputs(self) -> "Pipeline":
        """Load a previous run's products from ``out_dir`` (fresh-session resume).

        Reads the fit table and residual image written by :meth:`write_outputs`.
        Catalog-level diagnostics work directly on ``self.table``; image-based
        diagnostics additionally need :meth:`load_data`.
        """
        from astropy.io import fits

        if getattr(self, "run_config", None) is None:
            raise RuntimeError("load_outputs requires a config-driven Pipeline")
        if self.f_fit_table.exists():
            self.table = Table.read(self.f_fit_table)
            logger.info("loaded %s (%d rows)", self.f_fit_table.name, len(self.table))
        else:
            logger.warning("no fit table %s", self.f_fit_table)
        if self.f_residual.exists():
            self.residuals = [fits.getdata(self.f_residual)]
            logger.info("loaded %s", self.f_residual.name)
        else:
            logger.warning("no residual image %s", self.f_residual)
        if self.f_templates.exists():
            self.template_table = Table.read(self.f_templates)
            logger.info(
                "loaded %s (%d templates)", self.f_templates.name, len(self.template_table)
            )
        else:
            self.template_table = None
            logger.warning("no template table %s (older run?)", self.f_templates)
        return self

    def _allocate_residual(self, image: np.ndarray, ifilt: int) -> np.ndarray:
        """Zero-filled residual accumulator, file-backed where possible.

        The residual is written to :attr:`f_residual` at the end of the run
        either way, so mapping that file's data section costs no extra disk
        and turns a full reference-grid array of dirty anonymous pages (3.5 GB
        on a MINERVA field) into file pages the kernel can flush and evict.
        The access pattern suits it: scattered writes over scene bounding
        boxes, then one sequential subtract, then stamp-sized reads.

        Falls back to anonymous memory for API-driven runs (no output path)
        and for bands past the first, which have nowhere distinct to live.
        """
        if getattr(self, "run_config", None) is None or ifilt != 1:
            return np.zeros(image.shape, dtype=image.dtype)
        try:
            return self._residual_memmap(image.shape)
        except OSError as exc:
            logger.warning(
                "could not map %s (%s); accumulating the residual in memory",
                self.f_residual.name, exc,
            )
            return np.zeros(image.shape, dtype=image.dtype)

    def _residual_memmap(self, shape: tuple[int, int]) -> np.memmap:
        """Create ``f_residual`` at full size and map its data section.

        Writes the FITS header, then extends the file with ``truncate`` --
        which leaves it sparse, so blocks are allocated only as pages are
        written, and a trial patch costs the patch. The map is big-endian
        because that is what FITS stores; numpy handles the mixed byte order
        in the accumulate and the subtract, and :meth:`write_outputs` then has
        only the header left to finish.
        """
        from astropy.io import fits

        cfg = self.run_config
        path = self.f_residual
        path.parent.mkdir(parents=True, exist_ok=True)
        with _quiet_hierarch_warnings():
            hdr = fits.getheader(cfg.sci_hi).copy()
            for key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                        "EXTEND", "BSCALE", "BZERO"):
                hdr.remove(key, ignore_missing=True, remove_all=True)
            hdu = fits.PrimaryHDU(data=np.zeros((1, 1), dtype=np.float32), header=hdr)
            hdu.header["NAXIS1"] = int(shape[1])
            hdu.header["NAXIS2"] = int(shape[0])
            head = hdu.header.tostring(padding=True).encode("ascii")
        nbytes = int(shape[0]) * int(shape[1]) * 4
        total = len(head) + nbytes
        with open(path, "wb") as fh:
            fh.write(head)
            fh.truncate(-(-total // 2880) * 2880)  # FITS pads to 2880 blocks
        res = np.memmap(path, dtype=">f4", mode="r+", offset=len(head), shape=shape)
        logger.info("residual accumulating into %s (%s, file-backed)",
                    path.name, human_bytes(nbytes))
        return res

    def _scene_pixels_needed(self) -> bool:
        """Whether anything after the solve still reads a scene's band pixels.

        ``Scene.plot`` and ``Scene.residual`` mask on ``scene.weights``, so the
        weight map survives the end of the solve only until the scene figures
        are drawn. A run driven directly through the API (no ``run_config``)
        keeps everything: the caller owns the instance and may plot at any
        point.
        """
        cfg = getattr(self, "run_config", None)
        return cfg is None or bool(cfg.scene_plots)

    def _release_scene_weights(self, scenes: Sequence["Scene"] | None = None) -> None:
        """Drop the band weight map from scenes once nothing reads it.

        The weights are last read by ``Templates.predicted_errors`` and by the
        scene figures; on the upsample path they are a full reference-grid
        array (3.5 GB on a MINERVA field) that every ``Scene`` holds a
        reference to, so without this they would stay alive to the end of the
        process. ``Scene.solve`` refuses to run without them, which is the
        intended failure: the fit is over.

        Args:
            scenes: One band's scenes. Defaults to every band recorded on the
                instance.
        """
        bands = [scenes] if scenes is not None else (getattr(self, "all_scenes", []) or [])
        released = 0
        for band in bands:
            for s in band:
                if getattr(s, "weights", None) is not None:
                    s.weights = None
                    released += 1
        if released:
            logger.info(
                "released the band weight map from %d scene(s); memory: %.1f GB",
                released, memory(),
            )

    def _template_fit_table(self) -> Table:
        """Per-template fit state of the first fitted band as a flat table.

        Records what a deterministic template rebuild cannot re-derive:
        per-component fitted amplitudes/errors, the applied astrometric
        shifts, and scene membership.
        """
        scene_of = {id(t): s.id for s in (self.scenes or []) for t in s.templates}
        rows = []
        for t in self.all_templates[0]:
            pid = t.id_parent if getattr(t, "parent_id", None) is not None else t.id
            x, y = t.position_original
            dx, dy = (float(v) for v in t.shifted[:2])
            rows.append(
                (int(t.id), int(pid), float(x), float(y), dx, dy,
                 float(t.flux), float(t.err), int(scene_of.get(id(t), 0)))
            )
        return Table(
            rows=rows,
            names=["id", "id_parent", "x", "y", "dx", "dy", "flux", "err", "id_scene"],
        )

    # -- inspection --------------------------------------------------------
    def __repr__(self) -> str:
        cfg = getattr(self, "run_config", None)
        name = f" {cfg.name!r}" if cfg is not None else ""
        if getattr(self, "table", None) is not None:
            stage = "fitted"
        elif getattr(self, "images", None) is None:
            stage = "configured"
        else:
            stage = "loaded"
        nimg = len(self.images) if getattr(self, "images", None) is not None else 0
        cat = getattr(self, "catalog", None)
        nsrc = len(cat) if cat is not None else 0
        return f"<Pipeline{name} [{stage}] images={nimg} sources={nsrc}>"

    def info(self) -> str:
        """Print and return a summary of config, inputs, data, maps, and results.

        Usable at every stage: after :meth:`from_config` it reports the input
        files (existence, size, shape from the FITS headers, frame counts)
        without reading pixel data; after :meth:`load_data` it adds the loaded
        images/segmap/catalog; after :meth:`run` the fit products.
        """
        from astropy.io import fits

        lines = [repr(self)]
        cfg = getattr(self, "run_config", None)
        if cfg is not None:
            lines.append(f"config: out_dir={self.out_dir}")
            keys = ["sci_hi", "wht_hi", "segmap", "catalog", "sci_lo", "wht_lo",
                    "csv_hi", "csv_lo"]
            for key in keys:
                if key == "wht_hi":
                    try:  # unset -> derived from sci_hi; report what will be used
                        path = self.resolve_wht_hi()
                    except FileNotFoundError:
                        lines.append(f"  {key:8s} MISSING  (unset, no '_wht' sibling of sci_hi)")
                        continue
                else:
                    path = Path(getattr(cfg, key))
                if not path.exists():
                    lines.append(f"  {key:8s} MISSING  {path}")
                    continue
                desc = f"{human_bytes(path.stat().st_size):>10s}"
                try:
                    if path.suffix == ".csv":
                        desc += f"  {sum(1 for _ in open(path)) - 1} frames"
                    elif key == "catalog":
                        desc += f"  {fits.getheader(path, 1)['NAXIS2']} rows"
                    else:
                        hdr = fits.getheader(path)
                        desc += f"  {hdr['NAXIS2']}x{hdr['NAXIS1']}"
                except Exception as exc:
                    desc += f"  (unreadable: {exc})"
                lines.append(f"  {key:8s} {desc}  {path}")
            for label, f in (
                ("psf_hi", self.f_psf_hi),
                ("psf_lo", self.f_psf_lo),
                ("kernel", self.f_kernel),
            ):
                state = "cached" if f.exists() else "not built"
                lines.append(f"  map {label:6s} {state}  {f.name}")
            for label, f in (
                ("config", self.f_config),
                ("table", self.f_fit_table),
                ("residual", self.f_residual),
            ):
                state = "present" if f.exists() else "absent"
                lines.append(f"  out {label:8s} {state}  {f.name}")

        if getattr(self, "images", None) is None:
            lines.append("data: not loaded (load_data() reads images and catalog)")
        else:
            lines.append("data:")
            for i, img in enumerate(self.images):
                if img is None:
                    continue
                ps = self._pixel_scale_arcsec(self.wcs[i]) if self.wcs is not None else None
                pstxt = f"  {ps * 1000:.0f} mas/pix" if ps else ""
                wht = self.weights[i] if self.weights is not None else None
                whttxt = "  +weight" if wht is not None else ""
                lines.append(
                    f"  image[{i}]  {img.shape[0]}x{img.shape[1]} {img.dtype}{pstxt}{whttxt}"
                )
            seg = np.asarray(self.segmap)
            lines.append(f"  segmap    {seg.shape[0]}x{seg.shape[1]}  max label {int(seg.max())}")
            cat = getattr(self, "catalog", None)
            if cat is not None:
                cols = ", ".join(cat.colnames[:10])
                more = f" (+{len(cat.colnames) - 10} more)" if len(cat.colnames) > 10 else ""
                lines.append(f"  catalog   {len(cat)} rows: {cols}{more}")

        for label in ("prm_hi", "prm_lo", "prm_kern"):
            prm = getattr(self, label, None)
            if prm is None:
                continue
            stamp = ""
            if getattr(prm, "psfs", None) is not None:
                stamp = f", stamps {np.asarray(prm.psfs).shape[-1]} px"
            lines.append(f"  {label:8s} {len(prm.regions)} regions{stamp}")

        table = getattr(self, "table", None)
        if table is not None:
            fluxcols = [c for c in table.colnames if c.startswith("flux_")]
            lines.append(
                f"results: table {len(table)} rows ({', '.join(fluxcols)}); "
                f"{len(getattr(self, 'residuals', []))} residual image(s)"
            )
            if getattr(self, "all_scenes", None):
                lines.append(f"  scenes per band: {[len(s) for s in self.all_scenes]}")

        text = "\n".join(lines)
        print(text)
        return text

    def plot_inputs(
        self,
        *,
        sources: bool = True,
        save: str | os.PathLike | None = None,
    ):
        """Quicklook of the loaded inputs: hi-res, lo-res, weight, and segmap.

        Shows ``images[0]``, the last low-resolution image, that image's
        inverse-variance weight map (panel left blank when no weights are
        loaded), and the segmentation map.  Requires loaded data (after
        :meth:`load_data` or the array constructor).

        Args:
            sources: Overlay catalog positions on the hi-res panel.
            save: Optional path to save the figure to.

        Returns:
            Tuple of the created figure and its flat array of axes.
        """
        import matplotlib.pyplot as plt
        from photutils.segmentation import SegmentationImage

        if getattr(self, "images", None) is None:
            raise RuntimeError("no data loaded; call load_data() first")

        img_hi = self.images[0]
        img_lo = self.images[-1]
        wht_lo = self.weights[-1] if self.weights is not None else None
        seg = np.asarray(self.segmap)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12 * img_hi.shape[0] / img_hi.shape[1]))
        axes = axes.flatten()

        self._imshow_scaled(axes[0], img_hi)
        axes[0].set_title(f"images[0] hi-res {img_hi.shape[0]}x{img_hi.shape[1]}")
        cat = getattr(self, "catalog", None)
        if sources and cat is not None:
            axes[0].scatter(
                cat["x"], cat["y"], s=8, facecolors="none", edgecolors="tab:red", lw=0.5
            )
        self._imshow_scaled(axes[1], img_lo)
        axes[1].set_title(f"images[{len(self.images) - 1}] lo-res {img_lo.shape[0]}x{img_lo.shape[1]}")
        if wht_lo is not None:
            self._imshow_scaled(axes[2], wht_lo)
            axes[2].set_title("weight (inverse variance)")
        else:
            axes[2].set_axis_off()
        if seg.max() > 0:
            axes[3].imshow(
                seg, origin="lower", cmap=SegmentationImage(seg).cmap, interpolation="nearest"
            )
        else:
            axes[3].imshow(seg, origin="lower", cmap="gray")
        axes[3].set_title(f"segmap (max label {int(seg.max())})")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=180, bbox_inches="tight")
        return fig, axes

    def _scene_shift_samples(
        self, scene, *, at: Sequence[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample one scene's total shift field.

        Without ``at``, the number of samples follows the Chebyshev order of
        the scene's shift basis: ``2**order``, laid out as ``nx x ny`` with the
        longer side along the scene's longer axis. Order 0 gives a single
        sample at the scene center, order 1 two spread along that axis, order 2
        a 2x2 grid.

        ``Scene.shifts`` holds only the last astrometric iteration, so the
        field is refit at the same order to the accumulated
        ``Template.shifted`` values, which are the total applied offsets.

        Args:
            scene: A :class:`~mophongo.scene.Scene` that solved for shifts.
            at: Evaluate at this single ``(x, y)`` instead of on the grid.

        Returns:
            ``(xy, dxy)`` in reference-image pixels, both ``(n, 2)``, or None
            when the scene has no usable shift solution.
        """
        from .astrometry import cheb_basis, n_terms

        shifts = getattr(scene, "shifts", None)
        basis = getattr(scene, "shift_basis", None)
        if shifts is None or basis is None or len(shifts) < 2 or not scene.templates:
            return None
        _, (x0, y0), (Sx, Sy) = basis
        # invert n_terms(order) = (order+1)(order+2)/2 on the dx half of shifts
        p = len(shifts) // 2
        order = int(round((np.sqrt(8.0 * p + 1.0) - 3.0) / 2.0))
        if order < 0 or n_terms(order) != p:
            logger.warning(
                "scene %s: %d shift coefficients match no Chebyshev order; skipped",
                getattr(scene, "id", -1), len(shifts),
            )
            return None

        pos = np.array([t.position_original for t in scene.templates], dtype=float)
        sh = np.array([t.shifted[:2] for t in scene.templates], dtype=float)
        ok = np.isfinite(pos).all(axis=1) & np.isfinite(sh).all(axis=1)
        if not ok.any():
            return None
        design = np.vstack(
            [cheb_basis((px - x0) / Sx, (py - y0) / Sy, order) for px, py in pos[ok]]
        )
        beta, *_ = np.linalg.lstsq(design, sh[ok], rcond=None)  # (p, 2)

        if at is not None:
            xs, ys = np.atleast_1d(float(at[0])), np.atleast_1d(float(at[1]))
        else:
            # sample grid: 2**order points spread over the scene's own extent
            nx, ny = 2 ** ((order + 1) // 2), 2 ** (order // 2)
            span = pos[ok].max(axis=0) - pos[ok].min(axis=0)
            if span[1] > span[0]:
                nx, ny = ny, nx

            def step(n: int) -> np.ndarray:
                return np.zeros(1) if n == 1 else np.linspace(-0.5, 0.5, n)

            gu, gv = np.meshgrid(step(nx), step(ny))
            xs = pos[ok][:, 0].mean() + gu.ravel() * span[0] * _SHIFT_SAMPLE_SPREAD
            ys = pos[ok][:, 1].mean() + gv.ravel() * span[1] * _SHIFT_SAMPLE_SPREAD

        phi = np.vstack(
            [cheb_basis((x - x0) / Sx, (y - y0) / Sy, order) for x, y in zip(xs, ys)]
        )
        return np.column_stack([xs, ys]), phi @ beta

    def plot_shift_field(
        self,
        *,
        save: str | os.PathLike | None = None,
        arrow_frac: float = 0.05,
    ):
        """Map of the fitted astrometric shift field over the whole field.

        One arrow per sample point of every scene that solved for astrometry
        (see :meth:`_scene_shift_samples` for the sampling), drawn from the
        template position toward where the source is measured in the fitted
        band. Shifts are sub-pixel, so arrows carry a common magnification and
        the legend arrow gives the true scale; the scene id labels each scene
        in light gray.

        Args:
            save: Optional path to save the figure to.
            arrow_frac: Median arrow length as a fraction of the field span.

        Returns:
            ``(fig, ax)``, or None when no scene solved for astrometry.
        """
        import matplotlib.pyplot as plt

        wcs = self.wcs[0] if getattr(self, "wcs", None) is not None else None
        if wcs is None:
            logger.warning("no reference WCS; skipping the shift field plot")
            return None

        xy, dxy, ids, anchors = [], [], [], []
        for s in self.scenes or []:
            sampled = self._scene_shift_samples(s)
            if sampled is None:
                continue
            pix, shift = sampled
            xy.append(pix)
            dxy.append(shift)
            ids.append(int(s.id))
            # label sits by the first arrow, which for order 0 is the scene centre
            anchors.append(pix[0])
        if not xy:
            logger.info("no scene solved for astrometry; no shift field plot")
            return None
        xy, dxy = np.vstack(xy), np.vstack(dxy)
        anchors = np.vstack(anchors)

        # positions and shift vectors on the sky; RA differences are tiny, so
        # wrap only guards a field straddling RA = 0
        ra, dec = wcs.wcs_pix2world(xy[:, 0], xy[:, 1], 0)
        ra2, dec2 = wcs.wcs_pix2world(xy[:, 0] + dxy[:, 0], xy[:, 1] + dxy[:, 1], 0)
        dra = (ra2 - ra + 180.0) % 360.0 - 180.0
        ddec = dec2 - dec
        ra_lab, dec_lab = wcs.wcs_pix2world(anchors[:, 0], anchors[:, 1], 0)

        # display in raw RA/Dec degrees with aspect 1/cos(dec) so that angles
        # are undistorted; all angular lengths below carry the same cos factor
        cosd = float(np.cos(np.deg2rad(np.mean(dec))))
        ang = np.hypot(dra * cosd, ddec)
        span = max((ra.max() - ra.min()) * cosd, dec.max() - dec.min()) or 60.0 / 3600.0
        # scale off a high percentile, not the median: a single scene with a
        # runaway solution would otherwise shrink every other arrow to nothing
        ref_ang = float(np.percentile(ang, 90)) if np.any(ang > 0) else 0.0
        gain = arrow_frac * span / ref_ang if ref_ang > 0 else 1.0

        fig, ax = plt.subplots(figsize=(9, 8))
        # Arrows carry a common magnification, so length alone cannot be read
        # off the plot: colour them by their true magnitude and give the bar
        # the units. The key below still sets the length scale; together they
        # say both "how long is an arrow" and "how big is this one".
        mag = np.hypot(dxy[:, 0], dxy[:, 1])
        q = ax.quiver(
            ra, dec, dra * gain, ddec * gain, mag,
            angles="xy", scale_units="xy", scale=1,
            cmap="viridis", width=0.003, headwidth=4, headlength=5,
        )
        pscale_cb = self._pixel_scale_arcsec(wcs)
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("|shift| (pix)" + (f'   [1 pix = {pscale_cb:.3f}"]'
                                          if pscale_cb else ""))
        for sid, rl, dl in zip(ids, np.atleast_1d(ra_lab), np.atleast_1d(dec_lab)):
            ax.text(rl, dl, str(sid), color="0.7", fontsize=7, ha="left", va="bottom")
        if ref_ang > 0:
            ref_pix = float(np.percentile(mag, 90))
            pscale = self._pixel_scale_arcsec(wcs)
            key = f"{ref_pix:.3f} pix" + (f' = {ref_pix * pscale:.3f}"' if pscale else "")
            # quiverkey draws along x, whose data unit is RA degrees
            ax.quiverkey(
                q, 1.0 - arrow_frac, 1.02, arrow_frac * span / cosd, key,
                labelpos="W", coordinates="axes", color="k", fontproperties={"size": 9},
            )
        ax.set_aspect(1.0 / cosd)
        ax.invert_xaxis()
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(
            f"{self.run_config.name if getattr(self, 'run_config', None) else 'shift field'}"
            f": {len(ids)} scenes, {len(xy)} samples, "
            f"median {np.median(mag):.3f} pix, max {mag.max():.3f} pix",
            loc="left",
        )
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=180, bbox_inches="tight")
        return fig, ax

    # -- step 4: outputs ---------------------------------------------------
    def write_outputs(self) -> "Pipeline":
        """Write residual FITS, fit table, and scene diagnostics."""
        from astropy.io import fits

        if self.table is None:
            raise RuntimeError("run() first")
        cfg = self.run_config
        stem = self.out_dir / cfg.name
        # residual is on the hi-res reference grid (upsample path)
        with _quiet_hierarch_warnings():
            if isinstance(self.residuals[0], np.memmap):
                # already written: run() accumulated straight into the file's
                # data section (see _residual_memmap), so only the pages still
                # held by the kernel are outstanding
                self.residuals[0].flush()
                logger.info("residual flushed to %s", self.f_residual.name)
            else:
                fits.writeto(
                    self.f_residual,
                    self.residuals[0],
                    fits.getheader(cfg.sci_hi),
                    overwrite=True,
                )
            self.table.write(self.f_fit_table, overwrite=True)

            # per-template fit state: everything the solve produced that a
            # rebuild cannot re-derive (component amplitudes, astrometric
            # shifts, scenes)
            if getattr(self, "all_templates", None):
                self._template_fit_table().write(self.f_templates, overwrite=True)

        scene_dir = self.out_dir / "scenes"
        if cfg.scene_plots and self.scenes:
            scene_dir.mkdir(parents=True, exist_ok=True)

        # Saturated stars' segment ids: kept out of the display scale of every
        # OTHER scene's image panel (their brightness would dominate it) and
        # nulled in that scene's residual panel; see Scene.plot.
        sat_ids = [
            int(t.id)
            for s in self.scenes
            for t in s.templates
            if getattr(t, "is_saturated", False)
        ]
        rows = []
        for s in self.scenes:
            xy = np.mean([t.position_original for t in s.templates], axis=0)
            ra, dec = self.wcs[0].wcs_pix2world([xy], 0)[0]
            # total applied shift at the scene center, NaN where none was fitted
            sampled = self._scene_shift_samples(s, at=xy)
            dx, dy = sampled[1][0] if sampled is not None else (np.nan, np.nan)
            rows.append(
                (s.id, len(s.templates), int(s.is_bright.sum()), ra, dec,
                 float(dx), float(dy),
                 int(s.astrom_niter),
                 -1 if s.astrom_converged is None else int(not s.astrom_converged),
                 float(s.astrom_step if s.astrom_step is not None else np.nan))
            )
            if cfg.scene_plots:
                import matplotlib.pyplot as plt

                fig, _ = s.plot(
                    self.images[0], self.segmap, display_sig=5,
                    null_segments=sat_ids,
                )
                fig.savefig(scene_dir / f"{cfg.name}_scene_{s.id}.png", dpi=300)
                plt.close(fig)
        scene_table = Table(
            rows=rows,
            names=["id", "n_templates", "is_bright", "ra", "dec", "dx", "dy",
                   "astrom_niter", "flag_astrom", "astrom_step"],
        )
        viewer = cfg.minerva_viewer
        if viewer is None:
            viewer = f"{str(cfg.name).split('_')[0].lower()}/{cfg.minerva_release}"
        if viewer:
            scene_table["minerva_link"] = [
                f"https://minerva.colorado.edu/{viewer.strip('/')}/"
                f"?ra={ra:.7f}&dec={dec:.7f}&zoom=7"
                for ra, dec in zip(scene_table["ra"], scene_table["dec"])
            ]
        scene_table.write(
            f"{stem}_scene_catalog.csv", format="ascii.csv", overwrite=True
        )

        # Two full-field views of the partition, answering different
        # questions, side by side in one figure and sharing one colour per
        # scene. The left panel paints every segment with the colour of the
        # scene that fitted it, so it says which source went where; it reads
        # the mosaic and the segmap to do it. The right draws each scene as
        # the hull of its templates with its id, so it says where the scenes
        # are and how big they got -- pure vector, no raster, no decimation.
        if cfg.scene_plots and self.scenes:
            from .verification import save_scene_partition

            save_scene_partition(
                self.images[0], self.segmap, self.scenes,
                f"{stem}_scenes.png",
            )

        # shift field: only exists when astrometry was actually solved
        out = self.plot_shift_field(save=f"{stem}_shift_field.png")
        if out is not None:
            import matplotlib.pyplot as plt

            plt.close(out[0])

        # Everything that reads a scene's band pixels has now run. The stamps
        # come last on purpose: writing them is the run's other memory peak
        # (two full template sets, plus one stamp in flight), and the band
        # weight map -- a full reference-grid array on the upsample path -- is
        # dead from the moment the fluxes were solved. Releasing it here keeps
        # it out of that peak. run() does the same release earlier for runs
        # that plot nothing at all.
        self._release_scene_weights()
        if cfg.save_stamps:
            with _quiet_hierarch_warnings():
                self.write_stamps()

        logger.info("outputs written to %s", self.out_dir)
        return self

    def _stamps_header(self, ifilt: int, nsrc: int) -> "fits.Header":
        """Primary header of a stamps file: minimal pointers, no duplication.

        Everything else already has its own save data — the run config in its
        JSON file, the PSF/kernel maps in ``<name>_*.geojson``, WCS and grids
        in the input images, fit results in ``<name>_fit_table.fits``.  The
        header only names the run (to locate those files) and records the grid
        shapes :meth:`load_fit` uses to reject a stale stamps file.
        """
        from astropy.io import fits

        hdr = fits.Header()
        hdr["NSRC"] = (nsrc, "number of SOURCES rows")
        hdr["IFILT"] = (ifilt, "fitted image index of the *_lo columns")
        if getattr(self, "run_config", None) is not None:
            hdr["RUNNAME"] = (
                self.run_config.name,
                "run whose json/geojson save data these stamps use",
            )
        ny, nx = self.images[0].shape
        hdr["NX_HI"] = (nx, "reference-grid width [pix]")
        hdr["NY_HI"] = (ny, "reference-grid height [pix]")
        ny, nx = self.images[ifilt].shape
        hdr["NX_LO"] = (nx, "fitting-grid width [pix]")
        hdr["NY_LO"] = (ny, "fitting-grid height [pix]")
        return hdr

    def _band_psfs(
        self, ifilt: int
    ) -> tuple[np.ndarray | PSFRegionMap | None, np.ndarray | PSFRegionMap | None]:
        """The hi/lo PSF inputs (map or static array) for band ``ifilt``.

        Falls back to the cached hi-res geojson map for config-driven runs
        whose in-memory ``psfs[0]`` is unset.
        """
        psfs = self.psfs if self.psfs is not None else []
        psf_hi = psfs[0] if len(psfs) > 0 else None
        if psf_hi is None:
            psf_hi = getattr(self, "prm_hi", None)
        if (
            psf_hi is None
            and getattr(self, "run_config", None) is not None
            and self.f_psf_hi.exists()
        ):
            psf_hi = self.prm_hi = PSFRegionMap.from_geojson(str(self.f_psf_hi))
        psf_lo = psfs[ifilt] if len(psfs) > ifilt else None
        return psf_hi, psf_lo

    def write_stamps(
        self,
        path: str | os.PathLike | None = None,
        *,
        ifilt: int = 1,
    ) -> Path:
        """Write the per-source template stamps to one FITS file.

        Stamps keep their native, per-source sizes: the ``SOURCES`` binary
        table stores each template flattened in a variable-length array column
        (``tmpl_hi``, ``tmpl_lo``) next to its shape, grid origin, and source
        position, so nothing is padded to a common size.  Data that already
        have their own save files are not duplicated here: PSFs stay in the
        cached ``<name>_psf_*.geojson`` region maps and each source only
        carries its region key (``key_psf_hi``/``key_psf_lo``; 0 for a static
        PSF, -1 when the band has none), the run/fit configuration stays in
        the run's JSON, and the primary header holds just the pointers and
        grid shapes (:meth:`_stamps_header`).  Together with the fit table
        and residual this file restores the post-run state via
        :meth:`load_fit`; :meth:`read_stamps` gets the stamps back as 2D
        arrays.

        ``SOURCES`` columns:

        - ``id, x, y``: source id and reference-grid position
        - ``flux, err``: fitted amplitude and uncertainty
        - ``tmpl_hi, ny_hi, nx_hi, x0_hi, y0_hi, xs_hi, ys_hi``: hi-res
          template pixels (flattened, reshape to ``(ny, nx)``), the
          original-grid pixel of ``data[0, 0]`` (``x0, y0``), and the source
          position on that grid (``xs, ys``)
        - ``tmpl_lo, ny_lo, nx_lo, x0_lo, y0_lo, xs_lo, ys_lo``: same for the
          convolved template on the fitting grid
        - ``key_psf_hi, key_psf_lo``: psf_key into the band's PSF region map
        - ``flag_hi, flag, id_parent, id_scene, ee_psf_lo, ee_tmpl,
          err_pred, shift_x, shift_y``: per-template fit metadata, restored
          by :meth:`load_fit`

        Args:
            path: Output file.  Defaults to ``<out_dir>/<name>_stamps.h5``
                for config-driven runs.
            ifilt: Fitted image index (1-based, as elsewhere).

        Returns:
            Path of the written file.
        """
        from astropy.io import fits

        if not getattr(self, "all_templates", None):
            raise RuntimeError("run() first")
        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        if path is None:
            if getattr(self, "run_config", None) is None:
                raise ValueError("path is required when not running from a config")
            path = self.out_dir / f"{self.run_config.name}_stamps.h5"
        path = Path(path)
        if path.suffix.lower() in (".fits", ".fit"):
            # stamps are HDF5 now; a .fits name on an HDF5 file would be a
            # trap for anything that opens it by extension
            path = path.with_suffix(".h5")
            logger.info("stamps are written as HDF5; using %s", path.name)

        conv = self.all_templates[ifilt - 1]
        if not conv:
            raise RuntimeError("no fitted templates to write stamps for")
        hi_by_id = {int(t.id): t for t in self.tmpls.templates}

        psf_hi, psf_lo = self._band_psfs(ifilt)
        for band, p in (("hi", psf_hi), ("lo", psf_lo)):
            if p is None:
                logger.warning("no %s-res PSF available; its stamp key is -1", band)

        def psf_key(
            p: np.ndarray | PSFRegionMap | None, ra: float | None, dec: float | None
        ) -> int:
            if p is None:
                return -1
            if isinstance(p, PSFRegionMap):
                key = p.lookup_key(ra, dec) if ra is not None and dec is not None else None
                return int(key) if key is not None else 0
            return 0

        wcs_hi = self.wcs[0] if self.wcs is not None else None
        rows: dict[str, list] = defaultdict(list)
        for t_lo in conv:
            t_hi = hi_by_id.get(int(t_lo.id))
            if t_hi is None:
                logger.warning(
                    "no hi-res template for source id %d; tmpl_hi is empty", int(t_lo.id)
                )
            src = t_hi if t_hi is not None else t_lo
            x, y = src.input_position_original
            ra = dec = None
            if wcs_hi is not None and t_hi is not None:
                ra, dec = (float(v) for v in wcs_hi.wcs_pix2world(x, y, 0))
            for tag, t in (("hi", t_hi), ("lo", t_lo)):
                if t is None:
                    shape = (0, 0)
                    x0 = y0 = -1
                    xs = ys = np.nan
                else:
                    # shape only: the pixels are written in the second pass
                    # below, straight into their slot in the dataset
                    shape = t.data.shape
                    # data[0, 0] sits at this original-grid pixel (may be
                    # negative for cutouts padded past the image edge)
                    x0, y0 = (int(v) for v in t._origin_original_true)
                    xs, ys = (float(v) for v in t.input_position_original)
                rows[f"ny_{tag}"].append(shape[0])
                rows[f"nx_{tag}"].append(shape[1])
                rows[f"x0_{tag}"].append(x0)
                rows[f"y0_{tag}"].append(y0)
                rows[f"xs_{tag}"].append(xs)
                rows[f"ys_{tag}"].append(ys)
            flux = getattr(t_lo, "flux", None)
            err = getattr(t_lo, "err", None)
            rows["id"].append(int(t_lo.id))
            rows["x"].append(float(x))
            rows["y"].append(float(y))
            rows["flux"].append(float(flux) if flux is not None else np.nan)
            rows["err"].append(float(err) if err is not None else np.nan)
            rows["key_psf_hi"].append(psf_key(psf_hi, ra, dec))
            rows["key_psf_lo"].append(psf_key(psf_lo, ra, dec))
            # per-template fit metadata, so load_fit restores the full state
            rows["flag_hi"].append(int(getattr(t_hi, "flag", 0)) if t_hi is not None else 0)
            rows["flag"].append(int(getattr(t_lo, "flag", 0)))
            rows["id_parent"].append(int(getattr(t_lo, "id_parent", None) or t_lo.id))
            rows["id_scene"].append(int(getattr(t_lo, "id_scene", 1)))
            rows["ee_psf_lo"].append(float(getattr(t_lo, "ee_psf_lo", np.nan)))
            rows["ee_tmpl"].append(float(getattr(t_lo, "ee_tmpl", np.nan)))
            rows["err_pred"].append(float(getattr(t_lo, "err_pred", np.nan)))
            shift = np.asarray(getattr(t_lo, "shifted", (0.0, 0.0)), dtype=float)
            rows["shift_x"].append(float(shift[0]))
            rows["shift_y"].append(float(shift[1]))

        # One flat float32 buffer per band plus per-source offsets, rather
        # than one dataset per source: 138,610 HDF5 datasets would spend more
        # on object headers than on pixels, and the ragged concatenation is
        # what both readers want anyway.
        #
        # Offsets come from the shapes recorded above, so each dataset is
        # created at its final size and every stamp is written straight into
        # its own slot. Collecting the flattened stamps in a list and
        # concatenating held a full extra copy of every stamp -- 12 GB on a
        # MINERVA field -- at the very end of the run, on top of the two
        # template sets that are still alive here.
        hdr = self._stamps_header(ifilt, len(conv))
        offs = {}
        for tag in ("hi", "lo"):
            sizes = (np.asarray(rows[f"ny_{tag}"], dtype=np.int64)
                     * np.asarray(rows[f"nx_{tag}"], dtype=np.int64))
            offs[tag] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        npix = int(offs["hi"][-1]) + int(offs["lo"][-1])
        with h5py.File(path, "w") as h5:
            for key in hdr:
                if key in ("COMMENT", "HISTORY", ""):
                    continue
                value = hdr[key]
                h5.attrs[key] = value if isinstance(value, (int, float, str)) else str(value)
            pixels = {}
            for tag in ("hi", "lo"):
                grp = h5.create_group(f"tmpl_{tag}")
                total = int(offs[tag][-1])
                # chunks bounded independently of stamp size: a 1 Mi-element
                # chunk is 4 MiB, and a stamp spans at most a few of them
                # uncompressed: gzip-1 bought 26% on a full field (11.2 ->
                # 8.3 GiB) for 206 s of the 1112 s run, and these files are
                # working products read back by the diagnostics, not archive.
                # Chunked anyway, so a single-source read still touches only
                # its own chunks.
                pixels[tag] = grp.create_dataset(
                    "pixels", shape=(total,), dtype="f4",
                    chunks=(min(1 << 20, max(total, 1)),),
                )
                grp.create_dataset("offset", data=offs[tag], dtype="i8")
            for i, t_lo in enumerate(conv):
                t_hi = hi_by_id.get(int(t_lo.id))
                for tag, t in (("hi", t_hi), ("lo", t_lo)):
                    o0, o1 = int(offs[tag][i]), int(offs[tag][i + 1])
                    if o1 > o0:
                        pixels[tag][o0:o1] = np.asarray(
                            t.data, dtype=np.float32
                        ).ravel()
            src = h5.create_group("sources")
            for name, values in rows.items():
                arr = np.asarray(values)
                if arr.dtype.kind not in "iuf":
                    arr = arr.astype(np.float64)
                src.create_dataset(name, data=arr)
        logger.info(
            "wrote %d sources (%s of template pixels, %s on disk) to %s",
            len(conv), human_bytes(npix * 4),
            human_bytes(Path(path).stat().st_size), path,
        )
        return path

    @staticmethod
    def read_stamps(path: str | os.PathLike) -> list[dict]:
        """Read a :meth:`write_stamps` file back into per-source dicts.

        Each dict holds the scalar ``SOURCES`` columns plus the 2D
        ``tmpl_hi``/``tmpl_lo`` arrays.  PSF stamps are not stored in the
        file; ``key_psf_hi``/``key_psf_lo`` index ``psfs`` of the band's
        cached PSF region map (``<name>_psf_*.geojson``).
        """
        path = Path(path)
        if h5py.is_hdf5(path):
            return Pipeline._read_stamps_h5(path)
        # a stamps file written before the HDF5 switch
        logger.info("reading legacy FITS stamps %s", path.name)
        return Pipeline._read_stamps_fits(path)

    @staticmethod
    def _read_stamps_h5(path: Path) -> list[dict]:
        out: list[dict] = []
        with h5py.File(path, "r") as h5:
            src = h5["sources"]
            cols = {name: src[name][:] for name in src}
            pix = {tag: h5[f"tmpl_{tag}"]["pixels"] for tag in ("hi", "lo")}
            off = {tag: h5[f"tmpl_{tag}"]["offset"][:] for tag in ("hi", "lo")}
            n = len(next(iter(cols.values()))) if cols else 0
            for i in range(n):
                rec = {name: values[i] for name, values in cols.items()}
                for tag in ("hi", "lo"):
                    flat = pix[tag][off[tag][i] : off[tag][i + 1]]
                    rec[f"tmpl_{tag}"] = np.asarray(flat, dtype=np.float32).reshape(
                        int(rec[f"ny_{tag}"]), int(rec[f"nx_{tag}"])
                    )
                out.append(rec)
        return out

    @staticmethod
    def _read_stamps_fits(path: Path) -> list[dict]:
        from astropy.io import fits

        out: list[dict] = []
        with fits.open(path) as hdul:
            data = hdul["SOURCES"].data
            for row in data:
                rec = {
                    name: row[name]
                    for name in data.names
                    if not name.startswith("tmpl_")
                }
                for tag in ("hi", "lo"):
                    rec[f"tmpl_{tag}"] = np.array(
                        row[f"tmpl_{tag}"], dtype=np.float32
                    ).reshape(int(row[f"ny_{tag}"]), int(row[f"nx_{tag}"]))
                out.append(rec)
        return out

    def _templates_from_stamps(self, path: Path, ifilt: int) -> None:
        """Rebuild ``tmpls`` and ``all_templates`` from a stamps file.

        Applies :meth:`run`'s grid transform (image upsampling) first, then
        reconstructs every template with :meth:`Template.from_stamp` and
        restores its fit metadata from the ``SOURCES`` columns.
        """
        from astropy.io import fits

        config = self.config
        # replicate run()'s pre-fit grid transform so shapes and coordinates
        # match the templates the stamps were written from
        k = bin_factor_from_wcs(self.wcs[0], self.wcs[ifilt]) if self.wcs is not None else 1
        self.fit_bin_factors.append(int(k))
        if k > 1 and config.multi_resolution_method == "upsample":
            logger.info("upsampling image %d by factor %d", ifilt, k)
            self.images[ifilt], _ = _upsample_boxed(
                self.images[ifilt], None, k, self.trial_box_lo
            )
            self.wcs[ifilt] = self.wcs[0]

        shape_hi = self.images[0].shape
        shape_lo = self.images[ifilt].shape
        wcs_hi = self.wcs[0] if self.wcs is not None else None
        wcs_lo = self.wcs[ifilt] if self.wcs is not None else None

        # header/attrs and rows through the format-dispatching readers, so an
        # HDF5 file and a pre-switch FITS one both restore the same state
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as h5:
                hdr = dict(h5.attrs)
        else:
            hdr = dict(fits.getheader(path))
        src = Pipeline.read_stamps(path)
        if True:
            if int(hdr.get("IFILT", ifilt)) != int(ifilt):
                raise ValueError(
                    f"stamps file {path.name} was written for ifilt={hdr['IFILT']}"
                )
            if (hdr.get("NX_HI"), hdr.get("NY_HI")) != (shape_hi[1], shape_hi[0]) or (
                hdr.get("NX_LO"), hdr.get("NY_LO")
            ) != (shape_lo[1], shape_lo[0]):
                raise ValueError(
                    f"stamps file {path.name} grids do not match the loaded "
                    "images; stale file? Delete it to regenerate."
                )
            buf_hi = np.zeros(shape_hi, dtype=np.float32)
            buf_lo = np.zeros(shape_lo, dtype=np.float32)
            hi_templates: list[Template] = []
            lo_templates: list[Template] = []
            for row in src:
                sid = int(row["id"])
                ny, nx = int(row["ny_hi"]), int(row["nx_hi"])
                if ny and nx:
                    t_hi = Template.from_stamp(
                        np.array(row["tmpl_hi"], dtype=np.float32).reshape(ny, nx),
                        (int(row["x0_hi"]), int(row["y0_hi"])),
                        (float(row["xs_hi"]), float(row["ys_hi"])),
                        shape_hi,
                        wcs=wcs_hi,
                        label=sid,
                        parent_image=buf_hi,
                    )
                    t_hi.flag = int(row["flag_hi"])
                    hi_templates.append(t_hi)
                ny, nx = int(row["ny_lo"]), int(row["nx_lo"])
                t_lo = Template.from_stamp(
                    np.array(row["tmpl_lo"], dtype=np.float32).reshape(ny, nx),
                    (int(row["x0_lo"]), int(row["y0_lo"])),
                    (float(row["xs_lo"]), float(row["ys_lo"])),
                    shape_lo,
                    wcs=wcs_lo,
                    label=sid,
                    parent_image=buf_lo,
                )
                t_lo.flux = float(row["flux"])
                t_lo.err = float(row["err"])
                t_lo.err_pred = float(row["err_pred"])
                t_lo.flag = int(row["flag"])
                t_lo.id_parent = int(row["id_parent"])
                t_lo.id_scene = int(row["id_scene"])
                t_lo.ee_psf_lo = float(row["ee_psf_lo"])
                t_lo.ee_tmpl = float(row["ee_tmpl"])
                t_lo.shifted = np.array(
                    [float(row["shift_x"]), float(row["shift_y"])], dtype=float
                )
                lo_templates.append(t_lo)

        tmpls = Templates()
        tmpls.original_shape = shape_hi
        tmpls.wcs = wcs_hi
        tmpls._templates = hi_templates
        self.tmpls = tmpls
        # stamps carry the post-extension pixels only, so both build stages are
        # the same object here (see _prepare_hi_templates)
        self.templates_extracted = tmpls
        self.templates_extended = tmpls
        self.templates = lo_templates
        self.all_templates = [lo_templates]
        logger.info(
            "restored %d hi-res + %d fitted templates from %s",
            len(hi_templates), len(lo_templates), path.name,
        )

    def load_fit(self, ifilt: int = 1) -> "Pipeline":
        """Restore the post-run state from written outputs without refitting.

        Counterpart of :meth:`load_data` (pre-run state): reads
        ``<name>_fit_table.fits`` and ``<name>_residual.fits`` written by
        :meth:`write_outputs`, rebuilds the fitted templates from
        ``<name>_stamps.h5``, and recreates the derived state (grid
        upsampling, model image) so the instance matches a completed
        :meth:`run`.  When the stamps file is missing it is regenerated
        through the same template path :meth:`run` uses — fluxes then come
        from the fit table — and written back to disk.

        Not restored: ``all_scenes`` (scene objects are not persisted; scene
        membership survives as ``id_scene``), and the pre-extension pixels of
        ``templates_extracted`` when loading from a stamps file (it then
        equals ``templates_extended``).  Regeneration reapplies the fitted
        per-template amplitudes and astrometric shifts from
        ``<name>_templates.fits`` when present; without it, fluxes come from
        the fit table and the rebuild is exact only for runs that applied no
        shifts.

        Args:
            ifilt: Fitted image index (1-based, as elsewhere).

        Returns:
            self, in the post-run state.
        """
        if getattr(self, "run_config", None) is None:
            raise RuntimeError("load_fit requires a config-driven pipeline")
        if not self.f_fit_table.exists() or not self.f_residual.exists():
            raise FileNotFoundError(
                f"run outputs not found under {self.out_dir}; expected "
                f"{self.f_fit_table.name} and {self.f_residual.name} — run() "
                "and write_outputs() first"
            )
        # data first: load_data() finishes construction via __init__, which
        # resets the product lists that load_outputs() fills
        if self.images is None:
            self.load_data()
        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        config = self.config
        self.load_outputs()
        residual = np.asarray(self.residuals[0], dtype=np.float32)

        self.fit_bin_factors = []
        self.all_scenes = []
        # prefer HDF5, fall back to a stamps file written before the switch
        stem = self.out_dir / f"{self.run_config.name}_stamps"
        f_stamps = next(
            (p for p in (stem.with_suffix(".h5"), stem.with_suffix(".fits")) if p.exists()),
            stem.with_suffix(".h5"),
        )
        if f_stamps.exists():
            self._templates_from_stamps(f_stamps, ifilt)
        else:
            logger.warning(
                "stamps file %s not found; regenerating templates through the "
                "run() template path", f_stamps.name,
            )
            cat = self._fit_catalog(config)
            self._prepare_hi_templates(cat, config)
            with self._phase("convolve templates"):
                templates, weights_i = self._convolved_templates(ifilt, config)
            # fitted amplitudes/errors/shifts: the saved per-template table is
            # exact (per component, pre-aggregation); the fit table is the
            # fallback for runs that predate it
            ttab = getattr(self, "template_table", None)
            if ttab is not None:
                by_id = {int(i): j for j, i in enumerate(ttab["id"])}
                for t in templates:
                    row = by_id.get(int(t.id))
                    if row is None:
                        continue
                    t.flux = float(ttab["flux"][row])
                    t.err = float(ttab["err"][row])
                    t.id_scene = int(ttab["id_scene"][row])
                    t.to_shift = np.array(
                        [float(ttab["dx"][row]), float(ttab["dy"][row])], dtype=float
                    )
                Templates.apply_template_shifts(templates)
            else:
                flux_col, err_col = f"flux_{ifilt}", f"err_{ifilt}"
                by_id = {int(i): j for j, i in enumerate(self.table["id"])}
                for t in templates:
                    row = by_id.get(int(t.id))
                    if row is None:
                        continue
                    if flux_col in self.table.colnames:
                        t.flux = float(self.table[flux_col][row])
                    if err_col in self.table.colnames:
                        t.err = float(self.table[err_col][row])
            # populate err_pred the same way run() does
            Templates.predicted_errors(templates, weights_i)
            self.all_templates = [templates]
            self.write_stamps(ifilt=ifilt)

        if self.images[ifilt].shape != residual.shape:
            raise ValueError(
                f"residual shape {residual.shape} does not match "
                f"image shape {self.images[ifilt].shape}"
            )
        self.residuals = [residual]
        self.model_images = _ModelImages(self)
        logger.info("post-run state restored from %s", self.out_dir)
        return self

    @contextmanager
    def log_run(self, path: str | Path | None = None):
        """Capture everything the run emits into ``<out_dir>/<name>.log``.

        Both ``logging`` records and bare ``print``/``tqdm`` output go to the
        file, since the package emits through both. The console is unchanged.
        Appends, so successive runs against one output directory accumulate
        rather than overwrite.

        Three channels are captured for the duration of the block: all
        library loggers (a file handler on the root logger, so astropy,
        drizzlepac, and stpsf records reach the file even when the caller
        configured logging first; the root level is raised to INFO if unset
        or higher), ``warnings.warn`` messages (via
        ``logging.captureWarnings``, reset first so an earlier hook does not
        shadow it), and teed stdout/stderr.  Each entry starts with a header
        recording the run name, timestamp, Python version, platform, and
        output directory, and ends with the elapsed time; if the block
        raises, a ``FAILED after <t>s`` line is written and the exception
        propagates.  All logging state is restored on exit.

        Args:
            path: Log file; parent directories are created. Defaults to
                ``<out_dir>/<name>.log`` for config-driven runs.

        Yields:
            Path of the log file.
        """
        import platform
        import sys
        import time
        import warnings

        path = Path(path) if path is not None else self.out_dir / f"{self.run_config.name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reentrant: `main` wraps whatever steps were named, and one of those
        # steps may be `all`, which opens its own block. Nesting would tee
        # every line twice and stack two handlers on the root logger, so an
        # inner call just hands back the path the outer one is already using.
        if getattr(self, "_log_run_path", None) is not None:
            yield self._log_run_path
            return
        self._log_run_path = path
        handle = open(path, "a", buffering=1)
        started = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"\n{'=' * 78}\nmophongo run {self.run_config.name}  {stamp}\n")
        handle.write(f"python {platform.python_version()} on {platform.platform()}\n")
        handle.write(f"out_dir {self.out_dir}\n{'=' * 78}\n")

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, handle), _Tee(old_err, handle)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
        )
        # Everything, not just mophongo: astropy, drizzlepac and stpsf log
        # through their own loggers, and a caller that ran basicConfig before
        # this point holds a handler bound to the *original* stderr, so those
        # records would never reach the file. One handler on the root logger
        # catches the lot, and captureWarnings routes warnings.warn there too.
        root = logging.getLogger()
        # remember whether a console handler existed before ours: if it did
        # (e.g. main()'s basicConfig), the package handler below must not be
        # added or every record would print twice on the console
        had_console_handler = bool(root.handlers)
        file_handler = logging.StreamHandler(handle)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        old_root_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        old_showwarning = warnings.showwarning
        # Long keywords from the input catalog's meta (PHOT_UNIT,
        # WEBBSTARFILT, ...) round-trip as HIERARCH cards by design; the
        # per-card VerifyWarning about it is pure noise at run scale. The write
        # paths filter these themselves as well, for the step-at-a-time
        # invocations that never enter this block.
        old_filters = warnings.filters[:]
        _ignore_hierarch_warnings()
        # captureWarnings is a latch: if some earlier code in this process left
        # it on, another True call is a no-op and whatever hook is currently
        # installed (e.g. pytest's) stays in place. Reset it first so the
        # logging hook is installed now, over the current showwarning.
        logging.captureWarnings(False)
        logging.captureWarnings(True)

        # Console output for package records, but only when nothing else is
        # consuming them. It writes to the real stdout rather than the tee:
        # the file copy already arrives through the root handler above, and
        # teeing here would duplicate every line in the file.
        pkg = logging.getLogger("mophongo")
        handler = None
        if not pkg.handlers and not had_console_handler:
            handler = logging.StreamHandler(old_out)
            handler.setFormatter(fmt)
            pkg.addHandler(handler)
            if pkg.level == logging.NOTSET:
                pkg.setLevel(logging.INFO)
        try:
            yield path
        except BaseException as exc:
            handle.write(f"FAILED after {time.time() - started:.1f}s: "
                         f"{type(exc).__name__}: {exc}\n")
            raise
        else:
            handle.write(f"finished in {time.time() - started:.1f}s\n")
        finally:
            if handler is not None:
                pkg.removeHandler(handler)
            root.removeHandler(file_handler)
            root.setLevel(old_root_level)
            # captureWarnings(False) first: it clears logging's internal "already
            # capturing" flag. Restoring showwarning alone leaves that flag set,
            # and the *next* run's captureWarnings(True) silently does nothing.
            logging.captureWarnings(False)
            warnings.showwarning = old_showwarning
            warnings.filters[:] = old_filters
            self._log_run_path = None
            sys.stdout, sys.stderr = old_out, old_err
            handle.close()

    @contextmanager
    def _phase(self, name: str):
        """Accumulate wall time under ``name`` for the end-of-run breakdown.

        Re-entering a name adds to it, so a phase inside a per-band loop
        reports its total across bands rather than the last one.
        """
        if not hasattr(self, "_timings"):
            self._timings: dict[str, float] = {}
        if name.startswith("step: "):
            # a caller is partitioning the whole invocation and will report
            # once at the end; run() must not report its own half-finished view
            self._cli_stepping = True
        started = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - started
            self._timings[name] = self._timings.get(name, 0.0) + dt

    def report_timings(self, total: float | None = None) -> None:
        """Log the per-section wall-clock breakdown, longest first."""
        timings = getattr(self, "_timings", None)
        if not timings:
            return
        # Two levels, reported separately: CLI steps partition the run, while
        # the fit's own phases sit inside one of them. Listing both in one
        # table would double-count and the percentages would exceed 100.
        steps = {k[len("step: "):]: v for k, v in timings.items() if k.startswith("step: ")}
        inner = {k: v for k, v in timings.items() if not k.startswith("step: ")}
        total = total if total is not None else sum(steps.values()) or sum(inner.values())
        width = max(len(k) for k in list(steps) + list(inner))

        def _table(title: str, items: dict[str, float], denom: float) -> None:
            if not items:
                return
            logger.info("%s (%s):", title, _fmt_hms(denom))
            for name, dt in sorted(items.items(), key=lambda kv: -kv[1]):
                logger.info("  %-*s  %9s  %5.1f%%", width, name,
                            _fmt_hms(dt), 100.0 * dt / max(denom, 1e-9))
            rest = denom - sum(items.values())
            if rest > 0.02 * denom:
                logger.info("  %-*s  %9s  %5.1f%%", width, "(other)",
                            _fmt_hms(rest), 100.0 * rest / max(denom, 1e-9))

        _table("time by step", steps, total)
        _table("time within the fit", inner, steps.get("fit", sum(inner.values())))

    def build_repair_cache(self) -> "Pipeline":
        """Run the saturation repair, write its cache, and stop before the fit.

        A campaign submits a field's bands together, and each would otherwise
        re-run the same repair: it depends on the detection image, its weight,
        ``pattern_hi``, the halo pattern, ``repair_kwargs`` and the trial box
        (:meth:`_repair_provenance`), and on nothing that varies between bands.
        Running it once per field turns that duplicated work into a cache hit,
        and removes the concurrent writes to a shared cache file that come with
        submitting the bands together.

        ``kernels=False`` stops :meth:`load_data` building a matching-kernel
        map, which is per-band and not shared.

        Needs the detection-band and halo ePSF grids to exist already. On a
        cluster whose compute nodes have no route to MAST — OzStar — build them
        on the login node first (``examples/ozstar/jobs/build_psfs.sh``); where
        compute has internet, :meth:`prep` does both in one job.
        """
        cfg = self.run_config
        if cfg is None:
            raise RuntimeError("build_repair_cache needs a config-driven run")
        if not cfg.repair_saturated:
            logger.info("repair_saturated is off: no cache to build")
            return self
        with self.log_run() as log_path:
            logger.info("logging this repair to %s", log_path)
            self.load_data(kernels=False)
        return self

    def prep(self) -> "Pipeline":
        """Build what every band of this field shares, and stop.

        :meth:`build_psfs` — which builds or reloads the ePSF grids and writes
        this band's PSF region maps, the ones ``seed_cache.sh`` links into the
        other bands' run directories — followed by
        :meth:`build_repair_cache`.

        One job, so this suits a cluster whose compute nodes can reach MAST.
        Where they cannot, run the two halves on the machines that can: grids
        on the login node, then the ``repair`` step as an ordinary job.
        """
        self.build_psfs()
        return self.build_repair_cache()

    def run_all(self) -> "Pipeline":
        """All steps in order: psfs, kernels, fit, outputs.

        Everything the run emits is also written to ``<out_dir>/<name>.log``.
        """
        with self.log_run() as log_path:
            logger.info("logging this run to %s", log_path)
            self.build_psfs()
            self.build_kernels()
            self.run()
            self.write_outputs()
        return self

    def _update_catalog_with_fluxes(
        self,
        cat: Table,
        templates: list[Template],
        fluxes: np.ndarray,
        errs: np.ndarray,
        err_pred: np.ndarray,
        throughput: float,
        idx: int,
        scene_ids: np.ndarray | list[int] | None = None,
        scene_flags: dict[int, int] | None = None,
    ) -> None:
        """Insert measured fluxes into the output catalog.

        Parameters
        ----------
        cat
            Catalog to update.
        templates
            Templates associated with the fitted sources.
        fluxes, errs, err_pred
            Model-template flux measurements and their uncertainties.
        throughput
            Filter-level finite-support PSF sum, used only for templates whose
            per-source ``ee_psf_lo`` is missing.  Total-flux columns divide
            model fluxes/errors by the encircled energy of the low-resolution
            PSF stamp at each source position, which is the one factor between
            the fitted amplitude and a total flux.  A value of 1 applies no
            missing-PSF-support correction.
        idx
            Index of the current image (used for column naming).
        scene_ids
            Scene id of each template, in ``templates`` order. Written as
            ``scene_<idx>`` so scene membership survives the run: scenes are not
            persisted otherwise, and without it ``load_fit`` cannot rebuild them
            and the scene diagnostics can only be regenerated by refitting.
            Sources with no template keep -1.
        scene_flags
            Scene-level astrometry verdict per scene id, inherited by every
            source fitted in that scene and written as ``flag_astrom_<idx>``
            (0 converged, 1 still moving when the pass budget ran out).
            Sources with no template, and scenes with no verdict, keep -1.
        """

        parent_ids = [
            tmpl.id_parent if getattr(tmpl, "parent_id", None) is not None else tmpl.id
            for tmpl in templates
        ]
        id_to_index = {id_: i for i, id_ in enumerate(cat["id"])}
        cat[f"flux_{idx}"] = self.config.bad_value
        cat[f"err_{idx}"] = self.config.bad_value
        cat[f"err_pred_{idx}"] = self.config.bad_value
        cat[f"throughput_{idx}"] = self.config.bad_value
        cat[f"flux_{idx}_total"] = self.config.bad_value
        cat[f"err_{idx}_total"] = self.config.bad_value
        cat[f"err_pred_{idx}_total"] = self.config.bad_value
        cat[f"scene_{idx}"] = -1
        # Scene-level astrometry verdict, inherited by every member source.
        # int16, not int8: astropy writes an int8 column as a FITS logical,
        # and -1 comes back as True.
        cat[f"flag_astrom_{idx}"] = np.full(len(cat), -1, dtype=np.int16)

        if not np.isfinite(throughput) or throughput <= 0.0:
            logger.warning(
                "filter %d: PSF throughput is %r (non-finite or <= 0); applying "
                "NO missing-flux correction (=1). flux_%d_total will equal "
                "flux_%d for sources without a per-source ee_psf_lo.",
                idx, throughput, idx, idx,
            )
            throughput = 1.0
        throughput = float(throughput)

        flux_sum: defaultdict[int, float] = defaultdict(float)
        err_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_sum: defaultdict[int, float] = defaultdict(float)
        flux_total_sum: defaultdict[int, float] = defaultdict(float)
        err_total_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_total_sum: defaultdict[int, float] = defaultdict(float)
        # ee_psf_lo is measured on the drizzled stamp at each source position
        # (PSFRegionMap.get_ee_box, recorded by Templates.convolve_templates).
        # The filter-level mean is the fallback for templates that never saw a
        # PSF map.  ee_tmpl is deliberately not used here: the amplitude does
        # not scale with blanked wing flux (docs/ENCIRCLED_ENERGY.pdf).
        ee_used: list[float] = []
        # first template wins for a parent: deblend children of one source share
        # a scene, so any of them names it
        scene_of_parent: dict[int, int] = {}
        for k, (tmpl, pid, fl, er, ep) in enumerate(
            zip(templates, parent_ids, fluxes, errs, err_pred)
        ):
            if pid is None:
                continue
            if scene_ids is not None and k < len(scene_ids):
                scene_of_parent.setdefault(pid, int(scene_ids[k]))
            ee = getattr(tmpl, "ee_psf_lo", np.nan)
            if not np.isfinite(ee) or ee <= 0.0:
                ee = throughput
            ee = float(ee)
            ee_used.append(ee)
            flux_sum[pid] += fl
            err_sum[pid] = float(np.sqrt(err_sum[pid] ** 2 + er**2))
            err_pred_sum[pid] = float(np.sqrt(err_pred_sum[pid] ** 2 + ep**2))
            flux_total_sum[pid] += fl / ee
            err_total_sum[pid] = float(
                np.sqrt(err_total_sum[pid] ** 2 + (er / ee) ** 2)
            )
            err_pred_total_sum[pid] = float(
                np.sqrt(err_pred_total_sum[pid] ** 2 + (ep / ee) ** 2)
            )
        if ee_used:
            arr = np.asarray(ee_used)
            n_fallback = int(np.sum(arr == throughput))
            logger.info(
                "flux_%d_total divided by ee_psf_lo: median %.5f, range %.5f-%.5f "
                "over %d templates (%d fell back to the filter mean %.5f)",
                idx, float(np.median(arr)), float(arr.min()), float(arr.max()),
                arr.size, n_fallback, throughput,
            )
            # per-source EE missing on some sources is a broken propagation
            # chain (resampling/restore dropped ee_psf_lo), not a normal state:
            # their totals silently degrade to the filter-mean correction.
            if 0 < n_fallback:
                logger.warning(
                    "filter %d: %d of %d templates have no finite ee_psf_lo; "
                    "their flux_%d_total uses the filter-mean EE %.5f instead "
                    "of the per-source value",
                    idx, n_fallback, arr.size, idx, throughput,
                )

        for pid, fl in flux_sum.items():
            ci = id_to_index.get(pid)
            if ci is None:
                continue
            cat[f"flux_{idx}"][ci] = fl
            cat[f"err_{idx}"][ci] = err_sum[pid]
            cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]
            cat[f"flux_{idx}_total"][ci] = flux_total_sum[pid]
            cat[f"err_{idx}_total"][ci] = err_total_sum[pid]
            cat[f"err_pred_{idx}_total"][ci] = err_pred_total_sum[pid]
            cat[f"throughput_{idx}"][ci] = throughput
            sid = scene_of_parent.get(pid, -1)
            cat[f"scene_{idx}"][ci] = sid
            if scene_flags:
                cat[f"flag_astrom_{idx}"][ci] = scene_flags.get(sid, -1)

    # ------------------------------------------------------------------
    # template build scheme (extend_mode) resolution
    # ------------------------------------------------------------------
    #: Accepted constructor-override spellings -> canonical ``extend_mode``.
    _LEGACY_EXTEND_MODES = {
        None: "none",
        "none": "none",
        "default": "psf_wings",
        "psf_wings": "psf_wings",
        "psf": "psf_convolution",
        "psf_convolution": "psf_convolution",
        "psf_model": "psf_model",
        "wren": "wren",
        "classic": "classic",
    }

    def _resolve_extend_mode(self, config: FitConfig) -> str:
        """Return the active build scheme.

        ``Pipeline(extend_mode=...)`` overrides when given (used by tests and
        verification runs); otherwise ``FitConfig.extend_mode`` decides.
        """
        if self.extend_mode_override is not None:
            key = str(self.extend_mode_override).lower()
            if key not in self._LEGACY_EXTEND_MODES:
                raise ValueError(f"Unknown extend_mode {self.extend_mode_override!r}")
            return self._LEGACY_EXTEND_MODES[key]
        mode = str(getattr(config, "extend_mode", "default") or "default").lower()
        mode = EXTEND_MODE_ALIASES.get(mode, mode)
        if mode not in EXTEND_MODES:
            raise ValueError(f"Unknown extend_mode {mode!r}; expected one of {EXTEND_MODES}")
        return mode

    def _extend_scheme_kwargs(self, mode: str, config: FitConfig) -> dict:
        """``extract_templates`` kwargs for the ``'wren'``/``'classic'`` schemes.

        The two reference codes size their stamps globally before extraction.
        ``classic`` needs only the detection PSF; ``wren`` also needs the fill
        radius ``r_fill = max(R_ee, r_aper + kernel_half_width)`` so the
        template covers the measurement aperture plus a convolution margin.
        """
        if mode not in ("psf_wings", "wren", "classic"):
            return {}

        weights = self.weights if self.weights is not None else []
        det_weight = weights[0] if len(weights) > 0 else None
        if det_weight is None and getattr(self, "_det_weight_released", False):
            raise RuntimeError(
                f"extend_mode={mode!r} needs the detection-band weight map, which "
                "run() released after building the templates (it is as large as "
                "the detection mosaic and the fit never reads it again). Build a "
                "fresh Pipeline to fit these data again."
            )
        kwargs: dict = {
            "extend_mode": mode,
            "detection_psf": self._psf_for_template_extension(),
            "detection_weight": det_weight,
        }
        if mode == "psf_wings":
            kwargs["psf_wings"] = template_schemes.PsfWingsParams(
                snrlo_psf=float(config.psf_wings_snrlo),
                blend_p=float(config.psf_wings_blend_p),
                background_only=bool(config.extend_wings_background_only),
                rms=None if config.psf_wings_rms is None else float(config.psf_wings_rms),
            )
            return kwargs
        if mode == "classic":
            kwargs["classic"] = template_schemes.ClassicParams(
                tmpl_snrlo=float(config.classic_tmpl_snrlo),
                rms=None if config.classic_rms is None else float(config.classic_rms),
            )
            return kwargs

        # Detection-grid aperture radius (scalar apertures only; a per-band
        # array leaves it None, which only disables the flux_beyond_aper
        # crowding bookkeeping).
        r_ap = None
        scalar_ap = np.isscalar(config.aperture_diam) and not isinstance(config.aperture_diam, str)
        if scalar_ap:
            if config.aperture_units == "arcsec":
                pscale = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)
                if pscale:
                    r_ap = 0.5 * float(config.aperture_diam) / pscale
            else:
                r_ap = 0.5 * float(config.aperture_diam)

        # Largest matching-kernel effective half-width over the fitted bands
        # (95% encircled radius of |K|, not the zero-padded array size).
        kernel_hw = 0.0
        for kern in (self.kernels or []):
            arr = None
            if isinstance(kern, PSFRegionMap):
                arr = np.asarray(kern.psfs[0], dtype=float) if len(kern.psfs) else None
            elif kern is not None:
                arr = np.asarray(kern, dtype=float)
            if arr is not None and arr.ndim == 2 and np.abs(arr).sum() > 0:
                try:
                    kernel_hw = max(kernel_hw, template_schemes.psf_ee_radius_pix(np.abs(arr), 0.95))
                except ValueError:  # pragma: no cover - degenerate kernel
                    pass

        psf_rep = template_schemes.representative_psf(
            kwargs["detection_psf"], float(config.wren_ee_fraction)
        )
        ee_reach = template_schemes.psf_ee_radius_pix(psf_rep, float(config.wren_ee_fraction))
        r_fill = template_schemes.wren_fill_radius(
            psf_rep,
            ee_fraction=float(config.wren_ee_fraction),
            aperture_radius_pix=r_ap,
            kernel_half_width=kernel_hw,
        )
        logger.info(
            "Template build scheme 'wren': fill radius %.1f pix, PSF-wing reach "
            "%.1f pix @ %.0f%% EE", r_fill, ee_reach, 100.0 * float(config.wren_ee_fraction)
        )
        kwargs["wren"] = template_schemes.WrenParams(
            max_radius_pix=float(r_fill),
            psf_ee_radius_pix=float(ee_reach),
            aperture_radius_pix=None if r_ap is None else float(r_ap),
            ee_fraction=float(config.wren_ee_fraction),
            fit_snrlo_psf=float(config.wren_fit_snrlo_psf),
            wings_snr_psf=float(config.wren_wings_snr_psf),
            blend_p=float(config.wren_blend_p),
            blend_annulus=float(config.wren_blend_annulus),
            bg_rms=None if config.wren_bg_rms is None else float(config.wren_bg_rms),
        )
        return kwargs

    def _apply_extension_pass(self, tmpls: Templates, mode: str, config: FitConfig) -> None:
        """Run the post-extraction extension for ``'psf_convolution'``/``'psf_model'``."""
        if mode == "psf_convolution":
            tmpls.extend_with_psf(
                self._psf_for_template_extension(),
                skip_deblended=bool(config.skip_template_extension_for_deblended),
                background_only=bool(config.extend_wings_background_only),
                inplace=True,
            )
        elif mode == "psf_model":
            tmpls.extend_with_psf_model(
                self._psf_for_template_extension(),
                mode="model",
                skip_deblended=bool(config.skip_template_extension_for_deblended),
                inplace=True,
            )

    def _pixel_scale_arcsec(self, w: WCS | None) -> float | None:
        try:
            if w is None:
                return None
            # (dy, dx) scale; pick x
            return float(proj_plane_pixel_scales(w)[0] * 3600.0)
        except Exception:
            return None

    def _gaussian_fwhm_pix(self, psf: np.ndarray | None) -> float | None:
        if psf is None:
            return None
        try:
            from .utils import measure_shape

            mask = psf > (0.0 if np.min(psf) >= 0 else np.median(psf))
            _, _, sx, sy, _ = measure_shape(psf.astype(np.float32), mask.astype(bool))
            return 2.354820045 * float(np.sqrt(sx * sy))
        except Exception:
            return None

    def _resolve_image_ap_radius_pix(self, idx: int, cfg: _FitConfig) -> float:
        """
        Diameter source: cfg.aperture_diam
        - float/int => same for all images
        - np.ndarray(len(images)-1) => per image (idx>=1), pick [idx-1]
        - None => 1.5 × FWHM of PSF[idx] (in *pixels* of image idx),
                    fallback 3.0 pixels if PSF is missing.
        Units: cfg.aperture_units ("arcsec" or "pix")
        """
        diam = None
        if isinstance(cfg.aperture_diam, (int, float)):
            diam = float(cfg.aperture_diam)
        elif isinstance(cfg.aperture_diam, np.ndarray):
            # array corresponds to images[1:], so use [idx-1]
            if cfg.aperture_diam.size != (len(self.images) - 1):
                raise ValueError("aperture_diam array must have len(images)-1 elements")
            diam = float(cfg.aperture_diam[idx - 1])  # idx>=1 by construction here

        if diam is None:
            # default: 1.5×FWHM of this image PSF (pixels)
            psf_i = None
            if self.psfs is not None and len(self.psfs) > idx:
                psf_i = self.psfs[idx]
                if isinstance(psf_i, np.ndarray):
                    fwhm_pix = self._gaussian_fwhm_pix(psf_i)
                else:
                    # PSFRegionMap: use the first PSF as a representative
                    try:
                        fwhm_pix = self._gaussian_fwhm_pix(psf_i.psfs[0])
                    except Exception:
                        fwhm_pix = None
            else:
                fwhm_pix = None
            rad_pix = 1.5 * fwhm_pix if fwhm_pix and fwhm_pix > 0 else 3.0
            logger.info(f"Using aperture diam 1.5x fwhm {2*rad_pix:.2f} pix for image {idx}")
            return float(rad_pix)

        # convert diameter to pixels if needed
        if cfg.aperture_units.lower().startswith("arc"):
            pscale = self._pixel_scale_arcsec(self.wcs[idx] if self.wcs is not None else None)
            if not pscale or pscale <= 0:
                raise ValueError("aperture_diam in arcsec requires valid WCS for each image")
            return float(diam / (2.0 * pscale))
        else:
            return float(diam / 2.0)  # already in pixels

    def _resolve_catalog_ap_radius_pix(
        self, cat: Table, cfg: _FitConfig, r_default: float | None = None
    ) -> dict[int, float]:
        """
        Return per-source catalog aperture *radius in pixels of the reference image (idx=0)*.

        Source:
        - str => table column name with per-source *diameters*
        - float/int => fixed *diameter* for all sources
        - None => default 1.5 × FWHM of PSF[0] in pixels (fallback 3.0)

        Units: cfg.aperture_units ("arcsec" or "pix")
        """
        # get reference pixel scale
        pscale_ref = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)

        out: dict[int, float] = {}

        # if no catalog, default to r_default for all (if given)
        if cfg.aperture_catalog is None:
            for i, _ in enumerate(cat["id"]):
                out[int(cat["id"][i])] = r_default
            return out

        # get from catalog
        if isinstance(cfg.aperture_catalog, (int, float)):
            diam = float(cfg.aperture_catalog)
            if cfg.aperture_units.lower().startswith("arc"):
                if not pscale_ref or pscale_ref <= 0:
                    raise ValueError("aperture_catalog in arcsec requires valid ref WCS")
                rad = diam / (2.0 * pscale_ref)
            else:
                rad = diam / 2.0
            for i, _ in enumerate(cat["id"]):
                out[int(cat["id"][i])] = float(rad)
            return out

        # string column name
        col = str(cfg.aperture_catalog)
        if col not in cat.colnames:
            raise ValueError(f"aperture_catalog column '{col}' not found in table")
        if cfg.aperture_units.lower().startswith("arc"):
            if not pscale_ref or pscale_ref <= 0:
                raise ValueError("aperture_catalog in arcsec requires valid ref WCS")
            for i, _ in enumerate(cat["id"]):
                diam = float(cat[col][i])
                out[int(cat["id"][i])] = float(diam / (2.0 * pscale_ref))
        else:
            for i, _ in enumerate(cat["id"]):
                diam = float(cat[col][i])
                out[int(cat["id"][i])] = float(diam / 2.0)

        return out

    def _aperture_sum_on_template(
        self, tmpl: Template, radius_pix: float,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Exact aperture sum on a template image centered on its own center.

        ``offset`` displaces the aperture from the catalog position — pass
        the template's accumulated astrometric shift so the aperture follows
        the source the template was resampled onto.
        """
        x0 = tmpl.input_position_cutout[0] + float(offset[0])
        y0 = tmpl.input_position_cutout[1] + float(offset[1])
        aper = CircularAperture((float(x0), float(y0)), r=float(radius_pix))
        phot = aperture_photometry(tmpl.data, aper, method="exact")
        return float(phot["aperture_sum"][0])

    def _totcor_cat(self, cat: Table) -> dict[int, float]:
        """Catalog-side aperture-to-total per source id (band-independent).

            totcor_cat = (f_kron / f_aper) / EE_H(k * R_kron)

        This is the flux-estimator report's ``tcorH``, renamed: the
        detection catalog's Kron-to-aperture flux ratio times the inverse
        encircled energy of the high-resolution PSF at the scaled circularized
        Kron radius.  Computed only when the three ``cat_*_col`` FitConfig
        knobs name existing catalog columns; otherwise empty.  The Kron radius
        column is in arcsec; ``cat_kron_k`` scales it (SExtractor AUTO: 2.5).
        Written once to the band-independent ``totcor_cat`` catalog column.
        """
        cfg = self.config
        cols = (cfg.cat_kron_flux_col, cfg.cat_aper_flux_col, cfg.cat_kron_radius_col)
        source_cat = self.catalog
        if source_cat is None or any(c is None for c in cols):
            return {}
        missing = [c for c in cols if c not in source_cat.colnames]
        if missing:
            logger.warning("totcor_cat skipped: catalog lacks column(s) %s", missing)
            return {}
        psf_hi, _ = self._band_psfs(1)
        if psf_hi is None:
            logger.warning("totcor_cat skipped: no high-resolution PSF available")
            return {}
        pscale = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)
        if not pscale or pscale <= 0:
            logger.warning("totcor_cat skipped: no reference pixel scale")
            return {}

        if "totcor_cat" not in cat.colnames:
            cat["totcor_cat"] = cfg.bad_value
        id_to_row = {int(i): k for k, i in enumerate(cat["id"])}
        wcs_hi = self.wcs[0] if self.wcs is not None else None

        out: dict[int, float] = {}
        n_bad = 0
        for row in source_cat:
            sid = int(row["id"])
            ci = id_to_row.get(sid)
            if ci is None:
                continue
            f_kron = float(row[cfg.cat_kron_flux_col])
            f_aper = float(row[cfg.cat_aper_flux_col])
            r_kron = float(row[cfg.cat_kron_radius_col])  # arcsec
            if not (np.isfinite(f_kron) and np.isfinite(f_aper) and f_aper > 0
                    and np.isfinite(r_kron) and r_kron > 0):
                n_bad += 1
                continue
            ra = dec = None
            if wcs_hi is not None:
                ra, dec = (float(v) for v in wcs_hi.wcs_pix2world(
                    float(row["x"]), float(row["y"]), 0))
            stamp = psf_hi.get_psf(ra, dec) if isinstance(psf_hi, PSFRegionMap) else np.asarray(psf_hi)
            if stamp is None:
                n_bad += 1
                continue
            r_pix = float(cfg.cat_kron_k) * r_kron / pscale
            cy, cx = (stamp.shape[0] - 1) / 2.0, (stamp.shape[1] - 1) / 2.0
            aper = CircularAperture((cx, cy), r=max(r_pix, 0.5))
            ee_h = float(aperture_photometry(stamp, aper, method="exact")["aperture_sum"][0])
            tot = float(np.nansum(stamp))
            if not (np.isfinite(ee_h) and ee_h > 0 and tot > 0):
                n_bad += 1
                continue
            ee_h = min(ee_h / tot, 1.0)
            tcc = (f_kron / f_aper) / ee_h
            out[sid] = tcc
            cat["totcor_cat"][ci] = tcc
        if n_bad:
            logger.warning(
                "totcor_cat: %d of %d sources skipped (non-finite Kron/aperture "
                "inputs or unusable PSF); their ap_flux_cat stays unset",
                n_bad, len(source_cat),
            )
        return out

    def _add_aperture_photometry(
        self,
        cat: Table,
        templates: list[Template],  # post-conv templates (current band)
        fluxes: np.ndarray,  # best-fit per-template fluxes
        residual: np.ndarray,  # residual image (same grid as ref if you upsampled)
        psf: np.ndarray | PSFRegionMap | None,
        idx: int,  # current image index (>=1)
        throughput: float = np.nan,  # filter-mean EE fallback for ee_psf_lo
    ) -> None:
        """
        Aperture photometry on (model + residual), with the classic-mophongo
        (IDL subphot) correction names.  With ``src_tmpl`` the unit-normalized
        high-resolution composite ``H`` and ``src_img`` the unit-normalized
        band-convolved composite ``H*K``:

            ap_hi = aper(src_tmpl, R)      EE of the hi-res composite at R
            ap_lo = aper(src_img,  R)      EE of the band-convolved composite at R
            psfcor = ap_hi/ap_lo            hi -> lo band EE ratio (shape corr.)
            totcor = 1/(ap_lo*ee_psf_lo)   aperture -> genuine total: ap_lo is
                                          the model-support EE and ee_psf_lo
                                          the recorded flux fraction of the
                                          finite PSF support itself, the same
                                          factor ``flux_<i>_total`` divides by

        (IDL's ``totcor`` is 1/ap_lo alone: reconstruct it from the written
        columns as ``totcor * ee_psf_lo`` for like-for-like comparisons.)

        Writes:
        ap_flux_{idx}        – raw aperture sum on model+residual
        ee_psf_lo_{idx}      – per-source box EE used (fallback: filter mean)
        stampcor_{idx}       – 1/ap_lo alone: aperture to the total of the
                               model's own finite support, NO EE factor.  Named
                               ``stampcor`` and not ``tot*`` deliberately: a
                               correction that stops at the edge of the support
                               is not a total (flux_estimator_comparison.tex,
                               Sec. "Naming").  This is the like-for-like
                               quantity against classic IDL's released
                               ``totcor``, which is misnamed by the same rule —
                               but only when the two runs share a PSF support.
        totcor_{idx}         – 1/(ap_lo * ee_psf_lo): ``totcor`` earns the name
                               because it ALWAYS includes the beyond-support
                               EE, like a catalog aperture-to-total
        psfcor_{idx}         – ap_hi/ap_lo
        ap_flux_corr_{idx}   – ap_flux * totcor: total flux
        totcor_cat           – catalog-side aperture-to-total (band-independent),
                               (f_kron/f_aper) / EE_H(k*R_kron); needs the
                               ``cat_*_col`` FitConfig knobs (the flux-estimator
                               report's ``tcorH``, renamed)
        ap_flux_cat_{idx}    – ap_flux * psfcor * totcor_cat: total flux on the
                               detection catalog's Kron convention
        """
        from photutils.aperture import CircularAperture, aperture_photometry

        cfg = self.config
        id_to_row = {int(i): k for k, i in enumerate(cat["id"])}
        # pre-convolution composites for ap_hi, by id (reference grid, same
        # pixel scale as the fit grid on the upsample path)
        hi_by_id = {}
        if getattr(self, "tmpls", None) is not None:
            hi_by_id = {int(t.id): t for t in self.tmpls.templates}

        # ensure columns exist
        for name in (f"ap_model_{idx}", f"ap_flux_{idx}", f"ee_psf_lo_{idx}",
                     f"stampcor_{idx}", f"totcor_{idx}", f"psfcor_{idx}",
                     f"ap_flux_corr_{idx}", f"ap_flux_cat_{idx}"):
            if name not in cat.colnames:
                cat[name] = cfg.bad_value
        totcor_cat = self._totcor_cat(cat)

        # radii
        r_img_pix = self._resolve_image_ap_radius_pix(
            idx, cfg
        )  # same for all in this band (by design)
        # residual+model patch measurement (raw)
        for tmpl, fl in zip(templates, fluxes):
            pid = tmpl.id_parent if getattr(tmpl, "parent_id", None) is not None else tmpl.id
            row = id_to_row.get(int(pid))
            if row is None:
                continue

            # --- raw aperture flux on (model + residual) in the *current image* ---
            res_patch = residual[tmpl.slices_original]
            model_patch = fl * tmpl.data[tmpl.slices_cutout]
            patch = res_patch + model_patch

            # Aperture at the *fitted* position: the astrometric passes moved
            # both the source and its resampled template off the catalog
            # position (median ~1.3 fit-pix), and an off-centre aperture
            # loses EE. subphot does the same for its raw flux (xaper = xc-p).
            sx, sy = (float(v) for v in getattr(tmpl, "shifted", (0.0, 0.0))[:2])
            x0 = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start + sx
            y0 = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start + sy
            aper_img = CircularAperture((float(x0), float(y0)), r=float(r_img_pix))
            phot = aperture_photometry(patch, aper_img, method="exact")
            ap_raw = float(phot["aperture_sum"][0])

            # ap_lo (times the post-conv total): the post-conv total rather
            # than 1.0 accounts for any flux lost at the template boundary
            # during convolution.
            num = float(tmpl.data.sum())
            # same fitted-position convention as ap_raw: the template was
            # resampled onto the shifted source, so the aperture follows it
            den = self._aperture_sum_on_template(tmpl, r_img_pix, offset=(sx, sy))

            ap_model = fl * den  # aperture flux on model only (for info)

            inv_ap_b = num / den if (np.isfinite(num) and np.isfinite(den) and den > 0) else 1.0
            # per-source box EE of the finite PSF support (filter-mean fallback)
            ee = float(getattr(tmpl, "ee_psf_lo", np.nan))
            if not (np.isfinite(ee) and ee > 0.0):
                ee = float(throughput)
            if not (np.isfinite(ee) and ee > 0.0):
                ee = 1.0
            totcor = inv_ap_b / ee
            ap_corr = ap_raw * totcor

            # psfcor = ap_hi/ap_lo: aperture sum of the unit-normalized pre-conv
            # composite over that of the convolved one, both at R on the
            # reference grid (identical grids on the upsample path)
            psfcor = np.nan
            t_hi = hi_by_id.get(int(tmpl.id))
            if t_hi is not None:
                num_hi = float(t_hi.data.sum())
                ap_f = self._aperture_sum_on_template(t_hi, r_img_pix)
                if np.isfinite(ap_f) and ap_f > 0 and num_hi > 0 and den > 0 and num > 0:
                    psfcor = (ap_f / num_hi) / (den / num)

            cat[f"ap_model_{idx}"][row] = ap_model
            cat[f"ap_flux_{idx}"][row] = ap_raw
            cat[f"ee_psf_lo_{idx}"][row] = ee
            cat[f"stampcor_{idx}"][row] = inv_ap_b
            cat[f"totcor_{idx}"][row] = totcor
            cat[f"psfcor_{idx}"][row] = psfcor
            cat[f"ap_flux_corr_{idx}"][row] = ap_corr
            tcc = totcor_cat.get(int(pid), np.nan)
            if np.isfinite(psfcor) and np.isfinite(tcc):
                cat[f"ap_flux_cat_{idx}"][row] = ap_raw * psfcor * tcc

    def _fit_catalog(self, config: _FitConfig) -> Table:
        """Output-catalog skeleton :meth:`run` fits into: id/x/y + provenance."""
        catalog = self.catalog
        if catalog is None:
            # use astropy to make catalog from image[0] + segmap
            logger.error("no catalog provided; generating one from the segmap")
            raise NotImplementedError("Catalog generation not implemented yet")
        cat = catalog.copy()
        keep_cols = ["id", "x", "y"]
        keep_cols.extend(
            col
            for col in ("is_deblended", "deblend_parent_label", "deblend_nchildren")
            if col in catalog.colnames
        )
        sat_cols = [c for c in catalog.colnames if c.startswith("FLAG_SATURATED_")]
        keep_cols.extend(sat_cols)
        cat = cat[keep_cols]
        if config.aperture_catalog is not None:
            cat[config.aperture_catalog] = catalog[config.aperture_catalog]
        return cat

    def _prepare_hi_templates(self, cat: Table, config: _FitConfig) -> list[Template]:
        """Build the hi-res templates: extract (or adopt prebuilt), flag, extend.

        Factored out of :meth:`run` so :meth:`load_fit` can regenerate stamps
        through the identical code path.  Sets ``tmpls``,
        ``templates_extracted`` and ``templates_extended`` on the instance and
        returns the template list.
        """
        images = self.images
        segmap = self.segmap
        wcs = self.wcs
        catalog = self.catalog
        psfs = self.psfs if self.psfs is not None else [None] * len(images)

        if self.input_templates is None:
            # One selector over the four build schemes (see EXTEND_MODES):
            # 'wren'/'classic' build their composite inside extract_templates,
            # 'psf'/'psf_model' run as a post-pass below.
            extend_mode = self._resolve_extend_mode(config)
            self.extend_mode = extend_mode
            self.tmpls = Templates()
            self.tmpls.extract_templates(
                images[0],
                segmap,
                list(zip(cat["x"], cat["y"])),
                wcs=wcs[0] if wcs is not None else None,
                dilate_segmap=config.template_dilate_segmap,
                **self._extend_scheme_kwargs(extend_mode, config),
            )
            if "is_deblended" in cat.colnames:
                is_deblended_by_id = {
                    int(row["id"]): bool(row["is_deblended"])
                    for row in cat
                }
                parent_by_id = {
                    int(row["id"]): int(row["deblend_parent_label"])
                    for row in cat
                } if "deblend_parent_label" in cat.colnames else {}
                nchildren_by_id = {
                    int(row["id"]): int(row["deblend_nchildren"])
                    for row in cat
                } if "deblend_nchildren" in cat.colnames else {}
                for tmpl in self.tmpls.templates:
                    tmpl_id = int(tmpl.id)
                    tmpl.is_deblended = is_deblended_by_id.get(tmpl_id, False)
                    tmpl.deblend_parent_label = parent_by_id.get(tmpl_id)
                    tmpl.deblend_nchildren = nchildren_by_id.get(tmpl_id, 1)
            # FLAG_SATURATED_* (any filter): isolate these templates into
            # their own scenes in :func:`mophongo.scene.generate_scenes`.
            # The flag value is the star's group id (lowest flagged segment
            # id): rows sharing a value are one star and are fit together in
            # one scene. Legacy 0/1 columns degrade to one scene per template.
            sat_cols = [c for c in cat.colnames if c.startswith("FLAG_SATURATED_")]
            if sat_cols:
                sat_by_id: dict[int, int] = {}
                for row in cat:
                    group = max(int(row[c]) for c in sat_cols)
                    if group:
                        sat_by_id[int(row["id"])] = group
                for tmpl in self.tmpls.templates:
                    group = sat_by_id.get(int(tmpl.id), 0)
                    tmpl.is_saturated = group != 0
                    tmpl.sat_group = group
            # Build-stage snapshots for the diagnostics. A snapshot is a full
            # second copy of every stamp -- 6 GB on a MINERVA field -- so take
            # one only where the two stages actually differ. The post-extraction
            # pass is the only thing that rewrites template pixels here, and it
            # runs for two of the five modes; for the rest (including the
            # default 'psf_wings', which builds its composite inside
            # extract_templates) both names alias `tmpls`, which is what they
            # held pixel-for-pixel anyway. Aliased snapshots are read-only by
            # contract: nothing downstream writes into them.
            self.templates_extracted = (
                deepcopy(self.tmpls)
                if extend_mode in ("psf_convolution", "psf_model")
                else self.tmpls
            )
            # Template extension is a shape operation. The extension code
            # normalizes finite PSF stamps to unit-sum shapes and keeps
            # native finite-support sums only as throughput metadata.
            self._apply_extension_pass(self.tmpls, extend_mode, config)
            self.templates_extended = self.tmpls
        else:
            if isinstance(self.input_templates, Templates):
                self.tmpls = deepcopy(self.input_templates)
            else:
                self.tmpls = Templates()
                self.tmpls._templates = [deepcopy(t) for t in self.input_templates]
            if not getattr(self.tmpls, "original_shape", None):
                self.tmpls.original_shape = images[0].shape
            if not getattr(self.tmpls, "wcs", None):
                self.tmpls.wcs = wcs[0] if wcs is not None else None
            # no extraction and no extension pass ran, so both stages are the
            # prebuilt templates themselves (see the snapshot note above)
            self.templates_extracted = self.tmpls
            self.templates_extended = self.tmpls
        templates = self.tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        if catalog is not None and "flag_star" in catalog.colnames:
            star_ids = set(int(r["id"]) for r in catalog if r["flag_star"] == 1)
            for t in templates:
                if int(t.id) in star_ids:
                    t.is_star = True
            # is_star only reaches the astrometry when the run asks for it:
            # unsaturated stars are the best anchors there is (FitConfig
            # .astrom_exclude_stars, default False).
            logger.info(
                "Marked %d templates as stars (%s from astrometry)",
                sum(t.is_star for t in templates),
                "excluded" if config.astrom_exclude_stars else "kept as anchors",
            )

        ndropped = len(cat) - len(templates)
        # @@@ this is because of reliance of x,y in catalog -> use segmap + weight?
        source = "prebuilt" if self.input_templates is not None else "extracted"
        logger.info("%d %s templates, dropped %d", len(templates), source, ndropped)
        logger.info("Pipeline (templates) memory: %.1f GB", memory())
        return templates

    def _convolved_templates(
        self, ifilt: int, config: _FitConfig
    ) -> tuple[list[Template], np.ndarray | None]:
        """Convolve/project the hi templates onto band ``ifilt``'s fitting grid.

        Factored out of :meth:`run` for reuse by :meth:`load_fit`.  Appends
        the band's bin factor to ``fit_bin_factors`` and, on the upsample
        path, replaces ``images[ifilt]``/``wcs[ifilt]`` with their
        reference-grid versions, exactly as :meth:`run` always did.  Returns
        the convolved templates and the (possibly upsampled) weight image.
        """
        images = self.images
        weights = self.weights
        wcs = self.wcs
        kernels = self.kernels if self.kernels is not None else [None] * len(images)

        weights_i = weights[ifilt] if weights is not None else None
        kernel = kernels[ifilt]
        if isinstance(kernel, PSFRegionMap):
            logger.info("using kernel lookup table %s", kernel.name)

        k = bin_factor_from_wcs(wcs[0], wcs[ifilt]) if wcs is not None else 1
        self.fit_bin_factors.append(int(k))

        # Native lo-band pixel scale, recorded before the upsample path
        # rebinds wcs[ifilt] to the reference WCS: the delivered PSF stamps
        # stay on the native grid, so PSFSZ/RCIRC metadata must use this.
        if not hasattr(self, "native_pscales"):
            self.native_pscales: dict[int, float | None] = {}
        self.native_pscales[ifilt] = self._pixel_scale_arcsec(
            wcs[ifilt] if wcs is not None else None
        )

        if k > 1:
            if config.multi_resolution_method == "upsample":
                logger.info("upsampling image %d by factor %d", ifilt, k)
                images[ifilt], weights_i = _upsample_boxed(
                    images[ifilt],
                    weights_i,
                    k,
                    self.trial_box_lo,
                )
                wcs[ifilt] = wcs[0]
            else:
                logger.info("downsampling templates and kernels by factor %d", k)
                tmpls_lo = Templates()
                tmpls_lo.original_shape = images[ifilt].shape
                tmpls_lo.wcs = wcs[ifilt]
                tmpls_lo._templates = [
                    t.downsample(k, wcs_lo=wcs[ifilt]) for t in self.tmpls._templates
                ]

                if isinstance(kernel, PSFRegionMap):
                    kernel.psfs = np.array([downsample_psf(psf, k) for psf in kernel.psfs])
                else:
                    kernel = downsample_psf(kernel, k)

        if k == 1 or config.multi_resolution_method == "upsample":
            # Shallow container, not a deepcopy: `prune_outside_weight` only
            # drops list entries (and records `wnorm`), and `convolve_templates`
            # with inplace=False copies each stamp as it goes, so nothing here
            # writes into the hi-res pixels. A deepcopy would hold a second
            # full set of stamps -- 6 GB on a MINERVA field -- for the whole
            # convolution.
            tmpls_lo = Templates()
            tmpls_lo.original_shape = self.tmpls.original_shape
            tmpls_lo.segmap = self.tmpls.segmap
            tmpls_lo.wcs = getattr(self.tmpls, "wcs", None)
            tmpls_lo._templates = list(self.tmpls._templates)

        if weights_i is not None:
            tmpls_lo.prune_outside_weight(weights_i)

        templates = tmpls_lo.convolve_templates(
            kernel, inplace=False, psf_lo=getattr(self, "prm_lo", None)
        )
        del tmpls_lo
        if k > 1 and config.multi_resolution_method == "upsample":
            dummy_image = np.zeros(images[ifilt].shape, dtype=np.byte)
            # project in place: a pre-projection stamp is dead the moment its
            # projection exists, and building a second list would hold both
            # full sets at once
            for i, t in enumerate(templates):
                templates[i] = t.project_to_block_replicated_grid(
                    k, parent_image=dummy_image
                )
                del t
        self.templates = templates
        logger.info("Pipeline (convolved) memory: %.1f GB", memory())

        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"
        return templates, weights_i

    def run(self, config: FitConfig | None = None) -> tuple[Table, list[np.ndarray]]:
        """Run photometry on the configured images.

        Returns
        -------
        Table
            Catalog containing flux measurements for each image.
        list of ndarray
            Residual images corresponding to each fitted image.
        """
        from . import utils
        import warnings

        # config-driven construction: load data + maps on first run()
        if getattr(self, "run_config", None) is not None:
            if self.images is None:
                self.load_data()
            elif self.kernels[-1] is None:
                # data pre-loaded with load_data(kernels=False): finish the maps
                self._ensure_maps()
                self.psfs[0] = self.prm_hi  # template extension reads psfs[0]
                self.psfs[-1] = self.prm_lo
                self.kernels[-1] = self.prm_kern

        images = self.images
        segmap = self.segmap
        catalog = self.catalog
        psfs = self.psfs
        weights = self.weights
        kernels = self.kernels
        psf_throughputs = self.psf_throughputs
        if kernels is None:
            kernels = [None] * len(images)
        if psfs is None:
            psfs = [None] * len(images)
        wcs = self.wcs
        if config is None:
            config = self.config
        else:
            self.config = config

        # snapshot the fully-explicit config next to the outputs, so a
        # finished run reopens later with from_config(out_dir)
        if getattr(self, "run_config", None) is not None:
            self.save_config()

        logger.info("Pipeline (start) memory: %.1f GB", memory())
        logger.info("Pipeline config: %s", config)

        # No whole-array finiteness sweep here. The image branch of the old
        # check was inverted (it fired only when the image was None, and then
        # np.isfinite(None) raises), and the weight branch touched all 876 Mpx
        # of the detection ivar to re-check a guard load_data:1649-1664 has
        # already applied -- it zeroes non-finite pixels in image AND weight
        # so they carry no information. Templates are still asserted finite
        # after extraction (_prepare_hi_templates).

        with self._phase("catalog"):
            cat = self._fit_catalog(config)

        with self._phase("extract templates"):
            templates = self._prepare_hi_templates(cat, config)

        # The detection-band inverse variance is read only while templates are
        # built -- the build schemes grade real data against the PSF by SNR --
        # and nothing in the fit touches it again. It is as large as the
        # detection mosaic, so release it here, and record that it went: a
        # second run() on this instance must fail rather than quietly rebuild
        # the templates without their weights.
        if (
            getattr(self, "run_config", None) is not None
            and weights is not None
            and len(weights) > 0
            and weights[0] is not None
        ):
            weights[0] = None
            self._det_weight_released = True
            logger.info(
                "released the detection-band weight map; memory: %.1f GB", memory()
            )

        residuals: list[np.ndarray] = []
        self.all_templates: list[Template] = []
        self.all_scenes: list[Scene] = []
        self.fit_bin_factors: list[int] = []
        self.model_images = _ModelImages(self)
        for ifilt in range(1, len(images)):
            scenes = []
            templates, weights_i = self._convolved_templates(ifilt, config)
            # @@@ split scenes here
            if not getattr(config, "run_scene_solver", True):
                raise ValueError(
                    "run_scene_solver=False is no longer supported; the scene solver "
                    "is the only fitting path"
                )
            templates_scene = templates
            with self._phase("generate scenes"):
              scenes, labels = generate_scenes(
                templates_scene,
                images[ifilt],
                weights_i,
                coupling_thresh=float(config.scene_coupling_thresh),
                max_size=config.scene_max_size,
                snr_thresh_astrom=float(config.snr_thresh_astrom),
                isolation_thresh=float(config.astrom_isolation_thresh),
                minimum_bright=int(config.scene_minimum_bright),
                max_merge_radius=float(getattr(config, "scene_max_merge_radius", np.inf)),
                exclude_stars=bool(config.astrom_exclude_stars),
            )
            # Assume each scene has .ra and .dec attributes (center coordinates)
            # Compute RA/Dec for each scene center using WCS
            if config.generate_scene_catalog:
                self.all_scenes.append(scenes)
                ras, decs = [], []
                for s in scenes:
                    xy_mean = np.mean([t.position_original for t in s.templates], axis=0)
                    if wcs[0] is not None:
                        ra, dec = wcs[0].wcs_pix2world([xy_mean], 0)[0]
                    else:
                        ra, dec = np.nan, np.nan
                    ras.append(ra)
                    decs.append(dec)

                scene_table = Table(
                    {
                        "id": [s.id for s in scenes],
                        "n_templates": [len(s.templates) for s in scenes],
                        "is_bright": [s.is_bright.sum() for s in scenes],
                        "ra": ras,
                        "dec": decs,
                    }
                )
                scene_table.write(
                    f"scene_catalog_{ifilt}.ecsv", format="ascii.ecsv", overwrite=True
                )
                logger.info("wrote scene catalog scene_catalog_%d.ecsv", ifilt)
                import sys

                sys.exit()

            for s in scenes:
                # per-scene detail at DEBUG; generate_scenes logs the summary
                logger.debug(f"Scene {s.id}: {len(s.templates)} (bright: {s.is_bright.sum()})")

            niter_scene = max(config.fit_astrometry_niter, 1)
            shift_tol = float(getattr(config, "astrom_shift_tol", 0.02))
            # the tolerance is in fit-grid pixels, which on the upsample path
            # is the hi-res grid and not the grid of the band being fitted --
            # report the angle too, so the log is unambiguous
            ps_fit = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)
            tol_txt = f"{shift_tol:.3g} pix" + (
                f" = {shift_tol * ps_fit * 1000:.1f} mas" if ps_fit else ""
            )
            # Scenes are independent across passes, not only within one:
            # solve() reads the scene's own templates and read-only slices of
            # the shared image and weights, and writes only to itself and to
            # those templates. So the whole refinement of one scene -- its
            # passes, its convergence test, and the flux-only pass that closes
            # it out -- is a self-contained unit of work, and the loop runs
            # scene-by-scene rather than synchronising every scene at each
            # pass. The results are identical either way; what goes is the
            # barrier, which is what stops a scene from being handed to a
            # worker process (see docs/SCALING_FIXED_MEMORY.md).
            logger.info(
                "[Scenes] solving %d scene(s), up to %d astrometric pass(es) "
                "each (tol %s)", len(scenes), niter_scene, tol_txt,
            )
            # A flux-only run (fit_astrometry_niter = 0) takes one pass through
            # the loop below -- solve() dispatches to the flux-only path on the
            # same flag -- and skips the closing re-solve.
            final_cfg = (
                replace(config, fit_astrometry_niter=0)
                if config.fit_astrometry_niter > 0
                else None
            )
            unconverged: list[Scene] = []
            for scn in scenes:
                scn.set_band(images[ifilt], weights_i, config=config)
                with self._phase("astrometry passes"):
                  for j in range(niter_scene):
                    prev = np.array([t.shifted[:2] for t in scn.templates], dtype=float)
                    scn.solve(config=config, apply_shifts=True)
                    cur = np.array([t.shifted[:2] for t in scn.templates], dtype=float)
                    step = (
                        float(np.max(np.abs(cur - prev)))
                        if prev.size and cur.shape == prev.shape
                        else 0.0
                    )
                    scn.astrom_step, scn.astrom_niter = step, j + 1
                    # a verdict only means something where shifts were fitted:
                    # a flux-only run, or a scene with too few bright anchors
                    # to carry a shift block, never moves, and reporting that
                    # as "converged" would claim an astrometric solution that
                    # was never solved for. Those keep astrom_converged None
                    # and flag -1 -- and stop here, since nothing will move
                    # them on a later pass either.
                    if scn.shifts is not None and len(scn.shifts) > 0:
                        scn.astrom_converged = step < shift_tol
                    if scn.astrom_converged is not False:
                        break
                if scn.astrom_converged is False:
                    unconverged.append(scn)
                # Each pass solved fluxes on the templates as they stood
                # *before* that pass's shift was applied, so the stored fluxes,
                # errors and model belong to a basis that no longer exists --
                # the last shift applied is never accounted for. Re-solve
                # fluxes once on the final templates, regardless of the
                # convergence verdict, so what is written is stationary for the
                # basis actually used to build the model, residual and stamps.
                # Shifts are left untouched: this is a flux-only pass.
                if final_cfg is not None:
                    with self._phase("final flux solve"):
                        scn.solve(config=final_cfg)
                logger.debug(
                    "[Scenes] scene %s: %d pass(es), last increment %.4f pix, %s",
                    scn.id, scn.astrom_niter, scn.astrom_step or 0.0,
                    "converged" if scn.astrom_converged
                    else ("no shift fitted" if scn.astrom_converged is None
                          else "still moving"),
                )

            niters = [s.astrom_niter for s in scenes]
            logger.info(
                "[Scenes] %d of %d scene(s) converged (< %s); passes run: "
                "median %d, max %d%s",
                sum(s.astrom_converged is True for s in scenes), len(scenes),
                tol_txt,
                int(np.median(niters)) if niters else 0, max(niters, default=0),
                "; fluxes re-solved on the final templates"
                if final_cfg is not None else "",
            )

            # Scenes still moving when the budget ran out: their shifts are the
            # last iterate, not a converged solution. Worth knowing which.
            if unconverged:
                slow = sorted(unconverged, key=lambda s: -(s.astrom_step or 0.0))
                logger.warning(
                    "[Scenes] %d of %d scene(s) did not converge in %d passes "
                    "(tol %s); worst: %s",
                    len(unconverged), len(scenes), niter_scene, tol_txt,
                    ", ".join(f"scene {s.id} ({s.astrom_step:.3f} pix)" for s in slow[:5]),
                )

            # Every shifted template holds its pre-shift pixels so that each
            # pass resamples the original instead of compounding the cubic
            # smoothing (Templates.apply_template_shifts). That is a second
            # full set of stamps -- 6 GB on a MINERVA field -- and the shifts
            # are settled here, with the residual about to allocate another
            # full-field array. Release them; apply_template_shifts refuses to
            # shift a released template rather than resample resampled data.
            released = sum(
                t.__dict__.pop("_data_unshifted", None) is not None for t in templates
            )
            if released:
                logger.info(
                    "released the pre-shift pixels of %d template(s); memory: %.1f GB",
                    released, memory(),
                )

            # build model in res first, then subtract from image. File-backed
            # when the run has an output path (see _allocate_residual);
            # np.zeros, not np.zeros_like, for the in-memory case: zeros_like
            # memsets every page, which on a trial run materialises the whole
            # reference grid.
            with self._phase("residual"):
              res = self._allocate_residual(images[ifilt], ifilt)
              for s in scenes:
                sl = _slices_from_bbox(s.bbox)
                res[sl] += s.model_image()  # adds models in place
              # then subtract from image to get residual, in place:
              # `images - res` would hold a third full-field array while the
              # model is still alive
              np.subtract(images[ifilt], res, out=res)

            fluxes = [t.flux for t in templates]
            errs = [t.err for t in templates]
            err_pred = Templates.predicted_errors(templates, weights_i)

            # Last read of the band's inverse variance in the fit itself.
            # Nothing after this point -- residual, aperture photometry,
            # catalog update, write_outputs -- touches it except the scene
            # figures, so a run that draws none can drop it here already;
            # write_outputs releases it for the rest, after the figures and
            # before the stamps.
            if not self._scene_pixels_needed():
                weights_i = None
                self._release_scene_weights(scenes)
            throughput = _filter_psf_throughput(
                psfs[ifilt] if psfs is not None else None,
                None if psf_throughputs is None else psf_throughputs[ifilt],
            )
            _record_psf_ee(
                cat,
                psfs[ifilt] if psfs is not None else None,
                getattr(self, "native_pscales", {}).get(ifilt)
                or self._pixel_scale_arcsec(
                    self.wcs[ifilt] if self.wcs is not None else None
                ),
                ifilt,
                throughput,
            )


            if config.aperture_diam is not None:
                pscale = self._pixel_scale_arcsec(
                    self.wcs[ifilt] if self.wcs is not None else None
                )
                r_img_pix = self._resolve_image_ap_radius_pix(ifilt, config)
                r_img_arcsec = r_img_pix * pscale
                cat["aper_" + str(ifilt)] = 2 * r_img_arcsec
            # Scene membership, taken off the scene objects themselves rather
            # than generate_scenes' labels: the scenes hold the very template
            # instances that were fitted, so identity is exact.
            scene_of_template = {
                id(t): s.id for s in scenes for t in getattr(s, "templates", [])
            }
            template_scene_ids = [scene_of_template.get(id(t), -1) for t in templates]
            # Convergence is a property of the scene, and every source fitted
            # in it shares the verdict: its shift came out of the same passes.
            scene_astrom_flags = {
                s.id: (-1 if s.astrom_converged is None else int(not s.astrom_converged))
                for s in scenes
            }
            self._update_catalog_with_fluxes(
                cat,
                templates,
                fluxes,
                errs,
                err_pred,
                throughput,
                ifilt,
                scene_ids=template_scene_ids,
                scene_flags=scene_astrom_flags,
            )
            with self._phase("aperture photometry"):
              self._add_aperture_photometry(
                cat,
                templates,
                fluxes,
                res,
                psfs[ifilt] if psfs is not None else None,
                ifilt,
                throughput=throughput,
            )

            self.residuals.append(res)
            #            self.fit.append(fitter)
            self.all_templates.append(templates)
            self.all_scenes.append(scenes)
        #            self.infos.append(info)

        logger.info("Pipeline (end) memory: %.1f GB",
                    psutil.Process(os.getpid()).memory_info().rss / 1e9)
        # the CLI reports once for the whole invocation; only report here when
        # run() was called directly, or the breakdown would print twice
        if not getattr(self, "_cli_stepping", False):
            self.report_timings()
        self.table = cat

        return self.table, self.residuals  # , self.all_templates, self.all_scenes

    @staticmethod
    def _template_for_source(templates: Sequence[Template], source_id: int) -> Template:
        """Return the template with ``id == source_id``."""
        for tmpl in templates:
            if int(tmpl.id) == int(source_id):
                return tmpl
        raise KeyError(f"source id {source_id} not found in templates")

    @staticmethod
    def _stamp_slices_for_template(
        tmpl: Template,
        image_shape: tuple[int, int],
        half_size: int | None = None,
    ) -> tuple[slice, slice]:
        """Return parent-image slices for a source diagnostic stamp."""
        if half_size is None:
            ysl, xsl = tmpl.slices_original
            return ysl, xsl

        cx, cy = tmpl.input_position_original
        x0 = max(0, int(round(cx)) - int(half_size))
        x1 = min(image_shape[1], int(round(cx)) + int(half_size) + 1)
        y0 = max(0, int(round(cy)) - int(half_size))
        y1 = min(image_shape[0], int(round(cy)) + int(half_size) + 1)
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
    def _template_on_stamp(tmpl: Template, stamp_slices: tuple[slice, slice]) -> np.ndarray:
        """Place a template on a local parent-image stamp."""
        ysl, xsl = stamp_slices
        stamp = np.zeros((ysl.stop - ysl.start, xsl.stop - xsl.start), dtype=float)
        tx0, ty0 = map(int, tmpl._origin_original_true)
        tx1 = tx0 + tmpl.data.shape[1]
        ty1 = ty0 + tmpl.data.shape[0]

        x0 = max(xsl.start, tx0)
        x1 = min(xsl.stop, tx1)
        y0 = max(ysl.start, ty0)
        y1 = min(ysl.stop, ty1)
        if x1 <= x0 or y1 <= y0:
            return stamp

        stamp[y0 - ysl.start : y1 - ysl.start, x0 - xsl.start : x1 - xsl.start] = tmpl.data[
            y0 - ty0 : y1 - ty0,
            x0 - tx0 : x1 - tx0,
        ]
        return stamp

    @staticmethod
    def _diagnostic_display_scale(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
        """Return a shared median/MAD display scale for comparable panels."""
        values = []
        for data in arrays:
            arr = np.asarray(data, dtype=float)
            finite = np.isfinite(arr)
            if np.any(finite):
                values.append(arr[finite])
        if not values:
            return 0.0, 1.0

        merged = np.concatenate(values)
        center = float(np.nanmedian(merged))
        scale = float(mad_std(merged, ignore_nan=True))
        if not np.isfinite(scale) or scale <= 0.0:
            nonzero = merged[merged != 0.0]
            if nonzero.size:
                scale = float(mad_std(nonzero, ignore_nan=True))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.nanpercentile(np.abs(merged - center), 95.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return center, scale

    @staticmethod
    def _imshow_scaled(
        ax,
        data: np.ndarray,
        *,
        cmap: str = "gray_r",
        symmetric: bool = False,
        center: float | None = None,
        scale: float | None = None,
    ) -> None:
        """Show a diagnostic stamp with inverted grayscale MAD scaling."""
        arr = np.asarray(data, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            ax.imshow(arr, cmap=cmap, origin="lower")
            return
        if center is None or scale is None:
            auto_center, auto_scale = Pipeline._diagnostic_display_scale([arr])
            if center is None:
                center = auto_center
            if scale is None:
                scale = auto_scale
        ax.imshow(arr - center, cmap=cmap, origin="lower", vmin=-5.0 * scale, vmax=5.0 * scale)

    def _segmap_on_stamp(self, stamp_slices: tuple[slice, slice]) -> np.ndarray:
        """Return the segmentation map on the same parent-image stamp grid."""
        ysl, xsl = stamp_slices
        return np.asarray(self.segmap)[ysl, xsl]

    @staticmethod
    def _snapshot_template(tmpl: Template) -> Template:
        """Return a template metadata copy with detached pixel data."""
        snap = deepcopy(tmpl)
        snap.data = np.array(tmpl.data, copy=True)
        return snap

    def _source_position(self, source_id: int) -> tuple[float, float]:
        """Return the catalog position for ``source_id``."""
        if not hasattr(self, "table"):
            raise RuntimeError("run the pipeline before source diagnostics")
        ids = np.asarray(self.table["id"], dtype=int)
        idx = np.flatnonzero(ids == int(source_id))
        if idx.size == 0:
            raise KeyError(f"source id {source_id} not found in fitted catalog")
        row = int(idx[0])
        return float(self.table["x"][row]), float(self.table["y"][row])

    def _psf_for_template_extension(self):
        """Return the detection-band PSF used to build/extend templates.

        Strictly ``psfs[0]``: templates live on the ``images[0]`` grid, so
        anything else is the wrong band on the wrong pixel scale. There is
        deliberately no fallback to another index -- substituting the low-res
        PSF produces plausible-looking templates with silently wrong wings and
        radii (reaches derived in lo-res pixels applied as hi-res ones).
        """
        if getattr(self, "run_config", None) is not None and (
            self.psfs is None or len(self.psfs) == 0 or self.psfs[0] is None
        ):
            # config-driven run: the detection map is cached, but only
            # build_psfs() loads it.
            if self.prm_hi is None:
                self.build_psfs()
            if self.psfs is not None and len(self.psfs) > 0:
                self.psfs[0] = self.prm_hi

        psf_hi = self.psfs[0] if self.psfs is not None and len(self.psfs) > 0 else None
        if psf_hi is None:
            raise ValueError(
                f"extend_mode={getattr(self, 'extend_mode', self.extend_mode_override)!r} "
                "requires the detection-band PSF in psfs[0] (the images[0] grid). "
                "No other index is substituted: a lower-resolution PSF would "
                "silently produce wrong template wings and wrong extension radii."
            )
        return psf_hi

    def _rebuild_source_stage_templates(
        self,
        source_id: int,
        *,
        ifilt: int,
    ) -> tuple[Template, Template, Template]:
        """Rebuild one source through extraction, extension, and convolution.

        This intentionally does not read intermediate templates saved during
        ``run``.  Template data are copied immediately after each operation so
        later in-place mutations cannot change the diagnostic panels.
        """
        pos = self._source_position(source_id)
        extend_mode = self._resolve_extend_mode(self.config)
        rebuilt = Templates()
        rebuilt.extract_templates(
            self.images[0],
            self.segmap,
            [pos],
            wcs=self.wcs[0] if self.wcs is not None else None,
            dilate_segmap=int(self.config.template_dilate_segmap),
            **self._extend_scheme_kwargs(extend_mode, self.config),
        )
        if not rebuilt.templates:
            raise KeyError(f"could not re-extract source id {source_id} for diagnostics")

        tmpl = rebuilt.templates[0]
        if self.catalog is not None:
            matches = np.where(np.asarray(self.catalog["id"], dtype=int) == int(source_id))[0]
            if matches.size:
                row = self.catalog[int(matches[0])]
                if "is_deblended" in self.catalog.colnames:
                    tmpl.is_deblended = bool(row["is_deblended"])
                if "deblend_parent_label" in self.catalog.colnames:
                    tmpl.deblend_parent_label = int(row["deblend_parent_label"])
                if "deblend_nchildren" in self.catalog.colnames:
                    tmpl.deblend_nchildren = int(row["deblend_nchildren"])
                sat_cols = [c for c in self.catalog.colnames if c.startswith("FLAG_SATURATED_")]
                if sat_cols:
                    group = max(int(row[c]) for c in sat_cols)
                    tmpl.is_saturated = group != 0
                    tmpl.sat_group = group
        before = self._snapshot_template(tmpl)

        work = Templates()
        work.original_shape = rebuilt.original_shape
        work.wcs = getattr(rebuilt, "wcs", self.wcs[0] if self.wcs is not None else None)
        work.segmap = rebuilt.segmap
        work._templates = [tmpl]
        # 'wren'/'classic' already built their composite in extract_templates
        # above, so only the post-pass modes do anything here.
        self._apply_extension_pass(work, extend_mode, self.config)
        tmpl_ext = work.templates[0]
        after = self._snapshot_template(tmpl_ext)

        kernels = self.kernels if self.kernels is not None else [None] * len(self.images)
        kernel = kernels[ifilt]
        work_conv = Templates()
        work_conv.original_shape = getattr(work, "original_shape", self.images[0].shape)
        work_conv.wcs = getattr(work, "wcs", self.wcs[0] if self.wcs is not None else None)
        work_conv._templates = [tmpl_ext]
        conv = work_conv.convolve_templates(kernel, inplace=False)[0]

        k = 1
        if hasattr(self, "fit_bin_factors") and len(self.fit_bin_factors) >= ifilt:
            k = int(self.fit_bin_factors[ifilt - 1])
        if k > 1 and self.config.multi_resolution_method == "upsample":
            dummy_image = np.zeros(self.images[ifilt].shape, dtype=np.byte)
            conv = conv.project_to_block_replicated_grid(k, parent_image=dummy_image)
        final = self._snapshot_template(conv)
        return before, after, final

    def diagnose_sources(
        self,
        source_ids: Sequence[int],
        *,
        ifilt: int = 1,
        half_size: int | None = None,
        save: str | os.PathLike | None = None,
    ):
        """Plot template-construction and fit-residual stages for source IDs.

        The columns are:

        1. high-resolution image stamp on the extracted-template footprint
        2. segmentation map over the same extracted-template footprint
        3. actual extracted template before extension, placed on that same stamp
        4. template after extension, placed on that same stamp
        5. template after matching-kernel convolution/projection
        6. low-resolution image stamp at the same fitting-grid location
        7. final best-fit model image at the same location, including neighbors
        8. final residual image

        Columns 3 and 4 share one display scale so the effect of the
        extension step is directly visible.  The segmentation panel shows the
        target source in gray and each neighbor in a distinct color, with
        label ids printed at the segment centroids.  Requires a completed
        :meth:`run` (or :meth:`load_fit`); the three template panels are
        re-derived for that one source through the primary extraction/
        extension/convolution path, so later in-place mutations cannot alter
        them — stored snapshots are the fallback for runs given externally
        supplied templates.

        Args:
            source_ids: Source ids, one figure row per id. Must not be empty.
            ifilt: Fitted image index (1-based, as elsewhere).
            half_size: Window half-size in pixels; None uses the extracted
                template's footprint.
            save: Optional path to save the figure to.

        Returns:
            Tuple of the created figure and its (nsrc, 8) axes array.
        """
        import matplotlib.pyplot as plt

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        if len(getattr(self, "residuals", [])) < ifilt:
            raise RuntimeError("run() or load_fit() before calling diagnose_sources")

        # in a resumed session (load_fit) there are no in-memory fit templates;
        # every source is rebuilt through the primary per-source path instead
        all_templates = getattr(self, "all_templates", None) or []
        final_templates = all_templates[ifilt - 1] if len(all_templates) >= ifilt else []
        residual = self.residuals[ifilt - 1]
        model = (
            self.model_images[ifilt - 1]
            if len(self.model_images) >= ifilt
            else self.images[ifilt] - residual
        )

        source_ids = [int(sid) for sid in source_ids]
        nsrc = len(source_ids)
        if nsrc == 0:
            raise ValueError("source_ids must not be empty")

        fig, axes = plt.subplots(nsrc, 8, figsize=(24, 3.2 * nsrc), squeeze=False)
        titles = [
            "hires image (ref grid)",
            "segmap",
            "extracted template (ref grid)",
            "after extension (ref grid)",
            "after conv/proj (fit grid)",
            "low-res image (fit grid)",
            "best-fit model (fit grid)",
            "residual (fit grid)",
        ]
        for ax, title in zip(axes[0], titles):
            ax.set_title(title)

        for row, source_id in enumerate(source_ids):
            try:
                before, after, final = self._rebuild_source_stage_templates(source_id, ifilt=ifilt)
            except Exception:
                if self.input_templates is None:
                    raise
                before = self._snapshot_template(
                    self._template_for_source(self.templates_extracted.templates, source_id)
                )
                after = self._snapshot_template(
                    self._template_for_source(self.templates_extended.templates, source_id)
                )
                final = self._snapshot_template(self._template_for_source(final_templates, source_id))
            stamp_slices = self._stamp_slices_for_template(before, self.images[0].shape, half_size)
            ysl, xsl = stamp_slices

            image_stamp = self.images[0][ysl, xsl]
            before_stamp = self._template_on_stamp(before, stamp_slices)
            after_stamp = self._template_on_stamp(after, stamp_slices)
            final_stamp = self._template_on_stamp(final, stamp_slices)
            lowres_stamp = self.images[ifilt][ysl, xsl]
            model_stamp = model[ysl, xsl]
            residual_stamp = residual[ysl, xsl]
            template_center, template_scale = self._diagnostic_display_scale(
                [before_stamp, after_stamp]
            )
            panel_specs = [
                (0, image_stamp, None, None),
                (2, before_stamp, template_center, template_scale),
                (3, after_stamp, template_center, template_scale),
                (4, final_stamp, None, None),
                (5, lowres_stamp, None, None),
                (6, model_stamp, None, None),
                (7, residual_stamp, None, None),
            ]
            for col, data, center, scale in panel_specs:
                self._imshow_scaled(axes[row, col], data, cmap="gray_r", center=center, scale=scale)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

            seg = self._segmap_on_stamp(stamp_slices)
            seg_rgba = np.zeros((*seg.shape, 4), dtype=float)
            seg_rgba[..., 3] = 1.0
            seg_rgba[seg == source_id] = (0.75, 0.75, 0.75, 1.0)
            neighbor_ids = [sid for sid in np.unique(seg[seg > 0]) if int(sid) != int(source_id)]
            neighbor_palette = np.array(
                [
                    (0.10, 0.85, 0.12, 1.0),
                    (0.95, 0.05, 0.05, 1.0),
                    (0.10, 0.20, 0.95, 1.0),
                    (0.95, 0.10, 0.85, 1.0),
                    (0.00, 0.80, 0.95, 1.0),
                    (1.00, 0.60, 0.05, 1.0),
                    (0.65, 0.15, 0.95, 1.0),
                    (0.90, 0.95, 0.05, 1.0),
                    (0.00, 0.65, 0.20, 1.0),
                    (0.95, 0.35, 0.35, 1.0),
                ],
                dtype=float,
            )
            for color_idx, neighbor_id in enumerate(neighbor_ids):
                seg_rgba[seg == neighbor_id] = neighbor_palette[color_idx % len(neighbor_palette)]
            axes[row, 1].imshow(seg_rgba, origin="lower", interpolation="nearest")
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])
            for label_id in np.unique(seg[seg > 0]):
                yy, xx = np.nonzero(seg == label_id)
                if not yy.size:
                    continue
                is_target = int(label_id) == int(source_id)
                axes[row, 1].text(
                    float(np.mean(xx)),
                    float(np.mean(yy)),
                    str(int(label_id)),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8 if is_target else 6,
                    fontweight="bold" if is_target else "normal",
                )
            axes[row, 0].set_ylabel(f"id {source_id}")

        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=180, bbox_inches="tight")
        return fig, axes

    def diagnose_bright_sources(
        self,
        *,
        n: int = 5,
        ifilt: int = 1,
        flux_column: str | None = None,
        half_size: int | None = None,
        save: str | os.PathLike | None = None,
    ):
        """Diagnose the ``n`` brightest fitted sources in one image."""
        if not hasattr(self, "table"):
            raise RuntimeError("run the pipeline before calling diagnose_bright_sources")
        if flux_column is None:
            flux_column = f"flux_{ifilt}"
        if flux_column not in self.table.colnames:
            raise KeyError(f"{flux_column!r} not found in fitted catalog")
        order = np.argsort(np.asarray(self.table[flux_column], dtype=float))[::-1]
        source_ids = [int(self.table["id"][idx]) for idx in order[: int(n)]]
        return self.diagnose_sources(source_ids, ifilt=ifilt, half_size=half_size, save=save)

    def diagnose_source(
        self,
        source_id: int,
        *,
        ifilt: int = 1,
        half_size: int | None = None,
        save: str | os.PathLike | None = None,
    ):
        """Plot the template and residual construction stages for one source."""
        return self.diagnose_sources(
            [int(source_id)],
            ifilt=ifilt,
            half_size=half_size,
            save=save,
        )

    def source_products(
        self,
        source_id: int,
        *,
        ifilt: int = 1,
        half_size: int | None = None,
    ) -> dict:
        """Collect everything the fit produced for one source.

        Works on the in-memory state after :meth:`run` or :meth:`load_fit` —
        nothing is re-extracted or re-convolved.  The stamps and cutouts share
        the source-centered window, so they overlay directly.

        Args:
            source_id: Catalog id.
            ifilt: Fitted image index (1-based, as elsewhere).
            half_size: Window half-size in pixels of each grid; None uses the
                template footprint.

        Returns:
            Dict with the template stamps (``tmpl_hi``, ``tmpl_lo``), matching
            cutouts (``img_hi``, ``segmap``, ``img_lo``, ``model``,
            ``residual``), the band PSFs at the source position (``psf_hi``,
            ``psf_lo``), fitted scalars (``flux``, ``err``, ``err_pred``,
            ``ee_psf_lo``, ``flag``, ``shift``, ``position``), the fit-table
            ``row``, and the window slices (``slices_hi``, ``slices_lo``).
            Hi-grid entries are None when the source has no hi-res template.
        """
        if not getattr(self, "all_templates", None):
            raise RuntimeError("run() or load_fit() first")
        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        t_lo = next(
            (t for t in self.all_templates[ifilt - 1] if int(t.id) == int(source_id)),
            None,
        )
        if t_lo is None:
            raise KeyError(f"source id {source_id} not found in fitted templates")
        t_hi = next(
            (t for t in self.tmpls.templates if int(t.id) == int(source_id)), None
        )

        out: dict[str, Any] = {"id": int(source_id), "ifilt": int(ifilt)}
        sl_lo = self._stamp_slices_for_template(t_lo, self.images[ifilt].shape, half_size)
        out["slices_lo"] = sl_lo
        out["tmpl_lo"] = self._template_on_stamp(t_lo, sl_lo)
        out["img_lo"] = np.asarray(self.images[ifilt][sl_lo])
        out["model"] = (
            np.asarray(self.model_images[ifilt - 1][sl_lo])
            if len(self.model_images) >= ifilt
            else None
        )
        out["residual"] = (
            np.asarray(self.residuals[ifilt - 1][sl_lo])
            if len(self.residuals) >= ifilt
            else None
        )
        if t_hi is not None:
            sl_hi = self._stamp_slices_for_template(t_hi, self.images[0].shape, half_size)
            out["slices_hi"] = sl_hi
            out["tmpl_hi"] = self._template_on_stamp(t_hi, sl_hi)
            out["img_hi"] = np.asarray(self.images[0][sl_hi])
            out["segmap"] = self._segmap_on_stamp(sl_hi)
        else:
            out["slices_hi"] = out["tmpl_hi"] = out["img_hi"] = out["segmap"] = None

        psf_hi, psf_lo = self._band_psfs(ifilt)
        src = t_hi if t_hi is not None else t_lo
        x, y = (float(v) for v in src.input_position_original)
        ra = dec = None
        if self.wcs is not None and self.wcs[0] is not None and t_hi is not None:
            ra, dec = (float(v) for v in self.wcs[0].wcs_pix2world(x, y, 0))

        def stamp(p: np.ndarray | PSFRegionMap | None) -> np.ndarray | None:
            if p is None:
                return None
            if isinstance(p, PSFRegionMap):
                return p.get_psf(ra, dec)
            return np.asarray(p)

        out["psf_hi"] = stamp(psf_hi)
        out["psf_lo"] = stamp(psf_lo)

        out["position"] = (x, y)
        out["flux"] = float(t_lo.flux)
        out["err"] = float(t_lo.err)
        out["err_pred"] = float(t_lo.err_pred)
        out["ee_psf_lo"] = float(t_lo.ee_psf_lo)
        out["flag"] = int(t_lo.flag)
        out["shift"] = tuple(float(v) for v in np.asarray(t_lo.shifted))
        out["row"] = None
        table = getattr(self, "table", None)
        if table is not None:
            match = np.flatnonzero(np.asarray(table["id"], dtype=int) == int(source_id))
            if match.size:
                out["row"] = table[int(match[0])]
        return out

    def show_sources(
        self,
        source_ids: int | Sequence[int],
        *,
        ifilt: int = 1,
        half_size: int | None = None,
        save: str | os.PathLike | None = None,
    ):
        """Quicklook of the fitted products and stamps, one row per source.

        Columns: hi-res image, hi-res template, convolved template, low-res
        image, best-fit model, residual, and the two PSF stamps at the source
        position.  Image, model, and residual share one display scale so the
        subtraction is judged by eye.  Works after :meth:`run` or
        :meth:`load_fit`.

        Args:
            source_ids: One id or a sequence of ids.
            ifilt: Fitted image index (1-based, as elsewhere).
            half_size: Window half-size in pixels; None uses each template's
                footprint.
            save: Optional path to save the figure to.

        Returns:
            Tuple of the created figure and its (nsrc, 8) axes array.
        """
        import matplotlib.pyplot as plt

        ids = [int(s) for s in np.atleast_1d(source_ids)]
        if not ids:
            raise ValueError("source_ids must not be empty")

        fig, axes = plt.subplots(
            len(ids), 8, figsize=(20, 2.6 * len(ids)), squeeze=False
        )
        titles = [
            "hi image", "tmpl_hi", "tmpl_lo", "lo image",
            "model", "residual", "psf_hi", "psf_lo",
        ]
        for ax, title in zip(axes[0], titles):
            ax.set_title(title)

        for row, sid in enumerate(ids):
            p = self.source_products(sid, ifilt=ifilt, half_size=half_size)
            shared = self._diagnostic_display_scale(
                [a for a in (p["img_lo"], p["model"], p["residual"]) if a is not None]
            )
            panels = [
                (p["img_hi"], None),
                (p["tmpl_hi"], None),
                (p["tmpl_lo"], None),
                (p["img_lo"], shared),
                (p["model"], shared),
                (p["residual"], shared),
                (p["psf_hi"], None),
                (p["psf_lo"], None),
            ]
            for col, (data, scale) in enumerate(panels):
                ax = axes[row, col]
                if data is None:
                    ax.set_axis_off()
                    continue
                if scale is None:
                    self._imshow_scaled(ax, data)
                else:
                    self._imshow_scaled(ax, data, center=scale[0], scale=scale[1])
                ax.set_xticks([])
                ax.set_yticks([])
            axes[row, 0].set_ylabel(
                f"id {sid}\nflux {p['flux']:.4g} ± {p['err']:.2g}", fontsize=8
            )

        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig, axes

    def diagnose_subphot(
        self,
        source_id: int,
        *,
        ifilt: int = 1,
        size: int | None = None,
        rlim: float | None = None,
        nsig: float = 3.0,
        sys_err: float = 0.02,
        photbin: int = 1,
        raper: float | None = None,
        save: str | os.PathLike | None = None,
    ) -> np.ndarray:
        """IDL subphot-style 6-panel diagnostic (img/tmpl/seg/model/res/clean).

        Pixel-for-pixel port of the legacy ``subphot.pro`` ``mkdiag``/``fptv``
        diagnostic so outputs compare 1-1 against IDL PNGs of the same source:
        same panel layout (2x3 at 2x nearest-neighbour zoom), byte scalings,
        background/rms estimator (aperture-scale block sums, 2-sigma clipped,
        ``prms = rms/na``), circular ``rlim`` fit mask on res/clean, and the
        distance-sorted 5-level grayscale segmap colouring. Works in-session
        after :meth:`run` and in a fresh session after :meth:`load_fit`
        (the source's template is rebuilt and the saved fitted flux/shift
        applied from the template table).

        Panels: ``img`` = low-res stamp at ``+-nsig*prms``; ``tmpl`` = hi-res
        template at ``median +- 8*robust_sigma``; ``seg`` = colour-cycled
        segmap minus ``0.1*mask``; ``model`` = full model at ``+-nsig*prms``;
        ``res`` = masked ``(img-model)/err`` at ``+-nsig``; ``clean`` = masked
        image minus neighbour models at ``+-nsig*prms``.

        Args:
            source_id: Catalog id of the source to centre on.
            ifilt: Fitted image index (fit grid must equal the reference grid).
            size: Stamp side in fit-grid pixels (IDL ``tsz``); default the
                source's template-footprint size, made odd. Pass the IDL tile
                size for exact comparisons.
            rlim: Fit-mask radius in pixels; default ``(size-1)/2``.
            nsig: Display stretch in sigma. Default 3 matches the survey-era
                ``subphot_nsigma=3`` runs (the IDL code default was 5).
            sys_err: Systematic error fraction in ``err = sqrt(prms^2 +
                (sys_err*model)^2)`` (IDL default 0.02).
            photbin: Optional SNR-preserving display binning of the
                photometry-based panels (IDL ``photbin``).
            raper: Aperture radius in pixels for the rms block size ``na``;
                default from the fit configuration.
            save: Optional PNG output path.

        Returns:
            The rendered RGB image as a ``(4*size, 6*size, 3)`` uint8 array.
        """
        from astropy.stats import sigma_clipped_stats
        from PIL import Image, ImageDraw, ImageFont

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        have_run = bool(getattr(self, "all_templates", None))
        if not have_run and len(getattr(self, "model_images", [])) < ifilt:
            raise RuntimeError("run() or load_fit() before calling diagnose_subphot")
        if self.images[ifilt].shape != self.images[0].shape:
            raise NotImplementedError(
                "diagnose_subphot requires the fit grid to match the reference grid"
            )

        if have_run:
            templates = self.all_templates[ifilt - 1]
            own = [
                t
                for t in templates
                if int(t.id_parent if getattr(t, "parent_id", None) is not None else t.id)
                == int(source_id)
            ]
            if not own:
                raise KeyError(f"source id {source_id} not found in fitted templates")
        else:
            # resumed session: rebuild this source's convolved template and
            # apply the saved fitted shift/flux from the template table
            templates = None
            _, _, final = self._rebuild_source_stage_templates(source_id, ifilt=ifilt)
            flux = None
            ttab = getattr(self, "template_table", None)
            if ttab is not None:
                rows = ttab[np.asarray(ttab["id_parent"], dtype=int) == int(source_id)]
                if len(rows):
                    final.to_shift[:] = [float(rows["dx"][0]), float(rows["dy"][0])]
                    Templates.apply_template_shifts([final])
                    flux = float(np.sum(rows["flux"]))
            if flux is None:
                idx = np.flatnonzero(np.asarray(self.table["id"], dtype=int) == int(source_id))
                flux = float(self.table[f"flux_{ifilt}"][int(idx[0])])
            final.flux = flux
            own = [final]

        if size is None:
            size = max(own[0].data.shape)
            size += (size + 1) % 2
        size = int(size)
        if rlim is None:
            rlim = (size - 1) / 2.0

        xc_full, yc_full = self._source_position(source_id)
        ny, nx = self.images[0].shape
        x1 = int(np.clip(round(xc_full - (size - 1) / 2.0), 0, nx - size))
        y1 = int(np.clip(round(yc_full - (size - 1) / 2.0), 0, ny - size))
        sl = (slice(y1, y1 + size), slice(x1, x1 + size))

        tphot = np.asarray(self.images[ifilt][sl], dtype=float).copy()
        ttmpl = np.asarray(self.images[0][sl], dtype=float)
        tseg = np.asarray(self.segmap[sl])
        res = np.asarray(self.residuals[ifilt - 1][sl], dtype=float).copy()
        model = np.asarray(self.model_images[ifilt - 1][sl], dtype=float)
        zero0 = tphot == 0.0

        w_full = self.weights[ifilt] if self.weights is not None else None
        if w_full is None:
            wht = np.ones_like(tphot)
        elif w_full.shape == self.images[ifilt].shape:
            wht = np.asarray(w_full[sl], dtype=float)
        else:
            k = self.images[ifilt].shape[0] // w_full.shape[0]
            wht = (
                np.repeat(np.repeat(np.asarray(w_full, dtype=float), k, 0), k, 1)[sl]
                * k**2
            )

        own_img = np.zeros_like(tphot)
        for t in own:
            own_img += float(t.flux) * self._template_on_stamp(t, sl)
        nn_img = model - own_img

        # background + per-pixel rms on aperture scales (IDL tile_stat on rbin)
        if raper is None:
            raper = self._resolve_image_ap_radius_pix(ifilt, self.config)
        na = max(int(raper * np.sqrt(np.pi) / np.sqrt(2.0)), 1)
        pos = wht > 0
        rms_glob = 1.0 / np.sqrt(np.median(wht[pos])) if np.any(pos) else float(mad_std(res))
        bgmask = (model / 15.0 < rms_glob) & (ttmpl != 0) & pos
        dif = np.where(bgmask, res, np.nan)
        nb = size // na
        bsum = dif[: nb * na, : nb * na].reshape(nb, na, nb, na).sum(axis=(1, 3))
        bsum = bsum[np.isfinite(bsum)]
        if bsum.size >= 5:
            _, bgmed1, bgrms1 = sigma_clipped_stats(
                bsum, sigma=2.0, maxiters=5, stdfunc="mad_std"
            )
            prms = float(bgrms1) / na
            tphot -= float(bgmed1) / na**2
            res = res - float(bgmed1) / na**2
        else:
            prms = rms_glob
        if not np.isfinite(prms) or prms <= 0.0:
            prms = rms_glob if rms_glob > 0 else 1.0
        scl = nsig * prms
        err_img = np.sqrt(prms**2 + (sys_err * model) ** 2)

        yy, xx = np.mgrid[0:size, 0:size]
        d = np.hypot(xx - (xc_full - x1), yy - (yc_full - y1))
        mask = ((d >= rlim) | zero0).astype(float)

        # segmap colouring: cycle 5 gray levels through fitted ids by distance
        if templates is not None:
            src = [(int(t.id), *t.position_original) for t in templates]
        else:
            tt = getattr(self, "template_table", None)
            tab = tt if tt is not None else self.table
            src = list(
                zip(
                    np.asarray(tab["id"], dtype=int),
                    np.asarray(tab["x"], dtype=float),
                    np.asarray(tab["y"], dtype=float),
                )
            )
        in_stamp = [
            s for s in src if x1 <= s[1] < x1 + size and y1 <= s[2] < y1 + size
        ]
        order = sorted(
            in_stamp,
            key=lambda s: (s[1] - xc_full) ** 2 + (s[2] - yc_full) ** 2,
        )
        lv = [0.2, 0.8, 0.4, 0.6, 1.0]
        tvseg = tseg.astype(float)
        for i, (sid, _, _) in enumerate(order):
            tvseg[tseg == sid] = lv[i % 5]

        panels = [
            _fptv_panel(tphot, mm=(-scl, scl), bin=photbin),
            _fptv_panel(ttmpl, fac=8.0),
            _fptv_panel(tvseg - 0.1 * mask, mm=(-0.2, 1.0)),
            _fptv_panel(model, mm=(-scl, scl), bin=photbin),
            _fptv_panel(res / err_img * (1.0 - mask), mm=(-nsig, nsig), bin=photbin),
            _fptv_panel((tphot - nn_img) * (1.0 - mask), mm=(-scl, scl), bin=photbin),
        ]

        t2 = 2 * size
        canvas = np.zeros((2 * t2, 3 * t2), dtype=np.uint8)
        for i, p in enumerate(panels):
            r, c = divmod(i, 3)
            canvas[r * t2 : (r + 1) * t2, c * t2 : (c + 1) * t2] = p[::-1]

        rgb = Image.fromarray(canvas).convert("RGB")
        draw = ImageDraw.Draw(rgb)
        try:
            import matplotlib

            font = ImageFont.truetype(
                str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"),
                15,
            )
        except Exception:
            font = ImageFont.load_default()
        labels = [["img", "tmpl", "seg"], ["model", "res", "clean"]]
        for r, row in enumerate(labels):
            for c, lab in enumerate(row):
                xy = (c * t2 + 5, r * t2 + 20)
                try:
                    draw.text(xy, lab, fill=(255, 255, 255), font=font, anchor="ls")
                except (TypeError, ValueError):
                    draw.text((xy[0], xy[1] - 12), lab, fill=(255, 255, 255), font=font)

        if save is not None:
            rgb.save(save)
        return np.asarray(rgb)

    def plot_result(
        self,
        ifilt: int = 1,
        scene_id: int | None = None,
        source_id: int | None = None,
        display_sig: float = 3.0,
    ) -> tuple["matplotlib.figure.Figure", np.ndarray]:
        """Plot the fitted image, model, residual, and color composite.

        The high-resolution template image (``images[0]``) is shown with scene
        overlays alongside the segmentation map, the selected low-resolution
        image, its model, and the residual. A Lupton RGB image combining the
        template and low-resolution images is also displayed.

        Args:
            ifilt: Index of the low-resolution image to display. Defaults to ``1``.
            scene_id: Optional scene identifier to zoom into. Defaults to ``None``.
            source_id: Optional source identifier to zoom into. Defaults to
                ``None``. Ignored if ``scene_id`` is provided.

        Returns:
            Tuple containing the created figure and the array of axes.
        """


        import matplotlib.pyplot as plt
        import numpy as np
        from copy import deepcopy
        from astropy.visualization import make_lupton_rgb
        from photutils.segmentation import SegmentationImage
        from astropy.table import Table

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("idx must be between 1 and len(images)-1")

        all_scenes = getattr(self, "all_scenes", None)
        scene_list = all_scenes[ifilt - 1] if all_scenes else None
        if not scene_list:
            raise RuntimeError(
                "plot_result needs the scenes of a completed run(); load_fit does "
                "not restore them (see its docstring)"
            )
        nscenes = len(scene_list)

        segmap = self.segmap
        segm = SegmentationImage(as_label_array(segmap))
        segmap_cmap = segm.cmap
        scene_cmap = deepcopy(segmap_cmap)
        scene_cmap.colors[0] = (1.0, 1.0, 1.0, 0.0)

        # Scene id per segment, painted onto the segmentation grid. Built
        # unconditionally: the old guard tested `hasattr(self, "scenes")`, which
        # has been permanently true since `scenes` became a property, so the map
        # was never built and the lookup below raised NameError.
        logger.info("Building scene map for diagnostics")
        scene_map = np.zeros_like(segmap, dtype=int)
        # One label -> index map for the whole loop. segm.get_index validates
        # each label with np.setdiff1d against the full label list, i.e. one
        # sort of every label in the field per template.
        label_index = {int(lab): i for i, lab in enumerate(segm.labels)}
        for scene in scene_list:
            for tmpl in scene.templates:
                iseg = label_index.get(int(tmpl.id))
                if iseg is None:
                    continue  # template with no surviving segment
                sl = segm.segments[iseg].slices
                scene_map[sl][segm.data[sl] == tmpl.id] = scene.id

        logger.info(f"Plotting image {ifilt} with {nscenes} scenes")

        mask: np.ndarray | None = None
        if scene_id is not None:
            mask = scene_map == scene_id
        elif source_id is not None:
            mask = segmap == source_id

        buf = 10
        if mask is not None and np.any(mask):
            ys, xs = np.where(mask)
            y0, y1 = max(ys.min() - buf, 0), min(ys.max() + buf, segmap.shape[0]) + 1
            x0, x1 = max(xs.min() - buf, 0), min(xs.max() + buf, segmap.shape[1]) + 1
        else:
            y0, x0 = 0, 0
            y1, x1 = segmap.shape

        sl_hi = (slice(y0, y1), slice(x0, x1))
        # A pipeline built straight from arrays has no WCS; fall back to the
        # shape ratio, which is the same integer factor for block-aligned grids.
        wcs_list = getattr(self, "wcs", None)
        if wcs_list is not None and wcs_list[0] is not None and wcs_list[ifilt] is not None:
            kbin = bin_factor_from_wcs(wcs_list[0], wcs_list[ifilt])
        else:
            kbin = max(1, int(round(self.images[0].shape[0] / self.images[ifilt].shape[0])))
        y0_lo, y1_lo, x0_lo, x1_lo = np.round(bin_remap([y0, y1, x0, x1], kbin)).astype(int)
        sl_lo = (slice(y0_lo, y1_lo), slice(x0_lo, x1_lo))

        img_hi = self.images[0]
        img_lo = self.images[ifilt]

        img_cut = img_lo[sl_lo]
        # run() and load_fit both populate model_images; the old fitter
        # object this used to reach through never existed on Pipeline.
        model_cut = self.model_images[ifilt - 1][sl_lo]

        tmpl_cut = img_hi[sl_hi]
        seg_cut = segmap[sl_hi]
        scenes_cut = scene_map[sl_hi]
        # @@@ for now assume upsampled residual image
        res_cut = self.residuals[ifilt - 1][sl_hi]

        # RGB composite using template as blue and low-res as red
        tmpl_cut_lo = block_reduce(tmpl_cut, kbin, func=np.mean)
        b = tmpl_cut_lo / np.nanstd(tmpl_cut_lo) if np.nanstd(tmpl_cut_lo) != 0 else tmpl_cut_lo
        r = img_cut / np.nanstd(img_cut) if np.nanstd(img_cut) != 0 else img_cut
        g = (r + b) / 2.0
        col_cut = make_lupton_rgb(r, g, b, stretch=display_sig / 1.5)

        # aspect is w/h
        aspect = img_cut.shape[1] / img_cut.shape[0]

        fig, ax = plt.subplots(3, 2, figsize=(10, 13 / aspect))
        ax = ax.flatten()
        images = [
            tmpl_cut,
            seg_cut,
            img_cut,
            model_cut,
            res_cut,
            col_cut,
        ]
        titles = [
            f"template + scenes",
            "segmap",
            f"image{ifilt}",
            f"model image{ifilt}",
            "residual",
            "color",
        ]

        for i, (im, title) in enumerate(zip(images, titles)):
            if title == "segmap":
                ax[i].imshow(im, origin="lower", cmap=segmap_cmap, interpolation="nearest")
                # if plotting a scene, overplot template id as text
                if scene_id is not None or source_id is not None:
                    for scene in scene_list:
                        if scene.id != scene_id:
                            continue
                        for tmpl in scene.templates:
                            x, y = tmpl.position_original - np.array([x0, y0])
                            ax[i].text(
                                x,
                                y,
                                str(tmpl.id),
                                color="white",
                                fontsize=6,
                                ha="center",
                                va="center",
                            )
            elif title == "color":
                ax[i].imshow(im, origin="lower", interpolation="nearest")
            else:
                ivalid = img_cut != 0
                v = (
                    display_sig * np.nanstd(img_cut[ivalid])
                    if np.any(np.isfinite(img_cut[ivalid]))
                    else 1.0
                )
                ax[i].imshow(im, origin="lower", cmap="gray", vmin=-v, vmax=v)
                if i == 0:
                    # set background of segmap to transparent
                    ax[i].imshow(
                        scenes_cut,
                        origin="lower",
                        cmap=scene_cmap,
                        alpha=0.5,
                        interpolation="nearest",
                    )
            ax[i].set_title(title)

        plt.tight_layout()
        return fig, ax


def run(
    images: Sequence[np.ndarray],
    segmap: np.ndarray,
    *,
    catalog: Table | None = None,
    psfs: Sequence[np.ndarray] | None = None,
    weights: Sequence[np.ndarray] | None = None,
    wht_images: Sequence[np.ndarray] | None = None,
    kernels: Sequence[np.ndarray | PSFRegionMap] | None = None,
    psf_throughputs: Sequence[float] | None = None,
    wcs: Sequence[WCS] | None = None,
    window: Window | None = None,
    extend_mode: str | None = None,
    extend_templates: str | None = None,
    templates: Templates | Sequence[Template] | None = None,
    config: FitConfig | None = None,
) -> tuple[Table, list[np.ndarray], Pipeline]:
    """Backward compatible wrapper for :class:`Pipeline`"""

    pipeline = Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=weights,
        wht_images=wht_images,
        kernels=kernels,
        psf_throughputs=psf_throughputs,
        wcs=wcs,
        window=window,
        extend_mode=extend_mode,
        extend_templates=extend_templates,
        templates=templates,
        config=config,
    )
    table, residuals = pipeline.run()
    return table, residuals, pipeline


STEPS = {
    "psfs": "build_psfs",
    "kernels": "build_kernels",
    # field-level preparation, shared by every band: 'prep' is psfs + repair in
    # one job; 'repair' is the second half alone, for clusters whose compute
    # nodes cannot reach MAST and must build the grids elsewhere first
    "prep": "prep",
    "repair": "build_repair_cache",
    "load": "load_data",
    "loadfit": "load_fit",
    "info": "info",
    "fit": "run",
    "outputs": "write_outputs",
    "all": "run_all",
}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: python -m mophongo.pipeline config.json [steps]."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Config-driven mophongo photometry run (see pipeline.RunConfig)"
    )
    ap.add_argument("config", help="JSON run config (see mophongo.pipeline.RunConfig)")
    # No `choices=` here: with nargs="*" argparse checks the collected list
    # against choices as a single value, so a bare invocation dies with
    # "invalid choice: []". Validate the steps by hand instead.
    ap.add_argument(
        "steps",
        nargs="*",
        metavar="step",
        help=f"steps to run, any of {', '.join(STEPS)} (default: all)",
    )
    args = ap.parse_args(argv)
    steps = args.steps or ["all"]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(
            f"invalid step: {', '.join(unknown)} (choose from {', '.join(STEPS)})"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipe = Pipeline.from_config(args.config)
    # Every invocation logs to <out_dir>/<name>.log, not just `all`. Naming
    # steps explicitly used to bypass log_run entirely, so a run whose console
    # output was not redirected left no record next to its own products --
    # exactly the runs worth having a record of. log_run is reentrant, so the
    # `all` step's own block nests harmlessly.
    with pipe.log_run() as log_path:
        logger.info("logging this run to %s", log_path)
        started = time.perf_counter()
        for step in steps:
            with pipe._phase(f"step: {step}"):
                getattr(pipe, STEPS[step])()
        pipe.report_timings(time.perf_counter() - started)


if __name__ == "__main__":
    main()
