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
import psutil
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence
from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict

from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, block_replicate, block_reduce
from astropy.stats import mad_std
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs.utils import proj_plane_pixel_scales

from .psf_map import PSFRegionMap
from .utils import as_label_array, bin_factor_from_wcs, downsample_psf, bin_remap
from .templates import Templates, Template, _slices_from_bbox
from .fit import FitConfig as _FitConfig
from .scene import generate_scenes

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # show info for *this* logger only
if not logger.handlers:  # avoid duplicate handlers on reloads
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(module)s.%(funcName)s: %(message)s"))
    logger.addHandler(handler)

memory = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9


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
    # mosaic used for DrizzlePSF footprints/grid of the hi-res side
    # (defaults to sci_hi; set when sci_hi is a derived template image)
    driz_hi: str | None = None
    # --- PSFs -------------------------------------------------------------
    psf_dir: str = "data/PSF"
    pattern_hi: str = ""  # STDPSF filename regex for the hi-res band
    pattern_lo: str = ""  # STDPSF filename regex for the lo-res band
    filter_lo: str = ""  # lo-res filter name, e.g. "f770w" (blur lookup)
    # PSF stamp size in arcsec; None = full native ePSF stamp as generated
    psf_size: float | None = 4.0
    psf_autobuild: bool = True  # generate missing PSF grids with PSFFactory
    psf_fov_arcsec: float | None = None  # PSFFactory field of view; None = backend default
    # extra Gaussian broadening of the lo-res model PSF (FWHM arcsec);
    # "default" = mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC[filter_lo],
    # a number = that value, None = no broadening
    psf_blur_fwhm: float | str | None = "default"
    # optional [n_frames_hi, n_frames_lo] sanity assert on the WCS csvs
    expect_frames: list[int] | None = None
    # --- templates --------------------------------------------------------
    # how to fill the template outside its segment: "psf_wings" (default) adds
    # the high-resolution PSF beyond the segmentation footprint, "psf_model"
    # replaces the template by the PSF, None leaves it truncated. Without an
    # extension the total flux is biased low, badly so for faint sources.
    extend_templates: str | None = "psf_wings"
    # --- preprocessing ----------------------------------------------------
    bg_filter_sigma: float = 64.0  # get_bg_and_ivar background filter
    footprint_filter: bool = True  # keep only sources with wht_lo > 0
    r_trial: float = 0.0  # trial-patch radius in arcmin; 0 = full run
    trial_center: list[float] | None = None  # [ra, dec] deg of the patch
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
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        from dataclasses import asdict

        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


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
    image_hi = block_replicate(image, k, conserve_sum=True).astype(np.float32)
    weight_hi = None
    if weight is not None:
        weight_hi = block_replicate(weight, k, conserve_sum=False).astype(np.float32) * k**2
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
    r"(?P<mjd>_MJD[^_]+)?_GRID(?P<n>\d+)_(?P<samp>OS\d+|DET)$"
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
        return float(np.nanmean(sums[valid]))
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
        extend_templates: str | None = None,
        templates: Templates | Sequence[Template] | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if psfs is not None and len(images) != len(psfs):
            raise ValueError("Number of images and PSFs must match")
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
        self.extend_templates = extend_templates
        self.input_templates = templates
        self.config = config

        if kernels is None:
            kernels = [None] * len(images)
        if psfs is None:
            psfs = [None] * len(images)

        self.residuals: list[np.ndarray] = []
        self.fit: list[np.ndarray] = []
        self.astro: list[np.ndarray] = []
        #        self.templates: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.tmpls: Templates | None = None
        self.templates_extracted: Templates | None = None
        self.templates_extended: Templates | None = None
        self.model_images: list[np.ndarray] = []

        if not hasattr(self, "run_config"):
            self.run_config = None

        print(f"Pipeline (init) memory: {memory():.1f} GB")

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
    @classmethod
    def from_config(cls, path: str | Path | RunConfig) -> "Pipeline":
        """Create a deferred Pipeline from a JSON run config.

        ``path`` may be the config JSON, a directory holding exactly one
        ``*.json`` (e.g. a finished run's ``out_dir``, which carries a copy
        of its config), or a :class:`RunConfig`. Data are loaded lazily:
        :meth:`run` (or :meth:`load_data`) reads the images and finishes
        construction. Relative paths inside the config still resolve
        against the process working directory.
        """
        if not isinstance(path, RunConfig):
            p = Path(path)
            if p.is_dir():
                candidates = sorted(p.glob("*.json"))
                if len(candidates) != 1:
                    found = [c.name for c in candidates] or "none"
                    raise FileNotFoundError(
                        f"expected exactly one run config JSON in {p}, found {found}"
                    )
                path = candidates[0]
        cfg = path if isinstance(path, RunConfig) else RunConfig.from_json(path)
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

    @property
    def scenes(self):
        """Scenes of the (first) fitted band, once :meth:`run` has completed."""
        return self.all_scenes[0] if getattr(self, "all_scenes", None) else None

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
                driz_image=str(cfg.driz_hi or cfg.sci_hi), csv_file=str(cfg.csv_hi)
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

    def _load_epsf(self, dpsf, pattern: str, csv: str, band: str) -> None:
        """Load the ePSF grids for one band, generating them if absent.

        ``load_jwst_stdpsf`` matches files under ``psf_dir`` against *pattern*
        and loads nothing when none match, so a missing grid would otherwise
        pass unnoticed and the run would continue without a PSF. When
        ``psf_autobuild`` is set the grids are generated from the band's
        exposure list; either way an empty result is an error.
        """
        cfg = self.run_config
        dpsf.epsf_obj.load_jwst_stdpsf(local_dir=str(cfg.psf_dir), filter_pattern=pattern)
        if dpsf.epsf_obj.epsf:
            logger.info(
                "%s-res band: loaded %d ePSF grid(s) matching %r",
                band, len(dpsf.epsf_obj.epsf), pattern,
            )
            return

        if not cfg.psf_autobuild:
            raise FileNotFoundError(
                f"no PSF grids under {cfg.psf_dir} match {pattern!r} for the "
                f"{band}-res band, and psf_autobuild is off"
            )

        from .psf_factory import PSFFactory

        kw = _psf_factory_kwargs(pattern)
        logger.warning(
            "no PSF grids match %r under %s; generating them from %s with "
            "PSFFactory(%s). This is slow.",
            pattern, cfg.psf_dir, csv,
            ", ".join(f"{k}={v!r}" for k, v in sorted(kw.items())),
        )
        Path(cfg.psf_dir).mkdir(parents=True, exist_ok=True)
        PSFFactory(outdir=str(cfg.psf_dir), fov_arcsec=cfg.psf_fov_arcsec, **kw).from_csv(
            str(csv), save=True
        )
        dpsf.epsf_obj.load_jwst_stdpsf(local_dir=str(cfg.psf_dir), filter_pattern=pattern)
        if not dpsf.epsf_obj.epsf:
            raise FileNotFoundError(
                f"PSFFactory ran for the {band}-res band but no file under "
                f"{cfg.psf_dir} matches {pattern!r}; the pattern and the "
                "generated filenames disagree"
            )
        logger.info(
            "%s-res band: generated and loaded %d ePSF grid(s)",
            band, len(dpsf.epsf_obj.epsf),
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

        The hi-res map is needed as well as the lo-res one: it is the PSF that
        :meth:`run` extends the templates with when ``extend_templates`` is set.
        """
        if self.prm_lo is None and self.f_psf_lo.exists():
            self.prm_lo = PSFRegionMap.from_geojson(str(self.f_psf_lo))
        if self.prm_hi is None and self.f_psf_hi.exists():
            self.prm_hi = PSFRegionMap.from_geojson(str(self.f_psf_hi))
        if self.prm_kern is None:
            self.build_kernels()

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
        tmpl_hi = fits.getdata(cfg.sci_hi)
        sci_lo = fits.getdata(cfg.sci_lo)
        wht_lo = fits.getdata(cfg.wht_lo)
        # Normalise the label dtype once, here at the boundary: releases differ
        # (MINERVA COSMOS ships float64 where UDS and EGS ship int32) and every
        # downstream SegmentationImage would otherwise have to defend itself.
        segmap = as_label_array(fits.getdata(cfg.segmap))
        cat = Table.read(cfg.catalog)

        if cfg.footprint_filter:
            scale_hi = proj_plane_pixel_scales(wcs_hi)[0]
            scale_lo = proj_plane_pixel_scales(wcs_lo)[0]
            k = round(float(scale_lo / scale_hi))
            ix = np.clip((np.asarray(cat["x"]) / k).astype(int), 0, wht_lo.shape[1] - 1)
            iy = np.clip((np.asarray(cat["y"]) / k).astype(int), 0, wht_lo.shape[0] - 1)
            cat = cat[wht_lo[iy, ix] > 0]
            logger.info("%d sources inside the lo-res footprint", len(cat))

        if cfg.r_trial and cfg.r_trial > 0:
            import astropy.units as u
            from astropy.coordinates import SkyCoord

            if not cfg.trial_center:
                raise ValueError("r_trial > 0 requires trial_center=[ra, dec]")
            coords = SkyCoord(
                ra=np.asarray(cat["ra"], float) * u.deg,
                dec=np.asarray(cat["dec"], float) * u.deg,
            )
            ref = SkyCoord(
                ra=cfg.trial_center[0] * u.deg, dec=cfg.trial_center[1] * u.deg
            )
            cat = cat[coords.separation(ref) < cfg.r_trial * u.arcmin]
            logger.info("r_trial=%.2f': %d sources", cfg.r_trial, len(cat))

        bg, ivar = get_bg_and_ivar(sci_lo, wht_lo, bg_filter_sigma=cfg.bg_filter_sigma)
        sci_fit = sci_lo - bg
        # zero non-finite pixels in image AND weight so they carry no information
        bad = ~np.isfinite(sci_fit)
        sci_fit[bad] = 0.0
        ivar[bad] = 0.0
        ivar[~np.isfinite(ivar)] = 0.0
        np.nan_to_num(tmpl_hi, copy=False)

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
            weights=[None, ivar],
            catalog=cat,
            psfs=[self.prm_hi, self.prm_lo],
            kernels=[None, self.prm_kern],
            wcs=[wcs_hi, wcs_lo],
            config=_FitConfig(**cfg.fit),
            extend_templates=cfg.extend_templates,
        )
        return self

    # -- inspection --------------------------------------------------------
    def __repr__(self) -> str:
        cfg = getattr(self, "run_config", None)
        name = f" {cfg.name!r}" if cfg is not None else ""
        if getattr(self, "images", None) is None:
            stage = "configured"
        elif getattr(self, "table", None) is not None:
            stage = "fitted"
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
            for key in ("sci_hi", "segmap", "catalog", "sci_lo", "wht_lo", "csv_hi", "csv_lo"):
                path = Path(getattr(cfg, key))
                if not path.exists():
                    lines.append(f"  {key:8s} MISSING  {path}")
                    continue
                desc = f"{path.stat().st_size / 1e6:8.1f} MB"
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

        if getattr(self, "images", None) is None:
            lines.append("data: not loaded — load_data() reads images and catalog")
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
                f"{len(self.residuals)} residual image(s)"
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

        Args:
            sources: Overlay catalog positions on the hi-res panel.
            save: Optional path to save the figure to.

        Returns:
            Tuple of the created figure and its flat array of axes.
        """
        import matplotlib.pyplot as plt
        from photutils.segmentation import SegmentationImage

        if getattr(self, "images", None) is None:
            raise RuntimeError("no data loaded — call load_data() first")

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

    # -- step 4: outputs ---------------------------------------------------
    def write_outputs(self) -> "Pipeline":
        """Write residual FITS, fit table, and scene diagnostics."""
        from astropy.io import fits

        if self.table is None:
            raise RuntimeError("run() first")
        cfg = self.run_config
        stem = self.out_dir / cfg.name
        # residual is on the hi-res reference grid (upsample path)
        fits.writeto(
            f"{stem}_residual.fits",
            self.residuals[0],
            fits.getheader(cfg.sci_hi),
            overwrite=True,
        )
        self.table.write(f"{stem}_fit_table.fits", overwrite=True)
        if cfg.save_stamps:
            self.write_stamps()

        rows = []
        for s in self.scenes:
            xy = np.mean([t.position_original for t in s.templates], axis=0)
            ra, dec = self.wcs[0].wcs_pix2world([xy], 0)[0]
            rows.append((s.id, len(s.templates), int(s.is_bright.sum()), ra, dec))
            if cfg.scene_plots:
                import matplotlib.pyplot as plt

                fig, _ = s.plot(self.images[0], self.segmap, display_sig=5)
                fig.savefig(f"{stem}_scene_{s.id}.png", dpi=300)
                plt.close(fig)
        scene_table = Table(
            rows=rows, names=["id", "n_templates", "is_bright", "ra", "dec"]
        )
        scene_table["minerva_link"] = [
            f"https://minerva.colorado.edu/?ra={ra}&dec={dec}&zoom=7"
            for ra, dec in zip(scene_table["ra"], scene_table["dec"])
        ]
        scene_table.write(
            f"{stem}_scene_catalog.csv", format="ascii.csv", overwrite=True
        )
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
            path: Output file.  Defaults to ``<out_dir>/<name>_stamps.fits``
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
            path = self.out_dir / f"{self.run_config.name}_stamps.fits"
        path = Path(path)

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
        vla = {"hi": [], "lo": []}
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
                    data = np.zeros((0, 0), dtype=np.float32)
                    x0 = y0 = -1
                    xs = ys = np.nan
                else:
                    data = np.asarray(t.data, dtype=np.float32)
                    # data[0, 0] sits at this original-grid pixel (may be
                    # negative for cutouts padded past the image edge)
                    x0, y0 = (int(v) for v in t._origin_original_true)
                    xs, ys = (float(v) for v in t.input_position_original)
                vla[tag].append(data.ravel())
                rows[f"ny_{tag}"].append(data.shape[0])
                rows[f"nx_{tag}"].append(data.shape[1])
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

        def vla_column(name: str, stamps: list[np.ndarray]) -> fits.Column:
            arr = np.empty(len(stamps), dtype=object)
            arr[:] = stamps
            return fits.Column(name=name, format="PE()", array=arr)

        int_fmt = {"id", "ny", "nx", "x0", "y0", "key", "flag"}
        columns = [vla_column(f"tmpl_{tag}", vla[tag]) for tag in ("hi", "lo")]
        columns += [
            fits.Column(
                name=name,
                format="K" if name.split("_")[0] in int_fmt else "D",
                array=np.asarray(values),
            )
            for name, values in rows.items()
        ]
        src_hdu = fits.BinTableHDU.from_columns(columns, name="SOURCES")

        hdr = self._stamps_header(ifilt, len(conv))
        fits.HDUList([fits.PrimaryHDU(header=hdr), src_hdu]).writeto(
            path, overwrite=True
        )
        npix = sum(a.size for a in vla["hi"]) + sum(a.size for a in vla["lo"])
        logger.info(
            "wrote %d sources (%.1f MB of template pixels) to %s",
            len(conv), npix * 4 / 1e6, path,
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
            print(f"upsampling image {ifilt} by factor {k}")
            self.images[ifilt], _ = _upsample_flux_conserving_image_and_ivar(
                self.images[ifilt], None, k
            )
            self.wcs[ifilt] = self.wcs[0]

        shape_hi = self.images[0].shape
        shape_lo = self.images[ifilt].shape
        wcs_hi = self.wcs[0] if self.wcs is not None else None
        wcs_lo = self.wcs[ifilt] if self.wcs is not None else None

        with fits.open(path) as hdul:
            hdr = hdul[0].header
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
            src = hdul["SOURCES"].data
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
        self.templates_extracted = deepcopy(tmpls)
        self.templates_extended = deepcopy(tmpls)
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
        ``<name>_stamps.fits``, and recreates the derived state (grid
        upsampling, model image) so the instance matches a completed
        :meth:`run`.  When the stamps file is missing it is regenerated
        through the same template path :meth:`run` uses — fluxes then come
        from the fit table — and written back to disk.

        Not restored: ``all_scenes`` (scenes are not persisted), and the
        pre-extension pixels of ``templates_extracted`` when loading from a
        stamps file (it then equals ``templates_extended``).  Regenerated
        stamps reproduce the fitted templates exactly only when the run
        applied no astrometric shifts.

        Args:
            ifilt: Fitted image index (1-based, as elsewhere).

        Returns:
            self, in the post-run state.
        """
        from astropy.io import fits

        if getattr(self, "run_config", None) is None:
            raise RuntimeError("load_fit requires a config-driven pipeline")
        cfg = self.run_config
        stem = self.out_dir / cfg.name
        f_table = Path(f"{stem}_fit_table.fits")
        f_residual = Path(f"{stem}_residual.fits")
        if not f_table.exists() or not f_residual.exists():
            raise FileNotFoundError(
                f"run outputs not found under {self.out_dir}; expected "
                f"{f_table.name} and {f_residual.name} — run() and "
                "write_outputs() first"
            )
        if self.images is None:
            self.load_data()
        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        config = self.config

        self.table = Table.read(f_table)
        residual = np.asarray(fits.getdata(f_residual), dtype=np.float32)

        self.fit_bin_factors = []
        self.all_scenes = []
        f_stamps = Path(f"{stem}_stamps.fits")
        if f_stamps.exists():
            self._templates_from_stamps(f_stamps, ifilt)
        else:
            logger.warning(
                "stamps file %s not found; regenerating templates through the "
                "run() template path", f_stamps.name,
            )
            cat = self._fit_catalog(config)
            self._prepare_hi_templates(cat, config)
            templates, weights_i = self._convolved_templates(ifilt, config)
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

        self.residuals = [residual]
        self.model_images = [self.images[ifilt] - residual]
        logger.info("post-run state restored from %s", self.out_dir)
        return self

    @contextmanager
    def log_run(self, path: str | Path | None = None):
        """Capture everything the run emits into ``<out_dir>/<name>.log``.

        Both ``logging`` records and bare ``print``/``tqdm`` output go to the
        file, since the package emits through both. The console is unchanged.
        Appends, so successive runs against one output directory accumulate
        rather than overwrite.
        """
        import platform
        import sys
        import time
        import warnings

        path = Path(path) if path is not None else self.out_dir / f"{self.run_config.name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
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
        file_handler = logging.StreamHandler(handle)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        old_root_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        old_showwarning = warnings.showwarning
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
        if not pkg.handlers:
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
            sys.stdout, sys.stderr = old_out, old_err
            handle.close()

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

        if not np.isfinite(throughput) or throughput <= 0.0:
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
            logger.info(
                "flux_%d_total divided by ee_psf_lo: median %.5f, range %.5f-%.5f "
                "over %d templates (%d fell back to the filter mean %.5f)",
                idx, float(np.median(arr)), float(arr.min()), float(arr.max()),
                arr.size, int(np.sum(arr == throughput)), throughput,
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
            cat[f"scene_{idx}"][ci] = scene_of_parent.get(pid, -1)

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

    def _aperture_sum_on_template(self, tmpl: Template, radius_pix: float) -> float:
        """Exact aperture sum on a template image centered on its own center."""
        x0 = tmpl.input_position_cutout[0]  # - tmpl.slices_cutout[1].start
        y0 = tmpl.input_position_cutout[1]  # - tmpl.slices_cutout[0].start
        aper = CircularAperture((float(x0), float(y0)), r=float(radius_pix))
        phot = aperture_photometry(tmpl.data, aper, method="exact")
        return float(phot["aperture_sum"][0])

    def _add_aperture_photometry(
        self,
        cat: Table,
        templates: list[Template],  # post-conv templates (current band)
        fluxes: np.ndarray,  # best-fit per-template fluxes
        residual: np.ndarray,  # residual image (same grid as ref if you upsampled)
        psf: np.ndarray | PSFRegionMap | None,
        idx: int,  # current image index (>=1)
    ) -> None:
        """
        Measure aperture flux on (model+residual) and PSF-correct it using
        the ratio of pre/post-convolution *template* aperture integrals:

            corr = F_cat(tmpl_ref_preconv) / F_img(tmpl_ref_postconv)

        Writes:
        ap_flux_{idx}       – raw aperture sum on model+residual
        ap_corr_{idx}       – correction factor
        ap_flux_corr_{idx}  – corrected flux
        """
        from photutils.aperture import CircularAperture, aperture_photometry

        cfg = self.config
        id_to_row = {int(i): k for k, i in enumerate(cat["id"])}

        # ensure columns exist
        for name in (f"ap_model_{idx}", f"ap_flux_{idx}", f"ap_corr_{idx}", f"ap_flux_corr_{idx}"):
            if name not in cat.colnames:
                cat[name] = cfg.bad_value

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

            x0 = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start
            y0 = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start
            aper_img = CircularAperture((float(x0), float(y0)), r=float(r_img_pix))
            phot = aperture_photometry(patch, aper_img, method="exact")
            ap_raw = float(phot["aperture_sum"][0])

            # --- correction from aperture flux to total flux via template EE -----------
            # numerator: total flux of the convolved template (= post-conv sum).
            # Using the post-conv total (rather than 1.0) accounts for any flux lost
            # at the template boundary during convolution.
            # denominator: flux of the convolved template within the photometric aperture.
            # corr = num/den = post_conv_total / post_conv_aperture = 1/EE_source_MIRI(r),
            # which converts ap_raw (partial aperture flux) to total flux.
            # Previously num used aperture_sum(pre_conv_template, r), which gave
            # EE_source_F444W(r) ~ 0.3 in the numerator and caused a ~1.2 mag offset.
            num = float(tmpl.data.sum())

            # denominator: current *convolved* template with *image* aperture
            den = self._aperture_sum_on_template(tmpl, r_img_pix)

            ap_model = fl * den  # aperture flux on model only (for info)

            # safe correction
            corr = num / den if (np.isfinite(num) and np.isfinite(den) and den > 0) else 1.0
            ap_corr = ap_raw * corr

            cat[f"ap_model_{idx}"][row] = ap_model
            cat[f"ap_flux_{idx}"][row] = ap_raw
            cat[f"ap_corr_{idx}"][row] = corr
            cat[f"ap_flux_corr_{idx}"][row] = ap_corr

    def _fit_catalog(self, config: _FitConfig) -> Table:
        """Output-catalog skeleton :meth:`run` fits into: id/x/y + provenance."""
        catalog = self.catalog
        if catalog is None:
            # use astropy to make catalog from image[0] + segmap
            print("No catalog provided, generating from segmap")
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
            self.tmpls = Templates()
            self.tmpls.extract_templates(
                images[0],
                segmap,
                list(zip(cat["x"], cat["y"])),
                wcs=wcs[0] if wcs is not None else None,
                dilate_segmap=config.template_dilate_segmap,
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
            sat_cols = [c for c in cat.colnames if c.startswith("FLAG_SATURATED_")]
            if sat_cols:
                sat_by_id: dict[int, bool] = {}
                for row in cat:
                    flagged = any(int(row[c]) != 0 for c in sat_cols)
                    sat_by_id[int(row["id"])] = flagged
                for tmpl in self.tmpls.templates:
                    tmpl.is_saturated = sat_by_id.get(int(tmpl.id), False)
            self.templates_extracted = deepcopy(self.tmpls)
            if self.extend_templates in {"psf", "psf_wings"}:
                psf_hi = psfs[0] if psfs is not None and psfs[0] is not None else None
                if psf_hi is None and psfs is not None and len(psfs) > 1:
                    psf_hi = psfs[1]
                if psf_hi is None:
                    raise ValueError(
                        f"extend_templates={self.extend_templates!r} requires a high-resolution PSF in psfs[0]"
                    )
                # Template extension is a shape operation. The extension code
                # normalizes finite PSF stamps to unit-sum shapes and keeps
                # native finite-support sums only as throughput metadata.
                logger.info(
                    "extending %d templates with PSF wings (%s, background_only=%s, "
                    "skip_deblended=%s)",
                    len(self.tmpls.templates),
                    getattr(psf_hi, "name", type(psf_hi).__name__),
                    bool(config.extend_wings_background_only),
                    bool(config.skip_template_extension_for_deblended),
                )
                self.tmpls.extend_with_psf_wings(
                    psf_hi,
                    skip_deblended=bool(config.skip_template_extension_for_deblended),
                    background_only=bool(config.extend_wings_background_only),
                    inplace=True,
                )
            elif self.extend_templates == "psf_model":
                psf_hi = psfs[0] if psfs is not None and psfs[0] is not None else None
                if psf_hi is None and psfs is not None and len(psfs) > 1:
                    psf_hi = psfs[1]
                if psf_hi is None:
                    raise ValueError(
                        "extend_templates='psf_model' requires a high-resolution PSF in psfs[0]"
                    )
                # Keep the same shape-vs-throughput convention as psf_wings.
                self.tmpls.extend_with_psf_model(
                    psf_hi,
                    mode="model",
                    skip_deblended=bool(config.skip_template_extension_for_deblended),
                    inplace=True,
                )
            elif self.extend_templates not in {None, "none"}:
                raise ValueError(f"Unknown template extension mode {self.extend_templates!r}")
            self.templates_extended = deepcopy(self.tmpls)
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
            self.templates_extracted = deepcopy(self.tmpls)
            self.templates_extended = deepcopy(self.tmpls)
        templates = self.tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        if catalog is not None and "flag_star" in catalog.colnames:
            star_ids = set(int(r["id"]) for r in catalog if r["flag_star"] == 1)
            for t in templates:
                if int(t.id) in star_ids:
                    t.is_star = True
            logger.info("Marked %d templates as stars (excluded from astrometry)", sum(t.is_star for t in templates))

        ndropped = len(cat) - len(templates)
        # @@@ this is because of reliance of x,y in catalog -> use segmap + weight?
        source = "prebuilt" if self.input_templates is not None else "extracted"
        print(f"Pipepline: {len(templates)} {source} templates, dropped {ndropped}.")
        print(f"Pipeline (templates) memory: {memory():.1f} GB")
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
            print(f"Using kernel lookup table {kernel.name}")

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
                print(f"upsampling image {ifilt} by factor {k}")
                images[ifilt], weights_i = _upsample_flux_conserving_image_and_ivar(
                    images[ifilt],
                    weights_i,
                    k,
                )
                wcs[ifilt] = wcs[0]
            else:
                print(f"Downsampling templates and kernels by factor {k}")
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
            tmpls_lo = deepcopy(self.tmpls)

        if weights_i is not None:
            tmpls_lo.prune_outside_weight(weights_i)

        templates = tmpls_lo.convolve_templates(
            kernel, inplace=False, psf_lo=getattr(self, "prm_lo", None)
        )
        if k > 1 and config.multi_resolution_method == "upsample":
            dummy_image = np.zeros(images[ifilt].shape, dtype=np.byte)
            templates = [
                t.project_to_block_replicated_grid(k, parent_image=dummy_image)
                for t in templates
            ]
        self.templates = templates
        print(f"Pipeline (convolved) memory: {memory():.1f} GB")

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
        from .fit import SparseFitter
        from . import utils
        import warnings

        # config-driven construction: load data + maps on first run()
        if getattr(self, "run_config", None) is not None:
            # record the executed config in out_dir so a finished run can be
            # reopened later with from_config(out_dir)
            self.run_config.to_json(self.out_dir / f"{self.run_config.name}.json")
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

        print(f"Pipeline (start) memory: {memory():.1f} GB")
        print(f"Pipeline config: {config}")

        # test for NaN values in images and weights
        for i in range(len(images)):
            if images[i] is None:
                assert np.all(np.isfinite(images[i])), "Image contains NaN values"
            if weights[i] is not None:
                assert np.all(np.isfinite(weights[i])), "Weights contain NaN values"

        cat = self._fit_catalog(config)

        templates = self._prepare_hi_templates(cat, config)

        residuals: list[np.ndarray] = []
        self.all_templates: list[Template] = []
        self.all_scenes: list[Scene] = []
        self.fit_bin_factors: list[int] = []
        self.model_images = []
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
                print(f"Wrote scene catalog scene_catalog_{ifilt}.ecsv")
                import sys

                sys.exit()

            for s in scenes:
                logger.info(f"Scene {s.id}: {len(s.templates)} (bright: {s.is_bright.sum()})")

            niter_scene = max(config.fit_astrometry_niter, 1)
            shift_tol = float(getattr(config, "astrom_shift_tol", 0.02))
            for j in range(niter_scene):
                logger.info(f"[Scenes] Running iteration {j+1} of {niter_scene}")
                max_step = 0.0
                for scn in scenes:
                    prev = np.array([t.shifted[:2] for t in scn.templates], dtype=float)
                    scn.set_band(images[ifilt], weights_i, config=config)
                    scn.solve(config=config, apply_shifts=True)
                    cur = np.array([t.shifted[:2] for t in scn.templates], dtype=float)
                    if prev.size and cur.shape == prev.shape:
                        max_step = max(max_step, float(np.max(np.abs(cur - prev))))
                logger.info(
                    f"[Scenes] iteration {j+1}: max shift increment {max_step:.4f} pix"
                )
                if config.fit_astrometry_niter > 0 and max_step < shift_tol:
                    logger.info(
                        f"[Scenes] shifts converged (< {shift_tol} pix) after {j+1} passes"
                    )
                    break

            # build model in res first, then subtract from image
            res = np.zeros_like(images[ifilt])
            for s in scenes:
                sl = _slices_from_bbox(s.bbox)
                res[sl] += s.model_image()  # adds models in place
            # then subtract from image to get residual
            res = images[ifilt] - res

            fluxes = [t.flux for t in templates]
            errs = [t.err for t in templates]
            err_pred = Templates.predicted_errors(templates, weights_i)
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
            self._update_catalog_with_fluxes(
                cat,
                templates,
                fluxes,
                errs,
                err_pred,
                throughput,
                ifilt,
                scene_ids=template_scene_ids,
            )
            self._add_aperture_photometry(
                cat,
                templates,
                fluxes,
                res,
                psfs[ifilt] if psfs is not None else None,
                ifilt,
            )

            self.residuals.append(res)
            self.model_images.append(images[ifilt] - res)
            #            self.fit.append(fitter)
            self.all_templates.append(templates)
            self.all_scenes.append(scenes)
        #            self.infos.append(info)

        print(f"Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")
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
        """Return the high-resolution PSF object used by template extension."""
        psfs = self.psfs if self.psfs is not None else []
        psf_hi = psfs[0] if len(psfs) > 0 and psfs[0] is not None else None
        if psf_hi is None and len(psfs) > 1:
            psf_hi = psfs[1]
        if psf_hi is None:
            raise ValueError(
                f"extend_templates={self.extend_templates!r} requires a high-resolution PSF"
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
        rebuilt = Templates()
        rebuilt.extract_templates(
            self.images[0],
            self.segmap,
            [pos],
            wcs=self.wcs[0] if self.wcs is not None else None,
            dilate_segmap=int(self.config.template_dilate_segmap),
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
                    tmpl.is_saturated = any(int(row[c]) != 0 for c in sat_cols)
        before = self._snapshot_template(tmpl)

        work = Templates()
        work.original_shape = rebuilt.original_shape
        work.wcs = getattr(rebuilt, "wcs", self.wcs[0] if self.wcs is not None else None)
        work.segmap = rebuilt.segmap
        work._templates = [tmpl]
        if self.extend_templates in {"psf", "psf_wings"}:
            work.extend_with_psf_wings(
                self._psf_for_template_extension(),
                skip_deblended=bool(self.config.skip_template_extension_for_deblended),
                background_only=bool(self.config.extend_wings_background_only),
                inplace=True,
            )
            tmpl_ext = work.templates[0]
        elif self.extend_templates == "psf_model":
            work.extend_with_psf_model(
                self._psf_for_template_extension(),
                mode="model",
                skip_deblended=bool(self.config.skip_template_extension_for_deblended),
                inplace=True,
            )
            tmpl_ext = work.templates[0]
        elif self.extend_templates in {None, "none"}:
            tmpl_ext = tmpl
        else:
            raise ValueError(f"Unknown template extension mode {self.extend_templates!r}")
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
        """
        import matplotlib.pyplot as plt

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        if self.templates_extracted is None or self.templates_extended is None:
            raise RuntimeError("run the pipeline before calling diagnose_sources")
        if len(self.all_templates) < ifilt:
            raise RuntimeError("run the pipeline before calling diagnose_sources")

        final_templates = self.all_templates[ifilt - 1]
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

    def plot_subphot(
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
        distance-sorted 5-level grayscale segmap colouring.

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

        if not getattr(self, "all_templates", None):
            raise RuntimeError("run the pipeline before calling plot_subphot")
        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("ifilt must be between 1 and len(images)-1")
        if self.images[ifilt].shape != self.images[0].shape:
            raise NotImplementedError(
                "plot_subphot requires the fit grid to match the reference grid"
            )

        templates = self.all_templates[ifilt - 1]
        own = [
            t
            for t in templates
            if int(t.id_parent if getattr(t, "parent_id", None) is not None else t.id)
            == int(source_id)
        ]
        if not own:
            raise KeyError(f"source id {source_id} not found in fitted templates")

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
        in_stamp = [
            t
            for t in templates
            if x1 <= t.position_original[0] < x1 + size
            and y1 <= t.position_original[1] < y1 + size
        ]
        order = sorted(
            in_stamp,
            key=lambda t: (t.position_original[0] - xc_full) ** 2
            + (t.position_original[1] - yc_full) ** 2,
        )
        lv = [0.2, 0.8, 0.4, 0.6, 1.0]
        tvseg = tseg.astype(float)
        for i, t in enumerate(order):
            tvseg[tseg == int(t.id)] = lv[i % 5]

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
        for scene in scene_list:
            for tmpl in scene.templates:
                try:
                    iseg = segm.get_index(tmpl.id)
                except (KeyError, ValueError):
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
        extend_templates=extend_templates,
        templates=templates,
        config=config,
    )
    table, residuals = pipeline.run()
    return table, residuals, pipeline


STEPS = {
    "psfs": "build_psfs",
    "kernels": "build_kernels",
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
    for step in steps:
        getattr(pipe, STEPS[step])()


if __name__ == "__main__":
    main()
