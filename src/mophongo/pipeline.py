"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

import json
import os
import psutil
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
from .utils import bin_factor_from_wcs, downsample_psf, bin_remap
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
    # extra Gaussian broadening of the lo-res model PSF (FWHM arcsec);
    # "default" = mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC[filter_lo],
    # a number = that value, None = no broadening
    psf_blur_fwhm: float | str | None = "default"
    # optional [n_frames_hi, n_frames_lo] sanity assert on the WCS csvs
    expect_frames: list[int] | None = None
    # --- preprocessing ----------------------------------------------------
    bg_filter_sigma: float = 64.0  # get_bg_and_ivar background filter
    footprint_filter: bool = True  # keep only sources with wht_lo > 0
    r_trial: float = 0.0  # trial-patch radius in arcmin; 0 = full run
    trial_center: list[float] | None = None  # [ra, dec] deg of the patch
    # --- fitting ----------------------------------------------------------
    fit: dict[str, Any] = field(default_factory=dict)  # FitConfig kwargs
    scene_plots: bool = True  # write per-scene diagnostic PNGs

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


def _per_source_chi2(
    residual: np.ndarray, weights: np.ndarray, templates: Sequence[Template]
) -> np.ndarray:
    """Compute template-weighted chi² for each template.

    For each template, computes the sum of squared, template-weighted residuals
    divided by the noise variance, normalized by the sum of template weights.

    Returns
    -------
    ndarray
        Array of template-weighted chi² values, one per template in ``templates``.
    """
    chi2 = np.zeros(len(templates), dtype=float)
    for i, tmpl in enumerate(templates):
        res = residual[tmpl.slices_original]
        tmpl_data = tmpl.data[tmpl.slices_cutout]
        ivar = weights[tmpl.slices_original]  # inverse variance
        mask = ivar > 0
        # Template-weighted chi²: sum((res * tmpl)^2 / var) / sum(tmpl^2)
        num = np.sum(mask * (res * tmpl_data) ** 2 * ivar)
        denom = np.sum(mask * tmpl_data**2)
        chi2[i] = num / denom if denom > 0 else 0.0
    return chi2


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

        Data are loaded lazily: :meth:`run` (or :meth:`load_data`) reads the
        images and finishes construction.
        """
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
            self.dpsf_hi.epsf_obj.load_jwst_stdpsf(
                local_dir=str(cfg.psf_dir), filter_pattern=cfg.pattern_hi
            )
            self.dpsf_lo.epsf_obj.load_jwst_stdpsf(
                local_dir=str(cfg.psf_dir), filter_pattern=cfg.pattern_lo
            )
            self._epsf_loaded = True

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
        if self.f_psf_hi.exists() and self.f_psf_lo.exists() and not overwrite:
            self.prm_hi = PSFRegionMap.from_geojson(str(self.f_psf_hi))
            self.prm_lo = PSFRegionMap.from_geojson(str(self.f_psf_lo))
            logger.info("loaded cached PSF maps from %s", self.out_dir)
            return self

        self._ensure_dpsfs(load_epsf=True)
        prm_hi, prm_lo = self._region_maps()
        prm_hi.psfs = self.dpsf_hi.get_psf_radec(
            self._centroids(prm_hi), **self._size_kw()
        )
        prm_lo.psfs = self._drizzle_lo_blurred(self._centroids(prm_lo))
        blur = self._blur_fwhm()
        if blur:
            logger.info('applied %.3f" FWHM Gaussian broadening to lo-res PSFs', blur)
        prm_hi.to_file(self.f_psf_hi)
        prm_lo.to_file(self.f_psf_lo)
        self.prm_hi, self.prm_lo = prm_hi, prm_lo
        return self

    # -- step 2: matching-kernel map --------------------------------------
    def build_kernels(self, overwrite: bool = False) -> "Pipeline":
        """Build (or reload) the matching-kernel map on the hi/lo overlay."""
        from . import utils

        if self.f_kernel.exists() and not overwrite:
            self.prm_kern = PSFRegionMap.from_geojson(str(self.f_kernel))
            logger.info("loaded cached kernel map %s", self.f_kernel)
            return self

        self._ensure_dpsfs(load_epsf=True)
        prm_hi_geom, prm_lo_geom = self._region_maps()
        prm_kern = prm_hi_geom.overlay_with(prm_lo_geom)
        pos = self._centroids(prm_kern)

        psf_hi = self.dpsf_hi.get_psf_radec(pos, **self._size_kw())
        psf_lo = self._drizzle_lo_blurred(pos)

        pixel_ratio = round(self.dpsf_lo.driz_pscale / self.dpsf_hi.driz_pscale)
        kernels = [
            utils.matching_kernel(p_hi, p_lo, pixel_ratio=pixel_ratio)
            for p_hi, p_lo in zip(psf_hi, psf_lo)
        ]
        prm_kern.psfs = np.asarray(kernels)
        prm_kern.to_file(self.f_kernel)
        self.prm_kern = prm_kern
        return self

    def _ensure_maps(self) -> None:
        """Load the cached lo-res PSF map and build the kernel map if missing."""
        if self.prm_lo is None and self.f_psf_lo.exists():
            self.prm_lo = PSFRegionMap.from_geojson(str(self.f_psf_lo))
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
        segmap = fits.getdata(cfg.segmap)
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

        # finish construction: regular __init__ on the loaded products
        Pipeline.__init__(
            self,
            [tmpl_hi, sci_fit],
            segmap,
            weights=[None, ivar],
            catalog=cat,
            psfs=[None, self.prm_lo],
            kernels=[None, self.prm_kern],
            wcs=[wcs_hi, wcs_lo],
            config=_FitConfig(**cfg.fit),
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

    def run_all(self) -> "Pipeline":
        """All steps in order: psfs, kernels, fit, outputs."""
        self.build_psfs()
        self.build_kernels()
        self.run()
        self.write_outputs()
        return self

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _add_templates_for_bad_fits(
        self,
        templates: list[Template],
        tmpls_lo: Templates,
        psf: np.ndarray | PSFRegionMap | None,
        weights: np.ndarray | None,
        fitter: "SparseFitter",
        image: np.ndarray,
        fitter_cls,
        config: _FitConfig,
    ) -> tuple[list[Template], "SparseFitter"]:
        """Add secondary templates for poorly fitted sources.

        Parameters
        ----------
        templates
            Current list of templates used in the fit.
        tmpls_lo
            Base templates prior to convolution. Used when adding new
            components.
        psf
            PSF image for the current low-resolution frame.
        weights
            Weight map corresponding to ``image``.
        fitter
            Fitter instance from the initial solve.
        image
            Image data being modelled.
        fitter_cls
            Fitter class used to instantiate a new fitter if additional
            templates are required.
        config
            Fit configuration options.

        Returns
        -------
        list[Template], SparseFitter
            Possibly extended template list and a corresponding fitter
            instance.
        """

        if not (
            (config.multi_tmpl_psf_core or config.multi_tmpl_colour)
            and psf is not None
            and weights is not None
        ):
            fitter._ata = None
            return templates, fitter

        res = fitter.residual()
        chi_nu = _per_source_chi2(res, weights, templates)
        bad_idx = np.where(chi_nu > config.multi_tmpl_chi2_thresh)[0]
        if bad_idx.size > 0:
            logger.info("Adding %d new templates for poor fits", bad_idx.size)
            for bi in bad_idx:
                parent = templates[bi]
                if config.multi_tmpl_psf_core:
                    stamp = _extract_psf_at(parent, psf)
                    add_tmpl = tmpls_lo.add_component(parent, stamp, "psf")
                    templates.append(add_tmpl)
            fitter = fitter_cls(templates, image, weights, config)
        else:
            fitter._ata = None
        return templates, fitter

    def _update_catalog_with_fluxes(
        self,
        cat: Table,
        templates: list[Template],
        fluxes: np.ndarray,
        errs: np.ndarray,
        err_pred: np.ndarray,
        throughput: float,
        idx: int,
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
            Filter-level finite-support PSF sum.  Total-flux columns divide
            model fluxes/errors by this value.  A value of 1 applies no
            missing-PSF-support correction.
        idx
            Index of the current image (used for column naming).
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

        if not np.isfinite(throughput) or throughput <= 0.0:
            throughput = 1.0
        throughput = float(throughput)

        flux_sum: defaultdict[int, float] = defaultdict(float)
        err_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_sum: defaultdict[int, float] = defaultdict(float)
        flux_total_sum: defaultdict[int, float] = defaultdict(float)
        err_total_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_total_sum: defaultdict[int, float] = defaultdict(float)
        for pid, fl, er, ep in zip(parent_ids, fluxes, errs, err_pred):
            if pid is None:
                continue
            flux_sum[pid] += fl
            err_sum[pid] = float(np.sqrt(err_sum[pid] ** 2 + er**2))
            err_pred_sum[pid] = float(np.sqrt(err_pred_sum[pid] ** 2 + ep**2))
            flux_total_sum[pid] += fl / throughput
            err_total_sum[pid] = float(
                np.sqrt(err_total_sum[pid] ** 2 + (er / throughput) ** 2)
            )
            err_pred_total_sum[pid] = float(
                np.sqrt(err_pred_total_sum[pid] ** 2 + (ep / throughput) ** 2)
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
        ap_flux_raw_{idx}  – raw aperture sum on model+residual
        ap_corr_{idx}      – correction factor
        ap_flux_{idx}      – corrected flux
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

    def run(self, config: FitConfig | None = None) -> tuple[Table, list[np.ndarray]]:
        """Run photometry on the configured images.

        Returns
        -------
        Table
            Catalog containing flux measurements for each image.
        list of ndarray
            Residual images corresponding to each fitted image.
        SparseFitter
            The fitter instance used for the final fit.
        """
        from .fit import SparseFitter
        from .astro_fit import GlobalAstroFitter
        from .astrometry import AstroCorrect
        from . import utils
        import warnings

        # config-driven construction: load data + maps on first run()
        if getattr(self, "run_config", None) is not None:
            if self.images is None:
                self.load_data()
            elif self.kernels[-1] is None:
                # data pre-loaded with load_data(kernels=False): finish the maps
                self._ensure_maps()
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

        if catalog is None:
            # use astropy to make catalog from image[0] + segmap
            print("No catalog provided, generating from segmap")
            raise NotImplementedError("Catalog generation not implemented yet")
        else:
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

        astro = AstroCorrect(config)
        residuals: list[np.ndarray] = []
        self.all_templates: list[Template] = []
        self.all_scenes: list[Scene] = []
        self.fit_bin_factors: list[int] = []
        self.model_images = []
        for ifilt in range(1, len(images)):
            weights_i = weights[ifilt] if weights is not None else None
            scenes = []

            kernel = kernels[ifilt]
            if isinstance(kernel, PSFRegionMap):
                print(f"Using kernel lookup table {kernel.name}")

            k = bin_factor_from_wcs(wcs[0], wcs[ifilt]) if wcs is not None else 1
            self.fit_bin_factors.append(int(k))

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

            templates = tmpls_lo.convolve_templates(kernel, inplace=False)
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

            # @@@ split scenes here
            # Optional scene-based solver: does not alter legacy path
            if getattr(config, "run_scene_solver", False):
                # Work on a copy of templates to avoid affecting legacy loop
                templates_scene = templates
                scenes, labels = generate_scenes(
                    templates_scene,
                    images[ifilt],
                    weights_i,
                    coupling_thresh=float(config.scene_coupling_thresh),
                    max_size=config.scene_max_size,
                    snr_thresh_astrom=float(config.snr_thresh_astrom),
                    minimum_bright=int(config.scene_minimum_bright),
                    max_merge_radius=float(getattr(config, "scene_max_merge_radius", np.inf)),
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

            else:
                print("Running legacy solver")
                # fitter_cls = (
                #     GlobalAstroFitter
                #     if (config.fit_astrometry_niter > 0 and config.fit_astrometry_joint)
                #     else SparseFitter
                # )
                fitter_cls = SparseFitter
                niter = max(config.fit_astrometry_niter, 1)
                for j in range(niter):
                    print(f"Running iteration {j+1} of {niter}")

                    fitter = fitter_cls(templates, images[ifilt], weights_i, config)
                    fluxes, errs, info = fitter.solve()
                    print(f"Pipeline (residual) memory: {memory():.1f} GB")

                    # if config.fit_astrometry_niter > 0 and not config.fit_astrometry_joint:
                    #     # @@@ this is very expensive. We dont need to form the whole residual image
                    #     # can do it on the stamps only
                    #     res = fitter.residual()
                    #     logger.info("fitting astrometry separately")
                    #     astro.fit(templates, res, fitter.solution)

                    if config.fit_astrometry_niter > 0 and config.fit_astrometry_joint:
                        Templates.apply_template_shifts(templates)

                res = fitter.residual()

                #            print("END of TEMPLATES FITTING")

                # one final flux only solve after astrometry
                # cfg_noshift = _FitConfig(**config.__dict__)
                # cfg_noshift.fit_astrometry_niter = 0
                # templates, fitter = self._add_templates_for_bad_fits(
                #     templates,
                #     tmpls_lo,
                #     psfs[ifilt] if psfs is not None else None,
                #     weights_i,
                #     fitter,
                #     images[ifilt],
                #     fitter_cls,
                #     config,
                # )

                # add soft non-negative priors if fluxes are < 0.0 and resolve.
                # note idx is relative to initial list of templates. But additional templates were added at the end, so idx still works

                # snr = np.divide(fluxes, errs, out=np.zeros_like(errs), where=errs > 0)
                # selneg = snr < config.negative_snr_thresh
                # if np.any(selneg):
                #     logger.info(
                #         f"{selneg.sum()} fluxes are negative, applying soft non-negative prior and resolving."
                #     )
                #     # this updates ata and atb, so we can resolve again
                #     scale = np.clip(-snr, 1.0, 5.0)  # more negative → tighter prior
                #     fitter.add_flux_priors(selneg, mu=0.0, sigma=(errs / scale))

                #            fluxes, errs, info = fitter.solve(config=cfg_noshift)

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
                self._pixel_scale_arcsec(
                    self.wcs[ifilt] if self.wcs is not None else None
                ),
                ifilt,
                throughput,
            )

            # calculate a full image residual from the scenes and their slice
            #            res_scene
            # if getattr(config, "run_scene_solver", False):
            #     # sanity check
            #     diff = np.abs(res - res_scene)
            #     maxdiff = np.nanmax(diff)
            #     if maxdiff > 1e-5 * np.nanmax(np.abs(res)):
            #         warnings.warn(f"Scene residual differs from full residual: max diff {maxdiff}")
            #     else:
            #         print(f"Scene residual matches full residual: max diff {maxdiff}")
            # #                res = res_scene
            # print("Done...")

            if config.aperture_diam is not None:
                pscale = self._pixel_scale_arcsec(
                    self.wcs[ifilt] if self.wcs is not None else None
                )
                r_img_pix = self._resolve_image_ap_radius_pix(ifilt, config)
                r_img_arcsec = r_img_pix * pscale
                cat["aper_" + str(ifilt)] = 2 * r_img_arcsec
            self._update_catalog_with_fluxes(
                cat,
                templates,
                fluxes,
                errs,
                err_pred,
                throughput,
                ifilt,
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
    def _stamp_slices_for_templates(
        templates: Sequence[Template],
        image_shape: tuple[int, int],
    ) -> tuple[slice, slice]:
        """Return parent-image slices covering all template footprints."""
        x0s: list[int] = []
        x1s: list[int] = []
        y0s: list[int] = []
        y1s: list[int] = []
        for tmpl in templates:
            tx0, ty0 = map(int, tmpl._origin_original_true)
            tx1 = tx0 + tmpl.data.shape[1]
            ty1 = ty0 + tmpl.data.shape[0]
            x0s.append(tx0)
            x1s.append(tx1)
            y0s.append(ty0)
            y1s.append(ty1)

        x0 = max(0, min(x0s))
        x1 = min(image_shape[1], max(x1s))
        y0 = max(0, min(y0s))
        y1 = min(image_shape[0], max(y1s))
        if x1 <= x0 or y1 <= y0:
            return templates[0].slices_original
        return slice(y0, y1), slice(x0, x1)

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

        import math

        import matplotlib.pyplot as plt
        import numpy as np
        from copy import deepcopy
        from astropy.visualization import make_lupton_rgb
        from photutils.segmentation import SegmentationImage
        from astropy.table import Table

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("idx must be between 1 and len(images)-1")

        nscenes = len(np.unique(self.fit[ifilt - 1].scene_ids))

        segmap = self.segmap
        segm = SegmentationImage(segmap)
        segmap_cmap = segm.cmap
        scene_cmap = deepcopy(segmap_cmap)
        scene_cmap.colors[0] = (1.0, 1.0, 1.0, 0.0)

        fitter = self.fit[ifilt - 1]

        if not hasattr(self, "scenes"):
            logger.info("Building scene map for diagnostics")
            scenes = np.zeros_like(segmap, dtype=int)
            # fitter.scene_ids
            for tmpl in fitter.templates:
                iseg = segm.get_index(tmpl.id)
                sl = segm.segments[iseg].slices
                scenes_slice = scenes[sl]
                scenes_slice[segm.data[sl] == tmpl.id] = tmpl.id_scene

        logger.info(f"Plotting image {ifilt} with {nscenes} scenes")

        mask: np.ndarray | None = None
        if scene_id is not None:
            mask = scenes == scene_id
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
        kbin = bin_factor_from_wcs(self.wcs[0], self.wcs[ifilt])
        y0_lo, y1_lo, x0_lo, x1_lo = np.round(bin_remap([y0, y1, x0, x1], kbin)).astype(int)
        sl_lo = (slice(y0_lo, y1_lo), slice(x0_lo, x1_lo))

        img_hi = self.images[0]
        img_lo = self.images[ifilt]

        img_cut = img_lo[sl_lo]
        model_cut = fitter.model_image()[sl_lo]

        tmpl_cut = img_hi[sl_hi]
        seg_cut = segmap[sl_hi]
        scenes_cut = scenes[sl_hi]
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
                    for tmpl in fitter.templates:
                        if tmpl.id_scene == scene_id:
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

    # # EXTREMELY SLOW
    # # block into tiles for faster access
    # store = zarr.storage.MemoryStore()
    # group = zarr.group(store=store)  # container
    # fast = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)  # fastest
    # tight = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)  # better ratio, still fast
    # # You can control threads with Blosc(nthreads=<N>) if desired.
    # for i in range(len(images)):
    #     if images[i] is not None:
    #         img = group.create_array(
    #             f"images/{i}",
    #             shape=(images[i].shape),
    #             chunks=(512, 512),
    #             dtype="float32",
    #             compressors=None,  # <- critical
    #             filters=None,  # <- critical
    #             overwrite=True,
    #             fill_value=0.0,
    #         )
    #         img[:] = images[i]
    #         images[i] = img

    #     if weights[i] is not None:
    #         wht = group.create_array(
    #             f"weights/{i}",
    #             shape=(weights[i].shape),
    #             chunks=(512, 512),
    #             dtype="float32",
    #             compressors=None,  # <- critical
    #             filters=None,  # <- critical
    #             overwrite=True,
    #             fill_value=0.0,
    #         )
    #         wht[:] = weights[i]
    #         weights[i] = wht

    # # print(f"Pipeline (blocked storage) memory: {memory():.1f} GB")


STEPS = {
    "psfs": "build_psfs",
    "kernels": "build_kernels",
    "load": "load_data",
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
    ap.add_argument(
        "steps",
        nargs="*",
        choices=list(STEPS),
        help="steps to run (default: all)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipe = Pipeline.from_config(args.config)
    for step in args.steps or ["all"]:
        getattr(pipe, STEPS[step])()


if __name__ == "__main__":
    main()
