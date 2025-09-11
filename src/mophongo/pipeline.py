"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

import os
import psutil
from typing import Sequence
from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict

from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, block_replicate, block_reduce
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
        if hasattr(tmpl, "wcs") and tmpl.wcs is not None:
            ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
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
        kernels: Sequence[np.ndarray | PSFRegionMap] | None = None,
        wcs: Sequence[WCS] | None = None,
        window: Window | None = None,
        extend_templates: str | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if psfs is not None and len(images) != len(psfs):
            raise ValueError("Number of images and PSFs must match")
        if weights is None and wht_images is not None:
            weights = wht_images
        if weights is not None and len(weights) != len(images):
            raise ValueError("Number of weight images must match number of images")

        if config is None:
            config = _FitConfig()

        self.images = images
        self.segmap = segmap
        self.catalog = catalog
        self.psfs = psfs
        self.weights = weights
        self.kernels = kernels
        self.wcs = wcs
        self.window = window
        self.extend_templates = extend_templates
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
        self.tmpls: Templates()

        print(f"Pipeline (init) memory: {memory():.1f} GB")

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
            Flux measurements and their uncertainties.
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

        flux_sum: defaultdict[int, float] = defaultdict(float)
        err_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_sum: defaultdict[int, float] = defaultdict(float)
        for pid, fl, er, ep in zip(parent_ids, fluxes, errs, err_pred):
            if pid is None:
                continue
            flux_sum[pid] += fl
            err_sum[pid] = float(np.sqrt(err_sum[pid] ** 2 + er**2))
            err_pred_sum[pid] = float(np.sqrt(err_pred_sum[pid] ** 2 + ep**2))

        for pid, fl in flux_sum.items():
            ci = id_to_index.get(pid)
            if ci is None:
                continue
            cat[f"flux_{idx}"][ci] = fl
            cat[f"err_{idx}"][ci] = err_sum[pid]
            cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]

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
        r_cat_pix_by_id = self._resolve_catalog_ap_radius_pix(cat, cfg, r_default=r_img_pix)

        # map parent id -> original (pre-convolution) template on the ref image
        ref_tmpls = {int(t.id): t for t in self.tmpls.templates}

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

            # --- PSF correction from templates (pre vs post conv) -----------------
            # numerator: ref pre-convolution template with *catalog* aperture (per source)
            tmpl_ref = ref_tmpls.get(int(pid))
            r_cat_pix = r_cat_pix_by_id.get(int(pid), np.nan)
            num = (
                self._aperture_sum_on_template(tmpl_ref, r_cat_pix)
                if (tmpl_ref and np.isfinite(r_cat_pix))
                else np.nan
            )

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

        images = self.images
        segmap = self.segmap
        catalog = self.catalog
        psfs = self.psfs
        weights = self.weights
        kernels = self.kernels
        if kernels is None:
            kernels = [None] * len(images)
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
            cat = cat["id", "x", "y"]  # minimal set required
            if config.aperture_catalog is not None:
                cat[config.aperture_catalog] = catalog[config.aperture_catalog]

        self.tmpls = Templates()
        self.tmpls.extract_templates(
            images[0],
            segmap,
            list(zip(cat["x"], cat["y"])),
            wcs=wcs[0] if wcs is not None else None,
        )
        templates = self.tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        ndropped = len(cat) - len(templates)
        # @@@ this is because of reliance of x,y in catalog -> use segmap + weight?
        print(f"Pipepline: {len(templates)} extracted templates, dropped {ndropped}.")
        print(f"Pipeline (templates) memory: {memory():.1f} GB")

        astro = AstroCorrect(config)
        residuals: list[np.ndarray] = []
        self.all_templates: list[Template] = []
        self.all_scenes: list[Scene] = []
        for ifilt in range(1, len(images)):
            weights_i = weights[ifilt] if weights is not None else None

            kernel = kernels[ifilt]
            if kernel is None:
                kernel = np.array([[1.0]])  # @@@ this shouldnt be necessary?
            elif isinstance(kernel, PSFRegionMap):
                print(f"Using kernel lookup table {kernel.name}")

            k = bin_factor_from_wcs(wcs[0], wcs[ifilt]) if wcs is not None else 1

            if k > 1:
                if config.multi_resolution_method == "upsample":
                    print(f"upsampling image {ifilt} by factor {k}")
                    images[ifilt] = block_replicate(images[ifilt], k, conserve_sum=True).astype(
                        np.float32
                    )
                    if weights_i is not None:
                        weights_i = block_replicate(weights[ifilt], k).astype(np.float32) * k**2
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
                for j in range(niter_scene):
                    logger.info(f"[Scenes] Running iteration {j+1} of {niter_scene}")
                    for scn in scenes:
                        scn.set_band(images[ifilt], weights_i, config=config)
                        scn.solve(config=config, apply_shifts=True)

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
            self._update_catalog_with_fluxes(cat, templates, fluxes, errs, err_pred, ifilt)
            self._add_aperture_photometry(
                cat,
                templates,
                fluxes,
                res,
                psfs[ifilt] if psfs is not None else None,
                ifilt,
            )

            self.residuals.append(res)
            #            self.fit.append(fitter)
            self.all_templates.append(templates)
            self.all_scenes.append(scenes)
        #            self.infos.append(info)

        print(f"Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")
        self.table = cat

        return self.table, self.residuals  # , self.all_templates, self.all_scenes

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
    wcs: Sequence[WCS] | None = None,
    window: Window | None = None,
    extend_templates: str | None = None,
    config: FitConfig | None = None,
) -> tuple[Table, list[np.ndarray], SparseFitter]:
    """Backward compatible wrapper for :class:`Pipeline`"""

    pipeline = Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=weights,
        wht_images=wht_images,
        kernels=kernels,
        wcs=wcs,
        window=window,
        extend_templates=extend_templates,
        config=config,
    )
    return pipeline.run()

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
