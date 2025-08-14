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

from .psf_map import PSFRegionMap
from .utils import bin_factor_from_wcs, downsample_psf
from .templates import Templates, Template
from .fit import FitConfig as _FitConfig

logger = logging.getLogger(__name__)


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
        wht_images: Sequence[np.ndarray] | None = None,
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
        self.wht_images = wht_images
        self.kernels = kernels
        self.wcs = wcs
        self.window = window
        self.extend_templates = extend_templates
        self.config = config

        self.residuals: list[np.ndarray] = []
        self.fit: list[np.ndarray] = []
        self.astro: list[np.ndarray] = []
        self.templates: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.tmpls: Templates()

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
            logger.info(
                "Adding %d new templates for poor fits", bad_idx.size
            )
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
            err_pred_sum[pid] = float(
                np.sqrt(err_pred_sum[pid] ** 2 + ep**2)
            )

        for pid, fl in flux_sum.items():
            ci = id_to_index.get(pid)
            if ci is None:
                continue
            cat[f"flux_{idx}"][ci] = fl
            cat[f"err_{idx}"][ci] = err_sum[pid]
            cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]

    def _add_aperture_photometry(
        self,
        cat: Table,
        templates: list[Template],
        fluxes: np.ndarray,
        residual: np.ndarray,
        psf: np.ndarray | PSFRegionMap | None,
        idx: int,
    ) -> None:
        """Add aperture photometry measurements to the catalog.

        The aperture flux is measured on the best-fit model for each source
        with the residuals added back in. A simple PSF correction factor is
        derived from the ratio of fluxes measured on the pre- and
        post-convolution PSF images.
        """

        from photutils.aperture import CircularAperture, aperture_photometry
        from .utils import measure_shape

        id_to_index = {id_: i for i, id_ in enumerate(cat["id"])}
        cat[f"ap_flux_{idx}"] = self.config.bad_value
        cat[f"ap_corr_{idx}"] = self.config.bad_value

        for tmpl, fl in zip(templates, fluxes):
            pid = tmpl.id_parent if getattr(tmpl, "parent_id", None) is not None else tmpl.id
            ci = id_to_index.get(pid)
            if ci is None:
                continue

            res_patch = residual[tmpl.slices_original]
            model_patch = fl * tmpl.data[tmpl.slices_cutout]
            patch = res_patch + model_patch

            try:
                psf_stamp = _extract_psf_at(tmpl, psf) if psf is not None else tmpl.data
                _, _, sx, sy, _ = measure_shape(psf_stamp, psf_stamp > 0)
                fwhm = 2.355 * float(np.mean([sx, sy]))
            except Exception:
                fwhm = 1.0
            radius = 1.5 * fwhm

            x0 = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start
            y0 = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start
            aper = CircularAperture((x0, y0), r=radius)
            phot = aperture_photometry(patch, aper, method="exact")
            ap_flux = float(phot["aperture_sum"][0])

            corr = 1.0
            if self.psfs is not None and self.psfs[0] is not None and psf is not None:
                psf_hi = _extract_psf_at(tmpl, self.psfs[0])
                psf_lo = _extract_psf_at(tmpl, psf)
                aper_psf_hi = aperture_photometry(
                    psf_hi,
                    CircularAperture(
                        (psf_hi.shape[1] / 2, psf_hi.shape[0] / 2), r=radius
                    ),
                    method="exact",
                )["aperture_sum"][0]
                aper_psf_lo = aperture_photometry(
                    psf_lo,
                    CircularAperture(
                        (psf_lo.shape[1] / 2, psf_lo.shape[0] / 2), r=radius
                    ),
                    method="exact",
                )["aperture_sum"][0]
                if aper_psf_lo != 0:
                    corr = float(aper_psf_hi / aper_psf_lo)

            cat[f"ap_flux_{idx}"][ci] = ap_flux
            cat[f"ap_corr_{idx}"][ci] = corr

    def run(self) -> tuple[Table, list[np.ndarray], SparseFitter]:
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
        wcs = self.wcs
        config = self.config

        memory = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9

        print(f"Pipeline (start) memory: {memory():.1f} GB")
        print(f"Pipeline config: {config}")

        cat = catalog.copy() if catalog is not None else None
        positions = list(zip(catalog["x"], catalog["y"])) if catalog is not None else []

        self.tmpls = Templates()
        self.tmpls.extract_templates(
            np.nan_to_num(images[0], copy=False, nan=0.0, posinf=0.0, neginf=0.0),
            segmap,
            positions,
            wcs=wcs[0] if wcs is not None else None,
        )
        templates = self.tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        ndropped = len(positions) - len(templates)
        # @@@ this is because of reliance of x,y in catalog -> use segmap?
        print(f"Pipepline: {len(templates)} extracted templates, dropped {ndropped}.")
        print(f"Pipeline (templates) memory: {memory():.1f} GB")

        astro = AstroCorrect(config)
        residuals: list[np.ndarray] = []
        for idx in range(1, len(images)):
            weights_i = weights[idx] if weights is not None else None

            kernel = None
            if kernels is not None:
                kernel = kernels[idx]
                if kernel is None:
                    kernel = np.array([[1.0]])
                elif isinstance(kernel, PSFRegionMap):
                    print(f"Using kernel lookup table {kernel.name}")

            if wcs is not None:
                k = bin_factor_from_wcs(wcs[0], wcs[idx])
            else:
                k = 1

            if k > 1:
                if config.multi_resolution_method == "upsample":
                    print(f"upsampling image {idx} by factor {k}")
                    images[idx] = block_replicate(images[idx], k, conserve_sum=True).astype(
                        np.float32
                    )
                    if weights_i is not None:
                        weights_i = block_replicate(weights[idx], k).astype(np.float32) * k**2
                    if wcs is not None:
                        wcs[idx] = wcs[0]
                else:
                    print(f"Downsampling templates and kernels by factor {k}")
                    tmpls_lo = Templates()
                    tmpls_lo.original_shape = images[idx].shape
                    tmpls_lo.wcs = wcs[idx]
                    tmpls_lo._templates = [
                        t.downsample(k, wcs_lo=wcs[idx]) for t in self.tmpls._templates
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

            assert np.all(np.isfinite(images[idx])), "Image contains NaN values"
            if weights_i is not None:
                assert np.all(np.isfinite(weights_i)), "Weights contain NaN values"
            for t in templates:
                assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

            fitter_cls = (
                GlobalAstroFitter
                if (config.fit_astrometry_niter > 0 and config.fit_astrometry_joint)
                else SparseFitter
            )
            fitter_cls = SparseFitter

            niter = max(config.fit_astrometry_niter, 1)
            for j in range(niter):
                print(f"Running iteration {j+1} of {niter}")

                fitter = fitter_cls(templates, images[idx], weights_i, config)
                fluxes, errs, info = fitter.solve()
                print(f"Pipeline (residual) memory: {memory():.1f} GB")

                if config.fit_astrometry_niter > 0 and not config.fit_astrometry_joint:
                    # @@@ this is very expensive. We dont need to form the whole residual image
                    # can do it on the stamps only
                    res = fitter.residual()
                    logger.info("fitting astrometry separately")
                    astro.fit(templates, res, fitter.solution)

                if config.fit_astrometry_niter > 0 and config.fit_astrometry_joint:
                    Templates.apply_template_shifts(templates)

            # one final flux only solve after astrometry
            cfg2 = _FitConfig(**config.__dict__)
            cfg2.fit_astrometry_niter = 0
            templates, fitter = self._add_templates_for_bad_fits(
                templates,
                tmpls_lo,
                psfs[idx] if psfs is not None else None,
                weights_i,
                fitter,
                images[idx],
                fitter_cls,
                config,
            )

            fluxes, errs, info = fitter.solve(config=cfg2)
            res = fitter.residual()
            fluxes, errs, info = fitter.solve()
            err_pred = fitter.predicted_errors()
            res = fitter.residual()

            print("Done...")

            self._update_catalog_with_fluxes(cat, templates, fluxes, errs, err_pred, idx)
            self._add_aperture_photometry(
                cat,
                templates,
                fluxes,
                res,
                psfs[idx] if psfs is not None else None,
                idx,
            )

            if "astro" in locals():
                self.astro.append(astro)
            self.residuals.append(res)
            self.fit.append(fitter)
            self.templates.append(templates)
            self.infos.append(info)

        print(f"Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")

        self.table = cat

        return self.table, self.residuals, self.fit

    def plot_result(
        self,
        idx: int = 1,
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
            idx: Index of the low-resolution image to display. Defaults to ``1``.
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

        if idx <= 0 or idx >= len(self.images):
            raise ValueError("idx must be between 1 and len(images)-1")

        nscenes = len(np.unique(self.fit[idx - 1].scene_ids))

        segmap = self.segmap
        segm = SegmentationImage(segmap)
        segmap_cmap = segm.cmap
        scene_cmap = deepcopy(segmap_cmap)
        scene_cmap.colors[0] = (1.0, 1.0, 1.0, 0.0)

        fitter = self.fit[idx - 1]

        if not hasattr(self, "scenes"):
            logger.info("Building scene map for diagnostics")
            scenes = np.zeros_like(segmap, dtype=int)
            # fitter.scene_ids
            for tmpl in fitter.templates:
                iseg = segm.get_index(tmpl.id)
                sl = segm.segments[iseg].slices
                scenes_slice = scenes[sl]
                scenes_slice[segm.data[sl] == tmpl.id] = tmpl.id_scene

        logger.info(f"Plotting image {idx} with {nscenes} scenes")

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
        kbin = bin_factor_from_wcs(self.wcs[0], self.wcs[idx])
        y0_lo, y1_lo, x0_lo, x1_lo = np.round(Template.bin_remap([y0, y1, x0, x1], kbin)).astype(
            int
        )
        sl_lo = (slice(y0_lo, y1_lo), slice(x0_lo, x1_lo))

        img_hi = self.images[0]
        img_lo = self.images[idx]

        img_cut = img_lo[sl_lo]
        model_cut = fitter.model_image()[sl_lo]

        tmpl_cut = img_hi[sl_hi]
        seg_cut = segmap[sl_hi]
        scenes_cut = scenes[sl_hi]
        # @@@ for now assume upsampled residual image
        res_cut = self.residuals[idx - 1][sl_hi]

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
            f"image{idx}",
            f"model image{idx}",
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
