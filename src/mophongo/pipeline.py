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
from astropy.nddata import Cutout2D

from .psf_map import PSFRegionMap
from .utils import bin_factor_from_wcs, downsample_psf

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
            from .fit import FitConfig as _FitConfig

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

        images = self.images
        segmap = self.segmap
        catalog = self.catalog
        psfs = self.psfs
        weights = self.weights
        kernels = self.kernels
        wcs = self.wcs
        config = self.config

        from .psf import PSF
        from .templates import Templates
        from .fit import SparseFitter
        from .astro_fit import GlobalAstroFitter
        from astropy.nddata import block_replicate, block_reduce
        from .astrometry import AstroCorrect
        from . import utils
        import warnings

        memory = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9

        print(f"Pipeline (start) memory: {memory():.1f} GB")
        print(f"Pipeline config: {config}")

        cat = catalog.copy() if catalog is not None else None
        positions = list(zip(catalog["x"], catalog["y"])) if catalog is not None else []

        tmpls = Templates()
        tmpls.extract_templates(
            np.nan_to_num(images[0], copy=False, nan=0.0, posinf=0.0, neginf=0.0),
            segmap,
            positions,
            wcs=wcs[0] if wcs is not None else None,
        )
        templates = tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        ndropped = len(positions) - len(templates)
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
                    images[idx] = block_replicate(images[idx], k, conserve_sum=True)
                    if weights_i is not None:
                        weights_i = block_replicate(weights[idx], k) * k**2
                    if wcs is not None:
                        wcs[idx] = wcs[0]
                else:
                    print(f"Downsampling templates and kernels by factor {k}")
                    tmpls_lo = Templates()
                    tmpls_lo.original_shape = images[idx].shape
                    tmpls_lo.wcs = wcs[idx]
                    tmpls_lo._templates = [
                        t.downsample(k, wcs_lo=wcs[idx]) for t in tmpls._templates
                    ]

                    if isinstance(kernel, PSFRegionMap):
                        kernel.psfs = np.array([downsample_psf(psf, k) for psf in kernel.psfs])
                    else:
                        kernel = downsample_psf(kernel, k)

            if k == 1 or config.multi_resolution_method == "upsample":
                tmpls_lo = deepcopy(tmpls)

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

            niter = max(config.fit_astrometry_niter, 1)
            for j in range(niter):
                print(f"Running iteration {j+1} of {niter}")

                fitter = fitter_cls(templates, images[idx], weights_i, config)
                fluxes, errs, info = fitter.solve()

                res = fitter.residual()
                print(f"Pipeline (residual) memory: {memory():.1f} GB")

                if config.fit_astrometry_niter > 0 and not config.fit_astrometry_joint:
                    logger.info("fitting astrometry separately")
                    astro.fit(templates, res, fitter.solution)

            fitter._ata = None
            fluxes, errs, info = fitter.solve()
            res = fitter.residual()

            if (
                (config.multi_tmpl_psf_core or config.multi_tmpl_colour)
                and psfs is not None
                and weights_i is not None
            ):
                chi_nu = _per_source_chi2(res, weights_i, templates)
                bad_idx = np.where(chi_nu > config.multi_tmpl_chi2_thresh)[0]
                print("Distribution >99% chi2:", np.percentile(chi_nu, [99]))
                if bad_idx.size > 0:
                    print(
                        f"Adding {len(bad_idx)} new templates for poor fits "
                        f"(chi^2/nu > {config.multi_tmpl_chi2_thresh})"
                    )
                    for bi in bad_idx:
                        parent = templates[bi]
                        if config.multi_tmpl_psf_core and psfs is not None:
                            stamp = _extract_psf_at(parent, psfs[idx])
                            add_tmpl = tmpls.add_component(parent, stamp, "psf")
                            templates.append(add_tmpl)

                    fitter = fitter_cls(templates, images[idx], weights_i, config)
                    if config.solve_method == "ata":
                        fluxes, errs, info = fitter.solve()
                    else:
                        fluxes, errs, info = fitter.solve_linear_operator()
                    res = fitter.residual()

            err_pred = fitter.predicted_errors()

            print("Done...")

            parent_ids = [
                tmpl.id_parent if getattr(tmpl, "parent_id", None) is not None else tmpl.id
                for tmpl in templates
            ]
            id_to_index = {id_: i for i, id_ in enumerate(cat["id"])}
            cat[f"flux_{idx}"] = config.bad_value
            cat[f"err_{idx}"] = config.bad_value
            cat[f"err_pred_{idx}"] = config.bad_value

            # using dict to accumulate fluxes and errors
            # should also save separate errors for each parent template
            flux_sum: defaultdict[int, float] = defaultdict(float)
            err_sum: defaultdict[int, float] = defaultdict(float)
            err_pred_sum: defaultdict[int, float] = defaultdict(float)
            for pid, fl, er, ep in zip(parent_ids, fluxes, errs, err_pred):
                if pid is None:
                    continue
                flux_sum[pid] += fl
                err_sum[pid] = np.sqrt(err_sum[pid] ** 2 + er**2)
                err_pred_sum[pid] = np.sqrt(err_pred_sum[pid] ** 2 + ep**2)

            for pid, fl in flux_sum.items():
                ci = id_to_index.get(pid)
                if ci is None:
                    continue
                cat[f"flux_{idx}"][ci] = fl
                cat[f"err_{idx}"][ci] = err_sum[pid]
                cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]

            if k > 1 and config.multi_resolution_method == "upsample":
                print(f"Downsampling residuals by factor {k}")
                res = block_reduce(res, k, func=np.sum)

            if "astro" in locals():
                self.astro.append(astro)
            self.residuals.append(res)
            self.fit.append(fitter)
            self.templates.append(templates)

        print(f"Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")

        self.catalog = cat

        return cat, self.residuals, self.fit


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
