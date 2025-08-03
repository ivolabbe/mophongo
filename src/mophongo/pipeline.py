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
import numpy as np
from collections import defaultdict

from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D


def _per_source_chi2(residual: np.ndarray, sigma: np.ndarray, templates: Sequence[Template]) -> np.ndarray:
    """Compute reduced chi-square for each template.

    Parameters
    ----------
    residual
        Residual image from the first-pass fit.
    sigma
        Standard deviation image (1/sqrt(weights)).
    templates
        List of templates used in the fit.

    Returns
    -------
    ndarray
        Array of chi^2/nu values, one per template in ``templates``.
    """

    chi = np.zeros(len(templates), dtype=float)
    for i, tmpl in enumerate(templates):
        y0, y1, x0, x1 = tmpl.bbox
        res = residual[y0:y1, x0:x1] / sigma[y0:y1, x0:x1]
        chi[i] = np.mean(res ** 2)
    return chi


def _extract_psf_at(tmpl: Template, psf: np.ndarray) -> np.ndarray:
    """Return a PSF stamp matching the template size."""
    from scipy.ndimage import shift

    ny, nx = tmpl.data.shape
    cx_psf, cy_psf = psf.shape[1] // 2, psf.shape[0] // 2

    xc, yc = tmpl.input_position_cutout
    dx = xc - (nx // 2)
    dy = yc - (ny // 2)

    shifted = shift(psf, shift=(dy, dx), order=3, mode="constant", cval=0.0, prefilter=False)
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
    #    fit_astrometry: bool = False,
    #    astrom_order: int = 1,
    config: FitConfig | None = None,
) -> tuple[Table, np.ndarray]:
    """Run photometry on a set of images.

    Parameters
    ----------
    images
        Sequence of image arrays. The first image defines the high-resolution
        detection plane used to build templates.
    segmap
        Segmentation map aligned with the images. If not catalog provided, all sources are fit.
    catalog
        Optional source catalog. Must provide ``y`` and ``x`` columns giving pixel
        positions. Performs only fitting of these sources, not all sources in Segmentation image. 
    psfs
        Point-spread functions matching each image.
    weights
        Optional sequence of weight arrays corresponding to ``images``.
    kernels
        Optional sequence of precomputed kernels or :class:`~mophongo.kernels.KernelLookup`
        objects corresponding to ``images``. If ``None``, matching kernels are
        computed from ``psfs``.

    Returns
    -------
    Table
        Table containing the input catalog columns plus measured fluxes for
        each image.
    ndarray
        Stack of residual images after subtracting the model from each input
        image. The array has shape ``(n_images, ny, nx)``.
    """

    if psfs is not None:
        if len(images) != len(psfs):
            raise ValueError("Number of images and PSFs must match")
    if weights is None and wht_images is not None:
        weights = wht_images
    if weights is not None and len(weights) != len(images):
        raise ValueError("Number of weight images must match number of images")

    from .psf import PSF
    from .templates import Templates
    from .fit import SparseFitter, FitConfig
    from .astro_fit import GlobalAstroFitter
    from .local_astrometry import (
        correct_astrometry_polynomial,
        correct_astrometry_gp,
    )
    from .psf_map import PSFRegionMap
    from . import utils
    import warnings

    print(f'Pipeline (start) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

    if config is None:
        config = FitConfig()
    print(f"Pipeline config: {config}")

    cat = catalog.copy()
    positions = list(zip(catalog["x"], catalog["y"]))


    # Step 1: Extract templates from the first image (alternatively, use models)
    tmpls = Templates()
    tmpls.extract_templates(images[0],
                            segmap,
                            positions,
                            wcs=wcs[0] if wcs is not None else None)
    print('Pipepline:', len(tmpls.templates), f'extracted templates, dropped {len(positions)-len(tmpls.templates)}.')
    print(f'Pipeline (templates) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

    if extend_templates == 'psf' and psfs is not None:
        tmpls.extend_with_psf_wings(psfs[0], inplace=True)

    tmpls.deduplicate()
    print(f'Templates: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

    residuals = []
    for idx in range(1, len(images)):

        kernel = None
        if kernels is not None and kernels[idx] is not None:
            kernel = kernels[idx]
            if isinstance(kernel, PSFRegionMap):
                print(f"Using kernel lookup table {kernel.name}")

        wcs_i = wcs[idx] if wcs is not None else None

        templates = tmpls.convolve_templates(kernel, inplace=True)
        print(f'Pipeline (convolved) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

        # assume weights are dominated by photometry image (for proper weights see sparse fitter, needes iteration)
        weights_i = weights[idx] if weights is not None else None

        fitter_cls = GlobalAstroFitter if (
            config.fit_astrometry
            and config.fit_astrometry_joint) else SparseFitter

        # every iteration will scale and shift the tmpl images
        # accumulated the shifts and scale are recorded in template attributes
        niter = config.fit_astrometry_niter if config.fit_astrometry else 1

        for j in range(niter):
            print(f"Running iteration {j+1} of {config.fit_astrometry_niter}")

            fitter = fitter_cls(templates, images[idx], weights_i, config)
            fluxes, errs, info = fitter.solve()
            print(f'Pipeline (solve) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')
            res = fitter.residual()
            print(f'Pipeline (residual) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

            if config.fit_astrometry and not config.fit_astrometry_joint:
                print('fitting astrometry separately')
                if config.astrom_model == "gp":
                    correct_astrometry_gp(
                        tmpls.templates,
                        res,
                        fitter.solution,
                        box_size=5,
                        snr_threshold=config.snr_thresh_astrom,
                        length_scale=500.0,
                    )
                else:
                    # this also applies the shifts to the templates
                    correct_astrometry_polynomial(
                        tmpls.templates,
                        res,
                        fitter.solution,
                        order=config.astrom_basis_order,
                        box_size=5,
                        snr_threshold=config.snr_thresh_astrom,
                    )

                # check if this call is ok, only makes sense if we rebuild the normal matrix
                # TODO: track this from the templates is_dirty flag
                fitter._ata = None  # @@@ do this properly
                fluxes, errs, info = fitter.solve()
                res = fitter.residual()

        # second pass: add extra templates for poor fits
        if (config.multi_tmpl_psf_core or config.multi_tmpl_colour) and psfs is not None:
            sigma_img = (
                np.sqrt(1.0 / weights_i)
                if weights_i is not None
                else np.ones_like(images[idx])
            )
            chi_nu = _per_source_chi2(res, sigma_img, templates)
            bad_idx = np.where(chi_nu > config.multi_tmpl_chi2_thresh)[0]
            if bad_idx.size > 0:
                for bi in bad_idx:
                    parent = templates[bi]
                    if config.multi_tmpl_psf_core and psfs is not None:
                        stamp = _extract_psf_at(parent, psfs[idx])
                        tmpls.add_component(parent, stamp, "psf")
                    # Placeholder for additional components (e.g. colour maps)
                templates = tmpls.templates
                fitter = fitter_cls(templates, images[idx], weights_i, config)
                fluxes, errs, info = fitter.solve()
                res = fitter.residual()

        err_pred = fitter.predicted_errors()

        print("Done...")

        # put fluxes into catalog based on template IDs
        parent_ids = [
            tmpl.parent_id if getattr(tmpl, "parent_id", None) is not None else tmpl.id
            for tmpl in templates
        ]
        id_to_index = {id_: i for i, id_ in enumerate(cat["id"])}
        cat[f"flux_{idx}"] = config.bad_value
        cat[f"err_{idx}"] = config.bad_value
        cat[f"err_pred_{idx}"] = config.bad_value

        flux_sum: defaultdict[int, float] = defaultdict(float)
        err_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_sum: defaultdict[int, float] = defaultdict(float)
        for pid, fl, er, ep in zip(parent_ids, fluxes, errs, err_pred):
            if pid is None:
                continue
            flux_sum[pid] += fl
            err_sum[pid] = np.sqrt(err_sum[pid] ** 2 + er ** 2)
            err_pred_sum[pid] = np.sqrt(err_pred_sum[pid] ** 2 + ep ** 2)

        for pid, fl in flux_sum.items():
            ci = id_to_index.get(pid)
            if ci is None:
                continue
            cat[f"flux_{idx}"][ci] = fl
            cat[f"err_{idx}"][ci] = err_sum[pid]
            cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]

        residuals.append(res)

    print(f'Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

    return cat, residuals, fitter
