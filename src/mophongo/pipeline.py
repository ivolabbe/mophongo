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

from astropy.table import Table
from astropy.wcs import WCS

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

#   tmpls.deduplicate()
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
            print( f"Running iteration {j+1} of {config.fit_astrometry_niter}")

            fitter = fitter_cls(templates, images[idx], weights_i, config)
            fluxes, info = fitter.solve()
            errs = fitter.flux_errors()
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
                        length_scale=500.0)
                else:
                    # this also applies the shifts to the templates
                    correct_astrometry_polynomial(
                        tmpls.templates,
                        res,
                        fitter.solution,
                        order=config.astrom_basis_order,
                        box_size=5,
                        snr_threshold=config.snr_thresh_astrom)
                    
                # check if this call is ok, only makes sense if we rebuild the normal matrix
                # TODO: track this from the templates is_dirty flag
                fitter._ata = None  # @@@ do this properly
                fluxes, info = fitter.solve()
                errs = fitter.flux_errors()

        err_pred = fitter.predicted_errors()
 
        print(f"Done...")
 
        # put fluxes into catalog based on template IDs
        tmpl_ids = [tmpl.id for tmpl in templates if tmpl.id is not None]
        id_to_index = {id_: i for i, id_ in enumerate(cat['id'])}
        tmpl_idx = [id_to_index.get(tid, None) for tid in tmpl_ids]
        cat[f"flux_{idx}"] = config.bad_value
        cat[f"err_{idx}"] = config.bad_value
        cat[f"err_pred_{idx}"] = config.bad_value
        cat[f"flux_{idx}"][tmpl_idx] = fluxes
        cat[f"err_{idx}"][tmpl_idx] = errs
        cat[f"err_pred_{idx}"][tmpl_idx] = err_pred

        residuals.append(res)

    print(f'Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB')

    return cat, residuals, fitter
