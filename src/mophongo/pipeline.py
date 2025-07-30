"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

# The ``templates`` and ``fit`` modules are expected to provide the following
# functions. They are imported lazily inside :func:`run_photometry` so that this
# module can be imported even if those modules are not yet implemented.


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

    if config is None:
        config = FitConfig()

    if catalog is None:
        segm = SegmentationImage(segmap)
        # check x,y, names
        catalog = SourceCatalog(
            images[0],
            SegmentationImage(segmap),
            error=np.sqrt(1.0 / weights[0]),
            wcs=WCS(header) if header is not None else None,
        )

    cat = catalog.copy()
    positions = list(zip(catalog["x"], catalog["y"]))

    # Step 1: Extract templates from the first image (alternatively, use models)
    tmpls = Templates()
    tmpls.extract_templates(images[0],
                            segmap,
                            positions,
                            wcs=wcs[0] if wcs is not None else None)

    if extend_templates == 'psf' and psfs is not None:
        tmpls.extend_with_psf_wings(psfs[0], inplace=True)


#   tmpls.deduplicate()

    residuals = []
    for idx in range(1, len(images)):

        kernel = None
        if kernels is not None and kernels[idx] is not None:
            kernel = kernels[idx]
            if isinstance(kernel, PSFRegionMap):
                print(f"Using kernel lookup table {kernel.name}")

        wcs_i = wcs[idx] if wcs is not None else None

        templates = tmpls.convolve_templates(kernel, inplace=True)
        print('Pipepline:', len(templates), 'orig templates')

        # assume weights are dominated by photometry image (for proper weights see sparse fitter, needes iteration)
        weights_i = weights[idx] if weights is not None else None

        fitter_cls = GlobalAstroFitter if (
            config.fit_astrometry
            and config.fit_astrometry_joint) else SparseFitter

        if config.fit_astrometry:
            # every iteration will scale and shift the tmpl images
            # accumulated the shifts and scale are recorded in template attributes
            for j in range(config.fit_astrometry_niter):
                print(
                    f"Running iteration {j+1} of {config.fit_astrometry_niter} for astrometry fitting"
                )

                fitter = fitter_cls(templates, images[idx], weights_i, config)
                fluxes, _ = fitter.solve()
                res = fitter.residual()
                errs = fitter.flux_errors()

                if not config.fit_astrometry_joint:
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
                            snr_threshold=config.snr_thresh_astrom)
        else:
            fitter = fitter_cls(templates, images[idx], weights_i, FitConfig())
            fluxes, _ = fitter.solve()
            res = fitter.residual()
            errs = fitter.flux_errors()
        print(f"Done...")

        fit_templates = fitter.templates
        # put fluxes into catalog based on template IDs
        tmpl_ids = [tmpl.id for tmpl in fit_templates]
        id_to_index = {id_: i for i, id_ in enumerate(cat['id'])}
        tmpl_idx = [id_to_index.get(tid, None) for tid in tmpl_ids]

        # fill arrays with bad_value
        full_flux = np.full(len(cat), config.bad_value)
        full_err = np.full(len(cat), config.bad_value)
        full_pred = np.full(len(cat), config.bad_value)
        for j, ci in enumerate(tmpl_idx):
            if ci is not None:
                full_flux[ci] = fluxes[j]
                full_err[ci] = errs[j]
                if weights_i is not None:
                    full_pred[ci] = pred[j]
        cat[f"flux_{idx}"] = full_flux
        cat[f"err_{idx}"] = full_err
        cat[f"err_pred_{idx}"] = full_pred

        residuals.append(res)

    return cat, residuals, fitter
