"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

from typing import Sequence

from .kernels import KernelLookup
from .astro_fit import GlobalAstroFitter
from .fit import FitConfig, SparseFitter

import numpy as np
from astropy.table import Table

# The ``templates`` and ``fit`` modules are expected to provide the following
# functions. They are imported lazily inside :func:`run_photometry` so that this
# module can be imported even if those modules are not yet implemented.


def run_photometry(
    images: Sequence[np.ndarray],
    segmap: np.ndarray,
    catalog: Table,
    psfs: Sequence[np.ndarray],
    wht_images: Sequence[np.ndarray] | None = None,
    *,
    extend_templates: str | None = None,
    kernels: Sequence[np.ndarray | KernelLookup] | None = None,
    fit_astrometry: bool = False,
    astrom_order: int = 3,
) -> tuple[Table, np.ndarray]:
    """Run photometry on a set of images.

    Parameters
    ----------
    images
        Sequence of image arrays. The first image defines the high-resolution
        detection plane used to build templates.
    segmap
        Segmentation map aligned with the images.
    catalog
        Source catalog. Must provide ``y`` and ``x`` columns giving pixel
        positions. New flux columns will be added to a copy of this table.
    psfs
        Point-spread functions matching each image.
    kernels
        Optional sequence of precomputed kernels or :class:`~mophongo.kernels.KernelLookup`
        objects corresponding to ``images``. If ``None``, matching kernels are
        computed from ``psfs``.
    fit_astrometry
        If ``True`` fit a smooth astrometric shift field simultaneously with
        fluxes.
    astrom_order
        Order of the 2-D Chebyshev basis used for the shift model.

    Returns
    -------
    Table
        Table containing the input catalog columns plus measured fluxes for
        each image.
    ndarray
        Stack of residual images after subtracting the model from each input
        image. The array has shape ``(n_images, ny, nx)``.
    """

    if len(images) != len(psfs):
        raise ValueError("Number of images and PSFs must match")
    if wht_images is not None and len(wht_images) != len(images):
        raise ValueError("Number of RMS images must match number of images")

    from .psf import PSF
    from .templates import Templates
    import warnings

    catalog = catalog.copy()
    positions = list(zip(catalog["x"], catalog["y"]))

    # Step 1: Extract templates from the first image (alternatively, use models)
    tmpl_psf = PSF.from_array(psfs[0])

    tmpls = Templates()
    tmpls.extract_templates(images[0], segmap, positions)
    if extend_templates == 'psf':
        tmpls.extend_with_psf_wings(tmpl_psf.array, inplace=True)

    cfg = FitConfig(
        fit_astrometry=fit_astrometry,
        astrom_basis_order=astrom_order,
    )

    residuals = []
    for idx in range(1, len(images)):

        if kernels is not None and kernels[idx] is not None:
            kernel = kernels[idx]
        else:
            kernel = tmpl_psf.matching_kernel(psfs[idx])

        templates = tmpls.convolve_templates(kernel, inplace=False)
        nflux = len(templates)

        # assume weights are dominated by photometry image (for proper weights see sparse fitter, needes iteration)
        weights = wht_images[idx] if wht_images is not None else None
        
        fitter_cls = GlobalAstroFitter if cfg.fit_astrometry else SparseFitter
        if fitter_cls is GlobalAstroFitter:
            fitter = fitter_cls(templates, images[idx], weights, segmap, cfg)
        else:
            fitter = fitter_cls(templates, images[idx], weights, cfg)
        fluxes, _ = fitter.solve(cfg)
        resid = fitter.residual()
        errs = fitter.flux_errors()
        if cfg.fit_astrometry:
            fluxes = fluxes[:nflux]
            errs = errs[:nflux]

        if len(tmpls.templates) != len(catalog):
            warnings.warn(
                f"Number of templates ({len(tmpls.templates)}) does not match number of sources in catalog ({len(catalog)})."
            )
            # Get template IDs from position_original
            tmpl_ids = [
                segmap[tmpl.position_original[::-1]]
                for tmpl in tmpls.templates
            ]
            # Map template IDs to their indices in the catalog
            id_to_index = {id_: i for i, id_ in enumerate(catalog['id'])}
            # Find catalog indices for each template
            tmpl_idx = [id_to_index.get(tid, None) for tid in tmpl_ids]

        if weights is not None:
            pred = fitter.predicted_errors()
            if cfg.fit_astrometry:
                pred = pred[:nflux]

        if len(tmpls.templates) == len(catalog):
            if weights is not None:
                catalog[f"err_pred_{idx}"] = pred
            catalog[f"flux_{idx}"] = fluxes
            catalog[f"err_{idx}"] = errs
            if weights is not None:
                catalog[f"err_{idx}"] = pred
        else:
            # fill arrays with NaN and map via tmpl_idx
            full_flux = np.full(len(catalog), np.nan)
            full_err = np.full(len(catalog), np.nan)
            full_pred = np.full(len(catalog), np.nan)
            for j, ci in enumerate(tmpl_idx):
                if ci is not None:
                    full_flux[ci] = fluxes[j]
                    full_err[ci] = errs[j]
                    if weights is not None:
                        full_pred[ci] = pred[j]
            catalog[f"flux_{idx}"] = full_flux
            catalog[f"err_{idx}"] = full_err
            if weights is not None:
                catalog[f"err_pred_{idx}"] = full_pred
        residuals.append(resid)

    return catalog, residuals, tmpls
