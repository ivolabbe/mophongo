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
    from .fit import SparseFitter
    import warnings

    catalog = catalog.copy()
    positions = list(zip(catalog["x"], catalog["y"]))

    # Step 1: Extract templates from the first image (alternatively, use models)
    tmpl_psf = PSF.from_array(psfs[0])
    tmpl_wht = 0.0 if wht_images[0].max() == 0.0 else wht_images[0]

    tmpls = Templates()
    tmpls.extract_templates(images[0], segmap, positions)
    if extend_templates == 'psf':
        tmpls.extend_with_psf_wings(tmpl_psf.array, inplace=True)

    residuals = []
    for idx in range(1,len(images)):

        # some wave of controlling window goes here
        kernel = tmpl_psf.matching_kernel(psfs[idx])

        templates = tmpls.convolve_templates(kernel, inplace=False)

        # assume weights are dominated by photometry image (for proper weights see sparse fitter, needes iteration)
        weights = wht_images[idx]
        
        fitter = SparseFitter(templates, images[idx], weights)
        fluxes, _ = fitter.solve()
        resid = fitter.residual()
        errs = fitter.flux_errors()

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
            catalog[f"err_pred_{idx}"] = pred

        catalog[f"flux_{idx}"] = fluxes
        catalog[f"err_{idx}"] = errs
        catalog[f"err_{idx}"] = pred
        residuals.append(resid)

    return catalog, residuals, tmpls
