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
    rms_images: Sequence[np.ndarray] | None = None,
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
    if rms_images is not None and len(rms_images) != len(images):
        raise ValueError("Number of RMS images must match number of images")

    from .psf import PSF
    from .templates import Templates
    from .fit import SparseFitter

    catalog = catalog.copy()
    positions = list(zip(catalog["y"], catalog["x"]))

    hires_image = images[0]
    hires_psf = PSF.from_array(psfs[0])

    residuals = []
    for idx, (image, psf_arr) in enumerate(zip(images, psfs)):
        psf = PSF.from_array(psf_arr)
        if idx == 0:
            kernel = np.zeros_like(psf_arr)
            cy, cx = psf_arr.shape[0] // 2, psf_arr.shape[1] // 2
            kernel[cy, cx] = 1.0
        else:
            kernel = hires_psf.matching_kernel(psf)
     
        tmpls = Templates.from_image(hires_image, segmap, positions, kernel)
        if extend_templates == "psf_dilation":
            tmpls.extend_with_psf_dilation(psf.array, kernel)
        elif extend_templates == "2d_moffat":
            tmpls.extend_with_moffat(kernel)

        weights = None
        if rms_images is not None and rms_images[idx] is not None:
            weights = 1.0 / rms_images[idx] ** 2

        fitter = SparseFitter(tmpls.templates, image, weights)
        fluxes, _ = fitter.solve()
        resid = fitter.residual()
        errs = fitter.flux_errors()
#        errs = fitter.predicted_errors()

        if weights is not None:
            pred = fitter.predicted_errors()
            catalog[f"err_pred_{idx}"] = pred
        catalog[f"flux_{idx}"] = fluxes
        catalog[f"err_{idx}"] = errs
        residuals.append(resid)

    return catalog, np.stack(residuals)
