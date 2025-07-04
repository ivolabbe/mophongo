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
) -> tuple[Table, np.ndarray]:
    """Run photometry on a set of images.

    Parameters
    ----------
    images
        Sequence of image arrays to be fit.
    segmap
        Segmentation map aligned with the images.
    catalog
        Source catalog. New flux columns will be added to a copy of this table.
    psfs
        Point-spread functions matching each image.

    Returns
    -------
    Table
        Table containing the input catalog columns plus measured fluxes.
    ndarray
        Residual image after subtracting the model from the data.
    """

    if len(images) != len(psfs):
        raise ValueError("Number of images and PSFs must match")

    # Lazy imports so this module can be imported without the heavy dependencies
    from . import templates, fit

    # Build PSF-matched templates for each object
    templates_list = templates.extract_templates(images, segmap, catalog, psfs)

    # Solve for object fluxes using a sparse linear solver
    flux_table, residual = fit.sparse_fit(images, templates_list, catalog, psfs)

    return flux_table, residual
