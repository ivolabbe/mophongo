from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .fit import fit_fluxes


@dataclass
class PhotometryResult:
    fluxes: np.ndarray
    model: np.ndarray
    residual: np.ndarray


def run_photometry(image: np.ndarray, positions: Sequence[Tuple[int, int]], psf: np.ndarray) -> PhotometryResult:
    """Run a minimal photometry pipeline.

    Parameters
    ----------
    image : ndarray
        Low-resolution image data.
    positions : sequence of (y, x)
        Pixel coordinates of sources.
    psf : ndarray
        PSF image centered at (0, 0) with odd dimensions.
    """
    fluxes, model, residual = fit_fluxes(image, positions, psf)
    return PhotometryResult(fluxes=fluxes, model=model, residual=residual)
