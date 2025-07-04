import numpy as np


def moffat_psf(shape, fwhm, beta=2.5):
    """Return a normalized circular Moffat PSF image.

    Parameters
    ----------
    shape : tuple of int
        (ny, nx) shape of the output PSF image.
    fwhm : float
        Full-width at half-maximum of the PSF in pixels.
    beta : float, optional
        Moffat beta parameter controlling the wings. Default is 2.5.
    """
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    rr2 = (x - cx) ** 2 + (y - cy) ** 2
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    psf = (1 + rr2 / alpha ** 2) ** (-beta)
    psf /= psf.sum()
    return psf
