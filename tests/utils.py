import numpy as np


def moffat_psf(shape, fwhm, beta=2.5):
    """Return a normalized Moffat PSF image."""
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    rr2 = (x - cx) ** 2 + (y - cy) ** 2
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    psf = (1 + rr2 / alpha ** 2) ** (-beta)
    psf /= psf.sum()
    return psf


def add_sources(image, positions, fluxes, psf):
    """Add scaled PSFs to an image."""
    ny, nx = image.shape
    ph, pw = psf.shape
    oy = ph // 2
    ox = pw // 2
    for (y, x), f in zip(positions, fluxes):
        y0 = y - oy
        x0 = x - ox
        y1 = y0 + ph
        x1 = x0 + pw
        if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
            raise ValueError("source too close to edge")
        image[y0:y1, x0:x1] += f * psf


def simulate_image(shape, positions, fluxes, psf, noise_std, rng):
    """Generate an image with sources and additive Gaussian noise."""
    image = np.zeros(shape)
    add_sources(image, positions, fluxes, psf)
    noise = rng.normal(scale=noise_std, size=shape)
    return image + noise


def fit_fluxes(image, positions, psf):
    """Fit fluxes for fixed source positions with linear least squares."""
    ny, nx = image.shape
    ph, pw = psf.shape
    oy = ph // 2
    ox = pw // 2
    nsrc = len(positions)
    A = np.zeros((ny * nx, nsrc))
    for i, (y, x) in enumerate(positions):
        stamp = np.zeros_like(image)
        stamp[y - oy : y - oy + ph, x - ox : x - ox + pw] = psf
        A[:, i] = stamp.ravel()
    b = image.ravel()
    fluxes, *_ = np.linalg.lstsq(A, b, rcond=None)
    model = A @ fluxes
    model_image = model.reshape(ny, nx)
    residual = image - model_image
    return fluxes, model_image, residual
