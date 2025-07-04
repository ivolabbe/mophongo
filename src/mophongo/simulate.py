import numpy as np


def add_sources(image, positions, fluxes, psf):
    """Add scaled PSFs to an image at integer pixel positions."""
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
    image = np.zeros(shape, dtype=float)
    add_sources(image, positions, fluxes, psf)
    noise = rng.normal(scale=noise_std, size=shape)
    return image + noise
