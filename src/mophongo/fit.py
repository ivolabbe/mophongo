import numpy as np


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
