import numpy as np
from astropy.table import Table

from mophongo.psf import PSF
from mophongo.templates import _convolve2d
import matplotlib.pyplot as plt


def make_simple_data():
    """Create a small synthetic dataset for pipeline tests."""
    ny, nx = 21, 21
    yx = [(7, 7), (14, 14)]
    fluxes = [2.0, 3.0]

    psf_hi = PSF.gaussian(7, 2.0, 2.0)
    psf_lo = PSF.gaussian(7, 3.0, 3.0)
    kernel = psf_hi.matching_kernel(psf_lo)

    segmap = np.zeros((ny, nx), dtype=int)
    for i, (y, x) in enumerate(yx, start=1):
        segmap[y - 3 : y + 4, x - 3 : x + 4] = i

    hires = np.zeros((ny, nx))
    for (y, x), f in zip(yx, fluxes):
        yy = slice(y - 3, y + 4)
        xx = slice(x - 3, x + 4)
        hires[yy, xx] += f * psf_hi.array

    lowres = _convolve2d(hires, kernel)

    catalog = Table({'y': [p[0] for p in yx], 'x': [p[1] for p in yx]})

    return [hires, lowres], segmap, catalog, [psf_hi.array, psf_lo.array], fluxes


def save_diagnostic_image(
    filename: str,
    hires: np.ndarray,
    lowres: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Save 2x2 diagnostic plot with grayscale images."""
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    data = [hires, lowres, model, residual]
    titles = ["hires", "lowres", "model", "residual"]
    for ax, img, title in zip(axes.ravel(), data, titles):
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
