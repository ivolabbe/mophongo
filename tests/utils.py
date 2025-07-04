import numpy as np
from astropy.table import Table

from mophongo.psf import PSF
from mophongo.templates import _convolve2d
import matplotlib.pyplot as plt


def make_simple_data():
    """Create a small synthetic dataset for pipeline tests."""
    ny, nx = 71, 71
    yx = [(7, 7), (14, 14)]
    fluxes = [2.0, 3.0]

    psf_hi = PSF.gaussian(11, 2.0, 2.0)
    psf_lo = PSF.gaussian(21, 5.0, 5.0)
    kernel = psf_hi.matching_kernel(psf_lo)

    segmap = np.zeros((ny, nx), dtype=int)
    for i, (y, x) in enumerate(yx, start=1):
        segmap[y - 5 : y + 6, x - 5 : x + 6] = i

    hires = np.zeros((ny, nx))
    for (y, x), f in zip(yx, fluxes):
        yy = slice(y - 5, y + 6)
        xx = slice(x - 5, x + 6)
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
    print('Saving diagnostic image to:', filename)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    data = [hires, lowres, model, residual]
    titles = ["hires", "lowres", "model", "residual"]
    # use the same scaling for all low resolution images in imshow
    vmin = min(np.min(img) for img in data)
    vmax = max(np.max(img) for img in data)
    for img in data:
        img -= vmin # normalize to [0, 1]
 
    for ax, img, title in zip(axes.ravel(), data, titles):
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
