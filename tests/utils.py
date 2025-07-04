import numpy as np
from astropy.table import Table

from mophongo.psf import PSF
from mophongo.templates import _convolve2d
import matplotlib.pyplot as plt


def _pad_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center pad ``array`` to ``shape`` with zeros."""
    ny, nx = array.shape
    ty, tx = shape
    py = (ty - ny) // 2
    px = (tx - nx) // 2
    return np.pad(array, ((py, ty - ny - py), (px, tx - nx - px)))


def make_simple_data(seed: int = 0) -> tuple[list[np.ndarray], np.ndarray, Table, list[np.ndarray], list[float]]:
    """Create a synthetic dataset with 10 well-separated sources."""
    rng = np.random.default_rng(seed)

    ny = nx = 101
    nsrc = 10

    hi_fwhm = 2.0
    lo_fwhm = 5.0 * hi_fwhm

    psf_hi = PSF.gaussian(9, hi_fwhm, hi_fwhm)
    psf_lo = PSF.gaussian(41, lo_fwhm, lo_fwhm)

    # Expand PSFs to common grid for kernel computation
    size = (
        max(psf_hi.array.shape[0], psf_lo.array.shape[0]),
        max(psf_hi.array.shape[1], psf_lo.array.shape[1]),
    )
    psf_hi_big = PSF.from_array(_pad_to(psf_hi.array, size))
    psf_lo_big = PSF.from_array(_pad_to(psf_lo.array, size))
    kernel = psf_hi_big.matching_kernel(psf_lo_big)

    margin = size[0] // 2 + 1
    segmap = np.zeros((ny, nx), dtype=int)
    positions = []
    while len(positions) < nsrc:
        y = rng.integers(margin, ny - margin)
        x = rng.integers(margin, nx - margin)
        if all(max(abs(y - py), abs(x - px)) > psf_hi.array.shape[0] for py, px in positions):
            positions.append((y, x))

    fluxes = rng.uniform(1.0, 5.0, size=nsrc).tolist()

    hires = np.zeros((ny, nx))
    for i, ((y, x), f) in enumerate(zip(positions, fluxes), start=1):
        r = psf_hi.array.shape[0] // 2
        yy = slice(y - r, y + r + 1)
        xx = slice(x - r, x + r + 1)
        segmap[yy, xx] = i
        hires[yy, xx] += f * psf_hi.array

    lowres = _convolve2d(hires, kernel)

    catalog = Table({'y': [p[0] for p in positions], 'x': [p[1] for p in positions]})

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
