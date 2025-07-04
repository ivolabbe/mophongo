import numpy as np
from astropy.table import Table

from mophongo.psf import PSF
from mophongo.templates import _convolve2d, Template
import matplotlib.pyplot as plt


def _pad_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center pad ``array`` to ``shape`` with zeros."""
    ny, nx = array.shape
    ty, tx = shape
    py = (ty - ny) // 2
    px = (tx - nx) // 2
    return np.pad(array, ((py, ty - ny - py), (px, tx - nx - px)))


def make_simple_data(
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, Table, list[np.ndarray], list[float]]:
    """Create a synthetic dataset with 10 well-separated sources."""
    rng = np.random.default_rng(seed)

    ny = nx = 101
    nsrc = 10

    hi_fwhm = 2.5
    lo_fwhm = 4.0 * hi_fwhm
    psf_hi = PSF.gaussian(11, hi_fwhm, hi_fwhm)
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
    # add small Gaussian noise to the low resolution image to mimic
    # more realistic data used in the pipeline tests
    lowres += rng.normal(scale=0.001, size=lowres.shape)

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
    print('Saving diagnostic image to:', filename)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    data = [hires, lowres, model, residual]
    titles = ["hires", "lowres", "model", "residual"]
    std = residual.std()
    vlim = 5 * std
    for ax, img, title in zip(axes.ravel(), data, titles):
        if title == "residual":
            ax.imshow(img, cmap="gray", origin="lower", vmin=-vlim, vmax=vlim)
        else:
            ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_psf_diagnostic(
    filename: str,
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    kernel: np.ndarray,
) -> None:
    """Visualize PSF matching."""
    conv = _convolve2d(psf_hi, kernel)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    data = [psf_hi, kernel, conv, psf_lo]
    titles = ["psf_hi", "kernel", "hi*kernel", "psf_lo"]
    for ax, img, title in zip(axes.ravel(), data, titles):
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_fit_diagnostic(
    filename: str,
    image: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Visualize SparseFitter results."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    std = residual.std()
    vlim = 5 * std
    data = [image, model, residual]
    titles = ["image", "model", "residual"]
    for ax, img, title in zip(axes, data, titles):
        if title == "residual":
            ax.imshow(img, cmap="gray", origin="lower", vmin=-vlim, vmax=vlim)
        else:
            ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_template_diagnostic(
    filename: str,
    hires: np.ndarray,
    templates: list[Template],
) -> None:
    """Show hires image and extracted templates."""
    n = len(templates)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    axes = np.atleast_1d(axes)
    axes[0].imshow(hires, cmap="gray", origin="lower")
    axes[0].set_title("hires")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i, tmpl in enumerate(templates, start=1):
        axes[i].imshow(tmpl.array, cmap="gray", origin="lower")
        axes[i].set_title(f"tmpl {i}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
