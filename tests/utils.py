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
) -> tuple[
    list[np.ndarray], np.ndarray, Table, list[np.ndarray], list[float], np.ndarray
]:
    """Create a synthetic dataset with 10 well-separated sources.

    The underlying truth image is composed of randomly oriented Gaussian
    ellipses of varying size. It is convolved with the high-resolution PSF
    before resampling to the low-resolution grid.
    """

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

    truth = np.zeros((ny, nx))
    hires = np.zeros((ny, nx))
    for i, ((y, x), f) in enumerate(zip(positions, fluxes), start=1):
        g_fwhm_x = rng.uniform(2.0, 4.0)
        g_fwhm_y = rng.uniform(2.0, 4.0)
        theta = rng.uniform(0, np.pi)
        g_size = 13
        gauss = _gaussian_psf(g_size, g_fwhm_x, g_fwhm_y, theta)
        r = g_size // 2
        yy = slice(y - r, y + r + 1)
        xx = slice(x - r, x + r + 1)
        segmap[yy, xx] = i
        truth[yy, xx] += f * gauss

    hires[:] = _convolve2d(truth, psf_hi.array)
    lowres = _convolve2d(hires, kernel)
    # add small Gaussian noise to the low resolution image to mimic
    # more realistic data used in the pipeline tests
    lowres += rng.normal(scale=0.001, size=lowres.shape)

    catalog = Table({'y': [p[0] for p in positions], 'x': [p[1] for p in positions]})

    return [hires, lowres], segmap, catalog, [psf_hi.array, psf_lo.array], fluxes, truth


def save_diagnostic_image(
    filename: str,
    truth: np.ndarray,
    hires: np.ndarray,
    lowres: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Save diagnostic plot with truth, model and residual."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    data = [truth, hires, lowres, model, residual]
    titles = ["truth", "hires", "lowres", "model", "residual"]
    std = residual.std()
    vlim = 5 * std
    for ax, img, title in zip(axes.ravel()[:5], data, titles):

        if title == "residual":
            ax.imshow(img, cmap="gray", origin="lower", vmin=-vlim, vmax=vlim)
        else:
            ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.ravel()[5].axis("off")

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_psf_diagnostic(
    filename: str,
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    kernel: np.ndarray,
) -> None:
    """Visualize PSF matching with residual and growth-curve ratio."""
    conv = _convolve2d(psf_hi, kernel)
    resid = psf_lo - conv

    def growth(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = img.shape
        cy, cx = (ny - 1) / 2, (nx - 1) / 2
        y, x = np.indices(img.shape)
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        order = np.argsort(r.ravel())
        r_sorted = r.ravel()[order]
        cum = np.cumsum(img.ravel()[order])
        cum /= cum[-1]
        return r_sorted, cum

    r_lo, g_lo = growth(psf_lo)
    _, g_conv = growth(conv)
    ratio = g_lo / g_conv

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    imgs = [psf_hi, kernel, conv, psf_lo, resid]
    titles = ["psf_hi", "kernel", "hi*kernel", "psf_lo", "residual"]
    for ax, img, title in zip(axes.ravel()[:5], imgs, titles):
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_ratio = axes.ravel()[5]
    ax_ratio.plot(r_lo, ratio)
    ax_ratio.set_xlabel("radius (pix)")
    ax_ratio.set_ylabel("lo/conv")
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
