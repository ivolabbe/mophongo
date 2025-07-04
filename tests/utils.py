import numpy as np
from astropy.table import Table


from mophongo.psf import PSF
from mophongo.templates import _convolve2d, Template
import matplotlib.pyplot as plt


def make_simple_data(
    seed: int = 1001,
    nsrc: int = 15,
) -> tuple[
    list[np.ndarray], np.ndarray, Table, list[np.ndarray], np.ndarray
]:
    """Create a synthetic dataset with nsrc well-separated sources.

    The underlying truth image is composed of randomly oriented Gaussian
    ellipses of varying size. It is convolved with the high-resolution PSF
    before resampling to the low-resolution grid.
    """

    rng = np.random.default_rng(seed)
    ny = nx = 101

    hi_fwhm = 2.5
    lo_fwhm = 4.0 * hi_fwhm
    psf_hi = PSF.gaussian(11, hi_fwhm, hi_fwhm)
    psf_lo = PSF.gaussian(41, lo_fwhm, lo_fwhm)

    kernel = psf_hi.matching_kernel(psf_lo)

    margin = kernel.shape[0] // 2 + 1
    segmap = np.zeros((ny, nx), dtype=int)
    segimg = np.zeros((ny, nx), dtype=float)  # to track current image values for segmentation
    positions = []
    while len(positions) < nsrc:
        y = rng.integers(margin, ny - margin)
        x = rng.integers(margin, nx - margin)
        if all(np.hypot(y - py, x - px) > psf_hi.array.shape[0] / 2 for py, px in positions):
            positions.append((y, x))

    fluxes = rng.uniform(1.0, 5.0, size=nsrc)
    g_fwhm_xs = []
    g_fwhm_ys = []
    thetas = []
    g_sizes = []
    truth = np.zeros((ny, nx))
    for i, ((y, x), f) in enumerate(zip(positions, fluxes), start=1):
        g_fwhm_x = rng.uniform(2.0, 6.0)
        g_fwhm_y = rng.uniform(2.0, 6.0)
        theta = rng.uniform(0, np.pi)
        # Expand g_size to fit the largest FWHM, with margin for orientation
        max_fwhm = max(g_fwhm_x, g_fwhm_y)
        g_size = int(np.ceil(6 * max_fwhm)) | 1  # ensure odd size
        gauss = PSF.gaussian(g_size, g_fwhm_x, g_fwhm_y, theta).array
        peak = gauss.max()
        mask = gauss > (peak / 1000.0)
        r = g_size // 2
        yy = slice(y - r, y + r + 1)
        xx = slice(x - r, x + r + 1)
        # Only assign segmap where new pixels are brighter than existing segimg
        seg_slice = (slice(max(0, y - r), min(ny, y + r + 1)),
                     slice(max(0, x - r), min(nx, x + r + 1)))
        gauss_crop = gauss[
            max(0, r - y):g_size - max(0, y + r + 1 - ny),
            max(0, r - x):g_size - max(0, x + r + 1 - nx)
        ]
        mask_crop = mask[
            max(0, r - y):g_size - max(0, y + r + 1 - ny),
            max(0, r - x):g_size - max(0, x + r + 1 - nx)
        ]
        segimg_crop = segimg[seg_slice]
        segmap_crop = segmap[seg_slice]
        brighter = (gauss_crop * f > segimg_crop) & mask_crop
        segimg_crop[brighter] = gauss_crop[brighter] * f
        segmap_crop[brighter] = i
        segimg[seg_slice] = segimg_crop
        segmap[seg_slice] = segmap_crop
        # Add to truth image
        truth[yy, xx] += f * gauss
        g_fwhm_xs.append(g_fwhm_x)
        g_fwhm_ys.append(g_fwhm_y)
        thetas.append(theta)
        g_sizes.append(g_size)

    hires = _convolve2d(truth, psf_hi.array)
    lowres = _convolve2d(hires, kernel)
    # add small Gaussian noise to the low resolution image to mimic
    # more realistic data used in the pipeline tests
    lowres += rng.normal(scale=0.001, size=lowres.shape)

    catalog = Table({
        'y': [p[0] for p in positions],
        'x': [p[1] for p in positions],
        'g_size': g_sizes,
        'g_fwhm_x': g_fwhm_xs,
        'g_fwhm_y': g_fwhm_ys,
        'theta': thetas,
        'flux_true': fluxes,
    })

    return [hires, lowres], segmap, catalog, [psf_hi.array, psf_lo.array], truth


def save_diagnostic_image(
    filename: str,
    truth: np.ndarray,
    hires: np.ndarray,
    lowres: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    segmap: np.ndarray = None,
) -> None:
    """Save diagnostic plot with truth, model, residual, and optionally segmentation map."""
    n_panels = 6 if segmap is not None else 5
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    data = [truth, hires, lowres, model, residual]
    titles = ["truth", "hires", "lowres", "model", "residual"]

    for ax, img, title in zip(axes.ravel()[:5], data, titles):
        if title == "residual":
            std = residual.std()
            vlim = 5 * std
            ax.imshow(img, cmap="gray", origin="lower", vmin=-vlim, vmax=vlim)
        else:
            ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    if segmap is not None:
        ax = axes.ravel()[5]
        ax.imshow(segmap, cmap="nipy_spectral", origin="lower")
        ax.set_title("segmap")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
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
        if title == "residual":
            ax.imshow(img, cmap="gray", origin="lower", vmin=-0.1 * psf_lo.max(), vmax=0.1 * psf_lo.max())
        else:
            ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_ratio = axes.ravel()[5]
    ax_ratio.plot(r_lo, ratio)
    ax_ratio.set_ylim(0.9, 1.1)
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


def save_flux_vs_truth_plot(
    filename: str,
    truth: np.ndarray,
    recovered: np.ndarray,
    label: str = "Recovered Flux",
) -> None:
    """Plot recovered flux vs truth and recovered/true vs truth, save to file."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Recovered vs True
    axes[0].scatter(truth, recovered, s=30)
    minval = min(truth.min(), recovered.min())
    maxval = max(truth.max(), recovered.max())
    axes[0].plot([minval, maxval], [minval, maxval], "k--", label="y=x")
    axes[0].set_xlabel("True Flux")
    axes[0].set_ylabel(label)
    axes[0].set_title("Recovered vs True Flux")
    axes[0].legend()

    # Panel 2: Ratio vs True
    ratio = np.array(recovered) / np.array(truth)
    axes[1].scatter(truth, ratio, s=30)
    axes[1].axhline(1.0, color="k", linestyle="--", label="ratio=1")
    axes[1].set_xlabel("True Flux")
    axes[1].set_ylabel("Recovered / True")
    axes[1].set_title("Flux Ratio vs True")
    axes[1].set_ylim(0.8, 1.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
