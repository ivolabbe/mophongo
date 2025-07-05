import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize

from mophongo.psf import PSF
from mophongo.templates import _convolve2d, Template
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_model_params, make_model_image
from photutils.segmentation import detect_sources, deblend_sources


def lupton_norm(img):
    vmin, vmax = img.min(), np.percentile(img, 90)
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())


def make_simple_data(
    seed: int = 1001,
    nsrc: int = 15,
) -> tuple[
    list[np.ndarray], np.ndarray, Table, list[np.ndarray], np.ndarray
]:
    """Create a synthetic dataset with ``nsrc`` sources.

    A truth image with randomly positioned Gaussian sources is generated
    using ``photutils.datasets``. The truth image is convolved with the
    high- and low-resolution PSFs to create matching images with Gaussian
    noise. A segmentation map is created from the high-resolution image by
    running ``detect_sources`` followed by ``deblend_sources``. The returned
    segmentation IDs correspond to the input catalog order, with any missing
    detections represented by a single labeled pixel at the source position.
    """

    rng = np.random.default_rng(seed)
    ny = nx = 101

    hi_fwhm = 2.5
    lo_fwhm = 4.0 * hi_fwhm
    psf_hi = PSF.gaussian(11, hi_fwhm, hi_fwhm)
    psf_lo = PSF.gaussian(41, lo_fwhm, lo_fwhm)

    params = make_model_params(
        (ny, nx),
        nsrc,
        x_name="x_mean",
        y_name="y_mean",
        min_separation=int(hi_fwhm * 4),
        border_size=psf_lo.array.shape[0] // 2,
        seed=rng,
        amplitude=(1.0, 10.0),
        x_stddev=(1.0, 4.0),
        y_stddev=(1.0, 4.0),
        theta=(0, np.pi),
    )

    truth = make_model_image(
        (ny, nx),
        Gaussian2D(),
        params,
        bbox_factor=6.0,
        x_name="x_mean",
        y_name="y_mean",
    )

    hires = _convolve2d(truth, psf_hi.array)
    lowres = _convolve2d(truth, psf_lo.array)

    flux_true = (
        params["amplitude"]
        * 2
        * np.pi
        * params["x_stddev"]
        * params["y_stddev"]
    )

    amp_min = float(params["amplitude"].min())
    noise_std = amp_min / 50.0
    hires += rng.normal(scale=noise_std, size=hires.shape)
    lowres += rng.normal(scale=noise_std, size=lowres.shape)

    threshold = noise_std * 5.0
    seg_detect = detect_sources(hires, threshold, npixels=5)
    segm = deblend_sources(
        hires,
        seg_detect,
        npixels=5,
        nlevels=32, 
        contrast=0.0001,
        progress_bar=False,
    )
    segdata = segm.data
    segmap = np.zeros_like(segdata, dtype=int)
    used = set()
    for idx, (y, x) in enumerate(zip(params["y_mean"], params["x_mean"]), start=1):
        iy = int(round(y))
        ix = int(round(x))
        if iy < 0 or iy >= ny or ix < 0 or ix >= nx:
            continue
        label = segdata[iy, ix]
        if label != 0 and label not in used:
            segmap[segdata == label] = idx
            used.add(label)
        else:
            segmap[iy, ix] = idx

    catalog = Table({
        "y": params["y_mean"],
        "x": params["x_mean"],
        "flux_true": flux_true,
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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (0, 0, truth,   "truth",   "gray",   lupton_norm(truth)),
        (0, 1, hires,   "hires",   "gray",   lupton_norm(hires)),
        (0, 2, segmap,  "segmap",  "nipy_spectral", None),
        (1, 0, lowres,  "lowres",  "gray",   lupton_norm(lowres)),
        (1, 1, model,   "model",   "gray",   lupton_norm(model)),
        (1, 2, residual,"residual","gray",   None),
    ]
    for row, col, img, title, cmap, norm in panels:
        ax = axes[row, col]
        if img is None:
            ax.axis("off")
            continue
        if title == "residual":
            std = residual.std()
            vlim = 5 * std
            ax.imshow(img, cmap=cmap, origin="lower", vmin=-vlim, vmax=vlim)
        elif norm is not None:
            ax.imshow(img, cmap=cmap, origin="lower", norm=norm)
        else:
            ax.imshow(img, cmap=cmap, origin="lower")
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
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    std = residual.std()
    vlim = 5 * std
    data = [image, model, residual]
    titles = ["image", "model", "residual"]
    for ax, img, title in zip(axes, data, titles):
        if title == "residual":
            ax.imshow(img, cmap="gray", origin="lower", vmin=-vlim, vmax=vlim)
        else:
            ax.imshow(img, cmap="gray", origin="lower", norm=lupton_norm(img))
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
    n = len(templates)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    axes = np.atleast_1d(axes)
    axes[0].imshow(hires, cmap="gray", origin="lower", norm=lupton_norm(hires))
    axes[0].set_title("hires")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i, tmpl in enumerate(templates, start=1):
        axes[i].imshow(tmpl.array, cmap="gray", origin="lower", norm=lupton_norm(tmpl.array))
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
    xlabel: str = "True Flux",
    ylabel: str = "Recovered Flux",
) -> None:
    """Plot recovered flux vs truth and recovered/true vs truth, save to file."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Recovered vs True
    axes[0].scatter(truth, recovered, s=30)
    minval = min(truth.min(), recovered.min())
    maxval = max(truth.max(), recovered.max())
    axes[0].plot([minval, maxval], [minval, maxval], "k--", label="y=x")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(label)
    axes[0].legend()

    # Panel 2: Ratio vs True
    ratio = np.array(recovered) / np.array(truth)
    axes[1].scatter(truth, ratio, s=30)
    axes[1].axhline(1.0, color="k", linestyle="--", label="ratio=1")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Recovered / True")
    axes[1].set_title("Flux Ratio vs True")
    axes[1].set_ylim(0.8, 1.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
