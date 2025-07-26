import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize

from mophongo.psf import PSF
from mophongo.templates import _convolve2d, Template
from mophongo.catalog import safe_dilate_segmentation

import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian2D
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
#from mophongo.photutils_deblend import deblend_sources
from photutils.datasets import make_model_image, make_model_params
from skimage.morphology import dilation, disk, max_tree

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from astropy.coordinates import SkyCoord


def lupton_norm(img):
    p = np.percentile(img, [1,99])
    vmin, vmax = -p[1]/20, p[1]
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.01))


def make_simple_data(
    seed: int = 11,
    nsrc: int = 100,
    size: int = 201,
    det_fwhm: float = 0,
    sigthresh: float = 2.0,
    peak_snr: float = 1.0,
    ndilate: int = 2,
    border_size: int = 10,
) -> tuple[
    list[np.ndarray], np.ndarray, Table, list[np.ndarray], np.ndarray, list[np.ndarray]
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

    nx = ny = size
    hi_fwhm = 2.0
    lo_fwhm = 5.0 * hi_fwhm

    # Use Moffat PSFs instead of Gaussian
    psf_hi = PSF.moffat(41, hi_fwhm, hi_fwhm, beta=3.0)  # Typical ground-based seeing
    # delta function
    #    psf_hi = PSF.gaussian(5,0.1,0.1).array.round()

    psf_lo = PSF.moffat(41, lo_fwhm, lo_fwhm, beta=2.5)  # Broader wings for low-res

    params = make_model_params(
        (ny, nx),
        nsrc,
        x_name="x_mean",
        y_name="y_mean",
        min_separation=int(hi_fwhm * 6),
        border_size=border_size,
        seed=rng,
        amplitude=(1.0, 100),
        x_stddev=(0.5, 4.0),
        y_stddev=(0.5, 4.0),
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
    # Add noise
    amp_min = float(params["amplitude"].min())
    noise_std = amp_min / peak_snr
    hires += rng.normal(scale=noise_std, size=hires.shape)
    lowres += rng.normal(scale=noise_std, size=lowres.shape)
    wht_hi = np.ones_like(hires) * 1.0/noise_std**2
    wht_lo = np.ones_like(lowres) * 1.0/noise_std**2

    # Segmentation map from hires image
    # Use Gaussian PSF for detection (keeping this as Gaussian for now)
    if det_fwhm>0:
        psf_det = PSF.gaussian(3,0.01,0.01)   # delta function = no smoothing
        detimg = _convolve2d(hires, psf_det.array)
    else:
        detimg = hires

    seg = detect_sources(detimg, threshold=sigthresh * noise_std, npixels=5)
    if ndilate > 0:
        seg.data = safe_dilate_segmentation(seg.data, selem=disk(ndilate))

    segm = deblend_sources(
        detimg,
        seg,
        npixels=5,
        nlevels=64,
        contrast=0.000001,
        progress_bar=False,
    #        compactness=0.0,
    )
    segdata = segm.data
    segmap = np.zeros_like(segdata, dtype=int)
    used = set()
    for (idx, y, x) in zip(params["id"], params["y_mean"], params["x_mean"]):
        iy = int(round(y))
        ix = int(round(x))
        if iy < 0 or iy >= ny or ix < 0 or ix >= nx:
            continue
        label = segdata[iy, ix]
        if label != 0 and label not in used:
            segmap[segdata == label] = idx
            used.add(label)
        else:
            # Create a 3x3 pixel segment for undetected sources
            y_min = max(0, iy - 1)
            y_max = min(ny, iy + 2)  # +2 because range is exclusive
            x_min = max(0, ix - 1)
            x_max = min(nx, ix + 2)
            segmap[y_min:y_max, x_min:x_max] = idx

    catalog = Table({
        "id": params["id"],
        "y": params["y_mean"],
        "x": params["x_mean"],
        "flux_true": flux_true,
    })

    return (
        [hires, lowres],
        segmap,
        catalog,
        [psf_hi.array, psf_lo.array],
        truth,
        [wht_hi, wht_lo],
    )


def save_diagnostic_image(
    filename: str,
    truth: np.ndarray,
    hires: np.ndarray,
    lowres: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    segmap: np.ndarray = None,
    catalog: Table = None,
    fitter=None,
) -> None:
    # Compute covariance matrix if fitter is provided
    if fitter is not None:
        # Get the full covariance matrix (inverse of ATA)
        try:
            from scipy.sparse.linalg import spsolve
            from scipy.sparse import eye
            n = fitter.ata.shape[0]
            cov_matrix = spsolve(fitter.ata, eye(n).tocsc()).toarray()
        except:
            # Fallback: convert to dense and use numpy
            ata_dense = fitter.ata.toarray()
            try:
                cov_matrix = np.linalg.inv(ata_dense)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                cov_matrix = np.linalg.pinv(ata_dense)

        # Use the actual covariance matrix
        cov_img = cov_matrix
    else:
        # Create a dummy matrix if no fitter provided
        n_sources = len(np.unique(segmap[segmap > 0])) if segmap is not None else 5
        cov_img = np.eye(n_sources) * 0.1  # Identity matrix with small values

    panels = [
        (0, 0, truth,   "truth",   "gray",   lupton_norm(truth)),
        (0, 1, hires,   "hires",   "gray",   lupton_norm(hires)),
        (0, 2, segmap,  "segmap",  "nipy_spectral", None),
        (1, 0, lowres,  "lowres",  "gray",   lupton_norm(lowres)),
        (1, 1, model,   "model",   "gray",   lupton_norm(model)),
        (1, 2, residual,"residual","gray",   None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, col, img, title, cmap, norm in panels:
        ax = axes[row, col]
        if img is None:
            ax.axis("off")
            continue
        if title == "residual":
            std = residual.std()
            vlim = 5 * std
            ax.imshow(img, cmap=cmap, origin="lower", vmin=-vlim, vmax=vlim)
        elif title == "segmap":
            ax.imshow(img, cmap=cmap, origin="lower")
            label_segmap(ax, segmap, catalog)
        elif norm is not None:
            ax.imshow(img, cmap=cmap, origin="lower", norm=norm)
        else:
            ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Covariance panel with proper scaling
    ax_cov = axes[1, 3]
    # Use symmetric colormap for covariance (can have positive and negative values)
    vmax = np.abs(cov_img).max()
    if vmax > 0:
        im = ax_cov.imshow(cov_img, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    else:
        im = ax_cov.imshow(cov_img, cmap="gray", origin="lower")

    ax_cov.set_title(f"Covariance Matrix ({cov_img.shape[0]}×{cov_img.shape[1]})")
    ax_cov.set_xlabel("Source Index")
    ax_cov.set_ylabel("Source Index")

    # Add colorbar with better formatting
    cbar = fig.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance", rotation=270, labelpad=15)

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
            ax.imshow(img, cmap="gray", origin="lower", vmin=-0.01*psf_lo.max(), vmax=0.01*psf_lo.max())
        else:
            ax.imshow(img, cmap="gray", origin="lower", norm=lupton_norm(img))

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
    templates: list,
    templates_conv: list,
    segmap: np.ndarray,
    catalog=None,
) -> None:
    n = len(templates)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    axes = np.atleast_2d(axes)

    # Row 1: Original templates
    for i, tmpl in enumerate(templates):
        axes[0, i].imshow(tmpl.data, cmap="gray", origin="lower", norm=lupton_norm(tmpl.data))
        axes[0, i].set_title(f"tmpl {i+1}")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

    # Row 2: Convolved templates
    for i, tmpl in enumerate(templates_conv):
        axes[1, i].imshow(tmpl.data, cmap="gray", origin="lower", norm=lupton_norm(tmpl.data))
        axes[1, i].set_title(f"conv {i+1}")
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

    # Row 3: Segmentation map with labels for each template's bbox
    for i, tmpl in enumerate(templates):
        y0, y1, x0, x1 = tmpl.bbox
        seg = segmap[y0:y1, x0:x1]
        axes[2, i].imshow(seg, cmap="nipy_spectral", origin="lower")
        if catalog is not None and len(templates) < 50:
            # Find the catalog index corresponding to this template
            idx = i + 1  # assuming 1-based segmap labels
            # Find the center of the bbox for label placement
            cy = (y1 - y0) // 2
            cx = (x1 - x0) // 2
            axes[2, i].text(cx, cy, str(idx), color="white", fontsize=12, ha="center", va="center", weight="bold")
        axes[2, i].set_title(f"seg {i+1}")
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_flux_vs_truth_plot(
    filename: str,
    truth: np.ndarray,
    recovered: np.ndarray,
    error: np.ndarray = None,
    label: str = "Recovered Flux",
    xlabel: str = "True Flux",
    ylabel: str = "Recovered Flux",
) -> None:
    """Plot recovered flux vs truth and recovered/true vs truth, save to file."""
    import matplotlib.pyplot as plt
    from scipy.stats import median_abs_deviation
    from scipy.optimize import curve_fit
    from astropy.stats import mad_std

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    #    print('TRUTH min max', truth.min(), truth.max())
    # Panel 1 (top-left): Recovered vs True
    axes[0, 0].scatter(truth, recovered, s=20, alpha=0.4)
    minval = min(truth.min(), recovered.min())
    maxval = max(truth.max(), recovered.max())
    axes[0, 0].plot([minval, maxval], [minval, maxval], "k--", label="y=x")
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_title(label)
    axes[0, 0].legend()

    # Panel 2 (top-right): Ratio vs True with fitted error envelope
    ratio = np.array(recovered) / np.array(truth)
    axes[0, 1].scatter(truth, ratio, s=20, alpha=0.4, label='Data')
    axes[0, 1].axhline(1.0, color="k", linestyle="--", label="ratio=1")

    # Fit flux vs error relationship if errors provided
    if error is not None:
        # Fit error as function of recovered flux: error = a * flux^b + c
        def error_model(flux, a, b, c):
            return a * flux**b + c

        # Filter out invalid data for fitting
        valid_mask = (recovered > 0) & (error > 0) & np.isfinite(recovered) & np.isfinite(error)
        if np.sum(valid_mask) > 10:  # Need enough points for fitting
            try:
                # Initial guess: linear relationship with small offset
                p0 = [error[valid_mask].mean() / recovered[valid_mask].mean(), 1.0, error[valid_mask].min()]
                popt, _ = curve_fit(error_model, recovered[valid_mask], error[valid_mask],
                                  p0=p0, maxfev=1000)
                a_fit, b_fit, c_fit = popt

                # Calculate fitted errors for all flux values
                error_fit = error_model(recovered, a_fit, b_fit, c_fit)

                # Calculate fitted errors for true flux values
                error_fit_true = error_model(truth, a_fit, b_fit, c_fit)

                # Calculate relative error envelope: 1 ± error_fit/truth
                rel_error_fit = error_fit_true / truth

                # Plot fitted error envelope
                flux_sorted_idx = np.argsort(truth)
                truth_sorted = truth[flux_sorted_idx]
                upper_envelope = (1 + rel_error_fit)[flux_sorted_idx]
                lower_envelope = (1 - rel_error_fit)[flux_sorted_idx]

                axes[0, 1].plot(truth_sorted, upper_envelope, 'orange', linewidth=2,
                               label=f'Fitted ±1σ envelope\nσ = {a_fit:.2e}×F^{b_fit:.2f}+{c_fit:.2e}')
                axes[0, 1].plot(truth_sorted, lower_envelope, 'orange', linewidth=2)

            except Exception as e:
                print(f"Error fitting flux-error relationship: {e}")
                # Fallback to simple error envelope
                e_tot = np.array(error) / np.array(recovered)
                axes[0, 1].scatter(truth, 1 + e_tot, s=5, alpha=0.4, color='orange', label='+/- 1σ')
                axes[0, 1].scatter(truth, 1 - e_tot, s=5, alpha=0.4, color='orange')
                error_fit = error  # Use original errors for SNR calculation
        else:
            # Fallback to simple error envelope
            e_tot = np.array(error) / np.array(recovered)
            axes[0, 1].scatter(truth, 1 + e_tot, s=5, alpha=0.4, color='orange', label='+/- 1σ')
            axes[0, 1].scatter(truth, 1 - e_tot, s=5, alpha=0.4, color='orange')
            error_fit = error

    # Calculate binned statistics
    truth_log = np.log10(truth)
    n_bins = 8
    bin_edges = np.percentile(truth, np.linspace(0, 100, n_bins + 1))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    bin_medians = []
    bin_mad_stds = []
    bin_counts = []

    for i in range(n_bins):
        mask = (truth >= bin_edges[i]) & (truth < bin_edges[i+1])
        if np.sum(mask) > 2:
            bin_ratio = ratio[mask]
            bin_medians.append(np.median(bin_ratio))
            bin_mad_stds.append(mad_std(bin_ratio))
            bin_counts.append(np.sum(mask))
        else:
            bin_medians.append(np.nan)
            bin_mad_stds.append(np.nan)
            bin_counts.append(0)

    # Plot binned statistics
    valid_bins = ~np.isnan(bin_medians)
    if np.any(valid_bins):
        axes[0, 1].errorbar(bin_centers[valid_bins],
                           np.array(bin_medians)[valid_bins],
                           yerr=np.array(bin_mad_stds)[valid_bins],
                           fmt='ko', capsize=5, capthick=2, alpha=0.5,
                           label=f'Binned median ± MAD', markersize=6)

    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel("Recovered / True")
    axes[0, 1].set_title("Flux Ratio vs True")
    axes[0, 1].set_ylim(0.7, 1.3)
    axes[0, 1].set_xscale('function', functions=(np.sqrt, lambda x: x**2))  # Square root scaling
    axes[0, 1].set_xlim(truth.min(), truth.max())
    axes[0, 1].legend()

    # Add SNR axis if errors are available
    if error is not None:
        ax2 = axes[0, 1].twiny()

        # Create SNR tick positions based on true flux values
        snr_ticks = [1, 3, 5, 10, 20, 50, 100]
        flux_ticks = []

        for snr in snr_ticks:
            if 'error_fit_true' in locals():
                # For each SNR level, find the flux where SNR = flux/error = snr
                test_fluxes = np.logspace(np.log10( max(truth.min(),1e-4) ), np.log10(truth.max()), 1000)
                test_errors = error_model(test_fluxes, a_fit, b_fit, c_fit)
                test_snr = test_fluxes / test_errors

                # Find flux closest to target SNR
                closest_idx = np.argmin(np.abs(test_snr - snr))
                flux_ticks.append(test_fluxes[closest_idx])
            else:
                # Use original error relationship
                snr_values = truth / error
                closest_idx = np.argmin(np.abs(snr_values - snr))
                flux_ticks.append(truth[closest_idx])

        # Filter ticks within plot range and apply sqrt scaling
        xlim = axes[0, 1].get_xlim()
        valid_ticks = [(flux, snr) for flux, snr in zip(flux_ticks, snr_ticks)
                      if xlim[0] <= flux <= xlim[1]]

        if valid_ticks:
            flux_tick_pos, snr_tick_labels = zip(*valid_ticks)

            # Apply square root transformation to match the bottom axis scaling
            flux_tick_pos_sqrt = [np.sqrt(flux) for flux in flux_tick_pos]

            # Set the SNR axis limits to match the transformed bottom axis
            ax2.set_xlim([np.sqrt(xlim[0]), np.sqrt(xlim[1])])
            ax2.set_xticks(flux_tick_pos_sqrt)
            ax2.set_xticklabels([f'{snr:.0f}' for snr in snr_tick_labels])
            ax2.set_xlabel('SNR (True Flux / Fitted Error)')

    # Panel 3 (bottom-left): Histogram of Residuals / Error with Gaussian fit
    if error is not None:
        residuals_over_error = (recovered - truth) / error

        # Create histogram
        bins = np.linspace(-5, 5, 31)
        counts, bin_edges, _ = axes[1, 0].hist(residuals_over_error, bins=bins, alpha=0.5, density=True,
                                              color='lightblue', edgecolor='black', linewidth=0.5)

        # Add zero residual line
        axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='zero residual')

        # Fit Gaussian to the data
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Initial guess
        x_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        p0 = [counts.max(), np.mean(residuals_over_error), np.std(residuals_over_error)]

        try:
            popt, _ = curve_fit(gaussian, x_centers, counts, p0=p0)
            amp_fit, mu_fit, sigma_fit = popt

            # Plot fitted Gaussian
            x_fit = np.linspace(-10, 10, 100)
            y_fit = gaussian(x_fit, amp_fit, mu_fit, sigma_fit)
            axes[1, 0].plot(x_fit, y_fit, 'g-', linewidth=2,
                           label=f'Fitted Gaussian\nμ={mu_fit:.3f}, σ={sigma_fit:.3f}')
        except:
            mu_fit, sigma_fit = np.mean(residuals_over_error), np.std(residuals_over_error)

        # Add vertical lines for ±1, ±3 sigma
        for sigma in [1, 3]:
            axes[1, 0].axvline(sigma, color='gray', linestyle='--', alpha=0.5)
            axes[1, 0].axvline(-sigma, color='gray', linestyle='--', alpha=0.5)

        # Calculate statistics
        mean_resid = np.mean(residuals_over_error)
        median_resid = np.median(residuals_over_error)
        std_resid = np.std(residuals_over_error)
        mad_resid = mad_std(residuals_over_error)

        # Add statistics text
        stats_text = f'Mean: {mean_resid:.3f}\nMedian: {median_resid:.3f}\nStd: {std_resid:.3f}\nMAD: {mad_resid:.3f}'
        if 'mu_fit' in locals():
            stats_text += f'\nFit μ: {mu_fit:.3f}\nFit σ: {sigma_fit:.3f}'

        axes[1, 0].text(0.05, 0.95, stats_text,
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        axes[1, 0].set_xlabel("(Recovered - True) / Error")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Residuals / Error Distribution")
        axes[1, 0].set_xlim(-10, 10)
        axes[1, 0].legend()

        # Panel 4 (bottom-right): (Recovered - True) / Error vs Recovered
        axes[1, 1].scatter(recovered, residuals_over_error, s=20, alpha=0.4)
        axes[1, 1].axhline(0, color="k", linestyle="--", label="zero residual")

        # Add ±1, ±3 sigma lines
        for sigma in [1, 3]:
            axes[1, 1].axhline(sigma, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].axhline(-sigma, color='gray', linestyle='--', alpha=0.5, label=f'±{sigma}σ')

        axes[1, 1].set_xlabel("Recovered Flux")
        axes[1, 1].set_ylabel("(Recovered - True) / Error")
        axes[1, 1].set_title("Residuals vs Recovered Flux")
        axes[1, 1].set_ylim(-10, 10)
        axes[1, 1].set_xscale('function', functions=(np.sqrt, lambda x: x**2))  # Square root scaling
        axes[1, 1].set_xlim(truth.min(), truth.max())
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def save_psf_fit_diagnostic(filename: str, psf: np.ndarray, model: np.ndarray) -> None:
    """Save diagnostic image comparing PSF data with a fitted model."""
    resid = psf - model
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    images = [psf, model, resid]
    titles = ["psf", "model", "resid"]
    lnorm = lupton_norm(psf)
    for ax, img, title in zip(axes, images, titles):
        #        if title == "resid":
        #            v = np.max(np.abs(resid))
        #           ax.imshow(img, cmap="gray", origin="lower", vmin=-v, vmax=v)
        #      else:
        ax.imshow(img, cmap="gray", origin="lower", norm=lnorm)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def label_segmap(ax, segmap, catalog, fontsize=10):
    for idx, (y, x) in enumerate(zip(catalog["y"], catalog["x"]), start=1):
        ax.text(x, y, str(idx), color="white", fontsize=fontsize, ha="center", va="center", weight="medium", alpha=0.7)

#%%
def make_testdata():
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.table import Table
    from reproject import reproject_interp
    import numpy as np
    """Create cutouts for UDS images based on user-defined parameters."""
    indir = '/Users/ivo/Astro/PROJECTS/MINERVA/data/v1.0/'
    outdir = '/Users/ivo/Astro/PROJECTS/MINERVA/data/v1.0/testdata/'

    # --- User parameters from the pasted image ---
    center_x_40mas = 23243
    center_y_40mas = 19388
    #    size_x_40mas = 1144
    #    size_y_40mas = 878
    size_x_40mas = 4144
    size_y_40mas = 2978

    center = (center_x_40mas, center_y_40mas)
    size = (size_y_40mas, size_x_40mas)  # Cutout2D expects (ny, nx)

    # --- File lists ---
    files_40mas = [
        ("uds-grizli-v8.0-minerva-v1.0-40mas-f444w-clear_drc_sci_skysubvar.fits",
         "uds-test-f444w_sci.fits"),
        ("uds-grizli-v8.0-minerva-v1.0-40mas-f444w-clear_drc_wht.fits",
         "uds-test-f444w_wht.fits"),
        ("uds-grizli-v8.0-minerva-v1.0-40mas-f115w-clear_drc_sci_skysubvar.fits",
         "uds-test-f115w_sci.fits"),
        ("uds-grizli-v8.0-minerva-v1.0-40mas-f115w-clear_drc_wht.fits",
         "uds-test-f115w_wht.fits"),
        ("LW_f277w-f356w-f444w_SEGMAP.fits", "uds-test-LW_seg.fits"),
        ("LW_f277w-f356w-f444w_opterr.fits", "uds-test-f444w_opterr.fits"),
        ("LW_f277w-f356w-f444w_optavg.fits", "uds-test-f444w_optavg.fits"),
    ]

    # --- Extract cutouts for 40mas images ---
    for infile, outfile in files_40mas:
        with fits.open(indir + infile) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
            wcs = WCS(hdr)
            center_radec = SkyCoord(*wcs.all_pix2world(center_x_40mas,
                                                       center_y_40mas, 0),
                                    unit='deg')
            cutout = Cutout2D(data, position=center_radec, size=size, wcs=wcs)
            hdr.update(cutout.wcs.to_header())
            hdu = fits.PrimaryHDU(cutout.data, header=hdr)
            hdu.writeto(outdir + outfile, overwrite=True)
            print(f"Saved {outdir+outfile}")

    # --- Extract cutout for 80mas image (2x2 binned), reproject to 40mas grid ---
    # THESE ARE 160mas !!


#    files_80mas = [
#        ("uds-lowres-all-f770w_drz_sci.fits", "uds-test-f770w_sci.fits"),
#        ("uds-lowres-all-f770w_drz_wht.fits", "uds-test-f770w_wht.fits"),
#    ]

    files_80mas = [
        ('uds-sbkgsub-v0.3-80mas-f770w_drz_sci.fits',
         "uds-test-f770w_sci.fits"),
        ('uds-sbkgsub-v0.3-80mas-f770w_drz_wht.fits',
         "uds-test-f770w_wht.fits"),
    ]
    # Use the WCS and shape from one of the 40mas cutouts as the target
    ref_cutout_file = outdir + files_40mas[0][1]
    with fits.open(ref_cutout_file) as ref_hdul:
        target_header = ref_hdul[0].header
        target_wcs = WCS(target_header)
        target_shape = ref_hdul[0].data.shape

    for infile, outfile in files_80mas:
        with fits.open(indir + infile) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
            hdr['KERNEL'] = ('square', 'Drizzle kernel')
            hdr['PIXFRAC'] = (0.75, 'Drizzle pixfrac')

            # Extract the cutout at the 80mas scale
            size_80mas = (size[0] // 2, size[1] // 2)
            cutout_80mas = Cutout2D(data,
                                    position=center_radec,
                                    size=size_80mas,
                                    wcs=WCS(hdr))
            # Reproject to 40mas grid
            reprojected_data, _ = reproject_interp(
                (cutout_80mas.data, cutout_80mas.wcs),
                output_projection=target_wcs,
                shape_out=target_shape)
            # update hdr with new wcs
            hdr.update(target_wcs.to_header())
            hdu = fits.PrimaryHDU(reprojected_data.astype(np.float32),
                                  header=hdr)
            hdu.writeto(outdir + outfile, overwrite=True)
            print(f"Saved {outdir + outfile} (registered to 40mas grid)")

# %%
