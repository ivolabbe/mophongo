import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize

from mophongo.psf import PSF
from mophongo.templates import _convolve2d, Template
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian2D
from photutils.segmentation import detect_sources, deblend_sources
from photutils.datasets import make_model_image, make_model_params
from skimage.morphology import dilation, disk

def lupton_norm(img):
    p = np.percentile(img, [1,99])
    vmin, vmax = -p[1]/20, p[1]
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.01))

def safe_dilate_segmentation(segmap, selem=disk(1)):
    result = np.zeros_like(segmap)
    for seg_id in np.unique(segmap):
        if seg_id == 0:
            continue  # skip background
        mask = segmap == seg_id
        dilated = dilation(mask, selem)
        # Only allow dilation into background
        dilated = np.logical_and(dilated, segmap == 0)
        result[dilated] = seg_id
        result[mask] = seg_id  # retain original
    return result

def make_simple_data(
    seed: int = 1,
    nsrc: int = 300,
    size: int = 501,     
    det_fwhm: float = 0,
    sigthresh: float = 2.0,
    ndilate: int = 2,
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
        min_separation=int(hi_fwhm * 4),
        border_size=psf_lo.array.shape[0] // 4,
        seed=rng,
        amplitude=(1.0, 100),
        x_stddev=(0.5, 5.0),
        y_stddev=(0.5, 5.0),
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
    noise_std = amp_min / 2.0
    hires += rng.normal(scale=noise_std, size=hires.shape)
    lowres += rng.normal(scale=noise_std, size=lowres.shape)
    rms_hi = np.ones_like(hires) * noise_std
    rms_lo = np.ones_like(lowres) * noise_std

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
        detimg, seg, npixels=5, nlevels=64, contrast=0.000001, progress_bar=False,
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
        [rms_hi, rms_lo],
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
        axes[0, i].imshow(tmpl.array, cmap="gray", origin="lower", norm=lupton_norm(tmpl.array))
        axes[0, i].set_title(f"tmpl {i+1}")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

    # Row 2: Convolved templates
    for i, tmpl in enumerate(templates_conv):
        axes[1, i].imshow(tmpl.array, cmap="gray", origin="lower", norm=lupton_norm(tmpl.array))
        axes[1, i].set_title(f"conv {i+1}")
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

    # Row 3: Segmentation map with labels for each template's bbox
    for i, tmpl in enumerate(templates):
        y0, y1, x0, x1 = tmpl.bbox
        seg = segmap[y0:y1, x0:x1]
        axes[2, i].imshow(seg, cmap="nipy_spectral", origin="lower")
        if catalog is not None:
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

    # Panel 1 (top-left): Recovered vs True
    axes[0, 0].scatter(truth, recovered, s=30, alpha=0.6)
    minval = min(truth.min(), recovered.min())
    maxval = max(truth.max(), recovered.max())
    axes[0, 0].plot([minval, maxval], [minval, maxval], "k--", label="y=x")
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_title(label)
    axes[0, 0].legend()

    # Panel 2 (top-right): Ratio vs True with binned statistics and error envelope
    ratio = np.array(recovered) / np.array(truth)
    axes[0, 1].scatter(truth, ratio, s=30, alpha=0.6, label='Data')
    axes[0, 1].axhline(1.0, color="k", linestyle="--", label="ratio=1")
    
    # Add error envelope if provided
    if error is not None:
        # Calculate relative error: e_tot = error / recovered
        e_tot = np.array(error) / np.array(recovered)
        
        # Plot upper and lower envelope: 1 ± e_tot
        axes[0, 1].scatter(truth, 1 + e_tot, s=5, alpha=0.5, color='orange', label='+/- 1σ')
        axes[0, 1].scatter(truth, 1 - e_tot, s=5, alpha=0.5, color='orange')
    
    # Calculate binned statistics
    # Create logarithmic bins for better coverage
    truth_log = np.log10(truth)
    n_bins = 8
    # make in linear space 
    bin_edges = np.percentile(truth, np.linspace(0, 100, n_bins + 1))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    bin_medians = []
    bin_mad_stds = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (truth >= bin_edges[i]) & (truth < bin_edges[i+1])
        if np.sum(mask) > 2:  # Need at least 3 points for meaningful statistics
            bin_ratio = ratio[mask]
            bin_medians.append(np.median(bin_ratio))
            bin_mad_stds.append(mad_std(bin_ratio))  # Use astropy's mad_std
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
                           fmt='ko', capsize=5, capthick=2, alpha=0.7,
                           label=f'Binned median ± MAD', markersize=6)
    
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel("Recovered / True")
    axes[0, 1].set_title("Flux Ratio vs True")
    axes[0, 1].set_ylim(0.7, 1.3)
    axes[0, 1].set_xscale('symlog', linthresh=2000)
    axes[0, 1].set_xlim(truth.min(), truth.max())
    axes[0, 1].legend()

    # Panel 3 (bottom-left): Histogram of Residuals / Error with Gaussian fit
    if error is not None:
        residuals_over_error = (recovered - truth) / error
        
        # Create histogram
        bins = np.linspace(-5, 5, 31)  # 30 bins from -5 to 5 sigma
        counts, bin_edges, _ = axes[1, 0].hist(residuals_over_error, bins=bins, alpha=0.7, density=True, 
                                              color='lightblue', edgecolor='black', linewidth=0.5)
        
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
        
        # Add vertical lines for ±1, ±2, ±3 sigma
        for sigma in [1, 2, 3]:
            axes[1, 0].axvline(sigma, color='gray', linestyle='--', alpha=0.5)
            axes[1, 0].axvline(-sigma, color='gray', linestyle='--', alpha=0.5)
        
        # Calculate statistics using astropy's mad_std
        mean_resid = np.mean(residuals_over_error)
        median_resid = np.median(residuals_over_error)
        std_resid = np.std(residuals_over_error)
        mad_resid = mad_std(residuals_over_error)  # Gaussian-scaled MAD
        
        # Add statistics text
        stats_text = f'Mean: {mean_resid:.3f}\nMedian: {median_resid:.3f}\nStd: {std_resid:.3f}\nMAD: {mad_resid:.3f}'
        if 'mu_fit' in locals():
            stats_text += f'\nFit μ: {mu_fit:.3f}\nFit σ: {sigma_fit:.3f}'
        
        axes[1, 0].text(0.05, 0.95, stats_text, 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[1, 0].set_xlabel("(Recovered - True) / Error")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Residuals / Error Distribution")
        axes[1, 0].set_xlim(-5, 5)
        axes[1, 0].legend()

        # Panel 4 (bottom-right): (Recovered - True) / Error vs Recovered
        axes[1, 1].scatter(recovered, residuals_over_error, s=30, alpha=0.6)
        axes[1, 1].axhline(0, color="k", linestyle="--", label="zero residual")
        
        # Add ±1, ±2, ±3 sigma lines
        for sigma in [1, 3]:
            axes[1, 1].axhline(sigma, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].axhline(-sigma, color='gray', linestyle='--', alpha=0.5, label=f'±{sigma}σ')
        
        axes[1, 1].set_xlabel("Recovered Flux")
        axes[1, 1].set_ylabel("(Recovered - True) / Error")
        axes[1, 1].set_title("Residuals vs Recovered Flux")
        axes[1, 1].set_ylim(-10, 10)
        axes[1, 1].set_xscale('symlog', linthresh=2000)
        axes[1, 1].set_xlim(recovered.min(), recovered.max())
        axes[1, 1].legend()
    else:
        # If no error provided, hide panels 3 and 4
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def save_psf_fit_diagnostic(filename: str, psf: np.ndarray, model: np.ndarray) -> None:
    """Save diagnostic image comparing PSF data with a fitted model."""
    resid = psf - model
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    images = [psf, model, resid]
    titles = ["psf", "model", "resid"]
    for ax, img, title in zip(axes, images, titles):
        if title == "resid":
            v = np.max(np.abs(resid))
            ax.imshow(img, cmap="gray", origin="lower", vmin=-v, vmax=v)
        else:
            ax.imshow(img, cmap="gray", origin="lower", norm=lupton_norm(psf))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def label_segmap(ax, segmap, catalog):
    for idx, (y, x) in enumerate(zip(catalog["y"], catalog["x"]), start=1):
        ax.text(x, y, str(idx), color="white", fontsize=10, ha="center", va="center", weight="bold")

