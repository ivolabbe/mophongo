"""Three-scenario flux-recovery sanity check with Moffat PSFs matched to F444W/F770W.

Purpose: isolate matching-kernel / simulation effects from the diffraction
structure of the real ePSFs. The Moffat FWHM and growth curve are tuned to
match the NIRCam LW (F444W) and MIRI (F770W) ePSFs on the 40 mas mosaic grid:

    hires ("F444W-like" stress test): FWHM = 2.00 pix  (80 mas), beta = 1.86
    lores ("F770W-like"): FWHM = 6.01 pix  (240 mas), beta = 1.75

Growth-curve RMS between these Moffats and the real ePSFs is < 5% over
r < 1.5 arcsec. See /tmp/moffat_vs_epsf.png (produced by tune_moffat.py).

Scenarios mirror the Moffat talk plots in 25.minerva/mf_mof.pdf:
  1. "no_extension"   – raw detection-based segmap (ndilate=0)
  2. "dilated3"       – segmap dilated by disk(3)      (ndilate=3)
  3. "psf_wing"       – dilated segmap + extend_templates="psf"

Each scenario saves:
  * diagnostic.png          (truth / hires / segmap / lowres / model / residual)
  * flux_ratio_lowres.png   (recovered/true vs true flux; Moffat analogue of PDF)
"""

import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
from photutils.segmentation import detect_sources, deblend_sources
from skimage.morphology import disk

import mophongo.pipeline as pipeline
from mophongo.fit import FitConfig
from mophongo.psf import PSF
from mophongo.templates import _convolve2d
from mophongo.catalog import safe_dilate_segmentation
from mophongo.utils import fftconvolve
from utils import save_diagnostic_image, save_flux_vs_truth_plot

# Moffat params matched to F444W/F770W on the 40 mas mosaic grid (see docstring).
FWHM_HI = 2.0
BETA_HI = 1.86
FWHM_LO = 6.01
BETA_LO = 1.75
PSF_SIZE = 61  # PSF/kernel stamp size, independent of simulated image size.


def _oversampled_gaussian_stamp(
    *,
    shape: tuple[int, int],
    x_mean: float,
    y_mean: float,
    x_stddev: float,
    y_stddev: float,
    theta: float,
    oversample: int = 8,
    bbox_factor: float = 8.0,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Return a native-pixel unit-amplitude Gaussian stamp and image bbox."""
    if oversample < 2:
        raise ValueError("oversample must be >= 2")

    ny, nx = shape
    radius = int(np.ceil(bbox_factor * max(x_stddev, y_stddev))) + 2
    x_center = int(round(x_mean))
    y_center = int(round(y_mean))
    x0 = max(0, x_center - radius)
    x1 = min(nx, x_center + radius + 1)
    y0 = max(0, y_center - radius)
    y1 = min(ny, y_center + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 0), dtype=float), (y0, y1, x0, x1), 0.0

    width = x1 - x0
    height = y1 - y0
    xs = x0 - 0.5 + (np.arange(width * oversample) + 0.5) / oversample
    ys = y0 - 0.5 + (np.arange(height * oversample) + 0.5) / oversample
    xx, yy = np.meshgrid(xs, ys)
    unit = Gaussian2D(
        amplitude=1.0,
        x_mean=x_mean,
        y_mean=y_mean,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=theta,
    )(xx, yy)
    stamp = (
        unit.reshape(height, oversample, width, oversample).sum(axis=(1, 3))
        / oversample**2
    )
    return stamp, (y0, y1, x0, x1), float(stamp.sum())


def _make_moffat_data(
    seed: int = 5,
    nsrc: int = 150,
    size: int = 301,
    ndilate: int = 0,
    peak_snr: float = 50.0,
    border_size: int = 10,
    fixed_sigma: float | None = None,
    source_oversample: int = 8,
    source_snr_range: tuple[float, float] | None = (1.0, 1000.0),
    source_snr_distribution: str = "linear",
    min_hires_snr: float | None = None,
    min_hires_snr_mode: str = "flux",
    detect_kernel_size: float = 3.5,
    detect_threshold: float = 2.0,
    detect_npixels: int = 5,
    deblend_nlevels: int = 32,
    deblend_contrast: float = 1e-4,
) -> tuple[list[np.ndarray], np.ndarray, Table, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Build a Moffat-PSF synthetic dataset with F444W/F770W-like resolution.

    Returns (images, segmap, catalog, psfs, truth, weights) exactly like
    ``utils.make_simple_data``, so the existing diagnostic helpers work
    unchanged.
    """
    from photutils.datasets import make_model_params

    rng = np.random.default_rng(seed)
    nx = ny = size

    psf_hi = PSF.moffat(PSF_SIZE, FWHM_HI, FWHM_HI, beta=BETA_HI)
    psf_lo = PSF.moffat(PSF_SIZE, FWHM_LO, FWHM_LO, beta=BETA_LO)

    max_source_sigma = float(fixed_sigma) if fixed_sigma is not None else 8.0
    render_border = int(np.ceil(8.0 * max_source_sigma)) + 3
    params = make_model_params(
        (ny, nx),
        nsrc,
        x_name="x_mean",
        y_name="y_mean",
        min_separation=int(FWHM_HI * 6),
        border_size=max(border_size, render_border),
        seed=rng,
        amplitude=(1.0, 200.0),
        x_stddev=(1.0, 1.0),
        y_stddev=(1.0, 1.0),
        theta=(0, np.pi),
    )
    if fixed_sigma is None:
        # Draw intrinsic source sizes from a log-uniform distribution: most
        # sources are compact, but the validation still includes extended objects.
        params["x_stddev"] = np.exp(rng.uniform(np.log(0.4), np.log(8.0), len(params)))
        params["y_stddev"] = np.exp(rng.uniform(np.log(0.4), np.log(8.0), len(params)))
    else:
        params["x_stddev"] = float(fixed_sigma)
        params["y_stddev"] = float(fixed_sigma)
    if source_snr_range is None:
        requested_snr = np.asarray(params["amplitude"], dtype=float)
        noise_std = float(requested_snr.min()) / peak_snr
        flux_noise_eff = np.ones(len(params), dtype=float)
        flux_noise_eff_hi = np.ones(len(params), dtype=float)
        flux_draw = requested_snr.copy()
        snr_true = requested_snr.copy()
        snr_hi_true = requested_snr.copy()
    else:
        snr_min, snr_max = source_snr_range
        if snr_min <= 0 or snr_max <= snr_min:
            raise ValueError("source_snr_range must be positive and increasing")
        if min_hires_snr_mode not in {"flux", "noise"}:
            raise ValueError("min_hires_snr_mode must be either 'flux' or 'noise'")
        noise_std = 1.0
        if source_snr_distribution == "linear":
            requested_snr = np.linspace(float(snr_min), float(snr_max), len(params))
        elif source_snr_distribution in {"log", "log_uniform"}:
            requested_snr = np.exp(
                rng.uniform(np.log(float(snr_min)), np.log(float(snr_max)), len(params))
            )
        else:
            raise ValueError(
                "source_snr_distribution must be one of 'linear', 'log', or 'log_uniform'"
            )
        rng.shuffle(requested_snr)
        flux_noise_eff = np.empty(len(params), dtype=float)
        flux_noise_eff_hi = np.empty(len(params), dtype=float)
        flux_draw = np.empty(len(params), dtype=float)
        snr_true = np.empty(len(params), dtype=float)
        snr_hi_true = np.empty(len(params), dtype=float)
    truth = np.zeros((ny, nx), dtype=float)
    flux_true_sampled = np.empty(len(params), dtype=float)
    amplitudes = np.empty(len(params), dtype=float)
    for i, row in enumerate(params):
        stamp, bbox, stamp_sum = _oversampled_gaussian_stamp(
            shape=truth.shape,
            x_mean=float(row["x_mean"]),
            y_mean=float(row["y_mean"]),
            x_stddev=float(row["x_stddev"]),
            y_stddev=float(row["y_stddev"]),
            theta=float(row["theta"]),
            oversample=source_oversample,
        )
        if stamp_sum <= 0:
            flux_true_sampled[i] = 0.0
            amplitudes[i] = 0.0
            flux_noise_eff[i] = np.inf
            flux_noise_eff_hi[i] = np.inf
            snr_true[i] = 0.0
            snr_hi_true[i] = 0.0
            continue

        unit_stamp = stamp / stamp_sum
        if source_snr_range is not None:
            unit_hires_profile = fftconvolve(unit_stamp, psf_hi.array, mode="full")
            unit_lowres_profile = fftconvolve(unit_stamp, psf_lo.array, mode="full")
            profile_norm2_hi = float(np.sum(unit_hires_profile**2))
            profile_norm2_lo = float(np.sum(unit_lowres_profile**2))
            flux_noise_eff_hi[i] = noise_std / np.sqrt(profile_norm2_hi)
            flux_noise_eff[i] = noise_std / np.sqrt(profile_norm2_lo)
            flux_draw[i] = requested_snr[i] * flux_noise_eff[i]
            if min_hires_snr is not None and min_hires_snr_mode == "flux":
                flux_draw[i] = max(flux_draw[i], float(min_hires_snr) * flux_noise_eff_hi[i])
            snr_true[i] = flux_draw[i] / flux_noise_eff[i]
            snr_hi_true[i] = flux_draw[i] / flux_noise_eff_hi[i]

        y0, y1, x0, x1 = bbox
        truth[y0:y1, x0:x1] += unit_stamp * flux_draw[i]
        flux_true_sampled[i] = float(np.sum(unit_stamp * flux_draw[i]))
        amplitudes[i] = float(flux_draw[i]) / (
            2.0 * np.pi * float(row["x_stddev"]) * float(row["y_stddev"])
        )
    params["amplitude"] = amplitudes

    hires = _convolve2d(truth, psf_hi.array)
    lowres = _convolve2d(truth, psf_lo.array)

    flux_true_analytic = (
        params["amplitude"] * 2 * np.pi * params["x_stddev"] * params["y_stddev"]
    )

    noise_std_hi = noise_std
    if (
        source_snr_range is not None
        and min_hires_snr is not None
        and min_hires_snr_mode == "noise"
    ):
        finite = np.isfinite(snr_hi_true) & (snr_hi_true > 0)
        if np.any(finite):
            scale = min(1.0, float(np.nanmin(snr_hi_true[finite])) / float(min_hires_snr))
            noise_std_hi = noise_std * scale
            flux_noise_eff_hi *= scale
            snr_hi_true = flux_draw / flux_noise_eff_hi

    hires += rng.normal(scale=noise_std_hi, size=hires.shape)
    lowres += rng.normal(scale=noise_std, size=lowres.shape)
    wht_hi = np.ones_like(hires) / noise_std_hi**2
    wht_lo = np.ones_like(lowres) / noise_std**2

    det_img = hires / noise_std
    kernel_pix = int(2 * detect_kernel_size) | 1
    kernel = Gaussian2DKernel(
        detect_kernel_size / 2.355,
        x_size=kernel_pix,
        y_size=kernel_pix,
    )
    smooth = fftconvolve(det_img, kernel.array, mode="same")
    seg = detect_sources(smooth, threshold=detect_threshold, npixels=detect_npixels)
    if seg is None:
        raise RuntimeError("Moffat mock detection found no sources")
    if ndilate > 0:
        seg.data = safe_dilate_segmentation(seg, selem=disk(ndilate))
    segm = deblend_sources(
        det_img,
        seg,
        npixels=detect_npixels,
        mode="exponential",
        nlevels=deblend_nlevels,
        contrast=deblend_contrast,
        connectivity=8,
        progress_bar=False,
    )
    segdata = segm.data
    segmap = np.zeros_like(segdata, dtype=int)
    used: set[int] = set()
    for idx, y, x in zip(params["id"], params["y_mean"], params["x_mean"]):
        iy, ix = int(round(y)), int(round(x))
        if not (0 <= iy < ny and 0 <= ix < nx):
            continue
        label = segdata[iy, ix]
        if label != 0 and label not in used:
            segmap[segdata == label] = idx
            used.add(label)
        else:
            y0, y1 = max(0, iy - 1), min(ny, iy + 2)
            x0, x1 = max(0, ix - 1), min(nx, ix + 2)
            segmap[y0:y1, x0:x1] = idx

    catalog = Table({
            "id": params["id"],
            "y": params["y_mean"],
            "x": params["x_mean"],
            "flux_true": flux_true_sampled,
            "flux_true_analytic": flux_true_analytic,
            "snr_true": snr_true,
            "snr_hi_true": snr_hi_true,
            "snr_requested": requested_snr,
            "flux_noise_eff": flux_noise_eff,
            "flux_noise_eff_hi": flux_noise_eff_hi,
            "pixel_noise_rms": np.full(len(params), noise_std),
            "pixel_noise_rms_hi": np.full(len(params), noise_std_hi),
            "pixel_noise_rms_lo": np.full(len(params), noise_std),
            "amplitude": params["amplitude"],
            "x_stddev": params["x_stddev"],
            "y_stddev": params["y_stddev"],
            "theta": params["theta"],
        })
    return [hires, lowres], segmap, catalog, [psf_hi.array, psf_lo.array], truth, [wht_hi, wht_lo]


@pytest.mark.parametrize(
    "scenario, ndilate, extend",
    [
        ("dilated0",     0, None),
        ("dilated1",     1, None),
        ("dilated2",     2, None),
        ("dilated3",     3, None),
        ("psf_wing",     3, "psf"),
    ],
)
def test_moffat_flux_recovery(tmp_path, scenario, ndilate, extend):
    """Run the pipeline on Moffat-PSF mocks and save PDF-style diagnostics.

    The test asserts only that the pipeline completes and the median flux
    ratio is close to unity; the primary artifacts are the saved plots,
    which are the object of the diagnostic.
    """
    out = tmp_path
    images, segmap, catalog, psfs, truth, wht = _make_moffat_data(
        seed=5, nsrc=150, size=904, ndilate=ndilate, peak_snr=50.0,
    )

    _, kernel_fit = PSF.from_array(psfs[0]).auto_matching_kernel_window(
        PSF.from_array(psfs[1]),
        fom="C^2",
        reg_lambda=1e-3,
        alpha_grid=np.linspace(0.02, 0.95, 24),
        beta_grid=np.linspace(0.05, 0.95, 23),
        core_radius=8.0,
        recenter=False,
        diagnostic_path=out / f"kernel_window_{scenario}.png",
        source_label="pseudo-F444W Moffat PSF",
        target_label="pseudo-F770W Moffat PSF",
        diagnostic_title=f"Moffat PSF kernel-window search: {scenario}",
        aperture_radius=8.0,
        return_result=True,
    )
    kernel = kernel_fit.kernel
    dirac = np.zeros((3, 3)); dirac[1, 1] = 1.0

    # Pipeline uses images[0] for template extraction and fits images[1:].
    # Feed hires twice so we get a self-consistency flux_1 alongside flux_2 (lowres).
    fit_images = [images[0], images[0], images[1]]
    fit_wht = [wht[0], wht[0], wht[1]]
    fit_kernels = [dirac, dirac, kernel]
    # psfs[0] is the high-resolution PSF template extension needs
    fit_psfs = [psfs[0], psfs[0], psfs[1]]
    table, resid, _ = pipeline.run(
        fit_images, segmap,
        catalog=catalog, weights=fit_wht, kernels=fit_kernels,
        psfs=fit_psfs,
        extend_templates=extend or "none",
        config=FitConfig(astrom_minimum_snr=0.0),
    )
    table["flux_true"] = catalog["flux_true"]

    # Diagnostic panel (truth / hires / segmap / lowres / model / residual)
    # residuals are returned for fitted images (indices 1,2) as resid[0], resid[1]
    model = images[1] - resid[1]
    save_diagnostic_image(
        out / f"diagnostic_{scenario}.png",
        truth, images[0], images[1], model, resid[1],
        segmap=segmap, catalog=catalog,
    )

    # Flux-ratio plot for the low-resolution channel (the PDF analogue)
    save_flux_vs_truth_plot(
        out / f"flux_ratio_{scenario}_lowres.png",
        np.asarray(table["flux_true"]),
        np.asarray(table["flux_2"]),
        error=np.asarray(table["err_2"]),
        label=f"Moffat F770W-like: {scenario}",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)",
    )
    save_flux_vs_truth_plot(
        out / f"flux_ratio_{scenario}_hires.png",
        np.asarray(table["flux_true"]),
        np.asarray(table["flux_1"]),
        error=np.asarray(table["err_1"]),
        label=f"Moffat F444W-like: {scenario}",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )

    # Summary statistics
    ratio_hi = np.asarray(table["flux_1"]) / np.asarray(table["flux_true"])
    ratio_lo = np.asarray(table["flux_2"]) / np.asarray(table["flux_true"])
    med_hi, med_lo = np.median(ratio_hi), np.median(ratio_lo)
    print(
        f"\n[{scenario}] median ratio  hires={med_hi:.4f}  lores={med_lo:.4f}  "
        f"(n={len(ratio_lo)} sources)"
    )
    # Record for comparison across scenarios:
    with open(out.parent / "moffat_summary.txt", "a") as f:
        f.write(
            f"{scenario:15s}  med_hi={med_hi:.4f}  med_lo={med_lo:.4f}  "
            f"p16={np.percentile(ratio_lo,16):.4f}  p84={np.percentile(ratio_lo,84):.4f}  "
            f"kernel_alpha={kernel_fit.alpha:.4f}  kernel_beta={kernel_fit.beta:.4f}  "
            f"kernel_score={kernel_fit.score:.4g}\n"
        )

    # Diagnostic test — do not enforce tight bias tolerance; the point is to
    # compare scenarios. Still sanity-check that nothing catastrophic happened.
    assert np.isfinite(med_hi) and np.isfinite(med_lo)
    assert 0.5 < med_hi < 1.5, f"hires ratio {med_hi} nowhere near unity"
    assert 0.5 < med_lo < 1.5, f"lores ratio {med_lo} nowhere near unity"
