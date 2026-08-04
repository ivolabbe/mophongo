"""Point spread function utilities.

This module provides a :class:`PSF` class which wraps a pixel grid
representation of a point spread function. Instances can be created from
analytic profiles (Moffat, Gaussian) or directly from a user supplied
array. A method is included to compute a matching kernel between two PSFs.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import logging
import os
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import shift as shift
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from drizzlepac import adrizzle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.data import download_file
from photutils.psf.matching import SplitCosineBellWindow, TukeyWindow
from photutils.centroids import centroid_quadratic

from tqdm import tqdm

from .utils import (
    measure_shape,
    get_wcs_pscale,
    get_slice_wcs,
    to_header,
    fit_kernel_fourier,
    fftconvolve,
    pad_to_shape,
    matching_kernel,
)
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord


logger = logging.getLogger(__name__)


@contextmanager
def _quiet_drizzle():
    """Silence routine drizzlepac/stwcs chatter during per-stamp PSF drizzles.

    Each ``DrizzlePSF.get_psf`` call drizzles one small output cutout per
    contributing frame. drizzlepac logs INFO lines per frame and warns that
    input points fall outside the output image — expected here because the
    evaluated ePSF input grid is deliberately larger than the cutout. With
    thousands of stamps this floods the log, so suppress below ERROR for the
    duration of the call.
    """
    names = ["drizzlepac", "stwcs", "stsci"]
    loggers = [logging.getLogger(n) for n in names]
    old_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, old_levels):
            lg.setLevel(level)


@dataclass
class GaussianFit:
    """Parameters describing a fitted Gaussian profile."""

    fwhm_x: float
    fwhm_y: float
    theta: float
    xc: float
    yc: float
    flux: float
    shape: tuple = None  # Store the original array shape

    def model(self) -> np.ndarray:
        """Generate the best fit Gaussian model."""
        from .utils import gaussian

        return gaussian(
            self.shape,
            self.fwhm_x,
            self.fwhm_y,
            self.theta,
            x0=self.xc,
            y0=self.yc,
            flux=self.flux,
        )


@dataclass
class MoffatFit:
    """Parameters describing a fitted Moffat profile."""

    fwhm_x: float
    fwhm_y: float
    beta: float
    theta: float
    xc: float
    yc: float
    flux: float
    shape: tuple = None  # Store the original array shape

    def model(self) -> np.ndarray:
        """Generate the best fit Moffat model."""
        from .utils import moffat

        return moffat(
            self.shape,
            self.fwhm_x,
            self.fwhm_y,
            self.beta,
            self.theta,
            x0=self.xc,
            y0=self.yc,
            flux=self.flux,
        )


@dataclass
class MatchingKernelWindowFit:
    """Result of a split-cosine-bell PSF matching window grid search."""

    alpha: float
    beta: float
    score: float
    kernel: np.ndarray
    matched_psf: np.ndarray
    score_grid: np.ndarray
    growth_error_grid: np.ndarray
    core_error_grid: np.ndarray
    l2_error_grid: np.ndarray
    kernel_regularization_grid: np.ndarray
    kernel_high_frequency_grid: np.ndarray
    kernel_cancellation_grid: np.ndarray
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    radii: np.ndarray
    target_growth: np.ndarray
    matched_growth: np.ndarray
    target_profile: np.ndarray
    matched_profile: np.ndarray


@dataclass
class MatchingKernelRegFit:
    """Result of a 1D regularization-parameter scan for a matching method."""

    method: str
    reg: float
    score: float
    kernel: np.ndarray
    matched_psf: np.ndarray
    reg_grid: np.ndarray
    score_grid: np.ndarray
    growth_error_grid: np.ndarray
    core_error_grid: np.ndarray
    l2_error_grid: np.ndarray
    kernel_regularization_grid: np.ndarray
    kernel_high_frequency_grid: np.ndarray
    kernel_cancellation_grid: np.ndarray
    radii: np.ndarray
    target_growth: np.ndarray
    matched_growth: np.ndarray
    target_profile: np.ndarray
    matched_profile: np.ndarray
    extra: dict[str, Any]


def _center_crop_even_axes_to_odd(arr: np.ndarray) -> np.ndarray:
    """Crop one leading pixel on even axes for optional odd-grid diagnostics."""
    y0 = 1 if arr.shape[0] % 2 == 0 else 0
    x0 = 1 if arr.shape[1] % 2 == 0 else 0
    if y0 == 0 and x0 == 0:
        return arr
    return arr[y0:, x0:]


def _prepare_psf_pair(
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    *,
    force_odd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize and pad PSFs to a common matching shape."""
    psf_hi = np.asarray(psf_hi, dtype=float).copy()
    psf_lo = np.asarray(psf_lo, dtype=float).copy()
    psf_hi[~np.isfinite(psf_hi)] = 0.0
    psf_lo[~np.isfinite(psf_lo)] = 0.0
    if psf_hi.shape != psf_lo.shape:
        shape = (max(psf_hi.shape[0], psf_lo.shape[0]), max(psf_hi.shape[1], psf_lo.shape[1]))
        psf_hi = pad_to_shape(psf_hi, shape)
        psf_lo = pad_to_shape(psf_lo, shape)
    if force_odd:
        psf_hi = _center_crop_even_axes_to_odd(psf_hi)
        psf_lo = _center_crop_even_axes_to_odd(psf_lo)
    return psf_hi, psf_lo


def _radius_image(shape: tuple[int, int]) -> np.ndarray:
    """Return pixel radius from the image center."""
    y, x = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    return np.hypot(x - cx, y - cy)


def _encircled_energy(image: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return encircled-energy profile at the requested radii."""
    radius = _radius_image(image.shape).ravel()
    values = np.asarray(image, dtype=float).ravel()
    order = np.argsort(radius)
    radius = radius[order]
    cumulative = np.cumsum(values[order])
    total = cumulative[-1]
    if total != 0:
        cumulative = cumulative / total
    return np.interp(radii, radius, cumulative, left=0.0, right=cumulative[-1])


def _radial_profile(image: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return annular mean profile at the requested radii."""
    radius = _radius_image(image.shape)
    values = np.asarray(image, dtype=float)
    edges = np.concatenate(
        [
            [0.0],
            0.5 * (radii[:-1] + radii[1:]),
            [radii[-1] + 0.5 * (radii[-1] - radii[-2])],
        ]
    )
    profile = np.empty_like(radii, dtype=float)
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (radius >= lo) & (radius < hi)
        profile[idx] = np.nanmean(values[mask]) if np.any(mask) else np.nan
    valid = np.isfinite(profile)
    if not np.all(valid) and np.any(valid):
        profile[~valid] = np.interp(radii[~valid], radii[valid], profile[valid])
    return profile


def _growth_curve_error(target: np.ndarray, matched: np.ndarray) -> float:
    """Return mean squared encircled-energy mismatch."""
    return float(np.nanmean((matched - target) ** 2))


def _growth_curve_ratio_for_plot(
    radii: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    min_radius: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite growth-curve ratio samples outside ``min_radius``."""
    radii = np.asarray(radii, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )
    mask = (radii > min_radius) & np.isfinite(ratio)
    return radii[mask], ratio[mask]


def _growth_curve_ratio_plot_samples(
    matched_psf: np.ndarray,
    target_psf: np.ndarray,
    radius_max: float,
    *,
    min_radius: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense diagnostic-only growth-ratio samples above ``min_radius``."""
    radius_max = float(radius_max)
    if not np.isfinite(radius_max) or radius_max <= min_radius:
        return np.array([], dtype=float), np.array([], dtype=float)

    r0 = np.nextafter(float(min_radius), np.inf)
    core_max = min(5.0, radius_max)
    parts: list[np.ndarray] = []
    if core_max > r0:
        parts.append(np.linspace(r0, core_max, 80))
    if radius_max > core_max:
        parts.append(np.geomspace(max(core_max, r0), radius_max, 120))
    radii = np.unique(np.concatenate(parts)) if parts else np.array([r0], dtype=float)
    return _growth_curve_ratio_for_plot(
        radii,
        _encircled_energy(matched_psf, radii),
        _encircled_energy(target_psf, radii),
        min_radius=min_radius,
    )


def _core_profile_error(
    target: np.ndarray,
    matched: np.ndarray,
    radii: np.ndarray,
    core_radius: float,
) -> float:
    """Return mean squared log-profile mismatch inside ``core_radius``."""
    mask = radii <= core_radius
    if not np.any(mask):
        mask = np.ones_like(radii, dtype=bool)
    floor = max(float(np.nanmax(target)) * 1e-6, np.finfo(float).tiny)
    target_log = np.log10(np.maximum(target[mask], floor))
    matched_log = np.log10(np.maximum(matched[mask], floor))
    return float(np.nanmean((matched_log - target_log) ** 2))


def _kernel_high_frequency_power(
    kernel: np.ndarray,
    *,
    high_frequency_radius: float = 0.7,
) -> float:
    """Return the fraction of kernel Fourier power near the Nyquist scale.

    ``high_frequency_radius`` is expressed in Nyquist units: 1 is the axis
    Nyquist frequency and sqrt(2) is the corner of the Fourier grid.  This
    penalizes pixel-scale ringing that can match the PSF pair but is unstable
    when applied to real templates.
    """
    kernel = np.asarray(kernel, dtype=float)
    power = np.abs(np.fft.fft2(np.fft.ifftshift(kernel))) ** 2
    fy = np.fft.fftfreq(kernel.shape[0]) / 0.5
    fx = np.fft.fftfreq(kernel.shape[1]) / 0.5
    qy, qx = np.meshgrid(fy, fx, indexing="ij")
    q = np.hypot(qx, qy)
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    high = q >= high_frequency_radius
    return float(np.sum(power[high]) / total)


def _kernel_flux_cancellation(kernel: np.ndarray) -> float:
    """Return excess absolute flux from positive/negative kernel oscillations."""
    kernel = np.asarray(kernel, dtype=float)
    net = abs(float(np.sum(kernel)))
    if net <= 0:
        return np.inf
    return float(max(0.0, np.sum(np.abs(kernel)) / net - 1.0))


def _kernel_regularization(
    kernel: np.ndarray,
    *,
    high_frequency_radius: float = 0.7,
    high_frequency_weight: float = 0.0,
    cancellation_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Return kernel stability penalty and its components."""
    high_frequency = _kernel_high_frequency_power(
        kernel,
        high_frequency_radius=high_frequency_radius,
    )
    cancellation = _kernel_flux_cancellation(kernel)
    regularization = (
        high_frequency_weight * high_frequency
        + cancellation_weight * cancellation**2
    )
    return float(regularization), float(high_frequency), float(cancellation)


_FOM_PRESETS: dict[str, dict[str, float]] = {
    "growth_core_only": {
        "kernel_high_frequency_weight": 0.0,
        "kernel_cancellation_weight": 0.0,
    },
    "growth_core_hf": {
        "kernel_high_frequency_weight": 1.0,
        "kernel_cancellation_weight": 0.0,
    },
    "growth_core_cancel": {
        "kernel_high_frequency_weight": 0.0,
        "kernel_cancellation_weight": 1.0,
    },
    "growth_core_hf_cancel": {
        "kernel_high_frequency_weight": 1.0,
        "kernel_cancellation_weight": 1.0,
    },
}

_FOM_ALIASES = {
    "default": "growth_core_cancel",
    "c": "growth_core_cancel",
    "c2": "growth_core_cancel",
    "c^2": "growth_core_cancel",
    "cancel": "growth_core_cancel",
    "cancellation": "growth_core_cancel",
    "hf": "growth_core_hf",
    "hf_cancel": "growth_core_hf_cancel",
    "none": "growth_core_only",
}

_KERNEL_WINDOW_BASE_ALPHA_COUNT = 23
_KERNEL_WINDOW_BASE_BETA_COUNT = 19
_PROFILE_XTICKS = np.array([1, 2, 4, 10, 20, 50, 100], dtype=float)
_GROWTH_RATIO_MIN_RADIUS_PIX = 0.7


def _kernel_window_default_grids(grid_oversample: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return default split-cosine-bell search grids."""
    factor = max(1, int(grid_oversample))
    alpha_count = (_KERNEL_WINDOW_BASE_ALPHA_COUNT - 1) * factor + 1
    beta_count = (_KERNEL_WINDOW_BASE_BETA_COUNT - 1) * factor + 1
    return (
        np.linspace(0.02, 0.90, alpha_count),
        np.linspace(0.05, 0.95, beta_count),
    )


def _resolve_kernel_window_fom(fom: str) -> tuple[str, dict[str, float]]:
    """Return optimizer weights for a named kernel-window figure of merit."""
    key = fom.strip().lower().replace("-", "_").replace(" ", "_")
    key = _FOM_ALIASES.get(key, key)
    if key not in _FOM_PRESETS:
        choices = ", ".join(sorted(_FOM_PRESETS | _FOM_ALIASES))
        raise ValueError(f"Unknown kernel window FOM {fom!r}. Expected one of: {choices}")
    return key, dict(_FOM_PRESETS[key])


def _log10_score_grid(score_grid: np.ndarray) -> np.ma.MaskedArray:
    """Return a masked log10 score grid for diagnostics."""
    score = np.asarray(score_grid, dtype=float)
    log_score = np.full_like(score, np.nan, dtype=float)
    valid = np.isfinite(score) & (score > 0)
    log_score[valid] = np.log10(score[valid])
    return np.ma.masked_invalid(log_score)


def _finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    """Percentile helper that is robust to fully masked diagnostic arrays."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    return float(np.nanpercentile(finite, percentile))


def _shared_asinh_scale(images: list[np.ndarray]) -> tuple[float, float, float]:
    """Return shared physical limit, linear width, and transformed limit."""
    abs_stack = np.concatenate([np.ravel(np.abs(np.asarray(image, dtype=float))) for image in images])
    finite = abs_stack[np.isfinite(abs_stack)]
    limit = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
    limit = max(limit, np.finfo(float).eps)
    stretch = limit / 120.0
    transformed_limit = float(np.arcsinh(limit / stretch))
    return limit, stretch, transformed_limit


def _add_shared_asinh_colorbar(
    fig,
    *,
    stretch: float,
    limit: float,
    mappable,
    cax_bounds: list[float] | tuple[float, float, float, float] = (0.86, 0.14, 0.018, 0.40),
) -> None:
    """Add one physical-value colorbar for shared asinh diagnostic panels."""
    ticks_physical = np.array([-limit, -limit / 10.0, 0.0, limit / 10.0, limit])
    ticks_transformed = np.arcsinh(ticks_physical / stretch)
    cax = fig.add_axes(cax_bounds)
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        ticks=ticks_transformed,
    )
    cbar.ax.set_yticklabels([f"{value:.2e}" for value in ticks_physical])
    cbar.set_label("image value")


def _save_matching_kernel_window_diagnostic(
    path: str | Path,
    result: MatchingKernelWindowFit,
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    *,
    fom_name: str,
    core_radius: float,
    source_label: str,
    target_label: str,
    reg_lambda: float,
    title: str | None = None,
    aperture_radius: float | None = None,
) -> None:
    """Write a production-style diagnostic for an optimized PSF matching window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.suffix == "":
        path = path / "diagnostic_window.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    radii = result.radii
    residual = result.matched_psf - psf_lo
    alpha_idx = int(np.argmin(np.abs(result.alpha_grid - result.alpha)))
    beta_idx = int(np.argmin(np.abs(result.beta_grid - result.beta)))
    opt_growth = float(result.growth_error_grid[beta_idx, alpha_idx])
    opt_core = float(result.core_error_grid[beta_idx, alpha_idx])
    opt_l2 = float(result.l2_error_grid[beta_idx, alpha_idx])
    opt_kernel_reg = float(result.kernel_regularization_grid[beta_idx, alpha_idx])
    opt_kernel_hf = float(result.kernel_high_frequency_grid[beta_idx, alpha_idx])
    opt_kernel_cancel = float(result.kernel_cancellation_grid[beta_idx, alpha_idx])

    image_panels = [
        (psf_hi, source_label),
        (psf_lo, target_label),
        (result.kernel, "kernel"),
        (result.matched_psf, f"K * {source_label}"),
        (residual, f"K * {source_label} - {target_label}"),
    ]
    physical_limit, stretch, transformed_limit = _shared_asinh_scale(
        [image for image, _ in image_panels]
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    score = _log10_score_grid(result.score_grid)
    score_values = score.filled(np.nan)
    im = axes[0, 0].imshow(
        score,
        origin="lower",
        aspect="auto",
        extent=[
            float(result.alpha_grid.min()),
            float(result.alpha_grid.max()),
            float(result.beta_grid.min()),
            float(result.beta_grid.max()),
        ],
        vmin=_finite_percentile(score_values, 1, -12.0),
        vmax=_finite_percentile(score_values, 95, 0.0),
    )
    axes[0, 0].plot(
        result.alpha,
        result.beta,
        "r*",
        ms=15,
        label=f"alpha={result.alpha:.3f}\nbeta={result.beta:.3f}\nscore={result.score:.3g}",
    )
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("beta")
    axes[0, 0].set_title(f"log10(FOM): {fom_name}")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    profile_radius_max = float(min(psf_hi.shape) / 2.0 - 1.0)
    axes[0, 1].plot(radii, result.target_profile, "k-", lw=2, label=target_label)
    axes[0, 1].plot(radii, result.matched_profile, "r-.", lw=1.5, label="matched")
    axes[0, 1].set_xscale("symlog", linthresh=1.0)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("radius [pix]")
    axes[0, 1].set_ylabel("annular mean intensity")
    axes[0, 1].set_title("radial profiles")
    axes[0, 1].set_xlim(0.9, profile_radius_max)
    profile_ticks = _PROFILE_XTICKS[_PROFILE_XTICKS <= profile_radius_max]
    axes[0, 1].set_xticks(profile_ticks)
    axes[0, 1].set_xticklabels([f"{tick:g}" for tick in profile_ticks])
    axes[0, 1].set_ylim(bottom=1e-7)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    growth_ratio_radii, growth_ratio = _growth_curve_ratio_plot_samples(
        result.matched_psf,
        psf_lo,
        profile_radius_max,
        min_radius=_GROWTH_RATIO_MIN_RADIUS_PIX,
    )
    growth_ticks = np.concatenate(
        ([float(_GROWTH_RATIO_MIN_RADIUS_PIX)], profile_ticks[profile_ticks > _GROWTH_RATIO_MIN_RADIUS_PIX])
    )
    axes[0, 2].plot(
        growth_ratio_radii,
        growth_ratio,
        "r-",
        label=f"SCB({result.alpha:.2f}, {result.beta:.2f})",
    )
    axes[0, 2].axhline(1.0, color="k", lw=1, alpha=0.7)
    axes[0, 2].axhline(0.98, color="0.45", lw=1, ls=":", alpha=0.8)
    axes[0, 2].axhline(1.02, color="0.45", lw=1, ls=":", alpha=0.8, label="+/-2%")
    if aperture_radius is not None:
        axes[0, 2].axvline(
            aperture_radius,
            color="0.3",
            lw=1.2,
            ls="--",
            alpha=0.8,
            label=f"r={aperture_radius:g} pix",
        )
    axes[0, 2].set_xscale("symlog", linthresh=1.0)
    axes[0, 2].set_xlabel("radius [pix]")
    axes[0, 2].set_ylabel("EE(match) / EE(target)")
    axes[0, 2].set_ylim(0.8, 1.2)
    axes[0, 2].set_xlim(_GROWTH_RATIO_MIN_RADIUS_PIX, profile_radius_max)
    axes[0, 2].set_xticks(growth_ticks)
    axes[0, 2].set_xticklabels([f"{tick:g}" for tick in growth_ticks])
    axes[0, 2].set_title("growth-curve ratio")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend(fontsize=8)

    def show_panel(ax, image: np.ndarray, panel_title: str) -> None:
        im = ax.imshow(
            np.arcsinh(np.asarray(image, dtype=float) / stretch),
            origin="lower",
            cmap="RdBu_r",
            vmin=-transformed_limit,
            vmax=transformed_limit,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        return im

    image_mappable = None
    for ax, (image, panel_title) in zip(
        [axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1]],
        image_panels,
    ):
        image_mappable = show_panel(ax, image, panel_title)
    _add_shared_asinh_colorbar(
        fig,
        stretch=stretch,
        limit=physical_limit,
        mappable=image_mappable,
        cax_bounds=(0.86, 0.14, 0.018, 0.40),
    )

    info_ax = axes[2, 2]
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.98,
        f"FOM preset: {fom_name}\n"
        "FOM = growth MSE + core MSE + lambda * R(K)\n"
        "C = sum(abs(K)) / abs(sum(K)) - 1\n\n"
        f"alpha={result.alpha:.3f} beta={result.beta:.3f}\n"
        f"score={result.score:.4g}\n"
        f"RMS residual={np.sqrt(np.mean(residual**2)):.4g}\n"
        f"growth MSE={opt_growth:.3g}\n"
        f"core log MSE={opt_core:.3g} inside r<={core_radius:.3g} pix\n"
        f"L2 image MSE={opt_l2:.3g}\n"
        f"lambda={reg_lambda:.3g}\n"
        f"kernel R={opt_kernel_reg:.3g}\n"
        f"  high-freq={opt_kernel_hf:.3g}\n"
        f"  cancellation={opt_kernel_cancel:.3g}",
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    fig.suptitle(title or "PSF matching kernel-window diagnostic", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.06, wspace=0.42, hspace=0.55)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_matching_kernel_regularization_diagnostic(
    path: str | Path,
    result: MatchingKernelRegFit,
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    *,
    source_label: str,
    target_label: str,
    title: str | None = None,
    aperture_radius: float | None = None,
    target_note: str | None = None,
) -> None:
    """Write the standard diagnostic for a scalar-regularized matching kernel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.suffix == "":
        path = path / f"diagnostic_{result.method}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    radii = result.radii
    residual = result.matched_psf - psf_lo
    opt_idx = int(np.argmin(np.abs(result.reg_grid - result.reg)))
    opt_growth = float(result.growth_error_grid[opt_idx])
    opt_core = float(result.core_error_grid[opt_idx])
    opt_l2 = float(result.l2_error_grid[opt_idx])
    opt_kernel_reg = float(result.kernel_regularization_grid[opt_idx])
    opt_kernel_hf = float(result.kernel_high_frequency_grid[opt_idx])
    opt_kernel_cancel = float(result.kernel_cancellation_grid[opt_idx])

    image_panels = [
        (psf_hi, source_label),
        (psf_lo, target_label),
        (result.kernel, "kernel"),
        (result.matched_psf, f"K * {source_label}"),
        (residual, f"K * {source_label} - {target_label}"),
    ]
    physical_limit, stretch, transformed_limit = _shared_asinh_scale(
        [image for image, _ in image_panels]
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    valid_score = np.isfinite(result.score_grid) & (result.score_grid > 0)
    axes[0, 0].semilogx(result.reg_grid, result.score_grid, "o-", color="C0", ms=4)
    axes[0, 0].axvline(
        result.reg,
        color="r",
        lw=1.2,
        ls="--",
        label=f"lambda*={result.reg:.2e}\nscore={result.score:.3g}",
    )
    axes[0, 0].set_xlabel("lambda")
    axes[0, 0].set_ylabel("FOM")
    if np.any(valid_score):
        axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(f"FOM vs lambda: {result.method}")
    axes[0, 0].grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8, loc="best")

    profile_radius_max = float(min(psf_hi.shape) / 2.0 - 1.0)
    profile_ticks = _PROFILE_XTICKS[_PROFILE_XTICKS <= profile_radius_max]
    axes[0, 1].plot(radii, result.target_profile, "k-", lw=2, label=target_label)
    axes[0, 1].plot(radii, result.matched_profile, "r-.", lw=1.5, label="matched")
    axes[0, 1].set_xscale("symlog", linthresh=1.0)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("radius [pix]")
    axes[0, 1].set_ylabel("annular mean intensity")
    axes[0, 1].set_title("radial profiles")
    axes[0, 1].set_xlim(0.9, profile_radius_max)
    axes[0, 1].set_xticks(profile_ticks)
    axes[0, 1].set_xticklabels([f"{tick:g}" for tick in profile_ticks])
    axes[0, 1].set_ylim(bottom=1e-7)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    growth_ratio_radii, growth_ratio = _growth_curve_ratio_plot_samples(
        result.matched_psf,
        psf_lo,
        profile_radius_max,
        min_radius=_GROWTH_RATIO_MIN_RADIUS_PIX,
    )
    growth_ticks = np.concatenate(
        ([float(_GROWTH_RATIO_MIN_RADIUS_PIX)], profile_ticks[profile_ticks > _GROWTH_RATIO_MIN_RADIUS_PIX])
    )
    axes[0, 2].plot(growth_ratio_radii, growth_ratio, "r-", label=f"lambda={result.reg:.2e}")
    axes[0, 2].axhline(1.0, color="k", lw=1, alpha=0.7)
    axes[0, 2].axhline(0.98, color="0.45", lw=1, ls=":", alpha=0.8)
    axes[0, 2].axhline(1.02, color="0.45", lw=1, ls=":", alpha=0.8, label="+/-2%")
    if aperture_radius is not None:
        axes[0, 2].axvline(
            aperture_radius,
            color="0.3",
            lw=1.2,
            ls="--",
            alpha=0.8,
            label=f"r={aperture_radius:g} pix",
        )
    axes[0, 2].set_xscale("symlog", linthresh=1.0)
    axes[0, 2].set_xlabel("radius [pix]")
    axes[0, 2].set_ylabel("EE(match) / EE(target)")
    axes[0, 2].set_ylim(0.8, 1.2)
    axes[0, 2].set_xlim(_GROWTH_RATIO_MIN_RADIUS_PIX, profile_radius_max)
    axes[0, 2].set_xticks(growth_ticks)
    axes[0, 2].set_xticklabels([f"{tick:g}" for tick in growth_ticks])
    axes[0, 2].set_title("growth-curve ratio")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend(fontsize=8)

    image_mappable = None
    for ax, (image, panel_title) in zip(
        [axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1]],
        image_panels,
    ):
        image_mappable = ax.imshow(
            np.arcsinh(np.asarray(image, dtype=float) / stretch),
            origin="lower",
            cmap="RdBu_r",
            vmin=-transformed_limit,
            vmax=transformed_limit,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
    _add_shared_asinh_colorbar(
        fig,
        stretch=stretch,
        limit=physical_limit,
        mappable=image_mappable,
        cax_bounds=(0.86, 0.14, 0.018, 0.40),
    )

    core_radius = float(result.extra.get("core_radius", np.nan))
    growth_weight = float(result.extra.get("growth_weight", np.nan))
    core_weight = float(result.extra.get("core_weight", np.nan))
    l2_weight = float(result.extra.get("l2_weight", np.nan))
    kernel_weight = float(result.extra.get("kernel_regularization_weight", np.nan))
    note = "" if target_note is None else f"\n\ndata: {target_note}"
    info_ax = axes[2, 2]
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.98,
        f"method: {result.method}\n"
        f"lambda={result.reg:.4g}\n"
        f"score={result.score:.4g}\n"
        f"growth MSE={opt_growth:.3g}\n"
        f"core log MSE={opt_core:.3g} inside r<={core_radius:.3g} pix\n"
        f"L2 image MSE={opt_l2:.3g}\n"
        f"RMS residual={np.sqrt(np.mean(residual**2)):.4g}\n\n"
        f"FOM weights: growth={growth_weight:g}, core={core_weight:g}, "
        f"L2={l2_weight:g}, kernel={kernel_weight:g}\n"
        f"kernel R={opt_kernel_reg:.3g}\n"
        f"  high-freq={opt_kernel_hf:.3g}\n"
        f"  cancellation={opt_kernel_cancel:.3g}"
        f"{note}",
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.suptitle(title or f"PSF matching kernel diagnostic - {result.method}", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.06, wspace=0.42, hspace=0.55)
    fig.savefig(path, dpi=180)
    plt.close(fig)


@dataclass
class PSF:
    """Discrete point spread function."""

    array: np.ndarray
    wcs: WCS | None = None
    pos: tuple[float, float] | None = None

    @property
    def data(self) -> np.ndarray:
        """Alias for the PSF array."""
        return self.array

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array, dtype=float)

    @classmethod
    def moffat(
        cls,
        size: int | tuple[int, int],
        fwhm_x: float,
        fwhm_y: float,
        beta: float,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Moffat PSF."""
        from .utils import moffat as moffat_psf

        return cls(moffat_psf(size, fwhm_x, fwhm_y, beta, theta))

    @classmethod
    def gaussian(
        cls,
        size: int | tuple[int, int],
        fwhm: float | tuple[float, float] | None = None,
        theta: float = 0.0,
    ) -> "PSF":
        """Create a normalized Gaussian PSF.

        Parameters
        ----------
        size : int or tuple of int
            Size of the PSF array.
        fwhm_x, fwhm_y : float, optional
            FWHM along x and y axes.
        fwhm : float or tuple, optional
            If given, overrides fwhm_x and fwhm_y. If tuple, interpreted as (fwhm_x, fwhm_y).
        theta : float, optional
            Rotation angle in radians.

        Returns
        -------
        PSF
            PSF instance with a Gaussian profile.
        """
        from .utils import gaussian

        # Handle fwhm as tuple or float
        if fwhm is not None:
            if isinstance(fwhm, (tuple, list)) and len(fwhm) == 2:
                fwhm_x, fwhm_y = fwhm
            else:
                fwhm_x = fwhm_y = fwhm

        return cls(gaussian(size, fwhm_x, fwhm_y, theta=theta))

    @classmethod
    def delta(cls, size: int = 3) -> "PSF":
        """Create a symmetric delta function PSF.

        Parameters
        ----------
        size : int, optional
            Length of each side of the square PSF array. ``size`` should be odd
            to center the delta pixel. Defaults to ``3``.

        Returns
        -------
        PSF
            PSF instance containing a single central pixel with unit flux.
        """

        array = np.zeros((size, size), dtype=float)
        cy = size // 2
        cx = size // 2
        array[cy, cx] = 1.0
        return cls(array)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "PSF":
        """Create a PSF from an arbitrary pixel array."""
        return cls(array)

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        position: tuple[float, float] | tuple["Quantity", "Quantity"] | None = None,
        *,
        search_boxsize: int | tuple[int, int] | None = None,
        fit_boxsize: int | tuple[int, int] = 5,
        size: int = 51,
        wcs: WCS | None = None,
        verbose: bool = False,
    ) -> "PSF":
        """Extract a PSF from ``data`` around an approximate star position.

        Parameters
        ----------
        data : ndarray
            Image containing the star.
        position : tuple of float or tuple of astropy Quantity
            Approximate ``(x, y)`` pixel coordinates of the star, or (ra, dec) as astropy Quantities (e.g. with unit deg).
        search_boxsize, fit_boxsize : int or tuple of int, optional
            Passed to :func:`photutils.centroids.centroid_quadratic`.
        size : int, optional
            Cutout size. The PSF will be a square array of this shape.
        wcs : astropy.wcs.WCS, optional
            WCS object for the image.

        Returns
        -------
        PSF
            PSF instance extracted from the image.
        """
        from astropy.nddata import Cutout2D
        from photutils.centroids import centroid_quadratic
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if position is None:
            # get center pixel of ndarray
            position_pix = ((data.shape[1] - 1) // 2, (data.shape[0] - 1) / 2)
        else:
            # If position is given as (Quantity, Quantity) and wcs is supplied, convert to pixel
            if hasattr(position[0], "unit") and hasattr(position[1], "unit") and wcs is not None:
                sky = SkyCoord(position[0], position[1])
                x, y = wcs.world_to_pixel(sky)
                position_pix = (x, y)
            else:
                position_pix = position

        # If either search_boxsize or fit_boxsize is None, skip recentering
        if search_boxsize is None:
            x_cen, y_cen = position_pix
        else:
            x_cen, y_cen = centroid_quadratic(
                data,
                xpeak=position_pix[0],
                ypeak=position_pix[1],
                fit_boxsize=fit_boxsize,
                search_boxsize=search_boxsize,
            )
        if verbose:
            if search_boxsize is not None:
                print(f"original position: ({position_pix})")
            print(f"Centroid position: ({x_cen}, {y_cen})")

        cut = Cutout2D(
            data,
            (x_cen, y_cen),  # Use unrounded center
            (size, size),
            mode="partial",
            wcs=wcs,
            fill_value=0.0,
            copy=True,
        )
        return cls(array=np.asarray(cut.data), wcs=cut.wcs, pos=cut.input_position_cutout)

    def matching_kernel(
        self,
        other: "PSF" | np.ndarray,
        window: object | None = None,
        *,
        recenter: bool = False,
        method: str = "window",
        reg: float = 1e-3,
        wavelet: str = "db4",
        levels: int = 3,
        threshold_factor: float = 3.0,
        noise_sigma: float | None = None,
        forward_wavelet_wiener: bool = True,
        signal_psd: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the convolution kernel that matches ``self`` to ``other``.

        Parameters
        ----------
        other : PSF or np.ndarray
            The target PSF. If an array is supplied, its flux is used as given.
        window : optional
            Fourier-domain window function used when ``method="window"``.
            Defaults to ``SplitCosineBellWindow(alpha=0.4, beta=0.1)``.
        recenter : bool, optional
            If ``True`` the resulting kernel is shifted to its centroid.
        method : str, optional
            ``"window"`` (default), ``"tikhonov"``, ``"wiener"`` or
            ``"forward"`` (ForWaRD Fourier+wavelet regularized deconvolution,
            Neelamani et al. 2004).
        reg : float, optional
            Regularization parameter for ``tikhonov``, ``wiener``, and
            ``forward``.  Scaled by ``max(|H_hi|^2)`` so it is dimensionless.
        wavelet, levels, threshold_factor, noise_sigma, forward_wavelet_wiener :
            ``forward``-only options.  ``threshold_factor`` controls the hard
            threshold on detail coefficients in units of estimated per-subband
            noise.  Pass ``forward_wavelet_wiener=False`` to skip step 5.
        signal_psd : np.ndarray, optional
            Signal power spectral density for ``method="wiener"``.

        Returns
        -------
        kernel : np.ndarray
            Convolution kernel that matches self to other.
        """
        psf_hi = self.array

        # Handle both PSF objects and numpy arrays
        if isinstance(other, PSF):
            psf_lo = other.array
        elif hasattr(other, "array"):
            psf_lo = np.asarray(other.array, dtype=float)
        else:
            psf_lo = np.asarray(other, dtype=float)
        psf_hi, psf_lo = _prepare_psf_pair(psf_hi, psf_lo)

        kernel = matching_kernel(
            psf_hi,
            psf_lo,
            window=window,
            recenter=recenter,
            method=method,
            reg=reg,
            wavelet=wavelet,
            levels=levels,
            threshold_factor=threshold_factor,
            noise_sigma=noise_sigma,
            forward_wavelet_wiener=forward_wavelet_wiener,
            signal_psd=signal_psd,
        )
        return kernel.astype(np.float32)

    def optimize_matching_kernel_window(
        self,
        other: "PSF" | np.ndarray,
        *,
        alpha_grid: np.ndarray | None = None,
        beta_grid: np.ndarray | None = None,
        grid_oversample: int = 2,
        core_radius: float | None = None,
        growth_weight: float = 1.0,
        core_weight: float = 1.0,
        l2_weight: float = 0.0,
        kernel_regularization_weight: float = 1e-3,
        kernel_high_frequency_radius: float = 0.7,
        kernel_high_frequency_weight: float = 0.0,
        kernel_cancellation_weight: float = 1.0,
        recenter: bool = False,
    ) -> MatchingKernelWindowFit:
        """Grid-search split-cosine-bell parameters for PSF matching.

        The optimizer evaluates ``SplitCosineBellWindow(alpha, beta)`` values
        satisfying ``alpha + beta <= 1``.  Each kernel is generated with the
        same :func:`mophongo.utils.matching_kernel` routine used elsewhere in
        the pipeline, then scored using squared encircled-energy mismatch,
        squared log radial-profile mismatch in the core, and a configurable
        kernel stability penalty.  The default stability term is signed-flux
        cancellation only, i.e. ``1e-3 * C(K)^2``.  High-frequency kernel
        power is still available as an optional additional term.

        Parameters
        ----------
        other : PSF or np.ndarray
            Target PSF that ``self`` should be convolved into.
        alpha_grid, beta_grid : np.ndarray, optional
            Split-cosine-bell grid values.  Defaults cover broad windows
            without including the singular endpoints.
        grid_oversample : int, optional
            Refinement factor for the default alpha/beta grids.  Defaults to
            ``2``, i.e. twice the historical default sampling. Ignored when
            explicit grids are supplied.
        core_radius : float, optional
            Maximum radius in pixels for the core radial-profile term.
            Defaults to one quarter of the PSF size.
        growth_weight, core_weight, l2_weight : float, optional
            Weights for the growth-curve, radial-core, and image-space mean
            square terms in the score.
        kernel_regularization_weight : float, optional
            Overall weight for the kernel stability term.
        kernel_high_frequency_radius : float, optional
            Fourier radius, in Nyquist units, above which kernel power is
            penalized as pixel-scale ringing.
        kernel_high_frequency_weight, kernel_cancellation_weight : float, optional
            Relative weights for high-frequency kernel power and excess
            positive/negative L1 flux cancellation.
        recenter : bool, optional
            Passed to :func:`mophongo.utils.matching_kernel`.  Defaults to
            ``False`` so already-centered PSFs are not shifted during scoring.

        Returns
        -------
        MatchingKernelWindowFit
            Best-fit parameters, kernel, matched PSF, and diagnostic grids.
        """
        psf_hi = np.asarray(self.array, dtype=float)
        psf_lo = other.array if isinstance(other, PSF) else np.asarray(other, dtype=float)
        psf_hi, psf_lo = _prepare_psf_pair(psf_hi, psf_lo)

        default_alpha_grid, default_beta_grid = _kernel_window_default_grids(grid_oversample)
        alpha_grid = default_alpha_grid if alpha_grid is None else np.asarray(alpha_grid, dtype=float)
        beta_grid = default_beta_grid if beta_grid is None else np.asarray(beta_grid, dtype=float)
        if core_radius is None:
            core_radius = min(psf_hi.shape) / 4.0

        radii = np.linspace(0.5, min(psf_hi.shape) / 2.0 - 1.0, 100)
        target_growth = _encircled_energy(psf_lo, radii)
        target_profile = _radial_profile(psf_lo, radii)

        score_grid = np.full((len(beta_grid), len(alpha_grid)), np.nan, dtype=float)
        growth_error_grid = np.full_like(score_grid, np.nan)
        core_error_grid = np.full_like(score_grid, np.nan)
        l2_error_grid = np.full_like(score_grid, np.nan)
        kernel_regularization_grid = np.full_like(score_grid, np.nan)
        kernel_high_frequency_grid = np.full_like(score_grid, np.nan)
        kernel_cancellation_grid = np.full_like(score_grid, np.nan)

        best: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
        for ib, beta in enumerate(beta_grid):
            for ia, alpha in enumerate(alpha_grid):
                if alpha < 0 or beta < 0 or alpha + beta > 1:
                    continue
                window = SplitCosineBellWindow(alpha=float(alpha), beta=float(beta))
                kernel = matching_kernel(psf_hi, psf_lo, window=window, recenter=recenter)
                matched = fftconvolve(psf_hi, kernel, mode="same")

                growth = _encircled_energy(matched, radii)
                profile = _radial_profile(matched, radii)
                growth_error = _growth_curve_error(target_growth, growth)
                core_error = _core_profile_error(target_profile, profile, radii, core_radius)
                l2_error = float(np.nanmean((matched - psf_lo) ** 2))
                kernel_regularization, kernel_high_frequency, kernel_cancellation = (
                    _kernel_regularization(
                        kernel,
                        high_frequency_radius=kernel_high_frequency_radius,
                        high_frequency_weight=kernel_high_frequency_weight,
                        cancellation_weight=kernel_cancellation_weight,
                    )
                )
                score = (
                    growth_weight * growth_error
                    + core_weight * core_error
                    + l2_weight * l2_error
                    + kernel_regularization_weight * kernel_regularization
                )

                growth_error_grid[ib, ia] = growth_error
                core_error_grid[ib, ia] = core_error
                l2_error_grid[ib, ia] = l2_error
                kernel_regularization_grid[ib, ia] = kernel_regularization
                kernel_high_frequency_grid[ib, ia] = kernel_high_frequency
                kernel_cancellation_grid[ib, ia] = kernel_cancellation
                score_grid[ib, ia] = score
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, ib, ia, kernel, matched)

        if best is None:
            raise ValueError("No valid split-cosine-bell window parameters were evaluated.")

        score, ib, ia, kernel, matched = best
        return MatchingKernelWindowFit(
            alpha=float(alpha_grid[ia]),
            beta=float(beta_grid[ib]),
            score=float(score),
            kernel=np.asarray(kernel),
            matched_psf=np.asarray(matched),
            score_grid=score_grid,
            growth_error_grid=growth_error_grid,
            core_error_grid=core_error_grid,
            l2_error_grid=l2_error_grid,
            kernel_regularization_grid=kernel_regularization_grid,
            kernel_high_frequency_grid=kernel_high_frequency_grid,
            kernel_cancellation_grid=kernel_cancellation_grid,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            radii=radii,
            target_growth=target_growth,
            matched_growth=_encircled_energy(matched, radii),
            target_profile=target_profile,
            matched_profile=_radial_profile(matched, radii),
        )

    def optimize_matching_kernel_regularization(
        self,
        other: "PSF" | np.ndarray,
        *,
        method: str = "tikhonov",
        reg_grid: np.ndarray | None = None,
        core_radius: float | None = None,
        growth_weight: float = 1.0,
        core_weight: float = 1.0,
        l2_weight: float = 0.0,
        kernel_regularization_weight: float = 1e-3,
        kernel_high_frequency_radius: float = 0.7,
        kernel_high_frequency_weight: float = 0.0,
        kernel_cancellation_weight: float = 1.0,
        recenter: bool = False,
        wavelet: str = "db4",
        levels: int = 3,
        threshold_factor: float = 3.0,
        noise_sigma: float | None = None,
        forward_wavelet_wiener: bool = True,
        signal_psd: np.ndarray | None = None,
        diagnostic_path: str | Path | None = None,
        source_label: str = "source PSF",
        target_label: str = "target PSF",
        diagnostic_title: str | None = None,
        aperture_radius: float | None = None,
        diagnostic_note: str | None = None,
    ) -> MatchingKernelRegFit:
        """Grid-search the scalar regularization parameter for non-windowed methods.

        Identical figure of merit to
        :meth:`optimize_matching_kernel_window`; mostly used to find the best
        ``reg`` for ``method="tikhonov" | "wiener" | "forward"``.

        If ``diagnostic_path`` is provided, the standard PSF matching
        diagnostic is written. Passing a directory writes
        ``diagnostic_<method>.png`` inside that directory.
        """
        psf_hi = np.asarray(self.array, dtype=float)
        psf_lo = other.array if isinstance(other, PSF) else np.asarray(other, dtype=float)
        psf_hi, psf_lo = _prepare_psf_pair(psf_hi, psf_lo)

        if reg_grid is None:
            reg_grid = np.logspace(-6, -1, 21)
        reg_grid = np.asarray(reg_grid, dtype=float)

        if core_radius is None:
            core_radius = min(psf_hi.shape) / 4.0

        radii = np.linspace(0.5, min(psf_hi.shape) / 2.0 - 1.0, 100)
        target_growth = _encircled_energy(psf_lo, radii)
        target_profile = _radial_profile(psf_lo, radii)

        n = len(reg_grid)
        score_grid = np.full(n, np.nan)
        growth_error_grid = np.full(n, np.nan)
        core_error_grid = np.full(n, np.nan)
        l2_error_grid = np.full(n, np.nan)
        kernel_regularization_grid = np.full(n, np.nan)
        kernel_high_frequency_grid = np.full(n, np.nan)
        kernel_cancellation_grid = np.full(n, np.nan)

        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for i, reg in enumerate(reg_grid):
            try:
                kernel = matching_kernel(
                    psf_hi,
                    psf_lo,
                    method=method,
                    reg=float(reg),
                    recenter=recenter,
                    wavelet=wavelet,
                    levels=levels,
                    threshold_factor=threshold_factor,
                    noise_sigma=noise_sigma,
                    forward_wavelet_wiener=forward_wavelet_wiener,
                    signal_psd=signal_psd,
                )
            except Exception as exc:
                logger.warning("matching_kernel failed at reg=%g: %s", reg, exc)
                continue
            if not np.all(np.isfinite(kernel)):
                continue
            matched = fftconvolve(psf_hi, kernel, mode="same")
            growth = _encircled_energy(matched, radii)
            profile = _radial_profile(matched, radii)
            growth_error = _growth_curve_error(target_growth, growth)
            core_error = _core_profile_error(target_profile, profile, radii, core_radius)
            l2_error = float(np.nanmean((matched - psf_lo) ** 2))
            kreg, khf, kcancel = _kernel_regularization(
                kernel,
                high_frequency_radius=kernel_high_frequency_radius,
                high_frequency_weight=kernel_high_frequency_weight,
                cancellation_weight=kernel_cancellation_weight,
            )
            score = (
                growth_weight * growth_error
                + core_weight * core_error
                + l2_weight * l2_error
                + kernel_regularization_weight * kreg
            )
            growth_error_grid[i] = growth_error
            core_error_grid[i] = core_error
            l2_error_grid[i] = l2_error
            kernel_regularization_grid[i] = kreg
            kernel_high_frequency_grid[i] = khf
            kernel_cancellation_grid[i] = kcancel
            score_grid[i] = score
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, i, kernel, matched)

        if best is None:
            raise ValueError(
                f"No valid {method!r} kernel produced over reg grid [{reg_grid.min()}, {reg_grid.max()}]."
            )

        score, i, kernel, matched = best
        result = MatchingKernelRegFit(
            method=method,
            reg=float(reg_grid[i]),
            score=float(score),
            kernel=np.asarray(kernel),
            matched_psf=np.asarray(matched),
            reg_grid=reg_grid,
            score_grid=score_grid,
            growth_error_grid=growth_error_grid,
            core_error_grid=core_error_grid,
            l2_error_grid=l2_error_grid,
            kernel_regularization_grid=kernel_regularization_grid,
            kernel_high_frequency_grid=kernel_high_frequency_grid,
            kernel_cancellation_grid=kernel_cancellation_grid,
            radii=radii,
            target_growth=target_growth,
            matched_growth=_encircled_energy(matched, radii),
            target_profile=target_profile,
            matched_profile=_radial_profile(matched, radii),
            extra={
                "wavelet": wavelet,
                "levels": levels,
                "threshold_factor": threshold_factor,
                "noise_sigma": noise_sigma,
                "forward_wavelet_wiener": forward_wavelet_wiener,
                "core_radius": float(core_radius),
                "growth_weight": float(growth_weight),
                "core_weight": float(core_weight),
                "l2_weight": float(l2_weight),
                "kernel_regularization_weight": float(kernel_regularization_weight),
            },
        )

        if diagnostic_path is not None:
            _save_matching_kernel_regularization_diagnostic(
                diagnostic_path,
                result,
                psf_hi,
                psf_lo,
                source_label=source_label,
                target_label=target_label,
                title=diagnostic_title,
                aperture_radius=aperture_radius,
                target_note=diagnostic_note,
            )

        return result

    def auto_matching_kernel_window(
        self,
        other: "PSF" | np.ndarray,
        *,
        fom: str = "c2",
        alpha_grid: np.ndarray | None = None,
        beta_grid: np.ndarray | None = None,
        grid_oversample: int = 2,
        core_radius: float | None = None,
        growth_weight: float = 1.0,
        core_weight: float = 1.0,
        l2_weight: float = 0.0,
        reg_lambda: float = 1e-3,
        kernel_high_frequency_radius: float = 0.7,
        recenter: bool = False,
        diagnostic_path: str | Path | None = None,
        source_label: str = "source PSF",
        target_label: str = "target PSF",
        diagnostic_title: str | None = None,
        aperture_radius: float | None = None,
        return_result: bool = False,
    ) -> SplitCosineBellWindow | tuple[SplitCosineBellWindow, MatchingKernelWindowFit]:
        """Return the optimized split-cosine-bell window for matching to ``other``.

        This is the lightweight API intended for production use.  It wraps
        :meth:`optimize_matching_kernel_window`, maps a named figure of merit
        to the regularization weights, and returns the
        ``SplitCosineBellWindow`` object that can be passed directly to
        :meth:`matching_kernel`.

        Parameters
        ----------
        other : PSF or np.ndarray
            Target PSF that ``self`` should be convolved into.
        fom : str, optional
            Figure-of-merit preset.  The default ``"c2"`` is an alias for
            ``"growth_core_cancel"`` and scores
            ``growth MSE + core MSE + 1e-3 * C(K)^2``.  Other choices are
            ``"growth_core_only"``, ``"growth_core_hf"``, and
            ``"growth_core_hf_cancel"``.
        alpha_grid, beta_grid, grid_oversample, core_radius, growth_weight,
        core_weight, l2_weight, reg_lambda, kernel_high_frequency_radius,
        recenter : optional
            Passed through to :meth:`optimize_matching_kernel_window`.
        diagnostic_path : str or Path, optional
            If provided, write a PNG diagnostic with the score grid, radial
            profile, growth curve, kernel, matched PSF, and residual panels.
        source_label, target_label, diagnostic_title, aperture_radius : optional
            Labels used only in the diagnostic figure.
        return_result : bool, optional
            If ``True``, return ``(window, result)`` where ``result`` is the
            full :class:`MatchingKernelWindowFit`.

        Returns
        -------
        SplitCosineBellWindow or tuple
            Optimized window, optionally paired with the full fit result.
        """
        fom_name, weights = _resolve_kernel_window_fom(fom)
        psf_hi = np.asarray(self.array, dtype=float)
        psf_lo = other.array if isinstance(other, PSF) else np.asarray(other, dtype=float)
        psf_hi_prepared, psf_lo_prepared = _prepare_psf_pair(psf_hi, psf_lo)
        if core_radius is None:
            core_radius = min(psf_hi_prepared.shape) / 4.0

        result = self.optimize_matching_kernel_window(
            psf_lo_prepared,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            grid_oversample=grid_oversample,
            core_radius=core_radius,
            growth_weight=growth_weight,
            core_weight=core_weight,
            l2_weight=l2_weight,
            kernel_regularization_weight=reg_lambda,
            kernel_high_frequency_radius=kernel_high_frequency_radius,
            kernel_high_frequency_weight=weights["kernel_high_frequency_weight"],
            kernel_cancellation_weight=weights["kernel_cancellation_weight"],
            recenter=recenter,
        )
        window = SplitCosineBellWindow(alpha=result.alpha, beta=result.beta)

        if diagnostic_path is not None:
            _save_matching_kernel_window_diagnostic(
                diagnostic_path,
                result,
                psf_hi_prepared,
                psf_lo_prepared,
                fom_name=fom_name,
                core_radius=core_radius,
                source_label=source_label,
                target_label=target_label,
                reg_lambda=reg_lambda,
                title=diagnostic_title,
                aperture_radius=aperture_radius,
            )

        if return_result:
            return window, result
        return window

    def matching_kernel_basis(
        self,
        other: "PSF" | np.ndarray,
        basis: np.ndarray,
        *,
        method: str = "lstsq",
        recenter: bool = True,
    ) -> np.ndarray:
        """Return convolution kernel using a Fourier basis fit."""

        psf_hi = self.array
        psf_lo = other.array if isinstance(other, PSF) else np.asarray(other, dtype=float)

        if psf_hi.shape != psf_lo.shape:
            ny = max(psf_hi.shape[0], psf_lo.shape[0])
            nx = max(psf_hi.shape[1], psf_lo.shape[1])
            shape = (ny, nx)
            psf_hi = pad_to_shape(psf_hi, shape)
            psf_lo = pad_to_shape(psf_lo, shape)

        if basis.shape[:2] != psf_hi.shape:
            basis = np.stack(
                [pad_to_shape(basis[:, :, i], psf_hi.shape) for i in range(basis.shape[2])],
                axis=2,
            )

        kernel, _ = fit_kernel_fourier(psf_hi, psf_lo, basis, method=method)
        if recenter:
            ycen, xcen = centroid_quadratic(kernel, fit_boxsize=5)
            if not np.isnan(ycen) and not np.isnan(xcen):
                cy = (kernel.shape[0] - 1) / 2
                cx = (kernel.shape[1] - 1) / 2
                kernel = shift(kernel, (cy - ycen, cx - xcen), order=3, mode="nearest")
        return kernel

    def _fit_profile(
        self, model_func, default_params, free_params, xc=None, yc=None, result_class=None
    ):
        """Shared fitting logic for both Gaussian and Moffat profiles."""
        from scipy.optimize import least_squares

        y, x = np.indices(self.array.shape)
        cy = (self.array.shape[0] - 1) / 2 if yc is None else yc
        cx = (self.array.shape[1] - 1) / 2 if xc is None else xc

        _, _, sigma_x, sigma_y, theta0 = measure_shape(
            self.array, np.ones_like(self.array, dtype=bool)
        )
        theta0 = ((theta0 + np.pi / 2) % np.pi) - np.pi / 2

        params = default_params.copy()
        params.update(
            {
                "fwhm_x": 2.355 * sigma_x,
                "fwhm_y": 2.355 * sigma_y,
                "theta": theta0,
                "xc": cx,
                "yc": cy,
                "flux": self.array.sum(),  # Initial flux estimate
            }
        )

        # Build optimization parameter list and mapping
        free_list = [p.strip() for p in free_params.split(",")]
        opt_params = []
        param_map = {}
        bounds_lower, bounds_upper = [], []

        for param in free_list:
            if param == "fwhm":  # Special case for symmetric fwhm
                # Use fwhm_x as the initial value for symmetric fitting
                opt_params.append(params["fwhm_x"])
                param_map[param] = len(opt_params) - 1
                bounds_lower.append(1e-3)
                bounds_upper.append(np.inf)
            elif param.startswith("fwhm") and param in params:
                opt_params.append(params[param])
                param_map[param] = len(opt_params) - 1
                bounds_lower.append(1e-3)
                bounds_upper.append(np.inf)
            elif param in params:
                opt_params.append(params[param])
                param_map[param] = len(opt_params) - 1

                # Set bounds based on parameter type
                if param == "beta":
                    bounds_lower.append(0.5)
                    bounds_upper.append(20.0)
                elif param == "theta":
                    bounds_lower.append(-np.pi / 2)
                    bounds_upper.append(np.pi / 2)
                elif param in ["xc", "yc"]:
                    max_val = self.array.shape[1 if param == "xc" else 0] - 1
                    bounds_lower.append(0)
                    bounds_upper.append(max_val)
                elif param == "flux":
                    bounds_lower.append(1e-10)
                    bounds_upper.append(np.inf)

        def residual(p):
            # Map optimization parameters back to model parameters
            current_params = params.copy()

            for param_name, idx in param_map.items():
                if param_name == "fwhm":  # Symmetric case
                    current_params["fwhm_x"] = current_params["fwhm_y"] = p[idx]
                else:
                    current_params[param_name] = p[idx]

            model = model_func(self.array.shape, **current_params)
            return (model - self.array).ravel()

        result = least_squares(residual, opt_params, bounds=(bounds_lower, bounds_upper))

        # Update parameters with fitted values
        for param_name, idx in param_map.items():
            if param_name == "fwhm":  # Symmetric case
                fwhm_val = float(result.x[idx])
                params["fwhm_x"] = params["fwhm_y"] = fwhm_val
            else:
                params[param_name] = float(result.x[idx])

        # Return result with appropriate parameter names
        result_params = {}
        for field_name in result_class.__annotations__:
            if field_name != "shape":  # Skip the shape field
                result_params[field_name] = params[field_name]

        # Add the shape information
        result_params["shape"] = self.array.shape

        return result_class(**result_params)

    def fit_moffat(
        self,
        free_params: str = "fwhm_x,fwhm_y,beta,theta,flux",
        xc: float = None,
        yc: float = None,
    ) -> MoffatFit:
        from .utils import moffat

        def model_func(shape, fwhm_x, fwhm_y, beta, theta, xc, yc, flux, **kwargs):
            return moffat(shape, fwhm_x, fwhm_y, beta, theta, x0=xc, y0=yc, flux=flux)

        return self._fit_profile(model_func, {"beta": 2.5}, free_params, xc, yc, MoffatFit)

    def fit_gaussian(
        self, free_params: str = "fwhm_x,fwhm_y,theta,flux", xc: float = None, yc: float = None
    ) -> GaussianFit:
        from .utils import gaussian

        def model_func(shape, fwhm_x, fwhm_y, theta, xc, yc, flux, **kwargs):
            return gaussian(shape, fwhm_x, fwhm_y, theta, x0=xc, y0=yc, flux=flux)

        return self._fit_profile(model_func, {}, free_params, xc, yc, GaussianFit)


def psf_matching_kernel_basis(
    psf_hi: np.ndarray,
    psf_lo: np.ndarray,
    basis: np.ndarray,
    *,
    recenter: bool = False,
) -> np.ndarray:
    """Match ``psf_hi`` to ``psf_lo`` using basis function fitting."""

    kernel, _ = fit_kernel_fourier(psf_hi, psf_lo, basis)
    if recenter:
        cy = (kernel.shape[0] - 1) / 2
        cx = (kernel.shape[1] - 1) / 2
        ycen, xcen = centroid_quadratic(kernel, xpeak=cx, ypeak=cy, fit_boxsize=5)
        if not np.isnan(ycen) and not np.isnan(xcen):
            kernel = shift(kernel, (cy - ycen, cx - xcen), order=3, mode="nearest")
        else:
            logger.warning("Centroiding failed, kernel not recentered.")
    return kernel


import re


# ---------------------------------------------------------------------
# Minimal EffectivePSF implementation (JWST STDPSF)
# ---------------------------------------------------------------------
# @@@ change this to an overloaded astropy PSF gridded model
# Note: STDPSF headers record slightly different knot positions across
# datasets (e.g. NIRCam LW GRID25 has been seen as both [1, 513, 1025, 1536,
# 2048] and [0, 512, 1024, 1536, 2048]). Likely cause: 1-indexed FITS pixel
# convention in the format spec vs 0-indexed convention some writers use
# (astropy-derived pipelines, etc.) — same physical tile positions, different
# integer offset. Always prefer IPSFX/JPSFY from the header; uniform knots
# are a last-resort fallback when those keywords are missing entirely.
def _read_stdpsf_grid(hdr) -> tuple[list[int], list[int]]:
    """Extract the spatial knot positions (1-indexed detector pixels) from a
    STDPSF header. Returns ``(xk, yk)`` with ``len(xk) == NXPSFS`` and
    ``len(yk) == NYPSFS``. Falls back to uniform knots if IPSFX*/JPSFY* are
    missing — covers GRID1/GRID9/GRID25 and any NXPSFS×NYPSFS layout.
    """
    nxps = int(hdr.get("NXPSFS", 1))
    nyps = int(hdr.get("NYPSFS", 1))
    try:
        xk = [int(hdr[f"IPSFX{i:02d}"]) for i in range(1, nxps + 1)]
    except KeyError:
        xk = list(np.linspace(1, 2048, max(nxps, 1)).astype(int))
    try:
        yk = [int(hdr[f"JPSFY{i:02d}"]) for i in range(1, nyps + 1)]
    except KeyError:
        yk = list(np.linspace(1, 2048, max(nyps, 1)).astype(int))
    return xk, yk


def _stdpsf_oversampling(hdr) -> int:
    """Return the STDPSF oversampling factor, falling back to detector pixels."""
    for key in ("OVERSAMP", "OVERSAMPX"):
        value = hdr.get(key)
        if value is not None:
            try:
                return max(1, int(round(float(value))))
            except (TypeError, ValueError):
                pass
    return 1


def _edge_taper_window(shape: tuple[int, int], width: float) -> np.ndarray:
    """Return a cosine taper that is zero on the finite ePSF grid edge."""
    if width <= 0:
        return np.ones(shape, dtype=np.float32)

    y, x = np.indices(shape, dtype=float)
    distance = np.minimum.reduce([x, y, shape[1] - 1 - x, shape[0] - 1 - y])
    window = np.ones(shape, dtype=np.float32)
    edge = distance < width
    window[edge] = 0.5 * (1.0 - np.cos(np.pi * distance[edge] / width))
    window[distance <= 0] = 0.0
    return window


def _taper_stdpsf_cube(data: np.ndarray, hdr, edge_taper_pixels: float | None) -> np.ndarray:
    """Apply an edge taper to a loaded STDPSF cube in its oversampled pixels.

    ``edge_taper_pixels`` is specified in native detector pixels.  The taper
    removes finite-grid edge flux; the loaded ePSF planes are not renormalized.
    """
    if edge_taper_pixels is None or edge_taper_pixels <= 0:
        return data

    oversampling = _stdpsf_oversampling(hdr)
    width = float(edge_taper_pixels) * oversampling
    window = _edge_taper_window(data.shape[:2], width)
    tapered = np.asarray(data, dtype=np.float32).copy()
    tapered *= window[:, :, None]
    return tapered


class EffectivePSF:

    def __init__(self, **kwargs):
        self.epsf = OrderedDict()
        self.epsf_meta: dict = {}   # per-key dict: {'nxps','nyps','xk','yk'}
        self.extended_epsf = {}
        self.extended_N = None

    #        if kwargs.get("jwst_stdpsf", True):
    #            self.load_jwst_stdpsf()

    def load_jwst_stdpsf(
        self,
        miri_filters=None,
        nircam_sw_filters=None,
        nircam_sw_detectors=None,
        nircam_lw_filters=None,
        nircam_lw_detectors=None,
        miri_extended=True,
        clip_negative=False,
        local_dir=None,
        filter_pattern=None,
        edge_taper_pixels: float | None = 4.0,
        use_astropy_cache=True,
        verbose=False,
    ):
        """Download JWST STDPSF models.

        Parameters
        ----------
        edge_taper_pixels : float or None, optional
            Width of the cosine taper applied once to each loaded ePSF plane,
            in native detector pixels.  The loader converts this to the
            oversampled STDPSF grid using the ``OVERSAMP`` header keyword.
            Set to ``None`` or ``0`` to leave the loaded grids unchanged.
        """

        # If local_dir is specified, use it to find files
        if local_dir is not None and filter_pattern is not None:
            self.filter_pattern = filter_pattern
            p = Path(local_dir)
            if not p.is_dir():
                raise FileNotFoundError(
                    f"local_dir does not exist: {p!s}"
                )
            files_dir = list(p.rglob("*.fits"))
            rx = re.compile(f"{filter_pattern}(?!_EXTENDED)")
            files = [f for f in files_dir if rx.search(os.path.basename(f))]
            if not files:
                logger.warning(
                    "load_jwst_stdpsf: no files matched pattern %r in %s "
                    "(found %d .fits files total). Loaded keys will be empty.",
                    filter_pattern, p, len(files_dir),
                )
            for f in files:
                with fits.open(f) as im:
                    h = im[0].header
                    if verbose:
                        hstr = (
                            f"{h.get('NAXIS1', '?')}x{h.get('NAXIS2', '?')}x{h.get('NAXIS3', '?')} "
                            f"{h.get('INSTRUME', '?')} "
                            f"{h.get('DETECTOR', '?')} "
                            f"{h.get('FILTER',   '?')} "
                            f"{float(h.get('MJD-AVG', 0.0)):6.1f}"
                        )
                        print(f"Loading {f} {hstr}")
                    data = np.array([d.T for d in im[0].data]).T
                    if clip_negative:
                        data[data < 0] = 0
                    data = _taper_stdpsf_cube(data, h, edge_taper_pixels)
                    key = os.path.basename(f).split(".fits")[0]
                    self.epsf[key] = data
                    xk, yk = _read_stdpsf_grid(h)
                    mjd_hdr = float(h.get("MJD-AVG", 0.0))
                    self.epsf_meta[key] = {
                        "nxps": len(xk),
                        "nyps": len(yk),
                        "xk": xk,
                        "yk": yk,
                        "oversample": int(h.get("OVERSAMP", 4)),
                        "mjd": mjd_hdr if mjd_hdr > 0 else None,
                    }
            return

        if miri_filters is None:
            miri_filters = [
                #                "F560W",
                "F770W",
                # "F1000W",
                # "F1130W",
                # "F1280W",
                # "F1500W",
                # "F1800W",
                # "F2100W",
                # "F2550W",
            ]
        if nircam_sw_filters is None:
            nircam_sw_filters = ["F200W"]
        if nircam_sw_detectors is None:
            nircam_sw_detectors = [
                "A1",
                "A2",
                "A3",
                "A4",
                "B1",
                "B2",
                "B3",
                "B4",
            ]
        if nircam_lw_filters is None:
            nircam_lw_filters = ["F444W"]
        if nircam_lw_detectors is None:
            nircam_lw_detectors = ["AL", "BL"]

        base = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/"
        miri_path = (
            "MIRI/EXTENDED/STDPSF_MIRI_{filter}_EXTENDED.fits"
            if miri_extended
            else "MIRI/STDPSF_MIRI_{filter}.fits"
        )

        for filt in miri_filters:
            url = base + miri_path.format(filter=filt)
            try:
                file_obj = download_file(url, cache=use_astropy_cache)
                with fits.open(file_obj) as im:
                    h = im[0].header
                    data = np.array([d.T for d in im[0].data]).T
                    if clip_negative:
                        data[data < 0] = 0
                    data = _taper_stdpsf_cube(data, h, edge_taper_pixels)
                    key = os.path.basename(url.split(".fits")[0])
                    self.epsf[key] = data
                    xk, yk = _read_stdpsf_grid(h)
                    self.epsf_meta[key] = {
                        "nxps": len(xk),
                        "nyps": len(yk),
                        "xk": xk,
                        "yk": yk,
                        "oversample": int(h.get("OVERSAMP", 4)),
                    }
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        sw_path = "NIRCam/SWC/{filter}/STDPSF_NRC{detector}_{filter}.fits"
        for filt in nircam_sw_filters:
            for det in nircam_sw_detectors:
                url = base + sw_path.format(filter=filt, detector=det)
                try:
                    file_obj = download_file(url, cache=use_astropy_cache)
                    with fits.open(file_obj) as im:
                        h = im[0].header
                        data = np.array([d.T for d in im[0].data]).T
                        if clip_negative:
                            data[data < 0] = 0
                        data = _taper_stdpsf_cube(data, h, edge_taper_pixels)
                        key = os.path.basename(url.split(".fits")[0])
                        self.epsf[key] = data
                        xk, yk = _read_stdpsf_grid(h)
                        self.epsf_meta[key] = {
                            "nxps": len(xk),
                            "nyps": len(yk),
                            "xk": xk,
                            "yk": yk,
                            "oversample": int(h.get("OVERSAMP", 4)),
                        }
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

        lw_path = "NIRCam/LWC/STDPSF_NRC{detector}_{filter}.fits"
        for filt in nircam_lw_filters:
            for det in nircam_lw_detectors:
                url = base + lw_path.format(filter=filt, detector=det)
                try:
                    file_obj = download_file(url, cache=use_astropy_cache)
                    with fits.open(file_obj) as im:
                        h = im[0].header
                        data = np.array([d.T for d in im[0].data]).T
                        if clip_negative:
                            data[data < 0] = 0
                        data = _taper_stdpsf_cube(data, h, edge_taper_pixels)
                        key = os.path.basename(url.split(".fits")[0])
                        key = key.replace(f"{det}_", f"{det}ONG_")
                        self.epsf[key] = data
                        xk, yk = _read_stdpsf_grid(h)
                        self.epsf_meta[key] = {
                            "nxps": len(xk),
                            "nyps": len(yk),
                            "xk": xk,
                            "yk": yk,
                            "oversample": int(h.get("OVERSAMP", 4)),
                        }
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

    # do this with PSFgriddedmodel.eval
    # and change hardcoded depenendence on grid size and detector oversampling
    # --- PSF evaluation -------------------------------------------------
    # Bilinear interp on the NXPSFS × NYPSFS spatial tile grid stored in the
    # STDPSF header. Uses self.epsf_meta[key] = {'nxps','nyps','xk','yk'} if
    # available (populated by load_jwst_stdpsf); else falls back to uniformly
    # spaced knots across [1, 2048]. Works for GRID1 (1 tile), GRID9 (3×3),
    # GRID25 (5×5), or any NXPSFS × NYPSFS layout.
    def _interp_grid(self, cube: np.ndarray, key: str,
                      x: float, y: float) -> np.ndarray:
        ntiles = cube.shape[2]
        meta = self.epsf_meta.get(key)
        if meta is not None:
            nxps = meta["nxps"]; nyps = meta["nyps"]
            xk = np.asarray(meta["xk"], float); yk = np.asarray(meta["yk"], float)
        else:
            nxps = nyps = int(np.sqrt(ntiles))
            xk = np.linspace(1, 2048, max(nxps, 1))
            yk = np.linspace(1, 2048, max(nyps, 1))
        if ntiles == 1 or nxps == 1 and nyps == 1:
            return cube[:, :, 0]
        rx = float(np.interp(x, xk, np.arange(nxps)))
        ry = float(np.interp(y, yk, np.arange(nyps)))
        ix0 = int(np.clip(np.floor(rx), 0, nxps - 2))
        iy0 = int(np.clip(np.floor(ry), 0, nyps - 2))
        ix1 = ix0 + 1; iy1 = iy0 + 1
        fx = rx - ix0; fy = ry - iy0
        return (
            (1 - fx) * (1 - fy) * cube[:, :, ix0 + iy0 * nxps]
            + fx * (1 - fy) * cube[:, :, ix1 + iy0 * nxps]
            + (1 - fx) * fy * cube[:, :, ix0 + iy1 * nxps]
            + fx * fy * cube[:, :, ix1 + iy1 * nxps]
        )

    def get_at_position(self, x, y, filter, rot90=0):
        """Interpolate the ePSF grid to a detector position."""
        if filter not in self.epsf:
            raise KeyError(
                f"no stpsf grid loaded for filter key {filter!r}. "
                f"Loaded keys: {sorted(self.epsf)}. "
                f"Call load_jwst_stdpsf(..., filter_pattern=...) for this filter."
            )
        epsf = self.epsf[filter]
        meta = self.epsf_meta.get(filter, {})
        self.eval_psf_oversample = int(meta.get("oversample", 4))

        self.eval_psf_type = "HST/Optical"

        if "MIRI" in filter:
            self.eval_psf_type = "MIRI"
            psf_xy = self._interp_grid(epsf, filter, x, y).T

        elif "NRC" in filter:
            self.eval_psf_type = "NRC"
            psf_xy = self._interp_grid(epsf, filter, x, y).T
        else:
            psf_xy = epsf[:, :, 0]

        if rot90 != 0:
            psf_xy = np.rot90(psf_xy, rot90)

        return psf_xy

    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        """Evaluate the PSF at sub‑pixel offsets."""
        from scipy.ndimage import map_coordinates

        if self.eval_psf_type in ["WFC3/IR", "HST/Optical"]:
            ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
            coords = np.array([50 + 4 * dx[ok], 50 + 4 * dy[ok]])
        else:
            sh = psf_xy.shape
            oversample = int(getattr(self, "eval_psf_oversample", 4))
            y0 = 0.5 * (sh[0] - 1)
            x0 = 0.5 * (sh[1] - 1)
            max_y = min(y0, sh[0] - 1 - y0) / oversample
            max_x = min(x0, sh[1] - 1 - x0) / oversample
            ok = (np.abs(dx) <= max_x) & (np.abs(dy) <= max_y)
            coords = np.array([y0 + oversample * dx[ok], x0 + oversample * dy[ok]])

        interp_map = map_coordinates(psf_xy, coords, order=3)
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map

        if extended_data is not None:
            ok = np.abs(dx) < self.extended_N
            ok &= np.abs(dy) < self.extended_N
            x0 = self.extended_N
            coords = np.array([x0 + dy[ok], x0 + dx[ok]])
            out[ok] += map_coordinates(extended_data, coords, order=0)

        return out


def to_header(wcs, add_naxis=True, relax=True, key=None):
    """Convert WCS to a FITS header with a few extra keywords."""
    hdr = wcs.to_header(relax=relax, key=key)
    if add_naxis:
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs.pixel_shape[0]
            hdr["NAXIS2"] = wcs.pixel_shape[1]
        elif hasattr(wcs, "_naxis1"):
            hdr["NAXIS"] = wcs.naxis
            hdr["NAXIS1"] = wcs._naxis1
            hdr["NAXIS2"] = wcs._naxis2

    if hasattr(wcs.wcs, "cd"):
        for i in [0, 1]:
            for j in [0, 1]:
                hdr[f"CD{i + 1}_{j + 1}"] = wcs.wcs.cd[i][j]

    if hasattr(wcs, "sip") and wcs.sip is not None:
        hdr["SIPCRPX1"], hdr["SIPCRPX2"] = wcs.sip.crpix
    return hdr


def get_slice_wcs(wcs, slx, sly):
    """Slice a WCS while propagating SIP and distortion keywords."""
    nx = slx.stop - slx.start
    ny = sly.stop - sly.start
    swcs = wcs.slice((sly, slx))

    if hasattr(swcs, "_naxis1"):
        swcs.naxis1 = swcs._naxis1 = nx
        swcs.naxis2 = swcs._naxis2 = ny
    else:
        swcs._naxis = [nx, ny]
        swcs._naxis1 = nx
        swcs._naxis2 = ny

    if hasattr(swcs, "sip") and swcs.sip is not None:
        for c in [0, 1]:
            swcs.sip.crpix[c] = swcs.wcs.crpix[c]

    acs = [4096 / 2, 2048 / 2]
    dx = swcs.wcs.crpix[0] - acs[0]
    dy = swcs.wcs.crpix[1] - acs[1]
    for ext in ["cpdis1", "cpdis2", "det2im1", "det2im2"]:
        if hasattr(swcs, ext):
            extw = getattr(swcs, ext)
            if extw is not None:
                extw.crval[0] += dx
                extw.crval[1] += dy
                setattr(swcs, ext, extw)
    return swcs


def stamp_encircled_energy(
    psf: np.ndarray,
    pscale: float,
    *,
    ee_fraction: float | None = None,
) -> dict[str, float]:
    """Measure the *realized* encircled energy of drizzled PSF stamp(s).

    Absolutely calibrated ePSFs drizzle onto an absolute flux scale, so a stamp
    sum *is* an encircled energy.  Measuring it here rather than predicting it
    from the native oversampled ePSF growth curve folds in everything that
    happens on the way to the mosaic grid: the size quantization and parity
    bump applied by :meth:`DrizzlePSF.get_psf_radec`, the drizzle kernel and
    ``pixfrac``, geometric distortion (the local pixel area, not the mean
    linear WCS scale), and the exposure stack actually covering the position.

    Note that the square stamp circumscribes the circle of the same diameter,
    so ``ee_box >= ee_circ`` always, by the corner flux.  ``ee_box`` is the
    quantity that converts a fitted template amplitude into a total flux
    (see :func:`mophongo.pipeline._filter_psf_throughput`); ``ee_circ`` is the
    quantity to compare against tabulated encircled-energy curves.

    Parameters
    ----------
    psf : np.ndarray
        One stamp ``(ny, nx)`` or a cube ``(..., ny, nx)``.  Non-finite pixels
        are treated as zero.  Cubes are reduced by averaging the per-stamp
        scalars, so a spatially varying PSF map returns its mean behaviour.
    pscale : float
        Stamp pixel scale in arcsec.
    ee_fraction : float, optional
        If given, also return the radius enclosing this absolute fraction.

    Returns
    -------
    dict
        ``ee_box``  full-stamp sum;
        ``ee_circ`` sum inside the inscribed circle ``r_circ``;
        ``r_circ``  inscribed-circle radius [arcsec], ``0.5 * min(ny, nx)``
        pixels from the stamp centre;
        ``r_ee``    radius [arcsec] enclosing ``ee_fraction``, or ``nan`` when
        ``ee_fraction`` is ``None`` or no stamp reaches it within ``r_circ``.
        This is the radius of the first radius-sorted pixel whose cumulative
        sum reaches the fraction -- the same convention as
        :meth:`DrizzlePSF._ee_fraction_to_arcsec` -- so it sits up to one
        pixel shell outside the continuous radius.
    """
    arr = np.asarray(psf, dtype=float)
    if arr.ndim < 2:
        raise ValueError(f"psf must be at least 2-D; got shape {arr.shape}")
    if not np.isfinite(pscale) or pscale <= 0.0:
        raise ValueError(f"pscale must be a positive scalar; got {pscale!r}")

    ny, nx = arr.shape[-2:]
    flat = np.where(np.isfinite(arr), arr, 0.0).reshape(-1, ny * nx)

    radius = _radius_image((ny, nx)).ravel()
    order = np.argsort(radius)
    r_sorted = radius[order] * float(pscale)
    r_circ = 0.5 * min(ny, nx) * float(pscale)
    i_circ = max(int(np.searchsorted(r_sorted, r_circ, side="right")) - 1, 0)

    # one stamp at a time: a region map can hold hundreds of large stamps, and
    # the sorted cumulative sum of the whole cube at once is a big allocation
    ee_box, ee_circ, radii = [], [], []
    for stamp in flat:
        cum = np.cumsum(stamp[order])
        ee_box.append(cum[-1])
        ee_circ.append(cum[i_circ])
        if ee_fraction is not None and cum[i_circ] >= ee_fraction:
            # tiny negative wing pixels can make the raw sum non-monotonic
            idx = int(np.searchsorted(np.maximum.accumulate(cum), ee_fraction))
            radii.append(r_sorted[min(idx, len(r_sorted) - 1)])

    return {
        "ee_box": float(np.mean(ee_box)),
        "ee_circ": float(np.mean(ee_circ)),
        "r_circ": r_circ,
        # averaged over the stamps that reach ``ee_fraction`` at all
        "r_ee": float(np.mean(radii)) if radii else float("nan"),
    }


# ---------------------------------------------------------------------
# Drizzle PSF class
# ---------------------------------------------------------------------


class DrizzlePSF:

    def __init__(
        self,
        flt_files=None,
        info=None,
        driz_image=None,
        driz_hdu=None,
        full_flt_weight=True,
        csv_file=None,
        epsf_obj=None,
    ):

        import warnings
        from astropy.wcs import FITSFixedWarning

        warnings.simplefilter("ignore", FITSFixedWarning)

        if info is None:
            info = self.read_wcs_csv(driz_image, csv_file=csv_file)

        self.flt_keys, self.wcs, self.footprint, self.hdrs = info
        self.flt_files = list({k[0] for k in self.flt_keys})

        if epsf_obj is None:
            #            epsf_obj = NEffectivePSF()
            epsf_obj = EffectivePSF()
        self.epsf_obj = epsf_obj

        if driz_hdu is None:
            self.driz_image = driz_image
            self.driz_header = fits.getheader(driz_image)
        else:
            self.driz_image = driz_image
            self.driz_header = driz_hdu.header

        self.driz_wcs = WCS(self.driz_header)
        self.driz_pscale = get_wcs_pscale(self.driz_wcs)
        self.driz_wcs.pscale = self.driz_pscale
        self.driz_footprint = Polygon(self.driz_wcs.calc_footprint())

        # Realized PSF-stamp metadata, filled by ``get_psf_radec`` from the
        # cube it actually produced (see that method's Attributes section).
        self.psf_size: float | None = None
        self.ee_box: float | None = None
        self.ee_circ: float | None = None
        self.r_circ: float | None = None
        self.r_ee: float | None = None
        self.ee_fraction_request: float | None = None

    def load_jwst_stdpsf(self, *args, edge_taper_pixels: float | None = 4.0, **kwargs):
        """Load JWST STDPSF grids through the DrizzlePSF interface.

        Parameters
        ----------
        edge_taper_pixels : float or None, optional
            Native-detector-pixel width of the ePSF edge taper applied once at
            load time. Defaults to 4 native pixels. Set to ``None`` or ``0`` to
            preserve the finite ePSF grid exactly.
        """
        return self.epsf_obj.load_jwst_stdpsf(
            *args,
            edge_taper_pixels=edge_taper_pixels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ePSF key resolution
    # ------------------------------------------------------------------
    def _resolve_epsf_key(
        self,
        pattern: str,
        flt_file: str,
        frame_mjd: float | None,
    ) -> str:
        """Map a user filter *pattern* to a loaded ePSF key.

        Steps:

        1. **Detector substitution.** ``NRC..`` in *pattern* is replaced with
           the actual NIRCam SCA decoded from *flt_file* (e.g. ``NRCA5``).
        2. **Regex match** against ``self.epsf_obj.epsf`` keys. *pattern*
           is treated as a regex (anchored), so wildcards like
           ``MJD.....`` or no MJD token at all both work.
        3. **MJD-nearest selection.** If multiple keys match and the loaded
           grids carry an ``MJD-AVG`` value, the key whose stored MJD is
           closest to *frame_mjd* wins. If *frame_mjd* is ``None`` or none
           of the matches have an MJD, the first match (sorted) is used.

        Returns
        -------
        str
            The resolved ePSF key. If no key matches, the substituted
            pattern itself is returned, preserving the prior behaviour that
            triggers a ``KeyError`` from ``get_at_position`` with a clear
            message.
        """
        # 1. Detector substitution. Regex-search the filename for a
        # NIRCam SCA token so long grizli rate-file names (e.g.
        # ``jw01234_03101_..._nrcalong_..._cal.fits``) decode correctly.
        if "NRC.." in pattern:
            m = re.search(r"nrc(?:a[1-5]|b[1-5]|along|blong)",
                          Path(flt_file).stem, flags=re.IGNORECASE)
            if m is None:
                raise ValueError(
                    f"Cannot decode NIRCam detector from {flt_file!r}"
                )
            det = m.group(0).upper().replace("ALONG", "A5").replace("BLONG", "B5")
            pattern = pattern.replace("NRC..", det)

        # 2. Regex-match against loaded keys
        if pattern in self.epsf_obj.epsf:
            return pattern  # exact literal match, no work needed

        try:
            rx = re.compile(f"^{pattern}$")
        except re.error:
            return pattern  # not a valid regex; let caller fail clearly
        matches = sorted(k for k in self.epsf_obj.epsf if rx.match(k))
        if not matches:
            return pattern
        if len(matches) == 1:
            return matches[0]

        # 3. MJD-nearest among matches
        if frame_mjd is None:
            return matches[0]
        scored = [
            (abs(self.epsf_obj.epsf_meta.get(k, {}).get("mjd") - frame_mjd), k)
            for k in matches
            if self.epsf_obj.epsf_meta.get(k, {}).get("mjd") is not None
        ]
        if not scored:
            return matches[0]
        scored.sort()
        return scored[0][1]

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    # WCS information from CSV
    # ---------------------------------------------------------------------
    @staticmethod
    def read_wcs_csv(drz_file: str, csv_file=None, auto_reconstruct: bool = True):
        """Read exposure WCS info from a CSV table.

        Parameters
        ----------
        drz_file
            Path to the drizzled mosaic FITS.
        csv_file
            Optional explicit ``_wcs.csv`` path. Default: derived from
            *drz_file* by stripping the ``_drz_sci/_drc_sci/_sci``
            suffix.
        auto_reconstruct
            If True (default) and the CSV is missing, regenerate it via
            :func:`mophongo.wcs_recon.reconstruct_wcs` (fetches public
            MAST cal-file header ranges).
        """
        if csv_file is None:
            csv_file = (
                drz_file.split("_drz_sci")[0]
                .split("_drc_sci")[0]
                .split("_sci")[0]
                + "_wcs.csv"
            )
        if not os.path.exists(csv_file):
            if auto_reconstruct:
                from .utils import reconstruct_wcs

                reconstruct_wcs(drz_file, out_csv=csv_file)
            else:
                raise FileNotFoundError(f"CSV file {csv_file} not found")

        tab = Table.read(csv_file, format="csv")
        flt_keys = []
        wcs_dict = {}
        footprints = {}
        hdrs = {}

        for row in tab:
            key = (row["file"], row["ext"])
            hdr = fits.Header()
            for col in tab.colnames:
                val = row[col]
                #                print(col, val, val is np.ma.masked, getattr(val, "masked", False))
                if val is np.ma.masked or getattr(val, "masked", False):
                    continue  # skip masked entries
                hdr[col] = val

            wcs = WCS(hdr, relax=True)
            get_wcs_pscale(wcs)
            wcs.expweight = hdr.get("EXPTIME", 1)

            flt_keys.append(key)
            hdrs[key] = hdr
            wcs_dict[key] = wcs
            footprints[key] = Polygon(wcs.calc_footprint())

        return flt_keys, wcs_dict, footprints, hdrs

    @staticmethod
    def _get_empty_driz(wcs):
        if hasattr(wcs, "pixel_shape") and wcs.pixel_shape is not None:
            sh = wcs.pixel_shape[::-1]
        else:
            if (not hasattr(wcs, "_naxis1")) and hasattr(wcs, "_naxis"):
                wcs._naxis1, wcs._naxis2 = wcs._naxis
            sh = (wcs._naxis2, wcs._naxis1)

        outsci = np.zeros(sh, dtype=np.float32)
        outwht = np.zeros(sh, dtype=np.float32)
        outctx = np.zeros(sh, dtype=np.int32)
        return outsci, outwht, outctx

    def get_driz_cutout(
        self,
        ra,
        dec,
        size=None,
        size_native=None,
        recenter=False,
        search_boxsize=11,
        fit_boxsize=5,
        cutout_data=None,
        verbose=False,
    ):
        """Return a drizzle Cutout2D, including WCS."""

        # default size to size of the ePSF model
        if size is None:
            if size_native is None:  # get from the first filter
                first_key, first_value = next(iter(self.epsf_obj.epsf.items()))
                # NOTE: hard-coded 4× oversampling. DET-sampled grids are upsampled
                # to OS4 in examples/make_stpsfs.ipynb to fit this assumption — see
                # the "Oversample DETECTOR-sampled PSFs to OS4" caveat there: that
                # upsample is spline interpolation, not a true optical OS4, so
                # drizzling via this path does not fully preserve sub-pixel phase.
                size_native = first_value.shape[0] / 4
                if verbose:
                    print(
                        f"Using native size {size_native} from {first_key} assuming 4x oversampling."
                    )

            size = size_native * self.wcs[self.flt_keys[0]].pscale / self.driz_pscale

        size_pix = int(round(size))

        xc, yc = self.driz_wcs.world_to_pixel_values(ra, dec)

        if cutout_data is None:
            data = fits.getdata(self.driz_image)
        else:
            data = cutout_data

        # if data is 3D cube or list of 2D images, take loop over and append to output list
        # check if data is a 2D array or a list of 2D arrays
        if not isinstance(data, list):
            if data.ndim == 2:
                data = [data]

        # Centroid on a LOCAL crop — photutils 1.12 centroid_quadratic
        # (core.py line 245) builds its design matrix in absolute pixel coords,
        # so x² terms of order 10⁸ blow the conditioning and return NaN on
        # large mosaics. Cropping first keeps indices small; add the offset back.
        if recenter:
            xc0, yc0 = xc, yc
            ny_img, nx_img = data[0].shape
            half = max(search_boxsize, fit_boxsize) + 2
            xi, yi = int(np.round(float(xc))), int(np.round(float(yc)))
            x0, x1 = max(0, xi - half), min(nx_img, xi + half + 1)
            y0, y1 = max(0, yi - half), min(ny_img, yi + half + 1)
            local = data[0][y0:y1, x0:x1]
            xc_l, yc_l = centroid_quadratic(
                local, xpeak=xc - x0, ypeak=yc - y0,
                fit_boxsize=fit_boxsize, search_boxsize=search_boxsize,
            )
            if np.isfinite(xc_l) and np.isfinite(yc_l):
                xc, yc = xc_l + x0, yc_l + y0
            else:
                logger.warning(
                    "centroid_quadratic returned NaN at (ra,dec)=(%.6f,%.6f); "
                    "falling back to WCS position (%.2f,%.2f)",
                    ra, dec, xc0, yc0,
                )
                xc, yc = xc0, yc0

        # get cutouts for all images
        cutout_list = []
        for data_i in data:
            cutout = Cutout2D(
                data_i,
                (xc, yc),
                (size_pix, size_pix),
                wcs=self.driz_wcs,
                mode="partial",
                fill_value=0.0,
                copy=True,
            )
            cutout_list.append(cutout)

        return cutout_list if len(cutout_list) > 1 else cutout_list[0]

    # ---------------------------------------------------------------
    def get_psf(
        self,
        ra,
        dec,
        filter=None,
        pixfrac=0.75,
        kernel="square",
        verbose=False,
        wcs_slice=None,
        get_extended=True,
        get_weight=False,
        ds9=None,
        npix=None,
        xphase=0,
        yphase=0,
        taper_alpha=None,
        return_hdul=False,
    ):
        """Drizzle a finite-integral PSF model at ``ra``, ``dec`` onto ``wcs_slice``.

        The returned stamp is the flux that falls on the requested output WCS
        footprint. It is not normalized to unit sum or rescaled after drizzle.
        Pipeline fitting code should convert this native stamp to a unit-sum
        shape when using it for template extension or matching kernels, and
        keep this stamp sum separately as the finite-support throughput.
        """
        if wcs_slice is None:
            wcs_slice = self.driz_wcs.copy()

        # default: adopt the filter pattern used to load the ePSF models
        if filter is None:
            filter = self.epsf_obj.filter_pattern

        outsci, outwht, outctx = self._get_empty_driz(wcs_slice)

        if npix is None:
            # Include every input pixel whose drizzle drop can overlap the
            # requested output footprint. The half-pixfrac margin is in input
            # pixels; without it, small cutouts miss edge-drop flux.
            half_out = 0.5 * max(outsci.shape)
            input_pscale = min(get_wcs_pscale(self.wcs[key]) for key in self.flt_keys)
            npix = int(np.ceil(half_out * self.driz_pscale / input_pscale + 0.5 * pixfrac))

        pix = np.arange(-npix, npix + 1)
        keys = [key for key in self.flt_keys if self.footprint[key].contains(Point(ra, dec))]
        if not keys and self.flt_keys:
            # nearest-frame fallback: a position outside every exposure
            # footprint (e.g. a region-map sliver centroid) gets the PSF of
            # the closest frame instead of an empty stamp
            nearest = min(
                self.flt_keys, key=lambda k: self.footprint[k].distance(Point(ra, dec))
            )
            dist = self.footprint[nearest].distance(Point(ra, dec)) * 3600.0
            logger.warning(
                f"Position {ra:.6f}, {dec:.6f} outside all exposure footprints; "
                f"using nearest frame {nearest[0]}[SCI,{nearest[1]}] ({dist:.1f} arcsec away)."
            )
            keys = [nearest]
        for key in keys:
            file, ext = key

            xy = self.wcs[key].all_world2pix([[ra, dec]], 0)[0]

            xyp = np.asarray(xy, dtype=int)
            dx = xy[0] - int(xy[0]) + xphase
            dy = xy[1] - int(xy[1]) + yphase
            chip_offset = 2051 if ext == 2 else 0

            frame_mjd = float(self.hdrs[key].get("mjd-avg",
                          self.hdrs[key].get("MJD-AVG", 0.0)) or 0.0) or None
            flt_filter = self._resolve_epsf_key(filter, file, frame_mjd)

            if verbose:
                print(
                    f"Position: {xy}, Filter: {flt_filter}, "
                    f"in frame: {file}[SCI,{ext}] mjd={frame_mjd}"
                )

            psf_xy = self.epsf_obj.get_at_position(
                xy[0], xy[1] + chip_offset, filter=flt_filter
            )
            yp, xp = np.meshgrid(pix - dy, pix - dx, indexing="ij")
            extended_data = (
                self.epsf_obj.extended_epsf.get(flt_filter) if get_extended else None
            )
            psf = self.epsf_obj.eval_ePSF(psf_xy, xp, yp, extended_data=extended_data)

            flt_weight = self.wcs[key].expweight
            N = npix
            slx = slice(xyp[0] - N, xyp[0] + N + 1)
            sly = slice(xyp[1] - N, xyp[1] + N + 1)
            if hasattr(flt_weight, "ndim") and flt_weight.ndim == 2:
                wslx = slice(xyp[0] - N + 32, xyp[0] + N + 1 + 32)
                wsly = slice(xyp[1] - N + 32, xyp[1] + N + 1 + 32)
                flt_weight = self.wcs[key].expweight[wsly, wslx]

            psf_wcs = get_slice_wcs(self.wcs[key], slx, sly)
            psf_wcs.pscale = get_wcs_pscale(self.wcs[key])

            with _quiet_drizzle():
                adrizzle.do_driz(
                    psf,
                    psf_wcs,
                    (psf * 0 + flt_weight).astype(outwht.dtype),
                    wcs_slice,
                    outsci,
                    outwht,
                    outctx,
                    1.0,
                    "cps",
                    1,
                    wcslin_pscale=psf_wcs.pscale,
                    uniqid=1,
                    pixfrac=pixfrac,
                    kernel=kernel,
                    fillval=0,
                    stepsize=10,
                    wcsmap=None,
                )

        # taper PSF to avoid discontinuities at the edges and ringing
        if taper_alpha is not None and taper_alpha > 0:
            # rtaper is maximum radial extent of drizzled footprint
            shape = int(np.sqrt((outwht > 0).sum()))
            tukey_taper = pad_to_shape(
                TukeyWindow(alpha=taper_alpha)((shape, shape)), outsci.shape
            )
            outsci *= tukey_taper

        if "psf" not in locals():
            logger.warning(
                f"No PSF found, position possibly outside footprint for {ra}, {dec} in filter {filter}. Returning empty output."
            )

        if return_hdul is True:
            return fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=outsci, header=to_header(wcs_slice)),
                ]
            )
        else:
            return outsci

    def get_psf_radec(
        self,
        positions: list[tuple[float, float]],
        *,
        filter: str | None = None,
        size: float | int | None = None,
        ee_fraction: float | None = None,
        size_quantum_arcsec: float = 0.160,
        parity: str = "even",
        verbose: bool = False,
        kernel: str = "square",
        pixfrac: float = 0.75,
    ) -> np.ndarray:
        """Return a cube of drizzled PSFs evaluated at given coordinates.

        Parameters
        ----------
        positions : list of tuple(float, float)
            World coordinate pairs ``(ra, dec)`` in degrees.
        filter : str, optional
            Filter key or regular expression selecting the PSF model.
        size : float or int, optional
            Cutout size. ``float`` → arcsec, ``int`` → drizzle pixels. If
            ``None`` (default), the size is derived from ``ee_fraction`` when
            provided, otherwise the native DrizzlePSF/ePSF stamp size is used.
        ee_fraction : float, optional
            Target **absolute** encircled-energy fraction, i.e. a fraction of
            the total flux of an infinitely sampled PSF. The loaded ePSFs are
            absolutely calibrated, so their finite stamp already encloses only
            ``EE_stamp < 1``; asking for more than that raises rather than
            silently returning the stamp edge. Use ``ee_fraction >= 1.0`` for
            the full stamp side length. The resulting arcsec size is rounded UP
            to the nearest multiple of ``size_quantum_arcsec`` so that paired
            filters at nested pscales share an integer pixel ratio (clean
            block-binning for kernel matching). Note that this sizing runs on
            the native oversampled ePSF growth curve, because it must produce a
            size *before* anything is drizzled; it is a request, and the
            delivered value is measured afterwards (see Attributes). The
            delivered arcsec size is cached on ``self.psf_size`` so another
            ``DrizzlePSF`` on a different pscale can reuse it via
            ``size=dpsf_other.psf_size``. Ignored when ``size`` is given.
            Defaults to ``None``: with ``size`` also ``None`` the native
            DrizzlePSF stamp size is used.
        size_quantum_arcsec : float, optional
            Grid quantum for the size rounding (default 0.160″ = 2 × 80 mas,
            so the 20 / 40 / 80 mas ladder always block-bins cleanly).
        parity : {"even", "odd", "any"}, optional
            Parity of the output ``size_pix``. Default ``"even"``: an
            odd-pixel size is bumped up by 1 so block-replicating by an
            even factor preserves the cutout shape parity (no half-pixel
            shift on resampling). ``"odd"`` bumps even sizes up by 1 — the
            old default; convenient when the cutout is used as a paste
            stamp because the centre pixel coincides with the requested
            ``(RA, Dec)``. ``"any"`` keeps the requested size as-is.
        verbose : bool, optional
            Emit progress information if ``True``.

        Returns
        -------
        np.ndarray
            Array of shape ``(Npos, size, size)`` containing finite-integral
            drizzled PSFs on the requested output footprint.

        Attributes
        ----------
        Every call overwrites the following, measured on the cube it just
        produced by :func:`stamp_encircled_energy` — realized, not requested,
        so drizzle distortion and the size quantization are already folded in:

        ``psf_size``
            Delivered stamp side [arcsec].
        ``ee_box``, ``ee_circ``, ``r_circ``, ``r_ee``
            Absolute encircled energy in the full square stamp and in the
            inscribed circle, that circle's radius, and the radius enclosing
            ``ee_fraction`` (``nan`` if not requested or not reached).
        ``ee_fraction_request``
            The ``ee_fraction`` argument, kept for provenance.

        These describe the cube **as returned**. A caller that modifies it
        afterwards — e.g. the Gaussian broadening in
        ``Pipeline._drizzle_lo_blurred``, which pushes flux out of the stamp —
        must re-measure with :func:`stamp_encircled_energy` on the final cube.
        """
        if size is None:
            if ee_fraction is None:
                size_pix = None
                self.psf_size = None
                if verbose:
                    logger.info("using native DrizzlePSF stamp size")
            else:
                size_arcsec = self._ee_fraction_to_arcsec(
                    ee_fraction, filter=filter, quantum=size_quantum_arcsec,
                )
                size_pix = int(round(size_arcsec / self.driz_pscale))
                self.psf_size = size_arcsec
                if verbose:
                    logger.info(
                        "ee_fraction=%.3f -> psf_size=%.3f\" (%d pix at p_out=%.3f\")",
                        ee_fraction, size_arcsec, size_pix, self.driz_pscale,
                    )
        elif not isinstance(size, (int, np.integer)):
            size_pix = int(round(size / self.driz_pscale))
            self.psf_size = size_pix * self.driz_pscale
        else:
            size_pix = int(size)
            self.psf_size = size_pix * self.driz_pscale

        if size_pix is not None:
            size_pix = int(np.maximum(9, size_pix))
            if parity == "even" and size_pix % 2 == 1:
                size_pix += 1
            elif parity == "odd" and size_pix % 2 == 0:
                size_pix += 1
            elif parity not in ("even", "odd", "any"):
                raise ValueError(
                    f"parity must be 'even', 'odd', or 'any'; got {parity!r}")
            self.psf_size = size_pix * self.driz_pscale

        psf_cube: list[np.ndarray] = []
        for ra, dec in tqdm(positions, desc="Drizzling PSFs"):
            cutout = self.get_driz_cutout(
                ra,
                dec,
                size=size_pix,
                verbose=verbose,
                recenter=False,
                search_boxsize=11,
            )

            psf = self.get_psf(
                ra=ra,
                dec=dec,
                filter=filter,
                wcs_slice=cutout.wcs,
                kernel=self.driz_header["KERNEL"] if "KERNEL" in self.driz_header else kernel,
                pixfrac=self.driz_header["PIXFRAC"] if "PIXFRAC" in self.driz_header else pixfrac,
                verbose=verbose,
                #                npix=size // 2,
            )
            psf_cube.append(psf)

        cube = np.asarray(psf_cube)
        self._record_realized_ee(cube, ee_fraction)
        return cube

    def _record_realized_ee(
        self, cube: np.ndarray, ee_fraction: float | None
    ) -> None:
        """Store delivered stamp size and encircled energy from ``cube``.

        Called by :meth:`get_psf_radec` on the cube it just drizzled, so the
        cached values describe what came out rather than what was asked for.
        """
        self.ee_fraction_request = ee_fraction
        if cube.ndim != 3 or cube.size == 0:
            # empty positions, or ragged stamps from edge-clipped cutouts
            logger.warning(
                "PSF cube has shape %s; leaving realized PSF metadata unset",
                cube.shape,
            )
            return

        self.psf_size = float(cube.shape[-1]) * self.driz_pscale
        ee = stamp_encircled_energy(
            cube, self.driz_pscale, ee_fraction=ee_fraction
        )
        self.ee_box = ee["ee_box"]
        self.ee_circ = ee["ee_circ"]
        self.r_circ = ee["r_circ"]
        self.r_ee = ee["r_ee"]

        logger.info(
            'psf_size=%.3f" (%d pix): realized ee_box=%.4f, ee_circ(%.3f")=%.4f',
            self.psf_size, cube.shape[-1], self.ee_box, self.r_circ, self.ee_circ,
        )
        if ee_fraction is not None and self.ee_circ < ee_fraction:
            logger.warning(
                'requested ee_fraction=%.4f but the drizzled stamp encloses '
                'only %.4f within r=%.3f"; the native ePSF growth curve '
                "over-promised (drizzle kernel, pixfrac, or distortion)",
                ee_fraction, self.ee_circ, self.r_circ,
            )

    def _ee_fraction_to_arcsec(self,
                               ee_fraction: float,
                               filter: str | None = None,
                               quantum: float = 0.160) -> float:
        """Growth-curve diameter [arcsec] enclosing ``ee_fraction``, quantized UP.

        Uses the first loaded stpsf grid (center-of-detector model). The grids
        are absolutely calibrated: the oversampled array sums to
        ``oversample**2 * EE_stamp`` with ``EE_stamp < 1``, the flux missing to
        infinity plus whatever the load-time edge taper removed. ``ee_fraction``
        is therefore interpreted against that absolute scale, and a request
        above ``EE_stamp`` raises instead of silently returning the stamp edge.

        For ``ee_fraction >= 1``, use the finite ePSF side length rather than the
        corner radius, so full-stamp requests do not add a sqrt(2) diagonal
        buffer. The size is rounded up to the nearest multiple of ``quantum``
        so the 20/40/80 mas ladder always block-bins cleanly.

        This is a *predictor*: it has to return a size before any stamp is
        drizzled, so it can only work on the native oversampled grid, and it
        answers with a circular diameter that then becomes a square side. The
        delivered encircled energy differs — quantizing up grows the radius,
        the square adds corner flux, and drizzling resamples through the real
        distortion — and is measured after the fact by
        :func:`stamp_encircled_energy`.
        """
        ep = self.epsf_obj.epsf
        if not ep:
            raise RuntimeError(
                "no stpsf grids loaded; call load_jwst_stdpsf(...) first"
            )
        # Pick a grid matching ``filter`` if given, else the first one.
        key = None
        if filter is not None:
            for k in ep:
                if re.search(filter, k):
                    key = k
                    break
        if key is None:
            key = next(iter(ep))
        arr = np.asarray(ep[key])          # (Ny, Nx, Npsf_in_grid)
        psf = arr[..., arr.shape[-1] // 2] if arr.ndim == 3 else arr

        # Oversampled pixel scale: stpsf OS4 ⇒ p_native / 4. Look up the native
        # detector pscale from the driz WCS's corresponding filter entry. We
        # fall back to assuming OS4 against the input detector pscale of the
        # first flt if nothing else is known.
        p_native = self.wcs[self.flt_keys[0]].pscale
        oversample = 4   # hard-coded alongside get_driz_cutout; see make_stpsfs.ipynb caveat
        p_os = p_native / oversample

        ny, nx = psf.shape
        if ee_fraction >= 1.0:
            diam_arcsec = max(ny, nx) * p_os
            n_quanta = int(np.ceil(diam_arcsec / quantum))
            return float(n_quanta * quantum)

        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.mgrid[:ny, :nx]
        r = np.hypot(xx - cx, yy - cy)
        order = np.argsort(r.ravel())
        # absolute encircled energy: the oversampled grid carries oversample**2
        # per unit of detector-sampled flux, so divide it back out.
        cum = np.cumsum(psf.ravel()[order]) / float(oversample**2)
        ee_stamp = float(cum[-1])
        if ee_fraction > ee_stamp:
            raise ValueError(
                f"ee_fraction={ee_fraction} exceeds the absolute encircled "
                f"energy of the finite ePSF stamp ({key}: EE_stamp="
                f"{ee_stamp:.4f} over {max(ny, nx) * p_os:.2f}\"). Request "
                f"ee_fraction <= {ee_stamp:.4f}, or ee_fraction >= 1.0 for the "
                f"full stamp side length."
            )
        idx = int(np.searchsorted(cum, ee_fraction))
        idx = min(idx, len(cum) - 1)
        diam_arcsec = 2.0 * r.ravel()[order][idx] * p_os

        n_quanta = int(np.ceil(diam_arcsec / quantum))
        return float(n_quanta * quantum)

    def register(
        self,
        ra: float,
        dec: float,
        filter: str,
        size: int = 15,
        max_iterations: int = 3,
        convergence_threshold: float = 0.05,
        verbose: bool = False,
        kernel: str = "square",
        pixfrac: float = 0.75,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """Register a PSF model to match the data centroid.

        Parameters
        ----------
        cutout : `~astropy.nddata.Cutout2D`
            Data cutout used for registration.
        filter : str
            Filter key or regexp identifying the PSF model.
        max_iterations : int, optional
            Maximum number of centering iterations.
        convergence_threshold : float, optional
            Convergence threshold in pixels.
        verbose : bool, optional
            If ``True`` emit progress information through the logger.

        Returns
        -------
        position : tuple of float
            Final world coordinate position ``(ra, dec)``.
        data : ndarray
            Data cutout used for registration.
        psf_model : ndarray
            Registered PSF model array.
        """

        cutout = self.get_driz_cutout(ra, dec, size=size, recenter=True)

        xi, yi = cutout.input_position_cutout
        for i in range(max_iterations):
            ri, di = cutout.wcs.pixel_to_world_values(xi, yi)

            psf = self.get_psf(
                ra=ri,
                dec=di,
                filter=filter,
                wcs_slice=cutout.wcs,
                kernel=self.driz_header["KERNEL"] if "KERNEL" in self.driz_header else kernel,
                pixfrac=self.driz_header["PIXFRAC"] if "PIXFRAC" in self.driz_header else pixfrac,
                verbose=verbose,
            )

            xc, yc = centroid_quadratic(
                psf,
                xpeak=cutout.input_position_cutout[0],
                ypeak=cutout.input_position_cutout[1],
                fit_boxsize=5,
            )

            if not np.isnan(yc) and not np.isnan(xc):
                dx = cutout.input_position_cutout[0] - xc
                dy = cutout.input_position_cutout[1] - yc
                dr = np.hypot(dx, dy)
            else:
                logger.warning("Centroiding failed, psf not recentered.")

            if verbose:
                print(f"Iteration {i + 1}: Centroid box5 shift: {dx:.3f}, {dy:.3f}, dr= {dr:.3f}")
            xi += dx
            yi += dy

            if dr < convergence_threshold:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
        else:
            if verbose:
                print(f"Maximum iterations {max_iterations} reached")

        ri, di = cutout.wcs.pixel_to_world_values(xi, yi)
        return (ri, di), cutout.data, psf

# ------------------------------------------------------------------
# EffectivePSF  —  now grid-agnostic (MIRI & NIRCam)
# ------------------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
import os, re
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file


class NEffectivePSF:
    """
    Minimal JWST STDPSF loader/evaluator that *learns* the grid break-points
    (IPSFX## / JPSFY##) from every cube it opens, so it works with any SIAF
    release.
    """

    # ──────────────────────────────────────────────────────────────
    def __init__(self):
        self.epsf = OrderedDict()  # key → (Ny, Nx, Ncube)
        self.grid_breaks = {}  # key → {'x':[...], 'y':[...]}
        self.extended_epsf = {}  # unchanged
        self.extended_N = None
        self.eval_psf_type = None  # set in get_at_position

    # ──────────────────────────────────────────────────────────────
    # 1. LOAD CUBES ─ exactly as before, but store the break-points
    # ──────────────────────────────────────────────────────────────
    def _store_cube(self, key, hdu, clip_negative=False):
        """Helper: transpose to (Ny,Nx,N), clip <0, save cube & breaks."""
        dat = np.array([d.T for d in hdu.data]).T
        if clip_negative:
            dat[dat < 0] = 0
        self.epsf[key] = dat

        hdr = hdu.header
        nxps = hdr.get("NXPSFS", 1)
        nyps = hdr.get("NYPSFS", 1)
        xk = [hdr[f"IPSFX{i:02d}"] for i in range(1, nxps + 1)]
        yk = [hdr[f"JPSFY{i:02d}"] for i in range(1, nyps + 1)]
        self.grid_breaks[key] = {"x": xk, "y": yk}

    def load_jwst_stdpsf(
        self,
        *,
        clip_negative=False,
        local_dir=None,
        filter_pattern=None,
        verbose=False,
    ):
        """Load cubes from STScI site *or* a local directory (unchanged API)."""
        # ─── Local directory mode ───────────────────────────────────────
        if local_dir and filter_pattern:
            self.filter_pattern = filter_pattern
            regex = re.compile(filter_pattern, re.IGNORECASE)
            for fp in Path(local_dir).rglob("*.fits"):
                if regex.search(fp.name):
                    with fits.open(fp) as hdul:
                        if verbose:
                            print(f"Loading {fp}")
                        key = fp.stem
                        self._store_cube(key, hdul[0], clip_negative)
            return

        # # ─── Remote STScI buckets ───────────────────────────────────────
        # base = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/"
        # get = lambda url: download_file(url, cache=use_astropy_cache)
        # # ---- MIRI ----
        # miri_fmt = ("MIRI/EXTENDED/STDPSF_MIRI_{filt}_EXTENDED.fits"
        #             if miri_extended else "MIRI/STDPSF_MIRI_{filt}.fits")

    # ──────────────────────────────────────────────────────────────
    # 2. GET AT POSITION ─ use stored break-points, not literals
    # ──────────────────────────────────────────────────────────────
    def get_at_position(self, x, y, filter, rot90=0):
        """Return the oversampled PSF at (x,y) detector coords."""
        epsf = self.epsf[filter]  # cube (Ny,Nx,N)
        br = self.grid_breaks[filter]  # {'x': [...], 'y': [...]}

        # Determine flavour
        self.eval_psf_type = "HST/Optical"
        if "MIRI" in filter:
            self.eval_psf_type = "MIRI"
        if "NRC" in filter:
            self.eval_psf_type = "NRC"

        # ---- generic 2×2 (MIRI) or 3×3 / 5×5 (NIRCam) bilinear blend
        xk, yk = br["x"], br["y"]
        nxps, nyps = len(xk), len(yk)
        ndet = int(np.sqrt(epsf.shape[2]))  # 3×3 → 3 etc.

        # 0-based fractional indices within the grid
        rx = np.interp(x, xk, np.arange(nxps)) - 0
        ry = np.interp(y, yk, np.arange(nyps)) - 0
        ix, iy = np.clip(rx.astype(int), 0, nxps - 2), np.clip(ry.astype(int), 0, nyps - 2)
        fx, fy = rx - ix, ry - iy

        # Bilinear combination
        psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, ix + iy * ndet]
        psf_xy += fx * (1 - fy) * epsf[:, :, ix + 1 + iy * ndet]
        psf_xy += (1 - fx) * fy * epsf[:, :, ix + (iy + 1) * ndet]
        psf_xy += fx * fy * epsf[:, :, ix + 1 + (iy + 1) * ndet]
        psf_xy = psf_xy.T  # your historical transpose

        if rot90:
            psf_xy = np.rot90(psf_xy, rot90)

        return psf_xy

    # ──────────────────────────────────────────────────────────────
    # 3. eval_ePSF unchanged
    # ──────────────────────────────────────────────────────────────
    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        from scipy.ndimage import map_coordinates

        if self.eval_psf_type in ("WFC3/IR", "HST/Optical"):
            ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
            coords = np.array([50 + 4 * dx[ok], 50 + 4 * dy[ok]])
        else:
            sz = (psf_xy.shape[0] - 1) // 4
            x0 = sz * 2
            cen = (x0 - 1) // 2
            ok = (np.abs(dx) <= cen) & (np.abs(dy) <= cen)
            coords = np.array([x0 + 4 * dx[ok], x0 + 4 * dy[ok]])

        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = map_coordinates(psf_xy, coords, order=3)

        # optional extended halo
        if extended_data is not None:
            ok2 = (np.abs(dx) < self.extended_N) & (np.abs(dy) < self.extended_N)
            coords = np.array([self.extended_N + dy[ok2], self.extended_N + dx[ok2]])
            out[ok2] += map_coordinates(extended_data, coords, order=0)
        return out


def jwst_header(dataset_prefix, detector="mirimage", suffix="cal", ext=0):
    """
    dataset_prefix: e.g. 'jw01837001001_06101_00002'
    detector: e.g. 'mirimage', 'nrca1', 'nrcblong', 'nrs1', ...
    suffix: 'cal' or 'rate'
    ext: 0 for primary, or 'SCI' / 1 etc.
    """
    filename = f"{dataset_prefix}_{detector}_{suffix}.fits"
    uri = f"mast:JWST/product/{filename}"
    url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
    # fsspec streaming: headers are fetched without pulling the whole file
    with fits.open(url, use_fsspec=True) as hdul:
        return hdul[ext].header, url


# Try CAL then RATE; return the first that exists
def jwst_probe_headers(dataset_prefix, detector="mirimage", try_suffixes=("cal", "rate"), ext=0):
    last_err = None
    for sfx in try_suffixes:
        try:
            hdr, url = jwst_header(dataset_prefix, detector=detector, suffix=sfx, ext=ext)
            return hdr, url
        except Exception as e:
            last_err = e
            continue
    raise last_err


# Example:
# hdr, url = jwst_probe_headers("jw01837001001_06101_00002", detector="mirimage", try_suffixes=("cal","rate"), ext=0)
# print(url); print(hdr.tostring(sep="\n")[:600])
