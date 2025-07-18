"""Utility functions for analytic profiles and shape measurements."""

from __future__ import annotations

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from shapely.geometry import Polygon
from scipy.special import hermite
from scipy.interpolate import PchipInterpolator
from astropy.utils import lazyproperty
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Moffat1D
from photutils.profiles import RadialProfile, CurveOfGrowth

# model based stuff 
def elliptical_moffat(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Moffat profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
    return amplitude * (1 + r2) ** (-beta)


def elliptical_gaussian(
    y: np.ndarray,
    x: np.ndarray,
    amplitude: float,
    fwhm_x: float,
    fwhm_y: float,
    theta: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Return an elliptical Gaussian profile evaluated on ``x`` and ``y`` grids."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    r2 = (xr / sigma_x) ** 2 + (yr / sigma_y) ** 2
    return amplitude * np.exp(-0.5 * r2)


def measure_shape(data: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``x_c``, ``y_c``, ``sigma_x``, ``sigma_y``, and ``theta`` of ``data``.

    Parameters
    ----------
    data : ndarray
        Pixel data.
    mask : ndarray
        Boolean mask selecting the object pixels.
    """
    y_idx, x_idx = np.indices(data.shape)
    flux = float(data[mask].sum())
    y_c = float((y_idx[mask] * data[mask]).sum() / flux)
    x_c = float((x_idx[mask] * data[mask]).sum() / flux)
    y_rel = y_idx - y_c
    x_rel = x_idx - x_c
    cov_xx = float((data[mask] * x_rel[mask] ** 2).sum() / flux)
    cov_yy = float((data[mask] * y_rel[mask] ** 2).sum() / flux)
    cov_xy = float((data[mask] * x_rel[mask] * y_rel[mask]).sum() / flux)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    sigma_x = float(np.sqrt(vals[0]))
    sigma_y = float(np.sqrt(vals[1]))
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return x_c, y_c, sigma_x, sigma_y, theta


def moffat(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    beta: float,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Moffat PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy
    
    # Convert flux to amplitude analytically
    # For a Moffat profile: flux = amplitude * pi * alpha_x * alpha_y / (beta - 1)
    # where alpha = fwhm / (2 * sqrt(2^(1/beta) - 1))
    factor = 2 ** (1 / beta) - 1
    alpha_x = fwhm_x / (2 * np.sqrt(factor))
    alpha_y = fwhm_y / (2 * np.sqrt(factor))
    amplitude = flux * (beta - 1) / (np.pi * alpha_x * alpha_y)
    
    psf = elliptical_moffat(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        beta,
        theta,
        x0,
        y0,
    )
    return psf


def gaussian(
    size: int | tuple[int, int],
    fwhm_x: float,
    fwhm_y: float,
    theta: float = 0.0,
    x0: float | None = None,
    y0: float | None = None,
    flux: float = 1.0,
) -> np.ndarray:
    """Return a 2-D elliptical Gaussian PSF with specified total flux."""
    if isinstance(size, int):
        ny = nx = size
    else:
        ny, nx = size

    y, x = np.mgrid[:ny, :nx]
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    if x0 is None:
        x0 = cx
    if y0 is None:
        y0 = cy
    
    # Convert flux to amplitude analytically
    # For a Gaussian profile: flux = amplitude * 2 * pi * sigma_x * sigma_y
    # where sigma = fwhm / (2 * sqrt(2 * ln(2)))
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    amplitude = flux / (2 * np.pi * sigma_x * sigma_y)
    
    psf = elliptical_gaussian(
        y,
        x,
        amplitude,
        fwhm_x,
        fwhm_y,
        theta,
        x0,
        y0,
    )
    return psf


import numpy as np
from scipy.special import eval_hermite      # physicists' Hermite

# ------------------------------------------------------------------
# 1. 2-D Gauss–Hermite basis (physicists' convention)
# ------------------------------------------------------------------
def gauss_hermite_basis(order: int, scales, size: int):
    """
    Return an (size, size, Nbasis) cube of 2-D Gauss–Hermite functions.
    H_{i}(x) H_{j}(y) e^{-(x²+y²)/2s²},   0 ≤ i+j ≤ order  for each scale s.
    Zeroth component is unit-sum, all others are zero-sum.
    """
    y, x = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2                        # geometric centre
    basis = []

    for s in scales:
        xn = (x - cx) / s
        yn = (y - cy) / s
        g  = np.exp(-0.5 * (xn**2 + yn**2))        # isotropic Gaussian

        for i in range(order + 1):
            for j in range(order + 1 - i):
                b = g * eval_hermite(i, xn) * eval_hermite(j, yn)

                if i == 0 and j == 0:
                    b /= b.sum()                   # unit DC
                else:
                    b -= b.mean()                  # kill residual DC

                basis.append(b)

    # Stack as (Ny, Nx, Nbasis)
    return np.stack(basis, axis=-1)


# ------------------------------------------------------------------
# 2. Fourier-space kernel fit
# ------------------------------------------------------------------
import numpy as np

def fit_kernel_fourier(img_hi, img_lo, basis):
    """
    Solve  FFT(img_hi) * FFT(basis_k) * c_k  =  FFT(img_lo)
    and return a centred real-space kernel.
    """
    n_pix = img_hi.size                    # = size²

    # 1.  Shift arrays so that the PSF/basis centre is at pixel (0,0) **before** FFT
    f_hi    = np.fft.fft2(np.fft.ifftshift(img_hi))
    f_lo    = np.fft.fft2(np.fft.ifftshift(img_lo))
    f_basis = np.fft.fft2(np.fft.ifftshift(basis, axes=(0, 1)), axes=(0, 1))

    # 2.  Build the least-squares matrix in Fourier space
    nb = basis.shape[-1]
    A  = (f_basis * f_hi[..., None] / n_pix).reshape(-1, nb)   # IDL normalisation
    b  = (f_lo / n_pix).ravel()

    A_ri = np.vstack([A.real, A.imag])
    b_ri = np.concatenate([b.real, b.imag])
    coeffs, *_ = np.linalg.lstsq(A_ri, b_ri, rcond=None)

    # 3.  Back to real space – already centred, so **no fftshift here**
    kernel = np.tensordot(basis, coeffs, axes=([-1], [0]))
    kernel /= kernel.sum()                 # final normalisation

    return kernel, coeffs


def multi_gaussian_basis(scales: list[float], size: int) -> np.ndarray:
    """Return a set of Gaussian basis functions with varying width."""

    gauss_list = [gaussian(size, s, s) for s in scales]
    basis = np.stack(gauss_list, axis=2)
    basis /= basis.sum(axis=(0, 1))
    return basis



# ---------------------------------------------------------------------
# Basic WCS utilities
# ---------------------------------------------------------------------
def get_wcs_pscale(wcs, set_attribute=True):
    """Pixel scale in arcsec from a ``WCS`` object."""
    from numpy.linalg import det

    if isinstance(wcs, fits.Header):
        wcs = WCS(wcs, relax=True)

    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
        detv = det(wcs.wcs.cd)
    else:
        detv = det(wcs.wcs.pc)

    pscale = np.sqrt(np.abs(detv)) * 3600.0
    if set_attribute:
        wcs.pscale = pscale
    return pscale


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


# ---------------------------------------------------------------------
# WCS information from CSV
# ---------------------------------------------------------------------
def read_wcs_csv(drz_file, csv_file=None):
    """Read exposure WCS info from a CSV table."""
    if csv_file is None:
        csv_file = (
            drz_file.split("_drz_sci")[0].split("_drc_sci")[0] + "_wcs.csv"
        )
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found")

    tab = Table.read(csv_file, format="csv")
    flt_keys = []
    wcs_dict = {}
    footprints = {}

    for row in tab:
        key = (row["file"], row["ext"])
        hdr = fits.Header()
        for col in tab.colnames:
            hdr[col] = row[col]

        wcs = WCS(hdr, relax=True)
        get_wcs_pscale(wcs)
        wcs.expweight = hdr.get("EXPTIME", 1)

        flt_keys.append(key)
        wcs_dict[key] = wcs
        footprints[key] = Polygon(wcs.calc_footprint())

    return flt_keys, wcs_dict, footprints


class CircularApertureProfile(RadialProfile):
    """Combined radial profile and curve of growth for a source.

    This class extends :class:`photutils.profiles.RadialProfile` by
    computing a matching :class:`photutils.profiles.CurveOfGrowth` and
    providing convenience methods for normalization, 1D model fitting
    and plotting.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Background subtracted image data.
    xycen : tuple of 2 floats, optional
        ``(x, y)`` pixel coordinate of the source centre. If None, use image center.
    radii : 1D array_like, optional
        Radii defining the edges for the radial profile annuli. If None, use [0, 0.5, 1, 2, 4, ...] pix.
    cog_radii : 1D array_like, optional
        Radii for the curve of growth apertures.  If `None`, the values
        of ``radii[1:]`` are used.
    recenter : bool, optional
        If True, recenter using centroid_quadratic. Default False.
    centroid_kwargs : dict, optional
        Passed to centroid_quadratic. Defaults to {'search_boxsize': 11, 'fit_boxsize': 5}.
    name : str, optional
        Name of the profile used for plot legends.
    norm_radius : float, optional
        Radius at which to normalise both profiles.
    error, mask, method, subpixels : optional
        Passed to :class:`photutils.profiles.RadialProfile` and
        :class:`photutils.profiles.CurveOfGrowth`.
    """

    def __init__(
        self,
        data: np.ndarray,
        xycen: tuple[float, float] | None = None,
        radii: np.ndarray | None = None,
        *,
        cog_radii: np.ndarray | None = None,
        recenter: bool = False,
        centroid_kwargs: dict = None,
        error: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        method: str = "exact",
        subpixels: int = 5,
        name: str | None = None,
        norm_radius: float | None = None,
    ) -> None:
        from photutils.centroids import centroid_quadratic

        # Set centroid kwargs defaults
        if centroid_kwargs is None:
            centroid_kwargs = {"search_boxsize": 11, "fit_boxsize": 5}

        # Default xycen: image center
        if xycen is None:
            ny, nx = data.shape
            xycen = ((nx - 1) / 2, (ny - 1) / 2)

        # Optionally recenter using centroid_quadratic
        if recenter:
            xp, yp = xycen
            xycen = centroid_quadratic(data, xpeak=xp, ypeak=yp, **centroid_kwargs)

        # Default radii: logarithmic bins from 0, 0.5, 1, 2, 4, ... up to edge of image
        if radii is None:
            ny, nx = data.shape
            maxrad = min(nx, ny) / 2
            radii = np.unique(
                np.concatenate([
                    np.array([0, 0.5, 1]),
                    np.logspace(np.log10(2), np.log10(maxrad), num=101)
                ])
            )
            radii = radii[radii <= maxrad]

        # Always set cog_radii = radii[1:] if not provided
        if cog_radii is None:
            cog_radii = radii[1:]

        super().__init__(
            data,
            xycen,
            radii,
            error=error,
            mask=mask,
            method=method,
            subpixels=subpixels,
        )

        self.cog = CurveOfGrowth(
            data,
            xycen,
            cog_radii,
            error=error,
            mask=mask,
            method=method,
            subpixels=subpixels,
        )

        self.name = name
        self.norm_radius = norm_radius

        if norm_radius is not None:
            self.normalize(norm_radius)

    def normalize(self, norm_radius: float | None = None) -> None:
        """Normalize the radial profile and curve of growth."""

        if norm_radius is not None:
            self.norm_radius = norm_radius

        if self.norm_radius is None:
            raise ValueError("norm_radius must be provided")

        rp_val = PchipInterpolator(self.radius, self.profile, extrapolate=False)(
            self.norm_radius
        )
        if np.isfinite(rp_val) and rp_val != 0:
            self.normalization_value *= rp_val
            self.__dict__["profile"] = self.profile / rp_val
            self.__dict__["profile_error"] = self.profile_error / rp_val

        cog_val = PchipInterpolator(
            self.cog.radius, self.cog.profile, extrapolate=False
        )(self.norm_radius)
        if np.isfinite(cog_val) and cog_val != 0:
            self.cog.normalization_value *= cog_val
            self.cog.__dict__["profile"] = self.cog.profile / cog_val
            self.cog.__dict__["profile_error"] = self.cog.profile_error / cog_val

    @lazyproperty
    def moffat_fit(self):
        """Return a 1D Moffat model fitted to the radial profile."""

        profile = self.profile[self._profile_nanmask]
        radius = self.radius[self._profile_nanmask]
        amplitude = float(np.max(profile))
        gamma = float(radius[np.argmax(profile < amplitude / 2)] or 1.0)
        m_init = Moffat1D(amplitude=amplitude, x_0=0.0, gamma=gamma, alpha=2.5)
        m_init.x_0.fixed = True
        fitter = TRFLSQFitter()
        return fitter(m_init, radius, profile)

    @lazyproperty
    def moffat_fwhm(self) -> float:
        """Full width at half maximum (FWHM) of the fitted Moffat."""

        fit = self.moffat_fit
        return 2 * fit.gamma.value * np.sqrt(2 ** (1 / fit.alpha.value) - 1)

    def cog_ratio(self, other: "CircularApertureProfile") -> np.ndarray:
        """Return the ratio of this curve of growth to another."""

        interp = PchipInterpolator(
            other.cog.radius, other.cog.profile, extrapolate=False
        )(self.cog.radius)
        return self.cog.profile / interp

    def _plot_radial_profile(self, ax, color="C0") -> None:
        label = self.name or "profile"
        ax.plot(self.radius, self.profile, label=label, color=color)
        ax.set_yscale("log")
        ax.set_xlabel("Radius (pix)")
        ax.set_ylabel("Normalized Profile")
        ax.axvline(self.gaussian_fwhm / 2, color="C1", ls="--", label="Gauss FWHM")
        ax.axvline(self.moffat_fwhm / 2, color="C2", ls=":", label="Moffat FWHM")
        ax.set_ylim(np.max(self.profile)/1e5, np.max(self.profile) * 1.3)
        ax.legend()

    def _plot_cog(self, ax, color="C0") -> None:
        label = self.name or "profile"
        ax.plot(self.cog.radius, self.cog.profile, label=label, color=color)
        ax.set_xlabel("Radius (pix)")
        ax.set_ylabel("Encircled Energy")
        if self.norm_radius is not None:
            r20 = self.cog.calc_radius_at_ee(0.2)
            r80 = self.cog.calc_radius_at_ee(0.8)
            ax.axvline(r20, color="C1", ls=":")
            ax.axvline(r80, color="C1", ls="--")
        ax.set_ylim(0, 1.05)
        ax.legend()

    def _plot_ratio(self, other: "CircularApertureProfile", ax) -> None:
        ratio = self.cog_ratio(other)
        ax.plot(self.cog.radius, ratio, color="C0")
        ax.axhline(1.0, color="0.5", ls="--")
        ax.set_xlabel("Radius (pix)")
        ax.set_ylabel("COG Ratio")
        ax.set_ylim(0.8, 1.2)

    def plot(
        self,
        *,
        compare_to: "CircularApertureProfile" | None = None,
        show: bool = False,
    ) -> tuple["matplotlib.figure.Figure", list]:
        """Plot radial profile and curve of growth."""

        import matplotlib.pyplot as plt

        ncols = 3 if compare_to is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

        # Main profile: blue
        self._plot_radial_profile(axes[0])
        self._plot_cog(axes[1])

        if compare_to is not None:
            # Compare profile: red
            compare_to._plot_radial_profile(axes[0], color="C3")
            compare_to._plot_cog(axes[1], color="C3")
            self._plot_ratio(compare_to, axes[2])

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
