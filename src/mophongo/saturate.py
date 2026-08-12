"""Saturation-hole detection and PSF-based core repair.

Locate interior weight=0 ``holes`` in a drizzled weight map, fit the local
STPSF amplitude to a donut ring around each hole, and replace the saturated
core pixels with the best-fit PSF model. Designed to run as preprocessing,
outside the photometry pipeline.

This module is image-only: it depends on a drizzled science / weight pair
and a :class:`mophongo.psf.DrizzlePSF`. It deliberately does **not** know
about segmentation maps — the output table carries ``xc, yc, r_out`` per
repaired source so a separate catalog routine (see
:func:`mophongo.catalog.merge_segments_at_holes`) can relabel a segmap
afterwards if needed.

Notes
-----
- Hole detection is parameter-free aside from a minimum-area cut: any
  zero-weight component that does not touch the image border is interior.
- The donut fit is a joint amplitude + sub-pixel shift fit: each iteration
  solves a linearised amplitude/gradient system (:func:`fit_amp_and_shift`),
  applies the recovered shift, and re-drizzles the PSF stamp at the refined
  center until the shift converges.
- The PSF beyond the empirical stamp is unknown; pixels outside PSF support
  are excluded from the fit. If ``r_in`` exceeds the PSF half-width, the
  source is skipped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.stats import mad_std
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    find_objects,
    label as nd_label,
)

from .psf import DrizzlePSF
from .utils import get_slice_wcs
from .utils import get_wcs_pscale

logger = logging.getLogger(__name__)

__all__ = [
    "find_wht_holes",
    "fit_psf_donut",
    "refine_center_from_donut",
    "repair_saturated_holes",
    "plot_repair_diagnostic",
]


# --------------------------------------------------------------------------
# WCS / PSF helpers
# --------------------------------------------------------------------------


def _slice_wcs(wcs: WCS, sly: slice, slx: slice) -> WCS:
    """Return a sub-WCS aligned with the cutout, with ``pscale`` set.

    Uses :func:`mophongo.utils.get_slice_wcs` so SIP/distortion are
    propagated correctly, and sets the ``pscale`` attribute that
    :meth:`DrizzlePSF.get_psf` reads.
    """
    sub = get_slice_wcs(wcs, slx, sly)
    sub.pixel_shape = (slx.stop - slx.start, sly.stop - sly.start)
    sub.pscale = get_wcs_pscale(sub)
    return sub


def _drizzle_psf_on_cut(
    dpsf: DrizzlePSF,
    ra: float,
    dec: float,
    sub_wcs: WCS,
    *,
    filter: str | None = None,
    pixfrac: float | None = None,
    kernel: str | None = None,
) -> np.ndarray:
    """Drizzle the ePSF onto a cutout WCS aligned with the science cutout."""
    px = pixfrac if pixfrac is not None else float(
        dpsf.driz_header.get("PIXFRAC", 0.75)
    )
    kn = kernel if kernel is not None else str(
        dpsf.driz_header.get("KERNEL", "square")
    )
    psf = dpsf.get_psf(
        ra=float(ra), dec=float(dec),
        filter=filter, wcs_slice=sub_wcs,
        pixfrac=px, kernel=kn,
    )
    return np.asarray(psf, dtype=np.float64)


# --------------------------------------------------------------------------
# 1. Hole detection
# --------------------------------------------------------------------------


def find_wht_holes(
    wht: np.ndarray,
    *,
    min_area: int = 1,
    eps_wht: float = 0.0,
    merge_radius: int = 0,
) -> Table:
    """Locate interior zero-weight regions ("holes") in a weight map.

    A hole is a connected component of ``wht <= eps_wht`` that does not
    touch the image border. Edge rejections, chip gaps, and outside-FOV
    regions touch the border and are dropped automatically.

    A heavily saturated star can produce several disconnected zero-weight
    blobs (the deblender fragments the saturated core when the dither
    pattern leaves a few good-weight islands inside). Set
    ``merge_radius > 0`` to morphologically close the binary hole mask
    before labeling, so all fragments of the same star get a single id.
    The reported ``area``/``r_equiv`` use the original (un-dilated) hole
    pixels — only the labeling is grouped.

    Parameters
    ----------
    wht
        2D weight image.
    min_area
        Minimum area in pixels for a hole to be reported.
    eps_wht
        Pixels with ``wht <= eps_wht`` are treated as zero-weight.
    merge_radius
        If > 0, dilate the hole mask by this radius before labeling so
        nearby fragments share a label.

    Returns
    -------
    Table
        Columns: ``id, yc, xc, area, r_equiv, ymin, ymax, xmin, xmax``.
    """
    nonzero = wht > eps_wht
    filled = binary_fill_holes(nonzero)
    holes = filled & ~nonzero
    if merge_radius > 0:
        merged = binary_dilation(holes, iterations=int(merge_radius))
        lab, n = nd_label(merged)
        # restrict labels to actual hole pixels for area/centroid stats
        lab = np.where(holes, lab, 0)
    else:
        lab, n = nd_label(holes)
    rows: list[tuple] = []
    if n > 0:
        objs = find_objects(lab)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            sub = (lab[sl] == i)
            area = int(sub.sum())
            if area < min_area:
                continue
            ys_local, xs_local = np.where(sub)
            ys = ys_local + sl[0].start
            xs = xs_local + sl[1].start
            yc = float(ys.mean())
            xc = float(xs.mean())
            r_eq = float(np.sqrt(area / np.pi))
            rows.append(
                (i, yc, xc, area, r_eq,
                 int(ys.min()), int(ys.max()),
                 int(xs.min()), int(xs.max()))
            )
    tbl = Table(
        rows=rows or None,
        names=["id", "yc", "xc", "area", "r_equiv",
               "ymin", "ymax", "xmin", "xmax"],
        dtype=[int, float, float, int, float, int, int, int, int],
    )
    return tbl


# --------------------------------------------------------------------------
# 2. Segment association
# --------------------------------------------------------------------------


# NOTE: segmentation-map awareness used to live here as
# ``match_segments_to_holes``. It now lives in ``catalog.py`` so this
# module stays purely about image-pixel repair.


# --------------------------------------------------------------------------
# 3. Donut amplitude fit
# --------------------------------------------------------------------------


def refine_center_from_donut(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    center: tuple[float, float],
    r_in: float,
    r_out: float,
    bad_mask: np.ndarray | None = None,
    n_iter: int = 3,
) -> tuple[float, float]:
    """Iteratively refine ``center`` by flux-weighted centroid of the donut.

    PSF wings are azimuthally symmetric about the true star center, so a
    flux-weighted centroid of the (background-subtracted) donut converges
    to the star center even when the saturation hole is asymmetric.
    Background is estimated as the median in an outer annulus.
    """
    yy, xx = np.indices(sci.shape)
    cy, cx = float(center[0]), float(center[1])
    for _ in range(n_iter):
        rr = np.hypot(yy - cy, xx - cx)
        ring = (rr >= r_in) & (rr <= r_out) & (wht > 0)
        if bad_mask is not None:
            ring &= ~bad_mask
        if ring.sum() < 8:
            break
        # background from outer 25% of the donut
        r_bg = r_in + 0.75 * (r_out - r_in)
        bg_ring = ring & (rr >= r_bg)
        bg = float(np.median(sci[bg_ring])) if bg_ring.sum() > 0 else 0.0
        d = sci[ring] - bg
        # use only positive (signal) pixels for the centroid
        m = d > 0
        if m.sum() < 8:
            break
        w = d[m]
        cy_new = float(np.sum(w * yy[ring][m]) / w.sum())
        cx_new = float(np.sum(w * xx[ring][m]) / w.sum())
        if abs(cy_new - cy) < 0.05 and abs(cx_new - cx) < 0.05:
            cy, cx = cy_new, cx_new
            break
        cy, cx = cy_new, cx_new
    return cy, cx


def fit_psf_donut(
    sci: np.ndarray,
    wht: np.ndarray,
    psf: np.ndarray,
    *,
    center: tuple[float, float],
    r_in: float,
    r_out: float,
    bad_mask: np.ndarray | None = None,
    min_pix: int = 10,
    fit_pedestal: bool = False,
) -> dict[str, Any]:
    """Fit ``data ≈ A·psf [+ C]`` on ring ``r_in ≤ r ≤ r_out``.

    Pixels with ``wht <= 0``, ``bad_mask == True``, or ``psf <= 0`` are
    excluded from the ring. With ``fit_pedestal=True`` an additive
    constant ``C`` is added.

    The pedestal and halo are **never** part of the model that gets
    subtracted/replaced; only ``A·psf`` is. They are reported so the
    caller can audit fit quality.

    Also computes ``ρ_psf``, the weighted Pearson correlation between the
    data and the PSF model over the ring — a dimensionless,
    amplitude-invariant shape metric. ``ρ_psf ≈ 1`` means the ring data
    follow the PSF shape (good fit); values well below one mean the light
    on the ring is not PSF-like.

    Returns
    -------
    dict with keys: ``amplitude, amp_err, chi2_red, pedestal,
    rho_psf, n_pix, ring_mask``.
    """
    yy, xx = np.indices(sci.shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    ring = (rr >= r_in) & (rr <= r_out) & (wht > 0) & (psf > 0)
    if bad_mask is not None:
        ring &= ~bad_mask
    n_pix = int(ring.sum())

    out = {
        "amplitude": float("nan"),
        "amp_err": float("nan"),
        "chi2_red": float("nan"),
        "pedestal": 0.0,
        "rho_psf": float("nan"),
        "n_pix": n_pix,
        "ring_mask": ring,
    }
    if n_pix < min_pix:
        return out

    w = wht[ring].astype(np.float64)
    d = sci[ring].astype(np.float64)
    p = psf[ring].astype(np.float64)

    def _pearson(d_arr: np.ndarray, p_arr: np.ndarray) -> float:
        """Weighted Pearson correlation between data and PSF on the ring.
        Amplitude-invariant, sensitive only to *shape* mismatch.
        ρ = 1 → perfect shape match; < 1 → mismatch (halo, kernel, pedestal).
        Restrict to PSF footprint (p > 1e-6 × peak) for robustness."""
        peak = float(p_arr.max()) if p_arr.size else 0.0
        if peak <= 0:
            return float("nan")
        m = p_arr > 1e-6 * peak
        if m.sum() < 5:
            return float("nan")
        wm = w[m]
        sw = float(wm.sum())
        if sw <= 0:
            return float("nan")
        d_m = d_arr[m]
        p_m = p_arr[m]
        d_bar = float(np.sum(wm * d_m) / sw)
        p_bar = float(np.sum(wm * p_m) / sw)
        dd = d_m - d_bar
        pp = p_m - p_bar
        cov = float(np.sum(wm * dd * pp))
        var_d = float(np.sum(wm * dd * dd))
        var_p = float(np.sum(wm * pp * pp))
        if var_d <= 0 or var_p <= 0:
            return float("nan")
        return cov / float(np.sqrt(var_d * var_p))

    def _solve(G: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        M = (G * w[:, None]).T @ G
        rhs = (G * w[:, None]).T @ d
        try:
            c = np.linalg.solve(M, rhs)
            Minv = np.linalg.inv(M)
            return c, Minv
        except np.linalg.LinAlgError:
            return None

    # ── basis columns: PSF + optional pedestal ──
    if not fit_pedestal:
        den = float(np.sum(w * p * p))
        if den <= 0 or not np.isfinite(den):
            return out
        A = float(np.sum(w * d * p) / den)
        resid = d - A * p
        chi2 = float(np.sum(w * resid * resid))
        out["amplitude"] = A
        out["amp_err"] = float(1.0 / np.sqrt(den))
        out["chi2_red"] = chi2 / max(n_pix - 1, 1)
        out["rho_psf"] = _pearson(d, p)
        return out

    G = np.column_stack([p, np.ones_like(p)])
    sol = _solve(G)
    if sol is None:
        return out
    c, Minv = sol
    A = float(c[0])
    if not np.isfinite(A):
        return out
    model = G @ c
    resid = d - model
    chi2 = float(np.sum(w * resid * resid))
    out["amplitude"] = A
    out["amp_err"] = float(np.sqrt(max(Minv[0, 0], 0.0)))
    out["chi2_red"] = chi2 / max(n_pix - 2, 1)
    out["rho_psf"] = _pearson(d, p)
    out["pedestal"] = float(c[1])
    return out


def fit_amp_and_shift(
    sci: np.ndarray,
    wht: np.ndarray,
    psf: np.ndarray,
    *,
    center: tuple[float, float],
    r_in: float,
    r_out: float,
    bad_mask: np.ndarray | None = None,
    min_pix: int = 20,
) -> dict[str, Any]:
    """One linearised step of joint amplitude + sub-pixel shift fitting.

    Solves ``data ≈ A·ψ + B·∂ψ/∂x + C·∂ψ/∂y`` on the donut by weighted
    linear LSQ. From the Taylor expansion ``ψ(x-dx, y-dy) ≈ ψ - dx·∂xψ
    - dy·∂yψ`` the shift is recovered as ``dx = -B/A``, ``dy = -C/A``.
    Caller is expected to apply the shift, re-drizzle the PSF, and call
    again until convergence.
    """
    yy, xx = np.indices(sci.shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    ring = (rr >= r_in) & (rr <= r_out) & (wht > 0) & (psf > 0)
    if bad_mask is not None:
        ring &= ~bad_mask
    n_pix = int(ring.sum())
    fail = {
        "A": float("nan"), "dx": 0.0, "dy": 0.0,
        "chi2_red": float("nan"), "n_pix": n_pix,
    }
    if n_pix < min_pix:
        return fail

    dpx = 0.5 * (np.roll(psf, -1, axis=1) - np.roll(psf, 1, axis=1))
    dpy = 0.5 * (np.roll(psf, -1, axis=0) - np.roll(psf, 1, axis=0))
    # boundary columns/rows from np.roll wrap around — zero them.
    dpx[:, 0] = 0.0
    dpx[:, -1] = 0.0
    dpy[0, :] = 0.0
    dpy[-1, :] = 0.0

    w = wht[ring].astype(np.float64)
    d = sci[ring].astype(np.float64)
    G = np.column_stack([
        psf[ring].astype(np.float64),
        dpx[ring].astype(np.float64),
        dpy[ring].astype(np.float64),
    ])
    GW = G * w[:, None]
    M = GW.T @ G
    rhs = GW.T @ d
    try:
        c = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return fail
    A, B, C = float(c[0]), float(c[1]), float(c[2])
    if not np.isfinite(A) or A == 0.0:
        return fail
    resid = d - G @ c
    chi2 = float(np.sum(w * resid * resid))
    return {
        "A": A,
        "dx": -B / A,
        "dy": -C / A,
        "chi2_red": chi2 / max(n_pix - 3, 1),
        "n_pix": n_pix,
    }


def _donut_significance(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    center: tuple[float, float],
    r_in: float,
    r_out: float,
    sky: float,
    sky_noise: float,
    bad_mask: np.ndarray | None = None,
    inner_frac: float = 0.4,
) -> tuple[float, int]:
    """Median (donut - sky) in units of ``sky_noise``.

    Uses only the inner ``inner_frac`` of the donut where a real PSF
    wing is brightest. For a saturated star this is many tens of sigma;
    for a low-coverage "bay" it is near zero.
    """
    yy, xx = np.indices(sci.shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    r_hi = r_in + float(inner_frac) * (r_out - r_in)
    ring = (rr >= r_in) & (rr <= r_hi) & (wht > 0)
    if bad_mask is not None:
        ring &= ~bad_mask
    n = int(ring.sum())
    if n < 5 or sky_noise <= 0:
        return float("nan"), n
    med = float(np.median(sci[ring]))
    return (med - sky) / sky_noise, n


# --------------------------------------------------------------------------
# 5. Repair orchestration
# --------------------------------------------------------------------------


@dataclass
class RepairDiagnostic:
    """Per-source diagnostic stamps and fit metadata.

    Collected by :func:`repair_saturated_holes` when
    ``return_diagnostics=True`` for holes that reached the fitting stage;
    holes rejected by the buffer-SNR pre-filter or the residual-fraction
    guard get a table row but no diagnostic, and a hole whose fit failed
    gets a stub diagnostic with ``ok=False``.

    Scalar fields: ``id, yc, xc, r_equiv, r_in, r_out`` (geometry);
    ``amplitude, chi2_red, n_pix, n_iter, shift_total, significance,
    center`` plus the no-shift comparison fit ``amplitude_noshift,
    chi2_red_noshift, center_noshift`` (fit results); ``resid_frac,
    ring_snr, buffer_snr, pedestal, rho_psf, data_to_model`` (quality
    metrics); ``fit_mode`` (``"donut"`` or ``"donut+pedestal"``),
    ``action_mode`` (``"repair"`` or ``"subtract"``), ``ok``, ``status``.
    Array fields hold the cutouts and masks used by
    :func:`plot_repair_diagnostic`.
    """

    id: int
    yc: float
    xc: float
    r_equiv: float
    r_in: float
    r_out: float
    amplitude: float
    chi2_red: float
    n_pix: int
    n_iter: int
    shift_total: tuple[float, float]
    significance: float
    center: tuple[float, float]
    # Comparison fit with shift held at (0, 0): drizzle PSF at the hole
    # centroid and fit amplitude only on a donut centered at the hole
    # centroid (no iterative refinement).
    amplitude_noshift: float
    chi2_red_noshift: float
    center_noshift: tuple[float, float]
    sci_cut: np.ndarray = field(repr=False)
    wht_cut: np.ndarray = field(repr=False)
    psf_cut_scaled: np.ndarray = field(repr=False)
    sci_repaired_cut: np.ndarray = field(repr=False)
    hole_mask: np.ndarray = field(repr=False)
    dilated_hole_mask: np.ndarray = field(repr=False)
    ring_mask: np.ndarray = field(repr=False)
    repair_mask: np.ndarray = field(repr=False)
    psf_cut_noshift_scaled: np.ndarray = field(repr=False)
    ring_mask_noshift: np.ndarray = field(repr=False)
    resid_frac: float = float("nan")
    ring_snr: float = float("nan")
    buffer_snr: float = float("nan")
    pedestal: float = 0.0
    rho_psf: float = float("nan")
    fit_mode: str = "donut"
    data_to_model: float = float("nan")
    action_mode: str = "repair"   # or "subtract"
    bad_resid_mask: np.ndarray = field(
        repr=False,
        default_factory=lambda: np.zeros((0, 0), dtype=bool),
    )
    ok: bool = True
    status: str = "ok"


def _native_psf_drz_size(
    dpsf: DrizzlePSF,
    filter_key: str | None = None,
    *,
    fallback: int = 400,
) -> int:
    """Native ePSF FOV in drizzle pixels.

    Used by the subtract path to size each cutout to the actual PSF
    support. For OS4 STDPSF the cube is stored at 4× oversampling, so
    native detector pixels = ``cube_shape / 4``. We then convert to
    drizzle pixels via the WCS pixel-scale ratio.
    """
    epsfs = getattr(dpsf.epsf_obj, "epsf", {})
    if not epsfs:
        return fallback
    cube = epsfs.get(filter_key) if filter_key in epsfs else next(iter(epsfs.values()))
    Ny, Nx = cube.shape[:2]
    native_pix = max(Ny, Nx) // 4
    drz_pscale = float(dpsf.driz_pscale)
    flt_pscale = float(dpsf.wcs[dpsf.flt_keys[0]].pscale)
    return int(np.ceil(native_pix * flt_pscale / drz_pscale))


def repair_saturated_holes(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    dpsf: DrizzlePSF,
    wcs: WCS,
    holes: Table | None = None,
    buffer: float = 2.0,
    factor: float = 2.5,
    fwhm_pix: float = 2.0,
    eps_wht: float = 0.0,
    return_diagnostics: bool = True,
    only_ids: list[int] | None = None,
    fit_shift: bool = True,
    max_shift_iter: int = 5,
    shift_tol: float = 0.05,
    merge_radius: int = 3,
    sat_significance: float = 10.0,
    hole_dilate: int = 2,
    max_resid_frac: float = 1.0,
    min_ring_snr: float = 5.0,
    min_buffer_snr: float = 200.0,
    max_shift_pix: float = 3.0,
    extended_max_data_to_model: float = 1.15,
    mode: str = "repair",
    psf_size_pix_subtract: int | None = None,
    psf_filter: str | None = None,
    psf_pixfrac: float | None = None,
    psf_kernel: str | None = None,
    sky_sample: int = 200_000,
    output_csv: str | Path | None = None,
    plot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Replace saturated cores with a PSF model fit on the surrounding donut.

    For each interior hole:
      ``r_in  = r_equiv + buffer``
      ``r_out = max(2 * fwhm_pix, factor * r_in)``

    The repair pipeline per hole is:

    1. **Buffer pre-filter.** Measure the SNR of the buffer ring around the
       hole; holes whose buffer falls below ``min_buffer_snr`` are skipped
       (low-coverage "bay", not a saturated star). ``sat_significance`` is
       recorded per hole but not applied as a filter.
    2. **Joint amplitude + sub-pixel shift fit.** Drizzle the STPSF onto
       the cutout WCS at the current ``(RA, Dec)``. Run
       :func:`fit_amp_and_shift`; update the position by ``(dx, dy)``;
       re-drizzle. Iterate until ``|dx|, |dy| < shift_tol``.
    3. **Repair.** Fill the dilated saturation footprint with ``A * PSF``;
       restore the weight map to the median donut weight in those
       pixels.

    Parameters
    ----------
    sci, wht
        2D science and weight arrays. Weights are inverse variance; only
        ``wht > eps_wht`` pixels are considered valid.
    dpsf
        :class:`mophongo.psf.DrizzlePSF` configured with the same drizzle
        WCS as ``sci`` and an ePSF model already loaded.
    wcs
        :class:`astropy.wcs.WCS` for ``sci``.
    holes
        Precomputed hole table from :func:`find_wht_holes`. ``None`` runs
        hole detection internally.
    only_ids
        Restrict processing to these hole ids.
    buffer
        Pixels added to ``r_equiv`` to form the donut inner radius ``r_in``.
    factor
        Multiplier on ``r_in`` for the donut outer radius ``r_out``.
    fwhm_pix
        PSF FWHM in pixels; sets the floor ``r_out >= 2 * fwhm_pix``.
    eps_wht
        Pixels with ``wht <= eps_wht`` count as zero-weight.
    return_diagnostics
        Collect per-source :class:`RepairDiagnostic` objects in the return
        value.
    fit_shift, max_shift_iter, shift_tol
        Iterative joint amplitude+position fit controls: ``fit_shift=False``
        for amplitude-only, at most ``max_shift_iter`` drizzle-and-fit
        iterations, converged when the per-iteration shift is below
        ``shift_tol`` pixels.
    merge_radius
        Dilation radius (pixels) used during hole detection so disconnected
        fragments of one saturated core share a label.
    sat_significance
        Minimum median-donut significance (in σ above sky) for the hole
        to be classed as a saturated star. Default 10σ. Currently not
        applied as a filter: the measured significance is recorded in the
        output and the acceptance filter uses ``min_buffer_snr`` instead.
    hole_dilate
        Dilation (pixels) of the zero-weight mask; the dilated footprint
        defines the repair region and the inner boundary of the fit ring.
    max_resid_frac
        Intended threshold for the residual fraction. Currently not
        consulted: the applied guard is hard-coded to reject fits with
        residual fraction above 1.0 (no action taken on the image).
    min_ring_snr
        Intended minimum median ring SNR. Currently not applied: the
        measured ring SNR is stored in the diagnostics only.
    min_buffer_snr
        Minimum median sky-subtracted flux of the buffer pixels, in units
        of the global sky noise, for the hole to be treated as genuine
        saturation. The main pre-filter.
    max_shift_pix
        Hard cap on the cumulative fitted position shift, in pixels.
    extended_max_data_to_model
        If ``Σ data / Σ (A·ψ)`` over the donut exceeds this ratio, refit
        with an additive pedestal to absorb host-galaxy flux (the pedestal
        is reported, never subtracted).
    mode
        ``"repair"`` fills the saturated core with the model;
        ``"subtract"`` removes ``A·ψ`` over the full cutout and blanks the
        core plus strongly discrepant residual pixels.
    psf_size_pix_subtract
        Cutout size in drizzle pixels for subtract mode. ``None`` uses
        ``min(400, native ePSF field of view)``.
    psf_filter, psf_pixfrac, psf_kernel
        Forwarded to :meth:`DrizzlePSF.get_psf`. ``None`` falls back to
        the dpsf defaults (drizzle-header ``PIXFRAC``/``KERNEL``).
    sky_sample
        Number of valid pixels sampled to estimate sky and sky_noise
        (``mad_std``). 0 → use all pixels.
    output_csv
        If given, write the fit table as CSV to this path.
    plot_dir
        If given, write per-source diagnostic PNGs to this directory
        (only for sources for which a fit was attempted).

    Returns
    -------
    dict
        ``{"sci", "wht", "fits", "diagnostics", "holes",
        "sky", "sky_noise"}``. ``"sci"`` is the repaired science image
        (``float32``), ``"wht"`` the repaired weight map. ``"fits"`` is an
        astropy Table with one row per hole, including rejected ones, with
        columns ``id, yc, xc, r_equiv, r_in, r_out, amplitude, amp_err,
        chi2_red, n_pix, n_iter, shift_x, shift_y, significance,
        buffer_snr, flux_added, pedestal, fit_mode, data_to_model,
        amplitude_noshift, chi2_red_noshift, ok, status``; ``ok=False``
        rows record why a hole was skipped in ``status``.
    """
    if holes is None:
        holes = find_wht_holes(wht, eps_wht=eps_wht, merge_radius=merge_radius)

    if only_ids is not None:
        keep = np.isin(np.asarray(holes["id"]), list(only_ids))
        holes = holes[keep]

    # Robust global sky and noise from valid pixels.
    valid_mask = wht > eps_wht
    valid = sci[valid_mask]
    if sky_sample and valid.size > sky_sample:
        rng = np.random.default_rng(0)
        valid = rng.choice(valid, size=sky_sample, replace=False)
    sky = float(np.median(valid)) if valid.size else 0.0
    sky_noise = float(mad_std(valid, ignore_nan=True)) if valid.size else 1.0
    if not np.isfinite(sky_noise) or sky_noise <= 0:
        sky_noise = 1.0
    logger.info(
        "[saturate] sky=%.4g  sky_noise=%.4g  (n_valid=%d)",
        sky, sky_noise, int(valid_mask.sum()),
    )

    sci_rep = sci.astype(np.float32, copy=True)
    wht_rep = wht.astype(np.float32, copy=True)

    H, W = sci.shape
    fit_rows: list[tuple] = []
    diags: list[RepairDiagnostic] = []
    hole_mask_full = (wht <= eps_wht)

    # In subtract mode the cutout is sized to capture the bright halo
    # and diffraction spikes (~5-8" for JWST), but capped well below
    # the native large-PSF FOV (30") because the wing flux beyond ~10"
    # is below sky and subtracting there just adds noise. Default 200
    # drz-pix half (=8" at 40mas) is a good compromise.
    if mode == "subtract":
        if psf_size_pix_subtract is not None:
            half_subtract = int(psf_size_pix_subtract // 2)
        else:
            half_subtract = min(
                200, _native_psf_drz_size(dpsf, psf_filter) // 2,
            )
    else:
        half_subtract = 0

    for row in holes:
        hid = int(row["id"])
        yc0 = float(row["yc"])
        xc0 = float(row["xc"])
        r_eq = float(row["r_equiv"])
        r_in = r_eq + float(buffer)
        r_out = max(2.0 * float(fwhm_pix), float(factor) * r_in)
        # Cutout sized to the donut (repair) or to the large-PSF FOV (subtract).
        # In BOTH modes the FIT itself uses only the [r_in, r_out] donut
        # — the larger cutout in subtract mode is just the canvas onto
        # which we subtract A·ψ at the end.
        half = max(int(np.ceil(r_out)) + 2, half_subtract)

        y_int = int(round(yc0))
        x_int = int(round(xc0))
        sly = slice(max(0, y_int - half), min(H, y_int + half + 1))
        slx = slice(max(0, x_int - half), min(W, x_int + half + 1))
        sci_cut = sci[sly, slx].astype(np.float64)
        wht_cut = wht[sly, slx].astype(np.float64)
        if sci_cut.size == 0:
            continue
        cy = yc0 - sly.start
        cx = xc0 - slx.start
        hole_mask_cut = hole_mask_full[sly, slx]
        # In subtract mode the cutout is large (~750 px) and likely
        # contains many unrelated wht=0 pixels (other saturated stars,
        # CRs, chip edges). Keep only the connected component of the
        # source's own hole — otherwise the dilation buffer ends up
        # sampling pixels far from the source and buffer_snr is biased
        # toward sky.
        if hole_mask_cut.any():
            cy_idx = int(round(yc0 - sly.start))
            cx_idx = int(round(xc0 - slx.start))
            cy_idx = max(0, min(hole_mask_cut.shape[0] - 1, cy_idx))
            cx_idx = max(0, min(hole_mask_cut.shape[1] - 1, cx_idx))
            lbl, _ = nd_label(hole_mask_cut)
            center_lbl = int(lbl[cy_idx, cx_idx])
            if center_lbl == 0:
                # Centre pixel itself is not in any hole; pick the
                # nearest connected component instead.
                ys, xs = np.where(hole_mask_cut)
                if ys.size:
                    d2 = (ys - cy_idx) ** 2 + (xs - cx_idx) ** 2
                    j = int(np.argmin(d2))
                    center_lbl = int(lbl[ys[j], xs[j]])
            if center_lbl > 0:
                hole_mask_cut = (lbl == center_lbl)
            else:
                hole_mask_cut = np.zeros_like(hole_mask_cut)
        # Dilate the wht=0 mask: the repair region and the inner boundary
        # of the fitting ring follow the actual hole shape rather than a
        # circle. This avoids misalignment when the saturation footprint
        # is irregular.
        if hole_dilate > 0 and hole_mask_cut.any():
            dilated_mask = binary_dilation(
                hole_mask_cut, iterations=int(hole_dilate),
            )
        else:
            dilated_mask = hole_mask_cut.copy()
        # Buffer pixels: the ``hole_dilate``-thick annulus added by the
        # dilation, sitting just outside the original wht=0 hole. For a
        # genuinely saturated star these pixels are still very bright
        # (the wings of the saturated core); for a 1-px CR or a low-
        # coverage bay near a galaxy edge they sit close to sky.
        # Median (sky-subtracted) flux in those pixels, expressed as
        # multiples of ``sky_noise``, separates the two cases cleanly.
        buffer_mask = dilated_mask & ~hole_mask_cut
        if buffer_mask.any() and sky_noise > 0:
            buffer_snr = float(
                (np.median(sci_cut[buffer_mask]) - sky) / sky_noise
            )
        else:
            buffer_snr = float("nan")

        # 1) Saturation pre-filter — the buffer-pixel SNR is the
        # cleanest single discriminator between real saturation cores
        # and CR / low-coverage blobs. Donut significance is recorded
        # for reference only (no longer a filter).
        sig, n_sig = _donut_significance(
            sci_cut, wht_cut,
            center=(cy, cx), r_in=r_in, r_out=r_out,
            sky=sky, sky_noise=sky_noise, bad_mask=dilated_mask,
        )
        if not np.isfinite(buffer_snr) or buffer_snr < min_buffer_snr:
            # Pre-filter: no fit attempted, so no diagnostic. The CSV row
            # records the rejection so the source is still accounted for.
            # ASCII only: the status column must survive a FITS table write.
            status = (f"low buffer SNR ({buffer_snr:.1f} sig) - "
                      f"not saturation")
            fit_rows.append(_failed_row(
                hid, yc0, xc0, r_eq, r_in, r_out, status,
                significance=sig, buffer_snr=buffer_snr,
            ))
            continue

        # Hard upper bound on cumulative shift (default 2 px). Prevents
        # the iterative fit from walking off the actual hole onto a
        # nearby galaxy when the local model is poor. Tying this to the
        # hole size was too restrictive for tiny holes — a flat cap is
        # both simpler and correct (the wht=0 hole is always within ~1
        # px of the true PSF centre to begin with).
        shift_cap = float(max_shift_pix)

        # 2) Iterative joint amp + shift fit.
        sub_wcs = _slice_wcs(wcs, sly, slx)
        try:
            ra, dec = (float(v) for v in sub_wcs.pixel_to_world_values(cx, cy))
        except Exception:
            ra, dec = (float(v) for v in wcs.pixel_to_world_values(xc0, yc0))

        psf_cut: np.ndarray | None = None
        n_iter = 0
        shift_total = np.zeros(2, dtype=float)
        last = {"A": float("nan"), "dx": 0.0, "dy": 0.0,
                "chi2_red": float("nan"), "n_pix": 0}
        for n_iter in range(1, max_shift_iter + 1):
            try:
                psf_cut = _drizzle_psf_on_cut(
                    dpsf, ra, dec, sub_wcs,
                    filter=psf_filter, pixfrac=psf_pixfrac, kernel=psf_kernel,
                )
            except Exception as exc:
                logger.warning("hole %d: drizzle PSF failed: %s", hid, exc)
                psf_cut = None
                break

            last = fit_amp_and_shift(
                sci_cut, wht_cut, psf_cut,
                center=(cy, cx), r_in=0.0, r_out=r_out,
                bad_mask=dilated_mask,
            )
            if not np.isfinite(last["A"]) or last["A"] <= 0:
                break
            if not fit_shift:
                break
            dx, dy = last["dx"], last["dy"]
            # Hard-cap cumulative shift so a bad local fit can't walk
            # the model off the actual hole.
            proposed = shift_total + np.array([dx, dy])
            mag = float(np.hypot(*proposed))
            if mag > shift_cap:
                # Scale the proposed cumulative shift down to the cap and
                # apply only the residual step that gets us there.
                proposed = proposed * (shift_cap / mag)
                dx = float(proposed[0] - shift_total[0])
                dy = float(proposed[1] - shift_total[1])
            cy += dy
            cx += dx
            shift_total += (dx, dy)
            if abs(dx) < shift_tol and abs(dy) < shift_tol:
                break
            try:
                ra, dec = (float(v) for v in sub_wcs.pixel_to_world_values(cx, cy))
            except Exception:
                break

        # Final amplitude-only check at converged PSF (records ring_mask).
        if psf_cut is None or not np.isfinite(last["A"]) or last["A"] <= 0:
            status = ("fit-fail" if last["A"] != last["A"]
                      else "negative amplitude")
            fit_rows.append(_failed_row(
                hid, yc0, xc0, r_eq, r_in, r_out, status,
                significance=sig, buffer_snr=buffer_snr,
                shift=tuple(shift_total),
            ))
            if return_diagnostics:
                diags.append(_stub_diag(
                    hid=hid, yc0=yc0, xc0=xc0, r_eq=r_eq,
                    r_in=r_in, r_out=r_out,
                    sci_cut=sci_cut, wht_cut=wht_cut,
                    hole_mask_cut=hole_mask_cut, dilated_mask=dilated_mask,
                    cy=cy, cx=cx, status=status, significance=sig,
                    shift_total=tuple(shift_total),
                    psf_cut_scaled=(psf_cut * 0 if psf_cut is not None else None),
                    buffer_snr=buffer_snr,
                ))
            continue

        # Re-drizzle once more at the converged (cy, cx) so the PSF
        # peak exactly matches the position used for the repair (the
        # last loop iteration applied a shift after its drizzle step).
        if fit_shift and (abs(last["dx"]) > 1e-3 or abs(last["dy"]) > 1e-3):
            try:
                ra, dec = (float(v) for v in sub_wcs.pixel_to_world_values(cx, cy))
                psf_cut = _drizzle_psf_on_cut(
                    dpsf, ra, dec, sub_wcs,
                    filter=psf_filter, pixfrac=psf_pixfrac, kernel=psf_kernel,
                )
            except Exception:
                pass

        final = fit_psf_donut(
            sci_cut, wht_cut, psf_cut,
            center=(cy, cx), r_in=0.0, r_out=r_out,
            bad_mask=dilated_mask,
        )
        A = float(final["amplitude"])
        if not np.isfinite(A) or A <= 0 or final["n_pix"] < 10:
            status = "final amp fit failed"
            fit_rows.append(_failed_row(
                hid, yc0, xc0, r_eq, r_in, r_out, status,
                significance=sig, buffer_snr=buffer_snr,
                shift=tuple(shift_total),
            ))
            if return_diagnostics:
                diags.append(_stub_diag(
                    hid=hid, yc0=yc0, xc0=xc0, r_eq=r_eq,
                    r_in=r_in, r_out=r_out,
                    sci_cut=sci_cut, wht_cut=wht_cut,
                    hole_mask_cut=hole_mask_cut, dilated_mask=dilated_mask,
                    cy=cy, cx=cx, status=status, significance=sig,
                    shift_total=tuple(shift_total),
                    psf_cut_scaled=psf_cut * 0,
                    ring_mask=final["ring_mask"],
                    buffer_snr=buffer_snr,
                ))
            continue

        # Residual-fraction filter (a.k.a. mean absolute deviation
        # relative to the model). χ² is dominated by systematics so it
        # is a poor acceptance metric; instead require ``Σ|d - A·ψ|``
        # in the ring to be a small fraction of ``Σ|A·ψ|``. When the
        # model can't account for the data — e.g. a hole that turned
        # out to be in the wing of a brighter neighbour — this ratio
        # blows up well above unity.
        ring_final = final["ring_mask"]
        if ring_final.any():
            model_ring = A * psf_cut[ring_final]
            data_ring = sci_cut[ring_final]
            r_ring = data_ring - model_ring
            denom = float(np.sum(np.abs(model_ring)))
            resid_frac = (float(np.sum(np.abs(r_ring))) / denom
                          if denom > 0 else float("inf"))
            # Median ring SNR: a real saturated star has bright wings;
            # a wing-of-neighbour speck has most ring pixels at sky.
            ring_snr = (float(np.median(data_ring) - sky) / sky_noise
                        if sky_noise > 0 else float("nan"))
            # Embedded-source check (AGN-in-galaxy):
            #   data_to_model = Σ data / Σ A·ψ  over the donut.
            # When > extended_max_data_to_model, the host's smooth flux
            # is biasing the amplitude. Refit on the dilation buffer
            # ring with a +pedestal term so ``A`` reflects the point
            # source alone.
            sum_data_ring = float(np.sum(data_ring))
            sum_model_ring = float(np.sum(model_ring))
            data_to_model = (
                sum_data_ring / sum_model_ring
                if sum_model_ring > 0 else float("inf")
            )
        else:
            resid_frac = float("nan")
            ring_snr = float("nan")
            data_to_model = float("nan")

        # Pedestal switch on the SAME donut [r_in, r_out] used for the
        # FOM. ``A`` and ``C`` come from a 2-column LSQ. The pedestal
        # absorbs an extended host-galaxy continuum so ``A`` reflects
        # the point source alone. Triggered whenever
        # ``Σdata/Σ(A·ψ) > extended_max_data_to_model`` over that ring.
        fit_mode = "donut"
        pedestal = 0.0
        if (np.isfinite(data_to_model)
                and data_to_model > extended_max_data_to_model):
            buf = fit_psf_donut(
                sci_cut, wht_cut, psf_cut,
                center=(cy, cx), r_in=0.0, r_out=r_out,
                bad_mask=dilated_mask, fit_pedestal=True,
            )
            A_buf = float(buf["amplitude"])
            if np.isfinite(A_buf) and A_buf > 0:
                A = A_buf
                pedestal = float(buf["pedestal"])
                fit_mode = "donut+pedestal"
                final = buf

        cumshift_mag = float(np.hypot(*shift_total))

        # Comparison: amplitude-only fit at the original hole centroid
        # (shift held at 0, 0). Drizzles a fresh PSF at the hole position.
        cy_h = yc0 - sly.start
        cx_h = xc0 - slx.start
        try:
            ra_h, dec_h = (
                float(v) for v in sub_wcs.pixel_to_world_values(cx_h, cy_h)
            )
            psf_noshift = _drizzle_psf_on_cut(
                dpsf, ra_h, dec_h, sub_wcs,
                filter=psf_filter, pixfrac=psf_pixfrac, kernel=psf_kernel,
            )
        except Exception as exc:
            logger.warning("hole %d: noshift drizzle failed: %s", hid, exc)
            psf_noshift = None
        if psf_noshift is not None:
            ns = fit_psf_donut(
                sci_cut, wht_cut, psf_noshift,
                center=(cy_h, cx_h), r_in=0.0, r_out=r_out,
                bad_mask=dilated_mask,
            )
            A_ns = float(ns["amplitude"])
            chi2_ns = float(ns["chi2_red"])
            ring_ns = ns["ring_mask"]
        else:
            A_ns = float("nan")
            chi2_ns = float("nan")
            ring_ns = np.zeros_like(hole_mask_cut, dtype=bool)
            psf_noshift = np.zeros_like(sci_cut)

        # 3) Apply the model. SAME FIT in both modes — only the action
        # differs:
        #   * mode='repair'   → fill the dilated saturation footprint
        #     with A·ψ; restore wht of those pixels to median-donut wht.
        #   * mode='subtract' → subtract A·ψ from the entire cutout
        #     (PSF wings, spikes, halo). Saturation cores and bad-residual
        #     pixels are blanked by setting both sci and wht to zero, so
        #     downstream photometry skips them.
        # Guard: if the residual fraction over the fit ring is >100%,
        # the model doesn't describe the data — leaving the image
        # untouched is safer than corrupting it. We still record the
        # row (with status=bad-fit) so the source is accounted for.
        if np.isfinite(resid_frac) and resid_frac > 1.0:
            status = (f"bad fit (|resid|/|fit|={resid_frac:.2f}>1) - "
                      f"no action")
            fit_rows.append(_failed_row(
                hid, yc0, xc0, r_eq, r_in, r_out, status,
                significance=sig, buffer_snr=buffer_snr,
                shift=tuple(shift_total),
            ))
            continue

        yy, xx = np.indices(sci_cut.shape)
        rr = np.hypot(yy - cy, xx - cx)
        inside = dilated_mask | hole_mask_cut
        sci_view = sci_rep[sly, slx]
        wht_view = wht_rep[sly, slx]
        flux_before = float(np.sum(sci_view[inside]))
        # Residual-driven mask used in subtract mode (sci=wht=0 there).
        # Region of validity:
        #   * pedestal mode (extended host) → restrict to the dilated
        #     saturation footprint. The wing is dominated by host
        #     galaxy flux so we must not mask it.
        #   * default mode → wherever the PSF model is above the
        #     surface-brightness threshold (so we have something
        #     meaningful to fit). Excludes pixels deep in the noise
        #     where the model contributes nothing.
        # Within the region, blank pixels where the absolute residual
        # exceeds 1.5 × sky_noise.
        resid_full = sci_cut.astype(np.float64) - A * psf_cut - pedestal
        sigma = float(sky_noise)
        # Mask region:
        #   pedestal mode → dilated saturation footprint only
        #   else          → within r_out AND A·ψ > 1 % of model peak
        # Mask criterion: |residual| > 1.5 × sky_noise within that region.
        if "pedestal" in fit_mode:
            mask_region = dilated_mask
        else:
            yy, xx = np.indices(sci_cut.shape)
            rr_local = np.hypot(yy - cy, xx - cx)
            model = A * psf_cut
            psf_peak = float(model.max()) if model.size else 0.0
            mask_region = (rr_local <= r_out) & (model > 1e-4 * psf_peak)
        bad_resid_mask = mask_region & (np.abs(resid_full) > 1.5 * sigma)

        if mode == "subtract":
            # Subtract A·ψ over the entire cutout footprint.
            # Pedestal is reported but NOT removed (matches existing
            # convention: pedestal absorbs host flux but the host stays).
            sub = sci_view.astype(np.float64) - A * psf_cut
            sci_view[:, :] = sub.astype(sci_view.dtype)
            # Blank wht=0 saturation cores AND the bad-residual
            # pixels: set both sci and wht to zero. Downstream
            # photometry will skip these pixels because wht=0.
            blank = hole_mask_cut | bad_resid_mask
            sci_view[blank] = 0
            wht_view[blank] = 0
            flux_after = float(np.nansum(sci_view[inside]))
            flux_added = flux_after - flux_before
        else:
            # Repair: replace inner pixels with model.
            sci_view[inside] = (A * psf_cut[inside]).astype(sci_view.dtype)
            flux_after = float(np.sum(sci_view[inside]))
            flux_added = flux_after - flux_before
            donut_w = wht_cut[(rr <= r_out) & (~dilated_mask) & (wht_cut > 0)]
            wht_fill = float(np.median(donut_w)) if donut_w.size else 1.0
            wht_view[inside] = wht_fill

        fit_rows.append((
            hid, yc0, xc0, r_eq, r_in, r_out,
            A, float(final["amp_err"]), float(final["chi2_red"]),
            int(final["n_pix"]), int(n_iter),
            float(shift_total[0]), float(shift_total[1]),
            float(sig), float(buffer_snr), float(flux_added),
            float(pedestal), fit_mode, float(data_to_model),
            A_ns, chi2_ns,
            True, "ok",
        ))

        if return_diagnostics:
            psf_ns_scaled = (psf_noshift * A_ns) if np.isfinite(A_ns) else (
                psf_noshift * 0.0
            )
            diags.append(RepairDiagnostic(
                id=hid, yc=yc0, xc=xc0, r_equiv=r_eq,
                r_in=r_in, r_out=r_out,
                amplitude=A,
                chi2_red=float(final["chi2_red"]),
                n_pix=int(final["n_pix"]),
                n_iter=int(n_iter),
                shift_total=(float(shift_total[0]), float(shift_total[1])),
                significance=float(sig),
                center=(float(cy), float(cx)),
                amplitude_noshift=A_ns,
                chi2_red_noshift=chi2_ns,
                center_noshift=(float(cy_h), float(cx_h)),
                sci_cut=sci_cut.copy(),
                wht_cut=wht_cut.copy(),
                psf_cut_scaled=psf_cut * A,
                sci_repaired_cut=sci_rep[sly, slx].copy(),
                hole_mask=hole_mask_cut.copy(),
                dilated_hole_mask=dilated_mask.copy(),
                ring_mask=final["ring_mask"].copy(),
                repair_mask=inside.copy(),
                psf_cut_noshift_scaled=psf_ns_scaled,
                ring_mask_noshift=ring_ns.copy(),
                resid_frac=float(resid_frac),
                ring_snr=float(ring_snr),
                buffer_snr=float(buffer_snr),
                pedestal=float(pedestal),
                rho_psf=float(final.get("rho_psf", float("nan"))),
                fit_mode=fit_mode,
                data_to_model=float(data_to_model),
                action_mode=mode,
                bad_resid_mask=bad_resid_mask.copy(),
                ok=True, status="ok",
            ))

    fits_tbl = Table(
        rows=fit_rows or None,
        names=[
            "id", "yc", "xc", "r_equiv", "r_in", "r_out",
            "amplitude", "amp_err", "chi2_red", "n_pix", "n_iter",
            "shift_x", "shift_y",
            "significance", "buffer_snr", "flux_added",
            "pedestal", "fit_mode", "data_to_model",
            "amplitude_noshift", "chi2_red_noshift",
            "ok", "status",
        ],
        dtype=[int, float, float, float, float, float,
               float, float, float, int, int,
               float, float,
               float, float, float,
               float, str, float,
               float, float,
               bool, str],
    )

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fits_tbl.write(out_path, format="csv", overwrite=True)
        logger.info("[saturate] wrote %d rows → %s", len(fits_tbl), out_path)

    if plot_dir is not None and diags:
        plot_root = Path(plot_dir)
        plot_root.mkdir(parents=True, exist_ok=True)
        for d in diags:
            try:
                plot_repair_diagnostic(
                    d,
                    to_file=str(
                        plot_root
                        / f"{d.action_mode}_{d.id:04d}.png"
                    ),
                )
            except Exception as exc:  # pragma: no cover — diagnostic only
                logger.warning("hole %d: plot failed: %s", d.id, exc)

    return {
        "sci": sci_rep,
        "wht": wht_rep,
        "fits": fits_tbl,
        "diagnostics": diags,
        "holes": holes,
        "sky": sky,
        "sky_noise": sky_noise,
    }


def _failed_row(hid, yc, xc, r_eq, r_in, r_out, status,
                *, significance: float = float("nan"),
                buffer_snr: float = float("nan"),
                shift: tuple[float, float] = (0.0, 0.0)):
    return (
        hid, yc, xc, r_eq, r_in, r_out,
        float("nan"), float("nan"), float("nan"), 0, 0,
        float(shift[0]), float(shift[1]),
        float(significance), float(buffer_snr), 0.0,
        0.0, "", float("nan"),
        float("nan"), float("nan"),
        False, status,
    )


def _stub_diag(
    *, hid, yc0, xc0, r_eq, r_in, r_out,
    sci_cut, wht_cut, hole_mask_cut, dilated_mask,
    cy, cx, status,
    significance=float("nan"),
    shift_total=(0.0, 0.0),
    psf_cut_scaled=None, sci_repaired_cut=None,
    amplitude=float("nan"), chi2_red=float("nan"),
    n_pix=0, n_iter=0, ring_mask=None, repair_mask=None,
    resid_frac=float("nan"),
    ring_snr=float("nan"),
    buffer_snr=float("nan"),
    amplitude_noshift=float("nan"), chi2_red_noshift=float("nan"),
    center_noshift=(0.0, 0.0),
    psf_cut_noshift_scaled=None, ring_mask_noshift=None,
) -> RepairDiagnostic:
    """Stub diagnostic for sources that were rejected."""
    Z = np.zeros_like(sci_cut)
    F = np.zeros(sci_cut.shape, dtype=bool)
    return RepairDiagnostic(
        id=hid, yc=yc0, xc=xc0, r_equiv=r_eq, r_in=r_in, r_out=r_out,
        amplitude=amplitude, chi2_red=chi2_red,
        n_pix=int(n_pix), n_iter=int(n_iter),
        shift_total=(float(shift_total[0]), float(shift_total[1])),
        significance=float(significance),
        center=(float(cy), float(cx)),
        amplitude_noshift=amplitude_noshift,
        chi2_red_noshift=chi2_red_noshift,
        center_noshift=(float(center_noshift[0]), float(center_noshift[1])),
        sci_cut=sci_cut.copy(), wht_cut=wht_cut.copy(),
        psf_cut_scaled=psf_cut_scaled if psf_cut_scaled is not None else Z,
        sci_repaired_cut=(sci_repaired_cut if sci_repaired_cut is not None
                          else sci_cut.copy()),
        hole_mask=hole_mask_cut.copy(),
        dilated_hole_mask=dilated_mask.copy(),
        ring_mask=ring_mask if ring_mask is not None else F,
        repair_mask=repair_mask if repair_mask is not None else F,
        psf_cut_noshift_scaled=(psf_cut_noshift_scaled
                                if psf_cut_noshift_scaled is not None else Z),
        ring_mask_noshift=(ring_mask_noshift if ring_mask_noshift is not None
                           else F),
        resid_frac=float(resid_frac),
        ring_snr=float(ring_snr),
        buffer_snr=float(buffer_snr),
        ok=False, status=status,
    )


# --------------------------------------------------------------------------
# 6. Diagnostic plot
# --------------------------------------------------------------------------


def plot_repair_diagnostic(
    diag: RepairDiagnostic,
    *,
    to_file: str | None = None,
    pixel_scale: float | None = None,
    offset: float = 2e-5,
    include_gradient: bool = False,
    include_flux: bool = True,
    include_floor: bool = True,
):
    """Ten-panel repair diagnostic, shifted vs no-shift side by side.

    Layout (2×5):
        Row 0:  data/A | A·ψ shifted | residual shifted | residual SNR |
                residual no-shift
        Row 1:  hole+fit-ring overlay | subtracted/repaired | 2× zoom |
                radial profile | SNR polar (r, θ)

    The 2D panels show the full source cutout as it was actually fit
    (already sized to the PSF FOV in subtract mode, donut-sized in
    repair mode); no extra cropping.
    """
    import matplotlib.pyplot as plt

    full_sci = diag.sci_cut
    full_psf_scl = diag.psf_cut_scaled
    full_psf_ns_scl = diag.psf_cut_noshift_scaled
    full_sci_rep = diag.sci_repaired_cut
    sci = full_sci
    psf_scl = full_psf_scl
    psf_ns_scl = full_psf_ns_scl
    sci_rep = full_sci_rep
    hole_mask = diag.hole_mask
    dilated_hole_mask = diag.dilated_hole_mask
    ring_mask = diag.ring_mask
    repair_mask = diag.repair_mask
    ring_mask_noshift = diag.ring_mask_noshift
    if np.isfinite(diag.amplitude) and diag.amplitude > 0:
        A = float(diag.amplitude)
    else:
        # rejected source — use a positive reference for the log display.
        med = float(np.nanmedian(np.abs(sci))) if sci.size else 1.0
        A = med if med > 0 else 1.0
    A_ns = diag.amplitude_noshift if np.isfinite(diag.amplitude_noshift) else 0.0

    # 2-dex log stretch chosen so sky (offset=2e-5 → log≈-4.7) lands at
    # the same 25 % gray level that 0 lands on in the residual panels —
    # consistent zero tone across panels. Tight range = high contrast on
    # wings/halo/spike structure.
    log_kw = dict(vmin=-5.2, vmax=-3.2, cmap="bone_r", origin="lower")
    fig, ax = plt.subplots(2, 5, figsize=(21, 8))

    # Row 0
    ax[0, 0].imshow(np.log10(np.maximum(sci / A, 0) + offset), **log_kw)
    ax[0, 0].set_title(f"data / A   (A={A:.3g})")

    ax[0, 1].imshow(np.log10(np.maximum(psf_scl / A, 0) + offset), **log_kw)
    ax[0, 1].set_title("A·ψ shifted")

    # Residual colour range scaled by mad_std over ALL valid pixels —
    # i.e., everywhere except the wht=0 hole (and blanked pixels in
    # subtract mode). Not restricted to the fit ring, so the colour
    # scale reflects the full image's noise.
    resid = (sci - psf_scl) / A
    valid = ~hole_mask & np.isfinite(resid)
    bad = getattr(diag, "bad_resid_mask", None)
    if bad is not None and bad.shape == resid.shape:
        valid &= ~bad
    finite = resid[valid]
    if finite.size > 10:
        med = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - med)))
        rng = max(0.7 * mad, 1e-4)
    else:
        mad = 0.05 / 5.0
        rng = 0.05
    # Linear residual stretch for grayscale: asymmetric so the noise
    # floor (residual ≈ 0) lands near the WHITE end of the colormap
    # (≈0.2 in colour-space), matching the look of the log panels.
    # Negative residuals down to −1×MAD still resolved; positives
    # visible out to +3×MAD.
    grey_kw = dict(vmin=-2.0 * mad, vmax=5.0 * mad, cmap="bone_r",
                   origin="lower")
    # Linear stretch, scaled to the residual's own MAD: the residual runs
    # from a fraction of a percent to a few percent of the star, so on the
    # data's log stretch it is a flat gray field. Linear resolves the
    # under/over-subtraction structure the panel exists to show.
    resid_show = np.where(hole_mask, np.nan, resid)
    ax[0, 2].imshow(resid_show, **grey_kw)
    ax[0, 2].contour(ring_mask.astype(float), levels=[0.5],
                     colors="black", linewidths=0.5)
    ax[0, 2].contour(hole_mask.astype(float), levels=[0.5],
                     colors="red", linewidths=0.5)
    ax[0, 2].set_title(
        f"(data − A·ψ) / A   shifted   "
        f"Σ|r|/Σ|fit|={diag.resid_frac:.2f}"
    )

    fit_label_repair = ("A·ψ + pedestal"
                        if "pedestal" in diag.fit_mode
                        else "A·ψ")
    ped_repair = (f"  pedestal={diag.pedestal:.3g}"
                  if abs(diag.pedestal) > 0 else "")

    # (0, 3): SNR map with per-radial-bin calibrated noise.
    #
    # Algorithm:
    #   resid          = sci − A·ψ                      [data units]
    #   σ_ivar         = 1 / √wht                       [data units]
    #   w_az(r,θ)      = max(A·ψ(r,θ), ε)                azimuthal weight = local model
    #
    # For each radial annulus r ± dr (excluding hole + wht=0):
    #   var_resid(r) = mad_std_w(resid)²      — robust w_az-weighted scatter
    #   var_ivar(r)  = mad_std_w(σ_ivar)²     — robust w_az-weighted ivar
    #   var_psf(r)   = max(var_resid(r) − var_ivar(r), 0)
    #                                    — PSF residual variance, bg subtracted
    #                                      in quadrature (positive definite)
    # Weighted mad_std uses cumulative-weight weighted median + weighted MAD.
    # Robust to neighbor source contamination; pure rms is not.
    #
    # Per-pixel PSF noise distributes the bin variance proportional to the
    # local azimuthal weight (concentrates noise where model is bright):
    #   σ_psf²(r,θ) = var_psf(r) · w_az(r,θ) / <w_az>(r)
    #
    # Total noise + SNR:
    #   σ_tot = √(σ_ivar² + σ_psf²)
    #   SNR   = resid / σ_tot
    #
    # By construction the rms of (resid / σ_tot) within each bin ≈ 1, so
    # outliers (real astrophysical sources / bad PSF mismatch peaks) pop
    # out as |SNR| ≫ 1 against a uniform calibrated background.
    wht_cut = diag.wht_cut
    sigma_ivar = np.where(wht_cut > 0, 1.0 / np.sqrt(np.maximum(wht_cut, 1e-30)),
                          np.nan)
    sigma_ivar_safe = np.where(np.isfinite(sigma_ivar), sigma_ivar, 0.0)
    resid_d = sci - psf_scl
    psf_peak = float(np.nanmax(psf_scl)) if np.isfinite(psf_scl).any() else 0.0
    eps = 1e-12 * (psf_peak if psf_peak > 0 else 1.0)
    w_az = np.maximum(psf_scl, eps)
    yy_n, xx_n = np.indices(sci.shape)
    cy_n, cx_n = diag.center
    rr_n = np.hypot(yy_n - cy_n, xx_n - cx_n)
    valid = (~hole_mask) & np.isfinite(resid_d) & (wht_cut > 0)
    rmax_n = float(rr_n[valid].max()) if valid.any() else float(rr_n.max())
    n_bins_snr = 24
    bin_edges = np.linspace(0.0, rmax_n, n_bins_snr + 1)
    sigma_psf2 = np.zeros_like(resid_d, dtype=np.float64)
    def _wmad_std(values: np.ndarray, weights: np.ndarray,
                  clip_sigma: float = 3.0, n_iter: int = 2) -> float:
        """Weighted mad_std with iterative σ-clip rejection.

        Iterative rejection makes the statistic robust to compact bright
        contaminants (e.g. neighbour sources falling in the radial bin):
        pass 1 = unclipped weighted mad_std → pass 2+ = drop pixels with
        |x − wmedian| > clip_sigma · prev_mad_std and recompute.
        """
        if values.size == 0:
            return float("nan")
        v = values
        w = weights
        prev = float("nan")
        for _ in range(max(1, n_iter)):
            idx = np.argsort(v)
            cw = np.cumsum(w[idx])
            if cw[-1] <= 0:
                return float("nan")
            med = float(v[idx][int(np.searchsorted(cw, cw[-1] * 0.5))])
            ad = np.abs(v - med)
            idx2 = np.argsort(ad)
            cw2 = np.cumsum(w[idx2])
            if cw2[-1] <= 0:
                return float("nan")
            mad_std = 1.4826 * float(ad[idx2][int(np.searchsorted(cw2, cw2[-1] * 0.5))])
            if not np.isfinite(prev):
                prev = mad_std
                # Reject and continue with another pass.
                keep = ad <= clip_sigma * mad_std if mad_std > 0 else np.ones_like(v, bool)
                if keep.sum() < 5 or keep.sum() == v.size:
                    return mad_std
                v = v[keep]
                w = w[keep]
                continue
            return mad_std
        return prev

    for k in range(n_bins_snr):
        rlo, rhi = bin_edges[k], bin_edges[k + 1]
        in_bin = (rr_n >= rlo) & (rr_n < rhi) & valid
        if in_bin.sum() < 5:
            continue
        w_b = w_az[in_bin].astype(np.float64)
        sw = float(w_b.sum())
        if sw <= 0:
            continue
        sd_resid = _wmad_std(resid_d[in_bin].astype(np.float64), w_b)
        sd_ivar = _wmad_std(sigma_ivar_safe[in_bin].astype(np.float64), w_b)
        if not (np.isfinite(sd_resid) and np.isfinite(sd_ivar)):
            continue
        var_psf_bin = max(sd_resid**2 - sd_ivar**2, 0.0)
        if var_psf_bin <= 0:
            continue
        mean_w = sw / float(in_bin.sum())
        sigma_psf2[in_bin] = var_psf_bin * (w_b / mean_w)
    sigma_tot = np.sqrt(sigma_ivar_safe**2 + sigma_psf2)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_map = np.where(sigma_tot > 0, resid_d / sigma_tot, np.nan)
    snr_show = np.where(hole_mask, np.nan, snr_map)
    rms_check = (float(np.sqrt(np.nanmean(snr_show[valid]**2)))
                 if valid.any() else float("nan"))
    snr_kw = dict(vmin=-1.5, vmax=4.5, cmap="bone_r", origin="lower")
    ax[0, 3].imshow(snr_show, **snr_kw)
    ax[0, 3].contour(ring_mask.astype(float), levels=[0.5],
                     colors="black", linewidths=0.5)
    ax[0, 3].contour(hole_mask.astype(float), levels=[0.5],
                     colors="red", linewidths=0.5)
    ax[0, 3].set_title(
        f"SNR   per-radial-bin calibrated  (bins={n_bins_snr})\n"
        f"σ_psf²(r,θ) = max(<r²>−<σ_iv²>,0)·w_az/<w_az>   "
        f"⟨SNR²⟩^½={rms_check:.2f}"
    )

    # (0, 4): residual NO-SHIFT (same log stretch as the data panels).
    if A_ns > 0:
        resid_ns = (sci - psf_ns_scl) / A
        resid_ns_show = np.where(hole_mask, np.nan,
                                 np.log10(np.maximum(resid_ns, 0) + offset))
        ax[0, 4].imshow(resid_ns_show, **log_kw)
        ax[0, 4].contour(ring_mask_noshift.astype(float), levels=[0.5],
                         colors="black", linewidths=0.5)
        ax[0, 4].contour(hole_mask.astype(float), levels=[0.5],
                         colors="red", linewidths=0.5)
        m = ring_mask_noshift
        if m.any():
            mod = psf_ns_scl[m]
            denom_ns = float(np.sum(np.abs(mod)))
            rfrac_ns = (float(np.sum(np.abs(sci[m] - mod))) / denom_ns
                        if denom_ns > 0 else float("nan"))
        else:
            rfrac_ns = float("nan")
        ax[0, 4].set_title(
            f"(data − A·ψ) / A   no-shift   Σ|r|/Σ|fit|={rfrac_ns:.2f}"
        )
    else:
        ax[0, 4].set_title("no-shift fit unavailable")

    # (1, 0): data with hole + ring overlays.
    ax[1, 0].imshow(np.log10(np.maximum(sci / A, 0) + offset), **log_kw)
    ax[1, 0].contour(hole_mask.astype(float), levels=[0.5],
                     colors="red", linewidths=1.0)
    ax[1, 0].contour(ring_mask.astype(float), levels=[0.5],
                     colors="cyan", linewidths=0.7)
    ax[1, 0].set_title("hole (red), fit ring (cyan)")

    from matplotlib.patches import Circle

    # ── PSF-rank running-mad_std significance mask ────────────────────
    # Sort valid pixels by PSF brightness (descending). At each rank k,
    # compute a sliding window of √N residual pixels and the parallel
    # window of predicted ivar σ. Use NON-robust RMS (not MAD) so that
    # spike-pixel outliers actually inflate the statistic — MAD is
    # robust and silently absorbed them.
    #
    #   rms(resid_w) / √N_w  <  σ_pred_w / √N_w
    #
    # √N_w cancels (algebraically equivalent to rms < σ_pred) but
    # framing as standard-error-of-mean comparison makes the intent
    # explicit: residual mean consistent with zero at the noise level.
    # Pixels beyond that rank are noise-dominated → masked.
    psf_pix = psf_scl.ravel()
    resid_pix = resid_d.ravel()
    sigma_ivar_pix = sigma_ivar_safe.ravel()
    valid_flat = (~hole_mask).ravel() & np.isfinite(resid_pix) & (psf_pix > 0)
    mask_keep_2d = np.zeros_like(psf_scl, dtype=bool)
    if valid_flat.sum() > 16:
        idx_valid = np.where(valid_flat)[0]
        order_in_valid = np.argsort(-psf_pix[idx_valid])
        order_idx = idx_valid[order_in_valid]
        n_valid = order_idx.size
        window = max(int(np.sqrt(n_valid)), 8)
        res_sorted = resid_pix[order_idx]
        ivar_sorted = sigma_ivar_pix[order_idx]
        running_se_resid = np.empty(n_valid)
        running_se_sigma = np.empty(n_valid)
        half_w = window // 2
        for k in range(n_valid):
            lo = max(0, k - half_w)
            hi = min(n_valid, lo + window)
            win_r = res_sorted[lo:hi]
            win_s = ivar_sorted[lo:hi]
            n_w = float(win_r.size)
            # NON-robust RMS — preserves outliers (spikes) instead of
            # absorbing them like the median-based MAD would.
            rms_w = float(np.sqrt(np.mean(win_r * win_r)))
            running_se_resid[k] = rms_w / np.sqrt(n_w)
            ss = win_s[win_s > 0]
            sig_w = (float(np.sqrt(np.mean(ss * ss)))
                     if ss.size else 0.0)
            running_se_sigma[k] = sig_w / np.sqrt(n_w)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(running_se_sigma > 0,
                             running_se_resid / running_se_sigma, np.inf)
        # Walk descending PSF rank. Stop at the FIRST rank where
        # ratio ≤ 1.5 (1.5 σ floor) — descent halted. Mask all pixels
        # with rank strictly above the cutoff (i.e., brighter PSF
        # pixels with residual ≥ 1.5 σ_pred). Spike pixels at later
        # ranks are NOT included (descent already stopped).
        below = ratio <= 1.5
        if below.any():
            k_cut = int(np.argmax(below))
            n_keep = k_cut
        else:
            n_keep = n_valid
        if n_keep > 0:
            mask_flat = mask_keep_2d.ravel()
            mask_flat[order_idx[:n_keep]] = True
            mask_keep_2d = mask_flat.reshape(psf_scl.shape)
        logger.info(
            "[mask] id=%d n_valid=%d window=%d  keep=%d (%.1f%%)  "
            "ratio[min/med/max]=%.2f/%.2f/%.2f",
            diag.id, n_valid, window,
            n_keep, 100.0 * n_keep / n_valid,
            float(np.nanmin(ratio)),
            float(np.nanmedian(ratio[np.isfinite(ratio)])),
            float(np.nanmax(ratio[np.isfinite(ratio)])),
        )
    else:
        mask_keep_2d = np.ones_like(psf_scl, dtype=bool)

    # (1, 1) / (1, 2): subtracted residual or repaired sci/A.
    # In subtract mode the panel shows a RESIDUAL → linear grayscale
    # ±3 × MAD (same MAD used by the top-row residual panels).
    # In repair mode the panel shows the post-fill image → log scale
    # (same as the data/model panels).
    if diag.action_mode == "subtract":
        sub_kw = grey_kw       # same asymmetric stretch as the residual panels
        # sci_rep already has both A·ψ and the halo subtracted in
        # subtract mode, so we display sci_rep/A directly. INVERT the
        # PSF-rank significance mask: hide pixels where the PSF
        # dominated (residual > 2 σ_bg), keep the surrounding "rest
        # of the image" so neighbours / background structure pop out
        # free of PSF-residual contamination.
        sub_view = sci_rep / A
        sub_view = np.where(~mask_keep_2d, sub_view, 0.0)
        ax[1, 1].imshow(sub_view, **sub_kw)
    else:
        ax[1, 1].imshow(np.log10(np.maximum(sci_rep / A, 0) + offset),
                        **log_kw)

    if diag.action_mode == "subtract":
        ax[1, 1].add_patch(Circle(
            (diag.center[1], diag.center[0]), diag.r_out,
            fill=False, edgecolor="lime", linewidth=0.7, alpha=0.5,
        ))
        ax[1, 1].set_title(
            f"subtracted: (data − A·ψ)/A   ±{rng:.1g}\n"
            f"fit={fit_label_repair}  mode={diag.fit_mode}{ped_repair}"
        )
    else:
        ax[1, 1].contour(repair_mask.astype(float), levels=[0.5],
                         colors="lime", linewidths=0.7)
        ax[1, 1].set_title(
            f"repaired sci / A  (lime = filled)\n"
            f"fit={fit_label_repair}  mode={diag.fit_mode}{ped_repair}"
        )

    # (1, 2): same image, 2× zoomed in on the source position.
    H_full, W_full = sci_rep.shape
    cy_src = max(0, min(H_full - 1, int(round(diag.center[0]))))
    cx_src = max(0, min(W_full - 1, int(round(diag.center[1]))))
    half = max(H_full, W_full) // 4
    sly_z = slice(max(0, cy_src - half), min(H_full, cy_src + half + 1))
    slx_z = slice(max(0, cx_src - half), min(W_full, cx_src + half + 1))
    if diag.action_mode == "subtract":
        zoom_view = sci_rep[sly_z, slx_z] / A
        zoom_view = np.where(~mask_keep_2d[sly_z, slx_z], zoom_view, 0.0)
        ax[1, 2].imshow(zoom_view, **sub_kw)
    else:
        ax[1, 2].imshow(
            np.log10(np.maximum(sci_rep[sly_z, slx_z] / A, 0) + offset),
            **log_kw,
        )
    if diag.action_mode == "subtract":
        # r_out circle, in the cropped frame.
        ax[1, 2].add_patch(Circle(
            (diag.center[1] - slx_z.start, diag.center[0] - sly_z.start),
            diag.r_out,
            fill=False, edgecolor="lime", linewidth=0.7, alpha=0.5,
        ))
        ax[1, 2].set_title(f"subtracted (2× zoom)   ±{rng:.1g}")
    else:
        repair_zoom = repair_mask[sly_z, slx_z]
        if repair_zoom.any():
            ax[1, 2].contour(repair_zoom.astype(float), levels=[0.5],
                             colors="lime", linewidths=0.7)
        ax[1, 2].set_title("repaired (2× zoom)")

    # Radial profile uses the FULL cutout (so we see the wing extent
    # in subtract mode where the cutout is large).
    yy_full, xx_full = np.indices(full_sci.shape)
    cy_full, cx_full = diag.center
    rr_full = np.hypot(yy_full - cy_full, xx_full - cx_full)
    rmax = max(diag.r_out * 1.4, float(rr_full.max()) * 0.95)
    rbins = np.linspace(0.0, rmax, 30)
    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    def _prof(arr):
        m = (rr_full <= rmax) & np.isfinite(arr)
        idx = np.digitize(rr_full[m], rbins) - 1
        out = np.full(len(rbins) - 1, np.nan)
        for k in range(len(rbins) - 1):
            sel = idx == k
            if sel.any():
                out[k] = float(np.nanmedian(arr[m][sel]))
        return out

    def _mad_prof(arr):
        m = (rr_full <= rmax) & np.isfinite(arr) & ~hole_mask
        idx = np.digitize(rr_full[m], rbins) - 1
        a = arr[m]
        out = np.full(len(rbins) - 1, np.nan)
        for k in range(len(rbins) - 1):
            sel = idx == k
            if sel.sum() > 3:
                v = a[sel]
                out[k] = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        return out

    def _pos(arr):
        return np.where(arr > 0, arr, np.nan)

    p_data = _pos(_prof(full_sci / A))
    p_psf = _pos(_prof(full_psf_scl / A)) if diag.amplitude > 0 else None
    p_rep = _pos(_prof(full_sci_rep / A)) if diag.amplitude > 0 else None
    p_mad = (_pos(_mad_prof((full_sci - full_psf_scl) / A))
             if diag.amplitude > 0 else None)

    # Inner boundary = max radius of the dilated wht=0 mask.
    if diag.dilated_hole_mask.any():
        r_dil = float(np.max(rr_full[diag.dilated_hole_mask]))
    else:
        r_dil = 0.0

    axp = ax[1, 3]
    axp.plot(rmid, p_data, "-o", color="C0", label="data", markersize=3)
    if p_psf is not None:
        axp.plot(rmid, p_psf, "-", color="C1", label="A·ψ shifted")
    if p_rep is not None:
        axp.plot(rmid, p_rep, "--", color="C2", label="repaired")
    if p_mad is not None:
        axp.plot(rmid, p_mad, ":", color="C3", label="MAD(resid)/A")
    axp.axvspan(0, r_dil, color="red", alpha=0.10)
    axp.axvspan(r_dil, diag.r_out, color="cyan", alpha=0.08)
    parts = [p_data] + ([p_psf] if p_psf is not None else []) \
                     + ([p_rep] if p_rep is not None else []) \
                     + ([p_mad] if p_mad is not None else [])
    finite = np.concatenate(parts)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size:
        axp.set_ylim(0.5 * finite.min(), 2.0 * finite.max())
        axp.set_yscale("log")
    axp.set_xlabel("radius (pix)")
    axp.set_ylabel("median flux / A")
    axp.legend(fontsize=8)
    axp.set_title(f"radial profile  n={diag.n_pix}")

    # (1, 4): polar-remapped SNR map using ONLY σ_ivar (no PSF-error
    # terms) — exposes the raw residual structure relative to the
    # background/exposure noise. Diffraction spikes appear as vertical
    # bands at fixed θ; rings as horizontal bands at fixed r.
    from scipy.ndimage import map_coordinates
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_ivar_map = np.where(sigma_ivar_safe > 0,
                                resid_d / sigma_ivar_safe, np.nan)
    snr_ivar_show = np.where(hole_mask, np.nan, snr_ivar_map)
    cy_pol, cx_pol = diag.center
    # Fixed radial range r=5-50 px so polar panels are directly
    # comparable across sources of different sizes.
    rmin_pol, rmax_pol = 5.0, 50.0
    n_r, n_th = 120, 240
    r_grid = np.linspace(rmin_pol, rmax_pol, n_r)
    th_grid = np.linspace(0.0, 2.0 * np.pi, n_th)
    R_pol, TH_pol = np.meshgrid(r_grid, th_grid, indexing="ij")
    ys = cy_pol + R_pol * np.sin(TH_pol)
    xs = cx_pol + R_pol * np.cos(TH_pol)
    snr_filled = np.where(np.isfinite(snr_ivar_show), snr_ivar_show, 0.0)
    snr_polar = map_coordinates(snr_filled, [ys, xs], order=1,
                                cval=np.nan, mode="constant")
    hole_filled = hole_mask.astype(float)
    hole_pol = map_coordinates(hole_filled, [ys, xs], order=1,
                               cval=1.0, mode="constant")
    snr_polar = np.where(hole_pol > 0.5, np.nan, snr_polar)
    finite_pol = snr_polar[np.isfinite(snr_polar)]
    if finite_pol.size > 50:
        v_lo, v_hi = (float(np.nanpercentile(finite_pol, 5.0)),
                      float(np.nanpercentile(finite_pol, 95.0)))
    else:
        v_lo, v_hi = -3.0, 9.0
    snr_polar_kw = dict(vmin=v_lo, vmax=v_hi, cmap="bone_r", origin="lower")
    ax[1, 4].imshow(snr_polar.T, extent=[rmin_pol, rmax_pol, 0.0, 360.0],
                    aspect="auto", **snr_polar_kw)
    ax[1, 4].axis("on")
    ax[1, 4].set_xlabel("radius (pix)")
    ax[1, 4].set_ylabel("θ (deg)")
    ax[1, 4].set_title(
        f"SNR polar (r, θ)   σ_ivar only   stretch=[{v_lo:.1f},{v_hi:.1f}]"
    )

    for r in range(2):
        for c in range(5):
            if (r, c) not in ((1, 3), (1, 4)):
                ax[r, c].axis("off")

    sx, sy = diag.shift_total
    fom = (
        f"Σdata/Σ(A·ψ)={diag.data_to_model:.2f}  →  "
        f"{'PSF + pedestal' if 'pedestal' in diag.fit_mode else 'PSF only'}"
    )
    extras = []
    if np.isfinite(diag.rho_psf):
        extras.append(f"ρ_psf={diag.rho_psf:+.3f}")
    extras_str = ("  |  " + "  ".join(extras)) if extras else ""
    head = (
        f"{diag.action_mode} id={diag.id}  ({diag.yc:.0f}, {diag.xc:.0f})  "
        f"A={diag.amplitude:.3g}  "
        f"r_eq={diag.r_equiv:.1f}  r_out={diag.r_out:.1f}  "
        f"buffer SNR={diag.buffer_snr:.1f}  "
        f"shift=({sx:+.2f},{sy:+.2f}) px  "
        f"n_iter={diag.n_iter}  |  decision FOM: {fom}{extras_str}"
    )
    if not diag.ok:
        head = f"REJECTED — {diag.status}\n{head}"
        fig.suptitle(head, color="red")
    else:
        fig.suptitle(head)
    fig.tight_layout()
    if to_file:
        fig.savefig(to_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


