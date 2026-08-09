"""Alternative template build schemes: the wren fork and the IDL classic code.

Self-contained ports, deliberately kept out of :mod:`mophongo.templates` so the
three schemes can be compared 1-1 and either alternative can be adapted or
deleted as a unit. Nothing here imports the fitting, catalog or pipeline
layers: every entry point takes plain numpy arrays and returns a composite
stamp plus a flat dict of per-source bookkeeping.

Dispatch lives in :meth:`mophongo.templates.Templates.extract_templates` via
``extend_mode``:

``default``
    Segment-masked detection data, unit sum. The current mophongo scheme; no
    code here.
``psf``
    ``default`` followed by :meth:`mophongo.templates.Templates.extend_with_psf`
    (template convolved with the detection PSF fills the zero pixels). No code
    here.
``wren``
    :func:`composite_wren` -- port of ``wren/dev-wren``
    ``templates.py::_extended_composite``: area-weighted ownership, one
    SNR-graded linear blend between the data and a core-anchored PSF model,
    support capped at the PSF ``ee_fraction`` radius, positivity clip before
    the unit-sum norm.
``classic``
    :func:`composite_classic` -- port of IDL ``subphot.pro::build_cube``
    (:294-330): exact segment data plus a least-squares-scaled PSF over the
    whole stamp, with a hard replacement by a pure point source below
    ``tmpl_snrlo``.

References
----------
IDL   ``subphot.pro::build_cube``, driven by ``legacy/autopilot/old/dophot.pro``
      (``phot.param`` sets ``tmpl_snrlo = 15.0``).
wren  ``wren/dev-wren:src/mophongo/templates.py::_extended_composite`` plus its
      sizing preamble in ``pipeline.py``.
Both are described in ``scratch/wren/template_comparison.tex``.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging

import numpy as np
from scipy.ndimage import find_objects, map_coordinates
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)

__all__ = [
    "WrenParams",
    "ClassicParams",
    "blend_weight",
    "build_ownership",
    "composite_classic",
    "composite_wren",
    "cutout_roi",
    "detection_rms",
    "psf_ee_radius_pix",
    "representative_psf",
    "robust_sigma",
    "sample_psf_on_stamp",
    "sky_sigma",
    "wren_fill_radius",
]


# ---------------------------------------------------------------------------
# scheme parameters
# ---------------------------------------------------------------------------


@dataclass
class WrenParams:
    """Knobs of the wren ``_extended_composite`` scheme.

    Attributes
    ----------
    max_radius_pix
        Ownership-contest disk radius and outer reach of the halo annuli, in
        detection pixels (wren's ``r_fill``). ``<= 0`` derives it from the
        detection PSF via :func:`wren_fill_radius`.
    psf_ee_radius_pix
        Hard cap on the composite support (wren's ``R95``). ``None`` derives it
        from the detection PSF at ``ee_fraction``.
    aperture_radius_pix
        Measurement-aperture radius on the detection grid; only used for the
        ``flux_beyond_aper`` crowding bookkeeping. ``None`` disables it.
    ee_fraction
        Encircled-energy fraction defining the support cap (``extend_template_ee``).
    fit_snrlo_psf
        Core-weight onset is ``1.5 * fit_snrlo_psf`` -- the same SNR at which
        the IDL code switches to a pure point source.
    wings_snr_psf
        Per-annulus weight onset.
    blend_p
        Rolloff exponent of the blend weight.
    blend_annulus
        Halo annulus width in arcsec (converted with the cutout WCS; falls back
        to 4 detection pixels without one).
    containment
        Detection-PSF stamp containment ``c_det`` used by ``flux_beyond_stamp``.
        wren reads it from ``PSFRegionMap.containment``; 1.0 disables it.
    """

    max_radius_pix: float = 0.0
    psf_ee_radius_pix: float | None = None
    aperture_radius_pix: float | None = None
    ee_fraction: float = 0.95
    fit_snrlo_psf: float = 10.0
    wings_snr_psf: float = 3.0
    blend_p: float = 2.0
    blend_annulus: float = 0.15
    containment: float = 1.0


@dataclass
class ClassicParams:
    """Knobs of the IDL ``build_cube`` scheme.

    Attributes
    ----------
    tmpl_snrlo
        In-segment SNR below which the template is *replaced* by a pure point
        source (``phot.param:91`` sets 15.0). ``<= 0`` disables the branch,
        matching IDL's ``keyword_set(tmpl_snrlo)`` guard.
    rms
        Detection-image noise entering that SNR. ``None`` measures it once per
        extraction with :func:`robust_sigma`, IDL's ``robust_sigma(ttmpl)``.
    force_psf
        IDL's ``/psf`` keyword: build every template as a pure point source.
    """

    tmpl_snrlo: float = 15.0
    rms: float | None = None
    force_psf: bool = False


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def robust_sigma(y: np.ndarray, zero: bool = False) -> float:
    """Biweight scale estimate, a port of IDL astrolib ``robust_sigma``.

    Used by the ``classic`` scheme, which measures its template SNR against
    ``robust_sigma`` of the detection tile. Returns 0.0 for a degenerate input
    (astrolib returns -1 when fewer than three points survive the 6-MAD cut;
    that is reported here as 0.0 so callers can simply test ``> 0``).
    """
    arr = np.asarray(y, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    eps = 1.0e-20
    y0 = 0.0 if zero else float(np.median(arr))
    mad = float(np.median(np.abs(arr - y0))) / 0.6745
    if mad < eps:
        mad = float(np.mean(np.abs(arr - y0))) / 0.8
    if mad < eps:
        return 0.0
    u = (arr - y0) / (6.0 * mad)
    uu = u * u
    q = uu <= 1.0
    if int(q.sum()) < 3:
        return 0.0
    n = float(arr.size)
    num = float(np.sum((arr[q] - y0) ** 2 * (1.0 - uu[q]) ** 4))
    den = float(np.sum((1.0 - uu[q]) * (1.0 - 5.0 * uu[q])))
    if den <= 1.0:
        return 0.0
    sigma = n * num / (den * (den - 1.0))
    return float(np.sqrt(sigma)) if sigma > 0 else 0.0


def _decimate(arr: np.ndarray, max_pixels: int) -> np.ndarray:
    """Strided view of ``arr`` with at most ``max_pixels`` elements.

    Both noise estimators here are global scalars that IDL/wren measured on a
    single tile; on a full mosaic the exact statistic would sort ~1e9 pixels
    for no gain in precision, so it is measured on a regular subsample.
    """
    arr = np.asarray(arr)
    if arr.size <= max_pixels:
        return arr
    step = int(np.ceil(np.sqrt(arr.size / float(max_pixels))))
    return arr[::step, ::step]


def detection_rms(image: np.ndarray, max_pixels: int = 4_000_000) -> float:
    """IDL ``tmpl_rms = robust_sigma(ttmpl)`` on the detection image.

    Measured over the image as given (sources included): the biweight scale
    downweights them, which is precisely IDL's intent.
    """
    return robust_sigma(_decimate(image, max_pixels))


def sky_sigma(
    image: np.ndarray,
    segmap: np.ndarray,
    n_clip: float = 3.0,
    n_iter: int = 3,
    max_pixels: int = 4_000_000,
) -> float | None:
    """Robust sky sigma from un-segmented pixels, sigma-clipped.

    The wren noise fallback when no detection weight map is supplied.
    ``segmap == 0`` pixels still hold source wings and undetected light, which
    would bias a plain MAD high; iterative clipping removes them.
    """
    image = _decimate(image, max_pixels)
    segmap = _decimate(segmap, max_pixels)
    bg = np.asarray(image)[np.asarray(segmap) == 0]
    bg = bg[np.isfinite(bg)]
    if bg.size < 10:
        return None
    med = float(np.median(bg))
    sig = 1.4826 * float(np.median(np.abs(bg - med)))
    for _ in range(int(n_iter)):
        if sig <= 0:
            break
        keep = np.abs(bg - med) < n_clip * sig
        if keep.sum() < 10 or keep.all():
            break
        bg = bg[keep]
        med = float(np.median(bg))
        sig = 1.4826 * float(np.median(np.abs(bg - med)))
    return sig if sig > 0 else None


def psf_ee_radius_pix(psf: np.ndarray, ee_fraction: float = 0.95) -> float:
    """Stamp-relative encircled-energy radius of ``psf``, in pixels.

    Pixels are ordered by distance from the stamp centre ``(shape - 1) / 2``
    and the cumulative sum is normalised by ``psf.sum()`` -- so the radius
    encloses ``ee_fraction`` of the *stamp*, not of the true total PSF. wren's
    ``utils.psf_ee_radius_pix`` normalises the same way (see
    ``template_comparison.tex`` Fig. 8); it interpolates a photutils curve of
    growth where this uses the exact ordered cumulative sum.
    """
    arr = np.asarray(psf, dtype=float)
    total = float(np.nansum(arr))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF must have a positive finite sum")
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0).ravel()
    order = np.argsort(r)
    cum = np.cumsum(np.nan_to_num(arr.ravel()[order]))
    idx = int(np.searchsorted(cum, float(ee_fraction) * cum[-1]))
    idx = min(idx, r.size - 1)
    return float(r[order][idx])


def representative_psf(psf, ee_fraction: float = 0.95) -> np.ndarray:
    """Return one PSF array standing in for a whole map.

    A plain array passes through. A :class:`~mophongo.psf_map.PSFRegionMap`
    (duck-typed on ``.psfs``) yields its *widest* region, so a single derived
    template size is large enough everywhere -- wren's choice in
    ``pipeline.py``.
    """
    stack = getattr(psf, "psfs", None)
    if stack is None:
        return np.asarray(psf, dtype=float)
    arrays = [np.asarray(p, dtype=float) for p in stack]
    if not arrays:
        raise ValueError("PSF map contains no PSFs")
    return max(arrays, key=lambda p: psf_ee_radius_pix(p, ee_fraction))


def sample_psf_on_stamp(
    psf: np.ndarray,
    shape: tuple[int, int],
    center: tuple[float, float],
    order: int = 1,
) -> np.ndarray:
    """Resample ``psf`` onto ``shape`` with its centre at ``center = (x, y)``.

    The PSF is normalised by its own sum *before* interpolation and is not
    renormalised afterwards, so flux lost off the stamp edge stays lost -- the
    convention of both reference codes (wren divides by ``psf_src.sum()``, IDL
    normalises ``tmplpsf`` upstream and shifts with ``missing=0``). The PSF
    centre is the geometric stamp centre ``(shape - 1) / 2``, not a centroid.

    ``order=1`` reproduces wren's bilinear ``map_coordinates``; ``order=3``
    stands in for IDL's ``interpolate(..., cubic=-0.5)``.
    """
    arr = np.asarray(psf, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("PSF must have a positive finite sum")
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    pcy = (arr.shape[0] - 1) / 2.0
    pcx = (arr.shape[1] - 1) / 2.0
    coords = np.array([pcy + (yy - center[1]), pcx + (xx - center[0])])
    return map_coordinates(arr / total, coords, order=order, mode="constant", cval=0.0)


def blend_weight(snr: float, thresh: float, p: float) -> float:
    """Data weight of the wren data/PSF blend: 1 at/above ``thresh``, power-law below.

    ``thresh`` is the *onset* of PSF blending (the weight saturates there), so
    the smooth rolloff reduces to the IDL hard switch in the limit ``p -> inf``.
    """
    if thresh <= 0:
        return 1.0
    if np.isnan(snr):
        return 0.0  # no usable SNR measurement -> defer to the PSF model
    ratio = max(float(snr), 0.0) / float(thresh)
    if ratio >= 1.0:
        return 1.0
    return float(ratio ** p)


def _disk_kernel(radius: float) -> np.ndarray:
    """Binary circular kernel used by the ownership convolution."""
    r = max(1, int(np.ceil(float(radius))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((xx ** 2 + yy ** 2) <= float(radius) ** 2).astype(np.float32)


def cutout_roi(
    boxes: list[tuple[float, float, int, int]],
    shape: tuple[int, int],
    step: int = 8,
) -> np.ndarray:
    """Coarse bool mask of every pixel some retained cutout will read.

    ``boxes`` is a list of ``(x, y, height, width)`` in original-image pixels,
    built with the same arithmetic the extraction loop uses. Boxes are rounded
    outward and coarsened by ``step``; marking extra territory can only keep
    extra labels in :func:`build_ownership`, never drop a relevant one.
    """
    ny, nx = shape
    roi = np.zeros((-(-ny // step), -(-nx // step)), dtype=bool)
    for x, y, height, width in boxes:
        y0 = max(int(np.floor(y - height / 2)), 0)
        y1 = min(int(np.ceil(y + height / 2)) + 1, ny)
        x0 = max(int(np.floor(x - width / 2)), 0)
        x1 = min(int(np.ceil(x + width / 2)) + 1, nx)
        roi[y0 // step: -(-y1 // step), x0 // step: -(-x1 // step)] = True
    return roi


def build_ownership(
    segmap: np.ndarray,
    radius: float,
    *,
    roi: np.ndarray | None = None,
    roi_step: int = 1,
) -> np.ndarray:
    """Global area-weighted ownership map (wren ``Templates._build_ownership``).

    Each background pixel goes to the segment with the largest area inside a
    disk of ``radius`` around it; segment pixels always keep their own label,
    so the partition is disjoint by construction. Strict ``>`` in the contest
    makes the lowest label win ties. Unlike a distance Voronoi this is
    area-weighted: a large segment wins more inter-source territory than a
    small one.

    ``roi`` (a bool mask coarsened by ``roi_step``, from :func:`cutout_roi`)
    restricts the loop to labels that can affect a pixel some template actually
    reads. A label writes only inside its own bbox padded by ``radius``, so the
    restricted result is *identical* there, not approximate.
    """
    disk = _disk_kernel(radius)
    pad = disk.shape[0] // 2
    ny, nx = segmap.shape
    best = np.zeros((ny, nx), dtype=np.float32)
    owner = np.asarray(segmap).copy()
    slices = find_objects(owner)  # index i -> label (i + 1)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        label = i + 1
        y0, y1 = max(0, sl[0].start - pad), min(ny, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(nx, sl[1].stop + pad)
        if roi is not None and not roi[
            y0 // roi_step: -(-y1 // roi_step),
            x0 // roi_step: -(-x1 // roi_step),
        ].any():
            continue
        sub = owner[y0:y1, x0:x1]
        # Round the area to an integer: fftconvolve of binary arrays carries
        # ~1e-15 noise that would otherwise break exact ties non-deterministically.
        area = np.rint(fftconvolve((segmap[y0:y1, x0:x1] == label).astype(np.float32), disk, mode="same"))
        b = best[y0:y1, x0:x1]
        upd = (area > b) & (segmap[y0:y1, x0:x1] == 0)
        sub[upd] = label
        b[upd] = area[upd]
    return owner


def wren_fill_radius(
    psf: np.ndarray,
    *,
    ee_fraction: float = 0.95,
    aperture_radius_pix: float | None = None,
    kernel_half_width: float = 0.0,
) -> float:
    """wren's ``r_fill = max(R_ee, r_aper + kernel_half_width)``, in pixels.

    The template must cover the measurement aperture plus a convolution margin,
    and never be smaller than the encircled-energy cap. Because
    ``r_fill >= R_ee`` by construction, the extra margin buys stamp area and
    ownership reach only: the composite support still stops at ``R_ee``.
    """
    r_fill = psf_ee_radius_pix(psf, ee_fraction)
    if aperture_radius_pix is not None and aperture_radius_pix > 0:
        r_fill = max(r_fill, float(aperture_radius_pix) + float(kernel_half_width))
    return float(r_fill)


def _region_snr(
    data: np.ndarray,
    ivar: np.ndarray | None,
    mask: np.ndarray,
    bg_rms: float | None,
) -> tuple[float, float]:
    """Integrated SNR and 1-sigma noise of ``data`` over ``mask`` (wren ``_region_snr``).

    The flux is clamped non-negative before dividing: a region with a genuinely
    negative net sum is a non-detection and should blend fully to the PSF
    rather than report a negative SNR that never crosses a threshold. Noise
    prefers the formal ``sqrt(sum 1/ivar)``, falling back to ``bg_rms*sqrt(n)``.
    """
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0
    flux = max(float(np.nansum(data[mask])), 0.0)
    noise = 0.0
    if ivar is not None:
        iv = np.asarray(ivar, dtype=float)[mask]
        good = iv > 0
        if good.any():
            noise = float(np.sqrt(np.sum(1.0 / iv[good])))
    if noise <= 0 and bg_rms and bg_rms > 0:
        noise = float(bg_rms) * np.sqrt(n)
    return (flux / noise if noise > 0 else 0.0), noise


# ---------------------------------------------------------------------------
# scheme 'classic': IDL subphot.pro::build_cube (:294-330)
# ---------------------------------------------------------------------------


def composite_classic(
    data: np.ndarray,
    seg: np.ndarray,
    label: int,
    psf_stamp: np.ndarray,
    *,
    params: ClassicParams,
    tmpl_rms: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build one template the way IDL ``build_cube`` does.

    .. code-block:: idl

        fpsf = (la_least_squares(bpsf[iseg], ttmpl[iseg]))[0] > 0.
        if fpsf le 0 then begin  m = bpsf  &  fpsf = 1.0
        end else m = mask_seg*ttmpl + fpsf*(1-mask_seg)*bpsf
        if total(ttmpl[iseg])/(sqrt(nseg)*tmpl_rms) lt tmpl_snrlo then m = bpsf*fpsf

    so ``m = S.D + f_psf (1-S).P`` with ``f_psf = sum_S P D / sum_S P^2`` floored
    at zero -- an ordinary least-squares fit of the PSF to the data inside the
    exact segment, which down-weights pixels where the PSF is faint (wren's
    ``A_src`` is a flux *ratio* instead). Below ``tmpl_snrlo`` the template is
    *replaced* by a point source, not blended towards one.

    Notes
    -----
    * The support is the whole stamp, including pixels of *other* segments:
      ``(1 - mask_seg)`` does not exclude neighbours. Caller-side sizing should
      therefore floor the cutout at the detection-PSF stamp, beyond which the
      resampled PSF is identically zero and the composite with it.
    * There is no positivity clip; negative sky inside the segment is kept,
      exactly as in IDL.
    * IDL's step 7 (``nmc = apermask(...)*mc/total(mc)``, ``subphot.pro:324``)
      normalises the *convolved* plane and only then applies a circular mask of
      radius ``ceil(ksz/2)``. That belongs to the convolution stage and is not
      reproduced here; mophongo normalises the unconvolved template and does
      not mask after convolving.

    Parameters
    ----------
    data
        Raw detection-image stamp.
    seg
        Segmentation stamp aligned with ``data``.
    label
        This source's segmentation label.
    psf_stamp
        Unit-sum detection PSF resampled onto the stamp at the source position
        (:func:`sample_psf_on_stamp` with ``order=3``).
    params
        :class:`ClassicParams`.
    tmpl_rms
        Detection-image noise entering the low-SNR test (IDL ``tmpl_rms``).

    Returns
    -------
    tuple
        ``(composite, info)``; ``info`` holds ``fpsf``, ``snr_seg``,
        ``flux_in_seg`` and ``added_flux`` (IDL's per-source log columns).
    """
    own = np.asarray(seg) == label
    n_seg = int(own.sum())
    P = np.asarray(psf_stamp, dtype=float)
    D = np.where(np.isfinite(data), data, 0.0).astype(float)

    denom = float(np.dot(P[own], P[own])) if n_seg else 0.0
    fpsf = max(float(np.dot(P[own], D[own])) / denom, 0.0) if denom > 0 else 0.0
    if fpsf <= 0.0:
        # Degenerate least-squares fit: IDL falls back to a bare point source.
        m = P.copy()
        fpsf = 1.0
    else:
        m = np.where(own, D, 0.0) + fpsf * np.where(own, 0.0, P)

    # Low-SNR replacement (subphot.pro:310-313); the SNR uses the raw data sum
    # in the segment, not the composite.
    snr_seg = np.nan
    if n_seg and tmpl_rms > 0:
        snr_seg = float(D[own].sum()) / (np.sqrt(n_seg) * float(tmpl_rms))
    replaced = bool(params.force_psf)
    if params.tmpl_snrlo > 0 and np.isfinite(snr_seg) and snr_seg < params.tmpl_snrlo:
        replaced = True
    if replaced:
        m = fpsf * P

    flux_in_seg = float(np.where(own, m, 0.0).sum())
    added_flux = float(m.sum() / flux_in_seg) if flux_in_seg > 0 else np.nan
    info = {
        "fpsf": float(fpsf),
        "snr_seg": float(snr_seg),
        "flux_in_seg": flux_in_seg,
        "added_flux": added_flux,
        "psf_replaced": replaced,
    }
    return m, info


# ---------------------------------------------------------------------------
# scheme 'wren': dev-wren templates.py::_extended_composite
# ---------------------------------------------------------------------------


def composite_wren(
    data: np.ndarray,
    seg: np.ndarray,
    owned: np.ndarray,
    label: int,
    psf_stamp: np.ndarray | None,
    ivar: np.ndarray | None,
    center: tuple[float, float],
    *,
    params: WrenParams,
    max_radius_pix: float,
    ee_reach_pix: float,
    annulus_pix: float,
    bg_rms: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build one template the way wren ``_extended_composite`` does.

    A single radial, SNR-weighted linear blend between the detection data and a
    core-anchored PSF model,

    .. math::

        H = W D + (1 - W) M, \\qquad M = A_{\\rm src} P,
        \\qquad A_{\\rm src} = \\frac{\\sum_S \\max(D, 0)}{\\sum_S P},

    over the source's owned territory. One weight ``w_core = w(S_seg;
    1.5*fit_snrlo_psf)`` for the whole segment, one weight ``w(S_k;
    wings_snr_psf)`` per halo annulus, the halo ladder forced monotone
    non-increasing outward and seeded at ``w_core`` -- so data trust never
    increases with radius and a faint core caps its own halo.

    The support is ``own | (owned background within ee_reach_pix)``; halo
    weights are measured out to ``max_radius_pix``, which is ``>= ee_reach_pix``
    by construction, so the extra margin only buys ownership reach.

    Parameters
    ----------
    data
        Raw detection-image stamp (NaNs allowed: they take the PSF model).
    seg, owned
        Segmentation stamp and this source's ownership mask
        (``owner_map == label``), aligned with ``data``.
    label
        This source's segmentation label.
    psf_stamp
        Unit-sum detection PSF resampled onto the stamp at the source position
        (:func:`sample_psf_on_stamp` with ``order=1``). ``None`` triggers wren's
        failure branch: raw data over the *larger* ``ext_data`` support, no
        blend and no truncation bookkeeping.
    ivar, bg_rms
        Detection inverse variance, or a scalar sky rms fallback.
    center
        Source position ``(x, y)`` in stamp pixels.
    max_radius_pix, ee_reach_pix, annulus_pix
        Resolved reaches in stamp pixels (``r_fill``, ``R_ee``, annulus width).

    Returns
    -------
    tuple
        ``(composite, info)``; ``info`` carries ``snr_seg``, ``w_core``,
        ``A_src``, ``f_cut``, ``flux_beyond_stamp``, ``flux_beyond_aper`` and
        the ``psf_extended`` / ``extend_failed`` flags.

    Notes
    -----
    The positivity clip and the unit-sum normalisation are the caller's job
    (wren clips before recording ``template_norm``, which biases the stored
    norm high over the ~855-pixel support -- see ``template_comparison.tex``
    Sec. 7.1).
    """
    seg = np.asarray(seg)
    own = seg == label
    owned = np.asarray(owned, dtype=bool)
    arr = np.asarray(data)
    finite = np.isfinite(arr)
    data_f = np.where(finite, arr, 0.0).astype(float)

    h, w = own.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xs, ys = float(center[0]), float(center[1])
    r2 = (xx - xs) ** 2 + (yy - ys) ** 2

    bg_owned = owned & (seg == 0)
    ext_data = own | (bg_owned & (r2 <= float(max_radius_pix) ** 2))
    ext_psf = own | (bg_owned & (r2 <= float(ee_reach_pix) ** 2))

    snr_seg, _ = _region_snr(arr, ivar, own, bg_rms)
    w_core = blend_weight(snr_seg, 1.5 * params.fit_snrlo_psf, params.blend_p)
    info: dict[str, float] = {
        "snr_seg": float(snr_seg),
        "w_core": float(w_core),
        # wren's _model_kron shortcut gate: majority-PSF sources skip the
        # photutils Kron measurement downstream.
        "apcor_from_psf": bool(snr_seg < params.fit_snrlo_psf),
        "A_src": 0.0,
        "f_cut": 0.0,
        "flux_beyond_stamp": 0.0,
        "flux_beyond_aper": 0.0,
        "psf_extended": False,
        "extend_failed": False,
    }

    if psf_stamp is None:
        info["extend_failed"] = True
        return data_f * ext_data, info
    psf_cut = np.asarray(psf_stamp, dtype=float)

    f_own_psf = float(psf_cut[own].sum())
    if f_own_psf < 1e-8:
        info["extend_failed"] = True
        return data_f * ext_data, info

    A_src = float(np.maximum(data_f[own], 0.0).sum()) / f_own_psf
    M = A_src * psf_cut
    info["A_src"] = A_src

    # Halo weights: one per radial annulus over owned background within r_fill.
    halo_mask = bg_owned & (r2 <= float(max_radius_pix) ** 2)
    halo_ok = halo_mask & finite
    bin_idx = (np.sqrt(r2) / float(annulus_pix)).astype(int)
    if halo_mask.any():
        n_bins = int(bin_idx[halo_mask].max()) + 1
        flux_k = np.bincount(bin_idx[halo_ok], weights=arr[halo_ok], minlength=n_bins)[:n_bins]
        n_k = np.bincount(bin_idx[halo_ok], minlength=n_bins)[:n_bins].astype(float)
        if ivar is not None:
            good = halo_ok & (np.asarray(ivar) > 0)
            inv_k = np.bincount(
                bin_idx[good], weights=1.0 / np.asarray(ivar, dtype=float)[good], minlength=n_bins
            )[:n_bins]
            good_n_k = np.bincount(bin_idx[good], minlength=n_bins)[:n_bins]
        else:
            inv_k = np.zeros(n_bins)
            good_n_k = np.zeros(n_bins)
        noise_k = np.where(good_n_k > 0, np.sqrt(inv_k), 0.0)
        if bg_rms and bg_rms > 0:
            noise_k = np.where(noise_k > 0, noise_k, bg_rms * np.sqrt(np.maximum(n_k, 0.0)))
        snr_k = np.zeros(n_bins)
        has_noise = noise_k > 0
        snr_k[has_noise] = np.maximum(flux_k[has_noise], 0.0) / noise_k[has_noise]
        w_k = np.array([blend_weight(s, params.wings_snr_psf, params.blend_p) for s in snr_k])
        w_k[n_k <= 0] = 1.0  # empty annulus: no constraint -> inherits the running minimum
        # Monotone non-increasing outward, seeded at w_core.
        w_k = np.minimum.accumulate(np.concatenate(([w_core], w_k)))[1:]
    else:
        w_k = np.zeros(0)

    W = np.zeros(arr.shape, dtype=float)
    W[own] = w_core
    if halo_mask.any():
        W[halo_mask] = w_k[np.clip(bin_idx[halo_mask], 0, len(w_k) - 1)]
    # Beyond max_radius_pix but inside ee_reach: W stays 0 -> pure PSF model.
    # Non-finite data pixels likewise take the model, whatever their annulus.
    W[~finite] = 0.0
    H = np.where(ext_psf, W * data_f + (1.0 - W) * M, 0.0)

    info["psf_extended"] = bool(w_core < 1.0 or (w_k.size and np.any(w_k < 1.0)))

    # Truncation bookkeeping. f_cut is the PSF fraction inside the support H is
    # actually built over, so the faint limit sum(H) == A_src*f_cut holds exactly.
    c_det = float(params.containment)
    if not (np.isfinite(c_det) and c_det > 0):
        c_det = 1.0
    f_cut = float(psf_cut[ext_psf].sum())
    info["f_cut"] = f_cut
    info["flux_beyond_stamp"] = max(A_src * (1.0 / c_det - f_cut), 0.0)

    # Crowding deficit: a neighbour's ownership boundary can truncate ext_psf
    # inside the measurement aperture, where H is identically zero even though
    # this source's PSF tail genuinely extends there. H_corr - H == A_src*psf_cut
    # exactly outside ext_psf, so the delta is the aperture sum of psf_cut
    # restricted to outside the support.
    r_ap = params.aperture_radius_pix
    if r_ap is not None and r_ap > 0:
        r_ap = float(r_ap)
        y0, y1 = max(int(np.floor(ys - r_ap)), 0), min(int(np.ceil(ys + r_ap)) + 1, h)
        x0, x1 = max(int(np.floor(xs - r_ap)), 0), min(int(np.ceil(xs + r_ap)) + 1, w)
        if (~ext_psf[y0:y1, x0:x1]).any():
            from photutils.aperture import CircularAperture

            overlap = CircularAperture((xs, ys), r=r_ap).to_mask(method="exact").multiply(
                psf_cut * ~ext_psf
            )
            delta = float(overlap.sum()) if overlap is not None else 0.0
            info["flux_beyond_aper"] = max(A_src * delta, 0.0)

    return H, info
