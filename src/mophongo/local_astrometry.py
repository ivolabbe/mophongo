"""
local_astrometry.py — per-source shift measurement and smooth-field modelling
----------------------------------------------------------------------------

Drop-in replacement that adds a second centroiding mode based on the peak of
the *normalised cross-correlation* map.

* `FitConfig.astrom_centroid`
      "centroid"   – original quadratic-centroid difference (default)  
      "correlation"– peak of the NCC map ( centroid_quadratic on NCC )

If the quadratic centroid fails (returns NaNs) **either** mode falls back
to centre-of-mass (`centroid_com`).  Measurements whose fallback still
returns NaNs are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com, centroid_quadratic
from scipy.ndimage import shift as nd_shift
from scipy.signal import correlate2d
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from scipy.signal                 import fftconvolve

from . import astrometry
from .templates import Template


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_centroid_quadratic(img, x0, y0, box):
    """Quadratic centroid with COM fallback."""
    cx, cy = centroid_quadratic(img, xpeak=x0, ypeak=y0, fit_boxsize=box)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = centroid_com(img)
    return cx, cy

# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
def _xcorr_shift(model: np.ndarray,
                 tmpl : np.ndarray,
                 centre: Tuple[float, float],
                 box   : int = 5) -> np.ndarray:
    """
    Sub-pixel shift from the peak of the normalised cross-correlation
        cc = (model ⋆ tmpl) / (‖model‖‖tmpl‖)

    A quadratic centroid is attempted around the integer peak; on failure
    we fall back to a centre-of-mass estimate.
    """
    # full correlation (same size as inputs)
    cc = fftconvolve(model, tmpl[::-1, ::-1], mode="same")

    # integer-pixel peak
    j, i = np.unravel_index(np.argmax(cc), cc.shape)

    # restrict to a small postage stamp around the peak
    sl_y = slice(max(0, j-box), min(cc.shape[0], j+box+1))
    sl_x = slice(max(0, i-box), min(cc.shape[1], i+box+1))
    sub  = cc[sl_y, sl_x]

    # refined centroid
    try:
        cx, cy = centroid_quadratic(sub, xpeak=i-sl_x.start, ypeak=j-sl_y.start)
    except Exception:    # numerical failure
        cx, cy = np.nan, np.nan

    if np.isnan(cx) or np.isnan(cy):
        cx, cy = centroid_com(sub)

    if np.isnan(cx) or np.isnan(cy):
        return np.array([np.nan, np.nan])

    # shift relative to *template* frame centre
    xc, yc = centre
    dx = (sl_x.start + cx) - xc
    dy = (sl_y.start + cy) - yc
    return np.array([dx, dy])


# ──────────────────────────────────────────────────────────────────────
# raw measurements
# ──────────────────────────────────────────────────────────────────────
def measure_template_shifts(
    templates     : Sequence[Template],
    coeffs        : np.ndarray,
    residual      : np.ndarray,
    *,
    box_size      : int   = 5,
    snr_threshold : float = 7.0,
    method        : str   = "quadratic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (positions, dx, dy, weights) for all usable templates.
    ``method`` ∈ {"quadratic", "correlation"}.
    """
    positions, dx, dy, wt = [], [], [], []

    method = method.lower()
    use_xcorr = (method == "correlation")

    for tmpl, ampl in zip(templates, coeffs):
        if tmpl.err <= 0:                # bad error → skip
            continue
        snr = tmpl.flux / tmpl.err
        if snr < snr_threshold:
            continue

        # positions in *parent* and *cut-out* frames
        x_img, y_img = tmpl.input_position_original
        x_cut, y_cut = tmpl.input_position_cutout

        # postage-stamps (3× box for safety)
        cut_res  = Cutout2D(residual, position=(x_img, y_img),
                            size=3*box_size+1, mode="partial")
        cut_tmpl = Cutout2D(tmpl.data, position=(x_cut, y_cut),
                            size=3*box_size+1, mode="partial")

        model = ampl * cut_tmpl.data + cut_res.data
        centre = cut_tmpl.input_position_cutout  # (xc, yc) in cut_tmpl

        if use_xcorr:
            shift = _xcorr_shift(model, cut_tmpl.data, centre, box=box_size)
        else:
            # classic quadratic centroid on individual frames
            try:
                cx_m, cy_m = centroid_quadratic(model, *centre, fit_boxsize=box_size)
                cx_t, cy_t = centroid_quadratic(cut_tmpl.data, *centre, fit_boxsize=box_size)
            except Exception:
                cx_m = cy_m = cx_t = cy_t = np.nan

            if np.isnan(cx_m) or np.isnan(cy_m) or \
               np.isnan(cx_t) or np.isnan(cy_t):
                cx_m, cy_m = centroid_com(model)
                cx_t, cy_t = centroid_com(cut_tmpl.data)

            shift = np.array([cx_m - cx_t, cy_m - cy_t])

        if np.isnan(shift).any():
            continue

        positions.append((x_img, y_img))
        dx.append(float(shift[0]))
        dy.append(float(shift[1]))
        wt.append(snr**2)

    return (np.asarray(positions), np.asarray(dx), np.asarray(dy),
            np.asarray(wt))


def measure_template_shifts_old(
    templates: Sequence[Template],
    coeffs: np.ndarray,
    residual: np.ndarray,
    *,
    box_size: int = 5,
    snr_threshold: float = 5.0,
    method: str = "centroid",  # "centroid" | "correlation"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate per-template image ⇄ model shift.

    Parameters
    ----------
    method : {"centroid","correlation"}
        How to derive the local offset (see module docstring).
    """

    if method not in {"centroid", "correlation"}:
        raise ValueError("method must be 'centroid' or 'correlation'")

    pos, dx, dy, w = [], [], [], []

    for tmpl, amp in zip(templates, coeffs):
        if tmpl.flux <= 0 or tmpl.err <= 0:
            continue
        snr = tmpl.flux / tmpl.err
        if snr < snr_threshold:
            continue

        x_pix, y_pix = tmpl.input_position_original          # parent coords
        x_loc, y_loc = tmpl.input_position_cutout            # cut-out coords
        stamp_res    = Cutout2D(residual, (x_pix, y_pix), 3 * box_size + 1)
        stamp_tmp    = Cutout2D(tmpl.data, (x_loc, y_loc), 3 * box_size + 1)

        model  = amp * stamp_tmp.data + stamp_res.data       # local model
        xc, yc = stamp_tmp.input_position_cutout             # nominal centre

        # ------------------------------------------------------------------
        #  mode A – difference of centroids
        # ------------------------------------------------------------------
        if method == "centroid":
            cx_m, cy_m = _safe_centroid_quadratic(model,       xc, yc, box_size)
            cx_t, cy_t = _safe_centroid_quadratic(stamp_tmp.data, xc, yc, box_size)

        # ------------------------------------------------------------------
        #  mode B – peak of the normalised cross-correlation
        # ------------------------------------------------------------------
        else:  # "correlation"
            tmpl0 = stamp_tmp.data.astype(float) - stamp_tmp.data.mean()
            mod0  = model.astype(float) - model.mean()
            denom = tmpl0.std() * mod0.std()
            if denom == 0:          # pathological, skip
                continue
            ncc  = correlate2d(mod0 / denom, tmpl0, mode="same", boundary="fill")
            py, px = np.unravel_index(np.argmax(ncc), ncc.shape)
            cx_m, cy_m = _safe_centroid_quadratic(ncc, px, py, box_size)
            cx_t, cy_t = xc, yc      # template centre by construction

        shift = np.array([cx_m - cx_t, cy_m - cy_t], float)
        if np.isnan(shift).any():
            continue

        pos.append((x_pix, y_pix))
        dx.append(shift[0])
        dy.append(shift[1])
        w.append(snr ** 2)

    return (
        np.asarray(pos, float),
        np.asarray(dx, float),
        np.asarray(dy, float),
        np.asarray(w, float),
    )


# ---------------------------------------------------------------------------
#   main façade
# ---------------------------------------------------------------------------
@dataclass
class AstroCorrect:
    """Smooth, local astrometric correction."""

    cfg: "FitConfig"

    _predict: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False,
        default=lambda p: (np.zeros(len(p)), np.zeros(len(p)))
    )

    # --------------------------------------------------------------------
    #  evaluate
    # --------------------------------------------------------------------
    def __call__(
        self, x: float | np.ndarray, y: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if y is None:
            pts = np.asarray(x, float).reshape(-1, 2)
            shape = pts.shape[:-1]
        else:
            pts = np.c_[np.asarray(x, float).ravel(), np.asarray(y, float).ravel()]
            shape = np.broadcast(x, y).shape
        dx, dy = self._predict(pts)
        return dx.reshape(shape), dy.reshape(shape)

    # --------------------------------------------------------------------
    #  fit field & update templates
    # --------------------------------------------------------------------
    def fit(
        self,
        templates: Sequence[Template],
        residual : np.ndarray,
        coeffs   : np.ndarray,
    ) -> None:

        # 1. collect raw measurements
        astrom_kw = self.cfg.astrom_kwargs.get(self.cfg.astrom_model.lower(), {})
        pos, dx, dy, w = measure_template_shifts(
            templates, coeffs, residual,
            box_size      = astrom_kw.pop("box_size", 7),
            snr_threshold = self.cfg.snr_thresh_astrom,
            method        = self.cfg.astrom_centroid.lower(),
        )

        if pos.size == 0:                       # nothing usable
            self._predict = lambda p: (np.zeros(len(p)), np.zeros(len(p)))
            return

        # 2. fit smooth field
        if self.cfg.astrom_model.lower() == "poly":
            self._predict = self._fit_polynomial(
                pos, dx, dy, w,
                shape = residual.shape,
                **astrom_kw,
            )
        elif self.cfg.astrom_model.lower() == "gp":
            self._predict = self._fit_gp(pos, dx, dy, w, **astrom_kw)
        else:
            raise ValueError(f"Unknown astrom_model '{self.cfg.astrom_model}'")

        # 3. apply shifts to templates
        dxx, dyy = self(np.array([t.input_position_original for t in templates]))
        for tmpl, dxi, dyi in zip(templates, dxx, dyy):
            if abs(dxi) < 1e-2 and abs(dyi) < 1e-2:
                continue
            x0, y0 = tmpl.input_position_original
            tmpl.data = nd_shift(
                tmpl.data, (dyi, dxi),
                order=3, mode="constant", cval=0.0, prefilter=True,
            )
            tmpl.input_position_original = (x0 - dxi, y0 - dyi)
            tmpl.shift += [dxi, dyi]

    # --------------------------------------------------------------------
    #  polynomial & GP helpers
    # --------------------------------------------------------------------
    def _fit_polynomial(
        self,
        pos  : np.ndarray,
        dx   : np.ndarray,
        dy   : np.ndarray,
        w    : np.ndarray,
        *,
        order: int = 2,
        shape: tuple[int, int],
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:

        phi = np.array([
            astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order)
            for x, y in pos
        ])
        W       = np.diag(w)
        coeff_x = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dx))
        coeff_y = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dy))

        def predict(p):
            phi_p = np.array([
                astrometry.cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order)
                for x, y in p
            ])
            return phi_p @ coeff_x, phi_p @ coeff_y

        return predict

    def _fit_gp(
        self,
        pos : np.ndarray,
        dx  : np.ndarray,
        dy  : np.ndarray,
        w   : np.ndarray,
        *,
        length_scale: float = 300.0,
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:

        err  = 1 / np.sqrt(w)
        base = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale, (100.0, 5000.0))
        gp   = GaussianProcessRegressor(
            base + WhiteKernel(err.mean() ** 2, (1e-6, 1e2)),
            alpha=err**2, normalize_y=True, n_restarts_optimizer=5, random_state=0,
        )
        gpx, gpy = clone(gp), clone(gp)
        gpx.fit(pos, dx)
        gpy.fit(pos, dy)

        return lambda p: (gpx.predict(p), gpy.predict(p))
