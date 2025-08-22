"""Astrometry utilities for local shift measurement and modelling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from astropy.table import Table
from numpy.polynomial.chebyshev import chebval
from photutils.centroids import centroid_com, centroid_quadratic
from scipy.ndimage import shift as nd_shift
from scipy.signal import correlate2d, fftconvolve
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from .templates import Template

import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="photutils.centroids.core")


def cheb_basis(x: float, y: float, order: int) -> np.ndarray:
    """Return Chebyshev basis values T_i(x)T_j(y) up to ``order``."""
    u = 2 * x - 1.0
    v = 2 * y - 1.0
    tx = [chebval(u, [0] * i + [1]) for i in range(order + 1)]
    ty = [chebval(v, [0] * j + [1]) for j in range(order + 1)]
    phi = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            phi.append(tx[i] * ty[j])
    return np.array(phi, dtype=float)


def n_terms(order: int) -> int:
    """Number of 2-D Chebyshev terms for ``order``."""
    return (order + 1) * (order + 2) // 2


def make_gradients(templates):
    """Return per-template gradient images on the full frame."""
    gx, gy = [], []
    for tmpl in templates:
        dy, dx = np.gradient(tmpl.data.astype(float))
        gxi = deepcopy(tmpl)
        gyi = deepcopy(tmpl)
        gxi.data = dx
        gyi.data = dy
        gx.append(gxi)
        gy.append(gyi)
    return gx, gy


def basis_matrix(templates, shape, order):
    """Evaluate basis functions at template centres."""
    h, w = shape
    mat = np.zeros((len(templates), n_terms(order)), dtype=float)
    for i, tmpl in enumerate(templates):
        x, y = tmpl.input_position_original
        mat[i] = cheb_basis(x / (w - 1), y / (h - 1), order)
    return mat


def _safe_centroid_quadratic(img, x0, y0, box):
    """Quadratic centroid with COM fallback."""
    cx, cy = centroid_quadratic(img, xpeak=x0, ypeak=y0, fit_boxsize=box)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = centroid_com(img)
    return cx, cy


def _xcorr_shift(
    model: np.ndarray, tmpl: np.ndarray, centre: Tuple[float, float], box: int = 5
) -> np.ndarray:
    """Sub-pixel shift from the peak of the normalised cross-correlation."""
    cc = fftconvolve(model, tmpl[::-1, ::-1], mode="same")
    j, i = np.unravel_index(np.argmax(cc), cc.shape)
    sl_y = slice(max(0, j - box), min(cc.shape[0], j + box + 1))
    sl_x = slice(max(0, i - box), min(cc.shape[1], i + box + 1))
    sub = cc[sl_y, sl_x]
    try:
        cx, cy = centroid_quadratic(sub, xpeak=i - sl_x.start, ypeak=j - sl_y.start)
    except Exception:
        cx, cy = np.nan, np.nan
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = centroid_com(sub)
    if np.isnan(cx) or np.isnan(cy):
        return np.array([np.nan, np.nan])
    xc, yc = centre
    dx = (sl_x.start + cx) - xc
    dy = (sl_y.start + cy) - yc
    return np.array([dx, dy])


# ---------------------------------------------------------------------------
# Raw measurements
# ---------------------------------------------------------------------------


def measure_template_shifts(
    templates: Sequence[Template],
    coeffs: np.ndarray,
    residual: np.ndarray,
    *,
    box_size: int = 5,
    snr_threshold: float = 7.0,
    method: str = "quadratic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, dx, dy and weights for usable templates."""
    positions, dx, dy, wt = [], [], [], []

    method = method.lower()
    use_xcorr = method == "correlation"

    for tmpl, ampl in zip(templates, coeffs):
        if tmpl.err <= 0:
            continue
        snr = tmpl.flux / tmpl.err
        if snr < snr_threshold:
            continue

        x_img, y_img = tmpl.input_position_original
        x_cut, y_cut = tmpl.input_position_cutout

        cut_res = Cutout2D(
            residual, position=(x_img, y_img), size=3 * box_size + 1, mode="partial"
        )
        cut_tmpl = Cutout2D(
            tmpl.data, position=(x_cut, y_cut), size=3 * box_size + 1, mode="partial"
        )

        model = ampl * cut_tmpl.data + cut_res.data
        centre = cut_tmpl.input_position_cutout

        if use_xcorr:
            shift = _xcorr_shift(model, cut_tmpl.data, centre, box=box_size)
        else:
            try:
                cx_m, cy_m = centroid_quadratic(model, *centre, fit_boxsize=box_size)
                cx_t, cy_t = centroid_quadratic(cut_tmpl.data, *centre, fit_boxsize=box_size)
            except Exception:
                cx_m = cy_m = cx_t = cy_t = np.nan

            if np.isnan(cx_m) or np.isnan(cy_m) or np.isnan(cx_t) or np.isnan(cy_t):
                cx_m, cy_m = centroid_com(model)
                cx_t, cy_t = centroid_com(cut_tmpl.data)

            shift = np.array([cx_m - cx_t, cy_m - cy_t])

        if np.isnan(shift).any():
            continue

        positions.append((x_img, y_img))
        dx.append(float(shift[0]))
        dy.append(float(shift[1]))
        wt.append(snr**2)

    return (np.asarray(positions), np.asarray(dx), np.asarray(dy), np.asarray(wt))


def measure_template_shifts_old(
    templates: Sequence[Template],
    coeffs: np.ndarray,
    residual: np.ndarray,
    *,
    box_size: int = 5,
    snr_threshold: float = 5.0,
    method: str = "centroid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-template image ⇄ model shift."""

    if method not in {"centroid", "correlation"}:
        raise ValueError("method must be 'centroid' or 'correlation'")

    pos, dx, dy, w = [], [], [], []

    for tmpl, amp in zip(templates, coeffs):
        if tmpl.flux <= 0 or tmpl.err <= 0:
            continue
        snr = tmpl.flux / tmpl.err
        if snr < snr_threshold:
            continue

        x_pix, y_pix = tmpl.input_position_original
        x_loc, y_loc = tmpl.input_position_cutout
        stamp_res = Cutout2D(residual, (x_pix, y_pix), 3 * box_size + 1)
        stamp_tmp = Cutout2D(tmpl.data, (x_loc, y_loc), 3 * box_size + 1)

        model = amp * stamp_tmp.data + stamp_res.data
        xc, yc = stamp_tmp.input_position_cutout

        if method == "centroid":
            cx_m, cy_m = _safe_centroid_quadratic(model, xc, yc, box_size)
            cx_t, cy_t = _safe_centroid_quadratic(stamp_tmp.data, xc, yc, box_size)
        else:
            tmpl0 = stamp_tmp.data.astype(float) - stamp_tmp.data.mean()
            mod0 = model.astype(float) - model.mean()
            denom = tmpl0.std() * mod0.std()
            if denom == 0:
                continue
            ncc = correlate2d(mod0 / denom, tmpl0, mode="same", boundary="fill")
            py, px = np.unravel_index(np.argmax(ncc), ncc.shape)
            cx_m, cy_m = _safe_centroid_quadratic(ncc, px, py, box_size)
            cx_t, cy_t = xc, yc

        shift = np.array([cx_m - cx_t, cy_m - cy_t], float)
        if np.isnan(shift).any():
            continue

        pos.append((x_pix, y_pix))
        dx.append(shift[0])
        dy.append(shift[1])
        w.append(snr**2)

    return (
        np.asarray(pos, float),
        np.asarray(dx, float),
        np.asarray(dy, float),
        np.asarray(w, float),
    )


def fit_polynomial_field(
    pos: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    w: np.ndarray,
    *,
    order: int,
    shape: tuple[int, int],
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Fit a polynomial shift field and return a prediction function."""

    phi = np.array([cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order) for x, y in pos])
    W = np.diag(w)
    coeff_x = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dx))
    coeff_y = np.linalg.solve(phi.T @ W @ phi, phi.T @ (w * dy))

    def predict(p):
        phi_p = np.array([cheb_basis(x / (shape[1] - 1), y / (shape[0] - 1), order) for x, y in p])
        return phi_p @ coeff_x, phi_p @ coeff_y

    return predict


# ---------------------------------------------------------------------------
#   main façade
# ---------------------------------------------------------------------------


@dataclass
class AstroCorrect:
    """Smooth, local astrometric correction."""

    cfg: "FitConfig"

    _predict: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, default=lambda p: (np.zeros(len(p)), np.zeros(len(p)))
    )

    @staticmethod
    def build_poly_predictor(
        coeffs: np.ndarray,
        x_cen: float,
        y_cen: float,
        order: int,
        Sx: float = 1.0,
        Sy: float = 1.0,
    ):
        p = len(cheb_basis(0.0, 0.0, order))
        bx = np.asarray(coeffs[:p], float)
        by = np.asarray(coeffs[p : 2 * p], float) if coeffs.size >= 2 * p else np.zeros(p)

        def predict(x, y=None):
            if y is None:
                pts = np.asarray(x, float).reshape(-1, 2)
                shape = pts.shape[:-1]
                u = (pts[:, 0] - x_cen) / Sx
                v = (pts[:, 1] - y_cen) / Sy
            else:
                X = np.asarray(x, float)
                Y = np.asarray(y, float)
                shape = np.broadcast(X, Y).shape
                u = (X.ravel() - x_cen) / Sx
                v = (Y.ravel() - y_cen) / Sy
            Phi = np.vstack([cheb_basis(ui, vi, order) for ui, vi in zip(u, v)])
            dx = Phi @ bx
            dy = Phi @ by
            return dx.reshape(shape), dy.reshape(shape)

        return predict

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

    def fit(
        self,
        templates: Sequence[Template],
        residual: np.ndarray,
        coeffs: np.ndarray,
    ) -> None:
        astrom_kw = self.cfg.astrom_kwargs.get(self.cfg.astrom_model.lower(), {})
        pos, dx, dy, w = measure_template_shifts(
            templates,
            coeffs,
            residual,
            box_size=astrom_kw.pop("box_size", 7),
            snr_threshold=self.cfg.snr_thresh_astrom,
            method=self.cfg.astrom_centroid.lower(),
        )

        if pos.size == 0:
            self._predict = lambda p: (np.zeros(len(p)), np.zeros(len(p)))
            return

        if self.cfg.astrom_model.lower() == "poly":
            self._predict = fit_polynomial_field(
                pos, dx, dy, w, order=astrom_kw.pop("order", 2), shape=residual.shape
            )
        elif self.cfg.astrom_model.lower() == "gp":
            self._predict = self._fit_gp(pos, dx, dy, w, **astrom_kw)
        else:
            raise ValueError(f"Unknown astrom_model '{self.cfg.astrom_model}'")

        dxx, dyy = self(np.array([t.input_position_original for t in templates]))
        for tmpl, dxi, dyi in zip(templates, dxx, dyy):
            if abs(dxi) < 1e-2 and abs(dyi) < 1e-2:
                continue
            x0, y0 = tmpl.input_position_original
            tmpl.data = nd_shift(
                tmpl.data,
                (dyi, dxi),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            tmpl.shifted += [dxi, dyi]

    def _fit_gp(
        self,
        pos: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        w: np.ndarray,
        *,
        length_scale: float = 300.0,
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        err = 1 / np.sqrt(w)
        base = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale, (length_scale / 2, length_scale * 2)
        )
        gp = GaussianProcessRegressor(
            base + WhiteKernel(err.mean() ** 2, (1e-6, 1e2)),
            alpha=err**2,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=0,
        )
        gpx, gpy = clone(gp), clone(gp)
        gpx.fit(pos, dx)
        gpy.fit(pos, dy)

        return lambda p: (gpx.predict(p), gpy.predict(p))


# ---------------------------------------------------------------------------
#   AstroMap – image-to-image shift mapping
# ---------------------------------------------------------------------------


@dataclass
class AstroMap:
    """Map relative shifts between two images using segmentation labels."""

    order: int = 2
    snr_threshold: float = 5.0
    method: str = "quadratic"
    box_size: int = 5

    _predict: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, default=lambda p: (np.zeros(len(p)), np.zeros(len(p)))
    )

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

    def fit(self, img1: np.ndarray, img2: np.ndarray, catalog: Table, **kwargs) -> None:
        pos, dx, dy, wt = self._measure(img1, img2, catalog, **kwargs)
        self.pos = pos
        self.dxy = np.column_stack((dx, dy))
        if pos.size == 0:
            self._predict = lambda p: (np.zeros(len(p)), np.zeros(len(p)))
            return
        self._predict = fit_polynomial_field(pos, dx, dy, wt, order=self.order, shape=img1.shape)

    def _measure(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        catalog: Table,
        snr_threshold: float = 5.0,
        snr_key: str = "snr",
        wcs1: WCS = None,
        wcs2: WCS = None,
        pixel_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Measure shifts between two images at catalog positions.
        If WCS objects are provided, cutouts are centered on the same sky position (RA, Dec).
        """
        logger.info(
            f"Measuring shifts between images with {len(catalog)} sources and SNR > {snr_threshold}"
        )
        if snr_key not in catalog.colnames:
            catalog[snr_key] = snr_threshold
        positions, dx, dy, w = [], [], [], []
        for row in catalog[catalog[snr_key] > snr_threshold]:
            if wcs1 is not None and wcs2 is not None:
                ra, dec = row["ra"], row["dec"]
                # Convert pixel to sky in img1, then to pixel in img2
                x_c, y_c = wcs1.wcs_world2pix(ra, dec, 0)
                x2, y2 = wcs2.wcs_world2pix(ra, dec, 0)
                # Use same bounding box in pixels, but center on sky position
                cut1 = Cutout2D(img1, (x_c, y_c), 3 * self.box_size + 1, mode="partial", wcs=wcs1)
                #             print(x_c, y_c, x2, y2)
                cut2 = Cutout2D(img2, (x2, y2), 3 * self.box_size + 1, mode="partial", wcs=wcs2)
                centre1 = cut1.input_position_cutout
                centre2 = cut2.input_position_cutout
            else:
                x_c, y_c = row["x"], row["y"]
                cut1 = Cutout2D(img1, (x_c, y_c), 3 * self.box_size + 1, mode="partial")
                cut2 = Cutout2D(img2, (x_c, y_c), 3 * self.box_size + 1, mode="partial")
                centre1 = centre2 = cut1.input_position_cutout

            if self.method.lower() == "correlation" and (wcs1 is None or wcs2 is None):
                shift = _xcorr_shift(cut2.data, cut1.data, centre2, box=self.box_size)
            else:
                try:
                    cx1, cy1 = _safe_centroid_quadratic(
                        cut1.data, centre1[0], centre1[1], self.box_size
                    )
                    cx2, cy2 = _safe_centroid_quadratic(
                        cut2.data, centre2[0], centre2[1], self.box_size
                    )
                except Exception:
                    continue
                if wcs1 is not None and wcs2 is not None:
                    # Convert to pixel shifts
                    r1, d1 = cut1.wcs.wcs_pix2world(cx1, cy1, 0)
                    r2, d2 = cut2.wcs.wcs_pix2world(cx2, cy2, 0)
                    cx2, cy2 = cut1.wcs.wcs_world2pix(r2, d2, 0)

                # shift = np.array([r1-r2, d2 - d1]) * 3600 times cos delta for ra
                shift = np.array([cx2 - cx1, cy2 - cy1]) * pixel_scale
            if np.isnan(shift).any():
                continue
            positions.append((x_c, y_c))
            dx.append(float(shift[0]))
            dy.append(float(shift[1]))
            w.append(row[snr_key] ** 2)
        return (
            np.asarray(positions),
            np.asarray(dx),
            np.asarray(dy),
            np.asarray(w),
        )
