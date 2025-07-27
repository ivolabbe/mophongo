from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.sparse import eye
from scipy.sparse.linalg import cg

from .fit import SparseFitter, FitConfig
from .templates import Template
from . import astrometry


def _make_gradients(templates: list[Template]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gx_list: list[np.ndarray] = []
    gy_list: list[np.ndarray] = []
    for tmpl in templates:
        gy, gx = np.gradient(tmpl.data[tmpl.slices_cutout].astype(float))
        gx_list.append(gx)
        gy_list.append(gy)
    return gx_list, gy_list


def _basis_matrix(templates: list[Template], segmap: np.ndarray, order: int) -> np.ndarray:
    n = len(templates)
    k = astrometry.n_terms(order)
    h, w = segmap.shape
    phi = np.empty((n, k), dtype=float)
    for i, tmpl in enumerate(templates):
        x, y = tmpl.position_original
        phi[i] = astrometry.cheb_basis(x / (w - 1), y / (h - 1), order)
    return phi


def _collapse_gradients(
    templates: list[Template],
    gx_list: list[np.ndarray],
    gy_list: list[np.ndarray],
    phi: np.ndarray,
    k: int,
    shape: tuple[int, int],
) -> tuple[list[Template], list[Template]]:
    gx_templates = [np.zeros(shape, dtype=float) for _ in range(k)]
    gy_templates = [np.zeros(shape, dtype=float) for _ in range(k)]
    for i, tmpl in enumerate(templates):
        sl = tmpl.slices_original
        for j in range(k):
            w = phi[i, j]
            if w == 0:
                continue
            gx_templates[j][sl] += w * gx_list[i]
            gy_templates[j][sl] += w * gy_list[i]
    tmpl_gx = [Template(g, (shape[1] / 2, shape[0] / 2), shape) for g in gx_templates]
    tmpl_gy = [Template(g, (shape[1] / 2, shape[0] / 2), shape) for g in gy_templates]
    return tmpl_gx, tmpl_gy


class GlobalAstroFitter(SparseFitter):
    """Sparse fitter with optional global astrometric corrections."""

    def __init__(
        self,
        templates: list[Template],
        image: np.ndarray,
        weights: np.ndarray | None,
        segmap: np.ndarray,
        config: FitConfig,
    ) -> None:
        super().__init__(templates, image, weights, config)

        if not self.config.fit_astrometry:
            return

        order = self.config.astrom_basis_order
        k = astrometry.n_terms(order)
        self.basis_order = order
        self.n_alpha = k
        self.n_flux = len(templates)

        gx, gy = _make_gradients(templates)
        phi = _basis_matrix(templates, segmap, order)
        GX, GY = _collapse_gradients(templates, gx, gy, phi, k, image.shape)

        self.templates.extend(GX + GY)

    def _apply_shifts(self, alpha: np.ndarray, beta: np.ndarray) -> None:
        order = self.basis_order
        for tmpl in self.templates[: self.n_flux]:
            h, w = self.image.shape
            x_pix, y_pix = tmpl.position_original
            phi = astrometry.cheb_basis(x_pix / (w - 1), y_pix / (h - 1), order)
            dx = float(np.dot(alpha, phi))
            dy = float(np.dot(beta, phi))
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                continue
            tmpl.data[:, :] = nd_shift(
                tmpl.data,
                (dy, dx),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            tmpl.position_original = (x_pix - dx, y_pix - dy)

    def solve(self, config: FitConfig | None = None):
        cfg = config or self.config
        ata = self.ata
        atb = self.atb
        if cfg.fit_astrometry and cfg.reg_astrom is not None:
            diag_idx = np.arange(self.n_flux, self.n_flux + 2 * self.n_alpha)
            ata = ata.tolil()
            ata[diag_idx, diag_idx] = ata[diag_idx, diag_idx] + cfg.reg_astrom
            ata = ata.tocsr()
        if cfg.reg and cfg.reg > 0:
            ata = ata + eye(ata.shape[0]) * cfg.reg
        x, info = cg(ata, atb, **cfg.cg_kwargs)
        if cfg.positivity:
            x = np.where(x < 0, 0, x)
        if cfg.fit_astrometry:
            alpha = x[self.n_flux : self.n_flux + self.n_alpha]
            beta = x[self.n_flux + self.n_alpha : self.n_flux + 2 * self.n_alpha]
            self._apply_shifts(alpha, beta)
            sf = SparseFitter(self.templates[: self.n_flux], self.image, self.weights, cfg)
            fluxes, _ = sf.solve(cfg)
            x[: self.n_flux] = fluxes
        self.solution = x
        return x, info

