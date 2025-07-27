"""Wrapper fitter adding global astrometry parameters."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.sparse import eye
from scipy.sparse.linalg import cg

from .fit import SparseFitter, FitConfig
from .templates import Template
from . import astrometry


class GlobalAstroFitter(SparseFitter):
    """Sparse fitter with Chebyshev astrometric offsets."""

    def __init__(
        self,
        templates: List[Template],
        image: np.ndarray,
        weights: np.ndarray | None,
        segmap: np.ndarray,
        config: FitConfig,
    ) -> None:
        super().__init__(templates, image, weights, config)
        if not config.fit_astrometry:
            return

        order = config.astrom_basis_order
        k = astrometry.n_terms(order)
        self.basis_order = order
        self.n_alpha = k
        self.n_flux = len(templates)

        gx, gy = astrometry.make_gradients(templates, image.shape)
        phi = astrometry.basis_matrix(templates, segmap, order)
        GX, GY = astrometry.collapse_gradients(gx, gy, phi, k, image.shape)

        self.templates.extend(
            [Template(GX[i], (image.shape[1] / 2, image.shape[0] / 2), image.shape) for i in range(k)]
            + [Template(GY[i], (image.shape[1] / 2, image.shape[0] / 2), image.shape) for i in range(k)]
        )

    def _apply_shifts(self, alpha: np.ndarray, beta: np.ndarray) -> None:
        order = self.basis_order
        n_alpha = self.n_alpha
        h, w = self.image.shape
        for tmpl in self.templates[: self.n_flux]:
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
        if cfg.reg and cfg.reg > 0:
            ata = ata + eye(ata.shape[0]) * cfg.reg
        if cfg.fit_astrometry and cfg.reg_astrom is not None:
            start = self.n_flux
            diag_idx = np.arange(start, start + 2 * self.n_alpha)
            ata = ata.tolil()
            ata[diag_idx, diag_idx] += cfg.reg_astrom
            ata = ata.tocsr()
        x, info = cg(ata, atb, **cfg.cg_kwargs)
        if cfg.positivity:
            x = np.where(x < 0, 0, x)
        if cfg.fit_astrometry:
            alpha = x[self.n_flux : self.n_flux + self.n_alpha]
            beta = x[self.n_flux + self.n_alpha : self.n_flux + 2 * self.n_alpha]
            self._apply_shifts(alpha, beta)
            flux_only, _ = super().solve(cfg)
            x[: self.n_flux] = flux_only
        self.solution = x
        return x, info

