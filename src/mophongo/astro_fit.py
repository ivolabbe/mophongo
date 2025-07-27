"""Wrapper fitter adding global astrometry parameters."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.sparse import eye, diags
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
        _orig_templates = list(templates)
        _n_flux = len(_orig_templates)

        super().__init__(_orig_templates, image, weights, config)
        if not config.fit_astrometry:
            self.n_flux = _n_flux
            return

        order = config.astrom_basis_order
        k = astrometry.n_terms(order)
        self.basis_order = order
        self.n_alpha = k
        self.n_flux = _n_flux

        gx, gy = astrometry.make_gradients(_orig_templates, image.shape)
        phi = astrometry.basis_matrix(_orig_templates, segmap, order)
        GX, GY = astrometry.collapse_gradients(gx, gy, phi, k, image.shape)
        GX = np.atleast_3d(GX)
        GY = np.atleast_3d(GY)
        cy, cx = image.shape[0] / 2, image.shape[1] / 2
        for i in range(k):
            t = Template(GX[i], (cx, cy), image.shape)
            t.position_original = (cx, cy)
            self.templates.append(t)
        for i in range(k):
            t = Template(GY[i], (cx, cy), image.shape)
            t.position_original = (cx, cy)
            self.templates.append(t)

    # ------------------------------------------------------------------ #
    # override to return only the flux-component uncertainties
    def predicted_errors(self):
        errs = super().predicted_errors()
        return errs[: self.n_flux]

    def flux_errors(self):
        errs = super().flux_errors()
        return errs[: self.n_flux]

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
            reg_diag = np.zeros(ata.shape[0], dtype=float)
            reg_diag[start : start + 2 * self.n_alpha] = cfg.reg_astrom
            ata = ata + diags(reg_diag, 0, format="csr")
        x, info = cg(ata, atb, **cfg.cg_kwargs)
        if cfg.positivity:
            x = np.where(x < 0, 0, x)
        if cfg.fit_astrometry:
            alpha = x[self.n_flux : self.n_flux + self.n_alpha]
            beta = x[self.n_flux + self.n_alpha : self.n_flux + 2 * self.n_alpha]
            self._apply_shifts(alpha, beta)
            # Return only flux components
            flux_only = x[: self.n_flux]
            self.solution = x  # Store full solution including astrometry
            return flux_only, info
        self.solution = x
        return x, info

