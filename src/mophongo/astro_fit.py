"""Wrapper fitter adding global astrometry parameters."""

from __future__ import annotations


import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from scipy.ndimage import shift as nd_shift
from scipy.sparse import eye, diags, csr_matrix
from scipy.sparse.linalg import cg,  spilu, LinearOperator

from .fit import SparseFitter, FitConfig
from .templates import Template, Templates
from . import astrometry


class GlobalAstroFitter(SparseFitter):
    """
    Sparse photometry fitter with global Chebyshev astrometric offsets.

    * N flux templates  →  individual coefficients f_i
    * K Chebyshev terms →  global coefficients α_k, β_k
    """

    # ------------------------------------------------------------
    # 1.  constructor
    # ------------------------------------------------------------
    def __init__(
        self,
        templates: list[Template],
        image: np.ndarray,
        weights: np.ndarray | None,
        config: FitConfig 
    ):

        # ---------- flux part ----------
        super().__init__(list(templates), image, weights, config)

        print('GlobalAstroFitter: templates in', len(templates))
        if not self.config.fit_astrometry_niter > 0:
            print("WARNING: GlobalAstroFitter was created without astrometry enabled.")
            return      # nothing more to do

        # ---------- astrometry part ----------
        order = config.astrom_basis_order
        self.basis_order = order
        self.n_alpha = astrometry.n_terms(order)   # α_k  (β_k shares the same K)

        # get estimate for the flux and errors to scale the gradients and keep only high S/N sources for astrometry
        flux, rms = self.flux_and_rms()
        
        # 1. per-object S/N estimate        
        if self.config.snr_thresh_astrom > 0:
            good = (flux / rms) >= self.config.snr_thresh_astrom
        else:
            good = np.ones(self.n_flux, dtype=bool)
        
        print(f"GlobalAstroFitter: {self.n_flux} templates and {np.sum(good)} with S/N >= {self.config.snr_thresh_astrom} used for astrometry")
        if not np.any(good): 
            print("WARNING: No templates with S/N >= threshold, pick 10 brightest.")
            # fall back to quick flux estimates
            flux = Templates.quick_flux(self.templates, self.image)
            good = np.zeros_like(flux, dtype=bool)
            good[np.argsort(flux)[-min(10, len(flux)):]] = True
            
        # 1. per-object gradients
        gx_i, gy_i = astrometry.make_gradients(templates)

        # 2. Chebyshev basis evaluated at object centres
        Φ = astrometry.basis_matrix(templates, image.shape, order)    #  (N, K)

        # 3. build tiny templates for each term & each object
        self._big2small_col = []     # length = len(self.templates) after extension

        # We want to keep the matrix sparse, so we need to keep track of 
        # locations of the stamps in the big matrix, and then we can collapse them later
        # collapsing meaning that we trace their entrix in the sparse matrix and sum over 
        # all objects and fit for a global polynomial shift at each Chebyshev term
        # note we need to scale the gradients by the initial flux estimate
        for k in range(self.n_alpha):
            for i in np.where(good)[0]:
                f_est = flux[i]                     # per-object scale

                gx_tile = deepcopy(gx_i[i])
                gx_tile.data *= f_est * Φ[i, k]           # *** SCALE ***
                gx_tile.is_flux = False
                gx_tile.col_idx = self.n_flux + k
                self.templates.append(gx_tile)
                self._big2small_col.append(gx_tile.col_idx)

                gy_tile = deepcopy(gy_i[i])
                gy_tile.data *= f_est * Φ[i, k]           # *** SCALE ***
                gy_tile.is_flux = False
                gy_tile.col_idx = self.n_flux + self.n_alpha + k
                self.templates.append(gy_tile)
                self._big2small_col.append(gy_tile.col_idx)


    def _collapse(self):
        """Sum columns that share the same compact parameter index."""
        A_big, b_big = self._ata.tocsr(), self._atb
        cols_full    = np.fromiter((t.col_idx for t in self.templates), int)
        uniq, inv    = np.unique(cols_full, return_inverse=True)   # <— mapping

        S = csr_matrix((np.ones_like(inv), (np.arange(inv.size), inv)),
                    shape=(inv.size, uniq.size))                # selector

        self._ata_comp = (S.T @ A_big @ S).tocsr()
        self._atb_comp = (S.T @ b_big).ravel()          # 1-D
        self._compact2full = uniq                  # keep for scatter-back
        
    # ------------------------------------------------------------
    # 3.  solve   # keep track of valid fluxes through ID, not n_flux
    # ------------------------------------------------------------
    def solve(
        self,
        config: FitConfig | None = None,
        x_w0: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        cfg   = config or self.config

        # build big normal matrix once, this as a shift entry for every template
        _ = self.ata, self.atb          # triggers build_normal_matrix()

        # compact it leaving only the chebyshev terms
        self._collapse()

        A = self._ata_comp.copy()
        b = self._atb_comp.copy()

        # apply regularization by introducting a small diagonal term
        if cfg.reg:
            A = A + cfg.reg * eye(A.shape[0])

        if cfg.reg_astrom:
            # Parameters live in the *compact* space; map them back to their original
            # labels (stored in ``self._compact2full``).  Anything whose original label
            # is  ≥ n_flux  is a Chebyshev-shift coefficient.
            mask_non_flux = self._compact2full >= self.n_flux        # boolean array
            r = np.zeros(A.shape[0], dtype=float)
            r[mask_non_flux] = cfg.reg_astrom                        # α_k, β_k only
            A = A + diags(r, 0, format="csr")

        # ------------------------------------------------------------------
        # symmetric column-and-row whitening  (A → D⁻¹ A D⁻¹,  b → D⁻¹ b)
        # ------------------------------------------------------------------
        d = np.sqrt(A.diagonal())               # √(Ajj)  (never negative)
        d[d == 0.0] = 1.0                       # protect empty columns
        Dinv = diags(1.0 / d, 0, format="csr")  # D⁻¹

        A_w = Dinv @ A @ Dinv                   # still sparse, still SPD
        b_w = Dinv @ b
        
        # cg_kwargs = dict(cfg.cg_kwargs)
        # if cg_kwargs.get("M") is None and A_w.nnz > 10 * A_w.shape[0]:
        #     try:
        #         ilu = spilu(A_w.tocsc(), drop_tol=1e-4, fill_factor=10)
        #         cg_kwargs["M"] = LinearOperator(A_w.shape, ilu.solve)
        #     except Exception as err:
        #         logger.warning("ILU preconditioner failed: %s", err)

        # WARNING: this is a warm start: but only useful if the templates are the same
#        x_w0 = getattr(self, "x_w0", None)  
#        x_w, info = cg(A_w, b_w, x0=x_w0, **cfg.cg_kwargs)
        x_w, info = cg(A_w, b_w, **cfg.cg_kwargs)
        self.x_w = x_w

        # ---------- expand back to the *full* parameter space -------------------
        P_full     = self.n_flux + 2 * self.n_alpha        # N + 2K
        x_full     = np.zeros(P_full, dtype=float)
        e_full     = np.zeros(P_full, dtype=float)

        x_full[self._compact2full] = x_w / d        # un-whiten + scatter
        e_full[self._compact2full] = self._flux_errors(A_w) / d     # un-whiten errors ? 

        # positivity on fluxes only
        if cfg.positivity:
            x_full[:self.n_flux] = np.maximum(0, x_full[:self.n_flux])

        self.solution = x_full[:self.n_flux]             # fluxes, corresponds to original templates
        self.solution_err = e_full[:self.n_flux]  # flux errors, corresponds to original templates

        # apply sub-pixel shifts so residual() uses updated templates
        self.alpha    = x_full[self.n_flux : self.n_flux + self.n_alpha]  # α_k  (size K)
        self.beta     = x_full[self.n_flux + self.n_alpha : ]             # β_k  (size K)
        
        # only apply shifts if no nans in the solution
        if np.isfinite(self.alpha).any() or np.isfinite(self.beta).any():
            self._apply_shifts()             # <── one call, no duplication
        else:            
            print("WARNING: NaN in astrometric solution, not applying shifts.")
            self.config.fit_astrometry_niter = 0  # disable further iterations

        # update the templates with the fitted fluxes, errors     
        for tmpl, flux, err in zip(self._orig_templates, self.solution, self.solution_err):
            if np.isfinite(flux): tmpl.flux = flux 
            if np.isfinite(err): tmpl.err = err

        return self.solution, self.solution_err, info

    # ------------------------------------------------------------
    # helper – evaluate polynomial at any point
    # ------------------------------------------------------------
    def shift_at(self, x_pix: float, y_pix: float) -> tuple[float, float]:
        """
        Return (dx, dy) in pixels at detector coordinate (x_pix, y_pix).
        If astrometric fitting was not enabled it returns (0, 0).
        """
        if self.alpha is None:                 # nothing fitted
            return 0.0, 0.0

        h, w = self.image.shape
        phi = astrometry.cheb_basis(x_pix / (w - 1),
                                    y_pix / (h - 1),
                                    self.basis_order)
        return float(self.alpha @ phi), float(self.beta @ phi)


    # ------------------------------------------------------------
    # private – apply the stored shifts to all *original flux* templates
    # ------------------------------------------------------------
    def _apply_shifts(self) -> None:
        """
        Warp flux templates in-place according to the current α, β arrays.
        No-op when astrometry is off.
        """
        if self.alpha is None:
            return

        # note: shift all templates in original list
        # are also referenced by the solution templates list 
        # so this can be fed into the next iteration
        for i, tmpl in enumerate(self._orig_templates):
            x0, y0 = tmpl.input_position_original
            dx, dy = self.shift_at(x0, y0)
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                continue

            # the shifts are those needed to bring the measurement image to the template
            # but we apply them to the template, so subtract them
            tmpl.data[:] = nd_shift(
                tmpl.data, (-dy, -dx), 
                order=3, mode="constant", cval=0.0, 
                prefilter=True
            )
    
            tmpl.input_position_original = (x0 - dx, y0 - dy)
            
            # accumulate the shift and scale in the template, in case of iterative fitting
            tmpl.shift += [-dx, -dy]
