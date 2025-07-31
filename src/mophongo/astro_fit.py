"""Wrapper fitter adding global astrometry parameters."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.sparse import eye, diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import cg

from .fit import SparseFitter, FitConfig
from .templates import Template
from . import astrometry
from copy import deepcopy


# 0. quick per-source S/N directly from the measurement image
def estimate_snr(tmpl, weights):
    sl = tmpl.slices_original
    flux = np.sum(tmpl.data[tmpl.slices_cutout] * weights[sl])          # matched-filter sum
    noise = np.sqrt(np.sum(weights[sl]))                                # inverse-variance weights
    return flux / noise

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
        super().__init__(list(templates), image, weights)
        self.config = config 
        
        self.n_flux = len(templates)
        print('GlobalAstroFitter: templates in', len(templates))
        self.templates = [tmpl for tmpl in self.templates if tmpl.id is not None]
        self.n_flux = len(self.templates)  # update n_flux to match the templates
        print('GlobalAstroFitter: templates which are flux tempaltes: ', len(templates))

        for i, tmpl in enumerate(self.templates):
            tmpl.is_flux = True
            tmpl.pidx = i

        if not config.fit_astrometry:
            return      # nothing more to do

        # ---------- astrometry part ----------
        order = config.astrom_basis_order
        K     = astrometry.n_terms(order)
        self.basis_order = order
        self.n_alpha = K        # α_k  (β_k shares the same K)

        # get estimate for the flux and errors to scale the gradients and keep only high S/N sources for astrometry
        if self.templates[0].flux != 0:
            flux = [t.flux for t in self.templates]
        else:
            flux = self.quick_flux()[0:self.n_flux]
        rms = self.predicted_errors()[0:self.n_flux]  
        
        # 1. per-object S/N estimate        
        if self.config.snr_thresh_astrom > 0:
            good = (flux / rms) >= self.config.snr_thresh_astrom
        else:
            good = np.ones(self.n_flux, dtype=bool)

        if not np.any(good):
            good[:] = True
        
        print(f"GlobalAstroFitter: {self.n_flux} templates and {np.sum(good)} with S/N >= {self.config.snr_thresh_astrom} used for astrometry")

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
        for k in range(K):
            for i in np.where(good)[0]:
                f_est = flux[i]                     # per-object scale

                gx_tile = deepcopy(gx_i[i])
                gx_tile.data *= f_est * Φ[i, k]           # *** SCALE ***
                gx_tile.is_flux = False
                gx_tile.pidx = self.n_flux + k
                self.templates.append(gx_tile)
                self._big2small_col.append(self.n_flux + k)

                gy_tile = deepcopy(gy_i[i])
                gy_tile.data *= f_est * Φ[i, k]           # *** SCALE ***
                gy_tile.is_flux = False
                gy_tile.pidx = self.n_flux + K + k
                self.templates.append(gy_tile)
                self._big2small_col.append(self.n_flux + K + k)


    def _collapse(self):
        A_big = self._ata.tocsr()
        b_big = self._atb

        rows = np.arange(A_big.shape[0])
        cols = np.fromiter((t.pidx for t in self.templates), dtype=int)

        P = cols.max() + 1 if cols.size else 0

        S = csr_matrix(
            (np.ones_like(rows), (rows, cols)), shape=(A_big.shape[0], P)
        )

        self._ata_comp = (S.T @ A_big @ S).tocsr()
        self._atb_comp = (S.T @ b_big).ravel()

        self.n_flux = int(sum(t.is_flux for t in self.templates))
        
    # ------------------------------------------------------------
    # 3.  solve   # keep track of valid fluxes through ID, not n_flux
    # ------------------------------------------------------------
    def solve(self, config: FitConfig | None = None):
        cfg   = config or self.config

        # build big normal matrix once, this as a shift entry for every template
        _ = self.ata, self.atb          # triggers build_normal_matrix()

        # compact it the first time we call solve(), summing over all objects
        # and leaving only the chebyshev terms
        if not hasattr(self, "_ata_comp"):
            self._collapse()

        A = self._ata_comp.copy()
        b = self._atb_comp.copy()

        # apply regularization by introducting a small diagonal term
        if cfg.reg:
            A = A + cfg.reg * eye(A.shape[0])

        if cfg.reg_astrom:
            # build one dense 1-D vector, then convert to a diagonal sparse matrix
            d = np.zeros(A.shape[0], dtype=float)
            d[self.n_flux:] = cfg.reg_astrom     # only α_k , β_k rows/cols
            A = A + diags(d, 0, format="csr")

        # should we whiten SparseFitter too?
        # ------------------------------------------------------------------
        # symmetric column-and-row whitening  (A → D⁻¹ A D⁻¹,  b → D⁻¹ b)
        # ------------------------------------------------------------------
        d = np.sqrt(A.diagonal())               # √(Ajj)  (never negative)
        d[d == 0.0] = 1.0                       # protect empty columns
        Dinv = diags(1.0 / d, 0, format="csr")  # D⁻¹

        A_w = Dinv @ A @ Dinv                   # still sparse, still SPD
        b_w = Dinv @ b
        
        # preconditioner
        if "M" not in cfg.cg_kwargs:
            ilu = spilu(A_w, drop_tol=1e-5, fill_factor=10)
            M = LinearOperator(A_w.shape, ilu.solve)
            print('preconditioner:', M.shape)

        y, info = cg(A_w, b_w, **cfg.cg_kwargs) # solve whitened system
        x = Dinv @ y                            # un-whiten

        # fit is done, junk the astrometry templates 
#        self.n_flux = np.sum([1 for tmpl in self.templates if tmpl.id is not None])
        # update n_flux to match the templates

        # positivity on fluxes only
        if cfg.positivity:
            x[:self.n_flux] = np.maximum(0, x[:self.n_flux])

        # keep the flux vector for caller-side inspection
        self.x = x 
        self.solution = x[:self.n_flux]

        # apply sub-pixel shifts so residual() uses updated templates
        if cfg.fit_astrometry:
            K          = self.n_alpha
            self.alpha = x[self.n_flux       : self.n_flux + K]
            self.beta  = x[self.n_flux + K   : ]
            self._apply_shifts()             # <── one call, no duplication
        else:
            self.alpha = self.beta = None


        return self.solution, info


    # ------------------------------------------------------------------ #
    # override to return only the flux-component uncertainties
    def predicted_errors(self):
        errs = super().predicted_errors()
        return errs[: self.n_flux]

    def flux_errors(self):
        errs = super().flux_errors()
        return errs[: self.n_flux]

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
    # private – apply the stored shifts to all *flux* templates
    # ------------------------------------------------------------
    def _apply_shifts(self) -> None:
        """
        Warp flux templates in-place according to the current α, β arrays.
        No-op when astrometry is off.
        """
        if self.alpha is None:
            return

        for i, tmpl in enumerate(self.templates[: self.n_flux]):
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
    
            tmpl.shifted_position_original = (x0 - dx, y0 - dy)
            
            # accumulate the shift and scale in the template, in case of iterative fitting
            tmpl.shift += [-dx, -dy]
