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
        config: FitConfig,
    ):
        # ---------- flux part ----------
        super().__init__(list(templates), image, weights, config)
        self.n_flux = len(templates)

        if not config.fit_astrometry:
            return      # nothing more to do

        # ---------- astrometry part ----------
        order = config.astrom_basis_order
        K     = astrometry.n_terms(order)
        self.basis_order = order
        self.n_alpha = K        # α_k  (β_k shares the same K)

        if config.snr_thresh_astrom > 0:
            snr = np.array([estimate_snr(t, self.weights) for t in templates])
            good = snr >= config.snr_thresh_astrom
        else:
            good = np.ones(len(templates), dtype=bool)  
        
        print(f"GlobalAstroFitter: {np.sum(good)} templates with S/N >= {config.snr_thresh_astrom} used for astrometry")

        # 1. per-object gradients
        gx_i, gy_i = astrometry.make_gradients(templates)

        # 2. Chebyshev basis evaluated at object centres
        Φ = astrometry.basis_matrix(templates, image.shape, order)    #  (N, K)

        # 3. build tiny templates for each term & each object
        self._big2small_col = []     # length = len(self.templates) after extension
        self._param_size    = self.n_flux + 2 * K   # compact column count

        # We want to keep the matrix sparse, so we need to keep track of 
        # locations of the stamps in the big matrix, and then we can collapse them later
        # collapsing meaning that we trace their entrix in the sparse matrix and sum over 
        # all objects and fit for a global polynomial shift at each Chebyshev term
        for k in range(K):
            # gather GX_k and GY_k tiles (one per object)
            for i in np.where(good)[0]:
                tile = deepcopy(gx_i[i])
                tile.data *= Φ[i, k]
                self.templates.append(tile)
                # register this column belongs to α_k
                self._big2small_col.append(self.n_flux + k)

            for i in np.where(good)[0]:
                tile = deepcopy(gy_i[i])
                tile.data *= Φ[i, k]
                self.templates.append(tile)
                # register belongs to β_k
                self._big2small_col.append(self.n_flux + K + k)

        # map of first-seen compact column for every parameter index
        self._keep_col = {}
        for big, small in enumerate(self._big2small_col, start=self.n_flux):
            self._keep_col.setdefault(small, big)


    def _collapse(self):
        A_big = self._ata.tocsr()
        b_big = self._atb
        P     = self._param_size
        M     = A_big.shape[0]

        # selector S:  shape (M , P) , one 1 per row
        data  = np.ones(M, dtype=np.float64)
        rows  = np.arange(M)
        cols  = np.concatenate((
                    np.arange(self.n_flux),          # flux → same index
                    np.asarray(self._big2small_col)  # gradient tiles
                ))
        S = csr_matrix((data, (rows, cols)), shape=(M, P))

        A_cmp = S.T @ A_big @ S           # still sparse
        b_cmp = S.T @ b_big

        self._ata_comp = A_cmp.tocsr()
        self._atb_comp = np.asarray(b_cmp).ravel()
        
    # ------------------------------------------------------------
    # 3.  solve
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

        x, info = cg(A, b, **cfg.cg_kwargs)

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

        for tmpl in self.templates[: self.n_flux]:
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
            tmpl.shift += [-dx, -dy]



#------  OBSOLETE 
# 
# 
#   # ------------------------------------------------------------
    # 2.  collapse big → compact matrix / rhs
    # ------------------------------------------------------------
    def _collapse_slow(self):
        """Collapse identical columns/rows (α_k, β_k tiles) into one."""
        A_big = self._ata.tocsr()
        b_big = self._atb

        P = self._param_size                   # N + 2K
        A_cmp = lil_matrix((P, P))
        b_cmp = np.zeros(P, dtype=float)

        # flux columns are unique already
        A_cmp[:self.n_flux, :self.n_flux] = A_big[:self.n_flux, :self.n_flux]
        b_cmp[:self.n_flux]              = b_big[:self.n_flux]

        # accumulate tiles
        for big, small in enumerate(self._big2small_col, start=self.n_flux):
            # rows
            b_cmp[small] += b_big[big]
            # columns (symmetric)

            # diagonal ------------------------------------------------------
            A_cmp[small, small]        += A_big[big, big]      # ← NEW
            # off-diagonals with the flux block
            A_cmp[small, :self.n_flux]               += A_big[big, :self.n_flux]
            A_cmp[:self.n_flux, small]               += A_big[:self.n_flux, big]
            for big2, small2 in enumerate(self._big2small_col, start=self.n_flux):
                if small2 < small:      # lower triangle as well
                    A_cmp[small, small2] += A_big[big, big2]
                    A_cmp[small2, small] += A_big[big2, big]

        self._ata_comp = A_cmp.tocsr()
        self._atb_comp = b_cmp
