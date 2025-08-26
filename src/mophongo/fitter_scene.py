from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
import logging
import numpy as np
from scipy.sparse import csr_matrix, diags, eye, bmat
from scipy.sparse.linalg import cg

from .fit import FitConfig

logger = logging.getLogger(__name__)


@dataclass
class SceneFitter:
    """Stateless solver for per-scene normal equations."""

    def solve(
        self,
        A: csr_matrix,
        b: np.ndarray,
        AB: csr_matrix | None = None,
        BB: csr_matrix | None = None,
        bB: np.ndarray | None = None,
        config: Optional[FitConfig] = None,
    ) -> SimpleNamespace:
        """Solve a scene's normal system.

        Parameters
        ----------
        A, b
            Flux–flux normal matrix and right-hand side.
        AB, BB, bB
            Optional shift cross-terms.  When provided the extended system is
            solved for both fluxes and shift coefficients.
        config
            Optional :class:`FitConfig` controlling solver behaviour.
        """
        if AB is not None and BB is not None and bB is not None:
            flux, err, shifts, info = self._solve_scenes_with_shifts(
                A, b, AB, BB, bB, config
            )
        else:
            flux, err, info = self.solve_scene(A, b, config)
            shifts = None
        return SimpleNamespace(flux=flux, err_pred=err, shifts=shifts, info=info)

    # ------------------------------------------------------------------
    def solve_scene(
        self, A: csr_matrix, b: np.ndarray, config: Optional[FitConfig] = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve ``A x = b`` for flux parameters using conjugate gradient."""
        cfg = config or FitConfig()
        A = A.tocsr()
        reg = cfg.reg
        if reg > 0:
            A = A + eye(A.shape[0], format="csr") * reg

        d = np.sqrt(np.maximum(A.diagonal(), 1e-12))
        Dinv = diags(1.0 / d, 0, format="csr")
        A_w = Dinv @ A @ Dinv
        b_w = Dinv @ b

        x_w, info = cg(A_w, b_w, **cfg.cg_kwargs)
        x = x_w / d
        err = self._flux_errors(A_w) / d
        return x, err, {"cg_info": info}

    def _solve_scenes_with_shifts(
        self,
        A: csr_matrix,
        b: np.ndarray,
        AB: csr_matrix,
        BB: csr_matrix,
        bB: np.ndarray,
        config: Optional[FitConfig] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Solve augmented system including shift parameters."""
        A_full = bmat([[A, AB], [AB.T, BB]], format="csr")
        b_full = np.concatenate([b, bB])
        x_full, err_full, info = self.solve_scene(A_full, b_full, config)
        n_flux = A.shape[0]
        flux = x_full[:n_flux]
        err = err_full[:n_flux]
        shifts = x_full[n_flux:]
        return flux, err, shifts, info

    @staticmethod
    def _flux_errors(A: csr_matrix) -> np.ndarray:
        """Approximate 1-σ uncertainties from whitened normal matrix."""
        return 1.0 / np.sqrt(np.maximum(A.diagonal(), 1e-12))
