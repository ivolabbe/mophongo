from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

logger = logging.getLogger(__name__)


@dataclass
class SceneFitter:
    """Stateless solver for scene normal equations.

    The fitter whitens the flux block of the normal matrix, solves the
    system using conjugate gradients and returns unwhitened fluxes and
    their 1σ uncertainties. Optionally, an additional shift block can be
    supplied which is solved jointly with the fluxes.
    """

    @staticmethod
    def solve(
        A: sp.spmatrix,
        b: np.ndarray,
        *,
        reg: float = 0.0,
        AB: sp.spmatrix | None = None,
        BB: sp.spmatrix | None = None,
        bB: np.ndarray | None = None,
        cg_kwargs: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Solve ``A x = b`` with optional shift block.

        Parameters
        ----------
        A
            Flux normal matrix (unwhitened).
        b
            Right hand side.
        reg
            Diagonal regularisation strength.
        AB, BB, bB
            Optional blocks coupling the fluxes to shift parameters.
        cg_kwargs
            Extra keyword arguments passed to :func:`scipy.sparse.linalg.cg`.

        Returns
        -------
        alpha, err, beta, info
            Unwhitened fluxes, their 1σ errors, optional shift coefficients
            and the CG exit flag.
        """

        cg_kwargs = cg_kwargs or {}

        Areg = A + reg * sp.eye(A.shape[0], format="csr")
        d = np.sqrt(Areg.diagonal())
        Dinv = sp.diags(1.0 / d, 0, format="csr")
        A_w = Dinv @ Areg @ Dinv
        b_w = Dinv @ b

        if AB is not None:
            if BB is None or bB is None:
                raise ValueError("AB provided but BB or bB missing")
            AB_w = Dinv @ AB
            top = sp.hstack([A_w, AB_w], format="csr")
            bottom = sp.hstack([AB_w.T, BB], format="csr")
            sys = sp.vstack([top, bottom], format="csr")
            rhs = np.concatenate([b_w, bB])
        else:
            sys = A_w
            rhs = b_w

        xw, info = cg(sys, rhs, **cg_kwargs)
        if info != 0:
            logger.warning("CG did not converge: info=%d", info)

        if AB is not None:
            alpha_w = xw[: A.shape[0]]
            beta = xw[A.shape[0] :]
        else:
            alpha_w = xw
            beta = None

        alpha = alpha_w / d

        try:
            cov = np.linalg.inv(sys.toarray())
            err_w = np.sqrt(np.diag(cov)[: A.shape[0]])
        except Exception:
            logger.exception("Failed to compute covariance")
            err_w = np.full(A.shape[0], np.nan)
        err = err_w / d

        return alpha, err, beta, info
