from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
import logging
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

logger = logging.getLogger(__name__)


@dataclass
class SceneFitter:
    """Stateless solver for per-scene linear systems."""

    def solve(self,
              A: LinearOperator,
              b: np.ndarray,
              config: Optional[object] = None) -> SimpleNamespace:
        """Solve ``A x ≈ b`` using LSQR.

        Parameters
        ----------
        A : LinearOperator
            Whitened design matrix describing the scene.
        b : ndarray
            Whitened image vector.
        config : optional
            Placeholder for future configuration options.
        """
        result = lsqr(A, b)
        x = result[0]
        err = None
        if hasattr(A, "matrix"):
            AtA = A.matrix.T @ A.matrix
            try:
                cov = np.linalg.pinv(AtA)
                err = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:  # pragma: no cover - fallback
                err = np.full(len(x), np.nan)
        return SimpleNamespace(flux=x, err_pred=err, shifts=None, info={})
