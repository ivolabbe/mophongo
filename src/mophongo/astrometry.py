# ---------------------------------------------------------------------
# astrometry.py  –  utilities for building smooth shift models
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebval


def cheb_basis(x: float, y: float, order: int) -> np.ndarray:
    """Return Chebyshev basis values T_i(x)T_j(y) up to ``order``.

    Parameters
    ----------
    x, y : float
        Normalised coordinates in the unit square ``[0, 1]``.
    order : int
        Maximum polynomial order.

    Returns
    -------
    ndarray
        1-D array of length ``n_terms(order)`` containing the basis values.
    """
    u = 2.0 * x - 1.0
    v = 2.0 * y - 1.0
    terms: list[float] = []
    for i in range(order + 1):
        tx = chebval(u, [0] * i + [1])
        for j in range(order + 1 - i):
            ty = chebval(v, [0] * j + [1])
            terms.append(tx * ty)
    return np.array(terms, dtype=float)


def n_terms(order: int) -> int:
    """Return number of terms in a triangular 2-D Chebyshev basis."""
    return (order + 1) * (order + 2) // 2

