"""Utilities for building smooth astrometric shift models."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebval


def cheb_basis(x: float, y: float, order: int) -> np.ndarray:
    """Return Chebyshev basis values T_i(x)T_j(y) up to ``order``."""
    u = 2 * x - 1.0
    v = 2 * y - 1.0
    tx = [chebval(u, [0] * i + [1]) for i in range(order + 1)]
    ty = [chebval(v, [0] * j + [1]) for j in range(order + 1)]
    phi = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            phi.append(tx[i] * ty[j])
    return np.array(phi, dtype=float)


def n_terms(order: int) -> int:
    """Number of 2-D Chebyshev terms for ``order``."""
    return (order + 1) * (order + 2) // 2


def make_gradients(templates, shape):
    """Return per-template gradient images on the full frame."""
    gx, gy = [], []
    for tmpl in templates:
        dy, dx = np.gradient(tmpl.data.astype(float))
        gxi = np.zeros(shape, dtype=float)
        gyi = np.zeros(shape, dtype=float)
        gxi[tmpl.slices_original] = dx[tmpl.slices_cutout]
        gyi[tmpl.slices_original] = dy[tmpl.slices_cutout]
        gx.append(gxi)
        gy.append(gyi)
    return gx, gy


def basis_matrix(templates, segmap, order):
    """Evaluate basis functions at template centres."""
    h, w = segmap.shape
    mat = np.zeros((len(templates), n_terms(order)), dtype=float)
    for i, tmpl in enumerate(templates):
        x, y = tmpl.position_original
        mat[i] = cheb_basis(x / (w - 1), y / (h - 1), order)
    return mat


def collapse_gradients(gx, gy, phi, k, shape):
    """Collapse per-object gradients into global templates."""
    GX = [np.zeros(shape, dtype=float) for _ in range(k)]
    GY = [np.zeros(shape, dtype=float) for _ in range(k)]
    for i in range(len(gx)):
        for j in range(k):
            GX[j] += phi[i, j] * gx[i]
            GY[j] += phi[i, j] * gy[i]
    return GX, GY
