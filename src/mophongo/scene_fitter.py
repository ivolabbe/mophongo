from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu
from types import SimpleNamespace

logger = logging.getLogger(__name__)


import numpy as np
import scipy.sparse as sp

# from scipy.sparse.csgraph import connected_components
from .fit import FitConfig as FitConfig


def _positive_diagonal_scale(matrix: sp.spmatrix) -> float:
    """Return a finite positive diagonal scale for ridge regularization."""
    diag = np.asarray(matrix.diagonal(), dtype=float)
    positive = diag[np.isfinite(diag) & (diag > 0)]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _finite_nonnegative(value: float, default: float = 0.0) -> float:
    """Return a finite non-negative scalar, falling back for invalid input."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val) or val < 0:
        return default
    return val


def _slice_intersection(
    a: tuple[slice, slice], b: tuple[slice, slice]
) -> tuple[slice, slice] | None:
    y0 = max(a[0].start, b[0].start)
    y1 = min(a[0].stop, b[0].stop)
    x0 = max(a[1].start, b[1].start)
    x1 = min(a[1].stop, b[1].stop)
    if y0 >= y1 or x0 >= x1:
        return None
    return slice(y0, y1), slice(x0, x1)


def build_normal(
    templates: List[Template],
    image: np.ndarray,
    weights: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, "STRtree"]:
    """Stateless clone of SparseFitter.build_normal_tree: returns (ATA, ATb, rtree)."""
    from shapely.geometry import box
    from shapely.strtree import STRtree

    n = len(templates)
    ata = sp.lil_matrix((n, n))
    atb = np.zeros(n)

    # diagonals + RHS + bboxes
    boxes = []
    norms = np.empty(n, dtype=float)
    for i, tmpl in enumerate(templates):
        sl_i = tmpl.slices_original
        cut_i = tmpl.data[tmpl.slices_cutout]
        w_i = weights[sl_i]
        img_i = image[sl_i]

        # diag and rhs
        wi = float(np.sum(cut_i * w_i * cut_i))
        bi = float(np.sum(cut_i * w_i * img_i))
        norms[i] = wi
        ata[i, i] = wi
        atb[i] = bi

        # bbox geometry
        y0, y1, x0, x1 = tmpl.bbox
        boxes.append(box(x0, y0, x1, y1))

    # off-diagonals via STRtree overlap queries
    tree = STRtree(boxes)
    for i, geom in enumerate(boxes):
        sl_i = templates[i].slices_original
        for j in tree.query(geom):
            j = int(j)
            if j <= i:
                continue
            inter = _slice_intersection(sl_i, templates[j].slices_original)
            if inter is None:
                continue

            # map intersection to local cutouts
            ti = templates[i]
            tj = templates[j]
            sl_i_local = (
                slice(
                    inter[0].start - sl_i[0].start + ti.slices_cutout[0].start,
                    inter[0].stop - sl_i[0].start + ti.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start - sl_i[1].start + ti.slices_cutout[1].start,
                    inter[1].stop - sl_i[1].start + ti.slices_cutout[1].start,
                ),
            )
            sl_j = templates[j].slices_original
            sl_j_local = (
                slice(
                    inter[0].start - sl_j[0].start + tj.slices_cutout[0].start,
                    inter[0].stop - sl_j[0].start + tj.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start - sl_j[1].start + tj.slices_cutout[1].start,
                    inter[1].stop - sl_j[1].start + tj.slices_cutout[1].start,
                ),
            )

            w = weights[inter]
            arr_i = ti.data[sl_i_local]
            arr_j = tj.data[sl_j_local]
            val = float(np.sum(arr_i * arr_j * w))
            ata[i, j] = val
            ata[j, i] = val

    return ata.tocsr(), atb, tree


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
        AB: sp.spmatrix | None = None,
        BB: sp.spmatrix | None = None,
        bB: np.ndarray | None = None,
        config: Optional[FitConfig] = None,
        cg_kwargs: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Solve ``A x = b`` with optional shift block.

        Parameters
        ----------
        A
            Flux normal matrix (unwhitened).
        b
            Right hand side.
        config
            Solver configuration. ``reg_flux`` regularizes the flux block,
            and ``reg_astrom`` regularizes only the shift block.
        AB, BB, bB
            Optional blocks coupling the fluxes to shift parameters.

        Returns
        -------
        alpha, err, beta, info
            Unwhitened fluxes, their 1σ errors, optional shift coefficients
            and the solver exit flag (always 0 for the direct solver).
        """
        # Flux regularization must use only the photometric ridge.
        # Astrometric regularization is applied only to the shift block below.
        scale_A = _positive_diagonal_scale(A)
        reg_flux = _finite_nonnegative(getattr(config, "reg_flux", 0.0))
        lam_A = reg_flux if reg_flux > 0 else 1e-6 * scale_A
        Areg = A + sp.eye(A.shape[0], format="csr") * lam_A

        if AB is not None and BB is not None and bB is not None:
            scale_BB = _positive_diagonal_scale(BB)
            reg_astrom = _finite_nonnegative(getattr(config, "reg_astrom", 1e-4))
            lam_b = reg_astrom * scale_BB
            BBreg = BB + sp.eye(BB.shape[0], format="csr") * lam_b

            flux, err, shifts, info = SceneFitter._solve_flux_and_shifts(
                Areg, b, AB, BBreg, bB, config
            )
        else:
            flux, err, info = SceneFitter.solve_flux(Areg, b, config)
            shifts = None

        return SimpleNamespace(flux=flux, err=err, shifts=shifts, info=info)

    @staticmethod
    def solve_flux(
        A: sp.spmatrix, b: np.ndarray, config: Optional[FitConfig] = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve ``A x = b`` for flux parameters using conjugate gradient."""
        cfg = config or FitConfig()
        A = A.tocsr()

        d = np.sqrt(np.maximum(A.diagonal(), 1e-12))
        Dinv = sp.diags(1.0 / d, 0, format="csr")
        A_w = Dinv @ A @ Dinv
        b_w = Dinv @ b

        x_w = spsolve(A_w, b_w)
        info = 0
        x = x_w / d
        err = SceneFitter._flux_errors(A_w) / d

        if cfg.positivity:
            x = np.maximum(0.0, x)

        return x, err, {"cg_info": info}

    @staticmethod
    def _solve_flux_and_shifts(
        A: sp.spmatrix,
        b: np.ndarray,
        AB: sp.spmatrix,
        BB: sp.spmatrix,
        bB: np.ndarray | None = None,
        config: Optional[FitConfig] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        cfg = config or FitConfig()
        A = A.tocsr()
        AB = AB.tocsr()
        BB = BB.tocsr()

        # --- flux whitening
        d = np.sqrt(np.maximum(A.diagonal(), 1e-12))
        Dinv = sp.diags(1.0 / d, 0, format="csr")

        # --- shift whitening
        L = np.linalg.cholesky(BB.toarray())
        Linv = np.linalg.inv(L)

        A_w = Dinv @ A @ Dinv
        # whiten flux-shift coupling with inverse Cholesky factor
        AB_w = (Dinv @ AB) @ Linv.T
        #        AB_w = (Dinv @ AB) @ Linv.T
        BB_wI = sp.eye(BB.shape[0], format="csr")
        b_w = Dinv @ b
        bB_w = Linv @ bB

        # --- joint solve in whitened variables
        K = sp.bmat([[A_w, AB_w], [AB_w.T, BB_wI]], format="csr")
        rhs = np.concatenate([b_w, bB_w])
        sol = spsolve(K, rhs)
        info = 0

        na = A.shape[0]
        xw = sol[:na]
        betaw = sol[na:]

        # unwhiten
        x = xw / d
        beta = np.linalg.solve(L.T, betaw)

        # Marginalize over the shift block: Schur complement S_w = A_w − AB_w AB_wᵀ
        # (BB_wI = I after whitening). Errors are sqrt(diag(S_w⁻¹)) / d.
        if sp.issparse(AB_w):
            S_w = (A_w - AB_w @ AB_w.T).toarray()
        else:
            S_w = A_w.toarray() - AB_w @ AB_w.T
        err = SceneFitter._flux_errors(S_w) / d

        if cfg.positivity:
            x = np.maximum(0.0, x)

        return x, err, beta, {"cg_info": int(info)}

    @staticmethod
    def _flux_errors(A, dense_threshold: int = 500) -> np.ndarray:
        """Return ``sqrt(diag(A^{-1}))`` — 1-σ errors with off-diagonal coupling.

        ``A`` is expected SPD (a whitened normal matrix or its Schur
        complement). For scene-scale problems (``n ≤ dense_threshold``)
        a single dense inversion is fastest; above that we factor once
        with sparse LU and back-solve unit columns one at a time.
        """
        n = A.shape[0]
        if n <= dense_threshold:
            M = A.toarray() if sp.issparse(A) else np.asarray(A)
            diag = np.diag(np.linalg.inv(M))
        else:
            Acsc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
            factor = splu(Acsc)
            diag = np.empty(n)
            e = np.zeros(n)
            for i in range(n):
                e[i] = 1.0
                diag[i] = factor.solve(e)[i]
                e[i] = 0.0
        return np.sqrt(np.maximum(diag, 1e-12))
