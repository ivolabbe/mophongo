from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from types import SimpleNamespace

logger = logging.getLogger(__name__)


import numpy as np
import scipy.sparse as sp

# from scipy.sparse.csgraph import connected_components
from .fit import FitConfig as FitConfig


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

        if AB is not None and BB is not None and bB is not None:
            flux, err, shifts, info = SceneFitter._solve_flux_and_shifts(A, b, AB, BB, bB, config)
        else:
            flux, err, info = SceneFitter.solve_flux(A, b, config)
            shifts = None

        return SimpleNamespace(flux=flux, err=err, shifts=shifts, info=info)

    @staticmethod
    def solve_flux(
        self, A: sp.spmatrix, b: np.ndarray, config: Optional[FitConfig] = None
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
        err = SceneFitter._flux_errors(A_w) / d
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

        # --- regularize BEFORE building the scalings
        lam_x = max(cfg.reg, 0.0)
        diagB = np.asarray(BB.diagonal()).ravel()
        finite_pos = np.isfinite(diagB) & (diagB > 0)
        scaleB = float(np.median(diagB[finite_pos])) if finite_pos.any() else 1.0
        lam_b = float(getattr(cfg, "reg_astrom", 1e-4)) * scaleB

        Areg = A + sp.eye(A.shape[0], format="csr") * lam_x
        BBreg = BB + sp.eye(BB.shape[0], format="csr") * lam_b

        # --- flux whitening
        d = np.sqrt(np.maximum(Areg.diagonal(), 1e-12))
        Dinv = sp.diags(1.0 / d, 0, format="csr")

        # --- shift whitening
        L = np.linalg.cholesky(BBreg.toarray())
        Linv = np.linalg.inv(L)

        A_w = Dinv @ Areg @ Dinv
        # whiten flux-shift coupling with inverse Cholesky factor
        AB_w = (Dinv @ AB) @ Linv.T
        #        AB_w = (Dinv @ AB) @ Linv.T
        BB_wI = sp.eye(BB.shape[0], format="csr")
        b_w = Dinv @ b
        bB_w = Linv @ bB

        # --- joint solve in whitened variables
        K = sp.bmat([[A_w, AB_w], [AB_w.T, BB_wI]], format="csr")
        rhs = np.concatenate([b_w, bB_w])
        sol, info = cg(
            K,
            rhs,
            atol=0.0,
            rtol=cfg.cg_kwargs.get("rtol", 1e-6),
            maxiter=cfg.cg_kwargs.get("maxiter", 2000),
        )

        na = A.shape[0]
        xw = sol[:na]
        betaw = sol[na:]

        # unwhiten
        x = xw / d
        beta = np.linalg.solve(L.T, betaw)

        # errors: diagonal of Schur(A_w - AB_w AB_wᵀ) without mixing sparse/dense
        if sp.isspmatrix(AB_w):
            S_w_diag = (A_w - (AB_w @ AB_w.T)).diagonal()
        else:
            # AB_w is dense (nA x nB). diag(AB_w AB_wᵀ) = row-wise sum of squares
            S_w_diag = A_w.diagonal() - np.einsum("ij,ij->i", AB_w, AB_w)
        err = 1.0 / np.sqrt(np.maximum(S_w_diag, 1e-12)) / d

        return x, err, beta, {"cg_info": int(info)}

    @staticmethod
    def _flux_errors(A: csr_matrix) -> np.ndarray:
        """Approximate 1-σ uncertainties from whitened normal matrix."""
        return 1.0 / np.sqrt(np.maximum(A.diagonal(), 1e-12))
