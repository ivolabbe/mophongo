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
    for i, tmpl in enumerate(templates):
        sl_i = tmpl.slices_original
        cut_i = tmpl.data[tmpl.slices_cutout]
        w_i = weights[sl_i]
        img_i = image[sl_i]

        # diag and rhs, accumulated in float64. The operands are float32
        # stamps and weights, and np.sum without dtype= accumulates in the
        # input width -- so these entries carried ~7 significant digits into
        # a float64 matrix, and the whitening, Cholesky and spsolve chain
        # downstream inherited that. The sum is ~1e4 terms over one stamp
        # footprint, so the wider accumulator is free.
        wi = float(np.sum(cut_i * w_i * cut_i, dtype=np.float64))
        bi = float(np.sum(cut_i * w_i * img_i, dtype=np.float64))
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
            val = float(np.sum(arr_i * arr_j * w, dtype=np.float64))
            ata[i, j] = val
            ata[j, i] = val

    return ata.tocsr(), atb, tree


@dataclass
class SceneFitter:
    """Stateless solver for scene normal equations.

    The fitter whitens the flux block of the normal matrix, solves the
    system by direct sparse factorization (``scipy.sparse.linalg.spsolve``)
    and returns unwhitened fluxes and their 1σ uncertainties. Optionally,
    an additional shift block can be supplied which is solved jointly with
    the fluxes. All inputs arrive as arguments and results are returned,
    so the same instance serves every scene.
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Solve ``A x = b`` with optional shift block.

        The flux-block ridge follows ``config.reg_flux``: ``None`` (default)
        applies the adaptive ridge, ``1e-6`` times the median positive
        diagonal of ``A``; ``0.0`` applies none at all; a positive value is
        used as given. When shift blocks are supplied and non-empty, the shift block
        is regularized by ``config.astrom_reg`` times the median positive
        diagonal of ``BB`` and solved jointly; empty shift blocks (a scene
        with no bright member) fall back to flux-only.

        Parameters
        ----------
        A
            Flux normal matrix (unwhitened).
        b
            Right hand side.
        config
            Solver configuration. ``reg_flux`` regularizes the flux block,
            and ``astrom_reg`` regularizes only the shift block.
        AB, BB, bB
            Optional blocks coupling the fluxes to shift parameters.
        Returns
        -------
        SimpleNamespace
            Fields ``flux`` (unwhitened fluxes), ``err`` (1σ errors),
            ``shifts`` (shift coefficients, ``None`` on the flux-only path)
            and ``info`` (solver provenance; the solve is a direct sparse
            factorization, so there is no iteration count to report).
        """
        # Flux regularization must use only the photometric ridge; astrom_reg
        # is reserved for the shift block below. Three-state semantics:
        # None -> adaptive ridge 1e-6 * median positive diagonal (the
        # conditioning default), 0.0 -> genuinely no ridge, >0 -> that value.
        scale_A = _positive_diagonal_scale(A)
        reg_flux = getattr(config, "reg_flux", None)
        if reg_flux is None:
            lam_A = 1e-6 * scale_A
        else:
            lam_A = _finite_nonnegative(reg_flux)
        Areg = A + sp.eye(A.shape[0], format="csr") * lam_A

        # empty shift blocks (no bright member at all) fall back to flux-only
        if AB is not None and BB is not None and bB is not None and AB.shape[1] > 0:
            scale_BB = _positive_diagonal_scale(BB)
            astrom_reg = _finite_nonnegative(getattr(config, "astrom_reg", 1e-4))
            lam_b = astrom_reg * scale_BB
            BBreg = BB + sp.eye(BB.shape[0], format="csr") * lam_b

            flux, err, shifts, shift_cov, info = SceneFitter._solve_flux_and_shifts(
                Areg, b, AB, BBreg, bB, config
            )
        else:
            flux, err, info = SceneFitter.solve_flux(Areg, b, config)
            shifts, shift_cov = None, None

        return SimpleNamespace(
            flux=flux, err=err, shifts=shifts, shift_cov=shift_cov, info=info
        )

    @staticmethod
    def solve_flux(
        A: sp.spmatrix, b: np.ndarray, config: Optional[FitConfig] = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve ``A x = b`` for flux parameters (flux-only path).

        The matrix is whitened by its diagonal, ``A_w = D^-1 A D^-1`` with
        ``D = diag(sqrt(A_ii))``, solved directly, and unwhitened. Errors
        are ``sqrt(diag(A_w^-1)) / d``. If ``config.positivity`` is true,
        negative fluxes are clipped to zero after the solve (a post-hoc
        clamp, not a constrained NNLS solve).
        """
        cfg = config or FitConfig()
        A = A.tocsr()

        d = np.sqrt(np.maximum(A.diagonal(), 1e-12))
        Dinv = sp.diags(1.0 / d, 0, format="csr")
        A_w = Dinv @ A @ Dinv
        b_w = Dinv @ b

        x_w = spsolve(A_w, b_w)
        x = x_w / d
        err = SceneFitter._flux_errors(A_w) / d

        if cfg.positivity:
            x = np.maximum(0.0, x)

        return x, err, {"solver": "spsolve"}

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
        shift_cov = SceneFitter._shift_covariance(A_w, AB_w, Linv)

        if cfg.positivity:
            x = np.maximum(0.0, x)

        return x, err, beta, shift_cov, {"solver": "spsolve"}

    @staticmethod
    def _shift_covariance(
        A_w: sp.spmatrix, AB_w: sp.spmatrix, Linv: np.ndarray
    ) -> np.ndarray:
        """Covariance of the unwhitened shift coefficients, fluxes marginalized.

        The counterpart of :meth:`_flux_errors`, for the other block. In
        whitened variables the joint matrix is ``[[A_w, AB_w], [AB_wᵀ, I]]``,
        so the shift block of its inverse is
        ``(I − AB_wᵀ A_w⁻¹ AB_w)⁻¹`` -- the shift information *after* the
        fluxes have been allowed to absorb what they can. Unwhitening with
        ``beta = L⁻ᵀ betaw`` gives ``Linvᵀ · cov · Linv``.

        Ignoring the marginalization and taking ``BB⁻¹`` instead would be free
        (``Linvᵀ Linv``) but wrong by the flux-shift degeneracy, which reaches
        tens of percent once a neighbour sits about a FWHM from an anchor.

        Cost is one sparse factorization of the flux block plus ``nB``
        back-solves, and ``nB`` is 2 at the default order 0.

        Parameters
        ----------
        A_w
            Whitened flux block.
        AB_w
            Whitened flux-shift coupling, ``(nA, nB)``.
        Linv
            Inverse Cholesky factor of the shift block used to whiten it.

        Returns
        -------
        ndarray
            ``(nB, nB)`` covariance, or NaN where the factorization or the
            inverse fails.
        """
        ABd = AB_w.toarray() if sp.issparse(AB_w) else np.asarray(AB_w, dtype=float)
        nB = ABd.shape[1]
        if nB == 0:
            return np.zeros((0, 0), dtype=float)
        try:
            factor = splu((A_w.tocsc() if sp.issparse(A_w) else sp.csc_matrix(A_w)))
            Ainv_AB = np.column_stack([factor.solve(ABd[:, k]) for k in range(nB)])
            cov_w = np.linalg.inv(np.eye(nB) - ABd.T @ Ainv_AB)
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            logger.debug("[scenes] shift covariance unavailable", exc_info=True)
            return np.full((nB, nB), np.nan, dtype=float)
        cov = Linv.T @ cov_w @ Linv
        return cov if np.all(np.isfinite(cov)) else np.full((nB, nB), np.nan)

    @staticmethod
    def _flux_errors(A, dense_threshold: int = 500) -> np.ndarray:
        """Return ``sqrt(diag(A^{-1}))`` — 1-σ errors with off-diagonal coupling.

        ``A`` is expected SPD (a whitened normal matrix or its Schur
        complement). The sparse path -- factor once, back-solve one unit
        column at a time -- only pays when the matrix really is sparse, so the
        dispatch is on what ``A`` *is*, not on its size alone.

        That distinction matters because the two callers pass different
        things. The flux-only solve passes the whitened normal matrix, which is
        genuinely sparse (each template overlaps a handful of neighbours). The
        joint flux+shift solve passes the Schur complement
        ``S_w = A_w - AB_w AB_w^T``, and that outer product is fully populated,
        so ``S_w`` arrives as a dense ndarray. Routing it through
        ``csc_matrix`` + ``splu`` factorises a dense matrix as though it were
        sparse: for a 1718-template scene, 118 MB and n back-solves (5.1
        Gflop) against 47 MB and one LAPACK inversion (1.7 Gflop).
        """
        n = A.shape[0]
        if not sp.issparse(A):
            # dense in, dense out: converting to csc here would only add the
            # index arrays of an n^2-nonzero matrix on top of the values
            return np.sqrt(np.maximum(np.diag(np.linalg.inv(np.asarray(A))), 1e-12))
        if n <= dense_threshold:
            M = A.toarray()
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
