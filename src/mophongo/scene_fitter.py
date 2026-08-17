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
    def fnnls(
        A: sp.spmatrix | np.ndarray,
        b: np.ndarray,
        *,
        free: np.ndarray | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Non-negative least squares from the normal equations.

        Minimizes ``||M x - d||`` subject to ``x >= 0`` given only the Gram
        matrix ``A = M^T M`` and ``b = M^T d`` -- which is what a scene holds,
        so no design matrix has to be reconstructed. This is the Bro & de Jong
        (1997) "fast NNLS" rearrangement of Lawson-Hanson: the active-set logic
        is unchanged, but every inner least-squares problem is solved from the
        Gram submatrix rather than from ``M``.

        The difference from clipping matters most where template fitting is
        hardest. Forcing one template to zero changes what its *neighbours*
        should be: in a blend they took flux only because the negative one was
        there. The active set re-solves the survivors each time the passive set
        changes; a clip does not, and leaves the scene inconsistent.

        Args:
            A: Gram matrix, ``(n, n)``, symmetric positive semi-definite.
            b: Right-hand side ``M^T d``, ``(n,)``.
            free: Optional ``(n,)`` mask of components that are *not*
                constrained -- they stay in the passive set throughout and may
                go negative. This is what lets the joint flux+shift system use
                the same routine: a shift coefficient has no sign to respect,
                only the fluxes do.
            tol: Optimality tolerance on the KKT (gradient) test. Default
                scales with the problem: ``n * eps * max|diag(A)|``.
            max_iter: Outer-iteration cap. Default ``30 * n``, the
                Lawson-Hanson convention; hitting it warns and returns the
                current feasible point rather than raising, because a scene
                that fails to converge should still produce fluxes.

        Returns:
            ``(x, passive)``: the solution, and the boolean mask of components
            left free (not pinned at the bound). The mask is what callers need
            in order to say which errors describe a constrained parameter.
        """
        A_d = np.asarray(A.todense() if sp.issparse(A) else A, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        n = b.size
        if tol is None:
            diag_max = float(np.max(np.abs(np.diag(A_d)))) if n else 0.0
            tol = max(n * np.finfo(float).eps * diag_max, 1e-12)
        if max_iter is None:
            max_iter = 30 * max(n, 1)

        free = (np.zeros(n, dtype=bool) if free is None
                else np.asarray(free, dtype=bool))

        def _solve_passive(passive: np.ndarray) -> np.ndarray:
            out = np.zeros(n)
            idx = np.where(passive)[0]
            if idx.size == 0:
                return out
            sub = A_d[np.ix_(idx, idx)]
            try:
                out[idx] = np.linalg.solve(sub, b[idx])
            except np.linalg.LinAlgError:
                out[idx] = np.linalg.lstsq(sub, b[idx], rcond=None)[0]
            return out

        # Unconstrained components start in the passive set and never leave, so
        # the first solve already carries them.
        passive = free.copy()
        x = _solve_passive(passive) if passive.any() else np.zeros(n)
        x[~passive] = 0.0

        for _ in range(max_iter):
            w = b - A_d @ x  # -gradient of 0.5 x'Ax - b'x
            blocked = ~passive
            if not blocked.any() or np.all(w[blocked] <= tol):
                break
            # bring in the blocked component with the steepest descent
            candidates = np.where(blocked)[0]
            passive[candidates[np.argmax(w[candidates])]] = True

            # exact solve on the passive set, then walk back to feasibility
            for _ in range(max_iter):
                s = _solve_passive(passive)
                # only constrained components have a sign to violate
                check = passive & ~free
                if not check.any() or np.all(s[check] > 0):
                    x = s
                    break
                bad = np.where(check & (s <= 0))[0]
                denom = x[bad] - s[bad]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.where(denom != 0, x[bad] / denom, 0.0)
                alpha = float(np.min(ratios))
                x = x + alpha * (s - x)
                passive &= free | (x > tol)
                x[~passive] = 0.0
            else:
                logger.warning("fnnls inner loop hit %d iterations", max_iter)
                break
        else:
            logger.warning(
                "fnnls did not converge in %d iterations; returning the "
                "current feasible point", max_iter,
            )
        return x, passive

    @staticmethod
    def solve_flux(
        A: sp.spmatrix, b: np.ndarray, config: Optional[FitConfig] = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve ``A x = b`` for flux parameters (flux-only path).

        The matrix is whitened by its diagonal, ``A_w = D^-1 A D^-1`` with
        ``D = diag(sqrt(A_ii))``, solved directly, and unwhitened. Errors
        are ``sqrt(diag(A_w^-1)) / d``.

        ``config.fit_method`` then decides what happens to negatives:
        ``"lls"`` keeps them, ``"clip"`` clamps them to zero, ``"nnls"``
        re-solves under the constraint (:meth:`fnnls`). The unconstrained
        fluxes are returned whichever runs, in ``info["flux_uncon"]``: a
        clipped or pinned zero is not recoverable from the constrained answer,
        and faint-source statistics need the negative half of the noise
        distribution.
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
        info: dict = {"solver": "spsolve", "flux_uncon": x.copy()}

        method = str(cfg.fit_method).lower()
        if method == "nnls":
            # solved whitened, like the unconstrained path: the constraint
            # x >= 0 is preserved by the positive diagonal scaling
            x_w, passive = SceneFitter.fnnls(A_w, b_w)
            x = x_w / d
            info["solver"] = "fnnls"
            info["at_bound"] = ~passive
        elif method == "clip":
            x = np.maximum(0.0, x)
            info["at_bound"] = info["flux_uncon"] < 0.0

        return x, err, info

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
        na = A.shape[0]
        nb = rhs.size - na

        sol = spsolve(K, rhs)
        xw_uncon = sol[:na]

        solver = "spsolve"
        at_bound = None
        if str(cfg.fit_method).lower() == "nnls":
            # Same active set as the flux-only path, with the shift
            # coefficients held free: a shift has no sign to respect, only the
            # fluxes do. Solving the coupled system this way keeps the shifts
            # consistent with whichever fluxes survive, which is the whole
            # reason the two blocks are solved together.
            free = np.concatenate([np.zeros(na, bool), np.ones(nb, bool)])
            sol, passive = SceneFitter.fnnls(K, rhs, free=free)
            at_bound = ~passive[:na]
            solver = "fnnls"

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

        info = {"solver": solver, "flux_uncon": xw_uncon / d}
        if str(cfg.fit_method).lower() == "clip":
            x = np.maximum(0.0, x)
            info["at_bound"] = info["flux_uncon"] < 0.0
        elif at_bound is not None:
            info["at_bound"] = at_bound

        return x, err, beta, shift_cov, info

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
