from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from numpy.random import default_rng
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, lil_matrix, bmat
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import (
    LinearOperator,
    cg,
    lsqr,
    minres,
    spsolve_triangular,
)
from tqdm import tqdm

from .astrometry import cheb_basis, AstroCorrect, n_terms
from .templates import Template, Templates

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # show info for *this* logger only
if not logger.handlers:  # avoid duplicate handlers on reloads
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(module)s.%(funcName)s: %(message)s"))
    logger.addHandler(handler)

# full weights need to be calcuate like
# template_var = scipy.signal.fftconvolve(K**2, 1 / wht1, mode='same')  # same shape as template
# Iterate if needed (since A appears in w(x)):
# First fit using weights = wht2
# Compute A (amplitude)
# Recompute weights using full formula
# Refit using updated weights if you want accurate errors
# wht_tot = 1 / (1 / wht2 + A**2 * template_var)
# Pass wht_tot to your SparseFitter.
# Multiple templates: you must apply the same logic to each template independently. This means different pixels may have different total weights for each template, depending on each one's amplitude and support.
# Correlated templates (overlapping) require full covariance accounting; your current implementation approximates this by assuming per-template independence.
# If template noise is negligible, simplify to: weights = wht2 (as in your current default).
# Flux-dependent variance (via A^2) introduces mild nonlinearity; it's safe to fix A from initial fit for a single iteration.


@dataclass
class FitConfig:
    """Configuration options for :class:`SparseFitter`."""

    positivity: bool = True
    reg: float = 0.0
    bad_value: float = np.nan
    solve_method: str = "scene"  # 'all', 'scene', 'lo' (linear operator)
    cg_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"M": None, "maxiter": 500, "atol": 1e-6}
    )
    fit_covariances: bool = False

    fft_fast: float | bool = False  # False for full kernel, 0.1-1.0 for truncated FFT
    # condense fit astrometry flags into one: fit_astrometry_niter = 0, means not fitting astrometry
    fit_astrometry_niter: int = 2  # Number of astrometry refinement passes (0 → disabled)
    fit_astrometry_joint: bool = True  # Use joint astrometry fitting, or separate step
    # --- astrometry options -------------------------------------------------
    reg_astrom: float = 1e-4
    snr_thresh_astrom: float = 15.0  # 0 → keep all sources
    astrom_model: str = "gp"  # 'polynomial' or 'gp'
    astrom_centroid: str = "centroid"  # "centroid" (=old) | "correlation"
    astrom_kwargs: dict[str, dict] = field(
        default_factory=lambda: {"poly": {"order": 0}, "gp": {"length_scale": 400}}
    )
    #    astrom_kwargs={'poly': {'order': 2}, 'gp': {'length_scale': 400}}
    multi_tmpl_chi2_thresh: float = 5.0
    multi_tmpl_psf_core: bool = False
    multi_tmpl_colour: bool = False
    #    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    normal: str = "tree"  # 'loop' or 'tree'
    scene_merge_small: bool = True  # Merge small scenes before building bases
    # None → derive from astrometric model order in __post_init__
    # Minimum bright sources per scene. If None reverts to (n_poly+1)*(n_poly+2)
    scene_minimum_bright: int = 5
    negative_snr_thresh: float = -1.0  # Threshold for negative SNR fluxes to apply soft priors

    # Photometry aperture control:
    # - float/int: fixed aperture diameter size (in arcsec or pixels per `aperture_units`)
    # - str: column name in the input catalog for per-source aperture sizes
    # - None: fallback to 1.5 * FWHM (in pixels) measured from template
    aperture_diam: float | np.ndarray | None = None  # image measurement aperture (diameter)
    aperture_catalog: float | str | None = None  # catalog aperture (diameter or table column name)
    aperture_units: str = "arcsec"  # "arcsec" or "pix"

    # Internal options: don't change unless you know what you're doing
    block_size: int = 64  # Block size for tiled processing

    # scene processing
    run_scene_solver: bool = True  # Whether to run the scene solver at all
    scene_coupling_thresh: float = 1e-3  # 1% leakage threshold for scene splitting

    def __post_init__(self):
        # Derive scene_minimum_bright from astrometric polynomial order if not provided
        if self.scene_minimum_bright is None:
            try:
                poly_order = int(self.astrom_kwargs.get("poly", {}).get("order", 1))
            except Exception:
                poly_order = 1
            # default to 2x # of Chebyshev terms + 1
            n_poly = (poly_order + 1) * (poly_order + 2)
            self.scene_minimum_bright = n_poly + 1


def _diag_inv_hutch(A, k=32, rtol=1e-4, maxiter=None, seed=0):
    """
    Robust Hutchinson estimator of diag(A^{-1}).

    Strategy:
    1.   Try CG (fastest) with loose `rtol`; works if A is PD enough.
    2.   Fall back to MINRES (handles indef | singular) *without*
         raising on non-convergence – we just take the last iterate.
    3.   If both stall, add a ×10 diagonal jitter and restart *once*.
    """
    n = A.shape[0]
    maxiter = maxiter or 6 * n
    rng = default_rng(seed)
    acc = np.zeros(n)

    def _solve(rhs):
        # --- 1. CG attempt ------------------------------------------------
        x, flag = cg(A, rhs, rtol=rtol, atol=0, maxiter=maxiter)
        if flag == 0:
            return x
        # --- 2. MINRES rescue --------------------------------------------
        x, flag = minres(A, rhs, rtol=rtol, maxiter=maxiter)  # never raises
        if flag in (0, 1):  # converged or hit maxiter
            return x
        # --- 3. add jitter & restart once --------------------------------
        diag_boost = 1e-4 * np.median(A.diagonal())
        x, _ = cg(A + diag_boost * np.eye(n), rhs, rtol=rtol, atol=0, maxiter=maxiter)
        return x  # accept whatever we get

    for _ in range(k):
        z = rng.choice((-1.0, 1.0), size=n)
        x = _solve(z)
        acc += z * x

    return np.sqrt(np.abs(acc / k))


def sparse_cholesky(
    A: sp.spmatrix, *, use_rcm: bool = False, drop_tol: float = 0.0
) -> tuple[csc_matrix, np.ndarray]:
    """Sparse left-looking Cholesky factorization.

    Parameters
    ----------
    A
        Symmetric positive-definite matrix in sparse format.
    use_rcm
        Apply Reverse Cuthill–McKee permutation before factorization.
    drop_tol
        Drop entries with absolute value below this threshold (IC(0) when 0).

    Returns
    -------
    L, p
        ``L`` is the lower-triangular Cholesky factor in CSC format and
        ``p`` is the permutation applied such that ``A[p][:, p] = L @ L.T``.
    """

    if not sp.isspmatrix(A):
        raise TypeError("A must be a SciPy sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    Acsc = sp.tril(A.tocsc(), format="csc")
    n = Acsc.shape[0]

    if use_rcm:
        Apat = (Acsc + Acsc.T).tocsr()
        Apat.data[:] = 1.0
        p = reverse_cuthill_mckee(Apat, symmetric_mode=True)
    else:
        p = np.arange(n, dtype=np.int64)

    Aperm = Acsc[p, :][:, p].tocsc()
    indptr, indices, data = Aperm.indptr, Aperm.indices, Aperm.data

    Lcols: list[dict[int, float]] = [dict() for _ in range(n)]
    rowcols: list[list[int]] = [[] for _ in range(n)]

    for k in range(n):
        w: dict[int, float] = {}
        for t in range(indptr[k], indptr[k + 1]):
            i = indices[t]
            if i >= k:
                w[i] = float(data[t])

        for j in rowcols[k]:
            m = Lcols[j][k]
            if m == 0.0:
                continue
            for i, Lij in Lcols[j].items():
                if i <= j or i < k:
                    continue
                c = Lij * m
                if c != 0.0:
                    if i in w:
                        w[i] -= c
                    elif abs(c) > drop_tol:
                        w[i] = -c

        diag = w.get(k, 0.0)
        if diag <= 0.0:
            diag += 1e-15
        Lkk = float(np.sqrt(diag))
        if not np.isfinite(Lkk) or Lkk == 0.0:
            raise np.linalg.LinAlgError(f"Non-finite/zero pivot at column {k}")
        Lcols[k][k] = Lkk
        invLkk = 1.0 / Lkk

        for i in sorted(w.keys()):
            if i == k:
                continue
            val = w[i] * invLkk
            if drop_tol > 0.0 and abs(val) < drop_tol:
                continue
            Lcols[k][i] = val
            rowcols[i].append(k)

    nnz = sum(len(col) for col in Lcols)
    indptr_L = np.zeros(n + 1, dtype=int)
    indices_L = np.empty(nnz, dtype=int)
    data_L = np.empty(nnz, dtype=float)
    pos = 0
    for k in range(n):
        indptr_L[k] = pos
        for i, v in sorted(Lcols[k].items(), key=lambda x: x[0]):
            indices_L[pos] = i
            data_L[pos] = v
            pos += 1
    indptr_L[n] = pos
    L = csc_matrix((data_L, indices_L, indptr_L), shape=(n, n))
    return L, p


def make_sparse_chol_prec(
    A: sp.spmatrix, *, use_rcm: bool = False, drop_tol: float = 0.0
) -> LinearOperator:
    """Return ``LinearOperator`` applying ``(LLᵀ)⁻¹`` via sparse Cholesky."""

    L, p = sparse_cholesky(A, use_rcm=use_rcm, drop_tol=drop_tol)
    n = A.shape[0]
    invp = np.empty_like(p)
    invp[p] = np.arange(n, dtype=p.dtype)

    def _apply(v: np.ndarray) -> np.ndarray:
        vp = v[p]
        y = spsolve_triangular(L, vp, lower=True)
        w = spsolve_triangular(L.T, y, lower=False)
        return w[invp]

    return LinearOperator((n, n), matvec=_apply, rmatvec=_apply, dtype=A.dtype)


import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def build_scene_tree_from_normal(
    ATA: sp.spmatrix,
    ATb: np.ndarray,
    *,
    coupling_thresh: float = 0.03,  # 3% leakage threshold
    return_0_based: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Scene partition from normal-equation couplings.
    Connect i–j if the predicted cross-leakage between their diagonal-only
    fits exceeds `coupling_thresh`, then take connected components.

    Parameters
    ----------
    ATA : (n,n) sparse
        Un-whitened normal matrix (your `_ata`).
    ATb : (n,) array
        RHS (your `_atb`).
    coupling_thresh : float
        Edge if max(|A_ij α_j|/(A_ii|α_i|), |A_ij α_i|/(A_jj|α_j|)) >= threshold.
        0.02–0.05 works well; higher → more aggressive splitting.
    return_0_based : bool
        If True, labels are 0..K-1; else 1..K (default).

    Returns
    -------
    labels : (n) int array
        Scene id per template.
    nscene : int
        Number of scenes.
    """
    if not sp.isspmatrix(ATA):
        raise TypeError("ATA must be a SciPy sparse matrix")
    n = ATA.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), 0

    A = ATA.tocsr()
    d = A.diagonal().astype(float)
    # Numerical floor: if a diagonal is ~0 it should already have been pruned,
    # but keep it safe.
    eps_d = max(1e-30, 1e-12 * np.median(d[d > 0])) if np.any(d > 0) else 1e-30

    # Diagonal-only amplitudes
    alpha = np.divide(ATb, d, out=np.zeros_like(ATb, dtype=float), where=d > eps_d)
    abs_alpha = np.abs(alpha)

    # Work on strict upper triangle only
    # (coo is convenient to vectorize)
    Au = sp.triu(A, k=1).tocoo()
    if Au.nnz == 0:
        labs = np.arange(n, dtype=int)
        return (labs if return_0_based else labs + 1), n

    i = Au.row
    j = Au.col
    aij = np.abs(Au.data)

    di = d[i]
    dj = d[j]
    ai = abs_alpha[i]
    aj = abs_alpha[j]

    # r_ij = |A_ij α_j| / (A_ii |α_i| + eps),   r_ji = aij * ai / (denom_j + eps_j)
    denom_i = di * ai
    denom_j = dj * aj

    # add small stabilization only where denom ~ 0
    eps_i = np.where(denom_i > 0, 0.0, eps_d)
    eps_j = np.where(denom_j > 0, 0.0, eps_d)

    r_ij = aij * aj / (denom_i + eps_i)
    r_ji = aij * ai / (denom_j + eps_j)
    score = np.maximum(r_ij, r_ji)

    mask = score >= float(coupling_thresh)
    if not np.any(mask):
        labs = np.arange(n, dtype=int)
        return (labs if return_0_based else labs + 1), n

    ii = i[mask]
    jj = j[mask]
    # Build symmetric adjacency for the kept edges
    m = mask.sum()
    data = np.ones(m * 2, dtype=np.uint8)
    rows = np.concatenate([ii, jj])
    cols = np.concatenate([jj, ii])
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    nscene, labels0 = connected_components(adj, directed=False)
    return (labels0 if return_0_based else labels0 + 1), int(nscene)


from shapely.geometry import Point
from shapely.strtree import STRtree


def merge_small_scenes(
    labels: np.ndarray,
    templates: list[Template],
    bright_mask: np.ndarray,
    *,
    order: int = 1,
    minimum_bright: int | None = None,
    max_merge_radius: float = np.inf,  # pixels
    max_iter: int = 64,
) -> tuple[np.ndarray, int]:
    """
    Merge scenes below the bright threshold into their nearest scene.
    Uses Shapely 2.x STRtree.query_nearest (bulk) and unions all pairs per round.
    Returns (1-based labels, n_scenes).
    """
    if minimum_bright is None:
        minimum_bright = len(cheb_basis(0.0, 0.0, order))

    # Work with compact 0..K-1 labels for bincounts
    labs = np.unique(labels, return_inverse=True)[1]

    # Per-template positions & bright flags
    x = np.array([t.position_original[0] for t in templates], dtype=float)
    y = np.array([t.position_original[1] for t in templates], dtype=float)
    b = bright_mask.astype(np.int64)

    for _ in range(max_iter):
        counts = np.bincount(labs)
        K = counts.size
        if K <= 1:
            break

        valid = counts > 0
        ids = np.nonzero(valid)[0]
        if ids.size <= 1:
            break

        # Per-scene aggregates
        sumx = np.bincount(labs, weights=x, minlength=K)
        sumy = np.bincount(labs, weights=y, minlength=K)
        nbright = np.bincount(labs, weights=b, minlength=K).astype(int)

        cx = np.full(K, np.nan, dtype=float)
        cy = np.full(K, np.nan, dtype=float)
        cx[valid] = sumx[valid] / counts[valid]
        cy[valid] = sumy[valid] / counts[valid]

        under = np.where((nbright < minimum_bright) & valid)[0]
        if under.size == 0:
            break

        # Build STRtree over centroids of valid scenes (targets)
        pts = [Point(float(cx[i]), float(cy[i])) for i in ids]
        tree = STRtree(pts)

        # Query nearest for each underfilled scene (sources)
        q_pts = [Point(float(cx[i]), float(cy[i])) for i in under]

        if np.isfinite(max_merge_radius):
            pair_idx, _ = tree.query_nearest(
                q_pts,
                exclusive=True,
                return_distance=True,
                max_distance=float(max_merge_radius),
            )
            if pair_idx.size == 0:
                break
        else:
            pair_idx, _ = tree.query_nearest(q_pts, exclusive=True, return_distance=True)

        # Map query indices back to scene ids in [0..K-1]
        src = under[pair_idx[0].astype(int)]
        dst = ids[pair_idx[1].astype(int)]

        # Remove any accidental self-pairs (shouldn’t happen with exclusive=True)
        m = src != dst
        if not np.any(m):
            break
        src = src[m]
        dst = dst[m]

        # -------- union all pairs in one go (prevents A↔B label swaps) -------
        parent = np.arange(K, dtype=int)

        def find(a: int) -> int:
            # path compression
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u, v in zip(src, dst):
            ru, rv = find(u), find(v)
            if ru != rv:
                # union by simple heuristic: attach smaller index to larger
                if ru < rv:
                    parent[ru] = rv
                else:
                    parent[rv] = ru

        # Relabel all members by representative
        labs = np.fromiter((find(int(li)) for li in labs), dtype=int, count=labs.size)

        # loop: recompute aggregates on merged labels

    # Final compact relabel to 1..K (1-based)
    uniq, inv = np.unique(labs, return_inverse=True)
    new_labs = (inv + 1).astype(int)
    return new_labs, int(uniq.size)


def make_basis_per_scene(
    templates: list[Template],
    labels: np.ndarray,
    bright: np.ndarray,
    order: int = 1,
) -> tuple[
    list[Optional[np.ndarray]], dict[int, tuple[float, float]], dict[int, tuple[float, float]]
]:
    """Return per-template basis vectors for BRIGHT sources and per-scene center/scale."""
    basis: list[Optional[np.ndarray]] = [None] * len(templates)
    centers: dict[int, tuple[float, float]] = {}
    scales: dict[int, tuple[float, float]] = {}

    for sid in np.unique(labels):
        idx = np.where(labels == sid)[0]
        if idx.size == 0:
            continue

        # Prefer bright members to define center/scale; fall back to all in scene
        idx_b = [i for i in idx if bright[i]]
        use = idx_b if idx_b else idx

        xs = np.array([templates[i].position_original[0] for i in use], float)
        ys = np.array([templates[i].position_original[1] for i in use], float)

        x0 = float(xs.mean())
        y0 = float(ys.mean())
        # Half-range scaling → map roughly to [-1,1]
        Sx = 0.5 * float(xs.max() - xs.min()) if xs.size else 1.0
        Sy = 0.5 * float(ys.max() - ys.min()) if ys.size else 1.0
        # Pad a bit and guard against degenerate scenes
        Sx = max(1.0, 1.05 * Sx)
        Sy = max(1.0, 1.05 * Sy)

        centers[int(sid)] = (x0, y0)
        scales[int(sid)] = (Sx, Sy)

        for i in idx:
            if not bright[i]:
                continue
            x, y = templates[i].position_original
            u = (x - x0) / Sx
            v = (y - y0) / Sy
            basis[i] = cheb_basis(u, v, order)

    return basis, centers, scales


def assemble_scene_system_self_AB(
    idx: list[int],
    templates: list[Template],
    image: np.ndarray,
    weights: np.ndarray,
    basis_vals: list[Optional[np.ndarray]],
    *,
    alpha0: np.ndarray,  # <— NEW: per-template flux (unwhitened)
    order: int = 1,
    include_y: bool = True,
    ab_from_bright_only: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    # Which members have a shift basis in this scene?
    bright_in_idx = [g for g in idx if basis_vals[g] is not None]
    has_shift = len(bright_in_idx) >= 2
    if not has_shift:
        nA = len(idx)
        return sp.csr_matrix((nA, 0)), sp.csr_matrix((0, 0)), np.zeros(0, float)

    p = len(cheb_basis(0.0, 0.0, order))
    nA = len(idx)
    nB = p * (2 if include_y else 1)

    AB = sp.lil_matrix((nA, nB))
    BB = np.zeros((nB, nB), float)
    bB = np.zeros(nB, float)

    grad_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _gx_gy_for(g: int) -> tuple[np.ndarray, np.ndarray]:
        if g not in grad_cache:
            arr = templates[g].data.astype(float)
            if arr.shape[0] < 2 or arr.shape[1] < 2:
                gy = np.zeros_like(arr)
                gx = np.zeros_like(arr)
            else:
                gy, gx = np.gradient(arr)  # gy=d/dy, gx=d/dx
            grad_cache[g] = (gx, gy)
        return grad_cache[g]

    for row, g in enumerate(idx):
        ti = templates[g]
        sl = ti.slices_original
        tcut = ti.data[ti.slices_cutout]
        w = weights[sl]
        img = image[sl]

        Si = basis_vals[g]
        if (Si is None) and ab_from_bright_only:
            continue

        ai = float(alpha0[g])  # <-- flux scaling (pixels stay in dx/dy)

        Gx, Gy = _gx_gy_for(g)

        # Inner products
        gx_ip = float(np.sum(tcut * w * Gx[ti.slices_cutout]))
        AB[row, 0:p] += (-ai) * gx_ip * (Si if Si is not None else 0.0)
        if include_y:
            gy_ip = float(np.sum(tcut * w * Gy[ti.slices_cutout]))
            AB[row, p : 2 * p] += (-ai) * gy_ip * (Si if Si is not None else 0.0)

        if Si is not None:
            Gxx = float(np.sum(Gx[ti.slices_cutout] * w * Gx[ti.slices_cutout]))
            BB[0:p, 0:p] += (ai * ai) * Gxx * np.outer(Si, Si)
            if include_y:
                Gyy = float(np.sum(Gy[ti.slices_cutout] * w * Gy[ti.slices_cutout]))
                BB[p : 2 * p, p : 2 * p] += (ai * ai) * Gyy * np.outer(Si, Si)

            # RHS for beta (Gauss–Newton: J_beta^T W y; sign matches AB above)
            bB[0:p] += (-ai) * float(np.sum(Gx[ti.slices_cutout] * w * img)) * Si
            if include_y:
                bB[p : 2 * p] += (-ai) * float(np.sum(Gy[ti.slices_cutout] * w * img)) * Si

    return AB.tocsr(), sp.csr_matrix(BB), bB


def summarize_scenes(labels: np.ndarray) -> np.ndarray:
    """Log a brief summary of scene sizes."""

    counts = np.bincount(labels)[1:]  # skip 0 bin
    logger.info(
        "%d scenes (max=%d, median=%d, min=%d)",
        len(counts),
        counts.max(),
        int(np.median(counts)),
        counts.min(),
    )
    topk = np.argsort(counts)[::-1][:5]
    logger.info(
        "Top scenes by size: %s",
        [(int(cid), int(counts[cid])) for cid in topk],
    )
    return counts


def solve_scene_cg(
    ATA_w_csr: csr_matrix,
    ATb_w: np.ndarray,
    labels: np.ndarray,
    *,
    rtol: float = 1e-6,
    maxiter: int = 2000,
) -> tuple[np.ndarray, list[int]]:
    """Solve block-diagonal systems independently with CG.

    Assumes ``ATA_w_csr`` and ``ATb_w`` have been whitened beforehand.
    """

    x = np.zeros_like(ATb_w, dtype=float)
    infos: list[int] = []
    for sid in range(labels.max() + 1):
        idx = np.where(labels == sid)[0]
        if idx.size == 0:
            infos.append(0)
            continue

        A = ATA_w_csr[idx][:, idx].tocsr()
        b = ATb_w[idx]

        M = None
        sol, info = cg(A, b, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
        x[idx] = sol
        infos.append(int(info))

    return x, infos


class SparseFitter:
    """Build and solve sparse normal equations for photometry."""

    def __init__(
        self,
        templates: List[Template],
        image: np.ndarray,
        weights: np.ndarray | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if weights is None:
            weights = np.ones_like(image)

        self._orig_templates = templates  # keep original templates List object
        self.templates = templates.copy()  # work in list copy for fitting, modifying

        self.n_flux = len(templates)
        for i, tmpl in enumerate(self.templates):
            tmpl.is_flux = True
            tmpl.col_idx = i

        self.image = image
        self.weights = weights
        self.config = config or FitConfig()
        self._ata = None
        self._atb = None
        self.scene_ids = None  # scene ids for each template
        self.solution: np.ndarray | None = None

        # Identify high-S/N templates for astrometric shift fitting
        flux_est = Templates.quick_flux(self.templates, self.image)
        err_est = Templates.predicted_errors(self.templates, self.weights)
        snr = np.divide(
            flux_est,
            err_est,
            out=np.zeros_like(flux_est),
            where=err_est > 0,
        )

    #        self.orig_bright = snr > self.config.snr_thresh_astrom

    @staticmethod
    def _intersection(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int] | None:
        y0 = max(a[0], b[0])
        y1 = min(a[1], b[1])
        x0 = max(a[2], b[2])
        x1 = min(a[3], b[3])
        if y0 >= y1 or x0 >= x1:
            return None
        return y0, y1, x0, x1

    @staticmethod
    def _bbox_to_slices(bbox: Tuple[int, int, int, int]) -> Tuple[slice, slice]:
        """Convert integer bounding box to slices for array indexing."""
        y0, y1, x0, x1 = bbox
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
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

    def _weighted_norm(self, tmpl: Template) -> float:
        """Return the weighted L2 norm of ``tmpl``.
        The norm is computed by summing ``data * weight * data`` over the
        template support in the image space.
        """
        sl = tmpl.slices_original
        data = tmpl.data[tmpl.slices_cutout]
        w = self.weights[sl]
        wnorm = float(np.sum(data * w * data))
        tmpl.wnorm = wnorm
        return wnorm

    def build_normal(self) -> None:
        """Dispatch to the configured normal-matrix builder."""
        if getattr(self.config, "normal", "loop") == "tree":
            self.build_normal_tree()
        else:
            self.build_normal_matrix()

    def build_normal_tree(self) -> None:
        """Construct normal matrix using an STRtree to find overlaps."""
        from shapely.geometry import box
        from shapely.strtree import STRtree

        # scan for low norm templates but keep them for now
        norms = np.array([self._weighted_norm(t) for t in self.templates])
        tol = 1e-6 * np.median(norms)
        if np.sum(norms < tol) > 0:
            logger.info("Found %d templates with low norm.", np.sum(norms < tol))
        # self.templates = [self.templates[i] for i in keep]
        # norms = [norms[i] for i in keep]

        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        boxes = []
        for i, tmpl in enumerate(tqdm(self.templates, total=n, desc="Building Normal matrix")):
            sl_i = tmpl.slices_original
            data_i = tmpl.data[tmpl.slices_cutout]
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            atb[i] = np.sum(data_i * w_i * img_i)
            ata[i, i] = norms[i]

            y0, y1, x0, x1 = tmpl.bbox
            geom = box(x0, y0, x1, y1)
            boxes.append(geom)

        tree = STRtree(boxes)

        for i, geom in enumerate(boxes):
            sl_i = self.templates[i].slices_original
            for j in tree.query(geom):
                j = int(j)
                if j <= i:
                    continue
                inter = self._slice_intersection(sl_i, self.templates[j].slices_original)
                if inter is None:
                    continue
                w = self.weights[inter]
                sl_i_local = (
                    slice(
                        inter[0].start - sl_i[0].start + self.templates[i].slices_cutout[0].start,
                        inter[0].stop - sl_i[0].start + self.templates[i].slices_cutout[0].start,
                    ),
                    slice(
                        inter[1].start - sl_i[1].start + self.templates[i].slices_cutout[1].start,
                        inter[1].stop - sl_i[1].start + self.templates[i].slices_cutout[1].start,
                    ),
                )
                sl_j = self.templates[j].slices_original
                sl_j_local = (
                    slice(
                        inter[0].start - sl_j[0].start + self.templates[j].slices_cutout[0].start,
                        inter[0].stop - sl_j[0].start + self.templates[j].slices_cutout[0].start,
                    ),
                    slice(
                        inter[1].start - sl_j[1].start + self.templates[j].slices_cutout[1].start,
                        inter[1].stop - sl_j[1].start + self.templates[j].slices_cutout[1].start,
                    ),
                )
                arr_i = self.templates[i].data[sl_i_local]
                arr_j = self.templates[j].data[sl_j_local]
                val = np.sum(arr_i * arr_j * w)
                ata[i, j] = val
                ata[j, i] = val

        self._ata = ata.tocsr()
        self._atb = atb
        self.rtree = tree

    def model_image(self) -> np.ndarray:
        if self.solution is None:
            raise ValueError("Solve system first")
        model = np.zeros_like(self.image, dtype=float)
        for coeff, tmpl in zip(self.solution, self._orig_templates):
            model[tmpl.slices_original] += coeff * tmpl.data[tmpl.slices_cutout]
        model[self.weights <= 0 | np.isnan(self.weights)] = 0.0
        return model

    @property
    def ata(self):
        if self._ata is None:
            self.build_normal()
        return self._ata

    @property
    def atb(self):
        if self._atb is None:
            self.build_normal()
        return self._atb

    def add_flux_priors(self, idx, mu, sigma, *, floor=1e-12):
        """
        Add Gaussian flux priors to the UNwhitened normal:
            (x_i - mu_i)^2 / sigma_i^2  for i in sel.

        sel   : bool mask of length n or 1D integer indices
        mu    : scalar or array broadcastable to n (or |sel|); prior mean(s)
        sigma : scalar or array broadcastable to n (or |sel|); prior stddev(s)
        """
        import numpy as np
        import scipy.sparse as sp

        # Ensure normal is built (triggers pruning, etc., via properties)
        if self._ata is None or self._atb is None:
            _ = self.ata  # builds and caches
            _ = self.atb

        A = self._ata.tocsr()
        b = np.asarray(self._atb, dtype=float)
        n = b.shape[0]

        # normalize selection to integer indices
        if idx.size == 0:
            return
        nsel = len(idx)

        # broadcast mu/sigma to selected size
        mu_all = np.broadcast_to(mu, (nsel,)) if np.ndim(mu) else float(mu)
        sig_all = np.broadcast_to(sigma, (nsel,)) if np.ndim(sigma) else float(sigma)

        mu_sel = (mu_all if np.ndim(mu_all) else np.full(nsel, mu_all))[idx]
        sig_sel = (sig_all if np.ndim(sig_all) else np.full(nsel, sig_all))[idx]

        # guards
        sig_sel = np.maximum(np.asarray(sig_sel, float), floor)
        lam = 1.0 / (sig_sel**2)  # precisions

        # RHS: b[i] += λ_i * μ_i
        b[idx] += lam * np.asarray(mu_sel, float)

        # Diagonal: A_ii += λ_i
        diag_inc = np.zeros(n, float)
        diag_inc[idx] = lam
        A = A + sp.diags(diag_inc, 0, shape=A.shape, format="csr")

        # write back
        self._ata = A
        self._atb = b

    def solve(self, config: FitConfig | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Solve for template fluxes using conjugate gradient."""
        if config is None:
            config = self.config
        if config.solve_method == "scene":
            return self.solve_scene(config=config)
        else:
            return self.solve_all(config=config)

    def _solve_scenes_with_shifts(
        self,
        A_w: csr_matrix,  # prewhitened flux normal (D^-1 A D^-1)
        b_w: np.ndarray,  # prewhitened RHS (b / d)
        d: np.ndarray,  # whitening scales: sqrt(diag(A_reg))
        scene_ids: np.ndarray,
        templates: List[Template],  # precomputed/pruned list (self.templates)
        bright_mask: np.ndarray,  # precomputed mask aligned to `templates`
        *,
        order: int = 1,
        include_y: bool = True,
        ab_from_bright_only: bool = True,
        rtol: float = 1e-6,
        maxiter: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray, dict]], list[int]]:
        """
        Per-scene solve of joint block system in WHITENED flux space:

            [ A_w(scene)   AB_w ] [ x_w(scene) ] = [ b_w(scene) ]
            [  AB_wᵀ        BB  ] [   beta     ]   [     bB      ]

        Returns
        -------
        alpha : (n)  unwhitened fluxes (x_w / d)
        err   : (n)  unwhitened 1σ flux errors (from whitened Schur complement)
        betas : list[(scene_id, beta_vec)]
        infos : list[int]  CG/MINRES flags per scene
        """
        from scipy.sparse import diags, bmat

        cfg = self.config

        # Per-template polynomial bases (uses provided bright_mask)
        basis_vals, centers, scales = make_basis_per_scene(
            templates, scene_ids, bright_mask, order=order
        )

        n = A_w.shape[0]
        alpha = np.zeros(n, dtype=float)  # unwhitened fluxes
        err = np.zeros(n, dtype=float)
        betas: list[tuple[int, np.ndarray, dict]] = []
        infos: list[int] = []

        logger.debug("Solving %d scenes with shifts", len(np.unique(scene_ids)))

        for sid in np.unique(scene_ids):
            idx = np.where(scene_ids == sid)[0]
            if idx.size == 0:
                betas.append((int(sid), np.zeros(0, dtype=float)))
                infos.append(0)
                continue

            A_w_blk = A_w[idx][:, idx].tocsr()
            b_w_blk = b_w[idx]

            # α⁰ in **unwhitened** units: b_w/d  (since A_ii = d^2 and b = b_w*d)
            alpha0_scene = np.zeros(len(templates), float)
            alpha0_scene[idx] = b_w_blk / d[idx]

            AB, BB, bB = assemble_scene_system_self_AB(
                idx,
                templates,
                self.image,
                self.weights,
                basis_vals,
                order=order,
                include_y=include_y,
                ab_from_bright_only=ab_from_bright_only,
                alpha0=alpha0_scene,
            )

            if AB.shape[1] == 0:
                logger.debug("Scene %d: no shifts, solving flux-only", sid)
                # flux-only in this scene
                xw, info = cg(A_w_blk, b_w_blk, atol=0.0, rtol=rtol, maxiter=maxiter)
                alpha[idx] = xw / d[idx]
                err[idx] = self._flux_errors(A_w_blk) / d[idx]
                betas.append((int(sid), np.zeros(0, dtype=float)))
                infos.append(int(info))
                continue

            # Left-whiten AB rows to match whitened flux variables
            Dinv_scene = diags(1.0 / d[idx], 0, format="csr")
            AB_w = Dinv_scene @ AB  # (nA, nB); BB, bB stay in natural units

            # --- compute ridge once
            tau = float(getattr(self.config, "reg_astrom", 1e-4))
            diagB = BB.diagonal() if sp.issparse(BB) else np.diag(BB)  # safe both ways
            scaleB = np.median(diagB) if diagB.size and np.all(np.isfinite(diagB)) else 1.0
            lam = tau * scaleB

            # --- use the regularized sparse BB in K
            if lam != 0.0:
                BB_reg = BB + sp.eye(BB.shape[0], format="csr") * lam
            else:
                BB_reg = BB

            # Solve joint whitened system
            K = bmat([[A_w_blk, AB_w], [AB_w.T, BB_reg]], format="csr")
            rhs = np.concatenate([b_w_blk, bB])

            sol, info = cg(K, rhs, atol=0.0, rtol=rtol, maxiter=maxiter)
            if info > 0:
                sol, info = minres(K, rhs, rtol=rtol, maxiter=maxiter)

            na = idx.size
            xw_scene = sol[:na]
            beta_scene = sol[na:]
            alpha[idx] = xw_scene / d[idx]
            infos.append(int(info))

            # --- Flux errors via whitened Schur complement (stay in whitened flux space)
            BB_dense = BB_reg.toarray()
            try:
                # X solves: BB * X = AB_w.T   → X: (nB, nA)
                X = np.linalg.solve(BB_dense, AB_w.T.toarray())
            except np.linalg.LinAlgError:
                # fallback to pinv if BB is ill-conditioned
                X = np.linalg.pinv(BB_dense) @ AB_w.T.toarray()

            # Y = AB_w @ X  (nA × nA) — this is dense ndarray by design
            Y = AB_w @ X
            # Schur complement in whitened space; cast dense Y back to sparse before subtract
            S_w = (A_w_blk - sp.csr_matrix(Y)).tocsr()
            # Convert whitened errors to unwhitened: σ(x) = σ(x_w) / d
            err[idx] = self._flux_errors(S_w) / d[idx]

            # ----- save astrometry dx, dy shifts for this scene  < HERE > -----
            x_cen, y_cen = centers[int(sid)]
            Sx, Sy = scales[int(sid)]
            poly_shift_at = AstroCorrect.build_poly_predictor(
                beta_scene, x_cen, y_cen, order, Sx, Sy
            )

            # Evaluate at *input* positions (what you originally convolved/placed)
            pts = np.array([templates[i].position_original for i in idx], float)  # (na, 2)
            dx, dy = poly_shift_at(pts)  # -> (na,), (na,)

            # Record (lazy) shifts; apply later with AstroCorrect.apply_template_shifts(...) if desired
            # sign convention: shift is positive from image to template
            for k, i in enumerate(idx):
                templates[i].to_shift = np.array([dx[k], dy[k]])
            #                templates[i].is_dirty = True

            p = len(cheb_basis(0.0, 0.0, order))
            bx = beta_scene[:p]
            by = beta_scene[p : 2 * p]
            phi0 = cheb_basis(0.0, 0.0, order)
            mean_dx = float(phi0 @ bx)
            mean_dy = float(phi0 @ by)
            logger.debug(
                "Scene %s mean-shift near center ≈ (%.3f, %.3f) px", sid, mean_dx, mean_dy
            )

            betas.append(
                (
                    int(sid),
                    beta_scene,
                    {"center": (x_cen, y_cen), "scale": (Sx, Sy), "order": order},
                )
            )

            logger.debug(
                "Scene %d astrometry basis: center=(%.3f, %.3f) scale=(%.3f, %.3f) order=%d",
                sid,
                x_cen,
                y_cen,
                Sx,
                Sy,
                int(order),
            )
            logger.debug("betas for scene %d: %s", sid, betas[-1][1])
            logger.debug("Scene %d: solved with %d templates, info=%d", sid, len(idx), info)

        return alpha, err, betas, infos

    def solve_scene(self, config: FitConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve independent template scenes separately using CG.

        Templates are partitioned into connected scenes via their bounding
        box overlaps. The normal matrix is whitened once for the full system,
        and each scene is solved on this whitened matrix independently.
        """

        cfg = config or self.config

        # note templates are pruned here
        A, b = self.ata, self.atb

        # regularization: add a small ridge to the diagonal
        reg = cfg.reg
        if reg <= 0:
            reg = 1e-6 * np.median(A.diagonal())
        if reg > 0:
            A = A + eye(A.shape[0], format="csr") * reg

        # clamp diagonal to avoid numerical issues
        d = np.sqrt(np.maximum(A.diagonal(), reg))
        # whiten the normal equations
        # A_w = D^-1 A D^-1, b_w = b / d
        Dinv = diags(1.0 / d, 0, format="csr")
        A_w = Dinv @ A @ Dinv
        b_w = b / d

        # these are the templates we will use for the solve
        templates = self.templates
        idx = [t.col_idx for t in templates]
        is_bright = np.array(
            [(t.flux / t.err > cfg.snr_thresh_astrom) if t.err > 0 else False for t in templates]
        )

        # get scene ids from the normal equations
        # merge small scenes so enough bright sources are present
        if self.scene_ids is not None:
            scene_ids = self.scene_ids  # reuse if present (e.g repeated solves)
        else:
            scene_ids, nscene = build_scene_tree_from_normal(A, b, coupling_thresh=1e-4)
            if cfg.scene_merge_small:
                npoly = n_terms(cfg.astrom_kwargs.get("order", 1))
                minimum_bright = max(cfg.scene_minimum_bright, npoly * 2)
                scene_ids, nscene = merge_small_scenes(
                    scene_ids,
                    templates,
                    is_bright,
                    minimum_bright=minimum_bright,
                )
                summarize_scenes(scene_ids)
                self.scene_ids = scene_ids  # cache for later use

            for i, t in enumerate(templates):
                t.id_scene = int(scene_ids[i])  # assign scene id to each template

        # after you computed `scene_ids` (and before the flux-only branch)
        rtol = cfg.cg_kwargs.get("rtol", 1e-6)
        maxit = cfg.cg_kwargs.get("maxiter", 2000)
        if cfg.fit_astrometry_niter > 0 and cfg.fit_astrometry_joint:
            # x, err are unwhitened fluxes, betas are astrometric shifts
            x, err, betas, infos = self._solve_scenes_with_shifts(
                A_w,
                b_w,
                d,
                scene_ids,
                templates,
                is_bright,
                order=cfg.astrom_kwargs.get("poly", {}).get("order", 1),
                include_y=True,
                ab_from_bright_only=True,
                rtol=rtol,
                maxiter=maxit,
            )
            info = {"cg_info": infos, "nscene": nscene, "betas": betas}
            print("betas: ", getattr(info, "betas", None))
        else:
            x_w, info = solve_scene_cg(A_w, b_w, scene_ids, rtol=rtol, maxiter=maxit)
            x = x_w / d
            err = self._flux_errors(A_w) / d

        if cfg.positivity:
            x = np.maximum(0.0, x)

        x_full = np.zeros(self.n_flux, dtype=float)
        e_full = np.zeros(self.n_flux, dtype=float)
        x_full[idx] = x
        e_full[idx] = err

        self.solution = x_full
        self.solution_err = e_full
        for tmpl, flux, err in zip(self._orig_templates, x_full, e_full):
            tmpl.flux = flux
            tmpl.err = err
            tmpl.is_bright = bool(err > 0 and flux / err > cfg.snr_thresh_astrom)

        return x_full, e_full, {"cg_info": info}

    def residual(self) -> np.ndarray:
        return self.image - self.model_image()

    def quick_flux(self, templates: Optional[List[Template]] = None) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        if templates is None:
            templates = self._orig_templates
        return Templates.quick_flux(templates, self.image)

    def predicted_errors(self, templates: Optional[List[Template]] = None) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        if templates is None:
            templates = self._orig_templates
        return Templates.predicted_errors(templates, self.weights)

    def flux_and_rms(
        self, templates: Optional[List[Template]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return flux estimates and RMS errors for templates.

        Uses existing template fluxes when available; otherwise computes
        quick fluxes and predicted errors for the first ``n_flux`` templates.

        Args:
            templates: Optional list of templates to evaluate. Defaults to
                the original templates supplied to the fitter.

        Returns:
            Tuple ``(flux, rms)`` containing the flux estimates and
            corresponding RMS errors for each template.
        """
        if templates is None:
            templates = self._orig_templates

        if templates and templates[0].flux != 0:
            flux = np.array([t.flux for t in templates[: self.n_flux]])
        else:
            flux = self.quick_flux(templates)[: self.n_flux]

        rms = self.predicted_errors(templates)[: self.n_flux]
        return flux, rms

    def flux_errors(self) -> np.ndarray:
        """Return the 1-sigma flux uncertainties from the last solution."""
        if self.solution_err is None:
            raise ValueError("Solve system first")
        return self.solution_err

    def _flux_errors(self, A: csr_matrix) -> np.ndarray:
        """Return 1-sigma uncertainties for the fitted fluxes.
        This computes the diagonal of ``A`` :sup:`-1` using a SuperLU
        factorization when possible and falls back to a Hutchinson
        trace estimator otherwise.
        """
        eps_pd = 1e-4 * np.median(A.diagonal())
        A = A + eps_pd * eye(A.shape[0], format="csr")  # ensure PD

        # 0. cheap independent-pixel approximation?
        off = A.copy()
        off.setdiag(0)
        covar_power = np.sqrt((off.data**2).sum()) / A.diagonal().sum()
        if covar_power < 1e-3 or not self.config.fit_covariances:
            ibad = np.where(A.diagonal() < eps_pd)[0]
            if ibad.size > 0:
                print(f"Warning: {len(ibad)} pixels have low diagonal values in A")
            return np.sqrt(A.diagonal()) * covar_power
            return 1 / np.sqrt(A.diagonal())
        else:
            return _diag_inv_hutch(A, k=16, rtol=1e-4)

    @classmethod
    def fit(
        cls,
        templates: List[Template],
        image: np.ndarray,
        weights: np.ndarray | None = None,
        config: FitConfig | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method to solve for fluxes and return residuals."""
        fitter = cls(templates, image, weights, config)
        fluxes, _, _ = fitter.solve()
        resid = fitter.residual()
        return fluxes, resid


# ------------------------------------- OBSOLETE BELOW -------------------------------------
# ------------------------------------- OBSOLETE BELOW -------------------------------------


def assemble_scene_system_old(
    comp: list[int],
    templates: List[Template],
    image: np.ndarray,
    weights: np.ndarray,
    basis_vals: list[Optional[np.ndarray]],
    *,
    order: int = 1,
    include_y: bool = True,
    ab_from_bright_only: bool = True,
    tol_zero: float = 0.0,
):
    """Return block system for a single scene including shifts."""

    g2l = {g: i for i, g in enumerate(comp)}
    nA = len(comp)
    bright_in_comp = [g for g in comp if basis_vals[g] is not None]
    has_shift = len(bright_in_comp) >= 2
    p = len(cheb_basis(0.0, 0.0, order)) if has_shift else 0
    nB = p * (2 if include_y else 1)

    AA = lil_matrix((nA, nA))
    AB = lil_matrix((nA, nB))
    BB = np.zeros((nB, nB), dtype=float)
    bA = np.zeros(nA, dtype=float)
    bB = np.zeros(nB, dtype=float)

    for g_i in comp:
        iA = g2l[g_i]
        sl_i = templates[g_i].slices_original
        w = weights[sl_i]
        tt = templates[g_i].data[templates[g_i].slices_cutout]
        img = image[sl_i]
        bA[iA] += float(np.sum(tt * w * img))
        AA[iA, iA] = float(np.sum(tt * w * tt))

    grad_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if has_shift:
        for g in bright_in_comp:
            arr = templates[g].data.astype(float)
            if arr.shape[0] < 2 or arr.shape[1] < 2:
                gy = np.zeros_like(arr)
                gx = np.zeros_like(arr)
            else:
                gy, gx = np.gradient(arr)
            grad_cache[g] = (gx, gy)

    for a, g_i in enumerate(comp):
        iA = g2l[g_i]
        ti = templates[g_i]
        sli = ti.slices_original
        for g_j in comp:
            if g_j <= g_i:
                continue
            tj = templates[g_j]
            inter = SparseFitter._slice_intersection(sli, tj.slices_original)
            if inter is None:
                continue
            sl_i_local = (
                slice(
                    inter[0].start - sli[0].start + ti.slices_cutout[0].start,
                    inter[0].stop - sli[0].start + ti.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start - sli[1].start + ti.slices_cutout[1].start,
                    inter[1].stop - sli[1].start + ti.slices_cutout[1].start,
                ),
            )
            sl_j = tj.slices_original
            sl_j_local = (
                slice(
                    inter[0].start
                    - tmpl_j.slices_original[0].start
                    + tmpl_j.slices_cutout[0].start,
                    inter[0].stop
                    - tmpl_j.slices_original[0].start
                    + tmpl_j.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start
                    - tmpl_j.slices_original[1].start
                    + tmpl_j.slices_cutout[1].start,
                    inter[1].stop
                    - tmpl_j.slices_original[1].start
                    + tmpl_j.slices_cutout[1].start,
                ),
            )
            w = weights[inter]
            Ti = ti.data[sl_i_local]
            Tj = tj.data[sl_j_local]
            gij = float(np.sum(Ti * w * Tj))
            if abs(gij) > tol_zero:
                jA = g2l[g_j]
                AA[iA, jA] = gij
                AA[jA, iA] = gij

            if has_shift:
                Sj = basis_vals[g_j]
                if Sj is not None and (not ab_from_bright_only or basis_vals[g_i] is not None):
                    Gxj, Gyj = grad_cache[g_j]
                    gx = float(np.sum(Ti * w * Gxj[sl_j_local]))
                    AB[iA, 0:p] += gx * Sj
                    if include_y:
                        gy = float(np.sum(Ti * w * Gyj[sl_j_local]))
                        AB[iA, p : 2 * p] += gy * Sj

                Si = basis_vals[g_i]
                Sj = basis_vals[g_j]
                if (Si is not None) and (Sj is not None):
                    Gxi, Gyi = grad_cache[g_i]
                    Gxj, Gyj = grad_cache[g_j]
                    Gxx = float(np.sum(Gxi[sl_i_local] * w * Gxj[sl_j_local]))
                    BB[0:p, 0:p] += Gxx * np.outer(Si, Sj)
                    if include_y:
                        Gyy = float(np.sum(Gyi[sl_i_local] * w * Gyj[sl_j_local]))
                        BB[p : 2 * p, p : 2 * p] += Gyy * np.outer(Si, Sj)

    if has_shift:
        for g_j in comp:
            Sj = basis_vals[g_j]
            if Sj is None:
                continue
            tj = templates[g_j]
            slj = tj.slices_original
            w = weights[slj]
            img = image[slj]
            Gxj, Gyj = grad_cache[g_j]
            bB[0:p] += float(np.sum(Gxj[tj.slices_cutout] * w * img)) * Sj
            if include_y:
                bB[p : 2 * p] += float(np.sum(Gyj[tj.slices_cutout] * w * img)) * Sj

    return AA.tocsr(), AB.tocsr(), csr_matrix(BB), bA, bB


def build_scene_tree_old(
    templates: List[Template],
) -> tuple[np.ndarray, int]:
    """Label independent template groups using Shapely 2.x STRtree.query."""
    from shapely.geometry import box
    from shapely.strtree import STRtree
    from scipy.sparse import coo_matrix, csgraph
    import numpy as np

    def _bbox_to_box(t: Template):
        (ymin, ymax), (xmin, xmax) = t.bbox_original  # closed intervals
        return box(xmin, ymin, xmax, ymax)

    n = len(templates)
    if n == 0:
        return np.zeros(0, dtype=int), 0

    boxes = [_bbox_to_box(t) for t in templates]
    tree = STRtree(boxes)

    # Collect undirected edges (i<j) where boxes intersect
    ii: list[int] = []
    jj: list[int] = []
    for i, gi in enumerate(boxes):
        j_idx = tree.query(gi, predicate="intersects")  # ndarray[int]
        if j_idx.size == 0:
            continue
        sel = j_idx > i
        if np.any(sel):
            js = j_idx[sel].tolist()
            ii.extend([i] * len(js))
            jj.extend(js)

    if not ii:
        # no overlaps → each template is its own scene
        return (np.arange(n, dtype=int) + 1, n)

    adj = coo_matrix((np.ones(len(ii), dtype=np.uint8), (ii, jj)), shape=(n, n))
    adj = adj + adj.T
    nscene, labels = csgraph.connected_components(adj.tocsr(), directed=False)

    return labels.astype(int) + 1, int(nscene)


def merge_small_scenes_old(
    labels: np.ndarray,
    templates: list[Template],
    bright_mask: np.ndarray,
    *,
    order: int = 1,
    minimum_bright: int | None = None,
    max_merge_radius: float = np.inf,  # pixels; keep ∞ for no distance limit
) -> np.ndarray:

    labs = labels.copy()
    while True:
        uniq = np.unique(labs)
        if uniq.size <= 1:
            break

        # per-scene indices
        idx_lists = [np.where(labs == s)[0] for s in uniq]
        sizes = np.array([idx.size for idx in idx_lists])
        if sizes.min() == 0:
            # prune empty labels quickly
            nonempty = sizes > 0
            uniq = uniq[nonempty]
            idx_lists = [idx_lists[i] for i in np.where(nonempty)[0]]
            sizes = sizes[nonempty]
            if uniq.size <= 1:
                break

        # bright counts and centroids
        nbright = np.array([int(np.count_nonzero(bright_mask[idx])) for idx in idx_lists])
        cx = np.array(
            [np.mean([templates[i].position_original[0] for i in idx]) for idx in idx_lists]
        )
        cy = np.array(
            [np.mean([templates[i].position_original[1] for i in idx]) for idx in idx_lists]
        )

        # which scenes are under threshold?
        under = np.where(nbright < minimum_bright)[0]
        if under.size == 0:
            break

        # pick the *most* deficient scene to merge first (fewest brights)
        u = under[np.argmin(nbright[under])]

        # distances from that scene to all others
        dx = cx - cx[u]
        dy = cy - cy[u]
        d2 = dx * dx + dy * dy
        d2[u] = np.inf  # ignore self
        v = int(np.argmin(d2))

        if not np.isfinite(d2[v]) or d2[v] > max_merge_radius * max_merge_radius:
            # nothing reasonable to merge with; stop
            break

        # relabel: move all members of scene uniq[u] into scene uniq[v]
        labs[labs == uniq[u]] = uniq[v]

    # compact labels to 0..k-1
    unique_labels, new_labs = np.unique(labs, return_inverse=True)

    return new_labs

    def solve_scene_shifts(
        self,
        order: int = 1,
        *,
        include_y: bool = True,
        ab_from_bright_only: bool = True,
        rtol: float = 1e-6,
        maxiter: int = 2000,
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], dict]:
        """Solve scenes with additional polynomial shift terms.

        Returns
        -------
        flux : ndarray
            Best-fit fluxes for the original templates.
        betas : list[tuple[int, ndarray]]
            Per-scene shift coefficients.
        info : dict
            Solver diagnostics including CG convergence flags.
        """

        labels, nscene = build_scene_tree(self.templates)
        summarize_scenes(labels)
        basis_vals = make_basis_per_scene(self.templates, labels, self.bright_mask, order=order)

        alpha = np.zeros(len(self.templates), dtype=float)
        betas: list[tuple[int, np.ndarray]] = []
        infos: list[int] = []

        for sid in range(labels.max() + 1):
            comp = np.where(labels == sid)[0].tolist()
            if not comp:
                infos.append(0)
                continue
            AA, AB, BB, bA, bB = assemble_scene_system(
                comp,
                self.templates,
                self.image,
                self.weights,
                basis_vals,
                order=order,
                include_y=include_y,
                ab_from_bright_only=ab_from_bright_only,
            )
            if AB.shape[1] == 0:
                A = AA.tocsr()
                b = bA
                D = A.diagonal().copy()
                D[D == 0] = 1.0
                M = LinearOperator(A.shape, matvec=lambda v, D=D: v / D)
                sol, info = cg(A, b, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
                alpha_comp = sol
                beta_comp = np.zeros(0, dtype=float)
            else:
                K = bmat([[AA, AB], [AB.T, BB]], format="csr")
                rhs = np.concatenate([bA, bB])
                D = K.diagonal().copy()
                D[D == 0] = 1.0
                M = LinearOperator(K.shape, matvec=lambda v, D=D: v / D)
                sol, info = cg(K, rhs, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
                if info > 0:
                    sol, info = minres(K, rhs, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
                na = len(comp)
                alpha_comp = sol[:na]
                beta_comp = sol[na:]
            alpha[np.array(comp)] = alpha_comp
            betas.append((sid, beta_comp))
            infos.append(int(info))

        if self.config.positivity:
            alpha = np.maximum(0.0, alpha)

        x_full = np.zeros(self.n_flux, dtype=float)
        idx = [t.col_idx for t in self.templates]
        x_full[idx] = alpha
        e_full = np.zeros_like(x_full, dtype=float)
        self.solution = x_full
        for tmpl, flux in zip(self._orig_templates, x_full):
            tmpl.flux = flux

        info = {"nscene": nscene, "cg_info": infos, "betas": betas}
        return x_full, e_full, info


def build_normal_matrix(self) -> None:
    """Construct normal matrix using :class:`Template` objects."""
    # Compute weighted norms for all templates first

    norms = [self._weighted_norm(t) for t in self.templates]
    tol = 1e-8 * max(norms)

    keep = [i for i, n in enumerate(norms) if n > tol]
    print(f"Dropped {len(self.templates)-len(keep)} templates with low norm.")

    self.templates = [self.templates[i] for i in keep]
    norms = [norms[i] for i in keep]

    n = len(self.templates)
    ata = lil_matrix((n, n))
    atb = np.zeros(n)
    for i, tmpl_i in enumerate(tqdm(self.templates, total=n, desc="Building Normal matrix")):

        sl_i = tmpl_i.slices_original
        data_i = tmpl_i.data[tmpl_i.slices_cutout]
        w_i = self.weights[sl_i]
        img_i = self.image[sl_i]
        atb[i] = np.sum(data_i * w_i * img_i)
        ata[i, i] = norms[i]

        for j in range(i + 1, n):
            tmpl_j = self.templates[j]
            inter = self._slice_intersection(sl_i, tmpl_j.slices_original)
            if inter is None:
                continue
            w = self.weights[inter]
            sl_i_local = (
                slice(
                    inter[0].start - sl_i[0].start + tmpl_i.slices_cutout[0].start,
                    inter[0].stop - sl_i[0].start + tmpl_i.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start - sl_i[1].start + tmpl_i.slices_cutout[1].start,
                    inter[1].stop - sl_i[1].start + tmpl_i.slices_cutout[1].start,
                ),
            )
            sl_j_local = (
                slice(
                    inter[0].start
                    - tmpl_j.slices_original[0].start
                    + tmpl_j.slices_cutout[0].start,
                    inter[0].stop
                    - tmpl_j.slices_original[0].start
                    + tmpl_j.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start
                    - tmpl_j.slices_original[1].start
                    + tmpl_j.slices_cutout[1].start,
                    inter[1].stop
                    - tmpl_j.slices_original[1].start
                    + tmpl_j.slices_cutout[1].start,
                ),
            )
            arr_i = tmpl_i.data[sl_i_local]
            arr_j = tmpl_j.data[sl_j_local]
            val = np.sum(arr_i * arr_j * w)
            #                if val == 0: # cant prune < tol, because messes up global astrometry fit
            #                    continue
            ata[i, j] = val
            ata[j, i] = val

    self._ata = ata.tocsr()
    self._atb = atb
