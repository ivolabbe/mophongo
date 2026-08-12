from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import numpy as np
import scipy.sparse as sp

from .templates import Template, Templates
from .fit import FitConfig
from .scene_fitter import SceneFitter
from .astrometry import cheb_basis, AstroCorrect
from scipy.sparse.csgraph import connected_components
from .templates import _slices_from_bbox

logger = logging.getLogger(__name__)


def _bbox_union(templates: Sequence[Template]) -> Tuple[int, int, int, int]:
    """Return the union bounding box of ``templates``.

    Parameters
    ----------
    templates
        Sequence of :class:`~mophongo.templates.Template` objects.
    """
    y0 = min(t.bbox[0] for t in templates)
    y1 = max(t.bbox[1] for t in templates)
    x0 = min(t.bbox[2] for t in templates)
    x1 = max(t.bbox[3] for t in templates)
    return y0, y1, x0, x1


def _bbox_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _astrom_isolation_mask(A: sp.spmatrix, b: np.ndarray, thresh: float) -> np.ndarray:
    """Return bool mask: True where source contributes >= thresh of its own local flux.

    dominance[i] = (alpha0[i] * ATA[i,i]) /
                   (alpha0[i] * ATA[i,i] + sum_j alpha0[j] * |ATA[i,j]|)

    ATA[i,j] is the integral of T_i * T_j over the image, so alpha0[j] * ATA[i,j]
    is exactly the neighbor flux falling within source i's template footprint.
    """
    n = A.shape[0]
    diag = np.maximum(A.diagonal(), 1e-12)
    alpha0 = np.abs(b) / diag
    Au = sp.triu(A, k=1).tocoo()
    if Au.nnz == 0:
        return np.ones(n, dtype=bool)
    i, j, aij = Au.row, Au.col, np.abs(Au.data)
    neighbor_flux = np.zeros(n)
    np.add.at(neighbor_flux, i, alpha0[j] * aij)
    np.add.at(neighbor_flux, j, alpha0[i] * aij)
    self_flux = alpha0 * diag
    dominance = self_flux / np.maximum(self_flux + neighbor_flux, 1e-12)
    return dominance >= thresh


def build_scene_tree_from_normal(
    ATA: sp.spmatrix,
    ATb: np.ndarray,
    *,
    coupling_thresh: float = 0.01,  # 3% leakage threshold
    max_size: int | None = None,
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
    max_size : int, optional
        Soft cap on templates per component. A component over the cap is split
        by raising the threshold *locally* — bisecting over that component's
        own edge scores until its pieces fit — so strong couplings elsewhere
        in the field are never cut on behalf of one crowded region. The final
        local threshold is logged: it is the cross-scene leakage accepted
        inside that region. None (default) disables the cap.
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

    if max_size is not None and np.bincount(labels0).max() > int(max_size):
        sizes = np.bincount(labels0)
        next_label = int(labels0.max()) + 1
        t_hi = 0.0
        n_split = 0
        for comp in np.nonzero(sizes > int(max_size))[0]:
            # Kept edges live entirely inside one component, so testing one
            # endpoint suffices.
            em = mask & (labels0[i] == comp)
            ei, ej, es = i[em], j[em], score[em]
            memb = np.nonzero(labels0 == comp)[0]
            loc = np.full(n, -1, dtype=int)
            loc[memb] = np.arange(memb.size)
            li, lj = loc[ei], loc[ej]
            # Bisect over this component's own edge scores for the smallest
            # local threshold whose pieces all fit; dropping every edge leaves
            # singletons, so a solution always exists.
            cands = np.unique(es)
            lo, hi = 0, cands.size
            best_labs, best_t = None, 0.0
            while lo < hi:
                mid = (lo + hi) // 2
                keep = es >= cands[mid]
                data_m = np.ones(keep.sum() * 2, dtype=np.uint8)
                adj_m = sp.coo_matrix(
                    (
                        data_m,
                        (
                            np.concatenate([li[keep], lj[keep]]),
                            np.concatenate([lj[keep], li[keep]]),
                        ),
                    ),
                    shape=(memb.size, memb.size),
                ).tocsr()
                _, labs_m = connected_components(adj_m, directed=False)
                if np.bincount(labs_m).max() <= int(max_size):
                    hi = mid
                    best_labs, best_t = labs_m, float(cands[mid])
                else:
                    lo = mid + 1
            if best_labs is None:
                best_labs = np.arange(memb.size)  # drop all edges
                best_t = float(cands[-1]) if cands.size else 0.0
            labels0 = labels0.copy()
            labels0[memb] = next_label + best_labs
            next_label += int(best_labs.max()) + 1
            t_hi = max(t_hi, best_t)
            n_split += 1
        labels0 = np.unique(labels0, return_inverse=True)[1]
        nscene = int(labels0.max()) + 1
        # pre-merge component count: confusing at INFO (merge_small_scenes
        # absorbs most of these); only the final scene summary is reported
        logger.debug(
            "split %d component(s) over max_size=%d locally: threshold up to "
            "%.3g (floor %.3g), max component %d, %d pre-merge components",
            n_split,
            int(max_size),
            t_hi,
            float(coupling_thresh),
            int(np.bincount(labels0).max()),
            nscene,
        )

    return (labels0 if return_0_based else labels0 + 1), int(nscene)


from shapely.geometry import Point
from shapely.strtree import STRtree


def merge_small_scenes(
    labels: np.ndarray,
    templates: list[Template],
    bright_mask: np.ndarray,
    *,
    order: int = 1,
    minimum_bright: int = 10,
    max_merge_radius: float = np.inf,  # pixels
    max_iter: int = 64,
) -> tuple[np.ndarray, int]:
    """
    Merge scenes below the bright threshold into their nearest scene.
    Uses Shapely 2.x STRtree.query_nearest (bulk) and unions all pairs per round.
    Returns (1-based labels, n_scenes).
    """

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


def make_scene_basis(
    templates: List[Template],
    bright: np.ndarray,
    order: int = 1,
) -> tuple[
    List[Optional[np.ndarray]],
    tuple[float, float],  # center (x0, y0)
    tuple[float, float],  # scales (Sx, Sy)
]:
    """
    Build per-template polynomial bases for a *single* scene.

    Parameters
    ----------
    templates : list[Template]
        Templates belonging to one scene, in scene-local order.
    bright : (n,) bool array
        Bright mask aligned to `templates`. Only bright members get a basis.
    order : int
        Chebyshev polynomial order.

    Returns
    -------
    basis : list[Optional[np.ndarray]]
        For each template, either a basis vector (bright) or None (faint).
    center : (x0, y0)
        Scene center used for normalization.
    scales : (Sx, Sy)
        Half-range scales used to map positions roughly to [-1, 1].
    """
    bright = np.asarray(bright, dtype=bool)
    n = len(templates)
    basis: List[Optional[np.ndarray]] = [None] * n
    if n == 0:
        return basis, (0.0, 0.0), (1.0, 1.0)

    xs = np.array([t.position_original[0] for t in templates], dtype=float)
    ys = np.array([t.position_original[1] for t in templates], dtype=float)

    use = np.nonzero(bright)[0]
    if use.size == 0:
        # Fall back to all members if no brights in the scene
        use = np.arange(n)

    x0 = float(xs[use].mean())
    y0 = float(ys[use].mean())

    # Half-range scaling with a small pad, guard for degeneracy
    def _half_range(a):
        if a.size == 0:
            return 1.0
        return 0.5 * float(a.max() - a.min())

    Sx = max(1.0, 1.05 * _half_range(xs[use]))
    Sy = max(1.0, 1.05 * _half_range(ys[use]))

    for i in range(n):
        if not bright[i]:
            continue
        u = (xs[i] - x0) / Sx
        v = (ys[i] - y0) / Sy
        basis[i] = cheb_basis(u, v, order)

    return basis, (x0, y0), (Sx, Sy)


import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Tuple
from .templates import Template
from .astrometry import cheb_basis


def assemble_scene_system_AB(
    templates: List[Template],
    image: np.ndarray,
    weights: np.ndarray,
    basis_vals: List[Optional[np.ndarray]],
    *,
    alpha0: np.ndarray | float | None,  # per-template flux (unwhitened), scene-local
    order: int = 1,
    include_y: bool = True,
    leverage_cap: float | None = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Build the (A,B) coupling blocks and beta RHS for a *single scene*.

    These are the exact normal-equation blocks of the linearized joint
    design (``docs/fitting.md``). Writing the shift field as
    :math:`\\delta x_i = S_i \\cdot \\beta^x`, the model is

    .. math::
        m = \\sum_i \\alpha_i
            [T_i - (S_i \\beta^x) \\partial_x T_i - (S_i \\beta^y) \\partial_y T_i]

    so the design columns are :math:`A_j = T_j` for the fluxes and
    :math:`B_k = -\\sum_i \\alpha_i S_{ik} \\nabla T_i` for the shift
    coefficients, and the blocks are :math:`A^\\top W B`,
    :math:`B^\\top W B` and :math:`B^\\top W d`.

    Each :math:`B_k` sums over *every* bright anchor in the scene, so the
    blocks retain the cross-template terms: a flux row couples to gradients
    of its neighbours, distinct anchors couple to each other, and the x-y
    block is populated. Accumulating only each template's own products
    instead is exact for isolated anchors but, in a blend, mistakes the
    residual dipole of an overlapping neighbour for a shift.

    Parameters
    ----------
    templates
        Templates belonging to this scene (scene-local order).
    image, weights
        Full image and weight arrays (same shape); slicing is done per-template.
    basis_vals
        List aligned to `templates`; element i is either a basis vector (bright)
        or None (faint) for template i.
    alpha0
        Scene-local unwhitened flux seed(s). Can be:
          - array-like of shape (n_scene,)
          - scalar (broadcast to all)
          - None (treated as zeros)
    order
        Chebyshev polynomial order for the shift basis (only used for nB sizing).
    include_y
        If True, include ∂/∂y block (else only ∂/∂x).
    leverage_cap
        Quantile in (0, 1] at which to cap each anchor's leverage, or None
        to leave the weights alone.

        An anchor's information is ``I_i = a_i^2 <Gx,w,Gx>``, so leverage
        grows as flux squared and one bright source can carry a scene on its
        own. That is a problem when the source is extended with an asymmetric
        colour gradient: its residual is a dipole aligned with the template
        gradient, formally indistinguishable from a shift. Capping at
        ``I_cap = quantile(I, leverage_cap)`` scales that anchor down by
        ``wl_i = min(1, I_cap / I_i)`` while leaving the shift it implies
        untouched -- it still says what it says, it just counts less.

        The cap is a weight on the shift equations, not a rescaling of the
        anchor, so it enters the blocks at two different powers. ``AB`` and
        ``bB`` are linear in a derivative column and take ``wl_i``; ``BB`` is
        quadratic and takes ``sqrt(wl_i wl_j)``, which leaves ``wl_i`` on the
        diagonal. That is what keeps the implied shift
        ``dx_i = -bB_i / BB_ii`` invariant under capping, and it reduces to
        scaling that source's pixel weights exactly when nothing overlaps.
        The flux block is not touched, so photometry is unchanged.

        Note what this does *not* do: it cannot tell which anchor is wrong,
        so it clips the brightest, which are often the best anchors, and it
        does nothing in a scene whose offender is the only bright member.
        Cross-anchor robustness is the fix for that case (see TODO.md).

    Returns
    -------
    AB : csr_matrix (nA, nB)
    BB : csr_matrix (nB, nB)
    bB : ndarray (nB,)
    """
    nA = len(templates)
    if nA == 0:
        return sp.csr_matrix((0, 0)), sp.csr_matrix((0, 0)), np.zeros(0, float)

    # Determine if the scene has enough bright members to solve for shifts
    bright_idx = [i for i, S in enumerate(basis_vals) if S is not None]
    has_shift = len(bright_idx) >= 2
    if not has_shift:
        return sp.csr_matrix((nA, 0)), sp.csr_matrix((0, 0)), np.zeros(0, float)

    p = len(cheb_basis(0.0, 0.0, order))
    nB = p * (2 if include_y else 1)

    bB = np.zeros(nB, dtype=float)

    # Normalize/validate alpha0 → scene-local array
    if alpha0 is None:
        a = np.zeros(nA, dtype=float)
    elif np.isscalar(alpha0):
        a = np.full(nA, float(alpha0), dtype=float)
    else:
        a = np.asarray(alpha0, dtype=float)
        if a.shape != (nA,):
            raise ValueError(f"alpha0 must have shape ({nA},), got {a.shape}")

    # Cache gradients per local index
    grad_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _gx_gy_for(i_local: int) -> tuple[np.ndarray, np.ndarray]:
        if i_local not in grad_cache:
            arr = templates[i_local].data.astype(float)
            if arr.shape[0] < 2 or arr.shape[1] < 2:
                gy = np.zeros_like(arr)
                gx = np.zeros_like(arr)
            else:
                gy, gx = np.gradient(arr)  # gy=d/dy, gx=d/dx
            grad_cache[i_local] = (gx, gy)
        return grad_cache[i_local]

    # Per-anchor leverage, known before any solve: it depends on the
    # template, its flux seed and the weight map, not on the residual. So
    # the cap is a single pass here, unlike an IRLS scheme which has to see
    # the fit first.
    lev_w = np.ones(nA, dtype=float)
    if leverage_cap is not None and 0 < float(leverage_cap) <= 1:
        info = np.zeros(nA, dtype=float)
        for row in bright_idx:
            ti = templates[row]
            w = weights[ti.slices_original]
            Gx, Gy = _gx_gy_for(row)
            gxc = Gx[ti.slices_cutout]
            iso = float(np.sum(gxc * w * gxc))
            if include_y:
                gyc = Gy[ti.slices_cutout]
                iso = 0.5 * (iso + float(np.sum(gyc * w * gyc)))
            info[row] = a[row] ** 2 * iso
        pos = info[info > 0]
        if pos.size:
            cap = float(np.quantile(pos, float(leverage_cap)))
            if cap > 0:
                over = info > cap
                lev_w[over] = cap / info[over]
                if over.any():
                    logger.debug(
                        "[scenes] leverage cap q=%.2f: %d of %d anchor(s) "
                        "clipped, min weight %.3g",
                        float(leverage_cap), int(over.sum()), len(bright_idx),
                        float(lev_w[over].min()),
                    )

    # Scene-wide derivative columns B_k = sum_i (-a_i S_ik) grad(T_i),
    # accumulated over the union footprint of the bright anchors. Faint
    # members contribute no column of their own but still couple to these
    # through their AB row. Two weightings are carried so the leverage cap
    # keeps its per-anchor meaning (see `leverage_cap` above): `Bq` holds
    # sqrt(wl_i) and builds BB, `Bl` holds wl_i and builds AB and bB. They
    # are the same array unless some anchor is actually capped.
    y0 = min(templates[i].slices_original[0].start for i in bright_idx)
    y1 = max(templates[i].slices_original[0].stop for i in bright_idx)
    x0 = min(templates[i].slices_original[1].start for i in bright_idx)
    x1 = max(templates[i].slices_original[1].stop for i in bright_idx)
    logger.debug(
        "[scenes] shift columns: %d x (%d, %d) buffer, %.1f MB",
        nB, y1 - y0, x1 - x0, nB * (y1 - y0) * (x1 - x0) * 8 / 1e6,
    )

    Bq = np.zeros((nB, y1 - y0, x1 - x0), dtype=float)
    capped = bool(np.any(lev_w[bright_idx] < 1.0))
    Bl = np.zeros_like(Bq) if capped else Bq

    for i in bright_idx:
        ti = templates[i]
        Si = basis_vals[i]
        Gx, Gy = _gx_gy_for(i)
        gx = Gx[ti.slices_cutout]
        gy = Gy[ti.slices_cutout] if include_y else None
        sl = (
            slice(ti.slices_original[0].start - y0, ti.slices_original[0].stop - y0),
            slice(ti.slices_original[1].start - x0, ti.slices_original[1].stop - x0),
        )
        cq = -float(a[i]) * np.sqrt(lev_w[i])
        cl = -float(a[i]) * lev_w[i]
        for k in range(p):
            Bq[k][sl] += (cq * Si[k]) * gx
            if include_y:
                Bq[p + k][sl] += (cq * Si[k]) * gy
        if capped:
            for k in range(p):
                Bl[k][sl] += (cl * Si[k]) * gx
                if include_y:
                    Bl[p + k][sl] += (cl * Si[k]) * gy

    wbuf = weights[y0:y1, x0:x1]
    dbuf = image[y0:y1, x0:x1]

    # BB = B^T W B (cross-anchor and x-y terms included), bB = B^T W d
    BB = np.empty((nB, nB), dtype=float)
    for k in range(nB):
        bw = Bq[k] * wbuf
        for m in range(k, nB):
            BB[k, m] = BB[m, k] = float(np.sum(bw * Bq[m]))
        bB[k] = float(np.sum(Bl[k] * wbuf * dbuf))

    # AB = A^T W B, one row per template: a faint member's row is simply its
    # overlap with the anchors' derivative columns.
    AB = np.zeros((nA, nB), dtype=float)
    for row, tj in enumerate(templates):
        so = tj.slices_original
        ya, yb = max(so[0].start, y0), min(so[0].stop, y1)
        xa, xb = max(so[1].start, x0), min(so[1].stop, x1)
        if ya >= yb or xa >= xb:
            continue  # no overlap with any anchor
        sc = tj.slices_cutout
        tsl = (
            slice(sc[0].start + ya - so[0].start, sc[0].start + yb - so[0].start),
            slice(sc[1].start + xa - so[1].start, sc[1].start + xb - so[1].start),
        )
        tw = tj.data[tsl] * weights[ya:yb, xa:xb]
        bsl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        for k in range(nB):
            AB[row, k] = float(np.sum(tw * Bl[k][bsl]))

    return sp.csr_matrix(AB), sp.csr_matrix(BB), bB


def generate_scenes(
    templates: Sequence[Template],
    image: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    coupling_thresh: float = 0.01,
    max_size: int | None = None,
    snr_thresh_astrom: float = 7.0,
    isolation_thresh: float = 0.0,
    minimum_bright: int | None = None,
    max_merge_radius: float = np.inf,
    exclude_stars: bool = False,
    isolate_saturated: bool = True,
) -> tuple[List["Scene"], np.ndarray]:
    """
    Partition templates into independent Scenes using normal-equation couplings.

    Steps:
      1) build (ATA, ATb) from templates, image, weight
      2) build_scene_tree_from_normal(ATA, ATb, coupling_thresh)
      3) merge_small_scenes(labels, templates, bright_mask, order, max_merge_radius)
      4) create Scene objects with:
           - subset of templates
           - per-scene ATA, ATb blocks
           - links to image, weight

    Parameters
    ----------
    templates : Sequence[Template]
        Templates to partition.
    image : np.ndarray
        Fit image.
    weight : np.ndarray, optional
        Inverse-variance weights; ``None`` means unit weights.
    coupling_thresh : float
        Passed to :func:`build_scene_tree_from_normal`. The pipeline passes
        ``FitConfig.scene_coupling_thresh`` (default ``1e-3``).
    max_size : int, optional
        Soft per-scene template cap (pipeline passes
        ``FitConfig.scene_max_size``, default 800).
    snr_thresh_astrom : float
        Bright-anchor cut on the SNR proxy ``b_i / sqrt(A_ii)``.
    isolation_thresh : float
        If positive, a template only counts as a bright anchor when its own
        flux dominance within its footprint (self flux over self plus
        neighbor flux, from the full-field normal matrix) meets this
        fraction.
    minimum_bright : int, optional
        Minimum bright anchors per scene, forwarded to
        :func:`merge_small_scenes`. Pass an integer (the pipeline passes
        ``FitConfig.scene_minimum_bright``): the ``None`` default is
        forwarded unchanged and fails inside ``merge_small_scenes``.
    max_merge_radius : float
        Merge radius in pixels, forwarded to :func:`merge_small_scenes`.
    exclude_stars : bool
        Remove templates with ``is_star`` set from the bright-anchor mask.
    isolate_saturated : bool
        Move saturated/repaired templates into singleton scenes. Their PSF
        wings extend far beyond their segment and would corrupt the flux
        solution of every neighbor caught in the same coupling graph.

    Returns
    -------
    scenes : list[Scene]
        Scene objects with per-scene A/b attached as attributes (scene.A, scene.b).
    labels : ndarray (n_templates,)
        1-based scene labels for each template (after merge).
    """
    import numpy as np
    import scipy.sparse as sp
    from .scene_fitter import build_normal as build_normal_tree

    if weight is None:
        weight = np.ones_like(image, dtype=np.float32)

    # 1) Normal matrix from templates
    ATA, ATb, _ = build_normal_tree(list(templates), image, weight)  # csr, (n,), STRtree

    # 1b) Saturated templates are held out of the partitioning entirely.
    # A saturated star's wings couple to everything under them, so leaving
    # it in the graph glues its neighbours into one huge scene and then the
    # star is pulled back out again, leaving that scene shaped by a member
    # it no longer has. Partition the rest on its own, and give the star's
    # fragments their own scene afterwards.
    sat_mask = (
        np.asarray([bool(getattr(t, "is_saturated", False)) for t in templates], dtype=bool)
        if isolate_saturated
        else np.zeros(len(templates), dtype=bool)
    )
    keep = np.where(~sat_mask)[0]

    # 2) Initial scene labels from normal-equation couplings
    ATA_k = ATA[keep[:, None], keep].tocsr() if sat_mask.any() else ATA
    ATb_k = ATb[keep] if sat_mask.any() else ATb
    labels0_k, _ = build_scene_tree_from_normal(
        ATA_k, ATb_k, coupling_thresh=coupling_thresh, max_size=max_size,
        return_0_based=False,
    )

    # 3) Merge scenes that are too small in terms of "bright" members
    #    SNR proxy: snr_i ≈ b_i / sqrt(diag(A)_i)
    d = np.asarray(ATA.diagonal(), dtype=float)
    snr_proxy = np.divide(
        ATb, np.sqrt(np.maximum(d, 1e-12)), out=np.zeros_like(ATb, dtype=float), where=d > 0
    )
    bright_mask = np.asarray(snr_proxy > float(snr_thresh_astrom), dtype=bool)
    if exclude_stars:
        bright_mask &= ~np.array([t.is_star for t in templates], dtype=bool)
    if isolation_thresh > 0:
        # Count only isolated sources toward minimum_bright, so merged scenes
        # are guaranteed enough usable astrometric anchors. The full-field
        # normal matrix makes this dominance measure stricter (more honest)
        # than the per-scene one at solve time: out-of-scene neighbours still
        # count against a source here.
        bright_mask &= _astrom_isolation_mask(ATA, ATb, float(isolation_thresh))

    labels_k, nscene = merge_small_scenes(
        labels0_k,
        [templates[i] for i in keep],
        bright_mask[keep],
        minimum_bright=minimum_bright,
        max_merge_radius=max_merge_radius,
    )
    labels = np.zeros(len(templates), dtype=labels_k.dtype)
    labels[keep] = labels_k

    # 3b) Scenes for the saturated templates held out above. Templates
    # sharing a ``sat_group`` id (the star's core segment id from
    # FLAG_SATURATED_*) are the fragments of ONE star and go into ONE scene
    # together, fit jointly against the repaired image. Ungrouped saturated
    # templates (legacy 0/1 flags, or group id 1 which is indistinguishable
    # from a legacy flag) each get their own scene. These scenes never see
    # merge_small_scenes, so they are exempt from minimum_bright by
    # construction.
    if sat_mask.any():
        next_label = int(labels.max()) + 1
        group_label: dict[int, int] = {}
        n_alone = 0
        for i in np.where(sat_mask)[0]:
            group = int(getattr(templates[i], "sat_group", 0) or 0)
            if group > 1:
                if group not in group_label:
                    group_label[group] = next_label
                    next_label += 1
                labels[i] = group_label[group]
            else:
                labels[i] = next_label
                next_label += 1
                n_alone += 1
        logger.info(
            "held %d saturated template(s) out of the partitioning: "
            "%d star scene(s) + %d ungrouped",
            int(sat_mask.sum()), len(group_label), n_alone,
        )

    # 4) Instantiate per-scene objects with sub-blocks of ATA/ATb and links to data
    scenes: List[Scene] = []
    # labels are 1-based; build index lists
    for sid in range(1, labels.max() + 1):
        idx = np.where(labels == sid)[0]
        if idx.size == 0:
            continue

        # subset
        ts = [templates[i] for i in idx]
        for t in ts:
            t.id_scene = int(sid)
        A_s = ATA[idx[:, None], idx].tocsr()
        b_s = ATb[idx]

        scn = Scene(
            id=int(sid),
            templates=ts,
            fitter=SceneFitter(),  # minimal stateless fitter
            bbox=_bbox_union(ts),
            image=image,
            weights=weight,
            config=FitConfig(),  # default; caller can override later
        )

        # attach per-scene normal blocks
        scn.A = A_s  # flux block (csr_matrix)
        scn.b = b_s  # rhs (ndarray)
        scn.is_bright = bright_mask[idx]

        scenes.append(scn)

    if scenes:
        sizes = np.array([len(s.templates) for s in scenes])
        logger.info(
            "%d scenes for %d templates: sizes %d-%d (median %d), "
            "%d scene(s) without bright members",
            len(scenes), int(sizes.sum()), int(sizes.min()), int(sizes.max()),
            int(np.median(sizes)), int(sum(not s.is_bright.any() for s in scenes)),
        )
    return scenes, labels


@dataclass
class Scene:
    """Container for templates belonging to a single scene.

    Attributes
    ----------
    id : int
        1-based scene label.
    templates : list[Template]
        Scene members, in scene-local order.
    fitter : SceneFitter
        Stateless solver instance.
    bbox : tuple[int, int, int, int], optional
        Union bounding box ``(y0, y1, x0, x1)`` of the member templates.
    image, weights : np.ndarray, optional
        Full-frame band image and inverse-variance weights (sliced per
        template).
    config : FitConfig, optional
        Per-scene fit configuration.
    shift_basis : list, optional
        ``[basis, (x0, y0), (Sx, Sy)]`` stored by :meth:`solve` for shift
        evaluation.
    flux, err : np.ndarray, optional
        Declared but never filled by :meth:`solve`: per-source results are
        written onto ``solution`` and onto each template.
    shifts : np.ndarray, optional
        Fitted Chebyshev coefficients after a joint solve.
    is_bright : np.ndarray, optional
        Per-template bright-anchor mask.
    solution : SimpleNamespace, optional
        Full solver result (``flux``, ``err``, ``shifts``, ``info``).
    A, b, tree
        Scene-local normal block (``csr_matrix``), right-hand side
        (``ndarray``), and spatial index (``STRtree``); rebuilt from the
        current band by :meth:`solve` when absent.
    """

    id: int
    templates: List[Template]
    fitter: SceneFitter
    bbox: Tuple[int, int, int, int] | None = None
    image: np.ndarray | None = None
    weights: np.ndarray | None = None
    config: FitConfig | None = None
    shift_basis: List | None = None
    flux: np.ndarray | None = None
    err: np.ndarray | None = None
    shifts: np.ndarray | None = None
    is_bright: np.ndarray | None = None  # per-template
    #    info: int | None = None
    solution: SimpleNamespace | None = None
    # store per-scene normal blocks (scene-local ordering)
    A: sp.csr_matrix | None = None
    b: np.ndarray | None = None
    tree: STRtree | None = None  # STRtree over templates in this scene
    # astrometric-refinement bookkeeping, written by the caller driving the
    # passes (Pipeline.run): scenes converge at their own rate, so each one
    # records the increment of its last pass, how many passes it took, and
    # whether it got under astrom_shift_tol before the budget ran out.
    astrom_step: float | None = None
    astrom_niter: int = 0
    astrom_converged: bool | None = None

    def __post_init__(self) -> None:
        pass

    def set_band(
        self,
        image: np.ndarray,
        weight: np.ndarray | None = None,
        psf: np.ndarray | None = None,
        config: Optional[object] = None,
    ) -> None:
        """Cache per-band data for this scene.

        ``psf`` is accepted but currently unused. ``weight=None`` means unit
        weights; ``config``, if given, replaces the scene's fit
        configuration.
        """
        # cache I/O for this band
        self.image = image
        self.weights = np.ones_like(image, dtype=np.float32) if weight is None else weight
        if config is not None:
            self.config = config

    def solve(
        self,
        *,
        config: FitConfig | None = None,
        apply_shifts: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Solve the scene and return ``(flux, err, shifts, info)``.

        Rebuilds ``A``/``b`` from the current band if needed, recomputes the
        bright mask (SNR proxy above ``config.snr_thresh_astrom``, isolation
        above ``config.astrom_isolation_thresh``, optional star exclusion),
        then either solves flux-only (when ``config.fit_astrometry_joint``
        is false or ``fit_astrometry_niter <= 0``) or the joint flux+shift
        system. Results are stored on the scene (``solution``, ``shifts``)
        and on each template (``tmpl.flux``, ``tmpl.err``,
        ``tmpl.is_bright``).

        After a joint solve the fitted shift field is always evaluated at
        each template position, scaled by ``config.astrom_damping``, and
        stored on ``tmpl.to_shift``; ``apply_shifts=True`` additionally
        resamples the templates
        (:meth:`~mophongo.templates.Templates.apply_template_shifts`) and
        clears ``A``/``b`` so the next pass rebuilds them against the
        shifted templates. Scenes with fewer than two bright members fall
        back to flux-only and leave their templates unshifted (logged as a
        warning).
        """
        cfg = config or self.config or FitConfig()
        if self.image is None or self.weights is None:
            raise RuntimeError(
                "Scene image/weights not set. Call set_band() or generate_scenes()."
            )

        # ensure flux block is available or rebuild
        if self.A is None or self.b is None:
            # build normal from current band
            from .scene_fitter import build_normal

            self.A, self.b, self.tree = build_normal(self.templates, self.image, self.weights)

        A, b = self.A, self.b

        # bright mask: SNR cut + isolation cut (+ optional star exclusion)
        d = np.asarray(A.diagonal(), dtype=float)
        snr_proxy = np.divide(
            b, np.sqrt(np.maximum(d, 1e-12)), out=np.zeros_like(b, dtype=float), where=d > 0
        )
        isolated = _astrom_isolation_mask(A, b, float(cfg.astrom_isolation_thresh))
        self.is_bright = (snr_proxy > float(cfg.snr_thresh_astrom)) & isolated
        if getattr(cfg, "astrom_exclude_stars", False):
            self.is_bright &= ~np.array([t.is_star for t in self.templates], dtype=bool)

        # A saturated-star scene holds the fragments of ONE star: fit a
        # single rigid shift for the whole group. The usual anchor cuts
        # would disable astrometry here — the fragments fail the isolation
        # cut against each other (and star exclusion) — yet the star is by
        # far the brightest thing in its scene and its repaired centroid
        # can be off by a fraction of a pixel.
        sat_scene = bool(self.templates) and all(
            getattr(t, "is_saturated", False) for t in self.templates
        )
        if sat_scene:
            self.is_bright = np.ones(len(self.templates), dtype=bool)

        # flux-only path
        if (not cfg.fit_astrometry_joint) or int(getattr(cfg, "fit_astrometry_niter", 0)) <= 0:
            sol = SceneFitter.solve(A, b, config=cfg, **kwargs)
        else:
            # first guess solution from diagonal-only solution
            # used to correctly scale the AB/BB blocks
            alpha0 = np.divide(b, d, out=np.zeros_like(b, dtype=float), where=d > 0)
            # joint path: build basis and coupling blocks
            order = int(cfg.astrom_kwargs["poly"]["order"])  # assume defined in cfg
            if sat_scene:
                order = 0  # one constant (dx, dy) — the fragments move rigidly

            basis, (x0, y0), (Sx, Sy) = make_scene_basis(
                self.templates, self.is_bright, order=order
            )
            self.shift_basis = [basis, (x0, y0), (Sx, Sy)]

            AB, BB, bB = assemble_scene_system_AB(
                self.templates,
                self.image,
                self.weights,
                basis,
                alpha0=alpha0,
                order=order,
                include_y=True,
                leverage_cap=getattr(cfg, "astrom_leverage_cap", None),
            )
            # if no valid AB BB solve will fall back to flux-only
            # @@@ scenefitter.solve should not take config but regularization and cg_kwargs
            sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg, **kwargs)
            self.shifts = sol.shifts

            if self.shifts is not None and len(self.shifts) > 0:
                # record per object shift in templates
                predict = AstroCorrect.build_poly_predictor(self.shifts, x0, y0, order, Sx, Sy)
                pts = np.array([t.position_original for t in self.templates], dtype=float)
                dx, dy = predict(pts[:, 0], pts[:, 1])
                damp = float(getattr(cfg, "astrom_damping", 1.0))
                for k, tmpl in enumerate(self.templates):
                    tmpl.to_shift = damp * np.array(
                        [float(dx[k]), float(dy[k])], dtype=float
                    )

                # optionally apply shifts to templates now and clear A/b
                if apply_shifts:
                    Templates.apply_template_shifts(self.templates)
                    self.A, self.b = None, None

                sid = getattr(self, "id", -1)
                beta_scene = self.shifts
                p = len(cheb_basis(0.0, 0.0, order))
                bx = beta_scene[:p]
                by = beta_scene[p : 2 * p]
                phi0 = cheb_basis(0.0, 0.0, order)
                mean_dx = float(phi0 @ bx)
                mean_dy = float(phi0 @ by)
                logger.info(
                    "[Scenes] Scene %s shift at x0,y0 ≈ (%.3f, %.3f) px (applied x%.2f)",
                    sid, mean_dx, mean_dy, damp,
                )

                logger.debug(
                    "[Scenes] center=(%.3f, %.3f) scale=(%.3f, %.3f) order=%d",
                    x0,
                    y0,
                    Sx,
                    Sy,
                    int(order),
                )
                logger.debug(f"[scenes] betas {self.id}:{self.shifts}")
            else:
                # <2 bright members: shift blocks were empty and the solver
                # fell back to flux-only — leave templates unshifted.
                # TODO: consider merging this scene with a neighbor rather
                # than skipping. merge_small_scenes applies the same three
                # cuts against the full-field normal matrix, where a source
                # also competes with neighbours outside its scene, so a scene
                # that passed there can still fall short here.
                for tmpl in self.templates:
                    tmpl.to_shift = np.zeros(2, dtype=float)
                logger.warning(
                    "[Scenes] Scene %s: fewer than 2 sources pass the astrometric "
                    "anchor cuts (SNR > %g, isolation >= %g%s); astrometry skipped "
                    "for this scene.",
                    getattr(self, "id", -1),
                    float(cfg.snr_thresh_astrom),
                    float(cfg.astrom_isolation_thresh),
                    ", stars excluded" if getattr(cfg, "astrom_exclude_stars", False) else "",
                )

        # store solution
        self.solution = sol

        #        self.flux, self.err, self.info = sol.flux, sol.err, sol.shifts, sol.info
        for tmpl, flux, err, bright in zip(self.templates, sol.flux, sol.err, self.is_bright):
            tmpl.flux = flux
            tmpl.err = err
            tmpl.is_bright = bright

        return sol.flux, sol.err, sol.shifts, sol.info

    def shift_at(self, x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        """Evaluate the already-applied shift at positions ``(x, y)``.

        Nearest-template lookup; returns ``(dx, dy)`` arrays. Returns zeros
        unless the scene has both a shift fit and a spatial index — ``tree``
        is only populated when :meth:`solve` rebuilds ``A``/``b`` itself, so
        a scene straight out of :func:`generate_scenes` returns zeros on its
        first pass.
        """

        if self.shifts is None or self.shift_basis is None or self.tree is None:
            return np.zeros_like(x), np.zeros_like(y)

        # Ensure x, y are arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        # Convert coordinates to Shapely Point objects
        from shapely.geometry import Point

        pts = [Point(float(xi), float(yi)) for xi, yi in zip(x, y)]

        # Query nearest template(s) for each (x, y)
        nearest_idxs = self.tree.nearest(pts)
        # nearest_idxs: indices into self.templates

        # For each query point, get the shift of the nearest template
        shifts = np.zeros((len(pts), 2), dtype=float)
        for i, idx in enumerate(nearest_idxs):
            if hasattr(self.templates[idx], "to_shift"):
                shifts[i] = self.templates[idx].shifted
            else:
                shifts[i] = [0.0, 0.0]

        # If input was scalar, return scalars
        if np.isscalar(x) and np.isscalar(y):
            return float(shifts[0, 0]), float(shifts[0, 1])
        return shifts[:, 0], shifts[:, 1]

    @staticmethod
    def create_scene_graph(templates: List[Template]) -> np.ndarray:
        """Return zero-based connected-component labels from template overlaps."""
        n = len(templates)
        labels = np.full(n, -1, dtype=int)
        current = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            stack = [i]
            labels[i] = current
            while stack:
                j = stack.pop()
                for k in range(n):
                    if labels[k] >= 0:
                        continue
                    if Scene._overlaps(templates[j].bbox, templates[k].bbox):
                        labels[k] = current
                        stack.append(k)
            current += 1
        return labels

    @staticmethod
    def overlay_scene_graph(
        templates: List[Template], shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlay scene labels onto an empty image of ``shape``."""
        labels = Scene.create_scene_graph(templates)
        seg = np.zeros(shape, dtype=int)
        for lbl, tmpl in zip(labels, templates):
            y0, y1, x0, x1 = tmpl.bbox
            seg[y0:y1, x0:x1] = int(lbl) + 1
        return seg, labels

    @staticmethod
    def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        """Return True if bounding boxes ``a`` and ``b`` overlap."""
        y0a, y1a, x0a, x1a = a
        y0b, y1b, x0b, x1b = b
        return not (y1a <= y0b or y1b <= y0a or x1a <= x0b or x1b <= x0a)

    def plot(
        self,
        tmpl_image: np.ndarray,
        seg_image: np.ndarray,
        display_sig: float = 3.0,
        display_sig_by_title: dict[str, float] | None = None,
        residual_image: np.ndarray | None = None,
        null_segments: Sequence[int] | None = None,
        ax=None,
        **imshow_kwargs,
    ) -> tuple["matplotlib.figure.Figure", np.ndarray]:
        """Plot the six-panel scene diagnostic (template, image, model,
        segmap, residual, color composite), with the fitted shift field
        drawn as arrows on the model panel.

        Parameters
        ----------
        tmpl_image
            Full-frame high-resolution template image corresponding to
            ``self.image``.
        seg_image
            Full-frame segmentation map on the same grid as ``tmpl_image``.
        display_sig
            Sigma level used to scale grayscale panels. Defaults to ``3``.
        display_sig_by_title
            Optional per-panel overrides of ``display_sig``, keyed by panel
            title.
        residual_image
            Optional full-frame residual on the fit grid with *all* scene
            models subtracted. If given, the residual panel shows this global
            residual over the scene bbox. If ``None``, the panel falls back to
            ``self.residual()``, which subtracts only this scene's model and
            therefore still contains the (masked) light of sources belonging
            to other scenes.
        null_segments
            Segmentation labels of sources whose brightness would otherwise
            dominate a neighbouring scene's display — typically the saturated
            stars' segments. They are excluded from the *display scale* of
            the image panel but still drawn there, and nulled in the residual
            panel, where the fit residual under a saturated core is
            meaningless. Labels belonging to THIS scene's templates are never
            affected. The template and colour panels ignore the list.
        ax
            Optional array of matplotlib axes to draw on.
        **imshow_kwargs
            Additional keyword arguments forwarded to ``imshow`` for grayscale
            panels.

        Returns
        -------
        tuple
            Matplotlib figure and flattened array of axes.
        """

        from copy import deepcopy
        from astropy.visualization import make_lupton_rgb
        from photutils.segmentation import SegmentationImage
        import matplotlib.pyplot as plt
        from astropy.wcs.utils import proj_plane_pixel_scales
        from matplotlib.colors import ListedColormap

        if self.image is None or self.bbox is None:
            raise ValueError("Scene has no image data or bounding box")

        y0, y1, x0, x1 = self.bbox
        sl = _slices_from_bbox(self.bbox)
        tmpl_cut = tmpl_image[sl]
        seg_cut = seg_image[sl]
        img_cut = self.image[sl]

        # Foreign saturated segments stay visible in the image panel — it
        # shows the data as it is — but are kept out of its display scale,
        # which their brightness would otherwise flatten. The residual panel
        # nulls them: the fit residual under a saturated core is meaningless
        # and would set the whole panel's stretch.
        img_raw = img_cut  # un-nulled, for the colour composite below
        own_ids = {int(t.id) for t in self.templates}
        null_ids = [int(s) for s in (null_segments or []) if int(s) not in own_ids]
        null_mask = None
        if null_ids:
            mask = np.isin(seg_cut, np.asarray(null_ids))
            if mask.any():
                null_mask = mask

        scene_cut = np.zeros_like(seg_cut)
        scene_cut[seg_cut > 0] = int(self.id)

        segm = SegmentationImage(seg_cut)
        segmap_cmap = segm.cmap
        scene_cmap = deepcopy(segmap_cmap)
        scene_cmap.colors[0] = (1.0, 1.0, 1.0, 0.0)

        model_cut = self.model_image()
        if residual_image is not None:
            res_cut = np.asarray(residual_image)[sl].copy()
            res_cut[(self.weights[sl] <= 0) | np.isnan(self.weights[sl])] = 0.0
        else:
            res_cut = self.residual()
        if null_mask is not None and null_mask.any():
            res_cut = np.where(null_mask, 0.0, res_cut)

        b = tmpl_cut / np.nanstd(tmpl_cut) if np.nanstd(tmpl_cut) != 0 else tmpl_cut
        r = img_raw / np.nanstd(img_raw) if np.nanstd(img_raw) != 0 else img_raw
        g = (r + b) / 2.0
        col_cut = make_lupton_rgb(r, g, b, stretch=display_sig / 1.5)

        aspect = img_cut.shape[1] / img_cut.shape[0]

        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            ax = ax.flatten()
            created_fig = True
        else:
            fig = ax[0].figure
            created_fig = False

        # Create scene-specific segmap overlay for template panel
        scene_segmap = np.zeros_like(seg_cut)
        template_ids = [t.id for t in self.templates]  # Get all template IDs in this scene
        for template_id in template_ids:
            scene_segmap[seg_cut == template_id] = 1

        if residual_image is not None:
            # global residual already has every scene's model subtracted
            res_cut_masked = res_cut
        else:
            # Mask residual to only show pixels belonging to this scene
            res_cut_masked = res_cut.copy()
            # Set residual to zero where segmap shows sources NOT in this scene
            # (i.e., where seg_cut > 0 but scene_segmap == 0)
            other_sources_mask = (seg_cut > 0) & (scene_segmap == 0)
            res_cut_masked[other_sources_mask] = 0.0

        # Plot panels - use the masked residual
        images = [tmpl_cut, img_cut, model_cut, seg_cut, res_cut_masked, col_cut]
        titles = ["Template", "Image", "Model", "Segmap", "Residual", "Color"]

        # Pixels each panel's grayscale stretch is measured on; only the image
        # panel differs from what it displays (foreign saturated segments are
        # shown but excluded from its scale).
        scale_data = list(images)
        if null_mask is not None:
            scale_data[1] = img_cut[~null_mask]

        def _panel_std(ref: np.ndarray, fallback: np.ndarray) -> float:
            values = ref[ref != 0]
            if values.size == 0:
                values = fallback[fallback != 0]
            return float(np.std(values)) if values.size else 1.0

        for i, (img, title) in enumerate(zip(images, titles)):
            sig = (display_sig_by_title or {}).get(title, display_sig)
            if "Segmap" in title:
                ax[i].imshow(img, origin="lower", cmap=segmap_cmap, interpolation="nearest")
            elif "Residual" in title:  # residual
                std = _panel_std(scale_data[i], img)
                ax[i].imshow(
                    img,
                    origin="lower",
                    cmap="gray",
                    vmin=-sig * std,
                    vmax=sig * std,
                    **imshow_kwargs,
                )
                outside_fit = np.abs(model_cut) <= 0.0
                if np.any(outside_fit):
                    masked_outside = np.ma.masked_where(~outside_fit, np.ones_like(model_cut))
                    ax[i].imshow(
                        masked_outside,
                        origin="lower",
                        cmap=ListedColormap([(1.0, 1.0, 1.0, 0.28)]),
                        interpolation="nearest",
                    )
            elif "Color" in title:  # color
                ax[i].imshow(img, origin="lower", **imshow_kwargs)
            else:
                std = _panel_std(scale_data[i], img)
                ax[i].imshow(
                    img,
                    origin="lower",
                    cmap="gray",
                    vmin=-sig * std,
                    vmax=sig * std,
                    **imshow_kwargs,
                )

                # Overlay scene segmentation on template panel
                if "Template" in title:
                    # Create a masked array where 0 values are transparent
                    scene_overlay = np.ma.masked_where(scene_segmap == 0, scene_segmap)
                    ax[i].imshow(
                        scene_overlay, origin="lower", cmap="autumn", alpha=0.15, vmin=0, vmax=1
                    )

            ax[i].set_title(title)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        # Add shift field overlay on the model panel (index 3)
        if self.shifts is not None and self.shift_basis is not None and len(self.templates) > 0:
            model_ax = ax[2]

            # Create a coarse grid for displaying shifts
            h, w = model_cut.shape
            step = max(h // 7, w // 7, 10)  # ~15 arrows per dimension, minimum 10 pixels

            y_grid, x_grid = np.mgrid[step // 2 : h : step, step // 2 : w : step]
            dx_grid = np.zeros_like(x_grid, dtype=float)
            dy_grid = np.zeros_like(y_grid, dtype=float)

            # Get shifts at grid positions (convert to scene coordinates)
            for i in range(x_grid.shape[0]):
                for j in range(x_grid.shape[1]):
                    # Convert cutout coordinates to original image coordinates
                    x_orig = x_grid[i, j] + x0
                    y_orig = y_grid[i, j] + y0

                    try:
                        dx, dy = self.shift_at(x_orig, y_orig)
                        dx_grid[i, j] = dx
                        dy_grid[i, j] = dy
                    except:
                        # If shift_at fails, use zero shift
                        dx_grid[i, j] = 0.0
                        dy_grid[i, j] = 0.0

            # Scale arrows for visibility (make them ~1/20 of the image size)
            max_shift = np.sqrt(dx_grid**2 + dy_grid**2).max()
            if max_shift > 0:
                arrow_scale = min(h, w) / 20.0 / max_shift
                dx_display = dx_grid * arrow_scale
                dy_display = dy_grid * arrow_scale

                # Plot quiver arrows
                model_ax.quiver(
                    x_grid,
                    y_grid,
                    dx_display,
                    dy_display,
                    color="red",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    alpha=0.8,
                    width=0.003,
                    headwidth=3,
                    headlength=3,
                )

            # Add size bar to show 1 pixel scale
            # Try to get pixel scale from template WCS if available
            pixel_scale_arcsec = None
            if hasattr(self.templates[0], "wcs") and self.templates[0].wcs is not None:
                try:
                    scales = proj_plane_pixel_scales(self.templates[0].wcs)
                    pixel_scale_arcsec = float(scales[0] * 3600)  # convert to arcsec
                except:
                    pass

            # Position size bar in bottom-right corner
            bar_length = 1.0  # 1 pixel
            bar_x = w - 0.15 * w
            bar_y = 0.1 * h

            # Draw the size bar
            model_ax.plot(
                [bar_x, bar_x + bar_length],
                [bar_y, bar_y],
                color="white",
                linewidth=3,
                solid_capstyle="butt",
            )
            model_ax.plot(
                [bar_x, bar_x + bar_length],
                [bar_y, bar_y],
                color="black",
                linewidth=1,
                solid_capstyle="butt",
            )

            # Add label
            if pixel_scale_arcsec is not None:
                label = f'1 pix = {pixel_scale_arcsec:.3f}"'
            else:
                label = "1 pixel"

            model_ax.text(
                bar_x + bar_length / 2,
                bar_y - 0.03 * h,
                label,
                ha="center",
                va="top",
                color="white",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

            # Add shift scale indicator
            if max_shift > 0:
                med_dx = float(np.median(dx_grid))
                med_dy = float(np.median(dy_grid))
                shift_text = f"shift dx={med_dx:+.2f}, dy={med_dy:+.2f} pix"
                model_ax.text(
                    0.02,
                    0.98,
                    shift_text,
                    transform=model_ax.transAxes,
                    va="top",
                    ha="left",
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        if created_fig:
            plt.tight_layout()
            return fig, ax
        else:
            return fig, ax

    def model_image(self) -> np.ndarray:
        """Return the model image over the scene's bounding box."""
        if self.solution is None:
            raise RuntimeError("No solution available")
        bb = self.bbox
        model_scene = np.zeros((bb[1] - bb[0] + 1, bb[3] - bb[2] + 1), dtype=float)
        for t in self.templates:
            sl = t.slices_original
            sl_local_scene = (
                slice(sl[0].start - bb[0], sl[0].stop - bb[0]),
                slice(sl[1].start - bb[2], sl[1].stop - bb[2]),
            )
            model_scene[sl_local_scene] += t.flux * t.data[t.slices_cutout]
        return model_scene

    def residual(self) -> np.ndarray:
        """Return image-model residual over the scene's bounding box."""
        bb = self.bbox
        sl = _slices_from_bbox(bb)
        res_scene = self.image[sl] - self.model_image()
        res_scene[(self.weights[sl] <= 0) | np.isnan(self.weights[sl])] = 0.0
        return res_scene

