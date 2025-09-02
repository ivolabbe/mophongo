from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import numpy as np
import scipy.sparse as sp

from .templates import Template, Templates
from .fit import FitConfig
from .scene_fitter import SceneFitter
from .astrometry import cheb_basis, AstroCorrect, n_terms

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from .fit import FitConfig as FitConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # show info for *this* logger only
if not logger.handlers:  # avoid duplicate handlers on reloads
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(module)s.%(funcName)s: %(message)s"))
    logger.addHandler(handler)


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


def build_scene_tree_from_normal(
    ATA: sp.spmatrix,
    ATb: np.ndarray,
    *,
    coupling_thresh: float = 0.01,  # 3% leakage threshold
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


# scene_basis.py (or alongside your fitter helpers)

import numpy as np
from typing import List, Optional, Tuple
from .templates import Template
from .astrometry import cheb_basis


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
    ab_from_bright_only: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Build the (A,B) coupling blocks and beta RHS for a *single scene*.

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
    ab_from_bright_only
        If True, rows with Si=None (faint) do not contribute to AB; BB/bB
        still use only bright members (Si≠None).

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

    AB = sp.lil_matrix((nA, nB), dtype=float)
    BB = np.zeros((nB, nB), dtype=float)
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

    for row, ti in enumerate(templates):
        sl = ti.slices_original
        tcut = ti.data[ti.slices_cutout]
        w = weights[sl]
        img = image[sl]

        Si = basis_vals[row]
        if (Si is None) and ab_from_bright_only:
            continue

        ai = float(a[row])  # flux scaling (pixels remain in dx/dy)
        Gx, Gy = _gx_gy_for(row)

        # Inner products with template (weighted)
        gx_ip = float(np.sum(tcut * w * Gx[ti.slices_cutout]))
        AB[row, 0:p] += (-ai) * gx_ip * (Si if Si is not None else 0.0)
        if include_y:
            gy_ip = float(np.sum(tcut * w * Gy[ti.slices_cutout]))
            AB[row, p : 2 * p] += (-ai) * gy_ip * (Si if Si is not None else 0.0)

        if Si is not None:
            # BB accumulation (Gauss–Newton, uses gradients only on support)
            Gxx = float(np.sum(Gx[ti.slices_cutout] * w * Gx[ti.slices_cutout]))
            BB[0:p, 0:p] += (ai * ai) * Gxx * np.outer(Si, Si)

            if include_y:
                Gyy = float(np.sum(Gy[ti.slices_cutout] * w * Gy[ti.slices_cutout]))
                BB[p : 2 * p, p : 2 * p] += (ai * ai) * Gyy * np.outer(Si, Si)

            # RHS for beta (sign matches AB)
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


def generate_scenes(
    templates: Sequence[Template],
    image: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    coupling_thresh: float = 0.03,
    snr_thresh_astrom: float = 7.0,
    minimum_bright: int | None = None,
    max_merge_radius: float = np.inf,
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

    # 2) Initial scene labels from normal-equation couplings
    labels0, _ = build_scene_tree_from_normal(
        ATA, ATb, coupling_thresh=coupling_thresh, return_0_based=False
    )

    # 3) Merge scenes that are too small in terms of "bright" members
    #    SNR proxy: snr_i ≈ b_i / sqrt(diag(A)_i)
    d = np.asarray(ATA.diagonal(), dtype=float)
    snr_proxy = np.divide(
        ATb, np.sqrt(np.maximum(d, 1e-12)), out=np.zeros_like(ATb, dtype=float), where=d > 0
    )
    bright_mask = np.asarray(snr_proxy > float(snr_thresh_astrom), dtype=bool)

    labels, nscene = merge_small_scenes(
        labels0,
        list(templates),
        bright_mask,
        minimum_bright=minimum_bright,
        max_merge_radius=max_merge_radius,
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

    return scenes, labels


@dataclass
class Scene:
    """Container for templates belonging to a single scene."""

    id: int
    templates: List[Template]
    fitter: SceneFitter
    bbox: Tuple[int, int, int, int] | None = None
    image: np.ndarray | None = None
    weights: np.ndarray | None = None
    config: FitConfig | None = None
    shift_basis: np.ndarray | None = None
    flux: np.ndarray | None = None
    err: np.ndarray | None = None
    beta: np.ndarray | None = None
    is_bright: np.ndarray | None = None  # per-template
    info: int | None = None
    # store per-scene normal blocks (scene-local ordering)
    A: sp.csr_matrix | None = None
    b: np.ndarray | None = None

    def __post_init__(self) -> None:
        pass

    def set_band(
        self,
        image: np.ndarray,
        weight: np.ndarray | None = None,
        psf: np.ndarray | None = None,
        config: Optional[object] = None,
    ) -> None:
        """Cache per-band data for this scene."""
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
        """
        Solve this scene. If A/b are not provided and not cached, build them.
        Only build AB/BB/bB when cfg.fit_astrometry_joint is True; else flux-only.
        Stores results on the Scene (stateless fitter).
        """
        cfg = config or self.config or FitConfig()
        if self.image is None or self.weights is None:
            raise RuntimeError(
                "Scene image/weights not set. Call set_band() or generate_scenes()."
            )

        # ensure flux block is available
        if self.A is None or self.b is None:
            # build normal from current band
            from .scene_fitter import build_normal

            self.A, self.b, _ = build_normal(self.templates, self.image, self.weights)

        A, b = self.A, self.b

        # bright mask via SNR proxy against cfg.snr_thresh_astrom
        d = np.asarray(A.diagonal(), dtype=float)
        snr_proxy = np.divide(
            b, np.sqrt(np.maximum(d, 1e-12)), out=np.zeros_like(b, dtype=float), where=d > 0
        )
        self.is_bright = snr_proxy > float(cfg.snr_thresh_astrom)

        # flux-only path
        if not cfg.fit_astrometry_joint:
            sol = SceneFitter.solve(A, b, config=cfg, **kwargs)
        else:
            # first guess solution from diagonal-only solution
            # used to correctly scale the AB/BB blocks
            alpha0 = np.divide(b, d, out=np.zeros_like(b, dtype=float), where=d > 0)
            # joint path: build basis and coupling blocks
            order = int(cfg.astrom_kwargs["poly"]["order"])  # assume defined in cfg

            self.basis, (x0, y0), (Sx, Sy) = make_scene_basis(
                self.templates, self.is_bright, order=order
            )
            AB, BB, bB = assemble_scene_system_AB(
                self.templates,
                self.image,
                self.weights,
                self.basis,
                alpha0=alpha0,
                order=order,
                include_y=True,
                ab_from_bright_only=True,
            )
            # if no valid AB BB solve will fall back to flux-only
            sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg, **kwargs)
            self.shifts = sol.shifts

            # record per object shift in templates
            predict = AstroCorrect.build_poly_predictor(self.shifts, x0, y0, order, Sx, Sy)
            pts = np.array([t.position_original for t in self.templates], dtype=float)
            dx, dy = predict(pts[:, 0], pts[:, 1])
            for k, tmpl in enumerate(self.templates):
                tmpl.to_shift = np.array([float(dx[k]), float(dy[k])], dtype=float)

            if apply_shifts:
                Templates.apply_template_shifts(self.templates)

            sid = getattr(self, "id", -1)
            beta_scene = self.shifts
            p = len(cheb_basis(0.0, 0.0, order))
            bx = beta_scene[:p]
            by = beta_scene[p : 2 * p]
            phi0 = cheb_basis(0.0, 0.0, order)
            mean_dx = float(phi0 @ bx)
            mean_dy = float(phi0 @ by)
            logger.info(
                "[Scenes] Scene %s shift at x0,y0 ≈ (%.3f, %.3f) px", sid, mean_dx, mean_dy
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

        # store solution
        self.solution = sol
        #        self.flux, self.err, self.info = sol.flux, sol.err, sol.shifts, sol.info
        for tmpl, flux, err, bright in zip(self.templates, sol.flux, sol.err, self.is_bright):
            tmpl.flux = flux
            tmpl.err = err
            tmpl.is_bright = bright

        return sol.flux, sol.err, sol.shifts, sol.info

    def add_residuals(self, image: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Subtract model contributions from ``image`` in-place."""
        for c, tmpl in zip(coeffs, self.templates):
            sl = tmpl.slices_original
            cut = tmpl.data[tmpl.slices_cutout]
            image[sl] -= c * cut
        return image

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

    def plot(self, image: np.ndarray, ax=None, **imshow_kwargs):
        """Plot the scene on top of ``image``."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.imshow(image, origin="lower", **imshow_kwargs)
        for tmpl in self.templates:
            y0, y1, x0, x1 = tmpl.bbox
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="r")
            ax.add_patch(rect)
        return ax

    def residual(self) -> np.ndarray:
        """Return image-model residual over the scene's bounding box."""
        if self._solution is None:
            raise RuntimeError("No solution available")
        model = np.zeros_like(self._image)
        for amp, tmpl in zip(self._solution.flux, self.templates):
            model += tmpl.data[tmpl.slices_cutout] * amp
        return self._image - model

    # ------------------------------------------------------------------
    # Placeholders for future extensions
    # ------------------------------------------------------------------
    def augment_templates(self, thresh: float, mode: str = "psf_core") -> None:
        """Placeholder for residual-driven template augmentation."""
        return None
