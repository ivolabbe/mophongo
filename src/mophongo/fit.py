from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import logging
import numpy as np
from numpy.random import default_rng
from scipy.sparse import csr_matrix, diags, eye, lil_matrix, bmat
from scipy.sparse.linalg import LinearOperator, cg, lsqr, minres
from tqdm import tqdm

from .astrometry import cheb_basis
from .templates import Template, Templates

logger = logging.getLogger(__name__)

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

    positivity: bool = False
    # ``reg`` is interpreted as an absolute value when > 0. When set to 0 the
    # fitter will compute a default regularisation strength based on the median
    # of the normal matrix diagonal.
    reg: float = 0.0
    bad_value: float = np.nan
    solve_method: str = "ata"  # 'ata' or 'lo' (linear operator)
    cg_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"M": None, "maxiter": 500, "atol": 1e-6}
    )
    fit_covariances: bool = False  # Use simple fitting errors from diagonal of normal matrix
    fft_fast: float | bool = (
        False  # False for full kernel, float (0.1-1.0) for truncated FFT kernels
    )
    # condense fit astrometry flags into one: fit_astrometry_niter = 0, means not fitting astrometry
    fit_astrometry_niter: int = 2  # Number of astrometry refinement passes (0 → disabled)
    fit_astrometry_joint: bool = False  # Use joint astrometry fitting, or separate step
    # --- astrometry options -------------------------------------------------
    reg_astrom: float = 1e-4
    snr_thresh_astrom: float = 10.0  # 0 → keep all sources (current behaviour)
    astrom_model: str = "gp"  # 'polynomial' or 'gp'
    astrom_centroid: str = "centroid"  # "centroid" (=old) | "correlation"
    #    astrom_basis_order: int = 1
    astrom_kwargs: dict[str, dict] = field(
        default_factory=lambda: {"poly": {"order": 2}, "gp": {"length_scale": 500}}
    )
    #    astrom_kwargs={'poly': {'order': 2}, 'gp': {'length_scale': 400}}
    multi_tmpl_chi2_thresh: float = 5.0
    multi_tmpl_psf_core: bool = False
    multi_tmpl_colour: bool = False
    #    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    multi_resolution_method: str = "downsample"  # 'upsample' or 'downsample'
    normal: str = "tree"  # 'loop' or 'tree'


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


def build_components_strtree_labels(
    templates: List[Template],
) -> tuple[np.ndarray, int]:
    """Label independent template groups using an ``STRtree``.

    Parameters
    ----------
    templates
        Templates whose bounding boxes define the connectivity graph.

    Returns
    -------
    labels, ncomp
        ``labels`` maps each template index to a component id and ``ncomp`` is
        the total number of components.
    """
    from shapely.geometry import box
    from shapely.strtree import STRtree
    from scipy.sparse import coo_matrix, csgraph

    def _bbox_to_box(t: Template):
        (ymin, ymax), (xmin, xmax) = t.bbox_original  # closed intervals
        return box(xmin, ymin, xmax + 1, ymax + 1)

    n = len(templates)
    boxes = [_bbox_to_box(t) for t in templates]
    tree = STRtree(boxes)

    ii: list[int] = []
    jj: list[int] = []
    for i, gi in enumerate(boxes):
        for j in tree.query(gi):
            j = int(j)
            if j > i:
                ii.append(i)
                jj.append(j)

    if not ii:
        return np.arange(n, dtype=int), n

    adj = coo_matrix((np.ones(len(ii), dtype=np.uint8), (ii, jj)), shape=(n, n))
    adj = adj + adj.T
    ncomp, labels = csgraph.connected_components(adj, directed=False)
    return labels, int(ncomp)


def summarize_components(labels: np.ndarray) -> np.ndarray:
    """Log a brief summary of component sizes."""

    counts = np.bincount(labels)
    logger.info(
        "%d components (min=%d, median=%d, max=%d)",
        len(counts),
        counts.min(),
        int(np.median(counts)),
        counts.max(),
    )
    topk = np.argsort(counts)[::-1][:10]
    logger.debug(
        "Top components by size: %s",
        [(int(cid), int(counts[cid])) for cid in topk],
    )
    return counts


def solve_components_cg_with_labels(
    ATA_csr: csr_matrix,
    ATb: np.ndarray,
    labels: np.ndarray,
    *,
    rtol: float = 1e-6,
    maxiter: int = 2000,
) -> tuple[np.ndarray, list[int]]:
    """Solve block-diagonal systems independently with conjugate gradient."""

    x = np.zeros_like(ATb, dtype=float)
    infos: list[int] = []
    for cid in range(labels.max() + 1):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            infos.append(0)
            continue
        A = ATA_csr[idx][:, idx].tocsr()
        b = ATb[idx]
        D = A.diagonal().copy()
        D[D == 0] = 1.0
        M = LinearOperator(A.shape, matvec=lambda v, D=D: v / D)
        sol, info = cg(A, b, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
        x[idx] = sol
        infos.append(int(info))
    return x, infos

def make_basis_per_component(
    templates: List[Template],
    labels: np.ndarray,
    bright: np.ndarray,
    order: int = 1,
) -> list[Optional[np.ndarray]]:
    """Return per-template basis vectors for bright sources."""

    basis: list[Optional[np.ndarray]] = [None] * len(templates)
    for cid in range(labels.max() + 1):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        xs = np.array([templates[i].position_original[0] for i in idx], float)
        ys = np.array([templates[i].position_original[1] for i in idx], float)
        x0, y0 = xs.mean(), ys.mean()
        for i in idx:
            if bright[i]:
                x, y = templates[i].position_original
                basis[i] = cheb_basis(x - x0, y - y0, order)
    return basis


def assemble_component_system(
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
    """Return block system for a single component including shifts."""

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
                    inter[0].start - sl_j[0].start + tj.slices_cutout[0].start,
                    inter[0].stop - sl_j[0].start + tj.slices_cutout[0].start,
                ),
                slice(
                    inter[1].start - sl_j[1].start + tj.slices_cutout[1].start,
                    inter[1].stop - sl_j[1].start + tj.slices_cutout[1].start,
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
                if Sj is not None and (
                    not ab_from_bright_only or basis_vals[g_i] is not None
                ):
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
                        Gyy = float(
                            np.sum(Gyi[sl_i_local] * w * Gyj[sl_j_local])
                        )
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
                bB[p : 2 * p] += (
                    float(np.sum(Gyj[tj.slices_cutout] * w * img)) * Sj
                )

    return AA.tocsr(), AB.tocsr(), csr_matrix(BB), bA, bB

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
        self.templates = templates.copy()  # work in copy for fitting, modifying

        self.n_flux = len(templates)
        for i, tmpl in enumerate(self.templates):
            tmpl.is_flux = True
            tmpl.col_idx = i

        self.image = image
        self.weights = weights
        self.config = config or FitConfig()
        self._ata = None
        self._atb = None
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
        self.bright_mask = snr > self.config.snr_thresh_astrom

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
        return float(np.sum(data * w * data))

    def build_normal(self) -> None:
        """Dispatch to the configured normal-matrix builder."""
        if getattr(self.config, "normal", "loop") == "tree":
            self.build_normal_tree()
        else:
            self.build_normal_matrix()

    def build_normal_matrix(self) -> None:
        """Construct normal matrix using :class:`Template` objects."""
        # Compute weighted norms for all templates first

        norms = [self._weighted_norm(t) for t in self.templates]
        tol = 1e-8 * max(norms)

        # Prune templates with near-zero norm
        # valid: list[Template] = []
        # norms: list[float] = []
        # for tmpl, norm in zip(self.templates, norms_all):
        #     if norm < tol:
        #         valid.append(tmpl)
        #         norms.append(norm)

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

    def build_normal_tree(self) -> None:
        """Construct normal matrix using an STRtree to find overlaps."""
        from shapely.geometry import box
        from shapely.strtree import STRtree

        norms = [self._weighted_norm(t) for t in self.templates]
        tol = 1e-8 * max(norms)
        keep = [i for i, n in enumerate(norms) if n > tol]
        dropped = len(self.templates) - len(keep)
        if dropped:
            logger.info("Dropped %d templates with low norm.", dropped)
        self.templates = [self.templates[i] for i in keep]
        norms = [norms[i] for i in keep]

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

    def solve(
        self, config: FitConfig | None = None, x_w0: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for template fluxes using conjugate gradient."""
        cfg = config or self.config

        # build big normal matrix once, this as a shift entry for every template
        A, b = self.ata, self.atb  # triggers build_normal()

        # Guarantees strict positive definiteness after whitening
        # A symmetric matrix that is positive‐semi-definite but rank-deficient
        # can have eigenvalues down to 10⁻¹⁴–10⁻¹⁶ (numerical zero). Adding 10⁻⁸
        # shifts every eigenvalue by that amount, lifting them well above rounding
        # error yet staying ≪ typical diagonal (10⁰–10⁴ for sky+source units). The
        # induced bias in fluxes is therefore ≤ 10⁻⁸ negligible compared to
        # Poisson errors (∼10⁻²–10⁻³).
        reg = cfg.reg
        if reg <= 0:
            reg = 1e-4 * np.median(A.diagonal())
        if reg > 0:
            A = A + eye(A.shape[0], format="csr") * reg

        # detecting bad rows.
        bad = np.where(np.abs(A.diagonal()) < 1e-14 * np.max(A.diagonal()))[0]
        if bad.size:
            print(f"Eliminating {bad.size} nearly-zero diagonal rows before ILU", bad.size)

        eps = reg or 1e-10
        d = np.sqrt(np.maximum(A.diagonal(), eps))
        Dinv = diags(1.0 / d, 0, format="csr")

        A_w = Dinv @ A @ Dinv
        b_w = Dinv @ b

        # expand to full solution vector corresponding to _orig_templates
        idx = [t.col_idx for t in self.templates]

        # reuse prevous solution if available
        x_w0 = getattr(self, "x_w0", None)
        x_w, info = cg(A_w, b_w, x0=x_w0, **cfg.cg_kwargs)
        self.x_w = x_w

        # n_flux is always the full input length of the templates
        x_full = np.zeros(self.n_flux, dtype=float)
        e_full = np.zeros(self.n_flux, dtype=float)

        x_full[idx] = x_w / d  # un-whiten + scatter
        e_full[idx] = self._flux_errors(A_w) / d  # un-whiten errors

        if cfg.positivity:
            x_full[: self.n_flux] = np.maximum(0, x_full[: self.n_flux])

        self.solution = x_full[: self.n_flux]  # fluxes, corresponds to original templates
        self.solution_err = e_full[: self.n_flux]  # flux errors, corresponds to original templates

        # update the templates with the fitted fluxes, errors
        for tmpl, flux, err in zip(self._orig_templates, self.solution, self.solution_err):
            tmpl.flux = flux
            tmpl.err = err

        return self.solution, self.solution_err, info

    def solve_components(
        self, config: FitConfig | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve independent template groups separately using CG.

        Templates are partitioned into connected components via their bounding
        box overlaps. Each component is solved independently, allowing block
        diagonal systems to be handled efficiently.
        """

        cfg = config or self.config
        A, b = self.ata, self.atb

        reg = cfg.reg
        if reg <= 0:
            reg = 1e-4 * np.median(A.diagonal())
        if reg > 0:
            A = A + eye(A.shape[0], format="csr") * reg

        labels, ncomp = build_components_strtree_labels(self.templates)
        summarize_components(labels)

        rtol = cfg.cg_kwargs.get("rtol", 1e-6)
        maxit = cfg.cg_kwargs.get("maxiter", 2000)
        x, info = solve_components_cg_with_labels(
            A, b, labels, rtol=rtol, maxiter=maxit
        )

        err = np.zeros_like(x)
        for cid in range(ncomp):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            A_sub = A[idx][:, idx].tocsr()
            err[idx] = self._flux_errors(A_sub)

        if cfg.positivity:
            x = np.maximum(0.0, x)

        x_full = np.zeros(self.n_flux, dtype=float)
        e_full = np.zeros(self.n_flux, dtype=float)
        idx = [t.col_idx for t in self.templates]
        x_full[idx] = x
        e_full[idx] = err

        self.solution = x_full
        self.solution_err = e_full
        for tmpl, flux, err in zip(self._orig_templates, x_full, e_full):
            tmpl.flux = flux
            tmpl.err = err

        return x_full, e_full, {"cg_info": info, "ncomp": ncomp}

    def solve_components_shifts(
        self,
        order: int = 1,
        *,
        include_y: bool = True,
        ab_from_bright_only: bool = True,
        rtol: float = 1e-6,
        maxiter: int = 2000,
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], dict]:
        """Solve components with additional polynomial shift terms.

        Returns
        -------
        flux : ndarray
            Best-fit fluxes for the original templates.
        betas : list[tuple[int, ndarray]]
            Per-component shift coefficients.
        info : dict
            Solver diagnostics including CG convergence flags.
        """

        labels, ncomp = build_components_strtree_labels(self.templates)
        summarize_components(labels)
        basis_vals = make_basis_per_component(
            self.templates, labels, self.bright_mask, order=order
        )

        alpha = np.zeros(len(self.templates), dtype=float)
        betas: list[tuple[int, np.ndarray]] = []
        infos: list[int] = []

        for cid in range(labels.max() + 1):
            comp = np.where(labels == cid)[0].tolist()
            if not comp:
                infos.append(0)
                continue
            AA, AB, BB, bA, bB = assemble_component_system(
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
                    sol, info = minres(
                        K, rhs, M=M, atol=0.0, rtol=rtol, maxiter=maxiter
                    )
                na = len(comp)
                alpha_comp = sol[:na]
                beta_comp = sol[na:]
            alpha[np.array(comp)] = alpha_comp
            betas.append((cid, beta_comp))
            infos.append(int(info))

        if self.config.positivity:
            alpha = np.maximum(0.0, alpha)

        x_full = np.zeros(self.n_flux, dtype=float)
        idx = [t.col_idx for t in self.templates]
        x_full[idx] = alpha
        self.solution = x_full
        for tmpl, flux in zip(self._orig_templates, x_full):
            tmpl.flux = flux

        info = {"ncomp": ncomp, "cg_info": infos}
        return x_full, betas, info

    def solve_linear_operator(
        self,
        config: FitConfig | None = None,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve for template fluxes with LSQR on a matrix-free ``LinearOperator``.

        Parameters
        ----------
        config
            Optional :class:`FitConfig` overriding the fitter configuration.
        x0
            Optional starting vector for the iterative solver.

        Returns
        -------
        flux, err, info
            Best-fit coefficients, their 1-σ uncertainties (predicted-error
            approximation) and a dictionary with solver diagnostics.
        """

        cfg = config or self.config

        w_sqrt = np.sqrt(self.weights)
        n_cols = len(self.templates)
        img_shape = self.image.shape
        flat_size = self.image.size

        def _Av(x: np.ndarray) -> np.ndarray:
            """Apply ``A`` scaled by ``W^{1/2}``."""

            img = np.zeros_like(self.image, dtype=float)
            for coeff, tmpl in zip(x, self.templates):
                img[tmpl.slices_original] += coeff * tmpl.data[tmpl.slices_cutout]
            return (w_sqrt * img).ravel()

        def _ATv(y: np.ndarray) -> np.ndarray:
            """Apply ``Aᵀ`` scaled by ``W^{1/2}``."""

            y_img = w_sqrt * y.reshape(img_shape)
            out = np.zeros(n_cols, dtype=float)
            for i, tmpl in enumerate(self.templates):
                out[i] = np.sum(tmpl.data[tmpl.slices_cutout] * y_img[tmpl.slices_original])
            return out

        Aop = LinearOperator(
            shape=(flat_size, n_cols),
            matvec=_Av,
            rmatvec=_ATv,
            dtype=float,
        )

        b = (w_sqrt * self.image).ravel()

        atol = cfg.cg_kwargs.get("atol", 1e-6)
        maxit = cfg.cg_kwargs.get("maxiter", 500)
        result = lsqr(Aop, b, atol=atol, btol=atol, iter_lim=maxit, x0=x0)
        x_hat, istop, itn, resid = result[0], result[1], result[2], result[3]

        x_full = np.zeros(self.n_flux, dtype=float)
        idx = [t.col_idx for t in self.templates]
        x_full[idx] = x_hat

        if cfg.positivity:
            x_full = np.maximum(0.0, x_full)

        err_full = self.predicted_errors()

        self.solution = x_full
        self.solution_err = err_full
        for tmpl, f, e in zip(self._orig_templates, x_full, err_full):
            tmpl.flux, tmpl.err = f, e

        info = dict(istop=istop, itn=itn, resid_norm=resid)
        return x_full, err_full, info

    def solve_lo(
        self,
        config: FitConfig | None = None,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Alias for :meth:`solve_linear_operator` (LSQR solver)."""
        return self.solve_linear_operator(config=config, x0=x0)

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
        eps_pd = 1e-6 * np.median(A.diagonal())
        A = A + eps_pd * eye(A.shape[0], format="csr")  # ensure PD

        # 0. cheap independent-pixel approximation?
        off = A.copy()
        off.setdiag(0)
        covar_power = np.sqrt((off.data**2).sum()) / A.diagonal().sum()
        if covar_power < 1e-3 or not self.config.fit_covariances:
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
