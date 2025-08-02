from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import logging
import numpy as np
from scipy.sparse import lil_matrix, eye, diags, csr_matrix

from scipy.sparse.linalg import cg, spilu, LinearOperator
from tqdm import tqdm

from .templates import Template, Templates

logger = logging.getLogger(__name__)

# full weights need to be calcuate like
#template_var = scipy.signal.fftconvolve(K**2, 1 / wht1, mode='same')  # same shape as template
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
    cg_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"M": None, "maxiter": 500, "atol": 1e-6}
    )
    fit_astrometry: bool = False
    fit_astrometry_niter: int = 2     # Two passes for astrometry fitting
    astrom_basis_order: int = 1
    fit_astrometry_joint: bool = False  # Use joint astrometry fitting, or separate step
    reg_astrom: float = 1e-4
    snr_thresh_astrom: float = 10.0   # 0 → keep all sources (current behaviour)
    astrom_model: str = "polynomial"  # 'polynomial' or 'gp'


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
        self.templates = templates.copy() # work in copy for fitting, modifying 

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


    @staticmethod
    def _intersection(
            a: Tuple[int, int, int, int],
            b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int] | None:
        y0 = max(a[0], b[0])
        y1 = min(a[1], b[1])
        x0 = max(a[2], b[2])
        x1 = min(a[3], b[3])
        if y0 >= y1 or x0 >= x1:
            return None
        return y0, y1, x0, x1

    @staticmethod
    def _bbox_to_slices(
            bbox: Tuple[int, int, int, int]) -> Tuple[slice, slice]:
        """Convert integer bounding box to slices for array indexing."""
        y0, y1, x0, x1 = bbox
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
    def _slice_intersection(
            a: tuple[slice, slice],
            b: tuple[slice, slice]) -> tuple[slice, slice] | None:
        y0 = max(a[0].start, b[0].start)
        y1 = min(a[0].stop, b[0].stop)
        x0 = max(a[1].start, b[1].start)
        x1 = min(a[1].stop, b[1].stop)
        if y0 >= y1 or x0 >= x1:
            return None
        return slice(y0, y1), slice(x0, x1)

    def build_normal_matrix(self) -> None:
        """Construct normal matrix using :class:`Template` objects."""
        # Compute weighted norms for all templates first
        norms_all: list[float] = []
        for tmpl in self.templates:
            sl = tmpl.slices_original
            data = tmpl.data[tmpl.slices_cutout]
            w = self.weights[sl]
            norms_all.append(float(np.sum(data * w * data)))

        tol = 0.0
        if norms_all:
            tol = 1e-12 * max(norms_all)

        # Prune templates with near-zero norm
        valid: list[Template] = []
        norms: list[float] = []
        for tmpl, norm in zip(self.templates, norms_all):
            if norm < tol:
                logger.warning("Dropping template with low norm %.2e", norm)
                continue
            valid.append(tmpl)
            norms.append(norm)

        # Screen for highly correlated templates (duplicates)
        keep: list[int] = []
        data_arrays = [t.data[t.slices_cutout].ravel() for t in valid]
        for i, (arr_i, norm_i) in enumerate(zip(data_arrays, norms)):
            duplicate = False
            for j in keep:
                arr_j = data_arrays[j]
                if arr_j.size != arr_i.size:
                    continue
                cos_ij = float(np.dot(arr_i, arr_j) / (norm_i * norms[j]))
                if cos_ij > 0.999:
                    duplicate = True
                    logger.warning("Dropping nearly duplicate template %d", i)
                    break
            if not duplicate:
                keep.append(i)

        self.templates = [valid[i] for i in keep]
        norms = [norms[i] for i in keep]

        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)
        for i, tmpl_i in enumerate(
                tqdm(self.templates, total=n, desc="Building Normal matrix")):
            sl_i = tmpl_i.slices_original
            data_i = tmpl_i.data[tmpl_i.slices_cutout]
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            atb[i] = np.sum(data_i * w_i * img_i)
            ata[i, i] = norms[i]

            for j in range(i + 1, n):
                tmpl_j = self.templates[j]
                inter = self._slice_intersection(tmpl_i.slices_original,
                                                 tmpl_j.slices_original)
                if inter is None:
                    continue
                w = self.weights[inter]
                sl_i_local = (
                    slice(
                        inter[0].start - sl_i[0].start +
                        tmpl_i.slices_cutout[0].start, inter[0].stop -
                        sl_i[0].start + tmpl_i.slices_cutout[0].start),
                    slice(
                        inter[1].start - sl_i[1].start +
                        tmpl_i.slices_cutout[1].start, inter[1].stop -
                        sl_i[1].start + tmpl_i.slices_cutout[1].start),
                )
                sl_j_local = (
                    slice(
                        inter[0].start - tmpl_j.slices_original[0].start +
                        tmpl_j.slices_cutout[0].start,
                        inter[0].stop - tmpl_j.slices_original[0].start +
                        tmpl_j.slices_cutout[0].start),
                    slice(
                        inter[1].start - tmpl_j.slices_original[1].start +
                        tmpl_j.slices_cutout[1].start,
                        inter[1].stop - tmpl_j.slices_original[1].start +
                        tmpl_j.slices_cutout[1].start),
                )
                val = np.sum(tmpl_i.data[sl_i_local] *
                             tmpl_j.data[sl_j_local] * w)
                if val != 0.0:
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
        return model

    @property
    def ata(self):
        if self._ata is None:
            self.build_normal_matrix()
        return self._ata

    @property
    def atb(self):
        if self._atb is None:
            self.build_normal_matrix()
        return self._atb

    def solve(
        self,
        config: FitConfig | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for template fluxes using conjugate gradient."""
        cfg = config or self.config
        
        # build big normal matrix once, this as a shift entry for every template
        A, b = self.ata, self.atb          # triggers build_normal_matrix()

        reg = cfg.reg
        if reg <= 0:
            reg = 1e-4 * np.median(A.diagonal())
        if reg > 0:
            A = A + eye(A.shape[0], format="csr") * reg

        eps = reg or 1e-10
        d = np.sqrt(np.maximum(A.diagonal(), eps))
        Dinv = diags(1.0 / d, 0, format="csr")

        A_w = Dinv @ A @ Dinv
        b_w = Dinv @ b

        cg_kwargs = dict(cfg.cg_kwargs)
        if cg_kwargs.get("M") is None and A_w.nnz > 10 * A_w.shape[0]:
            try:
                ilu = spilu(A_w.tocsc(), drop_tol=1e-4, fill_factor=10)
                cg_kwargs["M"] = LinearOperator(A_w.shape, ilu.solve)
            except Exception as err:
                logger.warning("ILU preconditioner failed: %s", err)

        # expand to full solution vector corresponding to _orig_templates
        idx = [t.col_idx for t in self.templates]

        y, info = cg(A_w, b_w, **cg_kwargs)
        self.x = y

        # n_flux is always the full input length of the templates
        x_full     = np.zeros(self.n_flux, dtype=float)
        e_full     = np.zeros(self.n_flux, dtype=float)

        x_full[idx] = y / d                   # un-whiten + scatter
        e_full[idx] = self._flux_errors(A_w) / d  # un-whiten errors

        if cfg.positivity:
            x_full[:self.n_flux] = np.maximum(0, x_full[:self.n_flux])

        self.solution = x_full[:self.n_flux]      # fluxes, corresponds to original templates
        self.solution_err = e_full[:self.n_flux]  # flux errors, corresponds to original templates

        # update the templates with the fitted fluxes, errors     
        for tmpl, flux, err in zip(self._orig_templates, self.solution, self.solution_err):
            tmpl.flux = flux 
            tmpl.err = err

        return self.solution, self.solution_err, info

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

    def flux_errors(self) -> np.ndarray:
        """Return the 1-sigma flux uncertainties from the last solution."""
        if self.solution_err is None:
            raise ValueError("Solve system first")
        return self.solution_err

    def _flux_errors(self, A_csr: csr_matrix) -> np.ndarray:
        """Return 1-sigma uncertainties for the fitted fluxes."""
        eps = 1e-8
        A = A_csr + eps * eye(A_csr.shape[0], format="csr")
        try:  # Prefer sparse Cholesky if available
            from sksparse.cholmod import cholesky

            factor = cholesky(A, beta=eps)
            inv_diag = factor.inv().diagonal()
            return np.sqrt(inv_diag)
        except Exception as err:  # pragma: no cover - exercised in fallback
            logger.warning("cholmod failed (%s); falling back to SLQ", err)

        # Hutchinson stochastic trace estimator
        k = 32
        rng = np.random.default_rng(0)
        diag_est = np.zeros(A.shape[0])
        for _ in range(k):
            v = rng.choice([-1.0, 1.0], size=A.shape[0])
            x, _ = cg(A, v, tol=1e-6)
            diag_est += v * x
        diag_est = np.abs(diag_est / k)
        return np.sqrt(diag_est)

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
