from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import logging
import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import cg, splu
from tqdm import tqdm

from .templates import Template

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
    reg: float = 0.0
    cg_kwargs: Dict[str, Any] = field(default_factory=dict)
    fit_astrometry: bool = False
    fit_astrometry_niter: int = 2     # Two passes for astrometry fitting
    astrom_basis_order: int = 1
    fit_astrometry_joint: bool = False  # Use joint astrometry fitting, or separate step
    reg_astrom: float = 1e-4 
    snr_thresh_astrom: float = 10.0   # 0 → keep all sources (current behaviour)

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
        self.templates = templates
        self.image = image
        self.weights = weights
        self.config = config or FitConfig()
        self._ata = None
        self._atb = None
        self.solution: np.ndarray | None = None

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
    def _slice_intersection(a: tuple[slice, slice], b: tuple[slice, slice]) -> tuple[slice, slice] | None:
        y0 = max(a[0].start, b[0].start)
        y1 = min(a[0].stop, b[0].stop)
        x0 = max(a[1].start, b[1].start)
        x1 = min(a[1].stop, b[1].stop)
        if y0 >= y1 or x0 >= x1:
            return None
        return slice(y0, y1), slice(x0, x1)

    def build_normal_matrix(self) -> None:
        """Construct normal matrix using :class:`Template` objects."""
        # Drop templates that have zero weighted power
        valid: list[Template] = []
        norms: list[float] = []
        for tmpl in self.templates:
            sl = tmpl.slices_original
            data = tmpl.data[tmpl.slices_cutout]
            w = self.weights[sl]
            norm = float(np.sum(data * w * data))
            if norm == 0:
                logger.warning("Dropping template with zero weight")
                continue
            norms.append(norm)
            valid.append(tmpl)

        self.templates = valid

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
                inter = self._slice_intersection(tmpl_i.slices_original, tmpl_j.slices_original)
                if inter is None:
                    continue
                w = self.weights[inter]
                sl_i_local = (
                    slice(inter[0].start - sl_i[0].start + tmpl_i.slices_cutout[0].start,
                          inter[0].stop - sl_i[0].start + tmpl_i.slices_cutout[0].start),
                    slice(inter[1].start - sl_i[1].start + tmpl_i.slices_cutout[1].start,
                          inter[1].stop - sl_i[1].start + tmpl_i.slices_cutout[1].start),
                )
                sl_j_local = (
                    slice(inter[0].start - tmpl_j.slices_original[0].start + tmpl_j.slices_cutout[0].start,
                          inter[0].stop - tmpl_j.slices_original[0].start + tmpl_j.slices_cutout[0].start),
                    slice(inter[1].start - tmpl_j.slices_original[1].start + tmpl_j.slices_cutout[1].start,
                          inter[1].stop - tmpl_j.slices_original[1].start + tmpl_j.slices_cutout[1].start),
                )
                val = np.sum(
                    tmpl_i.data[sl_i_local] * tmpl_j.data[sl_j_local] * w
                )
                if val != 0.0:
                    ata[i, j] = val
                    ata[j, i] = val

        self._ata = ata.tocsr()
        self._atb = atb

    def model_image(self) -> np.ndarray:
        if self.solution is None:
            raise ValueError("Solve system first")
        model = np.zeros_like(self.image, dtype=float)
        for coeff, tmpl in zip(self.solution, self.templates):
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
    ) -> Tuple[np.ndarray, int]:
        """Solve for template fluxes using conjugate gradient."""
        cfg = config or self.config
        ata = self.ata
        atb = self.atb
        if cfg.reg and cfg.reg > 0:
            ata = ata + eye(ata.shape[0]) * cfg.reg
        x, info = cg(ata, atb, **cfg.cg_kwargs)
        if cfg.positivity:
            x = np.where(x < 0, 0, x)
        self.solution = x

        for i, t in enumerate(self.templates):  # Store the solution in each template
            t.flux = self.solution[i] 

        return x, info

    def residual(self) -> np.ndarray:
        return self.image - self.model_image()

    def quick_flux(self) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        flux = np.zeros(len(self.templates), dtype=float)
        for i, tmpl in enumerate(self.templates):
            tt = tmpl.data[tmpl.slices_cutout]
            img = self.image[tmpl.slices_original]
            ttsqs = np.sum(tt ** 2)
            flux[i] = np.sum(img * tt) / ttsqs if ttsqs > 0 else 0.0
            tmpl.quick_flux = flux[i]  # Store quick flux in the template for later use
        return flux
    
    def predicted_errors(self) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        pred = np.empty(len(self.templates), dtype=float)
        for i, tmpl in enumerate(self.templates):
            w = self.weights[tmpl.slices_original]
            pred[i] = 1.0 / np.sqrt(np.sum(w * tmpl.data[tmpl.slices_cutout] ** 2))
            tmpl.pred_err = pred[i]  # Store RMS in the template for later use
        return pred

    def flux_errors(self) -> np.ndarray:
        """Return 1-sigma uncertainties for the fitted fluxes."""
        if self.solution is None:
            raise ValueError("Solve system first")
        ata = self.ata.tocsc()
        # make positive definite to avoid singularities
        ata = ata + 1e-8 * eye(ata.shape[0], format="csc")
        try:
            solver = splu(ata)
        except RuntimeError:
            logger.warning("Normal matrix singular; using CG fallback for variances")
            return self._cg_variances()

        n = ata.shape[0]
        diag = np.empty(n, dtype=float)
        e = np.zeros(n, dtype=float)
        for i in range(n):
            e[i] = 1.0
            x = solver.solve(e)
            diag[i] = x[i]
            e[i] = 0.0
            self.templates[i].err = np.sqrt(diag[i])
        return np.sqrt(diag)

    def _cg_variances(self) -> np.ndarray:
        """Compute variances using conjugate gradient solves."""
        ata = self.ata
        n = ata.shape[0]
        diag = np.empty(n, dtype=float)
        e = np.zeros(n)
        for i in range(n):
            e[i] = 1.0
            x, _ = cg(ata, e)
            diag[i] = x[i]
            e[i] = 0.0
        return np.sqrt(diag)

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
        fluxes, _ = fitter.solve()
        resid = fitter.residual()
        return fluxes, resid
