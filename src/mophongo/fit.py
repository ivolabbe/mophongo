from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import cg, splu

from .templates import Template


@dataclass
class FitConfig:
    """Configuration options for :class:`SparseFitter`."""

    positivity: bool = False
    reg: float = 0.0
    cg_kwargs: Dict[str, Any] = field(default_factory=dict)


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

    def build_normal_matrix_old(self) -> None:
        """Construct A^T A and A^T b from templates and image."""
        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        for i, tmpl_i in enumerate(self.templates):
            sl_i = self._bbox_to_slices(tmpl_i.bbox)
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            temp_i = tmpl_i.array
            atb[i] = np.sum(temp_i * w_i * img_i)

            ata[i, i] = np.sum(temp_i * w_i * temp_i)

            for j in range(i + 1, n):
                tmpl_j = self.templates[j]
                inter = self._intersection(tmpl_i.bbox, tmpl_j.bbox)
                if inter is None:
                    continue
                y0, y1, x0, x1 = inter
                sl_inter = self._bbox_to_slices(inter)
                sl_i_local = (
                    slice(y0 - tmpl_i.bbox[0], y1 - tmpl_i.bbox[0]),
                    slice(x0 - tmpl_i.bbox[2], x1 - tmpl_i.bbox[2]),
                )
                sl_j_local = (
                    slice(y0 - tmpl_j.bbox[0], y1 - tmpl_j.bbox[0]),
                    slice(x0 - tmpl_j.bbox[2], x1 - tmpl_j.bbox[2]),
                )
                w = self.weights[sl_inter]
                val = np.sum(tmpl_i.array[sl_i_local] * tmpl_j.array[sl_j_local] * w)
                if val != 0.0:
                    ata[i, j] = val
                    ata[j, i] = val
        self._ata = ata.tocsr()
        self._atb = atb

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
        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        for i, tmpl_i in enumerate(self.templates):
            sl_i = tmpl_i.slices_original
            data_i = tmpl_i.array[tmpl_i.slices_cutout]
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            atb[i] = np.sum(data_i * w_i * img_i)
            ata[i, i] = np.sum(data_i * w_i * data_i)

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
                    tmpl_i.array[sl_i_local] * tmpl_j.array[sl_j_local] * w
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
            model[tmpl.slices_original] += coeff * tmpl.array[tmpl.slices_cutout]
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
        return x, info

    def model_image_old(self) -> np.ndarray:
        if self.solution is None:
            raise ValueError("Solve system first")
        model = np.zeros_like(self.image, dtype=float)
        for coeff, tmpl in zip(self.solution, self.templates):
            sl = self._bbox_to_slices(tmpl.bbox)
            model[sl] += coeff * tmpl.array
        return model

    def residual(self) -> np.ndarray:
        return self.image - self.model_image()

    def predicted_errors(self) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        pred = np.empty(len(self.templates), dtype=float)
        for i, tmpl in enumerate(self.templates):
            w = self.weights[tmpl.slices_original]
            pred[i] = 1.0 / np.sqrt(np.sum(w * tmpl.array[tmpl.slices_cutout] ** 2))
        return pred

    def flux_errors(self) -> np.ndarray:
        """Return 1-sigma uncertainties for the fitted fluxes."""
        if self.solution is None:
            raise ValueError("Solve system first")
        ata = self.ata.tocsc()
        solver = splu(ata)
        n = ata.shape[0]
        diag = np.empty(n, dtype=float)
        e = np.zeros(n, dtype=float)
        for i in range(n):
            e[:] = 0.0
            e[i] = 1.0
            x = solver.solve(e)
            diag[i] = x[i]
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
