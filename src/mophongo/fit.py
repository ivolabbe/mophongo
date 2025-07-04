from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import cg


@dataclass
class Template:
    """PSF-matched template cutout and its bounding box."""

    data: np.ndarray
    bbox: Tuple[slice, slice]


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
        a: Tuple[slice, slice], b: Tuple[slice, slice]
    ) -> Tuple[slice, slice] | None:
        y0 = max(a[0].start, b[0].start)
        y1 = min(a[0].stop, b[0].stop)
        x0 = max(a[1].start, b[1].start)
        x1 = min(a[1].stop, b[1].stop)
        if y0 >= y1 or x0 >= x1:
            return None
        return slice(y0, y1), slice(x0, x1)

    def build_normal_matrix(self) -> None:
        """Construct A^T A and A^T b from templates and image."""
        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        for i, tmpl_i in enumerate(self.templates):
            sl_i = tmpl_i.bbox
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            temp_i = tmpl_i.data
            atb[i] = np.sum(temp_i * w_i * img_i)

            ata[i, i] = np.sum(temp_i * w_i * temp_i)

            for j in range(i + 1, n):
                tmpl_j = self.templates[j]
                inter = self._intersection(sl_i, tmpl_j.bbox)
                if inter is None:
                    continue
                sl_i_local = (
                    slice(
                        inter[0].start - sl_i[0].start, inter[0].stop - sl_i[0].start
                    ),
                    slice(
                        inter[1].start - sl_i[1].start, inter[1].stop - sl_i[1].start
                    ),
                )
                sl_j_local = (
                    slice(
                        inter[0].start - tmpl_j.bbox[0].start,
                        inter[0].stop - tmpl_j.bbox[0].start,
                    ),
                    slice(
                        inter[1].start - tmpl_j.bbox[1].start,
                        inter[1].stop - tmpl_j.bbox[1].start,
                    ),
                )
                w = self.weights[inter]
                val = np.sum(tmpl_i.data[sl_i_local] * tmpl_j.data[sl_j_local] * w)
                if val != 0.0:
                    ata[i, j] = val
                    ata[j, i] = val
        self._ata = ata.tocsr()
        self._atb = atb

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

    def model_image(self) -> np.ndarray:
        if self.solution is None:
            raise ValueError("Solve system first")
        model = np.zeros_like(self.image, dtype=float)
        for coeff, tmpl in zip(self.solution, self.templates):
            sl = tmpl.bbox
            model[sl] += coeff * tmpl.data
        return model

    def residual(self) -> np.ndarray:
        return self.image - self.model_image()
