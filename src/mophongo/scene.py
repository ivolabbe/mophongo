from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from .templates import Template
from .scene_fitter import SceneFitter

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Container for templates belonging to a single scene."""

    templates: List[Template]
    fitter: SceneFitter
    shift_basis: np.ndarray | None = None
    bbox: Tuple[int, int, int, int] | None = None

    def solve(
        self,
        A: sp.spmatrix,
        b: np.ndarray,
        *,
        AB: sp.spmatrix | None = None,
        BB: sp.spmatrix | None = None,
        bB: np.ndarray | None = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Solve this scene's normal equations."""
        return self.fitter.solve(A, b, AB=AB, BB=BB, bB=bB, **kwargs)

    def add_residuals(self, image: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Subtract model contributions from ``image`` in-place."""
        for c, tmpl in zip(coeffs, self.templates):
            sl = tmpl.slices_original
            cut = tmpl.data[tmpl.slices_cutout]
            image[sl] -= c * cut
        return image

    @staticmethod
    def create_scene_graph(templates: List[Template]) -> np.ndarray:
        """Return connected-component labels for ``templates``."""
        n = len(templates)
        adj = sp.lil_matrix((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if Scene._overlaps(templates[i].bbox, templates[j].bbox):
                    adj[i, j] = 1
                    adj[j, i] = 1
        labels = sp.csgraph.connected_components(adj.tocsr(), directed=False)[1]
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
