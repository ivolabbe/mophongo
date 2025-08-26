from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
import logging
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import LinearOperator

from .templates import Template

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


def _bbox_overlap(a: Tuple[int, int, int, int],
                  b: Tuple[int, int, int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def build_overlap_graph(templates: Sequence[Template]) -> csr_matrix:
    """Construct an adjacency matrix encoding template overlap.

    Two templates are considered connected if their bounding boxes overlap.
    The returned matrix contains ones for overlapping pairs and zeros elsewhere.
    """
    n = len(templates)
    rows: List[int] = []
    cols: List[int] = []
    for i in range(n):
        bbox_i = templates[i].bbox
        for j in range(i + 1, n):
            if _bbox_overlap(bbox_i, templates[j].bbox):
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=int)
    mat = coo_matrix((data, (rows, cols)), shape=(n, n))
    mat = mat + mat.T
    return mat.tocsr()


def partition_scenes(adj: csr_matrix) -> List[List[int]]:
    """Return connected components of an adjacency matrix."""
    n_comp, labels = csgraph.connected_components(adj)
    groups: List[List[int]] = [[] for _ in range(n_comp)]
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)
    return groups


def merge_small_scenes(scenes: List[List[int]],
                       min_size: int,
                       max_size: int,
                       strategy: str | None = None) -> List[List[int]]:
    """Placeholder that currently returns ``scenes`` unchanged."""
    return [sc.copy() for sc in scenes]


def build_scene_tree_from_normal(templates: Sequence[Template],
                                 image: np.ndarray,
                                 weight: np.ndarray | None = None) -> List["Scene"]:
    """Partition ``templates`` into independent :class:`Scene` objects."""
    adj = build_overlap_graph(templates)
    groups = partition_scenes(adj)
    scenes: List[Scene] = []
    for sid, g in enumerate(groups):
        ts = [templates[i] for i in g]
        bbox = _bbox_union(ts)
        scenes.append(Scene(id=sid, templates=ts, bbox=bbox))
    return scenes


@dataclass
class Scene:
    """Container for a group of overlapping templates."""

    id: int
    templates: List[Template]
    bbox: Tuple[int, int, int, int]
    shift_basis: Optional[object] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        y0, y1, x0, x1 = self.bbox
        self._bbox_slices = (slice(y0, y1), slice(x0, x1))
        self._image: np.ndarray | None = None
        self._weight: np.ndarray | None = None
        self._solution = None

    @property
    def bbox_slices(self) -> Tuple[slice, slice]:
        return self._bbox_slices

    def set_band(self,
                 image: np.ndarray,
                 weight: np.ndarray | None = None,
                 psf: np.ndarray | None = None,
                 config: Optional[object] = None) -> None:
        """Cache per-band data for this scene."""
        self._image = image[self._bbox_slices]
        if weight is None:
            self._weight = np.ones_like(self._image)
        else:
            self._weight = weight[self._bbox_slices]
        self._psf = psf
        self._cfg = config

    def build_operator(self) -> LinearOperator:
        """Return a whitened linear operator for the scene."""
        if self._image is None or self._weight is None:
            raise RuntimeError("Band data has not been set")
        ny, nx = self._image.shape
        n_pix = ny * nx
        n_tmpl = len(self.templates)
        w = np.sqrt(self._weight).reshape(-1)
        mat = np.zeros((n_pix, n_tmpl), dtype=float)
        for j, tmpl in enumerate(self.templates):
            tmpl_data = tmpl.data[tmpl.slices_cutout]
            mat[:, j] = (tmpl_data * w.reshape(ny, nx)).ravel()
        self._operator_matrix = mat

        def mv(x: np.ndarray) -> np.ndarray:
            return mat @ x

        def rmv(y: np.ndarray) -> np.ndarray:
            return mat.T @ y

        op = LinearOperator((n_pix, n_tmpl), matvec=mv, rmatvec=rmv)
        op.matrix = mat  # stash for diagnostics
        return op

    def solve(self, fitter: "SceneFitter", config: Optional[object] = None):
        """Solve for template fluxes using ``fitter``."""
        if self._image is None or self._weight is None:
            raise RuntimeError("Band data has not been set")
        A = self.build_operator()
        b = (self._image * np.sqrt(self._weight)).ravel()
        sol = fitter.solve(A, b, config=config)
        self._solution = sol
        self.meta.update(getattr(sol, "info", {}))
        return sol

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

    def plot(self, *args, **kwargs) -> None:
        """Placeholder for diagnostic plotting."""
        return None
