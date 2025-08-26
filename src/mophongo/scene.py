from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
import logging
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse import csgraph

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


def _intersection(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int] | None:
    """Return the intersection of two integer bounding boxes."""
    y0 = max(a[0], b[0])
    y1 = min(a[1], b[1])
    x0 = max(a[2], b[2])
    x1 = min(a[3], b[3])
    if y0 >= y1 or x0 >= x1:
        return None
    return y0, y1, x0, x1


def _bbox_to_slices(bbox: Tuple[int, int, int, int]) -> Tuple[slice, slice]:
    """Convert integer bounding box to slices for array indexing."""
    y0, y1, x0, x1 = bbox
    return slice(y0, y1), slice(x0, x1)


def _slice_intersection(
    a: tuple[slice, slice],
    b: tuple[slice, slice],
) -> tuple[slice, slice] | None:
    """Return the intersection between two slice pairs."""
    y0 = max(a[0].start, b[0].start)
    y1 = min(a[0].stop, b[0].stop)
    x0 = max(a[1].start, b[1].start)
    x1 = min(a[1].stop, b[1].stop)
    if y0 >= y1 or x0 >= x1:
        return None
    return slice(y0, y1), slice(x0, x1)


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
        self._psf = psf

    # ------------------------------------------------------------------
    # System assembly and solving
    # ------------------------------------------------------------------
    def assemble_system(
        self,
    ) -> tuple[csr_matrix, np.ndarray, csr_matrix | None, csr_matrix | None, np.ndarray | None]:
        """Assemble the normal equations for this scene.

        Returns ``(A, b, AB, BB, bB)`` where ``A`` and ``b`` describe the
        flux parameters.  The remaining terms are ``None`` placeholders for
        optional shift parameters handled elsewhere.
        """
        if self._image is None or self._weight is None:
            raise RuntimeError("Band data has not been set")

        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        for i, tmpl in enumerate(self.templates):
            sl_i = tmpl.slices_original
            data_i = tmpl.data[tmpl.slices_cutout]
            w_i = self._weight[sl_i]
            img_i = self._image[sl_i]
            atb[i] = np.sum(data_i * w_i * img_i)
            ata[i, i] = np.sum(data_i * w_i * data_i)
            for j in range(i + 1, n):
                sl_j = self.templates[j].slices_original
                inter = _slice_intersection(sl_i, sl_j)
                if inter is None:
                    continue
                w = self._weight[inter]
                sl_i_local = (
                    slice(
                        inter[0].start - sl_i[0].start + tmpl.slices_cutout[0].start,
                        inter[0].stop - sl_i[0].start + tmpl.slices_cutout[0].start,
                    ),
                    slice(
                        inter[1].start - sl_i[1].start + tmpl.slices_cutout[1].start,
                        inter[1].stop - sl_i[1].start + tmpl.slices_cutout[1].start,
                    ),
                )
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
                arr_i = tmpl.data[sl_i_local]
                arr_j = self.templates[j].data[sl_j_local]
                val = np.sum(arr_i * arr_j * w)
                ata[i, j] = val
                ata[j, i] = val

        A = ata.tocsr()
        b = atb
        # Shift terms are constructed externally; return ``None`` placeholders.
        return A, b, None, None, None

    def solve(self, fitter: "SceneFitter", config: Optional[object] = None):
        """Solve for template fluxes using ``fitter``."""
        A, b, AB, BB, bB = self.assemble_system()
        sol = fitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=config)
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
