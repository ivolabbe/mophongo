from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, minres

from .fit import _diag_inv_hutch
from .templates import Template

logger = logging.getLogger(__name__)


class SceneFitter:
    """Minimal solver for normal equations using conjugate gradient.

    The fitter whitens the input design matrix and data using the provided
    weights, solves the normal equations with ``cg`` and returns fluxes and
    1-sigma uncertainties.  Optionally, a separate shift block can be supplied
    which will be solved jointly with the fluxes.
    """

    def fit(
        self,
        A: sp.spmatrix,
        data: np.ndarray,
        weights: np.ndarray,
        shift: sp.spmatrix | None = None,
        *,
        rtol: float = 1e-6,
        maxiter: int = 500,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for template amplitudes.

        Parameters
        ----------
        A:
            Design matrix with one column per template.
        data:
            Observed image data (flattened or 2-D).
        weights:
            Inverse-variance weights with the same shape as ``data``.
        shift:
            Optional shift basis matrix with one column per shift parameter.
        rtol:
            Relative tolerance for the iterative solver.
        maxiter:
            Maximum number of iterations for ``cg``.

        Returns
        -------
        flux, err[, shift_params]
            Best-fit fluxes and their uncertainties.  If ``shift`` is
            provided the array of shift parameters is returned as a third
            element.
        """

        b = np.asarray(data, dtype=float).ravel()
        w = np.sqrt(np.asarray(weights, dtype=float).ravel())
        if A.shape[0] != b.size:
            raise ValueError("Design matrix and data have incompatible shapes")

        Aw = sp.diags(w) @ A
        bw = w * b

        if shift is not None:
            Sw = sp.diags(w) @ shift
            n_flux = Aw.shape[1]
            n_shift = Sw.shape[1]

            def matvec(vec: np.ndarray) -> np.ndarray:
                x = vec[:n_flux]
                beta = vec[n_flux:]
                y = Aw @ x + Sw @ beta
                return np.concatenate([Aw.T @ y, Sw.T @ y])

            rhs = np.concatenate([Aw.T @ bw, Sw.T @ bw])
            N = LinearOperator((n_flux + n_shift, n_flux + n_shift), matvec=matvec)
            sol, info = cg(N, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
            if info > 0:
                sol, info = minres(N, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
            if info > 0:
                logger.warning("CG did not converge: info=%s", info)
            diag = _diag_inv_hutch(N)
            return sol[:n_flux], diag[:n_flux], sol[n_flux:]

        def matvec(vec: np.ndarray) -> np.ndarray:
            return Aw.T @ (Aw @ vec)

        rhs = Aw.T @ bw
        N = LinearOperator((Aw.shape[1], Aw.shape[1]), matvec=matvec)
        sol, info = cg(N, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
        if info > 0:
            sol, info = minres(N, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
        if info > 0:
            logger.warning("CG did not converge: info=%s", info)
        diag = _diag_inv_hutch(N)
        return sol, diag


@dataclass
class Scene:
    """Container for the data needed to solve a photometric scene."""

    templates: List[Template]
    image: np.ndarray
    weights: np.ndarray
    fitter: SceneFitter = field(default_factory=SceneFitter)
    shift: sp.spmatrix | None = None
    bbox: Tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.bbox is None and self.templates:
            y0 = min(t.bbox[0] for t in self.templates)
            y1 = max(t.bbox[1] for t in self.templates)
            x0 = min(t.bbox[2] for t in self.templates)
            x1 = max(t.bbox[3] for t in self.templates)
            self.bbox = (y0, y1, x0, x1)

    def solve(
        self, *, rtol: float = 1e-6, maxiter: int = 500
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the scene for template fluxes."""

        A = np.column_stack([t.data.ravel() for t in self.templates])
        A = sp.csr_matrix(A)
        result = self.fitter.fit(A, self.image, self.weights, self.shift, rtol=rtol, maxiter=maxiter)
        flux = result[0]
        err = result[1]
        shift_params = result[2] if len(result) > 2 else None
        for t, f, e in zip(self.templates, flux, err):
            t.flux = float(f)
            t.err = float(e)
        if shift_params is not None:
            return flux, err, shift_params
        return flux, err

    def add_residuals(self, residual: np.ndarray) -> np.ndarray:
        """Add model residuals to an array."""

        model = np.zeros_like(self.image, dtype=float)
        for t in self.templates:
            model += t.flux * t.data
        residual += self.image - model
        return residual

    @staticmethod
    def build_scene_graph(templates: List[Template]) -> dict[int, set[int]]:
        """Build adjacency graph based on template bounding-box overlap."""

        graph: dict[int, set[int]] = {i: set() for i in range(len(templates))}
        for i, t1 in enumerate(templates):
            y0, y1, x0, x1 = t1.bbox
            for j in range(i + 1, len(templates)):
                t2 = templates[j]
                y2, Y2, x2, X2 = t2.bbox
                if not (x1 <= x2 or X2 <= x0 or y1 <= y2 or Y2 <= y0):
                    graph[i].add(j)
                    graph[j].add(i)
        return graph

    @staticmethod
    def split(
        templates: List[Template], image: np.ndarray, weights: np.ndarray
    ) -> List["Scene"]:
        """Split templates into independent scenes."""

        graph = Scene.build_scene_graph(templates)
        seen: set[int] = set()
        scenes: List[Scene] = []
        for i in range(len(templates)):
            if i in seen:
                continue
            stack = [i]
            component: list[int] = []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                component.append(u)
                stack.extend(graph[u] - seen)
            subset = [templates[k] for k in component]
            scenes.append(Scene(subset, image, weights))
        return scenes

    def plot(self, ax=None):  # pragma: no cover - simple visualisation
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        for tmpl in self.templates:
            y0, y1, x0, x1 = tmpl.bbox
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="r")
            ax.add_patch(rect)
        return ax
