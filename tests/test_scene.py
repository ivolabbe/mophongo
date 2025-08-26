import numpy as np
import scipy.sparse as sp

from mophongo.scene import Scene, SceneFitter
from mophongo.templates import Template


def test_scene_fitter_basic():
    A = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    image = np.array([1.0, 2.0])
    weights = np.ones_like(image)
    fitter = SceneFitter()
    flux, err = fitter.fit(A, image, weights)
    assert np.allclose(flux, image)
    assert err.shape == (2,)


def _make_template(value: float, bbox):
    data = np.array([[value]], dtype=float)
    tmpl = Template(data, (0, 0), data.shape)
    tmpl.bbox_original = bbox
    return tmpl


def test_scene_split_and_solve():
    t1 = _make_template(1.0, ((0, 0), (0, 0)))
    t2 = _make_template(1.0, ((0, 0), (0, 0)))
    t3 = _make_template(1.0, ((2, 2), (2, 2)))
    image = np.zeros((3, 3), dtype=float)
    weights = np.ones_like(image)
    scenes = Scene.split([t1, t2, t3], image, weights)
    assert len(scenes) == 2

    img = np.array([[1.0, 0.0], [0.0, 2.0]])
    wht = np.ones_like(img)
    tmpl1 = Template(np.array([[1, 0], [0, 0]], dtype=float), (0.999, 0.999), (2, 2))
    tmpl2 = Template(np.array([[0, 0], [0, 1]], dtype=float), (1, 1), (2, 2))
    scene = Scene([tmpl1, tmpl2], img, wht)
    flux, err = scene.solve()
    assert np.allclose(flux, [1.0, 2.0])
    res = scene.add_residuals(np.zeros_like(img))
    model = tmpl1.data * flux[0] + tmpl2.data * flux[1]
    assert np.allclose(res, img - model)

