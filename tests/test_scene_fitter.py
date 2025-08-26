import numpy as np
import scipy.sparse as sp

from mophongo.scene_fitter import SceneFitter
from mophongo.scene import Scene
from mophongo.templates import Template


def test_scene_fitter_flux_only():
    A = sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    alpha, err, beta, info = SceneFitter.solve(A, b)
    expected = np.linalg.solve(A.toarray(), b)
    cov = np.linalg.inv(A.toarray())
    expected_err = np.sqrt(np.diag(cov))
    assert info == 0
    assert beta is None
    np.testing.assert_allclose(alpha, expected)
    np.testing.assert_allclose(err, expected_err)


def test_scene_fitter_with_shift_block():
    A = sp.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    AB = sp.csr_matrix([[1.0], [2.0]])
    BB = sp.csr_matrix([[3.0]])
    bB = np.array([0.5])
    alpha, err, beta, info = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB)
    M = np.block([[A.toarray(), AB.toarray()], [AB.T.toarray(), BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    np.testing.assert_allclose(alpha, dense[:2])
    np.testing.assert_allclose(beta, dense[2:])


def test_scene_graph_and_residuals():
    img = np.zeros((10, 10))
    size = (3, 3)
    t1 = Template(img, (2, 2), size, label=1)
    t2 = Template(img, (2, 3), size, label=2)
    t3 = Template(img, (7, 7), size, label=3)
    scene_labels = Scene.create_scene_graph([t1, t2, t3])
    assert scene_labels[t1.id - 1] == scene_labels[t2.id - 1]
    assert scene_labels[t1.id - 1] != scene_labels[t3.id - 1]
    seg, labels = Scene.overlay_scene_graph([t1, t2, t3], img.shape)
    assert seg[t1.bbox[0], t1.bbox[2]] == seg[t2.bbox[0], t2.bbox[2]]
    assert seg[t3.bbox[0], t3.bbox[2]] != seg[t1.bbox[0], t1.bbox[2]]
    # Residuals
    for tmpl in (t1, t2, t3):
        tmpl.data[...] = 1.0
    sc = Scene([t1, t2], SceneFitter())
    coeffs = np.array([2.0, 3.0])
    res = sc.add_residuals(np.zeros_like(img), coeffs)
    expected = np.zeros_like(img)
    expected[t1.slices_original] -= 2.0
    expected[t2.slices_original] -= 3.0
    np.testing.assert_array_almost_equal(res, expected)
