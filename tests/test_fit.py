import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.fit import Template, FitConfig, SparseFitter


def test_flux_recovery():
    img = np.zeros((4, 4), dtype=float)
    weights = np.ones_like(img)

    flux1, flux2 = 1.5, 2.5
    img[1, 1] = flux1
    img[2, 2] = flux2

    t1 = Template(data=np.array([[1.0]]), bbox=(slice(1, 2), slice(1, 2)))
    t2 = Template(data=np.array([[1.0]]), bbox=(slice(2, 3), slice(2, 3)))

    fitter = SparseFitter([t1, t2], img, weights, config=FitConfig())
    fitter.build_normal_matrix()
    x, info = fitter.solve()

    assert info == 0
    assert np.allclose(x, [flux1, flux2])
    model = fitter.model_image()
    assert np.allclose(model, img)


def test_ata_symmetry():
    img = np.zeros((3, 3), dtype=float)
    weights = np.ones_like(img)

    t1 = Template(data=np.ones((2, 2)), bbox=(slice(0, 2), slice(0, 2)))
    t2 = Template(data=np.ones((2, 2)), bbox=(slice(1, 3), slice(1, 3)))

    fitter = SparseFitter([t1, t2], img, weights, config=FitConfig())
    fitter.build_normal_matrix()
    ata = fitter.ata.toarray()
    assert np.allclose(ata, ata.T)
