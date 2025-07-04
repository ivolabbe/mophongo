import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.fit import FitConfig, SparseFitter
from mophongo.templates import Template
from utils import save_fit_diagnostic


def test_flux_recovery(tmp_path):
    img = np.zeros((4, 4), dtype=float)
    weights = np.ones_like(img)

    flux1, flux2 = 1.5, 2.5
    img[1, 1] = flux1
    img[2, 2] = flux2

    t1 = Template(array=np.array([[1.0]]), bbox=(1, 2, 1, 2))
    t2 = Template(array=np.array([[1.0]]), bbox=(2, 3, 2, 3))

    fitter = SparseFitter([t1, t2], img, weights, config=FitConfig())
    fitter.build_normal_matrix()
    x, info = fitter.solve()

    assert info == 0
    assert np.allclose(x, [flux1, flux2])
    model = fitter.model_image()
    assert np.allclose(model, img)
    fname = tmp_path / "fit.png"
    save_fit_diagnostic(fname, img, model, fitter.residual())
    assert fname.exists()


def test_ata_symmetry():
    img = np.zeros((3, 3), dtype=float)
    weights = np.ones_like(img)

    t1 = Template(array=np.ones((2, 2)), bbox=(0, 2, 0, 2))
    t2 = Template(array=np.ones((2, 2)), bbox=(1, 3, 1, 3))

    fitter = SparseFitter([t1, t2], img, weights, config=FitConfig())
    fitter.build_normal_matrix()
    ata = fitter.ata.toarray()
    assert np.allclose(ata, ata.T)
