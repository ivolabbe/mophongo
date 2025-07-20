import matplotlib
matplotlib.use('Agg')
import numpy as np
from mophongo.utils import (
    gaussian_laguerre_basis,
    difference_of_gaussians_basis,
    radial_bspline_basis,
    powerlaw_basis,
)


def test_basis_properties():
    size = 31

    gl = gaussian_laguerre_basis(1, [0, 2], [2.0], size)
    assert gl.shape[0] == size
    assert np.isclose(gl[:, :, 0].sum(), 1.0, atol=1e-6)
    assert np.allclose(gl[:, :, 1:].sum(axis=(0, 1)), 0.0, atol=1e-6)

    dog = difference_of_gaussians_basis([1.0, 2.0, 4.0], size)
    assert dog.shape[-1] == 2
    assert np.allclose(dog.sum(axis=(0, 1)), 0.0, atol=1e-6)

    bs = radial_bspline_basis(size, n_knots=4)
    assert bs.shape[0] == size
    assert np.isclose(bs[:, :, 0].sum(), 1.0, atol=1e-6)

    pw = powerlaw_basis([0.5, 1.0], size)
    assert pw.shape[-1] == 2
    assert np.allclose(pw.sum(axis=(0, 1)), 0.0, atol=1e-6)


