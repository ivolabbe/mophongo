import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mophongo.fit import FitConfig, SparseFitter
from mophongo.templates import Template, Templates
from utils import make_simple_data


def test_flux_and_rms_estimation():
    """SparseFitter.flux_and_rms matches quick flux and error estimates."""
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )

    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())

    flux, err = fitter.flux_and_rms()
    np.testing.assert_allclose(flux, fitter.quick_flux())
    np.testing.assert_allclose(err, fitter.predicted_errors())

    for tmpl in tmpls.templates:
        tmpl.flux = 42.0
    flux2, _ = fitter.flux_and_rms()
    assert np.all(flux2 == 42.0)


def test_solve_scene_matches_global():
    img = np.zeros((6, 6))
    weights = np.ones_like(img)

    t1 = Template(img, (2, 2), (3, 3))
    t1.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t2 = Template(img, (2, 3), (3, 3))
    t2.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t3 = Template(img, (5, 5), (1, 1))
    t3.data[:] = np.array([[1]])

    image = np.zeros_like(img)
    for flux, tmpl in zip([1.0, 2.0, 3.0], [t1, t2, t3]):
        image[tmpl.slices_original] += flux * tmpl.data[tmpl.slices_cutout]

    fitter_all = SparseFitter(
        [
            Template(img, (2, 2), (3, 3)),
            Template(img, (2, 3), (3, 3)),
            Template(img, (5, 5), (1, 1)),
        ],
        image,
        weights,
        FitConfig(),
    )
    fitter_all.templates[0].data[:] = t1.data
    fitter_all.templates[1].data[:] = t2.data
    fitter_all.templates[2].data[:] = t3.data
    flux_all, _, _ = fitter_all.solve()

    fitter_comp = SparseFitter(
        [
            Template(img, (2, 2), (3, 3)),
            Template(img, (2, 3), (3, 3)),
            Template(img, (5, 5), (1, 1)),
        ],
        image,
        weights,
        FitConfig(),
    )
    fitter_comp.templates[0].data[:] = t1.data
    fitter_comp.templates[1].data[:] = t2.data
    fitter_comp.templates[2].data[:] = t3.data
    flux_comp, _, _ = fitter_comp.solve_scene()

    np.testing.assert_allclose(flux_comp, flux_all, rtol=1e-6, atol=1e-6)
