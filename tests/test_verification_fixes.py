"""Regression tests for the code bugs surfaced by the 2026-08-11 docs
verification pass: NaN-weight masking, double background subtraction,
PSF.gaussian defaults, PSFRegionMap pickling, and id_scene assignment."""

import copy
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from shapely.geometry import box

from mophongo.catalog import Catalog
from mophongo.fit import FitConfig, SparseFitter
from mophongo.psf import PSF
from mophongo.psf_map import PSFRegionMap
from mophongo.scene import generate_scenes
from mophongo.templates import Templates
from utils import make_simple_data


def _fitted_templates():
    images, segmap, catalog, _psfs, _truth, rms = make_simple_data()
    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )
    weights = 1.0 / rms[1] ** 2
    return tmpls.templates, images[1], weights


def test_model_image_zeroes_nan_weight_pixels():
    """`weights <= 0 | isnan(weights)` used to parse as `weights <= (0|isnan)`,
    leaving NaN-weight pixels unmasked in the model image."""
    templates, image, weights = _fitted_templates()
    weights = weights.copy()
    weights[10:20, 10:20] = np.nan
    weights[30:40, 30:40] = -1.0

    fitter = SparseFitter(templates, image, weights, FitConfig())
    fitter.solution = np.ones(len(templates))

    model = fitter.model_image()
    assert np.all(model[10:20, 10:20] == 0.0)
    assert np.all(model[30:40, 30:40] == 0.0)


def test_detect_subtracts_background_once():
    """With estimate_background=True, run() subtracts the background from
    self.sci; _detect must not subtract it a second time."""
    images, _segmap, _catalog, _psfs, _truth, _rms = make_simple_data()
    pedestal = 5.0
    sci = images[0] + pedestal

    cat = Catalog(sci=sci, wht=np.ones_like(sci), estimate_background=True)
    # simulate what run() has done before calling _detect
    cat.background = np.full_like(sci, pedestal)
    cat.sci = sci - cat.background
    cat.ivar = np.ones_like(sci)
    cat._detect()

    # after the fix the detection image is the once-subtracted science image;
    # the double-subtraction bug gave images[0] - pedestal
    np.testing.assert_allclose(cat.det_img, images[0], rtol=0, atol=1e-10)


def test_detect_still_subtracts_user_background():
    """A user-supplied background level (estimate_background=False) is still
    removed inside _detect."""
    images, _segmap, _catalog, _psfs, _truth, _rms = make_simple_data()
    pedestal = 5.0
    sci = images[0] + pedestal

    cat = Catalog(sci=sci, wht=np.ones_like(sci), background=pedestal)
    cat.ivar = np.ones_like(sci)
    cat._detect()
    np.testing.assert_allclose(cat.det_img, images[0], rtol=0, atol=1e-10)


def test_psf_gaussian_requires_fwhm():
    with pytest.raises(ValueError, match="fwhm"):
        PSF.gaussian(31)


def test_psf_gaussian_fwhm_forms():
    sym = PSF.gaussian(31, 3.0).array
    pair = PSF.gaussian(31, 3.0, 3.0).array
    tup = PSF.gaussian(31, (3.0, 3.0)).array
    np.testing.assert_allclose(sym, pair)
    np.testing.assert_allclose(sym, tup)
    # elliptical: second positional argument is fwhm_y, not theta
    ell = PSF.gaussian(31, 2.0, 6.0).array
    assert ell[15, 13] != pytest.approx(ell[13, 15])


def test_psf_region_map_pickles_and_deepcopies():
    prm = PSFRegionMap.from_footprints(
        {"a": box(0, 0, 1, 1), "b": box(0.5, 0.5, 1.5, 1.5)}
    )
    prm.psfs = np.ones((len(prm.regions), 8, 8)) / 64.0

    for clone in (pickle.loads(pickle.dumps(prm)), copy.deepcopy(prm)):
        key = clone.lookup_key(0.25, 0.25)
        assert key == prm.lookup_key(0.25, 0.25)
        np.testing.assert_allclose(clone.get_psf(0.25, 0.25), prm.get_psf(0.25, 0.25))


def test_astrom_kwargs_not_mutated_by_astrocorrect_fit():
    from mophongo.astrometry import AstroCorrect

    templates, image, weights = _fitted_templates()
    cfg = FitConfig(astrom_model="poly")
    before = {k: dict(v) for k, v in cfg.astrom_kwargs.items()}

    astro = AstroCorrect(cfg)
    coeffs = np.ones(len(templates))
    astro.fit(templates, image * 0.0, coeffs)
    assert cfg.astrom_kwargs == before


def test_generate_scenes_assigns_id_scene():
    templates, image, weights = _fitted_templates()
    scenes, _labels = generate_scenes(
        templates, image, weights, coupling_thresh=1e-3, minimum_bright=3
    )
    assert len(scenes) > 0
    for s in scenes:
        for t in s.templates:
            assert t.id_scene == int(s.id)
    ids = {int(s.id) for s in scenes}
    assert len(ids) == len(scenes)
