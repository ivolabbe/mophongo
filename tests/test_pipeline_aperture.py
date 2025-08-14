import numpy as np
import pytest
from astropy.table import Table
from mophongo.pipeline import Pipeline
from mophongo.templates import Template


def test_aperture_photometry_helper():
    pl = Pipeline([np.zeros((5, 5))], np.zeros((5, 5)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({'id': [1]})
    tmpl = Template(np.ones((5, 5)), (2, 2), (5, 5), label=1)
    residual = np.zeros((5, 5))
    pl._add_aperture_photometry(cat, [tmpl], np.array([1.0]), residual, np.ones((5, 5)), 1)
    assert 'ap_flux_1' in cat.colnames
    assert 'ap_corr_1' in cat.colnames
    assert np.isfinite(cat['ap_flux_1'][0])
