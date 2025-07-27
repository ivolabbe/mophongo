import sys
import os

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))

import numpy as np
from mophongo import pipeline
from mophongo.fit import FitConfig
from utils import make_simple_data


def test_global_astrometry_reduces_residual(tmp_path):
    images, segmap, catalog, psfs, truth, wht = make_simple_data()

    tab0, res0, _ = pipeline.run(images, segmap, catalog=catalog, psfs=psfs, weights=wht)

    tab1, res1, _ = pipeline.run(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=wht,
        fit_astrometry=True,
        astrom_order=3,
    )

    assert res1[1].std() < 0.8 * res0[1].std()
