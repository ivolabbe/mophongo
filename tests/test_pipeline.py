import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np

from mophongo.pipeline import run_photometry
from utils import make_simple_data


def test_pipeline_flux_recovery():
    images, segmap, catalog, psfs, truth = make_simple_data()
    table, resid = run_photometry(images, segmap, catalog, psfs)

    for idx in range(len(psfs)):
        col = f"flux_{idx}"
        assert np.allclose(table[col], truth, rtol=1e-2)
    assert resid.shape[0] == len(images)
