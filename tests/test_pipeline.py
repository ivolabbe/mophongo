import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np

from mophongo.pipeline import run_photometry
from utils import make_simple_data, save_diagnostic_image


def test_pipeline_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth, truth_img = make_simple_data()
    table, resid = run_photometry(images, segmap, catalog, psfs)

    for idx in range(len(psfs)):
        col = f"flux_{idx}"
        assert np.allclose(table[col], truth, rtol=2e-2)
    assert resid.shape[0] == len(images)

    model = images[1] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(fname, truth_img, images[0], images[1], model, resid[1])
    assert fname.exists()
