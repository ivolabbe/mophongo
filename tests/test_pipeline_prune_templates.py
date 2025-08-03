import numpy as np
from astropy.table import Table
import mophongo.pipeline as pipeline


def test_pipeline_prunes_templates_with_zero_weight():
    hires = np.zeros((4, 4))
    lowres = np.zeros((4, 4))
    segmap = np.zeros((4, 4), dtype=int)
    segmap[0:2, 0:2] = 1
    segmap[2:4, 2:4] = 2
    hires[segmap > 0] = 1.0

    images = [hires, lowres]
    catalog = Table({"id": [1, 2], "x": [0, 3], "y": [0, 3]})
    w0 = np.ones_like(hires)
    w1 = np.ones_like(lowres)
    w1[2:4, 2:4] = 0
    weights = [w0, w1]
    kernels = [None, None]

    table, residuals, fitter = pipeline.run(images, segmap, catalog=catalog, weights=weights, kernels=kernels)

    assert len(fitter.templates) == 1
    assert np.isfinite(table["flux_1"][0])
    assert np.isnan(table["flux_1"][1])
