import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mophongo.pipeline as pipeline
from utils import make_simple_data
from mophongo.fit import FitConfig
import mophongo.utils as mutils


def test_pipeline_plot_result():
    images, segmap, catalog, psfs, _, wht = make_simple_data(nsrc=3, size=51)
    kernel = [mutils.matching_kernel(psfs[0], p) for p in psfs]
    kernel[0] = np.array([[1.0]])
    pl = pipeline.Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=wht,
        kernels=kernel,
        config=FitConfig(fit_astrometry_niter=0),
    )
    pl.run()
    fig, ax = pl.plot_result(idx=1)
    assert fig is not None
    plt.close(fig)
    fig2, ax2 = pl.plot_result(idx=1, source_id=int(catalog["id"][0]))
    assert fig2 is not None
    plt.close(fig2)
    fig3, ax3 = pl.plot_result(idx=1, scene_id=0)
    assert fig3 is not None
    plt.close(fig3)
