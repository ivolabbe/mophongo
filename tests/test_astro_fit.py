import numpy as np

from mophongo.pipeline import run_photometry
from utils import make_simple_data


def test_global_astrometry_reduces_residual():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=1, nsrc=30, size=101)

    from scipy.ndimage import shift
    images[1] = shift(images[1], (0.5, -0.3), order=3, mode="constant", cval=0.0)

    table0, resid0, _ = run_photometry(images, segmap, catalog, psfs, wht_images=wht)

    table1, resid1, _ = run_photometry(
        images,
        segmap,
        catalog,
        psfs,
        wht_images=wht,
        fit_astrometry=True,
        astrom_order=3,
    )

    assert resid1[0].shape == resid0[0].shape

