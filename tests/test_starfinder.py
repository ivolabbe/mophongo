import numpy as np
from mophongo.catalog import Catalog
from utils import make_simple_data


def test_find_stars_basic():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=0, nsrc=10, size=101)
    cat = Catalog(images[0], wht[0])
    cat.ivar = wht[0]
    stars = cat.find_stars(psf=psfs[0], sigma=1.0, r50_max_pix=10.0, elong_max=5.0, sharp_lohi=(0,2))
    assert len(stars) > 0
    assert {"x", "y", "flux", "snr"}.issubset(stars.colnames)
    assert np.all(np.isfinite(stars["snr"]))

