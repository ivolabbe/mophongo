#%%
import os
import numpy as np
from mophongo.catalog import Catalog
from astropy.io import fits
from utils import make_simple_data
import pytest

def test_find_stars_basic():
    print('basic test_find_stars')
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=0, nsrc=10, size=101)
    cat = Catalog(images[0], wht[0])
    cat.ivar = wht[0]
    stars = cat.find_stars(psf=psfs[0], sigma=1.0, r50_max_pix=10.0, elong_max=5.0, sharp_lohi=(0,2))
    assert len(stars) > 0
    assert {"x", "y", "flux", "snr"}.issubset(stars.colnames)
    assert np.all(np.isfinite(stars["snr"]))


@pytest.mark.skipif(not os.path.exists("../data/uds-test-f444w_sci.fits"), reason="Optional data missing")
def test_find_stars_real():
    print('Running test_find_stars_basic with real data')
    sci = fits.getdata("../data/uds-test-f444w_sci.fits")
    wht = fits.getdata("../data/uds-test-f444w_wht.fits")    

    psfs = [fits.getdata("../data/PSF/")]
    cat = Catalog(sci, wht)

    stars = cat.find_stars(psf=psfs[0], sigma=1.0, r50_max_pix=10.0, elong_max=5.0, sharp_lohi=(0,2))

    print(stars)
#cutout_data, obj, bg_level = clean_stamp(cutout.data, verbose=verbose, imshow=verbose)

# %%
