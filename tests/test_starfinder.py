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
    from mophongo.catalog import Catalog
    from mophongo.utils import clean_stamp    
    from astropy.io import fits
    from matplotlib import pyplot as plt

    print('Running test_find_stars_basic with real data')

    base = '../data/uds-test-f444w'
  
    params = {"kernel_size": 4, "detect_threshold": 3.0, "detect_npixels": 10, "deblend_mode": None}    
    cat = Catalog.from_fits(base+'_sci.fits', base+'_wht.fits', params=params)

    stars, idx = cat.find_stars(snr_min=300, r50_max=3.0, eccen_max=0.7, sharp_lohi=(0.1,1))

    fig,ax = plt.subplots(4, 4, figsize=(20, 20))
    ax = ax.flatten()
    for i,a in enumerate(ax):
        if i >= len(stars): break
        cat.show_stamp(stars['id'][i],keys=['snr','r50','eccentricity','sharpness'],ax=a, cmap='gray', alpha=0.3)

    for col in stars.colnames:
        if stars[col].dtype.kind in 'fc': stars[col].format = '.2f'
    stars['id','x','y','ra','dec','snr','r50','eccentricity','sharpness', 'point_like'].pprint_all()

    cutouts = cat.catalog.make_cutouts((201,201),fill_value=0)
    star_cutouts = [cutouts[j] for j in idx]
    star_clean = clean_stamp(star_cutouts[1].data,imshow=True)

    compare_psf_to_star(cutout_data, psf_data, kernel=kernel, Rnorm=2.0, pscale=pscale, to_file="compare.png")


#cutout_data, obj, bg_level = clean_stamp(cutout.data, verbose=verbose, imshow=verbose)

# %%
