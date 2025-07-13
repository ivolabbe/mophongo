import numpy as np
from astropy.io import fits
from mophongo.catalog import deblend_sources_color
from photutils.segmentation import SegmentationImage


def _make_color_blend():
    size = 51
    y, x = np.mgrid[0:size, 0:size]
    sigma = 2.0
    def g(amp, y0, x0):
        return amp * np.exp(-0.5 * ((x - x0)**2 + (y - y0)**2) / sigma**2)
    img1 = g(1.0, 25, 25) + g(0.2, 27, 25)
    img2 = g(0.2, 25, 25) + g(1.0, 27, 25)
    rng = np.random.default_rng(0)
    noise = 0.05
    img1 += rng.normal(scale=noise, size=img1.shape)
    img2 += rng.normal(scale=noise, size=img2.shape)
    ivar = np.ones_like(img1) / noise**2
    return img1, img2, ivar, ivar


def test_color_deblend_simulated():
    im1, im2, iv1, iv2 = _make_color_blend()
    seg = deblend_sources_color(
        im1,
        iv1,
        im2,
        iv2,
        detect_threshold=1.5,
        color_thresh=0.2,
        nsigma=2.0,
    )
    assert isinstance(seg, SegmentationImage)
    assert seg.nlabels >= 2


def test_color_deblend_real():
    sci1 = fits.getdata('data/uds-test-f444w_sci.fits')[100:200, 100:200]
    wht1 = fits.getdata('data/uds-test-f444w_wht.fits')[100:200, 100:200]
    sci2 = fits.getdata('data/uds-test-f770w_sci.fits')[100:200, 100:200]
    wht2 = fits.getdata('data/uds-test-f770w_wht.fits')[100:200, 100:200]
    seg = deblend_sources_color(sci1, wht1, sci2, wht2)
    assert seg.data.shape == sci1.shape
    assert seg.nlabels > 0
