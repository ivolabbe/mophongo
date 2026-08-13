"""Region-wise convolution of a full image with a PSFRegionMap."""
import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))

import geopandas as gpd
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon

from mophongo.psf_map import PSFRegionMap, convolve_fits
from mophongo.utils import fftconvolve


def _wcs(shape=(64, 64), pscale=1.0 / 3600):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-pscale, pscale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _kernel(sigma):
    y, x = np.mgrid[-6:7, -6:7]
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # float32, the width PSFRegionMap stores its stamps at. The reference
    # convolutions below have to use the same values the map holds, or they
    # differ at float32 kernel precision (~5e-9) and the tight tolerance that
    # makes these region-assignment tests meaningful would have to be dropped.
    return (k / k.sum()).astype(np.float32)


def _two_region_map(wcs, shape=(64, 64)):
    """Left and right halves of the image, each with its own kernel."""
    ny, nx = shape
    halves = []
    for x0, x1 in ((-0.5, nx / 2 - 0.5), (nx / 2 - 0.5, nx - 0.5)):
        corners = [(x0, -0.5), (x1, -0.5), (x1, ny - 0.5), (x0, ny - 0.5)]
        sky = [wcs.all_pix2world(px, py, 0) for px, py in corners]
        halves.append(Polygon([(float(a), float(d)) for a, d in sky]))
    gdf = gpd.GeoDataFrame({"psf_key": [0, 1]}, geometry=halves, crs="EPSG:4326")
    return PSFRegionMap(gdf, psfs=np.stack([_kernel(1.0), _kernel(2.5)]))


def test_convolve_image_applies_the_region_kernel_to_its_own_pixels():
    """Each half must match a full convolution with that half's kernel.

    Away from the seam -- further than the kernel reach -- the region cutout
    plus its buffer contains everything that contributes, so the stitched
    result has to equal the single-kernel convolution pixel for pixel.
    """
    shape = (64, 64)
    wcs = _wcs(shape)
    prm = _two_region_map(wcs, shape)

    rng = np.random.default_rng(3)
    image = rng.normal(size=shape)

    out = prm.convolve_image(image, wcs)
    ref0 = fftconvolve(image, _kernel(1.0), mode="same")
    ref1 = fftconvolve(image, _kernel(2.5), mode="same")

    # interior of each half, clear of the seam and the image edge.
    # atol 1e-7, not 1e-10: PSFRegionMap stores stamps float32, and
    # convolve_image renormalises the kernel it holds, so a float32 kernel
    # whose sum is 1 +/- 5e-9 shifts the output by ~3e-9. The test is about
    # which region's kernel reaches which pixels, and the two kernels differ
    # by O(0.1) -- asserted below -- so this still has four orders of margin
    # on both sides.
    assert np.allclose(out[8:-8, 8:24], ref0[8:-8, 8:24], atol=1e-7)
    assert np.allclose(out[8:-8, 40:-8], ref1[8:-8, 40:-8], atol=1e-7)
    # and the two kernels really differ, so the test is not vacuous
    assert not np.allclose(ref0[8:-8, 40:-8], ref1[8:-8, 40:-8], atol=1e-6)


def test_convolve_image_covers_every_pixel_and_conserves_flux():
    shape = (64, 64)
    wcs = _wcs(shape)
    prm = _two_region_map(wcs, shape)
    image = np.zeros(shape)
    image[20, 10] = 1.0   # in region 0
    image[40, 50] = 1.0   # in region 1

    out = prm.convolve_image(image, wcs)
    # unit-sum kernels, sources far from any edge: flux is preserved
    assert out.sum() == pytest.approx(2.0, rel=1e-6)
    # the two point sources took different widths
    assert out[20, 10] > out[40, 50]


def test_convolve_image_needs_psfs():
    wcs = _wcs()
    prm = _two_region_map(wcs)
    prm.psfs = None
    with pytest.raises(ValueError, match="no psfs"):
        prm.convolve_image(np.zeros((64, 64)), wcs)


def test_convolve_fits_round_trip(tmp_path):
    """File in, file out: same result as the array call, header carried over."""
    shape = (64, 64)
    wcs = _wcs(shape)
    prm = _two_region_map(wcs, shape)
    image = np.zeros(shape)
    image[32, 16] = 1.0

    sci = tmp_path / "sci.fits"
    header = wcs.to_header()
    header["BUNIT"] = "10*nJy"
    fits.writeto(sci, image, header)
    out = tmp_path / "sci_conv.fits"

    convolve_fits(sci, prm, out)

    with fits.open(out) as hdul:
        data, hdr = hdul[0].data, hdul[0].header
    assert data.shape == shape
    assert hdr["CONVNREG"] == 2
    assert hdr["BUNIT"] == "10*nJy"          # input header survives
    assert hdr["CONVMAP"]
    assert np.allclose(data, prm.convolve_image(image, wcs), atol=1e-10)


def test_convolve_fits_accepts_a_geojson_path(tmp_path):
    """The map can be named by file, the common case after a run."""
    shape = (64, 64)
    wcs = _wcs(shape)
    prm = _two_region_map(wcs, shape)
    prm.name = "demo_kernel"
    geojson = tmp_path / "demo_kernel.geojson"
    prm.to_file(geojson)          # writes the .fits stamp sidecar too

    image = np.zeros(shape)
    image[32, 16] = 1.0
    sci = tmp_path / "sci.fits"
    fits.writeto(sci, image, wcs.to_header())
    out = tmp_path / "sci_conv.fits"
    convolve_fits(sci, geojson, out)

    assert np.allclose(fits.getdata(out), prm.convolve_image(image, wcs),
                       atol=1e-10)
