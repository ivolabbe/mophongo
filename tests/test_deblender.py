import numpy as np
from photutils.segmentation import detect_sources
from mophongo.deblender import deblend_sources_symmetry, deblend_sources_hybrid
from photutils.segmentation import SegmentationImage


def _make_simple_blend():
    size = 31
    y, x = np.mgrid[0:size, 0:size]
    g1 = np.exp(-((x - 12) ** 2 + (y - 15) ** 2) / 4.0)
    g2 = np.exp(-((x - 18) ** 2 + (y - 15) ** 2) / 4.0)
    img = g1 + g2
    seg = detect_sources(img, 0.1, npixels=5)
    return img, seg


def test_symmetry_deblend_splits_sources():
    img, seg = _make_simple_blend()
    seg_deb = deblend_sources_symmetry(img, seg)
    assert isinstance(seg_deb, SegmentationImage)
    assert seg_deb.nlabels >= 2
    assert seg_deb.data.shape == img.shape


def test_hybrid_deblend_splits_sources():
    img, seg = _make_simple_blend()
    seg_deb = deblend_sources_hybrid(img, seg)
    assert isinstance(seg_deb, SegmentationImage)
    assert seg_deb.nlabels >= 2
    assert seg_deb.data.shape == img.shape
