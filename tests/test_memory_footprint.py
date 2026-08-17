"""Contracts behind the full-field memory pass (STATUS.md, 2026-08-13).

Each test pins a place where an array used to be duplicated. They are cheap
equivalence checks, not memory measurements: the point is that dropping the
duplicate did not change the numbers.
"""

import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

from copy import deepcopy

import numpy as np
import pytest
from astropy.nddata import block_replicate
from astropy.wcs import WCS

import mophongo.pipeline as pipeline
from mophongo.catalog import _mean_downsample, _valid_block_means
from mophongo.templates import Template, Templates


def _wcs(shape: tuple[int, int]) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-1.0e-5, 1.0e-5]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _noisy_field(seed: int = 3, shape: tuple[int, int] = (96, 112)):
    rng = np.random.default_rng(seed)
    sci = rng.normal(scale=1e-3, size=shape).astype(np.float32)
    wht = np.abs(rng.normal(loc=1e6, scale=1e5, size=shape)).astype(np.float32)
    # a blank edge and a bad pixel, so the valid masks actually differ
    wht[:8] = 0.0
    sci[20, 30] = np.nan
    wht[40, 50] = np.nan
    return sci, wht


# -- coarse block means -------------------------------------------------------


def test_valid_block_means_matches_the_masked_whole_array_form():
    sci, wht = _noisy_field()
    valid = np.isfinite(wht) & (wht > 0) & np.isfinite(sci)
    step = 8

    want_v = _mean_downsample(valid.astype(np.float32), step)
    want_s = _mean_downsample(np.where(valid, sci, 0.0).astype(np.float32), step)
    want_w = _mean_downsample(np.where(valid, wht, 0.0).astype(np.float32), step)
    got_v, got_s, got_w = _valid_block_means(sci, wht, valid, step)

    assert np.array_equal(want_v, got_v)
    assert np.array_equal(want_s, got_s)
    assert np.array_equal(want_w, got_w)


def test_valid_block_means_is_independent_of_the_band_height():
    sci, wht = _noisy_field(seed=5)
    valid = np.isfinite(wht) & (wht > 0) & np.isfinite(sci)
    ref = _valid_block_means(sci, wht, valid, 8, band_blocks=1000)
    for band in (1, 3, 4):
        got = _valid_block_means(sci, wht, valid, 8, band_blocks=band)
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)


def test_bg_and_ivar_boxed_passes_need_bg_through():
    sci, wht = _noisy_field(seed=9, shape=(128, 128))
    box = (16, 112, 16, 112)
    bg, ivar = pipeline.Pipeline._bg_and_ivar_boxed(
        sci, wht, box, bg_filter_sigma=64.0, need_bg=False
    )
    assert bg is None
    assert ivar.shape == sci.shape
    assert np.all(ivar[:16] == 0.0)  # outside the box, never calibrated


# -- flux-conserving upsample -------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4])
def test_upsample_matches_conserve_sum_and_stays_float32(k):
    rng = np.random.default_rng(2)
    img = rng.normal(size=(24, 30)).astype(np.float32)
    wht = np.abs(rng.normal(size=(24, 30))).astype(np.float32) * 1e6

    want_img = block_replicate(img, k, conserve_sum=True).astype(np.float32)
    want_wht = block_replicate(wht, k, conserve_sum=False).astype(np.float32) * k**2
    got_img, got_wht = pipeline._upsample_flux_conserving_image_and_ivar(img, wht, k)

    assert got_img.dtype == np.float32 and got_wht.dtype == np.float32
    assert np.array_equal(want_img, got_img)
    assert np.array_equal(want_wht, got_wht)
    assert got_img.sum() == pytest.approx(img.sum(), rel=1e-5)


# -- template copies ----------------------------------------------------------


def test_template_deepcopy_shares_the_parent_wcs_and_copies_everything_else():
    img = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
    w = _wcs(img.shape)
    tmpl = Template(img, (32.0, 32.0), (20, 20), wcs=w, label=4)
    tmpl.flux, tmpl.ee_psf_lo = 2.0, 0.9
    tmpl.to_shift[:] = [0.5, -0.25]

    copy = deepcopy(tmpl)

    assert copy.wcs_original is tmpl.wcs_original  # one parent WCS per set
    assert copy.wcs is not tmpl.wcs
    assert copy.data is not tmpl.data and np.array_equal(copy.data, tmpl.data)
    assert copy.to_shift is not tmpl.to_shift
    assert np.array_equal(copy.to_shift, tmpl.to_shift)
    assert (copy.id, copy.flux, copy.ee_psf_lo) == (tmpl.id, tmpl.flux, tmpl.ee_psf_lo)
    assert copy.slices_original == tmpl.slices_original
    assert copy.slices_cutout == tmpl.slices_cutout
    assert copy.bbox == tmpl.bbox

    copy.data[0, 0] = 999.0
    assert tmpl.data[0, 0] != 999.0


def test_shallow_templates_container_shares_stamps_and_segmap():
    """What _convolved_templates builds instead of deepcopy(self.tmpls)."""
    img = np.random.default_rng(1).normal(size=(64, 64)).astype(np.float32)
    seg = np.zeros(img.shape, dtype=np.int32)
    tmpls = Templates()
    tmpls.original_shape = img.shape
    tmpls.segmap = seg
    tmpls._templates = [
        Template(img, (20.0 + 8 * i, 20.0), (12, 12), label=i + 1) for i in range(4)
    ]

    shallow = Templates()
    shallow.original_shape = tmpls.original_shape
    shallow.segmap = tmpls.segmap
    shallow._templates = list(tmpls._templates)

    assert shallow.segmap is tmpls.segmap
    assert shallow._templates is not tmpls._templates
    assert all(a is b for a, b in zip(shallow._templates, tmpls._templates))

    # convolution copies each stamp, so the originals survive it untouched
    before = [t.data.copy() for t in tmpls._templates]
    kernel = np.zeros((5, 5))
    kernel[1, 2] = 1.0  # off-centre: not an identity kernel
    out = shallow.convolve_templates(kernel, inplace=False)
    assert all(o is not t for o, t in zip(out, tmpls._templates))
    assert all(np.array_equal(b, t.data) for b, t in zip(before, tmpls._templates))


def test_deepcopy_of_a_templates_container_duplicates_the_segmap():
    """Why the shallow container above matters: the segmap rides along."""
    tmpls = Templates()
    tmpls.segmap = np.zeros((32, 32), dtype=np.int32)
    assert deepcopy(tmpls).segmap is not tmpls.segmap


# -- derived model image ------------------------------------------------------


class _FakeRun:
    def __init__(self, images, residuals):
        self.images = images
        self.residuals = residuals


def test_model_images_are_derived_from_image_minus_residual():
    rng = np.random.default_rng(4)
    images = [rng.normal(size=(8, 8)) for _ in range(3)]
    residuals = [rng.normal(size=(8, 8)) for _ in range(2)]
    models = pipeline._ModelImages(_FakeRun(images, residuals))

    assert len(models) == 2
    for i in range(2):
        assert np.array_equal(models[i], images[i + 1] - residuals[i])
    assert np.array_equal(models[-1], images[2] - residuals[1])
    # repeated access returns the cached array rather than resubtracting
    assert models[1] is models[1]
    with pytest.raises(IndexError):
        models[2]


def test_model_images_is_empty_before_a_run():
    models = pipeline._ModelImages(_FakeRun([np.zeros((4, 4))], []))
    assert len(models) == 0
    with pytest.raises(IndexError):
        models[0]
