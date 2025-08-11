"""Tests for template downsampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.templates import Template
import pytest
from mophongo import utils as mutils

if not hasattr(mutils, "bin2d_mean"):
    pytest.skip("bin2d_mean not implemented", allow_module_level=True)

from mophongo.utils import bin2d_mean, downsample_psf


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("h,w", [(5, 7), (6, 6), (9, 12)])
@pytest.mark.parametrize("y0,x0", [(0, 0), (1, 2), (4, 5)])
def test_downsample_centroids(k: int, h: int, w: int, y0: int, x0: int) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(h, w))
    t_hi = Template.__new__(Template)
    t_hi.data = data
    t_hi.bbox_original = ((y0, y0 + h - 1), (x0, x0 + w - 1))
    t_hi.slices_original = (slice(y0, y0 + h), slice(x0, x0 + w))
    t_hi.slices_cutout = (slice(0, h), slice(0, w))
    t_hi.input_position_cutout = ((h - 1) / 2, (w - 1) / 2)
    t_hi.input_position_original = (y0 + (h - 1) / 2, x0 + (w - 1) / 2)

    t_lo = t_hi.downsample(k)

    shift = (k - 1) / 2.0
    y_expect = ((h - 1) / 2 - shift) / k
    x_expect = ((w - 1) / 2 - shift) / k
    y_lo, x_lo = t_lo.input_position_cutout
    assert np.allclose([y_lo, x_lo], [y_expect, x_expect])

    sy, sx = t_lo.slices_original
    assert (sy.start, sy.stop) == (t_lo.bbox[0], t_lo.bbox[1])
    assert (sx.start, sx.stop) == (t_lo.bbox[2], t_lo.bbox[3])

    ny_lo, nx_lo = t_lo.data.shape
    hi_trim = t_hi.data[: ny_lo * k, : nx_lo * k]
    assert np.allclose(
        hi_trim.sum(dtype=np.float64),
        t_lo.data.sum(dtype=np.float64) * k * k,
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("k", [2, 3])
def test_bin2d_mean_vs_numpy(k: int) -> None:
    arr = np.arange(7 * 11, dtype=float).reshape(7, 11)
    out = bin2d_mean(arr, k)
    ref = np.empty_like(out)
    for iy in range(out.shape[0]):
        for ix in range(out.shape[1]):
            block = arr[iy * k : (iy + 1) * k, ix * k : (ix + 1) * k]
            ref[iy, ix] = block.mean()
    assert np.allclose(out, ref)


@pytest.mark.parametrize("k", [2, 3])
def test_psf_downsample_center(k: int) -> None:
    size = 9
    x = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, x)
    psf = np.exp(-(X ** 2 + Y ** 2) / (2 * 1.5 ** 2))
    psf /= psf.sum()

    psf_lo = downsample_psf(psf, k)
    y, x = np.indices(psf_lo.shape)
    cy = (psf_lo * y).sum() / psf_lo.sum()
    cx = (psf_lo * x).sum() / psf_lo.sum()
    assert np.allclose([cy, cx], [(psf_lo.shape[0] - 1) / 2, (psf_lo.shape[1] - 1) / 2])

