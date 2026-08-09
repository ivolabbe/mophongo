"""Tests for the IDL subphot-style diagnostic (Pipeline.plot_subphot)."""

from __future__ import annotations

import numpy as np
import pytest

import mophongo.pipeline as pipeline
import mophongo.utils as mutils
from mophongo.pipeline import _bytscl, _fptv_panel

from utils import make_simple_data


def test_bytscl_matches_idl():
    a = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    b = _bytscl(a, 0.0, 1.0)
    # IDL bytscl: floor(255.9999 * clipped fraction) -> midpoint is 127
    assert b.dtype == np.uint8
    assert list(b) == [0, 0, 127, 255, 255]


def test_fptv_panel_shapes_and_binning():
    rng = np.random.default_rng(1)
    img = rng.normal(0, 1, (10, 10))
    p = _fptv_panel(img, mm=(-1, 1))
    assert p.shape == (20, 20)  # os=2 nearest-neighbour zoom
    assert p.dtype == np.uint8
    # binned panel: 2x2 blocks constant before the 2x zoom -> 4x4 blocks
    pb = _fptv_panel(img, mm=(-1, 1), bin=2)
    assert pb.shape == (20, 20)
    assert np.all(pb[:4, :4] == pb[0, 0])
    # auto scaling (median +- fac*robust sigma) must not crash on flat input
    _fptv_panel(np.zeros((10, 10)))


@pytest.fixture(scope="module")
def fitted_pipeline():
    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=3, nsrc=20, size=121, ndilate=2, peak_snr=2
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    _table, _resid, pipe = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=[None, kernel]
    )
    return pipe


def test_plot_subphot_layout_and_scalings(fitted_pipeline, tmp_path):
    pipe = fitted_pipeline
    sid = int(pipe.table["id"][0])
    size = 41
    out = tmp_path / f"subphot_{sid}.png"
    img = pipe.plot_subphot(sid, size=size, save=out)

    # 2x3 panels of size x size stamps at 2x zoom
    assert img.shape == (4 * size, 6 * size, 3)
    assert img.dtype == np.uint8
    assert out.exists()

    # res panel (bottom row, middle): corners are outside rlim -> masked,
    # zero maps to mid-gray 127 under the symmetric byte scaling
    t2 = 2 * size
    assert tuple(img[t2 + 1, 2 * t2 - 2]) == (127, 127, 127)
    # clean panel corner likewise
    assert tuple(img[t2 + 1, 3 * t2 - 2]) == (127, 127, 127)

    # white panel labels drawn in each panel's top-left corner
    assert img[:30, :60].max() >= 200  # "img"
    assert img[t2 : t2 + 30, :60].max() >= 200  # "model"


def test_plot_subphot_defaults_and_errors(fitted_pipeline):
    pipe = fitted_pipeline
    sid = int(pipe.table["id"][0])
    # default size: odd template-footprint side
    img = pipe.plot_subphot(sid)
    assert img.shape[0] % 4 == 0 and img.shape[0] // 4 % 2 == 1
    with pytest.raises(KeyError):
        pipe.plot_subphot(999999)
    with pytest.raises(ValueError):
        pipe.plot_subphot(sid, ifilt=0)
