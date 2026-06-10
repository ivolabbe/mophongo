"""Tests for :func:`mophongo.utils.lw_detection_coadd`."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from scipy.signal import fftconvolve

from mophongo.utils import lw_detection_coadd


def _gaussian_psf(sigma: float, size: int = 25) -> np.ndarray:
    k = Gaussian2DKernel(sigma, x_size=size, y_size=size).array.astype(float)
    return k / k.sum()


def _make_band(true_sci, psf_sigma, *, wht_value, rng, shape):
    psf = _gaussian_psf(psf_sigma, size=31)
    blurred = fftconvolve(true_sci, psf, mode="same")
    sigma = 1.0 / np.sqrt(wht_value)
    noisy = blurred + rng.normal(0.0, sigma, shape)
    wht = np.full(shape, wht_value, dtype=np.float64)
    return {"sci": noisy.astype(np.float64), "wht": wht, "psf": psf}


def test_coadd_recovers_target_psf_and_flux():
    rng = np.random.default_rng(2)
    shape = (96, 96)
    truth = np.zeros(shape, dtype=np.float64)
    # 5 bright point sources.
    yx = rng.integers(20, 76, size=(5, 2))
    for y, x in yx:
        truth[y, x] = 100.0

    target_sigma = 2.5  # target (e.g. F444W).
    target_psf = _gaussian_psf(target_sigma, size=31)

    b0 = _make_band(truth, 1.2, wht_value=4.0, rng=rng, shape=shape)
    b0["name"] = "F277W"
    b1 = _make_band(truth, 1.8, wht_value=4.0, rng=rng, shape=shape)
    b1["name"] = "F356W"
    b2 = _make_band(truth, target_sigma, wht_value=1.0, rng=rng, shape=shape)
    b2["name"] = "F444W"

    out = lw_detection_coadd(
        [b0, b1, b2],
        target_psf=target_psf,
        target_index=2,
        method="wiener",
        reg_grid=np.logspace(-5, -2, 8),
    )
    coadd, wcoadd = out["sci"], out["wht"]
    assert coadd.shape == shape
    assert wcoadd.shape == shape
    # Weight strictly increased over target-only.
    assert np.all(wcoadd >= b2["wht"])
    # Effective weight should beat any single band.
    assert wcoadd.mean() > max(b0["wht"].mean(), b1["wht"].mean(),
                               b2["wht"].mean())
    # Source peaks should still be recovered (within target PSF blur).
    target_blurred = fftconvolve(truth, target_psf, mode="same")
    # Coadd peaks should align with true source pixels reasonably well.
    for y, x in yx:
        local = coadd[y - 3 : y + 4, x - 3 : x + 4]
        assert local.max() > 0.4 * target_blurred[y, x]
    # info table has one entry per band.
    assert len(out["info"]) == 3
    # Target band records sum_k2=1.0 (no convolution).
    assert out["info"][2]["sum_k2"] == 1.0
    # Non-target bands recorded a real reg.
    assert np.isfinite(out["info"][0]["reg"])
    assert np.isfinite(out["info"][1]["reg"])


def test_coadd_shape_mismatch_raises():
    rng = np.random.default_rng(0)
    shape = (40, 40)
    truth = np.zeros(shape); truth[20, 20] = 50.0
    target_psf = _gaussian_psf(2.0, size=21)
    b0 = _make_band(truth, 1.0, wht_value=1.0, rng=rng, shape=shape)
    b0["name"] = "a"
    # Force a different shape for the second band.
    b1 = _make_band(truth, 1.0, wht_value=1.0, rng=rng, shape=shape)
    b1["sci"] = b1["sci"][:30, :30]
    b1["wht"] = b1["wht"][:30, :30]
    b1["name"] = "b"
    with pytest.raises(ValueError, match="shape"):
        lw_detection_coadd(
            [b0, b1], target_psf=target_psf,
            target_index=0, method="wiener",
        )


def test_coadd_writes_outputs(tmp_path):
    rng = np.random.default_rng(3)
    shape = (32, 32)
    truth = np.zeros(shape); truth[16, 16] = 50.0
    target_psf = _gaussian_psf(2.0, size=21)
    b0 = _make_band(truth, 1.5, wht_value=2.0, rng=rng, shape=shape)
    b0["name"] = "blue"
    b1 = _make_band(truth, 2.0, wht_value=1.0, rng=rng, shape=shape)
    b1["name"] = "red"
    out_sci = tmp_path / "coadd_sci.fits"
    out_wht = tmp_path / "coadd_wht.fits"
    lw_detection_coadd(
        [b0, b1], target_psf=target_psf,
        target_index=1, method="wiener",
        reg_grid=np.logspace(-4, -2, 4),
        output_sci=out_sci, output_wht=out_wht,
    )
    assert out_sci.exists() and out_sci.stat().st_size > 0
    assert out_wht.exists() and out_wht.stat().st_size > 0
