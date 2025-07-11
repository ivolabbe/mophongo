import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF, pad_to_shape
from mophongo.templates import _convolve2d
from utils import make_simple_data, save_psf_diagnostic, save_psf_fit_diagnostic

def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0,  rtol=0, atol=2e-3)


def test_psf_matching_kernel_properties(tmp_path):
    _, _, _, psfs, _, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_lo.array.shape

    # Pad psf_hi to kernel shape for diagnostics and convolution
    hi_pad = pad_to_shape(psf_hi.array, kernel.shape)
    conv = _convolve2d(hi_pad, kernel)
    np.testing.assert_allclose(conv, psf_lo.array, rtol=0, atol=2e-3)
    fname = tmp_path / "psf_kernel.png"
    save_psf_diagnostic(fname, hi_pad, psf_lo.array, kernel)
    assert fname.exists()


def test_psf_moffat_fit(tmp_path):
    psf = PSF.moffat(31, fwhm_x=3.2, fwhm_y=4.5, beta=2.8, theta=0.3)
    params = psf.fit_moffat()
    assert np.isclose(params.beta, 2.8, rtol=0.1)
    psf_fit = PSF.moffat(psf.array.shape, params.fwhm_x, params.fwhm_y, params.beta, params.theta)
    fname = tmp_path / "psf_fit_moffat.png"
    save_psf_fit_diagnostic(fname, psf.array, psf_fit.array)
    assert fname.exists()
    np.testing.assert_allclose(psf_fit.array, psf.array, rtol=0, atol=5e-2)


def test_psf_gaussian_fit(tmp_path):
    psf = PSF.gaussian(31, fwhm_x=2.5, fwhm_y=3.0, theta=0.2)
    params = psf.fit_gaussian()

    psf_fit = PSF.gaussian(psf.array.shape, params.fwhm_x, params.fwhm_y, params.theta)
    fname = tmp_path / "psf_fit_gaussian.png"
    save_psf_fit_diagnostic(fname, psf.array, psf_fit.array)
    assert fname.exists()
    np.testing.assert_allclose(psf_fit.array, psf.array, rtol=0, atol=5e-2)


def test_delta_psf_default():
    psf = PSF.delta()
    assert psf.array.shape == (3, 3)
    assert psf.array[1, 1] == 1.0
    np.testing.assert_allclose(psf.array.sum(), 1.0)
