import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF
from mophongo.templates import _convolve2d
from utils import make_simple_data, save_psf_diagnostic

def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0, rtol=1e-6)


def test_psf_matching_kernel_properties(tmp_path):
    _, _, _, psfs, _ = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_lo.array.shape

    def pad(arr, shape):
        py = (shape[0] - arr.shape[0]) // 2
        px = (shape[1] - arr.shape[1]) // 2
        return np.pad(arr, ((py, shape[0] - arr.shape[0] - py), (px, shape[1] - arr.shape[1] - px)))

    conv = _convolve2d(pad(psf_hi.array, kernel.shape), kernel)
    np.testing.assert_allclose(conv, psf_lo.array, rtol=1e-2, atol=5e-4)
    fname = tmp_path / "psf_kernel.png"
    save_psf_diagnostic(fname, psf_hi.array, psf_lo.array, kernel)
    assert fname.exists()


def test_psf_matching_kernel_different_sizes():
    _, _, _, psfs, _ = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])

    # downsample high-res PSF to enforce different shapes
    small_hi = PSF.from_array(psf_hi.array[::2, ::2])
    kernel = small_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_lo.array.shape

    def pad(arr, shape):
        py = (shape[0] - arr.shape[0]) // 2
        px = (shape[1] - arr.shape[1]) // 2
        return np.pad(arr, ((py, shape[0] - arr.shape[0] - py), (px, shape[1] - arr.shape[1] - px)))

    hi_pad = pad(small_hi.array, kernel.shape)
    lo_pad = pad(psf_lo.array, kernel.shape)
    conv = _convolve2d(hi_pad, kernel)
    np.testing.assert_allclose(conv, lo_pad, rtol=1e-2, atol=5e-4)
