import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF
from mophongo.templates import _convolve2d
from utils import save_psf_diagnostic


def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0, rtol=1e-6)


def test_psf_matching_kernel_properties(tmp_path):
    size = 15
    psf_hi = PSF.moffat(size, 2.0, 2.0, beta=2.5)
    psf_lo = PSF.moffat(size, 3.0, 3.0, beta=2.5)
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_hi.array.shape
    conv = _convolve2d(psf_hi.array, kernel)
    # kernel should transform psf_hi approximately into psf_lo
    np.testing.assert_allclose(conv, psf_lo.array, rtol=1e-2, atol=5e-4)
    fname = tmp_path / "psf_kernel.png"
    save_psf_diagnostic(fname, psf_hi.array, psf_lo.array, kernel)
    assert fname.exists()


def test_psf_matching_kernel_different_sizes():
    psf_hi = PSF.moffat(9, 2.0, 2.0, beta=2.5)
    psf_lo = PSF.moffat(21, 10.0, 10.0, beta=2.5)
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == (21, 21)

    def pad(arr, shape):
        py = (shape[0] - arr.shape[0]) // 2
        px = (shape[1] - arr.shape[1]) // 2
        return np.pad(arr, ((py, shape[0] - arr.shape[0] - py), (px, shape[1] - arr.shape[1] - px)))

    hi_pad = pad(psf_hi.array, kernel.shape)
    lo_pad = pad(psf_lo.array, kernel.shape)
    conv = _convolve2d(hi_pad, kernel)
    np.testing.assert_allclose(conv, lo_pad, rtol=1e-2, atol=5e-4)
