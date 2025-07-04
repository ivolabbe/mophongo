import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF, pad_to_shape
from mophongo.templates import _convolve2d
from utils import make_simple_data, save_psf_diagnostic

def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0,  rtol=0, atol=2e-3)


def test_psf_matching_kernel_properties(tmp_path):
    _, _, _, psfs, _ = make_simple_data()
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
