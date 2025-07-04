import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF


def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0, rtol=1e-6)


def test_psf_matching_kernel_properties():
    size = 15
    psf_hi = PSF.moffat(size, 2.0, 2.0, beta=2.5)
    psf_lo = PSF.moffat(size, 3.0, 3.0, beta=2.5)
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_hi.array.shape
    conv = np.fft.ifft2(np.fft.fft2(psf_hi.array) * np.fft.fft2(kernel)).real
    # kernel should transform psf_hi approximately into psf_lo
    np.testing.assert_allclose(conv, psf_lo.array, rtol=1e-2, atol=5e-4)
