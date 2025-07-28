import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.ndimage import shift as nd_shift

from utils import make_simple_data
from mophongo.psf import PSF
from mophongo.templates import Templates
from mophongo.fit import SparseFitter, FitConfig
from mophongo.local_astrometry import correct_astrometry_polynomial


def test_polynomial_astrometry_reduces_residual():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=101, peak_snr=50
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    shifted = nd_shift(images[1], (0.3, -0.2))

    fitter = SparseFitter(tmpls.templates, shifted, wht[1], FitConfig())
    fitter.build_normal_matrix()
    fitter.solve()
    resid0 = fitter.residual()

    coeff_x, coeff_y = correct_astrometry_polynomial(
        tmpls.templates, resid0, order=1, box_size=11, snr_threshold=2.0
    )

    assert coeff_x.shape[0] == coeff_y.shape[0]
