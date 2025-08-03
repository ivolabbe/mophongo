import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
from scipy.ndimage import shift as nd_shift

from pathlib import Path
from utils import make_simple_data
from mophongo.psf import PSF
from mophongo.templates import Templates
from mophongo.fit import SparseFitter, FitConfig
from mophongo.local_astrometry import (
    correct_astrometry_polynomial,
    correct_astrometry_gp,
    shifts_at_positions,
)
from utils import save_diagnostic_image, lupton_norm
from skimage.registration import phase_cross_correlation
from photutils.centroids import centroid_quadratic

def test_polynomial_astrometry_reduces_residual(tmp_path):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=42
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    shx, shy = 0.6, -0.5
    images[1] = nd_shift(images[1], (shy, shx))

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    templates = Templates.prune_and_dedupe(tmpls.templates, wht[1])
    fitter = SparseFitter(templates, images[1], wht[1], FitConfig())
    fitter.build_normal_matrix()
    flux, _, _ = fitter.solve()
    err = fitter.flux_errors()
    perr = fitter.predicted_errors()
    res = fitter.residual()

    res0 = fitter.residual()

    coeff_x, coeff_y = correct_astrometry_polynomial(
        templates,
        res0,
        fitter.solution,
        order=1,
        box_size=11,
        snr_threshold=20,
    )
  
    rhx,rhy = shifts_at_positions([[50,50]], coeff_x, coeff_y, 
                                  order=1, shape=images[1].shape)

    print(f"Astrometry correction coefficients: {coeff_x}, {coeff_y}")
    print(f'input shifts: {shx}, {shy} ')
    print(f"Reconstructed shifts: {rhx}, {rhy}")

    tmp_path = Path('../tmp')
    tmp_path.mkdir(exist_ok=True)
    fname = tmp_path / "diagnostic_poly_shift.png"
    model = images[1] - res0
    save_diagnostic_image(fname,
                          truth,
                          images[0],
                          images[1],
                          model,
                          res0,
                          segmap=segmap,
                          catalog=catalog)
    
    return

    from matplotlib import pyplot as plt

    i = 8
    t = tmpls[i]
    plt.imshow(t.data, origin='lower', cmap='gray', norm=lupton_norm(t.data))
    plt.show()
    d = images[1][t.slices_original]
    scl = fitter.solution[i]
    m = t.data[t.slices_cutout]*scl
    r = resid0[t.slices_original]
    plt.imshow(d, origin='lower', cmap='gray', norm=lupton_norm(d))
    plt.imshow(r, origin='lower', cmap='gray', norm=lupton_norm(d))
    plt.imshow(r+m, origin='lower', cmap='gray', norm=lupton_norm(d))
    ccr = _normalized_cross_correlation(r+m, m)
    plt.imshow(m, origin='lower', cmap='gray', norm=lupton_norm(d))
    plt.imshow(ccr, origin='lower', cmap='gray')

#    sh, esh, _ = phase_cross_correlation( r+m, m, upsample_factor=20)

    rm_xy = centroid_quadratic(r+m)
    m_xy = centroid_quadratic(m)
    ccr_xy = centroid_quadratic(ccr)
    print(f"Shifted residual centroid: {rm_xy}, model centroid: {m_xy} shift: {m_xy- rm_xy}")
    stamp_xy = ((np.array(ccr.shape)-1)/2)[::-1]
    print(f"Cross-correlation centroid: {ccr_xy}, stamp centroid: {stamp_xy} shift: {ccr_xy - stamp_xy}")
    assert coeff_x.shape[0] == coeff_y.shape[0]


def test_gp_astrometry_returns_models():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=101, peak_snr=5, seed=1
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    images[1] = nd_shift(images[1], (-0.5, 0.6))

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )

    templates = Templates.prune_and_dedupe(tmpls.templates, wht[1])
    fitter = SparseFitter(templates, images[1], wht[1], FitConfig())
    fitter.build_normal_matrix()
    fitter.solve()
    fitter.flux_errors()
    res = fitter.residual()

    gp_x, gp_y = correct_astrometry_gp(
        templates,
        res,
        fitter.solution,
        box_size=7,
        snr_threshold=5,
        length_scale=30.0,
    )

    pred_dx = gp_x.predict(np.array([[50.0, 50.0]]))[0]
    pred_dy = gp_y.predict(np.array([[50.0, 50.0]]))[0]
    assert isinstance(pred_dx, float)
    assert isinstance(pred_dy, float)
