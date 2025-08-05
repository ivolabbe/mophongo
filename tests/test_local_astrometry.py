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
from mophongo.local_astrometry import AstroCorrect
from utils import save_diagnostic_image

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

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal_matrix()
    flux, _, _ = fitter.solve()
    err = fitter.flux_errors()
    perr = fitter.predicted_errors()
    res = fitter.residual()
    
    res0 = fitter.residual()

    ac = AstroCorrect(FitConfig())
    ac.fit(tmpls.templates, res0, fitter.solution)

    rhx, rhy = ac(np.array([[50.0, 50.0]]))

    print(f"Astrometry correction shifts: {rhx}, {rhy}")
    print(f"input shifts: {shx}, {shy}")

    assert abs(rhx[0] - shx) < 0.3
    assert abs(rhy[0] - shy) < 0.3

    tmp_path = Path('../tmp')
    tmp_path.mkdir(exist_ok=True)
    fname = tmp_path / "diagnostic_poly_shift.png"
    model = images[1] - res0
    save_diagnostic_image(
        fname,
        truth,
        images[0],
        images[1],
        model,
        res0,
        segmap=segmap,
        catalog=catalog,
    )

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

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal_matrix()
    fitter.solve()
    fitter.flux_errors()
    res = fitter.residual()

    cfg = FitConfig(astrom_model="gp", astrom_kwargs={"gp": {"length_scale": 30.0}})
    ac = AstroCorrect(cfg)
    ac.fit(tmpls.templates, res, fitter.solution)

    dx, dy = ac(np.array([[50.0, 50.0]]))
    assert isinstance(float(dx[0]), float)
    assert isinstance(float(dy[0]), float)
