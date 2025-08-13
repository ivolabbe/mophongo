import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
from scipy.ndimage import shift as nd_shift

from pathlib import Path
from utils import make_simple_data
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from mophongo.fit import SparseFitter, FitConfig
from mophongo.astrometry import AstroCorrect, AstroMap, cheb_basis
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

    tmp_path = Path("../tmp")
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

    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)

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


def test_astromap_recovers_shift():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=7
    )
    shx, shy = 0.4, -0.3
    shifted = nd_shift(images[0], (shy, shx))
    amap = AstroMap(order=1, snr_threshold=3.0)
    amap.fit(images[0], shifted, segmap)
    dx, dy = amap(np.array([[75.0, 75.0]]))
    assert abs(dx[0] - shx) < 0.3
    assert abs(dy[0] - shy) < 0.3


def test_apply_template_shifts_uses_shift_field():
    data = np.zeros((7, 7))
    data[3, 3] = 1.0
    tmpl = Template(data, (3.0, 3.0), (7, 7), label=1)
    tmpl.shift = np.array([0.5, -0.25])

    AstroCorrect.apply_template_shifts([tmpl])

    expected = nd_shift(
        data,
        (-0.25, 0.5),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    assert np.allclose(tmpl.data, expected)
    assert np.allclose(tmpl.input_position_original, (2.5, 3.25))
    assert np.allclose(tmpl.shift, 0.0)


def test_build_poly_predictor_returns_expected_shift():
    order = 1
    betax = np.array([1.0, 0.2, -0.3])
    betay = np.array([-0.5, 0.1, 0.05])
    coeffs = np.concatenate([betax, betay])
    x0, y0 = 10.0, 20.0
    predict = AstroCorrect.build_poly_predictor(coeffs, x0, y0, order)

    x, y = 11.0, 19.0
    dx, dy = predict(x, y)
    phi = cheb_basis(x - x0, y - y0, order)
    assert np.allclose(dx, phi @ betax)
    assert np.allclose(dy, phi @ betay)
