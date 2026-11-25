"""Errors on the aperture fluxes.

Every aperture estimator is a multiple of the fitted amplitude plus a residual
sum, and its error column (``ap_err_<i>`` and friends) is built from the pixel
variance in the aperture, the template's share of it, and the two amplitude
errors. The test that matters is the last kind here: over many independent
noise realizations, the scatter of each flux must equal its reported error.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
from scipy.signal import fftconvolve

from mophongo import pipeline
from mophongo.fit import FitConfig
from mophongo.pipeline import _aperture_variances
from mophongo.psf import PSF
from mophongo.templates import Templates

PSCALE = 0.1  # arcsec / pixel
SIGMA = 0.5  # per-pixel noise of the fitted band
FLUX = 100.0

PAIRS = [
    ("ap_flux_1", "ap_err_1"),
    ("ap_flux_corr_1", "ap_err_corr_1"),
    ("ap_flux_cat_1", "ap_err_cat_1"),
    ("ap_flux_est3_1", "ap_err_est3_1"),
]


def _wcs(shape: tuple[int, int]) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-PSCALE / 3600.0, PSCALE / 3600.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _gaussian_kernel(size: int, fwhm: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size] - size // 2
    g = np.exp(-(xx**2 + yy**2) / (2 * (fwhm / 2.3548) ** 2))
    return g / g.sum()


def _run_grid(pair_sep: int = 0, kron: bool = True, seed: int = 3) -> Table:
    """Fit a grid of identical sources, each one an independent noise draw.

    Band 0 is sharp (FWHM 2 px) and noise-free; band 1 is band 0 convolved to
    FWHM 4 px plus white noise. With ``pair_sep`` the grid holds close pairs,
    whose convolved templates overlap, so the solver error exceeds the
    covariance-free one and each aperture reaches into the neighbor's segment.
    """
    n_side, step = (12, 44) if pair_sep else (16, 36)
    shape = (n_side * step, n_side * step)
    s_hi = 2.0 / 2.3548
    yy, xx = np.mgrid[: shape[0], : shape[1]]

    centers: list[tuple[float, float]] = []
    for j in range(n_side):
        for i in range(n_side):
            cx, cy = step // 2 + i * step, step // 2 + j * step
            if pair_sep:
                centers += [(cx, cy - pair_sep / 2), (cx, cy + pair_sep / 2)]
            else:
                centers.append((cx, cy))

    hi = np.zeros(shape)
    segmap = np.zeros(shape, dtype=np.int32)
    for k, (px, py) in enumerate(centers, start=1):
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        hi += FLUX * np.exp(-r2 / (2 * s_hi**2)) / (2 * np.pi * s_hi**2)
        own = r2 < 5**2
        if pair_sep:
            # strictly nearer to this member than to its partner, so the two
            # segments of a pair mirror each other
            partner_y = py + (pair_sep if k % 2 == 1 else -pair_sep)
            own &= np.abs(yy - py) < np.abs(yy - partner_y)
        segmap[own] = k

    kernel = _gaussian_kernel(31, np.sqrt(4.0**2 - 2.0**2))
    rng = np.random.default_rng(seed)
    lo = fftconvolve(hi, kernel, mode="same") + SIGMA * rng.standard_normal(shape)

    ids = list(range(1, len(centers) + 1))
    columns = {
        "id": ids,
        "x": [float(c[0]) for c in centers],
        "y": [float(c[1]) for c in centers],
    }
    phot = {"aperture_ee": 0.70, "units": "arcsec"}
    if kron:
        columns |= {
            "kron_flux": np.full(len(ids), 150.0),
            "aper_flux": np.full(len(ids), 100.0),
            "kron_radius": np.full(len(ids), 0.4),
        }
        phot |= {
            "kron_flux_col": "kron_flux",
            "aper_flux_col": "aper_flux",
            "kron_radius_col": "kron_radius",
        }
    catalog = Table(columns)
    tmpls = Templates.from_image(hi, segmap, list(zip(catalog["x"], catalog["y"])))
    table, _res, _pipe = pipeline.run(
        [hi, lo],
        segmap,
        catalog=catalog,
        weights=[np.ones(shape), np.full(shape, 1.0 / SIGMA**2)],
        kernels=[None, kernel],
        templates=tmpls,
        wcs=[_wcs(shape), _wcs(shape)],
        psfs=[PSF.gaussian(41, fwhm=2.0).array, PSF.gaussian(41, fwhm=4.0).array],
        config=FitConfig(
            fit_method="lls",
            reg_flux=0.0,
            fit_astrometry_niter=0,
            fit_astrometry_joint=False,
            astrom_minimum_snr=0.0,
            phot=phot,
        ),
    )
    return table


def _scatter_over_error(table: Table, flux: str, err: str, classes: np.ndarray) -> float:
    """Scatter of ``flux`` about its class mean, over the median reported error."""
    values = np.asarray(table[flux], dtype=float)
    dev = np.concatenate(
        [values[classes == c] - values[classes == c].mean() for c in np.unique(classes)]
    )
    return float(np.std(dev) / np.median(np.asarray(table[err], dtype=float)))


# --- the variance budget ----------------------------------------------------


def test_isolated_source_carries_the_pixel_noise_alone():
    """With err == err_pred the amplitude term cancels out of the raw sum."""
    var_ap, var_res = _aperture_variances(6.0, 6.0, 0.7, 0.7, err=3.0, err_pred=3.0)

    assert var_ap == pytest.approx(6.0)
    # the fit took (t * err_pred)^2 of the aperture's variance with it
    assert var_res == pytest.approx(6.0 - (0.7 * 3.0) ** 2)


def test_blending_adds_the_amplitude_uncertainty_it_causes():
    var_ap, var_res = _aperture_variances(6.0, 5.0, 0.7, 0.6, err=5.0, err_pred=3.0)

    assert var_ap == pytest.approx(6.0 + 0.7**2 * (5.0**2 - 3.0**2))
    # the residual term uses Omega and the covariance-free error only
    assert var_res == pytest.approx(5.0 - (0.6 * 3.0) ** 2)


def test_rounding_cannot_pull_a_variance_below_its_floor():
    # err a hair under err_pred: clipped, not subtracted
    var_ap, _ = _aperture_variances(6.0, 6.0, 0.7, 0.7, err=3.0 - 1e-9, err_pred=3.0)
    assert var_ap == pytest.approx(6.0)
    # a template share larger than the pixel variance allows: floor at zero
    _, var_res = _aperture_variances(6.0, 1.0, 0.7, 0.7, err=3.0, err_pred=3.0)
    assert var_res == 0.0


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 0.0, 0.7, 0.7, 3.0, 3.0),  # no valid pixel in the aperture
        (np.nan, 6.0, 0.7, 0.7, 3.0, 3.0),  # aperture off the patch
        (6.0, 6.0, 0.7, 0.7, 3.0, np.inf),  # template without weight
        (6.0, 6.0, 0.7, 0.7, np.nan, 3.0),
    ],
)
def test_no_error_is_reported_without_the_inputs_for_one(args):
    assert np.all(np.isnan(_aperture_variances(*args)))


# --- the columns, against the scatter they claim to describe ----------------


def test_errors_match_the_scatter_of_isolated_sources():
    """256 isolated sources are 256 noise realizations of one measurement."""
    table = _run_grid()
    classes = np.zeros(len(table), dtype=int)

    for flux, err in PAIRS:
        assert np.all(np.isfinite(table[err])), err
        # 256 draws fix a standard deviation to 4.4%
        ratio = _scatter_over_error(table, flux, err, classes)
        assert ratio == pytest.approx(1.0, abs=0.1), f"{flux}: scatter/error = {ratio:.3f}"

    # the residual sum is the term whose variance the fit reduces: check it
    # alone, since inside ap_flux_est3 the model term outweighs it
    alpha = np.asarray(
        table["ap_model_1"] / table["flux_1"] * table["psfcor_1"] * table["totcor_cat"]
    )
    var_res = np.asarray(table["ap_err_est3_1"]) ** 2 - (alpha * table["err_1"]) ** 2
    assert np.std(table["ap_res_1"]) == pytest.approx(np.sqrt(np.median(var_res)), rel=0.12)


def test_isolated_error_is_the_pixel_noise_in_the_aperture():
    """Uniform noise, no blending: sigma * sqrt(sum m^2), not sigma * sqrt(area)."""
    table = _run_grid()

    r_pix = float(table["aper_1"][0]) / 2.0 / PSCALE
    m = CircularAperture((20.0, 20.0), r=r_pix).to_mask(method="exact").data
    expected = SIGMA * np.sqrt(np.sum(m**2))

    np.testing.assert_allclose(table["ap_err_1"], expected, rtol=1e-6)
    # an edge pixel covered by half counts a quarter, so this is below the
    # sigma * sqrt(pi r^2) a sum of m / w would give
    assert expected < SIGMA * np.sqrt(np.pi) * r_pix


def test_corrected_errors_scale_with_their_fluxes():
    table = _run_grid()
    ap_err = np.asarray(table["ap_err_1"])

    np.testing.assert_allclose(table["ap_err_corr_1"], ap_err * table["totcor_1"], rtol=1e-12)
    np.testing.assert_allclose(
        table["ap_err_cat_1"], ap_err * table["psfcor_1"] * table["totcor_cat"], rtol=1e-12
    )
    # same fractional error as the flux it belongs to
    np.testing.assert_allclose(
        table["ap_err_corr_1"] / table["ap_flux_corr_1"],
        ap_err / table["ap_flux_1"],
        rtol=1e-12,
    )


def test_errors_stay_honest_for_blended_pairs():
    """Overlapping templates: the errors are an upper bound, and a close one."""
    table = _run_grid(pair_sep=3)
    classes = np.arange(len(table)) % 2  # the two members of a pair

    # the configuration does what it is meant to: blending inflates the fit error
    assert np.median(table["err_1"] / table["err_pred_1"]) > 1.03
    for flux, err in PAIRS:
        ratio = _scatter_over_error(table, flux, err, classes)
        assert 0.85 < ratio < 1.08, f"{flux}: scatter/error = {ratio:.3f}"


def test_catalog_tied_errors_need_the_catalog_tied_fluxes():
    """Without the Kron knobs the raw and corrected errors are still written."""
    table = _run_grid(kron=False)

    assert np.all(np.isfinite(table["ap_err_1"]))
    assert np.all(np.isfinite(table["ap_err_corr_1"]))
    assert np.all(np.isnan(table["ap_err_cat_1"]))
    assert np.all(np.isnan(table["ap_err_est3_1"]))
