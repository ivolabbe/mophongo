"""Smoke tests for MockMosaic WCS products and noise convention."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.stats import mad_std

from mophongo.mock_mosaic import MockMosaic, Pointing, drizzle_correlation_factor
from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap


def _build(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        center_radec=(34.5, -5.2),
        nircam_lw_frames={
            "f444w": [
                Pointing(ra=34.5, dec=-5.2, pa=0.0),
                Pointing(ra=34.5, dec=-5.2, pa=45.0),
            ]
        },
        miri_frames={"f770w": [Pointing(ra=34.5, dec=-5.2, pa=0.0)]},
    )
    return mock.write()


def test_mock_mosaic_writes_expected_rows_and_registration(tmp_path):
    paths = _build(tmp_path)
    assert paths["f444w"]["n_rows"] == 4
    assert paths["f770w"]["n_rows"] == 1
    assert paths["f444w"]["crpix"][0] == pytest.approx(2 * paths["f770w"]["crpix"][0] - 0.5)
    assert paths["f444w"]["size"] == tuple(2 * v for v in paths["f770w"]["size"])


def test_drizzlepsf_loads_mock_and_builds_region_map(tmp_path):
    paths = _build(tmp_path)
    dpsf = DrizzlePSF(
        driz_image=str(paths["f444w"]["fits"]),
        csv_file=str(paths["f444w"]["csv"]),
    )
    files = [k[0] for k in dpsf.flt_keys]
    assert any("nrcalong" in f for f in files)
    assert any("nrcblong" in f for f in files)

    prm = PSFRegionMap.from_footprints(dpsf.footprint, pa_tol=1.0)
    assert len(prm.regions) >= 3


def test_inject_noise_roundtrip(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        center_radec=(34.5, -5.2),
        nircam_lw_frames={
            "f444w": [
                Pointing(ra=34.5, dec=-5.2, pa=0.0),
                Pointing(ra=34.5, dec=-5.2, pa=45.0),
            ]
        },
        miri_frames={"f770w": [Pointing(ra=34.5, dec=-5.2, pa=0.0)]},
        noise_K={"f444w": 0.0768},
        noise_seed=0,
    )
    paths = mock.write()
    for filt in ("f444w", "f770w"):
        info = mock.inject_noise(filt, paths)
        sci = fits.getdata(info["sci"])
        wht = fits.getdata(info["wht"])
        wht_header = fits.getheader(info["wht"])
        mask = info["texp"] > 0
        np.testing.assert_allclose(wht[mask], 1.0 / info["sigma_pix"][mask] ** 2, rtol=1e-6)
        std = sci[mask] * np.sqrt(wht[mask])
        assert mad_std(std) == pytest.approx(1.0, rel=0.02)
        assert wht_header["WHTTYPE"] == "IVARPIX"
        assert wht_header["RNOISE"] == pytest.approx(info["R"])


def test_drizzle_correlation_factor():
    assert drizzle_correlation_factor(1.0, 1.0, 1.0) == pytest.approx(2 / 3)
    assert drizzle_correlation_factor(0.75, 0.063, 0.04) == pytest.approx(0.6077, abs=1e-3)


def test_mock_mosaic_psf_blur_conserves_flux_without_renormalizing(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        psf_gaussian_fwhm_arcsec={"f770w": 0.08},
    )
    yy, xx = np.mgrid[:33, :33] - 16.0
    psf = 0.7 * np.exp(-(yy**2 + xx**2) / (2 * 2.0**2))

    blurred = mock.blur_filter_psf("f770w", psf)

    assert mock.source_psf_normalization == "native"
    assert mock._psf_gaussian_fwhm_arcsec_for("f770w") == pytest.approx(0.08)
    assert mock._psf_gaussian_fwhm_for("f770w") == pytest.approx(1.0)
    # Fourier-domain Gaussian: flux-conserving, peak reduced, never renormalized
    assert blurred.sum() == pytest.approx(psf.sum(), rel=1e-8)
    assert blurred.max() < psf.max()
    # f444w has no configured blur: strict no-op
    assert mock.blur_filter_psf("f444w", psf) is psf


def test_mock_mosaic_f770w_blur_commutes_with_native_binning(tmp_path):
    """The blur operator must be grid-independent.

    Painted mock sources are blurred on the native 80 mas grid while the
    verification PSF/kernel maps are blurred on the 40 mas reference grid;
    any operator difference between the two paths appears as a spurious
    data-vs-model PSF mismatch ("donut" residuals) in the verification fits.
    """
    mock = MockMosaic(
        out_dir=tmp_path,
        psf_gaussian_fwhm_arcsec={"f770w": 0.08},
    )
    yy, xx = np.mgrid[:68, :68] - 33.5
    psf_40 = np.exp(-(yy**2 + xx**2) / (2 * 3.0**2))
    psf_80 = psf_40.reshape(34, 2, 34, 2).sum(axis=(1, 3))

    blur40_binned = mock.blur_filter_psf("f770w", psf_40, pscale=0.040)
    blur40_binned = blur40_binned.reshape(34, 2, 34, 2).sum(axis=(1, 3))
    blur80 = mock.blur_filter_psf("f770w", psf_80, pscale=0.080)

    np.testing.assert_allclose(blur40_binned, blur80, atol=1e-6 * blur80.max())

    # expected variance growth: sigma_out^2 = sigma_in^2 + (fwhm/2.355)^2
    r2 = ((np.mgrid[:34, :34][0] - 16.5) ** 2 + (np.mgrid[:34, :34][1] - 16.5) ** 2)
    var_in = float((psf_80 * r2).sum() / psf_80.sum()) / 2
    var_out = float((blur80 * r2).sum() / blur80.sum()) / 2
    sigma_blur_80 = (0.08 / 0.080) / 2.355
    assert var_out - var_in == pytest.approx(sigma_blur_80**2, rel=1e-3)


def test_gaussian_blur_psf_is_the_single_shared_operator(tmp_path):
    """Mock painting hook and real-data drivers must apply the identical blur.

    ``MockMosaic.blur_filter_psf`` (mock painting / verification kernel maps)
    and the driver-style direct call (``examples/run_770.py``) both delegate
    to ``gaussian_blur_psf`` with the shared
    ``DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC`` value — outputs must be
    bit-identical for the same grid.
    """
    from mophongo.mock_mosaic import (
        DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC,
        gaussian_blur_psf,
    )

    fwhm = DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC["f770w"]
    mock = MockMosaic(out_dir=tmp_path)

    rng = np.random.default_rng(3)
    cube = rng.random((3, 40, 40))
    via_mock = mock.blur_filter_psf("f770w", cube, pscale=0.080)
    via_driver = gaussian_blur_psf(cube, fwhm, 0.080)
    np.testing.assert_array_equal(via_mock, via_driver)

    # flux conservation of the shared operator
    np.testing.assert_allclose(
        via_driver.sum(axis=(1, 2)), cube.sum(axis=(1, 2)), rtol=1e-12
    )
