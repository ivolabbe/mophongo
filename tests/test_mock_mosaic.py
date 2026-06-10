"""Smoke tests for MockMosaic WCS products and noise convention."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter

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


def test_mock_mosaic_psf_blur_preserves_native_edge_loss(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        psf_gaussian_fwhm_arcsec={"f770w": 0.08},
    )
    psf = np.zeros((15, 15), dtype=float)
    psf[0, 0] = 1.0

    blurred = mock.blur_filter_psf("f770w", psf)

    assert mock.source_psf_normalization == "native"
    assert mock._psf_gaussian_fwhm_arcsec_for("f770w") == pytest.approx(0.08)
    assert mock._psf_gaussian_fwhm_for("f770w") == pytest.approx(1.0)
    assert blurred.sum() < psf.sum()


def test_mock_mosaic_f770w_blur_is_sampled_on_40mas_grid(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        psf_gaussian_fwhm_arcsec={"f770w": 0.08},
    )
    psf = np.zeros((17, 17), dtype=float)
    psf[8, 8] = 1.0

    blurred = mock.blur_filter_psf("f770w", psf)

    factor = 2
    up = np.repeat(np.repeat(psf, factor, axis=0), factor, axis=1) / factor**2
    manual_40mas = gaussian_filter(
        up,
        sigma=(0.08 / 0.040) / 2.355,
        mode="constant",
        cval=0.0,
        truncate=6.0,
    )
    manual_80mas = manual_40mas.reshape(17, factor, 17, factor).sum(axis=(1, 3))
    direct_80mas = gaussian_filter(
        psf,
        sigma=(0.08 / 0.080) / 2.355,
        mode="constant",
        cval=0.0,
        truncate=6.0,
    )

    assert blurred == pytest.approx(manual_80mas)
    assert np.max(np.abs(blurred - direct_80mas)) > 1e-2
