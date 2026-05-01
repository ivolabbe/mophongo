"""Round-trip: MockMosaic → DrizzlePSF loads CSV → footprints + PSFRegionMap."""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.mock_mosaic import MockMosaic, Pointing, drizzle_correlation_factor
from mophongo.psf import DrizzlePSF
from mophongo.psf_map import PSFRegionMap


def _build(tmp_path):
    mock = MockMosaic(
        out_dir=tmp_path,
        center_radec=(34.5, -5.2),
        nircam_lw_frames={"f444w": [
            Pointing(ra=34.5, dec=-5.2, pa=0.0),
            Pointing(ra=34.5, dec=-5.2, pa=45.0),
        ]},
        miri_frames={"f770w": [Pointing(ra=34.5, dec=-5.2, pa=0.0)]},
    )
    return mock.write()


def test_row_counts(tmp_path):
    paths = _build(tmp_path)
    # 2 F444W frames × (A + B) = 4; 1 MIRI frame × 1 = 1.
    assert paths["f444w"]["n_rows"] == 4
    assert paths["f770w"]["n_rows"] == 1


def test_drizzlepsf_loads(tmp_path):
    paths = _build(tmp_path)
    dpsf = DrizzlePSF(
        driz_image=str(paths["f444w"]["fits"]),
        csv_file=str(paths["f444w"]["csv"]),
    )
    assert len(dpsf.flt_keys) == 4
    # Filenames must encode detector so DrizzlePSF.get_psf can parse it.
    files = [k[0] for k in dpsf.flt_keys]
    assert any("nrcalong" in f for f in files)
    assert any("nrcblong" in f for f in files)


def test_nircam_lw_detector_separation(tmp_path):
    """A and B footprint centers should be separated by the SIAF-prescribed
    V-frame offset (~2.5 arcmin on sky between NRCA5_FULL and NRCB5_FULL)."""
    paths = _build(tmp_path)
    dpsf = DrizzlePSF(
        driz_image=str(paths["f444w"]["fits"]),
        csv_file=str(paths["f444w"]["csv"]),
    )
    # Pick both detectors from PA=0 frame (frame_id == 1).
    keys_pa0 = [k for k in dpsf.flt_keys if "_00001_" in k[0]]
    assert len(keys_pa0) == 2
    centers = np.array([dpsf.footprint[k].centroid.coords[0] for k in keys_pa0])
    sep_arcsec = np.hypot(
        (centers[0, 0] - centers[1, 0]) * np.cos(np.deg2rad(-5.2)) * 3600.0,
        (centers[0, 1] - centers[1, 1]) * 3600.0,
    )
    # NRCA5/NRCB5 V2 differ by ~175" ≈ 2.9'. Require order-of-magnitude match.
    assert 120.0 < sep_arcsec < 250.0


def test_pa_rotation_changes_footprint(tmp_path):
    """PA=0 and PA=45 footprints should not coincide."""
    paths = _build(tmp_path)
    dpsf = DrizzlePSF(
        driz_image=str(paths["f444w"]["fits"]),
        csv_file=str(paths["f444w"]["csv"]),
    )
    # Frame 1 = PA=0, Frame 2 = PA=45; compare A detectors.
    fp_pa0 = dpsf.footprint[("jw_mock_00001_nrcalong_rate.fits", 1)]
    fp_pa45 = dpsf.footprint[("jw_mock_00002_nrcalong_rate.fits", 1)]
    # Centroids stay at (roughly) field center, but the footprint polygons
    # must differ — intersection area < union area by a visible margin.
    inter = fp_pa0.intersection(fp_pa45).area
    union = fp_pa0.union(fp_pa45).area
    assert inter / union < 0.95  # clearly not identical


def test_half_pixel_alignment(tmp_path):
    """CRPIX_40 = 2·CRPIX_80 − 0.5 AND every CRPIX lands on the UDS-style
    X.5 half-integer grid (e.g. 17280.50, 8640.50) — not X.0, not X.25.
    """
    paths = _build(tmp_path)
    c40 = paths["f444w"]["crpix"]
    c80 = paths["f770w"]["crpix"]
    # Invariant relating the two scales.
    assert c40[0] == pytest.approx(2 * c80[0] - 0.5)
    assert c40[1] == pytest.approx(2 * c80[1] - 0.5)
    # Strict UDS-style X.5 snap at BOTH scales: 2·CRPIX is an odd integer.
    for v in (*c40, *c80):
        doubled = 2 * v
        assert abs(doubled - round(doubled)) < 1e-9, \
            f"CRPIX {v} is not a half-integer"
        assert round(doubled) % 2 == 1, \
            f"CRPIX {v} is not X.5 (UDS convention); got 2·CRPIX = {round(doubled)}"
    s40 = paths["f444w"]["size"]
    s80 = paths["f770w"]["size"]
    assert s40 == (s80[0] * 2, s80[1] * 2)


def test_reproduces_uds_wcs(tmp_path):
    """Explicit mosaic_crval/crpix/npix must be honoured bit-exactly, and the
    derived MIRI grid at 80 mas must match the UDS convention.
    """
    from mophongo.mock_mosaic import MockMosaic, Pointing
    center = (34.366, -5.200)
    mock = MockMosaic(
        out_dir=tmp_path, center_radec=center,
        nircam_lw_frames={'f444w': [Pointing(*center, pa=0.0)]},
        miri_frames={'f770w': [Pointing(*center, pa=0.0)]},
        mosaic_pscale='nircam_lw',
        mosaic_crval=(34.366, -5.200),
        mosaic_crpix=(17280.50, 12672.50),
        mosaic_npix=(34560, 25344),
    )
    paths = mock.write()
    assert paths['f444w']['size'] == (34560, 25344)
    assert paths['f444w']['crpix'] == (17280.50, 12672.50)
    assert paths['f770w']['size'] == (17280, 12672)
    assert paths['f770w']['crpix'] == (8640.50, 6336.50)


def test_block_bin_registration(tmp_path):
    """Pixel (i, j) at the 80 mas scale must point at the same sky position
    as the centre of the 2×2 block of 40 mas pixels (2i, 2i+1)×(2j, 2j+1).
    This is the practical test of half-pixel CRPIX registration.
    """
    import numpy as np
    paths = _build(tmp_path)
    w40 = paths["f444w"]["wcs"]; w80 = paths["f770w"]["wcs"]
    # Check a grid of 10×10 pixel positions at the coarse scale.
    xs_80 = np.linspace(10, paths["f770w"]["size"][0] - 10, 10)
    ys_80 = np.linspace(10, paths["f770w"]["size"][1] - 10, 10)
    X80, Y80 = np.meshgrid(xs_80, ys_80)
    ra_80, dec_80 = w80.wcs_pix2world(X80, Y80, 0)
    # Block-binned 40 mas pixel (i,j) centre = original (2i+0.5, 2j+0.5).
    X40 = 2 * X80 + 0.5
    Y40 = 2 * Y80 + 0.5
    ra_40, dec_40 = w40.wcs_pix2world(X40, Y40, 0)
    assert np.allclose(ra_80, ra_40, atol=1e-12)
    assert np.allclose(dec_80, dec_40, atol=1e-12)


def test_three_family_all_X5(tmp_path):
    """With SW + LW + MIRI all configured, every family's CRPIX is X.5."""
    from mophongo.mock_mosaic import MockMosaic, Pointing
    center = (34.5, -5.2)
    mock = MockMosaic(
        out_dir=tmp_path, center_radec=center,
        nircam_sw_frames={'f200w': [Pointing(*center, pa=0.0)]},
        nircam_lw_frames={'f444w': [Pointing(*center, pa=0.0)]},
        miri_frames     ={'f770w': [Pointing(*center, pa=0.0)]},
        mosaic_pscale='nircam_lw',
    )
    paths = mock.write()
    for fam in ('f200w', 'f444w', 'f770w'):
        for v in paths[fam]['crpix']:
            doubled = 2 * v
            assert abs(doubled - round(doubled)) < 1e-9, \
                f"{fam} CRPIX {v} is not half-integer"
            assert round(doubled) % 2 == 1, \
                f"{fam} CRPIX {v} is not X.5"


def test_sw_lw_miri_defaults(tmp_path):
    """Three-family mock: SW 20 mas, LW 40 mas, MIRI 80 mas, nested."""
    from mophongo.mock_mosaic import MockMosaic, Pointing
    center = (34.5, -5.2)
    mock = MockMosaic(
        out_dir=tmp_path, center_radec=center,
        nircam_sw_frames={'f200w': [Pointing(*center, pa=0.0)]},
        nircam_lw_frames={'f444w': [Pointing(*center, pa=0.0)]},
        miri_frames={'f770w': [Pointing(*center, pa=0.0)]},
    )
    paths = mock.write()
    # SW has 8 detectors, LW has 2, MIRI has 1.
    assert paths['f200w']['n_rows'] == 8
    assert paths['f444w']['n_rows'] == 2
    assert paths['f770w']['n_rows'] == 1
    # Default output pscales.
    assert paths['f200w']['pscale'] == pytest.approx(0.020)
    assert paths['f444w']['pscale'] == pytest.approx(0.040)
    assert paths['f770w']['pscale'] == pytest.approx(0.080)
    # Nested half-pixel alignment: size × pscale is identical across families.
    s20 = paths['f200w']['size'][0]; p20 = paths['f200w']['pscale']
    s40 = paths['f444w']['size'][0]; p40 = paths['f444w']['pscale']
    s80 = paths['f770w']['size'][0]; p80 = paths['f770w']['pscale']
    assert s20 == 4 * s80
    assert s40 == 2 * s80
    # CRPIX_20 = 4·CRPIX_80 - 1.5 on both axes.
    c20 = paths['f200w']['crpix']; c80 = paths['f770w']['crpix']
    assert c20[0] == pytest.approx(4 * c80[0] - 1.5)


def test_inject_noise_roundtrip(tmp_path):
    """inject_noise writes sci+wht and mad_std(sci/σ_pix) ≈ 1 by construction."""
    import numpy as np
    from astropy.io import fits
    from astropy.stats import mad_std
    from mophongo.mock_mosaic import MockMosaic, Pointing
    center = (34.5, -5.2)
    mock = MockMosaic(
        out_dir=tmp_path, center_radec=center,
        nircam_lw_frames={'f444w': [Pointing(*center, pa=0.0),
                                     Pointing(*center, pa=45.0)]},
        noise_K={'f444w': 0.0768},  # σ_nom = K / (p_out · √t_exp)
        noise_seed=0,
    )
    paths = mock.write()
    info = mock.inject_noise('f444w', paths)
    assert info['sci'].exists() and info['wht'].exists()
    sci = fits.getdata(info['sci'])
    wht = fits.getdata(info['wht'])
    mask = info['texp'] > 0
    # Standardized residual: noise / σ_pix should have mad_std ≈ 1.
    std = sci[mask] / (info['R'] / np.sqrt(wht[mask]))
    assert mad_std(std) == pytest.approx(1.0, rel=0.02)
    # Weight-map rms at deepest coverage matches R by construction.
    deep = info['texp'] == info['texp'].max()
    ratio = mad_std(sci[deep]) / (1.0 / np.sqrt(np.median(wht[deep])))
    assert ratio == pytest.approx(info['R'], rel=0.05)


def test_auto_size_encompasses_f444w(tmp_path):
    """Auto-sized mosaic must be the smallest rect covering all LW footprints."""
    import numpy as np
    from mophongo.mock_mosaic import MockMosaic, Pointing
    from mophongo.psf import DrizzlePSF
    from shapely.ops import unary_union
    center = (34.5, -5.2)
    mock = MockMosaic(
        out_dir=tmp_path, center_radec=center,
        nircam_lw_frames={'f444w': [Pointing(*center, pa=0.0),
                                     Pointing(*center, pa=45.0)]},
        # Intentionally no mosaic_size_arcsec override: auto-size from LW footprints.
    )
    paths = mock.write()
    dpsf = DrizzlePSF(driz_image=str(paths['f444w']['fits']),
                      csv_file=str(paths['f444w']['csv']))
    # Every footprint corner must land inside the mosaic (in pixel coords).
    nx, ny = paths['f444w']['size']
    mwcs = paths['f444w']['wcs']
    fp_union = unary_union(list(dpsf.footprint.values()))
    ra_min, dec_min, ra_max, dec_max = fp_union.bounds
    ra_corners  = [ra_min, ra_min, ra_max, ra_max]
    dec_corners = [dec_min, dec_max, dec_min, dec_max]
    xpix, ypix = mwcs.wcs_world2pix(ra_corners, dec_corners, 0)
    assert xpix.min() >= -0.5 and xpix.max() <= nx - 0.5
    assert ypix.min() >= -0.5 and ypix.max() <= ny - 0.5
    # Tightness: mosaic width should exceed bbox width by <= 4 × 80mas (rounding).
    w_bbox = (ra_max - ra_min) * np.cos(np.deg2rad(center[1])) * 3600
    h_bbox = (dec_max - dec_min) * 3600
    assert nx * paths['f444w']['pscale'] - w_bbox <= 4 * 0.080 + 1e-6
    assert ny * paths['f444w']['pscale'] - h_bbox <= 4 * 0.080 + 1e-6


def test_drizzle_correlation_factor():
    # r = 1 ⇒ R = 2/3 (boundary case)
    assert drizzle_correlation_factor(1.0, 1.0, 1.0) == pytest.approx(2 / 3)
    # UDS F444W (p_in=63mas, p_out=40mas, pixfrac=0.75) ⇒ matches empirical ~0.60
    R = drizzle_correlation_factor(0.75, 0.063, 0.04)
    assert R == pytest.approx(0.6077, abs=1e-3)
    # r << 1 ⇒ R → 1 (no suppression)
    assert drizzle_correlation_factor(0.1, 0.01, 1.0) == pytest.approx(1.0 - 0.001 / 3)


def test_psf_region_map(tmp_path):
    """Region map over the 4 NIRCam LW detector footprints should yield
    multiple distinct regions (A, B × 2 PAs + overlaps)."""
    paths = _build(tmp_path)
    dpsf = DrizzlePSF(
        driz_image=str(paths["f444w"]["fits"]),
        csv_file=str(paths["f444w"]["csv"]),
    )
    prm = PSFRegionMap.from_footprints(dpsf.footprint, pa_tol=1.0)
    assert len(prm.regions) >= 3
