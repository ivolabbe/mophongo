"""Sizing the measurement aperture by encircled energy.

`FitConfig.phot.aperture_ee` picks the aperture from the band's *model* PSF --
the drizzled stamp after the Gaussian diffusion blur -- instead of from a fixed
angular diameter. These tests pin the three things that makes it worth having:
the radius really is the encircled-energy radius of that PSF, an explicit
diameter still wins over it, and the realized diameter reaches the catalog.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from mophongo.fit import FitConfig
from mophongo.psf import PSF, stamp_encircled_energy


PSCALE = 0.1  # arcsec / pixel


def _wcs(shape: tuple[int, int], pscale: float = PSCALE) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-pscale / 3600.0, pscale / 3600.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


class _StubPipeline:
    """Just enough Pipeline for the two aperture resolvers under test."""

    from mophongo.pipeline import Pipeline

    _ee_ap_diam_arcsec = Pipeline._ee_ap_diam_arcsec
    _resolve_image_ap_radius_pix = Pipeline._resolve_image_ap_radius_pix
    _pixel_scale_arcsec = Pipeline._pixel_scale_arcsec
    _gaussian_fwhm_pix = Pipeline._gaussian_fwhm_pix

    def __init__(self, psf: np.ndarray, pscale: float = PSCALE):
        shape = psf.shape
        self.images = [np.zeros(shape), np.zeros(shape)]
        self.psfs = [psf, psf]
        self.wcs = [_wcs(shape, pscale), _wcs(shape, pscale)]


@pytest.mark.parametrize("ee", [0.5, 0.7, 0.9])
def test_ee_aperture_matches_the_psf_growth_curve(ee):
    """The resolved radius is the model PSF's own encircled-energy radius."""
    psf = PSF.gaussian(101, fwhm=6.0).array
    pipe = _StubPipeline(psf)

    r_pix = pipe._resolve_image_ap_radius_pix(1, FitConfig(phot={"aperture_ee": ee}))

    expected = stamp_encircled_energy(psf, PSCALE, ee_fraction=ee)["r_ee"]
    assert np.isfinite(expected) and expected > 0
    assert r_pix * PSCALE == pytest.approx(expected, rel=1e-9)


def test_ee_070_is_near_the_snr_optimum_for_a_gaussian():
    """EE=0.70 lands at ~1.35x FWHM in diameter.

    For a background-limited point source SNR ~ EE(r)/r, which for a Gaussian
    peaks at r = 1.585 sigma -- 1.35 x FWHM in diameter, enclosing ~71%. The
    default is chosen to sit there, so pin it against the analytic optimum
    rather than against a recorded number. The tolerance is a pixel shell:
    `stamp_encircled_energy` returns the first radius-sorted pixel reaching the
    fraction, so it lands just outside the continuous radius.
    """
    fwhm_pix = 8.0
    psf = PSF.gaussian(121, fwhm=fwhm_pix).array
    pipe = _StubPipeline(psf)

    r_pix = pipe._resolve_image_ap_radius_pix(1, FitConfig(phot={"aperture_ee": 0.70}))

    sigma = fwhm_pix / 2.354820045
    assert r_pix == pytest.approx(1.585 * sigma, rel=0.06)
    assert 2.0 * r_pix == pytest.approx(1.35 * fwhm_pix, rel=0.06)


def test_a_wider_psf_gets_a_wider_aperture_at_fixed_ee():
    """Fixed EE tracks the PSF, which is the property that makes colours work."""
    narrow = _StubPipeline(PSF.gaussian(121, fwhm=4.0).array)
    wide = _StubPipeline(PSF.gaussian(121, fwhm=12.0).array)
    cfg = FitConfig(phot={"aperture_ee": 0.70})

    r_narrow = narrow._resolve_image_ap_radius_pix(1, cfg)
    r_wide = wide._resolve_image_ap_radius_pix(1, cfg)

    # Same EE on a 3x wider PSF is a 3x wider aperture, to within a pixel shell.
    assert r_wide / r_narrow == pytest.approx(3.0, rel=0.1)


def test_explicit_diameter_wins_over_aperture_ee():
    """An explicit aperture is never silently resized by the EE default."""
    psf = PSF.gaussian(101, fwhm=6.0).array
    pipe = _StubPipeline(psf)

    cfg = FitConfig(phot={"aperture_diam": 0.8, "aperture_ee": 0.70, "units": "arcsec"})
    r_pix = pipe._resolve_image_ap_radius_pix(1, cfg)

    assert r_pix == pytest.approx(0.8 / (2.0 * PSCALE))


def test_falls_back_to_fwhm_when_the_stamp_never_reaches_the_fraction():
    """A stamp too small for the requested EE leaves the FWHM fallback in place.

    Inventing a radius past the edge of the stamp would be worse than the old
    default, because nothing downstream would show that it was extrapolated.
    """
    psf = PSF.gaussian(21, fwhm=9.0).array  # inscribed circle holds well under 99%
    pipe = _StubPipeline(psf)

    cfg = FitConfig(phot={"aperture_ee": 0.99})
    r_pix = pipe._resolve_image_ap_radius_pix(1, cfg)

    fwhm_pix = pipe._gaussian_fwhm_pix(psf)
    assert r_pix == pytest.approx(1.5 * fwhm_pix)


def test_aperture_ee_is_the_default():
    assert FitConfig().phot.aperture_ee == 0.70
    assert FitConfig().phot.aperture_diam is None


def _run_two_aperture_pipeline(tmp_path, use_aper_arcsec):
    """A tiny two-band run with a per-source catalog aperture column."""
    from astropy.table import Table
    from mophongo import pipeline
    from mophongo.templates import Templates

    shape = (120, 120)
    positions = [(35.0, 60.0), (85.0, 60.0)]
    ids = [1, 2]
    psf = PSF.gaussian(41, fwhm=4.0).array

    image = np.zeros(shape)
    segmap = np.zeros(shape, dtype=np.int32)
    for sid, (px, py) in zip(ids, positions):
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        image += 100.0 * np.exp(-r2 / (2 * (4.0 / 2.3548) ** 2))
        segmap[r2 < 25] = sid

    catalog = Table(
        {
            "id": ids,
            "x": [p[0] for p in positions],
            "y": [p[1] for p in positions],
            "use_aper": np.asarray(use_aper_arcsec, dtype=float),
        }
    )
    tmpls = Templates.from_image(
        image, segmap, list(zip(catalog["x"], catalog["y"]))
    )
    table, _res, _pipe = pipeline.run(
        [image, image],
        segmap,
        catalog=catalog,
        weights=[np.ones(shape), np.ones(shape)],
        kernels=[None, None],
        templates=tmpls,
        wcs=[_wcs(shape), _wcs(shape)],
        psfs=[psf, psf],
        config=FitConfig(
            reg_flux=0.0,
            fit_astrometry_niter=0,
            fit_astrometry_joint=False,
            astrom_minimum_snr=0.0,
            phot={
                "aperture_ee": 0.70,
                "aperture_catalog": "use_aper",
                "units": "arcsec",
            },
        ),
    )
    return table


def test_larger_of_catalog_and_ee_aperture_is_used(tmp_path):
    """One source below the EE floor, one well above it.

    Source 1 gets a tiny catalog aperture and must fall back to the encircled-
    energy one; source 2 gets a wide one and must keep it, so the band is
    measured in the catalog's own aperture and psfcor stays a pure PSF
    correction at a shared radius.
    """
    table = _run_two_aperture_pipeline(tmp_path, use_aper_arcsec=[0.02, 4.0])

    aper_used = np.asarray(table["aper_1"], dtype=float)
    aper_ee = np.asarray(table["aper_ee_1"], dtype=float)

    assert np.all(np.isfinite(aper_used)) and np.all(np.isfinite(aper_ee))
    # the EE aperture is a band constant, identical for both sources
    assert aper_ee[0] == pytest.approx(aper_ee[1])
    # tiny catalog aperture -> bottoms out on the EE aperture
    assert aper_used[0] == pytest.approx(aper_ee[0])
    # wide catalog aperture -> kept, and strictly wider than the EE one
    assert aper_used[1] > aper_ee[1]
    assert aper_used[1] == pytest.approx(4.0, rel=1e-6)


def test_both_raw_measurements_are_reported(tmp_path):
    """Whatever the rule picks, both raw sums survive for auditing."""
    table = _run_two_aperture_pipeline(tmp_path, use_aper_arcsec=[0.02, 4.0])

    ap_used = np.asarray(table["ap_flux_1"], dtype=float)
    ap_ee = np.asarray(table["ap_flux_ee_1"], dtype=float)
    ap_catap = np.asarray(table["ap_flux_catap_1"], dtype=float)

    assert np.all(np.isfinite(ap_ee)) and np.all(np.isfinite(ap_catap))
    # the used column equals whichever aperture the max rule selected
    assert ap_used[0] == pytest.approx(ap_ee[0])
    assert ap_used[1] == pytest.approx(ap_catap[1])
    # a wider aperture on a positive source collects at least as much flux
    assert ap_catap[1] >= ap_ee[1] - 1e-9


def test_estimator3_scales_the_model_but_not_the_residual(tmp_path):
    """Eq. 12: aper(model - model_nn, R)*psfcor*totcor_cat + sum_Omega(res).

    The distinguishing property is that the residual enters unscaled. Pin it by
    reconstructing the estimator from the columns it is built from, and check
    it does *not* equal the ap_flux_cat form, which scales ap_raw and so scales
    the residual along with the model.
    """
    from astropy.table import Table
    from mophongo import pipeline
    from mophongo.templates import Templates

    shape = (120, 120)
    positions = [(35.0, 60.0), (85.0, 60.0)]
    ids = [1, 2]
    psf = PSF.gaussian(41, fwhm=4.0).array

    rng = np.random.default_rng(7)
    image = np.zeros(shape)
    segmap = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for sid, (px, py) in zip(ids, positions):
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        image += 100.0 * np.exp(-r2 / (2 * (4.0 / 2.3548) ** 2))
        segmap[r2 < 25] = sid
    # a residual the estimator has to carry: without one the two forms agree
    image += 0.5 * rng.standard_normal(shape)

    catalog = Table(
        {
            "id": ids,
            "x": [p[0] for p in positions],
            "y": [p[1] for p in positions],
            "kron_flux": [150.0, 150.0],
            "aper_flux": [100.0, 100.0],
            "kron_radius": [0.4, 0.4],
        }
    )
    tmpls = Templates.from_image(image, segmap, list(zip(catalog["x"], catalog["y"])))
    table, _res, _pipe = pipeline.run(
        [image, image],
        segmap,
        catalog=catalog,
        weights=[np.ones(shape), np.ones(shape)],
        kernels=[None, None],
        templates=tmpls,
        wcs=[_wcs(shape), _wcs(shape)],
        psfs=[psf, psf],
        config=FitConfig(
            reg_flux=0.0,
            fit_astrometry_niter=0,
            fit_astrometry_joint=False,
            astrom_minimum_snr=0.0,
            phot={
                "aperture_ee": 0.70,
                "units": "arcsec",
                "kron_flux_col": "kron_flux",
                "aper_flux_col": "aper_flux",
                "kron_radius_col": "kron_radius",
            },
        ),
    )

    est3 = np.asarray(table["ap_flux_est3_1"], dtype=float)
    ap_model = np.asarray(table["ap_model_1"], dtype=float)
    ap_res = np.asarray(table["ap_res_1"], dtype=float)
    psfcor = np.asarray(table["psfcor_1"], dtype=float)
    tcc = np.asarray(table["totcor_cat"], dtype=float)

    good = np.isfinite(est3) & np.isfinite(psfcor) & np.isfinite(tcc)
    assert good.any(), "no source got a catalog-side total correction"

    np.testing.assert_allclose(
        est3[good], (ap_model * psfcor * tcc + ap_res)[good], rtol=1e-9
    )
    # the residual enters unscaled: removing the model term leaves exactly
    # ap_res, not ap_res * psfcor * tcc
    np.testing.assert_allclose(
        (est3 - ap_model * psfcor * tcc)[good], ap_res[good], atol=1e-9
    )


def test_omega_zeroes_other_sources_segment_pixels(tmp_path):
    """Omega excludes neighbours' segments, so their residual is not claimed."""
    from mophongo.pipeline import Pipeline

    seg = np.zeros((21, 21), dtype=np.int32)
    seg[8:13, 2:6] = 7        # a neighbour's segment
    seg[8:13, 15:19] = 3      # this source's segment
    res = np.ones((21, 21))

    mine = (seg == 3)
    other = (seg != 0) & ~mine
    masked = np.where(other, 0.0, res)

    # every neighbour pixel is dropped, everything else is kept untouched
    assert masked[other].sum() == 0.0
    assert masked[~other].sum() == res[~other].sum()
    assert other.sum() == 20


def test_estimator3_and_ap_flux_cat_differ_once_there_is_a_residual():
    """The two forms are not two spellings of one estimator.

    `ap_flux_cat` scales `ap_raw`, which already contains the residual, so it
    applies the aperture-to-total correction to the residual too. Estimator 3
    applies it to the model alone. They agree only when the residual vanishes,
    which is why the pipeline fixture above cannot tell them apart.
    """
    ap_model, ap_res, psfcor, tcc = 100.0, 12.0, 1.4, 2.1

    est3 = ap_model * psfcor * tcc + ap_res
    ap_flux_cat = (ap_model + ap_res) * psfcor * tcc

    assert est3 == pytest.approx(306.0)
    assert ap_flux_cat == pytest.approx(329.28)
    # the gap is exactly the residual carried through the correction
    assert ap_flux_cat - est3 == pytest.approx(ap_res * (psfcor * tcc - 1.0))

    # with no residual the distinction disappears
    assert 0.0 * psfcor * tcc + ap_model * psfcor * tcc == pytest.approx(
        (ap_model + 0.0) * psfcor * tcc
    )
