"""Tests for :mod:`mophongo.repair` (standalone saturation repair)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from shapely.geometry import Polygon

from mophongo import repair as repair_mod
from mophongo.psf import DrizzlePSF
from mophongo.utils import get_slice_wcs, get_wcs_pscale

SIZE = 256
PSCALE = 0.04
SIGMA = 2.3  # native pixels; FWHM ~ 5.4 px
AMP = 50.0
NOISE = 0.01
HOLE_R = 3.0


class _FakeEffectivePSF:
    """Minimal finite-integral ePSF provider (same shape as test_psf's)."""

    filter_pattern = "TEST"

    def __init__(self):
        self.epsf = OrderedDict({"TEST": np.ones((81, 81, 1), dtype=np.float32)})
        self.extended_epsf = {}

    def get_at_position(self, x, y, filter, rot90=0):
        return self.epsf[filter][:, :, 0]

    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        # Unit-integral gaussian so the drizzled stamp sums to ~1, matching
        # the STDPSF convention (amplitude ~ total flux) assumed by the
        # catalog-repair flux filter.
        norm = 1.0 / (2.0 * np.pi * SIGMA**2)
        psf = norm * np.exp(-0.5 * ((dx / SIGMA) ** 2 + (dy / SIGMA) ** 2))
        return psf.astype(np.float32)


def _make_test_wcs(size: int = SIZE, pscale: float = PSCALE) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2.0 + 0.5, size / 2.0 + 0.5]
    wcs.wcs.crval = [34.5, -5.2]
    wcs.wcs.cdelt = [-pscale / 3600.0, pscale / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.pixel_shape = (size, size)
    wcs.pscale = pscale
    return wcs


def _render_star_stamp(dpsf: DrizzlePSF, cy: int, cx: int, npix: int = 81):
    """Drizzle the fake ePSF onto an ``npix`` cutout centred on (cy, cx)."""
    wcs = dpsf.driz_wcs
    sly = slice(cy - npix // 2, cy - npix // 2 + npix)
    slx = slice(cx - npix // 2, cx - npix // 2 + npix)
    sub = get_slice_wcs(wcs, slx, sly)
    sub.pixel_shape = (npix, npix)
    sub.pscale = get_wcs_pscale(sub)
    ra, dec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
    stamp = np.asarray(dpsf.get_psf(
        ra=ra, dec=dec, filter="TEST", wcs_slice=sub,
        pixfrac=0.75, kernel="square",
    ), dtype=np.float64)
    return stamp, sly, slx


@pytest.fixture
def scene(tmp_path):
    """Synthetic mosaic with one saturated star + matching DrizzlePSF."""
    driz_wcs = _make_test_wcs()
    flt_wcs = _make_test_wcs()
    flt_wcs.expweight = 1.0
    header = driz_wcs.to_header()
    sci_path = tmp_path / "test-f444w_drc_sci.fits"
    wht_path = tmp_path / "test-f444w_drc_wht.fits"
    fits.writeto(sci_path, np.zeros((SIZE, SIZE), dtype=np.float32), header)

    key = ("synthetic_flt.fits", 1)
    info = (
        [key],
        {key: flt_wcs},
        {key: Polygon(flt_wcs.calc_footprint())},
        {key: fits.Header()},
    )
    dpsf = DrizzlePSF(driz_image=str(sci_path), info=info,
                      epsf_obj=_FakeEffectivePSF())

    cy = cx = SIZE // 2
    stamp, sly, slx = _render_star_stamp(dpsf, cy, cx)
    rng = np.random.default_rng(1)
    truth = np.zeros((SIZE, SIZE), dtype=np.float64)
    truth[sly, slx] += AMP * stamp
    sci = truth + rng.normal(0.0, NOISE, size=(SIZE, SIZE))
    wht = np.full((SIZE, SIZE), 1.0 / NOISE**2, dtype=np.float32)

    yy, xx = np.indices((SIZE, SIZE))
    hole = np.hypot(yy - cy, xx - cx) <= HOLE_R
    sci[hole] = 0.0
    wht[hole] = 0.0

    fits.writeto(sci_path, sci.astype(np.float32), header, overwrite=True)
    fits.writeto(wht_path, wht, header, overwrite=True)
    return {
        "sci_path": sci_path, "wht_path": wht_path, "dpsf": dpsf,
        "truth": truth, "hole": hole, "center": (cy, cx),
        "tmp_path": tmp_path,
    }


def _wedge_segmap_and_catalog(center, r_in=5.0, r_out=20.0, n_children=6):
    seg = np.zeros((SIZE, SIZE), dtype=np.int32)
    yc, xc = center
    yy, xx = np.indices(seg.shape)
    rr = np.hypot(yy - yc, xx - xc)
    theta = np.arctan2(yy - yc, xx - xc)
    wedge = (np.floor((theta + np.pi) / (2 * np.pi) * n_children).astype(int)
             % n_children)
    labels = list(range(10, 10 + n_children))
    for k, lbl in enumerate(labels):
        seg[(rr >= r_in) & (rr <= r_out) & (wedge == k)] = lbl
    cat = Table()
    cat["id"] = np.array(labels, dtype=np.int64)
    cat["x"] = np.full(n_children, float(xc), dtype=np.float64)
    cat["y"] = np.full(n_children, float(yc), dtype=np.float64)
    return seg, cat, labels


def test_repair_image_fills_hole(scene):
    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    assert res["sci_out"].exists()
    assert res["wht_out"].exists()
    assert res["csv_out"].exists()

    tbl = res["fits"]
    assert len(tbl) == 1
    assert bool(tbl["ok"][0])
    assert tbl["amplitude"][0] == pytest.approx(AMP, rel=0.1)

    cy, cx = scene["center"]
    sci_rep = fits.getdata(res["sci_out"])
    wht_rep = fits.getdata(res["wht_out"])
    hole = scene["hole"]
    assert np.all(wht_rep[hole] > 0)
    peak = float(scene["truth"][cy, cx])
    assert sci_rep[cy, cx] == pytest.approx(peak, rel=0.2)

    for path in (res["sci_out"], res["wht_out"]):
        hdr = fits.getheader(path)
        assert hdr["SATREPAI"] is True
        assert hdr["SATMODE"] == "repair"
        assert hdr["SATNFIX"] == 1
        assert hdr["SATFILT"] == "TEST"


def test_repair_image_subtract_mode(scene):
    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0, mode="subtract",
    )
    assert res["sci_out"].name.endswith("_subtracted.fits")
    assert res["wht_out"].name.endswith("_subtracted.fits")
    assert res["csv_out"].name.endswith("_saturate_subtract.csv")
    assert res["csv_out"].exists()

    # Saturated core stays blanked (sci = wht = 0), wings removed.
    sci_sub = fits.getdata(res["sci_out"])
    wht_sub = fits.getdata(res["wht_out"])
    hole = scene["hole"]
    assert np.all(wht_sub[hole] == 0)
    assert np.all(sci_sub[hole] == 0)
    cy, cx = scene["center"]
    wing = float(scene["truth"][cy, cx + 5])
    assert abs(float(sci_sub[cy, cx + 5])) < 0.2 * wing
    assert fits.getheader(res["sci_out"])["SATMODE"] == "subtract"


def test_repair_image_no_holes(scene):
    wht = fits.getdata(scene["wht_path"]).copy()
    wht[wht == 0] = 1.0
    fits.writeto(scene["wht_path"], wht, fits.getheader(scene["wht_path"]),
                 overwrite=True)
    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST", fwhm_pix=5.4,
    )
    assert len(res["fits"]) == 0
    assert res["sci_out"].exists()
    assert fits.getheader(res["sci_out"])["SATNFIX"] == 0


def test_repair_image_rejects_bad_mode(scene):
    with pytest.raises(ValueError, match="mode"):
        repair_mod.repair_image(
            scene["sci_path"], scene["wht_path"],
            dpsf=scene["dpsf"], filter_name="TEST", mode="sub",
        )


def test_psf_fwhm_measurement(scene):
    fwhm = repair_mod.psf_fwhm_pix(scene["dpsf"], "TEST")
    assert fwhm == pytest.approx(2.355 * SIGMA, rel=0.25)


def test_flag_catalog_merges_and_flags(scene):
    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])
    cat_path = scene["tmp_path"] / "catalog.fits"
    seg_path = scene["tmp_path"] / "segmap.fits"
    cat.write(cat_path)
    fits.writeto(seg_path, seg)

    out = repair_mod.flag_catalog(
        cat_path, seg_path, res["fits"],
        filter_name="TEST", fwhm_pix=5.4,
    )
    assert out["catalog_out"].exists()
    assert out["segmap_out"].exists()
    assert out["log_out"].exists()

    new_cat = Table.read(out["catalog_out"])
    flag = np.asarray(new_cat["FLAG_SATURATED_TEST"])
    assert flag.sum() == 1
    # children replaced by a single flagged parent
    assert len(new_cat) == 1
    assert int(new_cat["id"][0]) not in labels
    new_seg = fits.getdata(out["segmap_out"])
    assert not np.any(np.isin(new_seg, labels))
    # interior hole closed into the parent segment
    cy, cx = scene["center"]
    assert new_seg[cy, cx] == new_cat["id"][0]


def test_main_end_to_end(scene, monkeypatch):
    seg, cat, _ = _wedge_segmap_and_catalog(scene["center"])
    cat_path = scene["tmp_path"] / "catalog.fits"
    seg_path = scene["tmp_path"] / "segmap.fits"
    cat.write(cat_path)
    fits.writeto(seg_path, seg)
    out_dir = scene["tmp_path"] / "out"

    monkeypatch.setattr(
        repair_mod, "build_drizzle_psf",
        lambda *a, **k: (scene["dpsf"], "TEST"),
    )
    repair_mod.main([
        str(scene["sci_path"]), str(scene["wht_path"]),
        "--filter", "TEST",
        "--catalog", str(cat_path), "--segmap", str(seg_path),
        "--out-dir", str(out_dir),
        "--min-buffer-snr", "5",
    ])

    assert (out_dir / "test-f444w_drc_sci_repaired.fits").exists()
    assert (out_dir / "test-f444w_drc_wht_repaired.fits").exists()
    new_cat = Table.read(out_dir / "catalog_repaired.fits")
    assert np.asarray(new_cat["FLAG_SATURATED_TEST"]).sum() == 1


def test_main_requires_paired_catalog(scene):
    with pytest.raises(SystemExit):
        repair_mod.main([
            str(scene["sci_path"]), str(scene["wht_path"]),
            "--catalog", "cat.fits",
        ])


def test_main_requires_filter_for_catalog(tmp_path):
    # Filename without a filter token and no --filter: refuse the
    # catalog step up front instead of fabricating a flag-column name.
    with pytest.raises(SystemExit):
        repair_mod.main([
            str(tmp_path / "mosaic_sci.fits"), str(tmp_path / "mosaic_wht.fits"),
            "--catalog", "cat.fits", "--segmap", "seg.fits",
        ])
