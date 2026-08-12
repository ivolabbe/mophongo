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


def test_flag_catalog_merge_mode(scene):
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
        filter_name="TEST", fwhm_pix=5.4, merge=True,
    )
    assert out["catalog_out"].name == "catalog_repaired.fits"
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


def test_flag_saturated_segments(scene):
    from mophongo.catalog import flag_saturated_segments

    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])
    # Independent neighbour inside the stamp window: its own flux
    # dominates over the star wings there, so it must be kept.
    cy, cx = scene["center"]
    yy, xx = np.indices(seg.shape)
    neigh = np.hypot(yy - cy, xx - (cx + 37)) <= 4
    seg[neigh] = 99
    cat.add_row({"id": 99, "x": float(cx + 37), "y": float(cy)})
    sci = res["sci"].astype(np.float64).copy()
    sci[neigh] += 5.0

    stamp = repair_mod.drizzled_psf_stamp(scene["dpsf"], "TEST", npix=101)
    new_cat, new_seg, log = flag_saturated_segments(
        cat, seg, res["fits"],
        sci=sci, psf_stamp=stamp, filter_name="TEST", flux_frac=0.3,
    )

    gid = min(labels)
    flag = dict(zip(np.asarray(new_cat["id"]),
                    np.asarray(new_cat["FLAG_SATURATED_TEST"])))
    # all wedges of the star share the group id = lowest flagged label
    assert all(flag[lbl] == gid for lbl in labels)
    assert flag[99] == 0                           # neighbour kept
    assert len(new_cat) == len(cat)                # no rows dropped
    # flagged wing labels zeroed except within r_in of the star centre,
    # where the repaired pixels carry the group id; the enclosed core
    # (seg=0 before) is filled with the group id too
    assert np.any(new_seg == 99)                   # neighbour label kept
    assert new_seg[cy, cx] == gid                  # core filled
    wings = np.isin(seg, labels)
    assert set(np.unique(new_seg[wings])) <= {0, gid}

    flagged_rows = log[np.asarray(log["flagged"])]
    assert set(np.asarray(flagged_rows["seg_id"])) == set(labels)
    assert set(np.asarray(flagged_rows["group_id"])) == {gid}
    # Neighbour is either evaluated-and-kept or outside the model
    # support (never evaluated) — flagged in neither case.
    row99 = log[np.asarray(log["seg_id"]) == 99]
    assert not any(np.asarray(row99["flagged"]))


def test_flag_saturated_segments_truncated_stamp(scene):
    # An ePSF is zero beyond its native FOV. Flux outside the model
    # support (e.g. real diffraction spikes) must not dilute the ratio —
    # the comparison runs on support pixels only.
    from mophongo.catalog import flag_saturated_segments

    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"], r_out=40.0)
    stamp = repair_mod.drizzled_psf_stamp(scene["dpsf"], "TEST", npix=101)
    c = 101 // 2
    yy, xx = np.indices(stamp.shape)
    stamp[np.hypot(yy - c, xx - c) > 10] = 0.0  # truncate support at 10 px

    # Bright "spike" flux in the wedges beyond the support.
    cy, cx = scene["center"]
    yy, xx = np.indices(seg.shape)
    rr = np.hypot(yy - cy, xx - cx)
    sci = res["sci"].astype(np.float64).copy()
    sci[(seg > 0) & (rr > 10)] += 3.0

    new_cat, _, _ = flag_saturated_segments(
        cat, seg, res["fits"],
        sci=sci, psf_stamp=stamp, filter_name="TEST",
    )
    flag = dict(zip(np.asarray(new_cat["id"]),
                    np.asarray(new_cat["FLAG_SATURATED_TEST"])))
    assert all(flag[lbl] == min(labels) for lbl in labels)


def test_flag_saturated_segments_csv_roundtrip(scene, tmp_path):
    # 'ok' becomes the strings "True"/"False" after a CSV round-trip;
    # a naive astype(bool) would treat "False" as True.
    from mophongo.catalog import flag_saturated_segments

    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    csv = tmp_path / "fit.csv"
    res["fits"].write(csv, format="csv", overwrite=True)
    fit = Table.read(csv)
    fit["ok"] = ["False"] * len(fit)

    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])
    stamp = repair_mod.drizzled_psf_stamp(scene["dpsf"], "TEST", npix=101)
    new_cat, new_seg, log = flag_saturated_segments(
        cat, seg, fit, sci=res["sci"], psf_stamp=stamp, filter_name="TEST",
    )
    assert np.asarray(new_cat["FLAG_SATURATED_TEST"]).sum() == 0
    assert len(log) == 0


def test_repair_in_memory(scene, tmp_path):
    """Array-in/array-out flow: wht filled, wing labels kept, star grouped."""
    from astropy.wcs import WCS as _WCS

    sci = fits.getdata(scene["sci_path"]).astype(np.float32)
    wht = fits.getdata(scene["wht_path"]).astype(np.float32)
    wcs = _WCS(fits.getheader(scene["sci_path"]))
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])
    out_dir = tmp_path / "repaired"

    res = repair_mod.repair_in_memory(
        sci, wht,
        dpsf=scene["dpsf"], wcs=wcs, psf_pattern="TEST",
        catalog=cat, segmap=seg,
        out_dir=out_dir, fwhm_pix=5.4,
        min_buffer_snr=5.0, stamp_npix=101,
    )
    hole = scene["hole"]
    cy, cx = scene["center"]
    # inputs untouched, outputs repaired; wht cores filled (non-zero)
    assert np.all(wht[hole] == 0)
    assert np.all(res["wht"][hole] > 0)
    assert res["sci"][cy, cx] == pytest.approx(scene["truth"][cy, cx], rel=0.2)
    # zero_segments=False: wing labels keep their segments for templates,
    # the core carries the group id, and the flag column groups the star
    gid = min(labels)
    new_seg = res["segmap"]
    assert all(np.any(new_seg == lbl) for lbl in labels)
    assert new_seg[cy, cx] == gid
    flags = np.asarray(res["catalog"]["FLAG_SATURATED_TMPL"])
    assert set(flags[flags > 0]) == {gid}
    assert int((flags > 0).sum()) == len(labels)
    # diagnostics on disk
    assert (out_dir / "saturate_repair.fits").exists()
    assert (out_dir / "flag_log.csv").exists()
    assert (out_dir / "flag_diagnostic.png").exists()


def test_repair_star(scene, tmp_path):
    """One-star convenience: coordinate in, diagnostic + repaired cutout out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cy, cx = scene["center"]
    png = tmp_path / "star.png"
    out = repair_mod.repair_star(
        scene["sci_path"], scene["wht_path"],
        x=cx + 2.0, y=cy - 1.5,          # slightly off-centre on purpose
        dpsf=scene["dpsf"], wcs=None, psf_pattern="TEST",
        cutout=200, fwhm_pix=5.4, min_buffer_snr=5.0,
        to_file=png,
    )
    assert len(out["fit"]) == 1 and bool(out["fit"]["ok"][0])
    assert out["fit"]["amplitude"][0] == pytest.approx(AMP, rel=0.1)
    assert png.exists()
    sly, slx = out["slices"]
    hole_local = scene["hole"][sly, slx]
    assert np.all(out["wht"][hole_local] > 0)   # core wht filled

    # live figure mode
    out2 = repair_mod.repair_star(
        scene["sci_path"], scene["wht_path"],
        x=cx, y=cy, dpsf=scene["dpsf"], psf_pattern="TEST",
        cutout=200, fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    assert out2["fig"] is not None
    plt.close(out2["fig"])

    # far-away coordinates refuse loudly
    with pytest.raises(ValueError, match="search_radius"):
        repair_mod.repair_star(
            scene["sci_path"], scene["wht_path"],
            x=60.0, y=60.0, dpsf=scene["dpsf"], psf_pattern="TEST",
            cutout=200, search_radius=10.0, fwhm_pix=5.4,
        )


def test_hybrid_psf_stamp():
    """Core kept verbatim inside its support, halo continues outside."""
    n_core, n_halo = 41, 121
    yy, xx = np.indices((n_halo, n_halo))
    rr = np.hypot(yy - n_halo // 2, xx - n_halo // 2)
    halo = np.exp(-rr / 12.0)
    halo /= halo.sum()
    cy, cx = np.indices((n_core, n_core))
    rc = np.hypot(cy - n_core // 2, cx - n_core // 2)
    core = np.exp(-rc / 4.0)          # different (sharper) profile
    core[rc > 18] = 0.0               # finite support like a real ePSF
    core /= core.sum()

    hyb = repair_mod.hybrid_psf_stamp(core, halo)
    assert hyb.shape == halo.shape
    assert hyb.sum() == pytest.approx(1.0)
    # halo support present far outside the core stamp
    assert hyb[n_halo // 2, n_halo // 2 + 40] > 0
    # inside the graft region the profile follows the (renormalised) core:
    # compare shape via center-to-r ratio
    c = n_halo // 2
    ratio_hyb = hyb[c, c + 8] / hyb[c, c]
    ratio_core = core[n_core // 2, n_core // 2 + 8] / core[n_core // 2, n_core // 2]
    assert ratio_hyb == pytest.approx(ratio_core, rel=1e-3)


def test_repair_in_memory_hybrid_when_stamp_differs(scene, monkeypatch):
    """A distinct stamp PSF triggers the hybrid core+halo flag model."""
    from astropy.wcs import WCS as _WCS

    calls = []
    real_stamp = repair_mod.drizzled_psf_stamp

    def spy(dpsf, pattern, *, npix=201):
        calls.append((dpsf, pattern, npix))
        return real_stamp(dpsf, pattern, npix=npix)

    monkeypatch.setattr(repair_mod, "drizzled_psf_stamp", spy)

    sci = fits.getdata(scene["sci_path"]).astype(np.float32)
    wht = fits.getdata(scene["wht_path"]).astype(np.float32)
    wcs = _WCS(fits.getheader(scene["sci_path"]))
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])

    # separate DrizzlePSF object standing in for the 30" halo grids
    halo_dpsf = DrizzlePSF(
        driz_image=str(scene["sci_path"]),
        info=(scene["dpsf"].flt_keys, scene["dpsf"].wcs,
              scene["dpsf"].footprint, scene["dpsf"].hdrs),
        epsf_obj=_FakeEffectivePSF(),
    )
    res = repair_mod.repair_in_memory(
        sci, wht,
        dpsf=scene["dpsf"], wcs=wcs, psf_pattern="TEST",
        catalog=cat, segmap=seg,
        stamp_dpsf=halo_dpsf, stamp_pattern="TEST", stamp_npix=121,
        out_dir=None, fwhm_pix=5.4, min_buffer_snr=5.0, plots=False,
    )
    # two stamps drizzled: the halo model and the MJD-matched core
    assert len(calls) == 2
    assert calls[0][0] is halo_dpsf and calls[1][0] is scene["dpsf"]
    flags = np.asarray(res["catalog"]["FLAG_SATURATED_TMPL"])
    assert int((flags > 0).sum()) == len(labels)


def test_repair_in_memory_stamp_psf_override(scene, tmp_path, monkeypatch):
    """The flag model can come from a separate large-FOV PSF."""
    from astropy.wcs import WCS as _WCS

    calls = []
    real_stamp = repair_mod.drizzled_psf_stamp

    def spy(dpsf, pattern, *, npix=201):
        calls.append((dpsf, pattern, npix))
        return real_stamp(dpsf, pattern, npix=npix)

    monkeypatch.setattr(repair_mod, "drizzled_psf_stamp", spy)

    sci = fits.getdata(scene["sci_path"]).astype(np.float32)
    wht = fits.getdata(scene["wht_path"]).astype(np.float32)
    wcs = _WCS(fits.getheader(scene["sci_path"]))
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])

    large_dpsf = scene["dpsf"]  # stands in for the 30" GRID5 model
    res = repair_mod.repair_in_memory(
        sci, wht,
        dpsf=scene["dpsf"], wcs=wcs, psf_pattern="TEST",
        catalog=cat, segmap=seg,
        stamp_dpsf=large_dpsf, stamp_pattern="TEST", stamp_npix=101,
        out_dir=None, fwhm_pix=5.4, min_buffer_snr=5.0, plots=False,
    )
    # the flag stamp was drizzled from the override PSF
    assert calls and calls[-1][0] is large_dpsf and calls[-1][2] == 101
    flags = np.asarray(res["catalog"]["FLAG_SATURATED_TMPL"])
    assert int((flags > 0).sum()) == len(labels)


def test_flag_catalog_flag_mode_with_diagnostic(scene):
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

    stamp = repair_mod.drizzled_psf_stamp(scene["dpsf"], "TEST", npix=101)
    out = repair_mod.flag_catalog(
        cat_path, seg_path, res["fits"],
        filter_name="TEST", sci=res["sci"], psf_stamp=stamp,
        sci_before=fits.getdata(scene["sci_path"]),
    )
    assert out["catalog_out"].name == "catalog_flagged.fits"
    assert out["segmap_out"].name == "segmap_flagged.fits"
    assert out["log_out"].name == "catalog_flaglog.csv"
    assert out["diagnostic_out"] is not None
    assert out["diagnostic_out"].exists()

    new_cat = Table.read(out["catalog_out"])
    assert len(new_cat) == len(cat)
    flags = np.asarray(new_cat["FLAG_SATURATED_TEST"])
    assert int((flags > 0).sum()) == len(labels)
    assert set(flags[flags > 0]) == {min(labels)}


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
    # default catalog step is flag-only: rows kept, wedges flagged with
    # the shared group id, diagnostic written
    new_cat = Table.read(out_dir / "catalog_flagged.fits")
    assert len(new_cat) == 6
    flags = np.asarray(new_cat["FLAG_SATURATED_TMPL"])
    assert int((flags > 0).sum()) == 6
    assert set(flags[flags > 0]) == {10}
    assert (out_dir / "catalog_flag_diagnostic.png").exists()
    new_seg = fits.getdata(out_dir / "segmap_flagged.fits")
    # wedge labels zeroed; the enclosed core carries the group id
    assert set(np.unique(new_seg)) == {0, 10}
    cy, cx = scene["center"]
    assert new_seg[cy, cx] == 10


def test_main_merge_mode(scene, monkeypatch):
    seg, cat, _ = _wedge_segmap_and_catalog(scene["center"])
    cat_path = scene["tmp_path"] / "catalog.fits"
    seg_path = scene["tmp_path"] / "segmap.fits"
    cat.write(cat_path)
    fits.writeto(seg_path, seg)
    out_dir = scene["tmp_path"] / "out_merge"

    monkeypatch.setattr(
        repair_mod, "build_drizzle_psf",
        lambda *a, **k: (scene["dpsf"], "TEST"),
    )
    repair_mod.main([
        str(scene["sci_path"]), str(scene["wht_path"]),
        "--filter", "TEST",
        "--catalog", str(cat_path), "--segmap", str(seg_path),
        "--out-dir", str(out_dir),
        "--min-buffer-snr", "5", "--merge",
    ])

    new_cat = Table.read(out_dir / "catalog_repaired.fits")
    assert len(new_cat) == 1
    assert np.asarray(new_cat["FLAG_SATURATED_TMPL"]).sum() == 1


def test_main_requires_paired_catalog(scene):
    with pytest.raises(SystemExit):
        repair_mod.main([
            str(scene["sci_path"]), str(scene["wht_path"]),
            "--catalog", "cat.fits",
        ])



def _wing_stamp(npix=201, a=3.0, beta=1.2):
    """Unit-sum stamp with a power-law wing.

    The fixture's drizzled PSF is a pure Gaussian -- 1e-11 of its peak by
    r = 15 px -- so it has no halo to speak of. A Moffat-like wing is what
    a real saturated star's model looks like far from the core, which is
    what the two tests below are about.
    """
    c = npix // 2
    yy, xx = np.indices((npix, npix))
    rr = np.hypot(yy - c, xx - c)
    stamp = (1.0 + (rr / a) ** 2) ** (-beta)
    return stamp / stamp.sum()


def test_flag_group_id_is_the_core_segment(scene):
    """The group id names the segment on the star, not the lowest id.

    The id is also the label stamped on the filled core, so it has to
    belong to a row that sits on the star: a distant spike fragment with a
    lower id may be cut by the pipeline's footprint or trial-radius filter,
    and then nothing models the core.
    """
    from mophongo.catalog import flag_saturated_segments

    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    seg, cat, labels = _wedge_segmap_and_catalog(scene["center"])
    cy, cx = scene["center"]
    # a far spike fragment carrying a LOWER id than any core wedge
    yy, xx = np.indices(seg.shape)
    far = (np.abs(yy - cy) <= 1) & (xx - cx > 24) & (xx - cx < 40)
    seg[far] = 5
    cat.add_row({"id": 5, "x": float(cx + 32), "y": float(cy)})

    new_cat, new_seg, _log = flag_saturated_segments(
        cat, seg, res["fits"], sci=res["sci"].astype(np.float64),
        psf_stamp=_wing_stamp(), filter_name="TEST", sky_noise=0.002,
    )
    flag = dict(zip(np.asarray(new_cat["id"]),
                    np.asarray(new_cat["FLAG_SATURATED_TEST"])))
    assert flag[5] > 0, "the far fragment must still be flagged"
    gids = {v for v in flag.values() if v > 0}
    assert len(gids) == 1                 # one star, one group id
    assert gids.pop() == min(labels)      # a core wedge, not the lower id 5
    # and the core it labels is modelled: the group id owns the filled core
    assert int(new_seg[cy, cx]) == min(labels)


def test_halo_nsigma_flags_bright_spikes_the_ratio_test_misses(scene):
    """A segment the star's halo lights up is flagged on its own.

    A saturated star's real diffraction spikes run far above the ePSF's, so
    a spike segment can hold several times the model flux and fail the
    ratio test while the model there is still tens of sigma above the sky
    (measured on UDS: spike segments at frac 0.03-0.29 with the halo at
    6-94 sigma).
    """
    from mophongo.catalog import flag_saturated_segments

    res = repair_mod.repair_image(
        scene["sci_path"], scene["wht_path"],
        dpsf=scene["dpsf"], filter_name="TEST",
        fwhm_pix=5.4, min_buffer_snr=5.0,
    )
    seg, cat, _labels = _wedge_segmap_and_catalog(scene["center"])
    sci = res["sci"].astype(np.float64).copy()
    stamp = _wing_stamp()

    kw = dict(psf_stamp=stamp, filter_name="TEST", sky_noise=0.002)
    _c, _s, log0 = flag_saturated_segments(
        cat.copy(), seg.copy(), res["fits"], sci=sci, halo_nsigma=0.0, **kw)
    m0 = np.asarray(log0["seg_id"]) == 11
    model_sb = float(np.asarray(log0["model_flux"])[m0][0]
                     / np.asarray(log0["npix"])[m0][0])

    # make that wedge 20x brighter than the star model predicts there:
    # ratio collapses, model surface brightness is untouched
    spike = seg == 11
    sci[spike] += 20.0 * model_sb

    _c, _s, log_ratio = flag_saturated_segments(
        cat.copy(), seg.copy(), res["fits"], sci=sci, halo_nsigma=0.0, **kw)
    _c, _s, log_halo = flag_saturated_segments(
        cat.copy(), seg.copy(), res["fits"], sci=sci, halo_nsigma=5.0, **kw)

    def verdict(log):
        m = np.asarray(log["seg_id"]) == 11
        return (bool(np.asarray(log["flagged"])[m][0]),
                float(np.asarray(log["frac"])[m][0]),
                float(np.asarray(log["halo_sig"])[m][0]))

    hit_ratio, frac, _ = verdict(log_ratio)
    hit_halo, _, halo_sig = verdict(log_halo)
    assert frac < 0.3            # the ratio test cannot see it
    assert not hit_ratio
    assert halo_sig > 5.0        # but the model halo is bright there
    assert hit_halo
