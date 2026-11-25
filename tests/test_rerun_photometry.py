"""Redoing the aperture photometry of a finished run without refitting.

`Pipeline.rerun_photometry` (the `phot` step) restores a fit from its written
products and runs the aperture pass again. The fit is the expensive part, so
this is the route for a change to the photometry code or to `fit.phot`. What
has to hold: the same settings reproduce the table, new settings take effect,
and nothing the fit itself produced is touched.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import mophongo.pipeline as pipeline
import mophongo.utils as mutils
from mophongo.fit import FitConfig
from mophongo.pipeline import _aperture_column_names
from utils import make_simple_data

FIT_COLUMNS = ("flux_1", "err_1", "err_pred_1", "flux_1_total", "err_1_total", "scene_1")


@pytest.fixture()
def finished_run(tmp_path):
    """A fitted run with its products on disk, and a factory for fresh pipelines."""
    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    catalog["use_aper"] = 9.0  # a catalog aperture the first run does not use
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    run_cfg = pipeline.RunConfig(
        name="t", out_dir=str(tmp_path), sci_hi="hi.fits", segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
    )

    def fresh(**phot) -> pipeline.Pipeline:
        pipe = pipeline.Pipeline(
            [im.copy() for im in images],
            segmap,
            catalog=catalog,
            weights=[w.copy() for w in wht],
            kernels=[None, kernel],
            psfs=psfs,
            config=FitConfig(fit_astrometry_niter=0, phot=phot),
        )
        pipe.run_config = run_cfg
        pipe.out_dir = tmp_path
        return pipe

    first = fresh()
    first.run()
    # the products write_outputs leaves behind, written directly because
    # write_outputs wants the input FITS files for its headers and figures
    fits.writeto(first.f_residual, first.residuals[0], overwrite=True)
    first.table.write(first.f_fit_table, overwrite=True)
    scenes = fits.BinTableHDU(Table({"id": [1, 2], "niter": [3, 4]}).as_array(), name="SCENES")
    fits.append(first.f_fit_table, scenes.data, scenes.header)
    first.write_stamps()
    return first, fresh


def test_same_settings_reproduce_the_table(finished_run):
    first, fresh = finished_run

    redone = fresh().rerun_photometry()

    written = Table.read(redone.f_fit_table, hdu=1)
    assert written.colnames == first.table.colnames
    for col in first.table.colnames:
        if first.table[col].dtype.kind == "f":
            # templates come back from float32 stamps
            np.testing.assert_allclose(
                written[col], first.table[col], rtol=2e-5, atol=1e-6, equal_nan=True,
                err_msg=col,
            )
        else:
            assert np.all(written[col] == first.table[col]), col
    assert np.isfinite(written["ap_err_1"]).any()


def test_new_settings_change_the_apertures_and_nothing_else(finished_run):
    first, fresh = finished_run

    redone = fresh(aperture_diam=6.0, units="pix").rerun_photometry()

    new = redone.table
    fitted = np.isfinite(first.table["flux_1"])
    # a different aperture, so different aperture sums
    assert not np.allclose(new["ap_flux_1"][fitted], first.table["ap_flux_1"][fitted])
    # the fit is not redone: its columns are the ones that were written
    for col in FIT_COLUMNS:
        np.testing.assert_array_equal(
            np.asarray(new[col]), np.asarray(first.table[col]), err_msg=col
        )


def test_catalog_aperture_added_after_the_fit_is_picked_up(finished_run):
    first, fresh = finished_run
    assert "use_aper" not in first.table.colnames

    redone = fresh(aperture_catalog="use_aper", units="pix").rerun_photometry()

    new = redone.table
    fitted = np.isfinite(new["flux_1"])
    np.testing.assert_array_equal(np.asarray(new["use_aper"]), 9.0)
    # the two raw sums now come from two different apertures
    assert not np.allclose(new["ap_flux_catap_1"][fitted], new["ap_flux_ee_1"][fitted])


def test_replaced_table_is_archived_and_the_scene_extension_carried_over(finished_run):
    first, fresh = finished_run
    before = Table.read(first.f_fit_table, hdu=1)

    redone = fresh(aperture_diam=6.0, units="pix").rerun_photometry()

    # the fit's own table is version 1, kept under that number
    kept = Table.read(redone.f_fit_table.with_name("t_fit_table.v1.fits"), hdu=1)
    np.testing.assert_array_equal(np.asarray(kept["ap_flux_1"]), np.asarray(before["ap_flux_1"]))
    with fits.open(redone.f_fit_table) as hdul:
        assert "SCENES" in hdul
        assert list(hdul["SCENES"].data["niter"]) == [3, 4]
        header = hdul[1].header
        assert header["TABVER"] == 2
        assert "PHOTVER" in header
        # the settings of this pass, not of the fit, travel with the table
        assert "aperture_diam=6.0" in header["PHOTCFG"]


def test_every_pass_keeps_every_earlier_table(finished_run):
    """Versioned, never overwritten: three tables after two passes."""
    first, fresh = finished_run
    original = np.asarray(Table.read(first.f_fit_table, hdu=1)["ap_flux_1"])

    fresh(aperture_diam=6.0, units="pix").rerun_photometry()
    second = np.asarray(Table.read(first.f_fit_table, hdu=1)["ap_flux_1"])
    third_pipe = fresh(aperture_diam=9.0, units="pix").rerun_photometry()

    def column(name: str) -> np.ndarray:
        return np.asarray(Table.read(first.f_fit_table.with_name(name), hdu=1)["ap_flux_1"])

    np.testing.assert_array_equal(column("t_fit_table.v1.fits"), original)
    np.testing.assert_array_equal(column("t_fit_table.v2.fits"), second)
    assert not np.allclose(column("t_fit_table.fits"), second, equal_nan=True)
    assert fits.getheader(first.f_fit_table.with_name("t_fit_table.v2.fits"), 1)["TABVER"] == 2
    assert fits.getheader(third_pipe.f_fit_table, 1)["TABVER"] == 3
    assert sorted(p.name for p in first.out_dir.glob("t_fit_table*.fits")) == [
        "t_fit_table.fits", "t_fit_table.v1.fits", "t_fit_table.v2.fits",
    ]


def test_a_stale_value_cannot_outlive_the_pass(finished_run):
    """Columns the pass leaves unset must not keep what an earlier pass wrote.

    The catalog-tied columns are only filled where `totcor_cat` exists. A table
    made with the Kron columns set, redone without them, would otherwise keep
    its old `ap_flux_cat` and `totcor_cat` next to fluxes they no longer match.
    """
    first, fresh = finished_run
    stale = Table.read(first.f_fit_table, hdu=1)
    assert "ap_flux_cat_1" in _aperture_column_names(1)
    stale["ap_flux_cat_1"] = 123.0
    stale["ap_err_est3_1"] = 123.0
    stale["totcor_cat"] = 2.0
    stale.write(first.f_fit_table, overwrite=True)

    redone = fresh().rerun_photometry()

    assert np.all(np.isnan(redone.table["ap_flux_cat_1"]))
    assert np.all(np.isnan(redone.table["ap_err_est3_1"]))
    assert "totcor_cat" not in redone.table.colnames
