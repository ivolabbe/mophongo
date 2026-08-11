import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from astropy.table import Table
from astropy.nddata import Cutout2D
from photutils.psf.matching import SplitCosineBellWindow, TukeyWindow
import mophongo.utils as mutils

import mophongo.pipeline as pipeline
from mophongo.verification import build_source_recovery_table, prepare_psf_shape, wht_noise_check
from utils import (
    make_simple_data,
    save_diagnostic_image,
    save_flux_vs_truth_plot,
)


class _CaptureAxes:
    def __init__(self):
        self.kwargs = None

    def imshow(self, _data, **kwargs):
        self.kwargs = kwargs


def test_source_stage_template_diagnostic_uses_five_mad_scaling():
    ax = _CaptureAxes()
    stamp = np.array(
        [
            [-2.0, -1.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    pipeline.Pipeline._imshow_scaled(ax, stamp)

    scale = float(mad_std(stamp.ravel(), ignore_nan=True))
    assert ax.kwargs["vmin"] == -5.0 * scale
    assert ax.kwargs["vmax"] == 5.0 * scale


def test_lowres_upsampled_inverse_variance_preserves_native_predicted_error():
    rng = np.random.default_rng(123)
    factor = 2
    image_native = rng.normal(size=(5, 6)).astype(np.float32)
    weight_native = np.exp(rng.normal(size=(5, 6))).astype(np.float32)
    template_native = rng.random(size=(5, 6)).astype(np.float32)

    image_hi, weight_hi = pipeline._upsample_flux_conserving_image_and_ivar(
        image_native,
        weight_native,
        factor,
    )
    template_hi = np.kron(
        template_native,
        np.ones((factor, factor), dtype=np.float32) / factor**2,
    )

    native_inv_err = np.sqrt(np.sum(weight_native * template_native**2))
    hi_inv_err = np.sqrt(np.sum(weight_hi * template_hi**2))

    np.testing.assert_allclose(image_hi.sum(), image_native.sum(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        weight_hi[::factor, ::factor],
        weight_native * factor**2,
        rtol=1e-7,
        atol=0,
    )
    np.testing.assert_allclose(hi_inv_err, native_inv_err, rtol=1e-7, atol=0)


def test_pipeline_catalog_total_flux_columns_apply_psf_throughput():
    class DummyTemplate:
        id = 7
        parent_id = None

    pipe = object.__new__(pipeline.Pipeline)
    pipe.config = pipeline._FitConfig()
    cat = Table({"id": [7]})

    pipe._update_catalog_with_fluxes(
        cat,
        [DummyTemplate()],
        np.array([95.0]),
        np.array([9.5]),
        np.array([4.75]),
        0.95,
        1,
    )

    assert cat["throughput_1"][0] == 0.95
    assert cat["flux_1"][0] == 95.0
    assert cat["err_1"][0] == 9.5
    np.testing.assert_allclose(cat["flux_1_total"][0], 100.0)
    np.testing.assert_allclose(cat["err_1_total"][0], 10.0)
    np.testing.assert_allclose(cat["err_pred_1_total"][0], 5.0)


def test_filter_psf_throughput_uses_static_psf_sum():
    psf = np.full((5, 5), 0.95 / 25.0)

    np.testing.assert_allclose(pipeline._filter_psf_throughput(psf), 0.95)


def test_filter_psf_throughput_prefers_explicit_scalar():
    psf = np.ones((5, 5), dtype=float)

    assert pipeline._filter_psf_throughput(psf, explicit_throughput=0.93) == 0.93


def test_verification_psf_shape_and_recovery_table():
    psf = np.full((3, 3), 0.9 / 9.0)
    prepared = prepare_psf_shape(psf, "test")
    np.testing.assert_allclose(prepared.shape.sum(), 1.0)
    np.testing.assert_allclose(prepared.throughput, 0.9)

    fit = Table()
    fit["id"] = [1, 2]
    fit["flux_1"] = [90.0, 180.0]
    fit["err_1"] = [9.0, 18.0]
    fit["err_pred_1"] = [4.5, 9.0]
    fit["throughput_1"] = [0.9, 0.9]
    fit["flux_1_total"] = [100.0, 200.0]
    fit["err_1_total"] = [10.0, 20.0]
    fit["err_pred_1_total"] = [5.0, 10.0]
    fit["is_deblended"] = [False, True]
    truth = Table()
    truth["id"] = [1, 2]
    truth["flux_true"] = [100.0, 200.0]
    truth["snr"] = [20.0, 20.0]
    truth["is_point_source"] = [True, False]

    out = build_source_recovery_table(
        fit,
        truth,
        true_flux_col="flux_true",
        fitted_indices=[1],
        snr_col="snr",
    )

    np.testing.assert_allclose(out["ratio_1"], [1.0, 1.0])
    np.testing.assert_allclose(out["ratio_1_model"], [0.9, 0.9])
    np.testing.assert_allclose(out["pull_1_pred"], [0.0, 0.0])
    assert bool(out["is_deblended"][1])


def test_verification_wht_noise_check():
    rng = np.random.default_rng(123)
    truth = np.ones((64, 64), dtype=float)
    wht = np.full_like(truth, 4.0)
    sci = truth + rng.normal(scale=0.5, size=truth.shape)

    check = wht_noise_check(sci, truth, wht, filter_name="f000w")

    assert check.filter_name == "f000w"
    assert check.n_pix == truth.size
    assert 0.9 < check.std < 1.1


def test_pipeline_flux_recovery(tmp_path):
    #    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(seed=5, nsrc=300, size=501, ndilate=1, peak_snr=1)
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms)

    images, segmap, catalog, psfs, truth_img, wht = make_simple_data(
        seed=5, nsrc=151, size=301, ndilate=2, peak_snr=1.5
    )
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms, extend_templates='psf')

    # add the hires images as the first fitting image, so that we can compare fluxes
    images.insert(0, images[0])
    wht.insert(0, wht[0])
    psfs.insert(0, psfs[0])
    # images are: hires, hires, lowres
    # psfs are:   hires, hires, lowres
    # so this would add psf hires wings to templates, and result in a delta function for kernel
    dirac = lambda n: ((np.arange(n)[:, None] == n // 2) & (np.arange(n) == n // 2)).astype(float)

    kernel = [mutils.matching_kernel(psfs[0], psf) for psf in psfs]
    kernel[0] = kernel[1] = dirac(3)  # no kernel for the first image, it is the hires image
    table, resid, templates = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=kernel,
        # no detection PSF supplied here: this test predates the wings default
        config=pipeline._FitConfig(extend_mode="none"),
    )

    # @@@ sometimes flux_true is NEGATIVE?
    table["flux_true"] = catalog["flux_true"]  # add flux_true to the table

    # Plot for high-res (flux_0) vs truth
    flux_hi_plot = tmp_path / "flux_hi_vs_true.png"
    save_flux_vs_truth_plot(
        flux_hi_plot,
        table["flux_true"],
        table["flux_1"],
        error=table["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_hi_plot.exists()

    # Plot for low-res (flux_1) vs truth
    flux_lo_plot = tmp_path / "flux_lo_vs_true.png"
    save_flux_vs_truth_plot(
        flux_lo_plot,
        table["flux_true"],
        table["flux_2"],
        error=table["err_2"],  # Add error column
        label="Flux (lowres)",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_plot.exists()

    # Plot for flux_lo vs flux_hi with error propagation
    flux_lo_hi_plot = tmp_path / "flux_lo_vs_hi.png"
    # Calculate combined error for hires vs lowres comparison
    combined_error = np.sqrt(table["err_1"] ** 2 + table["err_2"] ** 2)
    save_flux_vs_truth_plot(
        flux_lo_hi_plot,
        table["flux_1"],
        table["flux_2"],
        error=combined_error,
        label="Flux (lowres) vs (hires)",
        xlabel="Recovered Flux (hires)",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_hi_plot.exists()

    # ----------------------------------- separate run for high-res, using the truth image as templates
    # images, segmap, catalog=catalog, psfs=psfs,  wht_images=wht)
    table_true, resid_hi, templates_true = pipeline.run(
        [truth_img, images[1]],
        segmap,
        catalog=catalog,
        kernels=[dirac(3), psfs[1]],
        weights=[np.zeros(wht[0].shape), wht[1]],
        config=pipeline._FitConfig(extend_mode="none"),
    )
    table_true["flux_true"] = catalog["flux_true"]
    # Plot for high-res (flux_0) vs truth
    flux_true_plot = tmp_path / "flux_hi_vs_true_truemodel.png"
    save_flux_vs_truth_plot(
        flux_true_plot,
        table_true["flux_true"],
        table_true["flux_1"],
        error=table_true["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_true_plot.exists()

    model = images[2] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(
        fname, truth_img, images[1], images[2], model, resid[1], segmap=segmap, catalog=catalog
    )
    fname = tmp_path / "diagnostic_hires.png"
    model = images[1] - resid[0]
    save_diagnostic_image(
        fname, truth_img, images[0], images[1], model, resid[0], segmap=segmap, catalog=catalog
    )

    fname = tmp_path / "diagnostic_hires_truemodel.png"
    model = images[1] - resid_hi[0]
    save_diagnostic_image(
        fname, truth_img, truth_img, images[1], model, resid_hi[0], segmap=segmap, catalog=catalog
    )
    assert fname.exists()

    # Report statistics for flux recovery
    for idx in range(1, len(psfs)):
        col = f"flux_{idx}"
        ratio = np.array(table[col]) / np.array(table["flux_true"])
        print(
            f"flux_{idx}/flux_true percentiles: 5th={np.percentile(ratio,5):.2f}, "
            f"16th={np.percentile(ratio,16):.2f}, 50th={np.percentile(ratio,50):.2f}, "
            f"84th={np.percentile(ratio,84):.2f}, 95th={np.percentile(ratio,95):.2f}"
        )

    # sanity check on propagated errors for low-res image
    from mophongo.psf import PSF
    from mophongo.templates import Templates

    psf_hi = PSF.from_array(psfs[1])
    psf_lo = PSF.from_array(psfs[2])
    kernel = psf_hi.matching_kernel(psf_lo)
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    noise_std = wht[1][0, 0]
    err_pred = np.array([noise_std / np.sqrt((t.data**2).sum()) for t in tmpls.templates])
    ratio_err = table["err_1"] / err_pred
    assert np.allclose(np.mean(ratio_err), 1.0, atol=3)

    # Write catalog with all columns formatted to 3 digits after the decimal
    for col in table.colnames:
        if table[col].dtype.kind in "fc":  # float or complex
            table[col].info.format = ".3f"

    cat_file = tmp_path / "photometry.cat"
    table.write(cat_file, format="ascii.commented_header")
    assert cat_file.exists()

    loaded = Table.read(cat_file, format="ascii.commented_header")
    assert len(loaded) == len(table)


def test_write_stamps_variable_size_single_file(tmp_path):
    from astropy.io import fits

    images, segmap, catalog, psfs, _truth_img, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    _table, _residuals, pipe = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=[None, kernel], psfs=psfs
    )

    path = tmp_path / "stamps.fits"
    out = pipe.write_stamps(path)
    assert out == path

    conv = pipe.all_templates[0]
    hi_by_id = {int(t.id): t for t in pipe.tmpls.templates}

    with fits.open(path) as hdul:
        src = hdul["SOURCES"].data
        assert hdul[0].header["NSRC"] == len(conv)
        assert len(src) == len(conv)
        # PSFs are not duplicated into the file: static PSFs carry key 0
        assert np.all(src["key_psf_hi"] == 0)
        assert np.all(src["key_psf_lo"] == 0)
        for row, t_lo in zip(src, conv):
            t_hi = hi_by_id[int(t_lo.id)]
            assert int(row["id"]) == int(t_lo.id)
            # native per-source sizes, no padding
            assert (row["ny_hi"], row["nx_hi"]) == t_hi.data.shape
            assert (row["ny_lo"], row["nx_lo"]) == t_lo.data.shape
            np.testing.assert_allclose(
                np.asarray(row["tmpl_lo"], dtype=np.float32).reshape(t_lo.data.shape),
                np.asarray(t_lo.data, dtype=np.float32),
                rtol=1e-6,
            )
            assert np.isclose(row["flux"], t_lo.flux)

    # reader helper reshapes templates and attaches the PSF stamps
    recs = pipeline.Pipeline.read_stamps(path)
    assert len(recs) == len(conv)
    rec = recs[0]
    assert rec["tmpl_hi"].shape == hi_by_id[int(conv[0].id)].data.shape

    # primary header holds only the pointers load_fit needs
    hdr = fits.getheader(path)
    assert (hdr["NX_HI"], hdr["NY_HI"]) == images[0].shape[::-1]
    assert (hdr["NX_LO"], hdr["NY_LO"]) == images[1].shape[::-1]


def test_source_products_and_show_sources(tmp_path):
    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    _table, _residuals, pipe = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=[None, kernel], psfs=psfs
    )

    tmpl = pipe.all_templates[0][0]
    sid = int(tmpl.id)
    p = pipe.source_products(sid)

    # window-aligned products on the fit grid
    assert (
        p["tmpl_lo"].shape == p["img_lo"].shape
        == p["model"].shape == p["residual"].shape
    )
    np.testing.assert_allclose(
        p["img_lo"] - p["model"], p["residual"], rtol=0, atol=1e-5
    )
    # hi-grid products present and window-aligned
    assert p["tmpl_hi"].shape == p["img_hi"].shape == p["segmap"].shape
    # scalars and PSFs come from the fitted template and band inputs
    assert np.isclose(p["flux"], tmpl.flux)
    assert np.isclose(p["err"], tmpl.err)
    np.testing.assert_allclose(p["psf_hi"], psfs[0])
    np.testing.assert_allclose(p["psf_lo"], psfs[1])
    assert int(p["row"]["id"]) == sid

    ids = [int(t.id) for t in pipe.all_templates[0][:2]]
    fig_path = tmp_path / "show_sources.png"
    fig, axes = pipe.show_sources(ids, save=fig_path)
    plt.close(fig)
    assert axes.shape == (2, 8)
    assert fig_path.exists()

    # scalar id works too
    fig, axes = pipe.show_sources(ids[0])
    plt.close(fig)
    assert axes.shape == (1, 8)


def test_load_fit_restores_post_run_state(tmp_path):
    from astropy.io import fits
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])

    run_cfg = pipeline.RunConfig(
        name="t", out_dir=str(tmp_path), sci_hi="hi.fits", segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
    )

    def fresh_pipe():
        pipe = pipeline.Pipeline(
            [im.copy() for im in images],
            segmap,
            catalog=catalog,
            weights=[w.copy() for w in wht],
            kernels=[None, kernel],
            psfs=psfs,
            config=FitConfig(fit_astrometry_niter=0),
        )
        pipe.run_config = run_cfg
        pipe.out_dir = tmp_path
        return pipe

    pipe1 = fresh_pipe()
    pipe1.run()
    # solver errors stay on the templates; predictions live in err_pred only
    for t in pipe1.all_templates[0]:
        row = pipe1.table[pipe1.table["id"] == int(t.id)][0]
        assert np.isclose(t.err, row["err_1"], rtol=1e-8)
        assert np.isclose(t.err_pred, row["err_pred_1"], rtol=1e-8)
    # write the outputs load_fit reads (write_outputs needs real input FITS
    # files for headers/scene plots, so write the three products directly)
    stem = tmp_path / "t"
    fits.writeto(f"{stem}_residual.fits", pipe1.residuals[0], overwrite=True)
    pipe1.table.write(f"{stem}_fit_table.fits", overwrite=True)
    stamps_path = pipe1.write_stamps()

    def assert_matches_run(pipe2):
        np.testing.assert_allclose(pipe2.residuals[0], pipe1.residuals[0], rtol=1e-6)
        np.testing.assert_allclose(
            pipe2.model_images[0], pipe1.model_images[0], rtol=1e-5, atol=1e-5
        )
        assert pipe2.fit_bin_factors == pipe1.fit_bin_factors
        for col in pipe1.table.colnames:
            if pipe1.table[col].dtype.kind in "fc":
                np.testing.assert_allclose(
                    pipe2.table[col], pipe1.table[col], rtol=1e-6, equal_nan=True
                )
            else:
                assert np.all(pipe2.table[col] == pipe1.table[col])
        live, rest = pipe1.all_templates[0], pipe2.all_templates[0]
        assert len(rest) == len(live)
        for a, b in zip(live, rest):
            assert int(b.id) == int(a.id)
            assert b.slices_original == a.slices_original
            assert b.slices_cutout == a.slices_cutout
            assert b._origin_original_true == a._origin_original_true
            assert b.input_position_original == a.input_position_original
            np.testing.assert_allclose(
                b.data, np.asarray(a.data, np.float32), rtol=1e-6, atol=1e-8
            )
            np.testing.assert_allclose(b.flux, a.flux, rtol=1e-6)
            np.testing.assert_allclose(b.err, a.err, rtol=1e-6)
            np.testing.assert_allclose(b.err_pred, a.err_pred, rtol=1e-6)
            np.testing.assert_allclose(b.shifted, a.shifted)
            np.testing.assert_allclose(b.ee_psf_lo, a.ee_psf_lo, equal_nan=True)
        hi_live = {int(t.id): t for t in pipe1.tmpls.templates}
        assert len(pipe2.tmpls.templates) == len(hi_live)
        for b in pipe2.tmpls.templates:
            a = hi_live[int(b.id)]
            assert b.slices_original == a.slices_original
            np.testing.assert_allclose(
                b.data, np.asarray(a.data, np.float32), rtol=1e-6, atol=1e-8
            )

    # --- restore from the stamps file: state matches the live run
    pipe2 = fresh_pipe()
    pipe2.load_fit()
    assert_matches_run(pipe2)
    for a, b in zip(pipe1.all_templates[0], pipe2.all_templates[0]):
        assert b.flag == a.flag
    # restored state drives the visualization helpers directly
    fig, axes = pipe2.show_sources(int(pipe2.all_templates[0][0].id))
    plt.close(fig)
    assert axes.shape == (1, 8)

    # --- delete the stamps file: load_fit regenerates it identically
    with fits.open(stamps_path) as hdul:
        src1 = Table(hdul["SOURCES"].data)
        tmpl1 = {
            tag: [np.asarray(row[f"tmpl_{tag}"], np.float32) for row in hdul["SOURCES"].data]
            for tag in ("hi", "lo")
        }
        hdr1 = hdul[0].header
    stamps_path.unlink()

    pipe3 = fresh_pipe()
    pipe3.load_fit()
    assert_matches_run(pipe3)
    assert stamps_path.exists()

    with fits.open(stamps_path) as hdul:
        src3 = Table(hdul["SOURCES"].data)
        hdr3 = hdul[0].header
        assert len(src3) == len(src1)
        for name in src1.colnames:
            if name.startswith("tmpl_"):
                continue
            if src1[name].dtype.kind in "fc":
                np.testing.assert_allclose(
                    src3[name], src1[name], rtol=1e-6, equal_nan=True, err_msg=name
                )
            else:
                assert np.all(src3[name] == src1[name]), name
        for tag in ("hi", "lo"):
            for row, ref in zip(hdul["SOURCES"].data, tmpl1[tag]):
                np.testing.assert_allclose(
                    np.asarray(row[f"tmpl_{tag}"], np.float32), ref, rtol=1e-6, atol=1e-8
                )
    assert hdr3["RUNNAME"] == hdr1["RUNNAME"] == "t"
    assert (hdr3["NX_HI"], hdr3["NY_HI"], hdr3["NX_LO"], hdr3["NY_LO"]) == (
        hdr1["NX_HI"], hdr1["NY_HI"], hdr1["NX_LO"], hdr1["NY_LO"]
    )


def test_pipeline_accepts_prebuilt_templates(tmp_path):
    images, segmap, catalog, psfs, _truth_img, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    from mophongo.templates import Templates

    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])))
    dirac = np.zeros((3, 3), dtype=float)
    dirac[1, 1] = 1.0
    table, residuals, pipe = pipeline.run(
        [images[0], images[0]],
        segmap,
        catalog=catalog,
        weights=[wht[0], wht[0]],
        kernels=[dirac, dirac],
        templates=tmpls,
        config=pipeline._FitConfig(extend_mode="none"),
    )
    assert len(table) == len(catalog)
    assert len(pipe.tmpls.templates) == len(tmpls.templates)
    assert len(residuals) == 1

    diagnostic_path = tmp_path / "source_stage_diagnostic.png"
    fig, axes = pipe.diagnose_source(int(table["id"][0]), save=diagnostic_path)
    plt.close(fig)
    assert diagnostic_path.exists()
    assert axes.shape[1] == 8


def test_pipeline_propagates_catalog_deblend_flag_to_templates():
    from mophongo.fit import FitConfig

    image = np.zeros((31, 31), dtype=np.float32)
    image[14:17, 14:17] = 1.0
    segmap = np.zeros_like(image, dtype=np.int32)
    segmap[14:17, 14:17] = 1
    catalog = Table()
    catalog["id"] = [1]
    catalog["x"] = [15.0]
    catalog["y"] = [15.0]
    catalog["is_deblended"] = [True]
    catalog["deblend_parent_label"] = [7]
    catalog["deblend_nchildren"] = [2]

    pipe = pipeline.Pipeline(
        [image, image],
        segmap,
        catalog=catalog,
        weights=[np.ones_like(image), np.ones_like(image)],
        psfs=[np.eye(3, dtype=float), np.eye(3, dtype=float)],
        kernels=[None, None],
        extend_templates="psf",
    )
    pipe.run(config=FitConfig(fit_astrometry_niter=0, template_dilate_segmap=0))

    tmpl = pipe.templates_extended.templates[0]
    assert tmpl.is_deblended
    assert tmpl.flag & pipeline.Template.FLAG_DEBLENDED
    assert tmpl.deblend_parent_label == 7
    assert tmpl.deblend_nchildren == 2
    assert tmpl.extension_mode == "psf"
    assert tmpl.extension_skip_reason == ""


def test_pipeline_can_skip_template_extension_for_deblended_sources():
    from mophongo.fit import FitConfig

    image = np.zeros((31, 31), dtype=np.float32)
    image[14:17, 14:17] = 1.0
    segmap = np.zeros_like(image, dtype=np.int32)
    segmap[14:17, 14:17] = 1
    catalog = Table()
    catalog["id"] = [1]
    catalog["x"] = [15.0]
    catalog["y"] = [15.0]
    catalog["is_deblended"] = [True]
    catalog["deblend_parent_label"] = [7]
    catalog["deblend_nchildren"] = [2]

    pipe = pipeline.Pipeline(
        [image, image],
        segmap,
        catalog=catalog,
        weights=[np.ones_like(image), np.ones_like(image)],
        psfs=[np.eye(3, dtype=float), np.eye(3, dtype=float)],
        kernels=[None, None],
        extend_templates="psf",
    )
    pipe.run(
        config=FitConfig(
            fit_astrometry_niter=0,
            template_dilate_segmap=0,
            skip_template_extension_for_deblended=True,
        )
    )

    tmpl = pipe.templates_extended.templates[0]
    assert tmpl.is_deblended
    assert tmpl.extension_mode == "none"
    assert tmpl.extension_skip_reason == "is_deblended"


def test_pipeline_prebuilt_native_templates_recover_scalar_fluxes():
    from mophongo.fit import FitConfig
    from mophongo.templates import Templates

    image = np.zeros((35, 35), dtype=np.float32)
    cutouts = [
        mutils.gaussian((9, 9), fwhm=2.0, flux=0.95).astype(np.float32),
        mutils.gaussian((9, 9), fwhm=2.8, flux=0.72).astype(np.float32),
    ]
    positions = [(10.0, 10.0), (24.0, 23.0)]
    ids = [101, 202]
    flux_true = np.array([2.0, 3.5], dtype=float)
    tmpls = Templates.from_cutout_models(
        cutouts,
        positions,
        ids,
        original_shape=image.shape,
        normalize=False,
    )
    for flux, tmpl in zip(flux_true, tmpls.templates):
        image[tmpl.slices_original] += flux * tmpl.data[tmpl.slices_cutout]

    catalog = Table({"id": ids, "x": [p[0] for p in positions], "y": [p[1] for p in positions]})
    table, residuals, _pipe = pipeline.run(
        [image, image],
        np.zeros_like(image, dtype=np.int32),
        catalog=catalog,
        weights=[np.ones_like(image), np.ones_like(image)],
        kernels=[None, None],
        psf_throughputs=[1.0, 0.95],
        templates=tmpls,
        config=FitConfig(
            reg_flux=0.0,
            fit_astrometry_niter=0,
            fit_astrometry_joint=False,
            aperture_diam=None,
            snr_thresh_astrom=0.0,
        ),
    )

    recovered = np.array([float(table[table["id"] == obj_id]["flux_1"][0]) for obj_id in ids])
    np.testing.assert_allclose(recovered, flux_true, rtol=0, atol=2e-5)
    recovered_total = np.array(
        [float(table[table["id"] == obj_id]["flux_1_total"][0]) for obj_id in ids]
    )
    np.testing.assert_allclose(recovered_total, flux_true / 0.95, rtol=0, atol=2e-5)
    np.testing.assert_allclose(table["throughput_1"], 0.95)
    np.testing.assert_allclose(residuals[0], 0.0, rtol=0, atol=2e-5)


def test_plot_result_uses_run_scenes(tmp_path):
    """plot_result must work off a completed run.

    It previously reached for ``self.fit``, which no Pipeline ever sets, and
    guarded its scene map on ``hasattr(self, "scenes")`` — permanently true
    since ``scenes`` became a property — so the map was never built and the
    lookup below it raised NameError. Nothing called it, so nothing caught this.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pytest
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
        config=FitConfig(fit_astrometry_niter=0),
    )
    pipe.run_config = pipeline.RunConfig(
        name="t", out_dir=str(tmp_path), sci_hi="hi.fits", segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
    )
    pipe.out_dir = tmp_path
    pipe.run()

    fig, ax = pipe.plot_result()
    plt.close(fig)
    assert ax.size >= 6

    # zooming to a scene exercises the scene map and the id labels
    scene_id = pipe.all_scenes[0][0].id
    fig, ax = pipe.plot_result(scene_id=scene_id)
    plt.close(fig)

    # without a run there are no scenes to plot, and it must say so
    bare = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
    )
    with pytest.raises(RuntimeError, match="completed run"):
        bare.plot_result()
