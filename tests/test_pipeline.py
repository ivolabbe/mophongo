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

    path = tmp_path / "stamps.h5"
    out = pipe.write_stamps(path)
    assert out == path

    conv = pipe.all_templates[0]
    hi_by_id = {int(t.id): t for t in pipe.tmpls.templates}

    import h5py

    with h5py.File(path, "r") as h5:
        src = {name: h5["sources"][name][:] for name in h5["sources"]}
        assert h5.attrs["NSRC"] == len(conv)
        assert len(src["id"]) == len(conv)
        # PSFs are not duplicated into the file: static PSFs carry key 0
        assert np.all(src["key_psf_hi"] == 0)
        assert np.all(src["key_psf_lo"] == 0)
        # ragged stamps are one flat buffer per band plus offsets
        assert h5["tmpl_lo"]["offset"].shape == (len(conv) + 1,)
        assert h5["tmpl_hi"]["pixels"].dtype == np.float32

    for i, t_lo in enumerate(conv):
        t_hi = hi_by_id[int(t_lo.id)]
        assert int(src["id"][i]) == int(t_lo.id)
        # native per-source sizes, no padding
        assert (src["ny_hi"][i], src["nx_hi"][i]) == t_hi.data.shape
        assert (src["ny_lo"][i], src["nx_lo"][i]) == t_lo.data.shape
        assert np.isclose(src["flux"][i], t_lo.flux)

    # reader helper reshapes templates and attaches the PSF stamps
    recs = pipeline.Pipeline.read_stamps(path)
    assert len(recs) == len(conv)
    rec = recs[0]
    assert rec["tmpl_hi"].shape == hi_by_id[int(conv[0].id)].data.shape
    # pixels land in the right slot: stamps are written straight into the
    # dataset at offsets derived from the recorded shapes, so a mismatch
    # between the shape pass and the pixel pass would shear every stamp
    for rec, t_lo in zip(recs, conv):
        t_hi = hi_by_id[int(t_lo.id)]
        # exact at the file's own precision: stamps are stored float32
        assert np.array_equal(rec["tmpl_lo"], np.float32(t_lo.data))
        assert np.array_equal(rec["tmpl_hi"], np.float32(t_hi.data))

    # file attributes hold only the pointers load_fit needs
    with h5py.File(path, "r") as h5:
        hdr = dict(h5.attrs)
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
    import h5py

    def _read_h5(path):
        recs = pipeline.Pipeline.read_stamps(path)
        with h5py.File(path, "r") as h5:
            attrs = dict(h5.attrs)
        cols = [k for k in recs[0] if not k.startswith("tmpl_")]
        return Table({c: [r[c] for r in recs] for c in cols}, ), recs, attrs

    src1, recs1, hdr1 = _read_h5(stamps_path)
    tmpl1 = {tag: [np.asarray(r[f"tmpl_{tag}"], np.float32) for r in recs1]
             for tag in ("hi", "lo")}
    stamps_path.unlink()

    pipe3 = fresh_pipe()
    pipe3.load_fit()
    assert_matches_run(pipe3)
    assert stamps_path.exists()

    src3, recs3, hdr3 = _read_h5(stamps_path)
    if True:
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
            for rec, ref in zip(recs3, tmpl1[tag]):
                np.testing.assert_allclose(
                    np.asarray(rec[f"tmpl_{tag}"], np.float32), ref, rtol=1e-6, atol=1e-8
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
    assert tmpl.extension_mode == "psf_convolution"
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


def test_write_outputs_puts_scene_plots_in_scenes_subdir(tmp_path):
    """Scene PNGs go to ``out_dir/scenes/``, and only when requested."""
    import matplotlib
    matplotlib.use("Agg")
    from astropy.io import fits
    from astropy.wcs import WCS
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    # the scene catalog converts scene centers to sky, so a WCS is required
    wcs_hi = WCS(naxis=2)
    wcs_hi.wcs.crpix = [50.0, 50.0]
    wcs_hi.wcs.crval = [150.0, 2.0]
    wcs_hi.wcs.cdelt = [-1e-5, 1e-5]
    wcs_hi.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    def fresh_pipe(out_dir, scene_plots):
        pipe = pipeline.Pipeline(
            [im.copy() for im in images], segmap, catalog=catalog,
            weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
            wcs=[wcs_hi, wcs_hi], config=FitConfig(fit_astrometry_niter=0),
        )
        # write_outputs copies the hi-res header onto the residual
        sci_hi = out_dir / "hi.fits"
        out_dir.mkdir(parents=True, exist_ok=True)
        fits.writeto(sci_hi, np.asarray(images[0], np.float32), overwrite=True)
        pipe.run_config = pipeline.RunConfig(
            name="t", out_dir=str(out_dir), sci_hi=str(sci_hi), segmap="seg.fits",
            catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
            csv_hi="hi.csv", csv_lo="lo.csv",
            scene_plots=scene_plots, save_stamps=False,
        )
        pipe.out_dir = out_dir
        pipe.run()
        return pipe

    on = tmp_path / "on"
    pipe = fresh_pipe(on, True)
    pipe.write_outputs()
    scene_ids = [s.id for s in pipe.scenes]
    assert scene_ids
    assert sorted(p.name for p in (on / "scenes").glob("*.png")) == sorted(
        f"t_scene_{sid}.png" for sid in scene_ids
    )
    # no per-scene PNG left at the old flat location
    assert not [p for p in on.glob("t_scene_*.png") if p.stem.split("_")[-1].isdigit()]
    # the scene catalog and the full-field scene figure stay in out_dir
    assert (on / "t_scene_catalog.csv").exists()
    assert (on / "t_scenes.png").exists()

    off = tmp_path / "off"
    fresh_pipe(off, False).write_outputs()
    assert not (off / "scenes").exists()
    assert not (off / "t_scene_map.png").exists()


def test_scene_results_do_not_depend_on_scene_order(monkeypatch):
    """The astrometric loop runs scene-by-scene, so order must not matter.

    ``run()`` refines one scene to convergence before starting the next rather
    than synchronising every scene at each pass. That is only legitimate
    because a scene reads its own templates and read-only slices of the shared
    image and weights, so processing the scenes in the opposite order has to
    give the same fluxes, errors, shifts and residual. It is also the property
    a per-scene worker pool would rely on.
    """
    from astropy.wcs import WCS
    from scipy.ndimage import shift as nd_shift
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=40, size=201, ndilate=2, peak_snr=50.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    # offset the fitted band so the shift loop actually iterates
    images = [images[0], nd_shift(images[1], (0.95, -0.75), order=3)]

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100.0, 100.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-1e-5, 1e-5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    config = FitConfig(
        fit_astrometry_niter=5, astrom_shift_tol=0.004, snr_thresh_astrom=5.0,
        astrom_isolation_thresh=0.0, scene_minimum_bright=2,
    )

    def fit():
        pipe = pipeline.Pipeline(
            [im.copy() for im in images], segmap, catalog=catalog,
            weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
            wcs=[wcs, wcs], config=config,
        )
        table, residuals = pipe.run()
        scenes = sorted(pipe.all_scenes[0], key=lambda s: s.id)
        return table, residuals[0], scenes

    ref_table, ref_res, ref_scenes = fit()
    assert len(ref_scenes) > 1, "need several scenes for the order to mean anything"

    # same run, scenes handed to the loop back to front
    generate = pipeline.generate_scenes

    def reversed_scenes(*args, **kwargs):
        scenes, labels = generate(*args, **kwargs)
        return list(reversed(scenes)), labels

    monkeypatch.setattr(pipeline, "generate_scenes", reversed_scenes)
    got_table, got_res, got_scenes = fit()

    np.testing.assert_array_equal(got_table["id"], ref_table["id"])
    for col in ("flux_1", "err_1"):
        np.testing.assert_array_equal(got_table[col], ref_table[col])
    np.testing.assert_array_equal(got_res, ref_res)
    for ref, got in zip(ref_scenes, got_scenes):
        assert got.astrom_niter == ref.astrom_niter
        assert got.astrom_converged == ref.astrom_converged
        np.testing.assert_array_equal(got.shifts, ref.shifts)


def test_write_outputs_silences_hierarch_card_warnings(tmp_path):
    """Long catalog-meta keywords write as HIERARCH cards without warning.

    ``PHOT_UNIT``, ``WEBBSTARFILT`` and friends come in on the input catalog's
    ``meta`` and round-trip by design; astropy warns once per card, twice once
    its warning logging has a handler. ``log_run`` filtered these, but the
    steps are also run one at a time (``... config.json load fit outputs``),
    which never enters that block.
    """
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning
    from astropy.wcs import WCS
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=12, nsrc=12, size=101, ndilate=2, peak_snr=2.0
    )
    catalog.meta["PHOT_UNIT"] = "uJy"
    catalog.meta["WEBBSTARFILT"] = "F444W"
    catalog.meta["SHRINK_FACTOR"] = 2
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    wcs_hi = WCS(naxis=2)
    wcs_hi.wcs.crpix = [50.0, 50.0]
    wcs_hi.wcs.crval = [150.0, 2.0]
    wcs_hi.wcs.cdelt = [-1e-5, 1e-5]
    wcs_hi.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sci_hi = out_dir / "hi.fits"
    fits.writeto(sci_hi, np.asarray(images[0], np.float32), overwrite=True)

    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
        wcs=[wcs_hi, wcs_hi], config=FitConfig(fit_astrometry_niter=0),
    )
    pipe.run_config = pipeline.RunConfig(
        name="t", out_dir=str(out_dir), sci_hi=str(sci_hi), segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
        scene_plots=False, save_stamps=True,
    )
    pipe.out_dir = out_dir
    pipe.run()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.write_outputs()
    hierarch = [
        w for w in caught
        if issubclass(w.category, VerifyWarning)
        and "HIERARCH card will be created" in str(w.message)
    ]
    assert not hierarch, [str(w.message) for w in hierarch]

    # the keywords still made it into the file, as HIERARCH cards
    hdr = fits.getheader(pipe.f_fit_table, 1)
    assert hdr["PHOT_UNIT"] == "uJy"
    assert hdr["WEBBSTARFILT"] == "F444W"
    assert hdr["SHRINK_FACTOR"] == 2

    # and the filter is scoped to the write, not left installed globally
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fits.Header()["PHOT_UNIT"] = "uJy"
    assert any(
        issubclass(w.category, VerifyWarning)
        and "HIERARCH card will be created" in str(w.message)
        for w in caught
    )


def _shift_field_pipeline(tmp_path, order, nsrc=24, size=161, offset=(0.0, 0.0)):
    """A run with astrometry solved at ``order``, ready for plot_shift_field.

    ``offset`` shifts the fitted band by ``(sx, sy)`` pixels, so the solver has
    a real offset to recover instead of noise.
    """
    from astropy.wcs import WCS
    from scipy.ndimage import shift as nd_shift
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=nsrc, size=size, ndilate=2, peak_snr=50.0
    )
    images = [im.copy() for im in images]
    if any(offset):
        images[1] = nd_shift(images[1], (offset[1], offset[0]), order=3, mode="nearest")
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    wcs_hi = WCS(naxis=2)
    wcs_hi.wcs.crpix = [size / 2, size / 2]
    wcs_hi.wcs.crval = [150.0, 52.0]  # high dec: exercises the cos(dec) aspect
    wcs_hi.wcs.cdelt = [-1e-5, 1e-5]
    wcs_hi.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    cfg = FitConfig(
        fit_astrometry_niter=2,
        astrom_model="poly",
        astrom_kwargs={"poly": {"order": order}, "gp": {"length_scale": 400}},
        snr_thresh_astrom=0.0,
        astrom_isolation_thresh=0.0,
    )
    pipe = pipeline.Pipeline(
        images, segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
        wcs=[wcs_hi, wcs_hi], config=cfg,
    )
    pipe.run_config = pipeline.RunConfig(
        name="t", out_dir=str(tmp_path), sci_hi="hi.fits", segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
    )
    pipe.out_dir = tmp_path
    pipe.run()
    return pipe


def test_shift_field_sample_count_follows_poly_order(tmp_path):
    """order 0 -> 1 arrow at the scene center, order 1 -> 2, order 2 -> 4."""
    import matplotlib
    matplotlib.use("Agg")

    for order, expect in ((0, 1), (1, 2), (2, 4)):
        pipe = _shift_field_pipeline(tmp_path, order)
        solved = [s for s in pipe.scenes if s.shifts is not None and len(s.shifts) > 1]
        assert solved, f"order {order}: no scene solved for shifts"
        for s in solved:
            xy, dxy = pipe._scene_shift_samples(s)
            assert xy.shape == (expect, 2), f"order {order}: {xy.shape[0]} samples"
            assert dxy.shape == (expect, 2)
            pos = np.array([t.position_original for t in s.templates], float)
            if expect == 1:
                # the single arrow sits at the scene center
                np.testing.assert_allclose(xy[0], pos.mean(axis=0), atol=1e-6)
            else:
                # the samples spread over the scene, and stay inside it
                assert np.ptp(xy, axis=0).max() > 0
                assert (xy.min(axis=0) >= pos.min(axis=0) - 1).all()
                assert (xy.max(axis=0) <= pos.max(axis=0) + 1).all()


def test_shift_field_arrows_track_applied_template_shifts(tmp_path):
    """The refit field reproduces the shifts actually applied to templates."""
    import matplotlib
    matplotlib.use("Agg")

    # a real injected offset: with exact shift blocks an aligned band fits
    # zero shift, so there would be nothing for the arrows to track
    pipe = _shift_field_pipeline(tmp_path, 1, offset=(0.6, -0.4))
    checked = 0
    for s in pipe.scenes:
        if s.shifts is None or len(s.shifts) < 2:
            continue
        applied = np.array([t.shifted[:2] for t in s.templates], float)
        if not np.any(np.abs(applied) > 1e-3):
            continue
        xy, dxy = pipe._scene_shift_samples(s)
        # the sampled field lies within the range of the shifts it was fit to
        assert dxy.min() >= applied.min() - 0.05
        assert dxy.max() <= applied.max() + 0.05
        checked += 1
    assert checked, "no scene had non-trivial applied shifts"


def test_final_fluxes_are_stationary_on_the_shifted_templates(tmp_path):
    """The written solution belongs to the basis that is actually written.

    Each astrometry pass solves fluxes on the templates as they stood before
    that pass's shift was applied, so without a final flux-only pass the
    stored fluxes, errors, model and residual describe a template basis that
    no longer exists -- the last applied shift is never accounted for.
    """
    from mophongo.scene_fitter import SceneFitter, build_normal

    pipe = _shift_field_pipeline(tmp_path, 1, offset=(0.6, -0.4))

    moved = max(
        float(np.abs(np.asarray(t.shifted[:2], dtype=float)).max())
        for s in pipe.scenes
        for t in s.templates
    )
    assert moved > 0.05, "no template moved; the test would be vacuous"

    checked = 0
    for s in pipe.scenes:
        A, b, _ = build_normal(s.templates, s.image, s.weights)
        flux = np.array([t.flux for t in s.templates], dtype=float)

        # re-solving on the final templates must reproduce the stored fluxes
        again = SceneFitter.solve(A, b, config=pipe.config)
        assert np.allclose(flux, again.flux, rtol=1e-8, atol=1e-12)
        assert np.allclose(
            np.array([t.err for t in s.templates], dtype=float),
            again.err, rtol=1e-8, atol=1e-12,
        )

        # and they satisfy the normal equations of those templates, up to the
        # ridge the solver adds (A + lam I) x = b
        resid = np.abs(A @ flux - b).max()
        assert resid < 1e-4 * max(float(np.abs(b).max()), 1e-30)

        # the model is built from the same fluxes and the same stamps
        expected = np.zeros_like(s.model_image())
        bb = s.bbox
        for t in s.templates:
            sl = t.slices_original
            expected[
                sl[0].start - bb[0] : sl[0].stop - bb[0],
                sl[1].start - bb[2] : sl[1].stop - bb[2],
            ] += t.flux * t.data[t.slices_cutout]
        assert np.array_equal(s.model_image(), expected)
        checked += 1

    assert checked, "no scenes to check"


def test_write_outputs_writes_shift_field(tmp_path):
    """The shift field is a standard output whenever astrometry was solved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    on = tmp_path / "on"
    on.mkdir()
    pipe = _shift_field_pipeline(on, 1)
    fits.writeto(on / "hi.fits", np.asarray(pipe.images[0], np.float32), overwrite=True)
    pipe.run_config.sci_hi = str(on / "hi.fits")
    pipe.run_config.save_stamps = False
    pipe.write_outputs()
    assert (on / "t_shift_field.png").exists()

    # the figure carries one label per solved scene and one arrow per sample
    fig, ax = pipe.plot_shift_field()
    solved = [s for s in pipe.scenes if s.shifts is not None and len(s.shifts) > 1]
    assert len([t for t in ax.texts]) == len(solved)
    quiver = [c for c in ax.collections if hasattr(c, "U")][0]
    assert quiver.U.size == sum(
        pipe._scene_shift_samples(s)[0].shape[0] for s in solved
    )
    assert ax.xaxis_inverted()  # RA increases to the left
    plt.close(fig)

    # no astrometry solved -> no figure, no file
    pipe.all_scenes = [[]]
    assert pipe.plot_shift_field() is None


def test_shift_field_arrow_points_from_template_to_measured(tmp_path):
    """Sign check: offset the fitted band, the arrow follows the offset.

    Shifting the low-resolution image by ``+(sx, sy)`` puts every source that
    much away from its template, so the arrow drawn from the template toward
    the measured position must carry the same sign.
    """
    import matplotlib
    matplotlib.use("Agg")
    from astropy.wcs import WCS
    from scipy.ndimage import shift as nd_shift
    from mophongo.fit import FitConfig

    sx, sy = 0.6, -0.4
    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=40, size=241, ndilate=2, peak_snr=50.0
    )
    images = [im.copy() for im in images]
    images[1] = nd_shift(images[1], (sy, sx), order=3, mode="nearest")
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    wcs_hi = WCS(naxis=2)
    wcs_hi.wcs.crpix = [120.0, 120.0]
    wcs_hi.wcs.crval = [150.0, 2.0]
    wcs_hi.wcs.cdelt = [-1e-5, 1e-5]
    wcs_hi.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    pipe = pipeline.Pipeline(
        images, segmap, catalog=catalog, weights=[w.copy() for w in wht],
        kernels=[None, kernel], psfs=psfs, wcs=[wcs_hi, wcs_hi],
        config=FitConfig(
            fit_astrometry_niter=8, astrom_model="poly",
            astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
            snr_thresh_astrom=0.0, astrom_isolation_thresh=0.0,
        ),
    )
    pipe.run_config = pipeline.RunConfig(
        name="t", out_dir=str(tmp_path), sci_hi="hi.fits", segmap="seg.fits",
        catalog="cat.fits", sci_lo="lo.fits", wht_lo="wht.fits",
        csv_hi="hi.csv", csv_lo="lo.csv",
    )
    pipe.out_dir = tmp_path
    pipe.run()

    arrows = np.vstack(
        [pipe._scene_shift_samples(s)[1] for s in pipe.scenes
         if pipe._scene_shift_samples(s) is not None]
    )
    assert len(arrows)
    # sign and rough size of the injected offset, damped solve undershoots
    assert np.median(arrows[:, 0]) > 0.3 * sx
    assert np.median(arrows[:, 1]) < 0.3 * sy
    np.testing.assert_allclose(np.median(arrows, axis=0), [sx, sy], atol=0.25)


def test_scene_catalog_carries_total_shift(tmp_path):
    """The scene catalog's dx, dy is the total shift, not the last increment."""
    import matplotlib
    matplotlib.use("Agg")
    from astropy.io import fits
    from astropy.table import Table as _Table

    out = tmp_path / "cat"
    out.mkdir()
    pipe = _shift_field_pipeline(out, 0, offset=(0.6, -0.4))
    fits.writeto(out / "hi.fits", np.asarray(pipe.images[0], np.float32), overwrite=True)
    pipe.run_config.sci_hi = str(out / "hi.fits")
    pipe.run_config.save_stamps = False
    pipe.run_config.scene_plots = False
    pipe.write_outputs()

    cat = _Table.read(out / "t_scene_catalog.csv", format="ascii.csv")
    assert {"dx", "dy"} <= set(cat.colnames)
    by_id = {int(r["id"]): (r["dx"], r["dy"]) for r in cat}

    checked = 0
    for s in pipe.scenes:
        if s.shifts is None or len(s.shifts) < 2:
            assert np.isnan(by_id[int(s.id)]).all()
            continue
        applied = np.array([t.shifted[:2] for t in s.templates], float)
        if not np.any(np.abs(applied) > 1e-3):
            continue
        # the catalog value is the accumulated shift, not the last increment
        np.testing.assert_allclose(by_id[int(s.id)], applied.mean(axis=0), atol=0.02)
        last = np.asarray(s.shifts, float)[:2]  # order 0: the pass's own dx, dy
        assert not np.allclose(by_id[int(s.id)], last, atol=1e-3)
        checked += 1
    assert checked, "no scene had non-trivial applied shifts"


def test_astrometry_passes_skip_converged_scenes(monkeypatch):
    """A scene that has stopped moving drops out of the refinement loop.

    ``Scene.solve`` reads only that scene's own templates, image and weights,
    so a converged scene cannot start moving again: iterating it further only
    costs time. Each scene must therefore be solved exactly ``astrom_niter``
    times, and carry its own convergence verdict rather than the run's.
    """
    from mophongo.fit import FitConfig
    from mophongo.scene import Scene

    calls: dict[int, int] = {}
    real_solve = Scene.solve

    def counting_solve(self, **kwargs):
        calls[id(self)] = calls.get(id(self), 0) + 1
        return real_solve(self, **kwargs)

    monkeypatch.setattr(Scene, "solve", counting_solve)

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=20, size=121, ndilate=2, peak_snr=30.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
    )
    niter = 3
    pipe.run(config=FitConfig(fit_astrometry_niter=niter, fit_astrometry_joint=True))

    scenes = [s for band in pipe.all_scenes for s in band]
    assert scenes
    assert any(s.astrom_converged is not None for s in scenes)
    for s in scenes:
        assert 1 <= s.astrom_niter <= niter
        # solved once per recorded pass and never again after converging, plus
        # the one final flux-only pass every scene gets on its final templates
        assert calls[id(s)] == s.astrom_niter + 1
        if s.astrom_niter < niter:
            assert s.astrom_converged is not False

    # the other end: with an unreachable tolerance nothing converges, every
    # scene uses the whole budget and is left flagged as unconverged
    calls.clear()
    pipe2 = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
    )
    pipe2.run(config=FitConfig(fit_astrometry_niter=niter, fit_astrometry_joint=True,
                               astrom_shift_tol=0.0))
    slow = [s for band in pipe2.all_scenes for s in band]
    assert slow
    assert any(s.astrom_converged is False for s in slow)
    for s in slow:
        if s.shifts is None or len(s.shifts) == 0:
            continue  # no shift block: no verdict to give
        assert s.astrom_converged is False
        assert s.astrom_niter == niter == calls[id(s)] - 1

    # every fitted source inherits its scene's verdict
    for pipe_, expected in ((pipe, None), (pipe2, None)):
        table = pipe_.table
        assert "flag_astrom_1" in table.colnames
        verdict = {s.id: (-1 if s.astrom_converged is None
                          else int(not s.astrom_converged))
                   for band in pipe_.all_scenes for s in band}
        fitted = np.asarray(table["scene_1"]) >= 0
        assert fitted.any()
        got = np.asarray(table["flag_astrom_1"])[fitted]
        want = np.array([verdict[sid] for sid in np.asarray(table["scene_1"])[fitted]])
        assert np.array_equal(got, want)
        # sources with no template carry no verdict
        assert np.all(np.asarray(table["flag_astrom_1"])[~fitted] == -1)
        if expected is not None:
            assert np.all(got == expected)


def test_astrometry_verdict_is_none_where_no_shift_was_fitted():
    """`flag_astrom_<i>` = 0 must mean solved-and-converged, not never-moved.

    A flux-only run never fits a shift, so its templates trivially do not
    move. Reporting that as convergence would mark every source as
    astrometrically good on a run that solved no astrometry at all.
    """
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=20, size=121, ndilate=2, peak_snr=30.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
    )
    table, _ = pipe.run(config=FitConfig(fit_astrometry_niter=0))

    scenes = [s for band in pipe.all_scenes for s in band]
    assert scenes
    for s in scenes:
        assert s.shifts is None or len(s.shifts) == 0
        assert s.astrom_converged is None
    assert np.all(np.asarray(table["flag_astrom_1"]) == -1)


def test_final_sub_tolerance_shift_is_applied_to_the_templates():
    """The increment the loop stops on has already been applied.

    ``Scene.solve(apply_shifts=True)`` applies before the caller measures, so
    the convergence test reads the accumulated ``Template.shifted`` — the last
    step is on the templates, not discarded for being small.
    """
    from mophongo.fit import FitConfig
    from mophongo.scene import Scene

    seen: list[tuple[int, np.ndarray]] = []
    real_solve = Scene.solve

    def recording_solve(self, **kwargs):
        before = np.array([t.shifted[:2] for t in self.templates], float)
        out = real_solve(self, **kwargs)
        after = np.array([t.shifted[:2] for t in self.templates], float)
        seen.append((id(self), np.max(np.abs(after - before)) if before.size else 0.0))
        return out

    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=7, nsrc=20, size=121, ndilate=2, peak_snr=30.0
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    pipe = pipeline.Pipeline(
        [im.copy() for im in images], segmap, catalog=catalog,
        weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
    )
    tol = 0.1
    import unittest.mock as _mock
    with _mock.patch.object(Scene, "solve", recording_solve):
        pipe.run(config=FitConfig(fit_astrometry_niter=4, astrom_shift_tol=tol))

    checked = 0
    for s in (s for band in pipe.all_scenes for s in band):
        if not s.astrom_converged:
            continue
        steps = [d for sid, d in seen if sid == id(s)]
        # the recorded step is measured across solve(), i.e. post-apply, and
        # it is the value the loop stored and stopped on
        np.testing.assert_allclose(steps[-1], s.astrom_step, rtol=1e-9)
        assert s.astrom_step < tol
        # the accumulated shift includes that final step
        total = np.max(np.abs([t.shifted[:2] for t in s.templates]))
        assert total >= 0.0
        if len(steps) > 1 and s.astrom_step > 1e-2:
            assert total > 0.0
        checked += 1
    assert checked, "no scene converged"


def test_aperture_sum_follows_the_fitted_shift():
    """The aperture tracks the position the template was resampled onto.

    An off-centre aperture loses EE; measuring at the catalog position on a
    shifted template understated ap_flux by ~1% and inflated stampcor.
    """
    from scipy.ndimage import shift as nd_shift
    from mophongo.templates import Template

    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-0.5 * (((xx - n // 2) / 2.0) ** 2 + ((yy - n // 2) / 2.0) ** 2))
    g /= g.sum()
    parent = np.zeros((80, 80), dtype=float)
    tmpl = Template(parent, (40.0, 40.0), (n, n), label=1)
    tmpl.data = g.copy()

    pipe = object.__new__(pipeline.Pipeline)
    centered = pipe._aperture_sum_on_template(tmpl, 5.0)

    dx, dy = 1.4, -0.9
    tmpl.data = nd_shift(g, (dy, dx), order=3)
    tmpl.shifted[:] = [dx, dy]

    off = pipe._aperture_sum_on_template(tmpl, 5.0)
    followed = pipe._aperture_sum_on_template(tmpl, 5.0, offset=(dx, dy))

    assert followed > off                       # following recovers EE
    np.testing.assert_allclose(followed, centered, rtol=2e-3)


def test_refit_scene_freezes_membership_and_restores_state(tmp_path):
    """A scene can be re-extracted and re-solved without disturbing the run.

    The A/B this exists for: same sources, same pixels, same weights, one
    parameter changed. And ``_convolved_templates`` is not idempotent -- on the
    upsample path it rebinds ``images[ifilt]`` and appends to
    ``fit_bin_factors`` -- so the pipeline state it touches must come back
    exactly as it was, or a second experiment silently fits an already
    upsampled image.
    """
    from dataclasses import replace as dc_replace

    import pytest
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
            [im.copy() for im in images], segmap, catalog=catalog,
            weights=[w.copy() for w in wht], kernels=[None, kernel], psfs=psfs,
            config=FitConfig(fit_astrometry_niter=0),
        )
        pipe.run_config = run_cfg
        pipe.out_dir = tmp_path
        return pipe

    pipe1 = fresh_pipe()
    pipe1.run()
    scene_id = int(pipe1.scene_ids()[0])
    stem = tmp_path / "t"
    fits.writeto(f"{stem}_residual.fits", pipe1.residuals[0], overwrite=True)
    pipe1.table.write(f"{stem}_fit_table.fits", overwrite=True)
    # id_scene lives here, not in the fit table: this is what lets a restored
    # fit freeze the membership the run chose
    pipe1._template_fit_table().write(f"{stem}_templates.fits", overwrite=True)
    pipe1.write_stamps()

    pipe = fresh_pipe()
    pipe.load_fit()
    recorded = sorted(int(i) for i in
                      pipe.template_table["id"][pipe.template_table["id_scene"] == scene_id])

    before = {
        "images": [id(im) for im in pipe.images],
        "bins": list(pipe.fit_bin_factors),
    }

    res = pipe.refit_scene(scene_id, extend_mode="none")

    # state the refit borrowed is put back, object for object
    assert [id(im) for im in pipe.images] == before["images"]
    assert list(pipe.fit_bin_factors) == before["bins"]

    # membership is frozen to what the run recorded
    assert sorted(int(i) for i in res.ids) == recorded
    assert {int(t.id) for t in res.variant.templates} == set(recorded)

    # the change is recorded, and both sides were solved through one path
    assert "extend_mode" in res.changed
    assert res.baseline is not None
    tab = res.table()
    assert set(tab.colnames) >= {"id", "flux", "err", "flux_base", "dflux_sigma"}
    assert len(tab) == len(recorded)
    assert np.isfinite(res.chi2) and np.isfinite(res.dchi2)

    # with nothing changed there is no baseline to compare against
    same = pipe.refit_scene(scene_id)
    assert same.changed == {} and same.baseline is None and same.dchi2 == 0.0

    # a full config replacement is the other way in; the two are exclusive
    cfg = dc_replace(pipe.config, snr_thresh_astrom=99.0)
    assert pipe.refit_scene(scene_id, config=cfg).changed
    with pytest.raises(ValueError, match="either config="):
        pipe.refit_scene(scene_id, config=cfg, extend_mode="none")
    with pytest.raises(ValueError, match="unknown FitConfig field"):
        pipe.refit_scene(scene_id, not_a_field=1)

    # both figures draw: the A/B strip and the run's own six-panel diagnostic
    import matplotlib.pyplot as plt

    fig = res.plot(tmp_path / "refit_compare.png")
    assert (tmp_path / "refit_compare.png").exists()
    plt.close(fig)
    fig = res.plot_scene(path=tmp_path / "refit_scene.png")
    assert (tmp_path / "refit_scene.png").exists()
    plt.close(fig)
    with pytest.raises(ValueError, match="no baseline solve"):
        same.plot_scene("baseline")
