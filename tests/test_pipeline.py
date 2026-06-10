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
        images, segmap, catalog=catalog, weights=wht, kernels=kernel
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
        extend_templates="psf_wings",
    )
    pipe.run(config=FitConfig(fit_astrometry_niter=0, template_dilate_segmap=0))

    tmpl = pipe.templates_extended.templates[0]
    assert tmpl.is_deblended
    assert tmpl.flag & pipeline.Template.FLAG_DEBLENDED
    assert tmpl.deblend_parent_label == 7
    assert tmpl.deblend_nchildren == 2
    assert tmpl.extension_mode == "psf_wings"
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
        extend_templates="psf_wings",
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
