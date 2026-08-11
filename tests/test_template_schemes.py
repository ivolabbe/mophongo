"""Tests for the alternative template build schemes ('wren', 'classic').

The reference behaviour asserted here is taken directly from the two source
codes (``wren/dev-wren:templates.py::_extended_composite`` and IDL
``subphot.pro::build_cube``), as documented in
``scratch/wren/template_comparison.tex``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mophongo import template_schemes as schemes
from mophongo.templates import Template, Templates


# ---------------------------------------------------------------- fixtures


def _gaussian(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return np.exp(-0.5 * r2 / sigma ** 2)


def _psf(size: int = 41, sigma: float = 2.0) -> np.ndarray:
    c = (size - 1) / 2.0
    psf = _gaussian((size, size), (c, c), sigma)
    return psf / psf.sum()


def _scene(
    shape: tuple[int, int] = (81, 81),
    positions=((40.0, 40.0),),
    amps=(100.0,),
    sigma: float = 2.0,
    seg_radius: float = 3.0,
):
    """Point sources convolved to ``sigma`` plus a tight segmentation map."""
    image = np.zeros(shape, dtype=float)
    segmap = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for i, ((x, y), amp) in enumerate(zip(positions, amps), start=1):
        image += amp * _gaussian(shape, (x, y), sigma)
        segmap[np.hypot(xx - x, yy - y) <= seg_radius] = i
    return image, segmap


# ---------------------------------------------------------------- helpers


def test_robust_sigma_matches_gaussian_scale():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 2.5, 20000)
    assert schemes.robust_sigma(y) == pytest.approx(2.5, rel=0.05)


def test_robust_sigma_ignores_bright_outliers():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 20000)
    y[:200] = 500.0  # a "source" in the tile
    assert schemes.robust_sigma(y) == pytest.approx(1.0, rel=0.1)


def test_psf_ee_radius_is_stamp_relative_and_monotone():
    psf = _psf(61, sigma=3.0)
    r50 = schemes.psf_ee_radius_pix(psf, 0.5)
    r95 = schemes.psf_ee_radius_pix(psf, 0.95)
    assert r50 < r95
    # A 2-D Gaussian encloses 50% within 1.177 sigma and 95% within 2.448 sigma.
    assert r50 == pytest.approx(1.177 * 3.0, rel=0.15)
    assert r95 == pytest.approx(2.448 * 3.0, rel=0.15)


def test_sample_psf_on_stamp_places_the_peak_at_the_requested_centre():
    psf = _psf(41, sigma=2.0)
    stamp = schemes.sample_psf_on_stamp(psf, (31, 31), (12.0, 20.0), order=1)
    assert np.unravel_index(np.argmax(stamp), stamp.shape) == (20, 12)
    # unit-sum before interpolation, no renormalisation afterwards
    assert stamp.sum() <= 1.0 + 1e-12


def test_build_ownership_is_disjoint_and_area_weighted():
    segmap = np.zeros((41, 61), dtype=np.int32)
    yy, xx = np.mgrid[0:41, 0:61]
    segmap[np.hypot(xx - 15, yy - 20) <= 6] = 1  # big
    segmap[np.hypot(xx - 45, yy - 20) <= 2] = 2  # small
    owner = schemes.build_ownership(segmap, radius=10.0)

    # segment pixels always keep themselves
    assert np.all(owner[segmap == 1] == 1)
    assert np.all(owner[segmap == 2] == 2)
    # the midpoint between the two goes to the larger segment
    assert owner[20, 30] == 1


def test_blend_weight_saturates_at_the_onset():
    assert schemes.blend_weight(30.0, 15.0, 2.0) == 1.0
    assert schemes.blend_weight(15.0, 15.0, 2.0) == 1.0
    assert schemes.blend_weight(5.0, 15.0, 2.0) == pytest.approx((5.0 / 15.0) ** 2)
    assert schemes.blend_weight(-3.0, 15.0, 2.0) == 0.0
    assert schemes.blend_weight(np.nan, 15.0, 2.0) == 0.0


# ---------------------------------------------------------------- classic


def test_classic_recovers_the_psf_amplitude_by_least_squares():
    """A pure point source: f_psf must be the source amplitude and the
    composite must equal the PSF model everywhere outside the segment."""
    psf = _psf(41, sigma=2.0)
    shape = (61, 61)
    cen = (30.0, 30.0)
    amp = 250.0
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, cen, order=3)
    data = amp * psf_stamp
    seg = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    seg[np.hypot(xx - cen[0], yy - cen[1]) <= 3] = 1

    comp, info = schemes.composite_classic(
        data, seg, 1, psf_stamp,
        params=schemes.ClassicParams(tmpl_snrlo=0.0), tmpl_rms=1.0,
    )
    assert info["fpsf"] == pytest.approx(amp, rel=1e-6)
    outside = seg != 1
    np.testing.assert_allclose(comp[outside], amp * psf_stamp[outside], rtol=1e-6, atol=0)
    np.testing.assert_allclose(comp[seg == 1], data[seg == 1], rtol=0, atol=1e-12)
    # added_flux = total / in-segment total = 1 / EE(segment) for a point source
    assert info["added_flux"] == pytest.approx(1.0 / psf_stamp[seg == 1].sum(), rel=1e-6)


def test_classic_low_snr_replaces_the_template_by_a_point_source():
    psf = _psf(41, sigma=2.0)
    shape = (61, 61)
    cen = (30.0, 30.0)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, cen, order=3)
    rng = np.random.default_rng(3)
    data = 1.0 * psf_stamp + rng.normal(0.0, 1e-3, shape)
    seg = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    seg[np.hypot(xx - cen[0], yy - cen[1]) <= 3] = 1

    comp, info = schemes.composite_classic(
        data, seg, 1, psf_stamp,
        params=schemes.ClassicParams(tmpl_snrlo=1e6), tmpl_rms=1e-3,
    )
    assert info["psf_replaced"] is True
    np.testing.assert_allclose(comp, info["fpsf"] * psf_stamp, rtol=0, atol=1e-12)


def test_classic_negative_least_squares_falls_back_to_a_bare_psf():
    psf = _psf(21, sigma=2.0)
    shape = (31, 31)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, (15.0, 15.0), order=3)
    seg = np.zeros(shape, dtype=np.int32)
    seg[14:17, 14:17] = 1
    data = -np.ones(shape)  # negative core -> LS amplitude <= 0

    comp, info = schemes.composite_classic(
        data, seg, 1, psf_stamp,
        params=schemes.ClassicParams(tmpl_snrlo=0.0), tmpl_rms=1.0,
    )
    assert info["fpsf"] == 1.0
    np.testing.assert_allclose(comp, psf_stamp, rtol=0, atol=1e-12)


def test_classic_keeps_negative_pixels_inside_the_segment():
    """IDL applies no positivity clip; only the run-path rewrite would have."""
    psf = _psf(21, sigma=2.0)
    shape = (31, 31)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, (15.0, 15.0), order=3)
    seg = np.zeros(shape, dtype=np.int32)
    seg[13:18, 13:18] = 1
    data = 100.0 * psf_stamp
    data[13, 13] = -5.0

    comp, _ = schemes.composite_classic(
        data, seg, 1, psf_stamp,
        params=schemes.ClassicParams(tmpl_snrlo=0.0), tmpl_rms=1.0,
    )
    assert comp[13, 13] == -5.0


# ---------------------------------------------------------------- wren


def _wren_inputs(shape=(81, 81), cen=(40.0, 40.0), amp=1000.0, sigma=2.0, ivar=1e4):
    psf = _psf(41, sigma=sigma)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, cen, order=1)
    data = amp * psf_stamp
    seg = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    seg[np.hypot(xx - cen[0], yy - cen[1]) <= 3] = 1
    owned = np.ones(shape, dtype=bool)
    return data, seg, owned, psf_stamp, np.full(shape, ivar), cen


def test_wren_high_snr_keeps_the_data_inside_the_segment():
    data, seg, owned, psf_stamp, ivar, cen = _wren_inputs()
    H, info = schemes.composite_wren(
        data, seg, owned, 1, psf_stamp, ivar, cen,
        params=schemes.WrenParams(), max_radius_pix=20.0,
        ee_reach_pix=12.0, annulus_pix=4.0,
    )
    assert info["w_core"] == 1.0
    np.testing.assert_allclose(H[seg == 1], data[seg == 1], rtol=1e-10, atol=0)
    # support is capped at ee_reach: nothing beyond it
    yy, xx = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    far = np.hypot(xx - cen[0], yy - cen[1]) > 12.0
    assert np.all(H[far & (seg == 0)] == 0.0)


def test_wren_faint_source_blends_towards_the_psf_model():
    data, seg, owned, psf_stamp, ivar, cen = _wren_inputs(amp=1.0, ivar=1.0)
    rng = np.random.default_rng(7)
    data = data + rng.normal(0.0, 1.0, data.shape)  # noise: data != a scaled PSF
    H, info = schemes.composite_wren(
        data, seg, owned, 1, psf_stamp, ivar, cen,
        params=schemes.WrenParams(), max_radius_pix=20.0,
        ee_reach_pix=12.0, annulus_pix=4.0,
    )
    assert 0.0 <= info["w_core"] < 1.0
    assert info["psf_extended"] is True
    # In the faint limit the composite approaches A_src * PSF over the support.
    model = info["A_src"] * psf_stamp
    core = seg == 1
    assert np.abs(H[core] - model[core]).sum() < np.abs(data[core] - model[core]).sum()


def test_wren_halo_weights_are_monotone_non_increasing():
    """A bright annulus outside a faint one must not regain data trust."""
    shape = (81, 81)
    cen = (40.0, 40.0)
    psf = _psf(41, sigma=2.0)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, cen, order=1)
    seg = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.hypot(xx - cen[0], yy - cen[1])
    seg[r <= 3] = 1
    ring = (r > 12) & (r <= 16)
    data = 1.0 * psf_stamp  # faint core and halo -> inner annuli SNR < 3
    data[ring] += 1e3  # bright ring far out: on its own it would get w_k = 1
    ivar = np.ones(shape)

    H, info = schemes.composite_wren(
        data, seg, np.ones(shape, bool), 1, psf_stamp, ivar, cen,
        params=schemes.WrenParams(), max_radius_pix=20.0,
        ee_reach_pix=20.0, annulus_pix=4.0,
    )
    assert info["w_core"] < 1.0
    ring_bg = ring & (seg == 0)
    # the ring inherits the running minimum from the annuli inside it, so the
    # composite there stays near the PSF model rather than the raw data
    assert H[ring_bg].max() < 1e-2 * data[ring_bg].max()


def test_wren_failed_psf_lookup_falls_back_to_raw_data():
    data, seg, owned, _psf_stamp, ivar, cen = _wren_inputs()
    H, info = schemes.composite_wren(
        data, seg, owned, 1, None, ivar, cen,
        params=schemes.WrenParams(), max_radius_pix=10.0,
        ee_reach_pix=5.0, annulus_pix=4.0,
    )
    assert info["extend_failed"] is True
    # ext_data (max_radius_pix), not the smaller ext_psf support
    yy, xx = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    r = np.hypot(xx - cen[0], yy - cen[1])
    assert np.any(H[(r > 5) & (r <= 10)] != 0.0)


def test_wren_flux_beyond_stamp_matches_the_truncated_psf_fraction():
    data, seg, owned, psf_stamp, ivar, cen = _wren_inputs()
    _H, info = schemes.composite_wren(
        data, seg, owned, 1, psf_stamp, ivar, cen,
        params=schemes.WrenParams(), max_radius_pix=8.0,
        ee_reach_pix=8.0, annulus_pix=4.0,
    )
    assert info["f_cut"] < 1.0
    assert info["flux_beyond_stamp"] == pytest.approx(
        info["A_src"] * (1.0 - info["f_cut"]), rel=1e-10
    )


# ---------------------------------------------------------------- dispatch


def test_extract_templates_rejects_an_unknown_mode():
    image, segmap = _scene()
    tmpls = Templates()
    with pytest.raises(ValueError, match="extend_mode"):
        tmpls.extract_templates(image, segmap, [(40.0, 40.0)], extend_mode="bogus")


@pytest.mark.parametrize("mode", ["wren", "classic"])
def test_build_schemes_require_a_detection_psf(mode):
    image, segmap = _scene()
    tmpls = Templates()
    with pytest.raises(ValueError, match="detection_psf"):
        tmpls.extract_templates(image, segmap, [(40.0, 40.0)], extend_mode=mode)


@pytest.mark.parametrize("mode", ["none", "psf", "wren", "classic"])
def test_every_mode_yields_a_unit_sum_template_with_a_stored_norm(mode):
    image, segmap = _scene(amps=(1000.0,))
    psf = _psf(41, sigma=2.0)
    tmpls = Templates()
    kw = {"extend_mode": "none"} if mode in {"none", "psf"} else {"extend_mode": mode, "detection_psf": psf}
    tmpls.extract_templates(image, segmap, [(40.0, 40.0)], dilate_segmap=0, **kw)
    if mode == "psf":
        tmpls.extend_with_psf(psf, inplace=True)
    [tmpl] = tmpls.templates
    assert tmpl.data.sum() == pytest.approx(1.0, rel=1e-10)
    assert tmpl.template_norm > 0.0


@pytest.mark.parametrize("mode", ["wren", "classic"])
def test_build_schemes_hold_far_more_of_a_point_source_than_the_bare_segment(mode):
    """The whole point of both schemes: EE(support) close to 1 rather than the
    segment's fraction (template_comparison.tex Fig. 1)."""
    image, segmap = _scene(amps=(1000.0,), seg_radius=2.0)
    psf = _psf(41, sigma=2.0)
    weight = np.full(image.shape, 1e4)

    plain = Templates()
    plain.extract_templates(image, segmap, [(40.0, 40.0)], dilate_segmap=0, extend_mode="none")
    ee_plain = float((plain.templates[0].data > 0).sum())

    built = Templates()
    built.extract_templates(
        image, segmap, [(40.0, 40.0)], dilate_segmap=0,
        extend_mode=mode, detection_psf=psf, detection_weight=weight,
    )
    tmpl = built.templates[0]
    assert tmpl.extension_mode == mode
    assert tmpl.extend_info  # per-source bookkeeping recorded
    assert float((tmpl.data != 0).sum()) > 4 * ee_plain


def test_wren_mode_clips_negatives_and_classic_mode_does_not():
    image, segmap = _scene(amps=(1000.0,))
    image[38, 38] = -50.0  # a negative sky pixel inside the segment
    psf = _psf(41, sigma=2.0)
    weight = np.full(image.shape, 1e4)

    out = {}
    for mode in ("wren", "classic"):
        tmpls = Templates()
        tmpls.extract_templates(
            image, segmap, [(40.0, 40.0)], dilate_segmap=0,
            extend_mode=mode, detection_psf=psf, detection_weight=weight,
            classic=schemes.ClassicParams(tmpl_snrlo=0.0),
        )
        out[mode] = tmpls.templates[0]

    assert out["wren"].data.min() >= 0.0
    assert out["classic"].data.min() < 0.0


def test_classic_mode_sizes_the_cutout_to_the_psf_stamp():
    """IDL pastes the PSF over the whole tile; the stamp footprint is the
    support, so the cutout must be at least the PSF stamp."""
    image, segmap = _scene(amps=(1000.0,))
    psf = _psf(41, sigma=2.0)
    tmpls = Templates()
    tmpls.extract_templates(
        image, segmap, [(40.0, 40.0)], dilate_segmap=0,
        extend_mode="classic", detection_psf=psf,
        classic=schemes.ClassicParams(tmpl_snrlo=0.0),
    )
    [tmpl] = tmpls.templates
    assert min(tmpl.data.shape) >= 41


def test_pipeline_extend_mode_selects_the_build_scheme():
    """One knob, four schemes, same inputs: the 1-1 comparison entry point."""
    from astropy.table import Table

    from mophongo import pipeline
    from mophongo.fit import FitConfig

    image, segmap = _scene(shape=(81, 81), amps=(1000.0,), seg_radius=3.0)
    image = image.astype(np.float32)
    psf = _psf(11, sigma=2.0)
    catalog = Table({"id": [1], "x": [40.0], "y": [40.0]})

    modes = {}
    for mode in ("none", "psf_wings", "psf", "wren", "classic"):
        pipe = pipeline.Pipeline(
            [image, image],
            segmap,
            catalog=catalog,
            weights=[np.ones_like(image), np.ones_like(image)],
            psfs=[psf, psf],
            kernels=[None, None],
            config=FitConfig(
                fit_astrometry_niter=0,
                template_dilate_segmap=0,
                extend_mode=mode,
                classic_tmpl_snrlo=0.0,
                psf_wings_snrlo=0.0,
            ),
        )
        # 'psf' is the legacy alias of 'psf_convolution'
        assert pipe.extend_mode == ("psf_convolution" if mode == "psf" else mode)
        pipe.run()
        tmpl = pipe.templates_extended.templates[0]
        modes[mode] = tmpl
        # 'psf_wings' deliberately sums to < 1 (neighbour wings dropped)
        assert tmpl.data.sum() == pytest.approx(1.0, rel=1e-6) or mode == "psf_wings"

    assert modes["none"].extension_mode == "none"
    assert modes["psf_wings"].extension_mode == "psf_wings"
    assert modes["psf"].extension_mode == "psf_convolution"
    assert modes["wren"].extension_mode == "wren"
    assert modes["classic"].extension_mode == "classic"
    # every extending scheme puts light outside the segment
    n_none = int(np.count_nonzero(modes["none"].data))
    for mode in ("psf_wings", "psf", "wren", "classic"):
        assert int(np.count_nonzero(modes[mode].data)) > n_none


def test_pipeline_legacy_extend_templates_still_selects_the_mode():
    from astropy.table import Table

    from mophongo import pipeline
    from mophongo.fit import FitConfig

    image, segmap = _scene(shape=(41, 41), positions=((20.0, 20.0),), amps=(100.0,))
    image = image.astype(np.float32)
    catalog = Table({"id": [1], "x": [20.0], "y": [20.0]})
    kwargs = dict(
        catalog=catalog,
        weights=[np.ones_like(image), np.ones_like(image)],
        psfs=[np.eye(3, dtype=float), np.eye(3, dtype=float)],
        kernels=[None, None],
    )

    assert pipeline.Pipeline([image, image], segmap, extend_templates="psf_wings", **kwargs).extend_mode == "psf_wings"
    assert pipeline.Pipeline([image, image], segmap, extend_templates="none", **kwargs).extend_mode == "none"
    assert pipeline.Pipeline([image, image], segmap, **kwargs).extend_mode == "psf_wings"  # config default
    with pytest.raises(ValueError, match="Unknown template extension mode"):
        pipeline.Pipeline([image, image], segmap, extend_templates="bogus", **kwargs)


def test_noise_estimators_survive_a_zero_padded_mosaic():
    """``Pipeline.load_data`` nan_to_num's the detection image, so no-coverage
    pixels are exact zeros. A majority-zero sample collapses the MAD to 0,
    which would zero every wren blend weight and disable IDL's low-SNR
    branch."""
    rng = np.random.default_rng(11)
    image = np.zeros((400, 400))
    image[:250, :250] = rng.normal(0.0, 0.05, (250, 250))  # 61% blank
    segmap = np.zeros(image.shape, dtype=np.int32)

    assert schemes.sky_sigma(image, segmap) == pytest.approx(0.05, rel=0.1)
    assert schemes.detection_rms(image) == pytest.approx(0.05, rel=0.1)
    # counting the blank margin as sky: the clipped MAD collapses to zero
    # outright, and the biweight scale comes out badly biased low
    all_px = np.ones(image.shape, dtype=bool)
    assert schemes.sky_sigma(image, segmap, valid=all_px) is None
    assert schemes.detection_rms(image, valid=all_px) < 0.5 * 0.05


def test_wren_refuses_to_run_without_any_noise_estimate():
    """Silently returning w=0 everywhere would make every template a bare
    point source, with no error and extend_failed False."""
    image = np.zeros((41, 41))
    image[20, 20] = 1.0
    segmap = np.zeros(image.shape, dtype=np.int32)
    segmap[19:22, 19:22] = 1
    psf = _psf(11, sigma=2.0)

    tmpls = Templates()
    with pytest.raises(ValueError, match="noise estimate"):
        tmpls.extract_templates(
            image, segmap, [(20.0, 20.0)], dilate_segmap=0,
            extend_mode="wren", detection_psf=psf,
        )
    # an explicit rms unblocks it
    tmpls.extract_templates(
        image, segmap, [(20.0, 20.0)], dilate_segmap=0,
        extend_mode="wren", detection_psf=psf,
        wren=schemes.WrenParams(bg_rms=0.01),
    )
    assert tmpls.templates[0].extend_info["snr_seg"] > 0


def test_classic_refuses_to_run_with_an_unusable_rms():
    image = np.zeros((41, 41))
    image[20, 20] = 1.0
    segmap = np.zeros(image.shape, dtype=np.int32)
    segmap[19:22, 19:22] = 1
    psf = _psf(11, sigma=2.0)

    tmpls = Templates()
    with pytest.raises(ValueError, match="detection rms"):
        tmpls.extract_templates(
            image, segmap, [(20.0, 20.0)], dilate_segmap=0,
            extend_mode="classic", detection_psf=psf,
        )
    # tmpl_snrlo=0 is IDL's keyword_set() guard: the branch is simply skipped
    tmpls.extract_templates(
        image, segmap, [(20.0, 20.0)], dilate_segmap=0,
        extend_mode="classic", detection_psf=psf,
        classic=schemes.ClassicParams(tmpl_snrlo=0.0),
    )
    assert tmpls.templates[0].extend_info["psf_replaced"] is False


def test_classic_ivar_noise_reduces_to_idl_scalar_for_uniform_weights():
    """IDL's sqrt(n_seg)*tmpl_rms IS sqrt(sum 1/ivar) when the noise is
    uniform, so the calibrated path must reproduce it exactly."""
    psf = _psf(21, sigma=2.0)
    shape = (41, 41)
    cen = (20.0, 20.0)
    psf_stamp = schemes.sample_psf_on_stamp(psf, shape, cen, order=3)
    seg = np.zeros(shape, dtype=np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    seg[np.hypot(xx - cen[0], yy - cen[1]) <= 3] = 1
    data = 100.0 * psf_stamp
    sigma = 0.7

    params = schemes.ClassicParams(tmpl_snrlo=0.0)
    _, scalar = schemes.composite_classic(
        data, seg, 1, psf_stamp, params=params, tmpl_rms=sigma
    )
    _, formal = schemes.composite_classic(
        data, seg, 1, psf_stamp, params=params,
        tmpl_rms=0.0, ivar=np.full(shape, 1.0 / sigma ** 2),
    )
    assert formal["snr_seg"] == pytest.approx(scalar["snr_seg"], rel=1e-12)


def test_classic_with_weights_needs_no_scalar_rms():
    """A calibrated ivar map supersedes IDL's per-tile scalar entirely."""
    image, segmap = _scene(amps=(1000.0,))
    psf = _psf(41, sigma=2.0)
    tmpls = Templates()
    tmpls.extract_templates(
        image, segmap, [(40.0, 40.0)], dilate_segmap=0,
        extend_mode="classic", detection_psf=psf,
        detection_weight=np.full(image.shape, 1e4),
    )
    assert tmpls.templates[0].extend_info["snr_seg"] > 0


def test_template_extension_refuses_a_lower_resolution_psf():
    """psfs[0] is the detection band and nothing else is substituted: a
    lo-res PSF would give wrong wings and wrong radii, silently."""
    from astropy.table import Table

    from mophongo import pipeline
    from mophongo.fit import FitConfig

    image, segmap = _scene(shape=(81, 81), amps=(1000.0,))
    image = image.astype(np.float32)
    catalog = Table({"id": [1], "x": [40.0], "y": [40.0]})

    for mode in ("psf_wings", "psf", "wren", "classic"):
        pipe = pipeline.Pipeline(
            [image, image],
            segmap,
            catalog=catalog,
            weights=[np.ones_like(image), np.ones_like(image)],
            psfs=[None, _psf(11, sigma=2.0)],  # detection PSF missing
            kernels=[None, None],
            config=FitConfig(
                fit_astrometry_niter=0, template_dilate_segmap=0, extend_mode=mode
            ),
        )
        with pytest.raises(ValueError, match="detection-band PSF in psfs"):
            pipe.run()


def test_extract_templates_does_not_dilate_by_default():
    image, segmap = _scene(amps=(1000.0,), seg_radius=2.0)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(40.0, 40.0)], extend_mode="none")
    tmpl = tmpls.templates[0]
    support = np.zeros(image.shape, bool)
    support[tmpl.slices_original] = tmpl.data[tmpl.slices_cutout] != 0
    assert int(support.sum()) == int((segmap == 1).sum())


def test_wren_ownership_splits_a_blend_between_neighbours():
    """Two neighbours must not both claim the background between them."""
    image, segmap = _scene(
        positions=((30.0, 40.0), (52.0, 40.0)), amps=(1000.0, 1000.0), seg_radius=3.0
    )
    psf = _psf(41, sigma=2.0)
    weight = np.full(image.shape, 1e4)
    tmpls = Templates()
    tmpls.extract_templates(
        image, segmap, [(30.0, 40.0), (52.0, 40.0)], dilate_segmap=0,
        extend_mode="wren", detection_psf=psf, detection_weight=weight,
    )
    a, b = tmpls.templates
    full_a = np.zeros(image.shape)
    full_b = np.zeros(image.shape)
    full_a[a.slices_original] = a.data[a.slices_cutout]
    full_b[b.slices_original] = b.data[b.slices_cutout]
    overlap = (full_a > 0) & (full_b > 0)
    # supports are disjoint by construction (segment pixels + owned background)
    assert not overlap.any()
