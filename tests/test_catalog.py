"""Source masking, background bias and ivar calibration in ``get_bg_and_ivar``,
plus the deblend bookkeeping ``Catalog`` owns.

The estimator masks sources out of the background fit with a two-pass
detection on a coarse grid, then measures the residual scatter to calibrate
the weight map. Three properties have to hold together, and the source-free
scalar-noise case exercises none of them:

* the mask must cover sources, and grow (never shrink) with ``dilate``;
* the fitted background must be unbiased where there are no sources;
* the recovered ``sigma_true`` must track the real noise, including the
  correlated-noise and depth dependence it exists to absorb.

The last sections pin what each of the two outputs is allowed to mask, that a
big-endian mosaic is not copied on the way in, and that a deblended child
records the parent it came from.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mophongo.catalog import _mean_downsample, coarse_source_mask, get_bg_and_ivar

SIZE = 512


def _field(seed=11, *, n_src=150, bg_level=0.05, bg_grad=0.03, corr=0.0, depth=1.0):
    """Correlated noise + a linear background + Gaussian sources.

    ``wht`` is an honest inverse variance for the noise actually present, so a
    correct estimator recovers ``sigma_true`` = 1 when ``corr`` is zero.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (SIZE, SIZE))
    if corr > 0:
        noise = gaussian_filter(noise, corr)
        noise /= noise.std()
    noise *= depth

    _y, x = np.mgrid[0:SIZE, 0:SIZE]
    bg_true = bg_level + bg_grad * (x / SIZE) * depth

    src = np.zeros((SIZE, SIZE))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for _ in range(n_src):
        xc, yc = rng.uniform(10, SIZE - 10, 2)
        sig = rng.uniform(1.0, 2.5)
        amp = 10 ** rng.uniform(0.0, 1.6) * depth
        src += amp * np.exp(-0.5 * (((xx - xc) / sig) ** 2 + ((yy - yc) / sig) ** 2))

    sci = (noise + bg_true + src).astype(np.float32)
    wht = np.full((SIZE, SIZE), 1.0 / depth**2, dtype=np.float32)
    return sci, wht, bg_true.astype(np.float32), src.astype(np.float32), depth


def _sigma_true(ivar, wht):
    """Invert the ivar rescaling to recover the measured ``sigma_true``."""
    ok = (wht > 0) & (ivar > 0)
    return float(np.sqrt(np.median(wht[ok] / ivar[ok])))


def test_background_is_unbiased_on_a_source_free_field():
    """No sources: the fit must track the true background, not the noise."""
    sci, wht, bg_true, _src, _d = _field(n_src=0)
    bg, ivar = get_bg_and_ivar(sci, wht)

    assert np.abs(np.mean(bg - bg_true)) < 0.02
    assert _sigma_true(ivar, wht) == pytest.approx(1.0, abs=0.05)


def test_sources_are_masked_and_background_stays_unbiased():
    """With sources present the background must not absorb their flux."""
    sci, wht, bg_true, src, _d = _field(n_src=150)
    bg, ivar = get_bg_and_ivar(sci, wht)

    # measured where the sources are faintest, so any leaked flux shows up
    quiet = _mean_downsample(src, 8) < 0.05 * src.max()
    bg_bin = _mean_downsample(bg_true, 8)
    bias = float(np.mean((_mean_downsample(bg, 8) - bg_bin)[quiet]))
    assert abs(bias) < 0.05, f"background biased by {bias:+.4f} sigma"
    assert _sigma_true(ivar, wht) == pytest.approx(1.0, abs=0.10)


def _coarse_det(sci, wht, step=8):
    """The median-subtracted coarse detection image and its robust sigma."""
    s_bin = _mean_downsample(sci, step)
    w_bin = _mean_downsample(wht, step)
    det = s_bin * np.sqrt(w_bin)
    med0 = float(np.median(det))
    sigma0 = float(np.median(np.abs(det - med0)) * 1.4826)
    return det - med0, sigma0


def test_mask_covers_source_flux():
    """Most injected flux must land under the mask, not in the background."""
    sci, wht, _bg_true, src, _d = _field(n_src=150)
    det, sigma0 = _coarse_det(sci, wht)
    mask = coarse_source_mask(det, sigma0)

    src_bin = _mean_downsample(src, 8)
    covered = float(src_bin[mask].sum() / src_bin.sum())
    assert covered > 0.95, f"only {covered:.1%} of source flux is masked"


def test_pure_noise_mask_stays_small():
    """A source-free field must leave the background sample nearly intact.

    With an unnormalized smoothing kernel the bright pass was compared
    against a threshold a factor of 29 too low and flagged ~47% of pure
    noise; the inverted dilation then hid that by re-admitting it.
    """
    sci, wht, _bg, _src, _d = _field(n_src=0)
    det, sigma0 = _coarse_det(sci, wht)
    occupancy = float(coarse_source_mask(det, sigma0).mean())
    assert occupancy < 0.05, f"mask covers {occupancy:.1%} of a pure noise field"


def test_dilation_grows_the_exclusion():
    """More dilation must mask more, never less.

    Dilating the background mask instead of the source mask inverts this: a
    larger ``dilate`` grew the background *into* the sources, re-admitting
    100% of a 5-pixel source's own pixels at ``dilate=3``.

    The masks are not nested, because ``dilate`` also sets the radius of the
    bright pass's smoothing kernel and so changes what is detected in the
    first place; only the total extent is monotone.
    """
    sci, wht, _bg, _src, _d = _field(n_src=150)
    det, sigma0 = _coarse_det(sci, wht)

    sizes = {
        d: int(coarse_source_mask(det, sigma0, dilate=d).sum()) for d in (1, 2, 3, 5)
    }
    assert sizes[1] < sizes[2] < sizes[3] < sizes[5], sizes


def test_dilation_of_a_compact_source_keeps_its_own_pixels():
    """A source's own pixels must stay masked at every dilation radius.

    A regression guard rather than a reproduction: dilating the *background*
    mask re-admits 100% of an isolated r<=2 source, but inside this estimator
    the bright pass's smoothing halo already inflates the segment by the
    kernel radius, so the inverted dilation ate the halo instead of the
    source and this property survived. The defect shows up in mask coverage
    and in the ``dilate`` dependence instead.
    """
    for radius in (1, 2, 3, 5, 8):
        det = np.zeros((81, 81), dtype=np.float32)
        yy, xx = np.mgrid[0:81, 0:81]
        core = (yy - 40) ** 2 + (xx - 40) ** 2 <= radius * radius
        det[core] = 50.0
        mask = coarse_source_mask(det, 1.0, min_npixels_bright=1)
        assert mask[core].all(), (
            f"r={radius}: {100 * (1 - mask[core].mean()):.0f}% of the source "
            "was re-admitted as background"
        )


@pytest.mark.parametrize("depth", [1.0, 0.25])
def test_ivar_calibration_follows_depth(depth):
    """The recovered scaling must be depth independent when wht is honest."""
    sci, wht, _bg, _src, _d = _field(n_src=150, depth=depth)
    _bg_img, ivar = get_bg_and_ivar(sci, wht)
    assert _sigma_true(ivar, wht) == pytest.approx(1.0, abs=0.10)


def test_correlated_noise_is_absorbed_into_sigma_true():
    """Correlated noise must inflate sigma_true, which is what it is for."""
    sci_u, wht, _b, _s, _d = _field(n_src=0, corr=0.0)
    sci_c, _wht, _b2, _s2, _d2 = _field(n_src=0, corr=1.5)

    _bg, ivar_u = get_bg_and_ivar(sci_u, wht)
    _bg2, ivar_c = get_bg_and_ivar(sci_c, wht)

    assert _sigma_true(ivar_u, wht) == pytest.approx(1.0, abs=0.05)
    # smoothing on a 1.5 px scale correlates the block-averaged noise well
    # above what the weight map claims
    assert _sigma_true(ivar_c, wht) > 2.0


# ---------------------------------------------------------------------------
# non-finite input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_single_bad_pixel_does_not_poison_the_estimate(bad):
    """One non-finite science pixel must not reach the outputs.

    Without a finite-science mask the block mean carries it into the median
    and MAD, and both returned arrays came back entirely NaN.
    """
    sci, wht, _bg, _src, _d = _field(n_src=0)
    clean_bg, clean_ivar = get_bg_and_ivar(sci, wht)

    sci = sci.copy()
    sci[100, 100] = bad
    bg, ivar = get_bg_and_ivar(sci, wht)

    assert np.isfinite(bg).all()
    assert np.isfinite(ivar).all()
    # and the calibration is essentially the one from the clean field
    assert _sigma_true(ivar, wht) == pytest.approx(
        _sigma_true(clean_ivar, wht), rel=0.02
    )
    assert np.abs(bg - clean_bg).max() < 0.05


def test_non_finite_border_is_excluded_not_propagated():
    """A NaN border must cost those rows, not the whole image."""
    sci, wht, _bg, _src, _d = _field(n_src=0)
    sci = sci.copy()
    sci[:4, :] = np.nan
    bg, ivar = get_bg_and_ivar(sci, wht)

    assert np.isfinite(bg).all()
    assert np.isfinite(ivar).all()
    assert _sigma_true(ivar, wht) == pytest.approx(1.0, abs=0.15)


def test_zero_weight_border_is_excluded():
    """Zero-weight pixels carry no information and must not be calibrated on."""
    sci, wht, _bg, _src, _d = _field(n_src=0)
    wht = wht.copy()
    wht[:16, :] = 0.0
    sci = sci.copy()
    sci[:16, :] = 1e6  # garbage under zero weight

    bg, ivar = get_bg_and_ivar(sci, wht)
    assert np.isfinite(bg).all()
    assert np.isfinite(ivar).all()
    assert np.all(ivar[:16, :] == 0)
    assert _sigma_true(ivar, wht) == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize(
    "sci_fill, wht_fill",
    [(0.0, 1.0), (np.nan, 1.0), (0.0, 0.0), (np.nan, 0.0)],
)
def test_degenerate_input_does_not_raise(sci_fill, wht_fill):
    """Blank, all-invalid and zero-weight images must return finite arrays."""
    sci = np.full((128, 128), sci_fill, dtype=np.float32)
    wht = np.full((128, 128), wht_fill, dtype=np.float32)

    bg, ivar = get_bg_and_ivar(sci, wht)
    assert bg.shape == sci.shape and ivar.shape == sci.shape
    assert np.isfinite(bg).all()
    assert np.isfinite(ivar).all()


def test_masked_out_nan_does_not_spread_through_the_smoothing():
    """``NaN * 0`` is ``NaN``: excluding a pixel by mask is not enough."""
    from mophongo.catalog import bg_gaussian_normalized

    img = np.ones((64, 64), dtype=np.float32)
    img[32, 32] = np.nan
    mask = np.ones((64, 64), dtype=bool)
    mask[32, 32] = False  # already excluded

    out = bg_gaussian_normalized(img, mask, sigma=4.0)
    assert np.isfinite(out).all()
    assert out == pytest.approx(np.ones_like(out), abs=1e-5)


def test_smoothing_threshold_matches_the_smoothed_noise():
    """The bright threshold must be in units of the noise it is compared to.

    The shipped code convolved with an unnormalized disk, whose white-noise
    RMS is sqrt(N) sigma, but scaled the threshold by the normalized
    kernel's 1/sqrt(N) — a factor of N = 29 for ``disk(3)``.
    """
    from scipy.signal import fftconvolve
    from skimage.morphology import disk

    sci, wht, _bg, _src, _d = _field(n_src=0)
    det, sigma0 = _coarse_det(sci, wht)

    kern = disk(3).astype(np.float32)
    kern_norm = kern / kern.sum()
    detc = fftconvolve(det, kern_norm, mode="same")
    assert np.std(detc) == pytest.approx(
        np.sqrt((kern_norm**2).sum()) * sigma0, rel=0.15
    )

    # the unnormalized pairing the code used to apply is off by N
    unnorm = fftconvolve(det, kern, mode="same")
    coded = np.sqrt((kern**2).sum()) / kern.sum() * sigma0
    assert np.std(unnorm) / coded == pytest.approx(kern.sum(), rel=0.15)


# ---------------------------------------------------------------------------
# what the two outputs are allowed to mask, and what the input may cost
# ---------------------------------------------------------------------------


def _awkward_field(seed=7, shape=(200, 160)):
    """Science + weight pair with the awkward pixels: zero, NaN weight, NaN sci."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0.0, 2e-3, shape).astype(np.float32)
    sci[40:52, 40:52] += 0.08  # something for the source mask to find
    wht = np.full(shape, 3e5, np.float32)
    wht[:4] = 0.0
    wht[-3:] = np.nan
    sci[15, 15] = np.nan
    return sci, wht


def test_get_bg_and_ivar_does_not_copy_on_byte_order():
    """Big-endian input gives identical output and is not cast up front.

    FITS stores big-endian, so a memory-mapped mosaic arrives as '>f4'.
    Casting it to native float32 would copy the whole array -- 3.5 GB per
    input on a MINERVA detection grid -- for a difference nothing downstream
    needs.
    """
    from mophongo.catalog import _as_float

    big = np.zeros((4, 4), dtype=">f4")
    assert _as_float(big) is big
    # a non-float input still has to be converted
    assert _as_float(np.zeros((2, 2), np.int32)).dtype == np.float32

    sci, wht = _awkward_field()
    for need_bg in (True, False):
        bg_n, ivar_n = get_bg_and_ivar(sci, wht, need_bg=need_bg)
        bg_b, ivar_b = get_bg_and_ivar(
            sci.astype(">f4"), wht.astype(">f4"), need_bg=need_bg
        )
        assert np.array_equal(ivar_n, ivar_b)
        assert ivar_n.dtype == ivar_b.dtype == np.float32
        if need_bg:
            assert np.array_equal(bg_n, bg_b)
            assert bg_n.dtype == np.float32
        else:
            assert bg_n is None and bg_b is None


def test_get_bg_and_ivar_masks_ivar_and_background_differently():
    """ivar needs finite science; the background only needs a good weight.

    A lone bad science pixel carries no information, so its inverse variance
    is zeroed -- but the background surface there is still defined, and
    punching a hole in a smooth fit would be worse than keeping it.

    ``need_bg=False`` must not change the weights it returns alongside.
    """
    sci, wht = _awkward_field()
    bg, ivar = get_bg_and_ivar(sci, wht, need_bg=True)

    assert ivar[0, 0] == 0.0        # zero weight
    assert ivar[-1, 0] == 0.0       # NaN weight
    assert ivar[15, 15] == 0.0      # NaN science
    assert bg[0, 0] == 0.0          # zero weight
    assert bg[-1, 0] == 0.0         # NaN weight
    assert bg[15, 15] != 0.0        # NaN science: background still defined

    bg_none, ivar_none = get_bg_and_ivar(sci, wht, need_bg=False)
    assert bg_none is None
    assert np.array_equal(ivar, ivar_none)


# ---------------------------------------------------------------------------
# deblend bookkeeping
# ---------------------------------------------------------------------------


def test_deblend_label_info_marks_children_split_from_parent():
    from mophongo.catalog import _deblend_label_info

    parent = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 2, 2],
        ],
        dtype=int,
    )
    final = np.array(
        [
            [3, 3, 0, 5, 5],
            [3, 4, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ],
        dtype=int,
    )

    info = _deblend_label_info(final, parent)

    assert info[3] == (1, 2, True)
    assert info[4] == (1, 2, True)
    assert info[5] == (2, 1, False)
