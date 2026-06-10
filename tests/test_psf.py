from pathlib import Path
import sys
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve as scipy_fftconvolve
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import (
    DrizzlePSF,
    EffectivePSF,
    PSF,
    _growth_curve_ratio_for_plot,
    _growth_curve_ratio_plot_samples,
    _kernel_regularization,
    pad_to_shape,
)
from mophongo.templates import _convolve2d
from mophongo.utils import (
    convolve2d as utils_convolve2d,
    fit_kernel_fourier,
    fftconvolve,
    gaussian,
    matching_kernel as utils_matching_kernel,
    resize_flux_conserving_inter_cubic,
)
from utils import make_simple_data


class _FakeEffectivePSF:
    """Minimal finite-integral ePSF provider for DrizzlePSF tests."""

    filter_pattern = "TEST"

    def __init__(self):
        self.epsf = OrderedDict({"TEST": np.ones((81, 81, 1), dtype=np.float32)})
        self.extended_epsf = {}
        self.last_eval_sum = np.nan

    def get_at_position(self, x, y, filter, rot90=0):
        return self.epsf[filter][:, :, 0]

    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        sigma = 2.3
        psf = 0.95 * np.exp(-0.5 * ((dx / sigma) ** 2 + (dy / sigma) ** 2))
        self.last_eval_sum = float(np.sum(psf, dtype=np.float64))
        return psf.astype(np.float32)


def _make_test_wcs(size: int = 128, pscale: float = 0.04) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2.0 + 0.5, size / 2.0 + 0.5]
    wcs.wcs.crval = [34.5, -5.2]
    wcs.wcs.cdelt = [-pscale / 3600.0, pscale / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.pixel_shape = (size, size)
    wcs.pscale = pscale
    return wcs


def _make_synthetic_dpsf(
    tmp_path,
    *,
    driz_pscale: float = 0.04,
    input_pscale: float | None = None,
) -> tuple[DrizzlePSF, _FakeEffectivePSF]:
    driz_wcs = _make_test_wcs(pscale=driz_pscale)
    flt_wcs = _make_test_wcs(pscale=input_pscale or driz_pscale)
    flt_wcs.expweight = 1.0
    header = driz_wcs.to_header()
    path = tmp_path / "synthetic_driz.fits"
    fits.writeto(path, np.zeros((128, 128), dtype=np.float32), header, overwrite=True)

    key = ("synthetic_flt.fits", 1)
    info = (
        [key],
        {key: flt_wcs},
        {key: Polygon(flt_wcs.calc_footprint())},
        {key: fits.Header()},
    )
    epsf = _FakeEffectivePSF()
    return DrizzlePSF(driz_image=str(path), info=info, epsf_obj=epsf), epsf


def test_psf_matching_kernel_properties():
    _, _, _, psfs, _, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    assert kernel.shape == psf_lo.array.shape
    conv = _convolve2d(pad_to_shape(psf_hi.array, kernel.shape), kernel)
    np.testing.assert_allclose(conv, psf_lo.array, rtol=0, atol=3e-3)


def test_fftconvolve_preserves_even_kernel_alignment():
    image = np.zeros((8, 8))
    image[4, 4] = 1.0
    kernel = np.zeros((4, 4))
    kernel[2, 2] = 1.0

    centered = fftconvolve(image, kernel, mode="same")
    scipy_same = scipy_fftconvolve(image, kernel, mode="same")

    assert np.unravel_index(np.argmax(centered), centered.shape) == (4, 4)
    assert np.unravel_index(np.argmax(scipy_same), scipy_same.shape) == (5, 5)


def test_fftconvolve_full_matches_scipy_full():
    image = np.arange(25, dtype=float).reshape(5, 5)
    kernel = np.array([[0.0, 2.0], [1.0, -0.5]])

    np.testing.assert_allclose(
        fftconvolve(image, kernel, mode="full"),
        scipy_fftconvolve(image, kernel, mode="full"),
        rtol=0,
        atol=1e-12,
    )


def test_convolve2d_uses_true_centered_convolution_for_asymmetric_kernel():
    image = np.zeros((9, 9))
    image[4, 4] = 1.0
    kernel = np.zeros((3, 3))
    kernel[0, 2] = 1.0

    conv = _convolve2d(image, kernel)
    expected = fftconvolve(image, kernel, mode="same")
    correlation_like = fftconvolve(image, kernel[::-1, ::-1], mode="same")

    np.testing.assert_allclose(conv, expected, rtol=0, atol=1e-12)
    assert np.unravel_index(np.argmax(conv), conv.shape) != np.unravel_index(
        np.argmax(correlation_like), correlation_like.shape
    )


def test_convolve2d_preserves_even_kernel_alignment():
    image = np.zeros((8, 8))
    image[4, 4] = 1.0
    kernel = np.zeros((4, 4))
    kernel[2, 2] = 1.0

    conv = _convolve2d(image, kernel)

    assert np.unravel_index(np.argmax(conv), conv.shape) == (4, 4)


def test_utils_convolve2d_uses_shared_convolution_for_asymmetric_kernel():
    image = np.zeros((9, 9))
    image[4, 4] = 1.0
    kernel = np.zeros((3, 3))
    kernel[0, 2] = 1.0

    conv = utils_convolve2d(image, kernel)
    expected = fftconvolve(image, kernel, mode="same")
    correlation_like = fftconvolve(image, kernel[::-1, ::-1], mode="same")

    assert conv.shape == image.shape
    np.testing.assert_allclose(conv, expected, rtol=0, atol=1e-12)
    assert np.unravel_index(np.argmax(conv), conv.shape) != np.unravel_index(
        np.argmax(correlation_like), correlation_like.shape
    )


def test_utils_convolve2d_uses_shared_convolution_for_even_kernel():
    image = np.zeros((8, 8))
    image[4, 4] = 1.0
    kernel = np.zeros((4, 4))
    kernel[2, 2] = 1.0

    conv = utils_convolve2d(image, kernel)
    expected = fftconvolve(image, kernel, mode="same")

    assert conv.shape == image.shape
    np.testing.assert_allclose(conv, expected, rtol=0, atol=1e-12)
    assert np.unravel_index(np.argmax(conv), conv.shape) == (4, 4)


def test_opencv_cubic_resize_conserves_flux_and_nested_centroid():
    psf_lo = gaussian((64, 64), fwhm=5.5, x0=29.37, y0=33.61, flux=1.7)
    psf_hi = resize_flux_conserving_inter_cubic(psf_lo, 2.0)

    yy_lo, xx_lo = np.indices(psf_lo.shape, dtype=float)
    yy_hi, xx_hi = np.indices(psf_hi.shape, dtype=float)
    flux_lo = psf_lo.sum(dtype=np.float64)
    flux_hi = psf_hi.sum(dtype=np.float64)
    y_lo = float((psf_lo * yy_lo).sum() / flux_lo)
    x_lo = float((psf_lo * xx_lo).sum() / flux_lo)
    y_hi = float((psf_hi * yy_hi).sum() / flux_hi)
    x_hi = float((psf_hi * xx_hi).sum() / flux_hi)

    np.testing.assert_allclose(flux_hi, flux_lo, rtol=0, atol=5e-7)
    np.testing.assert_allclose([y_hi, x_hi], [2.0 * y_lo + 0.5, 2.0 * x_lo + 0.5], atol=2e-5)


def test_matching_kernel_preserves_input_flux_ratio_without_normalizing():
    source = gaussian((65, 65), fwhm=4.5, flux=2.0)
    target = source * 2.5
    kernel = utils_matching_kernel(source, target, recenter=False)

    np.testing.assert_allclose(kernel.sum(), target.sum() / source.sum(), rtol=0, atol=5e-10)


def test_growth_curve_ratio_plot_samples_start_above_point_seven_pixel():
    radii = np.array([0.0, 0.5, 0.7, 1.0, 1.5, 2.0])
    numerator = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 16.0])
    denominator = np.array([1.0, 0.0, 1.5, 2.0, 4.0, 8.0])

    plot_radii, ratio = _growth_curve_ratio_for_plot(radii, numerator, denominator)

    np.testing.assert_allclose(plot_radii, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(ratio, [2.0, 2.0, 2.0])


def test_growth_curve_ratio_plot_samples_dense_near_point_seven_pixel():
    source = PSF.gaussian(101, 2.0, 2.0).array
    target = PSF.gaussian(101, 3.0, 3.0).array

    plot_radii, ratio = _growth_curve_ratio_plot_samples(source, target, 50.0)

    assert plot_radii[0] > 0.7
    assert plot_radii[0] < 0.75
    assert np.count_nonzero(plot_radii < 1.0) > 3
    assert np.all(np.isfinite(ratio))


def test_unit_template_kernel_maps_to_native_target_flux():
    source_native = np.zeros((65, 65), dtype=float)
    source_native[32, 32] = 0.95
    target_native = np.zeros_like(source_native)
    target_native[32, 32] = 0.80
    source_unit = source_native / source_native.sum()
    flux_true = 2.0
    image = flux_true * target_native

    class UnitWindow:
        def __call__(self, shape):
            return np.ones(shape, dtype=float)

    native_kernel = utils_matching_kernel(
        source_native,
        target_native,
        window=UnitWindow(),
        recenter=False,
    )
    unit_source_kernel = utils_matching_kernel(
        source_unit,
        target_native,
        window=UnitWindow(),
        recenter=False,
    )

    native_kernel_template = fftconvolve(source_unit, native_kernel, mode="same")
    unit_kernel_template = fftconvolve(source_unit, unit_source_kernel, mode="same")

    def fit_flux(template):
        return float(np.sum(template * image) / np.sum(template * template))

    np.testing.assert_allclose(
        fit_flux(native_kernel_template),
        flux_true * source_native.sum(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(fit_flux(unit_kernel_template), flux_true, rtol=0, atol=1e-12)
    np.testing.assert_allclose(unit_source_kernel.sum(), target_native.sum(), rtol=0, atol=1e-12)


def test_psf_from_array_preserves_native_sum():
    arr = gaussian((21, 21), fwhm=3.0, flux=2.5)
    psf = PSF.from_array(arr)

    np.testing.assert_allclose(psf.array.sum(), arr.sum(), rtol=0, atol=1e-12)


def test_drizzlepsf_full_stamp_conserves_finite_epsf_flux(tmp_path):
    dpsf, epsf = _make_synthetic_dpsf(tmp_path, driz_pscale=0.04, input_pscale=0.08)
    ra, dec = dpsf.driz_wcs.pixel_to_world_values(64.0, 64.0)

    cube = dpsf.get_psf_radec(
        [(float(ra), float(dec))],
        filter="TEST",
        size=61,
        parity="any",
    )

    assert cube.shape == (1, 61, 61)
    np.testing.assert_allclose(
        np.sum(cube[0], dtype=np.float64),
        epsf.last_eval_sum,
        rtol=0,
        atol=2e-6,
    )


def test_effectivepsf_jwst_eval_uses_stpsf_plane_center():
    epsf = EffectivePSF()
    epsf.eval_psf_type = "NRC"
    epsf.eval_psf_oversample = 4
    yy, xx = np.indices((260, 260), dtype=float)
    psf_xy = np.exp(-0.5 * (((yy - 129.5) / 4.0) ** 2 + ((xx - 129.5) / 4.0) ** 2))

    value = epsf.eval_ePSF(psf_xy, np.array([[0.0]]), np.array([[0.0]]))[0, 0]
    expected = map_coordinates(psf_xy, np.array([[129.5], [129.5]]), order=3)[0]

    np.testing.assert_allclose(value, expected, rtol=0, atol=2e-6)


def test_drizzlepsf_partial_stamp_matches_full_stamp_crop(tmp_path):
    dpsf, _epsf = _make_synthetic_dpsf(tmp_path, driz_pscale=0.04, input_pscale=0.08)
    ra, dec = dpsf.driz_wcs.pixel_to_world_values(64.0, 64.0)

    full = dpsf.get_psf_radec([(float(ra), float(dec))], filter="TEST", size=61, parity="any")[0]
    central = dpsf.get_psf_radec([(float(ra), float(dec))], filter="TEST", size=9, parity="any")[0]
    y0 = (full.shape[0] - central.shape[0]) // 2
    x0 = (full.shape[1] - central.shape[1]) // 2

    np.testing.assert_allclose(
        central,
        full[y0:y0 + central.shape[0], x0:x0 + central.shape[1]],
        rtol=0,
        atol=0,
    )


def test_basis_kernel_fit_preserves_native_flux_ratio_without_normalizing():
    source = np.zeros((9, 9), dtype=float)
    source[4, 4] = 2.0
    target = np.zeros_like(source)
    target[4, 4] = 5.0
    basis = np.zeros((9, 9, 1), dtype=float)
    basis[4, 4, 0] = 1.0

    kernel, coeffs = fit_kernel_fourier(source, target, basis)

    np.testing.assert_allclose(coeffs[0], 2.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(kernel.sum(), target.sum() / source.sum(), rtol=0, atol=1e-12)


def test_kernel_regularization_defaults_to_cancellation_only():
    kernel = np.array(
        [
            [0.0, -0.25, 0.0],
            [-0.25, 2.0, -0.25],
            [0.0, -0.25, 0.0],
        ],
        dtype=float,
    )
    regularization, high_frequency, cancellation = _kernel_regularization(kernel)

    assert high_frequency > 0
    assert cancellation > 0
    np.testing.assert_allclose(regularization, cancellation**2)


def test_regularized_matching_kernel_writes_method_diagnostic(tmp_path):
    source = PSF.gaussian(31, 1.5, 1.5).array
    target = PSF.gaussian(31, 2.5, 2.5).array

    result = PSF.from_array(source).optimize_matching_kernel_regularization(
        PSF.from_array(target),
        method="wiener",
        reg_grid=np.array([1e-4, 1e-3]),
        l2_weight=1.0,
        diagnostic_path=tmp_path,
        source_label="F444W",
        target_label="F770W (target)",
        diagnostic_note="unit-test PSF pair",
    )

    assert result.method == "wiener"
    diagnostic = tmp_path / "diagnostic_wiener.png"
    assert diagnostic.exists()
    assert diagnostic.stat().st_size > 0


def test_regularized_matching_kernel_default_lambda_range():
    source = PSF.gaussian(21, 1.3, 1.3).array
    target = PSF.gaussian(21, 2.1, 2.1).array

    result = PSF.from_array(source).optimize_matching_kernel_regularization(
        PSF.from_array(target),
        method="wiener",
        l2_weight=1.0,
    )

    np.testing.assert_allclose(result.reg_grid[0], 1e-6)
    np.testing.assert_allclose(result.reg_grid[-1], 1e-1)
