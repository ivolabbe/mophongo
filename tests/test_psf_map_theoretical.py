"""Theoretical target PSFs and spatially varying deconvolution kernels."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from mophongo.psf import PSF, psf_core_centroid, psf_core_fwhm
from mophongo.psf_map import PSFRegionMap
from mophongo.utils import fftconvolve, pad_to_shape


def _source_map() -> PSFRegionMap:
    regions = gpd.GeoDataFrame(
        {"psf_key": [0, 1]},
        geometry=[box(10.0, 0.0, 10.01, 0.01), box(10.01, 0.0, 10.02, 0.01)],
        crs="EPSG:4326",
    )
    psfs = np.stack(
        [
            PSF.gaussian(32, fwhm=4.0, x0=15.2, y0=15.7).array,
            PSF.gaussian(32, fwhm=5.0, x0=15.8, y0=15.3).array,
        ]
    )
    return PSFRegionMap(regions, psfs=psfs, pscale=0.04, name="source")


def test_gaussian_psf_map_is_unit_sum_padded_and_phase_matched():
    source = _source_map()
    target = source.gaussian_psf_map(2.5, shape=64, phase_match=True)

    assert target.psfs.shape == (2, 64, 64)
    assert target.psfs.dtype == np.float32
    assert np.allclose(target.psfs.sum(axis=(1, 2)), 1.0, atol=2e-7)
    assert np.all(target.regions["target_fwhm_pix"] == 2.5)
    assert np.allclose(target.regions["target_fwhm"], 0.1)

    # The target is padded by 16 pixels on each side, but retains the source
    # core's region-dependent subpixel phase.
    for key in (0, 1):
        source_xy = np.asarray(psf_core_centroid(source.psfs[key]))
        target_xy = np.asarray(psf_core_centroid(target.psfs[key])) - 16.0
        assert np.allclose(target_xy, source_xy, atol=0.15)
        assert np.mean(psf_core_fwhm(target.psfs[key])) == pytest.approx(2.6, abs=0.08)


def test_moffat_psf_map_is_unit_sum_padded_and_phase_matched():
    source = _source_map()
    target = source.moffat_psf_map(
        2.5, beta=2.5, shape=64, phase_match=True
    )

    assert target.psfs.shape == (2, 64, 64)
    assert target.psfs.dtype == np.float32
    assert np.allclose(target.psfs.sum(axis=(1, 2)), 1.0, atol=2e-7)
    assert np.all(target.regions["target_model"] == "moffat")
    assert np.all(target.regions["target_fwhm_pix"] == 2.5)
    assert np.all(target.regions["target_beta"] == 2.5)
    assert np.all(target.regions["target_discrete_sum"] > 0.99)

    for key in (0, 1):
        source_xy = np.asarray(psf_core_centroid(source.psfs[key]))
        target_xy = np.asarray(psf_core_centroid(target.psfs[key])) - 16.0
        assert np.allclose(target_xy, source_xy, atol=0.15)
        assert np.mean(psf_core_fwhm(target.psfs[key])) == pytest.approx(
            2.7, abs=0.12
        )


def test_wiener_kernel_map_sharpens_and_records_stability_metrics(tmp_path):
    source = _source_map()
    target = source.gaussian_psf_map(2.5, shape=64, phase_match=True)
    kernels = source.matching_kernel_map(target, method="wiener", reg=1e-3)

    assert kernels.psfs.shape == (2, 64, 64)
    assert kernels.psfs.dtype == np.float32
    assert np.allclose(kernels.psfs.sum(axis=(1, 2)), 1.0, atol=3e-7)
    assert np.all(kernels.regions["kernel_noise_gain"] > 1.0)
    assert np.all(kernels.regions["kernel_l1"] > 1.0)
    assert np.all(kernels.regions["kernel_edge_l1"] >= 0.0)
    assert np.all(kernels.regions["response_negative_flux"] > 0.0)
    assert np.all(kernels.regions["response_l2_fraction"] > 0.0)
    assert np.all(np.abs(kernels.regions["response_shift_x_pix"]) < 0.08)
    assert np.all(np.abs(kernels.regions["response_shift_y_pix"]) < 0.08)

    for key in (0, 1):
        source_shape = source.psfs[key] / source.psfs[key].sum()
        source_padded = pad_to_shape(source_shape, (64, 64))
        response = fftconvolve(source_padded, kernels.psfs[key], mode="same")
        source_width = np.mean(psf_core_fwhm(source.psfs[key]))
        response_width = np.mean(psf_core_fwhm(response))
        assert response_width < source_width
        row = kernels.regions.loc[kernels.regions.psf_key == key].iloc[0]
        assert response_width == pytest.approx(
            np.mean([row.response_fwhm_x_pix, row.response_fwhm_y_pix]),
            rel=2e-5,
        )

    # Numeric target provenance must survive the GeoJSON/FITS round trip as
    # numbers, not object-column strings.
    path = tmp_path / "wiener_kernel.geojson"
    kernels.to_file(path)
    loaded = PSFRegionMap.from_geojson(path, pscale=0.04)
    assert loaded.regions["kernel_reg"].dtype.kind == "f"
    assert loaded.regions["target_fwhm_pix"].dtype.kind == "f"
    assert np.all(loaded.regions["target_fwhm_pix"] == 2.5)
    assert np.allclose(loaded.psfs.sum(axis=(1, 2)), 1.0, atol=3e-7)


def test_matching_kernel_map_rejects_misaligned_cube():
    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
    )
    source = PSFRegionMap(
        regions,
        psfs=np.stack([PSF.gaussian(17, fwhm=4.0).array] * 2),
    )
    with pytest.raises(ValueError, match="one PSF plane per key"):
        source.gaussian_psf_map(2.5)


def test_gaussian_psf_map_validates_requested_shape():
    source = _source_map()
    with pytest.raises(ValueError, match="smaller than source"):
        source.gaussian_psf_map(2.5, shape=16)
    with pytest.raises(ValueError, match="positive"):
        source.gaussian_psf_map(0.0)
    with pytest.raises(ValueError, match="greater than one"):
        source.moffat_psf_map(2.5, beta=1.0)
    with pytest.raises(ValueError, match="greater than one"):
        source.moffat_psf_map(2.5, beta=np.nan)
    target = source.gaussian_psf_map(2.5)
    with pytest.raises(TypeError, match="reg"):
        source.matching_kernel_map(target)
    with pytest.raises(ValueError, match="positive"):
        source.matching_kernel_map(target, reg=0.0)
    with pytest.raises(ValueError, match="positive"):
        source.matching_kernel_map(target, reg=-1e-3)

    target.pscale = 0.08
    with pytest.raises(ValueError, match="pixel scales differ"):
        source.matching_kernel_map(target, reg=1e-3)


def test_aggressive_wiener_kernel_preserves_float32_unit_dc():
    source = _source_map()
    target = source.gaussian_psf_map(2.5, shape=64)
    kernels = source.matching_kernel_map(target, reg=1e-6)

    sums = np.sum(kernels.psfs, axis=(1, 2), dtype=np.float64)
    assert np.allclose(sums, 1.0, rtol=0.0, atol=2e-6)
