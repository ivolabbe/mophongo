"""Tests for the config-driven Pipeline runs (Pipeline.from_config)."""

from __future__ import annotations

import json

import pytest

from mophongo.pipeline import Pipeline, RunConfig, STEPS

MINIMAL = {
    "name": "test_run",
    "out_dir": "out",
    "sci_hi": "hi.fits",
    "segmap": "seg.fits",
    "catalog": "cat.fits",
    "sci_lo": "lo.fits",
    "wht_lo": "wht.fits",
    "csv_hi": "hi_wcs.csv",
    "csv_lo": "lo_wcs.csv",
}


def _write_config(tmp_path, extra=None):
    data = dict(MINIMAL)
    data["out_dir"] = str(tmp_path / "out")
    if extra:
        data.update(extra)
    path = tmp_path / "run.json"
    path.write_text("# comment line\n" + json.dumps(data))
    return path


def test_config_roundtrip_with_comments(tmp_path):
    cfg = RunConfig.from_json(_write_config(tmp_path, {"psf_size": None, "r_trial": 0.5}))
    assert cfg.name == "test_run"
    assert cfg.psf_size is None
    assert cfg.r_trial == 0.5
    out = tmp_path / "echo.json"
    cfg.to_json(out)
    assert RunConfig.from_json(out) == cfg


def test_config_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown config keys.*psf_sise"):
        RunConfig.from_json(_write_config(tmp_path, {"psf_sise": 2.0}))


def test_blur_resolution_modes(tmp_path):
    pipe = Pipeline.from_config(_write_config(tmp_path, {"filter_lo": "f770w"}))
    from mophongo.mock_mosaic import DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC

    assert pipe._blur_fwhm() == DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC["f770w"]

    pipe.run_config.psf_blur_fwhm = 0.1
    assert pipe._blur_fwhm() == 0.1
    pipe.run_config.psf_blur_fwhm = None
    assert pipe._blur_fwhm() is None
    # unknown filter with "default" -> no broadening
    pipe.run_config.psf_blur_fwhm = "default"
    pipe.run_config.filter_lo = "f9999w"
    assert pipe._blur_fwhm() is None


def test_cache_paths_and_step_registry(tmp_path):
    pipe = Pipeline.from_config(_write_config(tmp_path))
    assert pipe.out_dir.exists()
    assert pipe.f_psf_hi.name == "test_run_psf_hi.geojson"
    assert pipe.f_kernel.parent == pipe.out_dir
    for step, method in STEPS.items():
        assert callable(getattr(pipe, method)), (step, method)


def test_run_all_calls_steps_in_order(tmp_path, monkeypatch):
    pipe = Pipeline.from_config(_write_config(tmp_path))
    calls: list[str] = []
    for name in ("build_psfs", "build_kernels", "run", "write_outputs"):
        monkeypatch.setattr(pipe, name, lambda name=name, **kw: calls.append(name))
    pipe.run_all()
    assert calls == ["build_psfs", "build_kernels", "run", "write_outputs"]


def test_size_kw_native_vs_arcsec(tmp_path):
    pipe = Pipeline.from_config(_write_config(tmp_path, {"psf_size": 4.0}))
    assert pipe._size_kw() == {"size": 4.0}
    pipe.run_config.psf_size = None
    assert pipe._size_kw() == {"size": None, "ee_fraction": None}


def test_normal_construction_unaffected():
    """Plain Pipeline(...) construction keeps run_config=None and no deferred state."""
    import numpy as np

    img = np.zeros((8, 8))
    seg = np.zeros((8, 8), dtype=int)
    pipe = Pipeline([img, img], seg)
    assert pipe.run_config is None
