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


def test_from_config_accepts_directory(tmp_path):
    """A directory holding exactly one config JSON resolves to that config,
    so a finished run reopens with from_config(out_dir)."""
    cfg_path = _write_config(tmp_path)
    run_dir = tmp_path / "finished"
    run_dir.mkdir()
    (run_dir / "test_run.json").write_text(cfg_path.read_text())

    pipe = Pipeline.from_config(run_dir)
    assert pipe.run_config.name == "test_run"

    # zero or several JSONs is ambiguous and must fail loudly
    with pytest.raises(FileNotFoundError, match="none"):
        Pipeline.from_config(tmp_path / "out")
    (run_dir / "second.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="second.json"):
        Pipeline.from_config(run_dir)


def test_save_config_writes_full_snapshot(tmp_path):
    from dataclasses import fields as dc_fields

    from mophongo.fit import FitConfig

    pipe = Pipeline.from_config(
        _write_config(tmp_path, {"fit": {"scene_minimum_bright": 3}})
    )
    out = pipe.save_config()
    assert out == pipe.f_config == pipe.out_dir / "test_run.json"

    # every RunConfig field explicit in the snapshot
    clean = "\n".join(
        ln for ln in out.read_text().splitlines() if not ln.lstrip().startswith("#")
    )
    data = json.loads(clean)
    assert set(data) == {f.name for f in dc_fields(RunConfig)}

    # every *used* FitConfig setting explicit, overrides preserved, and
    # round-trippable; unused scheme settings (wren_*/classic_*) are omitted
    cfg2 = RunConfig.from_json(out)
    unused = {
        f.name
        for f in dc_fields(FitConfig)
        if f.name.startswith(("wren_", "classic_"))
    }
    assert unused  # guard: the pruning groups exist in FitConfig
    assert set(cfg2.fit) == {f.name for f in dc_fields(FitConfig)} - unused
    assert cfg2.fit["scene_minimum_bright"] == 3
    FitConfig(**cfg2.fit)


def test_save_config_keeps_selected_scheme_settings(tmp_path):
    pipe = Pipeline.from_config(
        _write_config(tmp_path, {"fit": {"extend_mode": "wren"}})
    )
    fit = RunConfig.from_json(pipe.save_config()).fit
    assert fit["extend_mode"] == "wren"
    assert any(k.startswith("wren_") for k in fit)
    assert not any(k.startswith("classic_") for k in fit)

    pipe.run_config.fit = {"extend_mode": "classic"}
    fit = RunConfig.from_json(pipe.save_config()).fit
    assert any(k.startswith("classic_") for k in fit)
    assert not any(k.startswith("wren_") for k in fit)


def test_from_config_accepts_run_directory(tmp_path):
    pipe = Pipeline.from_config(_write_config(tmp_path))
    snap = pipe.save_config()

    # directory with a single json
    pipe2 = Pipeline.from_config(pipe.out_dir)
    assert pipe2.run_config.name == "test_run"

    # <dir>/<dirname>.json wins over other jsons
    d = tmp_path / "myrun"
    d.mkdir()
    (d / "myrun.json").write_text(snap.read_text())
    (d / "zz.json").write_text("not json")
    assert Pipeline.from_config(d).run_config.name == "test_run"

    # ambiguous directory raises
    (d / "myrun.json").rename(d / "aa.json")
    with pytest.raises(FileNotFoundError, match="aa.json"):
        Pipeline.from_config(d)


def test_load_outputs_resume(tmp_path):
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    pipe = Pipeline.from_config(_write_config(tmp_path))
    pipe.save_config()  # run() writes this snapshot; needed to resume from the dir
    Table({"id": [1, 2], "flux_1": [1.0, 2.0]}).write(pipe.f_fit_table)
    fits.writeto(pipe.f_residual, np.zeros((4, 4), dtype=np.float32))

    fresh = Pipeline.from_config(pipe.out_dir).load_outputs()
    assert len(fresh.table) == 2
    assert fresh.residuals[0].shape == (4, 4)
    assert "fitted" in repr(fresh)
    assert "results: table 2 rows" in fresh.info()
