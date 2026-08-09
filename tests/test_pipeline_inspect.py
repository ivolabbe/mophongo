"""Tests for pre-run data loading and pipeline inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from mophongo.pipeline import Pipeline, STEPS


def _wcs_header(n: int, pscale_deg: float) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [34.0, -5.0]
    w.wcs.crpix = [n / 2, n / 2]
    w.wcs.cdelt = [-pscale_deg, pscale_deg]
    return w.to_header()


@pytest.fixture()
def tiny_config(tmp_path):
    """Write a minimal but fully loadable config: 64px hi, 32px lo, 1 source."""
    n_hi, n_lo = 64, 32
    scale_hi = 0.04 / 3600.0
    rng = np.random.default_rng(42)
    hi = rng.normal(0, 1e-3, (n_hi, n_hi)).astype(np.float32)
    lo = rng.normal(0, 1e-3, (n_lo, n_lo)).astype(np.float32)
    seg = np.zeros((n_hi, n_hi), dtype=np.int32)
    seg[30:34, 30:34] = 1
    hi[30:34, 30:34] += 1.0
    wht = np.ones((n_lo, n_lo), dtype=np.float32)

    fits.writeto(tmp_path / "hi.fits", hi, _wcs_header(n_hi, scale_hi))
    fits.writeto(tmp_path / "seg.fits", seg, _wcs_header(n_hi, scale_hi))
    fits.writeto(tmp_path / "lo.fits", lo, _wcs_header(n_lo, 2 * scale_hi))
    fits.writeto(tmp_path / "wht.fits", wht, _wcs_header(n_lo, 2 * scale_hi))
    fits.writeto(
        tmp_path / "hi_wht.fits",
        np.full((n_hi, n_hi), 1e6, dtype=np.float32),
        _wcs_header(n_hi, scale_hi),
    )
    Table(
        {"id": [1], "x": [32.0], "y": [32.0], "ra": [34.0], "dec": [-5.0]}
    ).write(tmp_path / "cat.fits")
    for csv in ("hi_wcs.csv", "lo_wcs.csv"):
        (tmp_path / csv).write_text("file,crval1\nexp1.fits,34.0\n")

    cfg = {
        "name": "tiny",
        "out_dir": str(tmp_path / "out"),
        "sci_hi": str(tmp_path / "hi.fits"),
        "wht_hi": str(tmp_path / "hi_wht.fits"),
        "segmap": str(tmp_path / "seg.fits"),
        "catalog": str(tmp_path / "cat.fits"),
        "sci_lo": str(tmp_path / "lo.fits"),
        "wht_lo": str(tmp_path / "wht.fits"),
        "csv_hi": str(tmp_path / "hi_wcs.csv"),
        "csv_lo": str(tmp_path / "lo_wcs.csv"),
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return path


def test_info_before_load(tiny_config):
    pipe = Pipeline.from_config(tiny_config)
    assert "configured" in repr(pipe)
    text = pipe.info()
    assert "not loaded" in text
    assert "64x64" in text  # sci_hi shape from the header, no pixel data read
    assert "1 rows" in text  # catalog row count from the table header
    assert "1 frames" in text
    assert "not built" in text  # no psf/kernel caches yet


def test_info_reports_missing_files(tmp_path):
    cfg = {
        "name": "missing",
        "out_dir": str(tmp_path / "out"),
        "sci_hi": str(tmp_path / "nope.fits"),
        "segmap": str(tmp_path / "nope.fits"),
        "catalog": str(tmp_path / "nope.fits"),
        "sci_lo": str(tmp_path / "nope.fits"),
        "wht_lo": str(tmp_path / "nope.fits"),
        "csv_hi": str(tmp_path / "nope.csv"),
        "csv_lo": str(tmp_path / "nope.csv"),
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    text = Pipeline.from_config(path).info()
    assert "MISSING" in text


def test_load_data_without_kernels(tiny_config):
    pipe = Pipeline.from_config(tiny_config).load_data(kernels=False)
    assert pipe.images[0].shape == (64, 64)
    assert pipe.images[1].shape == (32, 32)
    assert pipe.kernels[-1] is None
    assert pipe.weights[-1] is not None
    assert len(pipe.catalog) == 1
    assert "loaded" in repr(pipe)
    text = pipe.info()
    assert "image[0]" in text
    assert "40 mas/pix" in text
    assert "catalog   1 rows" in text


def test_detection_ivar_read_only_for_the_snr_weighted_schemes(tiny_config):
    """weights[0] is read only by the build schemes that weight data against a
    PSF model by SNR; a full-field hi-res weight map is as big as the mosaic."""
    cfg = json.loads(tiny_config.read_text())

    def _load(fit_kwargs):
        cfg["fit"] = fit_kwargs
        path = tiny_config.parent / "run_ivar.json"
        path.write_text(json.dumps(cfg))
        return Pipeline.from_config(path).load_data(kernels=False)

    assert _load({}).weights[0] is not None          # 'default' is a wings scheme
    assert _load({"extend_mode": "none"}).weights[0] is None
    assert _load({"extend_mode": "psf"}).weights[0] is None
    assert _load({"extend_mode": "default"}).weights[0] is not None
    assert _load({"extend_mode": "wren"}).weights[0] is not None
    assert _load({"extend_mode": "classic"}).weights[0] is not None


def test_wht_hi_is_derived_from_sci_hi_when_unset(tiny_config, tmp_path):
    """Standard grizli naming: <root>_sci.fits -> <root>_wht.fits."""
    cfg = json.loads(tiny_config.read_text())
    hdr = _wcs_header(64, 0.04 / 3600.0)
    fits.writeto(tmp_path / "drc_sci.fits", np.ones((64, 64), dtype=np.float32), hdr)
    fits.writeto(tmp_path / "drc_wht.fits", np.full((64, 64), 1e6, np.float32), hdr)
    cfg["sci_hi"] = str(tmp_path / "drc_sci.fits")
    del cfg["wht_hi"]
    path = tiny_config.parent / "run_derived.json"
    path.write_text(json.dumps(cfg))
    assert Pipeline.from_config(path).resolve_wht_hi() == tmp_path / "drc_wht.fits"


def test_run_without_a_detection_weight_map_is_refused(tiny_config):
    """A run with no wht_hi has no calibrated detection noise. Refuse it rather
    than degrade to one sky-sigma scalar for the whole mosaic."""
    cfg = json.loads(tiny_config.read_text())
    del cfg["wht_hi"]  # sci_hi is "hi.fits": no _sci.fits -> _wht.fits sibling
    path = tiny_config.parent / "run_nowht.json"
    path.write_text(json.dumps(cfg))

    pipe = Pipeline.from_config(path)
    with pytest.raises(FileNotFoundError, match="no detection-band weight map"):
        pipe.resolve_wht_hi()
    with pytest.raises(FileNotFoundError, match="no detection-band weight map"):
        pipe.load_data(kernels=False)
    assert "wht_hi   MISSING" in pipe.info()


def test_wht_hi_must_exist_when_set(tiny_config, tmp_path):
    cfg = json.loads(tiny_config.read_text())
    cfg["wht_hi"] = str(tmp_path / "nope.fits")
    path = tiny_config.parent / "run_badwht.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError, match="wht_hi does not exist"):
        Pipeline.from_config(path).resolve_wht_hi()


def test_load_data_default_ensures_maps(tiny_config, monkeypatch):
    pipe = Pipeline.from_config(tiny_config)
    called: list[int] = []
    monkeypatch.setattr(pipe, "_ensure_maps", lambda: called.append(1))
    pipe.load_data()
    assert called


def test_run_finishes_maps_after_preload(tiny_config, monkeypatch):
    """run() on a kernels=False preload must finish the maps before fitting."""

    class Stop(Exception):
        pass

    pipe = Pipeline.from_config(tiny_config).load_data(kernels=False)

    def fake_ensure():
        raise Stop()

    monkeypatch.setattr(pipe, "_ensure_maps", fake_ensure)
    with pytest.raises(Stop):
        pipe.run()


def test_plot_inputs(tiny_config, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipe = Pipeline.from_config(tiny_config).load_data(kernels=False)
    out = tmp_path / "inputs.png"
    fig, axes = pipe.plot_inputs(save=out)
    assert len(axes) == 4
    assert out.exists()
    plt.close(fig)


def test_plot_inputs_requires_load(tiny_config):
    pipe = Pipeline.from_config(tiny_config)
    with pytest.raises(RuntimeError, match="load_data"):
        pipe.plot_inputs()


def test_plain_pipeline_repr_and_info():
    img = np.zeros((8, 8))
    seg = np.zeros((8, 8), dtype=int)
    pipe = Pipeline([img, img], seg)
    assert "loaded" in repr(pipe)
    text = pipe.info()
    assert "image[0]" in text
    assert "segmap" in text


def test_cli_steps_include_inspection():
    assert STEPS["load"] == "load_data"
    assert STEPS["info"] == "info"


def test_run_saves_config_snapshot(tiny_config, monkeypatch):
    """run() must snapshot the fully-explicit config before fitting."""

    class Stop(Exception):
        pass

    pipe = Pipeline.from_config(tiny_config).load_data(kernels=False)
    monkeypatch.setattr(pipe, "_ensure_maps", lambda: None)

    def fake_save():
        raise Stop()

    monkeypatch.setattr(pipe, "save_config", fake_save)
    with pytest.raises(Stop):
        pipe.run()


def test_load_fit_offline_diagnostics(tiny_config, monkeypatch):
    """Fresh-session resume: load_outputs + load_fit + diagnose_subphot from files."""
    pipe = Pipeline.from_config(tiny_config)
    pipe.save_config()
    # fake run products: zero residual on the reference grid + fit/template tables
    Table(
        {"id": [1], "x": [32.0], "y": [32.0], "flux_1": [2.5], "err_1": [0.1]}
    ).write(pipe.f_fit_table)
    fits.writeto(pipe.f_residual, np.zeros((64, 64), dtype=np.float32))
    Table(
        rows=[(1, 1, 32.0, 32.0, 0.0, 0.0, 2.5, 0.1, 1)],
        names=["id", "id_parent", "x", "y", "dx", "dy", "flux", "err", "id_scene"],
    ).write(pipe.f_templates)

    fresh = Pipeline.from_config(pipe.out_dir)
    monkeypatch.setattr(fresh, "_ensure_maps", lambda: None)  # no PSF machinery
    fresh.load_fit()

    assert fresh.template_table is not None
    assert fresh.images[1].shape == (64, 64)  # upsampled onto the ref grid
    assert fresh.model_images[0].shape == (64, 64)
    img = fresh.diagnose_subphot(1, size=21)
    assert img.shape == (4 * 21, 6 * 21, 3)
    assert img.dtype == np.uint8
