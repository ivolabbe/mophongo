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
    Table(
        {"id": [1], "x": [32.0], "y": [32.0], "ra": [34.0], "dec": [-5.0]}
    ).write(tmp_path / "cat.fits")
    for csv in ("hi_wcs.csv", "lo_wcs.csv"):
        (tmp_path / csv).write_text("file,crval1\nexp1.fits,34.0\n")

    cfg = {
        "name": "tiny",
        "out_dir": str(tmp_path / "out"),
        "sci_hi": str(tmp_path / "hi.fits"),
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


def test_detection_ivar_loaded_only_for_the_snr_weighted_schemes(tiny_config):
    """weights[0] is read only by the build schemes that weight data against a
    PSF model by SNR; a full-field hi-res weight map is as big as the mosaic."""
    cfg = json.loads(tiny_config.read_text())
    # standard grizli naming, so wht_hi resolves without being set
    hi = Path(cfg["sci_hi"])
    wht_hi = hi.with_name(hi.name.replace(".fits", "_sci.fits"))
    fits.writeto(wht_hi, np.ones((64, 64), dtype=np.float32), _wcs_header(64, 0.04 / 3600.0))
    fits.writeto(
        str(wht_hi).replace("_sci.fits", "_wht.fits"),
        np.full((64, 64), 1e6, dtype=np.float32),
        _wcs_header(64, 0.04 / 3600.0),
    )
    cfg["sci_hi"] = str(wht_hi)

    def _load(fit_kwargs):
        cfg["fit"] = fit_kwargs
        path = tiny_config.parent / "run_ivar.json"
        path.write_text(json.dumps(cfg))
        return Pipeline.from_config(path).load_data(kernels=False)

    assert _load({}).weights[0] is None
    assert _load({"extend_mode": "psf"}).weights[0] is None
    assert _load({"extend_mode": "wren"}).weights[0] is not None
    assert _load({"extend_mode": "classic"}).weights[0] is not None


def test_detection_ivar_warns_when_no_hi_weight_exists(tiny_config, caplog):
    cfg = json.loads(tiny_config.read_text())
    cfg["fit"] = {"extend_mode": "wren"}  # sci_hi is "hi.fits": no _sci -> _wht sibling
    path = tiny_config.parent / "run_nowht.json"
    path.write_text(json.dumps(cfg))
    with caplog.at_level("WARNING"):
        pipe = Pipeline.from_config(path).load_data(kernels=False)
    assert pipe.weights[0] is None
    assert "scalar sky sigma" in caplog.text


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
