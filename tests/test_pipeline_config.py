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
    trial = {"center": [34.4, -5.26], "radius": 0.5}
    cfg = RunConfig.from_json(
        _write_config(tmp_path, {"psf_size": None, "trial": trial})
    )
    assert cfg.name == "test_run"
    assert cfg.psf_size is None
    assert cfg.trial_geometry() == ((34.4, -5.26), 0.5, 60.0)
    out = tmp_path / "echo.json"
    cfg.to_json(out)
    assert RunConfig.from_json(out) == cfg


def test_retired_trial_keys_raise(tmp_path):
    """r_trial/trial_center were folded into `trial`; say so, don't ignore."""
    for extra in ({"r_trial": 0.5}, {"trial_center": [34.4, -5.26]}):
        with pytest.raises(ValueError, match="replaced by a single `trial`"):
            RunConfig.from_json(_write_config(tmp_path, extra))


def test_trial_geometry_validates():
    def cfg(trial):
        return RunConfig(**{**MINIMAL, "trial": trial})

    assert cfg(None).trial_geometry() is None
    # radius 0 means a full-field run whatever the centre says
    assert cfg({"center": [1, 2], "radius": 0}).trial_geometry() is None
    with pytest.raises(ValueError, match="center"):
        cfg({"radius": 1.0}).trial_geometry()
    with pytest.raises(ValueError, match="unknown trial keys"):
        cfg({"center": [1, 2], "radius": 1.0, "radius_arcmin": 2}).trial_geometry()
    assert cfg(
        {"center": [1, 2], "radius": 1.5, "margin": 30.0}
    ).trial_geometry() == ((1.0, 2.0), 1.5, 30.0)


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


def test_residual_memmap_is_the_output_file(tmp_path):
    """The residual accumulator writes through to ``f_residual``.

    run() accumulates scene models straight into the output file's data
    section rather than into anonymous memory, so what write_outputs has left
    to do is flush. The file must be a valid FITS image with the detection
    band's header, and reading it back must give the accumulated pixels.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    hi = tmp_path / "hi.fits"
    w = WCS(naxis=2)
    w.wcs.crpix = [8.0, 6.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-1e-5, 1e-5]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    hdr["FILTER"] = "F770W"
    fits.writeto(hi, np.zeros((11, 13), dtype=np.float32), hdr)

    pipe = Pipeline.from_config(_write_config(tmp_path, {"sci_hi": str(hi)}))
    res = pipe._residual_memmap((11, 13))
    assert isinstance(res, np.memmap)
    assert res.shape == (11, 13)

    res[:] = 0.0
    res[3, 4] = 2.5
    res[10, 12] = -1.25
    res.flush()

    got, got_hdr = fits.getdata(pipe.f_residual, header=True)
    assert got.shape == (11, 13)
    assert got[3, 4] == 2.5 and got[10, 12] == -1.25
    assert got.sum() == 1.25
    # the detection band's WCS and provenance ride along
    assert got_hdr["FILTER"] == "F770W"
    assert WCS(got_hdr).wcs.crval[0] == 150.0


def test_residual_allocation_falls_back_without_a_config():
    """API-driven runs (no out_dir) keep the residual in memory."""
    import numpy as np

    from mophongo import pipeline as pl

    pipe = pl.Pipeline.__new__(pl.Pipeline)
    pipe.run_config = None
    img = np.zeros((5, 6), dtype=np.float32)
    res = pipe._allocate_residual(img, 1)
    assert not isinstance(res, np.memmap)
    assert res.shape == img.shape and res.dtype == img.dtype


def test_repair_patches_do_not_write_through_to_the_input(tmp_path):
    """Replaying a repair patch table leaves the input mosaics untouched.

    load_data applies the patches to fresh maps of sci_hi/segmap rather than
    holding the repaired mosaics, which is only safe because astropy maps a
    read-only HDU copy-on-write.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    from mophongo.pipeline import _apply_repair_patches, _read_image

    sci_path = tmp_path / "sci.fits"
    seg_path = tmp_path / "seg.fits"
    sci0 = np.arange(48, dtype=np.float32).reshape(6, 8)
    seg0 = np.zeros((6, 8), dtype=np.int32)
    fits.writeto(sci_path, sci0)
    fits.writeto(seg_path, seg0)

    patches = Table({
        "y": np.array([1, 4], np.int32), "x": np.array([2, 7], np.int32),
        "sci": np.array([-9.0, 3.5], np.float32),
        "wht": np.array([0.0, 0.0], np.float32),
        "seg": np.array([7, 7], np.int64),
    })
    sci = _read_image(sci_path)
    seg = _read_image(seg_path)
    _apply_repair_patches(patches, sci, seg)

    assert sci[1, 2] == -9.0 and sci[4, 7] == 3.5
    assert seg[1, 2] == 7 and seg[4, 7] == 7
    # everything else is the original
    untouched = np.ones((6, 8), bool)
    untouched[[1, 4], [2, 7]] = False
    assert np.array_equal(np.asarray(sci)[untouched], sci0[untouched])

    del sci, seg
    assert np.array_equal(fits.getdata(sci_path), sci0)
    assert np.array_equal(fits.getdata(seg_path), seg0)


def test_trial_pixel_box_and_partial_read(tmp_path):
    """Only the trial box is read, into a full-shape array.

    The point of the box is that a trial run costs the patch, not the mosaic,
    while every pixel coordinate keeps its full-frame meaning.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from mophongo.pipeline import _read_image, _trial_pixel_box

    ny, nx = 600, 800
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-1 / 3600.0, 1 / 3600.0]  # 1 arcsec / pixel
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    data = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data, header=w.to_header()).writeto(path)

    # 1 arcmin radius + 60 arcsec margin at 1 arcsec/pix -> 120 px half-width
    box = _trial_pixel_box(w, (ny, nx), (150.0, 2.0), 1.0, 60.0)
    y0, y1, x0, x1 = box
    # 2 * 120 px of half-width, +/- a pixel of rounding at the box edges
    assert abs((y1 - y0) - 241) <= 1 and abs((x1 - x0) - 241) <= 1
    assert y0 < ny / 2 < y1 and x0 < nx / 2 < x1

    full = _read_image(path)
    part = _read_image(path, box)

    assert part.shape == full.shape          # full-frame coordinates preserved
    assert np.array_equal(part[y0:y1, x0:x1], full[y0:y1, x0:x1])
    outside = np.ones((ny, nx), dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert not part[outside].any()           # nothing outside the box was read

    # a box clipped by the image edge still works
    edge = _trial_pixel_box(w, (ny, nx), (150.0, 2.0), 30.0, 0.0)
    assert edge == (0, ny, 0, nx)
