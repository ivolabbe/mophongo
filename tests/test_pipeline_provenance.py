"""Cache provenance, PSF-grid autobuild settings, and the run log."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from mophongo.pipeline import (
    Pipeline,
    _provenance,
    _provenance_matches,
    _psf_factory_kwargs,
    _stamp_provenance,
)
from mophongo.psf_map import PSFRegionMap


def _region_map(n_psf: int = 2) -> PSFRegionMap:
    prm = PSFRegionMap.from_footprints(
        {"a": box(0, 0, 1, 1), "b": box(0.5, 0.5, 1.5, 1.5)}
    )
    prm.psfs = np.ones((len(prm.regions), 8, 8)) / 64.0
    return prm


# --------------------------------------------------------------- patterns
@pytest.mark.parametrize(
    "pattern, expected",
    [
        (
            r"UDS_NRC.._F444W_MJD\d+_GRID25_OS4",
            dict(prefix="UDS", num_psfs=25, oversample=4,
                 use_detsampled_psf=False, include_mjd=True),
        ),
        (
            r"UDS_MIRI_F770W_MJD\d+_GRID9_OS4",
            dict(prefix="UDS", num_psfs=9, oversample=4,
                 use_detsampled_psf=False, include_mjd=True),
        ),
        (
            "COSMOS_NRCA5_F444W_GRID1_DET",
            dict(prefix="COSMOS", num_psfs=1, oversample=4,
                 use_detsampled_psf=True, include_mjd=False),
        ),
        (
            r"UDS_NRC.._F444W_MJD\d+_FOV30_GRID1_OS4",
            dict(prefix="UDS", num_psfs=1, oversample=4,
                 use_detsampled_psf=False, include_mjd=True,
                 fov_arcsec=30.0, include_fov=True),
        ),
    ],
)
def test_psf_factory_kwargs_recovers_generator_settings(pattern, expected):
    """The settings must come from the pattern the loader will search with."""
    assert _psf_factory_kwargs(pattern) == expected


def test_psf_factory_kwargs_rejects_unparseable_pattern():
    with pytest.raises(ValueError, match="cannot derive PSFFactory settings"):
        _psf_factory_kwargs("not-a-stdpsf-pattern")


def test_psf_factory_filename_fov_token_round_trips():
    """include_fov names parse back to the same generator settings."""
    from mophongo.psf_factory import PSFFactory

    fac = PSFFactory(prefix="UDS", num_psfs=1, oversample=4,
                     fov_arcsec=30.0, include_mjd=True, include_fov=True)
    name = fac.filename(detector="NRCA5", filt="F444W", mjd=59967.2)
    assert name == "UDS_NRCA5_F444W_MJD59967_FOV30_GRID1_OS4.fits"
    kw = _psf_factory_kwargs(name[:-5])
    assert kw["fov_arcsec"] == 30.0 and kw["include_fov"] is True
    # without include_fov the token is absent (legacy naming unchanged)
    fac_plain = PSFFactory(prefix="UDS", num_psfs=25, fov_arcsec=4.0,
                           include_mjd=True)
    assert fac_plain.filename(detector="NRCA5", filt="F444W", mjd=59967.2) \
        == "UDS_NRCA5_F444W_MJD59967_GRID25_OS4.fits"


def test_repair_halo_pattern_derivation():
    from mophongo.pipeline import Pipeline, RunConfig

    obj = Pipeline.__new__(Pipeline)
    obj.run_config = RunConfig(
        name="x", out_dir="x", sci_hi="a.fits", segmap="s.fits",
        catalog="c.fits", sci_lo="lo.fits", wht_lo="w.fits",
        csv_hi="h.csv", csv_lo="l.csv",
        pattern_hi=r"UDS_NRC.._F444W_MJD\d+_GRID25_OS4",
    )
    assert obj._repair_halo_pattern() == \
        r"UDS_NRC.._F444W_MJD\d+_FOV30_GRID1_OS4"
    obj.run_config.pattern_hi = "garbage"
    assert obj._repair_halo_pattern() == ""


# ------------------------------------------------------------- provenance
def test_provenance_round_trips_through_geojson(tmp_path):
    prm = _region_map()
    want = {"pattern": r"UDS_MIRI_F770W_MJD\d+_GRID9_OS4",
            "psf_size": 4.0, "blur_fwhm": 0.28}
    _stamp_provenance(prm, **want)
    path = tmp_path / "psf.geojson"
    prm.to_file(str(path))

    reloaded = PSFRegionMap.from_geojson(str(path))
    assert _provenance_matches(reloaded, want) is None
    assert _provenance(reloaded, "pattern") == want["pattern"]


@pytest.mark.parametrize("field, value", [
    ("psf_size", 8.0),
    ("blur_fwhm", 0.5),
    ("pattern", "SOMETHING_ELSE"),
])
def test_provenance_detects_each_changed_field(tmp_path, field, value):
    prm = _region_map()
    want = {"pattern": "P", "psf_size": 4.0, "blur_fwhm": 0.28}
    _stamp_provenance(prm, **want)
    path = tmp_path / "psf.geojson"
    prm.to_file(str(path))

    reloaded = PSFRegionMap.from_geojson(str(path))
    assert _provenance_matches(reloaded, {**want, field: value}) == field


def test_provenance_treats_an_unstamped_map_as_stale():
    """Maps written before provenance existed must not be reused silently."""
    prm = _region_map()
    assert _provenance_matches(prm, {"psf_size": 4.0}) == "psf_size"


# ---------------------------------------------------------------- run log
class _StubPipeline(Pipeline):
    """Only what log_run touches, so the log can be tested without data."""

    def __init__(self, out_dir: Path) -> None:  # noqa: D107
        self.out_dir = Path(out_dir)

        class _Cfg:
            name = "demo"

        self.run_config = _Cfg()


def test_log_run_captures_prints_and_log_records(tmp_path):
    pipe = _StubPipeline(tmp_path)
    with pipe.log_run() as path:
        print("a bare print")
        logging.getLogger("mophongo.demo").info("a log record")

    text = Path(path).read_text()
    assert path == tmp_path / "demo.log"
    assert "a bare print" in text
    assert "a log record" in text
    assert "mophongo run demo" in text
    assert "finished in" in text


def test_log_run_captures_other_libraries_and_warnings(tmp_path):
    """The file must hold every library's records, not just mophongo's.

    astropy, drizzlepac and stpsf log through their own loggers, and a caller
    that configured logging before the run holds a handler bound to the
    original stderr, so those records used to miss the file entirely.
    """
    import warnings

    logging.basicConfig(level=logging.INFO)  # as mophongo.pipeline.main does
    pipe = _StubPipeline(tmp_path)
    with pipe.log_run() as path:
        logging.getLogger("astropy").info("third party record")
        logging.getLogger("drizzlepac.haputils").warning("third party warning")
        # Call the installed hook rather than warnings.warn: pytest's warnings
        # plugin intercepts warn() and the warning would never be dispatched.
        # This is what captureWarnings(True) is expected to have replaced.
        warnings.showwarning("a python warning", UserWarning, "some_module.py", 1)

    text = Path(path).read_text()
    assert "third party record" in text
    assert "third party warning" in text
    assert "a python warning" in text


def test_log_run_captures_warnings_on_every_run(tmp_path):
    """Warning capture must survive a previous run in the same process.

    ``captureWarnings(True)`` is a no-op while logging still believes it is
    capturing, so a teardown that restores ``showwarning`` without clearing that
    flag leaves the second and later runs writing no warnings at all.
    """
    import warnings

    pipe = _StubPipeline(tmp_path)
    for run in ("first", "second"):
        with pipe.log_run() as path:
            warnings.showwarning(f"{run} warning", UserWarning, "some_module.py", 1)

    text = Path(path).read_text()
    assert "first warning" in text
    assert "second warning" in text


def test_log_run_does_not_duplicate_package_records(tmp_path):
    """Package records appear once, not once per handler."""
    pipe = _StubPipeline(tmp_path)
    with pipe.log_run() as path:
        logging.getLogger("mophongo.demo").info("say this once")

    assert Path(path).read_text().count("say this once") == 1


def test_log_run_leaves_logging_as_it_found_it(tmp_path):
    """Handlers, level and warning capture are all restored afterwards."""
    import warnings

    root = logging.getLogger()
    before = (list(root.handlers), root.level, warnings.showwarning)

    pipe = _StubPipeline(tmp_path)
    with pipe.log_run():
        pass

    assert list(root.handlers) == before[0]
    assert root.level == before[1]
    assert warnings.showwarning is before[2]


def test_log_run_records_a_failure_and_reraises(tmp_path):
    pipe = _StubPipeline(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with pipe.log_run():
            raise RuntimeError("boom")

    text = (tmp_path / "demo.log").read_text()
    assert "FAILED" in text and "RuntimeError: boom" in text


def test_log_run_restores_streams_and_appends(tmp_path):
    import sys

    pipe = _StubPipeline(tmp_path)
    before_out, before_err = sys.stdout, sys.stderr
    for _ in range(2):
        with pipe.log_run():
            print("run")
    assert (sys.stdout, sys.stderr) == (before_out, before_err)
    # appended, not overwritten
    assert (tmp_path / "demo.log").read_text().count("mophongo run demo") == 2


def test_psf_factory_parallel_defaults_and_job_uniqueness(tmp_path, monkeypatch):
    """One job per (detector, date), each writing its own filename.

    That uniqueness is what makes the build parallelisable at all: the jobs
    share nothing and never target the same file. Parallelism across
    *patterns* is a different matter -- every band of a field derives the same
    ``pattern_hi``, so two bands building at once do collide, which is why a
    campaign builds one band of a field first.
    """
    import types

    from mophongo import psf_factory as pf

    assert pf.PSFFactory().workers == 1, "serial by default; callers opt in"

    csv = tmp_path / "uds_f444w_wcs.csv"
    csv.write_text(
        "file,mjd-avg\njw01_nrca5.fits,60000.2\njw02_nrca5.fits,60005.7\n"
    )

    built = []

    def fake_build(**kw):
        built.append((kw["detector"], kw["date"]))
        return types.SimpleNamespace(
            data=np.zeros((1, 4, 4), dtype=np.float32),
            grid_xypos=np.array([[10, 10]]),
            meta={"detector": kw["detector"], "filter": kw["filter"]},
        )

    # JWSTBackend is a frozen dataclass whose build() calls this module
    # function, so patch there rather than on the instance
    import mophongo.jwst_psf as jp
    monkeypatch.setattr(jp, "build_jwst_psf", fake_build)

    out = tmp_path / "PSF"
    fac = pf.PSFFactory(prefix="UDS", outdir=str(out), num_psfs=1, oversample=4)
    fac.from_csv(csv, date_mode="all", save=True)

    # F444W is NIRCam LW, so both long-wave detectors are built: the job
    # count is detectors x unique integer MJDs, each in its own file
    assert sorted(built) == [("NRCA5", 60000.0), ("NRCA5", 60006.0),
                             ("NRCB5", 60000.0), ("NRCB5", 60006.0)]
    assert len(set(built)) == len(built)
    assert len(sorted(out.glob("*.fits"))) == 4

    # and a second run skips what already exists rather than rebuilding it
    built.clear()
    fac.from_csv(csv, date_mode="all", save=True)
    assert built == []


def test_psf_workers_and_provenance_reach_the_factory(tmp_path, monkeypatch):
    """``RunConfig.psf_workers``/``psf_provenance`` are wired, not decorative."""
    from mophongo.pipeline import RunConfig

    assert RunConfig.psf_workers == 1
    assert RunConfig.psf_provenance == "warn"
