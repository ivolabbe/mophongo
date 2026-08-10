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
    ],
)
def test_psf_factory_kwargs_recovers_generator_settings(pattern, expected):
    """The settings must come from the pattern the loader will search with."""
    assert _psf_factory_kwargs(pattern) == expected


def test_psf_factory_kwargs_rejects_unparseable_pattern():
    with pytest.raises(ValueError, match="cannot derive PSFFactory settings"):
        _psf_factory_kwargs("not-a-stdpsf-pattern")


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
