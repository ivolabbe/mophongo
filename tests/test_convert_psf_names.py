"""The standalone PSF converter must agree with mophongo's own provenance.

`examples/canfar/jobs/convert_psf_names.py` deliberately imports nothing from
mophongo: a run's venv now lives inside that run and is built by setup_env.sh,
so on a fresh tree there is nothing to import from when the grids need
converting. The cost of that independence is a second copy of the provenance
rule, and the only thing keeping the copy honest is this test.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mophongo.jwst_psf import read_stdpsf_provenance, write_stdpsf
from mophongo.psf_factory import grid_provenance

SCRIPT = Path(__file__).parent.parent / "examples/canfar/jobs/convert_psf_names.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_psf_names", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("date_mode,fov", [
    ("all", None), ("all", 30.0), ("cluster", 4.0), ("modal", 8.1),
])
def test_standalone_provenance_matches_grid_provenance(tmp_path, date_mode, fov):
    """Same cards, value for value, so a converted grid is not seen as stale."""
    csv = tmp_path / "uds_f444w_wcs.csv"
    csv.write_text("file,mjd-avg\njw01.fits,60000.0\njw02.fits,60011.5\n")

    mine = _load().provenance(csv, date_mode, fov)
    theirs = grid_provenance(csv, date_mode, fov)
    assert mine == theirs


def test_converter_renames_and_stamps_so_the_grid_reads_as_current(tmp_path):
    """End to end: rename carries the header FOV, cards read back intact."""
    mod = _load()
    psf_dir = tmp_path / "PSF"
    psf_dir.mkdir()
    csv = tmp_path / "uds_f444w_wcs.csv"
    csv.write_text("file,mjd-avg\njw01.fits,60000.0\n")

    old = psf_dir / "UDS_NRCA5_F444W_MJD60000_GRID25_OS4.fits"
    write_stdpsf(old, np.zeros((1, 5, 5), dtype=np.float32),
                 xgrid=np.array([10]), ygrid=np.array([10]),
                 detector="NRCA5", filt="F444W", overwrite=True)
    # stpsf records the field of view it built at; the name is taken from it
    with __import__("astropy.io.fits", fromlist=["fits"]).open(old, mode="update") as hdul:
        hdul[0].header["FOV"] = 4.0889634

    assert mod.main([str(psf_dir), "--csv", str(csv), "--apply"]) == 0

    new = psf_dir / "UDS_NRCA5_F444W_MJD60000_FOV4_GRID25_OS4.fits"
    assert new.exists() and not old.exists()
    assert read_stdpsf_provenance(new) == grid_provenance(csv, "all", None)


def test_converter_without_csv_renames_but_does_not_stamp(tmp_path):
    """Renaming alone leaves a run still treating the grid as stale."""
    mod = _load()
    psf_dir = tmp_path / "PSF"
    psf_dir.mkdir()
    old = psf_dir / "UDS_MIRI_F770W_MJD60000_GRID9_OS4.fits"
    write_stdpsf(old, np.zeros((1, 5, 5), dtype=np.float32),
                 xgrid=np.array([10]), ygrid=np.array([10]),
                 detector="MIRIM", filt="F770W", overwrite=True)
    with __import__("astropy.io.fits", fromlist=["fits"]).open(old, mode="update") as hdul:
        hdul[0].header["FOV"] = 8.0969428

    assert mod.main([str(psf_dir), "--apply"]) == 0
    new = psf_dir / "UDS_MIRI_F770W_MJD60000_FOV8_GRID9_OS4.fits"
    assert new.exists()
    assert read_stdpsf_provenance(new) == {}
