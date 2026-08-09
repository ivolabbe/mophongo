"""Telescope-agnostic PSF factory.

High-level entry point for generating model PSFs that match a given dataset.
The factory holds reusable configuration (grid size, oversampling, output
directory, ...) and dispatches the actual PSF construction to a per-telescope
backend selected from :data:`BACKENDS`.

Adding a new telescope (HST, Roman, ...) only requires writing a new backend
module exposing the same small interface as :class:`mophongo.jwst_psf.JWSTBackend`
and registering it here -- no changes to existing JWST code.

Backend protocol
----------------
Each backend object exposes:

* ``name``                 -- short telescope name (``"JWST"``, ``"HST"``, ...)
* ``detect(filename)``     -- bool: is this filename from this telescope?
* ``decode_filename(name)``-- ``(instrument, detector)`` from a rate-file name
* ``filter_from_path(p)``  -- ``"F444W"`` etc. from a path
* ``detectors_for_filter(instrument, filter)`` -- list of detectors that see
  the filter (empty list = use detector decoded from CSV)
* ``build(**kw) -> GriddedPSFModel`` -- the actual PSF builder

Example
-------
>>> from mophongo.psf_factory import PSFFactory
>>> fac = PSFFactory(prefix="UDS", outdir="data/PSF", fov_arcsec=8.0)
>>> fac.from_csv("data/uds-f444w_wcs.csv", num_psfs=9, save=True)
>>> # explicit single PSF
>>> grid = fac.build(telescope="JWST", instrument="NIRCAM",
...                  detector="NRCA5", filter="F444W", date=60000.0)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from astropy.time import Time

from .jwst_psf import jwst_backend, write_stdpsf

logger = logging.getLogger(__name__)

__all__ = ["PSFFactory", "dates_from_csv", "BACKENDS"]


# ──────────────────────────────────────────────────────────────────────────
# Backend protocol + registry
# ──────────────────────────────────────────────────────────────────────────
class PSFBackend(Protocol):
    name: str

    def detect(self, filename: str) -> bool: ...
    def decode_filename(self, name: str) -> tuple[str, str]: ...
    def filter_from_path(self, path: str | os.PathLike) -> str: ...
    def detectors_for_filter(self, instrument: str, filt: str) -> list[str]: ...
    def build(self, **kwargs: Any): ...


BACKENDS: dict[str, PSFBackend] = {jwst_backend.name: jwst_backend}
# Future: BACKENDS["HST"] = hst_backend; BACKENDS["ROMAN"] = roman_backend


def _detect_telescope(filename: str) -> str:
    for b in BACKENDS.values():
        if b.detect(filename):
            return b.name
    raise ValueError(f"No registered backend recognises filename {filename!r}")


# ──────────────────────────────────────────────────────────────────────────
# Date selection
# ──────────────────────────────────────────────────────────────────────────
def _modal_mjd(mjd: np.ndarray, span: float = 5.0) -> float:
    """Centre of the densest ``span``-day window in ``mjd``."""
    arr = np.sort(np.asarray(mjd, dtype=float))
    best_cnt, best_i = 0, 0
    for i, v in enumerate(arr):
        cnt = np.searchsorted(arr, v + span) - i
        if cnt > best_cnt:
            best_cnt, best_i = cnt, i
    return float(arr[best_i] + span / 2)


def _cluster_mjds(mjd: np.ndarray, delta_day: float = 2.0) -> list[float]:
    """Minimum-cardinality 1-D interval cover with radius ``delta_day``.

    Greedy left-to-right: anchor each cluster at the leftmost uncovered
    frame, absorb every frame within ``2*delta_day`` of it, then advance.
    Returned anchor is the midpoint ``(min+max)/2`` of each cluster, which
    is L∞-optimal and always satisfies the ``delta_day`` radius constraint.
    """
    arr = np.sort(np.asarray(mjd, dtype=float))
    centres: list[float] = []
    i, n = 0, len(arr)
    while i < n:
        lo = arr[i]
        j = int(np.searchsorted(arr, lo + 2 * delta_day, side="right"))
        members = arr[i:j]
        centres.append(float((members[0] + members[-1]) / 2.0))
        i = j
    return centres


def dates_from_csv(
    csv_path: str | os.PathLike,
    mode: str | float | Time = "modal",
    *,
    span: float = 5.0,
    delta_day: float = 2.0,
    column: str = "mjd-avg",
) -> list[float]:
    """Return list of MJDs from a CSV rate-file listing.

    Parameters
    ----------
    csv_path
        Path to a ``*_wcs.csv`` file with a ``mjd-avg`` column.
    mode
        One of:

        * ``'modal'``  -- centre of densest ``span``-day window (1 date)
        * ``'median'`` / ``'mean'`` -- summary statistic (1 date)
        * ``'cluster'`` -- minimum number of dates such that every frame is
          within ``delta_day`` of some date (greedy 1-D interval cover)
        * ``'all'``    -- one date per unique integer MJD
        * a literal MJD (float), ISO string, or :class:`astropy.time.Time`
    span
        Window width in days used by ``'modal'``.
    delta_day
        Cluster radius in days used by ``'cluster'``. Every frame is
        guaranteed within ``delta_day`` of its assigned PSF date.
    column
        CSV column to read MJDs from.
    """
    import pandas as pd

    if isinstance(mode, Time):
        return [float(mode.mjd)]
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return [float(mode)]
    if isinstance(mode, str) and mode not in {"modal", "median", "mean", "all", "cluster"}:
        return [float(Time(mode).mjd)]

    tab = pd.read_csv(csv_path)
    if column not in tab.columns:
        raise ValueError(f"CSV {csv_path} missing column '{column}'")
    mjds = np.asarray(tab[column], dtype=float)

    if mode == "modal":
        return [_modal_mjd(mjds, span=span)]
    if mode == "median":
        return [float(np.median(mjds))]
    if mode == "mean":
        return [float(np.mean(mjds))]
    if mode == "cluster":
        return _cluster_mjds(mjds, delta_day=delta_day)
    if mode == "all":
        return sorted({float(int(round(m))) for m in mjds})
    raise ValueError(f"Unknown date mode: {mode!r}")


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PSFFactory:
    """Configure once, produce many PSFs.

    Holds reusable defaults; per-call overrides allowed on :meth:`build` and
    :meth:`from_csv`. Telescope dispatch is automatic when a CSV is supplied;
    for :meth:`build`, the ``telescope`` kwarg picks the backend.

    Attributes
    ----------
    prefix
        Filename prefix (e.g. project tag ``'UDS'``).
    outdir
        Output directory for saved FITS files. Created on demand.
    num_psfs, oversample, fov_arcsec, use_detsampled_psf
        Default PSF build parameters.
    include_mjd
        If True (default), embed the integer MJD in saved filenames as
        ``..._MJD{int}.fits``. Set False to reproduce legacy filenames that
        omitted the MJD token.
    overwrite, verbose
        Standard knobs.
    """

    prefix: str = "STDPSF"
    outdir: str | os.PathLike | None = None
    num_psfs: int = 1
    oversample: int = 4
    fov_arcsec: float | None = None
    use_detsampled_psf: bool = False
    date_mode: str | float | Time = "modal"
    span: float = 5.0
    delta_day: float = 2.0
    include_mjd: bool = True
    overwrite: bool = False
    verbose: bool = False

    # ── filename helper ────────────────────────────────────────────────
    def filename(
        self,
        *,
        detector: str,
        filt: str,
        num_psfs: int | None = None,
        use_detsampled_psf: bool | None = None,
        mjd: float | None = None,
    ) -> str:
        """Build canonical PSF filename.

        Order: ``{prefix}_{DET}_{FILT}[_MJD{int}]_GRID{N}_{OS4|DET}.fits``.
        The MJD slot sits between the physical-identity prefix
        (project/detector/filter/epoch) and the sampling tail
        (grid layout / oversampling factor). Stripping ``_MJD\\d+`` yields
        the canonical sampling key used by :class:`DrizzlePSF` for
        nearest-MJD lookup.
        """
        n = num_psfs if num_psfs is not None else self.num_psfs
        det_samp = use_detsampled_psf if use_detsampled_psf is not None else self.use_detsampled_psf
        sampling = "DET" if det_samp else "OS4"
        parts = [self.prefix, detector, filt]
        if self.include_mjd and mjd is not None:
            parts.append(f"MJD{int(round(mjd))}")
        parts += [f"GRID{n}", sampling]
        return "_".join(p for p in parts if p) + ".fits"

    # ── low-level: one explicit PSF ────────────────────────────────────
    def build(
        self,
        *,
        telescope: str,
        instrument: str,
        filter: str,
        detector: str | None = None,
        date: float | str | Time | None = None,
        num_psfs: int | None = None,
        oversample: int | None = None,
        fov_arcsec: float | None = None,
        use_detsampled_psf: bool | None = None,
        save: bool = False,
        **backend_kw: Any,
    ):
        """Build one PSF / PSF grid. Returns the grid object; optionally writes it."""
        backend = BACKENDS.get(telescope.upper())
        if backend is None:
            raise ValueError(f"Unknown telescope {telescope!r}; known: {list(BACKENDS)}")

        n = num_psfs if num_psfs is not None else self.num_psfs
        os_ = oversample if oversample is not None else self.oversample
        fov = fov_arcsec if fov_arcsec is not None else self.fov_arcsec
        det_samp = use_detsampled_psf if use_detsampled_psf is not None else self.use_detsampled_psf

        grid = backend.build(
            instrument=instrument,
            filter=filter,
            detector=detector,
            date=date,
            num_psfs=n,
            oversample=os_,
            fov_arcsec=fov,
            use_detsampled_psf=det_samp,
            verbose=self.verbose,
            **backend_kw,
        )

        if save:
            mjd = float(date.mjd) if isinstance(date, Time) else (float(date) if date is not None else None)
            self._save(grid, detector=detector or instrument, filt=filter,
                       num_psfs=n, use_detsampled_psf=det_samp, mjd=mjd)
        return grid

    # ── high-level: drive everything from a CSV listing ───────────────
    def from_csv(
        self,
        csv_path: str | os.PathLike,
        *,
        detector: str | Sequence[str] | None = None,
        date_mode: str | float | Time | Iterable | None = None,
        span: float | None = None,
        delta_day: float | None = None,
        num_psfs: int | None = None,
        oversample: int | None = None,
        fov_arcsec: float | None = None,
        use_detsampled_psf: bool | None = None,
        save: bool = True,
    ) -> list:
        """Build PSFs matching a grizli ``*_wcs.csv`` rate-file listing.

        One PSF file is produced per ``(detector, date)`` pair. The
        instrument, filter and detector list are inferred from the CSV.

        Parameters
        ----------
        csv_path
            Path to a CSV containing at least ``file`` and ``mjd-avg`` columns.
        detector
            Override the detector list (string or sequence). Default = auto.
        date_mode
            See :func:`dates_from_csv`. May also be an iterable of modes /
            literals to combine (e.g. ``("modal", "median")``). Default =
            factory's ``date_mode`` (``'modal'``).
        span
            Window width (days) for ``'modal'``. Default = factory's ``span``.
        delta_day
            Cluster radius (days) for ``'cluster'``. Default = factory's
            ``delta_day`` (2.0). One PSF is generated per cluster.
        num_psfs, oversample, fov_arcsec, use_detsampled_psf
            Per-call overrides of the factory defaults.
        save
            If True, write each grid to ``outdir`` using
            :meth:`filename`.
        """
        import pandas as pd

        tab = pd.read_csv(csv_path)
        if "file" not in tab.columns:
            raise ValueError(f"CSV {csv_path} missing column 'file'")

        telescope = _detect_telescope(tab["file"].iloc[0])
        backend = BACKENDS[telescope]
        instrument, det_from_file = backend.decode_filename(tab["file"].iloc[0])
        filt = backend.filter_from_path(csv_path)

        # Detector list
        if detector is None:
            det_list = backend.detectors_for_filter(instrument, filt) or [det_from_file]
        elif isinstance(detector, str):
            det_list = [detector.upper()]
        else:
            det_list = [d.upper() for d in detector]

        # Date selection (fall back to factory defaults)
        mode = date_mode if date_mode is not None else self.date_mode
        sp = span if span is not None else self.span
        dd = delta_day if delta_day is not None else self.delta_day
        if isinstance(mode, (list, tuple, np.ndarray)) and not isinstance(mode, str):
            mjds: list[float] = []
            for d in mode:
                mjds.extend(dates_from_csv(csv_path, d, span=sp, delta_day=dd))
        else:
            mjds = dates_from_csv(csv_path, mode, span=sp, delta_day=dd)

        n = num_psfs if num_psfs is not None else self.num_psfs
        os_ = oversample if oversample is not None else self.oversample
        fov = fov_arcsec if fov_arcsec is not None else self.fov_arcsec
        det_samp = use_detsampled_psf if use_detsampled_psf is not None else self.use_detsampled_psf

        if self.verbose:
            logger.info(
                "%s filter=%s detectors=%s dates(MJD)=%s",
                telescope, filt, ",".join(det_list),
                ",".join(f"{m:.1f}" for m in mjds),
            )

        grids = []
        for det in det_list:
            for mjd in mjds:
                outpath = self._outpath(detector=det, filt=filt, num_psfs=n,
                                        use_detsampled_psf=det_samp, mjd=mjd)
                if save and outpath is not None and outpath.exists() and not self.overwrite:
                    logger.info("Skipping existing %s", outpath)
                    continue

                grid = backend.build(
                    instrument=instrument,
                    filter=filt,
                    detector=det,
                    date=float(mjd),
                    num_psfs=n,
                    oversample=os_,
                    fov_arcsec=fov,
                    use_detsampled_psf=det_samp,
                    verbose=self.verbose,
                )
                grids.append(grid)
                if save:
                    self._save(grid, detector=det, filt=filt, num_psfs=n,
                               use_detsampled_psf=det_samp, mjd=mjd)
        return grids

    # ── internals ──────────────────────────────────────────────────────
    def _outpath(self, **kw) -> Path | None:
        if self.outdir is None:
            return None
        return Path(self.outdir) / self.filename(**kw)

    def _save(self, grid, **fname_kw) -> Path | None:
        outpath = self._outpath(**fname_kw)
        if outpath is None:
            return None
        outpath.parent.mkdir(parents=True, exist_ok=True)
        write_stdpsf(outpath, grid, overwrite=self.overwrite, verbose=self.verbose)
        return outpath
