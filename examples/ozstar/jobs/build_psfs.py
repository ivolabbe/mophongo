#!/usr/bin/env python
"""Pre-build every ePSF grid a config's fit will ask for.

Run on the login node, which is the only machine with both internet and the
module stack. Compute nodes cannot do this: ``PSFFactory`` builds MJD-tagged
grids and ``stpsf`` resolves each exposure date to a wavefront OPD by querying
MAST, and a compute node has neither DNS nor a route.

Because this is a *dedicated* step and not grid generation inside a fit, the
whole work list is known up front. The ``(pattern, csv)`` pairs are therefore
deduplicated across configs: a field's F444W set is built once however many
bands share it, rather than once per band, and since each pattern is handled
exactly once no two workers can write the same filename. The pipeline's
autobuild has to serialise a field's bands precisely because it cannot see the
whole list. That matters because the shared grids dominate -- F444W is 25 PSFs
per grid across every epoch, plus the 30" halo grids, against 9 per grid for a
band's own MIRI set.

Three grid families, and the last two are why this exists rather than a bare
``python -m mophongo.pipeline <cfg> psfs``:

* ``pattern_hi``  - the F444W photometry grids, shared by every band of a field;
* ``pattern_lo``  - the band's MIRI grids;
* the 30" halo grids, when ``repair_saturated`` is on. Saturation repair runs
  inside ``load_data``, not ``build_psfs``, so the ``psfs`` step never touches
  them - and the pipeline's own fallback catches ``FileNotFoundError`` and
  ``ValueError`` only, while a MAST lookup on a sealed node raises
  ``ConnectionError``. A fit would therefore die rather than degrade.

``PSFFactory`` is called directly, rather than through the pipeline's
autobuild, for two reasons. It takes ``date_mode`` explicitly, so the grids are
per-date whatever the deployed mophongo defaults to; and the autobuild only
fires when *nothing* matches the pattern, so a band left holding one grid from
an earlier single-date build would never gain the rest. The factory itself
skips files that already exist, so re-running is cheap and additive.

    python build_psfs.py [--date-mode all] <run>/<name>_ozstar.json [more ...]
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_psfs")


def halo_pattern(pattern_hi: str) -> str:
    """The 30" halo pattern derived from ``pattern_hi``.

    Mirrors ``Pipeline._repair_halo_pattern``: keep prefix/detector/filter,
    swap the sampling tail for the single-position 30" layout.
    """
    from mophongo.pipeline import _PSF_PATTERN_RE

    m = _PSF_PATTERN_RE.match(pattern_hi.strip())
    if m is None:
        return ""
    return (f"{m.group('prefix')}_{m.group('det')}_{m.group('filt')}"
            r"_MJD\d+_FOV30_GRID1_OS4")


def build_one_date(job: tuple[str, str, str, float | None, float]) -> int:
    """Build one epoch of one pattern. Runs in a worker process.

    A literal MJD as ``date_mode`` selects exactly that epoch; see
    ``psf_factory.dates_from_csv``. Module-level and taking a plain tuple so it
    survives pickling to a spawned worker.
    """
    from mophongo.pipeline import _psf_factory_kwargs
    from mophongo.psf_factory import PSFFactory

    pattern, csv, psf_dir, fov_default, mjd = job
    kw = _psf_factory_kwargs(pattern)          # same derivation the fit uses,
    fov = kw.pop("fov_arcsec", fov_default)    # so the filenames match
    before = len(list(Path(psf_dir).glob("*.fits")))
    PSFFactory(outdir=psf_dir, fov_arcsec=fov, date_mode=mjd,
               **kw).from_csv(csv, save=True)
    return len(list(Path(psf_dir).glob("*.fits"))) - before


def build_for_pattern(pattern: str, csv: str, psf_dir: str, date_mode: str,
                      fov_default: float | None, workers: int = 1) -> int:
    """Build every grid implied by ``pattern`` and the exposure list.

    The fan-out is over epochs and lives here rather than inside
    ``PSFFactory``: the run tree pulls mophongo from GitHub main, so a toolkit
    script cannot assume a parameter that has not landed there yet. Each epoch
    writes its own filenames, so the workers cannot collide.
    """
    from mophongo.psf_factory import dates_from_csv

    dates = dates_from_csv(csv, mode=date_mode)
    log.info("  %s: %d date(s) from %s", pattern, len(dates), Path(csv).name)
    jobs = [(pattern, csv, psf_dir, fov_default, float(m)) for m in dates]
    if workers <= 1 or len(jobs) == 1:
        return sum(build_one_date(j) for j in jobs)

    added = 0
    # spawn: stpsf and its OPD state do not survive a fork cleanly
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(build_one_date, j): j for j in jobs}
        for fut in as_completed(futures):
            mjd = futures[fut][4]
            try:
                added += fut.result()
            except Exception as exc:  # noqa: BLE001 - one epoch must not stop the rest
                log.error("  FAILED MJD%d: %s: %s", mjd, type(exc).__name__, exc)
    return added


def pattern_tasks(configs: list[Path]) -> list[tuple[str, str, str, float | None]]:
    """Every distinct ``(pattern, csv, psf_dir, fov)`` the configs imply.

    Deduplicated across configs, which is the whole point of doing this as a
    dedicated step rather than inside the fits. A field's ``pattern_hi`` and
    halo patterns are identical for all of its bands, so the F444W set is built
    once here instead of once per band -- and because each pattern is handled
    exactly once, by one process pool, no two workers can write the same
    filename. The pipeline's autobuild has to serialise a field's bands
    precisely because it cannot see the whole work list; this can.
    """
    from mophongo.pipeline import RunConfig

    tasks: list[tuple[str, str, str, float | None]] = []
    seen: set[tuple[str, str]] = set()
    for cfg_path in configs:
        cfg = RunConfig.from_json(str(cfg_path))
        psf_dir = str(cfg.psf_dir)
        Path(psf_dir).mkdir(parents=True, exist_ok=True)
        pairs = [(cfg.pattern_hi, str(cfg.csv_hi)), (cfg.pattern_lo, str(cfg.csv_lo))]
        if getattr(cfg, "repair_saturated", False):
            pat = cfg.repair_psf_pattern or halo_pattern(cfg.pattern_hi)
            if pat and pat != cfg.pattern_hi:
                pairs.append((pat, str(cfg.csv_hi)))
        for pattern, csv in pairs:
            if not pattern or (pattern, csv) in seen:
                continue
            seen.add((pattern, csv))
            tasks.append((pattern, csv, psf_dir, cfg.psf_fov_arcsec))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes per pattern; one (detector, date) grid is "
                         "one job, so this scales with cores. 0 = every core "
                         "this session may use. Note that on a login node that "
                         "is a cgroup cap (four here) and `nproc` under-reports "
                         "it, because the site sets OMP_NUM_THREADS=1")
    ap.add_argument("--date-mode", default="all",
                    help="one grid per unique integer MJD by default; see "
                         "psf_factory.dates_from_csv")
    args = ap.parse_args()

    # sched_getaffinity, not cpu_count: the cgroup cap is what we actually get.
    workers = args.workers or len(os.sched_getaffinity(0))

    tasks = pattern_tasks(args.configs)
    log.info("%d distinct pattern(s) from %d config(s), %d worker(s)",
             len(tasks), len(args.configs), workers)

    total = 0
    start_all = time.time()
    for i, (pattern, csv, psf_dir, fov) in enumerate(tasks, 1):
        start = time.time()
        log.info("=== [%d/%d] %s", i, len(tasks), pattern)
        try:
            added = build_for_pattern(pattern, csv, psf_dir, args.date_mode,
                                      fov, workers)
        except Exception as exc:  # noqa: BLE001 - one pattern must not stop the rest
            log.error("FAILED %s: %s: %s", pattern, type(exc).__name__, exc)
            continue
        total += added
        log.info("=== [%d/%d] %s done in %.1f min, +%d grid(s) (%d total, "
                 "%.1f min elapsed)", i, len(tasks), pattern,
                 (time.time() - start) / 60, added, total,
                 (time.time() - start_all) / 60)
    log.info("built %d grid(s) in %.1f min", total, (time.time() - start_all) / 60)
    log.info("PSF_BUILD_DONE")


if __name__ == "__main__":
    main()
