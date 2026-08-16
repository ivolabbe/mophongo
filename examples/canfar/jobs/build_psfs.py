#!/usr/bin/env python
"""Pre-build every ePSF grid a campaign's configs will ask for.

Identical in ``examples/canfar/jobs`` and ``examples/ozstar/jobs``: finding and
building the grids is the same problem on both platforms, and only the wrapper
differs. OzStar must run it on the login node, the one machine with both
internet and the module stack -- ``stpsf`` resolves each epoch's wavefront by
querying MAST, and a compute node has neither DNS nor a route. CANFAR compute
has outbound internet, so there it is an ordinary container job and can be
sharded over several.

The unit of work is one ``(pattern, epoch)``: one grid, one filename. Because
this is a dedicated step rather than grid generation inside a fit, the whole
list is known up front, so it is deduplicated across configs -- a field's F444W
set is enumerated once however many bands share it -- and every entry has a
distinct output name. Nothing has to be serialised, and ``--shard K/N`` lets
several processes or containers divide the list without talking to each other:
each derives the same ordered list and takes every Nth entry.

That is what a fit cannot do. The autobuild inside a band sees its own two
patterns and has no idea what the other sixteen will want, so two bands of a
field building at once race on the same F444W filenames.

Epochs already on disk are dropped before the split, so the shards divide the
work that is actually left. Three grid families are covered:

* ``pattern_hi``  - the F444W photometry grids, shared by every band of a field;
* ``pattern_lo``  - the band's MIRI grids;
* the 30" halo grids, when ``repair_saturated`` is on. Saturation repair runs
  inside ``load_data``, not ``build_psfs``, so the ``psfs`` step never touches
  them.

``PSFFactory`` is called directly rather than through the pipeline's autobuild:
it takes ``date_mode`` explicitly, so the grids are per-date whatever the
deployed mophongo defaults to. The factory skips files that already exist, so
re-running is cheap and additive.

    python build_psfs.py [--date-mode all] [--shard 1/6] <config.json> [...]
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


def epoch_tasks(tasks: list[tuple[str, str, str, float | None]],
                date_mode: str) -> list[tuple[str, str, str, float | None, float]]:
    """Expand ``(pattern, csv, psf_dir, fov)`` into one entry per epoch.

    One grid is one filename, so the epoch is the smallest unit that can be
    handed to a worker -- or to another container -- with no possibility of a
    collision. Ordered deterministically, so every shard of a campaign derives
    the same list and slicing it is all the coordination they need.

    Epochs already on disk are dropped here rather than inside the factory, so
    the shards divide the work that is actually left.
    """
    from mophongo.psf_factory import dates_from_csv

    out: list[tuple[str, str, str, float | None, float]] = []
    for pattern, csv, psf_dir, fov in tasks:
        dates = dates_from_csv(csv, mode=date_mode)
        have = {int(m.group(1))
                for f in Path(psf_dir).glob("*.fits")
                if re.search(pattern.replace("_GRID", r"(?:_FOV\d+)?_GRID", 1), f.name)
                for m in [re.search(r"_MJD(\d+)", f.name)] if m}
        missing = [d for d in dates if int(round(d)) not in have]
        log.info("  %s: %d date(s), %d already built", pattern, len(dates),
                 len(dates) - len(missing))
        out += [(pattern, csv, psf_dir, fov, float(d)) for d in missing]
    return sorted(out, key=lambda t: (t[0], t[4]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes per pattern; one (detector, date) grid is "
                         "one job, so this scales with cores. 0 = every core "
                         "this session may use, read from sched_getaffinity "
                         "rather than cpu_count: the container's cgroup cap is "
                         "what we actually get")
    ap.add_argument("--date-mode", default="all",
                    help="one grid per unique integer MJD by default; see "
                         "psf_factory.dates_from_csv")
    ap.add_argument("--shard", default="1/1", metavar="K/N",
                    help="build only shard K of N of the whole work list. "
                         "Every shard derives the same deduplicated list from "
                         "the same configs and takes every Nth item, so the "
                         "shards are disjoint by construction and no two "
                         "processes -- in this container or another -- can "
                         "target one filename. This is what lets a campaign "
                         "fan the grids over as many jobs as it likes rather "
                         "than one per field")
    args = ap.parse_args()

    k, _, n = args.shard.partition("/")
    k, n = int(k), int(n or 1)
    if not 1 <= k <= n:
        raise SystemExit(f"--shard K/N needs 1 <= K <= N (got {args.shard})")

    # sched_getaffinity, not cpu_count: the cgroup cap is what we actually get.
    workers = args.workers or len(os.sched_getaffinity(0))

    # The unit of work is one (pattern, epoch): one grid, one filename. The
    # whole list is known here, so it is enumerated and deduplicated once and
    # then sliced -- the F444W set a field's seven bands share appears in it
    # exactly once, whichever shard happens to own each of its epochs.
    epochs = epoch_tasks(pattern_tasks(args.configs), args.date_mode)
    mine = epochs[k - 1::n]
    log.info("%d grid(s) over %d config(s); shard %d/%d takes %d, %d worker(s)",
             len(epochs), len(args.configs), k, n, len(mine), workers)

    if not mine:
        log.info("nothing to build")
        log.info("PSF_BUILD_DONE")
        return

    # Serially, before any fan-out: build_jwst_psf resolves each epoch's
    # wavefront by querying MAST and caching it under STPSF_PATH, and workers
    # doing that concurrently race to write the same cache files -- which
    # surfaces as "Empty or corrupt FITS file" on whichever epoch lost.
    from mophongo.psf_factory import prewarm_opds

    for inst in ("NIRCAM", "MIRI"):
        dates = [t[4] for t in mine if f"_{inst[:4]}" in t[0].upper()
                 or (inst == "NIRCAM" and "_NRC" in t[0].upper())]
        if dates:
            try:
                prewarm_opds(inst, sorted(set(dates)))
            except Exception as exc:  # noqa: BLE001 - the worker will fetch it
                log.warning("  OPD prewarm for %s failed (%s); workers will fetch",
                            inst, type(exc).__name__)

    total = 0
    start_all = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(mine)),
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(build_one_date, job): job for job in mine}
        for fut in as_completed(futures):
            pattern, _, _, _, mjd = futures[fut]
            done += 1
            try:
                total += fut.result()
            except Exception as exc:  # noqa: BLE001 - one epoch must not stop the rest
                log.error("FAILED %s MJD%d: %s: %s", pattern, mjd,
                          type(exc).__name__, exc)
                continue
            log.info("[%d/%d] %s MJD%d done (%d built, %.1f min elapsed)",
                     done, len(mine), pattern, mjd, total,
                     (time.time() - start_all) / 60)
    log.info("built %d grid(s) in %.1f min", total, (time.time() - start_all) / 60)
    log.info("PSF_BUILD_DONE")


if __name__ == "__main__":
    main()
