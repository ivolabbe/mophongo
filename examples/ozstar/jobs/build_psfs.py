#!/usr/bin/env python
"""Pre-build every ePSF grid a config's fit will ask for.

Run on the login node, which is the only machine with both internet and the
module stack. Compute nodes cannot do this: ``PSFFactory`` builds MJD-tagged
grids and ``stpsf`` resolves each exposure date to a wavefront OPD by querying
MAST, and a compute node has neither DNS nor a route.

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
import re
import time
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


def build_for_pattern(pattern: str, csv: str, psf_dir: str, date_mode: str,
                      fov_default: float | None) -> int:
    """Build every grid implied by ``pattern`` and the exposure list."""
    from mophongo.pipeline import _psf_factory_kwargs
    from mophongo.psf_factory import PSFFactory, dates_from_csv

    kw = _psf_factory_kwargs(pattern)          # same derivation the fit uses,
    fov = kw.pop("fov_arcsec", fov_default)    # so the filenames match
    want = len(dates_from_csv(csv, mode=date_mode))
    before = len(list(Path(psf_dir).glob("*.fits")))
    log.info("  %s: %d date(s) from %s", pattern, want, Path(csv).name)
    PSFFactory(outdir=psf_dir, fov_arcsec=fov, date_mode=date_mode,
               **kw).from_csv(csv, save=True)
    return len(list(Path(psf_dir).glob("*.fits"))) - before


def build_one(cfg_path: Path, date_mode: str) -> int:
    """Build the hi, lo and halo grids for one config; return grids added."""
    from mophongo.pipeline import RunConfig

    cfg = RunConfig.from_json(str(cfg_path))
    psf_dir = str(cfg.psf_dir)
    Path(psf_dir).mkdir(parents=True, exist_ok=True)
    added = 0
    for pattern, csv in ((cfg.pattern_hi, cfg.csv_hi), (cfg.pattern_lo, cfg.csv_lo)):
        if pattern:
            added += build_for_pattern(pattern, str(csv), psf_dir, date_mode,
                                       cfg.psf_fov_arcsec)
    if getattr(cfg, "repair_saturated", False):
        pat = cfg.repair_psf_pattern or halo_pattern(cfg.pattern_hi)
        if pat and pat != cfg.pattern_hi:
            added += build_for_pattern(pat, str(cfg.csv_hi), psf_dir, date_mode,
                                       cfg.psf_fov_arcsec)
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--date-mode", default="all",
                    help="one grid per unique integer MJD by default; see "
                         "psf_factory.dates_from_csv")
    args = ap.parse_args()

    for i, cfg_path in enumerate(args.configs, 1):
        start = time.time()
        log.info("=== [%d/%d] %s", i, len(args.configs), cfg_path.name)
        try:
            added = build_one(cfg_path, args.date_mode)
        except Exception as exc:  # noqa: BLE001 - one bad config must not stop the rest
            log.error("FAILED %s: %s: %s", cfg_path.name, type(exc).__name__, exc)
            continue
        log.info("=== [%d/%d] %s done in %.1f min, +%d grid(s)",
                 i, len(args.configs), cfg_path.name, (time.time() - start) / 60, added)
    log.info("PSF_BUILD_DONE")


if __name__ == "__main__":
    main()
