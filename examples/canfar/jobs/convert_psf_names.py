#!/usr/bin/env python
"""Convert ePSF grids to the FOV naming convention and stamp their provenance.

Standalone on purpose: astropy only, no mophongo import. A run's venv now
lives inside that run (``run<N>/config/venv``) and is built by ``setup_env.sh``,
so on a fresh tree there is nothing to import mophongo *from* when the grids
need converting. This has to work before that exists.

Two changes per grid, both needed before a run will leave it alone:

* **the name** gains ``_FOV{int}``, taken from the file's own ``FOV`` header --
  what the grid was actually built at. FOV4, FOV8 and FOV30 sets of the same
  GRID/OS layout then have distinct filenames and can share one directory.
* **the header** gains ``HIERARCH MPH`` cards recording the exposure list and
  date mode the grids came from. ``Pipeline._stale_psf_grids`` compares those
  against what the config asks for; a grid with no cards cannot be shown to
  agree with anything, counts as stale, and under ``psf_provenance="rebuild"``
  is rebuilt -- hours of MAST-bound work for files that were already correct.

The card values match ``mophongo.psf_factory.grid_provenance`` exactly (the
CSV's basename, the first 16 hex digits of its SHA-256, the date mode, and the
requested FOV formatted with ``%g``); ``tests/test_convert_psf_names.py``
pins that agreement so this copy cannot drift from the checker.

Usage::

    # rename only, no config needed
    python convert_psf_names.py <psf_dir>

    # rename and stamp, so a later run finds them current
    python convert_psf_names.py <psf_dir> --csv <exposures_wcs.csv> \\
        --date-mode all [--fov 30] [--pattern 'UDS_NRC.._F444W_MJD\\d+']

    # add --apply to make the changes; without it nothing is written

One ``--csv`` describes one grid family. The detection grids and a band's MIRI
grids come from different exposure lists, so convert them in separate runs and
use ``--pattern`` to select which files each applies to.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

from astropy.io import fits

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("convert_psf_names")

#: ``<stem>[_FOV<n>]_GRID<n>_<OS4|DET>.fits`` -- the canonical tail, with the
#: FOV token optional so an already-converted file is recognised, not doubled.
TAIL = re.compile(r"^(?P<stem>.+?)(?:_FOV\d+)?(?P<tail>_GRID\d+_(?:OS\d+|DET))\.fits$")


def csv_fingerprint(path: Path) -> str:
    """First 16 hex digits of the exposure list's SHA-256.

    Content, not path or mtime: the same listing re-downloaded elsewhere is
    the same input, and one rewritten in place with a new exposure is not.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def provenance(csv: Path, date_mode: str, fov: float | None) -> dict[str, str]:
    """The cards a run compares against. Mirrors ``grid_provenance``."""
    cards = {"csv": csv.name, "csvhash": csv_fingerprint(csv),
             "datemode": str(date_mode)}
    if fov is not None:
        cards["fov"] = f"{float(fov):g}"
    return cards


def plan(psf_dir: Path, pattern: str | None) -> list[tuple[Path, Path]]:
    """``(src, dst)`` for every grid to convert, dst == src when only stamping."""
    rx = re.compile(pattern) if pattern else None
    out = []
    for src in sorted(psf_dir.glob("*.fits")):
        if rx and not rx.search(src.name):
            continue
        m = TAIL.match(src.name)
        if not m:
            log.warning("  ? %s: no GRID/OS tail, left alone", src.name)
            continue
        try:
            fov_hdr = fits.getheader(src).get("FOV")
        except OSError as exc:
            log.warning("  ! %s unreadable (%s), left alone", src.name, exc)
            continue
        if fov_hdr is None:
            log.warning("  ! %s has no FOV header, left alone", src.name)
            continue
        out.append((src, psf_dir / "{}_FOV{}{}.fits".format(
            m.group("stem"), int(round(float(fov_hdr))), m.group("tail"))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psf_dir", type=Path)
    ap.add_argument("--csv", type=Path,
                    help="exposure list the grids were built from; without it "
                         "the files are renamed but not stamped, and a run "
                         "will still consider them stale")
    ap.add_argument("--date-mode", default="all",
                    help="date mode the grids were built with (default: all)")
    ap.add_argument("--fov", type=float, default=None,
                    help="FOV the config requests, when it sets one (e.g. 30 "
                         "for the halo grids). Omit when the config leaves it "
                         "unset -- the check then compares only csv and mode")
    ap.add_argument("--pattern", default=None,
                    help="only convert files whose name matches this regex")
    ap.add_argument("--apply", action="store_true",
                    help="make the changes; without it, print them and stop")
    args = ap.parse_args(argv)

    if not args.psf_dir.is_dir():
        log.error("no such directory: %s", args.psf_dir)
        return 2
    if args.csv is not None and not args.csv.exists():
        log.error("no such exposure list: %s", args.csv)
        return 2

    cards = provenance(args.csv, args.date_mode, args.fov) if args.csv else None
    jobs = plan(args.psf_dir, args.pattern)
    renames = sum(1 for src, dst in jobs if src != dst)
    log.info("%d grid(s): %d to rename, %d already named",
             len(jobs), renames, len(jobs) - renames)
    log.info("stamp: %s", cards or "NOT stamping (no --csv); runs will still "
                                   "treat these as stale")
    for src, dst in jobs[:5]:
        log.info("   %s%s", src.name,
                 "" if src == dst else f" -> {dst.name}")
    if len(jobs) > 5:
        log.info("   ... and %d more", len(jobs) - 5)

    if not args.apply:
        log.info("(dry run; pass --apply)")
        return 0

    stamped = moved = 0
    for src, dst in jobs:
        if cards:
            with fits.open(src, mode="update") as hdul:
                for key, value in cards.items():
                    hdul[0].header[f"HIERARCH MPH {key.upper()}"] = value
            stamped += 1
        if dst != src:
            if dst.exists():
                log.warning("  ! %s exists; %s left alone", dst.name, src.name)
                continue
            src.rename(dst)
            moved += 1
    log.info("done: %d stamped, %d renamed", stamped, moved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
