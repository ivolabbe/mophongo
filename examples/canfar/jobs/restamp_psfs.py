#!/usr/bin/env python
"""Rename existing ePSF grids to the FOV convention and stamp their provenance.

Grids built before 2026-08 carry neither the ``_FOV{int}`` token in their
filename nor the ``HIERARCH MPH`` provenance cards in their header. Both
matter to a run:

* the token is what lets FOV4, FOV8 and FOV30 sets of the same GRID/OS layout
  share one directory (``PSFFactory.filename``);
* the cards are what ``Pipeline._stale_psf_grids`` compares against
  ``grid_provenance(csv, date_mode, fov)``. A grid with no cards cannot be
  shown to agree with anything, so it counts as stale -- and with
  ``psf_provenance="rebuild"`` that means every band rebuilds its grids on
  first contact, hours of MAST-bound work for files that were already correct.

This script fixes both in place, so a later run finds the grids current and
leaves them alone.

It derives every value the way the pipeline does -- ``_psf_factory_kwargs``
for the FOV a pattern implies, ``grid_provenance`` for the cards -- rather
than reproducing the rules here, so the two cannot drift apart. The FOV in the
*filename* comes from each file's own ``FOV`` header, which is what the grid
was actually built at; the FOV in the *cards* is what the config asks for,
which is what the check compares.

Usage::

    python restamp_psfs.py <psf_dir> <config.json> [more configs ...]
    python restamp_psfs.py --apply <psf_dir> <config.json> [...]

Dry run by default: it prints what it would rename and stamp and changes
nothing. On CANFAR, run it with the run tree's interpreter so mophongo is
importable::

    $RUN/run$RUNNUM/config/venv/bin/python $RUN/jobs/restamp_psfs.py --apply \\
        $RUN/PSF $RUN/run$RUNNUM/config/uds_f770w_canfar.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from astropy.io import fits

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("restamp_psfs")

# The tail every canonical name ends with; the FOV token sits just before it.
TAIL = re.compile(r"^(?P<stem>.+?)(?:_FOV\d+)?(?P<tail>_GRID\d+_(?:OS\d+|DET))\.fits$")


def load_config(path: Path) -> dict:
    """Read a run config, tolerating the ``#`` comment lines these carry."""
    return json.loads(re.sub(r"(?m)^\s*#.*$", "", path.read_text()))


def patterns_of(cfg: dict) -> list[tuple[str, str]]:
    """``(pattern, csv)`` for every grid family a run of ``cfg`` will look up.

    The halo pattern is included because saturation repair runs inside
    ``load_data`` and needs its own 30" grids; a run that finds those stale
    rebuilds them exactly like the photometry grids.
    """
    from mophongo.pipeline import _PSF_PATTERN_RE

    out = []
    for key, csv_key in (("pattern_hi", "csv_hi"), ("pattern_lo", "csv_lo")):
        pattern, csv = cfg.get(key), cfg.get(csv_key)
        if pattern and csv:
            out.append((pattern, csv))
    if cfg.get("repair_saturated"):
        halo = cfg.get("repair_psf_pattern")
        if not halo and cfg.get("pattern_hi"):
            m = _PSF_PATTERN_RE.match(str(cfg["pattern_hi"]).strip())
            if m:
                halo = (f"{m.group('prefix')}_{m.group('det')}_{m.group('filt')}"
                        r"_MJD\d+_FOV30_GRID1_OS4")
        if halo and cfg.get("csv_hi"):
            out.append((halo, cfg["csv_hi"]))
    return out


def plan_for(psf_dir: Path, pattern: str, csv: str, cfg: dict) -> list[tuple[Path, Path, dict]]:
    """``(src, dst, cards)`` for each grid this pattern will look up."""
    from mophongo.jwst_psf import fov_agnostic_pattern, read_stdpsf_provenance
    from mophongo.pipeline import _psf_factory_kwargs
    from mophongo.psf_factory import grid_provenance

    kw = _psf_factory_kwargs(pattern)
    fov_cfg = kw.get("fov_arcsec", cfg.get("psf_fov_arcsec"))
    want = grid_provenance(csv, cfg.get("psf_date_mode", "all"), fov_cfg)
    rx = re.compile(fov_agnostic_pattern(pattern))

    plan = []
    for src in sorted(psf_dir.glob("*.fits")):
        if not rx.search(src.name):
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
        dst = psf_dir / "{}_FOV{}{}.fits".format(
            m.group("stem"), int(round(float(fov_hdr))), m.group("tail"))
        if read_stdpsf_provenance(src) == want and dst == src:
            continue                     # already named and stamped correctly
        plan.append((src, dst, want))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psf_dir", type=Path)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="make the changes; without it, print them and stop")
    args = ap.parse_args()

    if not args.psf_dir.is_dir():
        raise SystemExit(f"no such directory: {args.psf_dir}")

    seen: dict[Path, tuple[Path, dict]] = {}
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        log.info("%s", cfg_path.name)
        for pattern, csv in patterns_of(cfg):
            if not Path(csv).exists():
                log.warning("  ! %s missing; cannot hash it, skipping %s",
                            csv, pattern)
                continue
            plan = plan_for(args.psf_dir, pattern, csv, cfg)
            log.info("  %-52s %d grid(s)", pattern, len(plan))
            for src, dst, cards in plan:
                # a band shared by several configs is planned once; the first
                # config to claim a grid wins, and they agree by construction
                seen.setdefault(src, (dst, cards))

    if not seen:
        log.info("nothing to do")
        return

    renames = sum(1 for src, (dst, _) in seen.items() if src != dst)
    log.info("%d grid(s): %d rename, %d stamp only",
             len(seen), renames, len(seen) - renames)
    for src, (dst, cards) in list(seen.items())[:5]:
        log.info("   %s -> %s", src.name, dst.name)
        log.info("      %s", cards)
    if not args.apply:
        log.info("(dry run; pass --apply)")
        return

    for src, (dst, cards) in seen.items():
        with fits.open(src, mode="update") as hdul:
            for key, value in cards.items():
                hdul[0].header[f"HIERARCH MPH {key.upper()}"] = value
        if dst != src:
            if dst.exists():
                log.warning("  ! %s exists; %s left alone", dst.name, src.name)
                continue
            src.rename(dst)
    log.info("done: %d grid(s) stamped, %d renamed", len(seen), renames)


if __name__ == "__main__":
    main()
