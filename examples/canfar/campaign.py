#!/usr/bin/env python
"""Launch a whole mophongo campaign on CANFAR in one command.

Wraps the individual steps - upload, environment, config rewrite, staging, run -
so a full field or the entire MINERVA release can be started without driving
each stage by hand::

    python campaign.py                       # every config in ../minerva
    python campaign.py --fields uds cosmos   # only those fields
    python campaign.py --from stage          # inputs already pushed and built
    python campaign.py --from arcify --skip stage   # inputs already decompressed
    python campaign.py --dry-run             # print the plan, do nothing

To ship a code change into a campaign without rebuilding the venv, fast-forward
the run's checkout first and start the campaign after it::

    python submit.py sync
    python campaign.py --from arcify --skip stage

The submission is two phases, and the split is the point:

1. ``prep`` - one short job per field, every field at once, **waited on**. It
   builds that field's shared F444W and 30" halo ePSF grids and runs the
   saturation repair into a per-field cache.
2. ``run`` - every band of every field, fired off together and not waited on.

Everything phase 1 produces is shared by a field's bands and depends on the
detection band alone, so without it each band rebuilds the same grids, re-runs
the same repair, and several of them write one cache file at the same time.
Phase 1 costs one short job per field and buys a clean parallel fan-out.

Three more things this encodes, all learned the hard way:

* Fits are submitted and *not* waited on. A campaign must not depend on a
  laptop-side process staying alive; jobs live on CANFAR once submitted. Only
  the prep phase waits, because everything after it depends on the result.
* Staging runs one band per field to completion before the rest, because bands
  of a field share that field's F444W mosaic and segmap - several GB that would
  otherwise be decompressed once per band.
* With ``--skip prep``, a field whose PSF grids are missing runs one band alone
  first. With no grid matching the config pattern the pipeline builds one, and
  several bands of a field building the same grids at once race on a single
  ``psf_dir``.

Progress afterwards: ``submit.py status``, ``submit.py logs <id>``,
``submit.py fetch <name>``.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("campaign")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PSF_DIR = REPO / "data" / "PSF"
PYTHON = sys.executable
STEPS = ["push", "setup", "arcify", "seed", "stage", "prep", "run"]


#: Band a field's prep job runs. It builds the shared F444W and halo grids and
#: the saturation repair, so every other band of the field starts warm. F770W
#: by preference: it is the shortest MIRI band, so its own lo-res grids and
#: fit are the cheapest way to get the shared products built. Falls back to
#: whichever band sorts first when a field has no F770W.
PREP_BAND = "f770w"


def prep_leader(bands: list[str]) -> str:
    """The band whose job builds a field's shared products."""
    for name in bands:
        if name.split("_")[1:2] == [PREP_BAND] or f"_{PREP_BAND}" in name:
            return name
    return bands[0]


def run_step(args: list[str], dry: bool) -> None:
    """Run one toolkit command, stopping the campaign if it fails."""
    printable = " ".join(str(a) for a in args)
    log.info("+ %s", printable)
    if dry:
        return
    result = subprocess.run([PYTHON, *args], cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"failed: {printable}")


#: A per-band run config is named ``<field>_f<band>w.json``. The directory
#: also holds inputs that are not RunConfigs (``minerva_sed_fields.json``),
#: which would fail deep inside arcify with a confusing key error.
BAND_CONFIG = re.compile(r"^[a-z0-9]+_f\d+w$")


def configs_for(fields: list[str] | None) -> list[Path]:
    """The local MINERVA per-band configs to run, optionally by field."""
    found = [p for p in sorted((HERE.parent / "minerva").glob("*.json"))
             if BAND_CONFIG.match(p.stem)]
    if fields:
        found = [p for p in found if p.stem.split("_")[0] in fields]
    if not found:
        raise SystemExit("no configs matched")
    return found


def arc_psf_names() -> list[str]:
    """Grid filenames already sitting in the run tree's ``PSF/`` directory.

    The question ``has_shared_grids`` really asks is whether the *run* will
    find its grids, and the run reads ``$RUN/PSF`` on arc rather than the
    laptop. A field whose grids an earlier job built is complete there while
    the local directory has never held them, and serialising a leader for it
    costs a whole run - a full field, so hours. Nothing uploads grids any
    more, so arc is the only answer that counts.

    An unreadable listing returns empty, which falls back to the local check.
    """
    try:
        import submit  # local import: keeps --help and --dry-run off skaha
        vls = submit.VCP.parent / "vls"
        out = subprocess.run([str(vls), f"{submit.RUN_VOS}/PSF/"],
                             capture_output=True, text=True, timeout=180)
    except Exception as exc:  # noqa: BLE001 - any failure just means "unknown"
        log.warning("could not list the arc PSF dir (%s); using local grids only",
                    type(exc).__name__)
        return []
    if out.returncode != 0:
        log.warning("could not list the arc PSF dir; using local grids only")
        return []
    return out.stdout.split()


def has_shared_grids(cfg_path: Path, extra_names: list[str] | None = None) -> bool:
    """Whether the field's *hi-res* PSF grids already exist.

    ``extra_names`` are filenames known to be on arc already; they count the
    same as local ones, since the run reads the arc copy.

    Only ``pattern_hi`` matters here. Every band of a field matches the same
    F444W grids, so several bands building those at once write the same
    filenames into one ``psf_dir``. The per-band MIRI grids of ``pattern_lo``
    have distinct names and can be built concurrently without racing.

    With ``repair_saturated`` the 30" halo grids are shared the same way -
    every band of the field derives the same ``_FOV30_GRID1_OS4`` names from
    ``pattern_hi`` - so they count here too.
    """
    cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
    pattern = cfg.get("pattern_hi")
    if not pattern:
        return True
    patterns = [pattern]
    if cfg.get("repair_saturated"):
        # same derivation as Pipeline._repair_halo_pattern
        patterns.append(cfg.get("repair_psf_pattern")
                        or re.sub(r"_MJD.*$", r"_MJD\\d+_FOV30_GRID1_OS4", pattern))
    # Grid filenames carry an _FOV token that the configs' patterns do not, so
    # match the way the loader does (mophongo.jwst_psf.fov_agnostic_pattern is
    # the source of truth; inlined to keep this script free of the package).
    patterns = [p if ("_FOV" in p or "_GRID" not in p)
                else p.replace("_GRID", r"(?:_FOV\d+)?_GRID", 1)
                for p in patterns]
    names = [p.name for p in PSF_DIR.glob("*.fits")] + list(extra_names or [])
    return all(any(re.search(pat, n) for n in names) for pat in patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", help="restrict to these fields, e.g. uds cosmos")
    ap.add_argument("--from", dest="start", choices=STEPS, default="push",
                    help="skip everything before this step")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[],
                    help="drop these steps from the middle of the chain, e.g. "
                         "--skip stage when the inputs are already decompressed")
    ap.add_argument("--ram", type=int, default=None,
                    help="override the per-field default (64 GB, EGS 82)")
    ap.add_argument("--cores", type=int, default=None,
                    help="override; 4 by default, for a full field or a patch alike")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial radius in arcmin; 0 runs the full field")
    ap.add_argument("--suffix", default="",
                    help="append to every run name, e.g. _full, to keep outputs separate")
    ap.add_argument("--seed-from", default=None, metavar="SUFFIX",
                    help="seed PSF/kernel caches from the runs with this suffix "
                         "(use '' for the unsuffixed ones); skips rebuilding them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [s for s in STEPS[STEPS.index(args.start):] if s not in args.skip]
    cfgs = configs_for(args.fields)
    names = [c.stem + args.suffix for c in cfgs]
    log.info("campaign over %d config(s): %s", len(names), ", ".join(names))

    if "push" in todo:
        run_step(["submit.py", "push"], args.dry_run)
    if "setup" in todo:
        run_step(["submit.py", "setup"], args.dry_run)
    if "arcify" in todo:
        arcify = ["arcify.py", *[str(c) for c in cfgs]]
        if args.r_trial is not None:
            arcify += ["--r-trial", str(args.r_trial)]
        if args.suffix:
            arcify += ["--suffix", args.suffix]
        run_step(arcify, args.dry_run)
    if "seed" in todo and args.seed_from is not None:
        pairs = [f"{c.stem}{args.seed_from}:{c.stem}{args.suffix}" for c in cfgs]
        run_step(["submit.py", "seed", *pairs], args.dry_run)
    if "stage" in todo:
        run_step(["submit.py", "stage", *names], args.dry_run)
    # Left out unless overridden, so submit.py picks the per-field size rather
    # than one number applied to every field in the campaign.
    common = [] if args.ram is None else ["--ram", str(args.ram)]
    if args.cores is not None:
        common += ["--cores", str(args.cores)]

    # Group by field: everything below is per-field, because a field's bands
    # share its F444W grids, its halo grids and its saturation repair.
    by_field: dict[str, list[tuple[str, Path]]] = {}
    for name, cfg in zip(names, cfgs):
        by_field.setdefault(name.split("_")[0], []).append((name, cfg))

    # Phase 1: one prep job per field, all fields at once, waited on. Each
    # builds that field's shared ePSF grids and runs the saturation repair
    # into a per-field cache. It is short next to a fit, and it is what lets
    # phase 2 fire every band of every field in parallel: without it each band
    # rebuilds the same grids and re-runs the same repair, and several of them
    # write one cache file at the same time.
    if "prep" in todo:
        first = [prep_leader([n for n, _ in entries])
                 for entries in by_field.values()]
        log.info("prep: %d field(s), %s", len(first), ", ".join(first))
        run_step(["submit.py", "run", *first, "--step", "prep", *common],
                 args.dry_run)

    if "run" not in todo:
        return

    on_arc = [] if args.dry_run else arc_psf_names()
    for field, entries in by_field.items():
        bands = [n for n, _ in entries]
        # prep already built the grids; without it, fall back to the old
        # one-band-first order so concurrent builds do not race on psf_dir
        needs_build = ("prep" not in todo
                       and not has_shared_grids(entries[0][1], on_arc))
        if needs_build and len(bands) > 1:
            log.info("%s: shared grids missing (F444W or the 30\" halo), "
                     "running %s alone to build them", field, bands[0])
            run_step(["submit.py", "run", bands[0], *common], args.dry_run)
            run_step(["submit.py", "run", *bands[1:], *common, "--no-wait"], args.dry_run)
        else:
            run_step(["submit.py", "run", *bands, *common, "--no-wait"], args.dry_run)

    log.info("submitted. watch with:  python submit.py status")


if __name__ == "__main__":
    main()
