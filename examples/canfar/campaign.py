#!/usr/bin/env python
"""Launch a whole mophongo campaign on CANFAR in one command.

Wraps the individual steps - upload, environment, config rewrite, staging, run -
so a full field or the entire MINERVA release can be started without driving
each stage by hand::

    python campaign.py                       # every config in ../minerva
    python campaign.py --fields uds cosmos   # only those fields
    python campaign.py --from stage          # inputs already pushed and built
    python campaign.py --dry-run             # print the plan, do nothing

Three things this encodes, all learned the hard way:

* Runs are submitted and *not* waited on. A campaign must not depend on a
  laptop-side process staying alive; jobs live on CANFAR once submitted.
* Staging runs one band per field to completion before the rest, because bands
  of a field share that field's F444W mosaic and segmap - several GB that would
  otherwise be decompressed once per band.
* A field whose PSF grids are missing runs one band alone first. With no grid
  matching the config pattern the pipeline builds one, and several bands of a
  field building the same grids at once race on a single ``psf_dir``.

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
STEPS = ["push", "setup", "arcify", "seed", "stage", "run"]


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


def has_shared_grids(cfg_path: Path) -> bool:
    """Whether the field's *hi-res* PSF grids already exist locally.

    Only ``pattern_hi`` matters here. Every band of a field matches the same
    F444W grids, so several bands building those at once write the same
    filenames into one ``psf_dir``. The per-band MIRI grids of ``pattern_lo``
    have distinct names and can be built concurrently without racing.

    ``push`` uploads whatever is in ``data/PSF``, so local presence stands in
    for what the run will find on arc.

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
    names = [p.name for p in PSF_DIR.glob("*.fits")]
    return all(any(re.search(pat, n) for n in names) for pat in patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", help="restrict to these fields, e.g. uds cosmos")
    ap.add_argument("--from", dest="start", choices=STEPS, default="push",
                    help="skip everything before this step")
    ap.add_argument("--ram", type=int, default=48)
    ap.add_argument("--cores", type=int, default=None,
                    help="override; 2 by default, for a full field or a patch alike")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial radius in arcmin; 0 runs the full field")
    ap.add_argument("--suffix", default="",
                    help="append to every run name, e.g. _full, to keep outputs separate")
    ap.add_argument("--seed-from", default=None, metavar="SUFFIX",
                    help="seed PSF/kernel caches from the runs with this suffix "
                         "(use '' for the unsuffixed ones); skips rebuilding them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = STEPS[STEPS.index(args.start):]
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
    if "run" not in todo:
        return

    common = ["--ram", str(args.ram)]
    if args.cores is not None:
        common += ["--cores", str(args.cores)]

    # Group by field so a field missing its PSF grids can send one band first.
    by_field: dict[str, list[tuple[str, Path]]] = {}
    for name, cfg in zip(names, cfgs):
        by_field.setdefault(name.split("_")[0], []).append((name, cfg))

    for field, entries in by_field.items():
        bands = [n for n, _ in entries]
        needs_build = not has_shared_grids(entries[0][1])
        if needs_build and len(bands) > 1:
            log.info("%s: no F444W grids yet, running %s alone to build them", field, bands[0])
            run_step(["submit.py", "run", bands[0], *common], args.dry_run)
            run_step(["submit.py", "run", *bands[1:], *common, "--no-wait"], args.dry_run)
        else:
            run_step(["submit.py", "run", *bands, *common, "--no-wait"], args.dry_run)

    log.info("submitted. watch with:  python submit.py status")


if __name__ == "__main__":
    main()
