#!/usr/bin/env python
"""Launch a whole mophongo campaign on OzStar in one command.

Wraps the individual steps - config rewrite, upload, environment, staging, run
- so every band of every MINERVA field can be started at once::

    python campaign.py                          # every config in ../minerva
    python campaign.py --fields uds             # only that field
    python campaign.py --bands f770w            # only that band
    python campaign.py --from stage             # source and configs already up
    python campaign.py --dry-run                # print the plan, submit nothing

The whole campaign is submitted as one SLURM dependency graph and returns
immediately: staging a field is a ``datamover`` job, and that field's fits wait
on it with ``--dependency=afterok``. Nothing afterwards depends on the laptop.

Three things this encodes:

* staging is one job per *field*, not per band, because the bands of a field
  share the F444W mosaic, its weight map and the segmap;
* a field whose PSF grids are missing runs one band alone first, because with
  no grid matching the config pattern the pipeline builds one, and several
  bands of a field would build the same grids concurrently into one
  ``psf_dir``;
* ``afterok`` means a failed stage leaves its fits queued as
  ``DependencyNeverSatisfied`` until SLURM cancels them - which is the wanted
  behaviour, but reads as jobs silently disappearing if you do not know it.

Progress afterwards: ``submit.py status``, ``submit.py logs <name>``,
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

import ozroot
import submit

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("campaign")

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
STEPS = ["ozify", "push", "setup", "seed", "stage", "run"]

#: A per-band run config is named ``<field>_f<band>w.json``. The directory also
#: holds inputs that are not RunConfigs (``minerva_sed_fields.json``), which
#: would fail deep inside ozify with a confusing key error.
BAND_CONFIG = re.compile(r"^[a-z0-9]+_f\d+w$")


def run_step(args: list[str], dry: bool) -> None:
    """Run one toolkit command, stopping the campaign if it fails."""
    printable = " ".join(str(a) for a in args)
    log.info("+ %s", printable)
    if dry:
        return
    result = subprocess.run([PYTHON, *args], cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"failed: {printable}")


def configs_for(fields: list[str] | None, bands: list[str] | None) -> list[Path]:
    """The local MINERVA per-band configs to run."""
    found = [p for p in sorted((HERE.parent / "minerva").glob("*.json"))
             if BAND_CONFIG.match(p.stem)]
    if fields:
        found = [p for p in found if p.stem.split("_")[0] in fields]
    if bands:
        found = [p for p in found if p.stem.split("_")[1] in bands]
    if not found:
        raise SystemExit("no configs matched")
    return found


def psf_names_on_fred() -> list[str]:
    """Grid filenames already in the run tree's ``PSF/`` directory.

    Only the remote listing counts. Unlike the CANFAR toolkit, nothing here
    uploads local grids - ``setup`` clones the source and the grids are built
    on the node from the STPSF reference data - so a grid on the laptop says
    nothing about what the run will find.

    A missing or unreadable directory lists as empty, which costs one leader
    run and nothing else.
    """
    return submit.ssh(f"ls {ozroot.run_root()}/PSF 2>/dev/null", check=False).split()


def has_shared_grids(cfg_path: Path, names: list[str]) -> bool:
    """Whether the field's *hi-res* PSF grids already exist on /fred.

    Only ``pattern_hi`` matters. Every band of a field matches the same F444W
    grids, so several bands building those at once write the same filenames
    into one ``psf_dir``. The per-band MIRI grids of ``pattern_lo`` have
    distinct names and can be built concurrently without racing.

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
    return all(any(re.search(pat, n) for n in names) for pat in patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", help="restrict to these, e.g. uds cosmos")
    ap.add_argument("--bands", nargs="+", help="restrict to these, e.g. f770w")
    ap.add_argument("--from", dest="start", choices=STEPS, default="ozify",
                    help="skip everything before this step")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[],
                    help="drop these steps from the middle of the chain, e.g. "
                         "--skip stage when the inputs are already on /fred")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--mem", type=int, default=64, help="GB per run")
    ap.add_argument("--time", default="24:00:00", help="walltime per run")
    ap.add_argument("--stage-time", default="24:00:00", help="walltime per stage job")
    ap.add_argument("--branch", default="main", help="mophongo branch to clone")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial radius in arcmin; 0 runs the full field")
    ap.add_argument("--suffix", default="",
                    help="append to every run name, e.g. _trial, to keep outputs separate")
    ap.add_argument("--seed-from", default=None, metavar="SUFFIX",
                    help="seed PSF/kernel caches from the runs with this suffix "
                         "(use '' for the unsuffixed ones); skips rebuilding them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [s for s in STEPS[STEPS.index(args.start):] if s not in args.skip]
    cfgs = configs_for(args.fields, args.bands)
    names = [c.stem + args.suffix for c in cfgs]
    log.info("campaign over %d config(s): %s", len(names), ", ".join(names))
    log.info("run tree %s on %s", ozroot.run_root(), ozroot.ssh_target())

    if "ozify" in todo:
        ozify = ["ozify.py", *[str(c) for c in cfgs]]
        if args.r_trial is not None:
            ozify += ["--r-trial", str(args.r_trial)]
        if args.suffix:
            ozify += ["--suffix", args.suffix]
        run_step(ozify, args.dry_run)
    if "push" in todo:
        log.info("+ submit.push(%d configs)", len(names))
        if not args.dry_run:
            submit.push(names)
    if "setup" in todo:
        run_step(["submit.py", "setup", "--branch", args.branch], args.dry_run)
    if "seed" in todo and args.seed_from is not None:
        pairs = [f"{c.stem}{args.seed_from}:{c.stem}{args.suffix}" for c in cfgs]
        run_step(["submit.py", "seed", *pairs], args.dry_run)

    stage_ids: dict[str, str] = {}
    if "stage" in todo:
        for field, bands in submit.by_field(names).items():
            if args.dry_run:
                log.info("+ stage %s: %s", field, " ".join(bands))
                continue
            stage_ids[field] = submit.stage(bands, walltime=args.stage_time)[0]
    if "run" not in todo:
        return

    on_fred = [] if args.dry_run else psf_names_on_fred()
    for field, bands in submit.by_field(names).items():
        cfg = next(c for c, n in zip(cfgs, names) if n == bands[0])
        dep = stage_ids.get(field)
        leader_needed = not has_shared_grids(cfg, on_fred) and len(bands) > 1
        if args.dry_run:
            log.info("+ run %s%s%s", " ".join(bands),
                     f" after {dep or 'nothing'}",
                     " (first band alone: no shared PSF grids yet)" if leader_needed else "")
            continue
        if leader_needed:
            log.info("%s: shared grids missing (F444W or the 30\" halo), "
                     "running %s alone to build them", field, bands[0])
            leader = submit.run([bands[0]], after=dep, cores=args.cores,
                                mem=args.mem, walltime=args.time)[0]
            submit.run(bands[1:], after=leader, cores=args.cores,
                       mem=args.mem, walltime=args.time)
        else:
            submit.run(bands, after=dep, cores=args.cores, mem=args.mem,
                       walltime=args.time)

    log.info("submitted. watch with:  python submit.py status")


if __name__ == "__main__":
    main()
