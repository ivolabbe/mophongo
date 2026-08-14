#!/usr/bin/env python
"""Launch a whole mophongo campaign on CANFAR in one command.

Wraps the individual steps - upload, environment, config rewrite, staging, run -
so a full field or the entire MINERVA release can be started without driving
each stage by hand::

    python campaign.py                       # every config in ../minerva
    python campaign.py --fields uds cosmos   # only those fields
    python campaign.py --from stage          # inputs already pushed and built
    python campaign.py --from arcify --skip stage   # inputs already decompressed
    python campaign.py --skip psf repair     # grids and repair caches in place
    python campaign.py --dry-run             # print the plan, do nothing

To ship a code change into a campaign without rebuilding the venv, fast-forward
the run's checkout first and start the campaign after it::

    python submit.py sync
    python campaign.py --from arcify --skip stage

The submission is three phases, and the split is the point:

1. ``psf`` - one job per field, every field at once, **waited on**. It builds
   every ePSF grid the field's configs name: the shared F444W set, the 30"
   halo set, and each band's MIRI set, deduplicated across the configs and
   fanned out over epochs inside the job. Skipped when the grids are already
   on arc, which is the usual case after a release has been fitted once.
2. ``repair`` - one F444W-only job per field, every field at once, **waited
   on**. The saturation repair depends on the detection band alone, so one run
   of it serves every band of the field.
3. ``run`` - every band of every field, in a single dispatch, not waited on.

The first two phases exist so the third needs no order at all. Everything they
produce is shared by a field's bands and depends on the detection band alone,
so without them each band rebuilds the same grids, re-runs the same repair, and
several of them write one cache file at the same time. This is the same shape
``examples/ozstar`` uses, where the grid build has to be a separate step
regardless: its compute nodes cannot reach MAST.

Three more things this encodes, all learned the hard way:

* Fits are submitted and *not* waited on. A campaign must not depend on a
  laptop-side process staying alive; jobs live on CANFAR once submitted. Only
  the two preparation phases wait, because everything after them depends on
  the result.
* Staging runs one band per field to completion before the rest, because bands
  of a field share that field's F444W mosaic and segmap - several GB that would
  otherwise be decompressed once per band.
* The grid check reads the arc PSF directory, not the laptop: that is where the
  run will look, and a field whose grids an earlier job built is complete there
  while the local directory has never held them.

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
STEPS = ["push", "setup", "arcify", "seed", "psf", "stage", "repair", "run"]


#: Band a field's repair job runs. The repair depends on the detection band
#: alone, so any band of the field produces the same cache. F770W
#: by preference: it is the shortest MIRI band, so its own lo-res grids and
#: fit are the cheapest way to get the shared products built. Falls back to
#: whichever band sorts first when a field has no F770W.
PREP_BAND = "f770w"


def prep_leader(bands: list[str]) -> str:
    """The band whose job builds a field's shared repair cache."""
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

    The question ``has_all_grids`` really asks is whether the *run* will
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


def has_all_grids(cfg_path: Path, extra_names: list[str] | None = None) -> bool:
    """Whether every ePSF grid family this config needs is already on arc.

    ``extra_names`` are filenames known to be on arc; they count the same as
    local ones, since the run reads the arc copy.

    All three families, because this decides whether the ``psf`` step has work
    to do: the shared F444W set of ``pattern_hi``, the band's own MIRI set of
    ``pattern_lo``, and - with ``repair_saturated`` - the 30" halo set every
    band of the field derives from ``pattern_hi``.

    A family counts as present when *anything* matches it. That is deliberately
    weak: it cannot tell a complete set of epochs from one grid, which is what
    ``psf_provenance`` is for. It exists to skip a step that would otherwise
    queue three jobs to discover there is nothing to build.
    """
    cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
    pattern = cfg.get("pattern_hi")
    if not pattern:
        return True
    patterns = [pattern]
    if cfg.get("pattern_lo"):
        patterns.append(cfg["pattern_lo"])
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
    # Phase 1: every ePSF grid the release needs, one job per field, waited on.
    # Skipped when the grids are all there already, which is the usual case
    # once a release has been fitted once - the check reads arc, not the
    # laptop, since that is where the run will look.
    if "psf" in todo:
        on_arc = [] if args.dry_run else arc_psf_names()
        missing = [c.stem for c in cfgs if not has_all_grids(c, on_arc)]
        if missing or args.dry_run:
            log.info("psf: %d of %d config(s) still need grids (%s)",
                     len(missing), len(cfgs), ", ".join(missing[:4]) or "-")
            run_step(["submit.py", "psf", *names], args.dry_run)
        else:
            log.info("psf: every grid is on arc already; skipping")
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

    # Phase 2: the saturation repair, one F444W-only job per field, all fields
    # at once, waited on. It depends on the detection band alone, so one run
    # of it serves every band of the field; submitting the bands without it
    # means each re-runs the same repair and several write one cache file at
    # the same time.
    if "repair" in todo:
        first = [prep_leader([n for n, _ in entries])
                 for entries in by_field.values()]
        log.info("repair: %d field(s), %s", len(first), ", ".join(first))
        run_step(["submit.py", "run", *first, "--step", "repair", *common],
                 args.dry_run)

    if "run" not in todo:
        return

    # Phase 3: every band of every field, in one dispatch. Nothing is shared
    # that has not already been built, so there is no leader to elect and no
    # order to keep -- which is the point of paying for phases 1 and 2.
    all_bands = [n for entries in by_field.values() for n, _ in entries]
    log.info("run: %d band(s) in one dispatch", len(all_bands))
    run_step(["submit.py", "run", *all_bands, *common, "--no-wait"], args.dry_run)

    log.info("submitted. watch with:  python submit.py status")


if __name__ == "__main__":
    main()
