#!/usr/bin/env python
"""Launch a whole mophongo campaign on OzStar in one command.

Wraps the individual steps - config rewrite, upload, environment, grids,
staging, repair, run - so every band of every MINERVA field can be started at
once::

    python campaign.py                          # every config in ../minerva
    python campaign.py --fields uds             # only that field
    python campaign.py --bands f770w            # only that band
    python campaign.py --from stage             # source and configs already up
    python campaign.py --skip psf repair        # grids and caches in place
    python campaign.py --dry-run                # print the plan, submit nothing

The step names, the filters and the flags match ``examples/canfar/campaign.py``,
so one campaign reads the same way on either platform. What differs is who
enforces the order between the phases.

The submission is three phases, the same split CANFAR uses:

1. ``psf`` - every ePSF grid the configs name, built on the login node,
   **waited on**. Skipped when the grids are already on ``/fred``, which is the
   usual case once a release has been fitted once.
2. ``repair`` - one job per field, each depending on that field's staging. The
   saturation repair depends on the detection band alone, so one run of it
   fills the cache every band of the field then reloads.
3. ``run`` - every band of every field, each behind its field's repair.

The first two phases exist so the third needs no order at all. Everything they
produce is shared by a field's bands and depends on the detection band alone,
so without them each band rebuilds the same grids, re-runs the same repair, and
several of them write one cache file at the same time.

Where this differs from CANFAR:

* Phases 2 and 3 are **submitted, not waited on**. SLURM has dependencies, so
  the whole campaign goes up as one graph - staging a field is a ``datamover``
  job, its repair waits on that with ``--dependency=afterok``, and the band fits
  wait on the repair. On CANFAR the laptop has to block between phases because
  skaha sessions cannot depend on each other; here nothing after the submission
  depends on the laptop.
* Phase 1 *is* waited on, because it alone cannot be a SLURM job. stpsf resolves
  each exposure's date to a wavefront OPD by querying MAST, and compute nodes
  have no route to it, so the build is a detached process on the login node and
  nothing can be made to depend on it.
* ``afterok`` means a failed stage leaves its fits queued as
  ``DependencyNeverSatisfied`` until SLURM cancels them - which is the wanted
  behaviour, but reads as jobs silently disappearing if you do not know it.

The source is pulled once, before anything is submitted, and checked against
``--ref``. One pull rather than one per dispatch: a campaign has to run one
version of the code, not whatever ``main`` happened to be as each job went up.

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
REPO = HERE.parent.parent
PSF_DIR = REPO / "data" / "PSF"
PYTHON = sys.executable
STEPS = ["ozify", "push", "setup", "seed", "psf", "stage", "repair", "run"]

#: ``prep`` was this campaign's name for what CANFAR calls ``repair``, and both
#: ran the same pipeline step. Accepted so existing scripts keep working.
STEP_ALIASES = {"prep": "repair"}

#: A per-band run config is named ``<field>_f<band>w.json``. The directory also
#: holds inputs that are not RunConfigs (``minerva_sed_fields.json``), which
#: would fail deep inside ozify with a confusing key error.
BAND_CONFIG = re.compile(r"^[a-z0-9]+_f\d+w$")


#: Band a field's repair job runs. The repair depends on the detection band
#: alone, so any band of the field produces the same cache. F770W by
#: preference: it is the shortest MIRI band, so its own lo-res grids and fit are
#: the cheapest way to get the shared products built. Falls back to whichever
#: band sorts first when a field has no F770W.
PREP_BAND = "f770w"


def step_name(value: str) -> str:
    """Canonical step name, accepting the deprecated ``prep`` for ``repair``."""
    name = STEP_ALIASES.get(value, value)
    if name not in STEPS:
        raise argparse.ArgumentTypeError(
            f"unknown step {value!r}; choose from {', '.join(STEPS)}")
    return name


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

    The remote listing is the one that counts: the run reads
    ``<base>/PSF`` on ``/fred``, and a grid built there by an earlier campaign
    is present while the laptop has never held it. Local grids only count when
    ``--push-psf`` is going to ship them, which is where the caller adds them.

    A missing or unreadable directory lists as empty, which costs a grid build
    that finds nothing to do and nothing else.
    """
    return submit.ssh(f"ls {ozroot.run_root()}/PSF 2>/dev/null", check=False).split()


def grid_patterns(cfg_path: Path, shared_only: bool) -> list[str]:
    """The ePSF grid families one config needs, as regexes.

    ``shared_only`` drops ``pattern_lo``. Every band of a field matches the same
    ``pattern_hi`` grids, so bands building those at once write the same
    filenames into one ``psf_dir``; the per-band MIRI grids of ``pattern_lo``
    have distinct names and cannot race. So the leader decision asks only about
    the shared families, while the ``psf`` step asks about all of them.
    """
    cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
    pattern = cfg.get("pattern_hi")
    if not pattern:
        return []
    patterns = [pattern]
    if not shared_only and cfg.get("pattern_lo"):
        patterns.append(cfg["pattern_lo"])
    if cfg.get("repair_saturated"):
        # same derivation as Pipeline._repair_halo_pattern; the 30" halo grids
        # are shared by a field's bands exactly as pattern_hi is
        patterns.append(cfg.get("repair_psf_pattern")
                        or re.sub(r"_MJD.*$", r"_MJD\\d+_FOV30_GRID1_OS4", pattern))
    # Grid filenames carry an _FOV token that the configs' patterns do not, so
    # match the way the loader does (mophongo.jwst_psf.fov_agnostic_pattern is
    # the source of truth; inlined to keep this script free of the package).
    return [p if ("_FOV" in p or "_GRID" not in p)
            else p.replace("_GRID", r"(?:_FOV\d+)?_GRID", 1)
            for p in patterns]


def has_grids(cfg_path: Path, names: list[str], shared_only: bool = False) -> bool:
    """Whether every grid family this config needs is already present.

    A family counts as present when *anything* matches it. That is deliberately
    weak: it cannot tell a complete set of epochs from one grid, which is what
    ``psf_provenance`` is for. It exists to skip a build that would otherwise
    spend an hour discovering there is nothing to do.
    """
    return all(any(re.search(pat, n) for n in names)
               for pat in grid_patterns(cfg_path, shared_only))


def previous_run(name: str) -> str | None:
    """The run directory before this one, when the name ends in a number."""
    match = re.match(r"^(.*?)(\d+)$", name)
    if not match or int(match.group(2)) <= 1:
        return None
    return f"{match.group(1)}{int(match.group(2)) - 1}"


def write_run_readme(names: list[str], cfgs: list[Path], note: str,
                     dry: bool) -> None:
    """Write ``<run>/README.md``: what this run is, and how it differs.

    A run directory is the unit of versioning here, and a number says nothing
    about why it exists. This records the three things that actually change
    between runs -- the mophongo commit, the release versions the configs are
    pinned to, and whatever the person launching it types as ``--note`` -- and
    diffs the versions against the previous run's README so the difference is
    stated rather than reconstructed months later from file dates.

    Written before the work is submitted, so a run that dies halfway still says
    what it was trying to do. The copy under ``scratch/ozstar`` is kept as well
    as uploaded, since reading it back needs an ssh.
    """
    # The arc index and the version parser are the same problem on both
    # platforms, so canfar/arcify.py is imported rather than copied - the same
    # trick, and the same reason, as ozify.py.
    sys.path.append(str(HERE.parent / "canfar"))
    from arcify import config_versions

    run = ozroot.run_name()
    versions: dict[str, dict[str, str]] = {}
    for name, cfg_path in zip(names, cfgs):
        # the source config, not the rewritten one: ozify flattens the staged
        # inputs into data/, which drops the version directory the release is
        # actually named for
        cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
        versions[name.split("_")[0]] = config_versions(cfg)

    try:
        commit = submit.git("rev-parse", "--short", "HEAD")
    except SystemExit:  # a README is not worth failing a launch over
        commit = "unknown"

    lines = [f"# {run}", ""]
    if note:
        lines += [note, ""]
    lines += [f"- mophongo: `{commit}`",
              f"- bands: {len(names)} ({', '.join(sorted(names))})", "",
              "## Release versions", "",
              "| field | nircam | miri | catalog |", "|---|---|---|---|"]
    for field in sorted(versions):
        v = versions[field]
        lines.append(f"| {field} | {v.get('nircam', '-')} | {v.get('miri', '-')} "
                     f"| {v.get('catalog', '-')} |")

    # what changed against the run before, read from its own README
    before = previous_run(run)
    if before and not dry:
        previous = submit.ssh(
            f"cat {ozroot.base_root()}/{before}/README.md 2>/dev/null", check=False)
        if previous:
            changed = [ln for ln in lines if ln.startswith("| ")
                       and ln not in previous and not ln.startswith("| field")]
            lines += ["", f"## Against {before}", ""]
            lines += ([f"- changed: `{ln.strip()}`" for ln in changed]
                      or ["- same release versions"])

    text = "\n".join(lines) + "\n"
    if dry:
        log.info("%s README (dry run):\n%s", run, text)
        return
    submit.WORK.mkdir(parents=True, exist_ok=True)
    local = submit.WORK / "README.md"
    local.write_text(text)
    submit.upload([local], ozroot.run_root())
    log.info("wrote %s/README.md", ozroot.run_root())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", help="restrict to these, e.g. uds cosmos")
    ap.add_argument("--bands", nargs="+", help="restrict to these, e.g. f770w")
    ap.add_argument("--from", dest="start", type=step_name, default="ozify",
                    metavar="STEP",
                    help="skip everything before this step (%s)" % ", ".join(STEPS))
    ap.add_argument("--skip", nargs="+", type=step_name, default=[], metavar="STEP",
                    help="drop these steps from the middle of the chain, e.g. "
                         "--skip stage when the inputs are already on /fred")
    ap.add_argument("--cores", type=int, default=submit.DEFAULT_CORES)
    ap.add_argument("--mem", "--ram", dest="mem", type=int, default=None,
                    help="GB per run; default %d for every field%s" % (
                        submit.DEFAULT_MEM_GB,
                        "" if not submit.MEM_GB_BY_FIELD else
                        " except " + ", ".join(f"{k} {v}" for k, v in
                                               submit.MEM_GB_BY_FIELD.items())))
    ap.add_argument("--time", default="24:00:00", help="walltime per run")
    ap.add_argument("--repair-time", "--prep-time", dest="repair_time",
                    default="4:00:00",
                    help="walltime for the per-field repair job; it reads the "
                         "detection mosaic and fits the saturated cores, so it "
                         "is far shorter than a fit but not instant")
    ap.add_argument("--stage-time", default="24:00:00", help="walltime per stage job")
    # One knob, two spellings: the ref `setup` clones and the ref every job is
    # checked against have to be the same thing, and two flags that could
    # disagree would mean a campaign refusing to submit the branch it just
    # cloned. `--branch` is the older name.
    ap.add_argument("--ref", "--branch", dest="ref", default="main",
                    help="mophongo ref to clone, and the one the clone on /fred "
                         "must match before anything is submitted (default: main)")
    ap.add_argument("--force-stale", action="store_true",
                    help="submit even though the clone is not that ref")
    ap.add_argument("--push-psf", action="store_true",
                    help="ship the local ePSF grids with the configs; cheaper "
                         "than rebuilding the ones a laptop already has")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial radius in arcmin; 0 runs the full field")
    ap.add_argument("--suffix", default="",
                    help="append to every run name, e.g. _trial, to keep outputs separate")
    ap.add_argument("--seed-from", default=None, metavar="SUFFIX",
                    help="seed PSF/kernel caches from the runs with this suffix "
                         "(use '' for the unsuffixed ones); skips rebuilding them")
    ap.add_argument("--note", default="",
                    help="one line for <run>/README.md saying what this run "
                         "changes; the run directory alone records nothing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [s for s in STEPS[STEPS.index(args.start):] if s not in args.skip]
    cfgs = configs_for(args.fields, args.bands)
    names = [c.stem + args.suffix for c in cfgs]
    log.info("campaign over %d config(s): %s", len(names), ", ".join(names))
    log.info("run tree %s on %s", ozroot.run_root(), ozroot.ssh_target())
    write_run_readme(names, cfgs, args.note, args.dry_run)

    if "ozify" in todo:
        ozify = ["ozify.py", *[str(c) for c in cfgs]]
        if args.r_trial is not None:
            ozify += ["--r-trial", str(args.r_trial)]
        if args.suffix:
            ozify += ["--suffix", args.suffix]
        run_step(ozify, args.dry_run)
    if "push" in todo:
        log.info("+ submit.push(%d configs)%s", len(names),
                 " with the local PSF grids" if args.push_psf else "")
        if not args.dry_run:
            submit.push(names, submit.PSF_GLOBS if args.push_psf else None)
    if "setup" in todo:
        run_step(["submit.py", "setup", "--branch", args.ref], args.dry_run)
    if "seed" in todo and args.seed_from is not None:
        pairs = [f"{c.stem}{args.seed_from}:{c.stem}{args.suffix}" for c in cfgs]
        run_step(["submit.py", "seed", *pairs], args.dry_run)

    # Phase 1: every ePSF grid the release needs, on the login node, waited on.
    # This is the one phase a SLURM dependency cannot cover - the build queries
    # MAST for each exposure's OPD and compute nodes have no route to it - so it
    # is a detached process the laptop polls. Skipped when the grids are all
    # there already, which is the usual case once a release has been fitted
    # once; the check reads /fred, not the laptop, since that is where the run
    # will look.
    if "psf" in todo:
        known = [] if args.dry_run else psf_names_on_fred()
        if args.push_psf:
            known += [p.name for p in PSF_DIR.glob("*.fits")]
        missing = [c.stem for c in cfgs if not has_grids(c, known)]
        if args.dry_run:
            log.info("+ psf: build the grids for %d config(s) and wait", len(cfgs))
        elif missing:
            log.info("psf: %d of %d config(s) still need grids (%s)",
                     len(missing), len(cfgs), ", ".join(missing[:4]))
            submit.wait_for_psf_build(submit.build_psfs(names))
        else:
            log.info("psf: every grid is on /fred already; skipping")

    stage_ids: dict[str, str] = {}
    if "stage" in todo:
        for field, bands in submit.by_field(names).items():
            if args.dry_run:
                log.info("+ stage %s: %s", field, " ".join(bands))
                # so the plan below shows the dependency it would really carry
                stage_ids[field] = f"<{field}-stage>"
                continue
            stage_ids[field] = submit.stage(bands, walltime=args.stage_time)[0]

    if not any(s in todo for s in ("repair", "run")):
        return

    # One pull and one check for the whole campaign, before any job goes up.
    # Per-dispatch pulls would let a commit landing mid-submission split the
    # campaign across two versions of the code, and the check is what catches a
    # local commit that was never pushed: the cluster pulls GitHub, so an
    # unpushed fix is simply absent while every output looks normal.
    if not args.dry_run:
        submit.sync_src(args.ref)
        submit.check_src_current(args.ref, args.force_stale)

    # Phase 2: one saturation-repair job per field, all fields at once, each
    # depending only on its own staging. It writes the field's shared repair
    # cache, which every band of that field then hits instead of re-running the
    # identical repair and racing to write the same file.
    prep_ids: dict[str, str] = {}
    if "repair" in todo:
        for field, bands in submit.by_field(names).items():
            dep = stage_ids.get(field)
            if args.dry_run:
                log.info("+ repair %s (%s) after %s", field,
                         prep_leader(bands), dep or "nothing")
                # so the plan below shows the dependency it would really carry
                prep_ids[field] = f"<{field}-repair>"
                continue
            prep_ids[field] = submit.run(
                [prep_leader(bands)], after=dep, cores=args.cores,
                mem=args.mem, walltime=args.repair_time, step="repair",
                sync=False, ref=args.ref, force_stale=args.force_stale)[0]

    if "run" not in todo:
        return

    # Phase 3: every band of every field. Nothing is shared that the phases
    # above have not already built, so there is no order left to keep except
    # each field's own dependency chain.
    on_fred = [] if args.dry_run else psf_names_on_fred()
    for field, bands in submit.by_field(names).items():
        cfg = next(c for c, n in zip(cfgs, names) if n == bands[0])
        # every band waits on its field's repair when there was one, else on
        # its staging
        dep = prep_ids.get(field) or stage_ids.get(field)
        # with neither phase run, fall back to the old one-band-first order so
        # that concurrent autobuilds of the shared grids do not race
        leader_needed = ("psf" not in todo and "repair" not in todo
                         and not has_grids(cfg, on_fred, shared_only=True)
                         and len(bands) > 1)
        if args.dry_run:
            log.info("+ run %s%s%s", " ".join(bands),
                     f" after {dep or 'nothing'}",
                     " (first band alone: no shared PSF grids yet)" if leader_needed else "")
            continue
        if leader_needed:
            log.info("%s: shared grids missing (F444W or the 30\" halo), "
                     "running %s alone to build them", field, bands[0])
            leader = submit.run([bands[0]], after=dep, cores=args.cores,
                                mem=args.mem, walltime=args.time, sync=False,
                                ref=args.ref, force_stale=args.force_stale)[0]
            submit.run(bands[1:], after=leader, cores=args.cores, mem=args.mem,
                       walltime=args.time, sync=False, ref=args.ref,
                       force_stale=args.force_stale)
        else:
            submit.run(bands, after=dep, cores=args.cores, mem=args.mem,
                       walltime=args.time, sync=False, ref=args.ref,
                       force_stale=args.force_stale)

    log.info("submitted. watch with:  python submit.py status")


if __name__ == "__main__":
    main()
