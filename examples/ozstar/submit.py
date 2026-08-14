#!/usr/bin/env python
"""Drive mophongo runs on OzStar (Ngarrgu Tindebeek) over ssh and SLURM.

OzStar is the opposite of CANFAR in the two ways that shape this file. Compute
is ordinary ssh and ``sbatch`` rather than a REST API, so jobs chain with
``--dependency`` instead of a laptop-side process blocking on each stage. And
the MINERVA data are not there: everything a config names is copied from CANFAR
arc onto ``/fred`` first, by a job on the ``datamover`` partition, because
ordinary compute nodes have no internet at all.

    python submit.py cert                    # push the CADC proxy certificate
    python submit.py setup                   # clone mophongo, build the venv
    python submit.py sync                    # git pull, leaving the venv alone
    python submit.py push  uds_f770w ...     # upload job scripts and configs
    python submit.py stage uds_f770w ...     # datamover job: arc -> /fred/data
    python submit.py run   uds_f770w ...     # one SLURM job per config
    python submit.py status                  # the queue
    python submit.py logs  uds_f770w         # tail a job's log
    python submit.py fetch uds_f770w         # pull the small outputs down
    python submit.py cancel                  # scancel the campaign

``stage`` and ``run`` print the job ids they submit and accept ``--after``, so a
campaign is a dependency graph submitted in one go and nothing afterwards
depends on the laptop staying awake.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path

import ozroot

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("submit")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                     # mophongo/
# This directory is staging only - launch scripts and configs, all tracked and
# all small. Anything a command produces goes under scratch/, which is
# gitignored: outputs must never land in the staging tree.
WORK = REPO / "scratch" / "ozstar"
JOB_PREFIX = "moph"

#: Staging needs internet, which only these nodes have. It has to be given on
#: the sbatch command line: a site plugin reassigns the partition of a job that
#: names it in a #SBATCH directive alone, and the job then lands on a compute
#: node that cannot reach CANFAR. Fits leave the partition to the scheduler.
STAGE_PARTITION = "datamover"

#: Cores per fit. The fitting path is serial - a full-field run measured 6.1%
#: of 16 cores, i.e. about one - so these buy threaded BLAS in a few phases and
#: queue position everywhere else. Eight keeps some headroom for the dense
#: scene solves without paying 16 cores of fair-share for idle allocation.
DEFAULT_CORES = 8

#: GB per fit, by field. Peak memory is set by the detection grid and segmap,
#: which are per *field*, not per band: the four UDS bands measured 53.3, 55.6
#: and 57.4 GB, a spread of 4 GB. 72 GB leaves the worst of those at 80% with
#: about 15 GB of headroom, and still fits every milan (256 GB) and skylake
#: (191 GB) node, so it costs no scheduling reach over 64. EGS's detection grid
#: is about 1.4x UDS's, hence the larger request there. A run that exceeds its
#: request is killed with no Python traceback, so if a band dies without one,
#: check `sacct -j <id>.batch --format=MaxRSS` before anything else.
DEFAULT_MEM_GB = 72
MEM_GB_BY_FIELD = {"egs": 96}


def mem_for(name: str, override: int | None = None) -> int:
    """GB to request for one config."""
    if override is not None:
        return override
    return MEM_GB_BY_FIELD.get(name.split("_")[0], DEFAULT_MEM_GB)

#: Outputs small enough to be worth having locally. The residual, the stamps
#: and the scene PNGs stay on /fred; a full field writes tens of GB of them.
FETCH = ["{name}_fit_table.fits", "{name}_scene_catalog.csv", "{name}.log"]


# --- plumbing ---------------------------------------------------------------


def ssh(command: str, check: bool = True) -> str:
    """Run one command on the login node and return its stdout."""
    proc = subprocess.run(["ssh", ozroot.ssh_target(), command],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"ssh failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def ssh_stream(command: str) -> int:
    """Run one command on the login node with its output on the terminal."""
    return subprocess.run(["ssh", ozroot.ssh_target(), command]).returncode


def job_env(extra: dict[str, str] | None = None) -> str:
    """``VAR=value ...`` prefix for a job script run directly on the login node."""
    env = {"BASE": ozroot.base_root(), "RUN": ozroot.run_root(),
           "CFGDIR": ozroot.config_dir(), "DATA": ozroot.data_dir(),
           "PSFDIR": ozroot.psf_dir(), "BIN": ozroot.bin_dir(),
           "SRC": ozroot.src_dir(), "VENV": ozroot.venv_dir(),
           "VOS": ozroot.vos_dir(), "STPSF": ozroot.stpsf_dir(),
           **(extra or {})}
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def upload(paths: list[Path], dest: str) -> None:
    """Copy files to an absolute destination on the cluster with scp.

    Absolute rather than relative to the run tree: job scripts go to the shared
    ``bin/`` and configs to ``<run>/config/``, which are no longer siblings.

    scp rather than rsync: macOS ships openrsync, which does not have all of
    rsync's flags, and everything uploaded here is small - the mophongo source
    is cloned on the login node, not pushed.
    """
    if not paths:
        return
    ssh(f"mkdir -p {shlex.quote(dest)}")
    proc = subprocess.run(["scp", "-q", *[str(p) for p in paths],
                           f"{ozroot.ssh_target()}:{dest}/"])
    if proc.returncode != 0:
        raise SystemExit(f"scp failed ({proc.returncode})")


def sbatch(script: str, job_name: str, env: dict[str, str],
           after: str | None = None, extra: list[str] | None = None,
           log_dir: str | None = None) -> str:
    """Submit one job and return its id.

    Parameters go through ``--export`` rather than being baked into the script,
    so one script serves every config.

    ``log_dir`` is where the job's stdout lands, and for a fit it defaults to
    that band's ``out_dir``: the SLURM log belongs with the products it
    produced, alongside the pipeline's own ``<name>.log``, so a result and the
    record of how it was computed travel together. Jobs with no single output
    directory - staging a whole field, building grids - fall back to
    ``<run>/logs``. SLURM will not create the directory, so it is made here.
    """
    run = ozroot.run_root()
    exports = ",".join(f"{k}={v}" for k, v in
                       {"BASE": ozroot.base_root(), "RUN": run,
                        "CFGDIR": ozroot.config_dir(), "DATA": ozroot.data_dir(),
                        "PSFDIR": ozroot.psf_dir(), "BIN": ozroot.bin_dir(),
                        "SRC": ozroot.src_dir(), "VENV": ozroot.venv_dir(),
                        "VOS": ozroot.vos_dir(),
                        "STPSF": ozroot.stpsf_dir(), **env}.items())
    dest = log_dir or f"{run}/logs"
    flags = ["--parsable", f"--job-name={job_name}",
             f"--output={dest}/{job_name}-%j.out",
             f"--export=ALL,{exports}"]
    if after:
        flags.append(f"--dependency=afterok:{after}")
    flags += extra or []
    command = (f"mkdir -p {shlex.quote(dest)} && cd {shlex.quote(run)} && sbatch "
               + " ".join(shlex.quote(f) for f in flags)
               + f" {shlex.quote(ozroot.bin_dir())}/{script}")
    jobid = ssh(command).strip().split(";")[0]
    log.info("%-30s %s%s", job_name, jobid, f"  (after {after})" if after else "")
    return jobid


def by_field(names: list[str]) -> dict[str, list[str]]:
    """Group config names by field, preserving order."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.split("_")[0], []).append(name)
    return groups


# --- steps (importable by campaign.py, which needs the job ids) --------------


#: PSF grid patterns worth shipping when they already exist locally: the F444W
#: photometry grids, the 30" halo grids `repair_saturated` uses, and the
#: per-band MIRI grids.
PSF_GLOBS = ["*_NRC*_F444W_MJD*_GRID25_OS4.fits",
             "*_NRC*_F444W_MJD*_FOV30_GRID1_OS4.fits",
             "*_MIRI_*_MJD*_GRID9_OS4.fits"]


def push(names: list[str], psf_globs: list[str] | None = None) -> None:
    """Upload the job scripts and, if named, the rewritten configs.

    The mophongo source is not uploaded: ``setup`` clones it from GitHub on the
    login node, which reaches it far faster than a laptop uplink does.

    ``psf_globs`` additionally ships local ePSF grids. That is an optimisation,
    not a requirement - ``bin/build_psfs.sh`` builds whatever is missing - but
    building a grid needs a MAST query for the wavefront OPD and so can only
    happen on the login node, one field at a time. Grids that already exist on
    a laptop are much cheaper to copy than to rebuild.
    """
    # files only: importing a job module leaves a __pycache__ directory here,
    # and scp refuses a directory without -r
    scripts = sorted(p for p in (HERE / "jobs").glob("*") if p.is_file())
    upload(scripts, ozroot.bin_dir())
    ssh(f"chmod +x {shlex.quote(ozroot.bin_dir())}/*")
    files: list[Path] = []
    for name in names:
        for suffix in (f"{name}_ozstar.json", f"{name}_stage.tsv"):
            path = HERE / suffix
            if not path.exists():
                raise SystemExit(f"missing {path}; run ozify.py first")
            files.append(path)
    upload(files, ozroot.config_dir())
    log.info("pushed %d job script(s), %d config file(s)", len(scripts), len(files))
    if psf_globs:
        push_psf_grids(psf_globs)


def push_psf_grids(psf_globs: list[str]) -> None:
    """Ship local ePSF grids as one tarball.

    Streamed through a single ssh rather than scp'd file by file: the grids are
    a few hundred files of a few MB each, and per-file round trips over this
    link ran at about four files a minute against a couple of minutes for the
    whole tar. Extraction overwrites, so a partial earlier copy is repaired
    rather than left truncated.
    """
    psf_dir = HERE.parent.parent / "data" / "PSF"
    grids = sorted({g for pat in psf_globs for g in psf_dir.glob(pat)})
    if not grids:
        raise SystemExit(f"no PSF grids matched {psf_globs} under {psf_dir}")
    total = sum(g.stat().st_size for g in grids)
    dest = ozroot.psf_dir()
    ssh(f"mkdir -p {shlex.quote(dest)}")
    log.info("shipping %d PSF grid(s), %.0f MB", len(grids), total / 1e6)
    tar = subprocess.Popen(
        ["tar", "-cf", "-", "-C", str(psf_dir), *[g.name for g in grids]],
        stdout=subprocess.PIPE)
    unpack = subprocess.run(
        ["ssh", ozroot.ssh_target(), f"tar -xf - -C {shlex.quote(dest)}"],
        stdin=tar.stdout)
    tar.stdout.close()
    if tar.wait() != 0 or unpack.returncode != 0:
        raise SystemExit("PSF grid transfer failed")
    log.info("PSF grids on %s: %s", dest,
             ssh(f"ls {shlex.quote(dest)} | wc -l").strip())


def build_psfs(names: list[str]) -> str:
    """Start the login-node ePSF build and return its log path.

    Not a SLURM job, and it cannot become one: stpsf resolves each exposure's
    date to a wavefront OPD by querying MAST, compute nodes have no DNS or
    route, and datamover nodes (which do have internet) have no module tree to
    run mophongo from.

    Detached rather than streamed. The build runs for hours, and the first
    attempt died partway through when the driving ssh dropped and the remote
    bash took the SIGHUP with it. Poll the returned log with ``psf-log``.
    """
    script = f"{ozroot.bin_dir()}/build_psfs.sh"
    out = ssh(f"{job_env({'CFGS': ' '.join(names)})} bash {shlex.quote(script)}")
    log.info("%s", out.strip())
    for line in out.splitlines():
        if line.startswith("LOG="):
            return line[len("LOG="):].strip()
    raise SystemExit(f"PSF build did not report a log path: {out.strip()}")


def stage(names: list[str], after: str | None = None,
          walltime: str = "24:00:00") -> list[str]:
    """One datamover job per field; returns the job ids.

    Per field, not per band: the bands of a field share the F444W mosaic, its
    weight map and the segmap, several GB each that would otherwise cross the
    Pacific once per band. Within a job the destination list is deduplicated,
    and files already present are skipped, so resubmitting after a timeout
    resumes rather than restarts.
    """
    return [sbatch("stage.sh", f"{JOB_PREFIX}-stage-{field}",
                   {"CFGS": " ".join(bands)}, after=after,
                   extra=[f"--partition={STAGE_PARTITION}", f"--time={walltime}"])
            for field, bands in by_field(names).items()]


def run(names: list[str], after: str | None = None, cores: int = DEFAULT_CORES,
        mem: int | None = None, walltime: str = "24:00:00", sync: bool = True,
        step: str = "all") -> list[str]:
    """One SLURM job per config; returns the job ids.

    ``mem`` is GB and overrides the per-field default; leave it None so each
    field gets :func:`mem_for`, since peak memory follows the field's detection
    grid rather than the band.

    ``step`` is the pipeline step. ``"repair"`` runs a field's saturation
    repair into its shared cache and stops, which is what a campaign submits
    once per field before the bands.

    The source is pulled to the latest ``main`` first, unless ``sync`` is off.
    A run must not silently use whatever happened to be cloned weeks ago, and
    the pull is a second on the login node. It is safe mid-campaign: mophongo
    is installed editable, so running jobs keep the code they imported and only
    jobs that start afterwards pick this up.
    """
    if sync:
        sync_src()
    # the step goes in the job name so squeue can tell a field's prep job from
    # the band fits that follow it
    prefix = JOB_PREFIX if step == "all" else f"{JOB_PREFIX}-{step}"
    return [sbatch("run.slurm", f"{prefix}-{name.replace('_', '-')}",
                   {"CFG": name, "STEP": step}, after=after,
                   extra=[f"--cpus-per-task={cores}",
                          f"--mem={mem_for(name, mem)}g",
                          f"--time={walltime}"],
                   log_dir=ozroot.out_dir(name))
            for name in names]


def sync_src(branch: str = "main") -> None:
    """Pull the run tree's mophongo clone to the head of ``branch``."""
    script = f"{ozroot.bin_dir()}/sync_src.sh"
    code = ssh_stream(f"{job_env({'BRANCH': branch})} bash {shlex.quote(script)}")
    if code != 0:
        raise SystemExit(f"sync failed ({code})")


# --- commands ---------------------------------------------------------------


def do_cert(args: argparse.Namespace) -> None:
    """Copy the local CADC proxy certificate to OzStar.

    ``stage`` reads arc with the vos tools, which authenticate with this
    certificate and nothing else. It is valid for ten days, so a campaign that
    outlives one needs this again; the symptom is a staging job that fails
    immediately with a permission error.
    """
    cert = Path.home() / ".ssl/cadcproxy.pem"
    if not cert.exists():
        raise SystemExit(f"no certificate at {cert}; run scratch/canfar/canfar-cert.sh")
    ssh("mkdir -p ~/.ssl")
    proc = subprocess.run(["scp", "-q", str(cert),
                           f"{ozroot.ssh_target()}:.ssl/cadcproxy.pem"])
    if proc.returncode != 0:
        raise SystemExit("scp of the certificate failed")
    ssh("chmod 600 ~/.ssl/cadcproxy.pem")
    log.info("certificate installed, %s",
             ssh("openssl x509 -in ~/.ssl/cadcproxy.pem -noout -enddate").strip())


def do_push(args: argparse.Namespace) -> None:
    push(args.names, PSF_GLOBS if args.psf else None)


def do_psf(args: argparse.Namespace) -> None:
    build_psfs(args.names)


def do_setup(args: argparse.Namespace) -> None:
    """Clone mophongo and build the venv, on the login node.

    Not a SLURM job: compute nodes reach neither GitHub nor PyPI.
    """
    extra = {"BRANCH": args.branch}
    if args.rebuild:
        extra["REBUILD"] = "1"
    script = f"{ozroot.bin_dir()}/setup_env.sh"
    code = ssh_stream(f"{job_env(extra)} bash {shlex.quote(script)}")
    if code != 0:
        raise SystemExit(f"setup failed ({code})")


def do_sync(args: argparse.Namespace) -> None:
    """git pull the source in place; the venv and queued jobs are untouched."""
    sync_src(args.branch)


def do_seed(args: argparse.Namespace) -> None:
    """Link cached PSF/kernel maps from one run into another."""
    pairs = ",".join(f"{src}:{dst}" for src, dst in
                     (p.split(":", 1) for p in args.pairs))
    script = f"{ozroot.bin_dir()}/seed_cache.sh"
    ssh_stream(f"{job_env({'PAIRS': pairs})} bash {shlex.quote(script)}")


def do_stage(args: argparse.Namespace) -> None:
    stage(args.names, args.after, args.time)


def do_run(args: argparse.Namespace) -> None:
    run(args.names, args.after, args.cores, args.mem, args.time,
        sync=not args.no_sync, step=getattr(args, "step", "all"))


def push_arc(srcdir: str, dest: str, jobs: int = 6, compress: bool = False,
             dryrun: bool = False, walltime: str = "24:00:00") -> str:
    """Submit the datamover job that pushes a directory to CANFAR arc.

    A SLURM job, so it outlives the laptop, the ssh session and the terminal -
    which is the whole point for a transfer measured in hours.
    """
    return sbatch("push_arc.sh", f"{JOB_PREFIX}-push",
                  {"SRCDIR": srcdir, "DEST": dest, "JOBS": str(jobs),
                   "COMPRESS": "1" if compress else "0",
                   "DRYRUN": "1" if dryrun else "0"},
                  extra=[f"--partition={STAGE_PARTITION}", f"--time={walltime}"])


def do_push_arc(args: argparse.Namespace) -> None:
    push_arc(args.srcdir, args.dest, args.jobs, args.compress, args.dry_run,
             args.time)


def do_status(args: argparse.Namespace) -> None:
    fmt = "%.12i %.30j %.10P %.9T %.11M %.11l %.5C %.9m %R"
    print(ssh(f"squeue -u {ozroot.user()} -o {shlex.quote(fmt)}").rstrip())
    if args.done:
        print(ssh("sacct -X --format=JobID,JobName%30,State,Elapsed,MaxRSS,ReqMem "
                  f"-S {shlex.quote(args.done)}").rstrip())


def do_logs(args: argparse.Namespace) -> None:
    """Tail the newest log for a run name, a job name or a job id.

    Run names carry underscores (``uds_f770w``) and job names dashes
    (``moph-uds-f770w``), so both spellings are tried.
    """
    # A fit's log sits in its out_dir; jobs with no single output directory
    # (staging, grid builds) still land in <run>/logs, so search both.
    run = ozroot.run_root()
    roots = [f"{run}/*/*", f"{run}/logs"]
    globs = " ".join(f"{r}/*{p}*"
                     for r in roots
                     for p in dict.fromkeys([args.name, args.name.replace("_", "-")]))
    cmd = (f"f=$(ls -t {globs} 2>/dev/null | head -1); "
           f'[ -n "$f" ] || {{ echo "no log matching {args.name} under {run}"; exit 1; }}; '
           f'echo "--- $f"; tail -n {args.lines} "$f"')
    ssh_stream(cmd)


def do_fetch(args: argparse.Namespace) -> None:
    """Pull the small outputs down; the residual and the stamps stay on /fred."""
    root = ozroot.run_root()
    for name in args.names:
        dest = WORK / "out" / name
        dest.mkdir(parents=True, exist_ok=True)
        got = 0
        for template in FETCH:
            remote = f"{ozroot.out_dir(name)}/{template.format(name=name)}"
            proc = subprocess.run(["scp", "-q", f"{ozroot.ssh_target()}:{remote}",
                                   str(dest)], capture_output=True, text=True)
            if proc.returncode == 0:
                got += 1
            else:
                log.warning("  not on /fred: %s", Path(remote).name)
        log.info("%s -> %s (%d/%d files)", name, dest, got, len(FETCH))


def do_cancel(args: argparse.Namespace) -> None:
    """Cancel this campaign's jobs by name prefix, not the whole account."""
    rows = ssh(f"squeue -u {ozroot.user()} -h -o '%i %j'").splitlines()
    ids = [r.split()[0] for r in rows
           if len(r.split()) > 1 and r.split()[1].startswith(args.match)]
    if not ids:
        log.info("nothing queued matching %s", args.match)
        return
    ssh(f"scancel {' '.join(ids)}")
    log.info("cancelled %d job(s)", len(ids))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cert", help="copy the CADC proxy certificate to OzStar")
    p.set_defaults(func=do_cert)

    p = sub.add_parser("push", help="upload job scripts and rewritten configs")
    p.add_argument("names", nargs="*")
    p.add_argument("--psf", action="store_true",
                   help="also ship the local ePSF grids, saving a login-node build")
    p.set_defaults(func=do_push)

    p = sub.add_parser("psf", help="build missing ePSF grids on the login node")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_psf)

    p = sub.add_parser("setup", help="clone mophongo and build the venv")
    p.add_argument("--branch", default="main")
    p.add_argument("--rebuild", action="store_true",
                   help="delete and rebuild the venv (breaks jobs using it)")
    p.set_defaults(func=do_setup)

    p = sub.add_parser("sync", help="git pull the source, leaving the venv alone")
    p.add_argument("--branch", default="main")
    p.set_defaults(func=do_sync)

    p = sub.add_parser("seed", help="link cached PSF/kernel maps between runs")
    p.add_argument("pairs", nargs="+", metavar="SRC:DST")
    p.set_defaults(func=do_seed)

    p = sub.add_parser("stage", help="copy a field's inputs from arc to /fred")
    p.add_argument("names", nargs="+")
    p.add_argument("--after", help="SLURM job id to depend on")
    p.add_argument("--time", default="24:00:00")
    p.set_defaults(func=do_stage)

    p = sub.add_parser("run", help="one SLURM job per config")
    p.add_argument("names", nargs="+")
    p.add_argument("--step", default="all",
                   help="pipeline step (default: all). 'repair' runs the "
                        "field's saturation repair into its shared cache and "
                        "stops, so the bands that follow start warm")
    p.add_argument("--after", help="SLURM job id to depend on")
    p.add_argument("--cores", type=int, default=DEFAULT_CORES)
    p.add_argument("--mem", type=int, default=None,
                   help=f"GB; default is per field ({DEFAULT_MEM_GB}, "
                        + ", ".join(f"{k} {v}" for k, v in MEM_GB_BY_FIELD.items()) + ")")
    p.add_argument("--time", default="24:00:00")
    p.add_argument("--no-sync", action="store_true",
                   help="do not pull the latest main first (it is pulled by default, "
                        "so a run never uses a stale clone)")
    p.set_defaults(func=do_run)

    p = sub.add_parser("push-arc", help="datamover job: push a directory to CANFAR arc")
    p.add_argument("srcdir", help="local directory on /fred")
    p.add_argument("dest", help="arc: URI to push into")
    p.add_argument("--jobs", type=int, default=6,
                   help="parallel streams; 6 measured 14 MB/s against 1.25 for one")
    p.add_argument("--compress", action="store_true",
                   help="gzip before sending; pays on zero-heavy products "
                        "(residuals, stamps), not on dense ePSF grids (1.1x)")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent")
    p.add_argument("--time", default="24:00:00")
    p.set_defaults(func=do_push_arc)

    p = sub.add_parser("status", help="the queue")
    p.add_argument("--done", metavar="YYYY-MM-DD",
                   help="also list finished jobs since this date, with MaxRSS")
    p.set_defaults(func=do_status)

    p = sub.add_parser("logs", help="tail the newest log matching a name")
    p.add_argument("name")
    p.add_argument("-n", "--lines", type=int, default=60)
    p.set_defaults(func=do_logs)

    p = sub.add_parser("fetch", help="download the small outputs")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_fetch)

    p = sub.add_parser("cancel", help="cancel this campaign's queued jobs")
    p.add_argument("--match", default=JOB_PREFIX)
    p.set_defaults(func=do_cancel)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
