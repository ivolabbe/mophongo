#!/usr/bin/env python
"""Drive mophongo runs on the CANFAR Science Platform.

CANFAR compute is a REST API (`skaha`), not ssh: the transfer endpoint on port
64022 is SFTP only and cannot execute anything. Jobs are containers with `/arc`
mounted, so the MINERVA data are already there and nothing is uploaded except
the mophongo source and the PSF grids.

    python submit.py push                    # upload src + PSF grids (small)
    python submit.py setup                   # build the venv on /arc
    python submit.py stage  uds_f770w        # decompress that config's inputs
    python submit.py run    uds_f770w ...    # one job per config, concurrent
    python submit.py status                  # all sessions
    python submit.py logs   <session-id>
    python submit.py fetch  uds_f770w        # pull the small outputs down

Two API quirks are handled here: the installed client defaults to API v0, which
404s (hence ``version='v1'``), and ``args`` is whitespace-split into a YAML
sequence server side, so the command must be a single token. Parameters are
passed as environment variables instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from skaha.session import Session

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("submit")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                     # mophongo/


from runroot import run_root

RUN, RUN_VOS = run_root(REPO)   # /arc/... and its arc: URI
IMAGE = "images.canfar.net/skaha/jwst-notebook:25.07.25"
VCP = Path.home() / ".venvs/canfar/bin/vcp"
DONE = ("Succeeded", "Failed", "Completed", "Terminating")


def session() -> Session:
    return Session(version="v1")


def vcp(src: Path | str, dst: str, tries: int = 3) -> None:
    """Copy one file with ``vcp``. VOSpace flakes, so transient errors retry.

    A missing source is not transient: it raises ``FileNotFoundError`` at once
    rather than burning the retries.
    """
    for attempt in range(1, tries + 1):
        proc = subprocess.run([str(VCP), str(src), dst], capture_output=True, text=True)
        if proc.returncode == 0:
            return
        message = (proc.stderr or proc.stdout).strip()
        if "NodeNotFound" in message:
            raise FileNotFoundError(str(src))
        log.warning("  vcp attempt %d/%d failed (%d): %s", attempt, tries,
                    proc.returncode, message.splitlines()[-1:] or "")
        time.sleep(5)
    raise SystemExit(f"vcp failed after {tries} attempts: {src} -> {dst}")


def do_push(args: argparse.Namespace) -> None:
    """Upload the mophongo source and the PSF grids the configs reference.

    Only these two: every science input is already on arc. Tarred first because
    many small files over VOSpace are slow (about 3 MB/s).
    """
    tmp = HERE / "_upload"
    tmp.mkdir(exist_ok=True)

    src_tar = tmp / "mophongo_src.tgz"
    with tarfile.open(src_tar, "w:gz") as tar:
        for item in ["src", "pyproject.toml", "README.md"]:
            tar.add(REPO / item, arcname=item,
                    filter=lambda t: None if "__pycache__" in t.name else t)
    log.info("src  %.1f MB", src_tar.stat().st_size / 1e6)

    psf_tar = tmp / "psf.tar"
    grids = sorted(set(sum((list((REPO / "data" / "PSF").glob(pat))
                            for pat in args.psf_glob), [])))
    if not grids:
        raise SystemExit(f"no PSF grids matched {args.psf_glob}")
    with tarfile.open(psf_tar, "w") as tar:
        for grid in grids:
            tar.add(grid, arcname=grid.name)
    log.info("psf  %d grids, %.1f MB", len(grids), psf_tar.stat().st_size / 1e6)

    for path in (src_tar, psf_tar):
        log.info("uploading %s", path.name)
        vcp(path, f"{RUN_VOS}/{path.name}")

    subprocess.run([str(VCP.parent / "vmkdir"), f"{RUN_VOS}/jobs"], capture_output=True)
    for script in sorted((HERE / "jobs").glob("*.sh")):
        log.info("uploading jobs/%s", script.name)
        vcp(script, f"{RUN_VOS}/jobs/{script.name}")


def do_upload_cfg(names: list[str]) -> None:
    """Push the rewritten configs and their staging lists to arc."""
    for name in names:
        for suffix in (f"{name}_canfar.json", f"{name}_stage.tsv"):
            path = HERE / suffix
            if not path.exists():
                raise SystemExit(f"missing {path}; run arcify.py first")
            vcp(path, f"{RUN_VOS}/{suffix}")


def launch(name: str, script: str, cores: int, ram: int, env: dict[str, str]) -> str:
    """Start one headless job and return its session id."""
    ids = session().create(
        name=name, image=IMAGE, cores=cores, ram=ram, kind="headless",
        cmd="/bin/bash", args=f"{RUN}/jobs/{script}", env=env,
    )
    log.info("%-24s %s", name, ids[0] if ids else "FAILED")
    return ids[0] if ids else ""


def wait(ids: list[str], poll: int = 20, missing_tolerance: int = 3) -> dict[str, str]:
    """Block until every session reaches a terminal state.

    ``info`` returns an empty list both for a session the service has already
    reaped *and* for one just created that has not been registered yet, so an
    empty answer is only treated as terminal after several consecutive polls.
    Declaring it terminal on the first empty reply made wait() return instantly
    on a freshly submitted job.
    """
    pending = [i for i in ids if i]
    final: dict[str, str] = {}
    missing: dict[str, int] = {}
    while pending:
        for sid in list(pending):
            try:
                info = session().info([sid])
            except Exception as exc:  # noqa: BLE001 - transient API errors
                log.warning("  %s: info failed (%s), retrying", sid, type(exc).__name__)
                continue
            if not info:
                missing[sid] = missing.get(sid, 0) + 1
                if missing[sid] >= missing_tolerance:
                    final[sid] = "Gone"
                    pending.remove(sid)
                    log.info("%s Gone", sid)
                continue
            missing[sid] = 0
            status = info[0].get("status")
            if status in DONE:
                final[sid] = status
                pending.remove(sid)
                log.info("%s %s", sid, status)
        if pending:
            time.sleep(poll)
    return final


def first_log(ids: list[str]) -> str:
    """Log text of the first session that has any, or a note that none does."""
    for text in session().logs(ids).values():
        return text
    return "(no log available; the session may have been reaped)"


def do_setup(args: argparse.Namespace) -> None:
    # small footprint: this only unpacks and pip installs, and a modest
    # request schedules when the platform is busy
    ids = [launch("mophongo-setup", "setup_env.sh", 2, 8, {"RUN": RUN})]
    wait(ids)
    print(first_log(ids)[-2000:])


def session_name(*parts: str) -> str:
    """A skaha session name: alphanumerics and ``-`` only.

    Run names carry underscores and dots (``uds_f770w_v1.0``); skaha rejects
    both with a 400 that names the field but not the value, so normalise here
    rather than at each call site.
    """
    raw = "-".join(str(p) for p in parts if p)
    return re.sub(r"[^A-Za-z0-9-]+", "-", raw).strip("-")


def stage_job(name: str) -> str:
    return launch(session_name("mophongo-stage", name), "stage.sh", 2, 8,
                  {"RUN": RUN, "CFG": name})


def do_sync(args: argparse.Namespace) -> None:
    """Ship a code change without rebuilding the venv.

    ``setup`` deletes and recreates the venv, which breaks every run currently
    using it. mophongo is installed editable, so replacing the source is enough
    and is safe while other jobs are in flight.
    """
    ids = [launch("mophongo-sync", "update_src.sh", 1, 4, {"RUN": RUN})]
    wait(ids)
    print(tidy(first_log(ids), 10))


def do_seed(args: argparse.Namespace) -> None:
    """Copy cached PSF/kernel maps from one run into another.

    Those maps do not depend on the trial patch, so a full-field run can start
    from what a patch run of the same band already built instead of spending
    half an hour rebuilding them.
    """
    pairs = ",".join(f"{src}:{dst}" for src, dst in
                     (p.split(":", 1) for p in args.pairs))
    ids = [launch("mophongo-seed", "seed_cache.sh", 1, 4, {"RUN": RUN, "PAIRS": pairs})]
    wait(ids)
    print(tidy(first_log(ids), 25))


def do_stage(args: argparse.Namespace) -> None:
    """Stage inputs: one band per field first, then the rest concurrently.

    Bands of a field share that field's F444W mosaic and segmap, several GB
    each. Staging one band per field to completion first means every other band
    finds those already present rather than decompressing them again. The
    fields themselves share nothing, so their first bands run together.
    """
    do_upload_cfg(args.names)
    names = list(args.names)

    first: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()
    for name in names:
        field = name.split("_")[0]
        (rest if field in seen else first).append(name)
        seen.add(field)

    log.info("staging %d field leaders, then %d remaining bands",
             len(first), len(rest))
    ids = [stage_job(n) for n in first]
    wait(ids)
    if rest:
        later = [stage_job(n) for n in rest]
        wait(later)
        ids += later
    for sid, text in session().logs(ids).items():
        tail = tidy(text, 6)
        print(f"--- {sid}\n{tail}")


def tidy(text: str, lines: int = 40) -> str:
    """Drop tqdm progress spam, which otherwise buries the real output."""
    kept = [ln for ln in text.splitlines() if "it/s]" not in ln and "s/it]" not in ln]
    return "\n".join(kept[-lines:])


def cores_for(name: str, override: int | None) -> int:
    """Cores to request for one config.

    Two, for everything. Measured utilisation is about 0.2 of a core - the runs
    wait on ``/arc`` rather than compute, and the fitting path has no thread
    pool - so a larger request only idles allocation and takes longer to
    schedule when the platform is busy.
    """
    return 2 if override is None else override


def do_run(args: argparse.Namespace) -> None:
    do_upload_cfg(args.names)
    ids = []
    for name in args.names:
        cores = cores_for(name, args.cores)
        ids.append(launch(session_name("mophongo", name), "run.sh", cores, args.ram,
                          {"RUN": RUN, "CFG": name}))
    if args.no_wait:
        return
    final = wait(ids)
    for sid, text in session().logs(ids).items():
        status = final.get(sid)
        print(f"--- {sid} [{status}]\n{tidy(text)}")
        if status == "Failed" and "RUN_DONE" not in text:
            print("  note: a failure with no traceback is usually the container "
                  "being OOM-killed. A full-field run needs --ram 48; a run "
                  'with a "trial" patch reads only that patch and needs far '
                  "less.")


def do_status(args: argparse.Namespace) -> None:
    for info in session().fetch(kind="headless"):
        log.info("%-10s %-28s %s", info.get("id"), info.get("name"), info.get("status"))


def do_clean(args: argparse.Namespace) -> None:
    """Delete finished sessions so the listing stays readable.

    Only terminal ones: a Running job is never touched. Logs die with the
    session, so fetch anything you still want first.
    """
    sess = session()
    stale = [i["id"] for i in sess.fetch(kind="headless") if i.get("status") in DONE]
    if not stale:
        log.info("nothing to clean")
        return
    sess.destroy(stale)
    log.info("deleted %d finished session(s)", len(stale))


def do_logs(args: argparse.Namespace) -> None:
    for _, text in session().logs(args.ids).items():
        print(text)


def do_fetch(args: argparse.Namespace) -> None:
    """Pull the small outputs down; the multi-GB residuals stay on arc.

    Outputs are named after the run: ``<name>_fit_table.fits`` and ``<name>.log``.
    A file that is absent is reported and skipped rather than aborting the rest.
    """
    wanted = ["{name}_fit_table.fits", "{name}_scene_catalog.csv", "{name}.log"]
    for name in args.names:
        dest = HERE / "out" / name
        dest.mkdir(parents=True, exist_ok=True)
        got = 0
        for template in wanted:
            remote = f"{RUN_VOS}/out/{name}/{template.format(name=name)}"
            try:
                vcp(remote, str(dest))
                got += 1
            except FileNotFoundError:
                log.warning("  not on arc: %s", remote.rsplit("/", 1)[-1])
        log.info("%s -> %s (%d/%d files)", name, dest, got, len(wanted))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="upload mophongo source and PSF grids")
    # Every field's grids, not just UDS: a band with no matching grid falls back
    # to building them on the node, which is slow and, run concurrently by
    # several bands of a field, races on the same psf_dir. EGS has none locally
    # and has to build them regardless. The third glob is the 30" halo grid the
    # saturation repair uses to decide which segments a star dominates
    # (repair_saturated); without it the flag reach shrinks to the GRID25 field
    # of view and only the star's core gets flagged.
    p.add_argument("--psf-glob", nargs="+",
                   default=["*_NRC*_F444W_MJD*_GRID25_OS4.fits",
                            "*_NRC*_F444W_MJD*_FOV30_GRID1_OS4.fits",
                            "*_MIRI_*_MJD*_GRID9_OS4.fits"])
    p.set_defaults(func=do_push)

    p = sub.add_parser("setup", help="build the venv on /arc")
    p.set_defaults(func=do_setup)

    p = sub.add_parser("sync", help="replace the source in place, leaving the venv alone")
    p.set_defaults(func=do_sync)

    p = sub.add_parser("seed", help="copy cached PSF/kernel maps between runs")
    p.add_argument("pairs", nargs="+", metavar="SRC:DST")
    p.set_defaults(func=do_seed)

    p = sub.add_parser("stage", help="decompress a config's inputs on /arc")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_stage)

    p = sub.add_parser("run", help="run one job per config")
    p.add_argument("names", nargs="+")
    p.add_argument("--cores", type=int, default=None,
                   help="override; 2 by default, for a full field or a patch alike")
    # 48 GB standard. Runs peak near 34 GB on the UDS trial patches and a 16 GB
    # request is OOM-killed with no traceback; the headroom costs nothing, since
    # the quota's 32 GB is only a default and the nodes are far larger.
    p.add_argument("--ram", type=int, default=48)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=do_run)

    p = sub.add_parser("status", help="list headless sessions")
    p.set_defaults(func=do_status)

    p = sub.add_parser("clean", help="delete finished sessions (never a running one)")
    p.set_defaults(func=do_clean)

    p = sub.add_parser("logs", help="print session logs")
    p.add_argument("ids", nargs="+")
    p.set_defaults(func=do_logs)

    p = sub.add_parser("fetch", help="download the small outputs")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_fetch)

    args = ap.parse_args()
    if not (Path.home() / ".ssl/cadcproxy.pem").exists():
        sys.exit("no CADC certificate; run scratch/canfar/canfar-cert.sh first")
    args.func(args)


if __name__ == "__main__":
    main()
