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
import os
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


def canfar_user() -> str:
    """CADC username, from $CANFAR_USER or scratch/canfar/canfar.conf."""
    user = os.environ.get("CANFAR_USER")
    if user:
        return user
    conf = REPO / "scratch" / "canfar" / "canfar.conf"
    if conf.exists():
        match = re.search(r'^\s*CANFAR_USER\s*=\s*"?([^"\s]+)', conf.read_text(), re.M)
        if match and match.group(1) != "your_cadc_username":
            return match.group(1)
    sys.exit("set CANFAR_USER to your CADC username, or fill in "
             "scratch/canfar/canfar.conf")


USER = canfar_user()
RUN = f"/arc/home/{USER}/run"                 # run tree on arc (POSIX form)
RUN_VOS = f"arc:home/{USER}/run"              # same, VOSpace form
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


def wait(ids: list[str], poll: int = 20) -> dict[str, str]:
    """Block until every session reaches a terminal state.

    A session can disappear from the API entirely - reaped, or deleted by a
    concurrent ``clean`` - in which case ``info`` returns an empty list. Treat
    that as terminal rather than letting an IndexError kill a long campaign.
    """
    pending = [i for i in ids if i]
    final: dict[str, str] = {}
    while pending:
        for sid in list(pending):
            try:
                info = session().info([sid])
            except Exception as exc:  # noqa: BLE001 - transient API errors
                log.warning("  %s: info failed (%s), retrying", sid, type(exc).__name__)
                continue
            status = info[0].get("status") if info else "Gone"
            if status in DONE or status == "Gone":
                final[sid] = status
                pending.remove(sid)
                log.info("%s %s", sid, status)
        if pending:
            time.sleep(poll)
    return final


def do_setup(args: argparse.Namespace) -> None:
    ids = [launch("mophongo-setup", "setup_env.sh", 4, 16, {"RUN": RUN})]
    wait(ids)
    print(next(iter(session().logs(ids).values()))[-2000:])


def stage_job(name: str) -> str:
    return launch(f"mophongo-stage-{name.replace('_', '-')}", "stage.sh", 2, 8,
                  {"RUN": RUN, "CFG": name})


def do_sync(args: argparse.Namespace) -> None:
    """Ship a code change without rebuilding the venv.

    ``setup`` deletes and recreates the venv, which breaks every run currently
    using it. mophongo is installed editable, so replacing the source is enough
    and is safe while other jobs are in flight.
    """
    ids = [launch("mophongo-sync", "update_src.sh", 1, 4, {"RUN": RUN})]
    wait(ids)
    print(tidy(next(iter(session().logs(ids).values())), 10))


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
    """Cores to request for one config: 4 for a full field, 1 for a trial patch.

    Measured utilisation is about 0.2 of a core - the runs wait on ``/arc``
    rather than compute, and mophongo has no thread pool in the fitting path -
    so a large request only idles allocation that another job could use. Full
    fields get 4 because they carry many more sources through the solver.
    """
    if override is not None:
        return override
    cfg_path = HERE / f"{name}_canfar.json"
    try:
        r_trial = json.loads(cfg_path.read_text()).get("r_trial", 0.0)
    except (OSError, ValueError):
        return 4
    return 1 if r_trial else 4


def do_run(args: argparse.Namespace) -> None:
    do_upload_cfg(args.names)
    ids = []
    for name in args.names:
        cores = cores_for(name, args.cores)
        ids.append(launch(f"mophongo-{name.replace('_', '-')}", "run.sh", cores, args.ram,
                          {"RUN": RUN, "CFG": name}))
    if args.no_wait:
        return
    final = wait(ids)
    for sid, text in session().logs(ids).items():
        status = final.get(sid)
        print(f"--- {sid} [{status}]\n{tidy(text)}")
        if status == "Failed" and "RUN_DONE" not in text:
            print("  note: a failure with no traceback is usually the container "
                  "being OOM-killed. Memory scales with the mosaic size, not "
                  "r_trial, so keep --ram at 64 even for a small patch.")


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
    # and has to build them regardless.
    p.add_argument("--psf-glob", nargs="+",
                   default=["*_NRC*_F444W_MJD*_GRID25_OS4.fits",
                            "*_MIRI_*_MJD*_GRID9_OS4.fits"])
    p.set_defaults(func=do_push)

    p = sub.add_parser("setup", help="build the venv on /arc")
    p.set_defaults(func=do_setup)

    p = sub.add_parser("sync", help="replace the source in place, leaving the venv alone")
    p.set_defaults(func=do_sync)

    p = sub.add_parser("stage", help="decompress a config's inputs on /arc")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_stage)

    p = sub.add_parser("run", help="run one job per config")
    p.add_argument("names", nargs="+")
    p.add_argument("--cores", type=int, default=None,
                   help="override; default is 4 for a full field, 1 for a trial patch")
    # 64 GB standard. Runs peak near 34 GB on the UDS trial patches and a 16 GB
    # request is OOM-killed with no traceback; the headroom costs nothing, since
    # the quota's 32 GB is only a default and the nodes are far larger.
    p.add_argument("--ram", type=int, default=64)
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
