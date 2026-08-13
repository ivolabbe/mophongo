#!/usr/bin/env python
"""Drive mophongo runs on the CANFAR Science Platform.

CANFAR compute is a REST API (`skaha`), not ssh: the transfer endpoint on port
64022 is SFTP only and cannot execute anything. Jobs are containers with `/arc`
mounted, so the MINERVA data are already there and nothing is uploaded except
the mophongo source and the PSF grids.

    python submit.py push                    # upload src (from main) + PSF grids
    python submit.py setup                   # build the venv on /arc
    python submit.py stage  uds_f770w        # decompress that config's inputs
    python submit.py run    uds_f770w ...    # one job per config, concurrent
    python submit.py status                  # all sessions
    python submit.py kill                    # destroy them, stragglers included
    python submit.py logs   <session-id>
    python submit.py fetch  uds_f770w        # pull the small outputs down

Four API quirks are handled here. The installed client defaults to API v0,
which 404s (hence ``version='v1'``), and ``args`` is whitespace-split into a
YAML sequence server side, so the command must be a single token - parameters
are passed as environment variables instead. Session names come back with a
``-1`` replica index appended, so ``mophongo-uds-f770w-v1-0`` lists as
``mophongo-uds-f770w-v1-0-1``. And a session is not listed until the service
registers it, which is why ``wait`` tolerates an empty ``info`` and ``kill``
sweeps more than once.
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
# This directory is staging only - launch scripts and configs, all tracked and
# all small. Anything a command produces goes under scratch/, which is
# gitignored: outputs must never land in the staging tree.
WORK = REPO / "scratch" / "canfar"


from runroot import run_root

RUN, RUN_VOS = run_root(REPO)   # /arc/... and its arc: URI
IMAGE = "images.canfar.net/skaha/jwst-notebook:25.07.25"
VCP = Path.home() / ".venvs/canfar/bin/vcp"
DONE = ("Succeeded", "Failed", "Completed", "Terminating")


def session() -> Session:
    return Session(version="v1")


SRC_ITEMS = ["src", "pyproject.toml", "README.md"]


def git(*args: str) -> str:
    """Run git in the repo and return stdout, stripped."""
    proc = subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def resolve_ref(ref: str) -> str:
    """The sha a CANFAR run should use for ``ref``: the remote's, when there is one.

    A campaign must never ship a stale clone. Resolving through ``origin/<ref>``
    after a fetch means "latest main" is what actually reaches arc, whatever the
    local checkout happens to be sitting on, and without mutating the user's
    repo the way a pull would. A branch with no remote counterpart, or an
    unreachable network, falls back to the local ref and says so - an explicit
    sha (a commit, a tag) has no remote form and resolves unchanged.
    """
    local = git("rev-parse", ref)
    try:
        git("fetch", "--quiet", "origin", ref)
        remote = git("rev-parse", f"origin/{ref}")
    except SystemExit:
        log.warning("could not reach origin/%s; shipping the local %s (%s)",
                    ref, ref, local[:9])
        return local
    if remote != local:
        log.info("%s: local is %s, origin/%s is %s; shipping origin's",
                 ref, local[:9], ref, remote[:9])
    return remote


def build_src_tarball(dest: Path, ref: str, worktree: bool) -> str:
    """Write the source tarball; return a one-line description of what it holds.

    The source ships from ``git archive`` of a committed ref, not from the
    working tree. A campaign is 17 jobs running for hours, and tarring whatever
    happens to be on disk sends half-finished edits - another session's, or an
    editor mid-save - to every one of them, with nothing afterwards recording
    which code actually ran. ``--worktree`` is the escape hatch for trying an
    uncommitted change, and says so loudly.

    A ref behind its remote is a warning rather than an error: the remote may
    simply not have been fetched.
    """
    if worktree:
        dirty = git("status", "--porcelain", "--", *SRC_ITEMS).splitlines()
        with tarfile.open(dest, "w:gz") as tar:
            for item in SRC_ITEMS:
                tar.add(REPO / item, arcname=item,
                        filter=lambda t: None if "__pycache__" in t.name else t)
        log.warning("shipping the WORKING TREE, not a commit%s",
                    f"; {len(dirty)} uncommitted file(s) included" if dirty else "")
        for line in dirty:
            log.warning("    %s", line)
        return f"worktree on top of {git('rev-parse', '--short', 'HEAD')}"

    sha = resolve_ref(ref)
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz",
                    "-o", str(dest), sha, "--", *SRC_ITEMS], check=True)
    return f"{ref} {sha[:9]} {git('log', '-1', '--format=%cs %s', sha)}"


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

    ``psf.tar`` is unpacked by ``setup_env.sh`` and by nothing else, so pushing
    it before a ``sync`` (which only replaces the source) uploads several
    hundred MB that never reach ``$RUN/PSF``. Use ``--src-only`` to ship a code
    change on its own. The grids already on arc are not lost either way:
    ``PSFFactory`` skips any grid file that exists unless ``overwrite`` is set.
    """
    tmp = WORK / "_upload"
    tmp.mkdir(exist_ok=True)

    src_tar = tmp / "mophongo_src.tgz"
    version = build_src_tarball(src_tar, args.ref, args.worktree)
    log.info("src  %.1f MB  [%s]", src_tar.stat().st_size / 1e6, version)

    # Uploaded as ``.pending`` and promoted to ``SRC_VERSION`` by whichever job
    # unpacks the tarball. Stamping it here instead would record the version on
    # upload, and push-without-sync - the case worth catching - would then look
    # current while the jobs still imported the old source.
    version_file = tmp / "SRC_VERSION.pending"
    version_file.write_text(version + "\n")

    uploads = [src_tar, version_file]
    if args.src_only:
        log.info("psf  skipped (--src-only)")
    else:
        psf_tar = tmp / "psf.tar"
        grids = sorted(set(sum((list((REPO / "data" / "PSF").glob(pat))
                                for pat in args.psf_glob), [])))
        if not grids:
            raise SystemExit(f"no PSF grids matched {args.psf_glob}")
        with tarfile.open(psf_tar, "w") as tar:
            for grid in grids:
                tar.add(grid, arcname=grid.name)
        log.info("psf  %d grids, %.1f MB", len(grids), psf_tar.stat().st_size / 1e6)
        uploads.append(psf_tar)

    for path in uploads:
        log.info("uploading %s", path.name)
        vcp(path, f"{RUN_VOS}/{path.name}")

    subprocess.run([str(VCP.parent / "vmkdir"), f"{RUN_VOS}/jobs"], capture_output=True)
    # .py as well as .sh: a job whose logic does not fit in shell ships its
    # script alongside the wrapper that runs it.
    for script in sorted(p for p in (HERE / "jobs").iterdir()
                         if p.suffix in (".sh", ".py")):
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


def launch(name: str, script: str, cores: int, ram: int, env: dict[str, str],
           tries: int = 3) -> str:
    """Start one headless job and return its session id, or ``""``.

    Retried, because the service drops submissions under load: a batch of seven
    once lost three to HTTP 500s carrying
    ``JedisDataException: ERR max number of clients reached``. skaha swallows
    those and returns an empty list, so an unretried failure is a job that
    silently never existed.
    """
    for attempt in range(1, tries + 1):
        try:
            ids = session().create(
                name=name, image=IMAGE, cores=cores, ram=ram, kind="headless",
                cmd="/bin/bash", args=f"{RUN}/jobs/{script}", env=env,
            )
        except Exception as exc:  # noqa: BLE001 - transient service errors
            ids, why = [], type(exc).__name__
        else:
            why = "empty response"
        if ids and ids[0]:
            log.info("%-24s %s", name, ids[0])
            return ids[0]
        if attempt < tries:
            log.warning("%-24s submit failed (%s), retry %d/%d",
                        name, why, attempt, tries)
            time.sleep(10)
    log.error("%-24s FAILED after %d attempts", name, tries)
    return ""


def still_listed(sid: str) -> bool:
    """Whether the service still lists this session; ``True`` when unsure.

    An empty ``info`` is not evidence that a session is gone. The client
    swallows network errors and answers with an empty list, so a laptop that
    loses DNS looks exactly like a reaped job. This is a second, independent
    call, and only a *non-empty* listing that omits the id counts as proof:
    an empty listing could just as easily be the network. Anything else
    answers "still there", so the caller keeps waiting rather than declaring a
    running job dead.
    """
    try:
        rows = session().fetch(kind="headless")
    except Exception:  # noqa: BLE001 - transient API errors mean "unknown"
        return True
    return True if not rows else any(i.get("id") == sid for i in rows)


def wait(ids: list[str], poll: int = 20, missing_tolerance: int = 3) -> dict[str, str]:
    """Block until every session reaches a terminal state.

    ``info`` returns an empty list both for a session the service has already
    reaped *and* for one just created that has not been registered yet, so an
    empty answer is only treated as terminal after several consecutive polls
    *and* a listing that confirms it. Declaring it terminal on the first empty
    reply made wait() return instantly on a freshly submitted job; declaring it
    on the count alone made a two-minute DNS outage on the laptop report a
    running EGS job as ``Gone``, after which the campaign moved on and tried to
    start the bands that were waiting on the grids it was still building.
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
                    if still_listed(sid):
                        if missing[sid] % 15 == 0:
                            log.warning("  %s: unlisted by info for %d polls but "
                                        "still in the session listing; waiting",
                                        sid, missing[sid])
                        continue
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


def run_root_local() -> Path | None:
    """The run tree as a local path, when a writable /arc mount exposes it.

    Queue latency dominates small work here: a 1-core sync job has sat Pending
    for half an hour to do a few seconds of file copying. The sshfs mount is
    writable, not just readable, so the same unpack takes about twenty seconds
    from the laptop. ``$CANFAR_RUN_LOCAL`` names the mounted run tree
    explicitly; otherwise the documented mount of the user's home
    (``canfar-mount.sh /home/<user> ~/canfar_home``) is tried.
    """
    explicit = os.environ.get("CANFAR_RUN_LOCAL")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_dir() else None
    if not re.match(r"^/arc/home/[^/]+/", RUN):
        return None
    path = Path.home() / "canfar_home" / RUN.split("/", 4)[-1]
    return path if path.is_dir() else None


def do_sync(args: argparse.Namespace) -> None:
    """Ship a code change without rebuilding the venv.

    ``setup`` deletes and recreates the venv, which breaks every run currently
    using it. mophongo is installed editable, so replacing the source is enough
    and is safe while other jobs are in flight.

    Done through the /arc mount when there is one, since this is file movement
    rather than compute and a container would spend far longer queueing than
    working. ``--job`` forces the container. Either way the source is rewritten
    under any job that is running: as ``update_src.sh`` says, already-running
    jobs keep the code they imported, and only later ones pick this up.
    """
    local = None if args.job else run_root_local()
    if local is None:
        ids = [launch("mophongo-sync", "update_src.sh", 1, 4, {"RUN": RUN})]
        wait(ids)
        print(tidy(first_log(ids), 10))
        return

    # The same two steps as jobs/update_src.sh, in the same order: the version
    # is promoted only once the source it describes is actually unpacked.
    subprocess.run(["tar", "-xzf", str(local / "mophongo_src.tgz"),
                    "-C", str(local / "mophongo")], check=True)
    (local / "SRC_VERSION").write_text((local / "SRC_VERSION.pending").read_text())
    log.info("source updated through %s", local)
    log.info("mophongo: %s", (local / "SRC_VERSION").read_text().strip())


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

    Four, for everything. Measured utilisation is about 0.2 of a core - the runs
    wait on ``/arc`` rather than compute, and the fitting path has no thread
    pool - so the extra cores are headroom rather than throughput, at the price
    of waiting longer to be scheduled when the platform is busy.
    """
    return 4 if override is None else override


#: GB per field, for fields that need more than DEFAULT_RAM.
FIELD_RAM = {"egs": 82}
DEFAULT_RAM = 64


def ram_for(name: str, override: int | None) -> int:
    """GB to request for one config.

    64 as standard: a full-field run peaks near 46 GB during the fit, and the
    output stage adds to that afterwards - the stamps file alone was 10.1 GB of
    template pixels on COSMOS - so 48 left no room and every band died writing
    its diagnostics.

    EGS gets 82. Its detection grid is 1221 Mpx against UDS's 876, and it
    reached 49.4 GB before the writes had even started.
    """
    if override is not None:
        return override
    return FIELD_RAM.get(str(name).split("_")[0].lower(), DEFAULT_RAM)


def arc_src_version() -> str:
    """The ``SRC_VERSION`` recorded in the run tree, or ``""`` if unreadable."""
    tmp = WORK / "_upload"
    tmp.mkdir(exist_ok=True)
    dest = tmp / "SRC_VERSION.arc"
    dest.unlink(missing_ok=True)
    try:
        vcp(f"{RUN_VOS}/SRC_VERSION", str(dest))
    except (FileNotFoundError, SystemExit):
        return ""
    return dest.read_text().strip() if dest.exists() else ""


def check_src_current(ref: str, force: bool) -> None:
    """Refuse to run against source that is not the current ``ref``.

    ``push`` only uploads the tarball; ``setup`` and ``sync`` are what unpack
    it. Doing the first without the second leaves every job importing the
    previous campaign's code, and the outputs look entirely normal afterwards,
    so this is checked rather than trusted. It has already happened once: a
    campaign was submitted against source four hours older than the memory fix
    it depended on.
    """
    want = resolve_ref(ref)[:9]
    have = arc_src_version()
    if have and want in have:
        log.info("source on arc: %s", have)
        return
    problem = (f"no SRC_VERSION in {RUN_VOS}" if not have
               else f"arc has [{have}], local {ref} is {want}")
    if force:
        log.warning("%s (continuing anyway: --force-stale)", problem)
    else:
        raise SystemExit(f"refusing to run: {problem}. "
                         f"Run 'submit.py push --src-only' then 'submit.py sync'.")


def do_run(args: argparse.Namespace) -> None:
    check_src_current(args.ref, args.force_stale)
    do_upload_cfg(args.names)
    step = getattr(args, "step", "all")
    ids = []
    for name in args.names:
        cores = cores_for(name, args.cores)
        # the step goes in the session name so `status` can tell a field's
        # prep job from the band fits that follow it
        label = "mophongo" if step == "all" else f"mophongo-{step}"
        ids.append(launch(session_name(label, name), "run.sh", cores,
                          ram_for(name, args.ram),
                          {"RUN": RUN, "CFG": name, "STEP": step}))
    # A dropped submission is a band that quietly never runs, and with
    # --no-wait nothing downstream would notice until the outputs were missing.
    dropped = [n for n, sid in zip(args.names, ids) if not sid]
    if dropped:
        raise SystemExit("submission failed for: " + ", ".join(dropped)
                         + "\nThe others are running; resubmit just these.")
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


def do_plots(args: argparse.Namespace) -> None:
    """Redraw scene figures for runs that fitted but died rendering them.

    Not a rerun: ``jobs/scene_plots.sh`` restores the finished state with
    ``load_fit`` and draws the figures from it. The stamps file is rebuilt when
    it is missing or truncated, since the figures need the templates; nothing
    else the run wrote is touched.
    """
    check_src_current(args.ref, args.force_stale)
    do_upload_cfg(args.names)
    ids = [launch(session_name("mophongo-plots", name), "scene_plots.sh",
                  cores_for(name, args.cores), ram_for(name, args.ram),
                  {"RUN": RUN, "CFG": name})
           for name in args.names]
    dropped = [n for n, sid in zip(args.names, ids) if not sid]
    if dropped:
        raise SystemExit("submission failed for: " + ", ".join(dropped))
    if args.no_wait:
        return
    final = wait(ids)
    for sid, text in session().logs(ids).items():
        print(f"--- {sid} [{final.get(sid)}]\n{tidy(text)}")


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


def do_kill(args: argparse.Namespace) -> None:
    """Destroy sessions, sweeping until the listing stays empty.

    One pass is not enough. ``fetch`` does not report a session the service has
    accepted but not yet registered, exactly as ``wait`` documents for
    ``info``, so a campaign submitted with ``--no-wait`` can be listed as six
    jobs, destroyed, and then show ten more minutes later. Those stragglers
    carry the same run names as whatever is submitted next and would write into
    the same ``out/`` directories, so sweep until ``--sweeps`` consecutive
    passes come back empty.

    ``--keep`` protects a session by name substring; the default protects the
    ``sync`` job, which is usually the one shipping the code the next campaign
    needs.
    """
    sess = session()
    # An empty --keep entry is a substring of every name and would spare the
    # whole listing, silently destroying nothing. Read it as "keep nothing".
    keep = [k for k in args.keep if k]
    killed = 0
    quiet = 0
    while quiet < args.sweeps:
        rows = [i for i in sess.fetch(kind="headless")
                if args.match in i.get("name", "")
                and not any(k in i.get("name", "") for k in keep)]
        if rows:
            quiet = 0
            for info in rows:
                log.info("killing %-28s %s", info.get("name"), info.get("status"))
            sess.destroy([i["id"] for i in rows])
            killed += len(rows)
        else:
            quiet += 1
            if quiet < args.sweeps:
                log.info("clear (%d/%d); watching for late registrations",
                         quiet, args.sweeps)
                time.sleep(args.interval)
    log.info("destroyed %d session(s)", killed)


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
        dest = WORK / "out" / name
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
    p.add_argument("--src-only", action="store_true",
                   help="skip the PSF grids; only setup unpacks them, so a push "
                        "before sync ships several hundred MB for nothing")
    p.add_argument("--ref", default="main",
                   help="git ref whose source is shipped (default: main)")
    p.add_argument("--worktree", action="store_true",
                   help="ship the working tree instead of a commit, uncommitted "
                        "changes and all; for trying a fix, not for a campaign")
    p.set_defaults(func=do_push)

    p = sub.add_parser("setup", help="build the venv on /arc")
    p.set_defaults(func=do_setup)

    p = sub.add_parser("sync", help="replace the source in place, leaving the venv alone")
    p.add_argument("--job", action="store_true",
                   help="use a container even when the /arc mount is available; "
                        "the mount does the same work without the queue wait")
    p.set_defaults(func=do_sync)

    p = sub.add_parser("seed", help="copy cached PSF/kernel maps between runs")
    p.add_argument("pairs", nargs="+", metavar="SRC:DST")
    p.set_defaults(func=do_seed)

    p = sub.add_parser("stage", help="decompress a config's inputs on /arc")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=do_stage)

    p = sub.add_parser("run", help="run one job per config")
    p.add_argument("names", nargs="+")
    p.add_argument("--step", default="all",
                   help="pipeline step (default: all). 'prep' builds the "
                        "field's shared PSF grids and repair cache and stops, "
                        "so the bands that follow start warm")
    p.add_argument("--cores", type=int, default=None,
                   help="override; 4 by default, for a full field or a patch alike")
    # Per field: 64 GB standard, 82 for EGS (see ram_for). An under-sized
    # request is OOM-killed with no traceback, which reads as a silent failure
    # rather than an error, and the headroom costs nothing - the quota's 32 GB
    # is a default, not a cap, and the nodes report hundreds of GB.
    p.add_argument("--ram", type=int, default=None,
                   help="override the per-field default (64 GB, EGS 82)")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--ref", default="main",
                   help="git ref the arc source must match (default: main)")
    p.add_argument("--force-stale", action="store_true",
                   help="submit even though the arc source is not that ref")
    p.set_defaults(func=do_run)

    p = sub.add_parser("plots", help="redraw scene figures from a finished run")
    p.add_argument("names", nargs="+")
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--ram", type=int, default=None,
                   help="override the per-field default (64 GB, EGS 82)")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--ref", default="main",
                   help="git ref the arc source must match (default: main)")
    p.add_argument("--force-stale", action="store_true",
                   help="submit even though the arc source is not that ref")
    p.set_defaults(func=do_plots)

    p = sub.add_parser("status", help="list headless sessions")
    p.set_defaults(func=do_status)

    p = sub.add_parser("clean", help="delete finished sessions (never a running one)")
    p.set_defaults(func=do_clean)

    p = sub.add_parser("kill", help="destroy sessions, sweeping for late registrations")
    p.add_argument("--match", default="mophongo",
                   help="only sessions whose name contains this (default: mophongo)")
    p.add_argument("--keep", nargs="*", default=["sync"],
                   help="name substrings to spare (default: sync)")
    p.add_argument("--sweeps", type=int, default=3,
                   help="consecutive empty passes required before stopping")
    p.add_argument("--interval", type=int, default=60, help="seconds between passes")
    p.set_defaults(func=do_kill)

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
