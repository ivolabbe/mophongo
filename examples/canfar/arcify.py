#!/usr/bin/env python
"""Rewrite a local mophongo run config to run on CANFAR ARC storage.

The MINERVA mosaics, segmaps and catalogs already live on
``arc:projects/minerva``, so a CANFAR run copies nothing from the laptop: it
only needs the same config with its paths pointed at ``/arc``.

Two wrinkles make this more than a string substitution:

* almost everything on arc is gzipped, and the pipeline wants plain FITS, so
  those files are decompressed once into a per-run ``data/`` directory.
  Uncompressed files are referenced in place and never copied.
* the MIRI frame tables ship as ``*_f770_wcs.csv`` while the filter parser wants
  the ``f770w`` spelling.

Usage::

    python arcify.py ../minerva/uds_f770w.json [more configs ...]

writes ``<name>_canfar.json`` and ``<name>_stage.tsv`` next to this script. The
TSV is the copy list consumed by ``jobs/stage.sh`` inside the container.

arc is read only to place an input no generated pair here already places, so a
release that has been arcified before is rewritten without touching CANFAR, and
a release that has moved on costs a listing of the subtrees that changed.
Asking whether the release itself has moved on is a separate, deliberate act:
``--check-versions`` does it and rewrites nothing, and ``--reindex`` forces the
listing for a file that moved under an unchanged name.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

from vos import Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("arcify")

#: A version belongs in the run directory, not in the run name. `run2/uds/
#: uds_f770w` beats `run2/uds/uds_f770w_v2`: the latter repeats the version in
#: every path and every output filename, and makes two attempts at one release
#: impossible to compare without renaming files. `--suffix` remains for genuine
#: variants of a run (a `_trial` patch beside the full field), so a suffix that
#: merely looks like a version is refused rather than silently accepted.
VERSION_SUFFIX = re.compile(r"^_?v?[\d.]+[a-z]?$", re.I)


def check_suffix(suffix: str) -> str:
    """Reject a suffix that is really a version; the run directory carries that."""
    if suffix and VERSION_SUFFIX.match(suffix):
        raise SystemExit(
            f"refusing --suffix {suffix!r}: the version is the run directory. "
            f"Use CANFAR_RUNNUM=run<N> instead, so outputs stay "
            "<run>/<field>/<field>_<band> and nothing repeats the version."
        )
    return suffix


from runroot import run_number, run_root

REPO = Path(__file__).resolve().parent.parent.parent

ARC = "arc:projects/minerva"


def arc_run() -> str:
    """Run tree on arc; ``$CANFAR_RUN`` overrides the release default.

    Resolved on demand rather than at import so that ``examples/ozstar``, which
    reuses the arc indexing helpers below but has no CANFAR run tree, does not
    have to set ``$CANFAR_USER`` to import this module.
    """
    return run_root(REPO)[0]

# Config keys whose values are input file paths.
PATH_KEYS = ["sci_hi", "wht_hi", "segmap", "catalog", "csv_hi", "sci_lo", "wht_lo", "csv_lo"]


def roots_for(local_path: str) -> list[str]:
    """arc subtrees that could hold the file named by a local staged path.

    The staged tree mirrors the arc release versions as
    ``.../data/<FIELD>/<version>/...``, so the field and version are read off
    the path itself rather than hardcoded. That keeps this working when the
    releases move on.

    ``n3.0``   -> ``<field>/mosaics/nircam/n3.0/{grizli,bkgsub}``
    ``m3.1``   -> ``<field>/mosaics/miri/m3.1``
    ``n3.0_v1.2`` (or any ``*_v*``) -> ``<field>/catalogs/<version>/...``
    """
    parts = Path(local_path).parts
    fields = {"UDS": "uds", "COSMOS": "cosmos", "EGS": "egs"}
    for i, part in enumerate(parts):
        if part in fields and i + 1 < len(parts):
            field, version = fields[part], parts[i + 1]
            break
    else:
        return []

    # local-only suffix on some segmap directories
    version = re.sub(r"_SEC$", "", version)

    # Some products were staged flat, straight under the field directory, so the
    # component after the field is the file itself. The release version is in
    # the filename either way: MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_...
    VERSION_DIR = re.compile(r"[nm][\d.]+|n[\d.]+(?:_m[\d.]+)?_v[\d.]+")
    if not VERSION_DIR.fullmatch(version):
        match = re.search(r"_(n[\d.]+(?:_m[\d.]+)?_v[\d.]+)_", Path(local_path).name)
        if not match:
            return []
        version = match.group(1)

    if re.fullmatch(r"n[\d.]+", version):
        return [f"{ARC}/{field}/mosaics/nircam/{version}/grizli",
                f"{ARC}/{field}/mosaics/nircam/{version}/bkgsub"]
    if re.fullmatch(r"m[\d.]+", version):
        return [f"{ARC}/{field}/mosaics/miri/{version}"]
    return [f"{ARC}/{field}/catalogs/{version}",
            f"{ARC}/{field}/catalogs/{version}/ACS+WEBB_chi-mean",
            f"{ARC}/{field}/catalogs/{version}/ACS+WEBB_chi-mean/ancillary"]


#: Where each kind of release version is enumerated on arc, and the shape of a
#: version directory there. Mirrors the mapping ``roots_for`` walks in the
#: other direction.
RELEASE_TREES = {
    "nircam": ("mosaics/nircam", re.compile(r"^n[\d.]+$")),
    "miri": ("mosaics/miri", re.compile(r"^m[\d.]+$")),
    "catalog": ("catalogs", re.compile(r"^n[\d.]+(?:_m[\d.]+)?_v[\d.]+$")),
}


def _version_key(version: str) -> list[tuple[int, ...]]:
    """Sort key for a release version, numerically per component.

    ``n3.0`` and ``n10.0`` sort the way a human means, and so do the compound
    catalog versions: ``n3.0_m3.1_v1.2.1`` compares component by component.
    """
    return [tuple(int(n) for n in re.findall(r"\d+", part))
            for part in version.split("_")]


def _version_shape(version: str) -> tuple[str, ...]:
    """The letters of a version: ``n3.0_m3.1_v1.2.1`` -> ``('n', 'm', 'v')``.

    Versions are comparable only within a shape. A NIRCam-only catalog
    (``n3.0_v1.3``) and one built with MIRI (``n3.0_m3.1_v1.2.1``) are
    different products, not two points on one sequence, and ordering them
    against each other invents an answer.
    """
    return tuple(re.sub(r"[\d.]", "", part) for part in version.split("_"))


def newest_like(version: str, candidates: list[str]) -> str:
    """The newest candidate of the same shape as ``version``, or ``""``."""
    same = [c for c in candidates if _version_shape(c) == _version_shape(version)]
    return max(same, key=_version_key) if same else ""


def versions_on_arc(client: Client, field: str, kind: str) -> list[str]:
    """Release versions of one kind present for one field, oldest first."""
    subdir, shape = RELEASE_TREES[kind]
    try:
        children = client.get_children_info(f"{ARC}/{field}/{subdir}")
    except Exception as exc:  # noqa: BLE001 - a missing subtree is not fatal
        log.warning("  ! %s/%s: %s", field, subdir, exc)
        return []
    names = [getattr(c, "name", None) or c.uri.rstrip("/").rsplit("/", 1)[-1]
             for c in children]
    return sorted((n for n in names if shape.match(n)), key=_version_key)


def config_versions(cfg: dict) -> dict[str, str]:
    """The release versions one config is pinned to, by kind.

    Read back off the arc paths the config already carries, so this describes
    what a run would actually read rather than what a generator once intended.
    """
    got: dict[str, str] = {}
    for key, kind in (("csv_hi", "nircam"), ("csv_lo", "miri"),
                      ("catalog", "catalog"), ("segmap", "catalog")):
        value = str(cfg.get(key) or "")
        subdir, shape = RELEASE_TREES[kind]
        for part in Path(value).parts:
            if shape.match(part):
                got.setdefault(kind, part)
                break
        else:
            # staged files sit flat in data/, with the version in the filename
            match = re.search(r"[-_](n[\d.]+(?:_m[\d.]+)?_v[\d.]+)[-_]",
                              Path(value).name)
            if match and shape.match(match.group(1)):
                got.setdefault(kind, match.group(1))
    return got


def check_release_versions(cfg_paths: list[Path]) -> list[tuple[str, str, str, str]]:
    """Report configs pinned to something older than what arc now holds.

    Alert only: nothing is rewritten. Moving a campaign onto a new release is
    a deliberate act -- it changes the photometry, so it belongs to a new run
    number with a note saying why -- and it is not something a launch should
    do because a directory appeared upstream.

    Returns ``(config, kind, pinned, latest)`` for each version behind.
    """
    cert = Path.home() / ".ssl/cadcproxy.pem"
    if not cert.exists():
        raise SystemExit(f"no CADC certificate at {cert}; run remote/canfar-cert.sh first")
    client = Client(vospace_certfile=str(cert))

    latest: dict[tuple[str, str], list[str]] = {}
    behind: list[tuple[str, str, str, str]] = []
    for cfg_path in cfg_paths:
        cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
        field = str(cfg.get("name", cfg_path.stem)).split("_")[0]
        for kind, pinned in sorted(config_versions(cfg).items()):
            key = (field, kind)
            if key not in latest:
                latest[key] = versions_on_arc(client, field, kind)
            newest = newest_like(pinned, latest[key])
            if newest and _version_key(newest) > _version_key(pinned):
                behind.append((cfg_path.stem, kind, pinned, newest))
    return behind


def arc_index(client: Client, roots: list[str]) -> dict[str, str]:
    """Map basename -> POSIX ``/arc`` path for every file under ``roots``."""
    index: dict[str, str] = {}
    for root in roots:
        try:
            children = client.get_children_info(root)
        except Exception as exc:  # noqa: BLE001 - a missing subtree is not fatal
            log.warning("  ! %s: %s", root, exc)
            continue
        for child in children:
            name = getattr(child, "name", None) or child.uri.rstrip("/").rsplit("/", 1)[-1]
            if child.isdir() or child.islink():
                continue
            index.setdefault(name, f"/arc/{root.split(':', 1)[1]}/{name}")
        log.info("  indexed %s", root)
    return index


def resolve(basename: str, index: dict[str, str]) -> tuple[str, bool] | None:
    """Find ``basename`` on arc, returning ``(arc_path, is_gzipped)``.

    Tries the name itself, then the gzipped name, then the MIRI frame-table
    spelling (``f770w_wcs.csv`` is shipped as ``f770_wcs.csv``).
    """
    candidates = [basename, basename + ".gz"]
    alt = re.sub(r"_(f\d+)w_wcs\.csv$", r"_\1_wcs.csv", basename)
    if alt != basename:
        candidates += [alt, alt + ".gz"]
    for cand in candidates:
        if cand in index:
            return index[cand], cand.endswith(".gz")
    return None


def known_placements(out_dir: Path) -> dict[str, tuple[str, str]]:
    """``basename -> ("stage", arc source) | ("inplace", arc path)``.

    What arc is actually consulted for is where each input lives and whether it
    has to be decompressed or renamed on the way in. Both answers are already
    written down here by a previous run: ``<name>_stage.tsv`` names every file
    that gets staged and the arc path it comes from, and ``<name>_canfar.json``
    carries the arc path of every file read in place. Reading them back is what
    lets a release already arcified be rewritten with no certificate and no
    network.

    Basenames carry their release version
    (``MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits``), so a hit is a hit against
    the same release; a release that has moved on brings new basenames, misses,
    and is looked up. ``--reindex`` skips this for a file that moved on arc
    under an unchanged name.

    A staged file's rewritten value is ``<run>/data/<basename>``, which is not
    a source and must not be read back as one. It is excluded twice over: its
    basename is already in the manifest, and the run's data directory is the
    only place an input sits directly under a directory called ``data``.
    """
    placements: dict[str, tuple[str, str]] = {}
    for tsv in sorted(out_dir.glob("*_stage.tsv")):
        for line in tsv.read_text().splitlines():
            src, _, dst = line.partition("\t")
            if src.strip() and dst.strip():
                placements.setdefault(dst.strip(), ("stage", src.strip()))
    for generated in sorted(out_dir.glob("*_canfar.json")):
        try:
            cfg = json.loads(generated.read_text())
        except json.JSONDecodeError:  # a half-written file is not a cache
            continue
        for key in PATH_KEYS:
            value = str(cfg.get(key) or "")
            if not value.startswith("/arc/") or Path(value).parent.name == "data":
                continue
            placements.setdefault(Path(value).name, ("inplace", value))
    return placements


def wanted_inputs(cfg_paths: list[Path]) -> dict[str, str]:
    """``basename -> the local path it came from`` over every config.

    The local path is kept because :func:`roots_for` reads the field and the
    release version off it to decide which arc subtrees would have to be
    indexed for that file.
    """
    wanted: dict[str, str] = {}
    for cfg_path in cfg_paths:
        cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
        for key in PATH_KEYS:
            value = cfg.get(key) or ""
            if value:
                wanted.setdefault(Path(value).name, value)
    return wanted


def _repair_cache_name(cfg: dict) -> str:
    """Filename for the shared saturation-repair cache of one config.

    The repair depends on the detection band alone -- science image, weight,
    PSF pattern, kwargs, and the trial box (``Pipeline._repair_provenance``) --
    so every band of a field with the same geometry produces the same result
    and should share one cache. The name carries the field, which is what
    stops one field's cache being found stale by the next and overwritten:
    that is exactly what happened to the v1.0 campaign, where all three fields
    took turns rewriting a single ``out/repair_cache.fits``.

    The cache sits in the field directory of a numbered run, so a run and a
    geometry are already fixed by where it is. A trial patch still gets its
    geometry in the name: a patch and a full field repair different pixels,
    and nothing else distinguishes them if both are run under one run number.

    ``RunConfig.repair_cache_path`` defaults to ``'..'``, which resolves to one
    unnamed file per directory level -- fine for one field at a time, wrong for
    a release campaign that submits several.
    """
    field = str(cfg.get("name", "run")).split("_")[0]
    trial = cfg.get("trial")
    if not trial:
        return f"{field}_repair_cache.fits"
    ra, dec = (round(float(v), 5) for v in trial["center"])
    return f"{field}_r{float(trial['radius']):g}_{ra}{dec:+}_repair_cache.fits"


def arcify(cfg_path: Path, index: dict[str, str], out_dir: Path,
           r_trial: float | None = None, suffix: str = "",
           placements: dict[str, tuple[str, str]] | None = None,
           ) -> tuple[Path, Path]:
    """Write the arc-path config and its staging list for one local config.

    ``r_trial`` overrides the trial-patch radius (in arcmin) so a cheap smoke
    run can be generated from the same source config; ``suffix`` keeps its
    outputs separate from the full run's. The config's own ``trial.center``
    is kept — only the radius is overridden — so the patch stays where the
    source config put it.

    ``placements`` is the cache from :func:`known_placements`, consulted before
    ``index``; a basename it answers costs no arc lookup.
    """
    run = arc_run()
    raw = re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text())
    cfg = json.loads(raw)
    name = cfg.get("name", cfg_path.stem) + suffix
    if r_trial is not None:
        if r_trial <= 0:
            cfg["trial"] = None  # full field
        else:
            trial = dict(cfg.get("trial") or {})
            if not trial.get("center"):
                raise SystemExit(
                    f"{cfg_path.name}: --r-trial needs a trial.center in the "
                    'config: "trial": {"center": [ra, dec], "radius": <arcmin>}'
                )
            trial["radius"] = r_trial
            cfg["trial"] = trial
    cfg["name"] = name

    stage: list[tuple[str, str]] = []  # (arc source, staged basename)
    placements = placements or {}
    for key in PATH_KEYS:
        value = cfg.get(key)
        if not value:
            continue
        basename = Path(value).name
        known = placements.get(basename)
        if known is not None:
            how, source = known
        else:
            hit = resolve(basename, index)
            if hit is None:
                raise SystemExit(f"{cfg_path.name}: {key} not found on arc: {basename}")
            source, gzipped = hit
            # gzipped files are decompressed into the run's data dir, and an
            # uncompressed one whose arc name differs is copied there rather
            # than decompressed; both land under the name the config expects,
            # which is also what fixes the f770 -> f770w spelling. Anything
            # else is read in place and never copied.
            how = ("stage" if gzipped or Path(source).name != basename
                   else "inplace")
        if how == "stage":
            cfg[key] = f"{run}/data/{basename}"
            stage.append((source, basename))
        else:
            cfg[key] = source

    # Release layout: inputs and grids at the release root, products under a
    # numbered run and grouped by field. `..` from out_dir is the field
    # directory, so the repair cache lands beside the bands that share it.
    field = str(name).split("_")[0]
    cfg.setdefault("psf", {})["dir"] = f"{run}/PSF"
    cfg["out_dir"] = f"{run}/run{run_number()}/{field}/{name}"
    cfg["repair_cache_path"] = f"../{_repair_cache_name(cfg)}"

    cfg_out = out_dir / f"{name}_canfar.json"
    cfg_out.write_text(json.dumps(cfg, indent=2) + "\n")
    tsv_out = out_dir / f"{name}_stage.tsv"
    tsv_out.write_text("".join(f"{src}\t{dst}\n" for src, dst in stage))
    log.info("%-12s -> %s  (%d files to stage, %d read in place)",
             name, cfg_out.name, len(stage), len(PATH_KEYS) - len(stage))
    return cfg_out, tsv_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path, help="local RunConfig JSON files")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent,
                    help="where to write the rewritten configs")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial-patch radius in arcmin (0 = full field)")
    ap.add_argument("--suffix", default="",
                    help="append to the run name, e.g. _test, to keep outputs separate")
    ap.add_argument("--check-versions", action="store_true",
                    help="report configs pinned to an older release than arc "
                         "now holds, and rewrite nothing")
    ap.add_argument("--reindex", action="store_true",
                    help="list arc even for inputs the generated configs and "
                         "manifests here already place; use if a file has "
                         "moved on arc under the same name")
    args = ap.parse_args()

    if args.check_versions:
        behind = check_release_versions(args.configs)
        if not behind:
            log.info("every config is pinned to the newest release on arc")
            return
        log.warning("%d config/version pair(s) are behind arc:", len(behind))
        for name, kind, pinned, newest in behind:
            log.warning("  %-14s %-8s %s -> %s", name, kind, pinned, newest)
        log.warning("Nothing was rewritten. Moving to a new release changes the "
                    "photometry, so it belongs in a new run: bump $CANFAR_RUNNUM, "
                    "point the source configs at the new version, re-run arcify "
                    "and stage, and say what changed in the run's README.")
        return

    check_suffix(args.suffix)

    # arc answers two questions per input - where it lives, and whether it is
    # staged or read in place - and a previous run wrote both answers down
    # here. So place what the generated configs and manifests can place, and
    # list arc for the rest: nothing at all when this release has been arcified
    # before, and only the changed subtrees when it has moved on.
    placements = {} if args.reindex else known_placements(args.out_dir)
    wanted = wanted_inputs(args.configs)
    unresolved = {b: v for b, v in wanted.items() if b not in placements}

    index: dict[str, str] = {}
    if not unresolved:
        log.info("all %d input(s) are placed by the configs and manifests here; "
                 "not reading arc", len(wanted))
    else:
        why = (f"{len(unresolved)} of {len(wanted)} input(s) are not placed by "
               "anything here, so arc has to be listed to find them: "
               + ", ".join(sorted(unresolved)[:3])
               + (", ..." if len(unresolved) > 3 else ""))
        cert = Path.home() / ".ssl/cadcproxy.pem"
        if not cert.exists():
            raise SystemExit(f"no CADC certificate at {cert}; run "
                             f"remote/canfar-cert.sh first. {why}")
        # Present is not the same as valid. An expired certificate lists every
        # subtree as a warning and leaves an empty index, which then reads as
        # "not found on arc" against the release rather than against the cert.
        if subprocess.run(["openssl", "x509", "-in", str(cert), "-noout",
                           "-checkend", "0"], capture_output=True).returncode:
            raise SystemExit(f"{cert} has expired; run "
                             f"remote/canfar-cert.sh --force. {why}")
        # The subtrees those files could be in, each indexed once: the bands of
        # a field overlap almost completely.
        roots: list[str] = []
        for value in unresolved.values():
            for root in roots_for(value):
                if root not in roots:
                    roots.append(root)
        log.info("%d of %d input(s) not placed here; indexing %d arc subtrees:",
                 len(unresolved), len(wanted), len(roots))
        index = arc_index(Client(vospace_certfile=str(cert)), roots)
        # arc_index warns per subtree and carries on, because one missing
        # subtree is not fatal. Every subtree failing is a different thing -
        # an expired certificate, usually - and without this it surfaces as
        # "sci_hi not found on arc", which sends you looking at the release.
        if not index:
            raise SystemExit(
                f"none of the {len(roots)} arc subtrees listed (see the "
                "warnings above). The usual cause is an expired certificate: "
                f"{cert} exists, which is all the check above can see. Run "
                "remote/canfar-cert.sh --force and try again."
            )

    for cfg_path in args.configs:
        arcify(cfg_path, index, args.out_dir, args.r_trial, args.suffix,
               placements)


if __name__ == "__main__":
    main()
