#!/usr/bin/env python
"""Rewrite a local mophongo run config to run on OzStar, and list its inputs.

Unlike CANFAR, where ``/arc`` is mounted inside every container and most inputs
are read in place, OzStar has no view of the MINERVA release at all: every file
a config names has to be copied to ``/fred`` first. So this writes two things
per config:

* ``<name>_ozstar.json`` - the config with every input path pointed at
  ``<base>/data``, ``psf_dir`` at ``<base>/PSF`` and ``out_dir`` at
  ``<base>/<run>/<field>/<name>``. Data and grids sit above the run
  directory because they are stable across catalog versions; see
  :mod:`ozroot` for the tree;
* ``<name>_stage.tsv`` - ``<vospace uri>\\t<destination basename>`` for every
  one of those inputs, which ``jobs/stage.sh`` copies from CANFAR and, where
  the arc copy is gzipped, decompresses.

Finding the files on arc is the same problem ``examples/canfar/arcify.py``
solves, so its index is reused rather than duplicated: ``roots_for`` maps a
local staged path to the arc subtrees that could hold it, and ``resolve``
handles the gzipped names and the ``_f770_wcs.csv`` / ``_f770w_wcs.csv``
spelling mismatch.

Usage::

    python ozify.py ../minerva/uds_f770w.json [more configs ...]
    python ozify.py ../minerva/uds_f770w.json --r-trial 1.5 --suffix _trial
    python ozify.py ../minerva/*.json --check-versions   # scan, rewrite nothing

Needs a CADC proxy certificate locally (only to *list* arc; the copying itself
happens on OzStar with the certificate pushed by ``submit.py cert``).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from vos import Client

HERE = Path(__file__).resolve().parent
# The arc index is the same problem for both platforms, so canfar/arcify.py is
# imported rather than copied. Appended, not prepended: arcify imports canfar's
# own `runroot`, and this directory's equivalent is deliberately named
# `ozroot` so the two cannot shadow each other.
sys.path.append(str(HERE.parent / "canfar"))

from arcify import (  # noqa: E402
    PATH_KEYS, _repair_cache_name, arc_index, check_release_versions, resolve,
    roots_for,
)

import ozroot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ozify")

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
            f"Use OZSTAR_RUN=run<N> instead, so outputs stay "
            "<run>/<field>/<field>_<band> and nothing repeats the version."
        )
    return suffix



def vos_uri(arc_path: str) -> str:
    """``/arc/projects/minerva/...`` -> ``arc:projects/minerva/...``.

    The vos tools address arc by URI and the SSH endpoint by chrooted POSIX
    path; ``arc_index`` returns the POSIX form, and ``vcp`` wants the URI.
    """
    return "arc:" + arc_path[len("/arc/"):]


def ozify(cfg_path: Path, index: dict[str, str], out_dir: Path,
          r_trial: float | None = None, suffix: str = "") -> tuple[Path, Path]:
    """Write the OzStar config and its staging list for one local config.

    ``r_trial`` overrides the trial-patch radius in arcmin (0 means the full
    field) and ``suffix`` keeps a trial run's outputs separate from the full
    one's. The config's own ``trial.center`` is kept - only the radius moves -
    so the patch stays where the source config put it.
    """
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

    stage: list[tuple[str, str]] = []  # (vospace source, staged basename)
    for key in PATH_KEYS:
        value = cfg.get(key)
        if not value:
            continue
        basename = Path(value).name
        hit = resolve(basename, index)
        if hit is None:
            raise SystemExit(f"{cfg_path.name}: {key} not found on arc: {basename}")
        arc_path, _gzipped = hit
        # Everything is copied, compressed or not: /fred has no view of arc.
        # The destination carries the name the config expects, which also fixes
        # the f770 -> f770w frame-table spelling.
        cfg[key] = f"{ozroot.data_dir()}/{basename}"
        stage.append((vos_uri(arc_path), basename))

    # data/ and PSF/ live above the run directory: they are stable across
    # catalog versions, so a new run re-fits the same mosaics and reuses the
    # same grids rather than re-staging 64 GB and rebuilding 400 grids.
    cfg.setdefault("psf", {})["dir"] = ozroot.psf_dir()
    cfg["out_dir"] = ozroot.out_dir(name)
    # Per-field saturation-repair cache, one level above out_dir. The repair
    # depends only on detection-side inputs, so a field's bands share it: band
    # one fits the saturated cores, the rest reload. Naming it after the field
    # keeps fields apart even when they run concurrently -- the v1.0b campaign
    # had a single shared cache and COSMOS paid for the repair twice because
    # EGS overwrote it in between. Never a correctness problem (the cache
    # records sci_hi and its mtime, and a foreign one is rejected as stale),
    # but wasted work and a read-while-writing race.
    cfg["repair_cache_path"] = f"../{_repair_cache_name(cfg)}"

    cfg_out = out_dir / f"{name}_ozstar.json"
    cfg_out.write_text(json.dumps(cfg, indent=2) + "\n")
    tsv_out = out_dir / f"{name}_stage.tsv"
    tsv_out.write_text("".join(f"{src}\t{dst}\n" for src, dst in stage))
    log.info("%-18s -> %s  (%d files to stage)", name, cfg_out.name, len(stage))
    return cfg_out, tsv_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", type=Path, help="local RunConfig JSON files")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="where to write the rewritten configs")
    ap.add_argument("--r-trial", type=float, default=None,
                    help="override the trial-patch radius in arcmin (0 = full field)")
    ap.add_argument("--suffix", default="",
                    help="append to the run name, e.g. _trial, to keep outputs separate")
    ap.add_argument("--check-versions", action="store_true",
                    help="report configs pinned to an older release than arc "
                         "now holds, and rewrite nothing")
    args = ap.parse_args()

    # Reading arc is a laptop operation and needs only the certificate; the
    # datamover partition exists to *copy* the files to /fred, not to see them.
    # So this is the same check as `arcify.py --check-versions`, answered the
    # same way, and it costs the cluster nothing.
    if args.check_versions:
        behind = check_release_versions(args.configs)
        if not behind:
            log.info("every config is pinned to the newest release on arc")
            return
        log.warning("%d config/version pair(s) are behind arc:", len(behind))
        for name, kind, pinned, newest in behind:
            log.warning("  %-14s %-8s %s -> %s", name, kind, pinned, newest)
        log.warning("Nothing was rewritten. Moving to a new release changes the "
                    "photometry, so it belongs in a new run: bump $OZSTAR_RUN, "
                    "point the source configs at the new version, re-run ozify, "
                    "and re-stage - /fred holds a copy of each input, so a new "
                    "release is new files rather than an in-place update.")
        return

    check_suffix(args.suffix)

    cert = Path.home() / ".ssl/cadcproxy.pem"
    if not cert.exists():
        raise SystemExit(f"no CADC certificate at {cert}; run ~/bin/remote/canfar-cert.sh first")


    # Collect the subtrees every config needs, then index each one once: the
    # bands of a field overlap almost completely.
    roots: list[str] = []
    for cfg_path in args.configs:
        cfg = json.loads(re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text()))
        for key in PATH_KEYS:
            for root in roots_for(cfg.get(key) or ""):
                if root not in roots:
                    roots.append(root)
    log.info("indexing %d arc subtrees:", len(roots))
    index = arc_index(Client(vospace_certfile=str(cert)), roots)

    for cfg_path in args.configs:
        ozify(cfg_path, index, args.out_dir, args.r_trial, args.suffix)


if __name__ == "__main__":
    main()
