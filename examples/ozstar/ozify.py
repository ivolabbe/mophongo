#!/usr/bin/env python
"""Rewrite a local mophongo run config to run on OzStar, and list its inputs.

Unlike CANFAR, where ``/arc`` is mounted inside every container and most inputs
are read in place, OzStar has no view of the MINERVA release at all: every file
a config names has to be copied to ``/fred`` first. So this writes two things
per config:

* ``<name>_ozstar.json`` - the config with every input path pointed at
  ``$RUN/data``, ``psf_dir`` at ``$RUN/PSF`` and ``out_dir`` at
  ``$RUN/out/<name>``;
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

from arcify import PATH_KEYS, arc_index, resolve, roots_for  # noqa: E402

from ozroot import run_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ozify")


def vos_uri(arc_path: str) -> str:
    """``/arc/projects/minerva/...`` -> ``arc:projects/minerva/...``.

    The vos tools address arc by URI and the SSH endpoint by chrooted POSIX
    path; ``arc_index`` returns the POSIX form, and ``vcp`` wants the URI.
    """
    return "arc:" + arc_path[len("/arc/"):]


def ozify(cfg_path: Path, index: dict[str, str], out_dir: Path, run: str,
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
        cfg[key] = f"{run}/data/{basename}"
        stage.append((vos_uri(arc_path), basename))

    cfg["psf_dir"] = f"{run}/PSF"
    cfg["out_dir"] = f"{run}/out/{name}"

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
    args = ap.parse_args()

    cert = Path.home() / ".ssl/cadcproxy.pem"
    if not cert.exists():
        raise SystemExit(f"no CADC certificate at {cert}; run canfar-cert.sh first")

    run = run_root()

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
        ozify(cfg_path, index, args.out_dir, run, args.r_trial, args.suffix)


if __name__ == "__main__":
    main()
