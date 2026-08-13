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
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from vos import Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("arcify")

from runroot import run_root

REPO = Path(__file__).resolve().parent.parent.parent

ARC = "arc:projects/minerva"


def arc_run() -> str:
    """Run tree on arc; ``$CANFAR_RUN`` overrides the default home location.

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


def arcify(cfg_path: Path, index: dict[str, str], out_dir: Path,
           r_trial: float | None = None, suffix: str = "") -> tuple[Path, Path]:
    """Write the arc-path config and its staging list for one local config.

    ``r_trial`` overrides the trial-patch radius (in arcmin) so a cheap smoke
    run can be generated from the same source config; ``suffix`` keeps its
    outputs separate from the full run's. The config's own ``trial.center``
    is kept — only the radius is overridden — so the patch stays where the
    source config put it.
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
    for key in PATH_KEYS:
        value = cfg.get(key)
        if not value:
            continue
        basename = Path(value).name
        hit = resolve(basename, index)
        if hit is None:
            raise SystemExit(f"{cfg_path.name}: {key} not found on arc: {basename}")
        arc_path, gzipped = hit
        if gzipped:
            # decompress once into the run's data dir, under the name the
            # config expects (which also fixes the f770 -> f770w spelling)
            cfg[key] = f"{run}/data/{basename}"
            stage.append((arc_path, basename))
        elif Path(arc_path).name != basename:
            # uncompressed but misnamed: copy rather than decompress
            cfg[key] = f"{run}/data/{basename}"
            stage.append((arc_path, basename))
        else:
            cfg[key] = arc_path  # read in place, no copy

    cfg["psf_dir"] = f"{run}/PSF"
    cfg["out_dir"] = f"{run}/out/{name}"

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
    args = ap.parse_args()

    cert = Path.home() / ".ssl/cadcproxy.pem"
    if not cert.exists():
        raise SystemExit(f"no CADC certificate at {cert}; run canfar-cert.sh first")

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
        arcify(cfg_path, index, args.out_dir, args.r_trial, args.suffix)


if __name__ == "__main__":
    main()
