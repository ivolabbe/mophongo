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

# Run tree on arc. Everything the run writes lives under here, in the user's
# own home rather than the shared project space.
RUN = "/arc/home/ilabbe/run"

# arc subtrees searched for the config's input files, as VOSpace URIs.
ARC_ROOTS = [
    "arc:projects/minerva/uds/mosaics/nircam/n3.0/grizli",
    "arc:projects/minerva/uds/mosaics/nircam/n3.0/bkgsub",
    "arc:projects/minerva/uds/mosaics/miri/m3.1",
    "arc:projects/minerva/uds/mosaics/miri/m3.0",
    "arc:projects/minerva/uds/catalogs/n3.0_v1.2/ACS+WEBB_chi-mean/ancillary",
    "arc:projects/minerva/uds/catalogs/n3.0_m3.1_v1.2.1/ACS+WEBB_chi-mean",
]

# Config keys whose values are input file paths.
PATH_KEYS = ["sci_hi", "wht_hi", "segmap", "catalog", "csv_hi", "sci_lo", "wht_lo", "csv_lo"]


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
    outputs separate from the full run's.
    """
    raw = re.sub(r"(?m)^\s*#.*$", "", cfg_path.read_text())
    cfg = json.loads(raw)
    name = cfg.get("name", cfg_path.stem) + suffix
    if r_trial is not None:
        cfg["r_trial"] = r_trial
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
            cfg[key] = f"{RUN}/data/{basename}"
            stage.append((arc_path, basename))
        elif Path(arc_path).name != basename:
            # uncompressed but misnamed: copy rather than decompress
            cfg[key] = f"{RUN}/data/{basename}"
            stage.append((arc_path, basename))
        else:
            cfg[key] = arc_path  # read in place, no copy

    cfg["psf_dir"] = f"{RUN}/PSF"
    cfg["out_dir"] = f"{RUN}/out/{name}"

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
    log.info("indexing arc:")
    index = arc_index(Client(vospace_certfile=str(cert)), ARC_ROOTS)

    for cfg in args.configs:
        arcify(cfg, index, args.out_dir, args.r_trial, args.suffix)


if __name__ == "__main__":
    main()
