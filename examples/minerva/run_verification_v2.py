"""Verification v2: both UDS MIRI legs on the merged scheme code, consolidated.

Reruns the two verification legs of ``scratch/wren/verification.py`` on the
current HEAD (least-squares ``psf_wings`` build scheme, ee_psf_lo propagation,
extend_mode naming) and consolidates everything under a versioned directory::

    examples/minerva/verification/v2/
      README.md                    what this version is, code commit, results
      verification_summary.json    combined machine-readable summary
      driver.log                   this script's full log
      runs/uds_<band>/             real-data fits (r<3'), own <name>.log inside
      uds_monu/                    IDL leg: compare_idl_vs_python_<band>.png
                                   + idl_summary.json + idl_compare.log
      uds_sims/                    mock leg: uds_<band>/ outputs + summary_all
                                   .json + mock_flux_ratio/mock_wiener pngs
                                   + mock.log

v1 (pre-merge, convolution-fill wings on the real-data leg) stays untouched
next to it. Sequential on purpose: one r<3' band peaks at tens of GB.

Usage (from the repository root)::

    .venv/bin/python examples/minerva/run_verification_v2.py \
        [fits|idl|mock ...] [--version vN] [--scheme NAME]

``--version`` names the output directory under ``verification/`` (default
``v2``); ``--scheme`` forces one template build scheme (``psf_wings``,
``wren``, ``classic``, ...) into both legs — the real-data fit configs get it
as ``fit.extend_mode`` and the mock scenario runs it directly. Without
``--scheme`` the configs' own default applies (``psf_wings``).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "examples" / "minerva"))
# make_compare_idl_python (the IDL leg) lives with the report sources
sys.path.insert(0, str(BASE / "scratch" / "wren"))

VER_ROOT = BASE / "examples" / "minerva" / "verification"
# set by main() from --version/--scheme; module-level so every helper sees them
V2 = VER_ROOT / "v2"
RUNS = V2 / "runs"
IDL_OUT = V2 / "uds_monu"
MOCK_OUT = V2 / "uds_sims"
SCHEME: str | None = None
PSF_DIR: str | None = None
PSF_SIZE: float | None = None
R_TRIAL: float | None = None
SRC_RUNS = BASE / "examples" / "minerva"
BANDS = ["f770w", "f1280w", "f1500w", "f1800w"]
PY = sys.executable


def _set_version(version: str, scheme: str | None,
                 psf_dir: str | None = None, psf_size: float | None = None,
                 r_trial: float | None = None) -> None:
    global V2, RUNS, IDL_OUT, MOCK_OUT, SCHEME, PSF_DIR, PSF_SIZE, R_TRIAL
    V2 = VER_ROOT / version
    RUNS = V2 / "runs"
    IDL_OUT = V2 / "uds_monu"
    MOCK_OUT = V2 / "uds_sims"
    SCHEME = scheme
    PSF_DIR = psf_dir
    PSF_SIZE = psf_size
    R_TRIAL = r_trial

log = logging.getLogger("v2")


def _setup_logging() -> None:
    V2.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(V2 / "driver.log")):
        handler.setFormatter(fmt)
        logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=BASE,
        capture_output=True, text=True,
    ).stdout.strip()


def prep_configs() -> None:
    """Copy the four run configs into v2 with absolute out_dirs, seed caches."""
    RUNS.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        name = f"uds_{band}"
        src_cfg = SRC_RUNS / f"{name}.json"
        out_dir = RUNS / name
        out_dir.mkdir(exist_ok=True)
        text = src_cfg.read_text()
        # only the out_dir is rewritten; every data path is already absolute
        text = text.replace(f'"out_dir": "{name}"', f'"out_dir": "{out_dir}"')
        # the psf_wings build scheme requires the detection weight map, and
        # the generated configs predate the wht_hi field; the automatic
        # _sci->_wht substitution cannot see through the _bkgsub suffix
        if '"wht_hi"' not in text:
            for line in text.splitlines():
                if '"sci_hi"' in line:
                    sci = line.split('"')[3]
                    wht = sci.replace("_drc_sci_bkgsub.fits", "_drc_wht.fits")
                    if wht != sci and Path(wht).exists():
                        text = text.replace(line, line + f'\n  "wht_hi": "{wht}",')
                    else:
                        raise FileNotFoundError(f"no detection wht for {sci}")
                    break
        # the generated configs spell repair_psf_pattern in the legacy
        # OS4_GRID5 order, which matches no file and defeats autobuild
        # (see TODO.md); respell to the canonical GRID5_OS4 so the repair
        # grids are autobuilt into this run's psf_dir and saturated stars
        # are flagged and isolated as in production
        text = text.replace(
            '"repair_psf_pattern": "UDS_NRC.._F444W_OS4_GRID5"',
            '"repair_psf_pattern": "UDS_NRC.._F444W_GRID5_OS4"',
        )
        if SCHEME is not None:
            assert '"extend_mode"' not in text
            text = text.replace('"fit": {', f'"fit": {{\n    "extend_mode": "{SCHEME}",')
        if PSF_DIR is not None:
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if '"psf_dir"' in line:
                    lines[k] = f'  "psf_dir": "{Path(PSF_DIR).resolve()}",'
            text = "\n".join(lines)
        if PSF_SIZE is not None:
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if '"psf_size"' in line:
                    lines[k] = f'  "psf_size": {PSF_SIZE},'
            text = "\n".join(lines)
        if R_TRIAL is not None:
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if '"r_trial"' in line:
                    lines[k] = f'  "r_trial": {R_TRIAL},'
            text = "\n".join(lines)
        (RUNS / f"{name}.json").write_text(text)
        # seed the PSF/kernel caches from the newest prior copy anywhere —
        # other verification versions or the production run dirs. Each
        # .geojson pairs with a sibling .fits (PSFRegionMap.to_file); both
        # are needed, a geojson alone loads with psfs=None. Provenance is
        # checked on load (pattern, psf_size, blur, kernel method), so a
        # non-matching seed is rebuilt rather than silently reused: seeding
        # is always safe, at worst useless.
        for stem in (f"{name}_psf_hi", f"{name}_psf_lo", f"{name}_kernel"):
            dest = out_dir / f"{stem}.geojson"
            if dest.exists():
                continue
            candidates = sorted(
                (p for p in [
                    *VER_ROOT.glob(f"v*/runs/{name}/{stem}.geojson"),
                    SRC_RUNS / name / f"{stem}.geojson",
                ] if p.exists() and p.parent != out_dir and p.with_suffix(".fits").exists()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if candidates:
                src = candidates[0]
                shutil.copy2(src, dest)
                shutil.copy2(src.with_suffix(".fits"), dest.with_suffix(".fits"))
                log.info("seeded %s from %s", stem, src.parent)
        log.info("prepared %s (caches: %d geojson + %d fits)", name,
                 len(list(out_dir.glob("*.geojson"))),
                 len([f for f in out_dir.glob("*.fits") if "psf" in f.name or "kernel" in f.name]))


def run_fits() -> None:
    """Real-data fits, one subprocess per band (memory isolation)."""
    for band in BANDS:
        name = f"uds_{band}"
        cfg = RUNS / f"{name}.json"
        band_log = RUNS / f"run_{name}.log"
        t0 = time.time()
        log.info("fit %s starting (log: %s)", name, band_log)
        with open(band_log, "w") as fh:
            rc = subprocess.run(
                [PY, "-m", "mophongo.pipeline", str(cfg)],
                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        log.info("fit %s finished rc=%d in %.1f min", name, rc, (time.time() - t0) / 60)
        if rc != 0:
            raise RuntimeError(f"fit {name} failed (rc={rc}); see {band_log}")


def run_idl_leg() -> list[dict]:
    """IDL comparison against the v2 fit tables; outputs into uds_monu/."""
    import numpy as np

    import make_compare_idl_python as cmp
    from mophongo.pipeline import RunConfig

    IDL_OUT.mkdir(parents=True, exist_ok=True)
    cmp.RUNS = RUNS
    cmp.OUT = IDL_OUT
    leg_log = logging.FileHandler(IDL_OUT / "idl_compare.log")
    leg_log.setFormatter(logging.Formatter("%(message)s"))
    cmp.log.addHandler(leg_log)

    summaries = []
    for band in cmp.discover():
        log.info("idl compare %s", band)
        trio = cmp.matched(band)
        if trio is None:
            continue
        r_trial = RunConfig.from_json(RUNS / f"uds_{band}.json").r_trial
        path, panel_stats = cmp.figure(band, *trio, r_trial)
        entry = {"band": band, "n_matched": len(trio[0]),
                 "figure": path.name}
        for key, (med, sd, n) in panel_stats.items():
            if np.isfinite(med):
                entry[key] = {"median": round(float(med), 4),
                              "sd": round(float(sd), 4), "n": int(n)}
        summaries.append(entry)
        log.info("  wrote %s", path)
    cmp.log.removeHandler(leg_log)
    (IDL_OUT / "idl_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def run_mock_leg() -> list[dict]:
    """Injected-truth mocks into uds_sims/, with the report figures collected."""
    import run_verification as rv

    MOCK_OUT.mkdir(parents=True, exist_ok=True)
    rv.OUT_ROOT = MOCK_OUT
    leg_log = logging.FileHandler(MOCK_OUT / "mock.log")
    leg_log.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(leg_log)

    scheme = SCHEME or "psf_wings"
    summaries = []
    try:
        for band in BANDS:
            t0 = time.time()
            try:
                summaries.append(rv.run_band(band, scheme=scheme))
            except Exception:
                log.exception("mock %s failed", band)
            log.info("mock %s done in %.1f min", band, (time.time() - t0) / 60)
            src_dir = MOCK_OUT / f"uds_{band}"
            for src, dest in [
                (src_dir / f"template_extension_{scheme}" / f"flux_ratio_{scheme}_lowres.png",
                 MOCK_OUT / f"mock_flux_ratio_{band}.png"),
                (src_dir / "diagnostic_wiener.png", MOCK_OUT / f"mock_wiener_{band}.png"),
            ]:
                if src.exists():
                    shutil.copy2(src, dest)
                else:
                    log.warning("missing mock figure %s", src)
    finally:
        logging.getLogger().removeHandler(leg_log)
    (MOCK_OUT / "summary_all.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main(argv: list[str]) -> None:
    version, scheme, psf_dir, psf_size, r_trial = "v2", None, None, None, None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--version":
            version = next(it)
        elif a == "--scheme":
            scheme = next(it)
        elif a == "--psf-dir":
            psf_dir = next(it)
        elif a == "--psf-size":
            psf_size = float(next(it))
        elif a == "--r-trial":
            r_trial = float(next(it))
        else:
            args.append(a)
    _set_version(version, scheme, psf_dir, psf_size, r_trial)
    _setup_logging()
    global BANDS
    picked = [a for a in args if a in BANDS]
    if picked:
        BANDS = picked
    steps = [a for a in args if a in ("fits", "idl", "mock")] or ["fits", "idl", "mock"]
    head = _git_head()
    log.info("verification %s on commit %s; scheme: %s; steps: %s",
             version, head, scheme or "config default (psf_wings)", ", ".join(steps))

    out: dict = {"commit": head, "steps": steps, "scheme": scheme or "psf_wings"}
    if "fits" in steps:
        prep_configs()
        run_fits()
    if "idl" in steps:
        out["idl"] = run_idl_leg()
    if "mock" in steps:
        out["mock"] = run_mock_leg()

    (V2 / "verification_summary.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    log.info("combined summary -> %s", V2 / "verification_summary.json")
    if out.get("idl"):
        for s in out["idl"]:
            c, c25 = s.get("c", {}), s.get("c_snr25", {})
            log.info("idl  %-8s est1 %+.2f +- %.2f  SNR>25 %+.2f +- %.2f  (n=%d)",
                     s["band"], c.get("median", float("nan")),
                     c.get("sd", float("nan")), c25.get("median", float("nan")),
                     c25.get("sd", float("nan")), s["n_matched"])
    if out.get("mock"):
        for s in out["mock"]:
            log.info("mock %-8s med %.4f  p16-p84 %.4f-%.4f  resid/noise %.3f",
                     s["band"], s["med_lo"], s["p16_lo"], s["p84_lo"],
                     s["resid_std_over_noise"])
    log.info("V2_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
