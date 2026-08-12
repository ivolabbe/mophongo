"""Injected-truth verification of the UDS MIRI setup, one mock per band.

Runs the standard realistic verification (two NIRCam LW detectors, six
dithers; two MIRI macro pointings, eight dithers each; ~800 injected sources
with truth fluxes in both bands) through the package verification framework,
once per UDS MIRI band, with the production settings of the real runs: the
band's STPSF grid, the band's extra Gaussian blur, the band's photometric
aperture, ``scene_max_size`` 800 / ``scene_max_merge_radius`` 1000, and
``extend_templates="psf_wings"``.

One deliberate aliasing: the framework's mock builder and scenario runner key
the low-resolution band by the internal name ``"f770w"`` throughout (frame
lists, truth columns, output filenames). Rather than renaming that plumbing,
each band is run with its own PSF pattern, blur and aperture *in that slot*,
so inside ``verification/uds_<band>/`` any ``f770w`` label means "the
low-resolution band", which is the band in the directory name. The physics
(PSF, blur, aperture, kernel) is fully per band; only the labels are not.
``build_wiener_psf_maps`` reads the blur back off the mock, so the PSF maps
carry the identical Fourier operator that painted the sources.

Usage (from ``examples/minerva/``)::

    ../../.venv/bin/python run_verification.py [band ...]   # default: all four
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mophongo.mock_mosaic import DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC
from mophongo.verification import (
    DEFAULT_WIENER_REG_GRID,
    build_realistic_two_detector_mock,
    build_wiener_psf_maps,
    run_pipeline_extension_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify")

ROOT = Path(__file__).resolve().parents[2]
PSF_DIR = ROOT / "data" / "PSF"
OUT_ROOT = Path(__file__).resolve().parent / "verification"

NSRC = 800
SEED = 42
SNR_RANGE = (3.0, 3000.0)

# production values of the real-data runs (make_minerva_configs.py)
APERTURE_DIAM_ARCSEC = {"f770w": 0.70, "f1280w": 1.20, "f1500w": 1.20, "f1800w": 1.50}
FIT_OVERRIDES_COMMON = {"scene_max_size": 800, "scene_max_merge_radius": 1000}

BANDS = ["f770w", "f1280w", "f1500w", "f1800w"]


def run_band(band: str, scheme: str = "psf_wings") -> dict:
    """Build the mock and run one build-scheme scenario for one band."""
    out = OUT_ROOT / f"uds_{band}"
    out.mkdir(parents=True, exist_ok=True)
    blur = DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC[band]
    pattern = f"UDS_MIRI_{band.upper()}_OS4_GRID1"
    log.info("=== %s: pattern %s, blur %.2f\", aperture %.2f\"",
             band, pattern, blur, APERTURE_DIAM_ARCSEC[band])

    mock, paths, noise_info, dpsfs, truth = build_realistic_two_detector_mock(
        out / "mock",
        psf_dir=PSF_DIR,
        nsrc=NSRC,
        snr_range=SNR_RANGE,
        seed=SEED,
        target_pattern=pattern,
        # the band's blur, keyed by the framework's internal lo-res slot name
        psf_gaussian_fwhm_arcsec={"f770w": blur},
    )
    psf_maps = build_wiener_psf_maps(
        mock, paths, dpsfs, out,
        psf_dir=PSF_DIR,
        reg_grid=DEFAULT_WIENER_REG_GRID,
        target_pattern=pattern,
        target_label=band.upper(),
    )
    result = run_pipeline_extension_scenario(
        scheme,
        out_dir=out,
        paths=paths,
        noise_info=noise_info,
        truth=truth,
        psf_maps=psf_maps,
        nsrc=NSRC,
        target_label=band.upper(),
        fit_overrides={
            "aperture_diam": APERTURE_DIAM_ARCSEC[band],
            **FIT_OVERRIDES_COMMON,
        },
    )
    summary = {"band": band, "scheme": scheme, "blur_fwhm_arcsec": blur,
               "aperture_diam_arcsec": APERTURE_DIAM_ARCSEC[band],
               "pattern": pattern, **result.summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("%s: med_lo %.4f  p16-p84 %.4f-%.4f  pull %.2f +- %.2f  resid/noise %.3f",
             band, summary["med_lo"], summary["p16_lo"], summary["p84_lo"],
             summary["pull_lo_median"], summary["pull_lo_std"],
             summary["resid_std_over_noise"])
    return summary


def main(argv: list[str]) -> None:
    want = [a for a in argv if a in BANDS] or BANDS
    OUT_ROOT.mkdir(exist_ok=True)
    summaries = []
    for band in want:
        try:
            summaries.append(run_band(band))
        except Exception:
            log.exception("%s failed", band)
    if summaries:
        (OUT_ROOT / "summary_all.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
    log.info("\n%-8s %-10s %-17s %-16s %-12s", "band", "med_lo", "p16-p84",
             "pull med+-std", "resid/noise")
    for s in summaries:
        log.info("%-8s %-10.4f %.4f-%.4f    %+.2f +- %.2f    %.3f",
                 s["band"], s["med_lo"], s["p16_lo"], s["p84_lo"],
                 s["pull_lo_median"], s["pull_lo_std"], s["resid_std_over_noise"])


if __name__ == "__main__":
    main(sys.argv[1:])
