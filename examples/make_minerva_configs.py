"""Generate one :class:`~mophongo.pipeline.RunConfig` JSON per MINERVA field and
MIRI band from whatever is staged under ``MINERVA/data``.

The configs follow ``uds_770_dr0.json``: F444W template on the 40 mas NIRCam
grid, the release segmap and SUPER catalog, and one 80 mas MIRI band as the
low-resolution image to fit. Everything that can be read off the staged files
is read off them -- frame counts from the WCS tables, the trial-patch centre
from the weight map -- so the generator does not carry hand-copied numbers.

Usage (from ``examples/``)::

    python make_minerva_configs.py            # all staged field/band pairs
    python make_minerva_configs.py uds egs    # selected fields

Outputs go to ``examples/minerva/<field>_<band>.json``.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.wcs import WCS
from scipy.ndimage import uniform_filter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("configs")

DATA = Path("/Users/ivo/Astro/PROJECTS/MINERVA/data")
PSF_DIR = Path(__file__).resolve().parent.parent / "data" / "PSF"
OUT = Path(__file__).resolve().parent / "minerva"

R_TRIAL_ARCMIN = 3.0  # trial-patch radius; set 0 in the config for a full run

# Photometric aperture diameter per MIRI band, arcsec. The four UDS values are
# the ones classic IDL subphot uses (Y. Asada; see examples/run_uds_770_wren.py),
# so raw aperture fluxes are directly comparable between the two codes. F560W,
# F1000W and F2100W have no IDL counterpart and are interpolated on the same
# roughly linear trend with wavelength.
APERTURE_DIAM_ARCSEC: dict[str, float] = {
    "f560w": 0.60,
    "f770w": 0.70,
    "f1000w": 0.90,
    "f1280w": 1.20,
    "f1500w": 1.20,
    "f1800w": 1.50,
    "f2100w": 1.70,
}


@dataclass(frozen=True)
class Release:
    """The staged directories of one field."""

    field: str  # arc/grizli field name, e.g. "uds"
    local: str  # local directory under MINERVA/data
    nircam: str  # NIRCam mosaic version directory, e.g. "n3.0"
    miri: str  # MIRI mosaic version directory, e.g. "m3.1"
    seg_dir: str  # release prefix of the directory holding the segmap
    cat_dir: str  # release prefix of the directory holding the SUPER catalog
    # Put every band on one trial patch instead of each band's own deepest spot.
    # Set for EGS, where the MIRI strips are so thin that per-band centres land
    # in different places and no source is measured in all seven bands.
    common_center: bool = False


RELEASES = [
    Release("uds", "UDS", "n3.0", "m3.1", "n3.0_v1.2", "n3.0_m3.1_v1.2.1"),
    Release("cosmos", "COSMOS", "n3.0", "m3.0", "n3.0_v1.0", "n3.0_m3.0_v1.0.1"),
    Release("egs", "EGS", "n2.0", "m2.1", "n2.0_v1.3", "n2.0_m2.1_v1.3.1",
            common_center=True),
]


def _one(paths: list[Path], what: str) -> Path | None:
    """Return the single match, or None with a warning."""
    if not paths:
        log.warning("  missing %s", what)
        return None
    if len(paths) > 1:
        log.warning("  %d matches for %s, taking %s", len(paths), what, paths[0].name)
    return paths[0]


def _find(root: Path, pattern: str, what: str) -> Path | None:
    """Glob ``pattern`` anywhere under ``root``."""
    if not root.exists():
        log.warning("  no directory %s", root)
        return None
    return _one(sorted(root.rglob(pattern)), what)


def _find_release(root: Path, prefer: str, pattern: str, what: str) -> Path | None:
    """Glob ``pattern`` under the field root, preferring the ``prefer`` release.

    The same product can sit in more than one release directory (and the
    directory name varies: ``n3.0_v1.2`` from CANFAR, ``n3.0_v1.2_SEC`` from
    the Drive delivery), so match on the release prefix rather than an exact
    directory name and fall back to any release that has the file.
    """
    if not root.exists():
        log.warning("  no directory %s", root)
        return None
    hits = sorted(root.rglob(pattern))
    return _one([h for h in hits if prefer in str(h)] or hits, what)


def n_frames(csv: Path) -> int:
    """Number of exposures in a grizli/MINERVA ``_wcs.csv`` frame table."""
    with open(csv) as fh:
        return sum(1 for _ in fh) - 1


# Coverage fractions accepted for the trial box, tried in order. The last two
# are for the thin EGS MIRI strips: F560W, F1280W and F1800W cover 3-6% of their
# mosaic and their best 6 arcmin box reaches only 0.24-0.31, so anything stricter
# sends them to a full-field run. A sparse patch still beats the whole field,
# since footprint_filter drops the uncovered sources anyway.
COVERAGE_STEPS = (0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.2)


def deepest_patch(wht_path: Path, radius_arcmin: float) -> tuple[float, float] | None:
    """Return ``(ra, dec)`` of the deepest well-covered patch of a weight map.

    The weight map is block-averaged to about 2 arcsec per pixel, smoothed with
    a box the size of the trial patch, and the peak is taken over positions
    where the box is covered. Full coverage is tried first and the requirement
    is relaxed in steps, because MIRI footprints can be fragmented enough that
    no completely covered box of that size exists -- notably in EGS. Returns
    None when the band has no usable patch at all, which the caller turns into
    a full-field run rather than a silently bogus centre.
    """
    with fits.open(wht_path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        wht = np.nan_to_num(hdu.data.astype(np.float32))
        wcs = WCS(hdu.header)
    scale = abs(wcs.pixel_scale_matrix[1, 1]) * 3600.0  # arcsec/pixel
    binf = max(1, int(round(2.0 / scale)))
    small = block_reduce(wht, binf, func=np.mean)
    box = max(3, int(round(2 * radius_arcmin * 60.0 / (scale * binf))))
    covered = uniform_filter((small > 0).astype(np.float32), box, mode="constant", cval=0.0)
    # mean over the covered pixels, so a partly covered box is not penalised
    # twice (once by the coverage cut, once by averaging in its zeros)
    depth = uniform_filter(small, box, mode="constant", cval=0.0) / np.maximum(covered, 1e-6)

    for want in COVERAGE_STEPS:
        scored = np.where(covered >= want, depth, 0.0)
        if scored.max() > 0:
            break
    else:
        log.warning("  no covered patch of %.1f arcmin: full-field run", 2 * radius_arcmin)
        return None

    iy, ix = np.unravel_index(int(np.argmax(scored)), scored.shape)
    ra, dec = wcs.all_pix2world([[(ix + 0.5) * binf - 0.5, (iy + 0.5) * binf - 0.5]], 0)[0]
    nonzero = small[small > 0]
    pct = 100.0 * float((nonzero < depth[iy, ix]).mean())
    level = "" if want == COVERAGE_STEPS[0] else f", coverage relaxed to {want:.2f}"
    log.info(
        "  trial centre %.5f %.5f  (mean wht %.3g = %.0fth pct of the footprint%s)",
        ra, dec, depth[iy, ix], pct, level,
    )
    return float(ra), float(dec)


def common_patch(wht_paths: list[Path], radius_arcmin: float) -> tuple[float, float] | None:
    """Return ``(ra, dec)`` of the best patch covered by *every* band.

    Per-band centres put each band on its own deepest spot, which is fine in
    isolation but means no source is measured in all of them. This scores the
    intersection of the bands' footprints instead, so one patch serves the whole
    field and the bands are directly comparable.

    The bands of a MIRI release share a mosaic grid, so the intersection is a
    plain AND. Depth is normalised per band before scoring and the worst band is
    taken, so the centre is one that every band covers well rather than one a
    single deep band drags around. Returns None if the bands share no pixels.
    """
    masks, depths, wcs, binf, scale = [], [], None, 1, 1.0
    for path in wht_paths:
        with fits.open(path, memmap=True) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            wht = np.nan_to_num(hdu.data.astype(np.float32))
            if wcs is None:
                wcs = WCS(hdu.header)
                scale = abs(wcs.pixel_scale_matrix[1, 1]) * 3600.0
                binf = max(1, int(round(2.0 / scale)))
        small = block_reduce(wht, binf, func=np.mean)
        masks.append(small > 0)
        nonzero = small[small > 0]
        if nonzero.size == 0:
            return None
        depths.append(small / np.median(nonzero))

    overlap = np.logical_and.reduce(masks)
    if not overlap.any():
        log.warning("  bands share no covered pixels: per-band centres")
        return None

    box = max(3, int(round(2 * radius_arcmin * 60.0 / (scale * binf))))
    worst = np.min(np.stack(depths), axis=0) * overlap
    score = uniform_filter(worst.astype(np.float32), box, mode="constant", cval=0.0)
    coverage = uniform_filter(overlap.astype(np.float32), box, mode="constant", cval=0.0)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    ra, dec = wcs.all_pix2world([[(ix + 0.5) * binf - 0.5, (iy + 0.5) * binf - 0.5]], 0)[0]
    log.info(
        "  common centre %.5f %.5f  (%.1f%% of the box covered by all %d bands; "
        "the field's all-band overlap is %.2f%% of the mosaic)",
        ra, dec, 100 * coverage[iy, ix], len(wht_paths), 100 * overlap.mean(),
    )
    return float(ra), float(dec)


def band_configs(rel: Release) -> list[dict]:
    """Build one config dict per staged MIRI band of ``rel``."""
    root = DATA / rel.local
    log.info("=== %s", rel.local)

    sci_hi = _find(root / rel.nircam, "*-f444w-clear_drc_sci_bkgsub.fits", "F444W bkgsub sci")
    # The bkgsub sci breaks the automatic '_sci.fits' -> '_wht.fits' guess,
    # so name the weight map explicitly (needed by the SNR-weighted build
    # schemes and by repair_saturated).
    wht_hi = _find(root / rel.nircam, "*-f444w-clear_drc_wht.fits", "F444W wht")
    csv_hi = _find(root / rel.nircam, "*-f444w-clear_wcs.csv", "F444W wcs csv")
    # both detection flavours ship a segmap; the SUPER catalog is keyed to the
    # ACS+WEBB chi-mean one, so take that
    segmap = _find_release(root, rel.seg_dir, "*ACS+WEBB*SEGMAP.fits", "segmap")
    catalog = _find_release(root, rel.cat_dir, "*SUPER_CATALOG*.fits", "SUPER catalog")
    if not all([sci_hi, wht_hi, csv_hi, segmap, catalog]):
        log.warning("  %s incomplete, skipped", rel.local)
        return []

    miri_dir = root / rel.miri
    if not miri_dir.exists():
        log.warning("  no MIRI directory %s", miri_dir)
        return []

    # One centre for the whole field when asked for, so every band measures the
    # same sources. Only worth it where the bands overlap poorly enough that
    # per-band centres would land in disjoint places.
    shared = None
    if rel.common_center:
        wht_all = [
            sci.with_name(sci.name.replace("_drz_sci_extrabkg", "_drz_wht"))
            for sci in sorted(miri_dir.glob("*_drz_sci_extrabkg.fits"))
        ]
        shared = common_patch([w for w in wht_all if w.exists()], R_TRIAL_ARCMIN)

    configs = []
    for sci_lo in sorted(miri_dir.glob("*_drz_sci_extrabkg.fits")):
        band = sci_lo.name.split("-80mas-")[1].split("_")[0]  # e.g. "f770w"
        wht_lo = sci_lo.with_name(sci_lo.name.replace("_drz_sci_extrabkg", "_drz_wht"))
        csv_lo = _one(sorted((root / rel.miri).glob(f"*_{band}_wcs.csv")), f"{band} wcs csv")
        if not wht_lo.exists() or csv_lo is None:
            log.warning("  %s: missing weight or wcs csv, skipped", band)
            continue
        log.info("  %s", band)
        patch = shared or deepest_patch(wht_lo, R_TRIAL_ARCMIN)
        configs.append(
            {
                "name": f"{rel.field}_{band}",
                "out_dir": f"{rel.field}_{band}",
                "sci_hi": str(sci_hi),
                "wht_hi": str(wht_hi),
                "segmap": str(segmap),
                "catalog": str(catalog),
                "csv_hi": str(csv_hi),
                "sci_lo": str(sci_lo),
                "wht_lo": str(wht_lo),
                "csv_lo": str(csv_lo),
                "expect_frames": [n_frames(csv_hi), n_frames(csv_lo)],
                "psf_dir": str(PSF_DIR),
                "pattern_hi": rf"{rel.local}_NRC.._F444W_MJD\d+_GRID25_OS4",
                "pattern_lo": rf"{rel.local}_MIRI_{band.upper()}_MJD\d+_GRID9_OS4",
                "filter_lo": band,
                # 4.0 for every band: psf_size sets the hi-res support too,
                # and the F444W grids are only 4.09 arcsec across, so a larger
                # stamp would measure the grid edge (see TODO.md). The reddest
                # MIRI bands want more, once the NIRCam grids are regenerated
                # at a larger FOV.
                "psf_size": 4.0,
                # Stated explicitly rather than left to the default: these
                # exposure lists span up to four years across a dozen or more
                # epochs, the grids are MJD-tagged and looked up by nearest
                # date, and any mode that collapses the list ("modal" returns
                # a single date) throws that resolution away invisibly.
                "psf_date_mode": "all",
                # "warn" until the rebuild cost is affordable, then
                # "rebuild" (TODO.md). Never leave it "off": a grid built
                # another way is a silent photometric difference.
                "psf_provenance": "warn",
                "psf_blur_fwhm": "default",
                "footprint_filter": True,
                # In-memory saturation repair at load time: fill the wht=0
                # cores in the F444W template image with the fitted PSF,
                # flag the star-dominated segments (FLAG_SATURATED_TMPL
                # group ids -> one scene per star), and write the per-star
                # before/after comparison to <out_dir>/repaired/. Mosaics
                # on disk stay untouched.
                # The 30" halo grids for the flag model are derived from
                # pattern_hi ({prefix}_.._FOV30_GRID1_OS4) and built once
                # on demand, so no explicit repair_psf_pattern is needed.
                "repair_saturated": True,
                "repair_kwargs": {"min_buffer_snr": 200},
                "trial": (
                    {
                        "center": [round(patch[0], 5), round(patch[1], 5)],
                        "radius": R_TRIAL_ARCMIN,
                    }
                    if patch
                    else None
                ),
                "bg_filter_sigma": 64.0,
                "fit": {
                    "fit_astrometry_joint": True,
                    "scene_minimum_bright": 5,
                    # larger scenes before the local threshold bisection kicks
                    # in, and no long-range gluing of underfilled scenes
                    # (radius in 40 mas reference pixels; 1000 px = 40").
                    # The cap now also binds through merge_small_scenes, and a
                    # scene costs quadratically in its size (the joint solve's
                    # Schur complement is dense), so it is a real ceiling
                    # rather than advice; matches FitConfig's default.
                    "scene_max_size": 1000,
                    "scene_max_merge_radius": 1000,
                    "aperture_diam": APERTURE_DIAM_ARCSEC[band],
                },
                "scene_plots": True,
            }
        )
    return configs


HEADER = """\
# MINERVA {local} {band_up} template-fitting run (mophongo.pipeline.RunConfig).
#
# Generated by examples/make_minerva_configs.py from the staged release
# {nircam} NIRCam / {miri} MIRI / {cat_dir} catalog. Follows uds_770_dr0.json:
# F444W background-subtracted mosaic as the template image, the release segmap
# and SUPER catalog, one 80 mas MIRI band fitted.
#
# expect_frames are the row counts of the two WCS tables; trial.center is the
# deepest fully covered {r} arcmin patch of the MIRI weight map. Set
# "trial" to null for a full-field run.
#
# Run from examples/minerva/:  python -m mophongo.pipeline {name}.json
"""


def main(argv: list[str]) -> None:
    want = [a for a in argv if a in {r.field for r in RELEASES}]
    OUT.mkdir(exist_ok=True)
    written = 0
    for rel in RELEASES:
        if want and rel.field not in want:
            continue
        for cfg in band_configs(rel):
            path = OUT / f"{cfg['name']}.json"
            head = HEADER.format(
                local=rel.local,
                band_up=cfg["filter_lo"].upper(),
                nircam=rel.nircam,
                miri=rel.miri,
                cat_dir=rel.cat_dir,
                r=R_TRIAL_ARCMIN,
                name=cfg["name"],
            )
            path.write_text(head + json.dumps(cfg, indent=2) + "\n")
            written += 1
            log.info("  wrote %s", path.relative_to(OUT.parent))
    log.info("%d configs in %s", written, OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
