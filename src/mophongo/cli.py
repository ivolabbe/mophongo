"""Command-line access to the data products of a finished run.

Entry-point orchestrator in the spirit of :mod:`mophongo.repair`: every
subcommand is a thin wrapper over an existing :class:`~mophongo.pipeline.Pipeline`
or :class:`~mophongo.psf_map.PSFRegionMap` method, so no algorithmic logic
lives here.

Subcommands::

    mophongo psf    <map.geojson|run.json> RA DEC   # PSF/kernel stamp -> FITS
    mophongo stamps <run.json> ID [ID ...]          # one source's cutouts -> FITS
    mophongo diag   <run.json> ID [ID ...]          # per-source diagnostic PNG
    mophongo info   <run.json>                      # summarize a run
    mophongo run    <run.json> [step ...]           # run steps (see pipeline.STEPS)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# Region-map kind -> the RunConfig image whose header defines its pixel grid.
# The matching kernel is convolved into the high-resolution templates, so it
# lives on the detection grid like ``psf_hi``.
MAP_KINDS: dict[str, str] = {
    "psf_hi": "sci_hi",
    "psf_lo": "sci_lo",
    "kernel": "sci_hi",
}

# Provenance columns written on the region maps by the pipeline, mapped to
# FITS-legal (<=8 character) keywords.
PROVENANCE_KEYS: dict[str, tuple[str, str]] = {
    "pattern": ("PSFPATT", "STDPSF pattern of the source grids"),
    "psf_size": ("PSFSIZE", "requested stamp size [arcsec]"),
    "blur_fwhm": ("PSFBLUR", "extra Gaussian broadening [arcsec FWHM]"),
    "kernel_method": ("KERNMETH", "matching_kernel method"),
    "kernel_reg": ("KERNREG", "matching_kernel regularization"),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _card(header: fits.Header, key: str, value: Any, comment: str = "") -> None:
    """Set a header card, writing non-finite floats as undefined values.

    ``nan`` is a legitimate product value (an unmeasured encircled energy, an
    unregularized kernel) but is illegal in a FITS header, so it becomes an
    undefined card rather than being dropped silently.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        value = None
    header[key] = (value, comment)


def _resolve_map(path: str | Path, kind: str = "psf_lo") -> tuple[Path, str, Path | None]:
    """Resolve a map argument to a geojson file, its kind, and its run config.

    Args:
        path: Either a ``<name>_{psf_hi,psf_lo,kernel}.geojson`` written by
            :meth:`~mophongo.psf_map.PSFRegionMap.to_file`, or a run config
            JSON (or run directory) whose ``out_dir`` holds the cached maps.
        kind: Which map to take when ``path`` is a run config, or when the
            geojson name does not carry one of the standard suffixes.

    Returns:
        Tuple of the geojson path, the resolved kind, and the run config that
        produced it (``None`` when no config sits beside the map).
    """
    p = Path(path)
    if p.suffix == ".geojson":
        for name in MAP_KINDS:
            if p.stem.endswith(f"_{name}"):
                cfg_path = p.with_name(p.stem[: -(len(name) + 1)] + ".json")
                return p, name, cfg_path if cfg_path.exists() else None
        logger.warning(
            "%s does not carry a %s suffix; assuming kind=%r",
            p.name, "/".join(MAP_KINDS), kind,
        )
        return p, kind, None

    if p.suffix not in (".json", "") and not p.is_dir():
        raise ValueError(f"expected a .geojson map or a .json run config, got {p.name}")
    from .pipeline import Pipeline, RunConfig

    cfg_path = Pipeline._resolve_config_path(p)
    cfg = RunConfig.from_json(cfg_path)
    geojson = Path(cfg.out_dir) / f"{cfg.name}_{kind}.geojson"
    if not geojson.exists():
        # a relative out_dir resolves against the process working directory
        # (as everywhere else); fall back to the directory the config sits in,
        # which is where a finished run keeps its maps
        beside = cfg_path.parent / geojson.name
        if beside.exists():
            logger.info("%s not found; using %s", geojson, beside)
            geojson = beside
    return geojson, kind, cfg_path


def _reference_header(config_path: Path | None, kind: str) -> fits.Header | None:
    """Header of the mosaic whose pixel grid the map's stamps are drizzled on."""
    if config_path is None:
        return None
    from .pipeline import RunConfig

    image = Path(getattr(RunConfig.from_json(config_path), MAP_KINDS[kind]))
    if not image.exists():
        logger.warning("reference image %s not found; no WCS from the config", image)
        return None
    return fits.getheader(image)


def stamp_wcs(
    shape: tuple[int, int],
    ra: float,
    dec: float,
    *,
    ref_header: fits.Header | None = None,
    pixel_scale: float | None = None,
) -> WCS | None:
    """Celestial WCS centering a stamp of ``shape`` on ``(ra, dec)``.

    The stamp inherits the orientation and pixel scale of its parent grid when
    ``ref_header`` is given; ``pixel_scale`` builds a north-up, east-left
    tangent plane instead.

    Args:
        shape: Stamp shape ``(ny, nx)``.
        ra: Right ascension of the stamp center in degrees.
        dec: Declination of the stamp center in degrees.
        ref_header: Header of the parent mosaic; its CD/PC matrix is kept and
            only the reference pixel/value are moved. Any SIP distortion is
            dropped, since it is not valid away from the parent grid.
        pixel_scale: Stamp pixel scale in arcsec, used when ``ref_header`` is
            ``None``.

    Returns:
        The WCS, or ``None`` when neither input was given.
    """
    ny, nx = shape
    if ref_header is not None:
        w = WCS(ref_header).celestial.deepcopy()
        w.sip = None
        w.wcs.ctype = [str(c).replace("-SIP", "") for c in w.wcs.ctype]
    elif pixel_scale is not None:
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    else:
        return None
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.crval = [float(ra), float(dec)]
    w.wcs.set()
    return w


# ---------------------------------------------------------------------------
# psf / kernel stamps
# ---------------------------------------------------------------------------
def psf_to_fits(
    map_path: str | Path,
    ra: float,
    dec: float,
    out_path: str | Path | None = None,
    *,
    kind: str = "psf_lo",
    pixel_scale: float | None = None,
    overwrite: bool = True,
) -> Path:
    """Write the PSF or matching kernel at a sky position to a FITS file.

    Reads a cached :class:`~mophongo.psf_map.PSFRegionMap`, looks up the region
    containing ``(ra, dec)``, and writes that region's stamp with a WCS
    centered on the requested position. The stamp's encircled energy and the
    provenance of the map (PSF pattern, stamp size, broadening, kernel method
    and regularization) go into the header.

    Args:
        map_path: A ``*_psf_hi/_psf_lo/_kernel.geojson`` map, or a run config
            JSON (then ``kind`` selects the map).
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        out_path: Output FITS path; defaults to ``<map stem>_key<key>.fits``
            in the current directory.
        kind: Map to use when ``map_path`` is a run config.
        pixel_scale: Stamp pixel scale in arcsec. By default it comes from the
            map's reference mosaic, which also fixes the stamp orientation.
        overwrite: Overwrite an existing output file.

    Returns:
        The path written.
    """
    from .psf_map import PSFRegionMap

    geojson, kind, config_path = _resolve_map(map_path, kind)
    if not geojson.exists():
        raise FileNotFoundError(f"region map {geojson} not found")
    prm = PSFRegionMap.from_geojson(str(geojson))
    if prm.psfs is None:
        raise FileNotFoundError(
            f"no stamp cube beside {geojson.name}; expected "
            f"{geojson.with_suffix('.fits').name}"
        )

    key = prm.lookup_key(float(ra), float(dec))
    if key is None:
        logger.warning("no region at (%.6f, %.6f); using key 0", ra, dec)
        key = 0
    stamp = np.asarray(prm.psfs[int(key)], dtype=np.float32)

    ref_header = None if pixel_scale is not None else _reference_header(config_path, kind)
    w = stamp_wcs(stamp.shape, ra, dec, ref_header=ref_header, pixel_scale=pixel_scale)
    header = fits.Header() if w is None else w.to_header()
    if w is None:
        logger.warning(
            "no reference image and no --pixel-scale; writing %s without a WCS",
            geojson.stem,
        )
    _card(header, "MAPFILE", geojson.name[:60], "region map the stamp came from")
    _card(header, "MAPKIND", kind, "psf_hi, psf_lo, or kernel")
    _card(header, "PSFKEY", int(key), "region index in the map")
    _card(header, "RA_PSF", float(ra), "requested position [deg]")
    _card(header, "DEC_PSF", float(dec), "requested position [deg]")
    _card(header, "STAMPSUM", float(np.nansum(stamp)), "sum of the stamp")
    _card(header, "EE_BOX", float(prm.ee_box[int(key)]), "encircled energy in the stamp")
    _card(header, "EE_RLIM", float(prm.ee_rlim[int(key)]), "encircled energy within R_LIM")
    _card(header, "R_LIM", float(prm.r_lim), "inscribed-circle radius [arcsec]")
    _card(header, "PSCALE", float(prm.pscale), "stamp pixel scale [arcsec]")
    for column, (fits_key, comment) in PROVENANCE_KEYS.items():
        if column in prm.regions.columns and len(prm.regions):
            _card(header, fits_key, prm.regions[column].iloc[int(key)], comment)

    out = Path(out_path) if out_path else Path(f"{geojson.stem}_key{int(key)}.fits")
    fits.writeto(out, stamp, header, overwrite=overwrite)
    logger.info(
        "wrote %s: %s key %d, %dx%d, sum %.4f, EE(box) %.4f",
        out, kind, int(key), *stamp.shape, float(np.nansum(stamp)),
        float(prm.ee_box[int(key)]),
    )
    return out


# ---------------------------------------------------------------------------
# per-source products
# ---------------------------------------------------------------------------
def _native_lo_wcs(pipe: Any, ifilt: int) -> WCS | None:
    """WCS of the fitted band on its own grid (``pipe.wcs[ifilt]`` after an
    upsampling run is the reference grid, not the band's native one)."""
    cfg = getattr(pipe, "run_config", None)
    if cfg is not None and Path(cfg.sci_lo).exists():
        return WCS(fits.getheader(cfg.sci_lo))
    return pipe.wcs[ifilt] if getattr(pipe, "wcs", None) is not None else None


def source_stamps_to_fits(
    pipe: Any,
    source_id: int,
    out_path: str | Path,
    *,
    ifilt: int = 1,
    half_size: int | None = None,
    overwrite: bool = True,
) -> Path:
    """Write one source's fitted products to a multi-extension FITS file.

    Thin wrapper over :meth:`~mophongo.pipeline.Pipeline.source_products`: the
    cutouts it returns are written as image extensions carrying the sliced WCS
    of their parent grid, the two PSF stamps get a WCS centered on the source,
    and the fitted scalars go into the primary header (with the full fit-table
    row as a ``FITROW`` table extension).

    Args:
        pipe: Pipeline after :meth:`~mophongo.pipeline.Pipeline.run` or
            :meth:`~mophongo.pipeline.Pipeline.load_fit`.
        source_id: Catalog id.
        out_path: Output FITS path.
        ifilt: Fitted image index (1-based, as elsewhere).
        half_size: Window half-size in pixels; ``None`` uses the template
            footprint.
        overwrite: Overwrite an existing output file.

    Returns:
        The path written.
    """
    prod = pipe.source_products(source_id, ifilt=ifilt, half_size=half_size)

    wcs_hi = pipe.wcs[0] if getattr(pipe, "wcs", None) is not None else None
    wcs_lo = _native_lo_wcs(pipe, ifilt)
    x, y = prod["position"]
    ra = dec = None
    if wcs_hi is not None:
        ra, dec = (float(v) for v in wcs_hi.wcs_pix2world(x, y, 0))
    fit_wcs = pipe.wcs[ifilt] if getattr(pipe, "wcs", None) is not None else None

    hdr = fits.Header()
    _card(hdr, "ID", int(prod["id"]), "catalog id")
    _card(hdr, "IFILT", int(ifilt), "fitted image index")
    if getattr(pipe, "run_config", None) is not None:
        _card(hdr, "RUNNAME", pipe.run_config.name, "run these products come from")
    _card(hdr, "X", float(x), "source x on the reference grid [pix]")
    _card(hdr, "Y", float(y), "source y on the reference grid [pix]")
    if ra is not None:
        _card(hdr, "RA", ra, "source position [deg]")
        _card(hdr, "DEC", dec, "source position [deg]")
    _card(hdr, "FLUX", prod["flux"], "fitted template amplitude")
    _card(hdr, "ERR", prod["err"], "flux error")
    _card(hdr, "ERRPRED", prod["err_pred"], "predicted flux error")
    _card(hdr, "EEPSFLO", prod["ee_psf_lo"], "encircled energy of the band PSF")
    _card(hdr, "FLAG", int(prod["flag"]), "template flag")
    _card(hdr, "DX", prod["shift"][0], "applied astrometric shift [pix]")
    _card(hdr, "DY", prod["shift"][1], "applied astrometric shift [pix]")
    hdul = [fits.PrimaryHDU(header=hdr)]

    # (extension, product key, grid): reference- and fitting-grid cutouts
    # inherit the sliced parent WCS; the PSF stamps are re-centered on the
    # source, on the grid their own band was drizzled onto.
    layout = (
        ("IMG_HI", "img_hi", "hi"),
        ("SEGMAP", "segmap", "hi"),
        ("TMPL_HI", "tmpl_hi", "hi"),
        ("IMG_LO", "img_lo", "fit"),
        ("TMPL_LO", "tmpl_lo", "fit"),
        ("MODEL", "model", "fit"),
        ("RESID", "residual", "fit"),
        ("PSF_HI", "psf_hi", "psf_hi"),
        ("PSF_LO", "psf_lo", "psf_lo"),
    )
    for extname, key, grid in layout:
        data = prod.get(key)
        if data is None:
            continue
        data = np.asarray(data)
        w = None
        if grid == "hi" and wcs_hi is not None and prod["slices_hi"] is not None:
            w = wcs_hi.slice(prod["slices_hi"])
        elif grid == "fit" and fit_wcs is not None:
            w = fit_wcs.slice(prod["slices_lo"])
        elif grid in ("psf_hi", "psf_lo") and ra is not None:
            ref = wcs_hi if grid == "psf_hi" else wcs_lo
            w = stamp_wcs(
                data.shape, ra, dec,
                ref_header=None if ref is None else ref.to_header(),
            )
        ext_hdr = fits.Header() if w is None else w.to_header()
        ext_hdr["EXTNAME"] = extname
        dtype = np.int32 if key == "segmap" else np.float32
        hdul.append(fits.ImageHDU(data.astype(dtype), ext_hdr))

    if prod["row"] is not None:
        row = prod["row"]
        hdu = fits.table_to_hdu(row.table[row.index : row.index + 1])
        hdu.header["EXTNAME"] = "FITROW"
        hdul.append(hdu)

    out = Path(out_path)
    fits.HDUList(hdul).writeto(out, overwrite=overwrite)
    logger.info(
        "wrote %s: id %d, %d extensions, flux %.4g +/- %.4g",
        out, int(prod["id"]), len(hdul) - 1, prod["flux"], prod["err"],
    )
    return out


def source_diagnostic_png(
    pipe: Any,
    source_id: int,
    out_path: str | Path,
    *,
    ifilt: int = 1,
    style: str = "subphot",
    size: int | None = None,
    half_size: int | None = None,
) -> Path:
    """Write a per-source diagnostic image.

    Args:
        pipe: Pipeline after :meth:`~mophongo.pipeline.Pipeline.run` or
            :meth:`~mophongo.pipeline.Pipeline.load_fit`.
        source_id: Catalog id.
        out_path: Output PNG path.
        ifilt: Fitted image index (1-based, as elsewhere).
        style: ``"subphot"`` for the IDL-style 6-panel
            (:meth:`~mophongo.pipeline.Pipeline.diagnose_subphot`), ``"stages"``
            for the template-construction row
            (:meth:`~mophongo.pipeline.Pipeline.diagnose_source`).
        size: Stamp side in pixels, ``subphot`` style only.
        half_size: Window half-size in pixels, ``stages`` style only.

    Returns:
        The path written.
    """
    out = Path(out_path)
    if style == "subphot":
        pipe.diagnose_subphot(source_id, ifilt=ifilt, size=size, save=out)
    elif style == "stages":
        import matplotlib.pyplot as plt

        fig, _ = pipe.diagnose_source(
            source_id, ifilt=ifilt, half_size=half_size, save=out
        )
        plt.close(fig)
    else:
        raise ValueError(f"unknown diagnostic style {style!r}")
    logger.info("wrote %s (%s, id %d)", out, style, int(source_id))
    return out


def _load_fitted(config: str | Path, ifilt: int):
    """Restore a finished run from its config (loads the mosaics)."""
    from .pipeline import Pipeline

    return Pipeline.from_config(config).load_fit(ifilt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#: Placeholders for the ``RunConfig`` fields that have no default. They are
#: paths and a run name, so there is nothing sensible to default them to; the
#: written config is a template to fill in, not one that would run as-is.
_CONFIG_PLACEHOLDERS: dict[str, str] = {
    "name": "<run name, used as the output filename prefix>",
    "out_dir": "<output directory>",
    "sci_hi": "<high-resolution detection mosaic .fits>",
    "segmap": "<segmentation map .fits>",
    "catalog": "<detection catalog .fits>",
    "sci_lo": "<low-resolution science mosaic to fit .fits>",
    "wht_lo": "<low-resolution weight (inverse variance) map .fits>",
    "csv_hi": "<high-resolution exposure wcs .csv>",
    "csv_lo": "<low-resolution exposure wcs .csv>",
}


def write_default_config(path: str | Path, *, force: bool = False) -> Path:
    """Write a run config with every setting written out at its default.

    The `fit` block is a plain dict on :class:`~mophongo.pipeline.RunConfig`
    and defaults to empty, which would make a dumped default config say
    nothing about the fit at all. It is expanded to the full
    :class:`~mophongo.fit.FitConfig` defaults here, so the file lists every
    knob and its value rather than leaving the reader to find them in the
    source.

    The nine required fields have no defaults -- they are input paths and a run
    name -- so they are written as angle-bracket placeholders. The result is a
    template to fill in; it parses but does not run.

    Args:
        path: Output json.
        force: Overwrite an existing file instead of refusing.

    Returns:
        The path written.

    Raises:
        FileExistsError: If ``path`` exists and ``force`` is false. A config is
            hand-edited after it is generated, so silently replacing one would
            discard work.
    """
    import json
    from dataclasses import asdict
    from datetime import date

    from .fit import FitConfig
    from .pipeline import RunConfig

    out = Path(path)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to overwrite")

    cfg = RunConfig(**_CONFIG_PLACEHOLDERS)
    data = asdict(cfg)
    data["fit"] = asdict(FitConfig())

    header = (
        f"# mophongo default run config, written {date.today()} by\n"
        "# `mophongo config`. Every RunConfig, PsfConfig and FitConfig setting\n"
        "# is listed at its default value.\n"
        "#\n"
        "# The <angle bracket> entries have no default and must be filled in\n"
        "# before this will run. Paths are resolved relative to the config.\n"
        "#\n"
        "# Lines starting with '#' are comments and are stripped on load.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + json.dumps(data, indent=2, default=str) + "\n")
    logger.info("wrote default run config to %s", out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: ``mophongo <subcommand> ...``."""
    ap = argparse.ArgumentParser(
        prog="mophongo",
        description="Access the data products of a mophongo run.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_psf = sub.add_parser(
        "psf", help="write the PSF or matching kernel at a sky position to FITS"
    )
    p_psf.add_argument(
        "map", help="*_psf_hi/_psf_lo/_kernel.geojson map, or a run config json"
    )
    p_psf.add_argument("ra", type=float, help="right ascension [deg]")
    p_psf.add_argument("dec", type=float, help="declination [deg]")
    p_psf.add_argument("-o", "--out", default=None,
                       help="output FITS (default: <map stem>_key<key>.fits)")
    p_psf.add_argument("--map-kind", choices=list(MAP_KINDS), default="psf_lo",
                       help="map to take from a run config (default: psf_lo)")
    p_psf.add_argument("--pixel-scale", type=float, default=None,
                       help="stamp pixel scale in arcsec; default from the "
                            "map's reference mosaic, which also sets the "
                            "stamp orientation")

    p_stamps = sub.add_parser(
        "stamps", help="write one source's cutouts, PSFs, and fit row to FITS"
    )
    p_stamps.add_argument("config", help="run config json (or <out_dir>)")
    p_stamps.add_argument("ids", nargs="+", type=int, help="catalog ids")
    p_stamps.add_argument("--ifilt", type=int, default=1,
                          help="fitted image index (default: 1)")
    p_stamps.add_argument("--half-size", type=int, default=None,
                          help="window half-size in pixels (default: the "
                               "template footprint)")
    p_stamps.add_argument("-o", "--out-dir", default=None,
                          help="output directory (default: the run out_dir)")

    p_diag = sub.add_parser("diag", help="write a per-source diagnostic PNG")
    p_diag.add_argument("config", help="run config json (or <out_dir>)")
    p_diag.add_argument("ids", nargs="+", type=int, help="catalog ids")
    p_diag.add_argument("--ifilt", type=int, default=1,
                        help="fitted image index (default: 1)")
    p_diag.add_argument("--style", choices=("subphot", "stages"), default="subphot",
                        help="subphot 6-panel or template-construction stages "
                             "(default: subphot)")
    p_diag.add_argument("--size", type=int, default=None,
                        help="stamp side in pixels (subphot style)")
    p_diag.add_argument("--half-size", type=int, default=None,
                        help="window half-size in pixels (stages style)")
    p_diag.add_argument("-o", "--out-dir", default=None,
                        help="output directory (default: the run out_dir)")

    p_info = sub.add_parser("info", help="summarize a run's config, inputs, and products")
    p_info.add_argument("config", help="run config json (or <out_dir>)")

    p_run = sub.add_parser("run", help="run pipeline steps (see pipeline.STEPS)")
    p_run.add_argument("config", help="run config json (or <out_dir>)")
    p_run.add_argument("steps", nargs="*", metavar="step",
                       help="steps to run (default: all)")

    p_config = sub.add_parser(
        "config", help="write a run config with every setting at its default"
    )
    p_config.add_argument("out", help="output json, e.g. default.json")
    p_config.add_argument("--force", action="store_true",
                          help="overwrite an existing file")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.cmd == "psf":
        psf_to_fits(
            args.map, args.ra, args.dec, args.out,
            kind=args.map_kind, pixel_scale=args.pixel_scale,
        )
        return

    if args.cmd == "config":
        try:
            write_default_config(args.out, force=args.force)
        except FileExistsError as exc:
            ap.error(str(exc))  # a clean message, not a traceback
        return

    if args.cmd == "info":
        from .pipeline import Pipeline

        Pipeline.from_config(args.config).info()
        return

    if args.cmd == "run":
        from .pipeline import main as pipeline_main

        pipeline_main([args.config, *args.steps])
        return

    pipe = _load_fitted(args.config, args.ifilt)
    out_dir = Path(args.out_dir) if args.out_dir else pipe.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = pipe.run_config.name
    for source_id in args.ids:
        if args.cmd == "stamps":
            source_stamps_to_fits(
                pipe, source_id, out_dir / f"{name}_{source_id}_stamps.fits",
                ifilt=args.ifilt, half_size=args.half_size,
            )
        else:
            source_diagnostic_png(
                pipe, source_id, out_dir / f"{name}_{source_id}_{args.style}.png",
                ifilt=args.ifilt, style=args.style,
                size=args.size, half_size=args.half_size,
            )


if __name__ == "__main__":
    main()
