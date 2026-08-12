"""Standalone saturated-star repair for drizzled mosaics.

One runnable entry point around the image-level repair in
:mod:`mophongo.saturate` and the catalog/segmap repair in
:mod:`mophongo.catalog`, for users who want "repaired" FITS images (and
optionally a catalog with a saturated-star flag) without running the
photometry pipeline.

Usage::

    python -m mophongo.repair sci.fits wht.fits
    mophongo-repair sci.fits wht.fits --catalog cat.fits --segmap seg.fits

Outputs, written next to the science image (or to ``--out-dir``):

* ``<sci>_repaired.fits`` / ``<wht>_repaired.fits`` — science image with
  each saturated core replaced by its best-fit PSF model, and the weight
  map with those pixels restored (``<sci>_subtracted.fits`` etc. with
  ``mode="subtract"``).
* ``<sci>_saturate_<mode>.csv`` — per-hole fit table from
  :func:`mophongo.saturate.repair_saturated_holes`.
* with ``--catalog`` and ``--segmap``: ``<catalog>_flagged.fits`` /
  ``<segmap>_flagged.fits`` — segments dominated by a repaired star's
  PSF model (model flux > ``--flux-frac`` of the observed flux) get
  the star's group id in ``FLAG_SATURATED_TMPL`` on their existing
  catalog rows and are
  zeroed in the segmap; a per-star diagnostic PNG is written
  (:func:`mophongo.catalog.flag_saturated_segments`). With ``--merge``
  the children are instead merged into one parent row
  (:func:`mophongo.catalog.repair_saturated_catalog`, ``_repaired``
  outputs).

The PSF model is a drizzled STDPSF (:class:`mophongo.psf.DrizzlePSF`),
which needs the grizli exposure listing ``<root>_wcs.csv`` next to the
mosaic; a missing CSV is reconstructed from public MAST/S3 cal-file
headers (network access required). STDPSF grids are loaded from
``--psf-dir`` and, when none match, built on demand with
:class:`mophongo.psf_factory.PSFFactory` (requires ``stpsf`` reference
data).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .catalog import flag_saturated_segments, repair_saturated_catalog
from .psf import DrizzlePSF
from .saturate import repair_saturated_holes
from .utils import get_slice_wcs, get_wcs_pscale

logger = logging.getLogger(__name__)

__all__ = [
    "build_drizzle_psf",
    "drizzled_psf_stamp",
    "repair_image",
    "repair_in_memory",
    "flag_catalog",
    "main",
]


# --------------------------------------------------------------------------
# PSF setup
# --------------------------------------------------------------------------


def _default_csv(sci_path: Path) -> Path:
    """Grizli ``_wcs.csv`` path derived from the mosaic filename.

    Mirrors :meth:`mophongo.psf.DrizzlePSF.read_wcs_csv`.
    """
    stem = str(sci_path)
    for tok in ("_drz_sci", "_drc_sci", "_sci"):
        stem = stem.split(tok)[0]
    return Path(stem + "_wcs.csv")


def _infer_filter(path: Path) -> str | None:
    """Filter name from a filename (e.g. ``...f444w...`` → ``F444W``)."""
    from .psf_factory import BACKENDS

    for backend in BACKENDS.values():
        try:
            return backend.filter_from_path(path)
        except ValueError:
            continue
    return None


def build_drizzle_psf(
    sci_path: str | Path,
    *,
    csv: str | Path | None = None,
    psf_dir: str | Path | None = None,
    psf_pattern: str | None = None,
    filter_name: str | None = None,
    build_psf: bool = True,
) -> tuple[DrizzlePSF, str]:
    """Build a :class:`~mophongo.psf.DrizzlePSF` with an STDPSF loaded.

    Parameters
    ----------
    sci_path
        Drizzled science mosaic.
    csv
        Grizli ``_wcs.csv`` exposure listing. Default: derived from
        *sci_path*; reconstructed from public cal-file headers when the
        file is missing (network access).
    psf_dir
        Directory with STDPSF FITS grids. Default ``<sci dir>/PSF``.
    psf_pattern
        Regex matched against STDPSF filenames (and ePSF keys at fit
        time). Default: ``NRC.._<FILTER>`` for NIRCam, ``<FILTER>``
        otherwise.
    filter_name
        Filter (e.g. ``"F444W"``). Default: parsed from the filename.
    build_psf
        When no STDPSF in *psf_dir* matches, build one per detector at
        the modal exposure epoch via
        :meth:`mophongo.psf_factory.PSFFactory.from_csv`.

    Returns
    -------
    (dpsf, psf_pattern)
        The configured :class:`~mophongo.psf.DrizzlePSF` and the
        resolved pattern to pass as ``psf_filter``.
    """
    from .psf_factory import BACKENDS, PSFFactory

    sci_path = Path(sci_path)
    csv_path = Path(csv) if csv is not None else _default_csv(sci_path)
    psf_dir = Path(psf_dir) if psf_dir is not None else sci_path.parent / "PSF"

    # DrizzlePSF reconstructs a missing CSV from public cal-file headers.
    dpsf = DrizzlePSF(driz_image=str(sci_path), csv_file=str(csv_path))

    if filter_name is None:
        filter_name = _infer_filter(sci_path)
    if psf_pattern is None:
        if filter_name is None:
            raise ValueError(
                "Cannot infer the filter from the filename; pass "
                "filter_name/--filter or psf_pattern/--psf-pattern."
            )
        instrument = ""
        for backend in BACKENDS.values():
            try:
                instrument = backend.decode_filename(dpsf.flt_keys[0][0])[0]
                break
            except ValueError:
                continue
        if instrument.upper() == "NIRCAM":
            psf_pattern = f"NRC.._{filter_name.upper()}"
        else:
            psf_pattern = filter_name.upper()

    def _load() -> None:
        if psf_dir.is_dir():
            dpsf.load_jwst_stdpsf(
                local_dir=str(psf_dir), filter_pattern=psf_pattern,
            )

    _load()
    if not dpsf.epsf_obj.epsf and build_psf:
        logger.info(
            "[repair] no STDPSF matched %r in %s — building with PSFFactory",
            psf_pattern, psf_dir,
        )
        # date_mode='cluster' builds one grid per observation epoch —
        # same policy as the pipeline's PSF generation — and DrizzlePSF
        # then picks the nearest-MJD grid per contributing exposure.
        factory = PSFFactory(
            outdir=str(psf_dir), num_psfs=1, oversample=4,
            fov_arcsec=8.0, date_mode="cluster", include_mjd=True,
        )
        factory.from_csv(csv_path, save=True)
        _load()
    if not dpsf.epsf_obj.epsf:
        raise RuntimeError(
            f"No STDPSF grids matched {psf_pattern!r} in {psf_dir}. "
            "Provide --psf-dir with STDPSF FITS files or allow --build-psf."
        )
    logger.info(
        "[repair] loaded ePSF keys: %s", sorted(dpsf.epsf_obj.epsf),
    )
    return dpsf, psf_pattern


def drizzled_psf_stamp(
    dpsf: DrizzlePSF,
    psf_pattern: str,
    *,
    npix: int = 201,
) -> np.ndarray:
    """Sum-normalised PSF stamp drizzled at the mosaic centre."""
    wcs = dpsf.driz_wcs
    if wcs.array_shape is not None:
        H, W = wcs.array_shape
    else:
        H = int(dpsf.driz_header["NAXIS2"])
        W = int(dpsf.driz_header["NAXIS1"])
    npix = int(min(npix, H - 2, W - 2))
    if npix % 2 == 0:
        npix -= 1  # odd size keeps the stamp centre on a pixel
    cy, cx = H // 2, W // 2
    sly = slice(cy - npix // 2, cy - npix // 2 + npix)
    slx = slice(cx - npix // 2, cx - npix // 2 + npix)
    sub = get_slice_wcs(wcs, slx, sly)
    sub.pixel_shape = (npix, npix)
    sub.pscale = get_wcs_pscale(sub)
    ra, dec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
    stamp = np.asarray(dpsf.get_psf(
        ra=ra, dec=dec, filter=psf_pattern, wcs_slice=sub,
        pixfrac=float(dpsf.driz_header.get("PIXFRAC", 0.75)),
        kernel=str(dpsf.driz_header.get("KERNEL", "square")),
    ), dtype=np.float64)
    stamp[stamp < 0] = 0.0
    total = float(stamp.sum())
    if total <= 0:
        raise RuntimeError("PSF stamp has zero flux")
    return stamp / total


def psf_fwhm_pix(dpsf: DrizzlePSF, psf_pattern: str) -> float:
    """PSF FWHM in mosaic pixels, from the area above half maximum."""
    stamp = drizzled_psf_stamp(dpsf, psf_pattern, npix=101)
    area = int(np.sum(stamp >= 0.5 * float(stamp.max())))
    return 2.0 * float(np.sqrt(area / np.pi))


# --------------------------------------------------------------------------
# Image repair
# --------------------------------------------------------------------------


def _read_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    """First HDU with data from *path*."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"No image data in {path}")


def repair_image(
    sci_path: str | Path,
    wht_path: str | Path,
    *,
    filter_name: str | None = None,
    csv: str | Path | None = None,
    psf_dir: str | Path | None = None,
    psf_pattern: str | None = None,
    build_psf: bool = True,
    dpsf: DrizzlePSF | None = None,
    out_dir: str | Path | None = None,
    mode: str = "repair",
    fwhm_pix: float | None = None,
    plots: bool = False,
    **repair_kwargs: Any,
) -> dict[str, Any]:
    r"""Repair saturated cores in one drizzled mosaic and write the results.

    Parameters
    ----------
    sci_path, wht_path
        Drizzled science and weight FITS images.
    filter_name, csv, psf_dir, psf_pattern, build_psf
        PSF setup, see :func:`build_drizzle_psf`. Ignored when *dpsf* is
        given (then *psf_pattern* defaults to *filter_name*).
    dpsf
        Pre-built :class:`~mophongo.psf.DrizzlePSF` with ePSFs loaded.
    out_dir
        Output directory. Default: next to *sci_path*.
    mode
        ``"repair"`` (fill saturated cores with the PSF model) or
        ``"subtract"`` (remove the full PSF halo; see
        :func:`mophongo.saturate.repair_saturated_holes`).
    fwhm_pix
        PSF FWHM in mosaic pixels for the donut geometry. Default:
        measured from a drizzled PSF stamp.
    plots
        Write per-source diagnostic PNGs.
    \*\*repair_kwargs
        Forwarded to :func:`mophongo.saturate.repair_saturated_holes`
        (e.g. ``min_buffer_snr``, ``merge_radius``).

    Returns
    -------
    dict
        The :func:`~mophongo.saturate.repair_saturated_holes` result plus
        ``sci_out, wht_out, csv_out, filter, psf_pattern, fwhm_pix, dpsf``.
    """
    if mode not in ("repair", "subtract"):
        raise ValueError(f"mode must be 'repair' or 'subtract', got {mode!r}")
    sci_path = Path(sci_path)
    wht_path = Path(wht_path)
    out_dir = Path(out_dir) if out_dir is not None else sci_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if filter_name is None:
        filter_name = _infer_filter(sci_path)
    if dpsf is None:
        dpsf, psf_pattern = build_drizzle_psf(
            sci_path, csv=csv, psf_dir=psf_dir, psf_pattern=psf_pattern,
            filter_name=filter_name, build_psf=build_psf,
        )
    elif psf_pattern is None:
        if filter_name is None:
            raise ValueError("psf_pattern or filter_name required with dpsf")
        psf_pattern = filter_name.upper()

    sci, sci_hdr = _read_image(sci_path)
    wht, wht_hdr = _read_image(wht_path)
    wcs = WCS(sci_hdr)

    if fwhm_pix is None:
        fwhm_pix = psf_fwhm_pix(dpsf, psf_pattern)
        logger.info("[repair] measured PSF FWHM: %.2f pix", fwhm_pix)

    suffix = "_repaired" if mode == "repair" else "_subtracted"
    sci_out = out_dir / f"{sci_path.stem}{suffix}.fits"
    wht_out = out_dir / f"{wht_path.stem}{suffix}.fits"
    # Mode-specific names so a repair pass and a subtract pass in the
    # same directory do not overwrite each other's fit tables.
    csv_out = out_dir / f"{sci_path.stem}_saturate_{mode}.csv"
    plot_dir = out_dir / f"{sci_path.stem}_saturate_{mode}_png" if plots else None

    res = repair_saturated_holes(
        sci, wht, dpsf=dpsf, wcs=wcs,
        mode=mode, fwhm_pix=float(fwhm_pix),
        psf_filter=psf_pattern,
        output_csv=csv_out, plot_dir=plot_dir,
        **repair_kwargs,
    )
    n_ok = int(np.sum(res["fits"]["ok"])) if len(res["fits"]) else 0
    logger.info(
        "[repair] %d/%d holes fit and %sed",
        n_ok, len(res["fits"]), mode,
    )

    for hdr in (sci_hdr, wht_hdr):
        hdr["SATREPAI"] = (True, "mophongo.repair: saturated stars treated")
        hdr["SATMODE"] = (mode, "repair action mode")
        hdr["SATNFIX"] = (n_ok, "number of saturated stars fit")
        if filter_name:
            hdr["SATFILT"] = (filter_name.upper(), "filter used for the PSF")
    fits.writeto(sci_out, res["sci"], sci_hdr, overwrite=True)
    fits.writeto(wht_out, res["wht"], wht_hdr, overwrite=True)
    # FITS copy of the fit table: preserves column types (the CSV turns
    # the bool ``ok`` into strings on read-back).
    table_out = out_dir / f"{sci_path.stem}_saturate_{mode}.fits"
    res["fits"].write(table_out, overwrite=True)
    logger.info("[repair] wrote %s, %s", sci_out, wht_out)

    res.update(
        sci_out=sci_out, wht_out=wht_out, csv_out=csv_out,
        table_out=table_out,
        filter=filter_name, psf_pattern=psf_pattern,
        fwhm_pix=float(fwhm_pix), dpsf=dpsf,
    )
    return res


# --------------------------------------------------------------------------
# In-memory repair (pipeline integration)
# --------------------------------------------------------------------------


def repair_in_memory(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    dpsf: DrizzlePSF,
    wcs: WCS,
    psf_pattern: str,
    catalog: Table | None = None,
    segmap: np.ndarray | None = None,
    out_dir: str | Path | None = None,
    fwhm_pix: float | None = None,
    flux_frac: float = 0.3,
    min_snr: float = 5.0,
    stamp_npix: int | None = None,
    zero_segments: bool = False,
    plots: bool = True,
    **repair_kwargs: Any,
) -> dict[str, Any]:
    r"""Repair saturated stars and flag the catalog, all in memory.

    Array-in / array-out variant of :func:`repair_image` +
    :func:`flag_catalog` for callers that already hold the data — the
    photometry pipeline runs this on its loaded ``sci_hi / wht_hi /
    catalog / segmap`` so the mosaics on disk never need to be repaired
    beforehand. Nothing is written except diagnostics (fit table, flag
    log, per-star PNGs) to *out_dir*.

    Differences from the file-based flow, both because the products feed
    a photometry run instead of a catalog release:

    * ``zero_segments=False`` by default — flagged wing segments keep
      their labels so each catalog row still gets a template, and the
      shared ``FLAG_SATURATED_TMPL`` group id groups them into one scene
      per star. The undetected core is still filled with the group id.
    * the repaired ``wht`` is returned (cores carry the median donut
      weight) so downstream inverse-variance use sees valid pixels.

    Parameters
    ----------
    sci, wht
        Science and weight arrays; not modified in place.
    dpsf, wcs, psf_pattern
        PSF model, image WCS, and the ePSF key pattern for the fit.
    catalog, segmap
        Optional catalog + segmentation map to flag. Both or neither.
    out_dir
        Directory for diagnostics. ``None`` disables all file output.
    fwhm_pix
        PSF FWHM in pixels; measured from a drizzled stamp when None.
    flux_frac, min_snr
        Flag thresholds, see
        :func:`mophongo.catalog.flag_saturated_segments`.
    stamp_npix
        Flag-model stamp size in pixels. Default: the native ePSF field
        of view (the model reach — segments beyond it cannot flag).
    plots
        Write per-star repair PNGs and the flag diagnostic to *out_dir*.
    \*\*repair_kwargs
        Forwarded to :func:`mophongo.saturate.repair_saturated_holes`
        (e.g. ``min_buffer_snr``).

    Returns
    -------
    dict
        ``{"sci", "wht", "fits", ...}`` from the image repair, plus
        ``catalog``, ``segmap``, ``flag_log`` when a catalog was given
        (unchanged inputs when nothing was flagged).
    """
    from .catalog import flag_saturated_segments as _flag_segments
    from .saturate import _native_psf_drz_size

    if (catalog is None) != (segmap is None):
        raise ValueError("catalog and segmap must be given together")
    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if fwhm_pix is None:
        fwhm_pix = psf_fwhm_pix(dpsf, psf_pattern)
        logger.info("[repair] measured PSF FWHM: %.2f pix", fwhm_pix)

    res = repair_saturated_holes(
        np.asarray(sci), np.asarray(wht),
        dpsf=dpsf, wcs=wcs, mode="repair",
        fwhm_pix=float(fwhm_pix), psf_filter=psf_pattern,
        output_csv=(out_dir / "saturate_repair.csv" if out_dir else None),
        plot_dir=(out_dir / "saturate_png" if out_dir and plots else None),
        **repair_kwargs,
    )
    n_ok = int(np.sum(res["fits"]["ok"])) if len(res["fits"]) else 0
    logger.info("[repair] %d/%d holes fit and repaired (in memory)",
                n_ok, len(res["fits"]))
    if out_dir is not None:
        res["fits"].write(out_dir / "saturate_repair.fits", overwrite=True)

    if catalog is not None and n_ok:
        if stamp_npix is None:
            stamp_npix = _native_psf_drz_size(dpsf)
        stamp = drizzled_psf_stamp(dpsf, psf_pattern, npix=int(stamp_npix))
        new_cat, new_seg, flag_log = _flag_segments(
            catalog, segmap, res["fits"],
            sci=res["sci"], psf_stamp=stamp,
            flux_frac=flux_frac, min_snr=min_snr,
            zero_segments=zero_segments,
        )
        res.update(catalog=new_cat, segmap=new_seg, flag_log=flag_log)
        if out_dir is not None:
            flag_log.write(out_dir / "flag_log.csv", format="csv",
                           overwrite=True)
            n_flagged = (int(np.sum(np.asarray(flag_log["flagged"])))
                         if len(flag_log) else 0)
            if plots and n_flagged:
                from .verification import plot_saturated_flag_diagnostic

                plot_saturated_flag_diagnostic(
                    np.asarray(sci), res["sci"],
                    np.asarray(segmap), new_seg, flag_log,
                    out_path=out_dir / "flag_diagnostic.png",
                )
    elif catalog is not None:
        res.update(catalog=catalog, segmap=segmap, flag_log=None)
    return res


# --------------------------------------------------------------------------
# Catalog / segmap flagging
# --------------------------------------------------------------------------


def flag_catalog(
    catalog_path: str | Path,
    segmap_path: str | Path,
    fit_table: Table,
    *,
    filter_name: str = "TMPL",
    sci: np.ndarray | None = None,
    psf_stamp: np.ndarray | None = None,
    merge: bool = False,
    flux_frac: float = 0.3,
    fwhm_pix: float | None = None,
    out_dir: str | Path | None = None,
    sci_before: np.ndarray | None = None,
    diagnostic: bool = True,
    **catalog_kwargs: Any,
) -> dict[str, Any]:
    r"""Flag (or merge) the oversplit segments of repaired stars.

    Default is the non-destructive **flag** mode for catalogs built
    before the repair (:func:`mophongo.catalog.flag_saturated_segments`):
    every segment whose flux is dominated by the star model
    (``model > flux_frac x observed``) gets ``FLAG_SATURATED_<FILTER>=1``
    on its existing catalog row and is zeroed in the output segmap. No
    rows are dropped or added. Writes ``<catalog>_flagged.fits``,
    ``<segmap>_flagged.fits``, ``<catalog>_flaglog.csv`` and, by
    default, a per-star before/after diagnostic PNG
    (``<catalog>_flag_diagnostic.png``).

    With ``merge=True`` the destructive
    :func:`mophongo.catalog.repair_saturated_catalog` runs instead:
    child rows are dropped and replaced by one flagged parent, and the
    outputs are ``<catalog>_repaired.fits``, ``<segmap>_repaired.fits``
    and ``<catalog>_mergelog.csv``.

    Parameters
    ----------
    catalog_path, segmap_path
        Source catalog (FITS table) and segmentation map (FITS image).
    fit_table
        ``fits`` table from :func:`repair_image` /
        :func:`mophongo.saturate.repair_saturated_holes`.
    filter_name
        Suffix for the ``FLAG_SATURATED_<FILTER>`` column name.
        Default ``"TMPL"`` — the repair runs on the template band,
        which varies between surveys, so downstream code can rely on
        one fixed column name.
    sci, psf_stamp
        Science image on the segmap grid (use the repaired one) and a
        unit-sum PSF stamp. Required in flag mode; in merge mode they
        enable the neighbour-protection flux filter.
    merge
        Merge children into a parent row instead of flag-only.
    flux_frac
        Flag-mode threshold: flag a segment when
        ``model_flux > flux_frac * observed_flux``.
    fwhm_pix
        PSF FWHM in segmap pixels; required for ``merge=True``.
    sci_before
        Pre-repair science image for the diagnostic plot. Defaults to
        *sci*.
    diagnostic
        Write the flag-mode diagnostic PNG (skipped when nothing was
        flagged).
    \*\*catalog_kwargs
        Forwarded to the underlying catalog function (e.g. ``n_fwhm``
        and ``flux_frac_thresh`` in merge mode, ``id_col`` in both).

    Returns
    -------
    dict
        Flag mode: ``{"catalog", "segmap", "flag_log", "catalog_out",
        "segmap_out", "log_out", "diagnostic_out"}``. Merge mode:
        ``{"catalog", "segmap", "merge_log", "catalog_out",
        "segmap_out", "log_out"}``.
    """
    catalog_path = Path(catalog_path)
    segmap_path = Path(segmap_path)
    out_dir = Path(out_dir) if out_dir is not None else catalog_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = Table.read(catalog_path)
    seg = np.asarray(fits.getdata(segmap_path)).astype(np.int32)
    flag_col = f"FLAG_SATURATED_{filter_name.upper()}"

    if merge:
        if fwhm_pix is None:
            raise ValueError("fwhm_pix is required with merge=True")
        new_cat, new_seg, log = repair_saturated_catalog(
            cat, seg, fit_table,
            fwhm_pix=float(fwhm_pix), filter_name=filter_name,
            sci=sci, psf_stamp=psf_stamp,
            **catalog_kwargs,
        )
        logger.info(
            "[repair] catalog %d → %d rows, %d flagged in %s",
            len(cat), len(new_cat), int(np.sum(new_cat[flag_col])), flag_col,
        )
        suffix, log_name, log_key = "_repaired", "mergelog", "merge_log"
    else:
        if sci is None or psf_stamp is None:
            raise ValueError("sci and psf_stamp are required in flag mode")
        new_cat, new_seg, log = flag_saturated_segments(
            cat, seg, fit_table,
            sci=sci, psf_stamp=psf_stamp,
            filter_name=filter_name, flux_frac=flux_frac,
            **catalog_kwargs,
        )
        suffix, log_name, log_key = "_flagged", "flaglog", "flag_log"

    catalog_out = out_dir / f"{catalog_path.stem}{suffix}.fits"
    segmap_out = out_dir / f"{segmap_path.stem}{suffix}.fits"
    log_out = out_dir / f"{catalog_path.stem}_{log_name}.csv"
    new_cat.write(catalog_out, overwrite=True)
    fits.writeto(
        segmap_out, new_seg.astype(np.int32),
        fits.getheader(segmap_path), overwrite=True,
    )
    log.write(log_out, format="csv", overwrite=True)
    logger.info(
        "[repair] wrote %s, %s, %s", catalog_out, segmap_out, log_out,
    )

    out: dict[str, Any] = {
        "catalog": new_cat, "segmap": new_seg, log_key: log,
        "catalog_out": catalog_out, "segmap_out": segmap_out,
        "log_out": log_out,
    }
    if not merge:
        out["diagnostic_out"] = None
        n_flagged = int(np.sum(np.asarray(log["flagged"]))) if len(log) else 0
        if diagnostic and n_flagged:
            from .verification import plot_saturated_flag_diagnostic

            png = out_dir / f"{catalog_path.stem}_flag_diagnostic.png"
            plot_saturated_flag_diagnostic(
                sci_before if sci_before is not None else sci, sci,
                seg, new_seg, log, out_path=png,
            )
            logger.info("[repair] wrote %s", png)
            out["diagnostic_out"] = png
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``mophongo-repair sci.fits wht.fits``."""
    ap = argparse.ArgumentParser(
        prog="mophongo-repair",
        description=(
            "Repair saturated stars in a drizzled mosaic: fill wht=0 "
            "cores with a fitted PSF model and optionally flag the "
            "affected sources in a catalog (FLAG_SATURATED_TMPL)."
        ),
    )
    ap.add_argument("sci", help="drizzled science FITS image")
    ap.add_argument("wht", help="matching weight (inverse-variance) FITS image")
    ap.add_argument("--filter", dest="filter_name", default=None,
                    help="filter name, e.g. F444W (default: from filename)")
    ap.add_argument("--csv", default=None,
                    help="grizli _wcs.csv exposure listing "
                         "(default: derived from sci; reconstructed if missing)")
    ap.add_argument("--psf-dir", default=None,
                    help="directory with STDPSF grids (default: <sci dir>/PSF)")
    ap.add_argument("--psf-pattern", default=None,
                    help="regex for STDPSF filenames/keys "
                         "(default: NRC.._<FILTER> for NIRCam)")
    ap.add_argument("--no-build-psf", action="store_true",
                    help="fail instead of building missing STDPSFs with stpsf")
    ap.add_argument("--catalog", default=None,
                    help="source catalog FITS to flag (needs --segmap)")
    ap.add_argument("--segmap", default=None,
                    help="segmentation map FITS matching --catalog")
    ap.add_argument("--flux-frac", type=float, default=0.3,
                    help="flag a segment when the star model exceeds this "
                         "fraction of its observed flux (default: 0.3)")
    ap.add_argument("--merge", action="store_true",
                    help="merge flagged segments into one parent catalog row "
                         "(drops the child rows) instead of flag-only")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: next to sci)")
    ap.add_argument("--mode", choices=("repair", "subtract"), default="repair",
                    help="fill saturated cores (repair) or remove the full "
                         "PSF halo (subtract)")
    ap.add_argument("--fwhm-pix", type=float, default=None,
                    help="PSF FWHM in mosaic pixels (default: measured)")
    ap.add_argument("--min-buffer-snr", type=float, default=200.0,
                    help="saturation pre-filter threshold in sigma; lower "
                         "for bands with faint halos (default: 200)")
    ap.add_argument("--merge-radius", type=int, default=3,
                    help="merge wht=0 fragments within this radius (default: 3)")
    ap.add_argument("--n-fwhm", type=float, default=5.0,
                    help="catalog merge radius in FWHM (default: 5)")
    ap.add_argument("--plots", action="store_true",
                    help="write per-source diagnostic PNGs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if bool(args.catalog) != bool(args.segmap):
        ap.error("--catalog and --segmap must be given together")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    res = repair_image(
        args.sci, args.wht,
        filter_name=args.filter_name,
        csv=args.csv, psf_dir=args.psf_dir, psf_pattern=args.psf_pattern,
        build_psf=not args.no_build_psf,
        out_dir=args.out_dir, mode=args.mode,
        fwhm_pix=args.fwhm_pix, plots=args.plots,
        min_buffer_snr=args.min_buffer_snr,
        merge_radius=args.merge_radius,
    )

    if args.catalog:
        sci_before, _ = _read_image(Path(args.sci))
        merge_kwargs = {"n_fwhm": args.n_fwhm} if args.merge else {}
        flag_catalog(
            args.catalog, args.segmap, res["fits"],
            filter_name="TMPL",
            sci=res["sci"],
            psf_stamp=drizzled_psf_stamp(res["dpsf"], res["psf_pattern"], npix=401),
            merge=args.merge,
            flux_frac=args.flux_frac,
            fwhm_pix=res["fwhm_pix"],
            out_dir=args.out_dir,
            sci_before=sci_before,
            **merge_kwargs,
        )


if __name__ == "__main__":
    main()
