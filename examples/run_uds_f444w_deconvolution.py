#!/usr/bin/env python
"""Sharpen a modest MINERVA UDS F444W patch toward a 0.1 arcsec Gaussian.

This is a real-data experiment over Mophongo's checked-out APIs, not a second
deconvolution implementation.  It loads the production F444W PSFRegionMap,
uses ``gaussian_psf_map`` to define phase-matched theoretical targets, uses
``matching_kernel_map`` to derive the existing Wiener kernels, and applies
them with ``PSFRegionMap.convolve_image``.

The output is explicitly described as *regularized toward* the requested
target.  A regularized response is not the target itself; the summary and
figures report its measured width, negative ringing, and source-masked field
scatter.

This site-local driver requires a MINERVA config whose ``sci_hi``, ``wht_hi``,
and ``segmap`` point to aligned UDS mosaics plus the production F444W PSF map.
Run from the repository root, passing those paths when the ignored local
``examples/minerva`` products are not present::

    poetry run python examples/run_uds_f444w_deconvolution.py \
        --config /path/to/uds_f444w.json \
        --psf-map /path/to/uds_f444w_psf_hi.geojson
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import mad_std
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

from mophongo.pipeline import RunConfig
from mophongo.psf import psf_core_centroid, psf_core_fwhm
from mophongo.psf_map import PSFRegionMap
from mophongo.utils import fftconvolve, pad_to_shape


LOGGER = logging.getLogger(__name__)

DEFAULT_CENTER = (34.341327420893705, -5.2615397204619905)
DEFAULT_MINERVA_DIR = Path(__file__).with_name("minerva")
DEFAULT_PSF_MAP = DEFAULT_MINERVA_DIR / "uds_f770w" / "uds_f770w_psf_hi.geojson"
DEFAULT_CONFIG = DEFAULT_MINERVA_DIR / "uds_f770w.json"
DEFAULT_SCAN = np.array(
    [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    dtype=float,
)


def _image_hdu(hdul: fits.HDUList) -> int:
    """Return the first two-dimensional image HDU index."""
    for index, hdu in enumerate(hdul):
        if int(hdu.header.get("NAXIS", 0)) == 2:
            return index
    raise ValueError("FITS file contains no two-dimensional image HDU")


def _read_patch_geometry(
    path: str | Path,
    center: tuple[float, float],
    size: int,
    halo: int,
) -> tuple[np.ndarray, WCS, WCS, fits.Header, tuple[int, int, int, int]]:
    """Read a centered patch plus convolution halo with FITS section I/O."""
    with fits.open(path, memmap=True) as hdul:
        index = _image_hdu(hdul)
        hdu = hdul[index]
        header = hdu.header.copy()
        full_wcs = WCS(header)
        ny = int(header["NAXIS2"])
        nx = int(header["NAXIS1"])
        xcenter, ycenter = full_wcs.all_world2pix(center[0], center[1], 0)
        outer = int(size + 2 * halo)
        x0 = int(round(float(xcenter) - (outer - 1) / 2.0))
        y0 = int(round(float(ycenter) - (outer - 1) / 2.0))
        x1 = x0 + outer
        y1 = y0 + outer
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            raise ValueError(
                f"requested patch {(x0, x1, y0, y1)} falls outside {(nx, ny)}"
            )
        data = np.asarray(hdu.section[y0:y1, x0:x1], dtype=np.float32)

    outer_wcs = full_wcs.slice((slice(y0, y1), slice(x0, x1)))
    inner_wcs = outer_wcs.slice(
        (slice(halo, halo + size), slice(halo, halo + size))
    )
    return data, outer_wcs, inner_wcs, header, (y0, y1, x0, x1)


def _read_matching_section(
    path: str | Path,
    box: tuple[int, int, int, int],
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> tuple[np.ndarray, fits.Header, WCS]:
    """Read a known pixel box from a second FITS image."""
    y0, y1, x0, x1 = box
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[_image_hdu(hdul)]
        return (
            np.asarray(hdu.section[y0:y1, x0:x1], dtype=dtype),
            hdu.header.copy(),
            WCS(hdu.header),
        )


def _assert_aligned_wcs(
    reference: WCS,
    candidate: WCS,
    box: tuple[int, int, int, int],
    *,
    label: str,
    tolerance_pix: float = 1e-3,
) -> None:
    """Require two parent-image WCS grids to agree over the extracted box."""
    y0, y1, x0, x1 = box
    x = np.asarray([x0, x1 - 1, x0, x1 - 1, 0.5 * (x0 + x1 - 1)], dtype=float)
    y = np.asarray([y0, y0, y1 - 1, y1 - 1, 0.5 * (y0 + y1 - 1)], dtype=float)
    ra, dec = reference.all_pix2world(x, y, 0)
    x_check, y_check = candidate.all_world2pix(ra, dec, 0)
    separation = np.hypot(x_check - x, y_check - y)
    if not np.all(np.isfinite(separation)) or float(np.max(separation)) > tolerance_pix:
        raise ValueError(
            f"{label} WCS is not aligned with science grid: "
            f"maximum offset {float(np.nanmax(separation)):.6g} pixels"
        )


def _clip_psf_map(
    source: PSFRegionMap, wcs: WCS, *, name: str
) -> PSFRegionMap:
    """Clip a PSF map to a WCS footprint and carry its parent PSF planes."""
    footprint = Polygon(np.asarray(wcs.calc_footprint(), dtype=float))
    clipped = source.overlay_with(footprint)
    if source.regions.crs is not None:
        clipped.regions = clipped.regions.set_crs(source.regions.crs, allow_override=True)
    parents = np.asarray(clipped.regions["psf_key_1"], dtype=int)
    clipped.psfs = np.asarray(source.psfs[parents], dtype=np.float32)
    clipped.pscale = source.pscale
    clipped.name = name
    return clipped


def _region_key_image(prm: PSFRegionMap, wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize region keys on a small diagnostic patch."""
    keys = np.full(shape, -1, dtype=np.int32)
    ny, nx = shape
    for geom, key in zip(prm.regions.geometry, prm.regions["psf_key"]):
        polygon = shapely_transform(
            lambda x, y, z=None: tuple(
                wcs.all_world2pix(np.asarray(x), np.asarray(y), 0)
            ),
            geom,
        )
        bx0, by0, bx1, by1 = polygon.bounds
        x0 = max(int(np.floor(bx0)), 0)
        x1 = min(int(np.ceil(bx1)) + 1, nx)
        y0 = max(int(np.floor(by0)), 0)
        y1 = min(int(np.ceil(by1)) + 1, ny)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = contains_xy(polygon, xx.astype(float), yy.astype(float))
        keys[y0:y1, x0:x1][inside] = int(key)
    return keys


def _seam_mask(keys: np.ndarray, width: int = 3) -> np.ndarray:
    """Return a dilated mask around boundaries between valid region keys."""
    seam = np.zeros(keys.shape, dtype=bool)
    dx = (keys[:, 1:] != keys[:, :-1]) & (keys[:, 1:] >= 0) & (keys[:, :-1] >= 0)
    dy = (keys[1:, :] != keys[:-1, :]) & (keys[1:, :] >= 0) & (keys[:-1, :] >= 0)
    seam[:, 1:] |= dx
    seam[:, :-1] |= dx
    seam[1:, :] |= dy
    seam[:-1, :] |= dy
    return binary_dilation(seam, iterations=int(width))


def _field_rms_mask(
    image: np.ndarray,
    weight: np.ndarray,
    segmap: np.ndarray,
    *,
    dilation: int = 16,
) -> tuple[np.ndarray, float]:
    """Return a fixed source-masked field mask and its robust scatter.

    The release segmentation map avoids seeding a large dilation from random
    positive noise peaks.  This crowded patch still has no large area a full
    inverse-kernel radius from every source, so the statistic includes
    correlated background and residual ringing and is not an instrumental-
    noise estimate.
    """
    if image.shape != weight.shape or image.shape != segmap.shape:
        raise ValueError("science, weight, and segmentation shapes must match")
    finite = np.isfinite(image) & np.isfinite(weight) & (weight > 0.0)
    source = np.asarray(segmap) > 0
    source = binary_dilation(source, iterations=int(dilation))
    field = finite & ~source
    border = max(1, int(dilation))
    field[:border, :] = False
    field[-border:, :] = False
    field[:, :border] = False
    field[:, -border:] = False
    if int(field.sum()) < 1000:
        raise ValueError(
            f"source-masked field has only {int(field.sum())} pixels; "
            "choose a less crowded patch or reduce the mask dilation"
        )
    rms = float(mad_std(image[field], ignore_nan=True))
    return field, rms


def _empty_aperture_centers(
    segmap: np.ndarray,
    weight: np.ndarray,
    *,
    min_source_distance: int = 16,
    grid_step: int = 16,
    border: int = 17,
) -> np.ndarray:
    """Select deterministic empty-aperture centers from native inputs."""
    if segmap.shape != weight.shape:
        raise ValueError("segmentation and weight shapes must match")
    distance = distance_transform_edt(np.asarray(segmap) <= 0)
    yy, xx = np.mgrid[border : segmap.shape[0] - border : grid_step,
                      border : segmap.shape[1] - border : grid_step]
    valid = (
        (distance[yy, xx] >= float(min_source_distance))
        & np.isfinite(weight[yy, xx])
        & (weight[yy, xx] > 0.0)
    )
    centers = np.column_stack((yy[valid], xx[valid])).astype(int)
    if len(centers) < 100:
        raise ValueError(
            f"only {len(centers)} empty-aperture centers; choose a less crowded patch"
        )
    return centers


def _empty_aperture_sums(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    radius: int = 3,
    annulus: tuple[int, int] = (6, 10),
) -> np.ndarray:
    """Measure local-background-subtracted sums at fixed empty positions."""
    outer = int(annulus[1])
    dy, dx = np.mgrid[-outer : outer + 1, -outer : outer + 1]
    rr = np.hypot(dx, dy)
    aperture = rr <= float(radius)
    background = (rr >= float(annulus[0])) & (rr <= float(annulus[1]))
    sums = np.empty(len(centers), dtype=float)
    for index, (yc, xc) in enumerate(np.asarray(centers, dtype=int)):
        stamp = np.asarray(
            image[yc - outer : yc + outer + 1, xc - outer : xc + outer + 1],
            dtype=float,
        )
        if stamp.shape != rr.shape or not np.all(np.isfinite(stamp)):
            sums[index] = np.nan
            continue
        local_background = float(np.median(stamp[background]))
        sums[index] = float(np.sum(stamp[aperture] - local_background))
    return sums[np.isfinite(sums)]


def _aperture_flux(
    image: np.ndarray,
    wcs: WCS,
    position: tuple[float, float],
    pscale: float,
    radius_arcsec: float = 2.5,
) -> float:
    """Background-subtracted circular-aperture sum for the bright test star."""
    xc, yc = wcs.all_world2pix(position[0], position[1], 0)
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - float(xc), yy - float(yc)) * float(pscale)
    aperture = rr <= radius_arcsec
    annulus = (rr >= radius_arcsec + 0.3) & (rr <= radius_arcsec + 0.8)
    background = float(np.nanmedian(image[annulus]))
    return float(np.nansum(image[aperture] - background))


def _fits_header(
    wcs: WCS,
    source_header: fits.Header,
    *,
    target_fwhm: float | None = None,
    reg: float | None = None,
    kernel_size: int | None = None,
    source_map: str | None = None,
    metrics: dict[str, float] | None = None,
) -> fits.Header:
    """Return a compact WCS and deconvolution provenance header."""
    header = wcs.to_header(relax=True)
    for keyword in ("BUNIT", "FILTER", "TELESCOP", "INSTRUME"):
        if keyword in source_header:
            header[keyword] = source_header[keyword]
    if target_fwhm is not None:
        header["DECONV"] = (True, "regularized PSF deconvolution")
        header["TARGFWH"] = (float(target_fwhm), "requested Gaussian FWHM [arcsec]")
        header["KERNMETH"] = ("wiener", "Mophongo matching-kernel method")
        header["WREG"] = (float(reg), "dimensionless Wiener regularization")
        header["KERNSIZE"] = (int(kernel_size), "kernel side [pixels]")
        if source_map is not None:
            header["PSFMAP"] = (Path(source_map).name[:68], "source PSF region map")
        for keyword, value, comment in (
            ("WRMSGAIN", metrics.get("white_noise_gain"), "median white-noise RMS gain"),
            ("FLDRMSRT", metrics.get("field_rms_ratio"), "source-masked field RMS ratio"),
            ("FLDRMSLO", metrics.get("field_rms_ratio_min"), "mask-dilation RMS ratio minimum"),
            ("FLDRMSHI", metrics.get("field_rms_ratio_max"), "mask-dilation RMS ratio maximum"),
            ("APRMSRAT", metrics.get("empty_aperture_rms_ratio"), "empty-aperture RMS ratio"),
            ("RESPFWH", metrics.get("response_fwhm"), "median realized core FWHM [arcsec]"),
            ("RESPNEG", metrics.get("response_negative_flux"), "median negative response flux"),
        ):
            if value is not None and np.isfinite(value):
                header[keyword] = (float(value), comment)
        header["SUPLIM"] = (
            bool(metrics.get("support_limited", False)),
            "kernel edge-support diagnostic failed",
        )
        header["EDGEL1"] = (
            float(metrics.get("kernel_edge_l1_max", np.nan)),
            "maximum outer-edge absolute kernel L1",
        )
        header["EDGEL1F"] = (
            float(metrics.get("kernel_edge_l1_fraction_max", np.nan)),
            "maximum fractional outer-edge kernel L1",
        )
        header["SEGDIL"] = (
            int(metrics.get("field_dilation_pix", -1)),
            "segmentation dilation for field RMS",
        )
        header["FLDMASK"] = (
            float(metrics.get("field_mask_fraction", np.nan)),
            "unmasked fraction for field RMS",
        )
        header["NAPCTRL"] = (
            int(metrics.get("n_empty_apertures", 0)),
            "empty control apertures",
        )
    return header


def _line_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a core-centered horizontal profile of a PSF response."""
    xc, yc = psf_core_centroid(image)
    x = np.arange(image.shape[1], dtype=float)
    profile = map_coordinates(
        np.asarray(image, dtype=float),
        [np.full_like(x, yc), x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return x - xc, profile


def _comparison_figure(
    input_image: np.ndarray,
    outputs: dict[float, np.ndarray],
    summaries: dict[float, dict[str, float]],
    wcs: WCS,
    center: tuple[float, float],
    field_rms: float,
    path: Path,
) -> None:
    """Write full-patch and bright-star before/after panels."""
    images = [("native F444W", input_image)] + [
        (f"Wiener $\\lambda={reg:g}$", outputs[reg]) for reg in outputs
    ]
    ncol = len(images)
    fig, axes = plt.subplots(2, ncol, figsize=(5.1 * ncol, 9.0), squeeze=False)
    finite = np.isfinite(input_image)
    vmin, vmax = np.nanpercentile(input_image[finite], [0.5, 99.8])
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.03))
    xc, yc = wcs.all_world2pix(center[0], center[1], 0)
    xc = float(np.asarray(xc))
    yc = float(np.asarray(yc))
    half = 90
    y0, y1 = int(round(yc)) - half, int(round(yc)) + half
    x0, x1 = int(round(xc)) - half, int(round(xc)) + half

    star_arrays = [arr[y0:y1, x0:x1] for _, arr in images]
    star_limit = float(
        np.nanpercentile(np.concatenate([a[np.isfinite(a)] for a in star_arrays]), 99.9)
    )
    star_norm = ImageNormalize(
        vmin=-8.0 * field_rms,
        vmax=star_limit,
        stretch=AsinhStretch(0.02),
    )

    for column, (label, image) in enumerate(images):
        axes[0, column].imshow(image, origin="lower", cmap="gray", norm=norm)
        axes[0, column].set_title(label)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[1, column].imshow(
            image[y0:y1, x0:x1], origin="lower", cmap="gray", norm=star_norm
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
        if column == 0:
            axes[1, column].set_title('bright star: native')
        else:
            reg = list(outputs)[column - 1]
            summary = summaries[reg]
            axes[1, column].set_title(
                f"FWHM={summary['response_fwhm']:.3f}\"; "
                f"masked field RMS $\\times${summary['field_rms_ratio']:.2f}\n"
                f"negative response={summary['response_negative_flux']:.2f}"
            )
    axes[0, 0].set_ylabel("40.96 arcsec patch")
    axes[1, 0].set_ylabel("7.2 arcsec star cutout")
    fig.suptitle('MINERVA UDS F444W: regularized toward a 0.1" Gaussian', y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _tradeoff_figure(
    table: pd.DataFrame,
    responses: dict[float, np.ndarray],
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    pscale: float,
    target_fwhm: float,
    path: Path,
) -> None:
    """Write resolution/noise/ringing and realized-response diagnostics."""
    ordered = table.sort_values("reg")
    supported = ~ordered["support_limited"].astype(bool)
    stable = ordered.loc[supported]
    limited = ordered.loc[~supported]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))

    axes[0, 0].plot(
        stable["response_fwhm_arcsec"], stable["field_rms_ratio"], "o-"
    )
    axes[0, 0].plot(
        limited["response_fwhm_arcsec"],
        limited["field_rms_ratio"],
        "x",
        color="0.45",
        ms=8,
        mew=2,
        label="support-limited",
    )
    for row in ordered.itertuples():
        axes[0, 0].annotate(
            f"{row.reg:g}", (row.response_fwhm_arcsec, row.field_rms_ratio),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    axes[0, 0].axvline(target_fwhm, color="k", ls="--", label="requested target")
    axes[0, 0].set_xlabel("realized central-line FWHM [arcsec]")
    axes[0, 0].set_ylabel("source-masked field RMS / native")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(
        stable["response_fwhm_arcsec"], stable["response_negative_flux"], "o-"
    )
    axes[0, 1].plot(
        limited["response_fwhm_arcsec"],
        limited["response_negative_flux"],
        "x",
        color="0.45",
        ms=8,
        mew=2,
    )
    axes[0, 1].axvline(target_fwhm, color="k", ls="--")
    axes[0, 1].set_xlabel("realized central-line FWHM [arcsec]")
    axes[0, 1].set_ylabel("integrated negative response flux")
    axes[0, 1].grid(alpha=0.25)

    x, profile = _line_profile(source_psf)
    axes[1, 0].plot(x * pscale, profile / np.max(profile), label="native F444W")
    x, profile = _line_profile(target_psf)
    axes[1, 0].plot(x * pscale, profile / np.max(profile), "k--", label='0.1" target')
    for reg, response in responses.items():
        x, profile = _line_profile(response)
        axes[1, 0].plot(
            x * pscale, profile / np.max(profile), label=f"$\\lambda={reg:g}$"
        )
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].set_xlim(-0.8, 0.8)
    axes[1, 0].set_ylim(-0.12, 1.05)
    axes[1, 0].set_xlabel("offset [arcsec]")
    axes[1, 0].set_ylabel("central-line response / peak")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].loglog(
        stable["reg"], stable["white_noise_gain"], "o-", label="white-noise prediction"
    )
    axes[1, 1].loglog(
        stable["reg"], stable["field_rms_ratio"], "s-", label="masked field"
    )
    axes[1, 1].loglog(
        stable["reg"],
        stable["empty_aperture_rms_ratio"],
        "^-",
        label="empty apertures",
    )
    axes[1, 1].loglog(
        limited["reg"],
        limited["white_noise_gain"],
        "x",
        color="0.45",
        ms=8,
        mew=2,
        label="support-limited",
    )
    axes[1, 1].loglog(
        limited["reg"],
        limited["field_rms_ratio"],
        "x",
        color="0.45",
        ms=8,
        mew=2,
    )
    axes[1, 1].loglog(
        limited["reg"],
        limited["empty_aperture_rms_ratio"],
        "x",
        color="0.45",
        ms=8,
        mew=2,
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_xlabel("Wiener regularization $\\lambda$")
    axes[1, 1].set_ylabel("RMS amplification")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend()

    fig.suptitle("A narrower requested PSF trades resolution for noise and ringing")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    """Execute the UDS patch experiment and return its output directory."""
    config = RunConfig.from_json(args.config)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pscale = float(args.pixel_scale)
    halo = int(args.kernel_size // 2 + 2)
    outer_sci, outer_wcs, inner_wcs, source_header, box = _read_patch_geometry(
        config.sci_hi, tuple(args.center), int(args.size), halo
    )
    outer_wht, weight_header, weight_wcs = _read_matching_section(config.wht_hi, box)
    outer_seg, _, segmap_wcs = _read_matching_section(
        config.segmap, box, dtype=np.dtype(np.int64)
    )
    _assert_aligned_wcs(WCS(source_header), weight_wcs, box, label="weight")
    _assert_aligned_wcs(WCS(source_header), segmap_wcs, box, label="segmentation")
    if outer_sci.shape != outer_wht.shape or outer_sci.shape != outer_seg.shape:
        raise ValueError(
            "science/weight/segmentation patch shapes differ: "
            f"{outer_sci.shape}, {outer_wht.shape}, {outer_seg.shape}"
        )
    invalid = ~np.isfinite(outer_sci) | ~np.isfinite(outer_wht) | (outer_wht <= 0.0)
    if np.any(invalid):
        raise ValueError(
            f"test patch contains {int(invalid.sum())} invalid/zero-weight pixels; "
            "choose a fully covered patch or explicitly inpaint before deconvolution"
        )

    inner = (slice(halo, halo + args.size), slice(halo, halo + args.size))
    input_image = np.asarray(outer_sci[inner], dtype=np.float32)
    input_weight = np.asarray(outer_wht[inner], dtype=np.float32)
    input_segmap = np.asarray(outer_seg[inner], dtype=np.int64)
    sensitivity_dilations = sorted({12, int(args.seg_dilate), 24})
    field_stats = {
        dilation: _field_rms_mask(
            input_image, input_weight, input_segmap, dilation=dilation
        )
        for dilation in sensitivity_dilations
    }
    field_mask, input_rms = field_stats[int(args.seg_dilate)]
    aperture_centers = _empty_aperture_centers(
        input_segmap,
        input_weight,
        min_source_distance=int(args.aperture_source_distance),
    )
    input_aperture_sums = _empty_aperture_sums(input_image, aperture_centers)
    input_aperture_rms = float(mad_std(input_aperture_sums, ignore_nan=True))

    source_map = PSFRegionMap.from_geojson(args.psf_map, pscale=pscale)
    clipped = _clip_psf_map(source_map, outer_wcs, name="uds_f444w_patch_psf")
    target_map = clipped.gaussian_psf_map(
        float(args.target_fwhm) / pscale,
        shape=int(args.kernel_size),
        phase_match=True,
        name="uds_f444w_gaussian_target",
    )
    target_path = out_dir / "uds_f444w_gaussian_target.geojson"
    target_map.to_file(target_path)

    key_image = _region_key_image(clipped, outer_wcs, outer_sci.shape)[inner]
    output_keys = np.unique(key_image[key_image >= 0]).astype(int)
    if output_keys.size == 0:
        raise ValueError("inner test patch is outside the clipped PSF map")
    seam = _seam_mask(key_image)
    scan_regs = np.unique(np.concatenate([DEFAULT_SCAN, np.asarray(args.reg, dtype=float)]))
    save_regs = set(float(value) for value in args.reg)
    output_images: dict[float, np.ndarray] = {}
    output_summaries: dict[float, dict[str, float]] = {}
    response_examples: dict[float, np.ndarray] = {}
    rows: list[dict[str, float]] = []

    center_key = clipped.lookup_key(float(args.center[0]), float(args.center[1]))
    if center_key is None:
        raise ValueError("test center is outside the clipped PSF map")
    source_shape = np.asarray(clipped.psfs[int(center_key)], dtype=float)
    source_shape /= source_shape.sum()
    source_padded = pad_to_shape(source_shape, target_map.psfs.shape[-2:])
    target_example = np.asarray(target_map.psfs[int(center_key)], dtype=float)
    input_flux = _aperture_flux(input_image, inner_wcs, tuple(args.center), pscale)

    for reg in scan_regs:
        kernel_map = clipped.matching_kernel_map(
            target_map,
            method="wiener",
            reg=float(reg),
            name=f"uds_f444w_wiener_reg{reg:.0e}",
        )
        filtered_outer = kernel_map.convolve_image(
            outer_sci, outer_wcs, fill_value=np.nan
        )
        filtered = np.asarray(filtered_outer[inner], dtype=np.float32)
        finite = np.isfinite(filtered)
        field_valid = field_mask & finite
        output_rms = float(mad_std(filtered[field_valid], ignore_nan=True))
        field_ratios = []
        for mask, native_rms in field_stats.values():
            valid = mask & finite
            field_ratios.append(
                float(mad_std(filtered[valid], ignore_nan=True)) / native_rms
            )
        output_aperture_sums = _empty_aperture_sums(filtered, aperture_centers)
        output_aperture_rms = float(
            mad_std(output_aperture_sums, ignore_nan=True)
        )
        seam_field = field_valid & seam
        nonseam_field = field_valid & ~seam
        seam_ratio = float("nan")
        if seam_field.sum() > 100 and nonseam_field.sum() > 100:
            seam_ratio = float(
                mad_std(filtered[seam_field], ignore_nan=True)
                / mad_std(filtered[nonseam_field], ignore_nan=True)
            )
        output_flux = _aperture_flux(filtered, inner_wcs, tuple(args.center), pscale)
        region_table = kernel_map.regions
        output_region_table = region_table.loc[
            region_table["psf_key"].isin(output_keys)
        ]
        if len(output_region_table) != len(output_keys):
            raise ValueError("one or more inner-patch PSF keys are missing diagnostics")
        response_fwhm_pix = 0.5 * (
            np.asarray(output_region_table["response_fwhm_x_pix"], dtype=float)
            + np.asarray(output_region_table["response_fwhm_y_pix"], dtype=float)
        )
        summary = {
            "reg": float(reg),
            "n_output_regions": int(len(output_region_table)),
            "n_kernel_regions": int(len(region_table)),
            "response_fwhm_arcsec": float(np.nanmedian(response_fwhm_pix) * pscale),
            "white_noise_gain": float(
                np.nanmedian(
                    np.asarray(output_region_table["kernel_noise_gain"], dtype=float)
                )
            ),
            "field_rms_ratio": output_rms / input_rms,
            "field_rms_ratio_min": float(np.min(field_ratios)),
            "field_rms_ratio_max": float(np.max(field_ratios)),
            "field_dilation_pix": int(args.seg_dilate),
            "field_mask_fraction": float(np.mean(field_mask)),
            "empty_aperture_rms_ratio": output_aperture_rms / input_aperture_rms,
            "n_empty_apertures": int(len(input_aperture_sums)),
            "response_negative_flux": float(
                np.nanmedian(
                    np.asarray(output_region_table["response_negative_flux"], dtype=float)
                )
            ),
            "kernel_negative_flux": float(
                np.nanmedian(
                    np.asarray(output_region_table["kernel_negative_flux"], dtype=float)
                )
            ),
            "kernel_l1": float(
                np.nanmedian(np.asarray(output_region_table["kernel_l1"], dtype=float))
            ),
            "kernel_edge_l1_fraction_max": float(
                np.nanmax(
                    np.asarray(
                        output_region_table["kernel_edge_l1_fraction"], dtype=float
                    )
                )
            ),
            "kernel_edge_l1_max": float(
                np.nanmax(
                    np.asarray(output_region_table["kernel_edge_l1"], dtype=float)
                )
            ),
            "response_target_peak": float(
                np.nanmedian(
                    np.asarray(output_region_table["response_target_peak"], dtype=float)
                )
            ),
            "response_shift_max_pix": float(
                np.nanmax(
                    np.hypot(
                        np.asarray(
                            output_region_table["response_shift_x_pix"], dtype=float
                        ),
                        np.asarray(
                            output_region_table["response_shift_y_pix"], dtype=float
                        ),
                    )
                )
            ),
            "aperture_flux_ratio": output_flux / input_flux,
            "coverage_fraction": float(np.mean(finite)),
            "support_limited": bool(
                (
                    np.nanmax(
                        np.asarray(
                            output_region_table["kernel_edge_l1_fraction"], dtype=float
                        )
                    )
                    > 1e-3
                )
                or (
                    np.nanmax(
                        np.asarray(output_region_table["kernel_edge_l1"], dtype=float)
                    )
                    > 1e-2
                )
            ),
            "seam_field_rms_ratio": seam_ratio,
        }
        rows.append(summary)
        response_examples[float(reg)] = fftconvolve(
            source_padded, kernel_map.psfs[int(center_key)], mode="same"
        )

        if float(reg) in save_regs:
            output_images[float(reg)] = filtered
            output_summaries[float(reg)] = {
                "response_fwhm": summary["response_fwhm_arcsec"],
                "field_rms_ratio": summary["field_rms_ratio"],
                "response_negative_flux": summary["response_negative_flux"],
            }
            stem = f"uds_f444w_wiener_reg{reg:.0e}"
            kernel_map.to_file(out_dir / f"{stem}_kernel.geojson")
            header = _fits_header(
                inner_wcs,
                source_header,
                target_fwhm=float(args.target_fwhm),
                reg=float(reg),
                kernel_size=int(args.kernel_size),
                source_map=str(args.psf_map),
                metrics={
                    "white_noise_gain": summary["white_noise_gain"],
                    "field_rms_ratio": summary["field_rms_ratio"],
                    "field_rms_ratio_min": summary["field_rms_ratio_min"],
                    "field_rms_ratio_max": summary["field_rms_ratio_max"],
                    "field_dilation_pix": summary["field_dilation_pix"],
                    "field_mask_fraction": summary["field_mask_fraction"],
                    "empty_aperture_rms_ratio": summary[
                        "empty_aperture_rms_ratio"
                    ],
                    "n_empty_apertures": summary["n_empty_apertures"],
                    "response_fwhm": summary["response_fwhm_arcsec"],
                    "response_negative_flux": summary["response_negative_flux"],
                    "support_limited": summary["support_limited"],
                    "kernel_edge_l1_max": summary["kernel_edge_l1_max"],
                    "kernel_edge_l1_fraction_max": summary[
                        "kernel_edge_l1_fraction_max"
                    ],
                },
            )
            fits.writeto(out_dir / f"{stem}.fits", filtered, header, overwrite=True)

    summary_table = pd.DataFrame(rows).sort_values("reg")
    summary_table.to_csv(out_dir / "uds_f444w_deconvolution_summary.csv", index=False)
    input_header = _fits_header(inner_wcs, source_header)
    input_header["ORIGY0"] = (int(box[0] + halo), "row origin in full mosaic")
    input_header["ORIGX0"] = (int(box[2] + halo), "column origin in full mosaic")
    fits.writeto(
        out_dir / "uds_f444w_input.fits", input_image, input_header, overwrite=True
    )
    input_wht_header = _fits_header(inner_wcs, weight_header)
    input_wht_header.pop("BUNIT", None)
    input_wht_header["WHTROLE"] = (
        "INPUT",
        "inverse variance for native input only",
    )
    input_wht_header["DECVWHT"] = (
        False,
        "not propagated through signed kernels",
    )
    fits.writeto(
        out_dir / "uds_f444w_input_wht.fits",
        input_weight,
        input_wht_header,
        overwrite=True,
    )

    _comparison_figure(
        input_image,
        output_images,
        output_summaries,
        inner_wcs,
        tuple(args.center),
        input_rms,
        out_dir / "uds_f444w_deconvolution_comparison.png",
    )
    _tradeoff_figure(
        summary_table,
        {reg: response_examples[reg] for reg in output_images},
        source_padded,
        target_example,
        pscale,
        float(args.target_fwhm),
        out_dir / "uds_f444w_deconvolution_tradeoff.png",
    )

    native_fwhm = np.mean(psf_core_fwhm(source_padded)) * pscale
    target_sampled_fwhm = np.mean(psf_core_fwhm(target_example)) * pscale
    LOGGER.info(
        "wrote %s; native/target sampled FWHM %.3f/%.3f arcsec; %d PSF regions",
        out_dir,
        native_fwhm,
        target_sampled_fwhm,
        len(output_keys),
    )
    print(summary_table.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    return out_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--psf-map", type=Path, default=DEFAULT_PSF_MAP)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("scratch/uds_f444w_deconvolution")
    )
    parser.add_argument("--center", type=float, nargs=2, default=DEFAULT_CENTER)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--pixel-scale", type=float, default=0.04)
    parser.add_argument("--target-fwhm", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=512)
    parser.add_argument(
        "--seg-dilate",
        type=int,
        default=16,
        help="release-segmentation dilation used for the primary field RMS",
    )
    parser.add_argument(
        "--aperture-source-distance",
        type=int,
        default=16,
        help="minimum center distance from a segmented source [pixels]",
    )
    parser.add_argument(
        "--reg",
        type=float,
        nargs="+",
        default=[1e-3, 1e-4],
        help="regularizations whose FITS images and kernel maps are retained",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args())
