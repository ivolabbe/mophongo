"""Verification helpers for pipeline simulations and diagnostics.

The routines here are intentionally independent of a particular survey layout.
They encode reusable bookkeeping from the realistic pipeline checks: PSF
shape/throughput handling, standard PSF-kernel diagnostics, source-recovery
tables, flux-ratio diagnostics, and basic WHT sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.stats import mad_std
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS


DEFAULT_WIENER_REG_GRID = np.array(
    [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    dtype=float,
)
DEFAULT_F444W_PATTERN = "UDS_NRC.._F444W_OS4_GRID1"
DEFAULT_F770W_PATTERN = "UDS_MIRI_F770W_OS4_GRID1"
DEFAULT_NIRCAM_LW_DETECTORS = ("NRCA5", "NRCB5")
DEFAULT_MIRI_DETECTOR = ("MIRIM",)


@dataclass(frozen=True)
class PSFShape:
    """Unit-sum PSF shape plus finite-stamp throughput metadata."""

    shape: np.ndarray
    throughput: float


@dataclass(frozen=True)
class WHTNoiseCheck:
    """Summary of ``(sci - truth) * sqrt(wht)`` over valid pixels."""

    filter_name: str
    n_pix: int
    std: float
    mad: float
    median: float
    wht_type: str | None = None
    rnoise: float | None = None


@dataclass(frozen=True)
class WienerPSFMaps:
    """PSF maps, kernel map, and throughput metadata for a Wiener match."""

    source_map: Any
    target_map: Any
    kernel_map: Any
    wiener_lambda: float
    source_throughputs: np.ndarray
    target_throughputs: np.ndarray


@dataclass(frozen=True)
class PipelineScenarioResult:
    """Outputs from one verification pipeline scenario."""

    scenario: str
    pipeline: Any
    table: Table
    source_table: Table
    residuals: list[np.ndarray]
    residual_native: np.ndarray
    model_native: np.ndarray
    output_dir: Path
    summary: dict[str, float | str]


def prepare_psf_shape(psf: np.ndarray, label: str = "PSF") -> PSFShape:
    """Return a unit-sum PSF shape while preserving finite-stamp throughput.

    Do not renormalize the native PSF in-place.  The native finite-stamp sum is
    the throughput correction used for total-flux reporting.
    """

    arr = np.asarray(psf, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(f"{label} PSF has zero or non-finite finite-stamp sum")
    return PSFShape(shape=arr / total, throughput=total)


def filter_average_throughput(values: Sequence[float] | np.ndarray) -> float:
    """Return one filter-level throughput correction from finite PSF sums."""

    arr = np.asarray(values, dtype=float)
    good = np.isfinite(arr) & (arr > 0.0)
    if not np.any(good):
        return 1.0
    return float(np.nanmean(arr[good]))


def parse_regularization_grid(value: str | Sequence[float] | np.ndarray) -> np.ndarray:
    """Parse and validate a positive scalar regularization grid."""

    if isinstance(value, str):
        arr = np.array([float(item) for item in value.split(",") if item.strip()], dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("regularization grid must contain finite positive values")
    return arr


def offset_pointing(
    center: tuple[float, float],
    *,
    dx_arcsec: float,
    dy_arcsec: float,
    pa: float,
):
    """Return a JWST pointing offset from ``center`` by small sky offsets."""

    from mophongo.mock_mosaic import Pointing

    ra0, dec0 = map(float, center)
    dra = float(dx_arcsec) / (3600.0 * np.cos(np.deg2rad(dec0)))
    ddec = float(dy_arcsec) / 3600.0
    return Pointing(ra0 + dra, dec0 + ddec, pa=pa)


def native_phase_dither_pointings(
    center: tuple[float, float],
    *,
    family: str,
    n_dither: int,
    pa: float,
) -> list[Any]:
    """Generate deterministic pointings sampling native detector pixel phase."""

    from mophongo.mock_mosaic import NATIVE_PSCALE, Pointing

    n_dither = int(n_dither)
    if n_dither <= 1:
        return [Pointing(*center, pa=pa)]
    phases = np.array(
        [
            (0.00, 0.00),
            (0.50, 0.00),
            (0.00, 0.50),
            (0.50, 0.50),
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        ],
        dtype=float,
    )
    if n_dither > len(phases):
        raise ValueError("n_dither currently supports at most 8 deterministic phase samples")
    phases = phases[:n_dither]
    phases -= np.mean(phases, axis=0, keepdims=True)
    native_pscale = float(NATIVE_PSCALE[family])
    return [
        offset_pointing(
            center,
            dx_arcsec=float(dx_phase * native_pscale),
            dy_arcsec=float(dy_phase * native_pscale),
            pa=pa,
        )
        for dx_phase, dy_phase in phases
    ]


def nircam_lw_phase_pointings(
    center: tuple[float, float],
    *,
    pa: float = 0.0,
    n_dither: int = 6,
) -> list[Any]:
    """Return small NIRCam LW dithers with deterministic native-pixel phases."""

    return native_phase_dither_pointings(center, family="nircam_lw", n_dither=n_dither, pa=pa)


def miri_center_for_nircam_detector(
    center: tuple[float, float],
    *,
    detector: str,
    pa: float = 0.0,
    miri_detector: Sequence[str] = DEFAULT_MIRI_DETECTOR,
) -> tuple[float, float]:
    """Return a MIRI center whose footprint centroid matches one LW detector."""

    from shapely.ops import unary_union

    from mophongo.mock_mosaic import Pointing, _pointing_footprints

    target = unary_union(
        _pointing_footprints("nircam_lw", [Pointing(*center, pa=pa)], detector_keys=(detector,))
    )
    miri_at_center = unary_union(
        _pointing_footprints("miri", [Pointing(*center, pa=pa)], detector_keys=tuple(miri_detector))
    )
    return (
        float(center[0]) + float(target.centroid.x - miri_at_center.centroid.x),
        float(center[1]) + float(target.centroid.y - miri_at_center.centroid.y),
    )


def miri_two_macro_phase_pointings(
    center: tuple[float, float],
    *,
    pa: float = 0.0,
    nircam_detectors: Sequence[str] = DEFAULT_NIRCAM_LW_DETECTORS,
    miri_detector: Sequence[str] = DEFAULT_MIRI_DETECTOR,
    n_dither: int = 8,
) -> list[Any]:
    """Return two MIRI macro positions, each with deterministic phase dithers."""

    pointings: list[Any] = []
    for detector in nircam_detectors:
        macro_center = miri_center_for_nircam_detector(
            center,
            detector=str(detector),
            pa=pa,
            miri_detector=miri_detector,
        )
        pointings.extend(
            native_phase_dither_pointings(macro_center, family="miri", n_dither=n_dither, pa=pa)
        )
    return pointings


def write_pointing_summary(paths: dict[str, dict[str, Any]], out_dir: str | Path) -> None:
    """Record generated WCS CSV paths and frame counts for quick inspection."""

    out_dir = Path(out_dir)
    rows = []
    for filt, info in paths.items():
        rows.append(
            {
                "filter": filt,
                "wcs_csv": str(info["csv"]),
                "csv_rows": int(info["n_rows"]),
                "pscale_arcsec": float(info["pscale"]),
                "mosaic_fits": str(info["fits"]),
            }
        )
    Table(rows=rows).write(out_dir / "wcs_products.csv", overwrite=True)


def wht_noise_check(
    sci: np.ndarray,
    truth: np.ndarray,
    wht: np.ndarray,
    *,
    filter_name: str = "",
    header: fits.Header | None = None,
) -> WHTNoiseCheck:
    """Check that a WHT image is actual inverse variance per pixel."""

    sci = np.asarray(sci, dtype=float)
    truth = np.asarray(truth, dtype=float)
    wht = np.asarray(wht, dtype=float)
    mask = (wht > 0.0) & np.isfinite(sci) & np.isfinite(truth) & np.isfinite(wht)
    if not np.any(mask):
        raise ValueError("no finite positive-WHT pixels available for noise check")
    z = (sci[mask] - truth[mask]) * np.sqrt(wht[mask])
    return WHTNoiseCheck(
        filter_name=filter_name,
        n_pix=int(mask.sum()),
        std=float(np.std(z)),
        mad=float(mad_std(z, ignore_nan=True)),
        median=float(np.median(z)),
        wht_type=None if header is None else header.get("WHTTYPE"),
        rnoise=None if header is None else header.get("RNOISE"),
    )


def wht_noise_check_from_fits(
    sci_path: str | Path,
    truth_path: str | Path,
    wht_path: str | Path,
    *,
    filter_name: str = "",
) -> WHTNoiseCheck:
    """Load FITS products and run :func:`wht_noise_check`."""

    sci = fits.getdata(sci_path)
    truth = fits.getdata(truth_path)
    wht = fits.getdata(wht_path)
    header = fits.getheader(wht_path)
    return wht_noise_check(sci, truth, wht, filter_name=filter_name, header=header)


def build_realistic_two_detector_mock(
    out_dir: str | Path,
    *,
    psf_dir: str | Path,
    nsrc: int = 300,
    center: tuple[float, float] = (34.50, -5.20),
    pa: float = 0.0,
    snr_range: tuple[float, float] = (1.0, 500.0),
    sigma_range: tuple[float, float] = (1.0, 5.0),
    point_source_fraction: float = 0.10,
    seed: int = 42,
    image_size: int | None = None,
    source_pattern: str = DEFAULT_F444W_PATTERN,
    target_pattern: str = DEFAULT_F770W_PATTERN,
    nircam_detectors: Sequence[str] = DEFAULT_NIRCAM_LW_DETECTORS,
    miri_detector: Sequence[str] = DEFAULT_MIRI_DETECTOR,
    f770w_position_shift_xy: tuple[float, float] | None = None,
    psf_gaussian_fwhm_arcsec: float | dict[str, float] | None = None,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], Table]:
    """Build the standard two-detector F444W/F770W realistic verification mock.

    The setup mirrors the realistic scratch validation: two NIRCam LW
    detectors, six NIRCam phase dithers, two MIRI macro pointings aligned to
    the LW detectors, and eight MIRI phase dithers at each macro position.

    Parameters
    ----------
    psf_gaussian_fwhm_arcsec
        Extra Gaussian broadening of the injected PSFs (FWHM, arcsec; float or
        per-filter dict). ``None`` uses the ``MockMosaic`` default
        (``mock_mosaic.DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC``, e.g. F770W 0.08");
        pass ``0.0`` or ``{}`` to disable. Model-PSF chains fitting this mock
        (and real-data drivers such as ``examples/run_770.py``) must apply the
        same broadening before kernel construction.
    """

    from mophongo.mock_mosaic import MockMosaic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_kwargs: dict[str, Any] = {}
    if image_size is not None and int(image_size) > 0:
        npix = int(image_size)
        mosaic_kwargs = {
            "mosaic_npix": (npix, npix),
            "mosaic_crpix": ((npix + 1) / 2, (npix + 1) / 2),
        }

    mock = MockMosaic(
        out_dir=out_dir,
        center_radec=center,
        nircam_lw_frames={"f444w": nircam_lw_phase_pointings(center, pa=pa, n_dither=6)},
        miri_frames={
            "f770w": miri_two_macro_phase_pointings(
                center,
                pa=pa,
                nircam_detectors=nircam_detectors,
                miri_detector=miri_detector,
                n_dither=8,
            )
        },
        mosaic_pscale="nircam_lw",
        exptime={"f444w": 418.734, "f770w": 444.006},
        pixfrac={"nircam_lw": 0.75, "miri": 1.00},
        psf_size_arcsec={"f444w": 4.0, "f770w": 8.0},
        stpsf_patterns={"f444w": source_pattern, "f770w": target_pattern},
        detectors={"f444w": tuple(nircam_detectors), "f770w": tuple(miri_detector)},
        stpsf_dir=Path(psf_dir),
        snr_range=snr_range,
        source_sigma_pix=sigma_range,
        source_sigma_pscale=0.040,
        point_source_fraction=point_source_fraction,
        source_psf_normalization="native",
        apertures_arcsec=(0.32, 0.7),
        noise_seed=seed,
        **(
            {}
            if psf_gaussian_fwhm_arcsec is None
            else {"psf_gaussian_fwhm_arcsec": psf_gaussian_fwhm_arcsec}
        ),
        **mosaic_kwargs,
    )
    mock.to_json(out_dir / "mock_config.json")
    paths = mock.write()
    write_pointing_summary(paths, out_dir)
    noise_info = mock.inject_noise_all(paths)
    dpsfs = mock.load_drizzle_psfs(paths, psf_dir=Path(psf_dir))
    filter_position_offsets: dict[str, tuple[float, float]] | None = None
    if f770w_position_shift_xy is not None:
        dx, dy = float(f770w_position_shift_xy[0]), float(f770w_position_shift_xy[1])
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            raise ValueError("f770w_position_shift_xy is limited to +/-1 native F770W pixel per axis")
        filter_position_offsets = {"f770w": (dx, dy)}
    truth = mock.inject_point_sources(
        paths,
        dpsfs,
        n=int(nsrc),
        ref_filter="f770w",
        sample_filters=("f444w", "f770w"),
        filter_position_offsets_pix=filter_position_offsets,
        seed=seed,
    )
    truth.write(out_dir / "mock_truth.ecsv", format="ascii.ecsv", overwrite=True)
    mock.plot(save=out_dir / "mock_mosaic.png", dpi=180)
    return mock, paths, noise_info, dpsfs, truth


def apply_mock_filter_blur_on_grid(
    mock: Any,
    filter_name: str,
    psf: np.ndarray,
    *,
    grid_pscale: float,
) -> np.ndarray:
    """Apply MockMosaic extra PSF blur in angular units on the given PSF grid.

    Delegates to ``MockMosaic.blur_filter_psf`` so the PSF/kernel maps receive
    the exact same Fourier-domain Gaussian operator as the painted mock
    sources; any operator difference between the two paths shows up as a
    spurious data-vs-model PSF mismatch in the verification fits.
    """

    return mock.blur_filter_psf(filter_name, psf, pscale=float(grid_pscale))





def psf_centroid_info(psf: np.ndarray, prefix: str) -> dict[str, float]:
    """Measure a PSF stamp centroid without modifying the stamp."""

    from photutils.centroids import centroid_com

    arr = np.asarray(psf, dtype=float)
    py, px = np.unravel_index(np.nanargmax(arr), arr.shape)
    radius = 5
    y0, y1 = max(0, py - radius), min(arr.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(arr.shape[1], px + radius + 1)
    cx_sub, cy_sub = centroid_com(arr[y0:y1, x0:x1])
    if not (np.isfinite(cx_sub) and np.isfinite(cy_sub)):
        cy = float(py)
        cx = float(px)
    else:
        cy = float(y0 + cy_sub)
        cx = float(x0 + cx_sub)
    center_y = float(arr.shape[0] // 2)
    center_x = float(arr.shape[1] // 2)
    return {
        f"{prefix}_peak_y": float(py),
        f"{prefix}_peak_x": float(px),
        f"{prefix}_centroid_y": cy,
        f"{prefix}_centroid_x": cx,
        f"{prefix}_centroid_dy_from_array_center": cy - center_y,
        f"{prefix}_centroid_dx_from_array_center": cx - center_x,
    }


def build_wiener_psf_maps(
    mock: Any,
    paths: dict[str, dict[str, Any]],
    dpsfs: dict[str, Any],
    out_dir: str | Path,
    *,
    psf_dir: str | Path,
    reg_grid: Sequence[float] | np.ndarray = DEFAULT_WIENER_REG_GRID,
    kernel_grid_nside: int = 1,
    source_pattern: str = DEFAULT_F444W_PATTERN,
    target_pattern: str = DEFAULT_F770W_PATTERN,
    source_filter: str = "f444w",
    target_filter: str = "f770w",
    psf_size_arcsec: float = 8.0,
) -> WienerPSFMaps:
    """Build PSF region maps and a spatial Wiener kernel map.

    Native finite-stamp PSF sums are preserved as throughput metadata.  The
    maps used by the pipeline contain only unit-sum PSF shapes and matching
    kernels, following the package PSF shape/throughput convention.

    ``kernel_grid_nside`` is kept only for older callers and must be 1.
    Artificially subdividing PSF regions is not part of the standard
    verification path; PSFRegionMap already encodes the footprint overlap
    geometry needed for rotations and multi-frame coverage.
    """

    from mophongo.psf import PSF
    from mophongo.psf_map import PSFRegionMap
    from mophongo.utils import fftconvolve, matching_kernel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_grid = parse_regularization_grid(reg_grid)

    dpsf_source = dpsfs[source_filter]
    dpsf_target = dpsfs[target_filter]
    source_map = PSFRegionMap.from_footprints(dpsf_source.footprint, name=source_filter.upper()).overlay_with(
        dpsf_source.driz_footprint
    )
    target_map = PSFRegionMap.from_footprints(dpsf_target.footprint, name=target_filter.upper()).overlay_with(
        dpsf_target.driz_footprint
    )
    if int(kernel_grid_nside) != 1:
        raise ValueError(
            "kernel_grid_nside subdivision is no longer supported; "
            "PSFRegionMap already encodes the frame-overlap regions. "
            "Use kernel_grid_nside=1."
        )
    kernel_map = source_map.overlay_with(target_map)

    positions = [(float(geom.centroid.x), float(geom.centroid.y)) for geom in kernel_map.regions.geometry]
    dpsf_source.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=source_pattern)
    dpsf_target.epsf_obj.load_jwst_stdpsf(local_dir=str(psf_dir), filter_pattern=target_pattern)
    size_pix_source = int(round(float(psf_size_arcsec) / float(paths[source_filter]["pscale"])))
    wcs_slices = [
        dpsf_source.get_driz_cutout(ra, dec, size=size_pix_source, recenter=False).wcs
        for ra, dec in positions
    ]

    source_psfs = []
    target_psfs_on_source_grid = []
    source_kernel = dpsf_source.driz_header["KERNEL"] if "KERNEL" in dpsf_source.driz_header else "square"
    target_kernel = dpsf_target.driz_header["KERNEL"] if "KERNEL" in dpsf_target.driz_header else "square"
    source_pixfrac = dpsf_source.driz_header["PIXFRAC"] if "PIXFRAC" in dpsf_source.driz_header else 0.75
    target_pixfrac = dpsf_target.driz_header["PIXFRAC"] if "PIXFRAC" in dpsf_target.driz_header else 0.75
    for (ra, dec), wcs_slice in zip(positions, wcs_slices):
        source_psfs.append(
            dpsf_source.get_psf(
                ra,
                dec,
                filter=source_pattern,
                wcs_slice=wcs_slice,
                kernel=source_kernel,
                pixfrac=source_pixfrac,
            )
        )
        target_psfs_on_source_grid.append(
            dpsf_target.get_psf(
                ra,
                dec,
                filter=target_pattern,
                wcs_slice=wcs_slice,
                kernel=target_kernel,
                pixfrac=target_pixfrac,
            )
        )

    source_native = apply_mock_filter_blur_on_grid(
        mock,
        source_filter,
        np.asarray(source_psfs),
        grid_pscale=paths[source_filter]["pscale"],
    )
    target_native = apply_mock_filter_blur_on_grid(
        mock,
        target_filter,
        np.asarray(target_psfs_on_source_grid),
        grid_pscale=paths[source_filter]["pscale"],
    )

    source_shapes = []
    target_shapes = []
    source_throughputs = []
    target_throughputs = []
    for psf_source, psf_target in zip(source_native, target_native):
        source_shape = prepare_psf_shape(psf_source, source_filter.upper())
        target_shape = prepare_psf_shape(psf_target, target_filter.upper())
        source_shapes.append(source_shape.shape)
        target_shapes.append(target_shape.shape)
        source_throughputs.append(source_shape.throughput)
        target_throughputs.append(target_shape.throughput)
    source_throughputs_arr = np.asarray(source_throughputs, dtype=float)
    target_throughputs_arr = np.asarray(target_throughputs, dtype=float)

    reg_result = PSF.from_array(source_shapes[0]).optimize_matching_kernel_regularization(
        PSF.from_array(target_shapes[0]),
        method="wiener",
        reg_grid=reg_grid,
        recenter=False,
        growth_weight=1.0,
        core_weight=1.0,
        l2_weight=1.0,
        kernel_regularization_weight=1e-3,
        diagnostic_path=out_dir / "diagnostic_wiener.png",
        source_label=source_filter.upper(),
        target_label=f"{target_filter.upper()} (target)",
        diagnostic_title="PSF matching kernel diagnostic - wiener",
        diagnostic_note=(
            "two-detector representative region 0; PSF shapes are unit-sum for "
            "matching; finite-stamp sums are throughput metadata"
        ),
    )
    wiener_lambda = float(reg_result.reg)
    Table(
        {
            "wiener_lambda": reg_result.reg_grid,
            "score": reg_result.score_grid,
            "growth_error": reg_result.growth_error_grid,
            "core_error": reg_result.core_error_grid,
            "l2_error": reg_result.l2_error_grid,
            "kernel_regularization": reg_result.kernel_regularization_grid,
        }
    ).write(out_dir / "psf_kernel_wiener_lambda_scan.csv", overwrite=True)

    kernels = []
    rows = []
    for idx, (source, target, source_native_i) in enumerate(
        zip(source_shapes, target_shapes, source_native)
    ):
        kernel = matching_kernel(source, target, method="wiener", reg=wiener_lambda, recenter=False)
        matched = fftconvolve(source, kernel, mode="same")
        kernels.append(kernel)
        rows.append(
            {
                "region": idx,
                "wiener_lambda": wiener_lambda,
                "source_throughput": float(source_throughputs_arr[idx]),
                "source_shape_sum": float(np.sum(source)),
                "target_throughput": float(target_throughputs_arr[idx]),
                "target_shape_sum": float(np.sum(target)),
                "kernel_sum": float(np.sum(kernel)),
                "kernel_l1": float(np.sum(np.abs(kernel))),
                "matched_sum": float(np.sum(matched)),
                "matched_target_rms": float(np.sqrt(np.nanmean((matched - target) ** 2))),
                **psf_centroid_info(source_native_i, "source"),
                **psf_centroid_info(target, "target_common_grid"),
            }
        )

    source_map.psfs = np.asarray(source_shapes)
    target_map.psfs = np.asarray(target_shapes)
    kernel_map.psfs = np.asarray(kernels)
    source_map.to_file(out_dir / f"prm_{source_filter}_psf.geojson")
    target_map.to_file(out_dir / f"prm_{target_filter}_psf.geojson")
    kernel_map.to_file(out_dir / f"prm_{source_filter}_wiener_kernel_{target_filter}.geojson")
    Table(rows=rows).write(out_dir / "psf_kernel_wiener_results.csv", overwrite=True)
    return WienerPSFMaps(
        source_map=source_map,
        target_map=target_map,
        kernel_map=kernel_map,
        wiener_lambda=wiener_lambda,
        source_throughputs=source_throughputs_arr,
        target_throughputs=target_throughputs_arr,
    )


def _truth_lookup(truth: Table) -> dict[int, object]:
    return {int(row["id"]): row for row in truth}


def build_source_recovery_table(
    fit_table: Table,
    truth: Table,
    *,
    true_flux_col: str,
    fitted_indices: Iterable[int],
    snr_col: str | None = None,
    sigma_col: str | None = "source_sigma_pix",
    point_source_col: str | None = "is_point_source",
    template_extension: str | None = None,
) -> Table:
    """Build a reusable source-recovery table from a pipeline catalog.

    ``flux_<i>`` is preserved as ``flux_<i>_model``.  ``flux_<i>_total`` is read
    from the pipeline catalog when present, otherwise computed from
    ``throughput_<i>``.
    """

    truth_by_id = _truth_lookup(truth)
    ids = np.asarray(fit_table["id"], dtype=int)
    flux_true = np.array([float(truth_by_id[int(i)][true_flux_col]) for i in ids])

    out = Table()
    out["id"] = ids
    out["flux_true"] = flux_true
    if snr_col is not None and snr_col in truth.colnames:
        out[snr_col] = np.array([float(truth_by_id[int(i)][snr_col]) for i in ids])
    if sigma_col is not None and sigma_col in truth.colnames:
        out[sigma_col] = np.array([float(truth_by_id[int(i)][sigma_col]) for i in ids])
    if point_source_col is not None and point_source_col in truth.colnames:
        out[point_source_col] = np.array(
            [bool(truth_by_id[int(i)][point_source_col]) for i in ids],
            dtype=np.int8,
        )

    for name in ("is_deblended", "deblend_parent_label", "deblend_nchildren"):
        if name in fit_table.colnames:
            out[name] = fit_table[name]

    for idx in fitted_indices:
        flux_model = np.asarray(fit_table[f"flux_{idx}"], dtype=float)
        err_model = np.asarray(fit_table[f"err_{idx}"], dtype=float)
        err_pred_model = np.asarray(fit_table[f"err_pred_{idx}"], dtype=float)
        throughput = (
            np.asarray(fit_table[f"throughput_{idx}"], dtype=float)
            if f"throughput_{idx}" in fit_table.colnames
            else np.ones(len(fit_table), dtype=float)
        )
        throughput = np.where(np.isfinite(throughput) & (throughput > 0.0), throughput, 1.0)
        flux_total = (
            np.asarray(fit_table[f"flux_{idx}_total"], dtype=float)
            if f"flux_{idx}_total" in fit_table.colnames
            else flux_model / throughput
        )
        err_total = (
            np.asarray(fit_table[f"err_{idx}_total"], dtype=float)
            if f"err_{idx}_total" in fit_table.colnames
            else err_model / throughput
        )
        err_pred_total = (
            np.asarray(fit_table[f"err_pred_{idx}_total"], dtype=float)
            if f"err_pred_{idx}_total" in fit_table.colnames
            else err_pred_model / throughput
        )

        out[f"throughput_{idx}"] = throughput
        out[f"flux_{idx}_model"] = flux_model
        out[f"err_{idx}_model"] = err_model
        out[f"err_pred_{idx}_model"] = err_pred_model
        out[f"ratio_{idx}_model"] = flux_model / flux_true
        out[f"flux_{idx}_total"] = flux_total
        out[f"err_{idx}_total"] = err_total
        out[f"err_pred_{idx}_total"] = err_pred_total
        out[f"ratio_{idx}"] = flux_total / flux_true
        out[f"pull_{idx}_pred"] = (flux_total - flux_true) / err_pred_total
        out[f"pull_{idx}_cov"] = (flux_total - flux_true) / err_total

    if template_extension is not None:
        out["template_extension"] = np.array([template_extension] * len(fit_table))
    return out


def segment_weighted_positions(
    image: np.ndarray,
    segmap: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flux-weighted mean (x, y) of each segment's positive pixels on ``image``.

    Segments with no positive flux return NaN.
    """
    from scipy import ndimage

    img = np.asarray(image, dtype=float)
    w = np.where(np.isfinite(img) & (img > 0), img, 0.0)
    ids = np.asarray(ids, dtype=int)
    wsum = ndimage.sum_labels(w, labels=segmap, index=ids)
    yy, xx = np.mgrid[: segmap.shape[0], : segmap.shape[1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        xw = ndimage.sum_labels(w * xx, labels=segmap, index=ids) / wsum
        yw = ndimage.sum_labels(w * yy, labels=segmap, index=ids) / wsum
    bad = ~(wsum > 0)
    xw = np.where(bad, np.nan, xw)
    yw = np.where(bad, np.nan, yw)
    return xw, yw


def remap_detection_to_truth(
    sci_path: str | Path,
    wht_path: str | Path,
    truth: Table,
    *,
    ndilate: int = 0,
) -> tuple[np.ndarray, Table]:
    """Detect on F444W and remap detection labels onto truth-source ids.

    Catalog deblend metadata is propagated with the same column names used by
    :class:`mophongo.catalog.Catalog`: ``is_deblended``,
    ``deblend_parent_label``, and ``deblend_nchildren``.
    """

    from collections import Counter

    from photutils.segmentation import SegmentationImage
    from skimage.morphology import disk

    from mophongo.catalog import Catalog, safe_dilate_segmentation

    cat_obj = Catalog.from_fits(sci_path, wht_path, estimate_background=True, estimate_ivar=True)
    segdata = np.asarray(cat_obj.segmap.data)
    if int(ndilate) > 0:
        segdata = safe_dilate_segmentation(SegmentationImage(segdata), disk(int(ndilate)))
    segmap = np.zeros_like(segdata, dtype=np.int32)
    used: set[int] = set()
    ny, nx = segdata.shape
    keep = np.asarray(truth["valid_f444w"], dtype=bool) & np.asarray(truth["valid_f770w"], dtype=bool)
    truth_keep = truth[keep]
    detection_labels: list[int] = []
    for row in truth_keep:
        x = int(round(float(row["x_f444w"])))
        y = int(round(float(row["y_f444w"])))
        label = int(segdata[y, x]) if 0 <= x < nx and 0 <= y < ny else 0
        detection_labels.append(label)

    label_counts = Counter(label for label in detection_labels if label != 0)
    catalog_by_label = {int(row["id"]): row for row in cat_obj.table}
    catalog_colnames = set(cat_obj.table.colnames)
    cat = Table()
    cat["id"] = np.asarray(truth["id"], dtype=np.int32)
    cat["x"] = np.asarray(truth["x_f444w"], dtype=float)
    cat["y"] = np.asarray(truth["y_f444w"], dtype=float)
    cat = cat[keep]
    cat["detection_label"] = np.asarray(detection_labels, dtype=np.int32)
    cat["truth_sources_in_detection_label"] = np.asarray(
        [label_counts[label] if label != 0 else 0 for label in detection_labels],
        dtype=np.int32,
    )
    cat["is_deblended"] = np.asarray(
        [
            bool(catalog_by_label[label]["is_deblended"])
            if label in catalog_by_label and "is_deblended" in catalog_colnames
            else False
            for label in detection_labels
        ],
        dtype=bool,
    )
    cat["deblend_parent_label"] = np.asarray(
        [
            int(catalog_by_label[label]["deblend_parent_label"])
            if label in catalog_by_label and "deblend_parent_label" in catalog_colnames
            else 0
            for label in detection_labels
        ],
        dtype=np.int32,
    )
    cat["deblend_nchildren"] = np.asarray(
        [
            int(catalog_by_label[label]["deblend_nchildren"])
            if label in catalog_by_label and "deblend_nchildren" in catalog_colnames
            else 0
            for label in detection_labels
        ],
        dtype=np.int32,
    )
    cat["truth_sources_in_hires_detection"] = cat["truth_sources_in_detection_label"]
    cat["segmentation_detected"] = np.asarray([label != 0 for label in detection_labels], dtype=bool)

    # Collect the truth sources claiming each detection label. A label with
    # several truth members is a blended detection that photutils did not
    # deblend. Assign the whole segment to the BRIGHTEST member: its light
    # dominates the segment shape, so it must own the template. Assigning by
    # truth-catalog order can hand a bright source's segment to a faint
    # neighbour, which swaps the pair's fitted fluxes and biases the scene
    # astrometric shift fit. The remaining members get the 3x3 fallback.
    flux_col = "flux_f444w" if "flux_f444w" in truth_keep.colnames else None
    members_by_label: dict[int, list[tuple[float, int]]] = {}
    for row, label in zip(truth_keep, detection_labels):
        x = int(round(float(row["x_f444w"])))
        y = int(round(float(row["y_f444w"])))
        if label != 0 and 0 <= x < nx and 0 <= y < ny:
            flux = float(row[flux_col]) if flux_col else 0.0
            members_by_label.setdefault(label, []).append((flux, int(row["id"])))

    owner_by_label = {
        label: max(members)[1] for label, members in members_by_label.items()
    }
    for label, obj_id in owner_by_label.items():
        segmap[segdata == label] = obj_id
        used.add(label)

    # Undetected sources and non-owning blend members get a 3x3 stamp.
    for row, label in zip(truth_keep, detection_labels):
        obj_id = int(row["id"])
        if label != 0 and owner_by_label.get(label) == obj_id:
            continue
        x = int(round(float(row["x_f444w"])))
        y = int(round(float(row["y_f444w"])))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        y0, y1 = max(0, y - 1), min(ny, y + 2)
        x0, x1 = max(0, x - 1), min(nx, x + 2)
        segmap[y0:y1, x0:x1] = obj_id
    return segmap, cat


def actual_inverse_variance(
    noise_info: dict[str, Any] | None,
    filter_name: str,
    fallback_wht: np.ndarray,
) -> np.ndarray:
    """Return inverse variance for the actual mock pixel noise."""

    info = noise_info.get(filter_name, {}) if noise_info is not None else {}
    sigma = info.get("sigma_pix")
    if sigma is None:
        return np.asarray(fallback_wht, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma > 0, 1.0 / sigma**2, 0.0).astype(np.float32)


def residual_to_native(residual: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """Map a residual image back to the native science-image shape."""

    residual = np.asarray(residual, dtype=float)
    if residual.shape == native_shape:
        return residual
    fy = residual.shape[0] // native_shape[0]
    fx = residual.shape[1] // native_shape[1]
    if fy < 1 or fx < 1 or residual.shape != (native_shape[0] * fy, native_shape[1] * fx):
        raise ValueError(f"cannot map residual shape {residual.shape} to native shape {native_shape}")
    return block_reduce(residual, (fy, fx), func=np.sum)


def diagnostic_note(
    *,
    nsrc: int,
    sigma_range: tuple[float, float],
    point_source_fraction: float,
    template_dilate_segmap: int,
    wiener_lambda: float | None = None,
    template_extension: str | None = None,
    f770w_position_shift_xy: tuple[float, float] | None = None,
) -> str:
    """Return a compact caption for realistic two-detector verification output."""

    parts = [
        "Two-detector MockMosaic: F444W NRCA5+NRCB5, 6 NIRCam phase dithers",
        "F770W two macro pointings x 8 MIRI subpixel dithers",
        f"nsrc={int(nsrc)}",
        f"source sigma=[{sigma_range[0]:g}, {sigma_range[1]:g}] F444W pix",
        f"point sources={point_source_fraction:.0%}",
        f"pipeline template dilation={int(template_dilate_segmap)}",
        "PSF shapes are unit-sum for fitting; native stamp sums are throughput metadata",
    ]
    if wiener_lambda is not None:
        parts.append(f"Wiener lambda={float(wiener_lambda):.3g}")
    if template_extension is not None:
        parts.append(f"template_extension={template_extension}")
    if f770w_position_shift_xy is not None:
        parts.append(
            "F770W source-position shift="
            f"(dx={float(f770w_position_shift_xy[0]):.2f}, dy={float(f770w_position_shift_xy[1]):.2f}) pix"
        )
    return "; ".join(parts)


def run_pipeline_extension_scenario(
    scenario: str,
    *,
    out_dir: str | Path,
    paths: dict[str, dict[str, Any]],
    noise_info: dict[str, Any],
    truth: Table,
    psf_maps: WienerPSFMaps,
    mock_dilate_segmap: int = 2,
    template_dilate_segmap: int = 4,
    fit_astrometry_niter: int = 2,
    fit_background: bool = False,
    source_diagnostic_count: int = 10,
    full_diagnostic_highres_size: int | None = 3000,
    scene_diagnostic_count: int | None = 12,
    f770w_position_shift_xy: tuple[float, float] | None = None,
    nsrc: int | None = None,
    sigma_range: tuple[float, float] = (1.0, 5.0),
    point_source_fraction: float = 0.10,
    max_match_offset_pix: float = 3.0,
    fit_overrides: dict[str, Any] | None = None,
) -> PipelineScenarioResult:
    """Run one standard realistic verification scenario.

    Diagnostic images are emitted through existing package diagnostics only:
    ``Pipeline.diagnose_sources`` plus the standard flux-recovery helper.

    Args:
        fit_overrides: Extra :class:`~mophongo.fit.FitConfig` keyword overrides
            merged over the scenario defaults, e.g. a per-band
            ``aperture_diam`` or scene limits matching a production run.
    """

    import matplotlib.pyplot as plt

    from mophongo.catalog import get_bg_and_ivar
    from mophongo.fit import FitConfig
    from mophongo.pipeline import Pipeline

    scenario_dir = Path(out_dir) / f"template_extension_{scenario}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    sci_444 = paths["f444w"]["fits"]
    wht_444 = sci_444.with_name(sci_444.name.replace("_sci", "_wht"))
    sci_770 = paths["f770w"]["fits"]
    wht_770 = sci_770.with_name(sci_770.name.replace("_sci", "_wht"))
    segmap, cat = remap_detection_to_truth(
        sci_444,
        wht_444,
        truth,
        ndilate=int(mock_dilate_segmap),
    )

    img_444 = fits.getdata(sci_444).astype(np.float32)
    img_770 = fits.getdata(sci_770).astype(np.float32)
    wht_444_arr = fits.getdata(wht_444).astype(np.float32)
    wht_770_arr = fits.getdata(wht_770).astype(np.float32)
    ivar_444 = actual_inverse_variance(noise_info, "f444w", wht_444_arr)
    ivar_770 = actual_inverse_variance(noise_info, "f770w", wht_770_arr)
    if fit_background:
        bg_770, ivar_770 = get_bg_and_ivar(img_770, wht_770_arr, bg_filter_sigma=64.0)
    else:
        bg_770 = np.zeros_like(img_770, dtype=np.float32)
    wcs_444 = WCS(fits.getheader(sci_444))
    wcs_770 = WCS(fits.getheader(sci_770))

    fit_kwargs: dict[str, Any] = dict(
        reg_flux=0.0,
        fit_astrometry_niter=int(fit_astrometry_niter),
        fit_astrometry_joint=True,
        snr_thresh_astrom=0.0,
        scene_minimum_bright=10,
        aperture_diam=0.5,
        template_dilate_segmap=int(template_dilate_segmap),
    )
    fit_kwargs.update(fit_overrides or {})
    config = FitConfig(**fit_kwargs)
    pipe = Pipeline(
        [img_444, img_444, img_770 - bg_770],
        segmap,
        weights=[ivar_444, ivar_444, ivar_770],
        catalog=cat,
        psfs=[None, psf_maps.source_map, psf_maps.target_map],
        kernels=[None, None, psf_maps.kernel_map],
        psf_throughputs=[
            1.0,
            filter_average_throughput(psf_maps.source_throughputs),
            filter_average_throughput(psf_maps.target_throughputs),
        ],
        wcs=[wcs_444, wcs_444, wcs_770],
        extend_templates=None if scenario == "none" else scenario,
    )
    table, residuals = pipe.run(config=config)
    residual_native = residual_to_native(residuals[1], img_770.shape)
    model_native = img_770 - bg_770 - residual_native

    fits.writeto(scenario_dir / "segmap_truth_labels.fits", segmap, fits.getheader(sci_444), overwrite=True)
    fits.writeto(scenario_dir / "f770w_model.fits", model_native, fits.getheader(sci_770), overwrite=True)
    fits.writeto(scenario_dir / "f770w_residual.fits", residual_native, fits.getheader(sci_770), overwrite=True)

    source_table = build_source_recovery_table(
        table,
        truth,
        true_flux_col="flux_f770w",
        fitted_indices=(1, 2),
        snr_col="snr_f770w",
        template_extension=scenario,
    )

    # Guard against truth/template mismatches: a truth source whose segment
    # actually contains a different (usually brighter) source produces huge
    # flux ratios. Compare the flux-weighted position of each source's
    # segment on the detection image against the truth position and flag
    # rows beyond ``max_match_offset_pix``; flagged rows are excluded from
    # the flux-recovery plots below.
    fit_ids = np.asarray(table["id"], dtype=int)
    truth_by_id = _truth_lookup(truth)
    x_true = np.array([float(truth_by_id[i]["x_f444w"]) for i in fit_ids])
    y_true = np.array([float(truth_by_id[i]["y_f444w"]) for i in fit_ids])
    xw, yw = segment_weighted_positions(img_444, segmap, fit_ids)
    match_offset = np.hypot(xw - x_true, yw - y_true)
    position_matched = np.isfinite(match_offset) & (match_offset <= float(max_match_offset_pix))
    source_table["match_offset_pix"] = match_offset
    source_table["position_matched"] = position_matched
    source_table.write(scenario_dir / f"source_table_{scenario}.csv", overwrite=True)

    note = diagnostic_note(
        nsrc=int(nsrc if nsrc is not None else len(truth)),
        sigma_range=sigma_range,
        point_source_fraction=point_source_fraction,
        template_dilate_segmap=template_dilate_segmap,
        wiener_lambda=psf_maps.wiener_lambda,
        template_extension=scenario,
        f770w_position_shift_xy=f770w_position_shift_xy,
    )
    deblended_mask = (
        np.asarray(source_table["is_deblended"], dtype=bool)
        if "is_deblended" in source_table.colnames
        else None
    )
    is_deblended_by_id = (
        {int(row["id"]): bool(row["is_deblended"]) for row in cat}
        if "is_deblended" in cat.colnames
        else {}
    )
    stamp_grid, stamp_grid_labels = bright_source_residual_stamp_grid(
        residual_native,
        ivar_770,
        table["id"],
        _truth_lookup(truth),
        is_deblended_by_id,
        filt="f770w",
    )
    if full_diagnostic_highres_size is not None and int(full_diagnostic_highres_size) > 0:
        save_realistic_full_diagnostic(
            scenario_dir / f"full_diagnostic_{scenario}.png",
            paths=paths,
            img_444=img_444,
            img_770_bgsub=img_770 - bg_770,
            model_770=model_native,
            resid_770=residual_native,
            segmap=segmap,
            caption=note,
            stamp_grid=stamp_grid,
            stamp_grid_labels=stamp_grid_labels,
            highres_size=int(full_diagnostic_highres_size),
        )
    point_mask = (
        np.asarray(source_table["is_point_source"], dtype=bool)
        if "is_point_source" in source_table.colnames
        else None
    )
    sel = position_matched
    save_flux_recovery_plot(
        scenario_dir / f"flux_ratio_{scenario}_lowres.png",
        np.asarray(source_table["flux_true"], dtype=float)[sel],
        np.asarray(source_table["flux_2_total"], dtype=float)[sel],
        error=np.asarray(source_table["err_pred_2_total"], dtype=float)[sel],
        label=(
            "F770W fit: unit PSF-shape templates + throughput-corrected total flux; "
            f"nsrc={int(nsrc if nsrc is not None else len(truth))}, "
            f"sigma=[{sigma_range[0]:g}, {sigma_range[1]:g}], "
            f"template_extension={scenario}"
        ),
        xlabel="True Flux",
        ylabel="Recovered Total Flux (F770W)",
        snr_values=np.asarray(source_table["snr_f770w"], dtype=float)[sel]
        if "snr_f770w" in source_table.colnames
        else None,
        point_source_mask=point_mask[sel] if point_mask is not None else None,
        deblended_mask=deblended_mask[sel] if deblended_mask is not None else None,
        error_label="Predicted Error + 1% floor",
        systematic_error_fraction=0.01,
        caption=note,
    )
    save_flux_recovery_plot(
        scenario_dir / f"flux_ratio_{scenario}_hires.png",
        np.asarray(source_table["flux_true"], dtype=float)[sel],
        np.asarray(source_table["flux_1_total"], dtype=float)[sel],
        error=np.asarray(source_table["err_pred_1_total"], dtype=float)[sel],
        label=(
            "F444W self-fit: unit PSF-shape templates + throughput-corrected total flux; "
            f"nsrc={int(nsrc if nsrc is not None else len(truth))}, "
            f"sigma=[{sigma_range[0]:g}, {sigma_range[1]:g}], "
            f"template_extension={scenario}"
        ),
        xlabel="True Flux",
        ylabel="Recovered Total Flux (F444W)",
        snr_values=np.asarray(source_table["snr_f770w"], dtype=float)[sel]
        if "snr_f770w" in source_table.colnames
        else None,
        point_source_mask=point_mask[sel] if point_mask is not None else None,
        deblended_mask=deblended_mask[sel] if deblended_mask is not None else None,
        error_label="Predicted Error + 1% floor",
        systematic_error_fraction=0.01,
        caption=note,
    )

    diagnostic_ids = (
        {int(t.id) for t in pipe.templates_extracted.templates}
        & {int(t.id) for t in pipe.templates_extended.templates}
        & {int(t.id) for t in pipe.all_templates[1]}
    )
    flux_true = np.asarray(source_table["flux_true"], dtype=float)
    bright_ids = [
        int(source_table["id"][idx])
        for idx in np.argsort(flux_true)[::-1]
        if int(source_table["id"][idx]) in diagnostic_ids
    ][: int(source_diagnostic_count)]
    if bright_ids:
        fig, _ = pipe.diagnose_sources(
            bright_ids,
            ifilt=2,
            save=scenario_dir / f"source_stage_diagnostics_{scenario}.png",
        )
        plt.close(fig)

    scene_table = save_scene_diagnostics(
        pipe,
        img_444,
        segmap,
        scenario_dir / f"scene_diagnostics_{scenario}",
        max_scenes=scene_diagnostic_count,
    )

    noise = float(np.nanmedian(1.0 / np.sqrt(ivar_770[ivar_770 > 0])))
    ratio_1 = np.asarray(source_table["ratio_1"], dtype=float)[sel]
    ratio_2 = np.asarray(source_table["ratio_2"], dtype=float)[sel]
    pull_2 = np.asarray(source_table["pull_2_pred"], dtype=float)[sel]
    summary = {
        "template_extension": scenario,
        "n_fit": float(len(table)),
        "n_position_mismatched": float(np.sum(~position_matched)),
        "wiener_lambda": float(psf_maps.wiener_lambda),
        "n_source_diagnostics": float(len(bright_ids)),
        "med_hi": float(np.nanmedian(ratio_1)),
        "med_lo": float(np.nanmedian(ratio_2)),
        "p16_lo": float(np.nanpercentile(ratio_2, 16)),
        "p84_lo": float(np.nanpercentile(ratio_2, 84)),
        "pull_lo_median": float(np.nanmedian(pull_2)),
        "pull_lo_std": float(np.nanstd(pull_2)),
        "resid_std_over_noise": float(np.nanstd(residual_native) / noise),
    }
    if f770w_position_shift_xy is not None:
        native_dx = float(f770w_position_shift_xy[0])
        native_dy = float(f770w_position_shift_xy[1])
        scale_x = float(residuals[1].shape[1] / img_770.shape[1])
        scale_y = float(residuals[1].shape[0] / img_770.shape[0])
        expected_dx = native_dx * scale_x
        expected_dy = native_dy * scale_y
        if len(scene_table) > 0 and "median_dx_pix" in scene_table.colnames:
            recovered_dx = float(np.nanmedian(scene_table["median_dx_pix"]))
            recovered_dy = float(np.nanmedian(scene_table["median_dy_pix"]))
            error_dx = recovered_dx - expected_dx
            error_dy = recovered_dy - expected_dy
            recovered_ok = abs(error_dx) <= 0.5 and abs(error_dy) <= 0.5
        else:
            recovered_dx = np.nan
            recovered_dy = np.nan
            error_dx = np.nan
            error_dy = np.nan
            recovered_ok = False
        summary["f770w_shift_x_pix"] = native_dx
        summary["f770w_shift_y_pix"] = native_dy
        summary["f770w_shift_expected_dx_fitpix"] = expected_dx
        summary["f770w_shift_expected_dy_fitpix"] = expected_dy
        summary["f770w_shift_recovered_dx_fitpix"] = recovered_dx
        summary["f770w_shift_recovered_dy_fitpix"] = recovered_dy
        summary["f770w_shift_error_dx_fitpix"] = error_dx
        summary["f770w_shift_error_dy_fitpix"] = error_dy
        summary["f770w_shift_recovered_ok"] = recovered_ok
    return PipelineScenarioResult(
        scenario=scenario,
        pipeline=pipe,
        table=table,
        source_table=source_table,
        residuals=residuals,
        residual_native=residual_native,
        model_native=model_native,
        output_dir=scenario_dir,
        summary=summary,
    )


def diagnostic_lupton_norm(img: np.ndarray) -> ImageNormalize:
    """Return the same asinh display normalization used by standard diagnostics."""

    arr = np.asarray(img, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return ImageNormalize(vmin=0.0, vmax=1.0, stretch=AsinhStretch(0.01))
    p1, p99 = np.percentile(finite, [1, 99])
    return ImageNormalize(vmin=-p99 / 20.0, vmax=p99, stretch=AsinhStretch(0.01))


def label_segmap(ax: Any, segmap: np.ndarray, catalog: Table | None, fontsize: int = 10) -> None:
    """Label a segmentation map with catalog positions."""

    if catalog is None or "x" not in catalog.colnames or "y" not in catalog.colnames:
        return
    for idx, (y, x) in enumerate(zip(catalog["y"], catalog["x"]), start=1):
        ax.text(
            x,
            y,
            str(idx),
            color="white",
            fontsize=fontsize,
            ha="center",
            va="center",
            weight="medium",
            alpha=0.7,
        )


def save_diagnostic_image(
    filename: str | Path,
    truth: np.ndarray,
    hires: np.ndarray,
    lowres: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    *,
    segmap: np.ndarray | None = None,
    catalog: Table | None = None,
    caption: str | None = None,
    stamp_grid: np.ndarray | None = None,
    stamp_grid_title: str = "bright-source residual pulls",
    stamp_grid_labels: list[tuple[float, float, str, str]] | None = None,
    dpi: int | None = None,
) -> None:
    """Save the standard full-image truth/hires/lowres/model/residual diagnostic."""

    import matplotlib.pyplot as plt

    n_sources = len(np.unique(segmap[np.asarray(segmap) > 0])) if segmap is not None else 5
    cov_img = np.eye(n_sources) * 0.1
    panels = [
        (0, 0, truth, "truth", "gray", diagnostic_lupton_norm(truth)),
        (0, 1, hires, "hires", "gray", diagnostic_lupton_norm(hires)),
        (0, 2, segmap, "segmap", "nipy_spectral", None),
        (1, 0, lowres, "lowres", "gray", diagnostic_lupton_norm(lowres)),
        (1, 1, model, "model", "gray", diagnostic_lupton_norm(model)),
        (1, 2, residual, "residual", "gray", None),
    ]
    figsize = (16, 8)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    for row, col, img, title, cmap, norm in panels:
        ax = axes[row, col]
        if img is None:
            ax.axis("off")
            continue
        if title == "residual":
            finite = np.asarray(img, dtype=float)
            std = np.nanstd(finite[np.isfinite(finite)])
            vlim = 5.0 * std if np.isfinite(std) and std > 0 else 1.0
            ax.imshow(img, cmap=cmap, origin="lower", vmin=-vlim, vmax=vlim)
        elif title == "segmap":
            ax.imshow(img, cmap=cmap, origin="lower")
            label_segmap(ax, np.asarray(img), catalog)
        elif norm is not None:
            ax.imshow(img, cmap=cmap, origin="lower", norm=norm)
        else:
            ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_stamp = axes[0, 3]
    if stamp_grid is not None:
        im_stamp = ax_stamp.imshow(stamp_grid, cmap="RdBu_r", origin="lower", vmin=-5, vmax=5)
        ax_stamp.set_title(stamp_grid_title)
        ax_stamp.set_xticks([])
        ax_stamp.set_yticks([])
        if stamp_grid_labels is not None:
            for x, y, text, color in stamp_grid_labels:
                ax_stamp.text(
                    x,
                    y,
                    text,
                    color=color,
                    fontsize=7,
                    ha="left",
                    va="top",
                    weight="bold",
                    bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.0},
                )
        cbar_stamp = fig.colorbar(im_stamp, ax=ax_stamp, fraction=0.046, pad=0.04)
        cbar_stamp.set_label("Residual / noise", rotation=270, labelpad=15)
    else:
        ax_stamp.axis("off")

    ax_cov = axes[1, 3]
    vmax = np.abs(cov_img).max()
    if vmax > 0:
        im = ax_cov.imshow(cov_img, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    else:
        im = ax_cov.imshow(cov_img, cmap="gray", origin="lower")
    ax_cov.set_title(f"Covariance Matrix ({cov_img.shape[0]}x{cov_img.shape[1]})")
    ax_cov.set_xlabel("Source Index")
    ax_cov.set_ylabel("Source Index")
    cbar = fig.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance", rotation=270, labelpad=15)

    if caption:
        fig.text(0.01, 0.012, fill(caption, width=150), fontsize=8, ha="left", va="bottom")
        plt.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        plt.tight_layout()
    save_dpi = int(dpi) if dpi is not None else _diagnostic_pixel_sampling_dpi(
        [truth, hires, lowres, model, residual, segmap, stamp_grid],
        figsize=figsize,
        nrows=2,
        ncols=4,
    )
    fig.savefig(filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


def _diagnostic_pixel_sampling_dpi(
    panels: Iterable[np.ndarray | None],
    *,
    figsize: tuple[float, float],
    nrows: int,
    ncols: int,
    min_dpi: int = 150,
    max_dpi: int = 1200,
    oversample: float = 1.0,
) -> int:
    """Return a DPI high enough that diagnostic panels sample image pixels.

    ``oversample`` requests that many output pixels per displayed image pixel.
    """

    shapes = [np.asarray(panel).shape[:2] for panel in panels if panel is not None]
    if not shapes:
        return int(min_dpi)
    max_ny = max(int(shape[0]) for shape in shapes)
    max_nx = max(int(shape[1]) for shape in shapes)
    panel_width_in = float(figsize[0]) / float(ncols)
    panel_height_in = float(figsize[1]) / float(nrows)
    # Allow for titles, labels, colorbars, and tight-layout padding. The saved
    # PNG should have at least ``oversample`` output pixels per displayed image
    # pixel in the main panels, even for large full-diagnostic crops.
    needed_x = int(np.ceil(1.25 * oversample * max_nx / max(panel_width_in, 1e-6)))
    needed_y = int(np.ceil(1.25 * oversample * max_ny / max(panel_height_in, 1e-6)))
    return int(np.clip(max(min_dpi, needed_x, needed_y), min_dpi, max_dpi))


def bright_source_residual_stamp_grid(
    residual: np.ndarray,
    ivar: np.ndarray,
    ids: Iterable[int],
    truth_by_id: dict[int, Any],
    is_deblended_by_id: dict[int, bool],
    *,
    filt: str = "f770w",
    n_stamps: int = 16,
    half_size: int = 6,
) -> tuple[np.ndarray, list[tuple[float, float, str, str]]]:
    """Return residual/noise stamps plus labels marking deblended detections."""

    residual = np.asarray(residual, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    size = 2 * int(half_size) + 1
    selected = [
        int(obj_id)
        for obj_id in ids
        if int(obj_id) in truth_by_id and np.isfinite(float(truth_by_id[int(obj_id)][f"flux_{filt}"]))
    ]
    selected = sorted(
        selected,
        key=lambda obj_id: float(truth_by_id[obj_id][f"flux_{filt}"]),
        reverse=True,
    )[: int(n_stamps)]
    ncols = int(np.ceil(np.sqrt(max(1, len(selected)))))
    nrows = int(np.ceil(max(1, len(selected)) / ncols))
    grid = np.full(
        (nrows * size + max(0, nrows - 1), ncols * size + max(0, ncols - 1)),
        np.nan,
        dtype=float,
    )
    labels: list[tuple[float, float, str, str]] = []
    ny, nx = residual.shape
    for idx, obj_id in enumerate(selected):
        row = truth_by_id[obj_id]
        x = int(round(float(row[f"x_{filt}"])))
        y = int(round(float(row[f"y_{filt}"])))
        y0, y1 = max(0, y - half_size), min(ny, y + half_size + 1)
        x0, x1 = max(0, x - half_size), min(nx, x + half_size + 1)
        tile = np.full((size, size), np.nan, dtype=float)
        ty0 = y0 - (y - half_size)
        tx0 = x0 - (x - half_size)
        sigma_inv = np.sqrt(np.maximum(ivar[y0:y1, x0:x1], 0.0))
        tile[ty0 : ty0 + (y1 - y0), tx0 : tx0 + (x1 - x0)] = (
            residual[y0:y1, x0:x1] * sigma_inv
        )
        gy = idx // ncols
        gx = idx % ncols
        oy = gy * (size + 1)
        ox = gx * (size + 1)
        grid[oy : oy + size, ox : ox + size] = tile
        is_deblended = bool(is_deblended_by_id.get(obj_id, False))
        labels.append(
            (
                float(ox + 1),
                float(oy + size - 1),
                f"{obj_id}{' D' if is_deblended else ''}",
                "yellow" if is_deblended else "white",
            )
        )
    return grid, labels


def crop_from_origin(arr: np.ndarray, shape: tuple[int, int], origin_yx: tuple[float, float]) -> np.ndarray:
    """Crop ``arr`` to ``shape`` starting at ``origin_yx`` while staying in bounds."""

    ny = min(int(shape[0]), arr.shape[0])
    nx = min(int(shape[1]), arr.shape[1])
    y0 = min(max(0, int(np.floor(float(origin_yx[0])))), max(0, arr.shape[0] - ny))
    x0 = min(max(0, int(np.floor(float(origin_yx[1])))), max(0, arr.shape[1] - nx))
    return arr[y0 : y0 + ny, x0 : x0 + nx]


def segmap_lower_left_origin(segmap: np.ndarray) -> tuple[float, float]:
    """Return the lower-left origin of the detected-source footprint."""

    yy, xx = np.nonzero(np.asarray(segmap) > 0)
    if yy.size == 0:
        return (0.0, 0.0)
    return (float(np.min(yy)), float(np.min(xx)))


def save_realistic_full_diagnostic(
    path: str | Path,
    *,
    paths: dict[str, dict[str, Any]],
    img_444: np.ndarray,
    img_770_bgsub: np.ndarray,
    model_770: np.ndarray,
    resid_770: np.ndarray,
    segmap: np.ndarray,
    caption: str,
    stamp_grid: np.ndarray | None,
    stamp_grid_labels: list[tuple[float, float, str, str]] | None,
    highres_size: int,
) -> None:
    """Save the standard full-image diagnostic from the lower-left covered tile."""

    truth_444 = fits.getdata(paths["f444w"]["truth_fits"]).astype(np.float32)
    pixel_ratio = round(paths["f770w"]["pscale"] / paths["f444w"]["pscale"])
    requested_highres_size = max(int(highres_size), pixel_ratio)
    low_shape = (
        min(img_770_bgsub.shape[0], requested_highres_size // pixel_ratio),
        min(img_770_bgsub.shape[1], requested_highres_size // pixel_ratio),
    )
    high_shape = (low_shape[0] * pixel_ratio, low_shape[1] * pixel_ratio)
    high_origin = segmap_lower_left_origin(segmap)
    low_origin = (high_origin[0] / pixel_ratio, high_origin[1] / pixel_ratio)
    seg_crop = crop_from_origin(segmap, high_shape, high_origin)
    if not np.any(seg_crop > 0):
        high_origin = (0.0, 0.0)
        low_origin = (0.0, 0.0)
        seg_crop = crop_from_origin(segmap, high_shape, high_origin)
    save_diagnostic_image(
        path,
        crop_from_origin(truth_444, high_shape, high_origin),
        crop_from_origin(img_444, high_shape, high_origin),
        crop_from_origin(img_770_bgsub, low_shape, low_origin),
        crop_from_origin(model_770, low_shape, low_origin),
        crop_from_origin(resid_770, low_shape, low_origin),
        segmap=seg_crop,
        catalog=Table({"x": np.array([], dtype=float), "y": np.array([], dtype=float)}),
        caption=caption,
        stamp_grid=stamp_grid,
        stamp_grid_labels=stamp_grid_labels,
    )


def save_scene_diagnostics(
    pipe: Any,
    tmpl_image: np.ndarray,
    segmap: np.ndarray,
    out_dir: str | Path,
    *,
    scene_collection_index: int = -1,
    max_scenes: int | None = 12,
    display_sig: float = 5.0,
) -> Table:
    """Save existing ``Scene.plot`` diagnostics from a fitted pipeline run."""

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in out_dir.glob("scene_*.png"):
        stale_png.unlink()
    scene_groups = getattr(pipe, "all_scenes", [])
    if not scene_groups:
        table = Table()
        table.write(out_dir / "scene_catalog.csv", overwrite=True)
        return table
    scenes = scene_groups[int(scene_collection_index)]
    residual_groups = getattr(pipe, "residuals", None)
    residual_image = (
        residual_groups[int(scene_collection_index)]
        if residual_groups
        else None
    )
    save_scene_overview(
        tmpl_image,
        segmap,
        scenes,
        out_dir / "scene_overview.png",
    )
    rows = []
    for scene in scenes:
        shifts = []
        for tmpl in getattr(scene, "templates", []):
            shifted = getattr(tmpl, "shifted", None)
            if shifted is not None:
                arr = np.asarray(shifted, dtype=float)
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    shifts.append(arr[:2])
        shift_arr = np.asarray(shifts, dtype=float) if shifts else np.zeros((0, 2), dtype=float)
        if shift_arr.size:
            shift_norm = np.sqrt(np.sum(shift_arr**2, axis=1))
            med_dx = float(np.nanmedian(shift_arr[:, 0]))
            med_dy = float(np.nanmedian(shift_arr[:, 1]))
            max_shift = float(np.nanmax(shift_norm))
        else:
            med_dx = 0.0
            med_dy = 0.0
            max_shift = 0.0
        rows.append(
            {
                "id": int(scene.id),
                "n_templates": int(len(scene.templates)),
                "n_bright": int(np.sum(scene.is_bright)),
                "bbox_y0": int(scene.bbox[0]),
                "bbox_y1": int(scene.bbox[1]),
                "bbox_x0": int(scene.bbox[2]),
                "bbox_x1": int(scene.bbox[3]),
                "bbox_height_pix": int(scene.bbox[1] - scene.bbox[0] + 1),
                "bbox_width_pix": int(scene.bbox[3] - scene.bbox[2] + 1),
                "median_dx_pix": med_dx,
                "median_dy_pix": med_dy,
                "max_shift_pix": max_shift,
            }
        )
    table = Table(rows=rows)
    table.sort(["n_bright", "n_templates", "id"], reverse=True)
    table.write(out_dir / "scene_catalog.csv", overwrite=True)

    selected = scenes
    if max_scenes is not None:
        selected_ids = set(int(row["id"]) for row in table[: int(max_scenes)])
        selected = [scene for scene in scenes if int(scene.id) in selected_ids]
    for scene in selected:
        fig, _ = scene.plot(
            tmpl_image, segmap, display_sig=display_sig, residual_image=residual_image
        )
        scene_dpi = _diagnostic_pixel_sampling_dpi(
            [scene.model_image()],
            figsize=(15, 10),
            nrows=2,
            ncols=3,
            min_dpi=400,
            max_dpi=2400,
            oversample=2.0,
        )
        fig.savefig(out_dir / f"scene_{int(scene.id)}.png", dpi=scene_dpi, bbox_inches="tight")
        plt.close(fig)
    return table


def save_scene_overview(
    image: np.ndarray,
    segmap: np.ndarray,
    scenes: Sequence[Any],
    filename: str | Path,
    *,
    alpha: float = 0.42,
) -> None:
    """Save a full-field overview with segmentation colored by fitted scene."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    image = np.asarray(image, dtype=float)
    segmap = np.asarray(segmap)
    scene_map = np.zeros(segmap.shape, dtype=np.int32)
    for scene_index, scene in enumerate(scenes, start=1):
        scene_template_ids = {int(t.id) for t in getattr(scene, "templates", [])}
        if not scene_template_ids:
            continue
        mask = np.isin(segmap, list(scene_template_ids))
        scene_map[mask] = int(scene_index)

    finite = np.isfinite(image)
    med = float(np.nanmedian(image[finite])) if np.any(finite) else 0.0
    sig = float(np.nanstd(image[finite])) if np.any(finite) else 1.0
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0

    rng = np.random.default_rng(12345)
    n_colors = max(1, len(scenes))
    colors = np.zeros((n_colors + 1, 4), dtype=float)
    colors[0] = (1.0, 1.0, 1.0, 0.0)
    hue = np.linspace(0.0, 1.0, n_colors, endpoint=False)
    rng.shuffle(hue)
    import matplotlib.colors as mcolors

    for idx, h in enumerate(hue, start=1):
        rgb = mcolors.hsv_to_rgb((float(h), 0.58, 0.98))
        colors[idx] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))

    figsize = (14, 7)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, origin="lower", cmap="gray", vmin=med - 1.0 * sig, vmax=med + 8.0 * sig)
    ax.imshow(
        np.ma.masked_where(scene_map == 0, scene_map),
        origin="lower",
        cmap=ListedColormap(colors),
        interpolation="nearest",
        vmin=0,
        vmax=n_colors,
    )
    for scene_index, scene in enumerate(scenes, start=1):
        if getattr(scene, "bbox", None) is None:
            continue
        y0, y1, x0, x1 = scene.bbox
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color=colors[scene_index],
            linewidth=0.8,
            alpha=0.65,
        )
    ax.set_title(f"Scene overview ({len(scenes)} fitted scenes)")
    ax.set_xticks([])
    ax.set_yticks([])
    overview_dpi = _diagnostic_pixel_sampling_dpi(
        [image],
        figsize=figsize,
        nrows=1,
        ncols=1,
        min_dpi=180,
        max_dpi=900,
    )
    fig.savefig(filename, dpi=overview_dpi, bbox_inches="tight")
    plt.close(fig)


def save_flux_recovery_plot(
    filename: str | Path,
    truth: np.ndarray,
    recovered: np.ndarray,
    *,
    error: np.ndarray | None = None,
    label: str = "Recovered Flux",
    xlabel: str = "True Flux",
    ylabel: str = "Recovered Flux",
    snr_values: np.ndarray | None = None,
    point_source_mask: np.ndarray | None = None,
    deblended_mask: np.ndarray | None = None,
    error_label: str = "Error",
    systematic_error_fraction: float = 0.0,
    caption: str | None = None,
) -> None:
    """Write the standard four-panel flux recovery diagnostic PNG."""

    import matplotlib.pyplot as plt

    truth = np.asarray(truth, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    error = None if error is None else np.asarray(error, dtype=float)
    if error is not None and systematic_error_fraction > 0.0:
        error = np.hypot(error, systematic_error_fraction * np.abs(truth))
    ratio = recovered / truth
    point_source_mask = (
        np.zeros(len(truth), dtype=bool)
        if point_source_mask is None
        else np.asarray(point_source_mask, dtype=bool)
    )
    deblended_mask = (
        np.zeros(len(truth), dtype=bool)
        if deblended_mask is None
        else np.asarray(deblended_mask, dtype=bool)
    )
    base_mask = ~deblended_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, y, ylabel_i in (
        (axes[0, 0], recovered, ylabel),
        (axes[0, 1], ratio, "Recovered / True"),
    ):
        if np.any(base_mask):
            ax.scatter(truth[base_mask], y[base_mask], s=20, alpha=0.4, label="Data")
        ps = point_source_mask & base_mask
        if np.any(ps):
            ax.scatter(truth[ps], y[ps], s=26, color="tab:green", alpha=0.8, label="Point source")
        db = deblended_mask & ~point_source_mask
        if np.any(db):
            ax.scatter(
                truth[db], y[db], s=52, facecolors="none", edgecolors="tab:blue",
                linewidths=1.4, label="Deblended child",
            )
        dbps = deblended_mask & point_source_mask
        if np.any(dbps):
            ax.scatter(
                truth[dbps], y[dbps], s=52, facecolors="none", edgecolors="tab:green",
                linewidths=1.4, label="_Point source deblended child",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_i)
    minval = min(np.nanmin(truth), np.nanmin(recovered))
    maxval = max(np.nanmax(truth), np.nanmax(recovered))
    axes[0, 0].plot([minval, maxval], [minval, maxval], "k--", label="y=x")
    positive_min = max(minval, 1e-6)
    if maxval > positive_min:
        axes[0, 0].set_xscale("function", functions=(np.sqrt, lambda x: x**2))
        axes[0, 0].set_yscale("function", functions=(np.sqrt, lambda x: x**2))
        axes[0, 0].set_xlim(positive_min, maxval)
        axes[0, 0].set_ylim(positive_min, maxval)
    axes[0, 0].set_title(fill(label, width=52), fontsize=11)
    axes[0, 0].legend()
    axes[0, 1].axhline(1.0, color="k", linestyle="--", label="ratio=1")
    axes[0, 1].set_title("Flux Ratio vs True")
    axes[0, 1].set_ylim(0.7, 1.3)
    axes[0, 1].set_xscale("function", functions=(np.sqrt, lambda x: x**2))
    axes[0, 1].set_xlim(max(float(np.nanmin(truth[truth > 0])), 1e-6), float(np.nanmax(truth)))
    axes[0, 1].legend()

    if error is not None:
        rel = error / truth
        axes[0, 1].scatter(truth, 1.0 + rel, s=5, alpha=0.35, color="orange", label="+/- 1 sigma")
        axes[0, 1].scatter(truth, 1.0 - rel, s=5, alpha=0.35, color="orange")
        if snr_values is not None:
            ax2 = axes[0, 1].twiny()
            ticks = [1, 3, 5, 10, 20, 50, 100]
            flux_ticks = []
            snr_values = np.asarray(snr_values, dtype=float)
            for snr in ticks:
                idx = int(np.nanargmin(np.abs(snr_values - snr)))
                flux_ticks.append(truth[idx])
            xlim = axes[0, 1].get_xlim()
            valid = [(x, s) for x, s in zip(flux_ticks, ticks) if xlim[0] <= x <= xlim[1]]
            if valid:
                xs, labels = zip(*valid)
                ax2.set_xlim([np.sqrt(xlim[0]), np.sqrt(xlim[1])])
                ax2.set_xticks([np.sqrt(x) for x in xs])
                ax2.set_xticklabels([str(s) for s in labels])
                ax2.set_xlabel(f"SNR (True Flux / {error_label})")

        pulls = (recovered - truth) / error
        bins = np.linspace(-5, 5, 31)
        groups = [
            ("All", np.ones(len(pulls), dtype=bool), "0.7"),
        ]
        if snr_values is not None:
            groups = [
                ("All", np.ones(len(pulls), dtype=bool), "0.7"),
                ("SNR < 20", snr_values < 20.0, "tab:orange"),
                ("SNR >= 20", snr_values >= 20.0, "tab:blue"),
            ]
        for name, mask, color in groups:
            vals = pulls[np.asarray(mask, dtype=bool)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            axes[1, 0].hist(
                vals, bins=bins, density=True, alpha=0.22 if name == "All" else 0.42,
                color=color, histtype="stepfilled", label=name,
            )
            if name != "All" and vals.size >= 8:
                mu = float(np.nanmedian(vals))
                sig = float(mad_std(vals, ignore_nan=True))
                x = np.linspace(-10, 10, 300)
                if sig > 0 and np.isfinite(sig):
                    amp = 1.0 / (sig * np.sqrt(2.0 * np.pi))
                    axes[1, 0].plot(
                        x, amp * np.exp(-0.5 * ((x - mu) / sig) ** 2),
                        color=color, linewidth=2,
                        label=f"{name} MAD fit\nmu={mu:.3f}, sigma={sig:.3f}",
                    )
        for sigma in (1, 3):
            axes[1, 0].axvline(sigma, color="gray", linestyle="--", alpha=0.5)
            axes[1, 0].axvline(-sigma, color="gray", linestyle="--", alpha=0.5)
            axes[1, 1].axhline(sigma, color="gray", linestyle="--", alpha=0.5)
            axes[1, 1].axhline(-sigma, color="gray", linestyle="--", alpha=0.5, label=f"+/-{sigma} sigma")
        axes[1, 0].axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.8)
        axes[1, 0].set_xlim(-10, 10)
        axes[1, 0].set_xlabel(f"(Recovered - True) / {error_label}")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title(f"Residuals / {error_label} Distribution")
        axes[1, 0].legend()

        if np.any(base_mask):
            axes[1, 1].scatter(recovered[base_mask], pulls[base_mask], s=20, alpha=0.4, label="Data")
        ps = point_source_mask & base_mask
        if np.any(ps):
            axes[1, 1].scatter(
                recovered[ps], pulls[ps], s=26, color="tab:green", alpha=0.8, label="Point source"
            )
        db = deblended_mask & ~point_source_mask
        if np.any(db):
            axes[1, 1].scatter(
                recovered[db], pulls[db], s=52, facecolors="none", edgecolors="tab:blue",
                linewidths=1.4, label="Deblended child",
            )
        dbps = deblended_mask & point_source_mask
        if np.any(dbps):
            axes[1, 1].scatter(
                recovered[dbps],
                pulls[dbps],
                s=52,
                facecolors="none",
                edgecolors="tab:green",
                linewidths=1.4,
                label="_Point source deblended child",
            )
        axes[1, 1].axhline(0, color="k", linestyle="--", label="zero residual")
        axes[1, 1].set_xscale("function", functions=(np.sqrt, lambda x: x**2))
        pos = recovered[np.isfinite(recovered) & (recovered > 0)]
        if pos.size:
            axes[1, 1].set_xlim(max(float(np.nanmin(pos)), 1e-6), float(np.nanmax(pos)))
        axes[1, 1].set_ylim(-10, 10)
        axes[1, 1].set_xlabel("Recovered Flux")
        axes[1, 1].set_ylabel(f"(Recovered - True) / {error_label}")
        axes[1, 1].set_title("Residuals vs Recovered Flux")
        axes[1, 1].legend()

    if caption:
        fig.text(0.01, 0.012, fill(caption, width=140), fontsize=8, ha="left", va="bottom")
        plt.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_saturated_catalog_repair(
    sci: np.ndarray,
    seg_before: np.ndarray,
    seg_after: np.ndarray,
    cat_before: Table,
    cat_after: Table,
    merge_log: Table,
    *,
    out_path: str | Path,
    n_sources: int = 4,
    half_size: int | None = None,
    pad_factor: float = 1.8,
    id_col: str = "id",
    x_col: str = "x",
    y_col: str = "y",
    select_ids: Sequence[int] | None = None,
    asinh_a: float = 0.1,
    sci_percentiles: tuple[float, float] = (1.0, 99.5),
) -> Path:
    """Optical before/after diagnostic for
    :func:`mophongo.catalog.repair_saturated_catalog`.

    Per source: 3 panels — science cutout, segmap-before with each child
    outlined in its own colour, segmap-after with merged parent outlined
    in cyan. Catalog centroids overlaid: dropped children = red ``x``,
    new parent = cyan ``+``.

    Parameters
    ----------
    sci
        Full science image.
    seg_before, seg_after
        Segmentation maps before and after the repair.
    cat_before, cat_after
        Catalogs before and after. Must share ``id_col, x_col, y_col``.
    merge_log
        Table returned by
        :func:`mophongo.catalog.repair_saturated_catalog`. Required
        columns: ``parent_id, xc, yc, n_children, children``.
    out_path
        Output PNG path.
    n_sources
        Number of merge_log rows to plot (ignored when ``select_ids``
        given). Picked by largest ``n_children``.
    half_size
        Cutout half-size in pixels. If ``None``, picked per source
        from the parent segment area.
    pad_factor
        Cutout scale relative to the parent segment equivalent radius.
    select_ids
        Explicit list of ``parent_id`` values to plot.
    asinh_a, sci_percentiles
        Image stretch parameters.

    Returns
    -------
    Path
        Path of the written figure.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(merge_log) == 0:
        raise ValueError("merge_log is empty; nothing to plot")

    if select_ids is not None:
        sel = np.isin(np.asarray(merge_log["parent_id"]), list(select_ids))
        rows = merge_log[sel]
    else:
        order = np.argsort(-np.asarray(merge_log["n_children"]))
        rows = merge_log[order[: int(n_sources)]]

    if len(rows) == 0:
        raise ValueError("no merge_log rows selected to plot")

    H, W = sci.shape
    finite = np.isfinite(sci) & (sci != 0)
    if finite.any():
        vmin, vmax = np.percentile(sci[finite], list(sci_percentiles))
    else:
        vmin, vmax = 0.0, 1.0
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=asinh_a))

    cat_before_ids = np.asarray(cat_before[id_col])
    cat_after_ids = np.asarray(cat_after[id_col])

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for i, row in enumerate(rows):
        parent_id = int(row["parent_id"])
        xc = float(row["xc"])
        yc = float(row["yc"])
        children = [int(t) for t in str(row["children"]).split(",") if t]

        if half_size is None:
            area = int(np.sum(seg_after == parent_id))
            r_eq = max(8.0, float(np.sqrt(max(area, 1) / np.pi)))
            hs = int(np.ceil(pad_factor * r_eq))
        else:
            hs = int(half_size)
        y0 = max(0, int(round(yc)) - hs)
        y1 = min(H, int(round(yc)) + hs + 1)
        x0 = max(0, int(round(xc)) - hs)
        x1 = min(W, int(round(xc)) + hs + 1)

        sci_cut = sci[y0:y1, x0:x1]
        seg_b_cut = seg_before[y0:y1, x0:x1]
        seg_a_cut = seg_after[y0:y1, x0:x1]
        extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)

        ax = axes[i, 0]
        ax.imshow(sci_cut, origin="upper", cmap="gray", norm=norm, extent=extent)
        ax.set_title(
            f"parent_id={parent_id}  "
            f"n_children={int(row['n_children'])}  "
            f"({hs * 2 + 1}x{hs * 2 + 1} px)"
        )
        ax.set_xlabel("x"); ax.set_ylabel("y")

        ax = axes[i, 1]
        ax.imshow(sci_cut, origin="upper", cmap="gray", norm=norm, extent=extent)
        rng = np.random.default_rng(parent_id)
        colors = rng.uniform(0.3, 1.0, size=(len(children), 3))
        for k, lbl in enumerate(children):
            mask = (seg_b_cut == lbl)
            if not mask.any():
                continue
            ax.contour(
                mask.astype(np.uint8), levels=[0.5], extent=extent,
                colors=[tuple(colors[k])], linewidths=1.0,
                origin="upper",
            )
        child_in_cat = np.isin(cat_before_ids, np.array(children))
        if child_in_cat.any():
            ax.scatter(
                np.asarray(cat_before[x_col])[child_in_cat],
                np.asarray(cat_before[y_col])[child_in_cat],
                marker="x", c="red", s=42, linewidths=1.4,
                label="dropped children",
            )
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("before: oversplit children")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

        ax = axes[i, 2]
        ax.imshow(sci_cut, origin="upper", cmap="gray", norm=norm, extent=extent)
        parent_mask = (seg_a_cut == parent_id)
        if parent_mask.any():
            ax.contour(
                parent_mask.astype(np.uint8), levels=[0.5], extent=extent,
                colors="cyan", linewidths=1.4,
                origin="upper",
            )
        is_parent = cat_after_ids == parent_id
        if is_parent.any():
            ax.scatter(
                np.asarray(cat_after[x_col])[is_parent],
                np.asarray(cat_after[y_col])[is_parent],
                marker="+", c="cyan", s=120, linewidths=1.8,
                label="parent",
            )
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("after: merged parent (hole closed)")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
