#!/usr/bin/env python
"""Make the all-field MINERVA "mother of all SEDs" figure.

Every valid broadband measurement is converted from catalog ``F_nu`` to a
quantity proportional to ``F_lambda`` and painted as a top hat across the
half-maximum interval of its filter.  Each galaxy is normalized by a
data-only interpolation at rest 5000 Angstrom, then galaxies are nanmeaned in
bins with ``Delta z = 0.05 * (1 + z)``.  The output compares rest-frame and
observed-frame wavelength and includes a separate contributor-count image.
A companion reconstruction joins pivots only inside each galaxy's connected
valid filter coverage and includes a display-only continuum-residual panel.

Run from the repository root::

    poetry run python examples/minerva/plot_uds_sed_stack.py

The current COSMOS, EGS, and UDS catalogs and their exact row-aligned EAzY
``zout.fits`` tables are required.  A field manifest selects one distinct
catalog per field so per-filter run configs do not duplicate galaxies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import AsinhNorm, LogNorm, TwoSlopeNorm
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d

from mophongo.sed_stack import (
    SEDStack,
    combine_sed_stacks,
    filter_interval_wavelength_edges,
    fnu_to_flam_proxy,
    normalize_at_rest_wavelength,
    redshift_bin_edges,
    stack_filter_seds,
    stack_interpolated_seds,
)


logger = logging.getLogger("minerva_sed_stack")
HERE = Path(__file__).resolve().parent
DEFAULT_FILTERS = HERE / "uds_sed_filters.ecsv"
DEFAULT_FIELDS = HERE / "minerva_sed_fields.json"
DEFAULT_OUTPUT = HERE / "sed_stack" / "minerva_mother_of_all_seds"


FEATURES_UM = (
    (0.1216, "Lyα"),
    (0.4000, "4000 Å break"),
    (0.500684, "[O III] λ5007"),
    (0.656281, "Hα"),
    (1.6000, "1.6 μm bump"),
    (3.3000, "3.3 μm PAH"),
    (7.7000, "7.7 μm PAH"),
)


@dataclass(frozen=True)
class CatalogSEDs:
    """Selected MINERVA catalog arrays before 5000-Angstrom normalization."""

    ids: np.ndarray
    redshift: np.ndarray
    has_spec: np.ndarray
    flux_fnu: np.ndarray
    error_fnu: np.ndarray
    valid: np.ndarray
    input_rows: int
    use_phot_rows: int
    spec_redshifts: int
    valid_per_band: np.ndarray


@dataclass(frozen=True)
class FieldSpec:
    """One distinct MINERVA field catalog and its expected filter set."""

    name: str
    release: str
    catalog: Path
    photoz: Path
    bands: tuple[str, ...]


@dataclass(frozen=True)
class FieldSEDs:
    """Normalized union-filter arrays and provenance for one field."""

    spec: FieldSpec
    values: np.ndarray
    valid: np.ndarray
    redshift: np.ndarray
    input_rows: int
    use_phot_rows: int
    z_limited_rows: int
    normalized_rows: int
    spec_redshifts: int
    spec_normalized: int
    valid_per_band: np.ndarray
    normalized_valid_per_band: np.ndarray


def _commented_json(path: Path) -> dict:
    """Read a Mophongo JSON config whose full-line comments start with ``#``."""

    clean = "\n".join(
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    return json.loads(clean)


def load_field_manifest(path: Path, filters: Table) -> list[FieldSpec]:
    """Load and validate one current catalog specification per science field."""

    data = _commented_json(path)
    entries = data.get("fields")
    if not isinstance(entries, list) or not entries:
        raise ValueError("field manifest must contain a non-empty 'fields' list")
    known_bands = {str(band) for band in filters["band"]}
    fields = []
    names = set()
    catalogs = set()
    for entry in entries:
        name = str(entry["name"]).upper()
        if name in names:
            raise ValueError(f"duplicate field name in manifest: {name}")
        catalog = Path(entry["catalog"]).expanduser().resolve()
        photoz = Path(entry.get("photoz") or resolve_photoz(catalog)).expanduser().resolve()
        bands = tuple(str(band) for band in entry["bands"])
        if not bands or len(set(bands)) != len(bands):
            raise ValueError(f"{name} bands must be non-empty and unique")
        unknown = set(bands) - known_bands
        if unknown:
            raise ValueError(f"{name} manifest has unknown bands: {sorted(unknown)}")
        catalog_key = str(catalog)
        if catalog_key in catalogs:
            raise ValueError(f"catalog is repeated across fields: {catalog}")
        if not catalog.is_file():
            raise FileNotFoundError(f"{name} catalog not found: {catalog}")
        if not photoz.is_file():
            raise FileNotFoundError(f"{name} photo-z table not found: {photoz}")
        names.add(name)
        catalogs.add(catalog_key)
        fields.append(
            FieldSpec(
                name=name,
                release=str(entry["release"]),
                catalog=catalog,
                photoz=photoz,
                bands=bands,
            )
        )
    return fields


def catalog_from_config(path: Path) -> Path:
    """Resolve the catalog path recorded in a run config."""

    data = _commented_json(path)
    catalog = Path(data["catalog"])
    if not catalog.is_absolute():
        # Mophongo historically resolves config paths against the process CWD.
        catalog = (Path.cwd() / catalog).resolve()
    return catalog


def resolve_photoz(catalog: Path, explicit: Path | None = None) -> Path:
    """Find the exact EAzY table associated with ``catalog``."""

    if explicit is not None:
        return explicit.resolve()
    stem = catalog.stem
    candidates = (
        catalog.parent / "EAzY" / f"{stem}.eazypy.zout.fits",
        catalog.parent / f"{stem}.eazypy.zout.fits",
        catalog.parent
        / "ACS+WEBB_chi-mean"
        / "EAzY"
        / f"{stem}.eazypy.zout.fits",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "the catalog has no photo-z column and its exact EAzY table was not "
        f"found; tried:\n  {tried}\nPass --photoz explicitly."
    )


def _validate_row_alignment(catalog_data, photoz_data) -> None:
    """Reject a redshift table that is not the exact catalog counterpart."""

    if len(catalog_data) != len(photoz_data):
        raise ValueError(
            "catalog/photo-z row counts differ: "
            f"{len(catalog_data)} != {len(photoz_data)}"
        )
    if not np.array_equal(catalog_data["id"], photoz_data["id"]):
        raise ValueError(
            "catalog/photo-z IDs are not row-aligned; do not ID-join a "
            "different MINERVA detection catalog"
        )
    dra = (
        np.asarray(catalog_data["ra"], dtype=float)
        - np.asarray(photoz_data["ra"], dtype=float)
    ) * np.cos(np.deg2rad(np.asarray(catalog_data["dec"], dtype=float)))
    ddec = (
        np.asarray(catalog_data["dec"], dtype=float)
        - np.asarray(photoz_data["dec"], dtype=float)
    )
    separation_arcsec = np.hypot(dra, ddec) * 3600.0
    finite = np.isfinite(separation_arcsec)
    if not np.any(finite) or np.nanpercentile(separation_arcsec[finite], 99) > 0.05:
        raise ValueError(
            "catalog/photo-z sky positions disagree; the redshift table is "
            "not the exact counterpart of this photometric catalog"
        )


def load_catalog_seds(
    catalog: Path,
    photoz: Path,
    filters: Table,
    *,
    z_max: float,
    photoz_column: str = "z_phot",
) -> CatalogSEDs:
    """Load quality-masked HST+JWST fluxes from one MINERVA catalog.

    Negative finite fluxes are retained.  MIRI measurements additionally
    reject the catalog's persistence, edge, and high-scale-factor flags.
    Spectroscopic redshifts replace photo-z values where available.
    """

    logger.info("catalog: %s", catalog)
    logger.info("photo-z: %s", photoz)
    with fits.open(catalog, memmap=True) as catalog_hdul, fits.open(
        photoz, memmap=True
    ) as photoz_hdul:
        catalog_data = catalog_hdul[1].data
        photoz_data = photoz_hdul[1].data
        _validate_row_alignment(catalog_data, photoz_data)
        catalog_names = set(catalog_data.names)
        photoz_names = set(photoz_data.names)
        if photoz_column not in photoz_names:
            raise KeyError(f"photo-z table has no {photoz_column!r} column")

        z_phot = np.asarray(photoz_data[photoz_column], dtype=float)
        z_spec = np.asarray(catalog_data["z_spec"], dtype=float)
        has_spec = np.isfinite(z_spec) & (z_spec > 0)
        redshift = np.where(has_spec, z_spec, z_phot)
        use_phot = np.asarray(catalog_data["use_phot"]) > 0
        base = (
            use_phot
            & np.isfinite(redshift)
            & (redshift > 0)
            & (redshift <= z_max)
        )
        rows = np.flatnonzero(base)
        n_source = rows.size
        n_band = len(filters)
        flux = np.empty((n_source, n_band), dtype=np.float32)
        error = np.empty_like(flux)
        valid = np.zeros_like(flux, dtype=bool)

        high_scale = (
            np.asarray(catalog_data["FLAG_HIGH_SCLFACT"], dtype=bool)[rows]
            if "FLAG_HIGH_SCLFACT" in catalog_names
            else np.zeros(n_source, dtype=bool)
        )
        for band_index, filter_row in enumerate(filters):
            band = str(filter_row["band"])
            flux_column = f"f_{band}"
            error_column = f"e_{band}"
            missing = {flux_column, error_column} - catalog_names
            if missing:
                raise KeyError(f"catalog is missing filter columns {sorted(missing)}")
            band_flux = np.asarray(catalog_data[flux_column][rows], dtype=np.float32)
            band_error = np.asarray(catalog_data[error_column][rows], dtype=np.float32)
            band_valid = (
                np.isfinite(band_flux)
                & np.isfinite(band_error)
                & (band_error > 0)
            )
            coverage_column = f"w_{band}"
            if coverage_column in catalog_names:
                coverage = np.asarray(catalog_data[coverage_column][rows], dtype=float)
                band_valid &= np.isfinite(coverage) & (coverage > 0)

            if str(filter_row["instrument"]) == "JWST_MIRI":
                upper = band.upper()
                for flag_column in (
                    f"FLAG_PERSISTENCE_{upper}",
                    f"FLAG_EDGE_{upper}",
                ):
                    if flag_column in catalog_names:
                        band_valid &= ~np.asarray(
                            catalog_data[flag_column][rows], dtype=bool
                        )
                band_valid &= ~high_scale

            flux[:, band_index] = band_flux
            error[:, band_index] = band_error
            valid[:, band_index] = band_valid

        return CatalogSEDs(
            ids=np.asarray(catalog_data["id"][rows], dtype=np.int64),
            redshift=np.asarray(redshift[rows], dtype=np.float64),
            has_spec=np.asarray(has_spec[rows], dtype=bool),
            flux_fnu=flux,
            error_fnu=error,
            valid=valid,
            input_rows=len(catalog_data),
            use_phot_rows=int(np.count_nonzero(use_phot)),
            spec_redshifts=int(np.count_nonzero(has_spec[rows])),
            valid_per_band=np.sum(valid, axis=0, dtype=np.int64),
        )


def normalize_field_seds(
    spec: FieldSpec,
    union_filters: Table,
    *,
    z_max: float,
    photoz_column: str,
    rest_wavelength: float,
    min_snr: float,
    n_nearest: int,
) -> FieldSEDs:
    """Load, normalize, and pad one field onto the union-filter axis."""

    band_to_union = {
        str(band): index for index, band in enumerate(union_filters["band"])
    }
    local_indices = np.sort(
        np.array([band_to_union[band] for band in spec.bands], dtype=int)
    )
    local_filters = union_filters[local_indices]
    sed = load_catalog_seds(
        spec.catalog,
        spec.photoz,
        local_filters,
        z_max=z_max,
        photoz_column=photoz_column,
    )
    pivot = np.asarray(local_filters["pivot_angstrom"], dtype=float)
    flux_flam = fnu_to_flam_proxy(
        sed.flux_fnu, pivot, reference_wavelength=rest_wavelength
    )
    error_flam = fnu_to_flam_proxy(
        sed.error_fnu, pivot, reference_wavelength=rest_wavelength
    )
    normalization = normalize_at_rest_wavelength(
        flux_flam,
        error_flam,
        sed.valid,
        pivot,
        sed.redshift,
        rest_wavelength=rest_wavelength,
        min_snr=min_snr,
        n_nearest=n_nearest,
    )
    selected = normalization.selected
    n_selected = int(np.count_nonzero(selected))
    values = np.full((n_selected, len(union_filters)), np.nan, dtype=np.float32)
    valid = np.zeros(values.shape, dtype=bool)
    values[:, local_indices] = normalization.values[selected]
    valid[:, local_indices] = normalization.valid[selected]
    valid_per_band = np.zeros(len(union_filters), dtype=np.int64)
    normalized_valid_per_band = np.zeros(len(union_filters), dtype=np.int64)
    valid_per_band[local_indices] = sed.valid_per_band
    normalized_valid_per_band[local_indices] = np.sum(
        normalization.valid[selected], axis=0, dtype=np.int64
    )
    return FieldSEDs(
        spec=spec,
        values=values,
        valid=valid,
        redshift=sed.redshift[selected],
        input_rows=sed.input_rows,
        use_phot_rows=sed.use_phot_rows,
        z_limited_rows=len(sed.ids),
        normalized_rows=n_selected,
        spec_redshifts=sed.spec_redshifts,
        spec_normalized=int(np.count_nonzero(sed.has_spec[selected])),
        valid_per_band=valid_per_band,
        normalized_valid_per_band=normalized_valid_per_band,
    )


def _wave_table(edges_angstrom: np.ndarray) -> Table:
    """Wavelength-bin table in microns for a stack image axis."""

    edges_um = np.asarray(edges_angstrom, dtype=float) / 1.0e4
    return Table(
        {
            "wave_lo_um": edges_um[:-1],
            "wave_hi_um": edges_um[1:],
            "wave_um": np.sqrt(edges_um[:-1] * edges_um[1:]),
        }
    )


def linear_wavelength_edges(
    minimum_micron: float,
    maximum_micron: float,
    bin_width_angstrom: float,
) -> np.ndarray:
    """Build fixed-width wavelength-bin edges in Angstrom."""

    if (
        not np.isfinite(minimum_micron)
        or not np.isfinite(maximum_micron)
        or minimum_micron <= 0
        or maximum_micron <= minimum_micron
    ):
        raise ValueError("wavelength limits must be finite, positive, and ordered")
    if not np.isfinite(bin_width_angstrom) or bin_width_angstrom <= 0:
        raise ValueError("wavelength bin width must be finite and positive")
    lower = minimum_micron * 1.0e4
    upper = maximum_micron * 1.0e4
    n_bin = int(np.ceil((upper - lower) / bin_width_angstrom))
    return lower + bin_width_angstrom * np.arange(n_bin + 1, dtype=float)


def interpolated_wavelength_edges(
    filter_blue: np.ndarray,
    filter_pivot: np.ndarray,
    filter_red: np.ndarray,
    minimum_micron: float,
    maximum_micron: float,
    maximum_log10_step: float,
) -> np.ndarray:
    """Build a fine log-wavelength evaluation grid with physical boundaries."""

    lower = minimum_micron * 1.0e4
    upper = maximum_micron * 1.0e4
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0
        or upper <= lower
    ):
        raise ValueError("wavelength limits must be finite, positive, and ordered")
    if not np.isfinite(maximum_log10_step) or maximum_log10_step <= 0:
        raise ValueError("maximum log-wavelength step must be finite and positive")
    span = np.log10(upper / lower)
    n_bin = int(np.ceil(span / maximum_log10_step))
    base = np.geomspace(lower, upper, n_bin + 1)
    physical = np.concatenate(
        (
            np.asarray(filter_blue, dtype=float),
            np.asarray(filter_pivot, dtype=float),
            np.asarray(filter_red, dtype=float),
        )
    )
    physical = physical[(physical > lower) & (physical < upper)]
    if np.any(~np.isfinite(physical)) or np.any(physical <= 0):
        raise ValueError("filter wavelengths must be finite and positive")
    return np.unique(np.concatenate((base, physical)))


def write_stack_fits(
    path: Path,
    rest: SEDStack,
    observed: SEDStack,
    rest_interpolated: SEDStack,
    observed_interpolated: SEDStack,
    field_rest: list[SEDStack],
    field_observed: list[SEDStack],
    field_seds: list[FieldSEDs],
    filters: Table,
    *,
    n_input: int,
    n_use_phot: int,
    n_normalized: int,
    n_spec: int,
    norm_wavelength: float,
    norm_snr: float,
    norm_band_count: int,
    wavelength_bin_width: float,
    interpolation_log_step: float,
    interpolation_coincident_fraction: float,
    redshift_step: float,
    valid_per_band: np.ndarray,
    normalized_valid_per_band: np.ndarray,
) -> None:
    """Write stack means, counts, axes, filters, and provenance to FITS."""

    header = fits.Header()
    header["NINPUT"] = (n_input, "rows in the photometric catalog")
    header["NUSEPHOT"] = (n_use_phot, "catalog rows with use_phot > 0")
    header["NNORM"] = (n_normalized, "galaxies passing 5000-A normalization")
    header["NSPEC"] = (n_spec, "normalized sample using z_spec")
    header["NFIELD"] = (len(field_seds), "distinct MINERVA fields")
    header["NFILTER"] = (len(filters), "filters in union broadband SED")
    header["NORMWAVE"] = (norm_wavelength, "rest normalization wavelength [A]")
    header["NORMSNR"] = (norm_snr, "minimum normalization S/N")
    header["NORMBAND"] = (norm_band_count, "maximum bands in local normalization")
    header["DWAVE"] = (wavelength_bin_width, "rest wavelength bin width [Angstrom]")
    header["OBSGRID"] = ("FILTER", "observed cells use filter interval boundaries")
    header["INTPMOD"] = ("LOGLIN", "linear F_lambda versus log wavelength")
    header["INTPDLOG"] = (
        interpolation_log_step,
        "maximum observed interpolation grid step in log10 wavelength",
    )
    header["COINFR"] = (
        interpolation_coincident_fraction,
        "coincident-pivot tolerance / narrower filter width",
    )
    header["DZ1PZ"] = (redshift_step, "Delta z / (1 + z) at bin edge")
    header["COMMENT"] = "Means are unweighted nanmeans in linear normalized F_lambda."
    header["COMMENT"] = "Negative valid fluxes are retained; uncovered pixels are NaN."

    primary = fits.PrimaryHDU(header=header)
    image_hdus = []
    for name, stack in (("REST", rest), ("OBSERVED", observed)):
        mean_hdu = fits.ImageHDU(stack.mean.astype(np.float32), name=f"{name}_MEAN")
        mean_hdu.header["BUNIT"] = "F_lambda / F_lambda(5000A)"
        mean_hdu.header["FRAME"] = name.lower()
        count_hdu = fits.ImageHDU(stack.count.astype(np.int32), name=f"{name}_COUNT")
        count_hdu.header["BUNIT"] = "galaxies"
        image_hdus.extend((mean_hdu, count_hdu))
    for name, stack in (
        ("REST_INTERP", rest_interpolated),
        ("OBS_INTERP", observed_interpolated),
    ):
        mean_hdu = fits.ImageHDU(stack.mean.astype(np.float32), name=f"{name}_MEAN")
        mean_hdu.header["BUNIT"] = "F_lambda / F_lambda(5000A)"
        mean_hdu.header["MODEL"] = "piecewise linear in log wavelength"
        count_hdu = fits.ImageHDU(stack.count.astype(np.int32), name=f"{name}_COUNT")
        count_hdu.header["BUNIT"] = "galaxies"
        image_hdus.extend((mean_hdu, count_hdu))

    z_edges = rest.redshift_edges
    z_table = Table(
        {
            "z_lo": z_edges[:-1],
            "z_hi": z_edges[1:],
            "z_mid": np.expm1(
                0.5 * (np.log1p(z_edges[:-1]) + np.log1p(z_edges[1:]))
            ),
            "log1p_z_lo": np.log1p(z_edges[:-1]),
            "log1p_z_hi": np.log1p(z_edges[1:]),
            "n_galaxy": rest.galaxies_per_bin,
        }
    )
    filter_output = filters.copy()
    filter_output.meta.clear()
    filter_output["n_valid_input"] = np.asarray(valid_per_band, dtype=np.int64)
    filter_output["n_valid_normalized"] = np.asarray(
        normalized_valid_per_band, dtype=np.int64
    )
    field_table = Table(
        {
            "field": [field.spec.name for field in field_seds],
            "release": [field.spec.release for field in field_seds],
            "catalog": [str(field.spec.catalog) for field in field_seds],
            "photoz": [str(field.spec.photoz) for field in field_seds],
            "n_input": [field.input_rows for field in field_seds],
            "n_use_phot": [field.use_phot_rows for field in field_seds],
            "n_z_limited": [field.z_limited_rows for field in field_seds],
            "n_normalized": [field.normalized_rows for field in field_seds],
            "n_spec": [field.spec_normalized for field in field_seds],
            "n_filters": [len(field.spec.bands) for field in field_seds],
        }
    )
    field_filter_rows = []
    for field in field_seds:
        present = set(field.spec.bands)
        for filter_index, band in enumerate(filters["band"]):
            field_filter_rows.append(
                (
                    field.spec.name,
                    str(band),
                    str(band) in present,
                    field.valid_per_band[filter_index],
                    field.normalized_valid_per_band[filter_index],
                )
            )
    field_filter_table = Table(
        rows=field_filter_rows,
        names=(
            "field",
            "band",
            "present",
            "n_valid_input",
            "n_valid_normalized",
        ),
    )
    rest_field_mean = np.stack([stack.mean for stack in field_rest])
    rest_field_count = np.stack([stack.count for stack in field_rest])
    observed_field_mean = np.stack([stack.mean for stack in field_observed])
    observed_field_count = np.stack([stack.count for stack in field_observed])
    field_z_count = np.stack([stack.galaxies_per_bin for stack in field_rest])
    field_images = [
        fits.ImageHDU(rest_field_mean.astype(np.float32), name="FIELD_REST_MEAN"),
        fits.ImageHDU(rest_field_count.astype(np.int32), name="FIELD_REST_COUNT"),
        fits.ImageHDU(
            observed_field_mean.astype(np.float32), name="FIELD_OBSERVED_MEAN"
        ),
        fits.ImageHDU(
            observed_field_count.astype(np.int32), name="FIELD_OBSERVED_COUNT"
        ),
        fits.ImageHDU(field_z_count.astype(np.int32), name="FIELD_Z_COUNT"),
    ]
    for hdu in field_images[:4]:
        hdu.header["AXIS1"] = "wavelength"
        hdu.header["AXIS2"] = "redshift"
        hdu.header["AXIS3"] = "field; see FIELDS table"
    hdus = fits.HDUList(
        [
            primary,
            *image_hdus,
            *field_images,
            fits.BinTableHDU(z_table, name="REDSHIFT_BINS"),
            fits.BinTableHDU(_wave_table(rest.wavelength_edges), name="REST_WAVE"),
            fits.BinTableHDU(
                _wave_table(observed.wavelength_edges), name="OBSERVED_WAVE"
            ),
            fits.BinTableHDU(
                _wave_table(observed_interpolated.wavelength_edges),
                name="OBS_INTERP_WAVE",
            ),
            fits.BinTableHDU(filter_output, name="FILTERS"),
            fits.BinTableHDU(field_table, name="FIELDS"),
            fits.BinTableHDU(field_filter_table, name="FIELD_FILTERS"),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus.writeto(path, overwrite=True)


def _format_wave(value: float) -> str:
    """Compact micron tick label."""

    return f"{value:g}"


def _style_stack_axis(
    ax,
    wave_edges_angstrom: np.ndarray,
    redshift_edges: np.ndarray,
    *,
    xlabel: str,
    display_z_max: float | None = None,
) -> None:
    """Apply shared logarithmic-coordinate ticks to an image axis."""

    wave_edges_um = wave_edges_angstrom / 1.0e4
    wave_ticks = np.array([0.1, 0.2, 0.4, 0.7, 1, 2, 4, 7, 10, 20])
    wave_ticks = wave_ticks[
        (wave_ticks >= wave_edges_um[0]) & (wave_ticks <= wave_edges_um[-1])
    ]
    z_top = redshift_edges[-1] if display_z_max is None else display_z_max
    z_ticks = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20], dtype=float)
    z_ticks = z_ticks[z_ticks <= z_top]
    ax.set_xticks(np.log10(wave_ticks), [_format_wave(value) for value in wave_ticks])
    ax.set_yticks(np.log1p(z_ticks), [_format_wave(value) for value in z_ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("redshift")
    ax.set_ylim(np.log1p(redshift_edges[0]), np.log1p(z_top))
    ax.grid(axis="y", color="white", alpha=0.14, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _add_feature_guides(ax, frame: str, redshift_edges: np.ndarray) -> None:
    """Add broad rest-feature guides in the style of a stacked-spectrum plot."""

    color = "#35c3ca"
    if frame == "rest":
        blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        xmin, xmax = ax.get_xlim()
        for wavelength, label in FEATURES_UM:
            x = np.log10(wavelength)
            if xmin <= x <= xmax:
                ax.axvline(x, color=color, alpha=0.38, linewidth=0.75)
                ax.text(
                    x,
                    0.985,
                    label,
                    transform=blend,
                    rotation=90,
                    va="top",
                    ha="right",
                    color=color,
                    fontsize=7,
                )
        return

    y = np.linspace(np.log1p(redshift_edges[0]), np.log1p(redshift_edges[-1]), 512)
    one_plus_z = np.exp(y)
    xmin, xmax = ax.get_xlim()
    for wavelength, label in FEATURES_UM:
        x = np.log10(wavelength * one_plus_z)
        inside = (x >= xmin) & (x <= xmax)
        if not np.any(inside):
            continue
        ax.plot(x[inside], y[inside], color=color, alpha=0.38, linewidth=0.75)
        indices = np.flatnonzero(inside)
        label_index = indices[int(0.82 * (indices.size - 1))]
        ax.text(
            x[label_index],
            y[label_index],
            label,
            color=color,
            fontsize=7,
            rotation=27,
            ha="left",
            va="bottom",
            clip_on=True,
        )


def _draw_stack_mesh(ax, stack: SEDStack, data, *, cmap, norm):
    """Draw a stack with exact bin edges on logarithmic display coordinates."""

    return ax.pcolormesh(
        np.log10(stack.wavelength_edges / 1.0e4),
        np.log1p(stack.redshift_edges),
        data,
        shading="flat",
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )


def _redshift_interpolated_display(
    stack: SEDStack,
    data: np.ndarray,
    factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate finite redshift runs for rendering without crossing gaps."""

    if factor < 1:
        raise ValueError("display interpolation factor must be at least one")
    if factor == 1:
        return np.asarray(data), np.log1p(stack.redshift_edges)
    source = np.asarray(data, dtype=float)
    old_edges = np.log1p(stack.redshift_edges)
    old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    new_edges = np.linspace(old_edges[0], old_edges[-1], source.shape[0] * factor + 1)
    new_centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    display = np.full((new_centers.size, source.shape[1]), np.nan, dtype=np.float32)
    for column in range(source.shape[1]):
        finite = np.flatnonzero(np.isfinite(source[:, column]))
        if not finite.size:
            continue
        for run in np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1):
            target = (new_centers >= old_edges[run[0]]) & (
                new_centers <= old_edges[run[-1] + 1]
            )
            if run.size == 1:
                display[target, column] = source[run[0], column]
            else:
                display[target, column] = np.interp(
                    new_centers[target],
                    old_centers[run],
                    source[run, column],
                    left=source[run[0], column],
                    right=source[run[-1], column],
                )
    return display, new_edges


def _draw_interpolated_display(
    ax,
    stack: SEDStack,
    data: np.ndarray,
    *,
    cmap,
    norm,
    redshift_factor: int = 4,
):
    """Draw a reconstructed stack with masked display-only redshift smoothing."""

    display, log_redshift_edges = _redshift_interpolated_display(
        stack, data, factor=redshift_factor
    )
    return ax.pcolormesh(
        np.log10(stack.wavelength_edges / 1.0e4),
        log_redshift_edges,
        display,
        shading="flat",
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )


def _continuum_residual(
    stack: SEDStack,
    sigma_log10_wavelength: float,
) -> tuple[np.ndarray, float]:
    """Return an asinh-domain continuum residual within finite coverage runs."""

    if not np.isfinite(sigma_log10_wavelength) or sigma_log10_wavelength <= 0:
        raise ValueError("continuum width must be finite and positive")
    data = np.asarray(stack.mean, dtype=float)
    finite_values = data[np.isfinite(data)]
    if not finite_values.size:
        raise ValueError("cannot estimate a continuum from an empty stack")
    absolute = np.abs(finite_values)
    positive_scale = absolute[absolute > 0]
    softening = (
        max(0.02, 0.25 * float(np.percentile(positive_scale, 25)))
        if positive_scale.size
        else 0.02
    )
    transformed = np.full(data.shape, np.nan, dtype=float)
    finite = np.isfinite(data)
    transformed[finite] = np.arcsinh(data[finite] / softening)
    residual = np.full(data.shape, np.nan, dtype=np.float32)
    log_wave = np.log10(
        np.sqrt(stack.wavelength_edges[:-1] * stack.wavelength_edges[1:])
    )
    for row in range(data.shape[0]):
        indices = np.flatnonzero(finite[row])
        if not indices.size:
            continue
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if run.size == 1:
                residual[row, run] = 0.0
                continue
            x = log_wave[run]
            y = transformed[row, run]
            step = min(float(np.median(np.diff(x))), sigma_log10_wavelength / 8.0)
            n_uniform = max(2, int(np.ceil((x[-1] - x[0]) / step)) + 1)
            uniform_x = np.linspace(x[0], x[-1], n_uniform)
            uniform_y = np.interp(uniform_x, x, y)
            sigma_pixels = sigma_log10_wavelength / (uniform_x[1] - uniform_x[0])
            continuum = gaussian_filter1d(
                uniform_y, sigma=sigma_pixels, mode="nearest", truncate=4.0
            )
            residual[row, run] = np.asarray(
                y - np.interp(x, uniform_x, continuum), dtype=np.float32
            )
    return residual, softening


def plot_interpolated_stacks(
    path_png: Path,
    path_pdf: Path,
    rest: SEDStack,
    observed: SEDStack,
    *,
    n_galaxy: int,
    field_names: list[str],
    field_observed: list[SEDStack],
    continuum_sigma_dex: float,
    redshift_display_factor: int,
    features: bool,
    dpi: int,
) -> None:
    """Render smooth pivot-cell reconstructions plus a feature-enhanced view."""

    finite_values = np.concatenate(
        (rest.mean[np.isfinite(rest.mean)], observed.mean[np.isfinite(observed.mean)])
    )
    if not finite_values.size:
        raise ValueError("the interpolated stacks contain no finite pixels")
    vmin, vmax = np.percentile(finite_values, [1.0, 99.0])
    if vmax <= vmin:
        vmax = vmin + 1.0
    flux_norm = AsinhNorm(
        linear_width=max(0.04, 0.06 * (vmax - vmin)),
        vmin=vmin,
        vmax=vmax,
    )
    flux_cmap = plt.colormaps["gray"].copy()
    flux_cmap.set_bad("#101318")
    field_residual_stacks = []
    softening_values = []
    for field_stack in field_observed:
        field_residual, field_softening = _continuum_residual(
            field_stack, continuum_sigma_dex
        )
        field_residual_stacks.append(
            SEDStack(
                mean=field_residual,
                count=field_stack.count,
                wavelength_edges=field_stack.wavelength_edges,
                redshift_edges=field_stack.redshift_edges,
                galaxies_per_bin=field_stack.galaxies_per_bin,
            )
        )
        softening_values.append(field_softening)
    residual_stack = combine_sed_stacks(
        field_residual_stacks, minimum_count=0, minimum_fraction=0.0
    )
    residual = residual_stack.mean.copy()
    residual_count_floor = 100
    residual[residual_stack.count < residual_count_floor] = np.nan
    softening = float(np.median(softening_values))
    well_sampled = observed.galaxies_per_bin >= 100
    finite_residual = np.abs(residual[np.isfinite(residual)])
    residual_limit = (
        float(np.percentile(finite_residual, 98.0)) if finite_residual.size else 1.0
    )
    residual_limit = max(residual_limit, 1.0e-4)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    residual_cmap = plt.colormaps["RdBu_r"].copy()
    residual_cmap.set_bad("#d9d9d9")

    populated_rows = np.any(np.isfinite(rest.mean), axis=1) | np.any(
        np.isfinite(observed.mean), axis=1
    )
    populated = np.flatnonzero(populated_rows)
    if not populated.size:
        raise ValueError("the interpolated stacks have no populated redshift bins")
    display_edge_index = min(populated[-1] + 2, len(rest.redshift_edges) - 1)
    display_z_max = float(rest.redshift_edges[display_edge_index])
    display_redshift_edges = np.array([rest.redshift_edges[0], display_z_max])

    fig = plt.figure(figsize=(15.5, 12.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.08))
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    for ax, stack, frame, title in (
        (axes[0], rest, "rest", "Rest-frame piecewise-linear broadband reconstruction"),
        (
            axes[1],
            observed,
            "observed",
            "Observed-frame piecewise-linear broadband reconstruction",
        ),
    ):
        flux_image = _draw_interpolated_display(
            ax,
            stack,
            stack.mean,
            cmap=flux_cmap,
            norm=flux_norm,
            redshift_factor=redshift_display_factor,
        )
        _style_stack_axis(
            ax,
            stack.wavelength_edges,
            stack.redshift_edges,
            xlabel=f"{frame}-frame wavelength [micron]",
            display_z_max=display_z_max,
        )
        ax.set_title(title, loc="left", fontweight="medium")
        if features:
            _add_feature_guides(ax, frame, display_redshift_edges)
    flux_bar = fig.colorbar(
        flux_image, ax=axes, location="right", shrink=0.9, pad=0.012
    )
    flux_bar.set_label(
        r"mean $F_\lambda/F_\lambda(\mathrm{rest}\ 5000\,\AA)$ [asinh stretch]"
    )

    residual_ax = fig.add_subplot(grid[1, :])
    well_sampled_bins = np.flatnonzero(
        well_sampled & np.any(np.isfinite(residual), axis=1)
    )
    residual_z_max = (
        float(observed.redshift_edges[well_sampled_bins[-1] + 1])
        if well_sampled_bins.size
        else display_z_max
    )
    residual_redshift_edges = np.array(
        [observed.redshift_edges[0], residual_z_max]
    )
    residual_image = _draw_interpolated_display(
        residual_ax,
        observed,
        residual,
        cmap=residual_cmap,
        norm=residual_norm,
        redshift_factor=redshift_display_factor,
    )
    _style_stack_axis(
        residual_ax,
        observed.wavelength_edges,
        observed.redshift_edges,
        xlabel="observed-frame wavelength [micron]",
        display_z_max=residual_z_max,
    )
    residual_ax.set_title(
        "Observed frame — continuum-removed contrast (display only)",
        loc="left",
        fontweight="medium",
    )
    if features:
        _add_feature_guides(residual_ax, "observed", residual_redshift_edges)
    residual_bar = fig.colorbar(
        residual_image, ax=residual_ax, location="right", shrink=0.92, pad=0.012
    )
    residual_bar.set_label(
        f"asinh residual after {continuum_sigma_dex:g}-dex continuum smooth "
        f"(median softening={softening:.3g}; count ≥ {residual_count_floor})"
    )
    fig.suptitle(
        f"MINERVA {' + '.join(field_names)} — {n_galaxy:,} broadband SEDs",
        fontsize=16,
        fontweight="medium",
    )
    fig.supxlabel(
        "linear Fλ vs log wavelength between connected valid filter pivots; "
        "true half-maximum coverage gaps stay empty  |  "
        f"{redshift_display_factor}× redshift interpolation is rendering-only",
        fontsize=8.5,
        color="0.25",
    )
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, facecolor="white")
    fig.savefig(path_pdf, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_stacks(
    path_png: Path,
    path_pdf: Path,
    rest: SEDStack,
    observed: SEDStack,
    *,
    n_galaxy: int,
    n_filter: int,
    field_names: list[str],
    redshift_step: float,
    features: bool,
    dpi: int,
) -> None:
    """Render the rest/observed stack comparison with one shared stretch."""

    finite_values = np.concatenate(
        (rest.mean[np.isfinite(rest.mean)], observed.mean[np.isfinite(observed.mean)])
    )
    if not finite_values.size:
        raise ValueError("the stacks contain no finite pixels")
    vmin, vmax = np.percentile(finite_values, [1.0, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0
    linear_width = max(0.05, 0.08 * (vmax - vmin))
    norm = AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["gray"].copy()
    cmap.set_bad("#101318")

    populated_rows = np.any(np.isfinite(rest.mean), axis=1) | np.any(
        np.isfinite(observed.mean), axis=1
    )
    populated_indices = np.flatnonzero(populated_rows)
    if not populated_indices.size:
        raise ValueError("the stacks contain no populated redshift bins")
    display_edge_index = min(populated_indices[-1] + 2, len(rest.redshift_edges) - 1)
    display_z_max = float(rest.redshift_edges[display_edge_index])
    display_redshift_edges = np.array(
        [rest.redshift_edges[0], display_z_max], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.4), layout="constrained")
    for ax, stack, frame, title in (
        (axes[0], rest, "rest", "Rest-frame wavelength"),
        (axes[1], observed, "observed", "Observed wavelength"),
    ):
        image = _draw_stack_mesh(ax, stack, stack.mean, cmap=cmap, norm=norm)
        _style_stack_axis(
            ax,
            stack.wavelength_edges,
            stack.redshift_edges,
            xlabel=f"{frame}-frame wavelength [micron]",
            display_z_max=display_z_max,
        )
        ax.set_title(title, loc="left", fontweight="medium")
        if features:
            _add_feature_guides(ax, frame, display_redshift_edges)
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.88, pad=0.015)
    colorbar.set_label(
        r"mean  $F_\lambda/F_\lambda(\mathrm{rest}\ 5000\,\AA)$  [asinh stretch]"
    )
    fig.suptitle(
        f"MINERVA {' + '.join(field_names)} — "
        f"{n_galaxy:,} broadband SEDs stacked by redshift",
        fontsize=16,
        fontweight="medium",
    )
    fig.supxlabel(
        f"{n_filter} HST + JWST filters  |  "
        f"rest: fixed {np.median(np.diff(rest.wavelength_edges)):g} Å pixels  |  "
        "observed: filter-width cells  |  "
        f"Δz = {redshift_step:g}(1+z)  |  unweighted nanmean in linear Fλ",
        fontsize=8.5,
        color="0.25",
    )
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, facecolor="white")
    fig.savefig(path_pdf, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_coverage(
    path: Path,
    rest: SEDStack,
    observed: SEDStack,
    *,
    field_names: list[str],
    dpi: int,
) -> None:
    """Render per-pixel contributing-galaxy counts on a logarithmic scale."""

    max_count = max(int(np.max(rest.count)), int(np.max(observed.count)))
    norm = LogNorm(vmin=1, vmax=max(2, max_count))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#f3efe4")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), layout="constrained")
    for ax, stack, frame, title in (
        (axes[0], rest, "rest", "Rest-frame coverage"),
        (axes[1], observed, "observed", "Observed-frame coverage"),
    ):
        display = np.ma.masked_less_equal(stack.count, 0)
        image = _draw_stack_mesh(ax, stack, display, cmap=cmap, norm=norm)
        _style_stack_axis(
            ax,
            stack.wavelength_edges,
            stack.redshift_edges,
            xlabel=f"{frame}-frame wavelength [micron]",
        )
        ax.set_title(title, loc="left", fontweight="medium")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.88, pad=0.015)
    colorbar.set_label("contributing galaxies per pixel")
    fig.suptitle(
        f"MINERVA {' + '.join(field_names)} broadband-SED stack: sampling depth",
        fontsize=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_wide_stack(
    path: Path,
    stack: SEDStack,
    *,
    frame: str,
    n_galaxy: int,
    field_names: list[str],
    features: bool,
    dpi: int,
) -> None:
    """Render a wide, full-resolution-oriented stack in one wavelength frame."""

    if frame not in {"rest", "observed"}:
        raise ValueError("frame must be 'rest' or 'observed'")
    finite_values = stack.mean[np.isfinite(stack.mean)]
    if not finite_values.size:
        raise ValueError(f"the {frame}-frame stack contains no finite pixels")
    vmin, vmax = np.percentile(finite_values, [1.0, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = AsinhNorm(
        linear_width=max(0.05, 0.08 * (vmax - vmin)),
        vmin=vmin,
        vmax=vmax,
    )
    cmap = plt.colormaps["gray"].copy()
    cmap.set_bad("#101318")
    populated = np.flatnonzero(np.any(np.isfinite(stack.mean), axis=1))
    if not populated.size:
        raise ValueError(f"the {frame}-frame stack has no populated redshift bins")
    display_edge_index = min(populated[-1] + 2, len(stack.redshift_edges) - 1)
    display_z_max = float(stack.redshift_edges[display_edge_index])

    fig, ax = plt.subplots(figsize=(28, 8.5), layout="constrained")
    image = _draw_stack_mesh(ax, stack, stack.mean, cmap=cmap, norm=norm)
    _style_stack_axis(
        ax,
        stack.wavelength_edges,
        stack.redshift_edges,
        xlabel=f"{frame}-frame wavelength [micron]",
        display_z_max=display_z_max,
    )
    if features:
        _add_feature_guides(
            ax,
            frame,
            np.array([stack.redshift_edges[0], display_z_max]),
        )
    colorbar = fig.colorbar(image, ax=ax, location="right", shrink=0.9, pad=0.01)
    colorbar.set_label(
        r"mean  $F_\lambda/F_\lambda(\mathrm{rest}\ 5000\,\AA)$  [asinh stretch]"
    )
    pixel_description = (
        f"{np.median(np.diff(stack.wavelength_edges)):g} Å pixels"
        if frame == "rest"
        else "filter-width pixels (half-maximum intervals)"
    )
    ax.set_title(
        f"MINERVA {' + '.join(field_names)} — "
        f"{n_galaxy:,} broadband SEDs in {frame} wavelength — "
        f"{pixel_description}",
        loc="left",
        fontsize=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--filters", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photoz-column", default="z_phot")
    parser.add_argument("--z-max", type=float, default=20.0)
    parser.add_argument("--redshift-step", type=float, default=0.05)
    parser.add_argument("--norm-rest-angstrom", type=float, default=5000.0)
    parser.add_argument("--norm-snr-min", type=float, default=5.0)
    parser.add_argument("--norm-nearest-bands", type=int, default=3)
    parser.add_argument("--wavelength-bin-angstrom", type=float, default=100.0)
    parser.add_argument("--interpolation-observed-dlog10", type=float, default=0.0025)
    parser.add_argument("--interpolation-coincident-fraction", type=float, default=0.05)
    parser.add_argument("--continuum-sigma-dex", type=float, default=0.12)
    parser.add_argument("--redshift-display-factor", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--minimum-count", type=int, default=5)
    parser.add_argument("--minimum-fraction", type=float, default=0.01)
    parser.add_argument("--rest-wave-min-um", type=float, default=0.08)
    parser.add_argument("--rest-wave-max-um", type=float, default=20.0)
    parser.add_argument("--observed-wave-min-um", type=float, default=0.35)
    parser.add_argument("--observed-wave-max-um", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-features", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the stack data product and figures."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.wavelength_bin_angstrom <= 0:
        raise ValueError("--wavelength-bin-angstrom must be positive")
    if args.interpolation_observed_dlog10 <= 0:
        raise ValueError("--interpolation-observed-dlog10 must be positive")
    if args.interpolation_coincident_fraction < 0:
        raise ValueError("--interpolation-coincident-fraction must be non-negative")
    if args.continuum_sigma_dex <= 0:
        raise ValueError("--continuum-sigma-dex must be positive")
    if args.redshift_display_factor < 1:
        raise ValueError("--redshift-display-factor must be at least one")
    filters = Table.read(args.filters, format="ascii.ecsv")
    filters.sort("pivot_angstrom")
    field_specs = load_field_manifest(args.fields.resolve(), filters)
    blue = np.asarray(filters["blue_halfmax_angstrom"], dtype=float)
    red = np.asarray(filters["red_halfmax_angstrom"], dtype=float)
    field_seds = []
    for spec in field_specs:
        logger.info("loading %s (%s)", spec.name, spec.release)
        field = normalize_field_seds(
            spec,
            filters,
            z_max=args.z_max,
            photoz_column=args.photoz_column,
            rest_wavelength=args.norm_rest_angstrom,
            min_snr=args.norm_snr_min,
            n_nearest=args.norm_nearest_bands,
        )
        logger.info(
            "%s: %d/%d use_phot sources have 0 < z <= %.3g; %d normalize",
            spec.name,
            field.z_limited_rows,
            field.use_phot_rows,
            args.z_max,
            field.normalized_rows,
        )
        field_seds.append(field)

    values = np.concatenate([field.values for field in field_seds], axis=0)
    valid = np.concatenate([field.valid for field in field_seds], axis=0)
    redshift = np.concatenate([field.redshift for field in field_seds])
    n_normalized = len(redshift)
    n_spec_normalized = sum(field.spec_normalized for field in field_seds)
    if n_normalized == 0:
        raise ValueError("no galaxies pass the normalization requirements")
    logger.info(
        "%d all-field galaxies have bracketed positive rest-%.0f A "
        "normalization from up to %d bands at S/N >= %.1f",
        n_normalized,
        args.norm_rest_angstrom,
        args.norm_nearest_bands,
        args.norm_snr_min,
    )

    z_edges = redshift_bin_edges(args.z_max, args.redshift_step)
    rest_wave_edges = linear_wavelength_edges(
        args.rest_wave_min_um,
        args.rest_wave_max_um,
        args.wavelength_bin_angstrom,
    )
    observed_wave_edges = filter_interval_wavelength_edges(
        blue,
        red,
        minimum=args.observed_wave_min_um * 1.0e4,
        maximum=args.observed_wave_max_um * 1.0e4,
    )
    pivot = np.asarray(filters["pivot_angstrom"], dtype=float)
    observed_interpolated_edges = interpolated_wavelength_edges(
        blue,
        pivot,
        red,
        args.observed_wave_min_um,
        args.observed_wave_max_um,
        args.interpolation_observed_dlog10,
    )
    stack_kwargs = dict(
        values=values,
        valid=valid,
        redshift=redshift,
        filter_blue=blue,
        filter_red=red,
        redshift_edges=z_edges,
        chunk_size=args.chunk_size,
        minimum_count=args.minimum_count,
        minimum_fraction=args.minimum_fraction,
    )
    logger.info(
        "building rest-frame stack (%d fixed %.1f-A wavelength pixels)",
        len(rest_wave_edges) - 1,
        args.wavelength_bin_angstrom,
    )
    rest = stack_filter_seds(
        **stack_kwargs,
        wavelength_edges=rest_wave_edges,
        rest_frame=True,
    )
    logger.info(
        "building observed-frame stack (%d filter-boundary wavelength cells)",
        len(observed_wave_edges) - 1,
    )
    observed = stack_filter_seds(
        **stack_kwargs,
        wavelength_edges=observed_wave_edges,
        rest_frame=False,
    )
    field_rest = []
    field_observed = []
    field_rest_interpolated = []
    field_observed_interpolated = []
    band_to_union = {str(band): index for index, band in enumerate(filters["band"])}
    for field in field_seds:
        field_stack_kwargs = {
            **stack_kwargs,
            "values": field.values,
            "valid": field.valid,
            "redshift": field.redshift,
        }
        field_rest.append(
            stack_filter_seds(
                **field_stack_kwargs,
                wavelength_edges=rest_wave_edges,
                rest_frame=True,
            )
        )
        field_observed.append(
            stack_filter_seds(
                **field_stack_kwargs,
                wavelength_edges=observed_wave_edges,
                rest_frame=False,
            )
        )
        local_indices = np.sort(
            np.array([band_to_union[band] for band in field.spec.bands], dtype=int)
        )
        interpolation_kwargs = dict(
            values=field.values[:, local_indices],
            valid=field.valid[:, local_indices],
            redshift=field.redshift,
            pivot_wavelength=pivot[local_indices],
            filter_blue=blue[local_indices],
            filter_red=red[local_indices],
            redshift_edges=z_edges,
            coincident_fraction=args.interpolation_coincident_fraction,
            minimum_count=0,
            minimum_fraction=0.0,
        )
        field_rest_interpolated.append(
            stack_interpolated_seds(
                **interpolation_kwargs,
                wavelength_edges=rest_wave_edges,
                rest_frame=True,
            )
        )
        field_observed_interpolated.append(
            stack_interpolated_seds(
                **interpolation_kwargs,
                wavelength_edges=observed_interpolated_edges,
                rest_frame=False,
            )
        )

    rest_interpolated = combine_sed_stacks(
        field_rest_interpolated,
        minimum_count=args.minimum_count,
        minimum_fraction=args.minimum_fraction,
    )
    observed_interpolated = combine_sed_stacks(
        field_observed_interpolated,
        minimum_count=args.minimum_count,
        minimum_fraction=args.minimum_fraction,
    )

    prefix = args.output_prefix.resolve()
    fits_path = prefix.with_suffix(".fits")
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    coverage_path = prefix.with_name(prefix.name + "_coverage.png")
    rest_path = prefix.with_name(prefix.name + "_rest_100A.png")
    observed_path = prefix.with_name(prefix.name + "_observed_filter_width.png")
    interpolated_path = prefix.with_name(prefix.name + "_interpolated.png")
    interpolated_pdf_path = prefix.with_name(prefix.name + "_interpolated.pdf")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    write_stack_fits(
        fits_path,
        rest,
        observed,
        rest_interpolated,
        observed_interpolated,
        field_rest,
        field_observed,
        field_seds,
        filters,
        n_input=sum(field.input_rows for field in field_seds),
        n_use_phot=sum(field.use_phot_rows for field in field_seds),
        n_normalized=n_normalized,
        n_spec=n_spec_normalized,
        norm_wavelength=args.norm_rest_angstrom,
        norm_snr=args.norm_snr_min,
        norm_band_count=args.norm_nearest_bands,
        wavelength_bin_width=args.wavelength_bin_angstrom,
        interpolation_log_step=args.interpolation_observed_dlog10,
        interpolation_coincident_fraction=args.interpolation_coincident_fraction,
        redshift_step=args.redshift_step,
        valid_per_band=np.sum(
            [field.valid_per_band for field in field_seds], axis=0, dtype=np.int64
        ),
        normalized_valid_per_band=np.sum(
            [field.normalized_valid_per_band for field in field_seds],
            axis=0,
            dtype=np.int64,
        ),
    )
    plot_stacks(
        png_path,
        pdf_path,
        rest,
        observed,
        n_galaxy=n_normalized,
        n_filter=len(filters),
        field_names=[field.spec.name for field in field_seds],
        redshift_step=args.redshift_step,
        features=not args.no_features,
        dpi=args.dpi,
    )
    plot_coverage(
        coverage_path,
        rest,
        observed,
        field_names=[field.spec.name for field in field_seds],
        dpi=args.dpi,
    )
    plot_wide_stack(
        rest_path,
        rest,
        frame="rest",
        n_galaxy=n_normalized,
        field_names=[field.spec.name for field in field_seds],
        features=not args.no_features,
        dpi=args.dpi,
    )
    plot_interpolated_stacks(
        interpolated_path,
        interpolated_pdf_path,
        rest_interpolated,
        observed_interpolated,
        n_galaxy=n_normalized,
        field_names=[field.spec.name for field in field_seds],
        field_observed=field_observed_interpolated,
        continuum_sigma_dex=args.continuum_sigma_dex,
        redshift_display_factor=args.redshift_display_factor,
        features=not args.no_features,
        dpi=args.dpi,
    )
    plot_wide_stack(
        observed_path,
        observed,
        frame="observed",
        n_galaxy=n_normalized,
        field_names=[field.spec.name for field in field_seds],
        features=not args.no_features,
        dpi=args.dpi,
    )

    summary = {
        "field_manifest": str(args.fields.resolve()),
        "filter_metadata": str(args.filters.resolve()),
        "fields": [
            {
                "name": field.spec.name,
                "release": field.spec.release,
                "catalog": str(field.spec.catalog),
                "photoz": str(field.spec.photoz),
                "input_rows": field.input_rows,
                "use_phot_rows": field.use_phot_rows,
                "z_limited_rows": field.z_limited_rows,
                "normalized_rows": field.normalized_rows,
                "z_spec_rows_in_z_limited_sample": field.spec_redshifts,
                "z_spec_rows_in_normalized_sample": field.spec_normalized,
                "n_filters": len(field.spec.bands),
            }
            for field in field_seds
        ],
        "input_rows": sum(field.input_rows for field in field_seds),
        "use_phot_rows": sum(field.use_phot_rows for field in field_seds),
        "z_limited_rows": sum(field.z_limited_rows for field in field_seds),
        "normalized_rows": n_normalized,
        "z_spec_rows_in_z_limited_sample": sum(
            field.spec_redshifts for field in field_seds
        ),
        "z_spec_rows_in_normalized_sample": n_spec_normalized,
        "n_filters": len(filters),
        "n_redshift_bins": len(z_edges) - 1,
        "wavelength_bin_angstrom": args.wavelength_bin_angstrom,
        "observed_wavelength_grid": "filter_halfmax_boundaries",
        "interpolated_model": (
            "piecewise_linear_flam_vs_log_wavelength_within_connected_"
            "halfmax_coverage"
        ),
        "interpolation_observed_max_dlog10": args.interpolation_observed_dlog10,
        "interpolation_coincident_fraction": args.interpolation_coincident_fraction,
        "continuum_residual_sigma_dex": args.continuum_sigma_dex,
        "redshift_display_factor": args.redshift_display_factor,
        "n_rest_wavelength_bins": len(rest_wave_edges) - 1,
        "n_observed_wavelength_bins": len(observed_wave_edges) - 1,
        "n_observed_interpolated_bins": len(observed_interpolated_edges) - 1,
        "normalization_rest_angstrom": args.norm_rest_angstrom,
        "normalization_snr_min": args.norm_snr_min,
        "normalization_nearest_bands": args.norm_nearest_bands,
        "redshift_fractional_step": args.redshift_step,
        "minimum_count": args.minimum_count,
        "minimum_fraction": args.minimum_fraction,
        "products": {
            "stack_fits": str(fits_path),
            "comparison_png": str(png_path),
            "comparison_pdf": str(pdf_path),
            "coverage_png": str(coverage_path),
            "rest_100A_png": str(rest_path),
            "observed_filter_width_png": str(observed_path),
            "interpolated_png": str(interpolated_path),
            "interpolated_pdf": str(interpolated_pdf_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %s", fits_path)
    logger.info(
        "wrote %s, %s, %s, %s, and %s",
        png_path,
        coverage_path,
        rest_path,
        observed_path,
        interpolated_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
