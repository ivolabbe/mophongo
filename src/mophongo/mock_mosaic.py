"""Synthetic NIRCam (SW + LW) + MIRI mosaic generator for pipeline verification.

Produces per-filter ``_wcs.csv`` + mosaic FITS stub + optional Gaussian noise/wht
maps that drop into the real mophongo pipeline unchanged. Source injection is
done directly on the mosaic grid via :class:`DrizzlePSF.get_psf_radec`.

Noise model (count-rate drizzle convention)::

    σ_nominal(x,y) = K / (p_out · √t_exp(x,y))       [K: BUNIT · arcsec · √s]
    σ_pix(x,y)     = R(pixfrac, p_in, p_out) · σ_nominal(x,y)
    wht(x,y)       = p_out² · t_exp(x,y) / K²        ∝ p_out² · t_exp

``K`` is telescope+filter+background intrinsic (:data:`DEFAULT_NOISE_K`) and
``R`` is the Fruchter 2011 square-kernel correlation factor
(:func:`drizzle_correlation_factor`). The mock emits ``wht = 1/σ_nominal²``
directly, so ``σ_pix = R / √wht``. For real reductions whose wht uses a
different convention, :data:`DEFAULT_WHT_CALIB` gives the per-filter scalar
such that ``wht_real · wht_calib = 1/σ_nominal²`` — i.e. multiply the real
wht image by ``wht_calib`` to turn it into a mock-convention inverse-variance
map. See ``examples/MOCK_MOSAIC.md`` and ``examples/mock_test.ipynb`` for the
UDS v2.2 / v2.3 calibration.
"""

from __future__ import annotations

import dataclasses as _dc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysiaf
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.vectorized import contains as _shapely_contains

from ._mock_sip import (
    _SIP_NRCA1, _SIP_NRCA2, _SIP_NRCA3, _SIP_NRCA4,
    _SIP_NRCB1, _SIP_NRCB2, _SIP_NRCB3, _SIP_NRCB4,
    _SIP_NRCA5, _SIP_NRCB5, _SIP_MIRIM,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Pointing", "MockMosaic", "drizzle_correlation_factor",
    "NATIVE_PSCALE", "DEFAULT_OUTPUT_PSCALE",
    "DEFAULT_NOISE_K", "DEFAULT_WHT_CALIB",
]


# Native detector pixel scales [arcsec/pix].
NATIVE_PSCALE: dict[str, float] = {
    "nircam_sw": 0.031,
    "nircam_lw": 0.063,
    "miri":      0.110,
}

# Mosaic output pixel scales [arcsec/pix]; nested by factor 2 so 20/40/80 mas
# grids are pixel-aligned via the half-pixel CRPIX rule.
DEFAULT_OUTPUT_PSCALE: dict[str, float] = {
    "nircam_sw": 0.020,
    "nircam_lw": 0.040,
    "miri":      0.080,
}

# Per-filter noise constants K [BUNIT · arcsec · √s], calibrated from UDS
# 20″×12″ cutouts at (34.477414°, −5.2695644°); see mock_test.ipynb §6–§9.
DEFAULT_NOISE_K: dict[str, float] = {
    "f444w":   0.0768,  # UDS v2.2 NIRCam, pixfrac=0.75, p_out=40mas
    "f770w":   1.505,   # UDS v2.3 MIRI,   pixfrac=1.00, p_out=80mas
    "f1280w":  4.288,
    "f1500w":  5.404,
    "f1800w": 11.168,
}

# Per-filter wht calibration: multiply a real-data wht image by this scalar
# to convert it to the mock's pure-inverse-variance convention
# (``wht_real · wht_calib = 1/σ_nominal²``, so ``σ_pix = R/√(wht · wht_calib)``).
# = 1 when the real wht is already pure inverse variance (UDS F444W).
# UDS MIRI wht values are ~10⁶, giving wht_calib ~ 10⁻⁷–10⁻⁸.
DEFAULT_WHT_CALIB: dict[str, float] = {
    "f444w":  1.0,
    "f770w":  1.104e-7,   # = 1 / 3009²
    "f1280w": 2.668e-7,   # = 1 / 1936²
    "f1500w": 1.305e-7,   # = 1 / 2768²
    "f1800w": 3.230e-8,   # = 1 / 5564²
}

DEFAULT_PIXFRAC = 0.75
DEFAULT_DPI = 900

_DEFAULT_MJD_AVG = 59960.26
_DEFAULT_EXPTIME = 418.734

# Detector token embedded in filename; ``DrizzlePSF.get_psf`` parses this to
# pick the stdpsf detector.
_DET_TOKEN: dict[str, str] = {
    "NRCA1": "nrca1", "NRCA2": "nrca2", "NRCA3": "nrca3", "NRCA4": "nrca4",
    "NRCB1": "nrcb1", "NRCB2": "nrcb2", "NRCB3": "nrcb3", "NRCB4": "nrcb4",
    "NRCA5": "nrcalong", "NRCB5": "nrcblong",
    "MIRIM": "mirimage",
}

# (siaf_instrument, tie_aperture, [(detector_aperture, detector_key, sip_dict), ...])
_APERTURE_GROUPS: dict[str, tuple[str, str, list[tuple[str, str, dict]]]] = {
    "nircam_sw": (
        "NIRCam", "NRCALL_FULL",
        [
            ("NRCA1_FULL", "NRCA1", _SIP_NRCA1),
            ("NRCA2_FULL", "NRCA2", _SIP_NRCA2),
            ("NRCA3_FULL", "NRCA3", _SIP_NRCA3),
            ("NRCA4_FULL", "NRCA4", _SIP_NRCA4),
            ("NRCB1_FULL", "NRCB1", _SIP_NRCB1),
            ("NRCB2_FULL", "NRCB2", _SIP_NRCB2),
            ("NRCB3_FULL", "NRCB3", _SIP_NRCB3),
            ("NRCB4_FULL", "NRCB4", _SIP_NRCB4),
        ],
    ),
    "nircam_lw": (
        "NIRCam", "NRCALL_FULL",
        [
            ("NRCA5_FULL", "NRCA5", _SIP_NRCA5),
            ("NRCB5_FULL", "NRCB5", _SIP_NRCB5),
        ],
    ),
    "miri": (
        "MIRI", "MIRIM_FULL",
        [("MIRIM_FULL", "MIRIM", _SIP_MIRIM)],
    ),
}


# Square-kernel drizzle correlation factor (Fruchter 2011, PASP 123, 497):
#   r = pixfrac · p_in / p_out
#   R = 1 - r/3                   for r ≤ 1
#   R = (1/r) · (1 - 1/(3r))      for r ≥ 1
# R = σ_pix / (1/√wht) under pure inverse-variance wht.
def drizzle_correlation_factor(pixfrac: float,
                                input_pscale: float,
                                output_pscale: float) -> float:
    """Square-kernel drizzle pixel-to-resel rms ratio R ∈ (0, 1]."""
    r = pixfrac * input_pscale / output_pscale
    if r <= 1.0:
        return 1.0 - r / 3.0
    return (1.0 / r) * (1.0 - 1.0 / (3.0 * r))


# SIP has no constant/linear terms in our templates, so ``ap.sci_to_sky``'s
# Jacobian at the reference pixel equals the pure CD matrix.
def _cd_matrix(ap, xref: float, yref: float,
               eps: float = 1.0) -> tuple[np.ndarray, tuple[float, float]]:
    """Numerical CD matrix [deg/pix] at (xref, yref); returns (cd, (ra0, dec0))."""
    ra0, dec0 = ap.sci_to_sky(xref, yref)
    ra_x, dec_x = ap.sci_to_sky(xref + eps, yref)
    ra_y, dec_y = ap.sci_to_sky(xref, yref + eps)
    cos_d = np.cos(np.deg2rad(dec0))
    cd = np.array([
        [(ra_x - ra0) * cos_d / eps, (ra_y - ra0) * cos_d / eps],
        [(dec_x - dec0) / eps, (dec_y - dec0) / eps],
    ])
    return cd, (ra0, dec0)


def _pointing_footprints(group: str, pointings: list["Pointing"],
                          siaf_cache: dict[str, "pysiaf.Siaf"] | None = None,
                          ) -> list[Polygon]:
    """Sky polygons for every detector of every pointing in ``group``."""
    inst_name, tie_name, dets = _APERTURE_GROUPS[group]
    if siaf_cache is None:
        siaf = pysiaf.Siaf(inst_name)
    else:
        siaf = siaf_cache.setdefault(inst_name, pysiaf.Siaf(inst_name))
    tie = siaf[tie_name]
    polys: list[Polygon] = []
    for p in pointings:
        att = pysiaf.utils.rotations.attitude(tie.V2Ref, tie.V3Ref, p.ra, p.dec, p.pa)
        for ap_name, _, _ in dets:
            ap = siaf[ap_name]
            ap.set_attitude_matrix(att)
            nx, ny = int(ap.XSciSize), int(ap.YSciSize)
            xc = np.array([0.5, nx + 0.5, nx + 0.5, 0.5])
            yc = np.array([0.5, 0.5, ny + 0.5, ny + 0.5])
            ra_c, dec_c = ap.sci_to_sky(xc, yc)
            polys.append(Polygon(list(zip(np.asarray(ra_c).ravel(),
                                           np.asarray(dec_c).ravel()))))
    return polys


def _pointing_to_rows(group: str, ra: float, dec: float, pa_v3: float,
                      frame_id: int, mjd_avg: float, exptime: float,
                      siaf_cache: dict[str, "pysiaf.Siaf"] | None = None,
                      ) -> list[dict]:
    """Emit one row per detector for a single pointing."""
    inst_name, tie_name, dets = _APERTURE_GROUPS[group]
    if siaf_cache is None:
        siaf = pysiaf.Siaf(inst_name)
    else:
        siaf = siaf_cache.setdefault(inst_name, pysiaf.Siaf(inst_name))

    tie = siaf[tie_name]
    att = pysiaf.utils.rotations.attitude(tie.V2Ref, tie.V3Ref, ra, dec, pa_v3)

    rows: list[dict] = []
    for ap_name, det_key, sip in dets:
        ap = siaf[ap_name]
        ap.set_attitude_matrix(att)
        xref, yref = float(ap.XSciRef), float(ap.YSciRef)
        nx, ny = int(ap.XSciSize), int(ap.YSciSize)
        cd, (crval1, crval2) = _cd_matrix(ap, xref, yref)

        # Column order follows UDS ``uds-test-f444w_wcs.csv`` for readability;
        # we emit the subset that DrizzlePSF + PSFRegionMap actually consume.
        row = {
            "file": f"jw_mock_{frame_id:05d}_{_DET_TOKEN[det_key]}_rate.fits",
            "ext": 1,
            "exptime": exptime,
            "wcsaxes": 2,
            "crpix1": xref, "crpix2": yref,
            "cd1_1": float(cd[0, 0]), "cd1_2": float(cd[0, 1]),
            "cd2_1": float(cd[1, 0]), "cd2_2": float(cd[1, 1]),
            "cunit1": "deg", "cunit2": "deg",
            "ctype1": "RA---TAN-SIP", "ctype2": "DEC--TAN-SIP",
            "crval1": float(crval1), "crval2": float(crval2),
            "mjd-avg": mjd_avg,
            "radesys": "ICRS",
        }
        row.update(sip)   # a_order, a_*, b_order, b_*, sipcrpx1, sipcrpx2
        row["naxis"] = 2
        row["naxis1"] = nx
        row["naxis2"] = ny
        rows.append(row)
    return rows


def _write_wcs_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV; columns = union of row keys in first-seen order."""
    if not rows:
        raise ValueError("no rows to write")
    cols: list[str] = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    Table(rows=[[r.get(c, np.nan) for c in cols] for r in rows],
          names=cols).write(str(path), format="csv", overwrite=True)


def _aligned_mosaic_wcs(crval: tuple[float, float],
                        crpix: tuple[float, float],
                        pscale_arcsec: float,
                        size_pix: tuple[int, int],
                        radesys: str = "FK5") -> WCS:
    """TAN WCS with explicit CRVAL, CRPIX, pscale. size_pix = (nx, ny)."""
    scale = pscale_arcsec / 3600.0
    w = WCS(naxis=2)
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.radesys = radesys
    w.pixel_shape = tuple(size_pix)
    return w


# Half-pixel CRPIX rule for grid nesting: crpix_fine = r·crpix_ref − (r−1)/2
# guarantees that r×r block-binning the fine grid reproduces the coarse grid.
def nested_crpix(crpix_ref: tuple[float, float], ratio: int
                 ) -> tuple[float, float]:
    """CRPIX at a resolution ``ratio`` times finer than ``crpix_ref``."""
    shift = (ratio - 1) / 2.0
    return (ratio * crpix_ref[0] - shift, ratio * crpix_ref[1] - shift)


def _write_mosaic_stub(path: Path, wcs: WCS, shape: tuple[int, int],
                       extra_keys: dict | None = None) -> None:
    data = np.zeros(shape, dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["WCSAXES"] = 2
    hdr["CTYPE1"] = wcs.wcs.ctype[0]
    hdr["CTYPE2"] = wcs.wcs.ctype[1]
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL1"] = float(wcs.wcs.crval[0])
    hdr["CRVAL2"] = float(wcs.wcs.crval[1])
    hdr["CRPIX1"] = float(wcs.wcs.crpix[0])
    hdr["CRPIX2"] = float(wcs.wcs.crpix[1])
    for i in range(2):
        for j in range(2):
            hdr[f"CD{i + 1}_{j + 1}"] = float(wcs.wcs.cd[i, j])
    hdr["RADESYS"] = wcs.wcs.radesys or "FK5"
    hdr["LONPOLE"] = 180.0
    hdr["LATPOLE"] = float(wcs.wcs.crval[1])
    if extra_keys:
        for k, v in extra_keys.items():
            hdr[k] = v
    fits.PrimaryHDU(data=data, header=hdr).writeto(str(path), overwrite=True)


def _as_tuple(v):
    """Coerce a list/tuple (or ``None``) to tuple; passes non-sequences through."""
    if v is None or isinstance(v, tuple):
        return v
    return tuple(v) if isinstance(v, (list, tuple)) else v


@dataclass
class Pointing:
    """A single JWST pointing: (RA, Dec, PA_V3)."""

    ra: float
    dec: float
    pa: float


# F0xx/F1xx ≤ 212 → SW, ≤ 480 → LW, else MIRI.
def _family_of(filter_name: str) -> str:
    """Map filter name (e.g. ``'f444w'``) to aperture-group key."""
    name = filter_name.lower().strip()
    if name.startswith("f") and name[1:].rstrip("w").isdigit():
        num = int(name[1:].rstrip("w"))
        if num <= 212:
            return "nircam_sw"
        if num <= 480:
            return "nircam_lw"
    return "miri"


@dataclass
class MockMosaic:
    """Config + factory for synthetic NIRCam SW/LW + MIRI mosaic artifacts.

    Call :meth:`write` to emit per-filter wcs.csv + mosaic FITS stubs, then
    :meth:`inject_noise_all` for noise + wht maps. Mosaics are nested at
    20 / 40 / 80 mas (2× and 4× block-binnable).
    """

    out_dir: Path
    center_radec: tuple[float, float] = (34.5, -5.2)
    nircam_sw_frames: dict[str, list[Pointing]] = field(default_factory=dict)
    nircam_lw_frames: dict[str, list[Pointing]] = field(default_factory=dict)
    miri_frames:      dict[str, list[Pointing]] = field(default_factory=dict)

    # Reference family (key of :data:`DEFAULT_OUTPUT_PSCALE`). All other families
    # nest via the half-pixel CRPIX rule. npix/crpix/crval auto-fit the union
    # of all configured detector footprints when not set.
    mosaic_pscale: str = "nircam_lw"
    mosaic_npix:  tuple[int, int]     | None = None
    mosaic_crval: tuple[float, float] | None = None
    mosaic_crpix: tuple[float, float] | None = None

    mjd_avg: float = _DEFAULT_MJD_AVG
    # Scalar or per-filter dict for per-frame EXPTIME [s].
    exptime: float | dict[str, float] = _DEFAULT_EXPTIME
    # Overrides :data:`DEFAULT_NOISE_K`; empty ⇒ baked defaults.
    noise_K: dict[str, float] = field(default_factory=dict)
    # Scalar, or dict keyed by family or filter.
    pixfrac: float | dict[str, float] = DEFAULT_PIXFRAC
    noise_seed: int | None = None

    stpsf_dir: Path | None = None
    # Overrides :meth:`default_stpsf_pattern`.
    stpsf_patterns: dict[str, str] = field(default_factory=dict)

    # Source-injection defaults.
    snr_range: tuple[float, float] = (5.0, 5000.0)
    apertures_arcsec: tuple[float, ...] = (0.32, 0.7)
    psf_size_arcsec: float | dict[str, float] = 2.0
    bunit: str = "10.0*nanoJansky"

    def __post_init__(self) -> None:
        # Transient state populated by write/inject_noise_all/load_drizzle_psfs/
        # inject_point_sources so report()/plot() take no arguments.
        self._paths: dict = {}
        self._noise_info: dict = {}
        self._dpsfs: dict = {}
        self._patterns: dict[str, str] = {}
        self._truth = None

        # Coerce JSON-loaded lists back to the declared tuple types.
        self.out_dir = Path(self.out_dir)
        if self.stpsf_dir is not None:
            self.stpsf_dir = Path(self.stpsf_dir)
        for key in ("center_radec", "mosaic_crval", "mosaic_crpix",
                    "mosaic_npix", "snr_range", "apertures_arcsec"):
            setattr(self, key, _as_tuple(getattr(self, key)))
        for bucket in (self.nircam_sw_frames, self.nircam_lw_frames,
                       self.miri_frames):
            for filt, frames in list(bucket.items()):
                bucket[filt] = [p if isinstance(p, Pointing) else Pointing(**p)
                                for p in frames]
        if self.mosaic_pscale not in DEFAULT_OUTPUT_PSCALE:
            raise ValueError(
                f"mosaic_pscale must be one of {list(DEFAULT_OUTPUT_PSCALE)}, "
                f"got {self.mosaic_pscale!r}"
            )

    # ---- config I/O -------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict; empty override dicts are dropped."""
        def _enc(o):
            if isinstance(o, Pointing):
                return {"ra": o.ra, "dec": o.dec, "pa": o.pa}
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            if isinstance(o, dict):
                return {k: _enc(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_enc(x) for x in o]
            return o
        d = {k: _enc(v) for k, v in _dc.asdict(self).items()}
        for k in ("noise_K", "stpsf_patterns"):
            if d.get(k) == {}:
                d.pop(k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MockMosaic":
        return cls(**d)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "MockMosaic":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- per-filter resolvers --------------------------------------------

    def _pixfrac_for(self, filter_name: str, family: str) -> float:
        if isinstance(self.pixfrac, (int, float)):
            return float(self.pixfrac)
        return float(self.pixfrac.get(filter_name,
                                       self.pixfrac.get(family, DEFAULT_PIXFRAC)))

    def _exptime_for(self, filter_name: str) -> float:
        if isinstance(self.exptime, (int, float)):
            return float(self.exptime)
        return float(self.exptime.get(filter_name, _DEFAULT_EXPTIME))

    def _psf_size_for(self, filter_name: str) -> float:
        if isinstance(self.psf_size_arcsec, (int, float)):
            return float(self.psf_size_arcsec)
        return float(self.psf_size_arcsec.get(filter_name, 2.0))

    def _stpsf_pattern_for(self, filter_name: str) -> str:
        return self.stpsf_patterns.get(filter_name,
                                         self.default_stpsf_pattern(filter_name))

    def _iter_filters(self):
        """Yield ``(filter_name, family, frames)`` for every configured filter."""
        for filt, frames in self.nircam_sw_frames.items():
            yield filt, "nircam_sw", frames
        for filt, frames in self.nircam_lw_frames.items():
            yield filt, "nircam_lw", frames
        for filt, frames in self.miri_frames.items():
            yield filt, "miri", frames

    # Master grid lives at ``self.mosaic_pscale``; other families derive via the
    # half-pixel CRPIX rule. Auto size+crpix span the union of ALL filters.
    def _resolve_mosaic_wcs(self) -> dict:
        """Per-family ``(size, crpix, pscale)`` nested from ``mosaic_pscale``."""
        ref_fam = self.mosaic_pscale
        p_ref = DEFAULT_OUTPUT_PSCALE[ref_fam]
        crval = (tuple(self.mosaic_crval) if self.mosaic_crval is not None
                 else tuple(self.center_radec))

        all_polys: list[Polygon] = []
        cache: dict[str, "pysiaf.Siaf"] = {}
        for _, fam, frames in self._iter_filters():
            if frames:
                all_polys.extend(_pointing_footprints(fam, frames, cache))

        bounds = None
        if self.mosaic_npix is None or self.mosaic_crpix is None:
            if not all_polys:
                raise ValueError(
                    "auto-sizing requires at least one configured frame; "
                    "otherwise set mosaic_npix and mosaic_crpix explicitly"
                )
            bounds = unary_union(all_polys).bounds
            cos_d = np.cos(np.deg2rad(crval[1]))

        # CRPIX snap-to-grid: we want ALL nested scales (20/40/80mas) on the
        # UDS-style X.5 half-integer grid. This requires the reference CRPIX
        # to have the form (even).5, which we ensure by (a) rounding nx_ref up
        # to a multiple of 4 so nx_ref/2 is even, and (b) shifting the bbox
        # center by an even number of reference pixels.
        if self.mosaic_npix is not None:
            nx_ref, ny_ref = int(self.mosaic_npix[0]), int(self.mosaic_npix[1])
        else:
            ra_min, dec_min, ra_max, dec_max = bounds
            w_as = (ra_max - ra_min) * cos_d * 3600.0
            h_as = (dec_max - dec_min) * 3600.0
            nx_ref = int(np.ceil(w_as / p_ref))
            ny_ref = int(np.ceil(h_as / p_ref))
            nx_ref += (-nx_ref) % 4
            ny_ref += (-ny_ref) % 4

        if self.mosaic_crpix is not None:
            crpix_ref = tuple(self.mosaic_crpix)
        elif all_polys:
            ra_min, dec_min, ra_max, dec_max = bounds
            ra_c = 0.5 * (ra_min + ra_max)
            dec_c = 0.5 * (dec_min + dec_max)
            dx_pix = int(round((ra_c - crval[0]) * cos_d * 3600.0 / p_ref / 2)) * 2
            dy_pix = int(round((dec_c - crval[1]) * 3600.0 / p_ref / 2)) * 2
            crpix_ref = (nx_ref / 2 + 0.5 + dx_pix,
                         ny_ref / 2 + 0.5 - dy_pix)
        else:
            crpix_ref = (nx_ref / 2 + 0.5, ny_ref / 2 + 0.5)

        out: dict = {"crval": crval}
        for fam, p_fam in DEFAULT_OUTPUT_PSCALE.items():
            ratio = p_ref / p_fam
            nx = int(round(nx_ref * ratio))
            ny = int(round(ny_ref * ratio))
            out[fam] = ((nx, ny), nested_crpix(crpix_ref, ratio), p_fam)
        return out

    # ---- write CSV + mosaic stubs ----------------------------------------

    def write(self) -> dict[str, dict]:
        """Emit per-filter wcs.csv + empty mosaic FITS stub."""
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cache: dict[str, "pysiaf.Siaf"] = {}
        mwcs_spec = self._resolve_mosaic_wcs()
        crval = mwcs_spec["crval"]
        paths: dict[str, dict] = {}

        for filt, family, frames in self._iter_filters():
            if not frames:
                continue
            size, crpix, pscale = mwcs_spec[family]
            paths[filt] = self._write_filter(
                out, filt, family, frames, size, crpix, pscale, crval, cache,
            )
        self._paths = paths
        return paths

    def _write_filter(self, out: Path, filt: str, group: str,
                      frames: list[Pointing],
                      size_pix: tuple[int, int],
                      crpix: tuple[float, float],
                      pscale: float,
                      crval: tuple[float, float],
                      cache: dict) -> dict:
        rows: list[dict] = []
        exp = self._exptime_for(filt)
        for i, p in enumerate(frames, start=1):
            rows.extend(_pointing_to_rows(
                group, p.ra, p.dec, p.pa,
                frame_id=i, mjd_avg=self.mjd_avg, exptime=exp,
                siaf_cache=cache,
            ))
        csv_path = out / f"mock_{filt}_wcs.csv"
        _write_wcs_csv(csv_path, rows)

        mwcs = _aligned_mosaic_wcs(crval, crpix, pscale, size_pix)
        fits_path = out / f"mock_{filt}_sci.fits"
        pf = self._pixfrac_for(filt, group)
        _write_mosaic_stub(fits_path, mwcs, (size_pix[1], size_pix[0]),
                           extra_keys={"KERNEL": "square", "PIXFRAC": pf,
                                       "DRIZKERN": "square", "DRIZPIXF": pf})

        logger.info("wrote %s (%d rows) and %s (%d×%d)",
                    csv_path.name, len(rows), fits_path.name, *size_pix)
        return {"csv": csv_path, "fits": fits_path, "wcs": mwcs,
                "n_rows": len(rows), "crpix": crpix, "size": size_pix,
                "pscale": pscale, "family": group}

    # ---- noise injection -------------------------------------------------

    def inject_noise(self, filter_name: str,
                     paths: dict[str, dict],
                     K: float | None = None,
                     pixfrac: float | None = None,
                     seed: int | None = None,
                     bunit: str | None = None,
                     dpsf=None) -> dict:
        """Write ``<filter>_sci.fits`` (noise) + ``<filter>_wht.fits`` (1/var).

        ``t_exp(x,y) = Σ frame EXPTIMEs`` for frames whose footprint contains
        the pixel, then σ_pix = R·K/(p_out·√t_exp). ``wht = 1/σ_nominal²``.
        """
        from mophongo.psf import DrizzlePSF  # heavy local import
        from skimage.draw import polygon as sk_polygon

        if filter_name not in paths:
            raise KeyError(f"filter {filter_name!r} not in paths; run write() first")
        info = paths[filter_name]
        family = info["family"]
        p_out = info["pscale"]
        p_in = NATIVE_PSCALE[family]
        _pixfrac = (float(pixfrac) if pixfrac is not None
                    else self._pixfrac_for(filter_name, family))
        _seed = seed if seed is not None else self.noise_seed
        _bunit = bunit if bunit is not None else self.bunit

        if K is None:
            K = self.noise_K.get(filter_name, DEFAULT_NOISE_K.get(filter_name))
        if K is None:
            raise ValueError(
                f"no noise calibration K for filter {filter_name!r}; pass K= or set "
                f"MockMosaic.noise_K[{filter_name!r}] or DEFAULT_NOISE_K[{filter_name!r}]"
            )

        if dpsf is None:
            dpsf = DrizzlePSF(driz_image=str(info["fits"]), csv_file=str(info["csv"]))

        # Exposure-time map via per-footprint rasterization on the mosaic grid.
        nx, ny = info["size"]
        mwcs = info["wcs"]
        texp = np.zeros((ny, nx), dtype=np.float32)
        per_frame = self._exptime_for(filter_name)
        for poly in dpsf.footprint.values():
            ra_c, dec_c = np.asarray(poly.exterior.coords).T
            xpix, ypix = mwcs.wcs_world2pix(ra_c, dec_c, 0)
            rr, cc = sk_polygon(ypix, xpix, shape=(ny, nx))
            texp[rr, cc] += per_frame

        R = drizzle_correlation_factor(_pixfrac, p_in, p_out)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_nom = np.where(
                texp > 0, K / (p_out * np.sqrt(np.maximum(texp, 1e-30))), 0.0
            ).astype(np.float32)
        sigma_pix = (R * sigma_nom).astype(np.float32)

        rng = np.random.default_rng(_seed)
        noise = rng.normal(
            scale=np.where(sigma_pix > 0, sigma_pix, 1.0),
            size=sigma_pix.shape,
        ).astype(np.float32)
        noise[sigma_pix == 0] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            wht = np.where(sigma_nom > 0, 1.0 / sigma_nom ** 2, 0.0).astype(np.float32)

        sci_path = Path(info["fits"])
        with fits.open(sci_path) as h:
            hdr = h[0].header.copy()
        hdr["BUNIT"] = _bunit
        hdr["KERNEL"] = "square"
        hdr["PIXFRAC"] = _pixfrac
        hdr["DRIZKERN"] = "square"
        hdr["DRIZPIXF"] = _pixfrac
        fits.PrimaryHDU(data=noise, header=hdr).writeto(str(sci_path), overwrite=True)

        wht_path = sci_path.with_name(sci_path.name.replace("_sci", "_wht"))
        hdr_w = hdr.copy()
        hdr_w["BUNIT"] = f"1/({_bunit})^2"
        fits.PrimaryHDU(data=wht, header=hdr_w).writeto(str(wht_path), overwrite=True)

        logger.info(
            "injected noise in %s (R=%.3f, K=%.4g, p_out=%.3fas, p_in=%.3fas)",
            filter_name, R, K, p_out, p_in,
        )
        return {
            "sci": sci_path, "wht": wht_path, "texp": texp,
            "sigma_nom": sigma_nom, "sigma_pix": sigma_pix,
            "R": R, "K": K, "p_in": p_in, "p_out": p_out, "pixfrac": _pixfrac,
        }

    def inject_noise_all(self, paths: dict[str, dict], **kwargs) -> dict[str, dict]:
        """:meth:`inject_noise` for every filter with a known K."""
        family_lookup = {f: fam for f, fam, _ in self._iter_filters()}
        out: dict[str, dict] = {}
        for filt in paths:
            K = self.noise_K.get(filt, DEFAULT_NOISE_K.get(filt))
            if K is None:
                logger.info("skipping noise for %s (no K)", filt)
                continue
            pf = self._pixfrac_for(filt, family_lookup.get(filt, "nircam_lw"))
            out[filt] = self.inject_noise(filt, paths, K=K, pixfrac=pf, **kwargs)
        self._noise_info = out
        return out

    # ---- drizzle PSFs + source injection ---------------------------------

    @staticmethod
    def default_stpsf_pattern(filter_name: str) -> str:
        """Default ``load_jwst_stdpsf`` pattern for a filter."""
        fam = _family_of(filter_name)
        up = filter_name.upper()
        if fam == "miri":
            return f"UDS_MIRI_{up}_OS4_GRID1"
        return f"UDS_NRC.._{up}_OS4_GRID1"

    def load_drizzle_psfs(self, paths: dict[str, dict],
                          psf_dir: Path | str | None = None,
                          stpsf_patterns: dict[str, str] | None = None,
                          ) -> dict[str, "DrizzlePSF"]:
        """Per-filter ``DrizzlePSF`` with stdpsf grid loaded."""
        from mophongo.psf import DrizzlePSF
        d = Path(psf_dir if psf_dir is not None else (self.stpsf_dir or "data/PSF"))
        override = dict(stpsf_patterns or {})
        dpsfs: dict[str, "DrizzlePSF"] = {}
        patterns: dict[str, str] = {}
        for filt, info in paths.items():
            dp = DrizzlePSF(driz_image=str(info["fits"]), csv_file=str(info["csv"]))
            pat = override.get(filt) or self._stpsf_pattern_for(filt)
            dp.load_jwst_stdpsf(local_dir=str(d), filter_pattern=pat,
                                verbose=False)
            dpsfs[filt] = dp
            patterns[filt] = pat
        self._dpsfs = dpsfs
        self._patterns = patterns
        return dpsfs

    def sample_positions(self, n: int, dpsf: "DrizzlePSF",
                         seed: int | None = None,
                         oversample: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Rejection-sample ``n`` (ra, dec) inside the union of ``dpsf`` footprints."""
        coverage = unary_union(list(dpsf.footprint.values()))
        ra_min, dec_min, ra_max, dec_max = coverage.bounds
        rng = np.random.default_rng(seed if seed is not None else self.noise_seed)
        kept_ra: list[np.ndarray] = []
        kept_dec: list[np.ndarray] = []
        kept = 0
        while kept < n:
            batch = max(oversample * (n - kept), n)
            ra = rng.uniform(ra_min, ra_max, size=batch)
            dec = rng.uniform(dec_min, dec_max, size=batch)
            inside = _shapely_contains(coverage, ra, dec)
            kept_ra.append(ra[inside])
            kept_dec.append(dec[inside])
            kept += int(inside.sum())
        ra_out = np.concatenate(kept_ra)[:n]
        dec_out = np.concatenate(kept_dec)[:n]
        return ra_out, dec_out

    def inject_point_sources(self,
                             paths: dict[str, dict],
                             dpsfs: dict[str, "DrizzlePSF"],
                             n: int = 100,
                             snr_range: tuple[float, float] | None = None,
                             ref_filter: str = "f444w",
                             apertures_arcsec: tuple[float, ...] | None = None,
                             psf_size_arcsec: float | None = None,
                             seed: int | None = None,
                             ) -> "Table":
        """Inject ``n`` point sources with log-uniform matched-filter SNR.

        Positions are sampled inside the ``ref_filter`` coverage; per-source flux
        is set from the matched filter on the ``ref_filter`` wht map, and the
        same true flux is painted in every filter.
        """
        if ref_filter not in dpsfs:
            raise KeyError(f"ref_filter {ref_filter!r} not in dpsfs ({list(dpsfs)})")
        _snr = snr_range if snr_range is not None else self.snr_range
        _apr = apertures_arcsec if apertures_arcsec is not None else self.apertures_arcsec
        rng = np.random.default_rng(seed if seed is not None else self.noise_seed)

        # 1) Positions inside the reference filter's coverage.
        ra_src, dec_src = self.sample_positions(n, dpsfs[ref_filter],
                                                 seed=int(rng.integers(1 << 30)))
        pos_list = list(zip(ra_src, dec_src))

        # 2) Per-filter PSF stamps + in-coverage flags.
        #
        # Two corrections are applied here:
        #
        # (a) Sub-pixel re-centering. DrizzlePSF.get_psf has a per-filter
        #     ePSF-grid phase offset that would leak into the ground truth as
        #     a spurious inter-filter shift. Each stamp is rolled with
        #     bicubic interpolation so its centroid sits exactly at the
        #     requested sub-pixel offset from the integer paste pixel.
        #
        # (b) Normalization at the full PSF extent. The stpsf ePSF models
        #     plateau at sum ≈ 0.95–0.97 within a few arcsec; at the mock's
        #     stamp size (``psf_size_arcsec``) the stamp sum already equals
        #     that full-extent integral. We divide each stamp by that sum
        #     once, so the stored stamp is unit-normalized **at full extent**
        #     — identical to the convention in the rest of the codebase.
        #     Note this is NOT a per-crop renormalization: if a downstream
        #     consumer later crops the stamp tighter, the crop's sum is <1
        #     by design (``ee_fraction`` tracks EE at the cropped radius),
        #     and the template fit remains unbiased because flux is the
        #     scalar applied to the crop.
        #
        # Requirement: ``psf_size_arcsec`` must be large enough that the
        # stamp captures the full ePSF model footprint. 8″ satisfies this
        # for NIRCam LW and MIRI stpsf grids.
        from scipy.ndimage import shift as _ndi_shift
        from photutils.centroids import centroid_com as _com
        psfs: dict[str, np.ndarray] = {}
        inside: dict[str, np.ndarray] = {}
        for filt, dp in dpsfs.items():
            pat = self._patterns.get(filt) or self._stpsf_pattern_for(filt)
            size_filt = (psf_size_arcsec if psf_size_arcsec is not None
                         else self._psf_size_for(filt))
            cube = dp.get_psf_radec(pos_list, filter=pat, size=size_filt)
            sz = cube.shape[1]
            hs = sz // 2
            mwcs_f = paths[filt]["wcs"]
            xi_f, yi_f = mwcs_f.wcs_world2pix(ra_src, dec_src, 0)
            xi_r = np.round(xi_f).astype(int)
            yi_r = np.round(yi_f).astype(int)
            dx_sub = xi_f - xi_r
            dy_sub = yi_f - yi_r
            out = np.zeros_like(cube)
            in_filt = np.zeros(cube.shape[0], dtype=bool)
            for i in range(cube.shape[0]):
                s = cube[i].sum()
                if not (s > 1e-6):
                    continue
                py, px = np.unravel_index(np.argmax(cube[i]), cube[i].shape)
                r = 5
                sub = cube[i][max(0, py - r):py + r + 1,
                              max(0, px - r):px + r + 1]
                cx, cy = _com(sub)
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    continue
                cx += max(0, px - r)
                cy += max(0, py - r)
                tgt_x = hs + dx_sub[i]
                tgt_y = hs + dy_sub[i]
                stamp = _ndi_shift(
                    cube[i], (tgt_y - cy, tgt_x - cx),
                    order=3, mode="constant", cval=0.0, prefilter=True,
                )
                # normalize at full PSF extent (stamp covers the whole stpsf
                # model; s is the full-extent integral)
                out[i] = stamp / s
                in_filt[i] = True
            psfs[filt] = out
            inside[filt] = in_filt

        # 3) Reference-filter flux from target SNR via wht (= 1/σ_nominal²).
        ref_info = paths[ref_filter]
        wht_ref = fits.getdata(ref_info["fits"].with_name(
            ref_info["fits"].name.replace("_sci", "_wht")))
        mwcs_ref = ref_info["wcs"]
        ny, nx = wht_ref.shape
        xr, yr = mwcs_ref.wcs_world2pix(ra_src, dec_src, 0)
        xr_i = np.round(xr).astype(int)
        yr_i = np.round(yr).astype(int)

        target_snr = np.exp(rng.uniform(np.log(_snr[0]), np.log(_snr[1]), size=n))
        fluxes = np.zeros(n, dtype=np.float32)
        valid_ref = np.zeros(n, dtype=bool)
        sz_ref = psfs[ref_filter].shape[1]
        # hs is the stamp index that holds the source centroid (set by the
        # _ndi_shift above). Slice [yi_i - hs : yi_i - hs + sz] places it at
        # image y = yi_i + dy_sub, which matches the WCS prediction for both
        # odd and even sz.
        hs_ref = sz_ref // 2
        for i in range(n):
            if not inside[ref_filter][i]:
                continue
            y0, y1 = yr_i[i] - hs_ref, yr_i[i] - hs_ref + sz_ref
            x0, x1 = xr_i[i] - hs_ref, xr_i[i] - hs_ref + sz_ref
            if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                continue
            w_local = wht_ref[y0:y1, x0:x1]
            if (w_local <= 0).any():
                continue
            # SNR = F · √Σ(P² · wht).
            norm = float(np.sqrt(np.sum(psfs[ref_filter][i] ** 2 * w_local)))
            fluxes[i] = target_snr[i] / norm
            valid_ref[i] = True

        # 4) Paint into every filter; reuse flux, track per-filter validity.
        valid = {filt: valid_ref & inside[filt] for filt in dpsfs}
        per_filter_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for filt, dp in dpsfs.items():
            mwcs = paths[filt]["wcs"]
            xi, yi = mwcs.wcs_world2pix(ra_src, dec_src, 0)
            xi_i = np.round(xi).astype(int)
            yi_i = np.round(yi).astype(int)
            per_filter_xy[filt] = (xi, yi)
            sci = fits.getdata(paths[filt]["fits"]).astype(np.float32)
            sz = psfs[filt].shape[1]
            hs = sz // 2  # stamp index of the centroid (works for any parity)
            ny_f, nx_f = sci.shape
            for i in range(n):
                if not valid[filt][i]:
                    continue
                y0, y1 = yi_i[i] - hs, yi_i[i] - hs + sz
                x0, x1 = xi_i[i] - hs, xi_i[i] - hs + sz
                if y0 < 0 or x0 < 0 or y1 > ny_f or x1 > nx_f:
                    continue
                sci[y0:y1, x0:x1] += fluxes[i] * psfs[filt][i]
            with fits.open(paths[filt]["fits"]) as h:
                h[0].data = sci
                h.writeto(paths[filt]["fits"], overwrite=True)

        # 5) Truth catalog with total + aperture fluxes per filter.
        truth = Table()
        truth["id"] = np.arange(1, n + 1, dtype=np.int32)
        truth["ra"] = ra_src
        truth["dec"] = dec_src
        truth[f"snr_{ref_filter}"] = target_snr.astype(np.float32)
        for filt in dpsfs:
            xi, yi = per_filter_xy[filt]
            truth[f"x_{filt}"] = xi.astype(np.float32)
            truth[f"y_{filt}"] = yi.astype(np.float32)
            flux_filt = np.where(valid[filt], fluxes, np.nan).astype(np.float32)
            truth[f"flux_{filt}"] = flux_filt
            truth[f"flux_{filt}"].unit = self.bunit
            pscale = paths[filt]["pscale"]
            fracs = _aper_fractions(psfs[filt], pscale, _apr, valid[filt])
            for k, D in enumerate(_apr):
                col = f"flux_aper_D{int(D * 100):03d}_{filt}"
                truth[col] = (flux_filt * fracs[k]).astype(np.float32)
                truth[col].unit = self.bunit
            truth[f"valid_{filt}"] = valid[filt]
        self._truth = truth
        return truth

    # ---- one-shot pipeline + reporting -----------------------------------

    def build(self, n_sources: int = 200,
              psf_dir: Path | str | None = None,
              ref_filter: str = "f444w") -> tuple[dict, dict, dict, "Table"]:
        """write → inject_noise_all → load_drizzle_psfs → inject_point_sources."""
        paths = self.write()
        noise_info = self.inject_noise_all(paths)
        dpsfs = self.load_drizzle_psfs(paths, psf_dir=psf_dir)
        truth = self.inject_point_sources(paths, dpsfs, n=n_sources,
                                           ref_filter=ref_filter)
        truth.write(Path(self.out_dir) / "mock_truth.ecsv",
                    format="ascii.ecsv", overwrite=True)
        return paths, noise_info, dpsfs, truth

    def report(self) -> None:
        """Log per-filter mosaic shape, coverage, and valid-source counts."""
        if not self._paths:
            logger.warning("no mosaics yet; call write() or build() first.")
            return
        truth = self._truth
        for filt, info in self._paths.items():
            nx, ny = info["size"]
            p = info["pscale"]
            line = (f"{filt:>6s}: {nx:>5d}×{ny:<5d} @ {p * 1000:.0f} mas  "
                    f"({nx * p:6.1f}\"×{ny * p:6.1f}\")  family={info['family']}  "
                    f"exposures={info['n_rows']}")
            if truth is not None and f"valid_{filt}" in truth.colnames:
                v = int(truth[f"valid_{filt}"].sum())
                line += f"  valid={v}/{len(truth)}"
            logger.info(line)
            print(line)  # report() is an interactive convenience

    def plot(self, save: bool | str | Path = True,
             figsize: tuple | None = None,
             ref_snr: str = "snr_f444w",
             stretch_sigma: float = 2.0,
             dpi: int = DEFAULT_DPI):
        """Diagnostic plot: sci mosaic + detector footprints + truth sources."""
        if not self._paths:
            raise RuntimeError("no mosaics yet; call write() or build() first.")
        import matplotlib.pyplot as plt
        from astropy.stats import mad_std

        filters = list(self._paths)
        if figsize is None:
            figsize = (7 * len(filters), 7)
        fig, axes = plt.subplots(1, len(filters), figsize=figsize, squeeze=False)

        for ax, filt in zip(axes[0], filters):
            self._plot_panel(ax, filt, stretch_sigma, ref_snr)
        plt.tight_layout()

        if save:
            path = (Path(self.out_dir) / "mock_diagnostic.png"
                    if save is True else Path(save))
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info("wrote %s (dpi=%d)", path, dpi)
        return fig, axes

    def _plot_panel(self, ax, filt: str, stretch_sigma: float, ref_snr: str) -> None:
        import matplotlib.pyplot as plt
        from astropy.stats import mad_std

        info = self._paths[filt]
        sci = fits.getdata(info["fits"])
        mwcs = info["wcs"]
        nx, ny = info["size"]

        sig = mad_std(sci[sci != 0]) if (sci != 0).any() else 1.0
        ax.imshow(sci, origin="lower", cmap="gray",
                  vmin=-stretch_sigma * sig, vmax=stretch_sigma * sig)

        # Detector footprints: from the loaded DrizzlePSF if available, else
        # recomputed from pysiaf using the original pointings.
        if filt in self._dpsfs:
            polys = self._dpsfs[filt].footprint
        else:
            family = info["family"]
            frames = (self.nircam_sw_frames | self.nircam_lw_frames
                      | self.miri_frames).get(filt, [])
            polys = {}
            if frames:
                cache: dict = {}
                for i, p in enumerate(frames, start=1):
                    for j, poly in enumerate(_pointing_footprints(
                            family, [p], cache)):
                        polys[(f"frame{i}", j)] = poly

        cmap = plt.get_cmap("tab10")
        frame_ids = sorted({str(k[0]) for k in polys})
        colors = {fid: cmap(i % 10) for i, fid in enumerate(frame_ids)}
        for key, poly in polys.items():
            ra_c, dec_c = np.asarray(poly.exterior.coords).T
            xpix, ypix = mwcs.wcs_world2pix(ra_c, dec_c, 0)
            ax.plot(xpix, ypix, color=colors[str(key[0])], lw=1.2, alpha=0.85)

        n_src = 0
        if self._truth is not None and f"valid_{filt}" in self._truth.colnames:
            m = self._truth[f"valid_{filt}"]
            snr = (self._truth[ref_snr][m]
                   if ref_snr in self._truth.colnames else 10.0)
            # Marker size [pt²]: sqrt(SNR) scaling with a floor (~1″ on sky at
            # typical 7″-panel aspect) and a cap so bright sources don't overwhelm.
            s = np.clip(3.5 * np.sqrt(np.asarray(snr)), 8.0, 400.0)
            ax.scatter(self._truth[f"x_{filt}"][m],
                       self._truth[f"y_{filt}"][m],
                       s=s, edgecolor="white", facecolor="none", lw=0.25)
            n_src = int(m.sum())

        ax.set_title(f"{filt.upper()}  {nx}×{ny} @ {info['pscale'] * 1000:.0f} mas  "
                     f"({len(polys)} detectors, {n_src} sources)")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])


def _aper_fractions(psf_cube: np.ndarray, pscale_arcsec: float,
                    diameters_arcsec, valid: np.ndarray) -> np.ndarray:
    """Fraction of each unit-sum PSF inside circular apertures."""
    N, ny, nx = psf_cube.shape
    yy, xx = np.mgrid[:ny, :nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_pix = np.hypot(xx - cx, yy - cy)
    out = np.zeros((len(diameters_arcsec), N), dtype=np.float32)
    for k, D in enumerate(diameters_arcsec):
        mask = r_pix <= 0.5 * D / pscale_arcsec
        frac = (psf_cube * mask).reshape(N, -1).sum(axis=1)
        out[k] = np.where(valid, frac, np.nan)
    return out
