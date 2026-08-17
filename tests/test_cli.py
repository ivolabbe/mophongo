"""Tests for the command-line product accessors (mophongo.cli)."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from shapely.geometry import Polygon

import mophongo.pipeline as pipeline
import mophongo.utils as mutils
from mophongo.cli import (
    main,
    psf_to_fits,
    source_diagnostic_png,
    source_stamps_to_fits,
    stamp_wcs,
)
from mophongo.pipeline import _KERNEL_PSF_SOURCE
from mophongo.psf_map import PSFRegionMap

from utils import make_simple_data

RA, DEC = 150.0, 2.0
SCALE_HI = 0.04 / 3600.0


def _wcs(shape: tuple[int, int], pscale: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.crval = [RA, DEC]
    w.wcs.cdelt = [-pscale, pscale]
    return w


def _kernel(sigma: float) -> np.ndarray:
    y, x = np.mgrid[-6:7, -6:7]
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return k / k.sum()


@pytest.fixture()
def run_dir(tmp_path):
    """A loadable run directory: config, inputs, and three region maps.

    The inputs mirror ``test_pipeline_inspect.tiny_config`` (64 px detection
    grid, 32 px fitted grid, one source). Each map has two regions -- the east
    and west halves of the field, split at RA -- with a different stamp in
    each, so a lookup that ignored the position would return the wrong one.
    """
    n_hi, n_lo = 64, 32
    out = tmp_path / "out"
    out.mkdir()
    hi = np.zeros((n_hi, n_hi), np.float32)
    hi[30:34, 30:34] = 1.0
    seg = np.zeros((n_hi, n_hi), np.int32)
    seg[30:34, 30:34] = 1
    hdr_hi = _wcs((n_hi, n_hi), SCALE_HI).to_header()
    hdr_lo = _wcs((n_lo, n_lo), 2 * SCALE_HI).to_header()
    fits.writeto(tmp_path / "hi.fits", hi, hdr_hi)
    fits.writeto(tmp_path / "seg.fits", seg, hdr_hi)
    fits.writeto(tmp_path / "hi_wht.fits", np.full((n_hi, n_hi), 1e6, np.float32), hdr_hi)
    fits.writeto(tmp_path / "lo.fits", np.zeros((n_lo, n_lo), np.float32), hdr_lo)
    fits.writeto(tmp_path / "wht.fits", np.ones((n_lo, n_lo), np.float32), hdr_lo)
    ra, dec = WCS(hdr_hi).wcs_pix2world(32.0, 32.0, 0)
    Table({"id": [1], "x": [32.0], "y": [32.0],
           "ra": [float(ra)], "dec": [float(dec)]}).write(tmp_path / "cat.fits")
    for csv in ("hi_wcs.csv", "lo_wcs.csv"):
        (tmp_path / csv).write_text("file,crval1\nexp1.fits,150.0\n")

    halves = [
        Polygon([(RA, DEC - 1), (RA + 1, DEC - 1), (RA + 1, DEC + 1), (RA, DEC + 1)]),
        Polygon([(RA - 1, DEC - 1), (RA, DEC - 1), (RA, DEC + 1), (RA - 1, DEC + 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "psf_key": [0, 1],
            "kernel_method": ["wiener", "wiener"],
            "kernel_reg": [1e-3, 1e-3],
            "kernel_psf_source": [_KERNEL_PSF_SOURCE] * 2,
            "psf_size": [4.0, 4.0],
        },
        geometry=halves,
        crs="EPSG:4326",
    )
    prm = PSFRegionMap(gdf, psfs=np.stack([_kernel(1.0), _kernel(2.5)]))
    for kind in ("kernel", "psf_lo", "psf_hi"):
        prm.to_file(out / f"tiny_{kind}.geojson")

    cfg = {
        "name": "tiny",
        "out_dir": str(out),
        "sci_hi": str(tmp_path / "hi.fits"),
        "wht_hi": str(tmp_path / "hi_wht.fits"),
        "segmap": str(tmp_path / "seg.fits"),
        "catalog": str(tmp_path / "cat.fits"),
        "sci_lo": str(tmp_path / "lo.fits"),
        "wht_lo": str(tmp_path / "wht.fits"),
        "csv_hi": str(tmp_path / "hi_wcs.csv"),
        "csv_lo": str(tmp_path / "lo_wcs.csv"),
        "fit": {"extend_mode": "none"},
    }
    (out / "tiny.json").write_text(json.dumps(cfg))
    return out


def test_psf_cli_writes_the_region_stamp_with_the_reference_grid_wcs(run_dir, tmp_path):
    """The kernel map is on the detection grid, so its stamp inherits sci_hi."""
    out = tmp_path / "kern.fits"
    main(["psf", str(run_dir / "tiny_kernel.geojson"), str(RA + 0.1), str(DEC),
          "-o", str(out)])

    with fits.open(out) as hdul:
        data, hdr = hdul[0].data, hdul[0].header
    assert np.allclose(data, _kernel(1.0))  # east half -> key 0
    assert hdr["PSFKEY"] == 0
    assert hdr["MAPKIND"] == "kernel"
    assert hdr["KERNMETH"] == "wiener"
    assert hdr["STAMPSUM"] == pytest.approx(1.0, abs=1e-6)
    # PSF stamps are stored float32 (PSFRegionMap.__post_init__), so a stamp
    # normalised to unit sum and fully contained in its box -- which is what
    # this synthetic fixture is -- reports an encircled energy of 1 to within
    # float32 rounding rather than exactly 1. Real PSFs sit at 0.92-0.96 and
    # are nowhere near the bound.
    assert 0.0 < hdr["EE_BOX"] <= 1.0 + 1e-6

    w = WCS(hdr)
    ny, nx = data.shape
    ra, dec = w.wcs_pix2world((nx - 1) / 2, (ny - 1) / 2, 0)
    assert (float(ra), float(dec)) == pytest.approx((RA + 0.1, DEC), abs=1e-9)
    from astropy.wcs.utils import proj_plane_pixel_scales

    assert proj_plane_pixel_scales(w)[0] * 3600 == pytest.approx(0.04, rel=1e-6)


def test_psf_cli_from_a_run_config_picks_the_bands_own_grid(run_dir, tmp_path):
    """--map-kind psf_lo resolves the map and takes sci_lo's pixel scale."""
    out = tmp_path / "psf_lo.fits"
    psf_to_fits(run_dir / "tiny.json", RA - 0.1, DEC, out, kind="psf_lo")

    with fits.open(out) as hdul:
        data, hdr = hdul[0].data, hdul[0].header
    assert np.allclose(data, _kernel(2.5))  # west half -> key 1
    assert hdr["PSFKEY"] == 1
    from astropy.wcs.utils import proj_plane_pixel_scales

    assert proj_plane_pixel_scales(WCS(hdr))[0] * 3600 == pytest.approx(0.08, rel=1e-6)


def test_psf_pixel_scale_without_a_config(run_dir, tmp_path):
    """An explicit scale builds a north-up tangent plane, no config needed."""
    stray = tmp_path / "stray.geojson"
    PSFRegionMap.from_geojson(str(run_dir / "tiny_kernel.geojson")).to_file(stray)
    out = psf_to_fits(stray, RA + 0.1, DEC, tmp_path / "stray.fits", pixel_scale=0.1)

    hdr = fits.getheader(out)
    from astropy.wcs.utils import proj_plane_pixel_scales

    assert proj_plane_pixel_scales(WCS(hdr))[0] * 3600 == pytest.approx(0.1, rel=1e-6)
    assert hdr["MAPKIND"] == "psf_lo"  # unsuffixed name falls back to the default


def test_psf_finds_the_map_beside_a_relative_out_dir_config(run_dir, tmp_path):
    """A run directory works wherever the process runs from."""
    cfg = json.loads((run_dir / "tiny.json").read_text())
    cfg["out_dir"] = "somewhere/else"  # relative, resolves against the CWD
    (run_dir / "tiny.json").write_text(json.dumps(cfg))

    out = psf_to_fits(run_dir, RA + 0.1, DEC, tmp_path / "kern.fits", kind="kernel")
    assert fits.getheader(out)["PSFKEY"] == 0


def test_psf_missing_map_raises(run_dir):
    (run_dir / "tiny_psf_hi.geojson").unlink()
    with pytest.raises(FileNotFoundError, match="region map"):
        psf_to_fits(run_dir / "tiny.json", RA, DEC, kind="psf_hi")


def test_stamp_wcs_needs_a_scale_or_a_reference():
    assert stamp_wcs((5, 5), RA, DEC) is None


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Small array-driven fit, as in the other diagnostic tests."""
    images, segmap, catalog, psfs, _truth, wht = make_simple_data(
        seed=5, nsrc=15, size=101, ndilate=2, peak_snr=5
    )
    kernel = mutils.matching_kernel(psfs[0], psfs[1])
    w = _wcs(images[0].shape, SCALE_HI)
    _table, _resid, pipe = pipeline.run(
        images, segmap, catalog=catalog, weights=wht, kernels=[None, kernel],
        wcs=[w, w.deepcopy()],
        config=pipeline._FitConfig(extend_mode="none"),
    )
    return pipe


def test_source_stamps_fits_layout_and_wcs(fitted_pipeline, tmp_path):
    pipe = fitted_pipeline
    sid = int(pipe.table["id"][0])
    out = source_stamps_to_fits(pipe, sid, tmp_path / "src.fits", half_size=12)

    with fits.open(out) as hdul:
        names = [h.name for h in hdul[1:]]
        hdr = hdul[0].header
        lo_shape = hdul["IMG_LO"].data.shape
        lo_header = hdul["IMG_LO"].header.copy()
        row_id = int(hdul["FITROW"].data["id"][0])
    assert {"IMG_HI", "SEGMAP", "TMPL_HI", "IMG_LO", "TMPL_LO", "MODEL",
            "RESID", "FITROW"} <= set(names)
    assert hdr["ID"] == sid
    assert hdr["FLUX"] == pytest.approx(float(pipe.table["flux_1"][0]), rel=1e-5)
    assert row_id == sid
    assert lo_shape == (25, 25)

    # the cutout WCS must place the source at its own stamp position
    prod = pipe.source_products(sid, half_size=12)
    ysl, xsl = prod["slices_lo"]
    x, y = prod["position"]
    ra, dec = pipe.wcs[0].wcs_pix2world(x, y, 0)
    xs, ys = WCS(lo_header).wcs_world2pix(ra, dec, 0)
    assert (float(xs), float(ys)) == pytest.approx(
        (x - xsl.start, y - ysl.start), abs=1e-6
    )


@pytest.mark.parametrize("style", ["subphot", "stages"])
def test_source_diagnostic_png(fitted_pipeline, tmp_path, style):
    pipe = fitted_pipeline
    sid = int(pipe.table["id"][0])
    out = source_diagnostic_png(
        pipe, sid, tmp_path / f"{style}.png", style=style, size=41, half_size=12
    )
    assert out.exists() and out.stat().st_size > 0


def test_stamps_and_diag_cli_from_a_run_directory(run_dir, monkeypatch):
    """End to end: config -> load_fit -> one output file per id, named from the run."""
    from mophongo.pipeline import Pipeline

    # fake run products on the reference grid, as in the load_fit resume test
    Table({"id": [1], "x": [32.0], "y": [32.0], "flux_1": [2.5], "err_1": [0.1]}
          ).write(run_dir / "tiny_fit_table.fits")
    fits.writeto(run_dir / "tiny_residual.fits", np.zeros((64, 64), np.float32))
    Table(rows=[(1, 1, 32.0, 32.0, 0.0, 0.0, 2.5, 0.1, 1)],
          names=["id", "id_parent", "x", "y", "dx", "dy", "flux", "err", "id_scene"],
          ).write(run_dir / "tiny_templates.fits")
    monkeypatch.setattr(Pipeline, "_ensure_maps", lambda self: None)

    main(["stamps", str(run_dir), "1", "--half-size", "8"])
    main(["diag", str(run_dir), "1", "--size", "21"])

    with fits.open(run_dir / "tiny_1_stamps.fits") as hdul:
        names = [h.name for h in hdul[1:]]
        assert hdul[0].header["RUNNAME"] == "tiny"
        assert hdul[0].header["FLUX"] == pytest.approx(2.5)
        # the hi-res map is picked up from its cached geojson
        psf_hi_hdr = hdul["PSF_HI"].header.copy()
    assert {"IMG_HI", "IMG_LO", "MODEL", "RESID", "PSF_HI"} <= set(names)
    # the PSF stamp is centered on the source, not on the map or the mosaic
    cat = Table.read(json.loads((run_dir / "tiny.json").read_text())["catalog"])
    assert WCS(psf_hi_hdr).wcs.crval == pytest.approx(
        [cat["ra"][0], cat["dec"][0]], abs=1e-9
    )
    assert (run_dir / "tiny_1_subphot.png").exists()


def test_config_run_writes_the_residual_through_its_memmap(run_dir):
    """End-to-end config run: the residual file IS the accumulator.

    run() maps `f_residual` and accumulates scene models into it, so
    write_outputs only flushes. What lands on disk must equal image - model,
    and the band weight map must be gone from the scenes afterwards (nothing
    plots in this run, so nothing reads it again).
    """
    from pathlib import Path

    import numpy as np
    from mophongo.pipeline import Pipeline
    from mophongo.scene import _slices_from_bbox

    pipe = Pipeline.from_config(run_dir / "tiny.json")
    pipe.run()

    res = pipe.residuals[0]
    assert isinstance(res, np.memmap), "residual should be file-backed"
    assert Path(res.filename) == pipe.f_residual

    # this config plots scenes, so the weights survive run() -- the figures
    # still mask on them -- and write_outputs releases them once drawn
    assert pipe._scene_pixels_needed()
    scene = pipe.all_scenes[0][0]
    assert scene.weights is not None

    expected = np.asarray(res).copy()
    pipe.write_outputs()

    assert all(s.weights is None for s in pipe.all_scenes[0])
    # a scene still reports its residual without them, unmasked
    sl = _slices_from_bbox(scene.bbox)
    assert scene.residual().shape == scene.image[sl].shape

    on_disk = fits.getdata(pipe.f_residual)
    assert np.array_equal(on_disk, expected)
    assert on_disk.shape == pipe.images[1].shape
    # residual = image - model, and the model came from the fitted scenes
    model = pipe.model_images[0]
    assert np.allclose(on_disk, pipe.images[1] - model, atol=1e-6)
