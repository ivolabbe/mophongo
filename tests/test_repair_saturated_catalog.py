"""Tests for :func:`mophongo.catalog.repair_saturated_catalog`."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from mophongo.catalog import repair_saturated_catalog


def _build_oversplit_segmap(
    shape: tuple[int, int],
    center: tuple[float, float],
    n_children: int = 8,
    r_outer: float = 30.0,
    r_inner_hole: float = 8.0,
    first_label: int = 10,
) -> tuple[np.ndarray, list[int]]:
    """Build a segmap with an oversplit star: ``n_children`` angular wedges
    around ``center`` with an interior hole of radius ``r_inner_hole``.
    """
    seg = np.zeros(shape, dtype=np.int32)
    yc, xc = center
    yy, xx = np.indices(shape)
    dy = yy - yc
    dx = xx - xc
    rr = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)
    wedge = (
        np.floor((theta + np.pi) / (2 * np.pi) * n_children).astype(int)
        % n_children
    )
    in_annulus = (rr >= r_inner_hole) & (rr <= r_outer)
    labels = []
    for k in range(n_children):
        lbl = first_label + k
        mask = in_annulus & (wedge == k)
        seg[mask] = lbl
        labels.append(lbl)
    return seg, labels


def _build_catalog(labels: list[int], shape: tuple[int, int]) -> Table:
    rng = np.random.default_rng(0)
    n = len(labels)
    cat = Table()
    cat["id"] = np.array(labels, dtype=np.int64)
    cat["x"] = rng.uniform(0, shape[1], n).astype(np.float32)
    cat["y"] = rng.uniform(0, shape[0], n).astype(np.float32)
    cat["flux"] = rng.uniform(1.0, 5.0, n).astype(np.float32)
    cat["FLAG_PERSISTENCE_F444W"] = np.zeros(n, dtype=np.int8)
    return cat


def test_repair_basic_merge_and_close():
    shape = (120, 120)
    center = (60.0, 60.0)
    seg, child_labels = _build_oversplit_segmap(
        shape, center, n_children=8, r_outer=30.0, r_inner_hole=8.0,
        first_label=10,
    )

    # Add a real neighbor segment well outside merge circle.
    seg[100:110, 100:110] = 500
    cat = _build_catalog(child_labels + [500], shape)

    fit_table = Table()
    fit_table["id"] = np.array([1], dtype=np.int64)
    fit_table["xc"] = np.array([center[1]], dtype=np.float64)
    fit_table["yc"] = np.array([center[0]], dtype=np.float64)
    fit_table["ok"] = np.array([True], dtype=bool)

    new_cat, new_seg, log = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=4.0, n_fwhm=10.0,  # r_merge = 40 > r_outer=30
        filter_name="F444W",
    )

    parent_id = int(log["parent_id"][0])
    assert parent_id > seg.max()
    assert parent_id > int(np.asarray(cat["id"]).max())
    assert log["n_children"][0] == len(child_labels)

    # Children gone from segmap.
    for lbl in child_labels:
        assert not np.any(new_seg == lbl), f"child {lbl} still in segmap"
    # Parent occupies their pixels.
    assert np.any(new_seg == parent_id)

    # Interior hole closed: pixels at distance < r_inner_hole that were 0
    # are now parent_id.
    yy, xx = np.indices(shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    inside_hole = rr < 6.0  # well within original hole
    assert np.all(new_seg[inside_hole] == parent_id)

    # Real neighbor untouched.
    assert np.all(new_seg[100:110, 100:110] == 500)

    # Catalog: child rows dropped, parent row appended, neighbor kept.
    assert parent_id in set(int(i) for i in new_cat["id"])
    assert 500 in set(int(i) for i in new_cat["id"])
    for lbl in child_labels:
        assert lbl not in set(int(i) for i in new_cat["id"])

    # Flag column set correctly.
    flag = np.asarray(new_cat["FLAG_SATURATED_F444W"])
    assert flag[new_cat["id"] == parent_id][0] == 1
    assert flag[new_cat["id"] == 500][0] == 0

    # Position of parent row matches PSF fit centre.
    parent_row = new_cat[new_cat["id"] == parent_id][0]
    assert parent_row["x"] == pytest.approx(center[1])
    assert parent_row["y"] == pytest.approx(center[0])


def test_repair_skips_failed_fits():
    shape = (80, 80)
    seg, child_labels = _build_oversplit_segmap(
        shape, (40.0, 40.0), n_children=4, r_outer=20.0, r_inner_hole=5.0,
        first_label=1,
    )
    cat = _build_catalog(child_labels, shape)
    fit_table = Table()
    fit_table["id"] = [1, 2]
    fit_table["xc"] = [40.0, 10.0]
    fit_table["yc"] = [40.0, 10.0]
    fit_table["ok"] = [False, False]

    new_cat, new_seg, log = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=3.0, n_fwhm=8.0, filter_name="F444W",
    )
    # Nothing merged.
    assert len(log) == 0
    assert np.array_equal(new_seg, seg)
    assert "FLAG_SATURATED_F444W" in new_cat.colnames
    assert int(new_cat["FLAG_SATURATED_F444W"].sum()) == 0


def test_diagnostic_plot_written(tmp_path):
    """Smoke test for
    :func:`mophongo.verification.plot_saturated_catalog_repair`.
    """
    from mophongo.verification import plot_saturated_catalog_repair

    shape = (120, 120)
    center = (60.0, 60.0)
    seg, child_labels = _build_oversplit_segmap(
        shape, center, n_children=8, r_outer=30.0, r_inner_hole=8.0,
        first_label=10,
    )
    seg[100:110, 100:110] = 500
    cat = _build_catalog(child_labels + [500], shape)

    rng = np.random.default_rng(1)
    sci = rng.normal(0.0, 1.0, shape).astype(np.float32)
    sci[seg > 0] += 50.0

    fit_table = Table(
        {"id": [1], "xc": [center[1]], "yc": [center[0]], "ok": [True]}
    )
    new_cat, new_seg, log = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=4.0, n_fwhm=10.0, filter_name="F444W",
    )

    out = tmp_path / "diag.png"
    written = plot_saturated_catalog_repair(
        sci, seg, new_seg, cat, new_cat, log,
        out_path=out, n_sources=1,
    )
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_flux_fraction_filter_preserves_bright_neighbour():
    """A bright neighbour inside the merge radius whose own flux
    dominates over the saturated star's PSF should survive the merge.
    """
    shape = (200, 200)
    center = (100.0, 100.0)
    # Oversplit saturated star (spike fragments around centre).
    seg, child_labels = _build_oversplit_segmap(
        shape, center, n_children=8, r_outer=40.0, r_inner_hole=8.0,
        first_label=10,
    )
    # Bright neighbour at (100, 170) inside merge radius — its own label,
    # not part of the spike fragments.
    seg_neighbor_label = 500
    yy, xx = np.indices(shape)
    nbr_mask = (yy - 100) ** 2 + (xx - 170) ** 2 <= 25
    seg[nbr_mask] = seg_neighbor_label

    cat = _build_catalog(child_labels + [seg_neighbor_label], shape)

    # Build sci image: faint star wings (low) on top of a bright
    # neighbour (high). PSF stamp normalised; amplitude small.
    sci = np.zeros(shape, dtype=np.float64)
    # PSF stamp (Gaussian) sum-normalised at sci pixel scale.
    from astropy.convolution import Gaussian2DKernel

    psf_stamp = Gaussian2DKernel(2.5, x_size=81, y_size=81).array.astype(float)
    psf_stamp /= psf_stamp.sum()
    psf_amp = 1.0
    # Drop PSF model into sci at star centre.
    ph, pw = psf_stamp.shape
    yi = int(center[0]); xi = int(center[1])
    sci[yi - ph // 2: yi - ph // 2 + ph,
        xi - pw // 2: xi - pw // 2 + pw] += psf_amp * psf_stamp
    # Bright neighbour: 1000x the PSF wing contribution there.
    sci[nbr_mask] += 100.0

    fit_table = Table(
        {"id": [1], "xc": [center[1]], "yc": [center[0]],
         "ok": [True], "amplitude": [psf_amp]}
    )

    # WITHOUT filter — neighbour gets absorbed.
    cat_a, seg_a, log_a = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=3.0, n_fwhm=30.0,  # r=90 px covers neighbour at (100,170)
        filter_name="F444W",
    )
    assert seg_neighbor_label not in set(int(i) for i in cat_a["id"])

    # WITH filter — neighbour survives (psf*amp << bright neighbour flux).
    cat_b, seg_b, log_b = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=3.0, n_fwhm=30.0,
        filter_name="F444W",
        sci=sci, psf_stamp=psf_stamp, flux_frac_thresh=0.5,
    )
    assert seg_neighbor_label in set(int(i) for i in cat_b["id"])
    # Bright neighbour still has its own segmap label.
    assert np.any(seg_b == seg_neighbor_label)


def test_repair_no_segments_in_circle_is_noop():
    shape = (60, 60)
    seg = np.zeros(shape, dtype=np.int32)
    seg[5:10, 5:10] = 7  # far from the fit centre
    cat = Table(
        {"id": [7], "x": [7.0], "y": [7.0]}
    )
    fit_table = Table(
        {"id": [1], "xc": [50.0], "yc": [50.0], "ok": [True]}
    )
    new_cat, new_seg, log = repair_saturated_catalog(
        cat, seg, fit_table,
        fwhm_pix=2.0, n_fwhm=3.0, filter_name="F770W",
    )
    assert len(log) == 0
    assert np.array_equal(new_seg, seg)
    assert "FLAG_SATURATED_F770W" in new_cat.colnames
    assert int(new_cat["FLAG_SATURATED_F770W"].sum()) == 0
