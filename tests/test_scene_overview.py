"""Tests for the full-field scene map (verification.scene_label_map/overview)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mophongo.verification import save_scene_overview, scene_label_map


def _scene(sid: int, ids: list[int], bbox: tuple[int, int, int, int]):
    return SimpleNamespace(
        id=sid,
        templates=[SimpleNamespace(id=i) for i in ids],
        bbox=bbox,
    )


def _field(n: int = 32):
    """Four labeled blocks; scenes claim the first three, one is left out."""
    segmap = np.zeros((n, n), dtype=np.int32)
    segmap[2:10, 2:10] = 1
    segmap[2:10, 20:28] = 2
    segmap[20:28, 2:10] = 3
    segmap[20:28, 20:28] = 4
    scenes = [
        _scene(1, [1, 2], (2, 10, 2, 28)),
        _scene(2, [3], (20, 28, 2, 10)),
    ]
    return segmap, scenes


def test_scene_label_map_indexes_scenes_not_segments():
    segmap, scenes = _field()
    smap = scene_label_map(segmap, scenes)

    assert smap.shape == segmap.shape
    # both members of scene 1 carry the same color index
    assert set(np.unique(smap[segmap == 1])) == {1}
    assert set(np.unique(smap[segmap == 2])) == {1}
    assert set(np.unique(smap[segmap == 3])) == {2}
    # a segment no scene claimed, and the background, stay 0
    assert set(np.unique(smap[segmap == 4])) == {0}
    assert set(np.unique(smap[segmap == 0])) == {0}


def test_scene_label_map_ignores_ids_outside_the_segmap():
    """Deblend children and footprint-cut sources must not index out of range."""
    segmap, scenes = _field()
    scenes[0].templates.append(SimpleNamespace(id=9999))
    smap = scene_label_map(segmap, scenes)
    assert set(np.unique(smap)) == {0, 1, 2}


def test_scene_label_map_decimates_by_block_maximum():
    """Small segments must survive the decimation, not fall between samples."""
    segmap, scenes = _field()
    segmap[15, 15] = 3  # a one-pixel segment, invisible to plain subsampling
    smap = scene_label_map(segmap, scenes, step=4)

    assert smap.shape == (8, 8)
    assert smap[15 // 4, 15 // 4] == 2  # the scene that owns segment 3
    assert np.array_equal(smap[:2, :2], np.full((2, 2), 1))  # the 8x8 block


def test_scene_label_map_accepts_float_labels():
    """Some releases ship float segmentation maps (see load_data)."""
    segmap, scenes = _field()
    smap = scene_label_map(segmap.astype(np.float64), scenes)
    assert np.array_equal(smap, scene_label_map(segmap, scenes))


def test_save_scene_overview_decimates_a_large_field(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    segmap, scenes = _field(n=64)
    rng = np.random.default_rng(0)
    image = rng.normal(0, 1, segmap.shape)

    out = tmp_path / "scene_map.png"
    # max_side below the field size forces the display decimation path
    save_scene_overview(image, segmap, scenes, out, max_side=16)
    assert out.exists() and out.stat().st_size > 0

    small = tmp_path / "scene_map_full.png"
    save_scene_overview(image, segmap, scenes, small)
    assert small.exists() and small.stat().st_size > 0
