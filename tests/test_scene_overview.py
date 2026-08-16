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


def _blob_scene(sid: int, xy: np.ndarray):
    return SimpleNamespace(
        id=sid,
        templates=[SimpleNamespace(position_original=p) for p in np.asarray(xy, float)],
    )


def test_save_scene_blobs_draws_every_scene_and_labels_only_the_large(tmp_path):
    """One patch per scene, and only the ones big enough to hold a number.

    The blob map is vector: it never touches the mosaic, so its cost is the
    scene count rather than the field size. Small scenes are still drawn --
    they are the majority of a real partition -- but numbering them all turns
    the figure into a smear of grey digits.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mophongo.verification import save_scene_blobs

    rng = np.random.default_rng(0)
    scenes = [
        # one sprawling scene, comfortably over label_min_pix
        _blob_scene(1, rng.normal(500, 120, size=(40, 2))),
        # three compact ones, well under it
        _blob_scene(2, rng.normal(900, 3, size=(5, 2))),
        _blob_scene(3, np.array([[100.0, 100.0]])),          # single template
        # collinear and small: a hull would be a zero-area sliver, so this
        # takes the circle path, and 20 px is under the label gate
        _blob_scene(4, np.array([[200.0, 200.0], [210.0, 200.0],
                                 [220.0, 200.0]])),
    ]

    out = tmp_path / "blobs.png"
    save_scene_blobs(scenes, (1000, 1000), out)
    assert out.exists() and out.stat().st_size > 0

    # inspect the axes rather than the file: every scene gets a patch, and
    # the label count is the gate working
    fig, ax = plt.subplots()
    try:
        save_scene_blobs(scenes, (1000, 1000), tmp_path / "again.png")
    finally:
        plt.close(fig)

    captured = {}
    real_subplots = plt.subplots

    def spy(*a, **k):
        fig, ax = real_subplots(*a, **k)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = spy
    try:
        save_scene_blobs(scenes, (1000, 1000), tmp_path / "spy.png")
    finally:
        plt.subplots = real_subplots

    ax = captured["ax"]
    assert len(ax.patches) == len(scenes), "every scene is drawn"
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1"], "only the sprawling scene is labelled"
    assert ax.texts[0].get_color() == "0.75", "scene numbers are grey"


def test_save_scene_blobs_handles_an_empty_partition(tmp_path):
    """No scenes is a figure with no patches, not a traceback."""
    import matplotlib

    matplotlib.use("Agg")

    from mophongo.verification import save_scene_blobs

    out = tmp_path / "none.png"
    save_scene_blobs([], (100, 100), out)
    assert out.exists()


def test_save_scene_partition_draws_both_panels(tmp_path):
    """One figure, two panels, one colour per scene across both."""
    import matplotlib.pyplot as plt

    from mophongo.verification import _scene_colors, save_scene_partition

    rng = np.random.default_rng(3)
    image = rng.normal(size=(120, 140))
    segmap = np.zeros((120, 140), dtype=int)
    segmap[10:20, 10:20] = 1
    segmap[60:70, 90:100] = 2
    scenes = [
        SimpleNamespace(id=1, bbox=(10, 20, 10, 20),
                        templates=[SimpleNamespace(id=1, position_original=(15.0, 15.0))]),
        SimpleNamespace(id=2, bbox=(60, 70, 90, 100),
                        templates=[SimpleNamespace(id=2, position_original=(95.0, 65.0))]),
    ]

    out = tmp_path / "scenes.png"
    save_scene_partition(image, segmap, scenes, out)
    assert out.exists() and out.stat().st_size > 0

    # both panels are drawn, and the scene colours are shared rather than
    # each panel inventing its own
    assert _scene_colors(2).shape == (3, 4)
    assert np.array_equal(_scene_colors(2), _scene_colors(2))
    plt.close("all")
