"""Test that saturated templates land in their own scene."""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.scene import generate_scenes
from mophongo.templates import Templates


def _make_image_segmap(positions, shape=(80, 80), src_radius=2):
    image = np.zeros(shape, dtype=np.float32)
    segmap = np.zeros(shape, dtype=np.int32)
    yy, xx = np.indices(shape)
    for k, (y, x) in enumerate(positions, start=1):
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        mask = d2 <= src_radius ** 2
        image[mask] += 1.0
        segmap[mask] = k
    return image, segmap


def test_saturated_template_isolated():
    rng = np.random.default_rng(0)
    # 5 close-packed sources in a single coupling cluster.
    positions = [(20, 20), (22, 22), (24, 24), (26, 26), (28, 28)]
    image, segmap = _make_image_segmap(positions)
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)

    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    # mark template index 2 as saturated.
    tmpls.templates[2].is_saturated = True

    scenes, labels = generate_scenes(
        tmpls.templates, image, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    sat_lbl = int(labels[2])
    others = np.delete(labels, 2).astype(int)
    assert sat_lbl not in others.tolist(), (
        f"saturated template merged with neighbours: labels={labels.tolist()}"
    )
    assert int(np.unique(labels).size) >= 2


def test_isolate_saturated_off_default_behavior():
    rng = np.random.default_rng(1)
    positions = [(20, 20), (22, 22), (24, 24)]
    image, segmap = _make_image_segmap(positions)
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    tmpls.templates[1].is_saturated = True
    _, labels = generate_scenes(
        tmpls.templates, image, weight,
        coupling_thresh=0.0, minimum_bright=1, isolate_saturated=False,
    )
    # All three coupled (overlapping) sources merge into the same scene.
    assert int(np.unique(labels).size) == 1


def test_saturated_group_shares_one_scene():
    """Templates with the same sat_group are one star: one scene together."""
    rng = np.random.default_rng(3)
    positions = [(20, 20), (22, 22), (24, 24), (26, 26), (50, 50)]
    image, segmap = _make_image_segmap(positions)
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)

    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    # templates 1 and 2 are fragments of the same saturated star (group 2);
    # template 3 is a different saturated star with its own group.
    for i, group in ((1, 2), (2, 2), (3, 4)):
        tmpls.templates[i].is_saturated = True
        tmpls.templates[i].sat_group = group

    _, labels = generate_scenes(
        tmpls.templates, image, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    labels = labels.astype(int)
    assert labels[1] == labels[2]           # same star -> same scene
    assert labels[3] != labels[1]           # different star -> different scene
    normals = {labels[0], labels[4]}
    assert labels[1] not in normals and labels[3] not in normals


def test_saturated_without_group_gets_own_scene_each():
    """Legacy 0/1 flags (no group id) keep the one-scene-per-template rule."""
    rng = np.random.default_rng(4)
    positions = [(20, 20), (22, 22), (24, 24)]
    image, segmap = _make_image_segmap(positions)
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    for i in (1, 2):
        tmpls.templates[i].is_saturated = True
        tmpls.templates[i].sat_group = 1  # legacy boolean flag value
    _, labels = generate_scenes(
        tmpls.templates, image, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    assert labels[1] != labels[2]


def test_scene_plot_keeps_foreign_saturated_out_of_the_image_scale(tmp_path):
    """null_segments leaves the star in the image panel, out of its stretch."""
    import matplotlib
    matplotlib.use("Agg")

    rng = np.random.default_rng(5)
    positions = [(20, 20), (22, 22), (24, 24)]
    image, segmap = _make_image_segmap(positions)
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    tmpls.templates[2].is_saturated = True
    tmpls.templates[2].sat_group = 3

    scenes, labels = generate_scenes(
        tmpls.templates, image, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    by_sat = {any(t.is_saturated for t in s.templates): s for s in scenes}
    normal, sat_scene = by_sat[False], by_sat[True]
    sat_id = int(tmpls.templates[2].id)

    import matplotlib.pyplot as plt

    for scene in (normal, sat_scene):
        scene.solve()
        fig, _ = scene.plot(image, segmap, null_segments=[sat_id])
        plt.close(fig)
    assert sat_id not in {int(t.id) for t in normal.templates}
    assert sat_id in {int(t.id) for t in sat_scene.templates}

    # panel contents: the grayscale Image panel still shows the foreign
    # saturated segment, but does not let it set the display scale
    normal.solve()
    fig, axes = normal.plot(image, segmap, null_segments=[sat_id])
    y0, y1, x0, x1 = normal.bbox
    seg_cut = segmap[y0 : y1 + 1, x0 : x1 + 1]
    foreign = seg_cut == sat_id
    assert foreign.any(), "the foreign saturated segment is outside the bbox"

    def _panels(f, axs):
        names = ["Template", "Image", "Model", "Segmap", "Residual", "Color"]
        out = {}
        for i, name in enumerate(names):
            im = axs[i].get_images()[0]
            out[name] = (np.asarray(im.get_array()), im.get_clim())
        plt.close(f)
        return out

    nulled = _panels(fig, axes)
    fig2, axes2 = normal.plot(image, segmap, null_segments=None)
    plain = _panels(fig2, axes2)

    # the star is drawn either way ...
    assert plain["Image"][0][foreign].max() > 0, "test setup: star not visible"
    assert np.array_equal(nulled["Image"][0], plain["Image"][0]), (
        "Image panel pixels changed when null_segments was passed"
    )
    # ... but its brightness is kept out of the panel's stretch
    assert nulled["Image"][1][1] < plain["Image"][1][1], (
        "Image panel stretch still set by the foreign saturated star"
    )
    # the Color panel is unaffected by null_segments entirely
    assert np.array_equal(nulled["Color"][0], plain["Color"][0]), (
        "Color panel changed when null_segments was passed"
    )


def test_saturated_scene_fits_rigid_shift():
    """A saturated-star scene fits one shift even though its fragments
    would fail every astrometry anchor cut (isolation, star exclusion)."""
    from scipy.ndimage import shift as nd_shift

    from mophongo.fit import FitConfig

    rng = np.random.default_rng(6)
    # two blended fragments of one star
    positions = [(40, 38), (40, 44)]
    image, segmap = _make_image_segmap(positions, src_radius=3)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    for t in tmpls.templates:
        t.is_saturated = True
        t.sat_group = 1  # ungrouped is fine: the pair still shares one scene here
        t.is_star = True

    # fit image: same star shifted by (+0.3, +0.2) px + noise (small enough
    # for one linearised pass; production iterates fit_astrometry_niter=5)
    true_dx, true_dy = 0.3, 0.2
    fit_img = nd_shift(image, (true_dy, true_dx), order=3) * 25.0
    fit_img += rng.normal(0.0, 0.01, image.shape).astype(np.float32)
    weight = np.full(image.shape, 1e4, dtype=np.float32)

    # both fragments carry the same group -> one scene (group>1 required)
    for t in tmpls.templates:
        t.sat_group = 5
    scenes, _ = generate_scenes(
        tmpls.templates, fit_img, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    sat = [s for s in scenes if all(t.is_saturated for t in s.templates)]
    assert len(sat) == 1 and len(sat[0].templates) == 2
    scene = sat[0]

    cfg = FitConfig(
        fit_astrometry_joint=True, fit_astrometry_niter=1,
        astrom_exclude_stars=True,   # would normally veto the star
        astrom_isolation_thresh=0.7,  # blended pair fails this too
        astrom_damping=1.0,
    )
    scene.solve(config=cfg, apply_shifts=False)
    assert scene.shifts is not None and len(scene.shifts) > 0
    shifts = np.array([t.to_shift for t in scene.templates])
    # rigid: identical shift on both fragments (order 0)
    assert np.allclose(shifts[0], shifts[1], atol=1e-8)
    assert shifts[0][0] == pytest.approx(true_dx, abs=0.15)
    assert shifts[0][1] == pytest.approx(true_dy, abs=0.15)


def test_isolation_thresh_counts_only_isolated_toward_floor():
    """With isolation_thresh set, a blended pair must not count toward
    minimum_bright at merge time, so its scene merges into a neighbour."""
    rng = np.random.default_rng(2)
    # Blended pair (overlapping disks) far from an isolated bright source.
    positions = [(20, 20), (22, 21), (60, 60)]
    image, segmap = _make_image_segmap(positions, src_radius=3)
    image += rng.normal(0.0, 0.02, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])

    # Without isolation: pair members count as bright -> pair scene stands alone.
    _, labels = generate_scenes(
        tmpls.templates, image, weight,
        coupling_thresh=1e-3, minimum_bright=1, snr_thresh_astrom=1.0,
    )
    assert labels[0] == labels[1]  # blended pair coupled into one scene
    n_without = np.unique(labels).size

    # With a strict isolation cut the pair has no isolated member, so its
    # scene is under the floor and merges toward the isolated source.
    _, labels_iso = generate_scenes(
        tmpls.templates, image, weight,
        coupling_thresh=1e-3, minimum_bright=1, snr_thresh_astrom=1.0,
        isolation_thresh=0.95,
    )
    assert np.unique(labels_iso).size < n_without
    assert labels_iso[0] == labels_iso[2]
