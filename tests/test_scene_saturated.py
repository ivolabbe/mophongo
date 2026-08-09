"""Test that saturated templates land in their own scene."""

from __future__ import annotations

import numpy as np

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
