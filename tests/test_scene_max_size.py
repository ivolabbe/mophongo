"""`scene_max_size`: local threshold raise inside oversized components."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mophongo.scene import build_scene_tree_from_normal


def _chain_normal(n: int, off: float = 0.5) -> tuple[sp.csr_matrix, np.ndarray]:
    """Normal matrix of a 1-D chain: every neighbour pair coupled equally."""
    diag = np.ones(n)
    A = sp.diags([np.full(n - 1, off), diag, np.full(n - 1, off)], [-1, 0, 1]).tocsr()
    return A, np.ones(n)


def test_no_max_size_keeps_original_behaviour():
    A, b = _chain_normal(40)
    labels, nscene = build_scene_tree_from_normal(A, b, coupling_thresh=0.1)
    assert nscene == 1
    assert np.bincount(labels)[1:].max() == 40


def test_max_size_splits_oversized_component():
    # Chain with graded couplings so the local bisection has scores to work
    # with: score ~ 0.5/(1 + k/8) along the chain.
    n = 40
    off = 0.5 / (1.0 + np.arange(n - 1) / 8.0)
    A = sp.diags([off, np.ones(n), off], [-1, 0, 1]).tocsr()
    labels, nscene = build_scene_tree_from_normal(
        A, np.ones(n), coupling_thresh=0.01, max_size=10
    )
    sizes = np.bincount(labels)[1:]
    assert sizes.max() <= 10
    assert sizes.sum() == n
    assert nscene > 1


def test_max_size_is_a_ceiling_not_a_target():
    """Components already under the cap must come back unchanged."""
    A, b = _chain_normal(8)
    ref, nref = build_scene_tree_from_normal(A, b, coupling_thresh=0.1)
    got, ngot = build_scene_tree_from_normal(A, b, coupling_thresh=0.1, max_size=100)
    assert ngot == nref
    np.testing.assert_array_equal(got, ref)


def test_max_size_leaves_other_components_untouched():
    """Only the oversized component is split; a small strongly coupled pair
    elsewhere keeps its floor-threshold grouping."""
    # Block 1: chain of 12 (graded couplings); block 2: tight pair.
    n1, n = 12, 14
    off1 = 0.5 / (1.0 + np.arange(n1 - 1) / 4.0)
    A = sp.lil_matrix((n, n))
    A.setdiag(np.ones(n))
    for k, v in enumerate(off1):
        A[k, k + 1] = A[k + 1, k] = v
    A[12, 13] = A[13, 12] = 0.4  # tight pair, well above floor
    labels, _ = build_scene_tree_from_normal(
        A.tocsr(), np.ones(n), coupling_thresh=0.01, max_size=6
    )
    sizes = np.bincount(labels)[1:]
    assert sizes.max() <= 6
    assert labels[12] == labels[13]  # pair never split


# --- the cap must survive merge_small_scenes ---------------------------------
# build_scene_tree_from_normal splitting an oversized component is only half
# the job: merge_small_scenes then pulls the pieces back together chasing
# minimum_bright. Before max_size was threaded through, a run configured with
# scene_max_size=800 produced a 1718-template scene.


def _line_of_scenes(n_scenes, per_scene, spacing=50.0):
    """Templates in `n_scenes` well-separated clumps along x."""
    from types import SimpleNamespace

    templates, labels = [], []
    for s in range(n_scenes):
        for k in range(per_scene):
            templates.append(
                SimpleNamespace(position_original=(s * spacing + 0.1 * k, 0.0))
            )
            labels.append(s + 1)
    return templates, np.asarray(labels)


def test_merge_respects_max_size():
    from mophongo.scene import merge_small_scenes

    # 6 clumps of 10, none with enough bright members: unconstrained they all
    # collapse into one scene of 60.
    templates, labels = _line_of_scenes(6, 10)
    bright = np.zeros(len(templates), dtype=bool)

    free, n_free = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=None
    )
    assert np.bincount(free)[1:].max() == 60, "unconstrained merge should collapse"

    capped, n_capped = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=25
    )
    sizes = np.bincount(capped)[1:]
    assert sizes.max() <= 25, f"cap breached: {sizes.tolist()}"
    assert sizes.sum() == len(templates), "templates lost or duplicated"
    assert n_capped == len(sizes) > 1


def test_merge_cap_prefers_the_cap_over_minimum_bright():
    """A scene that cannot merge without breaching the cap stays underfilled."""
    from mophongo.scene import merge_small_scenes

    templates, labels = _line_of_scenes(3, 10)
    bright = np.zeros(len(templates), dtype=bool)
    # every scene is under minimum_bright, and any merge would exceed 10
    out, n = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=10
    )
    assert n == 3, "no merge is possible without breaching the cap"
    assert np.bincount(out)[1:].tolist() == [10, 10, 10]


def test_merge_still_merges_when_the_cap_allows_it():
    from mophongo.scene import merge_small_scenes

    templates, labels = _line_of_scenes(4, 10)
    bright = np.zeros(len(templates), dtype=bool)
    out, n = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=20
    )
    sizes = sorted(np.bincount(out)[1:].tolist())
    assert max(sizes) <= 20 and sum(sizes) == 40
    assert n < 4, "pairs should still merge under a cap of 20"


# --- extent cap: the shape the count cap leaves free --------------------------
# `max_size` bounds a scene's template count and says nothing about its shape.
# Components near the percolation threshold are dendritic, and a MINERVA UDS run
# produced a 25-template scene whose anchors spanned 4300 px. `max_extent` bounds
# the longer bounding-box side instead, by median bisection along that axis.


def _positions(labels_and_xy):
    x = np.array([p[0] for p in labels_and_xy], dtype=float)
    y = np.array([p[1] for p in labels_and_xy], dtype=float)
    return x, y


def _extents(labels, x, y):
    """Longest bbox side of every scene."""
    out = []
    for c in np.unique(labels):
        m = labels == c
        out.append(max(x[m].max() - x[m].min(), y[m].max() - y[m].min()))
    return np.asarray(out)


def test_spatial_split_bounds_the_longer_side():
    from mophongo.scene import _split_oversized_spatial

    # one long thin scene: 40 templates strung along x over 3900 px
    x, y = _positions([(100.0 * k, 0.0) for k in range(40)])
    labels = np.ones(40, dtype=int)

    out, n = _split_oversized_spatial(labels, x, y, max_extent=500.0)
    assert n > 1, "a 3900 px scene must split under a 500 px cap"
    assert _extents(out, x, y).max() <= 500.0
    assert np.bincount(out).sum() == 40, "templates lost or duplicated"


def test_spatial_split_squares_up_an_elongated_scene():
    """Bisecting the longer axis is what removes elongation, not just size."""
    from mophongo.scene import _split_oversized_spatial

    # 8 x 64 grid: 3150 px in x, 70 px in y -- 45:1 before, compact after
    pts = [(50.0 * i, 10.0 * j) for i in range(64) for j in range(8)]
    x, y = _positions(pts)
    labels = np.ones(len(pts), dtype=int)

    out, _ = _split_oversized_spatial(labels, x, y, max_extent=200.0)
    for c in np.unique(out):
        m = out == c
        dx = x[m].max() - x[m].min()
        dy = y[m].max() - y[m].min()
        assert max(dx, dy) <= 200.0
        # every piece is now within 4:1, against 45:1 for the input
        assert max(dx, dy) <= 4.0 * max(min(dx, dy), 10.0)


def test_spatial_split_leaves_compact_scenes_alone():
    from mophongo.scene import _split_oversized_spatial

    pts = [(10.0 * i, 10.0 * j) for i in range(5) for j in range(5)]
    x, y = _positions(pts)
    labels = np.repeat([1, 2], len(pts) // 2 + 1)[: len(pts)]
    out, n = _split_oversized_spatial(labels, x, y, max_extent=500.0)
    assert n == len(np.unique(labels))
    np.testing.assert_array_equal(
        np.unique(out, return_inverse=True)[1],
        np.unique(labels, return_inverse=True)[1],
    )


def test_spatial_split_terminates_on_coincident_positions():
    """Templates stacked on one pixel cannot be separated; must not spin."""
    from mophongo.scene import _split_oversized_spatial

    x = np.zeros(20)
    y = np.zeros(20)
    out, n = _split_oversized_spatial(np.ones(20, dtype=int), x, y, max_extent=1.0)
    assert n == 1, "a zero-extent scene is already under any positive cap"


def test_merge_radius_also_caps_the_merged_extent():
    """One knob: the merge radius must not let merging undo the split.

    A centroid-distance bound alone does not give this -- merging a wide
    scene with a near neighbour leaves a scene wider still -- so the veto
    tests the union bbox, not the pair separation.
    """
    from mophongo.scene import merge_small_scenes

    # 6 clumps spaced 50 px: unconstrained they collapse into one 250 px scene
    templates, labels = _line_of_scenes(6, 10, spacing=50.0)
    bright = np.zeros(len(templates), dtype=bool)

    free, _ = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_merge_radius=np.inf
    )
    assert np.bincount(free)[1:].max() == 60

    capped, _ = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_merge_radius=60.0
    )
    x = np.array([t.position_original[0] for t in templates])
    y = np.array([t.position_original[1] for t in templates])
    assert _extents(capped, x, y).max() <= 60.0
    assert np.bincount(capped)[1:].sum() == len(templates)


def test_infinite_merge_radius_changes_nothing():
    """np.inf must leave the pre-2026-08-13 partition byte for byte alone."""
    from mophongo.scene import merge_small_scenes

    templates, labels = _line_of_scenes(4, 10)
    bright = np.zeros(len(templates), dtype=bool)
    ref, nref = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=20
    )
    got, ngot = merge_small_scenes(
        labels, templates, bright, minimum_bright=5, max_size=20,
        max_merge_radius=np.inf,
    )
    assert ngot == nref
    np.testing.assert_array_equal(got, ref)
