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
